"""Sync registered model results and traces without rebuilding the frozen paper data.

Raw inputs are read only. Staging, checksums and the lock live in hidden/data/sync/.
Only validated public exports are copied to site/data/; the small tables go last.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import tempfile
import re

from anonymize_paths import anonymize
from export_traces import token_metric_path

ROOT = Path(__file__).resolve().parents[1]
CELL_FIELDS = {"status", "score", "post", "cost", "seconds", "explorer", "result_note"}


def read(path):
    if path.suffix == ".zst":
        return json.loads(subprocess.run(["zstd", "-dq", "--long=31", "-c", str(path)], check=True, capture_output=True).stdout)
    return json.loads(path.read_bytes())


def packed_artifact(path):
    """Large JSON files are stored as .json.zst by the results packer."""
    return next((p for p in (path, path.with_name(path.name + ".zst")) if p.is_file()), None)


def refresh_heartbeats(bench, archive):
    """Load available measurements for the exact, already-published result rows.

    The source hash guard keeps later upstream result corrections from being
    mixed into frozen website scores while filling holes in ancillary data.
    """
    changed = []
    for model in bench["models"]:
        key = model["key"]
        source = bench.get("result_sources", {}).get(key, {})
        if not source.get("source", "").startswith("results/"):
            continue
        folder = inside(archive, source["source"].removeprefix("results/"))
        updated = 0
        for repo in bench["repositories"]:
            rid = repo["id"]
            cell = bench["results"][rid].get(key, {})
            if cell.get("status") != "passed" or not repo.get("heartbeats"):
                continue
            result = folder / rid / "result.json"
            artifact = packed_artifact(folder / rid / "derived/heartbeats.json")
            if artifact is None or not result.is_file() or sha(result) != source.get("result_sha256", {}).get(rid):
                continue
            row, hb = read(result), read(artifact)
            if row["status"] != "passed" or row["metrics"]["post_lean_tokens"] != cell.get("post"):
                continue
            if hb.get("status") != "complete" or hb.get("body_heartbeats") is None:
                continue
            require(hb.get("repository", rid) == rid and hb.get("lean_tokens", cell["post"]) == cell["post"], f"{rid}/{key}: heartbeat endpoint mismatch")
            values = dict(heartbeats_after=round(hb["body_heartbeats"] / 1000),
                          heartbeats=round(100 * (1 - hb["body_heartbeats"] / 1000 / repo["heartbeats"]), 3))
            if any(cell.get(k) != value for k, value in values.items()):
                cell.update(values)
                updated += 1
        if updated:
            measured = [cells[key]["heartbeats"] for cells in bench["results"].values()
                        if cells.get(key, {}).get("status") == "passed" and "heartbeats_after" in cells[key]]
            model.update(heartbeats=round(mean(measured), 4), heartbeats_scored=len(measured),
                         heartbeats_improved=sum(v > 0 for v in measured))
            changed.append(dict(model=key, measurements=updated))
    return changed


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def origin_summary(bench, key, cells=None):
    """Equal-weight compression contributions, only with all passing pairs classified."""
    values = []
    for repo in bench["repositories"]:
        cell = cells[repo["id"]] if cells is not None else bench["results"][repo["id"]][key]
        if cell["status"] == "passed" and not cell.get("origin"):
            return None
        values.append((cell.get("origin", {}) if cell["status"] == "passed" else {}, repo["tokens"]))
    def part(*names):
        return round(mean(100 * sum(origin.get(name, 0) for name in names) / baseline for origin, baseline in values), 4)
    return dict(added=part("added"), dead_code=part("dead_code"), deleted=part("deleted"),
                proof_rewrite=part("proof_simplification", "structural_diff"), automation=part("automation"),
                syntax_optimization=part("syntax_optimization"), saved=part("dead_code", "deleted", "added", "proof_simplification", "structural_diff", "automation", "syntax_optimization"))


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes its input root: {relative}")
    return path


def native_prompt_text(fragments):
    parts = []

    def walk(value, key=None):
        if isinstance(value, str):
            if key in ("text", "content", "instructions") and not value.startswith("x-anthropic-billing-header:"):
                parts.append(value.strip("\n"))
        elif isinstance(value, list):
            for item in value:
                walk(item, key if isinstance(item, str) else None)
        elif isinstance(value, dict):
            for name, item in value.items():
                if name != "tools":
                    walk(item, name)

    for fragment in fragments:
        walk(fragment["value"], fragment["source"].rsplit(".", 1)[-1])
    return "\n\n".join(part for part in parts if part)


def logged_tools(capture):
    with (capture / "native.stdout.jsonl").open() as handle:
        for _, line in zip(range(20), handle):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            init = event.get("init") if event.get("event") == "init" else event if event.get("subtype") == "init" else None
            if init and isinstance(init.get("tools"), list):
                return init["tools"]
    return None


def settings(run):
    """Only publish settings recorded in the request; never inherit a model's defaults."""
    request = run["generation_parameters"].get("request") or {}
    output = [{"key": "effort", "label": "Reasoning effort", "value": run["reasoning_effort"], "how": "ours"}]
    for names, key, label in [
        (("maxOutputTokens", "max_output_tokens", "max_tokens"), "max_output", "Max output tokens"),
        (("temperature",), "temperature", "Temperature"),
        (("topP", "top_p"), "top_p", "Top-p"),
        (("topK", "top_k"), "top_k", "Top-k"),
    ]:
        value = next((request[n] for n in names if n in request), None)
        if value is not None:
            output.append(dict(key=key, label=label, value=value, how="default"))
    thinking = request.get("thinkingConfig") or {}
    if thinking.get("thinkingLevel"):
        require(thinking["thinkingLevel"].lower() == run["reasoning_effort"], "Thinking level differs from run effort")
        output.append(dict(key="thinking", label="Thinking level", value=thinking["thinkingLevel"], how="ours"))
    return output


def prepare(spec, bench, lean, archive):
    import yaml

    key, short = spec["key"], spec["short"]
    require(all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value) for value in (key, short)), "Invalid model key or short")
    live, frozen = inside(lean, spec["run"]), inside(archive, spec["source"])
    require(all(m["key"] == key or m["short"] != short for m in bench["models"]), f"Duplicate short: {short}")
    config_path = live / "run/config.snapshot.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    run = cfg["resolved"]
    require(run["model"] == key, f"{key}: config model mismatch")
    require(run["agent"]["prompt"] == bench["prompt"], f"{key}: task prompt differs from benchmark")
    require({k: run["container"][k] for k in bench["run"]["shared"]} == bench["run"]["shared"], f"{key}: container limits differ")
    require(cfg["system_prompt"]["mode"] == "native", f"{key}: expected native system prompt")
    prompt_path = inside(lean, cfg["system_prompt"]["snapshot"])
    snapshot = read(prompt_path)
    prompt_sha = hashlib.sha256(json.dumps(snapshot["fragments"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    require(prompt_sha == snapshot["native_instructions_sha256"] == cfg["system_prompt"]["sha256"], f"{key}: system prompt hash mismatch")
    require(snapshot["harness_version"] == run["harness_version"], f"{key}: system prompt harness mismatch")
    text = native_prompt_text(snapshot["fragments"])
    require(bool(text), f"{key}: empty native prompt")
    ids = {r["id"] for r in bench["repositories"]}
    paths = {p.parent.name: p for p in frozen.glob("*/result.json")}
    require(set(paths) == ids, f"{key}: result coverage differs: missing={sorted(ids - paths.keys())}, extra={sorted(paths.keys() - ids)}")
    model = {k: spec[k] for k in ("key", "short", "label", "provider", "logo", "color")}
    model.update(paper=False, effort=run["reasoning_effort"], harness=spec["config"]["harness"],
                 harness_version=run["harness_version"], origin=None, cost_note=spec.get("cost_note"))
    model["config"] = copy.deepcopy(spec["config"])
    model["config"].update(harness_version=run["harness_version"], settings=settings(run), system_prompt_chars=len(text),
        rates={("cache_write" if k == "cache_creation" else k): v for k, v in cfg["pricing"]["rates"].items()})
    jobs, cells, inputs, tools, result_hashes = [], {}, {config_path, prompt_path}, set(), {}
    for repo in bench["repositories"]:
        rid = repo["id"]
        path = paths[rid]
        row = read(path)
        require(row["repository"] == rid and row["model"] == key, f"{rid}: result identity mismatch")
        require(row["status"] in ("passed", "failed"), f"{rid}: postprocessing is incomplete")
        require(row["reasoning_effort"] == run["reasoning_effort"], f"{rid}: effort mismatch")
        require(row["metrics"]["baseline_lean_tokens"] == repo["tokens"], f"{rid}: baseline differs from benchmark")
        for field in ("cost_usd", "agent_seconds"):
            require(isinstance(row[field], (int, float)) and math.isfinite(row[field]) and row[field] >= 0, f"{rid}: invalid {field}")
        require(row["agent_seconds"] > 0, f"{rid}: missing agent duration")
        require(row["metrics"]["post_lean_tokens"] >= 0, f"{rid}: invalid final token count")
        require(row["metrics"]["lean_tokens_saved"] == repo["tokens"] - row["metrics"]["post_lean_tokens"], f"{rid}: inconsistent token totals")
        unpacked_path = live / rid / "result.json"
        unpacked = read(unpacked_path)
        comparable = copy.deepcopy(row)
        correction = comparable.pop("cost_correction", None)
        if correction:
            comparable["cost_usd"] = correction["recorded_cost_usd"]
        require(comparable == unpacked or row == unpacked, f"{rid}: packed and unpacked results disagree")
        playback_path = inside(live / rid, row["files"]["postprocessed_playback"])
        require(playback_path.name == "postprocessed-playback.json", f"{rid}: postprocessed playback is required")
        capture = playback_path.parent
        playback = read(playback_path)
        require(playback["baseline_lean_tokens"] == repo["tokens"] and playback["post_lean_tokens"] == row["metrics"]["post_lean_tokens"], f"{rid}: playback endpoint mismatch")
        require(playback.get("points"), f"{rid}: playback has no measured checkpoints")
        for name in ("standardized-trace.json", "native.stdout.jsonl"):
            require((capture / name).is_file(), f"{rid}: missing {name}")
            inputs.add(capture / name)
        inputs.update((path, unpacked_path, playback_path))
        if (live / rid / "prompt.txt").exists():
            inputs.add(live / rid / "prompt.txt")
        for point in playback["points"]:
            require(point.get("snapshot_path"), f"{rid}: checkpoint has no source snapshot")
            snapshot_path = inside(live / rid, point["snapshot_path"])
            require(snapshot_path.is_file(), f"{rid}: missing source snapshot: {snapshot_path.name}")
            inputs.add(snapshot_path)
            if point.get("incremental_diff_path"):
                diff = inside(live / rid, point["incremental_diff_path"])
                require(diff.is_file(), f"{rid}: missing checkpoint diff: {diff.name}")
                inputs.add(diff)
        for rollout in (live / rid / "native_rollout").glob("*.native-rollout.jsonl.gz"):
            inputs.add(rollout)
        tools.add(tuple(logged_tools(capture) or spec["config"].get("tools", [])))
        ok = row["status"] == "passed"
        cell = dict(status=row["status"], score=round(100 * row["metrics"]["lean_tokens_saved"] / repo["tokens"], 4) if ok else 0,
                    post=row["metrics"]["post_lean_tokens"], cost=round(row["cost_usd"], 3), seconds=round(row["agent_seconds"], 1), explorer=False)
        # Retain independently exported graphs only for an unchanged scored endpoint.
        previous = bench["results"][rid].get(key, {})
        if ok and previous.get("status") == "passed" and previous.get("post") == cell["post"]:
            for field in ("explorer", "origin"):
                if field in previous:
                    cell[field] = previous[field]
        heartbeat = packed_artifact(path.parent / "derived/heartbeats.json")
        if heartbeat is not None and ok and repo.get("heartbeats"):
            inputs.add(heartbeat)
            hb = read(heartbeat)
            if hb.get("status") == "complete" and hb.get("body_heartbeats") is not None:
                require(hb.get("repository", rid) == rid, f"{rid}: heartbeat repository mismatch")
                require(hb.get("lean_tokens", row["metrics"]["post_lean_tokens"]) == row["metrics"]["post_lean_tokens"], f"{rid}: heartbeat endpoint token mismatch")
                cell.update(heartbeats_after=round(hb["body_heartbeats"] / 1000),
                            heartbeats=round(100 * (1 - hb["body_heartbeats"] / 1000 / repo["heartbeats"]), 3))
        cells[rid] = cell
        result_hashes[rid] = sha(path)
        jobs.append(dict(repo=rid, model=key, status=row["status"], run=live / rid, capture=capture, result=row, recount_tokens=True))
    require(len(tools) == 1 and bool(next(iter(tools))), f"{key}: tool lists are missing or inconsistent")
    model["config"]["tools"] = list(next(iter(tools)))
    heartbeats = [c["heartbeats"] for c in cells.values() if "heartbeats" in c]
    model.update(score=round(mean(c["score"] for c in cells.values()), 4), cost=round(mean(j["result"]["cost_usd"] for j in jobs), 4),
        hours=round(mean(j["result"]["agent_seconds"] for j in jobs) / 3600, 4), passed=sum(c["status"] == "passed" for c in cells.values()),
        scored=len(ids), expected=len(ids), heartbeats=round(mean(heartbeats), 4) if heartbeats else None,
        heartbeats_improved=sum(v > 0 for v in heartbeats), heartbeats_scored=len(heartbeats))
    model["origin"] = origin_summary(bench, key, cells)
    provenance = dict(source="results/" + spec["source"], reasoning_effort=model["effort"], result_sha256=result_hashes,
                      system_prompt_sha256=prompt_sha, config_sha256=sha(config_path))
    return dict(spec=spec, model=model, cells=cells, jobs=jobs, inputs=inputs, provenance=provenance,
                prompt=dict(text=text, sha256=prompt_sha, captured=None))


def init_traces(stage, lean, data, shorts, cache):
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    sys.path.insert(0, str(lean / "scripts/analysis"))
    import plot_action_classes
    import export_traces as traces
    # The published baseline source index is sufficient for the paper's action classifier.
    def files(repo):
        return frozenset(json.loads(gzip.decompress((data / "repos" / repo / "source.json.gz").read_bytes())))
    plot_action_classes.local_repository_files = files
    traces._PAPER = plot_action_classes
    traces.LEANLEAN_CODE = lean
    traces._BUILDS = {}  # old paper replay builds are unrelated to the new runs
    traces.SHORT.update(shorts)
    traces.OUT = stage / "traces"


def export_trace(job):
    import export_traces
    result = export_traces.export_run(job)
    require(not result.get("error") and result.get("steps", 0) > 0, f"Trace export failed: {job['repo']}: {result.get('error')}")
    return job["repo"], job["model"], {k: result[k] for k in ("steps", "tools", "changes")}


def input_hashes(item, dependencies):
    return {str(p): sha(p) for p in sorted(item["inputs"] | dependencies)}


def valid_trace(data, rid, short):
    head_path = data / "traces" / rid / f"{short}.json.gz"
    pack_path = head_path.with_name(f"{short}.bin")
    if not head_path.is_file() or not pack_path.is_file():
        return False
    head = json.loads(gzip.decompress(head_path.read_bytes()))
    pack = pack_path.read_bytes()
    end = 0
    for offset, length in head["pack"]:
        require(offset == end and length > 0, f"Invalid trace pack offsets: {rid}/{short}")
        json.loads(gzip.decompress(pack[offset:offset + length]))
        end += length
    require(end == len(pack), f"Unindexed trace bytes: {rid}/{short}")
    return bool(head["steps"])


def publish(stage, data):
    """Install changed files, restoring the prior bytes if installation fails."""
    files = sorted((p for p in stage.rglob("*") if p.is_file()), key=lambda p: (len(p.relative_to(stage).parts) == 1, str(p)))
    changed = [p for p in files if not (data / p.relative_to(stage)).is_file() or p.read_bytes() != (data / p.relative_to(stage)).read_bytes()]
    with tempfile.TemporaryDirectory(prefix="rollback-", dir=stage.parent) as directory:
        backup, installed = Path(directory), []
        try:
            for source in changed:
                relative = source.relative_to(stage)
                target, old = data / relative, backup / relative
                if target.exists():
                    old.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(target, old)
                target.parent.mkdir(parents=True, exist_ok=True)
                installed.append(relative)
                os.replace(source, target)
        except BaseException:
            for relative in reversed(installed):
                old, target = backup / relative, data / relative
                if old.exists():
                    os.replace(old, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return len(changed)


def sync(args):
    manifest = read(args.manifest)
    require(manifest["schema_version"] == 1, "Unsupported model manifest schema")
    specs = manifest["models"]
    require(len({s["key"] for s in specs}) == len(specs), "Duplicate model keys")
    require(len({s["short"] for s in specs}) == len(specs), "Duplicate model shorts")
    if args.models:
        unknown = set(args.models) - {s["key"] for s in specs}
        require(not unknown, f"Unknown models: {sorted(unknown)}")
        specs = [s for s in specs if s["key"] in args.models]
    data, cache = args.data.resolve(), args.cache.resolve()
    bench, board, prompts = [read(data / f"{n}.json") for n in ("benchmark", "leaderboard", "system-prompts")]
    prepared = [prepare(spec, bench, args.leanlean_root.resolve(), args.archive_root.resolve()) for spec in specs]
    dependencies = {args.manifest.resolve(), Path(__file__).resolve(), Path(__file__).with_name("export_traces.py"),
                    Path(__file__).with_name("anonymize_paths.py"), args.leanlean_root.resolve() / "scripts/analysis/plot_action_classes.py",
                    token_metric_path(args.leanlean_root.resolve())}
    state_path = cache / "state.json"
    state = read(state_path) if state_path.exists() else {}
    pending, shorts, hashes = [], {}, {}
    trace_index = read(data / "traces/index.json")
    for item in prepared:
        model, spec = item["model"], item["spec"]
        key, short = model["key"], model["short"]
        item["inputs"].update(data / "repos" / j["repo"] / "source.json.gz" for j in item["jobs"])
        hashes[key] = input_hashes(item, dependencies)
        prior = state.get(key, {})
        current = not args.force and prior.get("inputs") == hashes[key]
        if current:
            current = all(valid_trace(data, j["repo"], short) and short in trace_index.get(j["repo"], {}) for j in item["jobs"])
            current = current and all((data / p).is_file() and sha(data / p) == h for p, h in prior.get("outputs", {}).items())
            current = current and prior.get("index") == {j["repo"]: trace_index.get(j["repo"], {}).get(short) for j in item["jobs"]}
        if not current:
            pending.extend(item["jobs"])
        shorts[key] = short
        bench["models"] = [model if m["key"] == key else m for m in bench["models"]] if any(m["key"] == key for m in bench["models"]) else bench["models"] + [model]
        for rid, cell in item["cells"].items():
            bench["results"][rid][key] = cell
        bands = {r["id"]: r["band"] for r in bench["repositories"]}
        bench["scale"][key] = {b["band"]: dict(
            score=round(mean(item["cells"][j["repo"]]["score"] for j in item["jobs"] if bands[j["repo"]] == b["band"]), 4),
            cost=round(mean(j["result"]["cost_usd"] for j in item["jobs"] if bands[j["repo"]] == b["band"]), 3)) for b in bench["bands"]}
        bench.setdefault("result_sources", {})[key] = item["provenance"]
        bench["updated_at"] = max(bench.get("updated_at", ""), spec["updated_at"])
        prompts[key] = item["prompt"]
        print(f"{model['label']} ({model['effort']}): {model['score']:.4f}%, ${model['cost']:.4f}/repo, {model['passed']}/{model['expected']} passed", flush=True)
    heartbeat_updates = refresh_heartbeats(bench, args.archive_root.resolve())
    if heartbeat_updates:
        print(f"Heartbeats: {sum(item['measurements'] for item in heartbeat_updates)} measurements refreshed across {len(heartbeat_updates)} models", flush=True)
    for rank, model in enumerate(sorted(bench["models"], key=lambda m: -m["score"]), 1):
        model["rank"] = rank
    board.update(models=bench["models"], updated_at=bench["updated_at"], results={rid: {key: {k: v for k, v in cell.items() if k in CELL_FIELDS} for key, cell in cells.items()} for rid, cells in bench["results"].items()})
    tables = {f"{name}.json": anonymize(value) for name, value in (("benchmark", bench), ("leaderboard", board), ("system-prompts", prompts))}
    stale_tables = [name for name, value in tables.items() if encoded(value) != (data / name).read_bytes()]
    print(f"{len(pending)} traces to export; {len(stale_tables)} tables to update", flush=True)
    if args.dry_run or args.check:
        return 1 if args.check and (pending or stale_tables) else 0
    with tempfile.TemporaryDirectory(prefix="stage-", dir=cache) as directory:
        stage = Path(directory)
        if pending:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_traces,
                    initargs=(stage, args.leanlean_root.resolve(), data, shorts, cache)) as pool:
                for n, (rid, key, summary) in enumerate(pool.map(export_trace, pending), 1):
                    short = shorts[key]
                    require(valid_trace(stage, rid, short), f"{rid}: invalid exported trace")
                    trace_index.setdefault(rid, {})[short] = summary
                    if n % 8 == 0 or n == len(pending):
                        print(f"Traces: {n}/{len(pending)}", flush=True)
        for item in prepared:
            require(input_hashes(item, dependencies) == hashes[item["model"]["key"]], "Inputs changed during export; rerun the sync")
        if pending:
            tables["traces/index.json"] = trace_index
        for name, value in tables.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded(value))
        count = publish(stage, data)
    for item in prepared:
        key, short = item["model"]["key"], item["model"]["short"]
        outputs = {str(p.relative_to(data)): sha(p) for j in item["jobs"] for p in
                   (data / "traces" / j["repo"] / f"{short}.json.gz", data / "traces" / j["repo"] / f"{short}.bin")}
        state[key] = dict(inputs=hashes[key], outputs=outputs, index={j["repo"]: trace_index[j["repo"]][short] for j in item["jobs"]})
    temp = state_path.with_suffix(".tmp")
    temp.write_bytes(encoded(state))
    os.replace(temp, state_path)
    print(f"Synced {len(prepared)} model(s); {count} public files changed. Inputs and cache remain gitignored.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="model keys from sync_models.json (default: all registered models)")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("sync_models.json"))
    parser.add_argument("--leanlean-root", type=Path, default=Path(os.environ.get("LEANLEAN_ROOT", Path.home() / "lean-lean")))
    parser.add_argument("--archive-root", type=Path, default=Path(os.environ.get("LEANLEAN_RESULTS_ROOT", Path.home() / "lean-lean-results")) / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "site/data")
    parser.add_argument("--cache", type=Path, default=ROOT / "hidden/data/sync")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and report changes without exporting")
    parser.add_argument("--check", action="store_true", help="exit 1 if a sync is needed, without publishing")
    parser.add_argument("--force", action="store_true", help="re-export traces even when inputs are unchanged")
    parser.add_argument("--skip-graphs", action="store_true", help="only sync result tables and traces")
    args = parser.parse_args()
    require(args.workers > 0, "--workers must be positive")
    args.cache.mkdir(parents=True, exist_ok=True)
    with (args.cache / "sync.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another sync is already running")
        try:
            result = sync(args)
            if args.skip_graphs:
                return result
            from sync_graphs import sync as sync_graphs
            graph_result = sync_graphs(args)
            return max(result, graph_result)
        except (ValueError, OSError, KeyError) as error:
            parser.exit(2, f"Sync aborted: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
