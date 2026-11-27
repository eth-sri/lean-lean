"""Export cached graph/diff pairs for registered models, without running Lean.

All source archives, classifier caches and baseline indices are validated before
installing any public files. Per-pair staging and fingerprints stay in hidden/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from anonymize_paths import anonymize
from sync_results import ROOT, encoded, inside, read, require, sha, publish

ORIGIN_KEYS = ("dead_code", "deleted", "added", "proof_simplification", "structural_diff", "automation", "syntax_optimization")
ACCOUNTING = None


def load_gzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def init_worker(lean):
    global ACCOUNTING
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lean), str(lean / "src")]
    import graph_data
    graph_data.DATASET = lean / "datasets/leanlean_20260914"
    path = lean / "scripts/analysis/refresh_website_accounting.py"
    spec = importlib.util.spec_from_file_location("website_source_accounting", path)
    ACCOUNTING = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ACCOUNTING)


def baseline(graph, source_dir):
    """Reconstruct the serializer's baseline using the site's existing node order."""
    names = graph["nodes"]["name"]
    member = {name: i for i, name in enumerate(names)}
    for index, aliases in graph.get("aliases", {}).items():
        for name in aliases:
            member[name] = int(index)
    offsets = load_gzip(source_dir / "source.json.gz")
    pack = (source_dir / "source.bin").read_bytes()
    sources = {name: json.loads(gzip.decompress(pack[offset:offset + size])) for name, (offset, size) in offsets.items()}
    nodes, lanes = graph["nodes"], graph["lanes"]
    return dict(names={name: i for i, name in enumerate(names)}, member=member,
        ranges=[[graph["files"][f] if f >= 0 else None, start, end] for f, start, end in zip(nodes["file"], nodes["start"], nodes["end"])],
        excluded=set(graph["excluded"]), sources=sources, positions=list(zip(nodes["x"], nodes["y"])),
        files=graph["files"], tokens=nodes["tokens"], world=graph["world"], per_lane=graph["per_lane"], lanes=lanes,
        lane_of=[min(range(len(lanes)), key=lambda i: abs(lanes[i][0] - y)) for y in nodes["y"]])


def scoped_sources(archive, record):
    from scripts.analysis.quantify_compression_categories import SourceIndex
    source = SourceIndex.from_archive(archive)
    scope = record.get("metric_scope", {})
    source.restrict(exclude_files=scope.get("excluded_files", []), exclude_dirs=scope.get("exclude_dirs", []), include_prefix=scope.get("include_prefix", ""))
    return source


def archive_sources(path):
    with tarfile.open(path) as archive:
        return {m.name.removeprefix("./"): archive.extractfile(m).read().decode("utf-8", "replace")
                for m in archive if m.isfile() and m.name.endswith(".lean")}


def export_pair(job):
    import graph_data
    rid, key, short = job["repo"], job["model"], job["short"]
    folder, installed = Path(job["directory"]), Path(job["installed"])
    work = Path(job["work"])
    pair_dir = work / "data/repos" / rid
    pair_dir.mkdir(parents=True, exist_ok=True)
    for name in ("graph.json.gz", "source.json.gz", "source.bin"):
        target = pair_dir / name
        if not target.exists():
            target.symlink_to(installed / name)
    graph = load_gzip(installed / "graph.json.gz")
    base = baseline(graph, installed)
    record = read(folder / "classification-record.json")
    before = scoped_sources(folder / "before/original-sources.tar.gz", record)
    after = scoped_sources(folder / "after/original-sources.tar.gz", record)
    for name in before.paths():
        require(base["sources"].get(name) == before._files[name], f"{rid}: baseline source differs: {name}")
    submitted = scoped_sources(Path(job["submitted"]), record)
    for name in after.paths():
        require(submitted._files.get(name) == after._files[name], f"{rid}: graph sources differ from submitted endpoint: {name}")
    require(set(after.paths()) == set(submitted.paths()), f"{rid}: submitted source coverage differs")
    require(all(n["name"] in base["member"] for n in read(folder / "before.json")["nodes"]), f"{rid}: baseline declaration mapping differs")
    record = dict(record, protected_declarations=[graph["nodes"]["name"][i] for i in graph["nodes"]["protected"]])
    programs = read(Path(job["program_cache"]) / key / f"{rid}.json")
    # Cached macro measurements only. This export cannot start Lean or write
    # measurements back into the upstream checkout.
    ACCOUNTING.program_sites.measure = lambda *args, **kwargs: programs
    outcome = ACCOUNTING.classifier.classify(rid, key, folder, record, {}, set(), program_cache=Path(job["program_cache"]))
    require(outcome is not None, f"{rid}: classifier produced no result")
    expected = job["canonical"]
    origin = dict(dead_code=outcome["dead_code"], deleted=outcome["detail"]["structural_removals"],
        added=-outcome["detail"]["structural_cost"], proof_simplification=outcome["proof_rewriting"],
        structural_diff=outcome["detail"]["structural_savings"], automation=outcome["automation"], syntax_optimization=outcome["syntax_optimization"])
    require(all(origin[k] == int(float(expected[k])) for k in ORIGIN_KEYS), f"{rid}: classification differs from frozen CSV")
    require(sum(origin.values()) == job["baseline"] - job["post"], f"{rid}: compression categories do not reconcile")
    graph_data.export_model(rid, short, key, folder, base, {"baseline": job["baseline"], "post": job["post"]}, pair_dir, classification=outcome)
    original = ACCOUNTING.classifier.classify
    ACCOUNTING.classifier.classify = lambda *args, **kwargs: copy.deepcopy(outcome)
    try:
        summary = ACCOUNTING.export(dict(repo=rid, model=key, short=short, directory=str(folder), record=str(folder / "classification-record.json"),
            cache=job["program_cache"], cache_model=key, site=str(work), stage=str(work / "data/repos")))
    finally:
        ACCOUNTING.classifier.classify = original
    payload = load_gzip(pair_dir / f"{short}.json.gz")
    require(len(payload["status"]) == len(graph["nodes"]["name"]), f"{rid}: graph node count differs")
    require(sum(f["before"] for f in payload["files"] if f["scored"]) == job["baseline"], f"{rid}: baseline file counts differ")
    require(sum(f["after"] for f in payload["files"] if f["scored"]) == job["post"], f"{rid}: final file counts differ")
    require(payload["outcome"]["scored_tokens_saved"] == job["baseline"] - job["post"], f"{rid}: scored endpoint differs")
    for suffix in (".json.gz", ".diff.json.gz"):
        path = pair_dir / f"{short}{suffix}"
        clean = anonymize(load_gzip(path))
        path.write_bytes(gzip.compress(encoded(clean), compresslevel=9, mtime=0))
    return dict(repo=rid, model=key, short=short, origin=origin, directory=str(pair_dir), accounting_version=2,
                files=len(payload["files"]), added=len(payload["added"]))


def prepare_jobs(spec, bench, lean, archive, data, cache):
    key = spec["key"]
    graph_root = inside(lean, spec["graphs"])
    with (graph_root / "classification-inputs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with (graph_root / "compression-origin/classification-repositories.csv").open() as handle:
        canonical = {r["repository"]: r for r in csv.DictReader(handle)}
    repos = {r["id"]: r for r in bench["repositories"]}
    require(len(rows) == len(repos) and {r["repository"] for r in rows} == set(repos), f"{key}: graph coverage differs")
    jobs = []
    for row in rows:
        rid = row["repository"]
        cell = bench["results"][rid][key]
        require(row["model"] == key and row["status"] == cell["status"], f"{rid}: graph result identity differs")
        if row["status"] != "passed":
            continue
        folder = inside(lean, row["graph_directory"])
        result_path = inside(archive, spec["source"]) / rid / "result.json"
        expected_hash = bench["result_sources"][key]["result_sha256"][rid]
        require(sha(result_path) == expected_hash, f"{rid}: archived result changed since publication")
        row_result = read(result_path)
        before, after = int(row["baseline_tokens"]), int(row["post_tokens"])
        require(row_result["status"] == "passed" and before == repos[rid]["tokens"] and after == cell["post"] == row_result["metrics"]["post_lean_tokens"], f"{rid}: graph endpoint mismatch")
        run = inside(lean, spec["run"]) / rid
        playback = inside(run, row_result["files"]["postprocessed_playback"])
        points = read(playback)["points"]
        selected = next(p for p in reversed(points) if p.get("endpoint_role") == "submitted_patch")
        submitted = inside(run, selected["snapshot_path"])
        program_cache = inside(lean, row["program_cache"])
        inputs = [folder / n for n in ("before.json", "after.json", "classification-record.json", "before/original-sources.tar.gz", "after/original-sources.tar.gz")]
        inputs += [graph_root / "classification-inputs.csv", graph_root / "compression-origin/classification-repositories.csv",
                   program_cache / key / f"{rid}.json", submitted, result_path]
        inputs += [data / "repos" / rid / n for n in ("graph.json.gz", "source.json.gz", "source.bin")]
        fixture = lean / "datasets/leanlean_20260914/repos" / rid / "comparator"
        manifest = read(fixture / "manifest.json")
        inputs += [fixture / "manifest.json", fixture / manifest["challenge"]["path"], fixture / manifest["configuration"]["path"]]
        jobs.append(dict(repo=rid, model=key, short=spec["short"], directory=str(folder), installed=str(data / "repos" / rid),
            work=str(cache / "pairs" / key / rid), submitted=str(submitted), program_cache=str(program_cache),
            baseline=before, post=after, canonical=canonical[rid], inputs=inputs))
    return jobs


def input_hashes(job, dependencies):
    return {str(p): sha(p) for p in sorted(set(job["inputs"]) | dependencies)}


def sync(args):
    data, cache, lean = args.data.resolve(), args.cache.resolve() / "graphs", args.leanlean_root.resolve()
    specs = [s for s in read(args.manifest)["models"] if s.get("graphs") and (not args.models or s["key"] in args.models)]
    if not specs:
        return 0
    cache.mkdir(parents=True, exist_ok=True)
    bench, board = read(data / "benchmark.json"), read(data / "leaderboard.json")
    models = {m["key"]: m for m in bench["models"]}
    missing = [s["key"] for s in specs if s["key"] not in models]
    if missing:
        require(args.dry_run or args.check, f"Sync model results before graphs: {missing}")
        print(f"Graph export awaits model result sync: {missing}")
        return int(args.check)
    init_worker(lean)
    modules = {Path(module.__file__).resolve() for name, module in sys.modules.items()
               if name.startswith(("scripts.analysis.", "leanlean.")) and getattr(module, "__file__", None)}
    dependencies = modules | {Path(__file__).resolve(), Path(__file__).with_name("graph_data.py"), Path(__file__).with_name("sync_results.py"),
        Path(__file__).with_name("anonymize_paths.py"), args.manifest.resolve(), lean / "scripts/analysis/refresh_website_accounting.py"}
    jobs = [job for spec in specs for job in prepare_jobs(spec, bench, lean, args.archive_root.resolve(), data, cache)]
    jobs = [j for j in jobs if not getattr(args, "only", None) or j["repo"] == args.only]
    state_path = cache / "state.json"
    state = read(state_path) if state_path.is_file() else {}
    fingerprints, pending, summaries = {}, [], []
    for job in jobs:
        identity = job["model"] + "/" + job["repo"]
        fingerprints[identity] = input_hashes(job, dependencies)
        prior = state.get(identity, {})
        outputs = [Path(job["work"]) / "data/repos" / job["repo"] / f"{job['short']}{suffix}" for suffix in (".json.gz", ".diff.json.gz")]
        cached = not args.force and prior.get("inputs") == fingerprints[identity] and all(p.is_file() and sha(p) == prior.get("outputs", {}).get(p.name) for p in outputs)
        if cached:
            summaries.append(prior["summary"])
        else:
            pending.append(job)
    print(f"Graphs: {len(jobs)} verified passing pairs; {len(pending)} to export", flush=True)
    if args.dry_run:
        return 0
    if pending and not args.check:
        errors = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(lean,)) as pool:
            futures = {pool.submit(export_pair, job): job for job in sorted(pending, key=lambda j: -j["baseline"])}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    summary = future.result()
                except Exception as error:
                    errors.append(f"{job['repo']}: {error!r}")
                    print(f"Graph export failed: {errors[-1]}", flush=True)
                    continue
                identity = job["model"] + "/" + job["repo"]
                outputs = [Path(summary["directory"]) / f"{job['short']}{suffix}" for suffix in (".json.gz", ".diff.json.gz")]
                state[identity] = dict(inputs=fingerprints[identity], outputs={p.name: sha(p) for p in outputs}, summary=summary)
                temp = state_path.with_suffix(".tmp")
                temp.write_bytes(encoded(state))
                os.replace(temp, state_path)
                summaries.append(summary)
                print(f"Graphs: {len(summaries)}/{len(jobs)} {job['repo']}", flush=True)
        require(not errors, f"{len(errors)} graph pairs failed; public files unchanged: {errors}")
    if args.check and pending:
        return 1
    for job in jobs:
        identity = job["model"] + "/" + job["repo"]
        require(input_hashes(job, dependencies) == fingerprints[identity], "Graph inputs changed during export; rerun sync")
    from sync_results import origin_summary
    with tempfile.TemporaryDirectory(prefix="publish-", dir=cache) as temp:
        stage = Path(temp)
        for summary in summaries:
            rid, key, short = summary["repo"], summary["model"], summary["short"]
            bench["results"][rid][key].update(explorer=True, origin=summary["origin"])
            board["results"][rid][key]["explorer"] = True
            target = stage / "repos" / rid
            target.mkdir(parents=True, exist_ok=True)
            for suffix in (".json.gz", ".diff.json.gz"):
                shutil.copyfile(Path(summary["directory"]) / f"{short}{suffix}", target / f"{short}{suffix}")
        for spec in specs:
            models[spec["key"]]["origin"] = origin_summary(bench, spec["key"])
        board["models"] = bench["models"]
        for name, value in (("benchmark", bench), ("leaderboard", board)):
            (stage / f"{name}.json").write_bytes(encoded(value))
        changed = [p for p in stage.rglob("*") if p.is_file() and (not (data / p.relative_to(stage)).is_file() or p.read_bytes() != (data / p.relative_to(stage)).read_bytes())]
        if args.check:
            return int(bool(changed))
        count = publish(stage, data)
    print(f"Graphs synced: {len(summaries)} pairs, {count} public files changed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("sync_models.json"))
    parser.add_argument("--leanlean-root", type=Path, default=Path(os.environ.get("LEANLEAN_ROOT", Path.home() / "lean-lean")))
    parser.add_argument("--archive-root", type=Path, default=Path(os.environ.get("LEANLEAN_RESULTS_ROOT", Path.home() / "lean-lean-results")) / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "site/data")
    parser.add_argument("--cache", type=Path, default=ROOT / "hidden/data/sync")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", help="validate/export one repository")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    require(args.workers > 0, "--workers must be positive")
    args.cache.mkdir(parents=True, exist_ok=True)
    with (args.cache / "sync.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
