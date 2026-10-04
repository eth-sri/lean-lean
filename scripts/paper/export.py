#!/usr/bin/env python3
"""Export the paper's CSVs from your own evaluation and postprocessing outputs.

The config (configs/paper/leanlean_20260914.yaml) names the dataset and, for each
model, the run directories ``output/evaluation/<dataset>/<provider>/<model>``
that eval.sh wrote and postprocessing.py verified, measured and replayed. It
also names the outputs of classify_compression.py. Nothing is run: this only
reads files and recomputes the paper's aggregates.

    python scripts/paper/export.py configs/paper/leanlean_20260914.yaml --output-dir output/paper-export
    python scripts/paper/render.py --data-dir output/paper-export/data --output-dir output/paper-from-runs

Rules are the paper's. A repository's verdict is its report_summary.json entry,
else its newest postprocessed playback; it must belong to the patch in
preds.json. Failed submissions score 0 and unresolved ones are left out, never
zeroed. Later run directories of a model override earlier ones per repository.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

BAND_ORDER = ("≤ 10k", "10k–50k", "50k–250k", "> 250k")
BAND_LABELS = dict(zip(BAND_ORDER, ("Compact", "Standard", "Large", "Massive")))
FAMILIES = {"Algebra & number theory": "#2B6F9C", "Geometry & topology": "#58A98B",
            "Analysis & dynamics": "#D99533", "Combinatorics & discrete math": "#9B77A8",
            "Logic & foundations": "#C75D63", "Applied & computational math": "#4E8D9D",
            "Probability & statistics": "#86A85F", "Computer science": "#D17C4B", "Physics": "#6C83B5"}
CATEGORIES = ("dead_code", "syntax_optimization", "automation", "proof_simplification", "structural_diff",
              "deleted", "added", "automation_simplification", "automation_structural")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def skipped_verification(record: dict) -> bool:
    """A checkpoint replay that built the endpoint but explicitly skipped its verification."""
    state = record.get("selected_state") or {}
    final = next((p for p in reversed(record.get("points", [])) if p.get("kind") == "final"), {})
    verification = final.get("lean_verify") or {}
    return (state.get("policy") == "submitted_final_state" and state.get("eligible") is False
            and state.get("build_passed") is True and state.get("lean_verify_passed") is False
            and verification.get("checked") is False and verification.get("skipped") == "checkpoint_lean_verify_disabled")


def score(record: dict) -> tuple[float | None, bool | None]:
    """None for an unresolved verdict, 0 for a failed submission, else the token reduction."""
    if skipped_verification(record):
        return None, None
    state = record.get("selected_state") or {}
    eligible, built, signatures = (state.get(k) for k in ("eligible", "build_passed", "signatures_preserved"))
    endpoint = eligible is True and built is True and signatures is None and state.get("lean_verify_passed") is True
    failure = isinstance(eligible, bool) and isinstance(built, bool) and (not eligible or not built)
    if not (all(isinstance(v, bool) for v in (eligible, built, signatures)) or endpoint or failure):
        return None, None
    if not (endpoint or (eligible is True and built is True and signatures is True)):
        return 0.0, False
    before, after = record.get("baseline_lean_tokens"), record.get("post_lean_tokens")
    if not (finite(before) and before > 0 and finite(after) and after >= 0):
        return None, None
    return 100 * (1 - after / before), True


# --- One model's selected endpoints ------------------------------------------------

def patch_hash(predictions: dict, repository: str) -> str | None:
    entry = predictions.get(repository) or {}
    return hashlib.sha256(entry["model_patch"].encode()).hexdigest() if "model_patch" in entry else None


@lru_cache(maxsize=None)
def report(run: Path) -> dict:
    return read_json(run / "report_summary.json").get("instances", {})


def verdict(run: Path, repository: str, expected: str | None) -> tuple[dict, dict, str]:
    """(record, playback, source): the report's verdict, else the newest matching playback's."""
    playback, source = {}, ""
    captures = sorted((run / repository / "playback").glob("capture_*/postprocessed-playback.json"),
                      key=lambda p: int(p.parent.name.split("_")[-1]))
    for path in captures:
        value = read_json(path)
        hashes = {p.get("submission_patch_sha256") for p in value.get("points", [])
                  if p.get("kind") == "final" and p.get("submission_patch_sha256")}
        if expected and hashes and hashes != {expected}:
            continue
        playback, source = value, str(path)
    latest = (report(run).get(repository) or {}).get("latest") or {}
    metrics = latest.get("metrics") or {}
    record = dict(selected_state=dict(eligible=latest.get("resolved"), build_passed=latest.get("build_passed"),
                                      signatures_preserved=latest.get("signatures_preserved"),
                                      lean_verify_passed=latest.get("lean_verify_passed")),
                  baseline_lean_tokens=metrics.get("baseline_lean_tokens"),
                  post_lean_tokens=metrics.get("post_lean_tokens"), final_cost_usd=latest.get("cost_total"))
    if score(record)[0] is not None:
        return record, playback, str(run / "report_summary.json")
    return playback, playback, source


def native_claude_cost(directory: Path, recorded) -> tuple[float | None, bool]:
    """Claude Code's own list-price total, and whether the native stream proves it.

    Resumed sessions repeat a cumulative total, so the last one counts. Without a
    terminal total, the recorded cost is complete when every message's usage at
    list prices adds up to it.
    """
    streams = sorted(directory.glob("playback/capture_*/native.stdout.jsonl"),
                     key=lambda p: int(p.parent.name.split("_")[-1]))
    if not streams:
        return recorded, False
    total = None
    for line in streams[-1].open():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("modelUsage") or {}
        if (event.get("type") == "result" and finite(event.get("total_cost_usd")) and usage
                and all(m.get("costBasis") == "list" for m in usage.values())):
            total = event["total_cost_usd"]
    if total is not None:
        return total, True
    if not finite(recorded):
        return recorded, False
    from scripts.analysis.reconstruct_native_claude_checkpoint_costs import TOLERANCE_USD, native_ledger
    ledger = native_ledger(streams[-1])
    return recorded, bool(ledger["message_count"]) and abs(ledger["total"] - recorded) <= TOLERANCE_USD


def usage_cost(trajectory: Path, pricing: dict) -> float | None:
    """Reprice every recorded response at fixed rates; cached input at its own rate if given."""
    cached_rate = pricing.get("cached_input_per_million_usd")
    tokens, observed = [0, 0, 0], False
    for attempt in read_json(trajectory).get("info", {}).get("attempts", []):
        for response in attempt.get("responses", []):
            usage = response.get("usage")
            if not isinstance(usage, dict):
                continue
            observed = True
            prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) if cached_rate is not None else 0
            tokens[0] += prompt - cached
            tokens[1] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            tokens[2] += cached
    if not observed:
        return None
    return (tokens[0] * pricing["input_per_million_usd"] + tokens[1] * pricing["output_per_million_usd"]
            + tokens[2] * (cached_rate or 0)) / 1_000_000


def endpoints(key: str, spec: dict, repositories: dict, *, recorded_cost: bool = False) -> list[dict]:
    """One row per dataset repository, from the last run directory that generated it."""
    runs = [(Path(run), read_json(Path(run) / "preds.json")) for run in spec["runs"]]
    rows = []
    for rid, repo in repositories.items():
        run, predictions = next(((r, p) for r, p in reversed(runs) if rid in p), runs[-1])
        expected = patch_hash(predictions, rid)
        record, playback, source = verdict(run, rid, expected)
        # A run re-postprocessed from an earlier one keeps the earlier verdict of the same patch.
        for older, older_predictions in reversed(runs):
            if score(record)[0] is None and older != run and patch_hash(older_predictions, rid) == expected:
                record, playback, source = verdict(older, rid, expected)
        value, passed = score(record)
        if value is not None and record.get("baseline_lean_tokens") != repo["stripped_tokens"]:
            value, passed = None, None  # measured against another baseline
        directory = run / rid
        cost = record.get("final_cost_usd")
        quality = "complete" if playback.get("final_cost_complete") is True else "unverified"
        if not recorded_cost and spec.get("cost") == "native_claude":
            cost, proven = native_claude_cost(directory, cost)
            quality = "complete" if proven else quality
        if not recorded_cost and spec.get("usage_pricing"):
            cost, quality = usage_cost(directory / f"{rid}.traj.json", spec["usage_pricing"]), "estimated"
        seconds = read_json(directory / "round_0_gen_metrics.json").get("wall_time_seconds")
        if not finite(seconds) and all(playback.get(k) == record.get(k) for k in ("baseline_lean_tokens", "post_lean_tokens")):
            seconds = playback.get("wall_time_seconds")
        rows.append(dict(model=key, repository=rid, token_band=repo["token_band"], generated=rid in predictions,
                         status="missing" if value is None else "passed" if passed else "failed",
                         score_pct=value, baseline_tokens=record.get("baseline_lean_tokens"),
                         post_tokens=record.get("post_lean_tokens"),
                         cost_usd=cost if finite(cost) and cost >= 0 else None, cost_quality=quality,
                         seconds=seconds if finite(seconds) and seconds > 0 else None,
                         selected_patch_sha256=expected, source=source, directory=str(directory)))
    return rows


# --- Dataset tables --------------------------------------------------------------

def dataset_tables(config: dict, issues: list) -> tuple[dict, dict]:
    from scripts.analysis.quantify_compression_categories import SourceIndex
    from leanlean.metrics.tokens import _IMPORT_COMMAND_RE, count_lean_tokens_in_source, remove_lean_comments
    dataset = Path(config["dataset"])
    family = {r["repository"]: r for r in read_csv(Path(config["subjects"]))}
    repositories, counts = {}, []
    for line in (dataset / "metadata.jsonl").read_text().splitlines():
        row = json.loads(line)
        rid = row["id"]
        strip = read_json(dataset / "repos" / rid / "prod-strip-report.json")
        tokens = strip["published_repository"]["metrics"]["lean_tokens"]
        if row["stripped_tokens"] != tokens["stripped"]:
            raise ValueError(f"Catalog and report disagree: {rid}")
        # Lines of code: the scored stripped sources without blank, comment or import lines.
        index = SourceIndex.from_directory(dataset / "repos" / rid / "stripped")
        scope, provenance = strip["published_repository"]["scope"], strip["protected_provenance"]
        index.restrict(exclude_dirs=scope.get("exclude_dirs", []), include_prefix=scope.get("target_dir", ""),
                       exclude_files=[provenance["registered_challenge_path"],
                                      provenance["registered_challenge_module"].replace(".", "/") + ".lean"])
        total, loc = 0, 0
        for text in (index._files[path] for path in index.paths()):  # paths() leaves out lakefile.lean
            total += count_lean_tokens_in_source(text)
            masked = remove_lean_comments(text, mask_strings=True).splitlines()
            loc += sum(bool(a.strip()) and not _IMPORT_COMMAND_RE.match(b)
                       for a, b in zip(remove_lean_comments(text).splitlines(), masked, strict=True))
        if total != tokens["stripped"]:
            issues.append(f"preprocessing-scale: {rid} sources count {total} tokens, not {tokens['stripped']}")
            loc = None
        repositories[rid] = dict(repository=rid, token_band=row["token_band"], raw_tokens=tokens["raw"],
                                 stripped_tokens=tokens["stripped"], core_theorems=row["protected_declarations"],
                                 loc=loc, primary_arxiv_tag=family[rid]["primary_arxiv"])
        counts.append(strip["dependency_graph"]["counts"])
    repos = list(repositories.values())
    scale = []
    for band in [*BAND_ORDER, None]:
        rs = [r for r in repos if band is None or r["token_band"] == band]
        locs = [r["loc"] for r in rs if r["loc"] is not None]
        complete = len(locs) == len(rs)
        scale.append(dict(scale=BAND_LABELS.get(band, "All"), token_band=band or "All", repositories=len(rs),
                          core_theorems_median=median(r["core_theorems"] for r in rs),
                          core_theorems_max=max(r["core_theorems"] for r in rs),
                          loc_median=median(locs) if complete else None, loc_max=max(locs) if complete else None,
                          lean_tokens_median=median(r["stripped_tokens"] for r in rs),
                          lean_tokens_max=max(r["stripped_tokens"] for r in rs), loc_repositories=len(locs)))
    union = sum(c["edges"] for c in counts)
    layers = [("Kernel (.olean)", "kernel_edges", "kernel_only_edges"), ("Source", "source_text_edges", "source_text_only_edges"),
              (".ilean", "ilean_edges", "ilean_only_edges"), ("Found by all three", "all_layers_edges", None),
              ("Union", "edges", None)]
    families = Counter(family[r["repository"]]["family"] for r in repos)
    tables = {
        "benchmark-repositories": repos,
        "benchmark-preprocessing-removal": [dict(repository=r["repository"], token_band=r["token_band"],
                                                 raw_tokens=r["raw_tokens"], stripped_tokens=r["stripped_tokens"],
                                                 reduction_pct=100 * (1 - r["stripped_tokens"] / r["raw_tokens"]))
                                            for r in repos],
        "preprocessing-scale": scale,
        "dependency-layers": [dict(layer=name, edges=sum(c[k] for c in counts),
                                   coverage_pct=100 * sum(c[k] for c in counts) / union,
                                   unique_edges=sum(c[u] for c in counts) if u else None,
                                   unique_share_pct=100 * sum(c[u] for c in counts) / union if u else None,
                                   repositories=len(counts)) for name, k, u in layers],
        "benchmark-primary-subject-families": [dict(family=f, repositories=families[f], color=c)
                                               for f, c in FAMILIES.items() if families[f]],
        "repository-subjects": [family[r["repository"]] for r in repos],
    }
    return repositories, tables


# --- Benchmark tables ------------------------------------------------------------------

def benchmark_tables(config: dict, repositories: dict, details: list[dict], issues: list) -> dict:
    models = config["models"]
    figure_models = [k for k, s in models.items() if not s.get("leaderboard_only")]
    summary = {}
    for key, spec in models.items():
        rows = [d for d in details if d["model"] == key]
        scored = [d for d in rows if d["status"] != "missing"]
        costs = [d["cost_usd"] for d in scored if d["cost_usd"] is not None]
        hours = [d["seconds"] / 3600 for d in scored if d["seconds"] is not None]
        # Python's mean of seconds, then hours, as the paper's export computed it.
        seconds = [d["seconds"] for d in scored if d["seconds"] is not None]
        summary[key] = dict(model=key, label=spec["label"], compression_pct=mean(d["score_pct"] for d in scored) if scored else None,
                            cost_per_task_usd=mean(costs) if scored and len(costs) == len(scored) else None,
                            duration_hours=mean(seconds) / 3600 if scored and len(hours) == len(scored) else None,
                            scored=len(scored), expected=len(rows), passed=sum(d["status"] == "passed" for d in rows),
                            generated=sum(d["generated"] for d in rows), cost_repositories=len(costs),
                            duration_repositories=len(hours), cost_note=spec.get("cost_note", ""),
                            unverified_cost_repositories=sum(d["cost_quality"] == "unverified" for d in scored))
        if len(scored) < len(rows):
            issues.append(f"leaderboard: {key}: {len(scored)}/{len(rows)} scored; unresolved results excluded, never zeroed")
    ranked = sorted(summary.values(), key=lambda r: -(r["compression_pct"] if r["compression_pct"] is not None else -math.inf))
    tables = {"leaderboard": [dict(rank=i, **r) for i, r in enumerate(ranked, 1)]}
    tables["introduction-results"] = [dict(model=k, label=s["label"], color=s["color"],
                                           mean_reduction_pct=summary[k]["compression_pct"],
                                           **{f: summary[k][f] for f in ("scored", "expected", "passed", "generated")})
                                      for k, s in models.items()]
    for stem, column, field, scale in (("benchmark-model-performance-by-scale", "mean_reduction_pct", "score_pct", 1),
                                       ("benchmark-model-cost-by-scale", "mean_cost_usd", "cost_usd", 1),
                                       ("benchmark-model-duration-by-scale", "mean_duration_hours", "seconds", 3600)):
        out = []
        for key in figure_models:
            for band in BAND_ORDER:
                cohort = [d for d in details if d["model"] == key and d["token_band"] == band and d["status"] != "missing"]
                values = [d[field] / scale for d in cohort if d[field] is not None]
                out.append(dict(model=key, label=models[key]["label"], color=models[key]["scale_color"], token_band=band,
                                **{column: sum(values) / len(values) if values and len(values) == len(cohort) else None},
                                scored=len(cohort), expected=sum(r["token_band"] == band for r in repositories.values()),
                                generated_total=summary[key]["generated"]))
        tables[stem] = out
    passing = [d for d in details if d["model"] in figure_models and d["status"] == "passed"]
    tables["compression-reduction-histograms-compression-reduction"] = [
        dict(model=d["model"], label=models[d["model"]]["label"], repository=d["repository"], baseline_tokens=d["baseline_tokens"],
             post_tokens=d["post_tokens"], reduction_pct=100 * (1 - d["post_tokens"] / d["baseline_tokens"])) for d in passing]
    tables["compression-task-cost-histogram"] = [
        dict(model=d["model"], label=models[d["model"]]["label"], repository=d["repository"], cost_usd=d["cost_usd"],
             cost_quality=d["cost_quality"]) for d in passing if d["cost_usd"] is not None]
    tables["compression-task-duration-histogram"] = [
        dict(model=d["model"], label=models[d["model"]]["label"], repository=d["repository"], seconds=d["seconds"],
             hours=d["seconds"] / 3600) for d in passing if d["seconds"] is not None]
    tables["unscored-repositories"] = [dict(model=d["model"], repository=d["repository"], generated=d["generated"],
                                            reason="No resolved verification verdict for selected endpoint")
                                       for d in details if d["status"] == "missing"]
    return tables


def heartbeat_tables(config: dict, details: list[dict]) -> dict:
    """Body-heartbeat reduction against the stripped baseline; failed submissions count 0."""
    method, field = "exact_lake_setup_sync_v2", "body_heartbeats"
    baselines = {}
    for path in config["heartbeats"]["baselines"]:
        for r in read_json(Path(path)).get("repositories", []):
            b = r.get("stripped") or {}
            if b.get("status") == "complete" and b.get("method") == method:
                baselines[r["repository"]] = b
    rows, summaries = [], []
    for key, spec in config["models"].items():
        values = []
        for d in (d for d in details if d["model"] == key):
            rid, b, value, status = d["repository"], baselines.get(d["repository"], {}), None, "pending_measurement"
            if d["status"] == "missing":
                status = "unscored_submission"
            elif not (b.get("lean_tokens") == d["baseline_tokens"] and finite(b.get(field)) and b[field] > 0):
                status = "missing_compatible_baseline"
            elif d["status"] == "failed":
                value, status = 0.0, "failed_submission_zero"
            else:
                # Later directories take precedence (gap-fill or corrected measurements).
                for directory in spec.get("heartbeats", spec["runs"]):
                    for name in ("heartbeats.json", "optimized.json"):
                        v = read_json(Path(directory) / rid / name)
                        if (v.get("status") == "complete" and v.get("method") == method
                                and v.get("lean_tokens") == d["post_tokens"] and finite(v.get(field)) and v[field] >= 0):
                            value, status = 100 * (1 - v[field] / b[field]), "measured"
            if value is not None:
                values.append(value)
            rows.append(dict(model=key, repository=rid, status=status, reduction_pct=value, field=field))
        summaries.append(dict(model=key, label=spec["label"], average_reduction_pct=mean(values) if values else None,
                              improved_repositories=sum(x > 0 for x in values),
                              improved_repositories_pct=100 * sum(x > 0 for x in values) / len(values) if values else None,
                              scored=len(values), expected=sum(d["model"] == key for d in details),
                              submission_scored=sum(d["model"] == key and d["status"] != "missing" for d in details),
                              field=field))
    summaries.sort(key=lambda r: -(r["average_reduction_pct"] if r["average_reduction_pct"] is not None else -math.inf))
    return {"heartbeat-leaderboard": [dict(rank=i, **r) for i, r in enumerate(summaries, 1)], "heartbeat-repositories": rows}


def union_tables(config: dict, repositories: dict, details: list[dict]) -> dict:
    """The merge arms and their Opus/Sol controls, all against the original stripped baseline."""
    spec = config["union"]
    ids = list(spec["repositories"])
    records = []
    for method in spec["controls"]:
        for d in (d for d in details if d["model"] == method and d["repository"] in ids):
            records.append(dict(method=method, repository=d["repository"], score_pct=d["score_pct"],
                                passed=None if d["status"] == "missing" else d["status"] == "passed",
                                original_tokens=repositories[d["repository"]]["stripped_tokens"],
                                starting_tokens=d["baseline_tokens"], post_tokens=d["post_tokens"]))
    for method, arm in spec["arms"].items():
        run = Path(arm["runs"][-1])
        predictions = read_json(run / "preds.json")
        for rid in ids:
            record, _, _ = verdict(run, rid, patch_hash(predictions, rid))
            value, passed = score(record)
            original = repositories[rid]["stripped_tokens"]
            if value is not None:
                value = 100 * (1 - record["post_lean_tokens"] / original) if passed else 0.0
            records.append(dict(method=method, repository=rid, score_pct=value, passed=passed, original_tokens=original,
                                starting_tokens=record.get("baseline_lean_tokens"), post_tokens=record.get("post_lean_tokens")))
    table = []
    for method, label in spec["labels"].items():
        values = [r["score_pct"] for r in records if r["method"] == method and r["score_pct"] is not None]
        table.append(dict(method=method, label=label, compression_pct=mean(values) if len(values) == len(ids) else None,
                          scored=len(values), expected=len(ids)))
    return {"union-ablation": table, "union-repositories": records}


def benchmark_extra_tables(config: dict, repositories: dict, details: list[dict]) -> dict:
    """Fable, Astra, Opus and Sol on the twelve repositories with a valid submission from each."""
    spec = config["benchmark_extra"]
    ids = list(spec["repositories"])
    subset = {rid: repositories[rid] for rid in ids}
    rows = []
    for key, model in spec["models"].items():
        if model.get("runs"):
            selected = endpoints(key, model, subset, recorded_cost=True)
        else:  # a benchmark model: its selected endpoints, at their recorded cost
            selected = endpoints(key, config["models"][key], subset, recorded_cost=True)
        for d in selected:
            if d["status"] != "passed":
                raise ValueError(f"benchmark-extra: {key}/{d['repository']} has no valid submission")
            rows.append(dict(model=key, label=model["label"], repository=d["repository"], status=d["status"],
                             baseline_tokens=d["baseline_tokens"], post_tokens=d["post_tokens"], score_pct=d["score_pct"],
                             selected_patch_sha256=d["selected_patch_sha256"], cost_usd=d["cost_usd"], seconds=d["seconds"]))
    board = []
    for key, model in spec["models"].items():
        cohort = [r for r in rows if r["model"] == key]
        costs = [r["cost_usd"] for r in cohort if r["cost_usd"] is not None]
        seconds = [r["seconds"] for r in cohort if r["seconds"] is not None]
        board.append(dict(model=key, label=model["label"], repositories=len(cohort),
                          compression_pct=mean(r["score_pct"] for r in cohort),
                          cost_per_task_usd=mean(costs) if len(costs) == len(cohort) else None,
                          duration_hours=mean(seconds) / 3600 if len(seconds) == len(cohort) else None,
                          cost_repositories=len(costs), duration_repositories=len(seconds)))
    board.sort(key=lambda r: (-r["compression_pct"], r["model"]))
    return {"benchmark-extra-leaderboard": [dict(rank=i, **r) for i, r in enumerate(board, 1)],
            "benchmark-extra-leaderboard-repositories": rows}


# --- Analyses on top of other public tools ---------------------------------------------

def macro(rows: list[dict]) -> dict:
    """Equal-weight means over repositories, as classify_compression.py aggregates them."""
    value = {k: sum(100 * float(r[k]) / (int(r["baseline_tokens"]) or 1) for r in rows) / len(rows) for k in CATEGORIES}
    value["saved"] = sum(100 * int(r["scored_tokens_saved"]) / (int(r["baseline_tokens"]) or 1) for r in rows) / len(rows)
    value["repositories"] = len(rows)
    value["proof_rewrites"] = value["automation"] + value["proof_simplification"] + value["structural_diff"]
    return value


def classification_tables(config: dict) -> dict:
    """classify_compression.py's outputs, and the prompt arms' means over their repositories."""
    directory = Path(config["classification"])
    tables = {name: read_csv(directory / f"{name}.csv") for name in ("classification-repositories", "diff_classification_macro")}
    spec = config.get("prompt_ablation")
    if spec:
        ids, arms, repos = set(spec["repositories"]), [], []
        for arm, entry in spec["arms"].items():
            rows = [r for r in read_csv(Path(entry["classification"]) / "classification-repositories.csv")
                    if r["repository"] in ids and r["model"] == entry.get("model", r["model"])]
            if len(rows) != len(ids):
                raise ValueError(f"prompt ablation: {arm} has {len(rows)}/{len(ids)} classified repositories")
            arms.append(dict(model=arm, label=entry["label"], **macro(rows)))
            repos += [dict(r, arm=arm) for r in rows]
        tables["opus-prompt-ablation"] = arms
        tables["opus-prompt-ablation-classification-repositories"] = repos
    return tables


def action_tables(config: dict, details: list[dict]) -> dict:
    """Ten-class action counts of every resolved selected trajectory (plot_action_classes.py)."""
    from scripts.analysis.plot_action_classes import CATEGORIES as CLASSES, TABLE_ORDER, actions, classify, starting_repository
    models = {k: s for k, s in config["models"].items() if not s.get("leaderboard_only")}
    rows = []
    for d in details:
        if d["model"] not in models or d["status"] == "missing":
            continue
        repo = starting_repository(Path(config["dataset"]), d["repository"])
        counts = Counter()
        for _, _, name, args in actions(Path(d["directory"])):
            category, _ = classify(name, args, repo)
            if category is not None:
                counts[category] += 1
        if counts:
            rows.append(dict(model=d["model"], repository=d["repository"], selected_patch_sha256=d["selected_patch_sha256"],
                             Actions=sum(counts.values()), **{c: counts[c] for c in CLASSES}))
    summary = []
    for key, spec in models.items():
        cohort = [r for r in rows if r["model"] == key]
        if cohort:
            summary.append(dict(model=key, label=spec["label"], trajectories=len(cohort),
                                **{c: mean(r[c] for r in cohort) for c in TABLE_ORDER}, total=mean(r["Actions"] for r in cohort)))
    return {"action-repositories": rows, "agent-actions": summary}


# --- Budget curves from the five-minute replay -------------------------------------------

def replay_points(config: dict, details: list[dict]) -> list[dict]:
    """Every replayed checkpoint build of each selected submission (postprocessing.py --replay)."""
    points = []
    for key, spec in config["models"].items():
        short = spec.get("budget")
        if not short:
            continue
        # The checkpoint index given to postprocessing.py --replay (select_checkpoint_windows.py).
        index = yaml.safe_load(Path(spec["checkpoints"]).read_text())["repositories"] if spec.get("checkpoints") else None
        for d in (d for d in details if d["model"] == key):
            selected = None if index is None else set(index.get(d["repository"], ()))
            replays = [Path(r) / d["repository"] for r in spec.get("replay", [])] or [Path(d["directory"])]
            for path in sorted(p for r in replays for p in r.glob("playback/capture_*/replayed-playback.json")):
                data = read_json(path)
                hashes = {p.get("submission_patch_sha256") for p in data.get("points", [])
                          if p.get("kind") == "final" and p.get("endpoint_role") == "submitted_patch"}
                if hashes != {d["selected_patch_sha256"]}:
                    continue
                observations = sorted((r["elapsed_seconds"], r["cumulative_cost_usd"])
                                      for r in data.get("checkpoint_observations", [])
                                      if r.get("elapsed_seconds") is not None and r.get("cumulative_cost_usd") is not None)
                seen = set()
                for p in data["points"]:
                    build = p.get("build")
                    if (p.get("kind") == "final" or p["edit_index"] in seen or not isinstance(build, dict)
                            or not isinstance(build.get("passed"), bool)
                            or (selected is not None and p["edit_index"] not in selected)):
                        continue
                    seen.add(p["edit_index"])
                    cost = p.get("cost_usd")
                    if spec.get("usage_pricing"):
                        cost = sum(usage_cost_tokens(u, spec["usage_pricing"]) for u in (p.get("usage_by_model") or {}).values()) \
                            if p.get("usage_by_model") else None
                    row = dict(model=short, canonical_model=key, repository=d["repository"],
                               selected_patch_sha256=d["selected_patch_sha256"], edit_index=p["edit_index"],
                               elapsed_minutes=p["elapsed_seconds"] / 60, cost_usd=cost if finite(cost) and cost >= 0 else None,
                               build_passed=build["passed"], compression_pct=p["lean_token_compression_pct"])
                    if short == "gemini":
                        # Gemini's cost is only known between two native observations.
                        i = bisect.bisect_right([t for t, _ in observations], p["elapsed_seconds"])
                        if i == 0 or i == len(observations):
                            raise ValueError(f"Unbracketed Gemini checkpoint cost: {d['repository']}/{p['edit_index']}")
                        row.update(cost_usd=observations[i - 1][1], cost_lower_usd=observations[i - 1][1],
                                   cost_upper_usd=observations[i][1])
                    points.append(row)
    return points


def usage_cost_tokens(usage: dict, pricing: dict) -> float:
    cached_rate = pricing.get("cached_input_per_million_usd")
    prompt, output = int(usage.get("prompt_tokens") or 0), int(usage.get("output_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0) if cached_rate is not None else 0
    return ((prompt - cached) * pricing["input_per_million_usd"] + output * pricing["output_per_million_usd"]
            + cached * (cached_rate or 0)) / 1_000_000


def latest_passing(points: list[dict], budget: float, axis: str) -> float:
    passing = sorted((p for p in points if p["build_passed"] and p[axis] is not None),
                     key=lambda p: (float(p[axis]), int(p["edit_index"])))
    index = bisect.bisect_right([float(p[axis]) for p in passing], budget) - 1
    return float(passing[index]["compression_pct"]) if index >= 0 else 0.0


def budget_table(config: dict, details: list[dict], points: list[dict]) -> list[dict]:
    """Mean score of the latest passing checkpoint within each time and dollar budget.

    A submission whose patch is empty has no checkpoints and scores 0 throughout.
    Time uses a five-minute grid; cost is evaluated at every passing checkpoint's
    cost, so its steps are exact. Gemini's bounds take every feasible cost.
    """
    order = ("gemini", "luna", "muse", "sol", "opus")
    keys = {s["budget"]: k for k, s in config["models"].items() if s.get("budget")}
    grouped = defaultdict(list)
    for p in points:
        grouped[p["model"], p["repository"]].append(p)
    empty = hashlib.sha256(b"").hexdigest()
    time_cohorts, cost_cohorts = {}, {}
    for model in order:
        replayed = sorted(r for m, r in grouped if m == model)
        zeros = []
        for d in (d for d in details if d["model"] == keys[model] and (model, d["repository"]) not in grouped):
            if d["selected_patch_sha256"] not in (None, empty):
                raise ValueError(f"Missing replay checkpoints for a nonempty patch: {model}/{d['repository']}")
            zeros.append(d["repository"])
        time_cohorts[model] = sorted(replayed + zeros)
        cost_cohorts[model] = sorted([r for r in replayed if any(p["cost_usd"] is not None for p in grouped[model, r])] + zeros)
    endpoint = math.ceil(max(float(p["cost_usd"]) for p in points if p["cost_usd"] is not None))
    rows = []
    for model in order:
        breakpoints = {0.0, float(endpoint)}
        for p in points:
            if p["model"] == model and p["build_passed"] and p["cost_usd"] is not None:
                breakpoints |= {float(p[k]) for k in ("cost_usd", "cost_lower_usd", "cost_upper_usd") if p.get(k) is not None}
        passing = {r: sorted((p for p in grouped[model, r] if p["build_passed"]), key=lambda p: int(p["edit_index"]))
                   for r in cost_cohorts[model]}
        for budget in sorted(breakpoints):
            ids = cost_cohorts[model]
            row = dict(axis="cost_usd", model=model, budget=budget,
                       score_pct=sum(latest_passing(grouped[model, r], budget, "cost_usd") for r in ids) / len(ids),
                       cohort_size=len(ids), lower_pct=None, upper_pct=None)
            if model == "gemini":
                low, high = [], []
                for r in ids:
                    ps = passing[r]
                    forced = max((i for i, p in enumerate(ps) if p["cost_upper_usd"] <= budget), default=-1)
                    feasible = [float(p["compression_pct"]) for i, p in enumerate(ps) if i >= forced and p["cost_lower_usd"] <= budget]
                    feasible += [0.0] if forced < 0 else []
                    low.append(min(feasible))
                    high.append(max(feasible))
                row.update(lower_pct=sum(low) / len(low), upper_pct=sum(high) / len(high))
            rows.append(row)
    last = math.ceil(max(p["elapsed_minutes"] for p in points) / 5) * 5
    for model in order:
        ids = time_cohorts[model]
        for budget in range(0, last + 1, 5):
            rows.append(dict(axis="elapsed_minutes", model=model, budget=budget,
                             score_pct=sum(latest_passing(grouped[model, r], budget, "elapsed_minutes") for r in ids) / len(ids),
                             cohort_size=len(ids), lower_pct=None, upper_pct=None))
    return rows


# --- Command line ------------------------------------------------------------------------

def export(config: dict, output: Path) -> list[str]:
    issues: list[str] = []
    repositories, tables = dataset_tables(config, issues)
    details = [d for key, spec in config["models"].items() for d in endpoints(key, spec, repositories)]
    tables["benchmark-repository-results"] = [{k: v for k, v in d.items() if k != "directory"} for d in details]
    tables.update(benchmark_tables(config, repositories, details, issues))
    if config.get("heartbeats"):
        tables.update(heartbeat_tables(config, details))
    if config.get("union"):
        tables.update(union_tables(config, repositories, details))
    if config.get("benchmark_extra"):
        tables.update(benchmark_extra_tables(config, repositories, details))
    if config.get("classification"):
        tables.update(classification_tables(config))
    if config.get("actions", True):
        tables.update(action_tables(config, details))
    if any(spec.get("budget") for spec in config["models"].values()):
        points = replay_points(config, details)
        tables["replay-checkpoint-builds"] = points
        tables["budget_log_pair"] = budget_table(config, details, points)
    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    pinned = {}
    for name, rows in tables.items():
        if rows:
            write_csv(data / f"{name}.csv", rows)
            pinned[f"{name}.csv"] = dict(rows=len(rows), sha256=hashlib.sha256((data / f"{name}.csv").read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(dict(issues=issues, tables=pinned), indent=2) + "\n")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/paper-export")
    args = parser.parse_args()
    for issue in export(yaml.safe_load(args.config.read_text()), args.output_dir):
        print("NOTE", issue)
    print(f"Wrote {args.output_dir / 'data'}")


if __name__ == "__main__":
    main()
