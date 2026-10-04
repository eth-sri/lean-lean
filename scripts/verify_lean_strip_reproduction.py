#!/usr/bin/env python3
"""Preprocess benchmark repositories with lean-strip and compare them with a published dataset.

Each repository of the configuration goes through the normal preprocessing
(source image, offline lean-strip, Comparator) and is then compared with the
same repository in the reference dataset: every file of the stripped tree, the
dropped declarations and the Lean token counts. An identical repository's output
and stripped image are discarded. A differing one keeps its output; its image is
kept only if Lean sources or dropped declarations differ.

The check runs as its own `<run>-verify` run (datasets/<run>-verify,
runs/preprocessing/<run>-verify, image tags :<run>-verify), so it never touches
a dataset or images the configuration itself produced. One record per
repository and a summary go to runs/preprocessing/<run>-verify/reproduction/.
Rerunning skips repositories that already have a record, except failed ones.

Usage:
    uv run python scripts/fetch_dataset.py    # the published release, as the reference
    uv run python scripts/verify_lean_strip_reproduction.py configs/preprocessing/leanlean_20260914.yaml \
        --reference datasets/leanlean_20260914
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.pipeline.lean_strip_preprocessing import (  # noqa: E402
    _atomic_json,
    check_tool_pin,
    load_config,
    prepare_targets,
    process_repository,
)
from leanlean.preprocessing.lean_strip_tool import resolve_lean_strip_tool  # noqa: E402


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] not in {".git", ".lake"}
    }


def compare(repository_dir: Path, reference_dir: Path) -> dict[str, Any]:
    new, old = _files(repository_dir / "stripped"), _files(reference_dir / "stripped")
    only_new, only_old = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    differs = sorted(path for path in set(new) & set(old) if new[path] != old[path])
    lean = lambda paths: [path for path in paths if path.endswith(".lean")]  # noqa: E731
    report = json.loads((repository_dir / "prod-strip-report.json").read_text())
    published = json.loads((reference_dir / "prod-strip-report.json").read_text())
    engine = json.loads((repository_dir / "lean-strip/state/report.json").read_text())["engine_report"]
    dropped_new, dropped_old = set(engine["dropped_decls"]), set(published.get("dropped_decls", []))
    tokens_new = report["published_repository"]["metrics"]["lean_tokens"]
    tokens_old = published["published_repository"]["metrics"]["lean_tokens"]
    result = {
        "tree_identical": not (only_new or only_old or differs),
        "lean_identical": not (lean(only_new) or lean(only_old) or lean(differs)),
        "files_only_new": only_new,
        "files_only_reference": only_old,
        "files_differ": differs,
        "dropped_identical": dropped_new == dropped_old,
        "dropped_only_new": sorted(dropped_new - dropped_old)[:200],
        "dropped_only_reference": sorted(dropped_old - dropped_new)[:200],
        "tokens": {"new": {k: tokens_new[k] for k in ("raw", "stripped")},
                   "reference": {k: tokens_old.get(k) for k in ("raw", "stripped")}},
        "comparator_passed": report["final_verification"]["passed"],
    }
    result["tokens_identical"] = result["tokens"]["new"] == result["tokens"]["reference"]
    result["identical"] = all(result[key] for key in
                              ("tree_identical", "dropped_identical", "tokens_identical", "comparator_passed"))
    return result


def verify(config, tool, target, reference: Path, records: Path) -> dict[str, Any]:
    instance_id = target.instance_id
    record_path = records / f"{instance_id}.json"
    if record_path.is_file():
        previous = json.loads(record_path.read_text())
        if previous["status"] != "failed":
            return previous
    started = time.time()
    record: dict[str, Any] = {"instance_id": instance_id, "lean_strip": tool.record()}
    repository_dir = config.output / "repos" / instance_id
    try:
        outcome = process_repository(config, tool, target)
        if outcome["discarded"]:
            record.update(status="discarded_by_lean_strip", identical=False)
        else:
            record.update(status="stripped", **compare(repository_dir, reference / "repos" / instance_id))
            report = json.loads((repository_dir / "prod-strip-report.json").read_text())
            record["lean_strip_seconds"] = report["lean_strip_seconds"]
            record["image"] = report["published_repository"]["variants"]["stripped"]["image"]
    except Exception as error:  # a failure is a finding; keep its output
        record.update(status="failed", identical=False, error=f"{type(error).__name__}: {error}")
    record["wall_seconds"] = round(time.time() - started, 1)
    if record.get("identical"):
        subprocess.run(["docker", "image", "rm", record["image"]], capture_output=True)
        shutil.rmtree(repository_dir, ignore_errors=True)
        record["output"] = "discarded"
    else:
        keep_image = record["status"] == "stripped" and not (record["lean_identical"] and record["dropped_identical"])
        if record.get("image") and not keep_image:
            subprocess.run(["docker", "image", "rm", record["image"]], capture_output=True)
        record["output"] = f"kept at {repository_dir.relative_to(ROOT)}" + ("" if keep_image else " (image removed)")
    _atomic_json(record_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--reference", type=Path, required=True, help="published dataset directory")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--recompare", action="store_true",
                        help="re-compare kept outputs of differing repositories without stripping again")
    args = parser.parse_args()
    config = load_config(args.config)
    # A separate run: never write into (or delete from) the configuration's own dataset.
    verify_id = f"{config.run_id}-verify"
    config = dataclasses.replace(
        config,
        run_id=verify_id,
        output=(ROOT / "datasets" / verify_id).resolve(),
        work=(ROOT / "runs/preprocessing" / verify_id).resolve(),
    )
    if args.reference.resolve() == config.output:
        parser.error("--reference must be a published dataset, not this check's own output")
    tool = resolve_lean_strip_tool()
    check_tool_pin(config, tool)
    reference = args.reference.resolve()
    records = config.work / "reproduction"
    records.mkdir(parents=True, exist_ok=True)
    if args.recompare:
        for path in sorted(records.glob("palomar__*.json")):
            record = json.loads(path.read_text())
            repository_dir = config.output / "repos" / record["instance_id"]
            if record["status"] != "stripped" or record["identical"] or not repository_dir.is_dir():
                continue
            record.update(compare(repository_dir, reference / "repos" / record["instance_id"]))
            if record["identical"]:
                subprocess.run(["docker", "image", "rm", record["image"]], capture_output=True)
                shutil.rmtree(repository_dir, ignore_errors=True)
                record["output"] = "discarded"
            elif record["lean_identical"] and record["dropped_identical"]:
                subprocess.run(["docker", "image", "rm", record["image"]], capture_output=True)
                record["output"] = f"kept at {repository_dir.relative_to(ROOT)} (image removed)"
            _atomic_json(path, record)
            print(f"{record['instance_id']}: {'IDENTICAL' if record['identical'] else 'DIFFERS'}", flush=True)
        return 0
    targets = [t for t in prepare_targets(config) if not args.only or t.instance_id in args.only]
    print(f"{config.run_id}: {len(targets)} repositories, lean-strip {tool.version} ({tool.commit[:12]}), "
          f"reference {reference}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(verify, config, tool, t, reference, records): t for t in targets}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            verdict = "IDENTICAL" if record.get("identical") else record["status"].upper()
            if record["status"] == "stripped" and not record["identical"]:
                verdict = "DIFFERS" + ("" if record["lean_identical"] else " (Lean sources)")
            print(f"[{time.strftime('%H:%M:%S')}] {record['instance_id']}: {verdict} "
                  f"({record['wall_seconds']}s)", flush=True)
    rows = [json.loads(path.read_text()) for path in sorted(records.glob("palomar__*.json"))]
    summary = {
        "run_id": config.run_id,
        "lean_strip": tool.record(),
        "reference": str(reference),
        "repositories": len(rows),
        "identical": sum(bool(r.get("identical")) for r in rows),
        "lean_identical": sum(bool(r.get("lean_identical")) for r in rows),
        "failed": sorted(r["instance_id"] for r in rows if r["status"] == "failed"),
        "differs": sorted(r["instance_id"] for r in rows if r["status"] == "stripped" and not r["identical"]),
    }
    _atomic_json(records / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "lean_strip"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
