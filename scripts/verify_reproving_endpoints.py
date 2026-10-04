#!/usr/bin/env python3
"""Offline lean_verify of reproving-ablation endpoint patches whose run images were retired.

Rebuilds each repository image from the pinned dataset tree, authors one
``leanlean_reproving_compile_comparison`` manifest per arm (the agent's own
recorded lean_verify outcome is pinned as the "old" result), then runs
scripts/run_reproving_compile.py on both arms.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from leanlean.evaluation_images import RECEIPTS_ENV, ensure_materialized_image_from_record

DATASET = ROOT / "datasets/reproving-ablation/dataset.yaml"
VARIANTS = {"baseline": "baseline", "post-compression": "post-compression"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("VARIANT", "SOURCE_OUTPUT"), required=True)
    parser.add_argument("--build-workers", type=int, default=4)
    parser.add_argument("--verify-workers", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("TMUX") and not args.prepare_only:
        raise SystemExit("run inside tmux")

    dataset = yaml.safe_load(DATASET.read_text())
    repos = {row["id"]: row for row in dataset["repositories"]}
    out_root = ROOT / "output/evaluation" / args.run_id
    state_root = ROOT / "runs/evaluation" / args.run_id
    state_root.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []

    for variant, source_output in args.arm:
        source = (ROOT / source_output).resolve()
        predictions_path = source / "preds.json"
        predictions = json.loads(predictions_path.read_text())
        receipts = out_root / "image_materialization" / variant
        os.environ[RECEIPTS_ENV] = str(receipts)
        arm_run_id = f"{args.run_id}-{variant}"

        def build(iid: str) -> None:
            materialization = repos[iid]["variants"][VARIANTS[variant]]["materialization"]
            tree = repos[iid]["variants"][VARIANTS[variant]]
            original = json.loads((source / "image_materialization" / f"{iid}.json").read_text())
            if original["tree_sha256"] != tree["tree_sha256"]:
                raise RuntimeError(f"{iid}: dataset tree drifted from the reproving run")
            ensure_materialized_image_from_record(iid, {
                "tag": f"leanlean-{iid}-stripped:{arm_run_id}",
                "source_tree": tree["cache_tree"],
                "tree_sha256": tree["tree_sha256"],
                "build_target": materialization["build_target"],
                "build_jobs": 8,
                "build_timeout_seconds": 14400,
                "persist": False,
                "source_isolation": "standalone_v3",
                "backend": materialization["backend"],
                "shared_environment": materialization["shared_environment"],
                "warm_build_cache": materialization["warm_build_cache"],
            })

        ids = sorted(predictions)
        with ThreadPoolExecutor(max_workers=args.build_workers) as pool:
            list(pool.map(build, ids))

        agent_results = {}
        for iid in ids:
            usage = source / iid / "agent_tool_usage.json"
            tool = (json.loads(usage.read_text()).get("tools") or {}).get("lean_verify") or {} if usage.exists() else {}
            passed = int(tool.get("succeeded") or 0) > 0
            agent_results[iid] = {"repo_build_passed": passed, "resolved": passed, "agent_lean_verify": tool}
        old_path = state_root / f"agent-recorded-lean-verify.{variant}.json"
        old_path.write_text(json.dumps(agent_results, indent=2, sort_keys=True) + "\n")

        contract = ROOT / f"datasets/reproving-ablation/repository-database.{variant}.json"
        rows = []
        for iid in ids:
            receipt = receipts / f"{iid}.json"
            rows.append({
                "id": iid,
                "image_id": json.loads(receipt.read_text())["image_id"],
                "tree_sha256": repos[iid]["variants"][VARIANTS[variant]]["tree_sha256"],
                "build_target": repos[iid]["variants"][VARIANTS[variant]]["materialization"]["build_target"],
                "materialization_receipt": {"path": rel(receipt), "sha256": digest(receipt)},
                "old_result": {
                    "path": rel(old_path), "sha256": digest(old_path), "key": iid,
                    "build_passed": agent_results[iid]["repo_build_passed"],
                    "resolved": agent_results[iid]["resolved"],
                },
            })
        spec = {
            "kind": "leanlean_reproving_compile_comparison",
            "schema_version": 1,
            "run_id": arm_run_id,
            "status": "prepared",
            "purpose": "offline lean_verify of agent endpoint patches; old_result = agent-recorded lean_verify",
            "source_model": args.source_model,
            "source_reasoning_effort": "xhigh",
            "source_run": rel(source),
            "dataset": {"manifest": rel(DATASET), "variant": variant, "repository_count": len(ids)},
            "predictions": {"path": rel(predictions_path), "sha256": digest(predictions_path)},
            "contract_database": {"path": rel(contract), "sha256": digest(contract), "visibility": "offline_verifier_only"},
            "repositories": rows,
            "parallelism": {"workers": args.verify_workers},
            "container": {
                "cpus": 8, "memory": "64g", "max_total_memory": f"{64 * args.verify_workers}g",
                "pids_limit": 4096, "timeout_seconds": 7200, "container_timeout": "3h",
                "cgroup_parent": "lean.slice", "network_policy": "none",
            },
            "evaluation": {
                "project_build_command": "LEAN_NUM_THREADS=8 lake build",
                "project_build_required": False,
                "lean_verify": True,
                "proof_length": False,
                "clear_project_build_cache": True,
                "success_criterion": "lean_verify",
                "offline_dependency_policy": "pinned_local_package_paths",
            },
            "model_visibility": {"Challenge.lean": False, "challenge.json": False, "comparator.json": False,
                                 "verifier_installed_after_model_submission": True},
            "monitoring": {"snapshots": True, "every_n_edits": 1, "retain_build_logs": True,
                           "retain_lean_verify_logs": True, "native_rollouts": "not_applicable_no_model"},
            "implementation": {"path": "scripts/run_reproving_compile.py",
                               "sha256": digest(ROOT / "scripts/run_reproving_compile.py")},
            "outputs": {
                "directory": rel(out_root / variant),
                "run_artifact": rel(state_root / variant / "run.json"),
                "summary": rel(out_root / variant / "summary.json"),
                "tmux_session": f"verify-{arm_run_id}",
            },
        }
        manifest = ROOT / "experiments" / f"{arm_run_id}.yaml"
        manifest.write_text(yaml.safe_dump(spec, sort_keys=False))
        subprocess.run([sys.executable, str(ROOT / "scripts/run_reproving_compile.py"), str(manifest), "--validate-only"], check=True)
        manifests.append(manifest)

    if args.prepare_only:
        return 0
    processes = []
    for manifest in manifests:
        log = (out_root / f"{manifest.stem}.log").open("w")
        processes.append(subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/run_reproving_compile.py"), str(manifest), "--inside-tmux"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        ))
    codes = [process.wait() for process in processes]
    print("compile exit codes:", codes, flush=True)
    return max(codes)


if __name__ == "__main__":
    raise SystemExit(main())
