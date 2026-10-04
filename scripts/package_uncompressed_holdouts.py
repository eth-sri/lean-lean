#!/usr/bin/env python3
"""Add the matching Stage 2 arm beside the compressed reproof dataset."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import yaml
from rich.progress import track
from scripts.package_valid_compressed_holdouts import _write_json, audit_sources, checked, pin
from leanlean.host_side_checkpointing import _write_source_archive
from leanlean.preprocessing.repositories import repository_definition_sha256
from leanlean.preprocessing.standardized_repositories import (
    source_tree_sha256, load_standardized_repository_database,
)
from leanlean.metrics.tokens import measure_repository_lean_tokens
from leanlean.dataset_bundle import load_dataset

DIRECTORY = ROOT / "datasets/compressed-sorried-holdouts-20260912-r1"
MANIFEST = ROOT / "experiments/datasets/uncompressed-sorried-holdouts-20260912-r1.yaml"
STAGE2 = ROOT / "datasets/holdout19-stage2-sorried-preprocessing-20260912-r1/repos_stage2"


def validate_stage2(report):
    if not (report.get("final_build_passed") is True
            and report.get("sorried_theorem_gate", {}).get("passed") is True
            and report.get("sorried_holdout_source_audit", {}).get("passed") is True):
        raise ValueError("Stage 2 source did not pass build and exact target audits")


def plan(path):
    if path.exists():
        raise ValueError("manifest already exists")
    cases = json.loads((DIRECTORY / "cases.json").read_text())
    records = []
    for case in cases:
        report_path = STAGE2 / case["repository"] / "prod-strip-report.json"
        report = json.loads(report_path.read_text())
        validate_stage2(report)
        records.append({
            "id": case["id"], "repository": case["repository"],
            "report": pin(report_path),
            "source": {"path": report["stripped_output_tree"],
                       "tree_sha256": report["stripped_output_tree_sha256"]},
        })
    manifest = {
        "kind": "leanlean_uncompressed_sorried_holdout_packaging",
        "schema_version": 1, "run_id": "uncompressed-sorried-holdouts-20260912-r1",
        "model": "none", "reasoning_effort": "none",
        "generator": "byte_identical_stage2_source_packaging",
        "repository_variant": "stripped", "rounds": 1, "workers": 1,
        "container_resources": {"enabled": False, "cpus": 4, "memory": "32g",
                                "network_policy": "model_proxy_only", "pids_limit": 4096},
        "timeout_seconds": 3600,
        "evaluation": {"source_gates": "retained_stage2_build_and_exact_sorry_audit",
                       "full_comparator_preflight": "not_yet_run_for_this_arm",
                       "final_reproof_verifier": "original_full_comparator_without_sorryAx",
                       "project_cache": "empty_rebuild_from_source", "launch_reproof": False},
        "monitoring": {"snapshots": True, "every_n_edits": 1, "native_traces": True,
                       "native_rollout_retention": True, "model_rollouts": "not_applicable",
                       "baseline": "stage2.sources.tar.gz", "terminal": "uncompressed.sources.tar.gz",
                       "intermediate": "none_no_source_edits"},
        "compressed_dataset": pin(DIRECTORY / "dataset.yaml"),
        "compressed_database": pin(DIRECTORY / "repository-database.json"),
        "output": str(DIRECTORY.relative_to(ROOT)),
        "tmux_session": "package-uncompressed-sorried-holdouts-20260912-r1",
        "cases": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def build(path):
    manifest = yaml.safe_load(path.read_text())
    if manifest["kind"] != "leanlean_uncompressed_sorried_holdout_packaging":
        raise ValueError("wrong manifest")
    monitor = manifest["monitoring"]
    if not (monitor["snapshots"] is True and monitor["every_n_edits"] == 1
            and monitor["native_traces"] is True and not manifest["evaluation"]["launch_reproof"]):
        raise ValueError("monitoring or reproof policy drift")
    compressed_path = checked(manifest["compressed_dataset"])
    database_path = checked(manifest["compressed_database"])
    output = (ROOT / manifest["output"]).resolve()
    if output != compressed_path.parent:
        raise ValueError("paired views must share the dataset directory")
    if (output / "uncompressed.yaml").exists():
        raise ValueError("uncompressed view already published")
    compressed = load_dataset(compressed_path)
    if [r["id"] for r in manifest["cases"]] != [r["id"] for r in compressed["repositories"]]:
        raise ValueError("paired membership mismatch")
    shutil.copy2(path, output / "run-manifest.uncompressed.yaml")
    dataset = copy.deepcopy(compressed)
    dataset["id"] = manifest["run_id"]
    by_id = {r["id"]: r for r in dataset["repositories"]}
    records = []
    for case in track(manifest["cases"], description="Packaging uncompressed sorried sources"):
        report_path = checked(case["report"])
        report = json.loads(report_path.read_text())
        validate_stage2(report)
        source = (ROOT / case["source"]["path"]).resolve()
        if not source.is_relative_to(ROOT / "datasets"):
            raise ValueError("source must be a dataset artifact")
        if source_tree_sha256(source) != case["source"]["tree_sha256"]:
            raise ValueError("Stage 2 source tree drift")
        directory = output / "repos" / case["id"] / "uncompressed"
        tree = directory / "source"
        files = {}
        for original in sorted(source.rglob("*")):
            relative = original.relative_to(source)
            if any(p in {".git", ".lake"} for p in relative.parts):
                continue
            if original.is_symlink():
                raise ValueError("source symlink is not supported")
            if original.is_file():
                target = tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                files[relative.as_posix()] = original.read_bytes()
        if source_tree_sha256(tree) != case["source"]["tree_sha256"]:
            raise ValueError("source copy changed bytes or modes")
        holdouts = report["removed_theorems"]
        audit = audit_sources(files, holdouts)
        expected = json.loads((output / "repos" / case["id"] / "targets.json").read_text())
        if {h["declaration"] for h in holdouts} != {h["declaration"] for h in expected}:
            raise ValueError("paired target mismatch")
        targets = [{"declaration": h["declaration"], "source_path": h["source_path"],
                    "source_command_with_sorry": h["source_command_with_sorry"]} for h in holdouts]
        _write_json(directory / "targets.json", targets)
        snapshots = directory / "monitoring"
        snapshots.mkdir(parents=True, exist_ok=True)
        _write_source_archive(snapshots / "stage2.sources.tar.gz", files)
        shutil.copy2(snapshots / "stage2.sources.tar.gz", snapshots / "uncompressed.sources.tar.gz")
        validation = {
            "build_passed": True, "build_cache_policy": report.get("final_build_cache_policy"),
            "exact_target_audit_passed": True, "target_audit": audit,
            "full_comparator_preflight_passed": None,
            "source_evidence": case["report"], "source_tree_sha256": case["source"]["tree_sha256"],
            "final_reproof_requires_original_comparator": True,
        }
        _write_json(directory / "validation.json", validation)
        row = by_id[case["id"]]
        row["source"].update(ablation_arm="uncompressed", endpoint_policy="stage2_before_model_compression", edit_index=None)
        variant = row["variants"]["stripped"]
        variant.update(cache_tree=str(tree.relative_to(ROOT)), tree_sha256=case["source"]["tree_sha256"],
                       lean_tokens=measure_repository_lean_tokens(tree))
        variant["final_verification"] = {
            "passed": None, "scope": "full_comparator_preflight_pending",
            "evidence": str((directory / "validation.json").relative_to(ROOT)),
        }
        row["reproof"].pop("target_file", None)
        row["reproof"].update(targets=str((directory / "targets.json").relative_to(ROOT)),
                             target_files=sorted({h["source_path"] for h in holdouts}))
        records.append({"id": case["id"], "repository": case["repository"],
                        "target_count": len(targets), "source": str(tree.relative_to(output)),
                        "targets": str((directory / "targets.json").relative_to(output)),
                        "source_tree_sha256": variant["tree_sha256"],
                        "lean_tokens": variant["lean_tokens"]})
        _write_json(output / "status.uncompressed.json", {"status": "packaging", "completed": len(records), "total": len(manifest["cases"])})
    database = json.loads(database_path.read_text())
    database["dataset_id"] = manifest["run_id"]
    new_database = output / "repository-database.uncompressed.json"
    _write_json(new_database, database)
    load_standardized_repository_database(new_database, repo_root=ROOT)
    dataset["repository_database"] = {
        **pin(new_database), "definition_sha256": repository_definition_sha256(database),
        "dataset_id": manifest["run_id"], "dataset_version": database["dataset_version"],
    }
    dataset["preprocessing"] = {
        "run_id": manifest["run_id"], "manifest": pin(output / "run-manifest.uncompressed.yaml"),
        "source_state": "v25_stage2_before_model_compression",
    }
    dataset["reproof"].update(
        source_state="uncompressed_stage2_with_sorried_holdouts",
        cases="cases.uncompressed.json",
        full_comparator_preflight="pending",
    )
    _write_json(output / "cases.uncompressed.json", records)
    candidate = output / "uncompressed.candidate.yaml"
    candidate.write_text(yaml.safe_dump(dataset, sort_keys=False))
    loaded = load_dataset(candidate)
    if loaded["repository_count"] != len(records):
        raise ValueError("published count mismatch")
    compressed_view = output / "compressed.yaml"
    if compressed_view.exists() and compressed_view.read_bytes() != compressed_path.read_bytes():
        raise ValueError("compressed view already exists with different content")
    shutil.copy2(compressed_path, compressed_view)
    candidate.replace(output / "uncompressed.yaml")
    _write_json(output / "status.uncompressed.json", {
        "status": "complete", "repositories": len(records),
        "targets": sum(r["target_count"] for r in records),
        "build_and_target_audits": "passed", "full_comparator_preflight": "pending",
        "reproof_launched": False, "monitoring": monitor,
    })
    (output / "PAIRED.md").write_text(
        "# Paired Reproof Datasets\n\n"
        "Select compressed.yaml or uncompressed.yaml via the dataset field in an eval.sh dataset config.\n\n"
        "Both views contain the same 15 repository IDs and 19 held-out theorem statements. "
        "Uncompressed means the v25 Stage 2 preprocessed source before model compression, "
        "not the raw upstream repository. Theorem locations can differ between arms; "
        "each arm has its own targets.json.\n\n"
        "Compressed sources have retained clean-build and original-Comparator interface checks. "
        "Uncompressed sources have retained Stage 2 build and exact-sorry audits; their separate "
        "full Comparator interface preflight remains pending. Both arms rebuild project code "
        "from empty caches during evaluation and require the strict original Comparator after reproof.\n\n"
        "Dataset configs:\n"
        "- configs/dataset/compressed-sorried-holdouts-20260912-r1.yaml\n"
        "- configs/dataset/uncompressed-sorried-holdouts-20260912-r1.yaml\n\n"
        "dataset.yaml remains a compatibility entry point for the compressed arm. "
        "No model calls are launched by packaging.\n"
    )
    print(f"Created paired views: {compressed_view} and {output / 'uncompressed.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=MANIFEST)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        plan(args.manifest.resolve())
        print(args.manifest.resolve())
    else:
        build(args.manifest.resolve())


if __name__ == "__main__":
    main()
