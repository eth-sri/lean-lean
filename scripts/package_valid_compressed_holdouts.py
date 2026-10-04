#!/usr/bin/env python3
"""Package byte-identical certified source-backed endpoints for later reproof."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import yaml
from rich.console import Console
from rich.progress import track
from leanlean.dataset_bundle import load_dataset
from leanlean.playback import _archive_source_files
from leanlean.preprocessing.repositories import file_sha256, repository_definition_sha256
from leanlean.preprocessing.standardized_repositories import (
    load_standardized_repository_database, source_tree_sha256,
)
from leanlean.shared_cache import artifact_tree_sha256
from leanlean.metrics.tokens import measure_repository_lean_tokens
from leanlean.palomar_comparator import resolve_palomar_contract, load_palomar_evidence
from leanlean.pipeline.transfer_dataset import localize_comparator

SOURCE = ROOT / "datasets/leanlean-holdouts21-s42_20260910_r1"
DEFAULT_MANIFEST = ROOT / "experiments/datasets/compressed-sorried-holdouts-20260912-r1.yaml"



def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def pin(path):
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": file_sha256(path)}


def checked(record):
    path = (ROOT / record["path"]).resolve()
    if not path.is_relative_to(ROOT) or file_sha256(path) != record["sha256"]:
        raise ValueError(f"input pin drift: {record['path']}")
    return path


def plan(path):
    if path.exists():
        raise ValueError("manifest already exists; use it without --plan")
    selected = {}
    for summary in sorted((ROOT / "output/reconstruction").glob(
        "leanlean-declaration-context-stage2-signatures-*/summary.json"
    )):
        for result in json.loads(summary.read_text()).get("results", []):
            if result.get("passed") is not True or result.get("signature_comparator_passed") is not True:
                continue
            case = result["case_id"]
            previous = selected.get(case)
            if previous is None or result["completed_at"] > previous[0]["completed_at"]:
                selected[case] = result, summary
    if len(selected) != 15:
        raise ValueError(f"expected 15 certified endpoints, found {len(selected)}")
    cases = []
    for case, (result, summary) in sorted(selected.items()):
        cases.append({
            "id": case, "repository": result["repository"],
            "summary": pin(summary),
            "archive": result["augmented_archive"],
            "baseline": result["baseline"],
            "payload": pin(ROOT / result["append_payload"]["path"]),
            "holdouts": pin(SOURCE / "repos" / case / "holdouts.json"),
            "source_report": pin(SOURCE / "repos" / case / "prod-strip-report.json"),
        })
    manifest = {
        "kind": "leanlean_compressed_sorried_holdout_packaging",
        "schema_version": 1, "run_id": "compressed-sorried-holdouts-20260912-r1",
        "model": "none", "reasoning_effort": "none",
        "generator": "byte_identical_certified_source_archive_packaging",
        "repository_variant": "stripped", "source_state": "compressed_with_sorried_holdouts",
        "rounds": 1, "workers": 1,
        "container_resources": {"enabled": False, "cpus": 4, "memory": "32g",
                                "pids_limit": 4096, "network_policy": "model_proxy_only"},
        "timeout_seconds": 3600,
        "evaluation": {
            "source_evidence": "existing_clean_build_and_original_comparator",
            "require_exact_archive_sha256": True, "reject_original_target_proofs": True,
            "final_reproof_verifier": "original_full_comparator_without_sorryAx",
            "project_cache": "empty_rebuild_from_source",
            "launch_reproof": False,
        },
        "monitoring": {
            "snapshots": True, "every_n_edits": 1, "native_traces": True,
            "native_rollout_retention": True, "model_rollouts": "not_applicable",
            "baseline": "compressed.sources.tar.gz", "terminal": "sorried.sources.tar.gz",
            "intermediate": "no_source_edits_byte_identical_copy",
        },
        "source_dataset": pin(SOURCE / "dataset.yaml"),
        "source_database": pin(SOURCE / "repository-database.json"),
        "output": "datasets/compressed-sorried-holdouts-20260912-r1",
        "tmux_session": "package-compressed-sorried-holdouts-20260912-r1",
        "cases": cases,
        "excluded": {
            "case07__holdout": "source transport failed to elaborate",
            "case10__holdout": "statement support source missing",
            "case11__holdout": "incompatible compressed OctV interface",
            "case13__holdout": "Stage 2 stripped build failed; retry still pending",
            "case14__holdout": "original Comparator rejected bandQuartic",
            "case19__holdout": "excluded proof-irrelevance case",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def audit_sources(files, holdouts, destination=None):
    texts = {name: data.decode("utf-8") for name, data in files.items() if name.endswith(".lean")}
    for name in files:
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or ".." in parts or any(p in {".lake", ".git"} for p in parts):
            raise ValueError(f"unsafe or cached source path: {name}")
        if name.endswith((".olean", ".ilean", ".o", ".so", ".a")):
            raise ValueError(f"compiled project artifact: {name}")
    audit = []
    for holdout in holdouts:
        target_source = texts[destination or holdout["source_path"]]
        statement = holdout["source_command_with_sorry"].strip()
        original = holdout["source_command"].strip()
        count = target_source.count(statement)
        original_count = sum(text.count(original) for text in texts.values())
        if count != 1 or original_count:
            raise ValueError(f"target source audit failed: {holdout['declaration']} ({count}, {original_count})")
        audit.append({"declaration": holdout["declaration"], "sorried_count": count,
                      "original_proof_count": original_count})
    for text in texts.values():
        if "import HoldoutRestore" in text or "import HoldoutTrusted" in text:
            raise ValueError("legacy compiled-declaration scaffold in source archive")
    return audit


def package_case(case, template, output):
    case_id = case["id"]
    summary_path = checked(case["summary"])
    matches = [r for r in json.loads(summary_path.read_text())["results"] if r["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"{case_id}: ambiguous validation result")
    result = matches[0]
    if not (result.get("passed") is True and result.get("signature_comparator_passed") is True
            and result.get("build_returncode") == 0 and not result.get("build_timed_out")):
        raise ValueError(f"{case_id}: missing clean build or signature gate")
    if result["augmented_archive"]["sha256"] != case["archive"]["sha256"]:
        raise ValueError(f"{case_id}: certified archive differs")
    archive, baseline, payload = checked(case["archive"]), checked(case["baseline"]), checked(case["payload"])
    holdouts = json.loads(checked(case["holdouts"]).read_text())
    source_report = json.loads(checked(case["source_report"]).read_text())
    files = _archive_source_files(archive)
    destination = result["append_payload"]["destination"]
    audit = audit_sources(files, holdouts, destination)
    directory = output / "repos" / case_id
    source_tree = directory / "source"
    for name, data in files.items():
        target = source_tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    snapshots = directory / "monitoring"
    snapshots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, snapshots / "sorried.sources.tar.gz")
    shutil.copy2(baseline, snapshots / "compressed.sources.tar.gz")
    shutil.copy2(payload, directory / "restored-statements.lean")
    _write_json(directory / "validation.json", {
        "passed": True, "source": "retained_exact_archive_validation",
        "receipt": result, "receipt_summary": case["summary"], "target_audit": audit,
        "packaged_archive_sha256": file_sha256(snapshots / "sorried.sources.tar.gz"),
    })
    targets = [
        {"declaration": h["declaration"], "source_path": destination,
         "source_command_with_sorry": h["source_command_with_sorry"],
         "original_proof_sha256": hashlib.sha256(h["source_command"].encode()).hexdigest()}
        for h in holdouts
    ]
    _write_json(directory / "targets.json", targets)
    db_row = copy.deepcopy(template)
    parent = copy.deepcopy(db_row["protected"]["provenance"]["parent"])
    parent["comparator"] = localize_comparator(
        parent["comparator"], case_id, case_id, directory / "comparator",
    )
    db_row["protected"] = {
        "declarations": sorted(set(db_row["protected"]["declarations"]) | {h["declaration"] for h in holdouts}),
        "provenance": parent,
    }
    db_row["source"]["standardization"]["comparator"] = copy.deepcopy(parent["comparator"])
    contract = resolve_palomar_contract(repo_root=ROOT, database_row=db_row)
    _, _, runtime_config = load_palomar_evidence(repo_root=ROOT, contract=contract)
    if "sorryAx" in json.loads(runtime_config)["permitted_axioms"]:
        raise ValueError(f"{case_id}: final verification permits sorry")
    materialization = copy.deepcopy(source_report["published_repository"]["variants"]["stripped"]["materialization"])
    env_archive = materialization["shared_environment"]["archive"]
    env_path = SOURCE / env_archive["path"]
    if not env_path.is_file() or env_path.stat().st_size != env_archive["bytes"]:
        raise ValueError(f"{case_id}: shared dependency archive missing")
    env_archive["path"] = str(env_path.relative_to(ROOT))
    empty_cache = directory / "empty-project-cache"
    empty_cache.mkdir(parents=True, exist_ok=True)
    if any(empty_cache.iterdir()):
        raise ValueError(f"{case_id}: project cache must be empty")
    materialization["warm_build_cache"] = {
        "path": str(empty_cache.relative_to(ROOT)), "sha256": artifact_tree_sha256(empty_cache),
    }
    tree_hash = source_tree_sha256(source_tree)
    tokens = measure_repository_lean_tokens(source_tree)
    runtime = {
        "id": case_id,
        "source": {"parent_repository": case["repository"], "ablation_arm": "compressed",
                   "endpoint_policy": result["baseline"]["selection_policy"],
                   "edit_index": result["baseline"].get("edit_index")},
        "variants": {"stripped": {
            "format": "theorem_reconstruction_repository",
            "cache_tree": str(source_tree.relative_to(ROOT)),
            "tree_sha256": tree_hash, "lean_tokens": tokens, "materialization": materialization,
            "final_verification": {"passed": True, "scope": "sorried_interface_preflight",
                                   "evidence": str((directory / "validation.json").relative_to(ROOT))},
        }},
        "reproof": {"targets": str((directory / "targets.json").relative_to(ROOT)),
                    "target_file": destination, "proofs_completed": False,
                    "verifier": "original_full_comparator", "permitted_sorry": False},
    }
    record = {
        "id": case_id, "repository": case["repository"], "target_count": len(targets),
        "source": str(source_tree.relative_to(output)), "target_file": destination,
        "targets": str((directory / "targets.json").relative_to(output)),
        "source_tree_sha256": tree_hash, "source_archive_sha256": case["archive"]["sha256"],
        "lean_tokens": tokens, "validation": str((directory / "validation.json").relative_to(output)),
        "endpoint_policy": result["baseline"]["selection_policy"],
        "edit_index": result["baseline"].get("edit_index"),
    }
    return runtime, db_row, record


def build(path):
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("kind") != "leanlean_compressed_sorried_holdout_packaging":
        raise ValueError("wrong packaging manifest")
    monitoring = manifest["monitoring"]
    if not (monitoring.get("snapshots") is True and monitoring.get("every_n_edits") == 1
            and monitoring.get("native_traces") is True):
        raise ValueError("required monitoring is disabled")
    if manifest["evaluation"]["launch_reproof"] is not False:
        raise ValueError("packaging must not launch models")
    checked(manifest["source_dataset"])
    source_db = json.loads(checked(manifest["source_database"]).read_text())
    templates = {row["instance_id"]: row for row in source_db["repositories"]}
    output = (ROOT / manifest["output"]).resolve()
    if not output.is_relative_to(ROOT / "datasets"):
        raise ValueError("output must be under datasets")
    if (output / "dataset.yaml").exists():
        raise ValueError("dataset already published; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output / "run-manifest.yaml")
    rows, database_rows, records = [], [], []
    for case in track(manifest["cases"], description="Packaging validated compressed sources"):
        runtime, db_row, record = package_case(case, templates[case["id"]], output)
        rows.append(runtime)
        database_rows.append(db_row)
        records.append(record)
        _write_json(output / "status.json", {"status": "packaging", "completed": len(rows),
                                            "total": len(manifest["cases"]), "monitoring": monitoring})
    database = {
        "kind": source_db["kind"], "schema_version": source_db["schema_version"],
        "dataset_id": manifest["run_id"], "dataset_version": "v1", "repositories": database_rows,
    }
    database_path = output / "repository-database.json"
    _write_json(database_path, database)
    load_standardized_repository_database(database_path, repo_root=ROOT)
    _write_json(output / "cases.json", records)
    dataset = {
        "kind": "leanlean_dataset", "schema_version": 1,
        "id": manifest["run_id"], "version": "v1", "default_variant": "stripped",
        "repository_count": len(rows), "repositories": rows,
        "repository_database": {
            **pin(database_path), "definition_sha256": repository_definition_sha256(database),
            "dataset_id": manifest["run_id"], "dataset_version": "v1",
        },
        "preprocessing": {"run_id": manifest["run_id"], "manifest": pin(output / "run-manifest.yaml"),
                          "source_dataset": manifest["source_dataset"]["path"]},
        "reproof": {
            "source_state": "compressed_with_sorried_holdouts", "cases": "cases.json",
            "target_count": sum(r["target_count"] for r in records),
            "final_verifier": "original_full_comparator", "project_build_cache": "empty",
            "dependency_archives": "pinned_references_to_existing_holdout_dataset",
            "launched": False, "excluded": manifest["excluded"],
        },
    }
    candidate = output / "dataset.candidate.yaml"
    candidate.write_text(yaml.safe_dump(dataset, sort_keys=False))
    loaded = load_dataset(candidate)
    if len(loaded["repositories"]) != len(manifest["cases"]):
        raise ValueError("dataset membership changed")
    for row in rows:
        variant = row["variants"]["stripped"]
        if source_tree_sha256(ROOT / variant["cache_tree"]) != variant["tree_sha256"]:
            raise ValueError("packaged source tree drift")
        cache = ROOT / variant["materialization"]["warm_build_cache"]["path"]
        if any(cache.iterdir()):
            raise ValueError("nonempty project cache")
    candidate.replace(output / "dataset.yaml")
    _write_json(output / "status.json", {
        "status": "complete", "repositories": len(rows), "targets": dataset["reproof"]["target_count"],
        "monitoring": monitoring, "reproof_launched": False,
    })
    table = "\n".join(
        f"| {r['id']} | {r['target_count']} | {r['target_file']} | {r['endpoint_policy']} |"
        for r in records
    )
    (output / "README.md").write_text(
        "# Compressed Holdout Reproof Dataset\n\n"
        f"{len(rows)} compressed repositories; {dataset['reproof']['target_count']} held-out theorems.\n\n"
        "Each repos/<case>/source tree is byte-identical to a clean-built, original-Comparator "
        "accepted sorried source archive. targets.json contains only the requested sorried commands. "
        "validation.json retains the exact validation receipt and hashes.\n\n"
        "Use dataset.yaml with the ordinary evaluation loader and variant stripped. This variant "
        "contains compressed endpoints with the source-based theorem restoration already applied. "
        "Replace the listed sorry proofs, then use lean_verify. Final verification uses the full "
        "original Comparator and does not permit sorryAx. A successful initial build is only "
        "an interface check, not a completed proof.\n\n"
        "Project build caches are empty. Runtime builds from source using pinned shared dependency "
        "environments referenced from the existing holdout dataset; retain that dataset alongside this one. "
        "No trusted theorem .olean restoration helper is used. No model reproof was launched.\n\n"
        "| Case | Targets | Target File | Compressed Endpoint |\n|---|---:|---|---|\n" + table + "\n"
    )
    Console().print(f"Published {len(rows)} repositories / {dataset['reproof']['target_count']} targets: {output / 'dataset.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.plan:
        plan(path)
        print(path)
    else:
        build(path)


if __name__ == "__main__":
    main()
