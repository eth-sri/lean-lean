"""Publish one self-contained, named-case dataset from verified holdout bundles."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import subprocess

import yaml

from leanlean.dataset_bundle import load_dataset
from leanlean.palomar_comparator import load_comparator_bundle, resolve_palomar_contract
from leanlean.pipeline.reconstruction import _atomic_text, _path
from leanlean.pipeline.reconstruction_assets import source_inventory
from leanlean.preprocessing.graph_artifact import graph_sha256
from leanlean.preprocessing.repositories import file_sha256, repository_definition_sha256
from leanlean.preprocessing.standardized_repositories import load_standardized_repository_database
from leanlean.preprocessing.theorem_holdout import theorem_command_with_sorry

ROOT = Path(__file__).resolve().parents[3]


def write_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"refusing to overwrite copied artifacts: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True)


def corrected_holdout(record: dict, alias: str) -> dict:
    result = copy.deepcopy(record)
    old = result["source_command_with_sorry"]
    new = theorem_command_with_sorry(result["source_command"])
    if result["source_file_with_sorry"].count(old) != 1:
        raise ValueError("holdout scaffold does not contain exactly one recorded statement")
    result["source_file_with_sorry"] = result["source_file_with_sorry"].replace(old, new, 1)
    result["source_command_with_sorry"] = new
    result["repository"] = alias
    return result


def localize_comparator(record: dict, original_id: str, alias: str, destination: Path) -> dict:
    _, manifest, challenge, configuration, registry = load_comparator_bundle(
        repo_root=ROOT, bundle_record=record, instance_id=original_id,
    )
    renamed = {**manifest, "instance_id": alias}
    # Only the transport identity changes. Registered statement/config bytes and
    # all upstream hashes stay exact; ordinary Comparator resolution checks them.
    for key, content in (("challenge", challenge), ("configuration", configuration), ("registry_record", registry)):
        _atomic_text(destination / manifest[key]["path"], content.decode())
    write_json(destination / "manifest.json", renamed)
    return {"path": str(destination.relative_to(ROOT)), "manifest": "manifest.json",
            "manifest_sha256": file_sha256(destination / "manifest.json")}


def publish(manifest: dict, manifest_path: Path) -> dict:
    output = _path(manifest["dataset"], "dataset").parent
    if (output / "dataset.yaml").exists():
        dataset = load_dataset(output / "dataset.yaml")
        if dataset["repository_count"] != len(manifest["cases"]):
            raise ValueError("existing transfer dataset membership drift")
        if file_sha256(output / "run-manifest.yaml") != file_sha256(manifest_path):
            raise ValueError("existing transfer dataset manifest drift")
        return dataset
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(output / "run-manifest.yaml", manifest_path.read_text())
    records, database_rows, case_records = [], [], []
    environment_pins = {}
    for case in manifest["cases"]:
        alias, original_id = case["id"], case["repository"]
        if not re.fullmatch(r"[a-z0-9_]+", alias):
            raise ValueError("unsafe transfer case ID")
        input_path = _path(case["dataset"], "case.dataset")
        if file_sha256(input_path) != case["dataset_sha256"]:
            raise ValueError(f"{alias}: input dataset drift")
        source_dataset = load_dataset(input_path)
        if len(source_dataset["repositories"]) != 1:
            raise ValueError("each input must be an independent verified ablation")
        runtime = source_dataset["repositories"][0]
        if runtime["id"] != original_id:
            raise ValueError("input repository identity differs")
        report_path = Path(runtime["preprocessing"]["report"])
        report = json.loads(report_path.read_text())
        if report.get("certification", {}).get("publishable") is not True:
            raise ValueError(f"{alias}: preprocessing certification did not pass")
        directory = output / "repos" / alias
        for name in ("raw", "stripped"):
            source = Path(runtime["variants"][name]["cache_tree"])
            if source_inventory(source)[1]:
                raise ValueError(f"{alias}: unexpected auxiliary project files")
            copy_tree(source, directory / name)
        materialization = runtime["variants"]["stripped"]["materialization"]
        copy_tree(Path(materialization["warm_build_cache"]["path"]), directory / "warm-build")
        env = materialization["shared_environment"]
        archive = env["archive"]
        pin = (env["image_id"], archive["sha256"])
        if env["id"] in environment_pins and environment_pins[env["id"]] != pin:
            raise ValueError("conflicting shared environment identity")
        if env["id"] not in environment_pins:
            copy_tree(Path(archive["path"]), output / "environments" / env["id"] / "image.tar.zst")
            environment_pins[env["id"]] = pin
        graph = json.loads(Path(runtime["preprocessing"]["dependency_graph"]["path"]).read_text())
        graph["repository"] = alias
        graph["sha256"] = graph_sha256(graph)
        write_json(directory / "dependency-graph.json", graph)
        published = report["published_repository"]
        published["id"] = alias
        published.setdefault("source", {})["parent_repository"] = original_id
        published["source"]["removal"] = case["removal"]
        published["source"]["category"] = case["category"]
        published["source"]["baseline_stripped_lean_tokens"] = case["baseline_stripped_lean_tokens"]
        published["preprocessing"]["dependency_graph"]["sha256"] = graph["sha256"]
        report["instance_id"] = alias
        report["dependency_graph"]["sha256"] = graph["sha256"]
        report["removed_theorems"] = [corrected_holdout(item, alias) for item in report["removed_theorems"]]
        report["consolidation"] = {"parent_dataset_sha256": case["dataset_sha256"],
            "parent_report_sha256": file_sha256(report_path), "original_repository": original_id,
            "policy": "byte_identical_sources_alias_metadata_and_correct_let_bound_statement_scaffold"}
        write_json(directory / "prod-strip-report.json", report)
        records.append({"id": alias, "report": f"repos/{alias}/prod-strip-report.json",
                        "report_sha256": file_sha256(directory / "prod-strip-report.json")})
        db = json.loads(Path(source_dataset["repository_database"]["path"]).read_text())
        row = copy.deepcopy(db["repositories"][0])
        row["instance_id"] = alias
        raw = row["source"]["raw_repository"]
        raw["path"] = str((directory / "raw").relative_to(ROOT))
        if "build_cache" in raw:
            raw["build_cache"]["path"] = raw["path"] + "/.lake"
        row["source"].pop("raw_image", None)
        row["source"]["standardization"]["source_manifest"] = str((output / "run-manifest.yaml").relative_to(ROOT))
        provenance = row["protected"]["provenance"]
        for name, section in (("parent", provenance["parent"]), ("current", provenance),
                              ("standard", row["source"]["standardization"])):
            if "comparator" in section:
                section["comparator"] = localize_comparator(section["comparator"], original_id, alias,
                                                            directory / "comparator" / name)
        resolve_palomar_contract(repo_root=ROOT, database_row=row)
        database_rows.append(row)
        holdouts = report["removed_theorems"]
        if len(holdouts) != 1 or holdouts[0]["declaration"] != case["theorem"]["declaration"]:
            raise ValueError("case target and held-out statement differ")
        write_json(directory / "holdout.json", holdouts[0])
        case_records.append({**case, "holdout": f"repos/{alias}/holdout.json",
                             "holdout_sha256": file_sha256(directory / "holdout.json"),
                             "stripped_lean_tokens": runtime["variants"]["stripped"]["lean_tokens"]})
    database = {"kind": "leanlean_standardized_repository_database", "schema_version": 2,
                "dataset_id": manifest["run_id"], "dataset_version": "v1", "repositories": database_rows}
    write_json(output / "repository-database.json", database)
    load_standardized_repository_database(output / "repository-database.json", repo_root=ROOT)
    write_json(output / "cases.json", case_records)
    dataset = {"kind": "leanlean_dataset", "schema_version": 2, "id": manifest["run_id"],
               "version": "v1", "default_variant": "stripped", "repository_count": len(records),
               "repositories": records, "discarded_repositories": [],
               "repository_database": {"path": "repository-database.json",
                   "sha256": file_sha256(output / "repository-database.json"),
                   "definition_sha256": repository_definition_sha256(database)},
               "preprocessing": {"run_id": manifest["run_id"], "manifest": {
                   "path": "run-manifest.yaml", "sha256": file_sha256(output / "run-manifest.yaml")}},
               "transfer": {"cases": "cases.json", "case_count": len(records),
                            "source_dataset": manifest["source_dataset"], "reproof_view": "reconstruction.yaml"}}
    # Validate the complete bundle before exposing its canonical dataset name.
    check_path = output / "dataset.candidate.yaml"
    _atomic_text(check_path, yaml.safe_dump(dataset, sort_keys=False))
    result = load_dataset(check_path)
    check_path.replace(output / "dataset.yaml")
    return result
