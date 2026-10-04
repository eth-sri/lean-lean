"""Explicit, content-pinned reuse across relocated self-contained datasets.

Equivalence only authorizes reading a submitted patch. Independent Lean
verification is still mandatory before preparing either proof-filling arm.
"""

import hashlib
import json
from pathlib import Path

from leanlean.dataset_bundle import load_dataset
from leanlean.palomar_comparator import load_palomar_evidence, resolve_palomar_contract
from leanlean.preprocessing.repositories import file_sha256
from leanlean.preprocessing.standardized_repositories import source_tree_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def repository_fingerprint(root: Path, dataset_path: Path, instance_id: str) -> str:
    dataset = load_dataset(dataset_path, verify_trees=False)
    row = next(row for row in dataset["repositories"] if row["id"] == instance_id)
    database_path = root / dataset["repository_database"]["path"]
    if file_sha256(database_path) != dataset["repository_database"]["sha256"]:
        raise ValueError("equivalence database drift")
    database_row = next(row for row in json.loads(database_path.read_text())["repositories"]
                        if row["instance_id"] == instance_id)
    trees = {}
    for name in ("raw", "stripped"):
        variant = row["variants"][name]
        actual = source_tree_sha256(root / variant["cache_tree"])
        if actual != variant["tree_sha256"]:
            raise ValueError(f"{instance_id}: {name} source tree drift")
        trees[name] = actual
    materialization = row["variants"]["stripped"]["materialization"]
    report = json.loads((root / row["preprocessing"]["report"]).read_text())
    holdouts = [{key: record[key] for key in (
        "declaration", "module", "source_path", "source_command",
        "source_command_with_sorry", "source_file_with_sorry")}
        for record in report["removed_theorems"]]
    contract = resolve_palomar_contract(repo_root=root, database_row=database_row)
    evidence = load_palomar_evidence(repo_root=root, contract=contract)
    return digest({"trees": trees, "scope": row["scope"], "holdouts": holdouts,
                   "protected": database_row["protected"]["declarations"],
                   "environment_image": materialization["shared_environment"]["image_id"],
                   "environment_archive": materialization["shared_environment"]["archive"]["sha256"],
                   "warm_build": materialization["warm_build_cache"]["sha256"],
                   "comparator": [hashlib.sha256(value).hexdigest() for value in evidence]})


def equivalence_record(root: Path, source: Path, target: Path, instance_ids: list[str]) -> dict:
    repositories = {}
    for instance_id in instance_ids:
        original = repository_fingerprint(root, source, instance_id)
        if original != repository_fingerprint(root, target, instance_id):
            raise ValueError(f"{instance_id}: saved compression inputs differ")
        repositories[instance_id] = original
    return {"kind": "leanlean_repository_equivalence", "schema_version": 1,
            "source_dataset": str(source.relative_to(root)), "source_dataset_sha256": file_sha256(source),
            "target_dataset": str(target.relative_to(root)), "target_dataset_sha256": file_sha256(target),
            "repositories": repositories}


def validate_equivalence(root: Path, pin: dict, source: Path, target: Path, instance_id: str) -> None:
    receipt_path = (root / pin["path"]).resolve()
    if not receipt_path.is_relative_to(root) or file_sha256(receipt_path) != pin["sha256"]:
        raise ValueError("dataset equivalence receipt drift")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["kind"], receipt["schema_version"]) != ("leanlean_repository_equivalence", 1):
        raise ValueError("unsupported dataset equivalence receipt")
    for role, actual in (("source", source), ("target", target)):
        if ((root / receipt[role + "_dataset"]).resolve() != actual.resolve()
                or file_sha256(actual) != receipt[role + "_dataset_sha256"]):
            raise ValueError("dataset equivalence manifest drift")
        if repository_fingerprint(root, actual, instance_id) != receipt["repositories"][instance_id]:
            raise ValueError("dataset equivalence repository drift")
