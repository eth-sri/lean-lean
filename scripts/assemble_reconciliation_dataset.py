#!/usr/bin/env python3
"""Assemble a self-contained schema-v2 bundle as a hardlink view of a source bundle.

A dataset subset builder cannot consume a composed dataset such
as `leanlean_20260914`: it requires a `preprocessing.config` record that a
`verified_source_composition` bundle does not carry.  It also copies every byte,
which for a 12-repository slice means ~37 GB of repositories and shared
environment archives.

Every path recorded inside a schema-v2 bundle is bundle-relative (`raw`,
`stripped`, `warm-build`, `environments/<id>/image.tar.zst`), so a bundle is
relocatable and a subset is just a narrower directory with a narrower manifest.
This assembles that directory with hardlinks, which costs no meaningful disk and
leaves the source bundle untouched: the reports and trees are byte-identical, so
their recorded hashes stay valid.

Reports are NOT rewritten here, and anything that edits them afterwards MUST
write a temporary file and rename it over the path.  An in-place write keeps the
inode and silently mutates the source bundle too (this happened on 2026-09-20 and
broke the report pins of leanlean_20260914 and leanlean_20260914_leanstral_pinned).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from leanlean.preprocessing.repositories import (  # noqa: E402
    file_sha256,
    repository_definition_sha256,
)


def _hardlink_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-al", str(source), str(destination)], check=True)


def _environments_for(report_path: Path) -> set[str]:
    report = json.loads(report_path.read_text())
    variants = report["published_repository"]["variants"]
    found: set[str] = set()
    for variant in variants.values():
        materialization = variant.get("materialization")
        if not isinstance(materialization, dict):
            continue
        environment = materialization.get("shared_environment")
        if isinstance(environment, dict) and environment.get("id"):
            found.add(str(environment["id"]))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--repositories", nargs="+", required=True)
    arguments = parser.parse_args()

    source_path = (REPO_ROOT / arguments.source).resolve()
    source_root = source_path.parent
    output_root = (REPO_ROOT / arguments.output).resolve()
    if output_root.exists():
        raise SystemExit(f"immutable output already exists: {output_root}")

    source = yaml.safe_load(source_path.read_text())
    if source.get("kind") != "leanlean_dataset" or source.get("schema_version") != 2:
        raise SystemExit("source must be a self-contained schema-v2 bundle")

    rows = {str(row["id"]): row for row in source["repositories"]}
    missing = [name for name in arguments.repositories if name not in rows]
    if missing:
        raise SystemExit(f"repositories absent from the source bundle: {missing}")

    database = json.loads(
        (source_root / str(source["repository_database"]["path"])).read_text()
    )
    database_rows = {
        str(row["instance_id"]): row for row in database["repositories"]
    }
    missing = [name for name in arguments.repositories if name not in database_rows]
    if missing:
        raise SystemExit(f"repositories absent from the source database: {missing}")

    output_root.mkdir(parents=True)
    environments: set[str] = set()
    repositories: list[dict[str, Any]] = []
    for instance_id in arguments.repositories:
        _hardlink_tree(source_root / "repos" / instance_id, output_root / "repos" / instance_id)
        report_path = output_root / "repos" / instance_id / "prod-strip-report.json"
        digest = file_sha256(report_path)
        if digest != rows[instance_id]["report_sha256"]:
            raise SystemExit(f"{instance_id}: report drift against the source bundle")
        environments |= _environments_for(report_path)
        repositories.append(
            {
                "id": instance_id,
                "report": f"repos/{instance_id}/prod-strip-report.json",
                "report_sha256": digest,
            }
        )

    for environment_id in sorted(environments):
        _hardlink_tree(
            source_root / "environments" / environment_id,
            output_root / "environments" / environment_id,
        )

    payload = {
        "kind": "leanlean_standardized_repository_database",
        "schema_version": database.get("schema_version", 1),
        "dataset_id": arguments.dataset_id,
        "dataset_version": arguments.dataset_version,
        "repositories": [database_rows[name] for name in arguments.repositories],
    }
    database_path = output_root / "repository-database.json"
    database_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    manifest = {
        "kind": "leanlean_dataset_composition",
        "schema_version": 1,
        "run_id": arguments.dataset_id,
        "dataset_id": arguments.dataset_id,
        "dataset_version": arguments.dataset_version,
        "method": "hardlink_subset_of_schema_v2_bundle",
        "source_dataset": str(source_path.relative_to(REPO_ROOT)),
        "source_dataset_sha256": file_sha256(source_path),
        "repositories": list(arguments.repositories),
        "shared_environments": sorted(environments),
    }
    manifest_path = output_root / "run-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    dataset = {
        "kind": "leanlean_dataset",
        "schema_version": 2,
        "id": arguments.dataset_id,
        "version": arguments.dataset_version,
        "default_variant": source.get("default_variant", "stripped"),
        "repository_count": len(repositories),
        "repositories": repositories,
        "discarded_repositories": [],
        "repository_database": {
            "path": "repository-database.json",
            "sha256": file_sha256(database_path),
            "definition_sha256": repository_definition_sha256(payload),
            "dataset_id": arguments.dataset_id,
            "dataset_version": arguments.dataset_version,
        },
        "preprocessing": {
            "run_id": arguments.dataset_id,
            "manifest": "run-manifest.yaml",
            "manifest_sha256": file_sha256(manifest_path),
            "source_dataset": str(source_path.relative_to(REPO_ROOT)),
        },
        "evaluation_materialization": source["evaluation_materialization"],
    }
    (output_root / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    print(f"assembled {output_root.relative_to(REPO_ROOT)}")
    print(f"  repositories: {len(repositories)}")
    print(f"  environments: {len(environments)}")


if __name__ == "__main__":
    main()
