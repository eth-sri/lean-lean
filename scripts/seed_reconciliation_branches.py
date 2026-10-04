#!/usr/bin/env python3
"""Seed a published dataset bundle with two-refactor reconciliation branches.

The evaluation container is built by `docker/evaluation/Dockerfile.shared-cache`,
which unconditionally runs `rm -rf .git && git init`.  A `.git` directory placed
in a dataset's `stripped` tree is therefore destroyed at materialization, so the
`main` / `other-refactor` branches the reconciliation prompt describes cannot be
shipped in the source tree.  They have to be rebuilt inside the image, which
`Dockerfile.shared-cache-reconciliation` does from two patch files.

This script stages those patch files into the bundle and records them, with
their hashes and provenance, on each repository's preprocessing report:

    published_repository.variants.stripped.materialization.reconciliation
      main:  {path, sha256, source_run, source_model}
      other: {path, sha256, source_run, source_model}

Model results are read from each run's `preds.json`, never from the per-repo
`round_0_patch.diff` files: those are an incomplete on-disk cache and are absent
for many repositories whose results are perfectly good.

Every patch is verified to apply against the bundle's own stripped tree using
the exact `git init` sequence the Dockerfile uses, so a clean run here means the
image build cannot fail on a bad patch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

KIND = "leanlean_reconciliation_seeding"
ROLES = ("main", "other")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def _load_preds(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: preds.json must be a mapping")
    return dict(payload)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _verify_patches(stripped: Path, patches: Mapping[str, Path], instance_id: str) -> None:
    """Replay the Dockerfile's init sequence and seed both branches for real."""

    with tempfile.TemporaryDirectory(prefix=f"seed-verify-{instance_id}-") as temporary:
        work = Path(temporary) / "tree"
        shutil.copytree(stripped, work, symlinks=True)
        shutil.rmtree(work / ".git", ignore_errors=True)
        for command in (
            ("init", "-q"),
            ("config", "user.name", "A U Thor"),
            ("config", "user.email", "author@example.com"),
        ):
            result = _git(work, *command)
            if result.returncode != 0:
                raise RuntimeError(f"{instance_id}: git {command[0]} failed: {result.stderr}")
        (work / ".gitignore").write_text(".lake/\n")
        for command in (("add", "."), ("commit", "-qm", "init")):
            result = _git(work, *command)
            if result.returncode != 0:
                raise RuntimeError(f"{instance_id}: git {command[0]} failed: {result.stderr}")
        init_commit = _git(work, "rev-parse", "HEAD").stdout.strip()

        steps = (
            ("branch", "-M", "main"),
            ("checkout", "-q", "-b", "other-refactor"),
            ("apply", "--whitespace=nowarn", str(patches["other"])),
            ("add", "-A"),
            ("commit", "-qm", "refactor"),
            ("checkout", "-q", "main"),
            ("apply", "--whitespace=nowarn", str(patches["main"])),
            ("add", "-A"),
            ("commit", "-qm", "refactor"),
        )
        for command in steps:
            result = _git(work, *command)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{instance_id}: seeding step 'git {' '.join(command)}' failed:\n"
                    f"{result.stderr}"
                )

        head = _git(work, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if head != "main":
            raise RuntimeError(f"{instance_id}: HEAD settled on {head!r}, expected 'main'")
        for branch in ("main", "other-refactor"):
            if _git(work, "rev-parse", "--verify", branch).returncode != 0:
                raise RuntimeError(f"{instance_id}: branch {branch} is missing after seeding")
            base = _git(work, "merge-base", branch, init_commit).stdout.strip()
            if base != init_commit:
                raise RuntimeError(f"{instance_id}: {branch} does not descend from the init commit")
        if _git(work, "diff", "--quiet", "main", "other-refactor").returncode == 0:
            raise RuntimeError(f"{instance_id}: the two refactors are identical")


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, Mapping):
        raise ValueError("seeding config must be a YAML mapping")
    if config.get("kind") != KIND or config.get("schema_version") != 1:
        raise ValueError("unsupported reconciliation seeding config")

    dataset_path = (REPO_ROOT / str(config["dataset"])).resolve()
    if not dataset_path.is_relative_to((REPO_ROOT / "datasets").resolve()):
        raise ValueError("dataset must live under datasets/")
    dataset_root = dataset_path.parent
    dataset = yaml.safe_load(dataset_path.read_text())
    if dataset.get("schema_version") != 2:
        raise ValueError("seeding requires a self-contained schema-v2 bundle")

    sources = _mapping(config.get("sources"), "sources")
    preds: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for role in ROLES:
        record = _mapping(sources.get(role), f"sources.{role}")
        preds_path = (REPO_ROOT / str(record["preds"])).resolve()
        preds[role] = _load_preds(preds_path)
        provenance[role] = {
            "preds": str(preds_path.relative_to(REPO_ROOT)),
            "run": str(record.get("run") or ""),
            "model": str(record.get("model") or ""),
        }

    requested = list(config.get("repositories") or [])
    rows = {str(row["id"]): row for row in dataset.get("repositories", [])}
    unknown = sorted(set(requested) - set(rows))
    if unknown:
        raise ValueError(f"repositories absent from the bundle: {unknown}")

    seeded: list[dict[str, Any]] = []
    for instance_id in requested:
        repository_dir = dataset_root / "repos" / instance_id
        stripped = repository_dir / "stripped"
        if not stripped.is_dir():
            raise ValueError(f"{instance_id}: stripped tree is missing")

        seed_dir = repository_dir / "reconciliation"
        seed_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for role in ROLES:
            entry = preds[role].get(instance_id)
            patch = (entry or {}).get("model_patch") or ""
            if not patch.strip():
                raise ValueError(
                    f"{instance_id}: {role} source has no model_patch in "
                    f"{provenance[role]['preds']}"
                )
            target = seed_dir / f"{role}.patch"
            target.write_text(patch)
            written[role] = target
            digests[role] = _sha256_text(patch)

        _verify_patches(stripped, written, instance_id)

        report_path = repository_dir / "prod-strip-report.json"
        report = json.loads(report_path.read_text())
        variants = report["published_repository"]["variants"]
        materialization = variants["stripped"].get("materialization")
        if not isinstance(materialization, Mapping):
            raise ValueError(f"{instance_id}: stripped variant has no materialization record")
        updated = copy.deepcopy(dict(materialization))
        updated["reconciliation"] = {
            role: {
                "path": str(written[role].relative_to(REPO_ROOT)),
                "sha256": digests[role],
                "source_run": provenance[role]["run"],
                "source_model": provenance[role]["model"],
                "source_preds": provenance[role]["preds"],
            }
            for role in ROLES
        }
        variants["stripped"]["materialization"] = updated
        report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        # Reports are hardlinked across dataset bundles, including the canonical
        # benchmark dataset.  An in-place write would mutate every bundle sharing
        # the inode and break their report_sha256 pins; write a temporary file and
        # rename it over the path so only this bundle's link is replaced.
        temporary = report_path.with_name(report_path.name + ".seed-tmp")
        temporary.write_text(report_text)
        os.replace(temporary, report_path)

        rows[instance_id]["report_sha256"] = _sha256_text(report_text)
        seeded.append(
            {
                "id": instance_id,
                "main_sha256": digests["main"],
                "other_sha256": digests["other"],
            }
        )

    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False))
    receipt = {
        "kind": KIND,
        "schema_version": 1,
        "dataset": str(dataset_path.relative_to(REPO_ROOT)),
        "sources": provenance,
        "repositories": seeded,
    }
    (dataset_root / "reconciliation-seeding.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    receipt = run(arguments.config)
    print(f"seeded {len(receipt['repositories'])} repositories")
    for row in receipt["repositories"]:
        print(f"  {row['id']:32} main={row['main_sha256'][:12]} other={row['other_sha256'][:12]}")


if __name__ == "__main__":
    main()
