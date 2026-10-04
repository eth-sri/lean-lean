#!/usr/bin/env python3
"""Isolate the reproving-ablation endpoint sources from the packaged holdouts.

Each of the 15 holdout repositories has two arms:

- baseline: the Stage-2 sorried source before model compression
  (``<packaged>/repos/<id>/uncompressed/source``), copied unchanged;
- post-compression: the submitted compressed endpoint with the holdout restored
  as ``sorry`` (``<packaged>/repos/<id>/source``), minus the benchmark
  scaffolding the reprover must not see: the comparator's challenge module,
  ``comparator.json`` and ``proof_length.py``.

Every isolated tree is audited: no scaffolding file remains and no Lean file
imports the challenge module. The receipt pins each tree with
``source_tree_sha256``. ``--check`` rebuilds everything in a temporary directory
and compares it against an existing receipt without writing to the dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leanlean.preprocessing.standardized_repositories import (  # noqa: E402
    source_tree_sha256,
    source_tree_stats,
)

SCAFFOLDING = ("comparator.json", "proof_length.py")
ARMS = {"baseline": "uncompressed/source", "post-compression": "source"}


def _challenge_path(source: Path) -> str:
    module = json.loads((source / "comparator.json").read_text())["challenge_module"]
    suffix = module.replace(".", "/") + ".lean"
    matches = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*.lean")
        if ".lake" not in path.relative_to(source).parts
        and path.relative_to(source).as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"{source}: challenge module {module!r} matches {matches}")
    return matches[0]


def _audit(tree: Path, challenge_module: str | None) -> dict:
    names = set(SCAFFOLDING)
    forbidden_files = sorted(
        path.relative_to(tree).as_posix()
        for path in tree.rglob("*")
        if path.is_file() and path.name in names
    )
    forbidden_imports = []
    if challenge_module:
        pattern = re.compile(rf"^\s*import\s+.*\b{re.escape(challenge_module)}\b", re.M)
        forbidden_imports = sorted(
            path.relative_to(tree).as_posix()
            for path in tree.rglob("*.lean")
            if pattern.search(path.read_text(errors="replace"))
        )
    return {
        "file_count": source_tree_stats(tree)["file_count"],
        "forbidden_files": forbidden_files,
        "forbidden_imports": forbidden_imports,
        "passed": not forbidden_files and not forbidden_imports,
    }


def isolate(packaged: Path, repositories: list[str], output: Path) -> dict:
    arms: dict[str, dict] = {}
    for arm, relative_source in ARMS.items():
        records = {}
        for repository in repositories:
            source = packaged / "repos" / repository / relative_source
            destination = output / "repos" / repository / arm / "source"
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copytree(source, destination, symlinks=True)
            removed: list[str] = []
            challenge_module = None
            if arm == "post-compression":
                challenge_module = json.loads((source / "comparator.json").read_text())["challenge_module"]
                removed = [_challenge_path(source), *SCAFFOLDING]
                for relative in removed:
                    (destination / relative).unlink()
            audit = _audit(destination, challenge_module)
            if not audit["passed"]:
                raise RuntimeError(f"{repository}/{arm} failed isolation audit: {audit}")
            records[repository] = {
                "audit": audit,
                "removed": removed,
                "source": str(source),
                "tree_sha256": source_tree_sha256(destination),
            }
        arms[arm] = {"repositories": records}
    return {
        "kind": "leanlean_isolated_endpoint_packaging",
        "schema_version": 1,
        "repository_count_per_arm": len(repositories),
        "arms": arms,
    }


def check(receipt_path: Path, packaged: Path) -> int:
    receipt = json.loads(receipt_path.read_text())
    repositories = sorted(receipt["arms"]["baseline"]["repositories"])
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = isolate(packaged, repositories, Path(directory))
    mismatches = []
    for arm, data in rebuilt["arms"].items():
        for repository, record in data["repositories"].items():
            expected = receipt["arms"][arm]["repositories"][repository]
            for key in ("tree_sha256", "removed", "audit"):
                if record[key] != expected[key]:
                    mismatches.append((arm, repository, key, record[key], expected[key]))
    for mismatch in mismatches:
        print("MISMATCH", *mismatch)
    cells = sum(len(data["repositories"]) for data in rebuilt["arms"].values())
    print(f"{cells - len({m[:2] for m in mismatches})}/{cells} isolated trees match {receipt_path}")
    return 1 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packaged", type=Path, default=ROOT / "datasets/compressed-sorried-holdouts-20260912-r1")
    parser.add_argument("--check", type=Path, metavar="RECEIPT", help="verify an existing isolation receipt")
    parser.add_argument("--output", type=Path, help="new dataset directory to write repos/ and the receipt into")
    parser.add_argument("repositories", nargs="*")
    args = parser.parse_args()
    if args.check:
        return check(args.check, args.packaged)
    if not args.output or not args.repositories:
        parser.error("--output and repositories are required unless --check is given")
    receipt = isolate(args.packaged, args.repositories, args.output)
    (args.output / "isolation-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"isolated {len(args.repositories)} repositories x {len(ARMS)} arms into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
