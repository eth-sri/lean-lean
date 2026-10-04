#!/usr/bin/env python3
"""Download the published LeanLean benchmark from Hugging Face and prepare it for the pipeline.

The dataset (eth-sri/lean-lean) is fetched at its pinned revision into
datasets/leanlean_20260914. The release holds metadata.jsonl and, per
repository, raw/, stripped/, comparator/{Challenge.lean, comparator.json} and
dependency-graph.json. Two steps then make it usable by evaluation and
postprocessing:

1. Hugging Face does not store file modes, but the tree hashes include them, so
   the modes in data/datasets/leanlean_20260914/file-modes.json are restored.
2. The pipeline's manifests are written next to the downloaded files:
   dataset.yaml, repository-database.json and, per repository,
   prod-strip-report.json, comparator/manifest.json and
   comparator/registry-record.json. They are derived from the release, the
   Palomar mapping in data/palomar/ and data/datasets/leanlean_20260914/overlay.json,
   which pins every tree and graph hash and holds the few values the release
   does not carry. registry-record.json is the full Palomar registry record,
   downloaded from the registry by the hash the mapping pins and cached under
   .cache/palomar-records/ for offline reruns. .leanlean-materialized.json
   lists these files.

Every raw and stripped tree is verified against its pinned hash, and the
dataset is loaded once as a check.

Usage:
    uv run python scripts/fetch_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.dataset_bundle import DATASET_KIND, load_dataset  # noqa: E402
from leanlean.identifiers import LEGACY_PREFIX, PREFIX  # noqa: E402
from leanlean.palomar_comparator import (  # noqa: E402
    COMPARATOR_BUNDLE_MANIFEST,
    COMPARATOR_BUNDLE_SCHEMA,
    VALIDATION_SCHEMA,
)
from leanlean.preprocessing.graph_artifact import GRAPH_SCHEMAS, graph_sha256  # noqa: E402
from leanlean.preprocessing.palomar_sources import _ensure_registry_record  # noqa: E402
from leanlean.preprocessing.repositories import (  # noqa: E402
    file_sha256,
    repository_definition_sha256,
)
from leanlean.preprocessing.standardized_repositories import (  # noqa: E402
    STANDARDIZATION_CONTRACT,
    STANDARDIZED_DATABASE_KIND,
    STANDARDIZED_DATABASE_SCHEMA_VERSION,
    load_standardized_repository_database,
    source_tree_sha256,
    source_tree_stats,
)

RECORDS = ROOT / "data/datasets/leanlean_20260914"
MODES = RECORDS / "file-modes.json"
OVERLAY = RECORDS / "overlay.json"
OUTPUT = ROOT / "datasets/leanlean_20260914"
MARKER = ".leanlean-materialized.json"
# Palomar registry records downloaded by their mapping-pinned hash; kept for offline reruns.
RECORD_CACHE = ".cache/palomar-records"
REPORT_KIND = "leanlean_published_preprocessing_report"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _write(path: Path, content: bytes) -> str:
    if not path.is_file() or path.read_bytes() != content:
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _relative_to_project(path: str, project_root: str | None) -> str:
    if not project_root:
        return path
    return PurePosixPath(path).relative_to(PurePosixPath(project_root.strip("/"))).as_posix()


def restore_modes(root: Path, record: dict) -> int:
    default = int(record["default_mode"], 8)
    modes = {path: int(mode, 8) for path, mode in record["modes"].items()}
    count = 0
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if not (Path(directory) == root and d == ".cache")]
        for name in files:
            path = Path(directory) / name
            path.chmod(modes.get(path.relative_to(root).as_posix(), default))
            count += 1
    return count


def _claim_output(output: Path, overlay: dict[str, Any]) -> None:
    """Refuse to write into a dataset this script did not fetch at the pinned revision."""

    if not output.is_relative_to(ROOT / "datasets"):
        raise SystemExit(f"--output must lie under {ROOT / 'datasets'}: the manifests pin repo-relative paths")
    marker = output / MARKER
    if output.is_dir() and any(output.iterdir()):
        previous = json.loads(marker.read_text()).get("revision") if marker.is_file() else None
        if previous != overlay["revision"]:
            found = f"revision {previous}" if previous else "a dataset this script did not fetch"
            raise SystemExit(f"{output} holds {found}; move it aside to fetch {overlay['revision'][:12]}")
        return
    output.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(_json_bytes({"dataset": overlay["dataset"], "revision": overlay["revision"],
                                    "status": "incomplete"}))


def _repository(
    output: Path,
    meta: dict[str, Any],
    pins: dict[str, Any],
    overlay: dict[str, Any],
    target: dict[str, Any],
    result: dict[str, Any],
    registry_url: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Write one repository's Comparator manifest, registry record and report; return its database row."""

    instance_id = meta["id"]
    directory = output / "repos" / instance_id
    relative = directory.relative_to(ROOT).as_posix()
    written: dict[str, str] = {}

    trees = {}
    for variant in ("raw", "stripped"):
        tree = directory / variant
        if source_tree_sha256(tree) != pins[variant]["tree_sha256"]:
            raise ValueError(f"{instance_id}: {variant}/ differs from its pinned tree hash")
        trees[variant] = {"tree_sha256": pins[variant]["tree_sha256"], **source_tree_stats(tree)}

    graph = json.loads((directory / "dependency-graph.json").read_text())
    if (
        graph.get("schema") not in GRAPH_SCHEMAS
        or graph.get("repository") != instance_id
        or graph_sha256(graph) != pins["dependency_graph_sha256"]
    ):
        raise ValueError(f"{instance_id}: dependency-graph.json differs from its pinned hash")
    names = [node["name"] for node in graph["nodes"]]
    kept = set(graph["policies"]["production"]["kept"])

    comparator = directory / "comparator"
    challenge = (comparator / "Challenge.lean").read_bytes()
    configuration = (comparator / "comparator.json").read_bytes()
    if (
        hashlib.sha256(challenge).hexdigest() != target["registered_challenge_sha256"]
        or hashlib.sha256(configuration).hexdigest() != target["registered_comparator_config_sha256"]
    ):
        raise ValueError(f"{instance_id}: Comparator files differ from the Palomar mapping")
    palomar_id = result["palomar_id"]
    toolchain = meta["lean_toolchain"]
    project_root = target["project_path"]
    declarations = sorted(result["theorem_names"] + result["definition_names"])
    if (
        target["palomar_ids"] != [palomar_id]
        or result["lean_toolchain"] != toolchain
        or meta["repository_url"] != f"https://github.com/{target['repository']}"
        or meta["commit"] != target["commit"]
        or meta["protected_declarations"] != len(declarations)
        or json.loads(configuration)["solution_module"] != target["registered_solution_module"]
    ):
        raise ValueError(f"{instance_id}: metadata.jsonl or comparator.json differs from the Palomar mapping")

    # The Comparator contract pins the full Palomar registry record, which the
    # release leaves out: download it by the hash the Palomar mapping pins.
    record_path = _ensure_registry_record(result, registry_url=registry_url,
                                          record_dir=output / RECORD_CACHE, timeout=120)
    record_sha256 = _write(comparator / "registry-record.json", record_path.read_bytes())
    if record_sha256 != result["record_sha256"]:
        raise ValueError(f"{instance_id}: Palomar registry record differs from the mapping")
    written[f"repos/{instance_id}/comparator/registry-record.json"] = record_sha256
    manifest = {
        # The published spelling (leanlean.identifiers), so that the manifest and
        # with it the Comparator contract hash equal the paper's byte for byte.
        "schema": LEGACY_PREFIX + COMPARATOR_BUNDLE_SCHEMA.removeprefix(PREFIX),
        "instance_id": instance_id,
        "palomar_id": palomar_id,
        "lean_toolchain": toolchain,
        "source": {"repository": target["repository"], "commit": target["commit"],
                   "project_root": project_root, "archive_sha256": pins["source_archive_sha256"]},
        "registry_record": {"path": "registry-record.json", "sha256": record_sha256},
        "challenge": {"path": "Challenge.lean", "sha256": target["registered_challenge_sha256"],
                      "registered_path": target["registered_challenge_path"],
                      "source_path": target["challenge_source"],
                      "module": target["registered_challenge_module"]},
        "configuration": {"path": "comparator.json", "sha256": target["registered_comparator_config_sha256"],
                          "registered_path": target["registered_comparator_config_path"],
                          "source_path": _relative_to_project(
                              target["registered_comparator_config_path"], project_root)},
        "solution": {"registered_path": target["registered_solution_path"],
                     "source_path": target["solution_source"],
                     "module": target["registered_solution_module"],
                     "sha256": target["registered_solution_sha256"]},
    }
    manifest_sha256 = _write(comparator / COMPARATOR_BUNDLE_MANIFEST, _json_bytes(manifest))
    written[f"repos/{instance_id}/comparator/{COMPARATOR_BUNDLE_MANIFEST}"] = manifest_sha256
    comparator_pin = {"path": f"{relative}/comparator", "manifest": COMPARATOR_BUNDLE_MANIFEST,
                      "manifest_sha256": manifest_sha256}

    registered = {
        "registered_challenge_module": target["registered_challenge_module"],
        "registered_challenge_path": target["registered_challenge_path"],
        "registered_challenge_sha256": target["registered_challenge_sha256"],
        "registered_comparator_config_path": target["registered_comparator_config_path"],
        "registered_comparator_config_sha256": target["registered_comparator_config_sha256"],
        "registered_solution_path": target["registered_solution_path"],
        "registered_solution_sha256": target["registered_solution_sha256"],
        "solution_source": target["solution_source"],
    }
    provenance = {
        "kind": "palomar_registry_main_results",
        "policy": "registry_exact_names_without_signature_scout",
        "registry": overlay["palomar_mapping"]["path"],
        "registry_sha256": overlay["palomar_mapping"]["sha256"],
        "registry_record": {"path": f"{relative}/comparator/registry-record.json", "sha256": record_sha256},
        "source_archive": {"repository": target["repository"], "commit": target["commit"],
                           "sha256": pins["source_archive_sha256"]},
        "comparator": comparator_pin,
        "palomar_ids": [palomar_id],
        "repository_roles": [result["repository_role"]],
        "registry_toolchains": [result["lean_toolchain"]],
        "source_toolchain": toolchain,
        "solution_module": target["registered_solution_module"],
        **registered,
        "original_registered_declarations": declarations,
        "override": None,
        "resolution_policy": "registry_solution_path_and_comparator_module_exact_fail_closed",
        "admission_audit": "Lean.collectAxioms_reject_sorryAx",
    }
    entry = target["registered_solution_module"]
    scope = {"entry_module": entry, "target_dir": "", "build_target": f"+{entry}", "exclude_dirs": []}
    row = {
        "instance_id": instance_id,
        "source": {
            "repository_url": meta["repository_url"],
            "commit": meta["commit"],
            "toolchain": toolchain,
            "raw_repository": {"path": f"{relative}/raw", **trees["raw"]},
            "standardization": {
                "contract": STANDARDIZATION_CONTRACT,
                "adapter": "palomar_registry",
                "clean_build_passed": True,
                "source_run_id": overlay["source_run_id"],
                "project_root": project_root,
                "entry_module": entry,
                "comparator": comparator_pin,
                **registered,
            },
        },
        "scopes": scope,
        "protected": {"declarations": declarations, "provenance": provenance},
    }

    contract_sha256 = pins["comparator"]["contract_sha256"]
    final_verification = {
        "passed": True,
        "status": "passed",
        "checks": {"declared_source_verifier": True, "stripped_image": True, "stripped_image_labels": True},
        "declared_verifier": {
            "applicable": True, "passed": True, "returncode": 0,
            "command": "lean_verify", "engine": "leanprover/comparator",
            "contract_schema": VALIDATION_SCHEMA, "contract_sha256": contract_sha256,
            "output_sha256": pins["comparator"]["output_sha256"],
            "invocation_policy": "single_final_comparator", "invocations": 1,
            "timeout_policy": "pinned_container_lifetime", "timeout_seconds": 43200,
        },
    }
    environment = overlay["shared_environments"][pins["shared_environment"]]
    raw_tokens, stripped_tokens = pins["raw"]["lean_tokens"], meta["stripped_tokens"]
    raw_words, stripped_words = pins["raw"]["words"], pins["stripped"]["words"]
    graph_record = {"schema": graph["schema"], "path": "dependency-graph.json",
                    "sha256": pins["dependency_graph_sha256"], "counts": graph["counts"]}
    published = {
        "id": instance_id,
        "source": {"adapter": "palomar_registry", "commit": meta["commit"],
                   "repository": meta["repository_url"], "toolchain": toolchain},
        "scope": scope,
        "protected": {
            "count": len(declarations),
            "declarations_sha256": hashlib.sha256(
                json.dumps(declarations, separators=(",", ":")).encode()).hexdigest(),
            "provenance": "source_database",
        },
        "metrics": {
            "declarations": {"raw": None, "stripped": pins["declarations_after_module_pruning"],
                             "dropped": len(names) - len(kept)},
            "lean_tokens": {"raw": raw_tokens, "stripped": stripped_tokens,
                            "reduction": round(1 - stripped_tokens / raw_tokens, 4)},
            "words": {"raw": raw_words, "stripped": stripped_words,
                      "reduction": round(1 - stripped_words / raw_words, 4)},
            "preprocessing_seconds": pins["preprocessing_seconds"],
        },
        "preprocessing": {
            "version": overlay["preprocessing"]["version"],
            "source_run_id": pins["source_run_id"],
            "grind_source_form": "original_grind_calls",
            "report": "prod-strip-report.json",
            "dependency_graph": graph_record,
        },
        "variants": {
            "raw": {"format": "standardized_raw_repository", "cache_tree": "raw",
                    **trees["raw"], "lean_tokens": raw_tokens},
            "stripped": {
                "format": "stripped_source_repository",
                "cache_tree": "stripped",
                **trees["stripped"],
                "lean_tokens": stripped_tokens,
                "final_verification": final_verification,
                "task_metadata": {
                    "validation_command": "lean_verify",
                    "validation_engine": "leanprover/comparator",
                    "pre_agent_validation": {"command": "lean_verify", "required_exit_code": 0,
                                             "preprocessing_gate_passed": True,
                                             "contract_sha256": contract_sha256},
                },
                # The paper's prebuilt environments are not part of the release: the
                # loader marks them unavailable and evaluation builds from stripped/.
                "materialization": {
                    "mode": "build_before_agent",
                    "backend": "shared_environment_v1",
                    "image_persistence": "ephemeral",
                    "build_target": scope["build_target"],
                    "shared_environment": {"id": pins["shared_environment"], **environment},
                    "warm_build_cache": {"path": "warm-build", "sha256": pins["warm_build_cache_sha256"]},
                },
            },
        },
    }
    report = {
        "kind": REPORT_KIND,
        "schema_version": 1,
        "instance_id": instance_id,
        "protected_provenance": provenance,
        "final_verification": final_verification,
        "dependency_graph": graph_record,
        # The declarations of raw/ that stripped/ drops, from the published graph.
        "dropped_decls": [name for index, name in enumerate(names) if index not in kept],
        "published_repository": published,
    }
    written[f"repos/{instance_id}/prod-strip-report.json"] = _write(
        directory / "prod-strip-report.json", _json_bytes(report))
    return row, written


def materialize(output: Path, overlay: dict[str, Any]) -> dict[str, str]:
    """Write the pipeline manifests of a downloaded release; return {path: sha256}."""

    mapping_path = ROOT / overlay["palomar_mapping"]["path"]
    if file_sha256(mapping_path) != overlay["palomar_mapping"]["sha256"]:
        raise ValueError(f"{mapping_path} differs from its pinned hash")
    mapping = json.loads(mapping_path.read_text())
    targets = {row["benchmark_id"]: row for row in mapping["targets"]}
    results = {row["palomar_id"]: row for row in mapping["results"]}
    metadata = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines() if line.strip()]
    pins = overlay["repositories"]
    ids = [row["id"] for row in metadata]
    if len(ids) != len(set(ids)) or set(ids) != set(pins):
        raise ValueError("metadata.jsonl lists other repositories than the overlay")

    rows, written = [], {}
    for meta in metadata:
        target = targets[meta["id"]]
        row, files = _repository(output, meta, pins[meta["id"]], overlay, target,
                                 results[target["palomar_ids"][0]], mapping["snapshot"]["registry_url"])
        rows.append(row)
        written.update(files)

    database = {
        "kind": STANDARDIZED_DATABASE_KIND,
        "schema_version": STANDARDIZED_DATABASE_SCHEMA_VERSION,
        "dataset_id": overlay["dataset_id"],
        "dataset_version": overlay["dataset_version"],
        "repositories": rows,
    }
    written["repository-database.json"] = _write(output / "repository-database.json", _json_bytes(database))
    manifest = {
        "kind": DATASET_KIND,
        "schema_version": 2,
        "id": overlay["dataset_id"],
        "version": overlay["dataset_version"],
        "default_variant": "stripped",
        "repository_count": len(rows),
        "repositories": [{"id": row["instance_id"], "report": f"repos/{row['instance_id']}/prod-strip-report.json",
                          "report_sha256": written[f"repos/{row['instance_id']}/prod-strip-report.json"]}
                         for row in rows],
        "discarded_repositories": [],
        "repository_database": {"path": "repository-database.json",
                                "sha256": written["repository-database.json"],
                                "definition_sha256": repository_definition_sha256(database)},
        "preprocessing": dict(overlay["preprocessing"]),
        "release": {"dataset": overlay["dataset"], "revision": overlay["revision"],
                    "overlay": OVERLAY.relative_to(ROOT).as_posix(), "overlay_sha256": file_sha256(OVERLAY)},
    }
    written["dataset.yaml"] = _write(
        output / "dataset.yaml", yaml.safe_dump(manifest, sort_keys=False, width=100).encode())
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="dataset directory (default: %(default)s)")
    parser.add_argument("--no-download", action="store_true",
                        help="prepare a copy of the pinned revision already in --output")
    args = parser.parse_args()
    modes = json.loads(MODES.read_text())
    overlay = json.loads(OVERLAY.read_text())
    if (modes["dataset"], modes["revision"]) != (overlay["dataset"], overlay["revision"]):
        raise SystemExit(f"{MODES} and {OVERLAY} pin different releases")
    output = args.output.resolve()
    _claim_output(output, overlay)

    if not args.no_download:
        from huggingface_hub import snapshot_download

        print(f"downloading {overlay['dataset']}@{overlay['revision'][:12]} -> {output}")
        snapshot_download(repo_id=overlay["dataset"], repo_type="dataset",
                          revision=overlay["revision"], local_dir=output)
    restore_modes(output, modes)
    written = materialize(output, overlay)
    (output / MARKER).write_bytes(_json_bytes({
        "dataset": overlay["dataset"], "revision": overlay["revision"], "status": "complete",
        "overlay_sha256": file_sha256(OVERLAY), "files": written,
    }))
    print(f"restored modes of {restore_modes(output, modes)} files and wrote {len(written)} manifests ({MARKER})")
    load_standardized_repository_database(output / "repository-database.json", repo_root=ROOT)
    dataset = load_dataset(output / "dataset.yaml", verify_trees=True)
    print(f"verified {len(dataset['repositories'])} repositories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
