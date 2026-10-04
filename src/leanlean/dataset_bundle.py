"""Load compact self-contained datasets into the established runtime shape."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import yaml

from leanlean.identifiers import same_identifier
from leanlean.preprocessing.graph_artifact import (
    GRAPH_SCHEMAS,
    graph_sha256,
)
from leanlean.preprocessing.repositories import file_sha256
from leanlean.preprocessing.standardized_repositories import (
    source_tree_sha256,
)
from leanlean.shared_cache import artifact_tree_sha256


DATASET_KIND = "leanlean_dataset"
BUNDLE_SCHEMA_VERSION = 2
IMAGE_MANIFEST_KIND = "leanlean_docker_image"
BUNDLE_GRAPH_SCHEMAS = {"preprocessing_dependency_graph_v1", "preprocessing_dependency_graph_v2"}


def _load_bundle_graph(path: Path) -> dict[str, Any]:
    """Frozen bundles retain their graph schema; never migrate pinned bytes.

    Both schemas are opaque evidence here. Callers still check repository,
    the full canonical hash, the embedded hash, and all published counts. The
    Hugging Face release publishes graphs without the embedded hash and the
    preprocessing image identity; their record pins the hash of that form.
    """
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") not in BUNDLE_GRAPH_SCHEMAS:
        raise ValueError("unsupported bundled dependency graph schema")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _member(root: Path, value: Any, name: str) -> Path:
    relative = Path(_text(value, name))
    if relative.is_absolute():
        raise ValueError(f"{name} must be bundle-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{name} escaped the dataset bundle")
    return path


def _pinned_file(root: Path, record: Mapping[str, Any], name: str) -> Path:
    path = _member(root, record.get("path"), f"{name}.path")
    expected = _text(record.get("sha256"), f"{name}.sha256")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{name} drift")
    return path


def _prebuilt_environment_absent(
    repository_dir: Path, dataset_root: Path, materialization: Mapping[str, Any]
) -> bool:
    """True when neither the warm build nor the shared environment archive is present."""

    warm = materialization.get("warm_build_cache")
    environment = materialization.get("shared_environment")
    archive = environment.get("archive") if isinstance(environment, Mapping) else None
    if not isinstance(warm, Mapping) or not isinstance(archive, Mapping):
        return False
    paths = [repository_dir / str(warm.get("path")), dataset_root / str(archive.get("path"))]
    present = [path.exists() for path in paths]
    if any(present) and not all(present):
        return False  # partially present: let the full checks report the drift
    return not any(present)


def _localized_path(
    repository_dir: Path,
    value: Any,
    expected: str,
    name: str,
) -> Path:
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}")
    path = _member(repository_dir, value, name)
    if not path.exists():
        raise ValueError(f"{name} is missing")
    return path


def _hydrate_repository(
    dataset_root: Path,
    record: Mapping[str, Any],
    *,
    verify_trees: bool,
) -> dict[str, Any]:
    instance_id = _text(record.get("id"), "dataset.repositories[].id")
    repository_dir = (dataset_root / "repos" / instance_id).resolve()
    if repository_dir.parent != (dataset_root / "repos").resolve():
        raise ValueError(f"unsafe repository ID: {instance_id!r}")
    expected_report = f"repos/{instance_id}/prod-strip-report.json"
    if record.get("report") != expected_report:
        raise ValueError(f"{instance_id}: noncanonical report path")
    report_path = _pinned_file(
        dataset_root,
        {"path": record.get("report"), "sha256": record.get("report_sha256")},
        f"{instance_id}.report",
    )
    report = json.loads(report_path.read_text())
    if not isinstance(report, Mapping):
        raise ValueError(f"{instance_id}: malformed preprocessing report")
    row = copy.deepcopy(
        dict(
            _mapping(
                report.get("published_repository"),
                f"{instance_id}.published_repository",
            )
        )
    )
    if row.get("id") != instance_id:
        raise ValueError(f"{instance_id}: published repository identity drift")

    variants = _mapping(row.get("variants"), f"{instance_id}.variants")
    if set(variants) not in ({"stripped"}, {"raw", "stripped"}):
        raise ValueError(f"{instance_id}: variants must be stripped, optionally with raw")
    # lean-strip datasets publish only the stripped tree; older bundles also
    # carry the isolated raw tree.
    raw = dict(_mapping(variants["raw"], f"{instance_id}.raw")) if "raw" in variants else None
    stripped = dict(
        _mapping(variants.get("stripped"), f"{instance_id}.stripped")
    )
    raw_path = (
        _localized_path(repository_dir, raw.get("cache_tree"), "raw", f"{instance_id}.raw")
        if raw is not None
        else None
    )
    stripped_path = _localized_path(
        repository_dir,
        stripped.get("cache_tree"),
        "stripped",
        f"{instance_id}.stripped",
    )
    if verify_trees:
        if raw is not None and source_tree_sha256(raw_path) != raw.get("tree_sha256"):
            raise ValueError(f"{instance_id}: raw tree drift")
        if source_tree_sha256(stripped_path) != stripped.get("tree_sha256"):
            raise ValueError(f"{instance_id}: stripped tree drift")

    image: Mapping[str, Any] | None = None
    image_path: Path | None = None
    materialization = stripped.get("materialization")
    if isinstance(materialization, Mapping) and _prebuilt_environment_absent(
        repository_dir, dataset_root, materialization
    ):
        # The published bundle omits the prebuilt shared environment and warm
        # build; evaluation must build the image from the source tree instead
        # (a dataset config's image_materialization).
        stripped["materialization"] = {
            **materialization,
            "unavailable": "prebuilt environment is not part of this copy of the bundle",
        }
    elif isinstance(materialization, Mapping):
        if any(
            key in stripped
            for key in (
                "image",
                "image_id",
                "image_manifest",
                "image_manifest_sha256",
            )
        ):
            raise ValueError(
                f"{instance_id}: ephemeral artifact published image metadata"
            )
        warm = dict(
            _mapping(
                materialization.get("warm_build_cache"),
                f"{instance_id}.warm_build_cache",
            )
        )
        warm_path = _localized_path(
            repository_dir,
            warm.get("path"),
            "warm-build",
            f"{instance_id}.warm_build_cache",
        )
        if verify_trees and artifact_tree_sha256(warm_path) != warm.get("sha256"):
            raise ValueError(f"{instance_id}: warm build cache drift")

        environment = dict(
            _mapping(
                materialization.get("shared_environment"),
                f"{instance_id}.shared_environment",
            )
        )
        environment_id = _text(
            environment.get("id"), f"{instance_id}.shared_environment.id"
        )
        archive = dict(
            _mapping(
                environment.get("archive"),
                f"{instance_id}.shared_environment.archive",
            )
        )
        expected_archive = (
            f"environments/{environment_id}/image.tar.zst"
        )
        archive_path = _localized_path(
            dataset_root,
            archive.get("path"),
            expected_archive,
            f"{instance_id}.shared_environment.archive",
        )
        if verify_trees and file_sha256(archive_path) != archive.get("sha256"):
            raise ValueError(f"{instance_id}: shared environment archive drift")
        expected_bytes = archive.get("bytes")
        if (
            isinstance(expected_bytes, int)
            and archive_path.stat().st_size != expected_bytes
        ):
            raise ValueError(
                f"{instance_id}: shared environment archive size drift"
            )
        warm["path"] = str(warm_path)
        archive["path"] = str(archive_path)
        environment["archive"] = archive
        materialization = dict(materialization)
        materialization["warm_build_cache"] = warm
        materialization["shared_environment"] = environment
        stripped["materialization"] = materialization
    else:
        image_path = _localized_path(
            repository_dir,
            stripped.get("image_manifest"),
            "docker-image.json",
            f"{instance_id}.image_manifest",
        )
        if file_sha256(image_path) != stripped.get("image_manifest_sha256"):
            raise ValueError(f"{instance_id}: Docker image manifest drift")
        image = json.loads(image_path.read_text())
        if (
            not isinstance(image, Mapping)
            or not same_identifier(image.get("kind"), IMAGE_MANIFEST_KIND)
            or image.get("schema_version") != 1
            or image.get("repository") != instance_id
            or image.get("variant") != "stripped"
            or image.get("tag") != stripped.get("image")
            or image.get("image_id") != stripped.get("image_id")
        ):
            raise ValueError(
                f"{instance_id}: Docker image manifest contract drift"
            )
        labels = _mapping(image.get("labels"), f"{instance_id}.image.labels")
        expected_labels = {
            "org.openai.leanlean.instance_id": instance_id,
            "org.openai.leanlean.repo_variant": "stripped",
            "org.openai.leanlean.run_id": image.get("source_run_id"),
        }
        if not all(
            labels.get(key) == value for key, value in expected_labels.items()
        ):
            raise ValueError(f"{instance_id}: Docker image labels drift")

    preprocessing = dict(
        _mapping(row.get("preprocessing"), f"{instance_id}.preprocessing")
    )
    if (
        image is not None
        and image.get("source_run_id") != preprocessing.get("source_run_id")
    ):
        raise ValueError(f"{instance_id}: Docker image source run drift")

    graph_record = dict(
        _mapping(
            preprocessing.get("dependency_graph"),
            f"{instance_id}.dependency_graph",
        )
    )
    graph_path = _localized_path(
        repository_dir,
        graph_record.get("path"),
        "dependency-graph.json",
        f"{instance_id}.dependency_graph",
    )
    graph = _load_bundle_graph(graph_path)
    if (
        graph.get("schema") not in GRAPH_SCHEMAS
        or graph.get("repository") != instance_id
        or graph_sha256(graph) != graph_record.get("sha256")
        or graph.get("sha256", graph_record.get("sha256")) != graph_record.get("sha256")
        or graph.get("counts") != graph_record.get("counts")
    ):
        raise ValueError(f"{instance_id}: dependency graph drift")

    stripped["cache_tree"] = str(stripped_path)
    if image_path is not None:
        stripped["image_manifest"] = str(image_path)
    graph_record["path"] = str(graph_path)
    preprocessing["dependency_graph"] = graph_record
    preprocessing["report"] = str(report_path)
    row["variants"] = {"stripped": stripped}
    if raw is not None:
        raw["cache_tree"] = str(raw_path)
        row["variants"]["raw"] = raw
    row["preprocessing"] = preprocessing
    return row


def load_dataset(
    path: Path,
    *,
    verify_trees: bool = True,
    repository_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a schema-v1 manifest or hydrate a self-contained schema-v2 bundle."""

    path = path.resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping")
    dataset = dict(value)
    if not same_identifier(dataset.get("kind"), DATASET_KIND):
        raise ValueError(f"{path}: unsupported dataset kind")
    if dataset.get("schema_version") == 1:
        return dataset
    if dataset.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported dataset schema")

    dataset_root = path.parent
    raw_records = dataset.get("repositories")
    if not isinstance(raw_records, list):
        raise ValueError("dataset.repositories must be a list")
    all_ids = [
        _text(_mapping(record, "dataset.repositories[]").get("id"), "dataset.repositories[].id")
        for record in raw_records
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("dataset contains duplicate repositories")
    selected_ids = list(repository_ids) if repository_ids is not None else None
    if selected_ids is not None:
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected dataset repositories contain duplicates")
        by_id = {
            _text(_mapping(record, "dataset.repositories[]").get("id"), "dataset.repositories[].id"): record
            for record in raw_records
        }
        unknown = sorted(set(selected_ids) - set(by_id))
        if unknown:
            raise ValueError(f"selected repositories are absent from dataset: {unknown}")
        hydrated_records = [by_id[instance_id] for instance_id in selected_ids]
    else:
        hydrated_records = raw_records
    rows = [
        _hydrate_repository(
            dataset_root,
            _mapping(record, "dataset.repositories[]"),
            verify_trees=verify_trees,
        )
        for record in hydrated_records
    ]
    if selected_ids is None and dataset.get("repository_count") != len(rows):
        raise ValueError("dataset.repository_count is inconsistent")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate repositories")

    discarded_records = dataset.get("discarded_repositories", [])
    if not isinstance(discarded_records, list):
        raise ValueError("dataset.discarded_repositories must be a list")
    discarded: list[str] = []
    for value in discarded_records:
        record = _mapping(value, "dataset.discarded_repositories[]")
        instance_id = _text(
            record.get("id"), "dataset.discarded_repositories[].id"
        )
        expected = f"repos/{instance_id}/prod-strip-report.json"
        if record.get("report") != expected:
            raise ValueError(f"{instance_id}: noncanonical discard report path")
        report_path = _pinned_file(
            dataset_root,
            {"path": record.get("report"), "sha256": record.get("report_sha256")},
            f"{instance_id}.discard_report",
        )
        report = json.loads(report_path.read_text())
        if (
            not isinstance(report, Mapping)
            or report.get("instance_id") != instance_id
            or report.get("discarded") is not True
            or report.get("total_decls") != 0
        ):
            raise ValueError(f"{instance_id}: discard report drift")
        repository_dir = report_path.parent
        if report.get("raw_source_tree") is not None:
            raw_record = _mapping(
                report.get("raw_source_tree"), f"{instance_id}.raw_source_tree"
            )
            raw_path = _localized_path(
                repository_dir,
                raw_record.get("path"),
                "raw",
                f"{instance_id}.raw",
            )
            if verify_trees and source_tree_sha256(raw_path) != raw_record.get(
                "tree_sha256"
            ):
                raise ValueError(f"{instance_id}: discarded raw tree drift")
        graph_record = _mapping(
            report.get("dependency_graph"),
            f"{instance_id}.dependency_graph",
        )
        graph_path = _localized_path(
            repository_dir,
            graph_record.get("path"),
            "dependency-graph.json",
            f"{instance_id}.dependency_graph",
        )
        graph = _load_bundle_graph(graph_path)
        if (
            graph.get("schema") not in GRAPH_SCHEMAS
            or graph.get("repository") != instance_id
            or graph_sha256(graph) != graph_record.get("sha256")
            or graph.get("sha256", graph_record.get("sha256")) != graph_record.get("sha256")
            or graph.get("counts") != graph_record.get("counts")
        ):
            raise ValueError(f"{instance_id}: discarded dependency graph drift")
        discarded.append(instance_id)
    if len(discarded) != len(set(discarded)) or set(discarded) & set(ids):
        raise ValueError("discarded repository membership is inconsistent")

    repository_database = dict(
        _mapping(dataset.get("repository_database"), "repository_database")
    )
    database_path = _pinned_file(
        dataset_root, repository_database, "repository_database"
    )
    database = _mapping(
        json.loads(database_path.read_text()), "repository_database payload"
    )
    if not same_identifier(database.get("kind"), "leanlean_standardized_repository_database"):
        raise ValueError("repository_database has an unsupported kind")
    database_rows = database.get("repositories")
    if not isinstance(database_rows, list):
        raise ValueError("repository_database.repositories must be a list")
    database_ids = [
        _text(
            record.get("instance_id"),
            "repository_database.repositories[].instance_id",
        )
        for record in database_rows
        if isinstance(record, Mapping)
    ]
    expected_database_ids = all_ids + discarded
    if (
        len(database_ids) != len(database_rows)
        or len(database_ids) != len(set(database_ids))
        or set(database_ids) != set(expected_database_ids)
    ):
        raise ValueError("repository_database membership drift")
    repository_database["path"] = str(database_path)
    repository_database["dataset_id"] = _text(
        database.get("dataset_id"), "repository_database.dataset_id"
    )
    repository_database["dataset_version"] = _text(
        database.get("dataset_version"), "repository_database.dataset_version"
    )
    preprocessing = copy.deepcopy(
        dict(_mapping(dataset.get("preprocessing"), "preprocessing"))
    )
    for key in ("manifest", "config"):
        record = preprocessing.get(key)
        if isinstance(record, Mapping):
            pinned = _pinned_file(dataset_root, record, f"preprocessing.{key}")
            preprocessing[key] = str(pinned)
            preprocessing[f"{key}_sha256"] = record["sha256"]

    raw_words = sum(
        int(row.get("metrics", {}).get("words", {}).get("raw") or 0)
        for row in rows
    )
    stripped_words = sum(
        int(row.get("metrics", {}).get("words", {}).get("stripped") or 0)
        for row in rows
    )
    token_rows = [
        row.get("metrics", {}).get("lean_tokens", {}) for row in rows
    ]
    has_lean_tokens = bool(token_rows) and all(
        isinstance(metrics.get("raw"), int)
        and isinstance(metrics.get("stripped"), int)
        for metrics in token_rows
    )
    raw_tokens = (
        sum(int(metrics["raw"]) for metrics in token_rows)
        if has_lean_tokens
        else None
    )
    stripped_tokens = (
        sum(int(metrics["stripped"]) for metrics in token_rows)
        if has_lean_tokens
        else None
    )
    hydrated = {
        "kind": DATASET_KIND,
        "schema_version": 1,
        "id": dataset.get("id"),
        "version": dataset.get("version"),
        "default_variant": dataset.get("default_variant"),
        "repository_count": len(rows),
        "summary": {
            "discarded_repository_count": len(discarded),
            "discarded_repositories": discarded,
            "protected_declarations": sum(
                int(row.get("protected", {}).get("count") or 0)
                for row in rows
            ),
            "words": {
                "raw": raw_words,
                "stripped": stripped_words,
                "reduction": (
                    round(1 - stripped_words / raw_words, 4)
                    if raw_words
                    else None
                ),
            },
            "lean_tokens": {
                "raw": raw_tokens,
                "stripped": stripped_tokens,
                "reduction": (
                    round(1 - stripped_tokens / raw_tokens, 4)
                    if isinstance(raw_tokens, int)
                    and isinstance(stripped_tokens, int)
                    and raw_tokens
                    else None
                ),
            },
        },
        "repository_database": repository_database,
        "preprocessing": preprocessing,
        "repositories": rows,
    }
    validate_dataset(hydrated)
    return hydrated


def _validate_variant(
    artifact: Mapping[str, Any],
    label: str,
    *,
    require_image: bool,
) -> None:
    """Require exact immutable source and runtime pins for one variant."""

    image = artifact.get("image")
    tree = artifact.get("cache_tree")
    materialization = artifact.get("materialization")
    if require_image and isinstance(materialization, Mapping):
        if image is not None or artifact.get("image_id") is not None:
            raise ValueError(
                f"{label}: ephemeral materialization cannot publish an image"
            )
        if (
            materialization.get("mode") != "build_before_agent"
            or materialization.get("backend") != "shared_environment_v1"
            or materialization.get("image_persistence") != "ephemeral"
        ):
            raise ValueError(f"{label}: invalid image materialization")
        warm = materialization.get("warm_build_cache")
        if (
            not isinstance(warm, Mapping)
            or not isinstance(warm.get("path"), str)
            or not warm["path"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(warm.get("sha256") or ""))
        ):
            raise ValueError(f"{label}: unpinned warm build cache")
        environment = materialization.get("shared_environment")
        archive = (
            environment.get("archive")
            if isinstance(environment, Mapping)
            else None
        )
        if (
            not isinstance(environment, Mapping)
            or not isinstance(environment.get("id"), str)
            or not environment["id"]
            or not isinstance(environment.get("image"), str)
            or not environment["image"]
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(environment.get("image_id") or ""),
            )
            or not isinstance(archive, Mapping)
            or archive.get("format") != "docker_image_save_zstd_v1"
            or not isinstance(archive.get("path"), str)
            or not archive["path"]
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(archive.get("sha256") or "")
            )
        ):
            raise ValueError(f"{label}: unpinned shared dependency environment")
    elif require_image:
        if not isinstance(image, str) or not image:
            raise ValueError(f"{label}: missing runnable image")
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(artifact.get("image_id", ""))
        ):
            raise ValueError(f"{label}: mutable image ID")
    elif image is not None:
        raise ValueError(f"{label}: raw transport image must not be published")
    if not isinstance(tree, str) or not tree or not re.fullmatch(
        r"[0-9a-f]{64}", str(artifact.get("tree_sha256", ""))
    ):
        raise ValueError(f"{label}: unpinned source tree")



def validate_dataset(dataset: Mapping[str, Any]) -> None:
    if dataset.get("kind") != DATASET_KIND or dataset.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest")
    rows = dataset.get("repositories")
    if not isinstance(rows, list) or dataset.get("repository_count") != len(rows):
        raise ValueError("dataset repository count is inconsistent")
    ids = []
    for value in rows:
        row = _mapping(value, "dataset.repository")
        instance_id = _text(row.get("id"), "dataset.repository.id")
        ids.append(instance_id)
        preprocessing = _mapping(
            row.get("preprocessing"), f"{instance_id}.preprocessing"
        )
        graph = _mapping(
            preprocessing.get("dependency_graph"),
            f"{instance_id}.preprocessing.dependency_graph",
        )
        if graph.get("schema") not in GRAPH_SCHEMAS or not re.fullmatch(
            r"[0-9a-f]{64}", str(graph.get("sha256") or "")
        ):
            raise ValueError(f"{instance_id}: invalid dependency graph reference")
        variants = _mapping(row.get("variants"), f"{instance_id}.variants")
        if set(variants) not in ({"stripped"}, {"raw", "stripped"}):
            raise ValueError(f"{instance_id}: variants must be stripped, optionally with raw")
        for name in sorted(variants):
            _validate_variant(
                _mapping(variants[name], f"{instance_id}.{name}"),
                f"{instance_id}.{name}",
                require_image=name == "stripped",
            )
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate repository IDs")
