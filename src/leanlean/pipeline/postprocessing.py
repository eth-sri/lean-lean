"""Artifact-first postprocessing for completed LeanLean evaluations."""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

import yaml

from leanlean.identifiers import same_identifier
from leanlean.pipeline import replay_recovery
from leanlean.timeout_policy import LEAN_VERIFY_TIMEOUT_SECONDS

from leanlean.action_trace import summarize_action_sequences
from leanlean.benchmarks.leanlean import (
    _diff_signatures,
    _modules_under_target,
    _render_task_proof_length_script,
    _resolve_requested_signature_names,
    _safe_repository_path,
)
from leanlean.capture_disposition import capture_is_scoring
from leanlean.submission_patch import drop_absent_deletions
from leanlean.dataset_bundle import load_dataset
from leanlean.environments.offline_replay import (
    DependencyLockViolation,
    OfflineReplayEnvironment as DockerEnvironment,
)
from leanlean.evaluation_images import (
    ensure_materialized_image_from_record,
    materialized_image_id,
    prune_buildkit_cache,
    retire_image_tags,
)
from leanlean.metrics.source_archive import measure_source_archive
from leanlean.playback import _archive_source_files
from leanlean.palomar_comparator import (
    VALIDATION_SCHEMA as PALOMAR_VALIDATION_SCHEMA,
    install_agent_verify_wrapper,
    install_palomar_tools,
    COMPARATOR_RUST_MIN_STACK,
    COMPARATOR_STACK_KIB,
    COMPARATOR_SUCCESS_MARKER,
    load_palomar_evidence,
    resolve_palomar_contract,
)
from leanlean.postprocessing_signatures import (
    VALIDATION_SCHEMA,
    ProtectedContract,
    collect_protected_axioms,
    collect_protected_signatures,
    compare_protected_axioms,
    compare_protected_type_compatibility,
    establish_protected_contract,
)
from leanlean.preprocessing.cache_isolation import render_container_cache_cleanup
from leanlean.preprocessing.repositories import file_sha256
from leanlean.run_action_summary import (
    ACTION_SUMMARY_SCHEMA_VERSION,
    build_run_action_summary,
)
from leanlean.subscription_status import classify_trajectory


MANIFEST_KIND = "leanlean_postprocessing_run"
RUN_ARTIFACT_KIND = "leanlean_postprocessing"
SOURCE_RUN_KIND = "leanlean_run"
SCHEMA_VERSION = 1
LEANLEAN_20260914_DATASET = "datasets/leanlean_20260914/dataset.yaml"
LEANLEAN_20260914_PROMPT_SHA256 = (
    "9e8b045d494936aba649587e30cff40f80fdaaad254b759a7c1f9e9af008e111"
)
DERIVED_PLAYBACK = "postprocessed-playback.json"
REPLAY_PLAYBACK = "replayed-playback.json"


def derived_playback_path(capture_dir: Path, *, replay: bool) -> Path:
    """Keep intermediate replay writes away from the final scoring report."""
    return capture_dir / (REPLAY_PLAYBACK if replay else DERIVED_PLAYBACK)


def _previous_playback_path(capture_dir: Path, *, replay: bool) -> Path:
    destination = derived_playback_path(capture_dir, replay=replay)
    if replay and not destination.exists():
        # Read legacy replay progress once; every subsequent write uses the
        # separate replay destination. Validation still checks source/policy.
        return capture_dir / DERIVED_PLAYBACK
    return destination

SUMMARY_FILENAME = "postprocess_summary.json"
BENCHMARK_FIXTURE_INTEGRITY_SCHEMA = (
    "leanlean_benchmark_fixture_integrity_v1"
)
AUTHORITATIVE_METRIC_SCHEMA = "leanlean_authoritative_lean_tokens_v1"
LEAN_VERIFY_RESULT_SCHEMA = "leanlean_replay_lean_verify_v1"
VERIFICATION_BUILD_POLICY = "clean_project_and_config_preserve_dependencies_v1"
WARM_BASELINE_BUILD_POLICY = "warm_stripped_baseline_incremental_v1"
INCREMENTAL_REPLAY_BUILD_POLICY = "incremental_replay_orphan_pruned_v1"
# Same builds, but lean_verify runs only on the submitted endpoint.
INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY = (
    "incremental_replay_orphan_pruned_endpoint_verify_only_v1"
)
INCREMENTAL_REPLAY_BUILD_ONLY_POLICY = "incremental_replay_orphan_pruned_build_only_v1"
CHECKPOINT_LEAN_VERIFY_SKIPPED = "checkpoint_lean_verify_disabled"
REPLAY_LEAN_VERIFY_SKIPPED = "replay_lean_verify_disabled"
IMPLEMENTATION_FILES = (
    "postprocessing.py",
    "postprocess.sh",
    "scripts/postprocess.py",
    "scripts/measure_heartbeats.py",
    "src/leanlean/heartbeat_cli.py",
    "src/leanlean/pipeline/evaluation.py",
    "src/leanlean/pipeline/postprocessing.py",
    "src/leanlean/preprocessing/cache_isolation.py",
    "src/leanlean/pipeline/replay_recovery.py",
    "src/leanlean/dataset_bundle.py",
    "src/leanlean/capture_disposition.py",
    "src/leanlean/benchmarks/leanlean.py",
    "src/leanlean/preprocessing/olean.py",
    "src/leanlean/environments/docker.py",
    "src/leanlean/environments/offline_replay.py",
    "src/leanlean/timeout_policy.py",
    "src/leanlean/evaluation_images.py",
    "src/leanlean/shared_cache.py",
    "docker/evaluation/Dockerfile.shared-cache",
    "src/leanlean/postprocessing_signatures.py",
    "src/leanlean/palomar_comparator.py",
    "scripts/palomar_landrun_wrapper.sh",
    "scripts/lean_verify",
    "src/leanlean/metrics/source_archive.py",
    "src/leanlean/metrics/tokens.py",
    "src/leanlean/playback.py",
    "src/leanlean/report_summary.py",
    "src/leanlean/run_action_summary.py",
)


@dataclass(frozen=True)
class PreparedPostprocessing:
    manifest_path: Path
    manifest: Mapping[str, Any]
    run_artifact_path: Path
    source_root: Path
    source_run_artifact_path: Path
    source_run: Mapping[str, Any]
    evaluation_manifest_path: Path
    evaluation_manifest: Mapping[str, Any]
    dataset_path: Path
    dataset: Mapping[str, Any]
    predictions_path: Path
    predictions: Mapping[str, Any]
    output_dir: Path
    replay: bool
    tmux_session: str


@dataclass(frozen=True)
class EvaluationAttempt:
    index: int
    run_id: str
    run_artifact_path: Path
    run_artifact: Mapping[str, Any]
    source_root: Path
    evaluation_manifest_path: Path
    evaluation_manifest: Mapping[str, Any]
    dataset_path: Path
    dataset: Mapping[str, Any]
    predictions_path: Path
    predictions: Mapping[str, Any]
    output_dir: Path
    repositories: tuple[str, ...]
    captures: tuple[Mapping[str, Any], ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(row: Mapping[str, Any], key: str, name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value


def _manifest_resource_overrides(
    manifest: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    if "resource_overrides" in manifest:
        overrides = _mapping(
            manifest.get("resource_overrides"), "manifest.resource_overrides"
        )
        values = (
            overrides.get("cgroup_parent"),
            overrides.get("container_memory"),
        )
    else:
        # Compatibility with manifests written before raw overrides were pinned.
        container = _mapping(manifest.get("container"), "manifest.container")
        values = (container.get("cgroup_parent"), container.get("memory"))
    result: list[str | None] = []
    for name, value in zip(("cgroup_parent", "container_memory"), values):
        if value is None or value == "":
            result.append(None)
        elif isinstance(value, str):
            result.append(value)
        else:
            raise ValueError(f"manifest resource override {name} must be a string or null")
    return result[0], result[1]


def _load_yaml(path: Path, name: str) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = yaml.safe_load(text)
        else:
            value = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    return _mapping(value, name)


def _dump(value: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(value), sort_keys=False, width=100)


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_dump(value))
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_exact(path: Path, content: str, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(
            f"postprocessing ID {run_id!r} already has a different {path.name}"
        )
    path.write_text(content)


def _relative_or_absolute(root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _canonical_materialization_paths(
    pins: Mapping[str, Any], *, source_root: Path
) -> dict[str, Any]:
    """Normalize paths without weakening artifact identity or mutating pins.

    Evaluation manifests store repository-relative paths; hydrated schema-v2
    datasets store absolute paths. Only the three filesystem fields may differ
    in spelling. Hashes, image IDs, archive sizes and other fields stay exact.
    """
    normalized = copy.deepcopy(dict(pins))
    for fields in (
        ("source_tree",),
        ("shared_environment", "archive", "path"),
        ("warm_build_cache", "path"),
    ):
        parent = normalized
        for field in fields[:-1]:
            child = parent.get(field)
            if not isinstance(child, dict):
                break
            parent = child
        else:
            field = fields[-1]
            if field in parent:
                parent[field] = str(
                    _resolve_path(
                        source_root,
                        _text(parent, field, "materialization." + ".".join(fields)),
                    )
                )
    return normalized


def _validate_materialization_pin(
    evaluation_pin: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    source_root: Path,
    instance_id: str,
    allow_relocated_paths: bool = False,
) -> dict[str, Any]:
    if evaluation_pin.get("backend", "dockerfile_network_v1") == "dockerfile_network_v1":
        execution_pin = copy.deepcopy(dict(evaluation_pin))
        if allow_relocated_paths:
            execution_pin["source_tree"] = artifact.get("cache_tree")
        actual = _canonical_materialization_paths(execution_pin, source_root=source_root)
        expected = _canonical_materialization_paths({
            "source_tree": artifact.get("cache_tree"),
            "tree_sha256": artifact.get("tree_sha256"),
        }, source_root=source_root)
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"{instance_id}: evaluation materialization pin drift for {key}")
        return execution_pin
    materialization = _mapping(
        artifact.get("materialization"), f"dataset.{instance_id}.materialization"
    )
    if evaluation_pin.get("backend") != "shared_environment_v1":
        raise ValueError(
            f"{instance_id}: unsupported postprocessing materialization backend"
        )
    environment = _mapping(
        materialization.get("shared_environment"),
        f"dataset.{instance_id}.shared_environment",
    )
    warm_cache = _mapping(
        materialization.get("warm_build_cache"),
        f"dataset.{instance_id}.warm_build_cache",
    )
    expected = {
        "source_tree": artifact.get("cache_tree"),
        "tree_sha256": artifact.get("tree_sha256"),
        "build_target": materialization.get("build_target"),
        "shared_environment": {
            key: environment[key]
            for key in ("id", "image", "image_id", "archive")
            if key in environment
        },
        "warm_build_cache": {
            key: warm_cache[key] for key in ("path", "sha256") if key in warm_cache
        },
    }
    execution_pin = copy.deepcopy(dict(evaluation_pin))
    if allow_relocated_paths:
        execution_pin["source_tree"] = artifact.get("cache_tree")
        execution_pin["shared_environment"]["archive"]["path"] = environment[
            "archive"
        ]["path"]
        execution_pin["warm_build_cache"]["path"] = warm_cache["path"]
    actual = _canonical_materialization_paths(execution_pin, source_root=source_root)
    expected = _canonical_materialization_paths(expected, source_root=source_root)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"{instance_id}: evaluation materialization pin drift for {key}"
            )
    return execution_pin


def _source_root(path: Path, experiment_relative: str) -> Path:
    relative = Path(experiment_relative)
    if relative.is_absolute():
        raise ValueError("evaluation experiment path must be repository-relative")
    for parent in path.resolve().parents:
        if (parent / relative).is_file():
            return parent
    raise ValueError(f"could not locate evaluation repository root for {path}")


def _resolve_source_run(reference: str | Path, repo_root: Path) -> tuple[Path, Path]:
    candidate = Path(reference)
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        by_id = repo_root / "runs/evaluation" / candidate / "run.yaml"
        if by_id.is_file():
            candidate = by_id
    elif not candidate.is_absolute():
        candidate = repo_root / candidate
    path = candidate.resolve()
    if not path.is_file():
        raise ValueError(f"evaluation run artifact not found: {path}")
    artifact = _load_yaml(path, "evaluation run artifact")
    experiment = _mapping(artifact.get("experiment"), "run.experiment")
    root = _source_root(path, _text(experiment, "manifest", "run.experiment"))
    return path, root


def _canonical_archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seconds(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([hms]?)", value.strip().lower())
    if not match:
        raise ValueError(f"unsupported timeout {value!r}")
    amount = int(match.group(1))
    return amount * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


def _postprocessing_build_timeout(container: Mapping[str, Any]) -> int:
    return max(
        LEAN_VERIFY_TIMEOUT_SECONDS,
        _seconds(_text(container, "timeout", "evaluation.container")),
    )


def _aggregate_memory(memory: Any, workers: int) -> str:
    value = str(memory)
    match = re.fullmatch(r"([1-9][0-9]*)([a-zA-Z]+)", value)
    if match is None:
        return f"{workers}x{value}"
    return f"{workers * int(match.group(1))}{match.group(2)}"


def _dataset_rows(dataset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = dataset.get("repositories")
    if not isinstance(rows, list):
        raise ValueError("dataset.repositories must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, "dataset repository")
        instance_id = _text(row, "id", "dataset repository")
        if instance_id in result:
            raise ValueError(f"duplicate dataset repository {instance_id}")
        result[instance_id] = row
    return result


def _repository_database_rows(
    dataset: Mapping[str, Any], dataset_path: Path
) -> dict[str, Mapping[str, Any]]:
    repository_root = dataset_path.parents[1]
    database = _mapping(
        dataset.get("repository_database"), "dataset.repository_database"
    )
    database_path = _resolve_path(
        repository_root,
        _text(database, "path", "dataset.repository_database"),
    )
    if file_sha256(database_path) != database.get("sha256"):
        raise ValueError(f"repository database drifted: {database_path}")
    payload = _load_yaml(database_path, "repository database")
    rows = payload.get("repositories")
    if not isinstance(rows, list):
        raise ValueError("repository database repositories must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, "repository database row")
        instance_id = _text(row, "instance_id", "repository database row")
        if instance_id in result:
            raise ValueError(f"duplicate repository database row {instance_id}")
        result[instance_id] = row
    return result


def _protected_declarations(
    dataset: Mapping[str, Any], dataset_path: Path
) -> dict[str, list[str]]:
    repository_root = dataset_path.parents[1]
    database = _mapping(
        dataset.get("repository_database"), "dataset.repository_database"
    )
    database_path = _resolve_path(
        repository_root,
        _text(database, "path", "dataset.repository_database"),
    )
    if file_sha256(database_path) != database.get("sha256"):
        raise ValueError(f"protected-declaration database drifted: {database_path}")
    payload = _load_yaml(database_path, "repository database")
    rows = payload.get("repositories")
    if not isinstance(rows, list):
        raise ValueError("repository database repositories must be a list")
    result: dict[str, list[str]] = {}
    for raw in rows:
        row = _mapping(raw, "repository database row")
        protected = _mapping(row.get("protected"), "repository protected declarations")
        declarations = protected.get("declarations")
        if declarations is None:
            file_pin = _mapping(protected.get("file"), "protected.file")
            signature_path = _resolve_path(
                repository_root, _text(file_pin, "path", "protected.file")
            )
            if file_sha256(signature_path) != file_pin.get("sha256"):
                raise ValueError(f"protected-signature file drifted: {signature_path}")
            signature_payload = _load_yaml(signature_path, "protected-signature file")
            declarations = [
                *list(signature_payload.get("signature_theorems") or []),
                *list(signature_payload.get("protected_definitions") or []),
            ]
        if not isinstance(declarations, list) or not all(
            isinstance(item, str) and item for item in declarations
        ):
            raise ValueError("protected.declarations must be a string list")
        result[_text(row, "instance_id", "repository database row")] = list(declarations)
    return result


def _input_capture_rows(
    output_dir: Path, *, repositories: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = set(repositories) if repositories is not None else None
    for path in sorted(output_dir.glob("*/playback/capture_*/playback.json")):
        instance_id = path.parents[2].name
        if selected is not None and instance_id not in selected:
            continue
        if not capture_is_scoring(path):
            continue
        capture = _load_yaml(path, "playback artifact")
        points = capture.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"{path}: playback points are missing")
        archives = []
        for point in points:
            point = _mapping(point, f"{path}: checkpoint")
            archive = path.parent / Path(_text(point, "snapshot_path", "checkpoint")).name
            if not archive.is_file():
                raise ValueError(f"checkpoint archive is missing: {archive}")
            expected = _text(point, "source_archive_sha256", "checkpoint")
            actual = _canonical_archive_sha256(archive)
            if actual != expected:
                raise ValueError(f"checkpoint archive digest mismatch: {archive}")
            archives.append({"path": archive.name, "source_archive_sha256": actual})
        rows.append(
            {
                "instance_id": str(capture.get("instance_id") or path.parents[2].name),
                "capture": path.parent.name,
                "playback": str(path.relative_to(output_dir)),
                "playback_sha256": file_sha256(path),
                "archives": archives,
            }
        )
    return rows


def _load_evaluation_attempt(
    path: Path,
    *,
    repo_root: Path,
    index: int,
    allow_partial: bool = False,
    repository_subset: Sequence[str] | None = None,
    allow_missing_repositories: bool = False,
) -> EvaluationAttempt:
    source_path, source_root = _resolve_source_run(path, repo_root)
    source_run = _load_yaml(source_path, "evaluation run artifact")
    if (
        source_run.get("kind") != SOURCE_RUN_KIND
        or source_run.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"{source_path}: unsupported evaluation run artifact")
    source_status = str(source_run.get("status") or "")
    if source_status not in {"complete", "failed"}:
        raise ValueError(
            f"{source_path}: source run is still active or unlaunched "
            f"(status {source_status!r})"
        )

    experiment_pin = _mapping(source_run.get("experiment"), "run.experiment")
    evaluation_path = _resolve_path(
        source_root, _text(experiment_pin, "manifest", "run.experiment")
    )
    if file_sha256(evaluation_path) != experiment_pin.get("sha256"):
        raise ValueError(f"evaluation manifest drifted: {evaluation_path}")
    evaluation = _load_yaml(evaluation_path, "evaluation manifest")
    if evaluation.get("kind") != "leanlean_evaluation_run":
        raise ValueError(f"{evaluation_path}: unsupported evaluation manifest")

    dataset_pin = _mapping(source_run.get("dataset"), "run.dataset")
    dataset_path = _resolve_path(
        source_root, _text(dataset_pin, "manifest", "run.dataset")
    )
    if file_sha256(dataset_path) != dataset_pin.get("sha256"):
        raise ValueError(f"dataset manifest drifted: {dataset_path}")
    dataset = (
        load_dataset(dataset_path)
        if repository_subset is None
        else load_dataset(dataset_path, repository_ids=repository_subset)
    )

    artifacts = _mapping(source_run.get("artifacts"), "run.artifacts")
    output_dir = _resolve_path(
        source_root, _text(artifacts, "output_directory", "run.artifacts")
    )
    predictions_path = _resolve_path(
        source_root, _text(artifacts, "predictions", "run.artifacts")
    )
    if not output_dir.is_dir() or not predictions_path.is_file():
        raise ValueError(f"evaluation output is incomplete: {output_dir}")
    try:
        predictions = json.loads(predictions_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            f"could not read predictions {predictions_path}: {error}"
        ) from error
    if not isinstance(predictions, Mapping):
        raise ValueError(
            f"predictions must be keyed by repository: {predictions_path}"
        )

    repositories = evaluation.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("evaluation.repositories must be a non-empty list")
    repository_ids = tuple(
        _text(
            _mapping(row, "evaluation repository"),
            "id",
            "evaluation repository",
        )
        for row in repositories
    )
    run_id = _text(source_run, "run_id", "run")
    evaluation_run_id = _text(evaluation, "run_id", "evaluation")
    if run_id != evaluation_run_id:
        raise ValueError(
            f"{source_path}: run ID {run_id!r} does not match "
            f"evaluation run ID {evaluation_run_id!r}"
        )
    execution = _mapping(source_run.get("execution"), "run.execution")
    for key in ("model", "reasoning_effort", "harness"):
        if execution.get(key) != evaluation.get(key):
            raise ValueError(
                f"{source_path}: recorded {key} does not match "
                "the evaluation manifest"
            )
    if int(execution.get("repositories", -1)) != len(repository_ids):
        raise ValueError(
            f"{source_path}: recorded repository count does not match "
            "the evaluation manifest"
        )
    prediction_ids = {str(key) for key in predictions}
    partial = allow_partial and source_status == "failed"
    if not prediction_ids <= set(repository_ids) or (
        not partial and prediction_ids != set(repository_ids)
    ):
        raise ValueError(
            f"{source_path}: prediction repository set does not match "
            "the evaluation manifest"
        )
    capture_rows = (
        _input_capture_rows(output_dir)
        if repository_subset is None
        else _input_capture_rows(output_dir, repositories=repository_subset)
    )
    captures = tuple(capture_rows)
    monitoring = _mapping(evaluation.get("monitoring"), "evaluation.monitoring")
    if monitoring.get("snapshots") is True:
        capture_ids = {str(row["instance_id"]) for row in captures}
        expected_capture_ids = set(repository_subset or repository_ids)
        if allow_missing_repositories:
            expected_capture_ids &= set(repository_ids)
        if not expected_capture_ids <= set(repository_ids):
            unknown = sorted(expected_capture_ids - set(repository_ids))
            raise ValueError(
                f"{source_path}: requested repositories are absent: {unknown}"
            )
        if not capture_ids <= expected_capture_ids or (
            not partial and capture_ids != expected_capture_ids
        ):
            raise ValueError(
                f"{source_path}: capture repository set does not match "
                "the evaluation manifest"
            )
    return EvaluationAttempt(
        index=index,
        run_id=run_id,
        run_artifact_path=source_path,
        run_artifact=source_run,
        source_root=source_root,
        evaluation_manifest_path=evaluation_path,
        evaluation_manifest=evaluation,
        dataset_path=dataset_path,
        dataset=dataset,
        predictions_path=predictions_path,
        predictions=predictions,
        output_dir=output_dir,
        repositories=repository_ids,
        captures=captures,
    )


def _validate_additional_attempt(
    root: EvaluationAttempt,
    attempt: EvaluationAttempt,
    *,
    allow_repository_union: bool = False,
) -> None:
    if (
        attempt.dataset_path != root.dataset_path
        or file_sha256(attempt.dataset_path) != file_sha256(root.dataset_path)
    ):
        raise ValueError(f"{attempt.run_id}: dataset differs from the first run")
    for key in ("model", "reasoning_effort", "harness", "rounds"):
        if attempt.evaluation_manifest.get(key) != root.evaluation_manifest.get(key):
            raise ValueError(
                f"{attempt.run_id}: {key} differs from the first run"
            )
    root_rows = {
        str(row["id"]): row
        for row in root.evaluation_manifest["repositories"]
    }
    selected = [
        instance_id
        for instance_id in root.repositories
        if instance_id in set(attempt.repositories)
    ]
    if not allow_repository_union and selected != list(attempt.repositories):
        raise ValueError(
            f"{attempt.run_id}: repositories are not an ordered first-run subset"
        )
    for raw in attempt.evaluation_manifest["repositories"]:
        row = _mapping(raw, "additional repository")
        instance_id = _text(row, "id", "additional repository")
        if instance_id not in root_rows:
            continue
        expected = _mapping(root_rows[instance_id], "root repository")
        materialized = isinstance(expected.get("materialization"), Mapping)
        for key in (
            "image",
            "image_id",
            "source_tree",
            "tree_sha256",
            "environment_id",
            "environment_image_id",
            "warm_build_cache_sha256",
            "materialization",
        ):
            actual_value, expected_value = row.get(key), expected.get(key)
            if key == "image" and materialized:
                # Each continuation deliberately owns a different ephemeral tag.
                continue
            if key == "materialization" and materialized:
                if isinstance(actual_value, Mapping):
                    actual_value = _canonical_materialization_paths(
                        actual_value, source_root=attempt.source_root
                    )
                    actual_value.pop("tag", None)
                expected_value = _canonical_materialization_paths(
                    expected_value, source_root=root.source_root
                )
                expected_value.pop("tag", None)
            if actual_value != expected_value:
                raise ValueError(
                    f"{attempt.run_id}: {instance_id} changed {key}"
                )


def _trajectory_selection(
    attempt: EvaluationAttempt,
    instance_id: str,
    *,
    allow_failed: bool = False,
) -> dict[str, Any] | None:
    prediction = attempt.predictions.get(instance_id)
    if not isinstance(prediction, Mapping) or not isinstance(
        prediction.get("model_patch"), str
    ):
        return None
    captures = [
        row for row in attempt.captures if row["instance_id"] == instance_id
    ]
    if not captures:
        return None
    trajectory = (
        attempt.output_dir / instance_id / f"{instance_id}.traj.json"
    )
    try:
        payload = json.loads(trajectory.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    failure = classify_trajectory(payload)
    if failure is not None and not allow_failed:
        return None
    info = payload.get("info")
    exit_status = info.get("exit_status") if isinstance(info, Mapping) else None
    if not isinstance(exit_status, str) or not exit_status:
        return None
    if exit_status in {"ExecutionFailed", "ProviderFailed", "QuotaExceeded"}:
        return None
    return {
        "run_id": attempt.run_id,
        "attempt_index": attempt.index,
        "output_directory": str(attempt.output_dir),
        "trajectory": _relative_or_absolute(
            attempt.source_root, trajectory
        ),
        "trajectory_sha256": file_sha256(trajectory),
    }


def _combine_runs(
    root: EvaluationAttempt,
    additional: list[EvaluationAttempt],
    *,
    allow_failed: bool = False,
    allow_unresolved: bool = False,
    repositories: Sequence[str] | None = None,
    allow_repository_union: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    attempts = [root, *additional]
    selected_repositories = list(repositories or root.repositories)
    if len(selected_repositories) != len(set(selected_repositories)):
        raise ValueError("postprocessing repository subset contains duplicates")
    available_repositories = {
        instance_id for attempt in attempts for instance_id in attempt.repositories
    }
    allowed_repositories = (
        available_repositories if allow_repository_union else set(root.repositories)
    )
    unknown = sorted(set(selected_repositories) - allowed_repositories)
    if unknown:
        raise ValueError(
            "postprocessing repository subset is absent from the root evaluation: "
            + ", ".join(unknown)
        )
    selected_repository_set = set(selected_repositories)
    selected_attempts: dict[str, EvaluationAttempt] = {}
    selection: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if attempt is not root:
            _validate_additional_attempt(
                root, attempt, allow_repository_union=allow_repository_union
            )
        for instance_id in attempt.repositories:
            if instance_id not in selected_repository_set:
                continue
            selected = _trajectory_selection(
                attempt, instance_id, allow_failed=allow_failed
            )
            if selected is not None:
                selected_attempts[instance_id] = attempt
                selection[instance_id] = selected
    unresolved = [
        instance_id
        for instance_id in selected_repositories
        if instance_id not in selected_attempts
    ]
    if unresolved and not allow_unresolved:
        raise ValueError(
            "explicit run combination has unresolved repositories: "
            + ", ".join(unresolved)
        )

    predictions: dict[str, Any] = {}
    captures: list[dict[str, Any]] = []
    for instance_id in selected_repositories:
        if instance_id not in selected_attempts:
            continue
        attempt = selected_attempts[instance_id]
        predictions[instance_id] = copy.deepcopy(attempt.predictions[instance_id])
        for raw in attempt.captures:
            if raw["instance_id"] != instance_id:
                continue
            row = dict(raw)
            row["source_run_id"] = attempt.run_id
            row["attempt_index"] = attempt.index
            row["output_directory"] = str(attempt.output_dir)
            captures.append(row)

    components = []
    for attempt in attempts:
        components.append(
            {
                "run_id": attempt.run_id,
                "attempt_index": attempt.index,
                "artifact": _relative_or_absolute(
                    attempt.source_root,
                    attempt.run_artifact_path,
                ),
                "artifact_sha256": file_sha256(attempt.run_artifact_path),
                "experiment": _relative_or_absolute(
                    attempt.source_root,
                    attempt.evaluation_manifest_path,
                ),
                "experiment_sha256": file_sha256(
                    attempt.evaluation_manifest_path
                ),
                "predictions": str(attempt.predictions_path),
                "predictions_sha256": file_sha256(attempt.predictions_path),
                "output_directory": str(attempt.output_dir),
                "model": attempt.evaluation_manifest.get("model"),
                "reasoning_effort": attempt.evaluation_manifest.get("reasoning_effort"),
                "harness": attempt.evaluation_manifest.get("harness"),
                "status": attempt.run_artifact.get("status"),
                "repositories": list(attempt.repositories),
                    "selected_repositories": [
                        instance_id
                        for instance_id in selected_repositories
                        if selected_attempts.get(instance_id) is attempt
                    ],
            }
        )
    return predictions, captures, selection, components


def _postprocessing_artifact_paths(
    root: Path,
    evaluation: Mapping[str, Any],
    *,
    postprocess_id: str,
    mode: str,
) -> tuple[Path, Path]:
    source_configs = evaluation.get("source_configs")
    logical_path: Path | None = None
    if isinstance(source_configs, list):
        config_keys = {
            str(item.get("role")): str(item.get("key"))
            for item in source_configs
            if isinstance(item, Mapping)
        }
        if config_keys.get("dataset") and config_keys.get("model"):
            logical_path = (
                Path(config_keys["dataset"]) / Path(config_keys["model"]) / mode
            )
            if logical_path.is_absolute() or ".." in logical_path.parts:
                raise ValueError("hierarchical postprocessing path is unsafe")
    if logical_path is None:
        # Temporary compatibility for pre-hierarchy evaluations.
        return (
            root / "experiments/postprocessing" / f"{postprocess_id}.yaml",
            root / "runs/postprocessing" / postprocess_id,
        )
    existing_manifest = root / "experiments/postprocessing" / logical_path / "manifest.yaml"
    if existing_manifest.exists():
        existing = _load_yaml(existing_manifest, "existing postprocessing manifest")
        if existing.get("postprocess_id") != postprocess_id:
            # A changed verifier gets a new immutable definition. Keep the old
            # completed run addressable while reusing matching checkpoint data.
            logical_path = logical_path / "revisions" / postprocess_id
    return (
        root / "experiments/postprocessing" / logical_path / "manifest.yaml",
        root / "runs/postprocessing" / logical_path,
    )


def _preprocessing_palomar_attestation(
    dataset_row: Mapping[str, Any], contract: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    preprocessing = _mapping(
        dataset_row.get("preprocessing"), "dataset.preprocessing"
    )
    report_path = Path(_text(preprocessing, "report", "dataset.preprocessing"))
    report = _mapping(json.loads(report_path.read_text()), "preprocessing report")
    variants = _mapping(dataset_row.get("variants"), "dataset.variants")
    artifact = _mapping(variants.get(variant), f"dataset.variants.{variant}")
    final = _mapping(artifact.get("final_verification"), "final_verification")
    verifier = _mapping(final.get("declared_verifier"), "declared_verifier")
    provenance = _mapping(
        report.get("protected_provenance"), "protected_provenance"
    )
    if (
        final.get("passed") is not True
        or final.get("status") != "passed"
        or not all(value is True for value in _mapping(
            final.get("checks"), "final_verification.checks"
        ).values())
        or verifier.get("passed") is not True
        or verifier.get("returncode") != 0
        or verifier.get("engine") != "leanprover/comparator"
        or not same_identifier(verifier.get("contract_schema"), PALOMAR_VALIDATION_SCHEMA)
    ):
        raise ValueError(
            f"{dataset_row.get('id')}: preprocessing baseline attestation failed"
        )
    identities = (
        ("comparator", "manifest_sha256", "comparator", "manifest_sha256"),
        ("registry_record", "sha256", "registry_record", "sha256"),
        ("source_archive", "sha256", "source_archive", "sha256"),
        ("source_archive", "repository", "source_archive", "repository"),
        ("source_archive", "commit", "source_archive", "commit"),
        ("challenge", "sha256", None, "registered_challenge_sha256"),
        ("configuration", "sha256", None, "registered_comparator_config_sha256"),
    )
    for contract_group, contract_key, provenance_group, provenance_key in identities:
        expected = _mapping(contract.get(contract_group), contract_group).get(contract_key)
        actual = (
            _mapping(provenance.get(provenance_group), provenance_group).get(provenance_key)
            if provenance_group
            else provenance.get(provenance_key)
        )
        if expected != actual:
            raise ValueError(
                f"{dataset_row.get('id')}: preprocessing baseline identity drift: "
                f"{contract_group}.{contract_key}"
            )
    return {
        "source": "pinned_preprocessing_final_verification",
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "stripped_tree_sha256": artifact.get("tree_sha256"),
        "preprocessing_contract_sha256": verifier.get("contract_sha256"),
        "current_contract_sha256": contract.get("sha256"),
        "verifier_output_sha256": verifier.get("output_sha256"),
        "verdict": "accepted",
    }


def resolve_run(
    reference: str | Path | Sequence[str | Path],
    *,
    repo_root: Path,
    replay: bool,
    workers: int | None = None,
    threads_per_repository: int | None = None,
    measure_heartbeats: bool = False,
    allow_relocated_dataset: bool = False,
    prefire: bool = False,
    ignore_incomplete: bool = False,
    endpoints_only: bool = False,
    repository_subset: Sequence[str] | None = None,
    allow_missing_subset: bool = False,
    repository_subset_source: Mapping[str, Any] | None = None,
    cgroup_parent: str | None = None,
    container_memory: str | None = None,
    warm_baseline_build: bool = False,
    save_lake_build: bool = False,
    incremental_replay: bool = False,
    skip_checkpoint_lean_verify: bool = False,
    build_only_replay: bool = False,
    checkpoint_selection: Mapping[str, Any] | None = None,
) -> PreparedPostprocessing:
    if endpoints_only and replay:
        raise ValueError("endpoints_only cannot be combined with replay")
    if incremental_replay and not replay:
        raise ValueError("incremental_replay requires replay")
    if skip_checkpoint_lean_verify and not incremental_replay:
        raise ValueError("skip_checkpoint_lean_verify requires incremental_replay")
    if build_only_replay and not incremental_replay:
        raise ValueError("build_only_replay requires incremental_replay")
    if build_only_replay and skip_checkpoint_lean_verify:
        raise ValueError("build_only_replay conflicts with skip_checkpoint_lean_verify")
    if checkpoint_selection is not None:
        if not replay:
            raise ValueError("checkpoint_selection requires replay")
        selected = checkpoint_selection.get("edit_indices")
        if not isinstance(selected, Mapping) or not all(
            isinstance(indices, list)
            and all(isinstance(index, int) and index > 0 for index in indices)
            for indices in selected.values()
        ):
            raise ValueError("checkpoint_selection.edit_indices must map repositories to positive edit indices")
    if warm_baseline_build and not endpoints_only:
        raise ValueError("warm_baseline_build requires endpoints_only")
    if save_lake_build and not warm_baseline_build:
        raise ValueError("save_lake_build requires warm_baseline_build")
    root = repo_root.resolve()
    references = (
        [reference] if isinstance(reference, (str, Path)) else list(reference)
    )
    if not references:
        raise ValueError("at least one evaluation run ID is required")
    attempts = [
        _load_evaluation_attempt(
            _resolve_source_run(item, root)[0], repo_root=root, index=index,
            allow_partial=ignore_incomplete or len(references) > 1,
            repository_subset=repository_subset,
            allow_missing_repositories=allow_missing_subset,
        )
        for index, item in enumerate(references)
    ]
    root_attempt, *additional_attempts = attempts
    requested_repositories = (
        list(repository_subset)
        if repository_subset is not None
        else list(root_attempt.repositories)
    )
    available_repositories = {
        instance_id
        for attempt in attempts
        for instance_id in attempt.repositories
    }
    missing_requested_repositories = [
        instance_id
        for instance_id in requested_repositories
        if instance_id not in available_repositories
    ]
    selected_repositories = [
        instance_id
        for instance_id in requested_repositories
        if instance_id in available_repositories
    ]
    if missing_requested_repositories and not allow_missing_subset:
        raise ValueError(
            "postprocessing repository subset is absent from the root evaluation: "
            + ", ".join(missing_requested_repositories)
        )
    if not selected_repositories:
        raise ValueError("postprocessing repository subset must not be empty")
    source_path = root_attempt.run_artifact_path
    prediction_payload, capture_rows, run_selection, component_runs = (
        _combine_runs(
            root_attempt,
            additional_attempts,
            allow_failed=prefire,
            allow_unresolved=ignore_incomplete or allow_missing_subset,
            repositories=selected_repositories,
            allow_repository_union=repository_subset is not None,
        )
    )
    source_run = root_attempt.run_artifact
    source_status = source_run.get("status")
    source_root = root_attempt.source_root
    evaluation_path = root_attempt.evaluation_manifest_path
    evaluation = root_attempt.evaluation_manifest
    dataset_pin = _mapping(source_run.get("dataset"), "run.dataset")
    dataset_path = root_attempt.dataset_path
    dataset = root_attempt.dataset
    output_dir = root_attempt.output_dir
    predictions = root_attempt.predictions_path
    monitoring = _mapping(evaluation.get("monitoring"), "evaluation.monitoring")
    prompt_sha256 = None
    prompt_ablation = evaluation.get("prompt_ablation")
    prompt_lock_required = (
        dataset_pin.get("manifest") == LEANLEAN_20260914_DATASET
        and prompt_ablation is None
    )
    if dataset_pin.get("manifest") == LEANLEAN_20260914_DATASET:
        agent = _mapping(evaluation.get("agent"), "evaluation.agent")
        prompt = _text(agent, "prompt", "evaluation.agent")
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if prompt_lock_required:
        if prompt_sha256 != LEANLEAN_20260914_PROMPT_SHA256:
            raise ValueError(
                "LeanLean 2026-09-14 resolved evaluation prompt mismatch: "
                f"expected {LEANLEAN_20260914_PROMPT_SHA256}, got {prompt_sha256}"
            )
        subset_prompt_sha256 = (repository_subset_source or {}).get(
            "canonical_prompt_sha256"
        )
        if (
            repository_subset_source is not None
            and subset_prompt_sha256 != LEANLEAN_20260914_PROMPT_SHA256
        ):
            raise ValueError(
                "LeanLean 2026-09-14 repository subset prompt mismatch: "
                f"expected {LEANLEAN_20260914_PROMPT_SHA256}, got {subset_prompt_sha256}"
            )
    if replay and monitoring.get("snapshots") is not True:
        raise ValueError("--replay requires an evaluation with monitoring.snapshots: true")

    repositories = evaluation.get("repositories")
    assert isinstance(repositories, list)
    repository_specs: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        for raw in attempt.evaluation_manifest["repositories"]:
            row = _mapping(raw, "evaluation repository")
            repository_specs[_text(row, "id", "evaluation repository")] = row
    expected_ids = [
        instance_id
        for instance_id in selected_repositories
        if instance_id in run_selection
    ]
    ignored_incomplete = [
        instance_id
        for instance_id in selected_repositories
        if instance_id not in run_selection
    ]
    if not expected_ids:
        raise ValueError("completed-only postprocessing found no valid repositories")
    patch_digests = {
        instance_id: hashlib.sha256(
            str(prediction_payload[instance_id]["model_patch"]).encode(
                "utf-8", errors="surrogateescape"
            )
        ).hexdigest()
        for instance_id in expected_ids
    }
    for row in capture_rows:
        row["model_patch_sha256"] = patch_digests[str(row["instance_id"])]

    dataset_rows = _dataset_rows(dataset)
    protected = _protected_declarations(dataset, dataset_path)
    database_rows = _repository_database_rows(dataset, dataset_path)
    palomar_contracts: dict[str, Mapping[str, Any]] = {}
    for instance_id in expected_ids:
        database_row = database_rows.get(instance_id)
        if database_row is None:
            raise ValueError(
                f"evaluation repository {instance_id} is absent from repository database"
            )
        row_protected = _mapping(
            database_row.get("protected"), f"database.{instance_id}.protected"
        )
        provenance = _mapping(
            row_protected.get("provenance"),
            f"database.{instance_id}.protected.provenance",
        )
        if provenance.get("kind") == "palomar_registry_main_results":
            palomar_contracts[instance_id] = resolve_palomar_contract(
                repo_root=root, database_row=database_row
            )
    if replay:
        missing_lean_verify = sorted(set(expected_ids) - set(palomar_contracts))
        if missing_lean_verify:
            raise ValueError(
                "--replay requires registered lean_verify support for every "
                "repository; missing: " + ", ".join(missing_lean_verify)
            )

    implementation = {
        relative: file_sha256(root / relative) for relative in IMPLEMENTATION_FILES
    }
    implementation_digest = hashlib.sha256(
        json.dumps(
            {
                "files": implementation,
                "palomar_contracts": {
                    key: value["sha256"]
                    for key, value in sorted(palomar_contracts.items())
                },
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    combination_digest = hashlib.sha256(
        json.dumps(
            {
                "inputs": component_runs,
                "measure_heartbeats": measure_heartbeats,
                "endpoints_only": endpoints_only,
                "allow_relocated_dataset": allow_relocated_dataset,
                "selection": run_selection,
                "prefire": prefire,
                "ignore_incomplete": ignore_incomplete,
                "repository_subset": requested_repositories,
                "allow_missing_subset": allow_missing_subset,
                "repository_subset_source": repository_subset_source,
                "cgroup_parent": cgroup_parent,
                "container_memory": container_memory,
                "warm_baseline_build": warm_baseline_build,
                "save_lake_build": save_lake_build,
                "incremental_replay": incremental_replay,
                "skip_checkpoint_lean_verify": skip_checkpoint_lean_verify,
                "build_only_replay": build_only_replay,
                "checkpoint_selection_sha256": (
                    checkpoint_selection.get("sha256")
                    if checkpoint_selection is not None
                    else None
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    mode = "replay" if replay else "report"
    source_run_id = _text(source_run, "run_id", "run")
    container = _mapping(evaluation.get("container"), "evaluation.container")
    parallelism = _mapping(evaluation.get("parallelism"), "evaluation.parallelism")
    worker_count = int(workers if workers is not None else parallelism.get("workers", 1))
    if worker_count < 1:
        raise ValueError("postprocessing workers must be at least 1")
    source_build_jobs = int(container.get("build_jobs") or container.get("cpus") or 1)
    thread_count = int(
        threads_per_repository
        if threads_per_repository is not None
        else source_build_jobs
    )
    if measure_heartbeats and worker_count > 6:
        raise ValueError("heartbeat postprocessing supports at most 6 workers")
    if thread_count < 1:
        raise ValueError("threads per repository must be at least 1")
    postprocess_id = (
        f"{source_run_id}_{mode}_w{worker_count}_t{thread_count}_"
        f"{implementation_digest[:8]}_{combination_digest[:8]}"
    )
    heartbeat_reuse_directories: list[str] = []
    heartbeat_reuse_artifact_count = 0
    if measure_heartbeats:
        expected_heartbeat_sources: dict[str, set[str]] = {}
        for capture in capture_rows:
            instance_id = str(capture["instance_id"])
            expected_heartbeat_sources.setdefault(instance_id, set()).update(
                str(archive["source_archive_sha256"])
                for archive in capture.get("archives") or []
                if archive.get("path") == "submitted.sources.tar.gz"
            )
        heartbeat_root = root / "output/heartbeats"
        prior_heartbeat_directories = [
            path
            for path in heartbeat_root.glob(f"{source_run_id}_*-heartbeats-v1")
            if path.is_dir()
            and path.name != f"{postprocess_id}-heartbeats-v1"
        ]

        def valid_heartbeat_count(directory: Path) -> int:
            count = 0
            for artifact_path in directory.glob("*/optimized.json"):
                try:
                    artifact = json.loads(artifact_path.read_text())
                except (OSError, ValueError):
                    continue
                repository_id = artifact_path.parent.name
                if (
                    isinstance(artifact, Mapping)
                    and artifact.get("repository") == repository_id
                    and artifact.get("status") in {"complete", "ineligible"}
                    and artifact.get("method") == "exact_lake_setup_sync_v2"
                    and artifact.get("source_archive_sha256")
                    in expected_heartbeat_sources.get(repository_id, set())
                ):
                    count += 1
            return count

        if prior_heartbeat_directories:
            scored = [
                (valid_heartbeat_count(path), path.stat().st_mtime_ns, path.name, path)
                for path in prior_heartbeat_directories
            ]
            heartbeat_reuse_artifact_count, _, _, selected = max(scored)
            if heartbeat_reuse_artifact_count:
                heartbeat_reuse_directories.append(
                    _relative_or_absolute(root, selected)
                )
    manifest_path, run_dir = _postprocessing_artifact_paths(
        root, evaluation, postprocess_id=postprocess_id, mode=mode
    )
    run_artifact_path = run_dir / "run.yaml"
    # tmux reads "." and ":" in a target as window/pane separators.
    tmux_session = re.sub(r"[.:]", "-", (
        f"postprocess-{source_run_id[:45]}-{mode}-w{worker_count}-t{thread_count}-"
        f"{implementation_digest[:6]}-{combination_digest[:6]}"
    ))
    build_timeout = _postprocessing_build_timeout(container)
    replay_rows = []
    dataset_variant = _text(dataset_pin, "variant", "evaluation.dataset")
    for instance_id in expected_ids:
        repository = _mapping(
            repository_specs[instance_id], "evaluation repository"
        )
        dataset_row = dataset_rows.get(instance_id)
        if dataset_row is None:
            raise ValueError(f"evaluation repository {instance_id} is absent from dataset")
        scope_value = dataset_row.get("scope")
        if not isinstance(scope_value, Mapping):
            database_row = database_rows.get(instance_id)
            if database_row is not None:
                scope_value = database_row.get("scopes")
        scope = _mapping(scope_value, f"dataset.{instance_id}.scope")
        replay_row: dict[str, Any] = {
            "id": instance_id,
            "build_command": (
                "lake build " + str(scope.get("build_target"))
                if scope.get("build_target")
                else "lake build"
            ),
            "metric_include_prefix": str(scope.get("target_dir") or ""),
            "exclude_dirs": list(scope.get("exclude_dirs") or []),
            "protected_declarations": protected.get(instance_id, []),
            "palomar_comparator": palomar_contracts.get(instance_id),
        }
        if instance_id in palomar_contracts and isinstance(dataset_row.get("preprocessing"), Mapping):
            replay_row["palomar_baseline_attestation"] = (
                _preprocessing_palomar_attestation(
                    dataset_row, palomar_contracts[instance_id], dataset_variant
                )
            )
        evaluation_materialization = repository.get("materialization")
        if isinstance(evaluation_materialization, Mapping):
            variants = _mapping(
                dataset_row.get("variants"), f"dataset.{instance_id}.variants"
            )
            artifact = _mapping(
                variants.get(dataset_variant),
                f"dataset.{instance_id}.variants.{dataset_variant}",
            )
            validated_materialization = _validate_materialization_pin(
                evaluation_materialization,
                artifact,
                source_root=source_root,
                instance_id=instance_id,
                allow_relocated_paths=allow_relocated_dataset,
            )
            # Reuse the image evaluation built from the same verified tree (with its
            # Lake dependencies, mathlib included) instead of rebuilding it per pass;
            # the builder reuses a tag only when its tree-hash labels match.
            tag = _text(evaluation_materialization, "tag", f"{instance_id}.materialization")
            replay_row["image"] = tag
            replay_row["materialization"] = {
                **copy.deepcopy(validated_materialization),
                "tag": tag,
                "persist": evaluation_materialization.get("persist") is not False,
                "build_jobs": thread_count,
                "build_timeout_seconds": build_timeout,
            }
        else:
            replay_row["image"] = _text(
                repository, "image", "evaluation repository"
            )
            replay_row["image_id"] = _text(
                repository, "image_id", "evaluation repository"
            )
        replay_row["monitoring_metadata"] = {
            "run_id": postprocess_id,
            "role": "postprocessing",
            "model": str(evaluation.get("model") or "unknown"),
            "instance_id": instance_id,
            "manifest": str(manifest_path),
        }
        replay_rows.append(replay_row)

    manifest: dict[str, Any] = {
        "kind": MANIFEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "postprocess_id": postprocess_id,
        "source_run": {
            "run_id": source_run_id,
            "artifact": _relative_or_absolute(root, source_path),
            "sha256": file_sha256(source_path),
            "repository_root": str(source_root),
            "experiment": _relative_or_absolute(root, evaluation_path),
            "experiment_sha256": file_sha256(evaluation_path),
            "dataset": _relative_or_absolute(root, dataset_path),
            "dataset_sha256": file_sha256(dataset_path),
            "output_directory": str(output_dir),
            "predictions": str(predictions),
            "predictions_sha256": file_sha256(predictions),
            "status": source_status,
            "generation_complete_verified": not ignored_incomplete,
        },
        "run_combination": {
            "policy": (
                "explicit_order_latest_valid_repository_ignore_incomplete"
                if ignore_incomplete
                else "explicit_order_latest_valid_repository"
            ),
            "automatic_discovery": False,
            "combined": len(component_runs) > 1,
            "digest": combination_digest,
            "inputs": component_runs,
            "selection": run_selection,
            "ignored_incomplete_repositories": ignored_incomplete,
        },
        "prompt_lock": {
            "required": prompt_lock_required,
            "sha256": prompt_sha256,
            "expected_sha256": (
                LEANLEAN_20260914_PROMPT_SHA256 if prompt_lock_required else None
            ),
            "verified": (
                prompt_sha256 == LEANLEAN_20260914_PROMPT_SHA256
                if prompt_lock_required
                else True
            ),
            **(
                {"ablation": prompt_ablation}
                if prompt_ablation is not None
                else {}
            ),
        },
        "repository_selection": {
            "policy": "explicit_ordered_subset",
            "explicit": repository_subset is not None,
            "requested_repositories": requested_repositories,
            "repositories": expected_ids,
            "missing_requested_repositories": missing_requested_repositories,
            "unresolved_requested_repositories": ignored_incomplete,
            "allow_missing_subset": allow_missing_subset,
            "excluded_source_repositories": [
                instance_id
                for instance_id in root_attempt.repositories
                if instance_id not in set(expected_ids)
            ],
            "source": copy.deepcopy(dict(repository_subset_source or {})),
        },
        "prefire": {
            "enabled": prefire,
            "purpose": "provisional cache warming before strict combined postprocessing",
            "failed_trajectories_may_be_selected": prefire,
        },
        "dataset_relocation": {
            "enabled": allow_relocated_dataset,
            "policy": "rebase_verified_source_archive_and_cache_paths_only",
            "identity_fields": [
                "tree_sha256",
                "shared_environment.image_id",
                "shared_environment.archive.sha256",
                "shared_environment.archive.bytes",
                "warm_build_cache.sha256",
            ],
        },
        "model": evaluation.get("model"),
        "reasoning_effort": evaluation.get("reasoning_effort"),
        "generator": evaluation.get("harness"),
        "repository_variant": dataset_pin.get("variant"),
        "rounds": evaluation.get("rounds"),
        "mode": mode,
        "repositories": replay_rows,
        "captures": capture_rows,
        "parallelism": {
            "workers": worker_count,
            "worker_scope": "one_repository",
            "threads_per_repository": thread_count,
        },
        "container": {
            "cpus": thread_count,
            "build_jobs": thread_count,
            "endpoints_only": endpoints_only,
            "source_evaluation_cpus": container.get("cpus"),
            "source_evaluation_build_jobs": container.get("build_jobs"),
            "memory": container_memory or container.get("memory"),
            "max_total_memory": _aggregate_memory(
                container_memory or container.get("memory"), worker_count
            ),
            "max_total_cpus": worker_count * thread_count,
            "source_evaluation_max_total_memory": container.get("max_total_memory"),
            "pids_limit": container.get("pids_limit"),
            "nofile_limit": 1048576,
            "offline_dependency_policy": "pinned_lake_packages_nested_v1",
            "cgroup_parent": cgroup_parent or container.get("cgroup_parent"),
            "network_policy": "model_proxy_only",
            "network_mode": "none",
            "build_timeout_seconds": build_timeout,
            "build_output_chars": 12000,
            "checkpoint_infrastructure_retries": replay_recovery.DEFAULT_CHECKPOINT_RETRIES,
            "verification_build_policy": (
                WARM_BASELINE_BUILD_POLICY
                if warm_baseline_build
                else INCREMENTAL_REPLAY_BUILD_ONLY_POLICY
                if build_only_replay
                else INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY
                if skip_checkpoint_lean_verify
                else INCREMENTAL_REPLAY_BUILD_POLICY
                if incremental_replay
                else VERIFICATION_BUILD_POLICY
            ),
            "warm_baseline_build": warm_baseline_build,
            "save_lake_build": save_lake_build,
            "incremental_replay": incremental_replay,
            "skip_checkpoint_lean_verify": skip_checkpoint_lean_verify,
            "build_only_replay": build_only_replay,
            "checkpoint_selection": (
                dict(checkpoint_selection) if checkpoint_selection is not None else None
            ),
            **(
                {"lake_build_output_directory": str(run_dir / "lake-builds")}
                if save_lake_build
                else {}
            ),
        },
        "resource_overrides": {
            "cgroup_parent": cgroup_parent,
            "container_memory": container_memory,
        },
        "comparison": (
            {
                "verification_command": (
                    "LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify"
                ),
                "result_schema": LEAN_VERIFY_RESULT_SCHEMA,
                "execution": (
                    "disabled_reuse_canonical_endpoint_validation"
                    if build_only_replay
                    else "after_successful_submitted_endpoint_build_only"
                    if skip_checkpoint_lean_verify
                    else "after_each_successful_checkpoint_build"
                ),
                "network_policy": "model_proxy_only",
                "network_mode": "none",
                "repository_count": len(palomar_contracts),
            }
            if replay
            else {
                "palomar_engine": "leanprover/comparator",
                "palomar_contract_schema": PALOMAR_VALIDATION_SCHEMA,
                "trusted_inputs": (
                    "registry-pinned Challenge.lean and original Comparator JSON "
                    "reintroduced before each authoritative checkpoint verification"
                ),
                "kernel_replay": (
                    "Lean builtin plus exact registered external-kernel policy"
                ),
                "network_policy": "model_proxy_only",
                "network_mode": "none",
                "palomar_repository_count": len(palomar_contracts),
            }
        ),
        "evaluation": {
            "rebuild_reports": True,
            "source_metrics": "every captured checkpoint",
            "build_submitted_final": True,
            "verification_build_policy": (
                WARM_BASELINE_BUILD_POLICY
                if warm_baseline_build
                else INCREMENTAL_REPLAY_BUILD_ONLY_POLICY
                if build_only_replay
                else INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY
                if skip_checkpoint_lean_verify
                else INCREMENTAL_REPLAY_BUILD_POLICY
                if incremental_replay
                else VERIFICATION_BUILD_POLICY
            ),
            "warm_baseline_build": warm_baseline_build,
            "save_lake_build": save_lake_build,
            "incremental_replay": incremental_replay,
            "skip_checkpoint_lean_verify": skip_checkpoint_lean_verify,
            "build_only_replay": build_only_replay,
            "checkpoint_selection": (
                dict(checkpoint_selection) if checkpoint_selection is not None else None
            ),
            **(
                {
                    "submitted_final_lean_verify": not build_only_replay,
                    "checkpoint_lean_verify": not (build_only_replay or skip_checkpoint_lean_verify),
                }
                if replay
                else {
                    "submitted_final_signature_validation": True,
                    "palomar_signature_engine": "official_comparator_exit_verdict",
                }
            ),
            "endpoints_only": endpoints_only,
            "build_edit_checkpoints": replay,
            "timed_out_checkpoint_builds": (
                "none" if endpoints_only else "all" if replay else "newest_to_oldest_until_green"
            ),
            "headline_metric": "macro_average_selected_state_word_compression",
            "failed_submission_score": 0,
            "timed_out_selection": (
                "submitted_final_state"
                if endpoints_only
                else (
                    "latest_build_and_lean_verify_green_checkpoint"
                    if replay
                    else "latest_build_and_signature_green_checkpoint"
                )
            ),
            **(
                {"lean_verify_failure_score": 0}
                if replay
                else {"signature_failure_score": 0}
            ),
        },
        "heartbeats": {
            "enabled": measure_heartbeats,
            "run_id": f"{postprocess_id}-heartbeats-v1",
            "postprocess_id": postprocess_id,
            "measurement": "stripped_to_submitted_final",
            "variants_computed": ["optimized"],
            "baseline_run_id": f"{re.sub(r'[^A-Za-z0-9_-]+', '-', str(dataset_pin.get('id') or dataset_path.stem)).strip('-')}-stripped-heartbeats-v1",
            "method": "exact_lake_setup_sync_v2",
            "unit": "raw_heartbeats",
            "clean_build": "reuse authoritative postprocessing clean build in the same live container",
            "primary_heartbeat_count": "body_heartbeats",
            "also_record": ["total_heartbeats", "import_heartbeats", "clean_build_seconds", "lean_tokens"],
            "lean_threads": 1,
            "elab_async": False,
            "repetitions": 1,
            "variation_warning_relative_range_pct": 5.0,
            "file_timeout_seconds": build_timeout,
            "output_directory": _relative_or_absolute(root, output_dir),
            "result_filename": "heartbeats.json",
            "reuse_policy": "exact_source_archive_method_consensus_v1",
            "reuse_selection": "maximum_exact_artifact_coverage_then_newest",
            "reuse_artifact_count": heartbeat_reuse_artifact_count,
            "reuse_directories": heartbeat_reuse_directories,
            "failures": "null counts; retain repository in coverage denominator",
        },
        "monitoring": {
            "snapshots": True,
            "every_n_edits": 1,
            "native_traces": "retained_from_source_evaluation",
            "baseline_snapshots": "retained_from_source_evaluation",
            "intermediate_snapshots": "retained_from_source_evaluation",
            "terminal_snapshots": "retained_from_source_evaluation",
        },
        "outputs": {
            "run_artifact": _relative_or_absolute(root, run_artifact_path),
            "summary": str(output_dir / ("replay_summary.json" if replay else SUMMARY_FILENAME)),
            "derived_playback": f"{output_dir}/*/playback/capture_*/{REPLAY_PLAYBACK if replay else DERIVED_PLAYBACK}",
            **(
                {"lake_builds": str(run_dir / "lake-builds")}
                if save_lake_build
                else {}
            ),
            "tmux_session": tmux_session,
        },
        "implementation": {
            "files": implementation,
            "palomar_contracts": {
                key: value["sha256"]
                for key, value in sorted(palomar_contracts.items())
            },
        },
    }
    return PreparedPostprocessing(
        manifest_path=manifest_path,
        manifest=manifest,
        run_artifact_path=run_artifact_path,
        source_root=source_root,
        source_run_artifact_path=source_path,
        source_run=source_run,
        evaluation_manifest_path=evaluation_path,
        evaluation_manifest=evaluation,
        dataset_path=dataset_path,
        dataset=dataset,
        predictions_path=predictions,
        predictions=prediction_payload,
        output_dir=output_dir,
        replay=replay,
        tmux_session=tmux_session,
    )


def write_run_definition(prepared: PreparedPostprocessing, *, repo_root: Path) -> None:
    content = _dump(prepared.manifest)
    postprocess_id = str(prepared.manifest["postprocess_id"])
    _write_exact(prepared.manifest_path, content, postprocess_id)
    if prepared.run_artifact_path.exists():
        existing = _load_yaml(prepared.run_artifact_path, "postprocessing run artifact")
        experiment_sha = _mapping(
            existing.get("experiment"), "postprocess.experiment"
        ).get("sha256")
        if experiment_sha != hashlib.sha256(content.encode()).hexdigest():
            raise RuntimeError(f"{prepared.run_artifact_path}: pins a different experiment")
        return
    artifact = {
        "kind": RUN_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "postprocess_id": postprocess_id,
        "source_run_id": prepared.source_run["run_id"],
        "mode": prepared.manifest["mode"],
        "status": "configured",
        "experiment": {
            "manifest": _relative_or_absolute(repo_root, prepared.manifest_path),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        },
        "source_run": dict(prepared.manifest["source_run"]),
        "artifacts": dict(prepared.manifest["outputs"]),
    }
    _atomic_yaml(prepared.run_artifact_path, artifact)


def update_run_status(
    path: Path, status: str, *, error: str | None = None, exit_code: int | None = None
) -> None:
    if status not in {"configured", "running", "complete", "failed"}:
        raise ValueError(f"unsupported postprocessing status {status!r}")
    artifact = dict(_load_yaml(path, "postprocessing run artifact"))
    if artifact.get("kind") != RUN_ARTIFACT_KIND:
        raise ValueError(f"{path}: invalid postprocessing run artifact")
    artifact["status"] = status
    lifecycle = dict(_mapping(artifact.get("lifecycle", {}), "postprocess.lifecycle"))
    if status == "running":
        lifecycle.setdefault("started_at", _utc_now())
        lifecycle["last_attempt_started_at"] = _utc_now()
        lifecycle["attempt_count"] = int(lifecycle.get("attempt_count") or 0) + 1
        lifecycle.pop("finished_at", None)
        lifecycle.pop("exit_code", None)
    elif status in {"complete", "failed"}:
        lifecycle["finished_at"] = _utc_now()
    if exit_code is not None:
        lifecycle["exit_code"] = int(exit_code)
    if error:
        lifecycle["error"] = error[:4000]
    else:
        lifecycle.pop("error", None)
    artifact["lifecycle"] = lifecycle
    _atomic_yaml(path, artifact)


def load_run_manifest(path: Path, *, repo_root: Path) -> PreparedPostprocessing:
    root = repo_root.resolve()
    manifest = _load_yaml(path.resolve(), "postprocessing manifest")
    if manifest.get("kind") != MANIFEST_KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported postprocessing manifest")
    combination = _mapping(
        manifest.get("run_combination"), "manifest.run_combination"
    )
    inputs = combination.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("manifest.run_combination.inputs must be a non-empty list")
    source_paths: list[Path] = []
    for index, raw in enumerate(inputs):
        item = _mapping(raw, f"manifest.run_combination.inputs[{index}]")
        source_path = _resolve_path(
            root,
            _text(item, "artifact", f"manifest.run_combination.inputs[{index}]"),
        )
        if file_sha256(source_path) != item.get("artifact_sha256"):
            raise ValueError(f"source run artifact drifted: {source_path}")
        source_paths.append(source_path)
    replay = manifest.get("mode") == "replay"
    parallelism = _mapping(manifest.get("parallelism"), "manifest.parallelism")
    prefire_config = _mapping(manifest.get("prefire", {}), "manifest.prefire")
    prefire = prefire_config.get("enabled") is True
    heartbeat_config = _mapping(manifest.get("heartbeats", {}), "manifest.heartbeats")
    measure_heartbeats = heartbeat_config.get("enabled") is True
    relocation_config = _mapping(
        manifest.get("dataset_relocation", {}), "manifest.dataset_relocation"
    )
    allow_relocated_dataset = relocation_config.get("enabled") is True
    ignore_incomplete = bool(
        combination.get("ignored_incomplete_repositories")
    )
    repository_selection = _mapping(
        manifest.get("repository_selection", {}), "manifest.repository_selection"
    )
    repository_subset = repository_selection.get(
        "requested_repositories", repository_selection.get("repositories")
    )
    repository_subset_explicit = repository_selection.get("explicit")
    allow_missing_subset = repository_selection.get("allow_missing_subset") is True
    if not isinstance(repository_subset, list) or not repository_subset or not all(
        isinstance(value, str) and value for value in repository_subset
    ):
        raise ValueError(
            "manifest.repository_selection.repositories must be a non-empty string list"
        )
    repository_subset_source = _mapping(
        repository_selection.get("source", {}), "manifest.repository_selection.source"
    )
    if not repository_subset_source:
        repository_subset_source = None
    if repository_subset_explicit is None:
        repository_subset_explicit = repository_subset_source is not None
    if not isinstance(repository_subset_explicit, bool):
        raise ValueError("manifest.repository_selection.explicit must be a boolean")
    source_config = (repository_subset_source or {}).get("config")
    source_sha256 = (repository_subset_source or {}).get("sha256")
    if source_config:
        source_config_path = _resolve_path(root, str(source_config))
        if file_sha256(source_config_path) != source_sha256:
            raise ValueError(
                f"repository subset source config drifted: {source_config_path}"
            )
        source_document = _load_yaml(source_config_path, "repository subset source")
        if source_document.get("repositories") != repository_subset:
            raise ValueError("repository subset source list drifted")
        source_agent = _mapping(
            source_document.get("agent"), "repository subset source.agent"
        )
        source_prompt = _text(
            source_agent, "prompt", "repository subset source.agent"
        )
        source_prompt_sha256 = hashlib.sha256(
            source_prompt.encode("utf-8")
        ).hexdigest()
        if source_prompt_sha256 != repository_subset_source.get(
            "canonical_prompt_sha256"
        ):
            raise ValueError("repository subset source prompt drifted")
    manifest_container = _mapping(
        manifest.get("container"), "manifest.container"
    )
    cgroup_parent_override, container_memory_override = (
        _manifest_resource_overrides(manifest)
    )
    prepared = resolve_run(
        source_paths,
        repo_root=root,
        replay=replay,
        workers=int(parallelism.get("workers", 1)),
        threads_per_repository=int(parallelism["threads_per_repository"]),
        prefire=prefire,
        measure_heartbeats=measure_heartbeats,
        allow_relocated_dataset=allow_relocated_dataset,
        ignore_incomplete=ignore_incomplete,
        endpoints_only=manifest.get("evaluation", {}).get("endpoints_only", False),
        repository_subset=(repository_subset if repository_subset_explicit else None),
        allow_missing_subset=allow_missing_subset,
        repository_subset_source=repository_subset_source,
        cgroup_parent=cgroup_parent_override,
        container_memory=container_memory_override,
        warm_baseline_build=manifest_container.get("warm_baseline_build") is True,
        save_lake_build=manifest_container.get("save_lake_build") is True,
        incremental_replay=manifest_container.get("incremental_replay") is True,
        skip_checkpoint_lean_verify=(
            manifest_container.get("skip_checkpoint_lean_verify") is True
        ),
        build_only_replay=manifest_container.get("build_only_replay") is True,
        checkpoint_selection=manifest_container.get("checkpoint_selection"),
    )
    if prepared.manifest != manifest:
        raise ValueError(f"{path}: postprocessing inputs or implementation drifted")
    return prepared


def _playback_timeout(capture_dir: Path) -> bool:
    trace = capture_dir / "standardized-trace.json"
    if not trace.is_file():
        return False
    try:
        payload = json.loads(trace.read_text())
    except (OSError, ValueError):
        return False
    source = payload.get("source")
    return isinstance(source, Mapping) and source.get("timed_out") is True


def _archive_for_point(capture_dir: Path, point: Mapping[str, Any]) -> Path:
    name = Path(_text(point, "snapshot_path", "checkpoint")).name
    path = capture_dir / name
    if not path.is_file():
        raise ValueError(f"checkpoint archive is missing: {path}")
    actual = _canonical_archive_sha256(path)
    expected = _text(point, "source_archive_sha256", "checkpoint")
    if actual != expected:
        raise ValueError(f"checkpoint archive digest mismatch: {path}")
    return path


def _patch_sha256(patch: str) -> str:
    return hashlib.sha256(
        patch.encode("utf-8", errors="surrogateescape")
    ).hexdigest()


def _submitted_endpoint(
    playback: Mapping[str, Any],
    capture_dir: Path,
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    patch: str,
    *,
    defer_build: bool = False,
) -> dict[str, Any]:
    """Return playback ending at the exact patch applied to the baseline."""

    result = copy.deepcopy(playback)
    points = result.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{capture_dir}: no checkpoint points")
    patch_sha = _patch_sha256(patch)
    existing = points[-1]
    if (
        isinstance(existing, Mapping)
        and existing.get("endpoint_role") == "submitted_patch"
    ):
        if existing.get("submission_patch_sha256") != patch_sha:
            raise ValueError(f"{capture_dir}: submitted endpoint patch digest drifted")
        _archive_for_point(capture_dir, existing)
        result["submitted_endpoint_captured"] = True
        result["submitted_endpoint_source_archive_sha256"] = existing.get(
            "source_archive_sha256"
        )
        return result

    if defer_build:
        # Endpoint construction itself needs a materialized image, before the
        # later validation/build fast path has a chance to request one.
        raise CaptureBuildRequired(str(repository["id"]))
    env = _build_environment(repository, container, 1)
    candidate = capture_dir / f".submitted-{time.time_ns()}.sources.tar.gz"
    destination = capture_dir / "submitted.sources.tar.gz"
    patch_path = "/tmp/leanlean-submitted.patch"
    try:
        baseline = _archive_for_point(capture_dir, points[0])
        env.restore_source_archive(baseline, ["."], timeout=900)
        correct, actual = _verify_restored_source(
            env,
            baseline,
            str(points[0]["source_archive_sha256"]),
            timeout=900,
        )
        if not correct:
            raise RuntimeError(
                "submitted endpoint baseline restore mismatch: " + actual
            )
        env.write_file_bytes(
            patch_path,
            patch.encode("utf-8", errors="surrogateescape"),
        )
        apply_command = (
            "cd /testbed && "
            f"{{ test ! -s {shlex.quote(patch_path)} || "
            f"git apply --binary --whitespace=nowarn {shlex.quote(patch_path)}; }}"
        )
        apply_timeout = min(int(container["build_timeout_seconds"]), 900)
        applied = env.execute(apply_command, timeout=apply_timeout)
        absent_deletions: list[str] = []
        if applied.get("returncode"):
            # A deletion of a path the baseline never had is a no-op, but git
            # apply rejects it and would discard the whole submission.  Retry
            # once without only those hunks; git apply is atomic on failure, so
            # the baseline is still pristine here.
            baseline_names = {
                name.removeprefix("./") for name in _archive_source_files(baseline)
            }
            if "__scoped_source__" not in baseline_names:
                filtered, absent_deletions = drop_absent_deletions(
                    patch, lambda path: path in baseline_names
                )
            if absent_deletions:
                env.write_file_bytes(
                    patch_path,
                    filtered.encode("utf-8", errors="surrogateescape"),
                )
                applied = env.execute(apply_command, timeout=apply_timeout)
        if applied.get("returncode"):
            result.update(
                {
                    "submitted_endpoint_captured": False,
                    "submitted_endpoint_capture_error": str(
                        applied.get("output") or "submitted patch did not apply"
                    )[-int(container["build_output_chars"]):],
                }
            )
            return result
        metadata = env.export_source_archive(
            candidate,
            ["."],
            timeout=900,
            retry_transient=False,
        )
        if destination.exists():
            if _canonical_archive_sha256(destination) != _canonical_archive_sha256(
                candidate
            ):
                raise RuntimeError(
                    f"refusing to replace a different submitted endpoint: {destination}"
                )
            candidate.unlink(missing_ok=True)
        else:
            os.replace(candidate, destination)

        submitted_files = {
            name.removeprefix("./"): payload
            for name, payload in _archive_source_files(destination).items()
        }
        monitored_files = {
            name.removeprefix("./"): payload
            for name, payload in _archive_source_files(
                _archive_for_point(capture_dir, points[-1])
            ).items()
        }
        changed_lean = sorted(
            name
            for name in set(monitored_files) | set(submitted_files)
            if name.endswith(".lean")
            and monitored_files.get(name) != submitted_files.get(name)
        )
        payload = destination.read_bytes()
        reference_parent = Path(str(points[-1]["snapshot_path"])).parent
        snapshot_path = str(reference_parent / destination.name)
        submitted_point = {
            "kind": "final",
            "endpoint_role": "submitted_patch",
            "edit_index": points[-1].get("edit_index"),
            "elapsed_seconds": points[-1].get("elapsed_seconds"),
            "cost_usd": points[-1].get("cost_usd"),
            "cost_boundary_exact": True,
            "exact": True,
            "matches_submission": True,
            "changed_files_from_monitored_final": changed_lean,
            "submission_patch_sha256": patch_sha,
            "absent_path_deletions_ignored": absent_deletions,
            "file_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(submitted_files.items())
            },
            "files": sorted(submitted_files),
            "snapshot_path": snapshot_path,
            "snapshot_format": "deterministic-source-tar-gzip-v1",
            "snapshot_artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "snapshot_artifact_bytes": len(payload),
            "snapshot_sha256": metadata["source_archive_sha256"],
            "snapshot_digest_basis": "canonical uncompressed tar stream",
            "snapshot_bytes": metadata["archive_bytes"],
            "source_archive_bytes": metadata["source_archive_bytes"],
            "source_archive_sha256": metadata["source_archive_sha256"],
            "capture_seconds": 0.0,
            "replay_status": "pending_postprocessing",
        }
        points[-1]["endpoint_role"] = "monitored_final_state"
        points[-1]["matches_submission"] = result.get(
            "final_snapshot_matches_submission"
        ) is True
        points.append(submitted_point)
        result["submitted_endpoint_captured"] = True
        result["submitted_endpoint_source_archive_sha256"] = metadata[
            "source_archive_sha256"
        ]
        if absent_deletions:
            result["submitted_endpoint_absent_path_deletions_ignored"] = (
                absent_deletions
            )
        return result
    finally:
        candidate.unlink(missing_ok=True)
        env.cleanup()


def _attach_source_metrics(
    playback: dict[str, Any],
    capture_dir: Path,
    repository: Mapping[str, Any],
    *,
    endpoints_only: bool = False,
    selected_edit_indices: Collection[int] | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(playback)
    points = result.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{capture_dir}: no checkpoint points")
    baseline_words = baseline_tokens = None
    include_prefix = str(repository.get("metric_include_prefix") or "")
    excludes = list(repository.get("exclude_dirs") or [])
    excluded_files: list[str] = []
    palomar = repository.get("palomar_comparator")
    if isinstance(palomar, Mapping):
        challenge = palomar.get("challenge")
        if isinstance(challenge, Mapping):
            excluded_files.append(
                _text(challenge, "source_path", "Palomar Challenge")
            )
    metric_indices = (
        {0, len(points) - 1}
        if endpoints_only
        else set(range(len(points)))
    )
    if selected_edit_indices is not None and not endpoints_only:
        # Sparse replay: measure only the baseline, the submitted endpoint and
        # the selected edit checkpoints.
        wanted = set(selected_edit_indices)
        metric_indices = {0, len(points) - 1} | {
            index
            for index, point in enumerate(points)
            if isinstance(point, dict)
            and (
                int(point.get("edit_index") or 0) in wanted
                or point.get("endpoint_role") == "submitted_patch"
            )
        }
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"{capture_dir}: malformed checkpoint {index}")
        if index not in metric_indices:
            continue
        archive = _archive_for_point(capture_dir, point)
        metrics = measure_source_archive(
            archive,
            exclude_dirs=excludes,
            exclude_files=excluded_files,
            include_prefix=include_prefix,
        )
        if index == 0:
            baseline_words = metrics.words
            baseline_tokens = metrics.lean_tokens
            if baseline_words <= 0 or baseline_tokens <= 0:
                raise ValueError(f"{capture_dir}: baseline source metrics are empty")
        assert baseline_words is not None and baseline_tokens is not None
        point.update(
            {
                "words": metrics.words,
                "words_saved": baseline_words - metrics.words,
                "word_compression_pct": round(
                    100 * (baseline_words - metrics.words) / baseline_words,
                    6,
                ),
                "lean_tokens": metrics.lean_tokens,
                "lean_tokens_saved": baseline_tokens - metrics.lean_tokens,
                "lean_token_compression_pct": round(
                    100
                    * (baseline_tokens - metrics.lean_tokens)
                    / baseline_tokens,
                    6,
                ),
                "lean_file_count": metrics.lean_file_count,
            }
        )
    final = points[-1]
    result.update(
        {
            "postprocessing_format": "leanlean-postprocessed-playback-v1",
            "postprocessed_at": _utc_now(),
            "source_playback_sha256": file_sha256(capture_dir / "playback.json"),
            "metric_basis": "host metrics from immutable captured source archives",
            "baseline_words": baseline_words,
            "post_words": final["words"],
            "words_saved": final["words_saved"],
            "baseline_lean_tokens": baseline_tokens,
            "post_lean_tokens": final["lean_tokens"],
            "lean_tokens_saved": final["lean_tokens_saved"],
        }
    )
    return result


def _preserve_checkpoint_results(
    result: dict[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    replay: bool,
) -> None:
    if not isinstance(previous, Mapping):
        return
    if previous.get("source_playback_sha256") != result.get("source_playback_sha256"):
        return
    previous_by_digest = {
        str(point.get("source_archive_sha256")): point
        for point in previous.get("points") or []
        if isinstance(point, Mapping)
    }
    for point in result.get("points") or []:
        if not isinstance(point, dict):
            continue
        old = previous_by_digest.get(str(point.get("source_archive_sha256")))
        if not isinstance(old, Mapping):
            continue
        if isinstance(old.get("build"), Mapping):
            point["build"] = copy.deepcopy(old["build"])
        if replay and isinstance(old.get("lean_verify"), Mapping):
            point["lean_verify"] = copy.deepcopy(old["lean_verify"])
        elif not replay and isinstance(old.get("signature_validation"), Mapping):
            point["signature_validation"] = copy.deepcopy(
                old["signature_validation"]
            )
        if isinstance(old.get("authoritative_metric"), Mapping):
            point["authoritative_metric"] = copy.deepcopy(
                old["authoritative_metric"]
            )


def _selected_state(
    playback: Mapping[str, Any], *, timed_out: bool, replay: bool
) -> dict[str, Any]:
    points = playback.get("points")
    assert isinstance(points, list) and points
    if not timed_out:
        point = points[-1]
        build_passed = playback.get("build_passed") is True
        verification_passed = (
            playback.get("lean_verify_passed") is True
            if replay
            else playback.get("signatures_preserved") is True
        )
        eligible = build_passed and verification_passed
        verification = (
            {"lean_verify_passed": verification_passed}
            if replay
            else {"signatures_preserved": verification_passed}
        )
        return {
            "policy": "submitted_final_state",
            "edit_index": point.get("edit_index"),
            "build_passed": build_passed,
            **verification,
            "eligible": eligible,
            "word_compression_pct": point.get("word_compression_pct") if eligible else 0.0,
            "lean_token_compression_pct": (
                point.get("lean_token_compression_pct") if eligible else 0.0
            ),
        }
    verification_key = "lean_verify" if replay else "signature_validation"
    verdict_key = "passed" if replay else "preserved"
    green = [
        point
        for point in points
        if isinstance(point, Mapping)
        and isinstance(point.get("build"), Mapping)
        and point["build"].get("passed") is True
        and isinstance(point.get(verification_key), Mapping)
        and point[verification_key].get(verdict_key) is True
    ]
    point = green[-1] if green else points[0]
    eligible = bool(green)
    return {
        "policy": (
            "latest_build_and_lean_verify_green_checkpoint"
            if replay
            else "latest_build_and_signature_green_checkpoint"
        ),
        "edit_index": point.get("edit_index"),
        "build_passed": eligible,
        **(
            {"lean_verify_passed": eligible}
            if replay
            else {"signatures_preserved": eligible}
        ),
        "eligible": eligible,
        "word_compression_pct": point.get("word_compression_pct") if eligible else 0.0,
        "lean_token_compression_pct": point.get("lean_token_compression_pct") if eligible else 0.0,
    }


def _build_environment(
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    point_count: int,
) -> DockerEnvironment:
    # Matches the preprocessing verification container: nanoda's kernel
    # re-check overflows the 8 MiB default main stack on large developments.
    stack_bytes = (
        max(int(container.get("lean_thread_stack_kib") or 0), COMPARATOR_STACK_KIB)
        * 1024
    )
    # Keep dead-container state until explicit cleanup records OOM/exit diagnostics.
    run_args = [
        f"--cpus={container['cpus']}",
        f"--memory={container['memory']}",
        f"--memory-swap={container['memory']}",
        f"--ulimit=stack={stack_bytes}:{stack_bytes}",
        f"--ulimit=nofile={int(container.get('nofile_limit', 1048576))}:{int(container.get('nofile_limit', 1048576))}",
    ]
    cgroup = str(container.get("cgroup_parent") or "")
    if cgroup:
        run_args.append(f"--cgroup-parent={cgroup}")
    timeout = int(container["build_timeout_seconds"])
    return DockerEnvironment(
        offline_lake=isinstance(repository.get("palomar_comparator"), Mapping),
        image=str(repository["image_id"]),
        cwd="/testbed",
        env={},
        forward_env=[],
        monitoring_metadata=dict(repository.get("monitoring_metadata") or {}),
        timeout=max(600, timeout),
        run_args=run_args,
        network_mode="none",
        network_policy="model_proxy_only",
        container_pids_limit=int(container["pids_limit"]),
        container_timeout=f"{max(600, point_count * timeout + 600)}s",
        container_grace_seconds=60,
    )


@contextlib.contextmanager
def _materialized_repository_image(
    repository: Mapping[str, Any],
):
    """Resolve one postprocessing image; retire it afterwards unless the evaluation keeps it."""

    resolved = dict(repository)
    instance_id = str(resolved["id"])
    materialization = resolved.get("materialization")
    retirement: list[dict[str, Any]] = []
    try:
        if isinstance(materialization, Mapping):
            ensure_materialized_image_from_record(instance_id, materialization)
            resolved["image_id"] = materialized_image_id(
                str(materialization["tag"])
            )
        yield resolved, retirement
    finally:
        if isinstance(materialization, Mapping) and not materialization.get("persist"):
            retirement.extend(retire_image_tags(str(materialization["tag"])))


def _verify_restored_source(
    env: DockerEnvironment, archive: Path, expected: str, *, timeout: int
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="postprocess-source-check-") as directory:
        exported_path = Path(directory) / "source.tar.gz"
        env.export_source_archive(
            exported_path, ["."], timeout=timeout
        )
        expected_files = {
            name.removeprefix("./"): payload
            for name, payload in _archive_source_files(archive).items()
        }
        actual_files = {
            name.removeprefix("./"): payload
            for name, payload in _archive_source_files(exported_path).items()
        }
    digest = hashlib.sha256()
    for name, payload in sorted(actual_files.items()):
        encoded = name.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    del expected  # the checkpoint tar digest was verified before restoration
    return actual_files == expected_files, digest.hexdigest()


def _palomar_validation_stub(
    repository: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    requested = sorted(set(repository.get("protected_declarations") or []))
    return {
        "schema": PALOMAR_VALIDATION_SCHEMA,
        "engine": "leanprover/comparator",
        "checked": False,
        "preserved": False,
        "named_interface_preserved": False,
        "semantic_integrity_preserved": False,
        "axioms_preserved": False,
        "contract_sha256": contract["sha256"],
        "requested_names": requested,
        "requested_count": len(requested),
        "protected_count": len(requested),
        "requested_definition_count": len(contract.get("definition_names") or []),
        "semantic_definition_count": len(contract.get("definition_names") or []),
        "dependency_definition_count": 0,
        "local_axiom_count": 0,
    }


def _copy_palomar_tools(
    env: DockerEnvironment, *, repo_root: Path, contract: Mapping[str, Any]
) -> None:
    install_palomar_tools(env, repo_root=repo_root, contract=contract)


def _registered_palomar_fixture_payloads(
    *,
    repo_root: Path,
    repository: Mapping[str, Any],
) -> dict[str, bytes]:
    """Return the exact agent-visible benchmark files for this repository."""

    contract = _mapping(
        repository.get("palomar_comparator"), "repository.palomar_comparator"
    )
    challenge_pin = _mapping(contract.get("challenge"), "challenge")
    challenge_source = _safe_repository_path(
        challenge_pin.get("source_path"), "registered Challenge path"
    )
    challenge, original_config, _runtime_config = load_palomar_evidence(
        repo_root=repo_root, contract=contract
    )
    proof_length = _render_task_proof_length_script(
        list(repository.get("exclude_dirs") or []),
        str(repository.get("metric_include_prefix") or ""),
        exclude_files=[challenge_source],
    ).encode("utf-8")
    return {
        challenge_source: challenge,
        "comparator.json": original_config,
        "proof_length.py": proof_length,
    }


def _benchmark_fixture_integrity(
    point: Mapping[str, Any],
    capture_dir: Path,
    repository: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Audit agent-visible helpers before the evaluator replaces them."""

    archive = _archive_for_point(capture_dir, point)
    actual_files = _archive_source_files(archive)
    expected_files = _registered_palomar_fixture_payloads(
        repo_root=repo_root,
        repository=repository,
    )
    files: dict[str, Any] = {}
    altered: list[str] = []
    for path, expected in expected_files.items():
        actual = actual_files.get(path)
        status = (
            "missing"
            if actual is None
            else "unchanged"
            if actual == expected
            else "modified"
        )
        if status != "unchanged":
            altered.append(path)
        files[path] = {
            "status": status,
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "actual_sha256": (
                hashlib.sha256(actual).hexdigest()
                if actual is not None
                else None
            ),
        }
    return {
        "schema": BENCHMARK_FIXTURE_INTEGRITY_SCHEMA,
        "checked": True,
        "preserved": not altered,
        "severity": "ok" if not altered else "warning",
        "altered_files": altered,
        "files": files,
        "evaluation_action": (
            "registered copies restored before authoritative verification "
            "and metric calculation"
        ),
        "score_uses_agent_copy": False,
    }


def _clear_verification_cache(env: DockerEnvironment, *, timeout: int) -> None:
    """Discard project/config artifacts after every restore, retaining pinned packages."""
    script = render_container_cache_cleanup(clear_project=True, clear_config=True)
    result = env.execute("python3 -c " + shlex.quote(script), timeout=min(timeout, 900))
    if result.get("returncode", 1) != 0 or result.get("timed_out"):
        raise RuntimeError(f"verification cache cleanup failed: {result.get('output', '')}")


def _prune_orphan_build_artifacts(env: DockerEnvironment, *, timeout: int) -> None:
    """Remove build artifacts whose module source is gone, keeping the rest."""
    script = render_container_cache_cleanup()
    result = env.execute("python3 -c " + shlex.quote(script), timeout=min(timeout, 900))
    if result.get("returncode", 1) != 0 or result.get("timed_out"):
        raise RuntimeError(f"orphan build artifact cleanup failed: {result.get('output', '')}")


def _prepare_verification_build(
    env: DockerEnvironment,
    container: Mapping[str, Any],
    *,
    timeout: int,
) -> dict[str, Any] | None:
    policy = str(container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY)
    if policy == WARM_BASELINE_BUILD_POLICY:
        result = env.execute(
            "test -d /testbed/.lake/build && "
            "find /testbed/.lake/build -type f -print -quit",
            timeout=min(timeout, 900),
        )
        sample = str(result.get("output") or "").strip()
        if result.get("returncode", 1) != 0 or result.get("timed_out") or not sample:
            raise RuntimeError(
                "stripped-baseline warm .lake/build is absent or empty: "
                + sample[-1000:]
            )
        return {
            "verified": True,
            "policy": policy,
            "sample_file": sample,
            "cache_source": "materialized_stripped_baseline_image",
        }
    if policy != VERIFICATION_BUILD_POLICY:
        raise ValueError(f"unsupported verification build policy {policy!r}")
    _clear_verification_cache(env, timeout=timeout)
    return None


def _install_replay_lean_verify(
    env: DockerEnvironment,
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Install the exact agent-facing verifier and its registry-pinned inputs."""

    contract = _mapping(
        repository.get("palomar_comparator"), "repository.palomar_comparator"
    )
    if not same_identifier(contract.get("schema"), PALOMAR_VALIDATION_SCHEMA):
        raise RuntimeError("replay requires a registered lean_verify contract")
    toolchain = env.read_file("/testbed/lean-toolchain").strip()
    if toolchain != contract.get("lean_toolchain"):
        raise RuntimeError(
            "candidate Lean toolchain differs from registered lean_verify toolchain: "
            f"{toolchain!r} != {contract.get('lean_toolchain')!r}"
        )
    challenge, original_config, runtime_config = load_palomar_evidence(
        repo_root=repo_root, contract=contract
    )
    challenge_pin = _mapping(contract.get("challenge"), "challenge")
    config_pin = _mapping(contract.get("configuration"), "configuration")
    challenge_source = _safe_repository_path(
        challenge_pin.get("source_path"), "registered Challenge path"
    )
    config_source = _safe_repository_path(
        config_pin.get("source_path"), "registered Comparator config path"
    )
    refresh_paths = [
        "/usr/local/bin/lean_verify",
        "/usr/local/bin/palomar-comparator",
        "/usr/local/bin/palomar-landrun",
        "/usr/local/bin/palomar-landrun-wrapper",
        "/usr/local/bin/palomar-lean4export",
        "/usr/local/bin/palomar-nanoda",
        "/tmp/leanlean-palomar-comparator-registered.json",
        "/tmp/leanlean-palomar-comparator.json",
        "/tmp/leanlean-palomar-comparator-defaults.json",
        "/tmp/leanlean-palomar-challenge.lean",
        "/tmp/leanlean-palomar-challenge-path",
        "/tmp/leanlean-palomar-build-jobs",
        "/testbed/" + challenge_source,
        "/testbed/" + config_source,
        "/testbed/comparator.json",
    ]
    refresh = env.execute(
        "chmod u+w "
        + " ".join(shlex.quote(path) for path in refresh_paths)
        + " 2>/dev/null || true",
        timeout=60,
    )
    if refresh.get("timed_out"):
        raise RuntimeError("timed out refreshing pinned lean_verify inputs")
    install_palomar_tools(env, repo_root=repo_root, contract=contract)
    install_agent_verify_wrapper(env, repo_root=repo_root)
    env.write_file_bytes("/testbed/" + challenge_source, challenge)
    env.write_file_bytes("/testbed/" + config_source, original_config)
    if config_source != "comparator.json":
        env.write_file_bytes("/testbed/comparator.json", original_config)
    env.write_file_bytes(
        "/tmp/leanlean-palomar-comparator-registered.json", original_config
    )
    env.write_file_bytes(
        "/tmp/leanlean-palomar-comparator.json", runtime_config
    )
    env.write_file_bytes(
        "/tmp/leanlean-palomar-comparator-defaults.json",
        (
            json.dumps(
                dict(
                    config_pin.get("runtime_defaults_applied") or {}
                )
            )
            + "\n"
        ).encode("utf-8"),
    )
    env.write_file_bytes(
        "/tmp/leanlean-palomar-challenge.lean", challenge
    )
    env.write_file_bytes(
        "/tmp/leanlean-palomar-challenge-path",
        (challenge_source + "\n").encode("utf-8"),
    )
    env.write_file_bytes(
        "/tmp/leanlean-palomar-build-jobs",
        (str(max(1, int(container["build_jobs"]))) + "\n").encode("utf-8"),
    )
    permissions = env.execute(
        "chmod 0755 /usr/local/bin/lean_verify && "
        "chmod 0444 /tmp/leanlean-palomar-comparator-registered.json "
        "/tmp/leanlean-palomar-comparator.json "
        "/tmp/leanlean-palomar-comparator-defaults.json "
        "/tmp/leanlean-palomar-challenge.lean "
        + shlex.quote("/testbed/" + challenge_source),
        timeout=60,
    )
    if permissions.get("returncode") or permissions.get("timed_out"):
        raise RuntimeError(
            "failed to install pinned lean_verify: "
            + str(permissions.get("output") or "")[-4000:]
        )


def _build_replay_point_with_lean_verify(
    env: DockerEnvironment,
    point: dict[str, Any],
    capture_dir: Path,
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Build one immutable checkpoint and optionally run lean_verify."""

    archive = _archive_for_point(capture_dir, point)
    expected = str(point["source_archive_sha256"])
    timeout = int(container["build_timeout_seconds"])
    output_chars = int(container["build_output_chars"])
    build_command = (
        f"LEAN_NUM_THREADS={container['build_jobs']} {repository['build_command']}"
    )
    verify_command = "LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify"
    build_policy = str(
        container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY
    )
    started = time.monotonic()
    verify_stub: dict[str, Any] = {
        "schema": LEAN_VERIFY_RESULT_SCHEMA,
        "checked": False,
        "passed": False,
        "command": verify_command,
    }
    try:
        env.restore_source_archive(archive, ["."], timeout=min(timeout, 900))
        correct, actual = _verify_restored_source(
            env, archive, expected, timeout=min(timeout, 900)
        )
        if not correct:
            raise RuntimeError(
                "restored source digest mismatch: "
                f"expected {expected}, found {actual}"
            )
        if build_policy in {
            INCREMENTAL_REPLAY_BUILD_POLICY,
            INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY,
            INCREMENTAL_REPLAY_BUILD_ONLY_POLICY,
        }:
            # Keep .lake/build from the previous checkpoint of this repository;
            # Lake rebuilds whatever changed.  Only artifacts whose source no
            # longer exists are dropped so a deleted module cannot resolve.
            _prune_orphan_build_artifacts(env, timeout=timeout)
        else:
            _clear_verification_cache(env, timeout=timeout)
        if build_policy != INCREMENTAL_REPLAY_BUILD_ONLY_POLICY:
            _install_replay_lean_verify(
                env, repository, container, repo_root=repo_root
            )
        build_started = time.monotonic()
        try:
            build_execution = env.execute(
                f"cd /testbed && {build_command}", timeout=timeout
            )
            build_timed_out = bool(build_execution.get("timed_out"))
        except subprocess.TimeoutExpired as error:
            partial = error.output or getattr(error, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            build_execution = {"returncode": 124, "output": partial}
            build_timed_out = True
        if build_timed_out:
            env.quiesce_after_agent_exit(timeout=15)
        build_returncode = int(build_execution.get("returncode", 1))
        build_output = str(build_execution.get("output") or "")
        build = {
            "passed": build_returncode == 0 and not build_timed_out,
            "verification_build_policy": build_policy,
            "returncode": build_returncode,
            "command": build_command,
            "duration_seconds": round(time.monotonic() - build_started, 3),
            "output": build_output[-output_chars:],
            "timed_out": build_timed_out,
            "source_tree_unchanged_before_verifier_install": correct,
            "source_archive_sha256_before_verifier_install": actual,
            "environment": "pinned_networkless_replay_container",
        }
        if not build["passed"]:
            return {
                "build": build,
                "lean_verify": {
                    **verify_stub,
                    "skipped": "checkpoint_build_failed",
                },
            }
        if build_policy == INCREMENTAL_REPLAY_BUILD_ONLY_POLICY:
            return {
                "build": build,
                "lean_verify": {
                    **verify_stub,
                    "passed": None,
                    "skipped": REPLAY_LEAN_VERIFY_SKIPPED,
                },
            }
        if (
            build_policy == INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY
            and point.get("endpoint_role") != "submitted_patch"
        ):
            return {
                "build": build,
                "lean_verify": {
                    **verify_stub,
                    "passed": None,
                    "skipped": CHECKPOINT_LEAN_VERIFY_SKIPPED,
                },
            }
        verify_started = time.monotonic()
        try:
            execution = env.execute(
                f"cd /testbed && {verify_command}", timeout=timeout
            )
            timed_out = bool(execution.get("timed_out"))
        except subprocess.TimeoutExpired as error:
            partial = error.output or getattr(error, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            execution = {"returncode": 124, "output": partial}
            timed_out = True
        if timed_out:
            env.quiesce_after_agent_exit(timeout=15)
        returncode = int(execution.get("returncode", 1))
        output = str(execution.get("output") or "")
        # lean_verify execs the Comparator; exit 0 without its closing line is
        # a truncated run, not an acceptance.
        completed = COMPARATOR_SUCCESS_MARKER in output
        interrupted = returncode in replay_recovery.INTERRUPTED_RETURN_CODES or (
            returncode == 0 and not timed_out and not completed
        )
        lean_verify = {
            **verify_stub,
            "checked": not timed_out and not interrupted,
            "passed": returncode == 0 and not timed_out and completed,
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - verify_started, 3),
            "output": output[-output_chars:],
            "timed_out": timed_out,
            "infrastructure_failure": interrupted,
            "executable": "/usr/local/bin/lean_verify",
        }
        return {"build": build, "lean_verify": lean_verify}
    except DependencyLockViolation as error:
        violation = {
            "package": error.package_name,
            "expected": error.expected,
            "actual": error.actual,
        }
        return {
            "build": {
                "passed": False,
                "verification_build_policy": build_policy,
                "returncode": 1,
                "output": str(error),
                "candidate_policy_violation": "dependency_lock",
                "dependency_lock_violation": violation,
                "environment": "pinned_networkless_replay_container",
            },
            "lean_verify": {
                **verify_stub,
                "checked": True,
                "candidate_policy_violation": "dependency_lock",
                "dependency_lock_violation": violation,
            },
        }
    except Exception as error:
        return {
            "build": {
                "passed": False,
                "returncode": 1,
                "command": build_command,
                "duration_seconds": round(time.monotonic() - started, 3),
                "output": str(error)[-output_chars:],
                "setup_failed": True,
                "environment": "pinned_networkless_replay_container",
            },
            "lean_verify": verify_stub,
        }


def _build_palomar_point(
    point: dict[str, Any],
    capture_dir: Path,
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    *,
    repo_root: Path,
    environment: DockerEnvironment | None = None,
) -> dict[str, Any]:
    contract = _mapping(
        repository.get("palomar_comparator"), "repository.palomar_comparator"
    )
    validation_stub = _palomar_validation_stub(repository, contract)
    archive = _archive_for_point(capture_dir, point)
    expected = str(point["source_archive_sha256"])
    timeout = int(container["build_timeout_seconds"])
    started = time.monotonic()
    env = environment
    owns_environment = env is None
    environment_kind = (
        "fresh_pinned_networkless_palomar_comparator_container"
        if owns_environment
        else "warm_repository_pinned_networkless_palomar_comparator_container"
    )
    command = (
        f"LEAN_NUM_THREADS={container['build_jobs']} "
        # Matches scripts/lean_verify: nanoda's kernel re-check recurses deeply.
        f"RUST_MIN_STACK={COMPARATOR_RUST_MIN_STACK} "
        "PALOMAR_LANDRUN_BIN=/usr/local/bin/palomar-landrun "
        "COMPARATOR_LANDRUN=/usr/local/bin/palomar-landrun-wrapper "
        "COMPARATOR_LEAN4EXPORT=/usr/local/bin/palomar-lean4export "
        "COMPARATOR_NANODA=/usr/local/bin/palomar-nanoda "
        "lake env /usr/local/bin/palomar-comparator "
        "/tmp/leanlean-palomar-comparator.json"
    )
    try:
        if env is None:
            env = _build_environment(repository, container, 1)
        env.restore_source_archive(archive, ["."], timeout=min(timeout, 900))
        correct, actual = _verify_restored_source(
            env, archive, expected, timeout=min(timeout, 900)
        )
        if not correct:
            raise RuntimeError(
                "restored source digest mismatch: "
                f"expected {expected}, found {actual}"
            )
        warm_cache = _prepare_verification_build(env, container, timeout=timeout)
        if not same_identifier(contract.get("schema"), PALOMAR_VALIDATION_SCHEMA):
            raise RuntimeError("unsupported Palomar Comparator contract schema")
        toolchain = env.read_file("/testbed/lean-toolchain").strip()
        if toolchain != contract.get("lean_toolchain"):
            raise RuntimeError(
                "candidate Lean toolchain differs from registered Palomar toolchain: "
                f"{toolchain!r} != {contract.get('lean_toolchain')!r}"
            )

        challenge, original_config, runtime_config = load_palomar_evidence(
            repo_root=repo_root, contract=contract
        )
        _copy_palomar_tools(env, repo_root=repo_root, contract=contract)
        challenge_pin = _mapping(contract.get("challenge"), "challenge")
        config_pin = _mapping(contract.get("configuration"), "configuration")
        challenge_source = _safe_repository_path(
            challenge_pin.get("source_path"), "registered Challenge path"
        )
        config_source = _safe_repository_path(
            config_pin.get("source_path"), "registered Comparator config path"
        )
        proof_length = _render_task_proof_length_script(
            list(repository.get("exclude_dirs") or []),
            str(repository.get("metric_include_prefix") or ""),
            exclude_files=[challenge_source],
        ).encode("utf-8")
        env.write_file_bytes(
            "/testbed/" + challenge_source,
            challenge,
        )
        env.write_file_bytes(
            "/testbed/" + config_source,
            original_config,
        )
        if config_source != "comparator.json":
            env.write_file_bytes("/testbed/comparator.json", original_config)
        env.write_file_bytes("/testbed/proof_length.py", proof_length)
        env.write_file_bytes(
            "/tmp/leanlean-palomar-comparator-registered.json",
            original_config,
        )
        env.write_file_bytes(
            "/tmp/leanlean-palomar-comparator.json", runtime_config
        )
        env.write_file_bytes(
            "/tmp/leanlean-palomar-comparator-defaults.json",
            (
                json.dumps(
                    dict(
                        (contract.get("configuration") or {}).get(
                            "runtime_defaults_applied"
                        )
                        or {}
                    )
                )
                + "\n"
            ).encode("utf-8"),
        )
        metric_command = (
            "LEANLEAN_BENCHMARK_INTERNAL=1 python3 proof_length.py"
        )
        metric_execution = env.execute(
            f"cd /testbed && {metric_command}",
            timeout=min(timeout, 900),
        )
        metric_output = str(metric_execution.get("output") or "").strip()
        if metric_execution.get("returncode") or not re.fullmatch(
            r"[0-9]+", metric_output
        ):
            raise RuntimeError(
                "trusted proof_length.py failed during authoritative evaluation: "
                + metric_output[-4000:]
            )
        measured_lean_tokens = int(metric_output)
        archive_lean_tokens = point.get("lean_tokens")
        if (
            not isinstance(archive_lean_tokens, int)
            or isinstance(archive_lean_tokens, bool)
            or measured_lean_tokens != archive_lean_tokens
        ):
            raise RuntimeError(
                "trusted proof_length.py disagrees with the immutable-archive "
                "metric: "
                f"{measured_lean_tokens} != {archive_lean_tokens!r}"
            )
        authoritative_metric = {
            "schema": AUTHORITATIVE_METRIC_SCHEMA,
            "checked": True,
            "matched": True,
            "lean_tokens": measured_lean_tokens,
            "immutable_archive_lean_tokens": archive_lean_tokens,
            "command": metric_command,
            "script_source": "evaluator-restored trusted implementation",
        }
        try:
            execution = env.execute(f"cd /testbed && {command}", timeout=timeout)
            timed_out = bool(execution.get("timed_out"))
        except subprocess.TimeoutExpired as error:
            partial = error.output or getattr(error, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            execution = {"returncode": 124, "output": partial}
            timed_out = True
        if timed_out:
            env.quiesce_after_agent_exit(timeout=15)
        returncode = int(execution.get("returncode", 1))
        output = str(execution.get("output") or "")
        # `docker exec` can report 0 for a Comparator that died mid-run (output
        # ending at "Building Challenge" or mid-`lake build`). Only the closing
        # line proves that both modules were built, exported and compared.
        completed = COMPARATOR_SUCCESS_MARKER in output
        accepted = returncode == 0 and not timed_out and completed
        interrupted = (returncode == 0 and not timed_out and not completed) or replay_recovery.infrastructure_failed({"build": {"passed": False, "returncode": returncode, "output": output}})
        checked = not timed_out and not interrupted
        requested = list(validation_stub["requested_names"])
        signature_validation = {
            **validation_stub,
            "checked": checked,
            "preserved": accepted,
            "named_interface_preserved": accepted,
            "requested_definitions_preserved": accepted,
            "semantic_definitions_preserved": accepted,
            "dependency_definitions_preserved": accepted,
            "local_axiom_declarations_preserved": accepted,
            "semantic_integrity_preserved": accepted,
            "exact_fingerprints_preserved": accepted,
            "fingerprints_preserved": accepted,
            "axioms_preserved": accepted,
            "removed": [],
            "changed": [],
            "requested_type_hash_changed": [],
            "requested_type_incompatible": requested if checked and not accepted else [],
            "requested_definition_changed": [],
            "requested_definition_removed": [],
            "semantic_definition_changed": [],
            "semantic_definition_removed": [],
            "dependency_definition_changed": [],
            "dependency_definition_removed": [],
            "comparator_rejected": checked and not accepted,
            "type_compatibility": {
                "engine": "leanprover/comparator declaration-closure comparison",
                "preserved": accepted,
            },
            "axiom_validation": {
                "engine": "leanprover/comparator permitted_axioms",
                "permitted_axioms": list(contract.get("permitted_axioms") or []),
                "preserved": accepted,
            },
            "official_comparator": {
                "returncode": returncode,
                "verdict": (
                    "accepted" if accepted else "infrastructure_failure" if interrupted else "timed_out" if timed_out else "rejected"
                ),
                "builtin_kernel": "required",
                "external_kernels": list(
                    _mapping(contract.get("configuration"), "configuration").get(
                        "external_kernels"
                    )
                    or []
                ),
                "nanoda_kernel": "as_registered",
                "tools": copy.deepcopy(dict(contract.get("tools") or {})),
                "output": output[-int(container["build_output_chars"]):],
            },
        }
        build = {
            "passed": accepted,
            "verification_build_policy": str(
                container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY
            ),
            "returncode": returncode,
            "command": command,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output": output[-int(container["build_output_chars"]):],
            "timed_out": timed_out,
            "infrastructure_failure": interrupted,
            "source_tree_unchanged_before_oracle_injection": correct,
            "source_archive_sha256_before_oracle_injection": actual,
            "environment": environment_kind,
            "comparison_engine": "leanprover/comparator",
            **({"warm_baseline_build_cache": warm_cache} if warm_cache else {}),
        }
        return {
            "build": build,
            "signature_validation": signature_validation,
            "authoritative_metric": authoritative_metric,
        }
    except DependencyLockViolation as error:
        violation = {
            "package": error.package_name,
            "expected": error.expected,
            "actual": error.actual,
        }
        return {
            "build": {
                "passed": False,
                "verification_build_policy": VERIFICATION_BUILD_POLICY,
                "returncode": 1,
                "output": str(error),
                "infrastructure_failure": False,
                "candidate_policy_violation": "dependency_lock",
                "dependency_lock_violation": violation,
                "environment": environment_kind,
                "comparison_engine": "pinned_dependency_policy",
            },
            "signature_validation": {
                **validation_stub,
                "checked": True,
                "candidate_policy_violation": "dependency_lock",
                "dependency_lock_violation": violation,
            },
        }
    except Exception as error:
        return {
            "build": {
                "passed": False,
                "returncode": 1,
                "command": command,
                "duration_seconds": round(time.monotonic() - started, 3),
                "output": str(error)[-int(container["build_output_chars"]):],
                "setup_failed": True,
                "source_tree_unchanged": False,
                "environment": environment_kind,
                "comparison_engine": "leanprover/comparator",
            },
            "signature_validation": validation_stub,
        }
    finally:
        if owns_environment and env is not None:
            env.cleanup()


def _palomar_contract_report(
    *,
    repo_root: Path,
    repository: Mapping[str, Any],
    baseline_point: dict[str, Any],
    capture_dir: Path,
    container: Mapping[str, Any],
    environment: DockerEnvironment | None = None,
) -> dict[str, Any]:
    contract = _mapping(
        repository.get("palomar_comparator"), "repository.palomar_comparator"
    )
    attestation = repository.get("palomar_baseline_attestation")
    if isinstance(attestation, Mapping):
        if (
            attestation.get("verdict") != "accepted"
            or attestation.get("current_contract_sha256") != contract.get("sha256")
        ):
            raise RuntimeError("pinned preprocessing baseline attestation drifted")
        return {
            "schema": PALOMAR_VALIDATION_SCHEMA,
            "sha256": contract["sha256"],
            "engine": "leanprover/comparator",
            "requested_names": sorted(repository.get("protected_declarations") or []),
            "theorem_names": list(contract.get("theorem_names") or []),
            "definition_names": list(contract.get("definition_names") or []),
            "permitted_axioms": list(contract.get("permitted_axioms") or []),
            "challenge": copy.deepcopy(dict(contract.get("challenge") or {})),
            "configuration": copy.deepcopy(dict(contract.get("configuration") or {})),
            "registry_record": copy.deepcopy(dict(contract.get("registry_record") or {})),
            "tools": copy.deepcopy(dict(contract.get("tools") or {})),
            "baseline_verdict": "accepted",
            "baseline_evidence": copy.deepcopy(dict(attestation)),
            "comparison": {"selection_gate": "official Comparator accepts"},
        }
    baseline = _build_palomar_point(
        baseline_point,
        capture_dir,
        repository,
        container,
        repo_root=repo_root,
        environment=environment,
    )
    build = _mapping(baseline.get("build"), "Palomar baseline build")
    validation = _mapping(
        baseline.get("signature_validation"), "Palomar baseline validation"
    )
    if build.get("passed") is not True or validation.get("preserved") is not True:
        raise RuntimeError(
            "registered Palomar baseline failed its own Comparator contract: "
            + str(build.get("output") or validation.get("error") or "unknown error")
        )
    return {
        "schema": PALOMAR_VALIDATION_SCHEMA,
        "sha256": contract["sha256"],
        "engine": "leanprover/comparator",
        "requested_names": sorted(repository.get("protected_declarations") or []),
        "theorem_names": list(contract.get("theorem_names") or []),
        "definition_names": list(contract.get("definition_names") or []),
        "permitted_axioms": list(contract.get("permitted_axioms") or []),
        "challenge": copy.deepcopy(dict(contract.get("challenge") or {})),
        "configuration": copy.deepcopy(dict(contract.get("configuration") or {})),
        "registry_record": copy.deepcopy(dict(contract.get("registry_record") or {})),
        "tools": copy.deepcopy(dict(contract.get("tools") or {})),
        "baseline_verdict": "accepted",
        "comparison": {
            "named_interface": "official Comparator declaration-closure equality",
            "axioms": "official Comparator permitted_axioms closure",
            "kernel": "Lean builtin plus exact registered external-kernel policy",
            "selection_gate": "official Comparator accepts",
        },
    }


def _signature_baseline(
    env: DockerEnvironment, repository: Mapping[str, Any]
) -> ProtectedContract:
    requested = set(repository.get("protected_declarations") or [])
    modules = _modules_under_target(env, str(repository.get("metric_include_prefix") or ""))
    initial = collect_protected_signatures(
        env,
        modules=modules,
        protected_names=requested,
    )
    resolution = _resolve_requested_signature_names(initial, requested)
    unresolved = resolution.get("unresolved") or []
    ambiguous = resolution.get("ambiguous") or {}
    if unresolved or ambiguous:
        raise RuntimeError(
            "protected declarations did not resolve in baseline: "
            f"unresolved={unresolved}, ambiguous={sorted(ambiguous)}"
        )
    return establish_protected_contract(
        env,
        modules=modules,
        requested_names=set(resolution["resolved"]),
    )


def _build_point(
    env: DockerEnvironment,
    point: dict[str, Any],
    capture_dir: Path,
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    contract: ProtectedContract,
) -> dict[str, Any]:
    archive = _archive_for_point(capture_dir, point)
    expected = str(point["source_archive_sha256"])
    timeout = int(container["build_timeout_seconds"])
    started = time.monotonic()
    requested_names = set(contract.requested_names)
    protected_names = set(contract.protected_names)
    semantic_definition_names = {
        name
        for name in contract.semantic_names
        if contract.signatures[name][0] != "theorem"
    }
    requested_definition_names = requested_names & semantic_definition_names
    dependency_definition_names = semantic_definition_names - requested_names
    local_axiom_names = set(contract.local_axiom_names)
    validation_stub: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "checked": False,
        "preserved": False,
        "named_interface_preserved": False,
        "semantic_integrity_preserved": False,
        "contract_sha256": contract.sha256,
        "requested_names": sorted(requested_names),
        "requested_count": len(requested_names),
        "protected_count": len(protected_names),
        "requested_definition_count": len(requested_definition_names),
        "semantic_definition_count": len(semantic_definition_names),
        "dependency_definition_count": len(dependency_definition_names),
        "local_axiom_count": len(contract.local_axiom_names),
    }
    try:
        env.restore_source_archive(archive, ["."], timeout=min(timeout, 900))
        correct, actual = _verify_restored_source(
            env, archive, expected, timeout=min(timeout, 900)
        )
        if not correct:
            raise RuntimeError(
                "restored source digest mismatch: "
                f"expected {expected}, found {actual}"
            )
        warm_cache = _prepare_verification_build(env, container, timeout=timeout)
        command = f"LEAN_NUM_THREADS={container['build_jobs']} {repository['build_command']}"
        try:
            result = env.execute(f"cd /testbed && {command}", timeout=timeout)
            timed_out = bool(result.get("timed_out"))
        except subprocess.TimeoutExpired as error:
            partial = error.output or getattr(error, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            result = {"returncode": 124, "output": partial}
            timed_out = True
        if timed_out:
            env.quiesce_after_agent_exit(timeout=15)
        unchanged, after = _verify_restored_source(
            env, archive, expected, timeout=min(timeout, 900)
        )
        build = {
            "passed": result.get("returncode", 1) == 0 and unchanged,
            "verification_build_policy": str(
                container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY
            ),
            "returncode": result.get("returncode"),
            "command": command,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output": str(result.get("output") or "")[
                -int(container["build_output_chars"]):
            ],
            "timed_out": timed_out,
            "source_tree_unchanged": unchanged,
            "source_archive_sha256_after_build": after,
            "environment": "pinned_networkless_checkpoint_container",
            **({"warm_baseline_build_cache": warm_cache} if warm_cache else {}),
        }
        signature_validation = dict(validation_stub)
        if build["passed"]:
            try:
                modules = _modules_under_target(
                    env, str(repository.get("metric_include_prefix") or "")
                )
                post_signatures = collect_protected_signatures(
                    env,
                    modules=modules,
                    protected_names=protected_names,
                )
                exact_fingerprint_difference = _diff_signatures(
                    contract.signatures,
                    post_signatures,
                    scout_names=protected_names,
                )
                type_compatibility = compare_protected_type_compatibility(
                    env,
                    modules=modules,
                    baseline_signatures=contract.signatures,
                    post_signatures=post_signatures,
                    type_sources=contract.type_sources,
                    protected_names=requested_names,
                )
                requested_definition_difference = _diff_signatures(
                    contract.signatures,
                    post_signatures,
                    scout_names=requested_definition_names,
                )
                semantic_definition_difference = _diff_signatures(
                    contract.signatures,
                    post_signatures,
                    scout_names=semantic_definition_names,
                )
                dependency_definition_difference = _diff_signatures(
                    contract.signatures,
                    post_signatures,
                    scout_names=dependency_definition_names,
                )
                local_axiom_difference = _diff_signatures(
                    contract.signatures,
                    post_signatures,
                    scout_names=local_axiom_names,
                )
                post_axioms = collect_protected_axioms(
                    env,
                    modules=modules,
                    protected_names=requested_names,
                )
                axiom_validation = compare_protected_axioms(
                    contract.axioms,
                    post_axioms,
                    protected_names=requested_names,
                )
                named_interface_preserved = (
                    type_compatibility.get("preserved") is True
                )
                requested_definitions_preserved = (
                    requested_definition_difference.get("preserved") is True
                )
                semantic_definitions_preserved = (
                    semantic_definition_difference.get("preserved") is True
                )
                dependencies_preserved = (
                    dependency_definition_difference.get("preserved") is True
                )
                local_axioms_preserved = (
                    local_axiom_difference.get("preserved") is True
                )
                axioms_preserved = axiom_validation.get("preserved") is True
                semantic_integrity_preserved = all(
                    (
                        named_interface_preserved,
                        semantic_definitions_preserved,
                        local_axioms_preserved,
                        axioms_preserved,
                    )
                )
                signature_validation = {
                    **validation_stub,
                    "checked": True,
                    "preserved": semantic_integrity_preserved,
                    "named_interface_preserved": named_interface_preserved,
                    "requested_definitions_preserved": requested_definitions_preserved,
                    "semantic_definitions_preserved": semantic_definitions_preserved,
                    "dependency_definitions_preserved": dependencies_preserved,
                    "local_axiom_declarations_preserved": local_axioms_preserved,
                    "semantic_integrity_preserved": semantic_integrity_preserved,
                    "exact_fingerprints_preserved": exact_fingerprint_difference.get("preserved") is True,
                    "fingerprints_preserved": exact_fingerprint_difference.get("preserved") is True,
                    "axioms_preserved": axioms_preserved,
                    "removed": exact_fingerprint_difference.get("root_removed") or [],
                    "changed": exact_fingerprint_difference.get("root_changed") or [],
                    "requested_type_hash_changed": sorted(
                        set(type_compatibility.get("definitionally_equal") or [])
                        | set(type_compatibility.get("incompatible") or {})
                    ),
                    "requested_type_incompatible": sorted(type_compatibility.get("incompatible") or {}),
                    "requested_definition_changed": requested_definition_difference.get("root_changed") or [],
                    "requested_definition_removed": requested_definition_difference.get("root_removed") or [],
                    "semantic_definition_changed": semantic_definition_difference.get("root_changed") or [],
                    "semantic_definition_removed": semantic_definition_difference.get("root_removed") or [],
                    "dependency_definition_changed": dependency_definition_difference.get("root_changed") or [],
                    "dependency_definition_removed": dependency_definition_difference.get("root_removed") or [],
                    "type_compatibility": type_compatibility,
                    "axiom_validation": axiom_validation,
                }
            except Exception as error:
                signature_validation["error"] = str(error)[:4000]
        return {"build": build, "signature_validation": signature_validation}
    except Exception as error:
        return {
            "build": {
                "passed": False,
                "returncode": 1,
                "command": str(repository["build_command"]),
                "duration_seconds": round(time.monotonic() - started, 3),
                "output": str(error)[-int(container["build_output_chars"]):],
                "setup_failed": True,
                "source_tree_unchanged": False,
                "environment": "pinned_networkless_checkpoint_container",
            },
            "signature_validation": validation_stub,
        }


def _contract_report(contract: ProtectedContract) -> dict[str, Any]:
    requested = set(contract.requested_names)
    return {
        "schema": VALIDATION_SCHEMA,
        "sha256": contract.sha256,
        "requested_names": sorted(requested),
        "protected_names": sorted(contract.protected_names),
        "semantic_definitions": sorted(set(contract.semantic_names) - requested),
        "local_axioms": sorted(contract.local_axiom_names),
        "baseline_axioms": {
            name: list(contract.axioms[name])
            for name in sorted(contract.requested_names)
        },
        "comparison": {
            "named_interface": "Lean Meta.isDefEq against serialized baseline Expr trees",
            "semantic_definitions": "proof-insensitive exact type and value fingerprints",
            "axioms": "per-target final closure must be a subset of baseline",
            "selection_gate": "build and semantic integrity",
        },
        "axiom_policy": "final_per_target_closure_must_be_subset_of_baseline",
    }


def _cached_validation_current(
    point: Mapping[str, Any],
    repository: Mapping[str, Any],
    *,
    contract_sha256: str | None = None,
    replay: bool = False,
    verification_build_policy: str = VERIFICATION_BUILD_POLICY,
) -> bool:
    build = point.get("build")
    if replay:
        verification = point.get("lean_verify")
        if (
            not isinstance(build, Mapping)
            or not isinstance(verification, Mapping)
            or build.get("timed_out") is True
            or verification.get("timed_out") is True
        ):
            return False
        recorded_policy = build.get("verification_build_policy")
        compatible_prior_policies = {
            INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY: {INCREMENTAL_REPLAY_BUILD_POLICY},
            INCREMENTAL_REPLAY_BUILD_ONLY_POLICY: {
                INCREMENTAL_REPLAY_BUILD_POLICY,
                INCREMENTAL_REPLAY_ENDPOINT_VERIFY_POLICY,
            },
        }
        if recorded_policy != verification_build_policy and recorded_policy not in compatible_prior_policies.get(verification_build_policy, set()):
            return False
        # A checkpoint built under the stronger policy has the same incremental
        # Lake build plus a recorded lean_verify result. Preserve that evidence
        # when switching to endpoint-only verification instead of rebuilding it.
        if verification.get("schema") != LEAN_VERIFY_RESULT_SCHEMA:
            return False
        # Verdicts recorded before the marker check passed on exit code alone.
        if verification.get("passed") is True and COMPARATOR_SUCCESS_MARKER not in str(
            verification.get("output") or ""
        ):
            return False
        return not replay_recovery.infrastructure_failed(
            {"build": build, "lean_verify": verification}
        )
    validation = point.get("signature_validation")
    if (
        not isinstance(build, Mapping)
        or not isinstance(validation, Mapping)
        or build.get("timed_out") is True
        or validation.get("timed_out") is True
    ):
        return False
    if build.get("verification_build_policy") != verification_build_policy:
        return False
    if not _evaluation_infrastructure_ok(build, validation):
        return False
    palomar = repository.get("palomar_comparator")
    expected_schema = (
        PALOMAR_VALIDATION_SCHEMA if isinstance(palomar, Mapping) else VALIDATION_SCHEMA
    )
    if (
        isinstance(palomar, Mapping)
        and build.get("candidate_policy_violation") != "dependency_lock"
    ):
        metric = point.get("authoritative_metric")
        if (
            not isinstance(metric, Mapping)
            or metric.get("schema") != AUTHORITATIVE_METRIC_SCHEMA
            or metric.get("checked") is not True
            or metric.get("matched") is not True
        ):
            return False
    if validation.get("schema") != expected_schema:
        return False
    # Verdicts recorded before the marker check passed on exit code alone.
    if (
        isinstance(palomar, Mapping)
        and build.get("passed") is True
        and COMPARATOR_SUCCESS_MARKER not in str(build.get("output") or "")
    ):
        return False
    if contract_sha256 is None and isinstance(palomar, Mapping):
        contract_sha256 = str(palomar.get("sha256") or "")
    requested = sorted(set(repository.get("protected_declarations") or []))
    if validation.get("requested_names") != requested:
        return False
    if (
        contract_sha256 is not None
        and validation.get("contract_sha256") != contract_sha256
    ):
        return False
    return True


def _lake_build_artifact_current(
    point: Mapping[str, Any],
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
) -> bool:
    if container.get("save_lake_build") is not True:
        return True
    root = container.get("lake_build_output_directory")
    artifact = point.get("lake_build_artifact")
    if not isinstance(root, str) or not isinstance(artifact, Mapping):
        return False
    repository_id = str(repository.get("id") or "")
    if not repository_id or Path(repository_id).name != repository_id:
        return False
    expected = (Path(root) / repository_id / "lake-build.tar.gz").resolve()
    try:
        actual = Path(str(artifact.get("path"))).resolve()
        archive_bytes = int(artifact.get("archive_bytes") or 0)
        tree_sha256 = str(artifact.get("tree_tar_sha256") or "")
        return (
            artifact.get("status") == "complete"
            and actual == expected
            and archive_bytes > 0
            and len(tree_sha256) == 64
            and actual.is_file()
            and actual.stat().st_size == archive_bytes
        )
    except (OSError, TypeError, ValueError):
        return False


def _export_lake_build(
    env: DockerEnvironment,
    point: dict[str, Any],
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
) -> None:
    if container.get("save_lake_build") is not True:
        return
    if not isinstance(point.get("build"), Mapping) or point["build"].get("passed") is not True:
        point["lake_build_artifact"] = {
            "status": "not_exported",
            "reason": "submitted_endpoint_build_did_not_pass",
        }
        return
    root = container.get("lake_build_output_directory")
    if not isinstance(root, str) or not root:
        raise ValueError("save_lake_build requires lake_build_output_directory")
    repository_id = str(repository.get("id") or "")
    if not repository_id or Path(repository_id).name != repository_id:
        raise ValueError(f"unsafe repository id for .lake/build export: {repository_id!r}")
    destination = Path(root).resolve() / repository_id / "lake-build.tar.gz"
    exported = env.export_directory_archive(
        destination,
        "/testbed/.lake/build",
        timeout=max(600, int(container["build_timeout_seconds"])),
    )
    point["lake_build_artifact"] = {
        "schema": "leanlean_lake_build_archive_v1",
        "status": "complete",
        "path": str(destination),
        "archive_sha256": exported["archive_sha256"],
        "tree_tar_sha256": exported["tree_tar_sha256"],
        "tree_tar_bytes": exported["tree_tar_bytes"],
        "archive_bytes": exported["archive_bytes"],
        "build_policy": point["build"].get("verification_build_policy"),
        "container_source": "/testbed/.lake/build",
    }


def _evaluation_infrastructure_ok(
    build: Mapping[str, Any] | None,
    signature_validation: Mapping[str, Any] | None,
) -> bool:
    """Separate evaluator failures from candidate build/signature failures."""

    return not replay_recovery.infrastructure_failed({
        "build": build, "signature_validation": signature_validation,
    })

def _heartbeat_result_path(
    heartbeat_config: Mapping[str, Any], repository_id: str
) -> Path:
    return (
        Path.cwd()
        / str(heartbeat_config["output_directory"])
        / repository_id
        / str(heartbeat_config.get("result_filename", "optimized.json"))
    ).resolve()


def _heartbeat_result_current(
    heartbeat_config: Mapping[str, Any] | None,
    repository_id: str,
    point: Mapping[str, Any] | None,
) -> bool:
    if not heartbeat_config or heartbeat_config.get("enabled") is not True or point is None:
        return True
    path = _heartbeat_result_path(heartbeat_config, repository_id)
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError):
        row = None
    if isinstance(row, Mapping) and (
        row.get("status") in {"complete", "ineligible"}
        and row.get("source_archive_sha256") == point.get("source_archive_sha256")
        and row.get("method") == heartbeat_config.get("method")
    ):
        return True


    if heartbeat_config.get("reuse_policy") != "exact_source_archive_method_consensus_v1":
        return False
    reuse_directories = heartbeat_config.get("reuse_directories") or []
    if not isinstance(reuse_directories, list):
        return False
    try:
        candidates = sorted(
            (Path.cwd() / str(directory) / repository_id / "optimized.json").resolve()
            for directory in reuse_directories
        )
    except OSError:
        return False
    matches: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        if candidate.resolve() == path:
            continue
        try:
            candidate_row = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        if (
            isinstance(candidate_row, dict)
            and candidate_row.get("repository") == repository_id
            and candidate_row.get("status") in {"complete", "ineligible"}
            and candidate_row.get("source_archive_sha256")
            == point.get("source_archive_sha256")
            and candidate_row.get("method") == heartbeat_config.get("method")
        ):
            matches.append((candidate, candidate_row))
    if not matches:
        return False

    def consensus_key(candidate_row: Mapping[str, Any]) -> str:
        return json.dumps(
            {
                key: candidate_row.get(key)
                for key in (
                    "status", "body_heartbeats", "total_heartbeats",
                    "import_heartbeats", "expected_file_count",
                    "measured_file_count", "unbuilt_sources",
                )
            },
            sort_keys=True,
        )

    if len({consensus_key(candidate_row) for _, candidate_row in matches}) != 1:
        return False
    source_path, source_row = matches[-1]
    reused = copy.deepcopy(source_row)
    reused["original_postprocess_id"] = source_row.get("postprocess_id")
    reused["postprocess_id"] = heartbeat_config.get("postprocess_id")
    reused["reuse"] = {
        "policy": heartbeat_config["reuse_policy"],
        "source_artifact": str(source_path),
        "source_artifact_sha256": file_sha256(source_path),
    }
    _atomic_json(path, reused)
    source_log = source_path.with_suffix(".log")
    destination_log = path.with_suffix(".log")
    if source_log.is_file() and not destination_log.exists():
        destination_log.write_bytes(source_log.read_bytes())
    return True

def _measure_postprocessing_heartbeats(
    env: DockerEnvironment,
    repository: Mapping[str, Any],
    point: Mapping[str, Any],
    heartbeat_config: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts import measure_heartbeats as heartbeat

    result_path = _heartbeat_result_path(
        heartbeat_config, str(repository["id"])
    )
    directory = result_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    previous = None
    if result_path.is_file():
        try:
            previous = json.loads(result_path.read_text())
        except (OSError, ValueError):
            pass
    contract = repository.get("palomar_comparator") or {}
    challenge = (contract.get("challenge") or {}).get("source_path")
    measurement_repository = {
        "id": str(repository["id"]),
        "exclude_files": [challenge] if challenge else [],
    }
    measurement_manifest = {
        "evaluation": {
            "repetitions": int(heartbeat_config["repetitions"]),
            "variation_warning_relative_range_pct": float(
                heartbeat_config["variation_warning_relative_range_pct"]
            ),
        },
        "timeout": {
            "file_seconds": int(heartbeat_config["file_timeout_seconds"]),
        },
    }
    if (
        not (point.get("build") or {}).get("passed")
        or not (point.get("signature_validation") or {}).get("preserved")
    ):
        measurement = {
            "status": "ineligible",
            "repository": repository["id"],
            "variant": "optimized",
            "method": heartbeat_config["method"],
            "source_archive_sha256": point.get("source_archive_sha256"),
            "body_heartbeats": None,
            "total_heartbeats": None,
            "import_heartbeats": None,
            "reason": "submitted endpoint did not pass postprocessing correctness gates",
        }
        heartbeat.atomic_json(result_path, measurement)
    else:
        measurement = heartbeat.measure_built_variant(
            env,
            measurement_repository,
            "optimized",
            measurement_manifest,
            directory,
            previous=previous,
            clean_build_seconds=(point.get("build") or {}).get("duration_seconds"),
            validation=point,
            lean_tokens=point.get("lean_tokens"),
            source_identity={
                "source_archive_sha256": point.get("source_archive_sha256"),
                "postprocess_id": heartbeat_config.get("postprocess_id"),
            },
            result_path=result_path,
        )
    return {
        "status": measurement.get("status"),
        "method": measurement.get("method"),
        "body_heartbeats": measurement.get("body_heartbeats"),
        "total_heartbeats": measurement.get("total_heartbeats"),
        "import_heartbeats": measurement.get("import_heartbeats"),
        "measured_file_count": measurement.get("measured_file_count"),
        "clean_build_reused": measurement.get("clean_build_reused"),
        "artifact": str(result_path),
        "error": measurement.get("error"),
    }


class CaptureBuildRequired(RuntimeError):
    """Signal that cache-only capture processing requires materialization."""


def process_capture(
    capture_dir: Path,
    repository: Mapping[str, Any],
    container: Mapping[str, Any],
    patch: str,
    *,
    replay: bool,
    heartbeat_config: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    defer_build: bool = False,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    source_path = capture_dir / "playback.json"
    source = json.loads(source_path.read_text())
    destination = derived_playback_path(capture_dir, replay=replay)
    previous_path = _previous_playback_path(capture_dir, replay=replay)
    previous: Mapping[str, Any] | None = None
    if previous_path.is_file():
        try:
            loaded = json.loads(previous_path.read_text())
            previous = loaded if isinstance(loaded, Mapping) else None
        except (OSError, ValueError):
            pass
    endpoint_playback = _submitted_endpoint(
        source,
        capture_dir,
        repository,
        container,
        patch,
        defer_build=defer_build,
    )
    result = _attach_source_metrics(
        endpoint_playback,
        capture_dir,
        repository,
        endpoints_only=bool(container.get("endpoints_only")),
        selected_edit_indices=(
            container["checkpoint_selection"]["edit_indices"].get(str(repository["id"]))
            if replay and isinstance(container.get("checkpoint_selection"), Mapping)
            else None
        ),
    )
    _preserve_checkpoint_results(result, previous, replay=replay)

    points = result["points"]
    palomar_contract = repository.get("palomar_comparator")
    uses_palomar = isinstance(palomar_contract, Mapping)
    for point in points:
        if replay:
            point.pop("signature_validation", None)
        if not _cached_validation_current(
            point,
            repository,
            replay=replay,
            verification_build_policy=str(
                container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY
            ),
        ) or not _lake_build_artifact_current(point, repository, container):
            point.pop("build", None)
            point.pop("lean_verify" if replay else "signature_validation", None)
            point.pop("authoritative_metric", None)
    if uses_palomar:
        for point in points:
            point["benchmark_fixture_integrity"] = _benchmark_fixture_integrity(
                point,
                capture_dir,
                repository,
                repo_root=resolved_repo_root,
            )
    timeline_points = [
        point
        for point in points
        if int(point.get("edit_index") or 0) > 0
        and point.get("endpoint_role") != "submitted_patch"
    ]
    submitted_point = next(
        (
            point
            for point in reversed(points)
            if point.get("endpoint_role") == "submitted_patch"
        ),
        None,
    )
    timed_out = _playback_timeout(capture_dir)
    if replay:
        build_points = [*timeline_points]
        selection = container.get("checkpoint_selection")
        if isinstance(selection, Mapping):
            # Sparse replay: build only the pre-selected edit checkpoints.
            selected = selection["edit_indices"].get(str(repository["id"]))
            if selected is None:
                raise ValueError(
                    f"{repository['id']}: missing from checkpoint_selection"
                )
            wanted = set(selected)
            build_points = [
                point
                for point in timeline_points
                if int(point.get("edit_index") or 0) in wanted
            ]
        if submitted_point is not None:
            build_points.append(submitted_point)
    else:
        # The final submission is the normal scoring state.  A timed-out run
        # needs only enough older checkpoints to find its newest build- and
        # signature-green recovery state; --replay remains the explicit way
        # to build the complete edit timeline.
        build_points = [submitted_point] if submitted_point is not None else []
        if timed_out and not container.get("endpoints_only"):
            build_points.extend(reversed(timeline_points))

    previous_matches_source = (
        isinstance(previous, Mapping)
        and previous.get("source_playback_sha256")
        == result.get("source_playback_sha256")
    )
    previous_by_digest = {
        str(point.get("source_archive_sha256")): point
        for point in (
            previous.get("points") if previous_matches_source else []
        ) or []
        if isinstance(point, Mapping)
    }
    results_by_digest = dict(previous_by_digest)
    needs_build = any(
        not isinstance(
            results_by_digest.get(str(point.get("source_archive_sha256"))),
            Mapping,
        )
        or not _cached_validation_current(
            results_by_digest[str(point.get("source_archive_sha256"))],
            repository,
            replay=replay,
            verification_build_policy=str(
                container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY
            ),
        )
        or not _lake_build_artifact_current(
            results_by_digest[str(point.get("source_archive_sha256"))],
            repository,
            container,
        )
        for point in build_points
    )
    if (
        submitted_point is not None
        and not _heartbeat_result_current(
            heartbeat_config, str(repository["id"]), submitted_point
        )
    ):
        needs_build = True
    if needs_build and defer_build:
        raise CaptureBuildRequired(str(repository["id"]))
    env = None
    contract: ProtectedContract | None = None
    if needs_build:
        if replay:
            if not uses_palomar:
                raise RuntimeError(
                    f"{repository['id']}: --replay requires a registered lean_verify command"
                )
            env = _build_environment(repository, container, len(build_points))
        elif uses_palomar:
            env = _build_environment(repository, container, len(build_points) + 1)
            try:
                result["signature_contract"] = _palomar_contract_report(
                    repo_root=resolved_repo_root,
                    repository=repository,
                    baseline_point=points[0],
                    capture_dir=capture_dir,
                    container=container,
                    environment=env,
                )
            except Exception:
                env.cleanup()
                raise
        else:
            env = _build_environment(repository, container, len(build_points))
            try:
                baseline_archive = _archive_for_point(capture_dir, points[0])
                env.restore_source_archive(baseline_archive, ["."], timeout=900)
                correct, actual = _verify_restored_source(
                    env,
                    baseline_archive,
                    str(points[0]["source_archive_sha256"]),
                    timeout=900,
                )
                if not correct:
                    raise RuntimeError(f"baseline restore digest mismatch: {actual}")
                contract = _signature_baseline(env, repository)
                result["signature_contract"] = _contract_report(contract)
            except Exception:
                env.cleanup()
                raise
    elif not replay and previous_matches_source and isinstance(
        previous.get("signature_contract"), Mapping
    ):
        previous_contract = previous["signature_contract"]
        requested = sorted(set(repository.get("protected_declarations") or []))
        expected_schema = (
            PALOMAR_VALIDATION_SCHEMA if uses_palomar else VALIDATION_SCHEMA
        )
        if (
            previous_contract.get("schema") == expected_schema
            and previous_contract.get("requested_names") == requested
            and (
                not uses_palomar
                or previous_contract.get("sha256") == palomar_contract.get("sha256")
            )
        ):
            result["signature_contract"] = copy.deepcopy(previous_contract)

    completed = 0
    try:
        for point in build_points:
            digest = str(point["source_archive_sha256"])
            cached = results_by_digest.get(digest)
            cached_current = isinstance(cached, Mapping) and _cached_validation_current(
                cached,
                repository,
                replay=replay,
                verification_build_policy=str(
                    container.get("verification_build_policy") or VERIFICATION_BUILD_POLICY
                ),
                contract_sha256=(
                    str(palomar_contract["sha256"])
                    if uses_palomar
                    else contract.sha256
                    if contract is not None
                    else None
                ),
            )
            cached_current = cached_current and _lake_build_artifact_current(
                cached, repository, container
            )
            cached_current = cached_current and (
                point is not submitted_point
                or _heartbeat_result_current(
                    heartbeat_config, str(repository["id"]), point
                )
            )
            if cached_current:
                point["build"] = copy.deepcopy(cached["build"])
                if replay:
                    point["lean_verify"] = copy.deepcopy(cached["lean_verify"])
                else:
                    point["signature_validation"] = copy.deepcopy(
                        cached["signature_validation"]
                    )
                if isinstance(cached.get("authoritative_metric"), Mapping):
                    point["authoritative_metric"] = copy.deepcopy(
                        cached["authoritative_metric"]
                    )
            else:
                if replay:
                    def build_attempt(current_environment):
                        return _build_replay_point_with_lean_verify(
                            current_environment,
                            point,
                            capture_dir,
                            repository,
                            container,
                            repo_root=resolved_repo_root,
                        )

                    def recreate_environment():
                        return _build_environment(
                            repository, container, len(build_points)
                        )

                    def record_infrastructure_failure(outcome, attempt):
                        point.update(outcome)
                        point.setdefault("infrastructure_attempts", []).append({
                            "at": _utc_now(),
                            "attempt": attempt,
                            "build": copy.deepcopy(outcome["build"]),
                            "lean_verify": copy.deepcopy(outcome.get("lean_verify")),
                        })
                        result["evaluation_infrastructure_ok"] = False
                        _atomic_json(destination, result)

                    try:
                        outcome, env = replay_recovery.retry_checkpoint(
                            env,
                            build_attempt,
                            recreate_environment,
                            record_infrastructure_failure,
                            retries=int(
                                container.get(
                                    "checkpoint_infrastructure_retries",
                                    replay_recovery.DEFAULT_CHECKPOINT_RETRIES,
                                )
                            ),
                        )
                    except replay_recovery.ReplayInfrastructureError as error:
                        raise replay_recovery.ReplayInfrastructureError(
                            f"{repository['id']} checkpoint {point.get('edit_index')}: {error}"
                        ) from error
                    point.update(outcome)
                elif uses_palomar:
                    def build_attempt(current_environment):
                        return _build_palomar_point(
                            point, capture_dir, repository, container,
                            repo_root=resolved_repo_root, environment=current_environment,
                        )

                    def recreate_environment():
                        replacement = _build_environment(repository, container, len(build_points) + 1)
                        try:
                            result["signature_contract"] = _palomar_contract_report(
                                repo_root=resolved_repo_root, repository=repository,
                                baseline_point=points[0], capture_dir=capture_dir,
                                container=container, environment=replacement,
                            )
                        except BaseException:
                            replacement.cleanup()
                            raise
                        return replacement

                    def record_infrastructure_failure(outcome, attempt):
                        point.update(outcome)
                        point.setdefault("infrastructure_attempts", []).append({
                            "at": _utc_now(), "attempt": attempt,
                            "build": copy.deepcopy(outcome["build"]),
                            "signature_validation": copy.deepcopy(outcome.get("signature_validation")),
                        })
                        result["evaluation_infrastructure_ok"] = False
                        _atomic_json(destination, result)

                    try:
                        outcome, env = replay_recovery.retry_checkpoint(
                            env, build_attempt, recreate_environment, record_infrastructure_failure,
                            retries=int(container.get("checkpoint_infrastructure_retries", replay_recovery.DEFAULT_CHECKPOINT_RETRIES)),
                        )
                    except replay_recovery.ReplayInfrastructureError as error:
                        raise replay_recovery.ReplayInfrastructureError(
                            f"{repository['id']} checkpoint {point.get('edit_index')}: {error}"
                        ) from error
                    point.update(outcome)
                else:
                    assert env is not None
                    assert contract is not None
                    point.update(
                        _build_point(
                            env,
                            point,
                            capture_dir,
                            repository,
                            container,
                            contract,
                        )
                    )
                if point is submitted_point:
                    assert env is not None
                    _export_lake_build(env, point, repository, container)
                if (
                    point is submitted_point
                    and heartbeat_config
                    and heartbeat_config.get("enabled") is True
                    and not _heartbeat_result_current(
                        heartbeat_config, str(repository["id"]), point
                    )
                ):
                    assert env is not None
                    point["heartbeat_measurement"] = _measure_postprocessing_heartbeats(
                        env, repository, point, heartbeat_config
                    )
                results_by_digest[digest] = point
            completed += 1
            result["checkpoint_build_attempt_count"] = sum(
                isinstance(item.get("build"), Mapping) for item in timeline_points
            )
            _atomic_json(destination, result)
            if progress is not None:
                progress(str(repository["id"]), completed, len(build_points))
            if (
                timed_out
                and not replay
                and isinstance(point.get("build"), Mapping)
                and point["build"].get("passed") is True
                and isinstance(point.get("signature_validation"), Mapping)
                and point["signature_validation"].get("preserved") is True
            ):
                break
    finally:
        if env is not None:
            env.cleanup()

    final_build = submitted_point.get("build") if submitted_point is not None else None
    final_verification = (
        submitted_point.get("lean_verify" if replay else "signature_validation")
        if submitted_point is not None
        else None
    )
    final_fixture_integrity = (
        submitted_point.get("benchmark_fixture_integrity")
        if submitted_point is not None
        else None
    )
    final_status = {
            "build_passed": isinstance(final_build, Mapping)
            and final_build.get("passed") is True,
            "evaluation_infrastructure_ok": not replay_recovery.infrastructure_failed({
                "build": final_build,
                ("lean_verify" if replay else "signature_validation"): final_verification,
            }),
            "benchmark_fixtures_preserved": (
                isinstance(final_fixture_integrity, Mapping)
                and final_fixture_integrity.get("preserved") is True
            )
            if uses_palomar
            else None,
            "benchmark_fixture_integrity": (
                copy.deepcopy(final_fixture_integrity)
                if isinstance(final_fixture_integrity, Mapping)
                else None
            ),
    }
    if replay:
        final_status["lean_verify_passed"] = (
            isinstance(final_verification, Mapping)
            and final_verification.get("passed") is True
        )
    else:
        final_status.update({
            "signatures_preserved": isinstance(final_verification, Mapping)
            and final_verification.get("preserved") is True,
            "named_signatures_preserved": isinstance(final_verification, Mapping)
            and final_verification.get("named_interface_preserved") is True,
            "semantic_integrity_preserved": isinstance(final_verification, Mapping)
            and final_verification.get("semantic_integrity_preserved") is True,
        })
    result.update(final_status)

    built = [
        point
        for point in timeline_points
        if isinstance(point.get("build"), Mapping)
    ]
    passed = [point for point in built if point["build"].get("passed") is True]
    verification_green = [
        point
        for point in passed
        if isinstance(
            point.get("lean_verify" if replay else "signature_validation"), Mapping
        )
        and point["lean_verify" if replay else "signature_validation"].get(
            "passed" if replay else "preserved"
        ) is True
    ]
    named_signature_green = [
        point
        for point in passed
        if isinstance(point.get("signature_validation"), Mapping)
        and point["signature_validation"].get("named_interface_preserved") is True
    ]
    replay_counts = (
        {"checkpoint_lean_verify_pass_count": len(verification_green)}
        if replay
        else {
            "checkpoint_signature_green_count": len(verification_green),
            "checkpoint_named_signature_green_count": len(named_signature_green),
        }
    )
    result.update(
        {
            "build_basis": (
                "postprocess.sh --replay; submitted final plus every monitored "
                "checkpoint; pinned image; network disabled"
                if replay
                else "postprocess.sh; submitted final; pinned image; network disabled"
            ),
            "monitor_final_matches_submission": source.get(
                "final_snapshot_matches_submission"
            ) is True,
            "edit_checkpoint_count": len(timeline_points),
            "every_edit_checkpoint_built": bool(timeline_points)
            and len(built) == len(timeline_points),
            "checkpoint_build_attempt_count": len(built),
            "checkpoint_build_pass_count": len(passed),
            "checkpoint_build_fail_count": len(built) - len(passed),
            **replay_counts,
            "all_checkpoint_builds_passed": bool(timeline_points)
            and len(passed) == len(timeline_points),
            "final_submission_build_attempted": isinstance(final_build, Mapping),
        }
    )
    result["provider_timed_out"] = timed_out
    result["selected_state"] = _selected_state(
        result,
        timed_out=timed_out and not container.get("endpoints_only", False),
        replay=replay,
    )
    _atomic_json(destination, result)
    return result


def _patch_file_count(patch: str) -> int:
    return len(re.findall(r"^diff --git ", patch, re.MULTILINE))


def _patch_sorry_count(patch: str) -> int:
    sorry = re.compile(r"(?<![A-Za-z_])(?:sorry|admit)(?![A-Za-z_])")
    net = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            net += len(sorry.findall(line[1:].split("--", 1)[0]))
        elif line.startswith("-"):
            net -= len(sorry.findall(line[1:].split("--", 1)[0]))
    return max(0, net)


def _write_combined_action_summary(
    prepared: PreparedPostprocessing,
) -> None:
    combination = _mapping(
        prepared.manifest.get("run_combination"), "manifest.run_combination"
    )
    selection = _mapping(combination.get("selection"), "run_combination.selection")
    summaries: dict[Path, Mapping[str, Any]] = {}
    sequences: dict[str, list[str]] = {}
    tool_instances: dict[str, Mapping[str, Any]] = {}
    lean_verify_samples: list[dict[str, Any]] = []
    selected_runs: dict[str, str] = {}
    for instance_id, raw in selection.items():
        selected = _mapping(raw, f"run_combination.selection.{instance_id}")
        output_dir = Path(
            _text(selected, "output_directory", "run combination selection")
        ).resolve()
        if output_dir not in summaries:
            summaries[output_dir] = build_run_action_summary(output_dir)
        summary = summaries[output_dir]
        instance = _mapping(summary.get("instances", {}), "action instances").get(
            instance_id
        )
        if isinstance(instance, Mapping):
            sequence = instance.get("sequence")
            if isinstance(sequence, list) and all(
                isinstance(item, str) for item in sequence
            ):
                sequences[str(instance_id)] = list(sequence)
        tools = _mapping(
            summary.get("benchmark_tools", {}), "benchmark tools"
        ).get("instances", {})
        if isinstance(tools, Mapping) and isinstance(
            tools.get(instance_id), Mapping
        ):
            tool_instances[str(instance_id)] = copy.deepcopy(
                tools[instance_id]
            )
        raw_samples = _mapping(
            summary.get("benchmark_tools", {}), "benchmark tools"
        ).get("duration_samples", {})
        if isinstance(raw_samples, Mapping):
            for sample in raw_samples.get("lean_verify") or []:
                if (
                    isinstance(sample, Mapping)
                    and sample.get("instance_id") == instance_id
                ):
                    lean_verify_samples.append(copy.deepcopy(dict(sample)))
        selected_runs[str(instance_id)] = _text(
            selected, "run_id", "run combination selection"
        )

    tool_totals: dict[str, dict[str, int]] = {}
    for tools in tool_instances.values():
        for tool, raw_counts in tools.items():
            if not isinstance(raw_counts, Mapping):
                continue
            totals = tool_totals.setdefault(str(tool), {})
            for field, value in raw_counts.items():
                if isinstance(value, int):
                    totals[str(field)] = totals.get(str(field), 0) + value
    summary = {
        "schema_version": ACTION_SUMMARY_SCHEMA_VERSION,
        "kind": "leanlean_run_action_summary",
        "generated_at": _utc_now(),
        "source": {
            "trace_count": len(sequences),
            "run_combination": copy.deepcopy(combination),
            "combined": combination.get("combined") is True,
            "combination_digest": combination.get("digest"),
            "selected_runs": selected_runs,
        },
        "benchmark_tools": {
            "totals": tool_totals,
            "instances": tool_instances,
            "duration_samples": {"lean_verify": lean_verify_samples},
        },
        **summarize_action_sequences(sequences),
    }
    action_filename = "replay_action_summary.json" if prepared.replay else "action_summary.json"
    _atomic_json(prepared.output_dir / action_filename, summary)


def _write_postprocessed_report_summary(
    prepared: PreparedPostprocessing,
    captures: Mapping[str, Mapping[str, Any]],
) -> None:
    """Publish the compact GUI index from authoritative postprocess results."""

    summary_path = prepared.output_dir / ("replay_report_summary.json" if prepared.replay else "report_summary.json")
    try:
        actions = json.loads(
            (prepared.output_dir / ("replay_action_summary.json" if prepared.replay else "action_summary.json")).read_text()
        )
    except (OSError, ValueError):
        actions = None
    cohort_bounds = (
        ("Compact", 10_000),
        ("Standard", 50_000),
        ("Large", 250_000),
        ("Massive", None),
    )

    def difficulty_cohort(lean_tokens: int) -> str:
        for label, upper_bound in cohort_bounds:
            if upper_bound is None or lean_tokens <= upper_bound:
                return label
        raise AssertionError("unreachable difficulty cohort")

    instances: dict[str, Any] = {}
    for instance_id, capture in sorted(captures.items()):
        points = capture.get("points") or []
        final = points[-1] if points and isinstance(points[-1], Mapping) else {}
        build = final.get("build") if isinstance(final, Mapping) else None
        signatures = (
            final.get("signature_validation")
            if isinstance(final, Mapping)
            else None
        )
        fixture_integrity = (
            final.get("benchmark_fixture_integrity")
            if isinstance(final, Mapping)
            else None
        )
        patch = str(prepared.predictions[instance_id]["model_patch"])
        baseline_words = int(capture.get("baseline_words") or 0)
        post_words = int(capture.get("post_words") or 0)
        baseline_tokens = int(capture.get("baseline_lean_tokens") or 0)
        post_tokens = int(capture.get("post_lean_tokens") or 0)
        edit_build_count = int(
            capture.get("checkpoint_build_attempt_count") or 0
        )
        final_build_count = int(
            capture.get("final_submission_build_attempted") is True
        )
        build_count = edit_build_count + final_build_count
        build_successes = int(
            capture.get("checkpoint_build_pass_count") or 0
        ) + int(capture.get("build_passed") is True)
        compact = {
            "instance_id": instance_id,
            "difficulty_cohort": difficulty_cohort(baseline_tokens),
            "round": 0,
            "source_run_id": capture.get("_source_run_id"),
            "source_output_directory": capture.get("_source_output_directory"),
            "report_path": capture.get("_derived_playback_path"),
            "resolved": capture.get("selected_state", {}).get("eligible") is True,
            "empty_patch": not patch.strip(),
            "apply_patch_ok": capture.get("submitted_endpoint_captured") is True,
            "model_submission": {
                "endpoint_captured": capture.get("submitted_endpoint_captured") is True,
                "patch_non_empty": bool(patch.strip()),
                "patch_files": _patch_file_count(patch),
            },
            "build_passed": capture.get("build_passed") is True,
            "build_ran": isinstance(build, Mapping),
            "signatures_preserved": capture.get("signatures_preserved") is True,
            "named_signatures_preserved": capture.get("named_signatures_preserved") is True,
            "semantic_integrity_preserved": capture.get("semantic_integrity_preserved") is True,
            "evaluation_infrastructure_ok": capture.get("evaluation_infrastructure_ok") is True,
            "benchmark_fixtures_preserved": (
                fixture_integrity.get("preserved") is True
                if isinstance(fixture_integrity, Mapping)
                else None
            ),
            "benchmark_fixture_integrity": (
                copy.deepcopy(fixture_integrity)
                if isinstance(fixture_integrity, Mapping)
                else None
            ),
            "signature_counts": {
                "root": (
                    signatures.get("protected_count")
                    if isinstance(signatures, Mapping)
                    else None
                ),
                "root_removed": len(signatures.get("removed") or [])
                if isinstance(signatures, Mapping)
                else 0,
                "root_changed": len(signatures.get("changed") or [])
                if isinstance(signatures, Mapping)
                else 0,
                "total_removed": len(signatures.get("removed") or [])
                if isinstance(signatures, Mapping)
                else 0,
                "total_changed": len(signatures.get("changed") or [])
                if isinstance(signatures, Mapping)
                else 0,
            },
            "n_sorry": _patch_sorry_count(patch),
            "patch_files": _patch_file_count(patch),
            "round_cost": float(capture.get("final_cost_usd") or 0.0),
            "cost_total": float(capture.get("final_cost_usd") or 0.0),
            "lake_builds": {
                "attempts": build_count,
                "successes": build_successes,
                "failures": build_count - build_successes,
            },
            "lake_builds_total": {
                "attempts": build_count,
                "successes": build_successes,
                "failures": build_count - build_successes,
            },
            "source_metrics_complete": baseline_words > 0 and baseline_tokens > 0,
            "metrics": {
                "lean_file_count": final.get("lean_file_count"),
                "baseline_words": baseline_words,
                "post_words": post_words,
                "compression_ratio": (
                    post_words / baseline_words if baseline_words else None
                ),
                "words_saved": baseline_words - post_words,
                "baseline_lean_tokens": baseline_tokens,
                "post_lean_tokens": post_tokens,
                "lean_token_ratio": (
                    post_tokens / baseline_tokens if baseline_tokens else None
                ),
                "lean_tokens_saved": baseline_tokens - post_tokens,
                "build_time_seconds": (
                    build.get("duration_seconds")
                    if isinstance(build, Mapping)
                    else None
                ),
                "files_modified": _patch_file_count(patch),
            },
        }
        if prepared.replay:
            compact.pop("signatures_preserved", None)
            compact.pop("named_signatures_preserved", None)
            compact.pop("semantic_integrity_preserved", None)
            compact.pop("signature_counts", None)
            compact["lean_verify_passed"] = (
                capture.get("lean_verify_passed") is True
            )
            compact["lean_verify_ran"] = isinstance(
                final.get("lean_verify"), Mapping
            )
        instances[instance_id] = {
            "latest_round": 0,
            "latest": compact,
            "rounds": {"0": copy.deepcopy(compact)},
        }

    latest = [entry["latest"] for entry in instances.values()]
    metrics = [entry["metrics"] for entry in latest]
    selected_states = {
        instance_id: captures[instance_id].get("selected_state", {})
        for instance_id in instances
    }
    difficulty_cohorts: dict[str, Any] = {}
    for cohort, _upper_bound in cohort_bounds:
        cohort_instances = [
            entry for entry in latest if entry["difficulty_cohort"] == cohort
        ]
        cohort_states = [
            selected_states[entry["instance_id"]] for entry in cohort_instances
        ]
        cohort_metrics = [entry["metrics"] for entry in cohort_instances]
        count = len(cohort_instances)
        difficulty_cohorts[cohort] = {
            "repositories": count,
            "valid_submissions": sum(
                state.get("eligible") is True for state in cohort_states
            ),
            "final_builds_passed": sum(
                entry["build_passed"] for entry in cohort_instances
            ),
            **(
                {
                    "lean_verify_passed": sum(
                        entry["lean_verify_passed"]
                        for entry in cohort_instances
                    )
                }
                if prepared.replay
                else {
                    "signatures_preserved": sum(
                        entry["signatures_preserved"]
                        for entry in cohort_instances
                    )
                }
            ),
            "macro_word_compression_pct": round(
                sum(float(state.get("word_compression_pct") or 0.0) for state in cohort_states)
                / count,
                6,
            ) if count else None,
            "macro_lean_token_compression_pct": round(
                sum(
                    float(state.get("lean_token_compression_pct") or 0.0)
                    for state in cohort_states
                ) / count,
                6,
            ) if count else None,
            "submitted_words_saved": sum(
                int(row["words_saved"]) for row in cohort_metrics
            ),
            "submitted_lean_tokens_saved": sum(
                int(row["lean_tokens_saved"]) for row in cohort_metrics
            ),
        }
    denominator = len(instances)
    headline = {
        "metric": "macro_average_selected_state_word_compression_pct",
        "value": round(
            sum(
                float(state.get("word_compression_pct") or 0.0)
                for state in selected_states.values()
            ) / denominator,
            6,
        ) if denominator else None,
        "lean_token_value": round(
            sum(
                float(state.get("lean_token_compression_pct") or 0.0)
                for state in selected_states.values()
            ) / denominator,
            6,
        ) if denominator else None,
        "denominator": denominator,
    }
    checkpoint_builds = {
        "edit_replay_requested": prepared.replay,
        "final_submissions": denominator,
        "final_attempted": sum(
            capture.get("final_submission_build_attempted") is True
            for capture in captures.values()
        ),
        "final_passed": sum(
            capture.get("build_passed") is True for capture in captures.values()
        ),
        **(
            {
                "final_lean_verify_passed": sum(
                    capture.get("lean_verify_passed") is True
                    for capture in captures.values()
                )
            }
            if prepared.replay
            else {
                "final_signature_green": sum(
                    capture.get("signatures_preserved") is True
                    for capture in captures.values()
                ),
                "final_named_signatures_preserved": sum(
                    capture.get("named_signatures_preserved") is True
                    for capture in captures.values()
                ),
            }
        ),
        "edit_checkpoints": sum(
            int(capture.get("edit_checkpoint_count") or 0)
            for capture in captures.values()
        ),
        "attempted": sum(
            int(capture.get("checkpoint_build_attempt_count") or 0)
            for capture in captures.values()
        ),
        "passed": sum(
            int(capture.get("checkpoint_build_pass_count") or 0)
            for capture in captures.values()
        ),
        "failed": sum(
            int(capture.get("checkpoint_build_fail_count") or 0)
            for capture in captures.values()
        ),
    }
    methodology = {
        "result_source": REPLAY_PLAYBACK if prepared.replay else DERIVED_PLAYBACK,
        "submitted_endpoint": (
            "exact submitted patch reconstructed on the immutable baseline"
        ),
        "final_validation": (
            "dataset-canonical build followed by the pinned lean_verify command"
            if prepared.replay
            else "dataset-canonical build plus protected declaration/signature validation"
        ),
        "headline_scoring": (
            (
                "equal-weight macro average; failed build or lean_verify scores zero"
                if prepared.replay
                else "equal-weight macro average; failed build or signature validation scores zero"
            )
        ),
        "source_metrics": (
            "non-comment, non-import Lean words and Lean tokens measured from source archives"
        ),
        "difficulty_cohorts": {
            "basis": "post-preprocessing baseline Lean-token count",
            "Compact": "<= 10,000",
            "Standard": "10,001-50,000",
            "Large": "50,001-250,000",
            "Massive": "> 250,000",
        },
        "agent_tool_usage": "action_summary.json benchmark_tools totals",
        "evaluation_policy": copy.deepcopy(
            prepared.manifest.get("evaluation", {})
        ),
    }
    totals = {
        "repos": len(instances),
        "predictions": len(prepared.predictions),
        "resolved": sum(entry["resolved"] for entry in latest),
        "build_passed": sum(entry["build_passed"] for entry in latest),
        **(
            {
                "lean_verify_passed": sum(
                    entry["lean_verify_passed"] for entry in latest
                )
            }
            if prepared.replay
            else {
                "signatures_preserved": sum(
                    entry["signatures_preserved"] for entry in latest
                ),
                "named_signatures_preserved": sum(
                    entry["named_signatures_preserved"] for entry in latest
                ),
                "semantic_integrity_preserved": sum(
                    entry["semantic_integrity_preserved"] for entry in latest
                ),
            }
        ),
        "evaluation_infrastructure_ok": sum(
            entry["evaluation_infrastructure_ok"] for entry in latest
        ),
        "benchmark_fixtures_preserved": sum(
            entry.get("benchmark_fixtures_preserved") is True
            for entry in latest
            if entry.get("benchmark_fixtures_preserved") is not None
        ),
        "benchmark_fixture_warnings": sum(
            entry.get("benchmark_fixtures_preserved") is False
            for entry in latest
        ),
        "repos_with_sorry": sum(entry["n_sorry"] > 0 for entry in latest),
        "cost_total": sum(float(entry["cost_total"]) for entry in latest),
        "source_metrics_complete": all(
            entry["source_metrics_complete"] for entry in latest
        ),
        "evaluation_complete": (
            len(instances) == len(prepared.predictions)
            and all(entry["build_ran"] or not entry["apply_patch_ok"] for entry in latest)
        ),
        "source_metric_repos": len(instances),
        "baseline_words": sum(int(row["baseline_words"]) for row in metrics),
        "words_saved": sum(int(row["words_saved"]) for row in metrics),
        "baseline_lean_tokens": sum(
            int(row["baseline_lean_tokens"]) for row in metrics
        ),
        "lean_tokens_saved": sum(
            int(row["lean_tokens_saved"]) for row in metrics
        ),
    }
    _atomic_json(
        summary_path,
        {
            "schema_version": 2,
            "kind": "leanlean_run_report_summary",
            "generated_at": _utc_now(),
            "source": {
                "basis": "authoritative postprocessed submitted endpoints",
                "postprocess_id": prepared.manifest["postprocess_id"],
                "run_combination": copy.deepcopy(prepared.manifest["run_combination"]),
                "sidecar_count": len(instances),
            },
            "totals": totals,
            "headline": headline,
            "final_evaluation": {
                "submitted": denominator,
                "submitted_endpoints_captured": sum(
                    entry["apply_patch_ok"] for entry in latest
                ),
                "non_empty_patches": sum(
                    not entry["empty_patch"] for entry in latest
                ),
                "build_attempted": checkpoint_builds["final_attempted"],
                "build_passed": checkpoint_builds["final_passed"],
                **(
                    {
                        "lean_verify_passed": checkpoint_builds[
                            "final_lean_verify_passed"
                        ]
                    }
                    if prepared.replay
                    else {
                        "signatures_preserved": checkpoint_builds[
                            "final_signature_green"
                        ]
                    }
                ),
                "valid_submissions": totals["resolved"],
            },
            "checkpoint_builds": checkpoint_builds,
            "difficulty_cohorts": difficulty_cohorts,
            "benchmark_tools": copy.deepcopy(
                (actions or {}).get("benchmark_tools", {})
                if isinstance(actions, Mapping)
                else {}
            ),
            "methodology": methodology,
            "completeness": {
                "complete": len(instances) == len(prepared.predictions),
                "prediction_repos": len(prepared.predictions),
                "report_repos": len(instances),
                "missing_reports": sorted(set(prepared.predictions) - set(instances)),
                "unexpected_reports": sorted(set(instances) - set(prepared.predictions)),
                "source_metrics_complete": all(
                    entry["source_metrics_complete"] for entry in latest
                ),
            },
            "actions": actions,
            "instances": instances,
        },
    )


def finalize_run(
    prepared: PreparedPostprocessing,
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    _write_combined_action_summary(prepared)
    repositories = {
        str(row["id"]): row for row in prepared.manifest["repositories"]
    }
    capture_rows = list(prepared.manifest["captures"])
    buildkit_cache_cleanup_needed = threading.Event()
    buildkit_cache_cleanup: dict[str, Any] = {"status": "not_needed"}

    def process(row: Mapping[str, Any]) -> dict[str, Any]:
        source_output = Path(
            str(row.get("output_directory") or prepared.output_dir)
        ).resolve()
        capture_dir = source_output / Path(str(row["playback"])).parent
        instance_id = str(row["instance_id"])
        repository = repositories[instance_id]
        materialization = repository.get("materialization")
        cached_result: dict[str, Any] | None = None
        cached_path = _previous_playback_path(capture_dir, replay=prepared.replay)
        if cached_path.is_file():
            try:
                loaded = json.loads(cached_path.read_text())
            except (OSError, ValueError):
                loaded = None
            if (
                isinstance(loaded, dict)
                and loaded.get("instance_id") == instance_id
                and loaded.get("source_playback_sha256") == row.get("playback_sha256")
                and isinstance(loaded.get("selected_state"), Mapping)
                and (not prepared.manifest["container"].get("endpoints_only")
                     or loaded["selected_state"].get("policy") == "submitted_final_state")
            ):
                points = [
                    point
                    for point in loaded.get("points") or []
                    if isinstance(point, Mapping)
                ]
                submitted = next(
                    (
                        point
                        for point in reversed(points)
                        if point.get("endpoint_role") == "submitted_patch"
                    ),
                    None,
                )
                selected_edit = (loaded.get("selected_state") or {}).get("edit_index")
                validation_points = [
                    point
                    for point in points
                    if point is submitted
                    or (
                        selected_edit is not None
                        and point.get("edit_index") == selected_edit
                        and isinstance(point.get("build"), Mapping)
                    )
                ]
                if (
                    submitted is not None
                    and validation_points
                    and all(
                        _cached_validation_current(
                            point,
                            repository,
                            replay=prepared.replay,
                            verification_build_policy=str(
                                prepared.manifest["container"].get(
                                    "verification_build_policy"
                                )
                                or VERIFICATION_BUILD_POLICY
                            ),
                        )
                        and _lake_build_artifact_current(
                            point, repository, prepared.manifest["container"]
                        )
                        for point in validation_points
                    )
                    and _heartbeat_result_current(
                        prepared.manifest.get("heartbeats"),
                        instance_id,
                        submitted,
                    )
                ):
                    cached_result = copy.deepcopy(loaded)

        def run_capture(
            resolved_repository: Mapping[str, Any],
            *,
            defer_build: bool,
        ) -> dict[str, Any]:
            return process_capture(
                capture_dir,
                resolved_repository,
                prepared.manifest["container"],
                str(prepared.predictions[instance_id]["model_patch"]),
                replay=prepared.replay,
                repo_root=prepared.source_root,
                progress=progress,
                heartbeat_config=prepared.manifest.get("heartbeats"),
                defer_build=defer_build,
            )

        if cached_result is not None:
            result = cached_result
            retirement = {
                "status": "not_materialized",
                "reason": "postprocessing_and_heartbeat_cache_current",
            }
            if progress is not None:
                progress(instance_id, 1, 1)
        else:
            try:
                result = run_capture(repository, defer_build=True)
                retirement = {
                    "status": "not_materialized",
                    "reason": "postprocessing_and_heartbeat_cache_current",
                }
            except CaptureBuildRequired:
                buildkit_cache_cleanup_needed.set()
                with _materialized_repository_image(repository) as (
                    resolved_repository,
                    retirement,
                ):
                    result = run_capture(resolved_repository, defer_build=False)
        if isinstance(materialization, Mapping) and "shared_environment" in materialization:
            shared_environment = _mapping(
                materialization.get("shared_environment"),
                f"{instance_id}.shared_environment",
            )
            warm_cache = _mapping(
                materialization.get("warm_build_cache"),
                f"{instance_id}.warm_build_cache",
            )
            result["image_materialization"] = {
                "backend": materialization.get("backend"),
                "environment_id": shared_environment.get("id"),
                "warm_build_cache_sha256": warm_cache.get("sha256"),
                "image_persistence": "ephemeral",
                "retirement": retirement,
            }
        elif isinstance(materialization, Mapping):
            # Images built from the published source tree (dockerfile_network_v1).
            result["image_materialization"] = {
                "backend": materialization.get("backend", "dockerfile_network_v1"),
                "tree_sha256": materialization.get("tree_sha256"),
                "image_persistence": "persist" if materialization.get("persist") else "ephemeral",
                "retirement": retirement,
            }
        result["_capture_name"] = str(row["capture"])
        result["_source_run_id"] = str(row["source_run_id"])
        result["_source_output_directory"] = str(source_output)
        destination = derived_playback_path(capture_dir, replay=prepared.replay)
        if cached_result is not None and cached_path != destination:
            _atomic_json(destination, result)
        result["_derived_playback_path"] = str(destination)
        return result

    workers = min(
        len(capture_rows), int(prepared.manifest["parallelism"]["workers"])
    )
    try:
        if workers <= 1:
            captures = [process(row) for row in capture_rows]
        else:
            captures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process, row): row for row in capture_rows}
                for future in concurrent.futures.as_completed(futures):
                    captures.append(future.result())
    finally:
        if buildkit_cache_cleanup_needed.is_set():
            buildkit_cache_cleanup = prune_buildkit_cache()
    newest: dict[str, dict[str, Any]] = {}
    for capture in captures:
        instance_id = str(capture["instance_id"])
        previous = newest.get(instance_id)
        if previous is None or str(capture["_capture_name"]) > str(
            previous["_capture_name"]
        ):
            newest[instance_id] = capture
    _write_postprocessed_report_summary(prepared, newest)
    scores = [float(item["selected_state"]["word_compression_pct"]) for item in newest.values()]
    token_scores = [
        float(item["selected_state"]["lean_token_compression_pct"])
        for item in newest.values()
    ]
    expected = len(prepared.manifest["repositories"])
    if len(scores) != expected:
        raise ValueError(f"postprocessed {len(scores)} repositories, expected {expected}")
    lake_build_inventory: list[dict[str, Any]] = []
    for instance_id, capture in sorted(newest.items()):
        submitted = next(
            (
                point
                for point in reversed(capture.get("points") or [])
                if point.get("endpoint_role") == "submitted_patch"
            ),
            None,
        )
        artifact = (
            submitted.get("lake_build_artifact")
            if isinstance(submitted, Mapping)
            else None
        )
        lake_build_inventory.append(
            {
                "repository": instance_id,
                **(
                    copy.deepcopy(dict(artifact))
                    if isinstance(artifact, Mapping)
                    else {"status": "missing"}
                ),
            }
        )
    lake_build_bytes = sum(
        int(item.get("archive_bytes") or 0)
        for item in lake_build_inventory
        if item.get("status") == "complete"
    )
    summary = {
        "kind": "leanlean_postprocess_summary",
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source_run_id": prepared.source_run["run_id"],
        "postprocess_id": prepared.manifest["postprocess_id"],
        "mode": prepared.manifest["mode"],
        "repositories": expected,
        "run_combination": copy.deepcopy(prepared.manifest["run_combination"]),
        "headline": {
            "metric": "macro_average_selected_state_word_compression_pct",
            "value": round(sum(scores) / expected, 6),
            "lean_token_value": round(sum(token_scores) / expected, 6),
            "denominator": expected,
            "policy": dict(prepared.manifest["evaluation"]),
        },
        "selection": {
            instance_id: capture["selected_state"]
            for instance_id, capture in sorted(newest.items())
        },
        "benchmark_fixture_integrity": {
            "policy": (
                "audit submitted agent-visible files, then restore registered "
                + (
                    "copies before lean_verify and token evaluation"
                    if prepared.replay
                    else "copies before authoritative Comparator and token evaluation"
                )
            ),
            "checked": sum(
                item.get("benchmark_fixtures_preserved") is not None
                for item in newest.values()
            ),
            "preserved": sum(
                item.get("benchmark_fixtures_preserved") is True
                for item in newest.values()
            ),
            "warnings": sorted(
                instance_id
                for instance_id, item in newest.items()
                if item.get("benchmark_fixtures_preserved") is False
            ),
        },
        "checkpoint_builds": {
            "edit_replay_requested": prepared.replay,
            "final_submissions": expected,
            "final_attempted": sum(
                item.get("final_submission_build_attempted") is True
                for item in captures
            ),
            "final_passed": sum(
                item.get("build_passed") is True for item in captures
            ),
            **(
                {
                    "final_lean_verify_passed": sum(
                        item.get("lean_verify_passed") is True
                        for item in captures
                    )
                }
                if prepared.replay
                else {
                    "final_signature_green": sum(
                        item.get("signatures_preserved") is True
                        for item in captures
                    ),
                    "final_named_signatures_preserved": sum(
                        item.get("named_signatures_preserved") is True
                        for item in captures
                    ),
                }
            ),
            "edit_checkpoints": sum(
                int(item.get("edit_checkpoint_count") or 0)
                for item in captures
            ),
            "attempted": sum(
                int(item.get("checkpoint_build_attempt_count") or 0)
                for item in captures
            ),
            "passed": sum(int(item.get("checkpoint_build_pass_count") or 0) for item in captures),
            "failed": sum(int(item.get("checkpoint_build_fail_count") or 0) for item in captures),
        },
        "lake_builds": {
            "requested": prepared.manifest["container"].get("save_lake_build") is True,
            "repositories": expected,
            "exported": sum(
                item.get("status") == "complete" for item in lake_build_inventory
            ),
            "missing_or_failed": sum(
                item.get("status") != "complete" for item in lake_build_inventory
            ),
            "archive_bytes_total": lake_build_bytes,
            "artifacts": lake_build_inventory,
        },
        "docker_build_cache_cleanup": buildkit_cache_cleanup,
        "artifacts": {
            "report_summary": str(prepared.output_dir / ("replay_report_summary.json" if prepared.replay else "report_summary.json")),
            "action_summary": str(prepared.output_dir / ("replay_action_summary.json" if prepared.replay else "action_summary.json")),
            "derived_playback": sorted(
                str(capture["_derived_playback_path"])
                for capture in captures
            ),
            "source_output_directories": sorted(
                {str(capture["_source_output_directory"]) for capture in captures}
            ),
            **(
                {"lake_builds": prepared.manifest["outputs"]["lake_builds"]}
                if prepared.manifest["container"].get("save_lake_build") is True
                else {}
            ),
        },
    }
    _atomic_json(prepared.output_dir / ("replay_summary.json" if prepared.replay else SUMMARY_FILENAME), summary)
    artifact = dict(_load_yaml(prepared.run_artifact_path, "postprocessing run artifact"))
    artifact["results"] = {
        "headline_word_compression_pct": summary["headline"]["value"],
        "headline_lean_token_compression_pct": summary["headline"]["lean_token_value"],
        "checkpoint_builds": summary["checkpoint_builds"],
    }
    _atomic_yaml(prepared.run_artifact_path, artifact)
    return summary


__all__ = [
    "DERIVED_PLAYBACK",
    "PreparedPostprocessing",
    "finalize_run",
    "load_run_manifest",
    "process_capture",
    "resolve_run",
    "update_run_status",
    "write_run_definition",
]
