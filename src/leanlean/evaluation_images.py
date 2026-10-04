"""Build and retire evaluation images from pinned optimized source trees."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from leanlean.environment_archives import ensure_environment_image
from leanlean.preprocessing.cache_isolation import sanitize_disposable_build_context
from leanlean.preprocessing.standardized_repositories import (
    source_tree_sha256,
)
from leanlean.shared_cache import artifact_tree_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALIZATION_ENV = "LEANLEAN_REPO_MATERIALIZATIONS"
RECEIPTS_ENV = "LEANLEAN_MATERIALIZATION_RECEIPTS_DIR"
EPHEMERAL_IMAGES_ENV = "LEANLEAN_EPHEMERAL_IMAGES"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def materialization_specs_from_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    values = os.environ if env is None else env
    raw = values.get(MATERIALIZATION_ENV)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {MATERIALIZATION_ENV} JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{MATERIALIZATION_ENV} must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for instance_id, record in parsed.items():
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(record, dict)
        ):
            raise ValueError(f"{MATERIALIZATION_ENV} contains an invalid entry")
        result[instance_id] = dict(record)
    return result


def _image_record(image: str) -> tuple[str, dict[str, str]] | None:
    inspected = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}\t{{json .Config.Labels}}",
            image,
        ],
        capture_output=True,
        text=True,
    )
    if inspected.returncode != 0:
        return None
    image_id, _, raw_labels = inspected.stdout.strip().partition("\t")
    try:
        labels = json.loads(raw_labels) if raw_labels else {}
    except ValueError:
        labels = {}
    return image_id, labels if isinstance(labels, dict) else {}


def _source_tree(record: Mapping[str, Any], instance_id: str) -> Path:
    relative = record.get("source_tree")
    expected = record.get("tree_sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{instance_id}: source_tree is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{instance_id}: tree_sha256 is invalid")
    path = (REPO_ROOT / relative).resolve()
    allowed = (
        (REPO_ROOT / "prod_repos").resolve(),
        (REPO_ROOT / "datasets").resolve(),
    )
    if not any(path.is_relative_to(root) for root in allowed):
        raise ValueError(f"{instance_id}: source_tree escaped durable artifacts")
    if not path.is_dir() or source_tree_sha256(path) != expected:
        raise ValueError(f"{instance_id}: optimized source tree drift")
    return path


def _artifact_tree(record: Mapping[str, Any], instance_id: str) -> Path:
    raw = record.get("warm_build_cache")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{instance_id}: warm_build_cache is missing")
    relative = raw.get("path")
    expected = raw.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{instance_id}: warm build cache path is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{instance_id}: warm build cache hash is invalid")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to((REPO_ROOT / "datasets").resolve()):
        raise ValueError(f"{instance_id}: warm build cache escaped datasets/")
    if artifact_tree_sha256(path) != expected:
        raise ValueError(f"{instance_id}: warm build cache drift")
    return path


def _copy_tree(source: Path, destination: Path) -> None:
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)],
        check=True,
    )


def _build_from_shared_environment(
    instance_id: str,
    record: Mapping[str, Any],
    source_tree: Path,
    tag: str,
) -> None:
    environment = record.get("shared_environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"{instance_id}: shared_environment is missing")
    base_tag = str(environment.get("image") or "")
    ensure_environment_image(environment, repo_root=REPO_ROOT)
    warm_cache = _artifact_tree(record, instance_id)
    with tempfile.TemporaryDirectory(
        prefix=f"leanlean-materialize-{instance_id}-",
        dir=REPO_ROOT / "runs",
    ) as temporary:
        context = Path(temporary)
        _copy_tree(source_tree, context / "source")
        from leanlean.preprocessing.palomar_sources import localize_lake_manifest
        localize_lake_manifest(context / "source")
        _copy_tree(warm_cache, context / "warm-build")
        cache_audit = sanitize_disposable_build_context(context)
        print(f"{instance_id}: excluded {len(cache_audit['removed_files'])} orphan project cache artifacts", flush=True)
        # A reconciliation repository ships two refactor patches that the image
        # replays into `main` and `other-refactor`.  The branches cannot be
        # shipped in the source tree: every materialization Dockerfile starts by
        # discarding `.git`.
        reconciliation = record.get("reconciliation")
        seed_args: list[str] = []
        dockerfile = "docker/evaluation/Dockerfile.shared-cache"
        if isinstance(reconciliation, Mapping):
            dockerfile = "docker/evaluation/Dockerfile.shared-cache-reconciliation"
            seed_dir = context / "seed"
            seed_dir.mkdir()
            for role, argument in (("main", "MAIN_PATCH_SHA256"), ("other", "OTHER_PATCH_SHA256")):
                seed = reconciliation[role]
                source = REPO_ROOT / str(seed["path"])
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest != str(seed["sha256"]):
                    raise RuntimeError(
                        f"{instance_id}: reconciliation {role} patch drift at {source}"
                    )
                shutil.copyfile(source, seed_dir / f"{role}.patch")
                seed_args += ["--build-arg", f"{argument}={digest}"]
        command = [
            "docker", "build", "--network=none", "--progress=plain",
            "--file", str(REPO_ROOT / dockerfile),
            "--tag", tag,
            # Docker treats a bare sha256 digest in FROM as a registry
            # repository name.  The local tag is safe here because its exact
            # image ID was verified immediately above.
            "--build-arg", f"BASE_IMAGE={base_tag}",
            "--build-arg", f"INSTANCE_ID={instance_id}",
            "--build-arg", f"SOURCE_TREE_SHA256={record['tree_sha256']}",
            "--build-arg", f"ENVIRONMENT_ID={environment.get('id', '')}",
            "--build-arg", f"BUILD_TARGET={record.get('build_target', '')}",
            "--build-arg", f"LEAN_BUILD_THREADS={int(record.get('build_jobs', 8))}",
            *seed_args,
            str(context),
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            timeout=int(record.get("build_timeout_seconds", 14400)),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{instance_id}: shared-cache image build failed with "
                f"exit code {result.returncode}"
            )


def _write_receipt(instance_id: str, receipt: Mapping[str, Any]) -> None:
    raw_dir = os.environ.get(RECEIPTS_ENV)
    if not raw_dir:
        return
    directory = Path(raw_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{instance_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def materialized_image_id(tag: str) -> str:
    """Resolve a materialized tag to its immutable local image ID."""

    found = _image_record(tag)
    if found is None:
        raise RuntimeError(f"materialized image is missing: {tag}")
    return found[0]


def ensure_materialized_image_from_record(
    instance_id: str, record: Mapping[str, Any]
) -> bool:
    """Build one repository image from an explicit pinned record."""

    source_tree = _source_tree(record, instance_id)
    tag = str(record.get("tag") or "")
    if not tag:
        raise ValueError(f"{instance_id}: materialized image tag is missing")
    expected_sha = str(record["tree_sha256"])
    existing = _image_record(tag)
    if existing is not None:
        image_id, labels = existing
        if (
            labels.get(
                "org.openai.leanlean.source_tree_sha256"
            ) == expected_sha
            and labels.get(
                "org.openai.leanlean.instance_id"
            ) == instance_id
        ):
            _write_receipt(
                instance_id,
                {
                    "kind": "leanlean_evaluation_image_materialization",
                    "schema_version": 1,
                    "instance_id": instance_id,
                    "status": "reused",
                    "image": tag,
                    "image_id": image_id,
                    "source_tree": str(record["source_tree"]),
                    "tree_sha256": expected_sha,
                    "agent_lifetime_started": False,
                    "recorded_at": _utc_now(),
                },
            )
            return True
        raise RuntimeError(f"{instance_id}: materialized image tag drift: {tag}")

    if record.get("backend") == "shared_environment_v1":
        started_at = _utc_now()
        started = time.perf_counter()
        _build_from_shared_environment(instance_id, record, source_tree, tag)
        seconds = round(time.perf_counter() - started, 3)
    else:
        minimum_free = int(record.get("minimum_free_disk_gib", 0)) * 1024**3
        if shutil.disk_usage(source_tree).free < minimum_free:
            raise RuntimeError(f"{instance_id}: insufficient free disk for isolated image assembly")
        command = [
            "docker",
            "build",
            "--progress=plain",
            "--file",
            str(REPO_ROOT / "docker/evaluation/Dockerfile"),
            "--tag",
            tag,
            "--build-arg",
            f"INSTANCE_ID={instance_id}",
            "--build-arg",
            f"SOURCE_TREE_SHA256={expected_sha}",
            "--build-arg",
            f"BUILD_TARGET={record.get('build_target', '')}",
            "--build-arg",
            f"LEAN_BUILD_THREADS={int(record.get('build_jobs', 8))}",
            "--build-arg",
            f"REPO_VARIANT={record.get('repo_variant', 'optimized')}",
        ]
        if record.get("cache") == "disabled":
            command.append("--no-cache")
        started_at = _utc_now()
        started = time.perf_counter()
        # Never send a user's local Lake packages/builds or Git history to Docker.
        with tempfile.TemporaryDirectory(prefix="leanlean-source-image-") as temporary:
            context = Path(temporary) / "source"
            shutil.copytree(source_tree, context, ignore=shutil.ignore_patterns(".git", ".lake"))
            if source_tree_sha256(context) != expected_sha:
                raise ValueError(f"{instance_id}: isolated build context source drift")
            result = subprocess.run(
                [*command, str(context)],
                cwd=REPO_ROOT,
                text=True,
                timeout=int(record.get("build_timeout_seconds", 14400)),
            )
        seconds = round(time.perf_counter() - started, 3)
        if result.returncode != 0:
            raise RuntimeError(
                f"{instance_id}: evaluation image build failed with "
                f"exit code {result.returncode}"
            )
    built = _image_record(tag)
    if built is None:
        raise RuntimeError(f"{instance_id}: built image is missing: {tag}")
    image_id, labels = built
    checks = {
        "instance_id": labels.get(
            "org.openai.leanlean.instance_id"
        ) == instance_id,
        "source_tree_sha256": labels.get(
            "org.openai.leanlean.source_tree_sha256"
        ) == expected_sha,
        "materialization": labels.get(
            "org.openai.leanlean.materialization"
        ) in {"build_before_agent_v1", "shared_environment_v1"},
    }
    if not all(checks.values()):
        raise RuntimeError(f"{instance_id}: built image label drift: {checks}")
    _write_receipt(
        instance_id,
        {
            "kind": "leanlean_evaluation_image_materialization",
            "schema_version": 1,
            "instance_id": instance_id,
            "status": "built",
            "image": tag,
            "image_id": image_id,
            "source_tree": str(record["source_tree"]),
            "tree_sha256": expected_sha,
            "build_target": str(record.get("build_target", "")),
            "build_started_at": started_at,
            "build_finished_at": _utc_now(),
            "build_seconds": seconds,
            "agent_lifetime_started": False,
        },
    )
    return True


def ensure_materialized_image(instance_id: str) -> bool:
    """Build one repository image before its agent container exists."""

    record = materialization_specs_from_env().get(instance_id)
    if record is None:
        return False
    return ensure_materialized_image_from_record(instance_id, record)


def retire_image_tags(*tags: str) -> list[dict[str, Any]]:
    results = []
    for tag in dict.fromkeys(value for value in tags if value):
        removed = subprocess.run(
            ["docker", "image", "rm", tag],
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "tag": tag,
                "removed": removed.returncode == 0,
                "detail": (removed.stdout or removed.stderr).strip()[-1000:],
            }
        )
    return results


def prune_buildkit_cache() -> dict[str, Any]:
    """Remove unused cache from the shared Docker builder after replay work."""

    command = ["docker", "builder", "prune", "--all", "--force"]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        detail = str(error)
        print(f"Docker BuildKit cache cleanup failed: {detail}", flush=True)
        return {
            "status": "failed",
            "scope": "all_unused_cache_on_default_docker_builder",
            "command": command,
            "returncode": None,
            "detail": detail,
        }
    detail = (result.stdout or result.stderr).strip()[-4000:]
    status = "complete" if result.returncode == 0 else "failed"
    print(
        f"Docker BuildKit cache cleanup {status}: {detail or 'no details'}",
        flush=True,
    )
    return {
        "status": status,
        "scope": "all_unused_cache_on_default_docker_builder",
        "command": command,
        "returncode": result.returncode,
        "detail": detail,
    }
