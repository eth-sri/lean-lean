#!/usr/bin/env python3
"""Plan and launch an immutable selective retry of an evaluation run."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from leanlean.pipeline.evaluation import resolve_config  # noqa: E402
from scripts import generate  # noqa: E402


ACTIVE_STATUSES = frozenset({"running", "waiting_for_quota"})
RETRYABLE_EXIT_STATUSES = frozenset(
    {"ExecutionFailed", "ProviderFailed", "QuotaExceeded"}
)


@dataclass(frozen=True)
class Attempt:
    config_path: Path
    run_id: str
    repositories: tuple[str, ...]
    output_dir: Path
    status: str
    tmux_session: str
    playback_required: bool


def _yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read YAML from {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping")
    return dict(value)


def _run_status(run_id: str) -> str:
    path = REPO_ROOT / "runs/evaluation" / run_id / "run.yaml"
    if not path.is_file():
        return ""
    value = _yaml(path)
    status = value.get("status")
    return str(status) if status is not None else ""


def _attempt(config_path: Path) -> Attempt:
    resolved = resolve_config(config_path, repo_root=REPO_ROOT)
    return Attempt(
        config_path=resolved.config_path,
        run_id=str(resolved.manifest["run_id"]),
        repositories=tuple(
            str(row["id"]) for row in resolved.manifest["repositories"]
        ),
        output_dir=resolved.output_dir,
        status=_run_status(str(resolved.manifest["run_id"])),
        tmux_session=resolved.tmux_session,
        playback_required=bool(
            resolved.engine_manifest.get("playback", {}).get("enabled", False)
        ),
    )


def _retry_index(root_run_id: str, path: Path) -> int | None:
    pattern = rf"^{re.escape(root_run_id)}_retry_r([1-9][0-9]*)$"
    try:
        config = _yaml(path)
    except ValueError:
        return None
    match = re.fullmatch(pattern, str(config.get("run_id", "")))
    return int(match.group(1)) if match else None


def _retry_configs(root_run_id: str) -> list[tuple[int, Path]]:
    directory = REPO_ROOT / "configs/evaluation"
    indexed = (
        (_retry_index(root_run_id, path), path)
        for path in directory.glob(f"{root_run_id}_retry_r*.yaml")
    )
    return sorted(
        (index, path) for index, path in indexed if index is not None
    )


def _tmux_is_alive(session: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _attempt_is_active(attempt: Attempt) -> bool:
    return attempt.status in ACTIVE_STATUSES and _tmux_is_alive(
        attempt.tmux_session
    )


def _retry_reason(attempt: Attempt, instance_id: str) -> str | None:
    if _attempt_is_active(attempt):
        return "active"
    if not generate._instance_has_existing_output(
        attempt.output_dir, instance_id
    ):
        return "missing_trajectory"
    exit_status = generate._get_existing_exit_status(
        attempt.output_dir, instance_id
    )
    if exit_status in RETRYABLE_EXIT_STATUSES:
        return str(exit_status)
    if not generate._instance_has_prediction(attempt.output_dir, instance_id):
        return "missing_prediction"
    if attempt.playback_required and not generate._instance_has_scoring_capture(
        attempt.output_dir, instance_id
    ):
        return "missing_scoring_capture"
    return None


def _subscription_failure_reasons(attempt: Attempt) -> dict[str, str]:
    """Return explicit API/quota failures recorded by the run wrapper."""

    path = attempt.output_dir / "subscription_status.json"
    if not path.is_file():
        return {}
    try:
        value = _yaml(path)
    except ValueError:
        return {}
    failures = value.get("failures")
    if not isinstance(failures, list):
        return {}
    retryable_kinds = {"provider_error", "quota_exhausted"}
    reasons: dict[str, str] = {}
    for failure in failures:
        if not isinstance(failure, Mapping):
            continue
        instance_id = failure.get("instance_id")
        kind = failure.get("kind")
        if (
            isinstance(instance_id, str)
            and isinstance(kind, str)
            and kind in retryable_kinds
        ):
            reasons[instance_id] = kind
    return reasons


def _write_retry_config(
    root: Attempt,
    root_config: Mapping[str, Any],
    repositories: list[str],
    index: int,
) -> Path:
    run_id = f"{root.run_id}_retry_r{index}"
    config = dict(root_config)
    config["run_id"] = run_id
    config["repositories"] = repositories
    parallelism = dict(config["parallelism"])
    parallelism["workers"] = min(
        int(parallelism["workers"]), len(repositories)
    )
    config["parallelism"] = parallelism
    path = root.config_path.parent / f"{run_id}.yaml"
    content = (
        f"# Auto-generated selective retry of {root.run_id}.\n"
        "# Only unresolved repositories are included; the parent run is immutable.\n"
        + yaml.safe_dump(config, sort_keys=False, width=100)
    )
    if path.exists() and path.read_text() != content:
        raise RuntimeError(
            f"retry config collision at {path}; refusing to overwrite it"
        )
    path.write_text(content)
    return path


def prepare_retry(config_path: Path) -> tuple[Path | None, dict[str, str]]:
    root = _attempt(config_path)
    root_config = _yaml(root.config_path)
    retry_configs = _retry_configs(root.run_id)
    attempts = [root]
    for _, retry_path in retry_configs:
        attempt = _attempt(retry_path)
        attempts.append(attempt)
        if attempt.status in {"", "configured"} and not _attempt_is_active(
            attempt
        ):
            reasons = {
                instance_id: "prepared_not_launched"
                for instance_id in attempt.repositories
            }
            return attempt.config_path, reasons

    latest = {instance_id: root for instance_id in root.repositories}
    for attempt in attempts[1:]:
        for instance_id in attempt.repositories:
            if instance_id not in latest:
                raise ValueError(
                    f"retry {attempt.run_id} contains repository {instance_id!r} "
                    f"that is absent from root run {root.run_id}"
                )
            latest[instance_id] = attempt

    active = {
        instance_id: attempt
        for instance_id, attempt in latest.items()
        if _attempt_is_active(attempt)
    }
    if active:
        sessions = sorted({attempt.tmux_session for attempt in active.values()})
        print("retry already active: " + ", ".join(sessions))
        return None, {instance_id: "active" for instance_id in active}

    reasons: dict[str, str] = {}
    for instance_id in root.repositories:
        attempt = latest[instance_id]
        reason = _subscription_failure_reasons(attempt).get(instance_id)
        if reason is None:
            reason = _retry_reason(attempt, instance_id)
        if reason is not None:
            reasons[instance_id] = reason
    if not reasons:
        print(f"nothing to retry: all {len(root.repositories)} repositories are complete")
        return None, {}

    next_index = retry_configs[-1][0] + 1 if retry_configs else 1
    path = _write_retry_config(
        root, root_config, list(reasons), next_index
    )
    return path, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    retry_path, reasons = prepare_retry(config_path)
    if retry_path is None:
        return 0

    relative = retry_path.relative_to(REPO_ROOT)
    print(f"retry config: {relative}")
    for instance_id, reason in reasons.items():
        print(f"  {instance_id}: {reason}")
    if args.prepare_only:
        return 0
    return subprocess.run(
        ["bash", str(REPO_ROOT / "eval.sh"), str(relative)],
        cwd=REPO_ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
