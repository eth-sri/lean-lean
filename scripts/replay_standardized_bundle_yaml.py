#!/usr/bin/env python3
"""Replay one immutable standardized trajectory bundle from a YAML manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from leanlean.environments.docker import DockerEnvironment
from leanlean.replay_analysis import write_edit_build_analysis
from leanlean.standardized_replay import (
    ReplayPolicy,
    StandardizedReplayEngine,
    _shell_asynchrony_reason,
    install_replay_tool_bundle,
    load_replay_bundle,
)


def _path(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _positive_int(document: dict[str, Any], field: str) -> int:
    value = int(document.get(field) or 0)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value

def _edit_replay_sequence(result: dict[str, Any]) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for point in result.get("points") or []:
        if not isinstance(point, dict):
            continue
        if int(point.get("edit_index") or 0) <= 0:
            continue
        if point.get("kind") == "unattributed_action_endpoint_change":
            continue
        snapshot = point.get("snapshot_artifact")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        diff = point.get("incremental_diff_artifact")
        diff = diff if isinstance(diff, dict) else {}
        build = point.get("build")
        build = build if isinstance(build, dict) else {}
        sequence.append(
            {
                "edit_index": point.get("edit_index"),
                "action_index": point.get("action_index"),
                "source_archive_sha256": point.get("source_archive_sha256"),
                "snapshot_archive_sha256": point.get("snapshot_archive_sha256"),
                "snapshot_artifact_sha256": snapshot.get("sha256"),
                "incremental_diff_sha256": diff.get("sha256"),
                "changed_files": point.get("changed_files"),
                "build_passed": build.get("passed"),
                "build_returncode": build.get("returncode"),
                "build_source_tree_unchanged": build.get(
                    "source_tree_unchanged"
                ),
                "cost_usd": point.get("cost_usd"),
                "marginal_cost_since_previous_edit_usd": point.get(
                    "marginal_cost_since_previous_edit_usd"
                ),
            }
        )
    return sequence


def _sequence_sha256(sequence: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        sequence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("replay manifest must be a YAML mapping")
    if document.get("kind") != "standardized_bundle_replay":
        raise ValueError("kind must be standardized_bundle_replay")
    if document.get("network_policy") != "none":
        raise ValueError("offline replay requires network_policy: none")

    bundle_path = _path(manifest_path.parent, document.get("bundle"), field="bundle")
    output_path = _path(manifest_path.parent, document.get("output"), field="output")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite replay result: {output_path}")
    evaluation = document.get("evaluation")
    if evaluation is None:
        evaluation = {"require_standardized_session_replay_exact": True}
    if evaluation is not None and not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a YAML mapping")
    bundle = load_replay_bundle(bundle_path)
    environment = bundle.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("replay bundle has no environment metadata")
    image = str(environment.get("image") or "")
    if not image:
        raise ValueError("replay bundle has no pinned image")
    expected_image = document.get("image")
    if expected_image is not None and str(expected_image) != image:
        raise ValueError("manifest image differs from replay bundle")
    bundled_image_id = str(environment.get("image_id") or "")
    manifest_image_id = str(document.get("image_id") or "")
    if bundled_image_id and manifest_image_id and bundled_image_id != manifest_image_id:
        raise ValueError("manifest image_id differs from replay bundle")
    image_id = bundled_image_id or manifest_image_id
    if not image_id.startswith("sha256:"):
        raise ValueError("offline replay requires an immutable sha256 image_id")

    cpus = _positive_int(document, "container_cpus")
    pids_limit = _positive_int(document, "container_pids_limit")
    action_timeout = _positive_int(document, "replay_action_timeout_seconds")
    memory = str(document.get("container_memory") or "")
    container_timeout = str(document.get("container_timeout") or "")
    if not memory or not container_timeout:
        raise ValueError("container_memory and container_timeout are required")
    cgroup_parent = str(document.get("container_cgroup_parent") or "")
    run_args = [
        "--rm",
        f"--cpus={cpus}",
        f"--memory={memory}",
        f"--memory-swap={memory}",
    ]

    verify_each_mutation = bool(document.get("verify_each_mutation"))
    reject_asynchronous_actions = bool(document.get("reject_asynchronous_actions"))
    asynchronous_actions: list[tuple[int, str]] = []
    for action in bundle["_trajectory"].get("actions") or []:
        if not isinstance(action, dict) or action.get("name") != "Bash":
            continue
        arguments = action.get("arguments")
        command = arguments.get("command") if isinstance(arguments, dict) else None
        reason = (
            _shell_asynchrony_reason(command)
            if isinstance(command, str)
            else None
        )
        if reason:
            asynchronous_actions.append(
                (int(action.get("action_index") or 0), reason)
            )
    if reject_asynchronous_actions and asynchronous_actions:
        rendered = ", ".join(
            f"action {index} ({reason})"
            for index, reason in asynchronous_actions
        )
        raise ValueError(
            "strict replay cannot establish closed edit boundaries for "
            f"asynchronous trajectory actions: {rendered}"
        )
    build_each_checkpoint = bool(document.get("build_each_checkpoint"))
    bundled_build_command = str(environment.get("build_command") or "")
    manifest_build_command = str(document.get("build_command") or "")
    if (
        manifest_build_command
        and bundled_build_command
        and manifest_build_command != bundled_build_command
    ):
        raise ValueError("manifest build_command differs from replay bundle")
    build_command = manifest_build_command or bundled_build_command
    if build_each_checkpoint and not build_command:
        raise ValueError(
            "build_each_checkpoint requires the bundle or manifest build_command"
        )
    build_timeout = int(document.get("build_timeout_seconds") or 3600)
    build_output_chars = int(document.get("build_output_chars") or 4000)
    if build_timeout <= 0 or build_output_chars <= 0:
        raise ValueError("build timeout and output size must be positive")

    retain_checkpoints = bool(document.get("retain_edit_checkpoints"))
    checkpoint_dir: Path | None = None
    if retain_checkpoints:
        checkpoint_value = document.get("checkpoint_output")
        if checkpoint_value is None:
            checkpoint_dir = output_path.parent / "checkpoints"
        else:
            checkpoint_dir = _path(
                manifest_path.parent,
                checkpoint_value,
                field="checkpoint_output",
            )
        if checkpoint_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite replay checkpoints: {checkpoint_dir}"
            )

    edit_evidence_settings = {
        "require_nonzero_edit_count",
        "require_every_edit_checkpoint_archived",
        "require_every_edit_checkpoint_diffed",
        "require_every_edit_checkpoint_built",
        "require_every_edit_checkpoint_costed",
        "require_replay_digest_sequence_equality",
    }
    strict_edit_evidence = any(
        evaluation.get(setting) is True for setting in edit_evidence_settings
    )
    if strict_edit_evidence and not verify_each_mutation:
        raise ValueError("strict edit evidence requires verify_each_mutation: true")
    if strict_edit_evidence and not reject_asynchronous_actions:
        raise ValueError(
            "strict edit evidence requires reject_asynchronous_actions: true"
        )
    if (
        evaluation.get("require_every_edit_checkpoint_archived") is True
        or evaluation.get("require_every_edit_checkpoint_diffed") is True
    ) and not retain_checkpoints:
        raise ValueError(
            "retained edit evidence requires retain_edit_checkpoints: true"
        )
    if (
        evaluation.get("require_every_edit_checkpoint_built") is True
        and not build_each_checkpoint
    ):
        raise ValueError(
            "per-edit build evidence requires build_each_checkpoint: true"
        )
    if strict_edit_evidence and str(
        document.get("terminal_reconciliation") or "none"
    ) != "none":
        raise ValueError("strict edit evidence forbids terminal reconciliation")
    if (
        evaluation.get("require_replay_digest_sequence_equality") is True
        and document.get("compare_against") is None
    ):
        raise ValueError("repeat digest equality requires compare_against")
    if cgroup_parent:
        run_args.append(f"--cgroup-parent={cgroup_parent}")

    replay_env = DockerEnvironment(
        image=image_id,
        cwd="/testbed",
        env={},
        forward_env=[],
        timeout=action_timeout,
        run_args=run_args,
        network_mode="none",
        network_policy="none",
        container_pids_limit=pids_limit,
        container_timeout=container_timeout,
        container_grace_seconds=120,
        refactor_rounds=0,
    )
    try:
        tool_environment = install_replay_tool_bundle(replay_env, environment)
        action_total = len(bundle["_trajectory"].get("actions") or [])
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        with progress:
            task = progress.add_task("Exact edit replay", total=action_total)

            def update_progress(event: dict[str, Any]) -> None:
                action_index = int(event.get("action_index") or 0)
                name = str(event.get("name") or "action")
                phase = str(event.get("phase") or "")
                suffix = ""
                if phase == "action_completed" and event.get("source_changed"):
                    suffix = " · snapshot + diff + build"
                progress.update(
                    task,
                    completed=action_index if phase == "action_completed" else None,
                    description=f"Exact edit replay · {action_index}/{action_total} · {name}{suffix}",
                )

            result = StandardizedReplayEngine(

                bundle_path,
                replay_env=replay_env,
                policy=ReplayPolicy(
                    bash=str(document.get("bash_replay_policy") or "all"),
                    terminal_reconciliation=str(
                        document.get("terminal_reconciliation") or "none"
                    ),
                    action_timeout_seconds=action_timeout,
                    verify_each_mutation=verify_each_mutation,
                    build_after_each_mutation=build_each_checkpoint,
                    build_command=build_command if build_each_checkpoint else "",
                    build_timeout_seconds=build_timeout,
                    build_output_chars=build_output_chars,
                    reject_asynchronous_actions=reject_asynchronous_actions,
                ),
                instance_id=str((document.get("repos") or [""])[0]),
                final_cost_usd=float(document.get("final_cost_usd") or 0.0),
                checkpoint_dir=checkpoint_dir,
                progress_callback=update_progress,
            ).replay()
            progress.update(task, completed=action_total)
        result["replay_tool_environment"] = tool_environment
    finally:
        replay_env.cleanup()

    edit_sequence = _edit_replay_sequence(result)
    result["edit_replay_sequence_sha256"] = _sequence_sha256(edit_sequence)
    result["edit_replay_sequence_length"] = len(edit_sequence)
    result["replay_digest_sequence_reproducible"] = None
    comparison_value = document.get("compare_against")
    if comparison_value is not None:
        comparison_path = _path(
            manifest_path.parent,
            comparison_value,
            field="compare_against",
        )
        comparison_result = json.loads(comparison_path.read_text(encoding="utf-8"))
        if not isinstance(comparison_result, dict):
            raise TypeError("compare_against result must be a JSON mapping")
        comparison_sequence = _edit_replay_sequence(comparison_result)
        comparison_sha256 = _sequence_sha256(comparison_sequence)
        prior_validation = comparison_result.get("offline_validation")
        prior_validation_passed = bool(
            isinstance(prior_validation, dict)
            and prior_validation.get("passed") is True
        )
        sequence_matches = edit_sequence == comparison_sequence
        result["replay_digest_sequence_reproducible"] = bool(
            sequence_matches and prior_validation_passed
        )
        result["repeat_validation"] = {
            "compare_against": str(comparison_path),
            "prior_validation_passed": prior_validation_passed,
            "prior_sequence_sha256": comparison_sha256,
            "current_sequence_sha256": result["edit_replay_sequence_sha256"],
            "sequence_matches": sequence_matches,
            "passed": result["replay_digest_sequence_reproducible"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exactness_fields = {
        "require_native_stream_exact": "native_stream_exact",
        "require_mutation_trace_exact": "mutation_trace_exact",
        "require_subagent_lineage_complete": "subagent_lineage_complete",
        "require_action_endpoint_digest_equality": (
            "trajectory_action_replay_exact"
        ),
        "require_submitted_diff_digest_equality": "source_diff_exact",
        "require_endpoint_reconstruction_exact": "endpoint_reconstruction_exact",
        "require_standardized_session_replay_exact": (
            "standardized_session_replay_exact"
        ),
        "require_action_boundary_sequence_complete": (
            "action_boundary_sequence_complete"
        ),
        "require_nonzero_edit_count": "nonzero_edit_count",
        "require_every_edit_checkpoint_archived": (
            "every_edit_checkpoint_archived"
        ),
        "require_every_edit_checkpoint_diffed": "every_edit_checkpoint_diffed",
        "require_every_edit_checkpoint_built": "every_edit_checkpoint_built",
        "require_every_edit_checkpoint_costed": "every_edit_checkpoint_costed",
        "require_action_cost_boundaries": "every_edit_checkpoint_costed",
        "require_all_checkpoint_builds_passed": "all_checkpoint_builds_passed",
        "require_replay_digest_sequence_equality": (
            "replay_digest_sequence_reproducible"
        ),
    }
    required = {
        result_field: result.get(result_field) is True
        for setting, result_field in exactness_fields.items()
        if evaluation.get(setting) is True
    }
    validation_passed = bool(required) and all(required.values())
    result["offline_validation"] = {
        "required_exactness": sorted(required),
        "checks": required,
        "passed": validation_passed,
    }
    analysis_value = document.get("analysis_output")
    analysis_output = (
        _path(manifest_path.parent, analysis_value, field="analysis_output")
        if analysis_value is not None
        else output_path.with_suffix("").with_name(
            output_path.with_suffix("").name + ".analysis"
        )
    )
    result["analysis_artifacts"] = write_edit_build_analysis(
        result, analysis_output
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not required:
        raise ValueError("evaluation must require at least one exactness check")
    if not validation_passed:
        failed = sorted(field for field, passed in required.items() if not passed)
        raise RuntimeError(
            "offline standardized replay failed required exactness checks: "
            + ", ".join(failed)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="standardized bundle replay YAML")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = replay_manifest(args.manifest)
    print(
        json.dumps(
            {
                "provider": result.get("provider"),
                "action_count": result.get("action_count"),
                "trajectory_action_replay_exact": result.get(
                    "trajectory_action_replay_exact"
                ),
                "source_diff_exact": result.get("source_diff_exact"),
                "standardized_session_replay_exact": result.get(
                    "standardized_session_replay_exact"
                ),
                "offline_validation": result.get("offline_validation"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
