"""Two-endpoint passive capture around an trace-utils native agent run."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leanlean.replay_analysis import write_edit_build_analysis
from leanlean.standardized_replay import (
    ReplayPolicy,
    StandardizedReplayEngine,
    StandardizedReplayError,
    _shell_asynchrony_reason,
    create_replay_bundle,
    install_replay_tool_bundle,
)
from leanlean.standardized_trace import (
    standardize_native_capture,
    write_standardized_trace,
)


@dataclass
class StandardizedRunRecorder:
    """Capture only before launch and after verified process quiescence.

    The caller launches ``trace-utils`` between :meth:`prepare` and :meth:`finish`.
    No source reads, native parsing, builds, or replay commands are performed in
    that interval.  ``finish`` consumes the immutable native capture produced by
    ``Agent(capture_mode="post_exit", trace=False)``.
    """

    env: Any
    output_dir: str | Path
    source_scope: Sequence[str]
    environment: Mapping[str, Any]
    instance_id: str = ""
    capture_timeout_seconds: int = 600

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser().resolve(strict=False)
        self.source_scope = tuple(str(path) for path in self.source_scope)
        if not self.source_scope:
            raise ValueError("source_scope cannot be empty")
        if self.capture_timeout_seconds <= 0:
            raise ValueError("capture_timeout_seconds must be positive")
        if not str(self.environment.get("image") or ""):
            raise ValueError("environment.image is required")
        if self.environment.get("network_policy") not in {
            "none",
            "model_proxy_only",
        }:
            raise ValueError("environment.network_policy must fail closed")
        if int(self.environment.get("pids_limit") or 0) <= 0:
            raise ValueError("environment.pids_limit must be positive")
        self._prepared = False
        self._terminal_source_captured = False
        self._terminal_captured = False
        self._finished = False
        self._baseline: dict[str, Any] = {}
        self._final: dict[str, Any] = {}
        self.bundle_path: Path | None = None
        self.standardized_trace: dict[str, Any] | None = None
        self.replay_result: dict[str, Any] | None = None

    def _capture(self, filename: str) -> dict[str, Any]:
        destination = self.output_dir / filename
        metadata = self.env.export_source_archive(
            destination,
            self.source_scope,
            timeout=self.capture_timeout_seconds,
        )
        digest = str(metadata.get("source_archive_sha256") or "")
        if not digest:
            raise RuntimeError("source archive capture returned no canonical digest")
        return {**metadata, "path": destination}

    def prepare(self) -> dict[str, Any]:
        if self._prepared:
            raise RuntimeError("standardized run recorder is already prepared")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._baseline = self._capture("baseline.sources.tar.gz")
        self._prepared = True
        return copy.deepcopy(self._baseline)

    def capture_terminal_source(self, *, post_exit_quiesced: bool) -> dict[str, Any]:
        if not self._prepared:
            raise RuntimeError("prepare must run before finish")
        if self._terminal_source_captured:
            raise RuntimeError("terminal source state is already captured")
        if not post_exit_quiesced:
            raise RuntimeError(
                "terminal source capture requires verified post-exit process quiescence"
            )
        # The terminal archive is deliberately captured before native parsing.
        self._final = self._capture("final.sources.tar.gz")
        self._terminal_source_captured = True
        return copy.deepcopy(self._final)

    def standardize_capture(
        self,
        native_capture_manifest: str | Path,
        *,
        codex_rollout_manifest: str | Path | None = None,
    ) -> dict[str, Any]:
        if not self._terminal_source_captured:
            raise RuntimeError(
                "terminal source capture must precede native standardization"
            )
        if self._terminal_captured:
            raise RuntimeError("native capture is already standardized")
        self.standardized_trace = standardize_native_capture(
            native_capture_manifest,
            codex_rollout_manifest=codex_rollout_manifest,
        )
        self._terminal_captured = True
        return copy.deepcopy(self.standardized_trace)

    def capture_terminal(
        self,
        native_capture_manifest: str | Path,
        *,
        post_exit_quiesced: bool,
        codex_rollout_manifest: str | Path | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible combined post-exit capture operation."""

        self.capture_terminal_source(post_exit_quiesced=post_exit_quiesced)
        return self.standardize_capture(
            native_capture_manifest,
            codex_rollout_manifest=codex_rollout_manifest,
        )

    def finish(
        self,
        native_capture_manifest: str | Path | None = None,
        *,
        submitted_patch: str,
        post_exit_quiesced: bool | None = None,
        codex_rollout_manifest: str | Path | None = None,
    ) -> Path:
        if self._finished:
            raise RuntimeError("standardized run recorder is already finished")
        if not self._terminal_captured:
            if native_capture_manifest is None or post_exit_quiesced is None:
                raise RuntimeError(
                    "capture_terminal must run before finish, or finish must receive "
                    "the native capture and quiescence proof"
                )
            self.capture_terminal(
                native_capture_manifest,
                post_exit_quiesced=post_exit_quiesced,
                codex_rollout_manifest=codex_rollout_manifest,
            )
        assert self.standardized_trace is not None
        write_standardized_trace(
            self.standardized_trace,
            self.output_dir / "standardized-trace.json",
        )
        self.bundle_path = create_replay_bundle(
            self.output_dir / "bundle",
            standardized_trace=self.standardized_trace,
            baseline_archive=self._baseline["path"],
            baseline_source_sha256=str(self._baseline["source_archive_sha256"]),
            final_archive=self._final["path"],
            final_source_sha256=str(self._final["source_archive_sha256"]),
            submitted_patch=submitted_patch,
            source_scope=self.source_scope,
            environment=self.environment,
        )
        audit = {
            "format": "code-harness-passive-capture-audit-v1",
            "agent_paused_for_checkpoints": False,
            "native_stream_parsed_during_agent_execution": False,
            "trajectory_standardization_phase": "post_exit",
            "agent_source_reads": "pre_run_and_post_exit_only",
            "post_exit_process_quiescence": True,
            "supplemental_codex_rollout_capture": (
                self.standardized_trace.get("codex_native_rollout") is not None
            ),
            "native_capture_format": self.standardized_trace.get("source", {}).get(
                "capture_format"
            ),
            "native_capture_kind": self.standardized_trace.get("source", {}).get(
                "kind"
            ),
            "baseline_source_archive_sha256": self._baseline["source_archive_sha256"],
            "final_source_archive_sha256": self._final["source_archive_sha256"],
            "standardized_trace_sha256": self.standardized_trace["trace_sha256"],
            "bundle_path": str(self.bundle_path),
        }
        (self.output_dir / "capture-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._finished = True
        return self.bundle_path

    def replay(
        self,
        *,
        policy: ReplayPolicy | None = None,
        final_cost_usd: float = 0.0,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if not self._finished or self.bundle_path is None:
            raise RuntimeError("finish must run before replay")
        spawner = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawner):
            raise TypeError("capture environment has no replay-environment factory")
        replay_policy = policy or ReplayPolicy()
        actions = self.standardized_trace.get("actions") if self.standardized_trace else []
        if replay_policy.reject_asynchronous_actions:
            asynchronous_actions: list[tuple[int, str]] = []
            for action in actions or []:
                if not isinstance(action, Mapping) or action.get("name") != "Bash":
                    continue
                arguments = action.get("arguments")
                command = (
                    arguments.get("command")
                    if isinstance(arguments, Mapping)
                    else None
                )
                reason = (
                    _shell_asynchrony_reason(command)
                    if isinstance(command, str)
                    else None
                )
                if reason:
                    asynchronous_actions.append(
                        (int(action.get("action_index") or 0), reason)
                    )
            if asynchronous_actions:
                rendered = ", ".join(
                    f"action {index} ({reason})"
                    for index, reason in asynchronous_actions
                )
                raise StandardizedReplayError(
                    "strict replay cannot establish closed edit boundaries for "
                    f"asynchronous trajectory actions: {rendered}"
                )
        mutation_upper_bound = sum(
            isinstance(action, Mapping)
            and action.get("replay_kind") in {"edit", "write", "patch", "bash"}
            for action in (actions or [])
        )
        per_mutation_seconds = replay_policy.action_timeout_seconds
        if replay_policy.build_after_each_mutation:
            per_mutation_seconds += replay_policy.build_timeout_seconds
        lifetime_seconds = max(
            600,
            self.capture_timeout_seconds,
            (mutation_upper_bound + 2) * per_mutation_seconds,
        )
        replay_env = spawner(lifetime_seconds=lifetime_seconds)
        try:
            tool_environment = install_replay_tool_bundle(replay_env, self.environment)
            self.replay_result = StandardizedReplayEngine(
                self.bundle_path,
                replay_env=replay_env,
                policy=replay_policy,
                instance_id=self.instance_id,
                final_cost_usd=final_cost_usd,
                checkpoint_dir=checkpoint_dir,
            ).replay()
            self.replay_result["replay_tool_environment"] = tool_environment
            self.replay_result["analysis_artifacts"] = write_edit_build_analysis(
                self.replay_result, self.output_dir / "edit-build-analysis"
            )
            payload = (
                json.dumps(self.replay_result, indent=2, sort_keys=True) + "\n"
            )
            (self.output_dir / "standardized-replay-result.json").write_text(
                payload,
                encoding="utf-8",
            )
            (self.output_dir / "playback.json").write_text(
                payload,
                encoding="utf-8",
            )
            return copy.deepcopy(self.replay_result)
        finally:
            replay_env.cleanup()


__all__ = ["StandardizedRunRecorder"]
