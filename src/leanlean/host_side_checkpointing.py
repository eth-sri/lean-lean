"""Host-side checkpointing of Lean source states.

The model process is never paused or instrumented. A baseline is captured
before launch, then the host reads the container's writable overlay at
completed-action boundaries and on a periodic reconciliation timer. Unchanged
states retain only a small checkpoint observation; changed states retain a
deterministic Lean-source archive and incremental diff for offline measurement/building.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import tarfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from leanlean.model.costs import cost_from_responses
from leanlean.antigravity_usage import apply_recorded_costs
from leanlean.playback import (
    ExternalPlaybackRecorder,
    _archive_source_files,
    _safe_float,
)
from leanlean.standardized_replay import _action_cost_ledger
from leanlean.standardized_trace import (
    _codex_rollout_turn_messages,
    _digest,
    standardize_events,
    write_standardized_trace,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_seconds(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).timestamp()
    except (TypeError, ValueError):
        return None


def _normalized_files(files: dict[str, bytes]) -> dict[str, bytes]:
    return {
        name.removeprefix("./"): payload
        for name, payload in files.items()
    }


def _write_source_archive(
    destination: Path, files: dict[str, bytes]
) -> dict[str, Any]:
    """Write one deterministic archive and hash its canonical tar bytes."""

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            out.addfile(info, io.BytesIO(payload))
    canonical = tar_buffer.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            compressed.write(canonical)
    return {
        "source_archive_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_archive_bytes": len(canonical),
        "archive_bytes": destination.stat().st_size,
        "lean_file_count": sum(name.endswith(".lean") for name in files),
    }


def _lean_tree_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(files.items()):
        if not name.endswith(".lean"):
            continue
        encoded = name.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass
class HostSideCheckpointRecorder(ExternalPlaybackRecorder):
    """Capture deduplicated Lean states without entering the live container."""

    reconciliation_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reconciliation_interval_seconds <= 0:
            raise ValueError("reconciliation_interval_seconds must be positive")
        self._checkpoint_lock = threading.RLock()
        self._checkpoint_stop = threading.Event()
        self._checkpoint_thread: threading.Thread | None = None
        self._baseline_files: dict[str, bytes] | None = None
        self._upper_testbed: Path | None = None
        self._preflight_files: dict[str, bytes] | None = None
        self._last_observed_lean_digest = ""
        self._checkpoint_events: list[dict[str, Any]] = []
        self._checkpoint_capture_failures: list[dict[str, Any]] = []
        self._codex_rollout_costs_exact = False
        self._native_events: list[dict[str, Any]] = []
        self._native_stdout_path: Path | None = None
        self._native_stdout_sha256 = hashlib.sha256()
        self._native_stdout_bytes = 0
        self._standardized_trace_path: Path | None = None
        self._live_standardized_trace: dict[str, Any] = {}
        self._live_standardization_failures: list[dict[str, Any]] = []
        self._provider_exit_recorded = False
        self._provider_timed_out = False
        self._provider_returncode: int | None = None
        self._provider_termination_at: str | None = None
        self._live_trace_finalized = False
        self._live_trace_postprocessed = False

    @staticmethod
    def _read_stable_file(path: Path) -> bytes:
        last_error: BaseException | None = None
        for _ in range(4):
            try:
                before = path.stat()
                payload = path.read_bytes()
                after = path.stat()
                if (
                    before.st_ino == after.st_ino
                    and before.st_size == after.st_size
                    and before.st_mtime_ns == after.st_mtime_ns
                ):
                    return payload
            except (FileNotFoundError, OSError) as exc:
                last_error = exc
            threading.Event().wait(0.01)
        raise RuntimeError(f"Could not stably read overlay file {path}") from last_error

    def _overlay_lean_state(self) -> dict[str, bytes]:
        if self._baseline_files is None or self._upper_testbed is None:
            raise RuntimeError("host-side checkpointing has no prepared baseline")
        state = copy.deepcopy(self._baseline_files)
        root = self._upper_testbed
        if not root.exists():
            return state

        for directory, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = sorted(
                name for name in dirnames if name not in {".git", ".lake"}
            )
            filenames = sorted(filenames)
            current = Path(directory)
            relative_dir = current.relative_to(root)
            prefix = "" if str(relative_dir) == "." else relative_dir.as_posix() + "/"

            if ".wh..wh..opq" in filenames:
                state = {
                    name: payload
                    for name, payload in state.items()
                    if not (name.endswith(".lean") and name.startswith(prefix))
                }

            for filename in filenames:
                if filename.startswith(".wh."):
                    if filename == ".wh..wh..opq":
                        continue
                    target = prefix + filename.removeprefix(".wh.")
                    target_prefix = target.rstrip("/") + "/"
                    state = {
                        name: payload
                        for name, payload in state.items()
                        if not (
                            name.endswith(".lean")
                            and (name == target or name.startswith(target_prefix))
                        )
                    }
                    continue
                path = current / filename
                if path.is_char_device():
                    # Rootless fuse-overlayfs represents a deletion as a 0:0
                    # character device at the deleted path rather than as a
                    # .wh.<name> entry.  Remove both files and directory trees
                    # from the reconstructed baseline state.
                    target = prefix + filename
                    target_prefix = target.rstrip("/") + "/"
                    state = {
                        name: payload
                        for name, payload in state.items()
                        if not (
                            name.endswith(".lean")
                            and (name == target or name.startswith(target_prefix))
                        )
                    }
                    continue
                if not filename.endswith(".lean"):
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                state[prefix + filename] = self._read_stable_file(path)
        return state

    def _capture_archive_candidate(self) -> tuple[Path, dict[str, Any], float]:
        if self.snapshot_dir is None:
            raise RuntimeError("host-side checkpointing requires an artifact directory")
        directory = Path(self.snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / (
            f".candidate-{id(self)}-{self.event_count}-{time.time_ns()}.tar.gz"
        )
        started = time.monotonic()

        if self._baseline_files is None:
            raw_baseline = directory / f".baseline-raw-{id(self)}.tar.gz"
            try:
                self.env.export_source_archive(
                    raw_baseline,
                    self._archive_paths(),
                    timeout=self.capture_timeout_seconds,
                )
                # Static non-Lean repository files are retained so every
                # captured state remains independently buildable. Runtime
                # observations below update only .lean content, so logs and
                # other scratch files never create checkpoints.
                self._baseline_files = _normalized_files(
                    _archive_source_files(raw_baseline)
                )
            finally:
                raw_baseline.unlink(missing_ok=True)
            upper_dir = self.env.container_upper_dir()
            self._upper_testbed = Path(upper_dir) / "testbed"
            files = copy.deepcopy(self._baseline_files)
        elif self._preflight_files is not None:
            files = self._preflight_files
            self._preflight_files = None
        else:
            files = self._overlay_lean_state()

        metadata = _write_source_archive(candidate, files)
        metadata.update(
            {
                "capture_backend": "host_container_upperdir",
                "runtime_container_process_created": False,
            }
        )
        return candidate, metadata, time.monotonic() - started

    def prepare(self) -> None:
        with self._checkpoint_lock:
            if self.snapshot_dir is None:
                raise RuntimeError(
                    "host-side checkpointing requires an artifact directory"
                )
            artifact_dir = Path(self.snapshot_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            self._native_stdout_path = artifact_dir / "native.stdout.jsonl"
            self._native_stdout_path.write_bytes(b"")
            self._standardized_trace_path = artifact_dir / "standardized-trace.json"
            super().prepare()
            self._last_observed_lean_digest = _lean_tree_digest(
                self._baseline_files or {}
            )
            self._persist_manifest(
                self._playback_result(
                    cost_basis="live provider usage; final cost pending"
                )
            )
            self._checkpoint_thread = threading.Thread(
                target=self._reconciliation_loop,
                name=f"lean-checkpoint-{self.instance_id or id(self)}",
                daemon=True,
            )
            self._checkpoint_thread.start()

    def _record_live_standardization_failure(
        self, phase: str, error: BaseException
    ) -> None:
        record = {
            "wall_timestamp": _utc_now(),
            "phase": phase,
            "error": str(error),
        }
        if not self._live_standardization_failures or (
            self._live_standardization_failures[-1].get("phase"),
            self._live_standardization_failures[-1].get("error"),
        ) != (phase, str(error)):
            self._live_standardization_failures.append(record)

    def _append_native_line(self, line: str) -> dict[str, Any] | None:
        try:
            payload = line.encode("utf-8", errors="surrogateescape")
            if not payload.endswith(b"\n"):
                payload += b"\n"
            if self._native_stdout_path is not None:
                with self._native_stdout_path.open("ab") as stream:
                    stream.write(payload)
            self._native_stdout_sha256.update(payload)
            self._native_stdout_bytes += len(payload)
        except Exception as exc:
            self._record_live_standardization_failure("raw_stream_append", exc)

        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            return None
        if not isinstance(event, dict):
            return None
        self._native_events.append(copy.deepcopy(event))
        return event

    @staticmethod
    def _checkpoint_trace_record(point: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "edit_index",
            "elapsed_seconds",
            "cost_usd",
            "cost_boundary_exact",
            "cost_estimated",
            "cost_join_basis",
            "native_event_sequence",
            "native_usage",
            "usage_by_model",
            "request_start_time_fallback_count",
            "untimed_request_count",
            "untimed_request_cost_usd",
            "recorded_cost_lower_bound_usd",
            "recorded_cost_upper_bound_usd",
            "kind",
            "action_index",
            "tool_use_id",
            "agent_id",
            "changed_files",
            "source_archive_sha256",
            "snapshot_path",
            "incremental_diff_path",
            "capture_seconds",
            "lean_tokens",
            "lean_tokens_saved",
            "lean_token_compression_pct",
            "build",
            "matches_submission",
        )
        return {
            key: copy.deepcopy(point.get(key))
            for key in keys
            if point.get(key) is not None
        }

    def _persist_standardized_trace(self) -> None:
        if self._standardized_trace_path is None:
            return
        try:
            source: dict[str, Any] = {
                "kind": "host-native-stream",
                "native_stream_exact": True,
                "native_event_payloads_exact": True,
                "normalization_phase": "post_exit",
                "agent_observable": False,
                "host_only": True,
                "raw_stream_path": self._archive_reference(
                    self._native_stdout_path.name
                )
                if self._native_stdout_path is not None
                else None,
                "raw_stream_sha256": self._native_stdout_sha256.hexdigest(),
                "raw_stream_bytes": self._native_stdout_bytes,
                "provider_hint": self.provider,
                "provider_stream_terminated": self._provider_timed_out,
                "provider_terminated_pending_tools": self._provider_timed_out,
                "pending_tool_terminal_status": "timed_out",
                "pending_tool_termination_reason": "evaluation_timeout",
                "provider_termination_at": self._provider_termination_at,
                "timed_out": self._provider_timed_out,
                "returncode": self._provider_returncode,
            }
            if self._provider_timed_out:
                source["pending_tool_count_evidence"] = sum(
                    (
                        len(self._claude_trace_builder.pending_tool_uses),
                        len(self._kimi_trace_builder.pending_tool_uses),
                    )
                )
            trace = standardize_events(
                "auto",
                copy.deepcopy(self._native_events),
                source=source,
                standardization_phase="post_exit",
            )
            detected_provider = trace.get("provider")
            trace["source"]["native_event_semantics_exact"] = (
                detected_provider != "generic-jsonl"
            )
            trace["source"]["forwards_subagent_stream"] = (
                True if detected_provider == "claude-code" else None
            )

            observations_by_tool = {
                str(event.get("tool_use_id")): event
                for event in self._checkpoint_events
                if event.get("tool_use_id")
            }
            observations_by_action = {
                int(event["action_index"]): event
                for event in self._checkpoint_events
                if event.get("action_index") is not None
            }
            for action in trace.get("actions") or []:
                observation = observations_by_tool.get(
                    str(action.get("tool_call_id") or "")
                )
                if observation is None and action.get("action_index") is not None:
                    observation = observations_by_action.get(
                        int(action["action_index"])
                    )
                if observation is not None:
                    action["source_observation"] = copy.deepcopy(observation)

            cost_evidence = getattr(self, "_gateway_cost_evidence", None) or getattr(self, "_codex_checkpoint_cost_evidence", None)
            if self._live_trace_finalized and cost_evidence:
                boundaries = {}
                for action in trace.get("actions") or []:
                    observation = action.get("source_observation") or {}
                    cost = observation.get("cumulative_cost_usd")
                    boundaries[str(action["action_index"])] = {
                        "cost_usd": cost, "cumulative_cost_usd": cost,
                        "cost_boundary_exact": False, "cost_estimated": cost is not None,
                        "cost_granularity": "request_timestamp", "turn_cost_usd": None,
                        "request_start_time_fallback_count": observation.get("request_start_time_fallback_count", 0),
                    }
                trace["recorded_cost_ledger"] = {
                    "format": "verified-replay-cost-ledger-v1", "boundaries": boundaries,
                    "summary": {
                        "captured_completed_turn_cost_usd": cost_evidence.get("captured_completed_turn_cost_usd"),
                        "captured_assistant_turn_count": cost_evidence.get("call_count", cost_evidence.get("request_count", 0)),
                        "all_captured_turn_costs_exact": False, "cost_granularity": "request_timestamp",
                        "cost_allocation": "recorded request usage; completion preferred, start fallback; no proration",
                        "cost_evidence": copy.deepcopy(cost_evidence),
                        "authoritative_final_cost_usd": self._authoritative_final_cost_usd,
                    },
                }
            trace["source_observations"] = copy.deepcopy(self._checkpoint_events)
            trace["source_checkpoints"] = [
                self._checkpoint_trace_record(point) for point in self.points
            ]
            trace["live_capture"] = {
                "provider_exit_recorded": self._provider_exit_recorded,
                "capture_finalized": self._live_trace_finalized,
                "offline_postprocessing_complete": self._live_trace_postprocessed,
                "native_event_count": len(self._native_events),
                "completed_action_count": self.event_count,
                "source_observation_count": len(self._checkpoint_events),
                "checkpoint_record_count": len(self.points),
                "unique_lean_state_count": len(
                    {
                        point["source_archive_sha256"]
                        for point in self.points
                        if point.get("source_archive_sha256")
                    }
                ),
                "edit_count": self.edit_count,
                "capture_failure_count": len(self._checkpoint_capture_failures),
                "standardization_failure_count": len(
                    self._live_standardization_failures
                ),
            }
            if isinstance(self._result, dict) and isinstance(
                self._result.get("submission_verification"), dict
            ):
                endpoint_exact = (
                    self._result.get("final_snapshot_matches_submission") is True
                )
                trace.setdefault("quality", {})[
                    "filesystem_reconstruction_exact"
                ] = endpoint_exact
                trace["live_capture"]["final_snapshot_matches_submission"] = (
                    endpoint_exact
                )
                trace["live_capture"]["submitted_endpoint_captured"] = (
                    self._result.get("submitted_endpoint_captured") is True
                )
                verification = self._result.get("submission_verification")
                if isinstance(verification, dict):
                    trace["endpoint_verification"] = copy.deepcopy(verification)
            trace.pop("trace_sha256", None)
            trace["trace_sha256"] = _digest(trace)
            write_standardized_trace(trace, self._standardized_trace_path)
            self._live_standardized_trace = trace
        except Exception as exc:
            # Raw JSONL and source checkpoints remain authoritative if optional
            # standardized-trace rendering fails during host finalization.
            self._record_live_standardization_failure("trace_persist", exc)

    def mark_provider_exit(self, *, timed_out: bool, returncode: int | None) -> None:
        with self._checkpoint_lock:
            self._provider_exit_recorded = True
            self._provider_timed_out = bool(timed_out)
            self._provider_returncode = returncode
            self._provider_termination_at = _utc_now() if timed_out else None
            self._live_trace_finalized = True
            self._persist_standardized_trace()
            self._persist_manifest(
                self._playback_result(
                    cost_basis="live provider usage; final cost pending"
                )
            )

    def _reconciliation_loop(self) -> None:
        while not self._checkpoint_stop.wait(self.reconciliation_interval_seconds):
            self._capture(
                files=[],
                kind="periodic_reconciliation",
                action_index=None,
                tool_use_id=None,
                agent_id=None,
            )

    def close(self) -> None:
        self._checkpoint_stop.set()
        thread = self._checkpoint_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.reconciliation_interval_seconds + 1.0))
        self._checkpoint_thread = None

    def _capture(
        self,
        *,
        files: list[str],
        kind: str,
        force: bool = False,
        pause_agent: bool = False,
        action_index: int | None = None,
        tool_use_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        del pause_agent
        with self._checkpoint_lock:
            observer_only = (
                kind in {"periodic_reconciliation", "host_stream_boundary"}
                and action_index is None
            )
            preflight_digest = ""
            if not force and self._baseline_files is not None:
                try:
                    preflight_files = self._overlay_lean_state()
                    preflight_digest = _lean_tree_digest(preflight_files)
                    if (
                        observer_only
                        and preflight_digest == self._last_observed_lean_digest
                    ):
                        return
                    self._preflight_files = preflight_files
                except Exception:
                    # The authoritative capture below retries stable reads and
                    # records any persistent failure with boundary metadata.
                    self._preflight_files = None
            boundary_wall_timestamp = _utc_now()
            boundary_elapsed_seconds = round(
                time.monotonic() - self.started_at, 3
            )
            before_edit_count = self.edit_count
            before_point_count = len(self.points)
            before_event_count = self.event_count
            error = ""
            try:
                super()._capture(
                    files=files,
                    kind=kind,
                    force=force,
                    pause_agent=False,
                    action_index=action_index,
                    tool_use_id=tool_use_id,
                    agent_id=agent_id,
                )
            except Exception as exc:
                error = str(exc)
                failure = {
                    "wall_timestamp": _utc_now(),
                    "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                    "kind": kind,
                    "action_index": action_index,
                    "tool_use_id": tool_use_id,
                    "agent_id": agent_id,
                    "error": error,
                }
                self._checkpoint_capture_failures.append(failure)
                if force:
                    raise
            finally:
                self._preflight_files = None

            if preflight_digest and not error:
                self._last_observed_lean_digest = preflight_digest
            if observer_only:
                # ExternalPlaybackRecorder counts every non-forced capture as
                # an event. Host observations are not completed provider actions,
                # so keep the action counter clean.
                self.event_count = before_event_count

            source_changed = self.edit_count > before_edit_count
            checkpoint = (
                self.points[-1]
                if source_changed and len(self.points) > before_point_count
                else None
            )
            if observer_only and not source_changed and not error:
                return
            self._checkpoint_events.append(
                {
                    "sequence": len(self._checkpoint_events) + 1,
                    "native_event_sequence": self._antigravity_usage.sequence if self._detected_stream_provider == "antigravity-cli" else None,
                    "wall_timestamp": boundary_wall_timestamp,
                    "elapsed_seconds": boundary_elapsed_seconds,
                    "kind": kind,
                    "action_index": action_index,
                    "tool_use_id": tool_use_id,
                    "agent_id": agent_id,
                    "files": sorted(set(files)),
                    "source_changed": source_changed,
                    "edit_index": checkpoint.get("edit_index") if checkpoint else None,
                    "source_archive_sha256": (
                        checkpoint.get("source_archive_sha256")
                        if checkpoint
                        else self._last_seen_archive_sha256
                    ),
                    "lean_tokens": (
                        checkpoint.get("lean_tokens") if checkpoint else None
                    ),
                    "lean_tokens_saved": (
                        checkpoint.get("lean_tokens_saved") if checkpoint else None
                    ),
                    "cumulative_cost_usd": round(
                        cost_from_responses(self._turn_responses), 8
                    ),
                    "cost_boundary_exact": (
                        self._detected_stream_provider == "claude-code"
                    ),
                    "capture_error": error or None,
                }
            )
            self._persist_manifest(
                self._playback_result(
                    cost_basis="live provider usage; final cost pending"
                )
            )

    def ingest_line(self, line: str) -> None:
        with self._checkpoint_lock:
            event = self._append_native_line(line)
            before_observations = len(self._checkpoint_events)
            super().ingest_line(line)
            if (
                event is not None
                and len(self._checkpoint_events) == before_observations
            ):
                self._capture(
                    files=[],
                    kind="host_stream_boundary",
                    action_index=None,
                    tool_use_id=None,
                    agent_id=None,
                )

    def _read_tokens(self, tree_oid: str = "", *, env: Any | None = None) -> int:
        """Skip legacy Python token counting; checkpoint validation is source/build based."""

        del tree_oid, env
        return 0

    def attach_codex_rollout_costs(self, rollout_path: str | Path) -> None:
        """Join exact completed-turn costs to Codex action checkpoints."""

        payload = gzip.decompress(Path(rollout_path).read_bytes())
        events = [
            json.loads(line)
            for line in payload.splitlines()
            if line.strip()
        ]
        if not events:
            raise RuntimeError("Codex rollout is empty")
        header = events[0].get("payload") if isinstance(events[0], dict) else None
        thread_id = str(
            (header or {}).get("id") or (header or {}).get("session_id") or "root"
        )
        messages = _codex_rollout_turn_messages(events, thread_id)
        native_to_call: dict[str, str] = {}
        native_call_starts: list[tuple[float, str]] = []
        for event in events:
            item = event.get("payload") if isinstance(event, dict) else None
            if not isinstance(item, dict):
                continue
            native_id = str(item.get("id") or "")
            call_id = str(item.get("call_id") or "")
            if native_id and call_id:
                native_to_call[native_id] = call_id
            if item.get("type") == "custom_tool_call" and call_id:
                timestamp = _timestamp_seconds(event.get("timestamp"))
                if timestamp is not None:
                    native_call_starts.append((timestamp, call_id))
        native_call_starts.sort()

        actions = []
        tool_events: list[dict[str, Any]] = []
        for event in self._checkpoint_events:
            action_index = event.get("action_index")
            tool_use_id = str(event.get("tool_use_id") or "")
            kind = str(event.get("kind") or "")
            if (
                action_index is None
                or not tool_use_id
                or kind in {"agent_message", "reasoning", "error", "unknown"}
            ):
                continue
            tool_events.append(event)
            cost_tool_call_id = native_to_call.get(tool_use_id)
            if not cost_tool_call_id:
                boundary_timestamp = _timestamp_seconds(
                    event.get("wall_timestamp")
                )
                if boundary_timestamp is not None:
                    candidates = [
                        call_id
                        for started_at, call_id in native_call_starts
                        if started_at <= boundary_timestamp
                    ]
                    if candidates:
                        cost_tool_call_id = candidates[-1]
                        event["cost_join_basis"] = (
                            "latest native code-mode call started before action "
                            "boundary"
                        )
            actions.append(
                {
                    "action_index": int(action_index),
                    "agent_id": thread_id,
                    "tool_call_id": cost_tool_call_id or tool_use_id,
                    "started_sequence": 0,
                }
            )
        boundaries, summary = _action_cost_ledger(
            {"messages": messages, "actions": actions}
        )
        if not summary.get("all_captured_turn_costs_exact"):
            raise RuntimeError("Codex rollout has inexact completed-turn usage")

        last_exact_cost = 0.0
        edit_costs: dict[int, float] = {}
        exact_action_count = 0
        for event in self._checkpoint_events:
            action_index = event.get("action_index")
            boundary = boundaries.get(int(action_index)) if action_index else None
            if boundary and boundary.get("cost_boundary_exact") is True:
                last_exact_cost = float(boundary["cost_usd"])
                event["turn_cost_usd"] = boundary.get("turn_cost_usd")
                event["cost_boundary_exact"] = True
                exact_action_count += 1
            event["cumulative_cost_usd"] = round(last_exact_cost, 8)
            edit_index = event.get("edit_index")
            if edit_index is not None:
                edit_costs[int(edit_index)] = last_exact_cost

        for point in self.points:
            edit_index = int(point.get("edit_index") or 0)
            if edit_index in edit_costs:
                point["cost_usd"] = round(edit_costs[edit_index], 8)
                point["cost_boundary_exact"] = True

        self._turn_responses = [
            {"model": message.get("model"), "usage": message.get("usage")}
            for message in messages
            if isinstance(message.get("usage"), dict)
        ]
        self._codex_rollout_costs_exact = bool(messages) and exact_action_count == len(
            tool_events
        )
        from leanlean.recorded_costs import prepare_requests, request_ledger, apply_request_ledger
        records = [{"event": "success", "ts": _timestamp_seconds(message.get("trace_timestamp")),
                    "litellm_call_id": message["id"],
                    "agent_context": {"accounting_scope": thread_id},
                    "response": {"model": message["model"], "usage": message["usage"]}}
                   for message in messages]
        ledger = request_ledger(prepare_requests(records))
        data = {"points": self.points, "checkpoint_observations": self._checkpoint_events}
        apply_request_ledger(data, ledger, final_complete=False,
                             source_evidence={"source": "codex_native_token_count"})
        self._codex_rollout_costs_exact = True  # Usage counters; source boundaries remain estimates.
        self._codex_checkpoint_cost_evidence = {
            "source": "codex_native_token_count", "request_count": ledger["request_count"],
            "untimed_request_count": data["untimed_request_count"],
            "captured_completed_turn_cost_usd": ledger["final_cost_usd"],
        }

    def attach_gateway_costs(self, records, *, scope_id: str) -> None:
        """Retain scoped journal evidence for the final timestamp join."""
        if not scope_id:
            raise ValueError("gateway checkpoint accounting requires an invocation scope")
        self._gateway_cost_records = list(records)
        self._gateway_cost_scope = scope_id

    def finish(self, authoritative_responses: list[dict[str, Any]]) -> dict[str, Any]:
        self.close()
        with self._checkpoint_lock:
            result = super().finish(authoritative_responses)
            final_cost = _safe_float(result.get("final_cost_usd"))
            final_elapsed = max(
                (
                    _safe_float(event.get("elapsed_seconds"))
                    for event in self._checkpoint_events
                ),
                default=0.0,
            )
            final_elapsed = max(final_elapsed, 1e-9)
            if (
                self._detected_stream_provider not in {"claude-code", "antigravity-cli"}
                and not self._codex_rollout_costs_exact
            ):
                for event in self._checkpoint_events:
                    event["cumulative_cost_usd"] = None
                    event["cost_boundary_exact"] = False
            opus_terminal = any(
                "opus" in str(response.get("model", ""))
                and (response.get("native_cost") or {}).get("basis") == "claude_code_list_price"
                for response in authoritative_responses if isinstance(response, dict)
            )
            if hasattr(self, "_gateway_cost_records"):
                from leanlean.model.litellm_wrapper.checkpoint_costs import join_checkpoint_costs
                try:
                    self._gateway_cost_evidence = join_checkpoint_costs(
                        self.points, self._checkpoint_events, self._gateway_cost_records,
                        scope_id=self._gateway_cost_scope, expected_total=final_cost,
                        terminal_authoritative=opus_terminal,
                    )
                except ValueError as error:
                    from leanlean.recorded_costs import invalidate_unmeasured_costs
                    self._checkpoint_cost_join_error = str(error)
                    invalidate_unmeasured_costs(
                        {"points": self.points, "checkpoint_observations": self._checkpoint_events}, str(error)
                    )
                    self._gateway_cost_evidence = {
                        "source": "invocation_scoped_gateway_journal", "error": str(error),
                        "timestamp_join_complete": False, "cost_prorated": False,
                    }
                else:
                    self._captured_completed_turn_cost_usd = self._gateway_cost_evidence["captured_completed_turn_cost_usd"]
                self._cost_prorated = False
                result["cost_basis"] = "invocation-scoped gateway usage joined by request timestamp; no time proration"
            self._result = self._playback_result(
                cost_basis=str(result.get("cost_basis") or ""),
                final_cost_usd=final_cost,
            )
            self._live_trace_finalized = True
            self._persist_standardized_trace()
            self._result = self._playback_result(
                cost_basis=str(result.get("cost_basis") or ""),
                final_cost_usd=final_cost,
            )
            if getattr(self, "_checkpoint_cost_join_error", None):
                self._result.update(
                    checkpoint_cost_join_error=self._checkpoint_cost_join_error,
                    cost_timeline_complete=False, every_edit_checkpoint_costed=False,
                    final_cost_complete=opus_terminal,
                    authoritative_final_cost_usd=final_cost if opus_terminal else None,
                )
            if opus_terminal:
                from leanlean.recorded_costs import use_authoritative_terminal_cost
                use_authoritative_terminal_cost(self._result, final_cost)
                self._persist_standardized_trace()
            if self._detected_stream_provider == "antigravity-cli":
                apply_recorded_costs(self._result, self._antigravity_usage)
                self._persist_standardized_trace()
            self._persist_manifest(self._result)
            return self._result

    def _verify_external_submission(
        self, replay_env: Any, patch: str
    ) -> dict[str, Any]:
        full_scope = super()._verify_external_submission(replay_env, patch)
        if full_scope.get("error") and "could not be applied" in str(
            full_scope.get("error")
        ).lower():
            return full_scope

        candidate = Path(self.snapshot_dir) / f".lean-verification-{id(self)}.tar.gz"
        try:
            replay_env.export_source_archive(
                candidate,
                self._archive_paths(),
                timeout=self.capture_timeout_seconds,
            )
            submitted_files = _archive_source_files(candidate)
        finally:
            candidate.unlink(missing_ok=True)
        baseline_files = _archive_source_files(self._archive_file(self.points[0]))
        captured_files = _archive_source_files(self._archive_file(self.points[-1]))
        evaluated_paths = sorted(
            name
            for name in set(baseline_files) | set(submitted_files)
            if name.endswith(".lean")
        )
        submitted_scope = {
            name: submitted_files[name]
            for name in evaluated_paths
            if name in submitted_files
        }
        captured_scope = {
            name: captured_files[name]
            for name in evaluated_paths
            if name in captured_files
        }
        submitted_digest = _lean_tree_digest(submitted_scope)
        captured_digest = _lean_tree_digest(captured_scope)
        changed_paths = sorted(
            name
            for name in evaluated_paths
            if submitted_files.get(name) != captured_files.get(name)
        )
        ignored_paths = sorted(
            name
            for name in captured_files
            if name.endswith(".lean") and name not in evaluated_paths
        )
        return {
            "checked": True,
            "matches": submitted_digest == captured_digest,
            "basis": "exact .lean path-and-byte equality over evaluated patch scope",
            "captured_lean_tree_sha256": captured_digest,
            "submitted_lean_tree_sha256": submitted_digest,
            "mismatched_lean_paths": changed_paths,
            "ignored_unsubmitted_lean_paths": ignored_paths,
            "full_scoped_source_verification": full_scope,
        }

    def replay_submission(self, patch: str) -> dict[str, Any]:
        """Archive the exact submitted endpoint without replaying checkpoints.

        Live evaluation owns generation and observation only. It reconstructs
        the submitted patch once from the immutable baseline so the final
        endpoint does not depend on monitor fidelity. Source metrics, builds,
        and protected-signature checks are deferred to postprocessing.
        """

        spawner = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawner):
            raise RuntimeError(
                "Host-side endpoint capture requires a replay environment factory"
            )
        if self.snapshot_dir is None:
            raise RuntimeError("Host-side endpoint capture requires snapshot storage")

        replay_env = None
        verification: dict[str, Any]
        submitted_point: dict[str, Any] | None = None
        try:
            replay_env = spawner(
                lifetime_seconds=max(600, self.capture_timeout_seconds * 3)
            )
            verification = self._verify_external_submission(replay_env, patch)
            if not verification.get("submitted_lean_tree_sha256"):
                raise RuntimeError(
                    str(verification.get("error") or "submission was not reconstructed")
                )

            candidate = Path(self.snapshot_dir) / (
                f".submitted-endpoint-{id(self)}.tar.gz"
            )
            try:
                replay_env.export_source_archive(
                    candidate,
                    self._archive_paths(),
                    timeout=self.capture_timeout_seconds,
                )
                submitted_files = _normalized_files(
                    _archive_source_files(candidate)
                )
            finally:
                candidate.unlink(missing_ok=True)

            destination = Path(self.snapshot_dir) / "submitted.sources.tar.gz"
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            metadata = _write_source_archive(temporary, submitted_files)
            if destination.exists():
                existing = _archive_source_files(destination)
                if _normalized_files(existing) != submitted_files:
                    raise RuntimeError(
                        "refusing to overwrite a different submitted endpoint: "
                        f"{destination}"
                    )
                temporary.unlink(missing_ok=True)
                metadata = {
                    **metadata,
                    "archive_bytes": destination.stat().st_size,
                }
            else:
                os.replace(temporary, destination)

            monitored_point = self.points[-1]
            monitored_files = {
                name: payload
                for name, payload in _normalized_files(
                    _archive_source_files(self._archive_file(monitored_point))
                ).items()
                if name.endswith(".lean")
            }
            monitored_point["endpoint_role"] = "monitored_final_state"
            monitored_point["matches_submission"] = verification.get("matches") is True
            changed_files = sorted(
                name
                for name in set(monitored_files) | set(submitted_files)
                if name.endswith(".lean")
                if monitored_files.get(name) != submitted_files.get(name)
            )
            payload = destination.read_bytes()
            submitted_point = {
                "kind": "final",
                "endpoint_role": "submitted_patch",
                "edit_index": self.edit_count,
                "elapsed_seconds": monitored_point.get("elapsed_seconds"),
                "cost_usd": monitored_point.get("cost_usd"),
                "cost_boundary_exact": True,
                "exact": True,
                "matches_submission": True,
                "changed_files_from_monitored_final": changed_files,
                "submission_patch_sha256": hashlib.sha256(
                    patch.encode("utf-8", errors="surrogateescape")
                ).hexdigest(),
                "file_sha256": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in sorted(submitted_files.items())
                },
                "files": sorted(submitted_files),
                "snapshot_path": self._archive_reference(destination.name),
                "snapshot_format": "deterministic-source-tar-gzip-v1",
                "snapshot_artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "snapshot_artifact_bytes": len(payload),
                "snapshot_sha256": metadata["source_archive_sha256"],
                "snapshot_digest_basis": "canonical uncompressed tar stream",
                "snapshot_bytes": metadata["archive_bytes"],
                "source_archive_bytes": metadata["source_archive_bytes"],
                "source_archive_sha256": metadata["source_archive_sha256"],
                "lean_tokens": None,
                "lean_tokens_saved": None,
                "lean_token_compression_pct": None,
                "capture_seconds": 0.0,
                "replay_status": "pending_postprocessing",
            }
            self.points.append(submitted_point)
        except Exception as exc:
            verification = {
                "checked": False,
                "matches": False,
                "basis": "exact submitted patch reconstructed from baseline",
                "error": str(exc),
            }
        finally:
            if replay_env is not None:
                replay_env.cleanup()

        prior = self._result or {}
        result = self._playback_result(
            cost_basis=str(prior.get("cost_basis") or "authoritative final cost"),
            final_cost_usd=_safe_float(prior.get("final_cost_usd")),
        )
        monitor_matches = verification.get("matches") is True
        result.update(
            {
                "quality": (
                    "endpoint_exact_host_side_checkpointing"
                    if submitted_point is not None and monitor_matches
                    else "endpoint_exact_with_monitor_divergence"
                    if submitted_point is not None
                    else "submitted_endpoint_capture_failed"
                ),
                "endpoint_basis": "exact submitted patch applied to immutable baseline",
                "submitted_endpoint_captured": submitted_point is not None,
                "submitted_endpoint_source_archive_sha256": (
                    submitted_point.get("source_archive_sha256")
                    if submitted_point is not None
                    else None
                ),
                "submission_verification": verification,
                "final_snapshot_matches_submission": monitor_matches,
                "build_basis": "pending postprocess.sh",
                "timeline_fidelity": (
                    "every structured trace event plus periodic reconciliation; "
                    "sub-interval transient states are not certified"
                ),
                "agent_observer_visibility": (
                    "no runtime process, marker, pause, build, or source write "
                    "in task container"
                ),
            }
        )
        self._result = result
        self._persist_standardized_trace()
        result["live_standardized_trace"] = self._live_trace_metadata()
        self._persist_manifest(result)
        return result

    def _live_trace_metadata(self) -> dict[str, Any]:
        trace = self._live_standardized_trace
        return {
            "enabled": self._standardized_trace_path is not None,
            "path": self._archive_reference(self._standardized_trace_path.name)
            if self._standardized_trace_path is not None
            else None,
            "format": trace.get("format"),
            "standardization_phase": trace.get("standardization_phase"),
            "trace_sha256": trace.get("trace_sha256"),
            "native_event_count": len(self._native_events),
            "raw_stream_path": self._archive_reference(
                self._native_stdout_path.name
            )
            if self._native_stdout_path is not None
            else None,
            "raw_stream_sha256": self._native_stdout_sha256.hexdigest(),
            "raw_stream_bytes": self._native_stdout_bytes,
            "agent_observable": False,
            "failure_count": len(self._live_standardization_failures),
            "failures": copy.deepcopy(self._live_standardization_failures),
        }

    def _playback_result(
        self, *, cost_basis: str, final_cost_usd: float | None = None
    ) -> dict[str, Any]:
        result = super()._playback_result(
            cost_basis=cost_basis,
            final_cost_usd=final_cost_usd,
        )
        unchanged = sum(
            event.get("source_changed") is False for event in self._checkpoint_events
        )
        result.update(
            {
                "capture_mode": "host_side_checkpointing",
                "snapshot_basis": (
                    "pre-launch baseline plus host-only container upper-layer "
                    "observations; unique .lean states only"
                ),
                "metric_basis": (
                    "exact source states and offline builds; legacy Lean token "
                    "counting disabled"
                ),
                "agent_paused_for_checkpoints": False,
                "agent_container_source_reads": "pre_launch_baseline_only",
                "agent_container_source_reads_during_execution": False,
                "agent_container_source_writes": False,
                "agent_container_git_writes": False,
                "agent_container_builds": False,
                "runtime_observer_container_process_count": 0,
                "reconciliation_interval_seconds": (
                    self.reconciliation_interval_seconds
                ),
                "checkpoint_observation_count": len(self._checkpoint_events),
                "unchanged_checkpoint_observation_count": unchanged,
                "capture_failure_count": len(self._checkpoint_capture_failures),
                "capture_failures": list(self._checkpoint_capture_failures),
                "checkpoint_observations": list(self._checkpoint_events),
                "timeline_exact": False,
                "timeline_fidelity": (
                    "every structured trace event plus periodic reconciliation"
                ),
                "codex_rollout_costs_exact": self._codex_rollout_costs_exact,
                "gateway_cost_evidence": getattr(self, "_gateway_cost_evidence", None),
                "detected_stream_provider": self._detected_stream_provider,
                "trajectory_standardization_phase": "post_exit",
                "native_stream_parsed_during_agent_execution": True,
                "live_standardized_trace": self._live_trace_metadata(),
            }
        )
        return result
