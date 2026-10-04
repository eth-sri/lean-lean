"""Pause-free capture and post-run replay of Claude Code tool actions."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from leanlean.model.costs import cost_from_responses
from leanlean.playback import (
    ExternalPlaybackRecorder,
    _paths_from_tool,
    _safe_float,
)


_NON_MUTATING_TOOLS = {
    "agent",
    "task",
    "taskoutput",
    "read",
    "glob",
    "grep",
    "webfetch",
    "websearch",
    "sendmessage",
    "taskcreate",
    "taskget",
    "tasklist",
    "taskupdate",
    "todowrite",
    "skill",
    "enterplanmode",
    "exitplanmode",
    "askuserquestion",
}
_REPLAYABLE_MUTATING_TOOLS = {"bash", "edit", "write"}
_BASH_REPLAY_POLICIES = {"all", "source_mutations_only", "edit_tools_only"}
_TERMINAL_RECONCILIATION_POLICIES = {"none", "submitted_patch"}
_SHELL_SOURCE_MUTATORS = {
    "apply_patch",
    "chgrp",
    "chmod",
    "chown",
    "patch",
    "cp",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}
_INTERPRETER_SOURCE_MUTATORS = {"python", "python3", "ruby"}
_GIT_SOURCE_MUTATORS = {
    "am",
    "apply",
    "checkout",
    "cherry-pick",
    "clean",
    "merge",
    "mv",
    "rebase",
    "reset",
    "restore",
    "revert",
    "rm",
}
_SHELL_SEPARATORS = {"&&", "||", ";", "|", "&"}
_KNOWN_UNSUPPORTED_MUTATING_TOOLS = {
    "multiedit",
    "notebookedit",
    "apply_patch",
    "str_replace",
    "str_replace_editor",
}


def _command_segment(tokens: list[str], start: int) -> list[str]:
    segment = []
    for token in tokens[start + 1 :]:
        if token in _SHELL_SEPARATORS:
            break
        segment.append(token)
    return segment


def _perl_source_mutation_reason(segment: list[str]) -> str | None:
    """Classify a Perl invocation without replaying proven read-only metrics."""

    code_arguments: list[str] = []
    for index, argument in enumerate(segment):
        if not argument.startswith("-"):
            continue
        option = argument.lstrip("-")
        if argument.startswith("-M") or argument.startswith("-m"):
            return "perl_module"
        if "i" in option:
            return "perl_in_place"
        if "e" in option.lower() and index + 1 < len(segment):
            code_arguments.append(segment[index + 1])
    if not code_arguments:
        return "perl_script"

    code = "\n".join(code_arguments)
    if re.search(
        r"\b(?:unlink|rename|chmod|chown|mkdir|rmdir|truncate|"
        r"system|exec|eval|syswrite|write)\b|"
        r"->\s*(?:spew|append|remove)\b|"
        r"\bqx\s*[\(\{/]|`",
        code,
        re.IGNORECASE,
    ):
        return "perl_program"
    for match in re.finditer(r"\bopen\b", code):
        tail = code[match.end() : match.end() + 160]
        if re.match(
            r"\s*(?:\(\s*)?(?:my\s+)?(?:\$?[A-Za-z_][A-Za-z0-9_]*)"
            r"\s*,\s*[\"']<[\"']\s*,",
            tail,
        ):
            continue
        return "perl_open"
    return None


def _shell_punctuation_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(
        command,
        posix=False,
        punctuation_chars="&<>|;()",
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _without_shell_heredoc_bodies(command: str) -> str:
    pending: list[tuple[str, bool]] = []
    retained: list[str] = []
    for line in command.splitlines(keepends=True):
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                pending.pop(0)
            retained.append("\n")
            continue
        retained.append(line)
        try:
            tokens = _shell_punctuation_tokens(line)
        except ValueError:
            continue
        for index, token in enumerate(tokens[:-1]):
            if token != "<<":
                continue
            raw_delimiter = tokens[index + 1]
            strip_tabs = raw_delimiter.startswith("-")
            if strip_tabs:
                raw_delimiter = raw_delimiter[1:]
            delimiter = raw_delimiter.strip("\"'")
            if delimiter:
                pending.append((delimiter, strip_tabs))
    return "".join(retained)


def _shell_has_source_output_redirection(command: str) -> bool:
    try:
        tokens = _shell_punctuation_tokens(
            _without_shell_heredoc_bodies(command)
        )
    except ValueError:
        return bool(
            re.search(r"(?<![0-9<>])>{1,2}(?![>&])\s*[^&|;\s]+", command)
        )
    output_operators = {">", ">>", "&>", ">|", "<>"}
    for index, token in enumerate(tokens):
        if token not in output_operators and token != ">&":
            continue
        target = tokens[index + 1] if index + 1 < len(tokens) else ""
        normalized = target.strip("\"'")
        if token == ">&" and (normalized.isdigit() or normalized == "-"):
            continue
        if normalized == "/dev/null" or normalized.startswith("/tmp/"):
            continue
        return True
    return False


def _bash_source_mutation_reason(command: str) -> str | None:
    """Return a conservative static reason that Bash may change source files."""

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        executable = PurePosixPath(token.strip("();{}")).name.lower()
        segment = _command_segment(tokens, index)
        if executable == "sed":
            if any(
                option == "--in-place"
                or (
                    option.startswith("-")
                    and not option.startswith("--")
                    and "i" in option[1:]
                )
                for option in segment
            ):
                return "sed_in_place"
        if executable == "perl":
            perl_reason = _perl_source_mutation_reason(segment)
            if perl_reason is not None:
                return perl_reason
        if executable == "git":
            subcommand = next(
                (
                    item.lower()
                    for item in segment
                    if item and not item.startswith("-")
                ),
                "",
            )
            if subcommand in _GIT_SOURCE_MUTATORS:
                return f"git_{subcommand}"
        if executable in _SHELL_SOURCE_MUTATORS:
            if executable == "tee":
                targets = [
                    argument
                    for argument in segment
                    if argument and not argument.startswith("-")
                ]
                if targets and all(
                    target == "/dev/null" or target.startswith("/tmp/")
                    for target in targets
                ):
                    continue
            return executable
        if executable in _INTERPRETER_SOURCE_MUTATORS:
            return executable
        if executable in {"bash", "sh", "zsh"}:
            command_index = next(
                (
                    offset
                    for offset, argument in enumerate(segment)
                    if argument.startswith("-") and "c" in argument[1:]
                ),
                None,
            )
            if command_index is not None:
                if command_index + 1 >= len(segment):
                    return f"{executable}_script"
                return _bash_source_mutation_reason(segment[command_index + 1])
            if any(argument and not argument.startswith("-") for argument in segment):
                return f"{executable}_script_file"
    if _shell_has_source_output_redirection(command):
        return "stdout_redirection"
    if re.search(r"\bfind\b[^\n;&|]*(?:-delete|-exec\s+[^\n;&|]+)", command):
        return "find_mutation"
    return None


@dataclass
class ClaudeActionPlaybackRecorder(ExternalPlaybackRecorder):
    """Capture no live checkpoints and reconstruct Claude actions afterward.

    The agent container is read exactly twice: once before Claude starts and
    once after its process has exited. Native stdout is buffered verbatim by the
    process runner; this recorder receives, persists, and parses it only after
    terminal source capture. Tool actions are then replayed in a disposable
    network-disabled container.
    """

    replay_action_timeout_seconds: int = 600
    bash_replay_policy: str = "all"
    terminal_reconciliation: str = "none"
    post_exit_quiesced: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bash_replay_policy not in _BASH_REPLAY_POLICIES:
            raise ValueError(
                "bash_replay_policy must be one of "
                + ", ".join(sorted(_BASH_REPLAY_POLICIES))
            )
        if self.terminal_reconciliation not in _TERMINAL_RECONCILIATION_POLICIES:
            raise ValueError(
                "terminal_reconciliation must be one of "
                + ", ".join(sorted(_TERMINAL_RECONCILIATION_POLICIES))
            )
        self._external_quality = "passive_actions_pending_replay"
        self._actions: list[dict[str, Any]] = []
        self._action_sequence = 0
        self._final_archive_metadata: dict[str, Any] = {}
        self._action_endpoint_metadata: dict[str, Any] = {}
        self._submission_endpoint_metadata: dict[str, Any] = {}
        self._raw_stdout_metadata: dict[str, Any] = {}
        self._replay_divergence_count = 0
        self._replayed_bash_action_count = 0
        self._skipped_bash_action_count = 0

    def _action_reference(self, filename: str) -> str:
        if self.snapshot_reference_prefix:
            return f"{self.snapshot_reference_prefix.rstrip('/')}/{filename}"
        return filename

    def _action_file(self, record: dict[str, Any]) -> Path:
        if self.snapshot_dir is None:
            raise RuntimeError("Passive Claude replay requires an artifact directory")
        return Path(self.snapshot_dir) / Path(str(record["action_path"])).name

    def _read_action(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = gzip.decompress(self._action_file(record).read_bytes())
        action = json.loads(payload.decode("utf-8"))
        if not isinstance(action, dict):
            raise RuntimeError("Malformed Claude action artifact")
        return action

    def _record_action(self, action: dict[str, Any]) -> None:
        if self.snapshot_dir is None:
            raise RuntimeError("Passive Claude replay requires an artifact directory")
        self._action_sequence += 1
        self.event_count += 1
        payload = json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8", errors="surrogateescape")
        filename = f"action_{self._action_sequence:05d}.json.gz"
        destination = Path(self.snapshot_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
                out.write(payload)
        name = str(action.get("name") or "unknown")
        self._actions.append(
            {
                "action_index": self._action_sequence,
                "tool_use_id": str(action.get("tool_use_id") or ""),
                "agent_id": str(action.get("agent_id") or "root"),
                "parent_tool_use_id": action.get("parent_tool_use_id"),
                "name": name,
                "status": str(action.get("status") or ""),
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "cost_usd": round(cost_from_responses(self._turn_responses), 8),
                "files": _paths_from_tool(name, action.get("input")),
                "action_path": self._action_reference(filename),
                "action_sha256": hashlib.sha256(payload).hexdigest(),
                "action_bytes": len(payload),
                "replay_status": "pending",
            }
        )
        self._persist_manifest(
            self._playback_result(
                cost_basis="live provider usage; final cost pending"
            )
        )

    def _ingest_line_after_exit(self, line: str) -> None:
        """Normalize one native event during post-exit processing."""

        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            return
        if not isinstance(event, dict):
            return
        ingested = self._claude_trace_builder.ingest(event)
        for normalized in ingested["messages"]:
            if normalized.get("role") != "assistant":
                continue
            self._record_claude_turn_response(normalized)
        for action in ingested["completed_actions"]:
            self._record_action(copy.deepcopy(action))

    def _persist_and_ingest_raw_stdout(self, raw_stdout: str) -> None:
        """Persist the native stream, then derive actions entirely post-exit."""

        if self.snapshot_dir is None:
            raise RuntimeError("Passive Claude replay requires an artifact directory")
        payload = raw_stdout.encode("utf-8", errors="surrogateescape")
        filename = "native.stdout.jsonl.gz"
        destination = Path(self.snapshot_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
                out.write(payload)
        self._raw_stdout_metadata = {
            "path": self._action_reference(filename),
            "format": "claude-native-stdout-jsonl-gzip-v1",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "parsed_after_agent_exit": True,
        }
        for line in raw_stdout.splitlines():
            self._ingest_line_after_exit(line)

    def _capture_final_archive(self) -> None:
        candidate, metadata, capture_seconds = self._capture_archive_candidate()
        destination = Path(self.snapshot_dir) / "final.sources.tar.gz"
        candidate.replace(destination)
        self._final_archive_metadata = {
            **metadata,
            "snapshot_path": self._archive_reference(destination.name),
            "snapshot_format": "deterministic-source-tar-gzip-v1",
            "snapshot_digest_basis": "canonical uncompressed tar stream",
            "capture_seconds": round(capture_seconds, 3),
        }

    def finish(
        self,
        authoritative_responses: list[dict[str, Any]],
        *,
        raw_stdout: str | None = None,
    ) -> dict[str, Any]:
        """Capture the terminal tree, then parse native stdout after Claude exits."""

        self._capture_final_archive()
        if raw_stdout is not None:
            self._persist_and_ingest_raw_stdout(raw_stdout)
        final_cost = cost_from_responses(authoritative_responses)
        observed_cost = max(
            (_safe_float(action.get("cost_usd")) for action in self._actions),
            default=0.0,
        )
        cost_basis = "recorded provider usage; terminal total separate; no proration"
        for item in self._actions:
            item["cost_boundary_exact"] = False
            item["cost_join_basis"] = "recorded_partial_usage"
            if observed_cost <= 0 and item.get("kind") != "baseline":
                item["cost_usd"] = None
                item["cost_join_basis"] = "missing_recorded_usage"
        result = self._playback_result(
            cost_basis=cost_basis,
            final_cost_usd=final_cost,
        )
        self._result = result
        self._persist_manifest(result)
        return result

    def _playback_result(
        self, *, cost_basis: str, final_cost_usd: float | None = None
    ) -> dict[str, Any]:
        result = super()._playback_result(
            cost_basis=cost_basis,
            final_cost_usd=final_cost_usd,
        )
        result.update(
            {
                "quality": self._external_quality,
                "capture_mode": "passive_action_replay",
                "metric_basis": (
                    "Lean tokens measured during post-run action replay"
                    if self._external_replayed
                    else "pending post-run action replay"
                ),
                "snapshot_basis": (
                    "one pre-run source archive, native completed Claude actions, "
                    "and one post-exit source archive"
                ),
                "build_basis": (
                    "post-run disposable network-disabled container from the "
                    "pinned baseline image"
                ),
                "agent_paused_for_checkpoints": False,
                "agent_launch_wrapped_for_playback": False,
                "source_reads_during_agent_execution": 0,
                "native_stream_parsed_during_agent_execution": False,
                "trajectory_standardization_phase": "post_exit",
                "agent_container_source_reads": "pre_run_and_post_exit_only",
                "agent_container_source_writes": False,
                "agent_container_git_writes": False,
                "agent_container_builds": False,
                "post_exit_process_quiescence": self.post_exit_quiesced,
                "capture_consistency": "no_runtime_checkpoints",
                "unpaused_action_capture_count": 0,
                "replay_divergence_count": self._replay_divergence_count,
                "bash_replay_policy": self.bash_replay_policy,
                "terminal_reconciliation": self.terminal_reconciliation,
                "replayed_bash_action_count": self._replayed_bash_action_count,
                "skipped_bash_action_count": self._skipped_bash_action_count,
                "native_action_timeline": {
                    "enabled": True,
                    "format": "claude-completed-action-gzip-v1",
                    "count": len(self._actions),
                    "actions": self._actions,
                },
            }
        )
        if self._final_archive_metadata:
            result["final_source_archive"] = dict(self._final_archive_metadata)
        if self._action_endpoint_metadata:
            result["action_replay_endpoint"] = dict(
                self._action_endpoint_metadata
            )
        if self._submission_endpoint_metadata:
            result["submitted_patch_endpoint"] = dict(
                self._submission_endpoint_metadata
            )
        if self._raw_stdout_metadata:
            result["raw_provider_stream"] = dict(self._raw_stdout_metadata)
        trace = self._claude_trace_builder.to_trace(include_messages=False)
        trace["filesystem_exactness"] = (
            "no live intermediate observations; offline action endpoint is "
            "verified against the post-exit source archive and submitted patch"
        )
        result["provider_action_trace"] = trace
        return result

    @staticmethod
    def _container_path(raw_path: Any) -> str:
        raw = str(raw_path or "")
        path = PurePosixPath(raw)
        if not path.is_absolute():
            path = PurePosixPath("/testbed") / path
        if not path.parts or ".." in path.parts:
            raise RuntimeError(f"unsafe Claude action path: {raw!r}")
        return path.as_posix()

    @staticmethod
    def _write_replay_file(replay_env: Any, path: str, content: str) -> None:
        writer = getattr(replay_env, "write_file_bytes", None)
        if callable(writer):
            writer(path, content.encode("utf-8", errors="surrogateescape"))
            return
        fallback = getattr(replay_env, "write_file", None)
        if not callable(fallback):
            raise RuntimeError("Replay environment cannot write reconstructed files")
        fallback(path, content)

    def _apply_edit(self, replay_env: Any, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("status") == "failed" or action.get("is_error"):
            return {
                "replay_returncode": 1,
                "observed_error": True,
                "outcome_matches": True,
                "skipped_failed_atomic_tool": True,
                "check_source_state": True,
            }
        arguments = action.get("input")
        if not isinstance(arguments, dict):
            raise RuntimeError("Claude Edit action has no object input")
        path = self._container_path(arguments.get("file_path"))
        old = arguments.get("old_string")
        new = arguments.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise RuntimeError("Claude Edit action has invalid old/new strings")
        current = replay_env.read_file(path)
        occurrences = current.count(old)
        replace_all = bool(arguments.get("replace_all"))
        if occurrences == 0 or (not replace_all and occurrences != 1):
            raise RuntimeError(
                f"Claude Edit precondition diverged for {path}: "
                f"found {occurrences} occurrences"
            )
        updated = current.replace(old, new) if replace_all else current.replace(old, new, 1)
        self._write_replay_file(replay_env, path, updated)
        return {
            "replay_returncode": 0,
            "observed_error": False,
            "outcome_matches": True,
            "check_source_state": True,
        }

    def _apply_write(self, replay_env: Any, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("status") == "failed" or action.get("is_error"):
            return {
                "replay_returncode": 1,
                "observed_error": True,
                "outcome_matches": True,
                "skipped_failed_atomic_tool": True,
                "check_source_state": True,
            }
        arguments = action.get("input")
        if not isinstance(arguments, dict):
            raise RuntimeError("Claude Write action has no object input")
        path = self._container_path(arguments.get("file_path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Claude Write action has no string content")
        self._write_replay_file(replay_env, path, content)
        return {
            "replay_returncode": 0,
            "observed_error": False,
            "outcome_matches": True,
            "check_source_state": True,
        }

    def _apply_bash(self, replay_env: Any, action: dict[str, Any]) -> dict[str, Any]:
        arguments = action.get("input")
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get("command"), str
        ):
            raise RuntimeError("Claude Bash action has no command")
        command = arguments["command"]
        mutation_reason = _bash_source_mutation_reason(command)
        if self.bash_replay_policy == "edit_tools_only" or (
            self.bash_replay_policy == "source_mutations_only"
            and mutation_reason is None
        ):
            self._skipped_bash_action_count += 1
            return {
                "replay_status": "skipped_bash_by_policy",
                "check_source_state": False,
                "bash_replay_policy": self.bash_replay_policy,
                "bash_mutation_reason": mutation_reason,
            }
        self._replayed_bash_action_count += 1
        observed_error = bool(
            action.get("status") == "failed" or action.get("is_error")
        )
        timed_out = False
        try:
            replayed = replay_env.execute(
                f"cd /testbed && {command}",
                timeout=self.replay_action_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial = exc.output or getattr(exc, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            replayed = {"returncode": 124, "output": partial}
        replay_error = replayed.get("returncode", 1) != 0
        return {
            "replay_returncode": replayed.get("returncode"),
            "observed_error": observed_error,
            "outcome_matches": replay_error == observed_error,
            "timed_out": timed_out,
            "output_tail": (replayed.get("output", "") or "")[
                -self.build_output_chars :
            ],
            "check_source_state": True,
            "bash_replay_policy": self.bash_replay_policy,
            "bash_mutation_reason": mutation_reason,
        }

    def _apply_action(
        self, replay_env: Any, action: dict[str, Any]
    ) -> dict[str, Any]:
        name = str(action.get("name") or "").lower()
        if name == "edit":
            return self._apply_edit(replay_env, action)
        if name == "write":
            return self._apply_write(replay_env, action)
        if name == "bash":
            return self._apply_bash(replay_env, action)
        if name in _NON_MUTATING_TOOLS:
            return {
                "replay_status": "skipped_non_mutating",
                "check_source_state": False,
            }
        if name in _KNOWN_UNSUPPORTED_MUTATING_TOOLS:
            raise RuntimeError(f"Unsupported mutating Claude tool: {action.get('name')}")
        raise RuntimeError(
            f"Unclassified Claude tool cannot be safely replayed: {action.get('name')}"
        )

    def _export_replay_digest(self, replay_env: Any, label: str) -> str:
        candidate = Path(self.snapshot_dir) / f".{label}-{id(self)}.tar.gz"
        try:
            metadata = replay_env.export_source_archive(
                candidate,
                self._archive_paths(),
                timeout=self.capture_timeout_seconds,
            )
            return str(metadata["source_archive_sha256"])
        finally:
            candidate.unlink(missing_ok=True)

    def _capture_action_endpoint(self, replay_env: Any) -> str:
        destination = Path(self.snapshot_dir) / "action-replay-final.sources.tar.gz"
        metadata = replay_env.export_source_archive(
            destination,
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        self._action_endpoint_metadata = {
            **metadata,
            "snapshot_path": self._archive_reference(destination.name),
            "snapshot_format": "deterministic-source-tar-gzip-v1",
            "snapshot_digest_basis": "canonical uncompressed tar stream",
        }
        return str(metadata["source_archive_sha256"])

    def _capture_submission_endpoint(self, replay_env: Any) -> str:
        destination = Path(self.snapshot_dir) / "submitted-patch-final.sources.tar.gz"
        metadata = replay_env.export_source_archive(
            destination,
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        self._submission_endpoint_metadata = {
            **metadata,
            "snapshot_path": self._archive_reference(destination.name),
            "snapshot_format": "deterministic-source-tar-gzip-v1",
            "snapshot_digest_basis": "canonical uncompressed tar stream",
        }
        return str(metadata["source_archive_sha256"])

    def _apply_submission(self, replay_env: Any, patch: str) -> str:
        replay_env.restore_source_archive(
            self._archive_file(self.points[0]),
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        if not patch:
            return self._capture_submission_endpoint(replay_env)
        patch_path = f"/tmp/leanlean-claude-submission-{id(self)}.patch"
        replay_env.write_file(patch_path, patch)
        applied = replay_env.execute(
            "cd /testbed && "
            f"{{ test ! -s {shlex.quote(patch_path)} || "
            f"git apply --binary --whitespace=nowarn {shlex.quote(patch_path)}; }}",
            timeout=False,
        )
        if applied.get("returncode"):
            raise RuntimeError(
                "Submitted patch could not be applied during Claude replay: "
                + applied.get("output", "")[-self.build_output_chars :]
            )
        return self._capture_submission_endpoint(replay_env)

    def _replay_actions(self, replay_env: Any, patch: str) -> dict[str, Any]:
        baseline = dict(self.points[0])
        replay_env.restore_source_archive(
            self._archive_file(baseline),
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        baseline_digest = str(baseline.get("source_archive_sha256") or "")
        baseline["lean_tokens"] = self._read_tokens(env=replay_env)
        baseline["lean_tokens_saved"] = 0
        baseline["replay_status"] = "complete"
        baseline["reconstruction"] = "pre_run_observation"
        baseline_build = self._run_replay_build(replay_env)
        if baseline_build is not None:
            baseline["build"] = baseline_build
        replay_points = [baseline]
        previous_digest = baseline_digest
        edit_index = 0
        self._replay_divergence_count = 0
        self._replayed_bash_action_count = 0
        self._skipped_bash_action_count = 0

        for record in self._actions:
            action = self._read_action(record)
            replay = self._apply_action(replay_env, action)
            record.update(replay)
            if replay.get("outcome_matches") is False:
                self._replay_divergence_count += 1
            if not replay.get("check_source_state"):
                record["replay_status"] = replay.get(
                    "replay_status", "skipped_non_mutating"
                )
                continue
            digest = self._export_replay_digest(
                replay_env, f"action-{record['action_index']}-check"
            )
            changed = digest != previous_digest
            record["source_changed"] = changed
            record["source_archive_sha256"] = digest
            record["replay_status"] = "complete"
            if not changed:
                continue
            edit_index += 1
            previous_digest = digest
            point = {
                "edit_index": edit_index,
                "action_index": record["action_index"],
                "tool_use_id": record.get("tool_use_id") or "",
                "agent_id": record.get("agent_id") or "root",
                "parent_tool_use_id": record.get("parent_tool_use_id"),
                "elapsed_seconds": record["elapsed_seconds"],
                "cost_usd": record["cost_usd"],
                "lean_tokens": self._read_tokens(env=replay_env),
                "lean_tokens_saved": None,
                "files": record.get("files") or [],
                "kind": f"claude_{record['name']}_offline_replay",
                "exact": False,
                "reconstruction": "ordered_completed_action_replay",
                "action_path": record["action_path"],
                "action_sha256": record["action_sha256"],
                "source_archive_sha256": digest,
                "replay_status": "complete",
            }
            build = self._run_replay_build(replay_env)
            if build is not None:
                point["build"] = build
            replay_points.append(point)

        action_digest = self._capture_action_endpoint(replay_env)
        captured_digest = str(
            self._final_archive_metadata.get("source_archive_sha256") or ""
        )
        submitted_digest = self._apply_submission(replay_env, patch)
        action_matches = bool(action_digest) and action_digest == captured_digest
        submission_matches = bool(submitted_digest) and submitted_digest == captured_digest
        joint_matches = action_matches and submission_matches
        reconciliation_applied = bool(
            self.terminal_reconciliation == "submitted_patch"
            and submission_matches
            and not action_matches
        )
        endpoint_matches = joint_matches or reconciliation_applied
        if reconciliation_applied:
            edit_index += 1
            replay_points.append(
                {
                    "edit_index": edit_index,
                    "action_index": None,
                    "tool_use_id": "",
                    "agent_id": "postprocessor",
                    "parent_tool_use_id": None,
                    "elapsed_seconds": 0.0,
                    "cost_usd": 0.0,
                    "lean_tokens": self._read_tokens(env=replay_env),
                    "lean_tokens_saved": None,
                    "files": [],
                    "kind": "terminal_submitted_patch_reconciliation",
                    "exact": True,
                    "reconstruction": "baseline_plus_verified_submitted_patch",
                    "snapshot_path": self._submission_endpoint_metadata[
                        "snapshot_path"
                    ],
                    "source_archive_sha256": submitted_digest,
                    "replay_status": "complete",
                }
            )
        verification = {
            "checked": True,
            "matches": endpoint_matches,
            "joint_action_and_submission_match": joint_matches,
            "source_diff_exact": submission_matches,
            "basis": "exact scoped source archive endpoint",
            "action_replay_matches_final": action_matches,
            "submitted_patch_matches_final": submission_matches,
            "trajectory_action_replay_exact": action_matches,
            "endpoint_reconstruction_exact": endpoint_matches,
            "terminal_reconciliation": self.terminal_reconciliation,
            "terminal_reconciliation_applied": reconciliation_applied,
            "captured_source_archive_sha256": captured_digest,
            "action_replay_source_archive_sha256": action_digest,
            "submitted_source_archive_sha256": submitted_digest,
            "intermediate_state_claim": (
                "offline reconstructed, not directly observed during agent execution"
            ),
        }

        self.points = replay_points
        self.edit_count = edit_index
        baseline_tokens = int(baseline["lean_tokens"])
        self.baseline_lean_tokens = baseline_tokens
        for point in self.points:
            tokens = int(point["lean_tokens"])
            point["lean_tokens_saved"] = baseline_tokens - tokens
            point["lean_token_compression_pct"] = (
                round(100 * (baseline_tokens - tokens) / baseline_tokens, 6)
                if baseline_tokens
                else 0.0
            )
        if self.points and self._actions:
            self.points[-1]["cost_usd"] = self._actions[-1]["cost_usd"]
            self.points[-1]["elapsed_seconds"] = self._actions[-1][
                "elapsed_seconds"
            ]
        if self.points:
            self.points[-1]["exact"] = verification["matches"]
            self.points[-1]["matches_submission"] = submission_matches

        self._external_replayed = True
        if joint_matches and self._replay_divergence_count == 0:
            self._external_quality = "endpoint_verified_passive_action_replay"
        elif joint_matches:
            self._external_quality = (
                "endpoint_verified_passive_action_replay_with_divergence"
            )
        elif reconciliation_applied:
            self._external_quality = (
                "endpoint_verified_with_submitted_patch_reconciliation"
            )
        else:
            self._external_quality = "invalid_final_tree_reconstruction"
        result = self._playback_result(
            cost_basis=str(
                self._result.get(
                    "cost_basis", "recorded provider usage; missing boundaries unknown"
                )
            ),
            final_cost_usd=_safe_float(self._result.get("final_cost_usd")),
        )
        result["submission_verification"] = verification
        result["final_snapshot_matches_submission"] = submission_matches
        result["endpoint_reconstruction_exact"] = endpoint_matches
        result["trajectory_action_replay_exact"] = action_matches
        result["source_diff_exact"] = verification["source_diff_exact"]
        self._result = result
        self._persist_manifest(result)
        return result

    def replay_submission(self, patch: str) -> dict[str, Any]:
        """Replay all completed Claude actions after the agent has exited."""

        spawner = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawner):
            raise RuntimeError(
                "Passive Claude replay requires a replay environment factory"
            )
        lifetime = max(
            600,
            len(self._actions)
            * (
                self.replay_action_timeout_seconds
                + self.capture_timeout_seconds
                + (self.build_timeout_seconds if self.build_command else 0)
                + 60
            ),
        )
        replay_env = None
        try:
            replay_env = spawner(lifetime_seconds=lifetime)
            self._prepared = False
            return self._replay_actions(replay_env, patch)
        except Exception as exc:
            self._external_quality = "passive_action_replay_failed"
            result = self._result or self._playback_result(
                cost_basis="passive action replay failed"
            )
            result["quality"] = self._external_quality
            result["replay_error"] = str(exc)
            result["final_snapshot_matches_submission"] = False
            self._result = result
            self._persist_manifest(result)
            return result
        finally:
            if replay_env is not None:
                replay_env.cleanup()
