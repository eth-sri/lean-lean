"""Portable, fail-closed replay of standardized coding-agent trajectories."""

from __future__ import annotations

import copy
import difflib
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from configs.model_constants import get_model_prices
from leanlean.claude_action_playback import (
    _bash_source_mutation_reason,
    _without_shell_heredoc_bodies,
)
from leanlean.metrics.tokens import count_lean_tokens_in_source
from leanlean.standardized_trace import STANDARDIZED_TRACE_FORMAT

REPLAY_BUNDLE_FORMAT = "code-harness-replay-bundle-v1"
_BASH_POLICIES = {"all", "source_mutations_only", "edit_tools_only"}
_RECONCILIATION_POLICIES = {"none", "submitted_patch"}


def _unwrap_captured_shell_command(command: str) -> str:
    """Remove one recorded shell argv wrapper before replay's own shell layer."""

    try:
        arguments = shlex.split(command, posix=True)
    except ValueError:
        return command
    if (
        len(arguments) == 3
        and PurePosixPath(arguments[0]).name in {"bash", "sh", "zsh"}
        and arguments[1].startswith("-")
        and "c" in arguments[1][1:]
    ):
        return arguments[2]
    return command


def _shell_asynchrony_reason(command: str) -> str | None:
    command = _without_shell_heredoc_bodies(command)
    try:
        lexer = shlex.shlex(
            command,
            # Non-POSIX mode preserves quote delimiters in word tokens, so a
            # literal quoted ``&`` cannot be confused with the shell background
            # operator. Punctuation operators are still split.
            posix=False,
            punctuation_chars="&<>|;()",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        tokens = []
    if "&" in tokens:
        return "background_operator"
    if "coproc" in tokens:
        return "coproc"
    if "disown" in tokens:
        return "disown"
    for index, token in enumerate(tokens):
        if PurePosixPath(token).name != "setsid":
            continue
        for option in tokens[index + 1 :]:
            if not option.startswith("-"):
                break
            if option == "--fork" or (
                option.startswith("-")
                and not option.startswith("--")
                and "f" in option[1:]
            ):
                return "detached_setsid"
    return None


class StandardizedReplayError(RuntimeError):
    """A bundle or action cannot be replayed without weakening its contract."""


def install_replay_tool_bundle(
    env: Any, environment: Mapping[str, Any]
) -> dict[str, Any]:
    """Recreate the agent container's pinned offline CLI tool environment."""

    generator = str(environment.get("generator") or "")
    if not generator:
        return {"status": "not_declared"}
    version = str(environment.get("cli_version") or "")
    if not version:
        raise StandardizedReplayError(
            f"replay bundle for {generator} has no pinned cli_version"
        )
    if generator == "codex_sub":
        from leanlean.generators.cli_agent import (
            install_codex_standalone_tools,
        )

        install_codex_standalone_tools(env, version)
    elif generator == "claude_code_sub":
        from leanlean.generators.cli_agent import (
            install_claude_standalone_tool,
        )

        install_claude_standalone_tool(env, version)
    else:
        raise StandardizedReplayError(
            f"unsupported standardized replay generator: {generator!r}"
        )
    return {
        "status": "installed",
        "generator": generator,
        "cli_version": version,
        "source": "pinned_offline_host_bundle",
        "network_used": False,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", errors="surrogateescape")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _usage_token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _assistant_turn_cost(
    message: Mapping[str, Any],
) -> tuple[float | None, dict[str, int]]:
    usage = message.get("usage")
    prices = get_model_prices(str(message.get("model") or ""))
    if not isinstance(usage, Mapping) or prices is None:
        return None, {}

    cache_read = _usage_token_count(usage.get("cache_read_input_tokens"))
    cache_creation = _usage_token_count(
        usage.get("cache_creation_input_tokens")
    )
    output_tokens = _usage_token_count(
        usage.get("output_tokens", usage.get("completion_tokens"))
    )
    if "prompt_tokens" not in usage and (
        "cache_read_input_tokens" in usage
        or "cache_creation_input_tokens" in usage
    ):
        input_tokens = _usage_token_count(usage.get("input_tokens"))
    else:
        prompt_tokens = _usage_token_count(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, Mapping):
            details = usage.get("input_tokens_details")
        if isinstance(details, Mapping):
            cache_read = max(
                cache_read,
                _usage_token_count(details.get("cached_tokens")),
            )
            cache_creation = max(
                cache_creation,
                _usage_token_count(details.get("cache_creation_tokens")),
            )
        input_tokens = max(0, prompt_tokens - cache_read - cache_creation)

    from leanlean.model.costs import cache_write_cost_adjustment
    input_price, output_price, cache_price, cache_creation_price = prices
    cost = (
        input_tokens * input_price
        + output_tokens * output_price
        + cache_read * cache_price
        + cache_creation * cache_creation_price
    ) / 1_000_000
    cost += cache_write_cost_adjustment(dict(usage), prices)
    return cost, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def _action_cost_ledger(
    trajectory: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    recorded = trajectory.get("recorded_cost_ledger")
    if isinstance(recorded, Mapping):
        if recorded.get("format") != "verified-replay-cost-ledger-v1":
            raise ValueError("unsupported recorded cost ledger")
        return ({int(key): value for key, value in recorded["boundaries"].items()},
                dict(recorded["summary"]))
    if trajectory.get("provider") == "antigravity-cli":
        from leanlean.antigravity_usage import AntigravityUsageJournal
        journal = AntigravityUsageJournal()
        for event in trajectory.get("events") or []:
            journal.ingest(event.get("payload", event))
        evidence = journal.evidence()
        boundaries = {}
        for action in trajectory.get("actions") or []:
            boundary = journal.tools.get(action.get("tool_call_id")) if journal.steps else None
            boundaries[int(action["action_index"])] = {
                "cost_usd": boundary["cost_usd"] if boundary else None,
                "cumulative_cost_usd": boundary["cost_usd"] if boundary else None,
                "native_usage": boundary["native_usage"] if boundary else None,
                "cost_boundary_exact": boundary is not None,
                "cost_granularity": "completed_native_step",
                "model_attribution": evidence["model_attribution"],
                "turn_cost_usd": None,
            }
        return boundaries, {
            "captured_completed_turn_cost_usd": evidence["recorded_cost_usd"],
            "captured_assistant_turn_count": len(journal.steps),
            "all_captured_turn_costs_exact": bool(journal.steps) and evidence["model_attribution"] == "per_step_model",
            "native_cost_evidence": evidence,
            "cost_granularity": "completed_native_step",
            "cost_allocation": "recorded native usage; no per-tool or per-edit proration",
        }

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for fallback_sequence, message in enumerate(trajectory.get("messages") or [], 1):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        if message.get("cost_accounting_exempt") is True:
            continue
        agent_id = str(message.get("trace_agent_id") or "root")
        sequence = _usage_token_count(
            message.get("trace_sequence", message.get("sequence"))
        ) or fallback_sequence
        message_id = str(message.get("id") or f"sequence-{sequence}")
        group = groups.setdefault(
            (agent_id, message_id),
            {
                "agent_id": agent_id,
                "message_id": message_id,
                "messages": [],
                "sequences": [],
                "tool_call_ids": [],
            },
        )
        group["messages"].append(message)
        group["sequences"].append(sequence)
        for block in message.get("content") or []:
            if (
                isinstance(block, Mapping)
                and block.get("type") == "tool_use"
                and block.get("id")
            ):
                group["tool_call_ids"].append(str(block["id"]))

    turns: list[dict[str, Any]] = []
    tool_turns: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        messages = list(group["messages"])
        representative = max(
            messages,
            key=lambda item: _usage_token_count(
                item.get("trace_sequence", item.get("sequence"))
            ),
        )
        usage_variants = {
            _canonical_bytes(item.get("usage")).decode(
                "utf-8", errors="surrogateescape"
            )
            for item in messages
            if isinstance(item.get("usage"), Mapping)
        }
        models = {
            str(item.get("model") or "")
            for item in messages
            if item.get("model")
        }
        cost, token_usage = _assistant_turn_cost(representative)
        tool_call_ids = sorted(set(group["tool_call_ids"]))
        turn = {
            "agent_id": group["agent_id"],
            "assistant_turn_id": group["message_id"],
            "first_sequence": min(group["sequences"]),
            "last_sequence": max(group["sequences"]),
            "timestamp": str(representative.get("trace_timestamp") or ""),
            "turn_cost_usd": cost,
            "token_usage": token_usage,
            "tool_call_ids": tool_call_ids,
            "shared_turn_tool_call_count": len(tool_call_ids),
            "cost_boundary_exact": (
                cost is not None
                and len(usage_variants) == 1
                and len(models) == 1
            ),
        }
        turns.append(turn)
        for tool_call_id in tool_call_ids:
            tool_turns[tool_call_id] = turn

    turns.sort(
        key=lambda item: (
            item["timestamp"] or "",
            item["first_sequence"],
            item["assistant_turn_id"],
        )
    )
    running_cost = 0.0
    all_turn_costs_exact = bool(turns)
    cumulative_cost_exact = True
    for turn in turns:
        turn_cost = turn["turn_cost_usd"]
        if turn_cost is None or turn["cost_boundary_exact"] is not True:
            all_turn_costs_exact = False
            cumulative_cost_exact = False
        else:
            running_cost += float(turn_cost)
        turn["cumulative_cost_usd"] = round(running_cost, 12)
        turn["cumulative_cost_exact"] = cumulative_cost_exact

    boundaries: dict[int, dict[str, Any]] = {}
    for action in trajectory.get("actions") or []:
        if not isinstance(action, Mapping):
            continue
        action_index = _usage_token_count(action.get("action_index"))
        tool_call_id = str(
            action.get("cost_tool_call_id") or action.get("tool_call_id") or ""
        )
        turn = tool_turns.get(tool_call_id)
        if turn is None:
            sequence = _usage_token_count(action.get("started_sequence"))
            candidates = [
                candidate
                for candidate in turns
                if candidate["agent_id"] == str(action.get("agent_id") or "root")
                and candidate["first_sequence"] <= sequence
            ]
            turn = candidates[-1] if candidates else None
        if turn is None:
            boundaries[action_index] = {
                "cost_usd": None,
                "cumulative_cost_usd": None,
                "turn_cost_usd": None,
                "cost_boundary_exact": False,
                "cost_granularity": "assistant_turn",
            }
            continue
        boundaries[action_index] = {
            "cost_usd": (
                turn["cumulative_cost_usd"]
                if turn["cumulative_cost_exact"]
                else None
            ),
            "cumulative_cost_usd": (
                turn["cumulative_cost_usd"] if turn["cumulative_cost_exact"] else None
            ),
            "turn_cost_usd": (
                round(float(turn["turn_cost_usd"]), 12)
                if turn["turn_cost_usd"] is not None
                else None
            ),
            "cost_boundary_exact": turn["cumulative_cost_exact"],
            "cost_granularity": "assistant_turn",
            "assistant_turn_id": turn["assistant_turn_id"],
            "shared_turn_tool_call_count": turn["shared_turn_tool_call_count"],
        }

    return boundaries, {
        "captured_completed_turn_cost_usd": round(running_cost, 12),
        "captured_assistant_turn_count": len(turns),
        "all_captured_turn_costs_exact": all_turn_costs_exact,
        "cost_granularity": "assistant_turn",
        "cost_allocation": "no per-tool or per-edit proration",
    }



def _copy_artifact(source: Path, destination: Path) -> dict[str, Any]:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "path": destination.name,
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }


def _source_artifact(
    source: str | os.PathLike[str],
    destination: Path,
    source_archive_sha256: str,
) -> dict[str, Any]:
    if not source_archive_sha256:
        raise StandardizedReplayError("source archive has no canonical source digest")
    result = _copy_artifact(Path(source).expanduser().resolve(), destination)
    result["source_archive_sha256"] = source_archive_sha256
    result["format"] = "deterministic-source-tar-gzip-v1"
    return result


def _trace_digest_valid(trace: Mapping[str, Any]) -> bool:
    expected = str(trace.get("trace_sha256") or "")
    body = copy.deepcopy(dict(trace))
    body.pop("trace_sha256", None)
    return bool(expected) and _sha256(_canonical_bytes(body)) == expected


def create_replay_bundle(
    destination: str | os.PathLike[str],
    *,
    standardized_trace: Mapping[str, Any],
    baseline_archive: str | os.PathLike[str],
    baseline_source_sha256: str,
    final_archive: str | os.PathLike[str],
    final_source_sha256: str,
    submitted_patch: str,
    source_scope: Sequence[str],
    environment: Mapping[str, Any],
) -> Path:
    """Create a portable evidence bundle without trusting external references."""

    if standardized_trace.get("format") != STANDARDIZED_TRACE_FORMAT:
        raise StandardizedReplayError("unsupported standardized trace format")
    if not _trace_digest_valid(standardized_trace):
        raise StandardizedReplayError("standardized trace digest is missing or invalid")
    root = Path(destination).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "replay-bundle.json"
    if manifest_path.exists():
        raise FileExistsError(f"replay bundle already exists: {manifest_path}")

    trajectory_payload = _canonical_bytes(standardized_trace) + b"\n"
    trajectory_path = root / "standardized-trace.json"
    trajectory_path.write_bytes(trajectory_payload)
    patch_payload = submitted_patch.encode("utf-8", errors="surrogateescape")
    patch_path = root / "submitted.patch"
    patch_path.write_bytes(patch_payload)
    artifacts: dict[str, Any] = {
        "trajectory": {
            "path": trajectory_path.name,
            "sha256": _sha256(trajectory_payload),
            "bytes": len(trajectory_payload),
            "format": STANDARDIZED_TRACE_FORMAT,
        },
        "baseline_source": _source_artifact(
            baseline_archive,
            root / "baseline.sources.tar.gz",
            baseline_source_sha256,
        ),
        "final_source": _source_artifact(
            final_archive,
            root / "final.sources.tar.gz",
            final_source_sha256,
        ),
        "submitted_patch": {
            "path": patch_path.name,
            "sha256": _sha256(patch_payload),
            "bytes": len(patch_payload),
            "format": "git-binary-unified-diff-v1",
        },
    }

    source = standardized_trace.get("source")
    native_path = (
        source.get("native_stream_path") if isinstance(source, Mapping) else None
    )
    if isinstance(native_path, str) and Path(native_path).is_file():
        native = _copy_artifact(
            Path(native_path).expanduser().resolve(), root / "native.stdout.jsonl"
        )
        expected = str(source.get("native_stream_sha256") or "")
        if native["sha256"] != expected:
            raise StandardizedReplayError(
                "standardized trace native-stream digest differs from captured artifact"
            )
        native["format"] = "provider-native-jsonl-v1"
        artifacts["native_stream"] = native
        capture_manifest = source.get("capture_manifest")
        if isinstance(capture_manifest, str) and Path(capture_manifest).is_file():
            capture_path = Path(capture_manifest).expanduser().resolve()
            try:
                capture_document = json.loads(capture_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise StandardizedReplayError(
                    f"native capture manifest cannot be bundled: {exc}"
                ) from exc
            if not isinstance(capture_document, Mapping):
                raise StandardizedReplayError("native capture manifest is malformed")
            copied_capture = _copy_artifact(
                capture_path,
                root / "native.capture.json",
            )
            copied_capture["format"] = str(capture_document.get("format") or "")
            artifacts["native_capture_manifest"] = copied_capture
            stderr_metadata = capture_document.get("stderr")
            if isinstance(stderr_metadata, Mapping):
                stderr_reference = stderr_metadata.get("path")
                if not isinstance(stderr_reference, str):
                    raise StandardizedReplayError(
                        "native capture stderr artifact has no path"
                    )
                stderr_path = (capture_path.parent / stderr_reference).resolve()
                if capture_path.parent not in stderr_path.parents:
                    raise StandardizedReplayError(
                        "native capture stderr artifact escapes its root"
                    )
                copied_stderr = _copy_artifact(
                    stderr_path,
                    root / "native.stderr.log",
                )
                if copied_stderr["sha256"] != stderr_metadata.get(
                    "sha256"
                ) or copied_stderr["bytes"] != stderr_metadata.get("bytes"):
                    raise StandardizedReplayError(
                        "native stderr differs from its capture manifest"
                    )
                copied_stderr["format"] = "provider-native-stderr-v1"
                artifacts["native_stderr"] = copied_stderr

    codex_rollout = standardized_trace.get("codex_native_rollout")
    if isinstance(codex_rollout, Mapping):
        rollout_manifest = codex_rollout.get("manifest_path")
        if not isinstance(rollout_manifest, str):
            raise StandardizedReplayError("Codex rollout manifest path is missing")
        copied_manifest = _copy_artifact(
            Path(rollout_manifest).expanduser().resolve(),
            root / "codex-rollouts" / "rollouts.json",
        )
        copied_manifest["path"] = "codex-rollouts/rollouts.json"
        if copied_manifest["sha256"] != codex_rollout.get("manifest_sha256"):
            raise StandardizedReplayError("Codex rollout manifest digest differs")
        copied_manifest["format"] = "codex-native-rollout-bundle-v1"
        artifacts["codex_rollout_manifest"] = copied_manifest
        try:
            rollout_document = json.loads(
                Path(rollout_manifest).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StandardizedReplayError(
                f"Codex rollout manifest cannot be bundled: {exc}"
            ) from exc
        streams = rollout_document.get("streams")
        if not isinstance(streams, list):
            raise StandardizedReplayError("Codex rollout manifest has no streams")
        stream_paths = {
            str(stream.get("session_id") or ""): stream.get("path")
            for stream in streams
            if isinstance(stream, Mapping)
        }
        rollout_artifacts = codex_rollout.get("artifacts")
        if not isinstance(rollout_artifacts, list) or not rollout_artifacts:
            raise StandardizedReplayError("Codex rollout session streams are missing")
        for index, metadata in enumerate(rollout_artifacts, 1):
            if not isinstance(metadata, Mapping):
                raise StandardizedReplayError("Codex rollout stream is malformed")
            source_path = Path(str(metadata.get("path") or "")).expanduser().resolve()
            filename = stream_paths.get(str(metadata.get("session_id") or ""))
            if (
                not isinstance(filename, str)
                or not filename.endswith(".jsonl.gz")
                or Path(filename).name != filename
                or source_path.name != filename
            ):
                raise StandardizedReplayError(
                    "Codex rollout manifest and captured stream names differ"
                )
            copied = _copy_artifact(
                source_path,
                root / "codex-rollouts" / filename,
            )
            copied["path"] = f"codex-rollouts/{filename}"
            if copied["sha256"] != metadata.get("archive_sha256") or copied[
                "bytes"
            ] != metadata.get("archive_bytes"):
                raise StandardizedReplayError(
                    "Codex rollout compressed artifact differs"
                )
            copied.update(
                {
                    "format": "codex-native-rollout-jsonl-gzip-v1",
                    "content_sha256": metadata.get("sha256"),
                    "content_bytes": metadata.get("bytes"),
                    "session_id": metadata.get("session_id"),
                }
            )
            artifacts[f"codex_rollout_{index:04d}"] = copied

    manifest = {
        "format": REPLAY_BUNDLE_FORMAT,
        "provider": standardized_trace.get("provider"),
        "session_id": standardized_trace.get("session_id"),
        "source_scope": [str(item) for item in source_scope],
        "environment": copy.deepcopy(dict(environment)),
        "artifacts": artifacts,
        "exactness_contract": {
            "native_stream_exact": standardized_trace.get("quality", {}).get(
                "native_stream_exact"
            ),
            "tool_trace_exact": standardized_trace.get("quality", {}).get(
                "tool_trace_exact"
            ),
            "mutation_trace_exact": standardized_trace.get("quality", {}).get(
                "mutation_trace_exact"
            ),
            "subagent_lineage_complete": standardized_trace.get("quality", {}).get(
                "subagent_lineage_complete"
            ),
            "endpoint_requires_digest_equality": True,
            "action_and_endpoint_exactness_reported_separately": True,
        },
    }
    temporary = root / ".replay-bundle.json.tmp"
    try:
        temporary.write_bytes(_canonical_bytes(manifest) + b"\n")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def _bundle_artifact(root: Path, metadata: Mapping[str, Any]) -> Path:
    reference = metadata.get("path")
    if not isinstance(reference, str) or not reference:
        raise StandardizedReplayError("bundle artifact has no path")
    path = (root / reference).resolve()
    if path != root and root not in path.parents:
        raise StandardizedReplayError(f"bundle artifact escapes its root: {reference}")
    if not path.is_file():
        raise StandardizedReplayError(f"bundle artifact is missing: {path}")
    payload = path.read_bytes()
    if int(metadata.get("bytes", -1)) != len(payload):
        raise StandardizedReplayError(f"bundle artifact byte count differs: {path}")
    if str(metadata.get("sha256") or "") != _sha256(payload):
        raise StandardizedReplayError(f"bundle artifact digest differs: {path}")
    return path


def load_replay_bundle(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a bundle only after verifying every retained artifact."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StandardizedReplayError(
            f"cannot load replay bundle {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("format") != REPLAY_BUNDLE_FORMAT:
        raise StandardizedReplayError("unsupported replay bundle format")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise StandardizedReplayError("replay bundle has no artifacts")
    paths = {
        name: _bundle_artifact(manifest_path.parent, metadata)
        for name, metadata in artifacts.items()
        if isinstance(metadata, Mapping)
    }
    required = {"trajectory", "baseline_source", "final_source", "submitted_patch"}
    if not required.issubset(paths):
        raise StandardizedReplayError(
            f"replay bundle is missing required artifacts: {sorted(required - set(paths))}"
        )
    trajectory = json.loads(paths["trajectory"].read_text(encoding="utf-8"))
    if not isinstance(trajectory, dict) or not _trace_digest_valid(trajectory):
        raise StandardizedReplayError("standardized trajectory digest differs")
    manifest["_root"] = manifest_path.parent
    manifest["_manifest_path"] = manifest_path
    manifest["_paths"] = paths
    manifest["_trajectory"] = trajectory
    return manifest


@dataclass(frozen=True)
class ReplayPolicy:
    """Replay cost/fidelity controls which never relax endpoint validation."""

    bash: str = "all"
    terminal_reconciliation: str = "none"
    action_timeout_seconds: int = 600
    verify_each_mutation: bool = False
    build_after_each_mutation: bool = False
    build_command: str = ""
    build_timeout_seconds: int = 3600
    build_output_chars: int = 4000
    reject_asynchronous_actions: bool = False

    def __post_init__(self) -> None:
        if self.bash not in _BASH_POLICIES:
            raise ValueError(f"bash must be one of {sorted(_BASH_POLICIES)}")
        if self.terminal_reconciliation not in _RECONCILIATION_POLICIES:
            raise ValueError(
                "terminal_reconciliation must be one of "
                f"{sorted(_RECONCILIATION_POLICIES)}"
            )
        if self.action_timeout_seconds <= 0:
            raise ValueError("action_timeout_seconds must be positive")
        if self.build_timeout_seconds <= 0:
            raise ValueError("build_timeout_seconds must be positive")
        if self.build_output_chars <= 0:
            raise ValueError("build_output_chars must be positive")
        if self.build_after_each_mutation and not self.build_command.strip():
            raise ValueError(
                "build_after_each_mutation requires a non-empty build_command"
            )


class StandardizedReplayEngine:
    """Replay canonical source mutations in a fresh, caller-provided sandbox."""

    def __init__(
        self,
        bundle: str | os.PathLike[str],
        *,
        replay_env: Any,
        policy: ReplayPolicy | None = None,
        instance_id: str = "",
        final_cost_usd: float = 0.0,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        checkpoint_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.bundle = load_replay_bundle(bundle)
        self.env = replay_env
        self.policy = policy or ReplayPolicy()
        self.instance_id = instance_id
        self.final_cost_usd = max(0.0, float(final_cost_usd))
        self.trajectory = self.bundle["_trajectory"]
        self.paths: dict[str, Path] = self.bundle["_paths"]
        self.scope = tuple(self.bundle.get("source_scope") or ())
        if not self.scope:
            raise StandardizedReplayError("replay bundle source scope is empty")
        self.checkpoint_dir = (
            Path(checkpoint_dir).expanduser().resolve(strict=False)
            if checkpoint_dir is not None
            else None
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self._base_commit = ""
        self._build_env: Any | None = None
    def _notify_progress(self, event: Mapping[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:  # noqa: BLE001
            return


    @staticmethod
    def _container_path(raw_path: Any) -> str:
        raw = str(raw_path or "")
        path = PurePosixPath(raw)
        if not path.is_absolute():
            path = PurePosixPath("/testbed") / path
        if ".." in path.parts:
            raise StandardizedReplayError(f"unsafe action path: {raw}")
        if not any(
            path == root or root in path.parents
            for root in (PurePosixPath("/testbed"), PurePosixPath("/tmp"))
        ):
            raise StandardizedReplayError(
                f"action path is outside the isolated replay roots: {raw}"
            )
        return path.as_posix()

    def _state_digest(self, label: str) -> str:
        with tempfile.TemporaryDirectory(prefix="standardized-replay-") as directory:
            destination = Path(directory) / f"{label}.sources.tar.gz"
            metadata = self.env.export_source_archive(
                destination,
                self.scope,
                timeout=self.policy.action_timeout_seconds,
            )
        digest = str(metadata.get("source_archive_sha256") or "")
        if not digest:
            raise StandardizedReplayError(
                "replay environment returned no source digest"
            )
        return digest

    @staticmethod
    def _archive_lean_tokens(path: Path) -> int:
        total = 0
        try:
            archive = tarfile.open(path, mode="r:gz")
        except tarfile.ReadError:
            # Unit/legacy single-file replay environments may expose the scoped
            # file directly as a deterministic gzip payload rather than a tar.
            return count_lean_tokens_in_source(
                gzip.decompress(path.read_bytes()).decode(
                    "utf-8", errors="replace"
                )
            )
        with archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or member_path.suffix != ".lean"
                    or member_path.name == "lakefile.lean"
                    or ".lake" in member_path.parts
                ):
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                total += count_lean_tokens_in_source(
                    stream.read().decode("utf-8", errors="replace")
                )
        return total

    @staticmethod
    def _archive_source_files(
        path: Path, scope: Sequence[str]
    ) -> dict[str, bytes]:
        try:
            archive = tarfile.open(path, mode="r:gz")
        except tarfile.ReadError:
            if len(scope) != 1:
                raise StandardizedReplayError(
                    "single-file checkpoint archive has ambiguous source scope"
                )
            return {str(scope[0]): gzip.decompress(path.read_bytes())}
        files: dict[str, bytes] = {}
        with archive:
            for member in archive:
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                name = PurePosixPath(member.name).as_posix().removeprefix("./")
                files[name] = stream.read()
        return files

    @staticmethod
    def _incremental_diff(
        before: Mapping[str, bytes], after: Mapping[str, bytes]
    ) -> tuple[str, list[str]]:
        changed_files: list[str] = []
        chunks: list[str] = []
        for name in sorted(set(before) | set(after)):
            old = before.get(name)
            new = after.get(name)
            if old == new:
                continue
            changed_files.append(name)
            if (old is not None and b"\0" in old) or (
                new is not None and b"\0" in new
            ):
                chunks.append(f"Binary files a/{name} and b/{name} differ\n")
                continue
            old_lines = (
                old.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
                if old is not None
                else []
            )
            new_lines = (
                new.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
                if new is not None
                else []
            )
            chunks.extend(
                line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                for line in difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{name}" if old is not None else "/dev/null",
                    tofile=f"b/{name}" if new is not None else "/dev/null",
                    lineterm="\n",
                )
            )
        return "".join(chunks), changed_files

    def _state_checkpoint(self, label: str) -> dict[str, Any]:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if self.checkpoint_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="standardized-replay-")
            destination = Path(temporary.name) / f"{safe_label}.sources.tar.gz"
        else:
            destination = self.checkpoint_dir / f"{safe_label}.sources.tar.gz"
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite replay checkpoint: {destination}"
                )
        try:
            metadata = self.env.export_source_archive(
                destination,
                self.scope,
                timeout=self.policy.action_timeout_seconds,
            )
            tokens = self._archive_lean_tokens(destination)
            source_files = self._archive_source_files(destination, self.scope)
            payload = destination.read_bytes()
            digest = str(metadata.get("source_archive_sha256") or "")
            if not digest:
                raise StandardizedReplayError(
                    "replay environment returned no source digest"
                )
            result: dict[str, Any] = {
                "source_archive_sha256": digest,
                "lean_tokens": tokens,
                "snapshot_format": "deterministic-source-tar-gzip-v1",
                "snapshot_digest_basis": "canonical uncompressed tar stream",
                "snapshot_archive_sha256": _sha256(payload),
                "snapshot_archive_bytes": len(payload),
                "_snapshot_files": source_files,
                "_snapshot_payload": payload,
            }
            if self.checkpoint_dir is not None:
                result["snapshot_artifact"] = {
                    "path": str(destination),
                    "sha256": _sha256(payload),
                    "bytes": len(payload),
                    "format": "deterministic-source-tar-gzip-v1",
                }
            return result
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _attach_incremental_diff(
        self,
        checkpoint: dict[str, Any],
        previous_files: Mapping[str, bytes],
        *,
        label: str,
    ) -> dict[str, bytes]:
        current_files = checkpoint.pop("_snapshot_files")
        diff, changed_files = self._incremental_diff(previous_files, current_files)
        payload = diff.encode("utf-8", errors="surrogateescape")
        artifact: dict[str, Any] = {
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "format": "unified-source-diff-v1",
        }
        if self.checkpoint_dir is not None:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
            destination = self.checkpoint_dir / f"{safe_label}.incremental.diff"
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite replay diff: {destination}"
                )
            destination.write_bytes(payload)
            artifact["path"] = str(destination)
        checkpoint["incremental_diff_artifact"] = artifact
        checkpoint["changed_files"] = changed_files
        return current_files

    def _restore_baseline(self) -> str:
        self.env.restore_source_archive(
            self.paths["baseline_source"],
            self.scope,
            timeout=self.policy.action_timeout_seconds,
        )
        digest = self._state_digest("baseline-check")
        expected = str(
            self.bundle["artifacts"]["baseline_source"].get("source_archive_sha256")
            or ""
        )
        if digest != expected:
            raise StandardizedReplayError(
                f"restored baseline digest differs: expected {expected}, found {digest}"
            )
        return digest

    def _write_bytes(self, path: str, content: bytes) -> None:
        writer = getattr(self.env, "write_file_bytes", None)
        if callable(writer):
            writer(path, content)
            return
        fallback = getattr(self.env, "write_file", None)
        if not callable(fallback):
            raise StandardizedReplayError("replay environment cannot write files")
        fallback(path, content.decode("utf-8", errors="surrogateescape"))

    def _apply_edit(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if action.get("status") == "failed" or action.get("is_error"):
            return {"replay_status": "skipped_failed_atomic_tool", "mutating": False}
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping):
            raise StandardizedReplayError("Edit action has no object arguments")
        path = self._container_path(arguments.get("file_path"))
        old = arguments.get("old_string")
        new = arguments.get("new_string")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise StandardizedReplayError("Edit action has invalid old/new strings")
        current = self.env.read_file(path)
        occurrences = current.count(old)
        replace_all = bool(arguments.get("replace_all"))
        if occurrences == 0 or (not replace_all and occurrences != 1):
            raise StandardizedReplayError(
                f"Edit precondition diverged for {path}: found {occurrences} occurrences"
            )
        updated = (
            current.replace(old, new) if replace_all else current.replace(old, new, 1)
        )
        self._write_bytes(path, updated.encode("utf-8", errors="surrogateescape"))
        return {"replay_status": "complete", "mutating": True}

    def _apply_write(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if action.get("status") == "failed" or action.get("is_error"):
            return {"replay_status": "skipped_failed_atomic_tool", "mutating": False}
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping) or not isinstance(
            arguments.get("content"), str
        ):
            raise StandardizedReplayError("Write action has invalid arguments")
        path = self._container_path(arguments.get("file_path"))
        self._write_bytes(
            path,
            arguments["content"].encode("utf-8", errors="surrogateescape"),
        )
        return {"replay_status": "complete", "mutating": True}

    def _apply_bash(self, action: Mapping[str, Any]) -> dict[str, Any]:
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping) or not isinstance(
            arguments.get("command"), str
        ):
            raise StandardizedReplayError("Bash action has no command")
        command = _unwrap_captured_shell_command(arguments["command"])
        mutation_reason = _bash_source_mutation_reason(command)
        asynchrony_reason = _shell_asynchrony_reason(command)
        if self.policy.reject_asynchronous_actions and asynchrony_reason:
            raise StandardizedReplayError(
                "strict replay rejects an action whose effects can outlive its "
                f"boundary: {asynchrony_reason}"
            )
        observed_error = bool(
            action.get("status") == "failed"
            or action.get("is_error")
            or (
                isinstance(action.get("observed_returncode"), int)
                and action.get("observed_returncode") != 0
            )
        )
        if self.policy.bash == "edit_tools_only" or (
            self.policy.bash == "source_mutations_only" and mutation_reason is None
        ):
            return {
                "replay_status": "skipped_bash_by_policy",
                "mutating": False,
                "bash_mutation_reason": mutation_reason,
            }
        observed_result = str(action.get("result") or "")
        observed_timed_out = bool(action.get("timed_out")) or bool(
            re.search(r"\b(?:timed out|timeout)\b", observed_result, re.IGNORECASE)
        )
        if observed_error and observed_timed_out:
            # Timed-out commands can hang again and do not have a trustworthy
            # completion boundary. Preserve the native failure and let terminal
            # digest equality reject any unreconstructed partial source effect.
            return {
                "replay_status": "skipped_timed_out_bash",
                "mutating": False,
                "bash_mutation_reason": mutation_reason,
                "observed_error": True,
                "observed_timed_out": True,
                "outcome_matches": None,
            }
        cwd = self._container_path(arguments.get("cwd") or "/testbed")
        replay_timed_out = False
        try:
            result = self.env.execute(
                f"cd {shlex.quote(cwd)} && {command}",
                timeout=self.policy.action_timeout_seconds,
            )
            replay_timed_out = bool(result.get("timed_out"))
        except subprocess.TimeoutExpired as exc:
            replay_timed_out = True
            partial = exc.output or getattr(exc, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            result = {"returncode": 124, "output": partial}
        replay_error = result.get("returncode", 1) != 0
        return {
            "replay_status": "complete",
            "mutating": mutation_reason is not None,
            "bash_mutation_reason": mutation_reason,
            "replay_returncode": result.get("returncode"),
            "observed_error": observed_error,
            "outcome_matches": replay_error == observed_error,
            "observed_timed_out": observed_timed_out,
            "replay_timed_out": replay_timed_out,
        }

    @classmethod
    def _standalone_change_diff(cls, change: Mapping[str, Any]) -> str:
        diff = str(change.get("diff") or "").rstrip("\n")
        if diff.startswith(("diff --git ", "--- ")):
            return diff + "\n"
        path = PurePosixPath(cls._container_path(change.get("path"))).relative_to(
            "/testbed"
        )
        kind = change.get("kind")
        move_path = kind.get("movePath") if isinstance(kind, Mapping) else None
        if move_path:
            destination = PurePosixPath(cls._container_path(move_path)).relative_to(
                "/testbed"
            )
            prefix = f"diff --git a/{path} b/{destination}\n"
            rename = f"rename from {path}\nrename to {destination}\n"
            if not diff:
                return f"{prefix}similarity index 100%\n{rename}"
            return f"{prefix}{rename}--- a/{path}\n+++ b/{destination}\n{diff}\n"
        if not diff:
            raise StandardizedReplayError("FileChange action omits its diff")
        kind_value = str(
            kind.get("type") if isinstance(kind, Mapping) else kind or ""
        ).lower()
        if "add" in kind_value or "create" in kind_value:
            before, after = "/dev/null", f"b/{path}"
        elif "delete" in kind_value or "remove" in kind_value:
            before, after = f"a/{path}", "/dev/null"
        else:
            before, after = f"a/{path}", f"b/{path}"
        return f"diff --git a/{path} b/{path}\n--- {before}\n+++ {after}\n{diff}\n"

    def _apply_content_change(self, change: Mapping[str, Any]) -> None:
        path = self._container_path(change.get("path"))
        content = change.get("content")
        kind = change.get("kind")
        kind_value = str(
            kind.get("type") if isinstance(kind, Mapping) else kind or ""
        ).lower()
        if not isinstance(content, str):
            raise StandardizedReplayError(
                "FileChange full-content action omits its content"
            )
        if "add" in kind_value or "create" in kind_value:
            self._write_bytes(
                path, content.encode("utf-8", errors="surrogateescape")
            )
            return
        if "delete" in kind_value or "remove" in kind_value:
            current = self.env.read_file(path)
            if current != content:
                raise StandardizedReplayError(
                    f"FileChange delete precondition diverged for {path}"
                )
            result = self.env.execute(
                f"rm -f -- {shlex.quote(path)}",
                timeout=self.policy.action_timeout_seconds,
            )
            if result.get("returncode"):
                raise StandardizedReplayError(
                    f"could not delete {path}: "
                    + str(result.get("output") or "")[-4000:]
                )
            return
        raise StandardizedReplayError(
            f"unsupported full-content FileChange kind: {kind_value!r}"
        )

    def _apply_patch_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        arguments = action.get("arguments")
        changes = arguments.get("changes") if isinstance(arguments, Mapping) else None
        if not isinstance(changes, list) or not changes:
            raise StandardizedReplayError("FileChange action has no changes")
        diff_changes = [
            change
            for change in changes
            if isinstance(change, Mapping) and isinstance(change.get("diff"), str)
        ]
        patch = "".join(
            self._standalone_change_diff(change)
            for change in diff_changes
        )
        if patch:
            self._apply_patch(patch, label=f"action-{action.get('action_index')}")
        content_changes = [
            change
            for change in changes
            if isinstance(change, Mapping) and not isinstance(change.get("diff"), str)
        ]
        for change in content_changes:
            self._apply_content_change(change)
        if len(diff_changes) + len(content_changes) != len(changes):
            raise StandardizedReplayError("FileChange action contains malformed changes")
        return {"replay_status": "complete", "mutating": True, "replay_returncode": 0}

    def _apply_patch(self, patch: str, *, label: str) -> dict[str, Any]:
        if not patch:
            return {"replay_status": "empty_patch", "mutating": False}
        path = f"/tmp/standardized-replay-{os.getpid()}-{label}.patch"
        self.env.write_file(path, patch)
        result = self.env.execute(
            "cd /testbed && git apply --binary --whitespace=nowarn "
            f"{shlex.quote(path)}",
            timeout=self.policy.action_timeout_seconds,
        )
        if result.get("returncode"):
            raise StandardizedReplayError(
                f"could not apply {label}: " + str(result.get("output") or "")[-4000:]
            )
        return {"replay_status": "complete", "mutating": True, "replay_returncode": 0}

    def _apply_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(action.get("replay_kind") or "unsupported")
        if kind == "non_mutating":
            return {"replay_status": "skipped_non_mutating", "mutating": False}
        if kind == "edit":
            return self._apply_edit(action)
        if kind == "write":
            return self._apply_write(action)
        if kind == "bash":
            return self._apply_bash(action)
        if kind == "patch":
            return self._apply_patch_action(action)
        if kind == "unresolved_command":
            return {
                "replay_status": "skipped_unresolved_command",
                "mutating": False,
                "source_effect_test": "terminal_action_endpoint_digest_equality",
            }
        raise StandardizedReplayError(
            f"unsupported potentially mutating action: {action.get('name')} ({kind})"
        )

    def _resolve_base_commit(self) -> str:
        if self._base_commit:
            return self._base_commit
        result = self.env.execute(
            "cd /testbed && git rev-parse HEAD",
            timeout=self.policy.action_timeout_seconds,
        )
        lines = str(result.get("output") or "").strip().splitlines()
        commit = lines[-1] if lines else ""
        if result.get("returncode") or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise StandardizedReplayError(
                "could not resolve replay baseline commit: "
                + str(result.get("output") or "")[-4000:]
            )
        self._base_commit = commit
        return commit

    def _create_tree_snapshot(self) -> str:
        base_commit = self._resolve_base_commit()
        index_path = f"/tmp/.standardized-replay-{os.getpid()}-{id(self)}.index"
        result = self.env.execute(
            "cd /testbed && "
            f"rm -f {shlex.quote(index_path)} {shlex.quote(index_path + '.lock')} && "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git read-tree "
            f"{shlex.quote(base_commit)} && "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git add -u -- . && "
            "git ls-files --others --exclude-standard -z -- . | "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git add "
            "--pathspec-from-file=- --pathspec-file-nul && "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git write-tree; "
            f"rc=$?; rm -f {shlex.quote(index_path)} "
            f"{shlex.quote(index_path + '.lock')}; exit $rc",
            timeout=self.policy.action_timeout_seconds,
        )
        lines = str(result.get("output") or "").strip().splitlines()
        tree_oid = lines[-1] if lines else ""
        if result.get("returncode") or not re.fullmatch(
            r"[0-9a-f]{40,64}", tree_oid
        ):
            raise StandardizedReplayError(
                "could not capture replay checkpoint tree: "
                + str(result.get("output") or "")[-4000:]
            )
        return tree_oid

    def _build_environment(self) -> Any:
        if self._build_env is not None:
            return self._build_env
        spawn = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawn):
            raise StandardizedReplayError(
                "checkpoint builds require a separate pinned replay sandbox"
            )
        action_count = max(1, len(self.trajectory.get("actions") or []))
        self._build_env = spawn(
            lifetime_seconds=max(
                600,
                action_count * self.policy.build_timeout_seconds + 600,
            )
        )
        return self._build_env

    def _run_build(
        self,
        *,
        snapshot_payload: bytes,
        source_archive_sha256: str,
    ) -> dict[str, Any] | None:
        if not self.policy.build_after_each_mutation:
            return None
        # The canonical source archive is already the checkpoint identity. Do
        # not run Git index/tree operations in the certified replay sandbox:
        # repository filters or hooks are allowed to have source side effects.
        # Builds receive only this archive in a separate container.
        tree_oid = None
        build_env = self._build_environment()
        with tempfile.TemporaryDirectory(
            prefix="standardized-replay-build-input-"
        ) as directory:
            archive = Path(directory) / "checkpoint.sources.tar.gz"
            archive.write_bytes(snapshot_payload)
            build_env.restore_source_archive(
                archive,
                self.scope,
                timeout=self.policy.action_timeout_seconds,
            )
            restored = build_env.export_source_archive(
                Path(directory) / "restored.sources.tar.gz",
                self.scope,
                timeout=self.policy.action_timeout_seconds,
            )
            restored_digest = str(restored.get("source_archive_sha256") or "")
            if restored_digest != source_archive_sha256:
                return {
                    "passed": False,
                    "returncode": 1,
                    "command": self.policy.build_command,
                    "duration_seconds": 0.0,
                    "output": (
                        "isolated build sandbox restored the wrong source digest: "
                        f"expected {source_archive_sha256}, found {restored_digest}"
                    ),
                    "tree_oid": tree_oid,
                    "setup_failed": True,
                    "source_tree_unchanged": False,
                    "environment": "isolated_replay_container",
                }

            started = time.monotonic()
            timed_out = False
            try:
                build = build_env.execute(
                    f"cd /testbed && {self.policy.build_command}",
                    timeout=self.policy.build_timeout_seconds,
                )
                timed_out = bool(build.get("timed_out"))
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                partial = exc.output or getattr(exc, "stdout", None) or ""
                if isinstance(partial, bytes):
                    partial = partial.decode("utf-8", errors="replace")
                build = {"returncode": 124, "output": partial}
            duration = time.monotonic() - started
            after = build_env.export_source_archive(
                Path(directory) / "after-build.sources.tar.gz",
                self.scope,
                timeout=self.policy.action_timeout_seconds,
            )
            unchanged = (
                str(after.get("source_archive_sha256") or "")
                == source_archive_sha256
            )
        return {
            "passed": build.get("returncode", 1) == 0,
            "returncode": build.get("returncode"),
            "command": self.policy.build_command,
            "duration_seconds": round(duration, 3),
            "output": str(build.get("output") or "")[
                -self.policy.build_output_chars :
            ],
            "tree_oid": tree_oid,
            "timed_out": timed_out,
            "source_tree_unchanged": unchanged,
            "environment": "isolated_replay_container",
        }

    def _cleanup_build_environment(self) -> None:
        if self._build_env is None:
            return
        try:
            self._build_env.cleanup()
        finally:
            self._build_env = None

    @staticmethod
    def _action_files(action: Mapping[str, Any]) -> list[str]:
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping):
            return []
        files: list[str] = []
        for key in ("file_path", "path"):
            value = arguments.get(key)
            if isinstance(value, str):
                files.append(value)
        changes = arguments.get("changes")
        if isinstance(changes, list):
            files.extend(
                str(change.get("path"))
                for change in changes
                if isinstance(change, Mapping) and change.get("path")
            )
        return sorted(set(files))

    def _submitted_reconstruction_in_place(self) -> tuple[str, str]:
        self._restore_baseline()
        patch = (
            self.paths["submitted_patch"]
            .read_bytes()
            .decode("utf-8", errors="surrogateescape")
        )
        basis = "submitted_patch"
        if not patch:
            terminal = self.trajectory.get("terminal_cumulative_diff")
            if isinstance(terminal, Mapping) and isinstance(terminal.get("diff"), str):
                patch = terminal["diff"]
                basis = "provider_terminal_cumulative_diff"
        self._apply_patch(patch, label="terminal-reconstruction")
        return self._state_digest("submitted-endpoint"), basis

    def _submitted_reconstruction(self) -> tuple[str, str]:
        """Prove the submitted endpoint in a fresh, isolated sandbox.

        Replayed Bash actions can commit, reset, or launch detached writers.
        Restoring scoped source files does not restore Git metadata or stop
        those writers, so applying the terminal patch in the action sandbox is
        inherently racy. A second pinned, networkless environment makes this
        proof depend only on the captured baseline and submitted patch.
        """

        spawn = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawn):
            raise StandardizedReplayError(
                "exact endpoint reconstruction requires a fresh replay sandbox"
            )
        endpoint_env = spawn(
            lifetime_seconds=max(600, 2 * self.policy.action_timeout_seconds)
        )
        try:
            endpoint_engine = type(self)(
                self.bundle["_manifest_path"],
                replay_env=endpoint_env,
                policy=self.policy,
                instance_id=self.instance_id,
                final_cost_usd=self.final_cost_usd,
            )
            return endpoint_engine._submitted_reconstruction_in_place()
        finally:
            endpoint_env.cleanup()

    def replay(self) -> dict[str, Any]:
        """Replay actions and prove exactness through canonical digest equality."""

        replay_started = time.monotonic()
        action_cost_boundaries, cost_summary = _action_cost_ledger(self.trajectory)
        baseline_digest = self._restore_baseline()
        baseline_checkpoint = self._state_checkpoint("0000-baseline")
        if baseline_checkpoint["source_archive_sha256"] != baseline_digest:
            raise StandardizedReplayError(
                "baseline metric checkpoint changed the restored source state"
            )
        previous_files = baseline_checkpoint.pop("_snapshot_files")
        baseline_checkpoint.pop("_snapshot_payload")
        baseline_tokens = int(baseline_checkpoint["lean_tokens"])
        expected_final = str(
            self.bundle["artifacts"]["final_source"].get("source_archive_sha256") or ""
        )
        outcomes: list[dict[str, Any]] = []
        points: list[dict[str, Any]] = [
            {
                "edit_index": 0,
                "action_index": 0,
                "elapsed_seconds": 0.0,
                "cost_usd": 0.0,
                "lean_tokens": baseline_tokens,
                "lean_tokens_saved": 0,
                "lean_token_compression_pct": 0.0,
                "files": [],
                "kind": "baseline",
                "exact": True,
                "intermediate_live_agent_state_exact": False,
                **baseline_checkpoint,
            }
        ]
        execution_failure_count = 0
        command_outcome_divergence_count = 0
        unsupported_count = 0
        unresolved_command_count = 0
        replayed_bash = 0
        skipped_bash = 0
        previous_digest = baseline_digest
        edit_index = 0

        previous_edit_cost = 0.0
        for action in self.trajectory.get("actions") or []:
            if not isinstance(action, Mapping):
                continue
            action_index = _usage_token_count(action.get("action_index"))
            action_started = time.monotonic()
            outcome = copy.deepcopy(dict(action))
            cost_boundary = action_cost_boundaries.get(
                action_index,
                {
                    "cost_usd": None,
                    "cumulative_cost_usd": None,
                    "turn_cost_usd": None,
                    "cost_boundary_exact": False,
                    "cost_granularity": "assistant_turn",
                },
            )
            outcome.update(cost_boundary)
            outcome["replay_started_elapsed_seconds"] = round(
                action_started - replay_started, 6
            )
            self._notify_progress(
                {
                    "phase": "action_started",
                    "action_index": action_index,
                    "name": action.get("name"),
                }
            )
            try:
                applied = self._apply_action(action)
                outcome.update(applied)
                if action.get("replay_kind") == "unresolved_command":
                    unresolved_command_count += 1
                outcome["mutation_apply_duration_seconds"] = round(
                    time.monotonic() - action_started, 6
                )
                if action.get("replay_kind") == "bash":
                    if str(applied.get("replay_status") or "").startswith("skipped_"):
                        skipped_bash += 1
                    else:
                        replayed_bash += 1
                if applied.get("outcome_matches") is False:
                    command_outcome_divergence_count += 1
                if applied.get("mutating") and (
                    self.policy.verify_each_mutation
                    or self.policy.build_after_each_mutation
                ):
                    checkpoint_label = f"{action_index:04d}-action-endpoint"
                    checkpoint = self._state_checkpoint(checkpoint_label)
                    snapshot_payload = checkpoint.pop("_snapshot_payload")
                    current_files = self._attach_incremental_diff(
                        checkpoint,
                        previous_files,
                        label=checkpoint_label,
                    )
                    digest = str(checkpoint["source_archive_sha256"])
                    outcome["source_archive_sha256"] = digest
                    outcome["source_changed"] = digest != previous_digest
                    outcome["checkpoint"] = copy.deepcopy(checkpoint)
                    previous_digest = digest
                    previous_files = current_files
                    if outcome["source_changed"]:
                        build = self._run_build(
                            snapshot_payload=snapshot_payload,
                            source_archive_sha256=digest,
                        )
                        if build is not None:
                            outcome["build"] = build
                        edit_index += 1
                        tokens = int(checkpoint["lean_tokens"])
                        boundary_cost = cost_boundary.get("cumulative_cost_usd")
                        marginal_cost = None
                        if (
                            cost_boundary.get("cost_boundary_exact") is True
                            and isinstance(boundary_cost, (int, float))
                        ):
                            marginal_cost = round(
                                float(boundary_cost) - previous_edit_cost, 12
                            )
                            previous_edit_cost = float(boundary_cost)
                        point = {
                            "edit_index": edit_index,
                            "action_index": action_index,
                            "action_name": action.get("name"),
                            "agent_id": action.get("agent_id"),
                            "tool_call_id": action.get("tool_call_id"),
                            "provider_call_id": action.get("provider_call_id"),
                            "elapsed_seconds": round(
                                time.monotonic() - replay_started, 3
                            ),
                            **cost_boundary,
                            "marginal_cost_since_previous_edit_usd": marginal_cost,
                            "lean_tokens": tokens,
                            "lean_tokens_saved": baseline_tokens - tokens,
                            "lean_token_compression_pct": (
                                round(
                                    100
                                    * (baseline_tokens - tokens)
                                    / baseline_tokens,
                                    6,
                                )
                                if baseline_tokens
                                else 0.0
                            ),
                            "files": list(checkpoint["changed_files"]),
                            "declared_action_files": self._action_files(action),
                            "kind": action.get("replay_kind"),
                            "exact": True,
                            "replay_checkpoint_exact": True,
                            "intermediate_live_agent_state_exact": False,
                            **checkpoint,
                        }
                        if build is not None:
                            point["build"] = copy.deepcopy(build)
                        points.append(point)
            except Exception as exc:  # noqa: BLE001
                outcome["replay_status"] = "failed"
                outcome["replay_error"] = str(exc)
                execution_failure_count += 1
                unsupported_count += 1
            outcome["replay_duration_seconds"] = round(
                time.monotonic() - action_started, 6
            )
            outcome["replay_completed_elapsed_seconds"] = round(
                time.monotonic() - replay_started, 6
            )
            outcomes.append(outcome)
            self._notify_progress({"phase": "action_completed", **outcome})

        action_checkpoint = self._state_checkpoint("9999-action-endpoint")
        action_checkpoint.pop("_snapshot_payload")
        self._attach_incremental_diff(
            action_checkpoint,
            previous_files,
            label="9999-action-endpoint",
        )
        action_digest = str(action_checkpoint["source_archive_sha256"])
        unattributed_endpoint_change = bool(
            (
                self.policy.verify_each_mutation
                or self.policy.build_after_each_mutation
            )
            and action_digest != previous_digest
        )
        if unattributed_endpoint_change:
            tokens = int(action_checkpoint["lean_tokens"])
            last_cost_boundary = (
                {
                    key: outcomes[-1].get(key)
                    for key in (
                        "cost_usd",
                        "cumulative_cost_usd",
                        "turn_cost_usd",
                        "cost_boundary_exact",
                        "cost_granularity",
                        "assistant_turn_id",
                        "shared_turn_tool_call_count",
                    )
                    if key in outcomes[-1]
                }
                if outcomes
                else {"cost_usd": 0.0, "cost_boundary_exact": True}
            )
            points.append(
                {
                    "edit_index": edit_index,
                    "action_index": len(outcomes),
                    "elapsed_seconds": round(time.monotonic() - replay_started, 3),
                    **last_cost_boundary,
                    "marginal_cost_since_previous_edit_usd": None,
                    "lean_tokens": tokens,
                    "lean_tokens_saved": baseline_tokens - tokens,
                    "lean_token_compression_pct": (
                        round(
                            100 * (baseline_tokens - tokens) / baseline_tokens,
                            6,
                        )
                        if baseline_tokens
                        else 0.0
                    ),
                    "files": list(action_checkpoint["changed_files"]),
                    "kind": "unattributed_action_endpoint_change",
                    "exact": False,
                    "replay_checkpoint_exact": True,
                    "intermediate_live_agent_state_exact": False,
                    **action_checkpoint,
                }
            )
        edit_points = [
            point
            for point in points
            if int(point.get("edit_index") or 0) > 0
            and point.get("kind") != "unattributed_action_endpoint_change"
        ]
        build_points = [
            point for point in edit_points if isinstance(point.get("build"), dict)
        ]
        build_failure_count = sum(
            not bool(point["build"].get("passed")) for point in build_points
        )
        first_failing_build_edit = next(
            (
                int(point["edit_index"])
                for point in build_points
                if not point["build"].get("passed")
            ),
            None,
        )
        archived_edit_checkpoint_count = sum(
            isinstance(point.get("snapshot_artifact"), Mapping)
            for point in edit_points
        )
        diffed_edit_checkpoint_count = sum(
            isinstance(point.get("incremental_diff_artifact"), Mapping)
            and bool(point["incremental_diff_artifact"].get("path"))
            and bool(point["incremental_diff_artifact"].get("bytes"))
            for point in edit_points
        )
        costed_edit_checkpoint_count = sum(
            point.get("cost_boundary_exact") is True for point in edit_points
        )
        captured_turn_cost = float(
            cost_summary["captured_completed_turn_cost_usd"]
        )
        authoritative_final_cost = (
            self.final_cost_usd if self.final_cost_usd > 0 else None
        )
        reported_final_cost = (
            authoritative_final_cost
            if authoritative_final_cost is not None
            else captured_turn_cost
        )
        unattributed_terminal_cost = (
            round(authoritative_final_cost - captured_turn_cost, 12)
            if authoritative_final_cost is not None
            else None
        )

        self._cleanup_build_environment()
        submitted_digest, reconciliation_basis = self._submitted_reconstruction()
        action_matches = action_digest == expected_final
        submitted_matches = submitted_digest == expected_final
        action_exact = bool(
            action_matches
            and not unattributed_endpoint_change
            and execution_failure_count == 0
            and unsupported_count == 0
            and unresolved_command_count == 0
        )
        command_outcomes_exact = bool(
            execution_failure_count == 0 and command_outcome_divergence_count == 0
        )
        reconciliation_applied = bool(
            self.policy.terminal_reconciliation == "submitted_patch"
            and submitted_matches
            and not action_exact
        )
        endpoint_exact = action_exact or reconciliation_applied
        quality = self.trajectory.get("quality")
        quality = quality if isinstance(quality, Mapping) else {}
        standardized_session_exact = bool(
            action_exact
            and quality.get("native_stream_exact") is True
            and quality.get("mutation_trace_exact") is True
            and quality.get("subagent_lineage_complete") is True
            and quality.get("provider_terminal_success") is True
        )
        if points:
            points[-1]["matches_captured_terminal"] = action_matches
        return {
            "format": "code-harness-standardized-replay-result-v1",
            "instance_id": self.instance_id,
            "quality": (
                "exact_offline_standardized_replay"
                if standardized_session_exact
                else "invalid_standardized_replay"
            ),
            "provider": self.trajectory.get("provider"),
            "session_id": self.trajectory.get("session_id"),
            "metric_basis": (
                "Lean tokens measured from every offline replayed mutation endpoint"
            ),
            "cost_basis": (
                "native completed assistant-turn usage at action boundaries; "
                "no per-tool or per-edit proration"
            ),
            "final_cost_usd": round(reported_final_cost, 12),
            "authoritative_final_cost_usd": (
                round(authoritative_final_cost, 12)
                if authoritative_final_cost is not None
                else None
            ),
            "unattributed_terminal_cost_usd": unattributed_terminal_cost,
            **cost_summary,
            "wall_time_seconds": round(time.monotonic() - replay_started, 3),
            "baseline_lean_tokens": baseline_tokens,
            "edit_count": edit_index,
            "nonzero_edit_count": edit_index > 0,
            "points": points,
            "build_command": self.policy.build_command,
            "build_basis": (
                "separate pinned networkless container for every offline replayed "
                "mutation endpoint"
            ),
            "build_cache_basis": (
                "sequential checkpoint builds reuse only the dedicated build "
                "container's pinned .lake cache; the certified replay container "
                "is never used for compilation"
            ),
            "build_attempt_count": len(build_points),
            "build_failure_count": build_failure_count,
            "every_edit_checkpoint_built": (
                edit_index > 0 and len(build_points) == edit_index
            ),
            "all_checkpoint_builds_passed": (
                edit_index > 0
                and len(build_points) == edit_index
                and build_failure_count == 0
            ),
            "first_failing_build_edit_index": first_failing_build_edit,
            "archived_edit_checkpoint_count": archived_edit_checkpoint_count,
            "diffed_edit_checkpoint_count": diffed_edit_checkpoint_count,
            "costed_edit_checkpoint_count": costed_edit_checkpoint_count,
            "every_edit_checkpoint_archived": (
                edit_index > 0 and archived_edit_checkpoint_count == edit_index
            ),
            "every_edit_checkpoint_diffed": (
                edit_index > 0 and diffed_edit_checkpoint_count == edit_index
            ),
            "every_edit_checkpoint_costed": (
                edit_index > 0 and costed_edit_checkpoint_count == edit_index
            ),
            "checkpoint_directory": (
                str(self.checkpoint_dir) if self.checkpoint_dir is not None else None
            ),
            "action_endpoint_checkpoint": action_checkpoint,
            "unattributed_action_endpoint_source_change": (
                unattributed_endpoint_change
            ),
            "reject_asynchronous_actions": (
                self.policy.reject_asynchronous_actions
            ),
            "bash_replay_policy": self.policy.bash,
            "terminal_reconciliation": self.policy.terminal_reconciliation,
            "actions": outcomes,
            "action_count": len(outcomes),
            "replayed_bash_action_count": replayed_bash,
            "skipped_bash_action_count": skipped_bash,
            "replay_divergence_count": execution_failure_count,
            "command_outcome_divergence_count": command_outcome_divergence_count,
            "command_outcomes_exact": command_outcomes_exact,
            "unsupported_action_count": unsupported_count,
            "unresolved_command_action_count": unresolved_command_count,
            "baseline_source_archive_sha256": baseline_digest,
            "captured_final_source_archive_sha256": expected_final,
            "action_replay_source_archive_sha256": action_digest,
            "submitted_source_archive_sha256": submitted_digest,
            "reconciliation_basis": reconciliation_basis,
            "endpoint_reconstruction_environment": (
                "fresh_pinned_networkless_replay_sandbox"
            ),
            "trajectory_action_replay_exact": action_exact,
            "action_source_endpoint_exact": action_matches,
            "action_boundary_sequence_complete": not unattributed_endpoint_change,
            "endpoint_reconstruction_exact": endpoint_exact,
            "source_diff_exact": submitted_matches,
            "standardized_session_replay_exact": standardized_session_exact,
            "native_stream_exact": quality.get("native_stream_exact") is True,
            "native_tool_trace_exact": quality.get("tool_trace_exact") is True,
            "mutation_trace_exact": quality.get("mutation_trace_exact") is True,
            "subagent_lineage_complete": (
                quality.get("subagent_lineage_complete") is True
            ),
            "provider_terminal_success": quality.get("provider_terminal_success")
            is True,
            "terminal_reconciliation_applied": reconciliation_applied,
            "exactness_basis": "canonical scoped source archive digest equality",
            "intermediate_state_claim": (
                "deterministic offline replay snapshot and incremental diff "
                "recorded after each source-changing action; these are exact "
                "replay states, not live-agent intermediate-state observations"
                if self.policy.verify_each_mutation
                else "offline replayed; only baseline and terminal endpoints verified"
            ),
        }


__all__ = [
    "REPLAY_BUNDLE_FORMAT",
    "ReplayPolicy",
    "StandardizedReplayEngine",
    "StandardizedReplayError",
    "create_replay_bundle",
    "load_replay_bundle",
]
