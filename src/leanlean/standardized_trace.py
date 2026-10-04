"""Post-exit standardization of native coding-harness trajectories.

The exact provider stream remains authoritative.  This module turns either an
exact ``harness-wrapper`` native capture or its older normalized JSONL trace
into one provider-neutral action/agent schema.  Importing an older trace is
supported for analysis, but is never mislabeled as byte-exact.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from leanlean.antigravity_trace import (
    is_antigravity_event,
    standardize_antigravity,
)
from leanlean.claude_code_trace import reconstruct_claude_trace
from leanlean.deepseek_harness_trace import (
    DeepSeekHarnessTraceBuilder,
    is_dsh_stream_record,
)
from leanlean.kimi_code_trace import KimiTraceBuilder
from leanlean.mistral_vibe_trace import (
    is_mistral_vibe_event,
    standardize_mistral_vibe,
)

STANDARDIZED_TRACE_FORMAT = "code-harness-standardized-trace-v1"
PROXY_NATIVE_CAPTURE_FORMAT = "harness-wrapper-native-capture-v1"
LEAN_NATIVE_CAPTURE_FORMAT = "code-harness-native-capture-v1"
CODEX_ROLLOUT_BUNDLE_FORMAT = "codex-native-rollout-bundle-v1"
_ROOT_AGENT_ID = "root"
CODEX_LONG_THREAD_ADVISORY = (
    "Heads up: Long threads and multiple compactions can cause the model"
)


def is_codex_nonterminal_advisory(message: object) -> bool:
    """Return whether a Codex error item is informational, not terminal."""

    return isinstance(message, str) and message.startswith(
        CODEX_LONG_THREAD_ADVISORY
    )
_CLAUDE_NON_MUTATING = {
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
    "taskstop",
    "taskupdate",
    "todowrite",
    "skill",
    "enterplanmode",
    "exitplanmode",
    "askuserquestion",
}
_KIMI_NON_MUTATING = _CLAUDE_NON_MUTATING | {"todolist"}


class StandardizationError(RuntimeError):
    """The capture cannot support a trustworthy standardized trajectory."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", errors="surrogateescape")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StandardizationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StandardizationError(f"expected JSON object at {path}")
    return value


def _artifact_path(root: Path, reference: Any) -> Path:
    if not isinstance(reference, str) or not reference:
        raise StandardizationError("capture artifact has no path")
    candidate = (root / reference).resolve()
    if candidate != root and root not in candidate.parents:
        raise StandardizationError(
            f"capture artifact escapes its directory: {reference}"
        )
    if not candidate.is_file():
        raise StandardizationError(f"capture artifact is missing: {candidate}")
    return candidate


def _verified_artifact(root: Path, metadata: Mapping[str, Any]) -> tuple[Path, bytes]:
    path = _artifact_path(root, metadata.get("path"))
    payload = path.read_bytes()
    expected_bytes = metadata.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != len(payload):
        raise StandardizationError(
            f"capture artifact byte count differs for {path}: "
            f"expected {expected_bytes}, found {len(payload)}"
        )
    expected_digest = str(metadata.get("sha256") or "")
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not expected_digest or actual_digest != expected_digest:
        raise StandardizationError(
            f"capture artifact digest differs for {path}: "
            f"expected {expected_digest or '<missing>'}, found {actual_digest}"
        )
    return path, payload


def _parse_jsonl(payload: bytes, source: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8", errors="surrogateescape"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StandardizationError(
                f"invalid provider JSONL at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise StandardizationError(
                f"non-object provider JSONL at {source}:{line_number}"
            )
        events.append(value)
    return events


def _event_records(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for sequence, event in enumerate(events, 1):
        parent = event.get("parent_tool_use_id")
        records.append(
            {
                "sequence": sequence,
                "type": str(event.get("type") or event.get("method") or "event"),
                "agent_id": str(parent) if parent else _ROOT_AGENT_ID,
                "parent_tool_use_id": parent,
                "payload": copy.deepcopy(dict(event)),
                "payload_sha256": _digest(event),
            }
        )
    return records


def detect_trace_provider(events: Iterable[Mapping[str, Any]]) -> str:
    """Detect a native CLI event dialect without consulting the model name."""

    saw_codex = False
    for event in events:
        if is_antigravity_event(event):
            return "antigravity-cli"
        event_type = str(event.get("type") or "")
        if is_dsh_stream_record(event):
            return "deepseek-harness"
        if is_mistral_vibe_event(event):
            return "mistral-vibe"
        if event_type == "result" or (
            event_type in {"assistant", "user"}
            and isinstance(event.get("message"), Mapping)
        ):
            return "claude-code"
        role = str(event.get("role") or "")
        if role == "assistant" and isinstance(event.get("tool_calls"), list):
            return "kimi-code"
        if role == "tool" and event.get("tool_call_id") is not None:
            return "kimi-code"
        if event_type in {
            "thread.started",
            "turn.started",
            "turn.completed",
            "turn.failed",
            "item.started",
            "item.completed",
        } or isinstance(event.get("item"), Mapping):
            saw_codex = True
    return "codex-cli" if saw_codex else "generic-jsonl"


def _standardize_generic(
    events: list[dict[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve an unknown stream exactly without inventing tool semantics."""

    timed_out = bool(source.get("timed_out"))
    returncode = source.get("returncode")
    provider_terminal_success = bool(not timed_out and returncode == 0)
    return {
        "provider": "generic-jsonl",
        "session_id": source.get("session_id"),
        "events": _event_records(events),
        "messages": [],
        "agents": [
            {
                "agent_id": _ROOT_AGENT_ID,
                "parent_agent_id": None,
                "spawn_sequence": None,
            }
        ],
        "actions": [],
        "terminal": {
            "status": (
                "timed_out"
                if timed_out
                else "completed"
                if provider_terminal_success
                else "failed"
            ),
            "timed_out": timed_out,
            "returncode": returncode,
        },
        "pending_tools": [],
        "quality": {
            "native_stream_exact": bool(source.get("native_stream_exact")),
            "tool_trace_exact": False,
            "mutation_trace_exact": False,
            "subagent_lineage_complete": False,
            "completed_action_order_exact": False,
            "filesystem_reconstruction_exact": None,
            "lineage_issues": [
                {
                    "issue": "unrecognized_native_event_schema",
                    "detail": (
                        "raw events are exact; action and subagent semantics were "
                        "not inferred"
                    ),
                }
            ],
            "provider_quality": "exact_raw_stream_unknown_semantics",
            "provider_terminal_success": provider_terminal_success,
            "all_actions_failed": False,
        },
    }


def _claude_replay_kind(name: Any) -> str:
    tool = str(name or "").lower()
    if tool in {"edit", "write", "bash"}:
        return tool
    if tool in _CLAUDE_NON_MUTATING:
        return "non_mutating"
    return "unsupported"


def _standardize_claude(
    events: list[dict[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    native = reconstruct_claude_trace(events)
    agents = copy.deepcopy(native["agents"])
    parent_by_agent = {
        str(agent.get("agent_id") or _ROOT_AGENT_ID): agent.get("parent_agent_id")
        for agent in agents
    }
    lineage_issues = [
        {
            "agent_id": agent.get("agent_id"),
            "issue": "missing_spawn_event",
        }
        for agent in agents
        if agent.get("agent_id") != _ROOT_AGENT_ID
        and agent.get("spawn_sequence") is None
    ]
    actions = []
    for action in native["actions"]:
        agent_id = str(action.get("agent_id") or _ROOT_AGENT_ID)
        actions.append(
            {
                "action_index": int(action["action_index"]),
                "tool_call_id": str(action.get("tool_use_id") or ""),
                "name": str(action.get("name") or "unknown"),
                "arguments": copy.deepcopy(action.get("input")),
                "result": copy.deepcopy(action.get("result")),
                "result_block": copy.deepcopy(action.get("result_block")),
                "status": str(action.get("status") or ""),
                "is_error": action.get("is_error"),
                "agent_id": agent_id,
                "parent_agent_id": parent_by_agent.get(agent_id),
                "parent_tool_use_id": action.get("parent_tool_use_id"),
                "started_sequence": action.get("started_sequence"),
                "completed_sequence": action.get("completed_sequence"),
                "ordering_basis": "native completed-result receipt order",
                "replay_kind": _claude_replay_kind(action.get("name")),
            }
        )
    exact_native = bool(source.get("native_stream_exact"))
    forwarded = bool(source.get("forwards_subagent_stream"))
    tool_trace_exact = bool(
        exact_native
        and forwarded
        and source.get("native_edit_payloads_exact", True) is True
        and native["quality"] == "ordered_complete_tool_trace"
        and not lineage_issues
    )
    terminal = copy.deepcopy(native.get("terminal") or {})
    all_actions_failed = bool(actions) and all(
        action.get("status") == "failed" or action.get("is_error") is True
        for action in actions
    )
    provider_terminal_success = (
        bool(terminal)
        and not bool(terminal.get("is_error") or terminal.get("error"))
        and not all_actions_failed
    )
    return {
        "provider": "claude-code",
        "session_id": native.get("terminal", {}).get("session_id")
        or source.get("session_id"),
        "events": _event_records(events),
        "messages": copy.deepcopy(native.get("messages") or []),
        "agents": agents,
        "actions": actions,
        "terminal": terminal,
        "pending_tools": copy.deepcopy(native.get("pending_tools") or []),
        "quality": {
            "native_stream_exact": exact_native,
            "tool_trace_exact": tool_trace_exact,
            "mutation_trace_exact": tool_trace_exact,
            "subagent_lineage_complete": not lineage_issues and forwarded,
            "completed_action_order_exact": exact_native,
            "filesystem_reconstruction_exact": None,
            "lineage_issues": lineage_issues,
            "provider_quality": native["quality"],
            "provider_terminal_success": provider_terminal_success,
            "all_actions_failed": all_actions_failed,
        },
    }


def _standardize_deepseek_harness(
    events: list[dict[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    builder = DeepSeekHarnessTraceBuilder()
    for event in events:
        builder.ingest(event)
    native = builder.to_trace(
        native_stream_exact=bool(source.get("native_stream_exact"))
    )
    actions = []
    for action in native["actions"]:
        name = str(action.get("name") or "unknown")
        replay_kind = (
            "bash"
            if name.lower() == "bash"
            else "edit"
            if name.lower() == "str_replace_editor"
            else "unsupported"
        )
        actions.append(
            {
                "action_index": int(action["action_index"]),
                "tool_call_id": str(action.get("tool_call_id") or ""),
                "name": name,
                "arguments": copy.deepcopy(action.get("arguments")),
                "result": copy.deepcopy(action.get("result")),
                "result_block": copy.deepcopy(action.get("result_block")),
                "status": str(action.get("status") or ""),
                "is_error": action.get("is_error"),
                "agent_id": str(action.get("agent_id") or _ROOT_AGENT_ID),
                "parent_agent_id": None,
                "parent_tool_use_id": None,
                "started_sequence": action.get("started_sequence"),
                "completed_sequence": action.get("completed_sequence"),
                "ordering_basis": "native tool-result receipt order",
                "replay_kind": replay_kind,
            }
        )
    terminal = copy.deepcopy(native.get("terminal") or {})
    timed_out = bool(source.get("timed_out"))
    returncode = source.get("returncode")
    provider_terminal_success = bool(
        terminal.get("success") and not timed_out and returncode in {None, 0}
    )
    terminal.update(
        {
            "timed_out": timed_out,
            "returncode": returncode,
            "success": provider_terminal_success,
        }
    )
    quality = copy.deepcopy(native.get("quality") or {})
    quality.update(
        {
            "filesystem_reconstruction_exact": None,
            "provider_terminal_success": provider_terminal_success,
        }
    )
    return {
        "provider": "deepseek-harness",
        "session_id": native.get("session_id") or source.get("session_id"),
        "events": _event_records(events),
        "messages": copy.deepcopy(native.get("messages") or []),
        "agents": [
            {
                "agent_id": _ROOT_AGENT_ID,
                "parent_agent_id": None,
                "spawn_sequence": None,
            }
        ],
        "actions": actions,
        "terminal": terminal,
        "pending_tools": copy.deepcopy(native.get("pending_tools") or []),
        "quality": quality,
    }


def _kimi_replay_kind(name: Any) -> str:
    tool = str(name or "").lower()
    if tool == "agentswarm":
        return "unsupported_subagent"
    if tool in _KIMI_NON_MUTATING:
        return "non_mutating"
    return _claude_replay_kind(name)


def _standardize_kimi(
    events: list[dict[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    builder = KimiTraceBuilder()
    actions: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    provider_errors: list[dict[str, Any]] = []
    session_id = source.get("session_id")
    for sequence, event in enumerate(events, 1):
        role = str(event.get("role") or "")
        event_type = str(event.get("type") or "")
        if role in {"assistant", "tool"}:
            messages.append(
                {
                    "sequence": sequence,
                    "role": role,
                    "content": copy.deepcopy(event.get("content")),
                    "tool_calls": copy.deepcopy(event.get("tool_calls") or []),
                    "tool_call_id": event.get("tool_call_id"),
                }
            )
        if session_id is None and event.get("session_id"):
            session_id = event.get("session_id")
        if role == "error" or event_type == "error" or event_type.endswith(".error"):
            provider_errors.append(copy.deepcopy(dict(event)))
        for completed in builder.ingest(event):
            name = str(completed.get("name") or "unknown")
            actions.append(
                {
                    "action_index": int(completed["action_index"]),
                    "tool_call_id": str(completed.get("tool_use_id") or ""),
                    "name": name,
                    "arguments": copy.deepcopy(completed.get("input")),
                    "result": copy.deepcopy(completed.get("result")),
                    "result_block": copy.deepcopy(completed.get("result_block")),
                    "status": str(completed.get("status") or ""),
                    "is_error": completed.get("is_error"),
                    "agent_id": _ROOT_AGENT_ID,
                    "parent_agent_id": None,
                    "parent_tool_use_id": None,
                    "started_sequence": completed.get("started_sequence"),
                    "completed_sequence": completed.get("completed_sequence"),
                    "ordering_basis": "native tool-result receipt order",
                    "replay_kind": _kimi_replay_kind(name),
                    "collaboration_event": name.lower() == "agentswarm",
                }
            )

    pending_tools = builder.pending_tool_uses
    swarm_actions = [
        action for action in actions if action.get("collaboration_event") is True
    ]
    lineage_issues = [
        {
            "tool_call_id": action["tool_call_id"],
            "issue": "subagent_actions_not_forwarded_by_kimi_stream",
        }
        for action in swarm_actions
    ]
    native_exact = bool(source.get("native_stream_exact"))
    tool_trace_exact = bool(
        native_exact
        and not pending_tools
        and not builder.duplicate_tool_call_ids
        and not builder.orphan_tool_result_ids
    )
    timed_out = bool(source.get("timed_out"))
    returncode = source.get("returncode")
    all_actions_failed = bool(actions) and all(
        action.get("status") == "failed" or action.get("is_error") is True
        for action in actions
    )
    provider_terminal_success = bool(
        not timed_out
        and returncode == 0
        and not provider_errors
        and not all_actions_failed
    )
    terminal = {
        "status": (
            "timed_out"
            if timed_out
            else "completed"
            if provider_terminal_success
            else "failed"
        ),
        "timed_out": timed_out,
        "returncode": returncode,
        "errors": provider_errors,
    }
    return {
        "provider": "kimi-code",
        "session_id": session_id,
        "events": _event_records(events),
        "messages": messages,
        "agents": [
            {
                "agent_id": _ROOT_AGENT_ID,
                "parent_agent_id": None,
                "spawn_sequence": None,
            }
        ],
        "actions": actions,
        "terminal": terminal,
        "pending_tools": pending_tools,
        "quality": {
            "native_stream_exact": native_exact,
            "tool_trace_exact": tool_trace_exact,
            "mutation_trace_exact": tool_trace_exact and not swarm_actions,
            "subagent_lineage_complete": not swarm_actions,
            "completed_action_order_exact": native_exact,
            "filesystem_reconstruction_exact": None,
            "lineage_issues": lineage_issues,
            "provider_terminal_success": provider_terminal_success,
            "provider_errors": provider_errors,
            "all_actions_failed": all_actions_failed,
            "duplicate_tool_call_ids": list(builder.duplicate_tool_call_ids),
            "orphan_tool_result_ids": list(builder.orphan_tool_result_ids),
        },
    }


def _codex_item_type(item: Mapping[str, Any]) -> str:
    native = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        str(item.get("type") or "item"),
    )
    return re.sub(r"[^a-z0-9]+", "_", native.lower()).strip("_")


def _codex_agent(
    item: Mapping[str, Any], event: Mapping[str, Any]
) -> tuple[str, str | None]:
    agent = item.get("agent_id") or item.get("agentId") or event.get("agent_id")
    parent = (
        item.get("parent_agent_id")
        or item.get("parentAgentId")
        or event.get("parent_agent_id")
    )
    return str(agent or _ROOT_AGENT_ID), str(parent) if parent else None


def _codex_changes(item: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    changes = [
        copy.deepcopy(dict(change))
        for change in item.get("changes") or []
        if isinstance(change, Mapping)
    ]
    exact = bool(changes) and all(
        isinstance(change.get("diff"), str) for change in changes
    )
    return changes, exact


def _codex_action(
    item: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    action_index: int,
    completed_sequence: int,
) -> dict[str, Any] | None:
    item_type = _codex_item_type(item)
    agent_id, parent_agent_id = _codex_agent(item, event)
    common = {
        "action_index": action_index,
        "tool_call_id": str(item.get("id") or ""),
        "status": str(item.get("status") or "completed"),
        "agent_id": agent_id,
        "parent_agent_id": parent_agent_id,
        "parent_tool_use_id": item.get("parent_tool_use_id"),
        "started_sequence": None,
        "completed_sequence": completed_sequence,
        "ordering_basis": "native item.completed receipt order",
        "result": item.get("aggregated_output", item.get("result")),
    }
    if item_type == "command_execution":
        return {
            **common,
            "name": "Bash",
            "arguments": {
                "command": item.get("command"),
                "cwd": item.get("cwd") or "/testbed",
            },
            "observed_returncode": item.get("exit_code", item.get("exitCode")),
            "replay_kind": "bash"
            if isinstance(item.get("command"), str)
            else "unsupported",
        }
    if item_type == "file_change":
        changes, exact = _codex_changes(item)
        return {
            **common,
            "name": "FileChange",
            "arguments": {"changes": changes},
            "contains_exact_edit_payload": exact,
            "replay_kind": "patch" if exact else "unsupported_file_change",
        }
    if item_type == "mcp_tool_call":
        server = str(item.get("server") or "")
        tool = str(item.get("tool") or item.get("name") or "")
        return {
            **common,
            "name": tool or server or item_type,
            "arguments": copy.deepcopy(item.get("arguments", item.get("input"))),
            "error": copy.deepcopy(item.get("error")),
            "is_error": bool(item.get("error"))
            or str(item.get("status") or "").lower() in {"failed", "error"},
            "mcp_server": server,
            "mcp_tool": tool,
            "replay_kind": "mcp",
        }
    if item_type == "tool_call":
        return {
            **common,
            "name": str(item.get("name") or item_type),
            "arguments": copy.deepcopy(item.get("arguments", item.get("input"))),
            "replay_kind": "unsupported_tool",
        }
    if item_type in {"web_search", "todo_list"}:
        return {
            **common,
            "name": str(item.get("name") or item_type),
            "arguments": copy.deepcopy(item),
            "replay_kind": "non_mutating",
        }
    if "collab" in item_type or "subagent" in item_type:
        return {
            **common,
            "name": str(item.get("name") or item_type),
            "arguments": copy.deepcopy(item),
            "replay_kind": "non_mutating",
            "collaboration_event": True,
        }
    return None


def _standardize_codex(
    events: list[dict[str, Any]], source: Mapping[str, Any], *, app_server: bool
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    agents: dict[str, dict[str, Any]] = {
        _ROOT_AGENT_ID: {
            "agent_id": _ROOT_AGENT_ID,
            "parent_agent_id": None,
            "spawn_sequence": None,
        }
    }
    terminal: dict[str, Any] = {}
    cumulative_diffs = []
    collaboration_without_lineage = []
    provider_errors = []
    provider_warnings = []
    for sequence, event in enumerate(events, 1):
        native_type = str(event.get("type") or event.get("method") or "event")
        if app_server:
            params = event.get("params")
            params = params if isinstance(params, Mapping) else {}
            if native_type == "turn/diff/updated" and isinstance(
                params.get("diff"), str
            ):
                cumulative_diffs.append(
                    {
                        "sequence": sequence,
                        "diff": params["diff"],
                        "diff_sha256": hashlib.sha256(
                            params["diff"].encode("utf-8", errors="surrogateescape")
                        ).hexdigest(),
                    }
                )
            item = params.get("item") if native_type == "item/completed" else None
            if native_type == "turn/completed":
                terminal = copy.deepcopy(dict(params))
        else:
            item = event.get("item") if native_type == "item.completed" else None
            if native_type in {"thread.started", "turn.completed", "turn.failed"}:
                terminal.update(copy.deepcopy(event))
        if isinstance(item, Mapping):
            item_type = _codex_item_type(item)
            if item_type == "error":
                message = copy.deepcopy(item.get("message") or dict(item))
                if is_codex_nonterminal_advisory(message):
                    provider_warnings.append(message)
                else:
                    provider_errors.append(message)
            if item_type == "agent_message":
                messages.append(
                    {
                        "sequence": sequence,
                        "role": "assistant",
                        "content": item.get("text", ""),
                    }
                )
            action = _codex_action(
                item,
                event,
                action_index=len(actions) + 1,
                completed_sequence=sequence,
            )
            if action is not None:
                actions.append(action)
                agent_id = action["agent_id"]
                agents.setdefault(
                    agent_id,
                    {
                        "agent_id": agent_id,
                        "parent_agent_id": action.get("parent_agent_id"),
                        "spawn_sequence": sequence,
                    },
                )
                if action.get("collaboration_event") and (
                    agent_id == _ROOT_AGENT_ID or not action.get("parent_agent_id")
                ):
                    collaboration_without_lineage.append(action["tool_call_id"])
        if native_type in {"turn.failed", "turn/failed", "error"}:
            provider_errors.append(copy.deepcopy(dict(event)))
    file_changes = [action for action in actions if action["name"] == "FileChange"]
    exact_file_changes = [
        action for action in file_changes if action.get("contains_exact_edit_payload")
    ]
    native_exact = bool(source.get("native_stream_exact"))
    terminal_diff = cumulative_diffs[-1] if cumulative_diffs else None
    edit_evidence = {
        "file_change_count": len(file_changes),
        "exact_file_change_count": len(exact_file_changes),
        "all_file_changes_include_diff": len(file_changes) == len(exact_file_changes),
        "cumulative_diff_count": len(cumulative_diffs),
        "terminal_cumulative_diff_exact": bool(terminal_diff and native_exact),
        "ordinary_exec_json_contains_exact_edits": (
            bool(file_changes) and len(file_changes) == len(exact_file_changes)
            if not app_server
            else None
        ),
    }
    terminal_status = str(
        terminal.get("status")
        or (
            terminal.get("turn", {}).get("status")
            if isinstance(terminal.get("turn"), Mapping)
            else ""
        )
        or ""
    ).lower()
    all_actions_failed = bool(actions) and all(
        str(action.get("status") or "").lower() == "failed" for action in actions
    )
    provider_terminal_success = (
        bool(terminal)
        and not provider_errors
        and not all_actions_failed
        and (terminal_status not in {"failed", "error", "cancelled", "canceled"})
    )
    return {
        "provider": "codex-app-server" if app_server else "codex-cli",
        "session_id": source.get("session_id")
        or terminal.get("thread_id")
        or terminal.get("threadId"),
        "events": _event_records(events),
        "messages": messages,
        "agents": list(agents.values()),
        "actions": actions,
        "terminal": terminal,
        "cumulative_diffs": cumulative_diffs,
        "terminal_cumulative_diff": terminal_diff,
        "codex_edit_evidence": edit_evidence,
        "quality": {
            "native_stream_exact": native_exact,
            "tool_trace_exact": bool(
                app_server
                and native_exact
                and not collaboration_without_lineage
                and edit_evidence["all_file_changes_include_diff"]
            ),
            "mutation_trace_exact": bool(
                app_server
                and native_exact
                and not collaboration_without_lineage
                and edit_evidence["all_file_changes_include_diff"]
            ),
            "subagent_lineage_complete": not collaboration_without_lineage,
            "completed_action_order_exact": native_exact,
            "filesystem_reconstruction_exact": None,
            "lineage_issues": [
                {"tool_call_id": item, "issue": "collaboration_event_has_no_lineage"}
                for item in collaboration_without_lineage
            ],
            "provider_terminal_success": provider_terminal_success,
            "provider_errors": provider_errors,
            "provider_warnings": provider_warnings,
            "all_actions_failed": all_actions_failed,
        },
    }


def _normalized_codex_path(value: Any) -> str:
    path = str(value or "")
    return path.removeprefix("/testbed/").lstrip("./")


def _codex_change_paths(changes: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            _normalized_codex_path(change.get("path"))
            for change in changes
            if change.get("path")
        )
    )


def _decode_codex_template_literal(
    source: str, *, raw: bool = False
) -> str:
    if not source.startswith("`"):
        raise StandardizationError("Codex command is not a template literal")
    result: list[str] = []
    position = 1
    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
        "v": "\v",
        "0": "\0",
        "\\": "\\",
        "`": "`",
        "$": "$",
    }
    while position < len(source):
        character = source[position]
        if character == "`":
            return "".join(result)
        if character != "\\":
            result.append(character)
            position += 1
            continue
        position += 1
        if position >= len(source):
            raise StandardizationError("Codex command template ends in an escape")
        escaped = source[position]
        if raw:
            result.extend(("\\", escaped))
            position += 1
            continue
        if escaped in {"\n", "\r"}:
            if escaped == "\r" and source[position + 1 : position + 2] == "\n":
                position += 1
            position += 1
            continue
        if escaped == "x":
            digits = source[position + 1 : position + 3]
            if len(digits) != 2 or not re.fullmatch(r"[0-9A-Fa-f]{2}", digits):
                raise StandardizationError("Codex command has an invalid hex escape")
            result.append(chr(int(digits, 16)))
            position += 3
            continue
        if escaped == "u":
            digits = source[position + 1 : position + 5]
            if len(digits) != 4 or not re.fullmatch(r"[0-9A-Fa-f]{4}", digits):
                raise StandardizationError(
                    "Codex command has an invalid Unicode escape"
                )
            result.append(chr(int(digits, 16)))
            position += 5
            continue
        result.append(escapes.get(escaped, escaped))
        position += 1
    raise StandardizationError("Codex command template literal is unterminated")


def _codex_js_string_field(source: str, field: str) -> str | None:
    pattern = re.compile(
        rf'(?<![A-Za-z0-9_$])(?:"{re.escape(field)}"|{re.escape(field)})\s*:\s*'
    )
    match = pattern.search(source)
    if match is None:
        return None
    value_source = source[match.end() :]
    raw_template = value_source.startswith("String.raw")
    if raw_template:
        value_source = value_source.removeprefix("String.raw").lstrip()
    if value_source.startswith("`"):
        value = _decode_codex_template_literal(
            value_source, raw=raw_template
        )
    elif value_source.startswith('"'):
        try:
            value, _ = json.JSONDecoder().raw_decode(value_source)
        except json.JSONDecodeError as exc:
            raise StandardizationError(
                f"Codex exec_command field {field!r} is malformed"
            ) from exc
    else:
        raise StandardizationError(
            f"Codex exec_command field {field!r} is not a static string"
        )
    if not isinstance(value, str):
        raise StandardizationError(
            f"Codex exec_command field {field!r} is not a string"
        )
    return value


def _codex_js_identifier_field(source: str, field: str) -> str | None:
    pattern = re.compile(
        rf'(?<![A-Za-z0-9_$])(?:"{re.escape(field)}"|{re.escape(field)})\s*:\s*'
        r'([A-Za-z_$][A-Za-z0-9_$]*)'
    )
    match = pattern.search(source)
    return match.group(1) if match else None


def _codex_js_has_shorthand_field(source: str, field: str) -> bool:
    return bool(
        re.search(
            rf"(?:^|[{{,])\s*{re.escape(field)}\s*(?=[,}}])",
            source,
        )
    )


def _codex_static_array_binding(source: str, name: str) -> list[Any] | None:
    """Decode a JSON-compatible ``const name = [...]`` binding."""

    binding = re.search(
        rf"\bconst\s+{re.escape(name)}\s*=\s*",
        source,
    )
    if binding is None:
        return None
    position = binding.end()
    if position >= len(source) or source[position] != "[":
        return None
    start = position
    depth = 0
    quote: str | None = None
    escaped = False
    while position < len(source):
        character = source[position]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character == '"':
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(source[start : position + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, list) else None
        position += 1
    return None


def _codex_destructured_command_values(
    source: str, command_name: str
) -> list[str] | None:
    """Expand commands from a static ``rows.map(([..., cmd]) => ...)`` cell."""

    mapping = re.search(
        r"\b([A-Za-z_$][A-Za-z0-9_$]*)\.map\s*\(\s*"
        r"\(\s*\[([^\]]+)\]\s*\)\s*=>\s*tools\.exec_command\s*\(",
        source,
    )
    if mapping is None:
        return None
    names = [item.strip() for item in mapping.group(2).split(",")]
    if command_name not in names:
        return None
    command_index = names.index(command_name)
    rows = _codex_static_array_binding(source, mapping.group(1))
    if not rows:
        return None
    commands: list[str] = []
    for row in rows:
        if (
            not isinstance(row, list)
            or command_index >= len(row)
            or not isinstance(row[command_index], str)
        ):
            return None
        commands.append(row[command_index])
    return commands


def _codex_template_has_interpolation(source: str) -> bool:
    if not source.startswith("`"):
        return False
    escaped = False
    for position, character in enumerate(source[1:], 1):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "`":
            return False
        elif character == "$" and source[position + 1 : position + 2] == "{":
            return True
    raise StandardizationError("Codex command template literal is unterminated")


def _codex_js_template_field_is_dynamic(source: str, field: str) -> bool:
    pattern = re.compile(
        rf'(?<![A-Za-z0-9_$])(?:"{re.escape(field)}"|{re.escape(field)})\s*:\s*'
    )
    match = pattern.search(source)
    if match is None:
        return False
    value_source = source[match.end() :]
    if value_source.startswith("String.raw"):
        value_source = value_source.removeprefix("String.raw").lstrip()
    return _codex_template_has_interpolation(value_source)


def _codex_string_binding_is_dynamic(source: str, name: str) -> bool:
    binding = re.search(
        rf'\bconst\s+{re.escape(name)}\s*=\s*(?:String\.raw)?\s*', source
    )
    if binding is None:
        raise StandardizationError(
            f"Codex exec_command string binding {name!r} is missing"
        )
    return _codex_template_has_interpolation(source[binding.end() :])


def _codex_resolve_string_binding(source: str, name: str) -> str:
    binding = re.search(
        rf'\bconst\s+{re.escape(name)}\s*=\s*(?:(String\.raw)\s*)?', source
    )
    if binding is None:
        raise StandardizationError(
            f"Codex exec_command string binding {name!r} is missing"
        )
    return (
        _codex_js_string_field(
            "value:" + source[binding.end() :],
            "value",
        )
        if binding.group(1) is None
        else _codex_js_string_field(
            "value:String.raw" + source[binding.end() :],
            "value",
        )
    ) or ""


def _codex_exec_requests(source: str) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    cursor = 0
    pattern = re.compile(r"\btools\.exec_command\s*\(")
    while match := pattern.search(source, cursor):
        position = match.end()
        while position < len(source) and source[position].isspace():
            position += 1
        if position >= len(source) or source[position] != "{":
            raise StandardizationError(
                "Codex tools.exec_command does not use an inline object literal"
            )
        start = position
        depth = 0
        quote: str | None = None
        escaped = False
        while position < len(source):
            character = source[position]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in {'"', "'", "`"}:
                # Some persisted Codex exec cells contain a tolerated trailing
                # quote on an otherwise bare option key (for example,
                # ``max_output_tokens":``). It is outside the command
                # payload and must not hide the object's closing brace.
                trailing_bare_key_quote = bool(
                    character in {'"', "'"}
                    and position > start
                    and re.match(r"[A-Za-z0-9_$]", source[position - 1])
                    and source[position + 1 : position + 2] == ":"
                )
                if not trailing_bare_key_quote:
                    quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    break
            position += 1
        if depth != 0 or position >= len(source):
            raise StandardizationError("Codex exec_command object is unterminated")
        object_source = source[start : position + 1]
        dynamic_template = _codex_js_template_field_is_dynamic(
            object_source, "cmd"
        )
        try:
            command = _codex_js_string_field(object_source, "cmd")
        except StandardizationError:
            command = None
        expanded_commands: list[str] | None = None
        if command is None:
            command_name = _codex_js_identifier_field(object_source, "cmd")
            if command_name is None and _codex_js_has_shorthand_field(
                object_source, "cmd"
            ):
                command_name = "cmd"
            if command_name is None:
                raise StandardizationError(
                    "Codex exec_command object has no static cmd field"
                )
            try:
                dynamic_template = _codex_string_binding_is_dynamic(
                    source, command_name
                )
                command = _codex_resolve_string_binding(source, command_name)
            except StandardizationError:
                expanded_commands = _codex_destructured_command_values(
                    source, command_name
                )
                if expanded_commands is None:
                    raise
                dynamic_template = False
                command = expanded_commands[0]
        try:
            workdir = _codex_js_string_field(object_source, "workdir")
        except StandardizationError:
            workdir_name = _codex_js_identifier_field(object_source, "workdir")
            if workdir_name is None and _codex_js_has_shorthand_field(
                object_source, "workdir"
            ):
                workdir_name = "workdir"
            if workdir_name is None:
                raise
            workdir = _codex_resolve_string_binding(source, workdir_name)
        else:
            if workdir is None:
                workdir_name = _codex_js_identifier_field(
                    object_source, "workdir"
                )
                if workdir_name is None and _codex_js_has_shorthand_field(
                    object_source, "workdir"
                ):
                    workdir_name = "workdir"
                workdir = (
                    _codex_resolve_string_binding(source, workdir_name)
                    if workdir_name is not None
                    else "/testbed"
                )
        requests.extend(
            {
                "command": expanded_command,
                "cwd": workdir,
                "dynamic_template": dynamic_template,
            }
            for expanded_command in (expanded_commands or [command])
        )
        cursor = position + 1
    return requests


def _codex_dynamic_command_is_read_only(command: str, source: str) -> bool:
    stripped = command.strip()
    mutating = re.search(
        r"(?:^|[\s;|&])(?:rm|mv|cp|install|touch|truncate|tee)\b"
        r"|(?:sed|perl)\s+[^\n;]*-[^\s;]*i"
        r"|(?:^|[^<])>(?![>&])"
        r"|apply_patch|write_text|write_bytes|unlink\s*\(|remove\s*\(",
        stripped,
    )
    if mutating:
        return False
    if re.match(
        r"^(?:rg\b|wc\s+-l\b|git\s+(?:show|diff|status|ls-files)\b|"
        r"sed\s+-n\b|awk\b|"
        r"perl\s+[^\n;]*-[^\s;]*n)",
        stripped,
    ):
        return True
    if stripped.startswith("${perl}"):
        return bool(re.search(r"const\s+perl\s*=.*perl[^`\n]*-[^`\n]*n", source))
    # Codex sometimes builds the Perl program in a JS variable and interpolates
    # it into this exact source-metric pipeline.  The template is dynamic, but
    # every shell stage is read-only.  Keep this deliberately narrow: allowing
    # arbitrary ``find`` templates would also admit mutating actions such as
    # ``-delete`` and ``-exec``.
    if re.fullmatch(
        r"find\s+[^\n|]+\s+-name\s+['\"]\*\.lean['\"]\s+-type\s+f\s+"
        r"-print0\s*\|\s*xargs\s+-0\s+cat\s*\|\s*"
        r"perl\s+-0777\s+-ne\s+['\"]\$\{[A-Za-z_$][A-Za-z0-9_$]*\}['\"]",
        stripped,
    ):
        return True
    if stripped.startswith("python3 -c '${script}'"):
        script_is_metric = bool(
            "const script = String.raw`" in source
            and "print(" in source
            and not re.search(
                r"\.write\s*\(|open\s*\([^\n]*,[^\n]*['\"]w|"
                r"subprocess|os\.(?:remove|rename|replace|unlink|system)",
                source,
            )
        )
        return script_is_metric
    lake_probe = re.fullmatch(
        r"lake\s+env\s+lean\s+\$\{([A-Za-z_$][A-Za-z0-9_$]*)\}",
        stripped,
    )
    if lake_probe is not None:
        variable = lake_probe.group(1)
        mapping = re.search(
            rf"\b([A-Za-z_$][A-Za-z0-9_$]*)\.map\s*\(\s*{re.escape(variable)}\s*=>",
            source,
        )
        values = (
            _codex_static_array_binding(source, mapping.group(1))
            if mapping is not None
            else None
        )
        return bool(
            values
            and all(
                isinstance(value, str)
                and bool(re.fullmatch(r"[A-Za-z0-9_./+-]+\.lean", value))
                and ".." not in PurePosixPath(value).parts
                for value in values
            )
        )
    ruby_metric = re.fullmatch(
        r"ruby\s+-e\s+\$\{JSON\.stringify\(([A-Za-z_$][A-Za-z0-9_$]*)\)\}",
        stripped,
    )
    if ruby_metric is not None:
        try:
            script = _codex_resolve_string_binding(source, ruby_metric.group(1))
        except StandardizationError:
            return False
        return bool(
            "File.read(" in script
            and ("puts " in script or "puts(" in script)
            and not re.search(
                r"\b(?:File|IO)\.(?:write|open|delete|unlink|rename|truncate|popen)\b|"
                r"\b(?:system|exec|spawn|fork)\s*\(|`",
                script,
            )
        )
    return False


def _verified_codex_rollout_stream(
    root: Path, metadata: Mapping[str, Any]
) -> tuple[Path, bytes]:
    path = _artifact_path(root, metadata.get("path"))
    archive = path.read_bytes()
    expected_archive = metadata.get("archive_bytes")
    if expected_archive is not None and int(expected_archive) != len(archive):
        raise StandardizationError(f"Codex rollout archive byte count differs: {path}")
    try:
        payload = gzip.decompress(archive)
    except (OSError, EOFError) as exc:
        raise StandardizationError(f"invalid Codex rollout gzip {path}: {exc}") from exc
    expected_bytes = int(metadata.get("bytes", -1))
    actual_digest = hashlib.sha256(payload).hexdigest()
    if expected_bytes != len(payload) or actual_digest != metadata.get("sha256"):
        raise StandardizationError(f"Codex rollout content digest differs: {path}")
    return path, payload


def _codex_rollout_turn_messages(
    events: list[dict[str, Any]], thread_id: str
) -> list[dict[str, Any]]:
    """Convert provider-native token counters into exact assistant turns."""

    model = ""
    for event in events:
        payload = event.get("payload")
        if (
            event.get("type") == "turn_context"
            and isinstance(payload, Mapping)
            and payload.get("model")
        ):
            model = str(payload["model"])
            break

    messages: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    pending_first_sequence: int | None = None
    previous_total_usage: dict[str, int] = {}
    for local_sequence, event in enumerate(events, 1):
        payload = event.get("payload")
        if event.get("type") == "turn_context" and isinstance(payload, Mapping) and payload.get("model"):
            model = str(payload["model"])
        if event.get("type") == "response_item" and isinstance(payload, Mapping):
            if pending_first_sequence is None:
                pending_first_sequence = local_sequence
            if payload.get("type") in {"custom_tool_call", "function_call"}:
                call_id = str(payload.get("call_id") or "")
                if call_id:
                    pending_call_ids.append(call_id)

        is_token_count = (
            event.get("type") == "event_msg"
            and isinstance(payload, Mapping)
            and payload.get("type") == "token_count"
        )
        if not is_token_count:
            continue
        info = payload.get("info")
        usage = None
        if isinstance(info, Mapping):
            total_usage = info.get("total_token_usage")
            if isinstance(total_usage, Mapping):
                usage = {
                    field: int(total_usage.get(field) or 0) - previous_total_usage.get(field, 0)
                    for field in (
                        "input_tokens",
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                        "output_tokens",
                    )
                }
                if any(value < 0 for value in usage.values()):
                    raise ValueError("native Codex cumulative usage counters decreased")
                previous_total_usage = {
                    field: int(total_usage.get(field) or 0) for field in usage
                }
            else:
                usage = info.get("last_token_usage")
        if isinstance(usage, Mapping) and model:
            call_ids = sorted(set(pending_call_ids))
            identity = json.dumps(
                {
                    "timestamp": str(event.get("timestamp") or ""),
                    "usage": dict(usage),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            native_turn_id = (
                "codex-token-count:" + hashlib.sha256(identity).hexdigest()
            )
            trace_sequence = pending_first_sequence or local_sequence
            messages.append(
                {
                    "role": "assistant",
                    "id": native_turn_id,
                    "model": model,
                    "usage": {
                        "prompt_tokens": int(usage.get("input_tokens") or 0),
                        "completion_tokens": int(usage.get("output_tokens") or 0),
                        "prompt_tokens_details": {
                            "cached_tokens": int(
                                usage.get("cached_input_tokens") or 0
                            ),
                            "cache_creation_tokens": int(
                                usage.get("cache_write_input_tokens") or 0
                            ),
                        },
                    },
                    "content": [
                        {"type": "tool_use", "id": call_id}
                        for call_id in call_ids
                    ],
                    "trace_agent_id": thread_id,
                    "trace_sequence": trace_sequence,
                    "trace_timestamp": str(event.get("timestamp") or ""),
                    "native_turn_id": native_turn_id,
                    "cost_accounting_source": "codex_native_token_count",
                }
            )
        pending_call_ids = []
        pending_first_sequence = None
    return messages


def _load_codex_rollout_bundle(
    manifest_path: str | os.PathLike[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_object(path)
    if manifest.get("format") != CODEX_ROLLOUT_BUNDLE_FORMAT:
        raise StandardizationError(
            f"unsupported Codex rollout bundle: {manifest.get('format')!r}"
        )
    streams = manifest.get("streams")
    if not isinstance(streams, list) or not streams:
        raise StandardizationError("Codex rollout bundle has no session streams")
    sessions: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    assistant_turn_messages: list[dict[str, Any]] = []
    safe_dynamic_read_only_command_template_count = 0
    unresolved_command_cell_count = 0
    seen_threads: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise StandardizationError("Codex rollout stream metadata is malformed")
        stream_path, payload = _verified_codex_rollout_stream(path.parent, stream)
        events = _parse_jsonl(payload, stream_path)
        if not events or events[0].get("type") != "session_meta":
            raise StandardizationError(
                f"Codex rollout has no session header: {stream_path}"
            )
        header = events[0].get("payload")
        if not isinstance(header, Mapping):
            raise StandardizationError(
                f"Codex rollout session header is malformed: {stream_path}"
            )
        thread_id = str(header.get("id") or header.get("session_id") or "")
        logical_session_id = str(header.get("session_id") or thread_id)
        parent_id = header.get("parent_thread_id")
        parent_id = str(parent_id) if parent_id else None
        if not thread_id or thread_id in seen_threads:
            raise StandardizationError(
                "Codex rollout thread IDs are missing or duplicated"
            )
        manifest_thread_id = str(
            stream.get("thread_id") or stream.get("session_id") or ""
        )
        if (
            manifest_thread_id != thread_id
            or stream.get("logical_session_id", logical_session_id)
            != logical_session_id
            or stream.get("parent_thread_id") != parent_id
        ):
            raise StandardizationError(
                f"Codex rollout manifest/header lineage differs for {thread_id}"
            )
        seen_threads.add(thread_id)
        sessions.append(
            {
                "agent_id": thread_id,
                "parent_agent_id": parent_id,
                "spawn_sequence": None,
                "source": copy.deepcopy(header.get("source")),
                "thread_source": copy.deepcopy(header.get("thread_source")),
                "logical_session_id": logical_session_id,
                "event_count": len(events),
                "stream_sha256": stream["sha256"],
            }
        )
        artifacts.append(
            {
                "path": str(stream_path),
                "sha256": stream["sha256"],
                "bytes": stream["bytes"],
                "archive_sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest(),
                "archive_bytes": stream.get("archive_bytes"),
                "session_id": thread_id,
                "logical_session_id": logical_session_id,
            }
        )
        assistant_turn_messages.extend(
            _codex_rollout_turn_messages(events, thread_id)
        )
        command_starts: dict[str, tuple[str, int]] = {}
        command_completions: dict[str, tuple[str, int]] = {}
        for local_sequence, event in enumerate(events, 1):
            event_payload = event.get("payload")
            if (
                event.get("type") == "response_item"
                and isinstance(event_payload, Mapping)
                and event_payload.get("type") == "custom_tool_call_output"
            ):
                call_id = str(event_payload.get("call_id") or "")
                if call_id:
                    command_completions[call_id] = (
                        str(event.get("timestamp") or ""),
                        local_sequence,
                    )
            if (
                event.get("type") == "response_item"
                and isinstance(event_payload, Mapping)
                and event_payload.get("type") == "custom_tool_call"
            ):
                call_id = str(event_payload.get("call_id") or "")
                if call_id:
                    command_starts[call_id] = (
                        str(event.get("timestamp") or ""), local_sequence
                    )
        for local_sequence, event in enumerate(events, 1):
            event_payload = event.get("payload")
            if (
                event.get("type") == "response_item"
                and isinstance(event_payload, Mapping)
                and event_payload.get("type") == "custom_tool_call"
                and event_payload.get("name") == "exec"
            ):
                source = event_payload.get("input")
                if not isinstance(source, str):
                    raise StandardizationError(
                        "Codex exec custom tool call has no source input"
                    )
                call_id = str(event_payload.get("call_id") or "")
                try:
                    requests = _codex_exec_requests(source)
                except StandardizationError as exc:
                    if "tools.exec_command" not in source:
                        raise
                    if not call_id:
                        raise StandardizationError(
                            "Codex unresolved exec_command cell has no provider "
                            "call ID"
                        ) from exc
                    completion = command_completions.get(call_id)
                    if completion is None:
                        raise StandardizationError(
                            f"Codex exec_command cell {call_id} has no completion"
                        ) from exc
                    commands.append(
                        {
                            "agent_id": thread_id,
                            "parent_agent_id": parent_id,
                            "timestamp": completion[0],
                            "started_timestamp": str(event.get("timestamp") or ""),
                            "local_sequence": completion[1],
                            "tool_call_id": f"{call_id}:opaque",
                            "started_sequence": local_sequence,
                            "provider_cell_call_id": call_id,
                            "request_ordinal": 0,
                            "status": str(
                                event_payload.get("status") or "completed"
                            ),
                            "command": None,
                            "cwd": "/testbed",
                            "unresolved_source": source,
                            "resolution_error": str(exc),
                            "resolved": False,
                        }
                    )
                    unresolved_command_cell_count += 1
                    continue
                if "tools.exec_command" in source and not requests:
                    raise StandardizationError(
                        "Codex exec cell contains an unparsed exec_command request"
                    )
                if requests and not call_id:
                    raise StandardizationError(
                        "Codex exec_command cell has no provider call ID"
                    )
                completion = command_completions.get(call_id)
                if requests and completion is None:
                    raise StandardizationError(
                        f"Codex exec_command cell {call_id} has no completion"
                    )
                for ordinal, request in enumerate(requests, 1):
                    if request["dynamic_template"]:
                        if not _codex_dynamic_command_is_read_only(
                            request["command"], source
                        ):
                            raise StandardizationError(
                                "Codex exec_command has an unresolved potentially "
                                f"mutating template in cell {call_id}"
                            )
                        safe_dynamic_read_only_command_template_count += 1
                        continue
                    commands.append(
                        {
                            "agent_id": thread_id,
                            "parent_agent_id": parent_id,
                            "timestamp": completion[0],
                            "started_timestamp": str(event.get("timestamp") or ""),
                            "local_sequence": completion[1],
                            "tool_call_id": f"{call_id}:{ordinal}",
                            "started_sequence": local_sequence,
                            "provider_cell_call_id": call_id,
                            "request_ordinal": ordinal,
                            "status": str(
                                event_payload.get("status") or "completed"
                            ),
                            "command": request["command"],
                            "cwd": request["cwd"],
                            "resolved": True,
                        }
                    )
            if (
                event.get("type") != "event_msg"
                or not isinstance(event_payload, Mapping)
                or event_payload.get("type") != "patch_apply_end"
            ):
                continue
            native_changes = event_payload.get("changes")
            if not isinstance(native_changes, Mapping):
                raise StandardizationError(
                    "Codex patch completion has no change mapping"
                )
            changes = []
            for change_path, value in native_changes.items():
                if not isinstance(value, Mapping):
                    raise StandardizationError(
                        "Codex patch completion has a malformed change"
                    )
                kind_type = str(value.get("type") or "update")
                unified_diff = value.get("unified_diff")
                content = value.get("content")
                content_exact = isinstance(content, str) and kind_type.lower() in {
                    "add",
                    "create",
                    "delete",
                    "remove",
                }
                if not isinstance(unified_diff, str) and not content_exact:
                    raise StandardizationError(
                        "Codex patch completion omits an exact diff/content payload"
                    )
                change = {
                    "path": str(change_path),
                    "kind": {
                        "type": kind_type,
                        "movePath": value.get("move_path"),
                    },
                }
                if isinstance(unified_diff, str):
                    change["diff"] = unified_diff
                if isinstance(content, str):
                    change["content"] = content
                changes.append(change)
            active_calls = [
                (started[1], call_id)
                for call_id, started in command_starts.items()
                if started[1] <= local_sequence
                and call_id in command_completions
                and local_sequence <= command_completions[call_id][1]
            ]
            provider_cell_call_id = (
                max(active_calls)[1] if active_calls else None
            )
            patches.append(
                {
                    "agent_id": thread_id,
                    "parent_agent_id": parent_id,
                    "timestamp": str(event.get("timestamp") or ""),
                    "local_sequence": local_sequence,
                    "tool_call_id": str(event_payload.get("call_id") or ""),
                    "status": str(event_payload.get("status") or "completed"),
                    "success": bool(event_payload.get("success")),
                    "changes": changes,
                    "provider_cell_call_id": provider_cell_call_id,
                    "started_sequence": (
                        command_starts[provider_cell_call_id][1]
                        if provider_cell_call_id is not None
                        else None
                    ),
                    "paths": _codex_change_paths(changes),
                }
            )
    roots = [session for session in sessions if session["parent_agent_id"] is None]
    logical_session_ids = {
        session["logical_session_id"] for session in sessions
    }
    missing_parents = sorted(
        session["agent_id"]
        for session in sessions
        if session["parent_agent_id"] is not None
        and session["parent_agent_id"] not in seen_threads
    )
    if len(roots) != 1 or missing_parents or len(logical_session_ids) != 1:
        raise StandardizationError(
            "Codex rollout bundle is not one complete rooted session tree"
        )
    root_session_id = str(manifest.get("root_session_id") or "")
    if root_session_id != roots[0]["agent_id"]:
        raise StandardizationError("Codex rollout root session ID differs from headers")
    logical_session_id = next(iter(logical_session_ids))
    if root_session_id != logical_session_id:
        raise StandardizationError(
            "Codex rollout root and logical session IDs differ"
        )
    manifest_logical_session_id = manifest.get("logical_session_id")
    if (
        manifest_logical_session_id is not None
        and str(manifest_logical_session_id) != logical_session_id
    ):
        raise StandardizationError(
            "Codex rollout manifest logical session ID differs from headers"
        )
    if int(manifest.get("session_count", len(streams))) != len(streams):
        raise StandardizationError("Codex rollout manifest session count differs")
    patches.sort(
        key=lambda item: (
            item["timestamp"],
            item["local_sequence"],
            item["agent_id"],
        )
    )
    canonical_patches: dict[str, dict[str, Any]] = {}
    duplicate_patch_count = 0
    for patch in patches:
        call_id = str(patch.get("tool_call_id") or "")
        if not call_id:
            raise StandardizationError("Codex patch completion has no call ID")
        canonical = canonical_patches.get(call_id)
        if canonical is None:
            canonical_patches[call_id] = patch
            patch["inherited_history_duplicate"] = False
            continue
        # Forked Codex rollout files contain the child's inherited transcript.
        # Those records are byte-for-byte evidence, but they are not new
        # filesystem mutations. A call ID is provider-native action identity:
        # retain every occurrence for stdout alignment, while replaying only
        # the first chronological occurrence. Fail closed if an inherited
        # copy disagrees with the original action payload.
        for field in ("status", "success", "changes", "paths"):
            if patch.get(field) != canonical.get(field):
                raise StandardizationError(
                    "Codex inherited patch history conflicts for call ID "
                    f"{call_id}"
                )
        patch["inherited_history_duplicate"] = True
        patch["canonical_agent_id"] = canonical["agent_id"]
        patch["canonical_timestamp"] = canonical["timestamp"]
        duplicate_patch_count += 1
    commands.sort(
        key=lambda item: (
            item["timestamp"],
            item["local_sequence"],
            item["agent_id"],
            item["request_ordinal"],
        )
    )
    canonical_commands: dict[str, dict[str, Any]] = {}
    duplicate_command_count = 0
    for command in commands:
        tool_call_id = command["tool_call_id"]
        canonical = canonical_commands.get(tool_call_id)
        if canonical is None:
            canonical_commands[tool_call_id] = command
            command["inherited_history_duplicate"] = False
            continue
        for field in (
            "status",
            "command",
            "cwd",
            "request_ordinal",
            "resolved",
            "resolution_error",
        ):
            if command.get(field) != canonical.get(field):
                raise StandardizationError(
                    "Codex inherited command history conflicts for call ID "
                    f"{tool_call_id}"
                )
        command["inherited_history_duplicate"] = True
        command["canonical_agent_id"] = canonical["agent_id"]
        command["canonical_timestamp"] = canonical["timestamp"]
        duplicate_command_count += 1
    bundle = {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "root_session_id": root_session_id,
        "logical_session_id": logical_session_id,
        "all_fresh_home_sessions_captured": manifest.get(
            "all_fresh_home_sessions_captured"
        )
        is True,
        "unique_patch_completion_count": len(canonical_patches),
        "inherited_patch_history_duplicate_count": duplicate_patch_count,
        "command_request_count": len(commands),
        "unique_command_request_count": len(canonical_commands),
        "inherited_command_history_duplicate_count": duplicate_command_count,
        "safe_dynamic_read_only_command_template_count": (
            safe_dynamic_read_only_command_template_count
        ),
        "unresolved_command_cell_count": unresolved_command_cell_count,
        "artifacts": artifacts,
    }
    canonical_turn_messages: dict[str, dict[str, Any]] = {}
    for message in sorted(
        assistant_turn_messages,
        key=lambda item: (
            item.get("trace_timestamp", ""),
            item.get("trace_agent_id", ""),
            item.get("trace_sequence", 0),
        ),
    ):
        native_turn_id = str(message["native_turn_id"])
        previous = canonical_turn_messages.get(native_turn_id)
        if previous is None:
            canonical_turn_messages[native_turn_id] = message
            continue
        for field in ("model", "usage"):
            if previous.get(field) != message.get(field):
                raise StandardizationError(
                    f"Codex inherited turn usage conflicts for {native_turn_id}"
                )
        previous_call_ids = {
            str(block.get("id"))
            for block in previous.get("content") or []
            if isinstance(block, Mapping) and block.get("id")
        }
        previous["content"].extend(
            block
            for block in message.get("content") or []
            if str(block.get("id") or "") not in previous_call_ids
        )
    bundle["_assistant_turn_messages"] = list(canonical_turn_messages.values())
    return bundle, sessions, patches, commands


def _enrich_codex_with_rollouts(
    trajectory: dict[str, Any], manifest_path: str | os.PathLike[str]
) -> None:
    bundle, sessions, patches, commands = _load_codex_rollout_bundle(manifest_path)
    assistant_turn_messages = bundle.pop("_assistant_turn_messages", [])
    root_session_id = bundle["root_session_id"]
    for message in trajectory.get("messages") or []:
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and not isinstance(message.get("usage"), Mapping)
        ):
            message["cost_accounting_exempt"] = True
    trajectory.setdefault("messages", []).extend(assistant_turn_messages)
    parent_by_session = {
        session["agent_id"]: session["parent_agent_id"] for session in sessions
    }
    public_file_actions = [
        action
        for action in trajectory.get("actions") or []
        if action.get("name") == "FileChange"
    ]
    unused = list(range(len(patches)))
    matched = 0
    for action in public_file_actions:
        arguments = action.get("arguments")
        changes = arguments.get("changes") if isinstance(arguments, Mapping) else []
        paths = _codex_change_paths(changes or [])
        match = next(
            (index for index in unused if patches[index]["paths"] == paths),
            None,
        )
        if match is None:
            continue
        unused.remove(match)
        matched += 1

    # ``codex exec --json`` emits only a lossy subset of subagent file-change
    # receipts. The verified persisted rollout tree is authoritative for exact
    # edits: create one replay action per unique provider-native patch call,
    # ordered by its first chronological ``patch_apply_end`` receipt. Public
    # file-change items remain preserved byte-for-byte in ``events`` but must
    # not define mutation coverage.
    canonical_file_actions: list[dict[str, Any]] = []
    for patch in patches:
        if patch.get("inherited_history_duplicate") is True:
            continue
        effectful = any(
            bool(change.get("diff"))
            or isinstance(change.get("content"), str)
            or bool(
                change.get("kind", {}).get("movePath")
                if isinstance(change.get("kind"), Mapping)
                else False
            )
            for change in patch["changes"]
            if isinstance(change, Mapping)
        )
        canonical_file_actions.append(
            {
                "action_index": 0,
                "tool_call_id": patch["tool_call_id"],
                "status": patch["status"],
                "agent_id": patch["agent_id"],
                "parent_agent_id": patch["parent_agent_id"],
                "parent_tool_use_id": None,
                "started_sequence": patch["started_sequence"],
                "cost_tool_call_id": patch["provider_cell_call_id"],
                "completed_sequence": patch["local_sequence"],
                "ordering_basis": (
                    "unique native rollout patch_apply_end timestamp order"
                ),
                "result": None,
                "name": "FileChange",
                "arguments": {"changes": copy.deepcopy(patch["changes"])},
                "contains_exact_edit_payload": True,
                "replay_kind": (
                    "patch" if patch["success"] and effectful else "non_mutating"
                ),
                "native_rollout_tool_call_id": patch["tool_call_id"],
                "native_rollout_timestamp": patch["timestamp"],
                "native_rollout_local_sequence": patch["local_sequence"],
                "native_rollout_inherited_history_duplicate": False,
                "native_rollout_effectful": effectful,
            }
        )
    canonical_command_actions: list[dict[str, Any]] = []
    for command in commands:
        if command.get("inherited_history_duplicate") is True:
            continue
        canonical_command_actions.append(
            {
                "action_index": 0,
                "tool_call_id": command["tool_call_id"],
                "status": command["status"],
                "agent_id": command["agent_id"],
                "parent_agent_id": command["parent_agent_id"],
                "parent_tool_use_id": None,
                "started_sequence": command["started_sequence"],
                "cost_tool_call_id": command["provider_cell_call_id"],
                "completed_sequence": command["local_sequence"],
                "ordering_basis": (
                    "unique native rollout exec_command completion timestamp order"
                ),
                "result": None,
                "name": "Bash",
                "arguments": {
                    "cwd": command["cwd"],
                    **(
                        {"command": command["command"]}
                        if command.get("resolved") is not False
                        else {
                            "source": command.get("unresolved_source"),
                            "resolution_error": command.get("resolution_error"),
                        }
                    ),
                },
                "observed_returncode": None,
                "replay_kind": (
                    "bash"
                    if command.get("resolved") is not False
                    else "unresolved_command"
                ),
                "native_rollout_tool_call_id": command["tool_call_id"],
                "native_rollout_timestamp": command["timestamp"],
                "native_rollout_started_timestamp": command[
                    "started_timestamp"
                ],
                "native_rollout_local_sequence": command["local_sequence"],
                "native_rollout_inherited_history_duplicate": False,
            }
        )
    canonical_mutation_actions = canonical_file_actions + canonical_command_actions
    canonical_mutation_actions.sort(
        key=lambda action: (
            action["native_rollout_timestamp"],
            action["native_rollout_local_sequence"],
            action["agent_id"],
            action["native_rollout_tool_call_id"],
        )
    )
    non_native_public_actions = [
        action
        for action in trajectory.get("actions") or []
        if action.get("name") not in {"FileChange", "Bash"}
    ]
    trajectory["actions"] = non_native_public_actions + canonical_mutation_actions
    for action_index, action in enumerate(trajectory["actions"], 1):
        previous_index = action.get("action_index")
        if previous_index:
            action["public_action_index"] = previous_index
        action["action_index"] = action_index
    # Ordinary ``codex exec --json`` omits an agent identifier from root items,
    # so the first pass uses a provider-neutral ``root`` placeholder. Once the
    # verified rollout tree is available, replace that placeholder with the
    # authoritative root session ID. Every normalized action must then point
    # into the captured session tree; otherwise lineage is not complete.
    for event in trajectory.get("events") or []:
        if event.get("agent_id") == _ROOT_AGENT_ID:
            event["agent_id"] = root_session_id
        if event.get("parent_agent_id") == _ROOT_AGENT_ID:
            event["parent_agent_id"] = root_session_id
    dangling_action_agents: set[str] = set()
    for action in trajectory.get("actions") or []:
        agent_id = str(action.get("agent_id") or _ROOT_AGENT_ID)
        if agent_id == _ROOT_AGENT_ID:
            agent_id = root_session_id
        action["agent_id"] = agent_id
        if agent_id in parent_by_session:
            action["parent_agent_id"] = parent_by_session[agent_id]
        else:
            dangling_action_agents.add(agent_id)
    lineage_complete = bool(
        bundle["all_fresh_home_sessions_captured"] and not dangling_action_agents
    )
    successful_patch_count = sum(bool(patch["success"]) for patch in patches)
    all_patches_successful = successful_patch_count == len(patches)
    alignment_complete = bool(
        matched == len(public_file_actions) == len(patches)
        and all_patches_successful
    )
    canonical_edit_trace_complete = bool(
        canonical_file_actions
        and len(canonical_file_actions) == bundle["unique_patch_completion_count"]
        and all_patches_successful
    )
    canonical_command_trace_complete = bool(
        len(canonical_command_actions) == bundle["unique_command_request_count"]
        and bundle["unresolved_command_cell_count"] == 0
    )
    canonical_mutation_trace_complete = bool(
        canonical_edit_trace_complete and canonical_command_trace_complete
    )
    trajectory["session_id"] = root_session_id
    trajectory["agents"] = sessions
    trajectory["codex_native_rollout"] = {
        **bundle,
        "session_count": len(sessions),
        "patch_completion_count": len(patches),
        "unique_patch_completion_count": bundle["unique_patch_completion_count"],
        "inherited_patch_history_duplicate_count": bundle[
            "inherited_patch_history_duplicate_count"
        ],
        "successful_patch_completion_count": successful_patch_count,
        "matched_stdout_file_change_count": matched,
        "ordered_path_alignment_complete": alignment_complete,
        "canonical_edit_trace_complete": canonical_edit_trace_complete,
        "canonical_command_trace_complete": canonical_command_trace_complete,
        "canonical_mutation_trace_complete": canonical_mutation_trace_complete,
    }
    trajectory["codex_edit_evidence"].update(
        {
            "supplemental_native_rollout_contains_exact_edits": bool(patches),
            "supplemental_patch_completion_count": len(patches),
            "supplemental_successful_patch_completion_count": successful_patch_count,
            "stdout_rollout_edit_alignment_complete": alignment_complete,
            "public_stdout_file_change_receipt_count": len(public_file_actions),
            "canonical_unique_file_change_action_count": len(
                canonical_file_actions
            ),
            "rollout_canonical_edit_trace_complete": canonical_edit_trace_complete,
            "canonical_unique_command_action_count": len(
                canonical_command_actions
            ),
            "rollout_canonical_command_trace_complete": (
                canonical_command_trace_complete
            ),
            "rollout_canonical_mutation_trace_complete": (
                canonical_mutation_trace_complete
            ),
        }
    )
    quality = trajectory["quality"]
    quality["subagent_lineage_complete"] = lineage_complete
    # The compressed rollout files retain the complete provider-native tool
    # transcript byte-for-byte.  The provider-neutral action list intentionally
    # contains only replay-relevant public CLI items, so do not label that full
    # normalized tool list exact.  Exact mutation coverage is a separate claim.
    quality["tool_trace_exact"] = False
    quality["provider_tool_artifact_exact"] = True
    quality["file_edit_trace_exact"] = bool(
        quality.get("native_stream_exact")
        and lineage_complete
        and canonical_edit_trace_complete
    )
    quality["command_trace_exact"] = bool(
        quality.get("native_stream_exact")
        and lineage_complete
        and canonical_command_trace_complete
    )
    quality["mutation_trace_exact"] = bool(
        quality["file_edit_trace_exact"]
        and quality["command_trace_exact"]
        and canonical_mutation_trace_complete
    )
    quality["completed_action_order_exact"] = False
    quality["mutation_ordering_basis"] = (
        "deduplicated native rollout patch and exec completion timestamp order"
    )
    quality["lineage_issues"] = (
        []
        if lineage_complete
        else [
            {
                "issue": "Codex rollout/action lineage is incomplete",
                "all_fresh_home_sessions_captured": bundle[
                    "all_fresh_home_sessions_captured"
                ],
                "dangling_action_agent_ids": sorted(dangling_action_agents),
            }
        ]
    )


def standardize_events(
    provider: str,
    events: list[dict[str, Any]],
    *,
    source: Mapping[str, Any] | None = None,
    standardization_phase: str = "post_exit",
) -> dict[str, Any]:
    """Standardize decoded native events without touching a live agent."""

    if standardization_phase not in {"post_exit", "live_host"}:
        raise StandardizationError(
            "standardization_phase must be post_exit or live_host"
        )
    source_metadata = dict(source or {})
    canonical_provider = provider.strip().lower().replace("_", "-")
    if canonical_provider in {"", "auto", "detect"}:
        canonical_provider = detect_trace_provider(events)
    if canonical_provider in {
        "antigravity",
        "antigravity-cli",
        "agy",
    }:
        body = standardize_antigravity(events, source_metadata)
    elif canonical_provider in {"claude", "claude-code", "anthropic"}:
        body = _standardize_claude(events, source_metadata)
    elif canonical_provider in {
        "deepseek",
        "deepseek-harness",
        "dsh",
    }:
        body = _standardize_deepseek_harness(events, source_metadata)
    elif canonical_provider in {"kimi", "kimi-code", "kimi-code-cli"}:
        body = _standardize_kimi(events, source_metadata)
    elif canonical_provider in {
        "mistral",
        "mistral-vibe",
        "mistral-vibe-cli",
        "vibe",
    }:
        body = standardize_mistral_vibe(events, source_metadata)
    elif canonical_provider in {"codex", "codex-cli", "openai-codex"}:
        body = _standardize_codex(events, source_metadata, app_server=False)
    elif canonical_provider in {"codex-app-server", "app-server"}:
        body = _standardize_codex(events, source_metadata, app_server=True)
    elif canonical_provider in {"generic", "generic-jsonl", "unknown"}:
        body = _standardize_generic(events, source_metadata)
    else:
        raise StandardizationError(
            f"unsupported standardized-trace provider: {provider}"
        )
    result = {
        "format": STANDARDIZED_TRACE_FORMAT,
        "standardization_phase": standardization_phase,
        "ordering_basis": "provider native JSONL receipt order",
        "source": source_metadata,
        **body,
    }
    result["trace_sha256"] = _digest(result)
    return result


def standardize_native_capture(
    manifest_path: str | os.PathLike[str],
    *,
    codex_rollout_manifest: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify and standardize an exact harness-native capture."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_object(path)
    if manifest.get("format") not in {
        PROXY_NATIVE_CAPTURE_FORMAT,
        LEAN_NATIVE_CAPTURE_FORMAT,
    }:
        raise StandardizationError(
            f"unsupported harness-wrapper capture format: {manifest.get('format')!r}"
        )
    stdout_metadata = manifest.get("stdout")
    if not isinstance(stdout_metadata, Mapping):
        raise StandardizationError("native capture manifest has no stdout artifact")
    stdout_path, stdout = _verified_artifact(path.parent, stdout_metadata)
    stderr_metadata = manifest.get("stderr")
    if isinstance(stderr_metadata, Mapping):
        _verified_artifact(path.parent, stderr_metadata)
    if (
        manifest.get("normalization_phase") != "post_exit"
        or manifest.get("native_stream_parsed_during_agent_execution") is not False
    ):
        raise StandardizationError("capture does not prove post-exit standardization")
    events = _parse_jsonl(stdout, stdout_path)
    source = {
        "kind": (
            "trace-utils-native-capture"
            if manifest["format"] == PROXY_NATIVE_CAPTURE_FORMAT
            else "leanlean-native-capture"
        ),
        "capture_format": manifest["format"],
        "capture_id": manifest.get("capture_id"),
        "capture_manifest": str(path),
        "native_stream_path": str(stdout_path),
        "native_stream_sha256": stdout_metadata["sha256"],
        "native_stream_bytes": stdout_metadata["bytes"],
        "native_stream_exact": True,
        "normalization_phase": "post_exit",
        "forwards_subagent_stream": bool(manifest.get("forwards_subagent_stream")),
        "native_edit_payloads_exact": bool(
            manifest.get(
                "native_edit_payloads_exact",
                str(manifest.get("harness") or "").lower()
                in {"claude", "claude-code", "anthropic"},
            )
        ),
        "edit_payload_evidence": manifest.get("edit_payload_evidence"),
        "session_id": manifest.get("session_id"),
        "model": manifest.get("model"),
        "returncode": manifest.get("returncode"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "timed_out": manifest.get("timed_out"),
    }
    trajectory = standardize_events(
        str(manifest.get("harness") or ""), events, source=source
    )
    if codex_rollout_manifest is not None:
        if trajectory.get("provider") != "codex-cli":
            raise StandardizationError(
                "supplemental Codex rollouts require a codex-cli native capture"
            )
        _enrich_codex_with_rollouts(trajectory, codex_rollout_manifest)
        trajectory["trace_sha256"] = _digest(
            {key: value for key, value in trajectory.items() if key != "trace_sha256"}
        )
    return trajectory


def standardize_proxy_native_capture(
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Backward-compatible trace-utils native-capture entry point."""

    return standardize_native_capture(manifest_path)


def _expected_proxy_fanout(harness: str, raw: Mapping[str, Any]) -> int:
    provider = harness.strip().lower().replace("_", "-")
    native_type = str(raw.get("type") or "event")
    if provider in {"claude", "claude-code", "anthropic"}:
        if native_type in {"assistant", "user"}:
            message = raw.get("message")
            blocks = message.get("content") if isinstance(message, Mapping) else None
            return max(1, len(blocks)) if isinstance(blocks, list) else 1
        if native_type == "result" and raw.get("usage") is not None:
            return 2
        return 1
    if provider in {"codex", "codex-cli", "openai-codex"}:
        item = raw.get("item")
        if (
            native_type == "item.completed"
            and isinstance(item, Mapping)
            and _codex_item_type(item)
            in {"command_execution", "mcp_tool_call", "web_search"}
            and item.get("aggregated_output", item.get("result")) is not None
        ):
            return 2
    return 1


def _proxy_trace_raw_events(
    trace_path: Path, harness: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_copies: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        trace_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StandardizationError(
                f"invalid trace-utils trace at line {line_number}"
            ) from exc
        event = record.get("event", record) if isinstance(record, Mapping) else None
        metadata = event.get("metadata") if isinstance(event, Mapping) else None
        raw = metadata.get("raw") if isinstance(metadata, Mapping) else None
        if not isinstance(raw, Mapping):
            raise StandardizationError(
                f"trace-utils trace event {line_number} does not preserve metadata.raw"
            )
        raw_copies.append(copy.deepcopy(dict(raw)))

    events: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    index = 0
    while index < len(raw_copies):
        raw = raw_copies[index]
        fanout = _expected_proxy_fanout(harness, raw)
        digest = _digest(raw)
        group = raw_copies[index : index + fanout]
        if len(group) == fanout and all(_digest(item) == digest for item in group):
            events.append(raw)
            index += fanout
            continue
        issues.append(
            {
                "trace_record": index + 1,
                "issue": "normalized_fanout_could_not_be_recovered",
                "expected_fanout": fanout,
            }
        )
        events.append(raw)
        index += 1
    return events, issues


def standardize_proxy_normalized_trace(
    trace_path: str | os.PathLike[str], *, harness: str
) -> dict[str, Any]:
    """Import an old trace-utils trace without making byte-exactness claims."""

    path = Path(trace_path).expanduser().resolve()
    events, issues = _proxy_trace_raw_events(path, harness)
    source = {
        "kind": "trace-utils-normalized-trace",
        "trace_path": str(path),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_stream_exact": False,
        "normalization_phase": "during_agent_execution",
        "forwards_subagent_stream": False,
        "recovery_issues": issues,
    }
    return standardize_events(harness, events, source=source)


def write_standardized_trace(
    trajectory: Mapping[str, Any], destination: str | os.PathLike[str]
) -> Path:
    """Atomically persist a canonical standardized trajectory."""

    if trajectory.get("format") != STANDARDIZED_TRACE_FORMAT:
        raise StandardizationError("cannot write an unknown standardized trace format")
    path = Path(destination).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_bytes(trajectory) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "CODEX_ROLLOUT_BUNDLE_FORMAT",
    "LEAN_NATIVE_CAPTURE_FORMAT",
    "PROXY_NATIVE_CAPTURE_FORMAT",
    "STANDARDIZED_TRACE_FORMAT",
    "StandardizationError",
    "standardize_events",
    "standardize_native_capture",
    "standardize_proxy_native_capture",
    "standardize_proxy_normalized_trace",
    "write_standardized_trace",
]
