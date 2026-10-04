"""Mistral Vibe public-history trace normalization.

Vibe 2.24 programmatic streaming emits each completed public-history entry once.
This module preserves that presentation-level contract without claiming that
subagent-internal actions, which are not forwarded by the stream, are present.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

ROOT_AGENT_ID = "root"
VIBE_PUBLIC_ENTRY_TYPES = {"message", "reasoning", "effect", "callback"}


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


def _value(value: Mapping[str, Any], snake: str, camel: str) -> Any:
    return value.get(camel, value.get(snake))


def is_mistral_vibe_event(event: Mapping[str, Any]) -> bool:
    """Recognize one Vibe public-history entry without using the model name."""

    return (
        str(event.get("type") or "") in VIBE_PUBLIC_ENTRY_TYPES
        and event.get("id") is not None
        and _value(event, "generation_status", "generationStatus") is not None
    )


def _event_records(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for sequence, event in enumerate(events, 1):
        session_id = _value(event, "session_id", "sessionId")
        records.append(
            {
                "sequence": sequence,
                "type": str(event.get("type") or "event"),
                "agent_id": ROOT_AGENT_ID,
                "provider_session_id": str(session_id) if session_id else None,
                "parent_tool_use_id": None,
                "payload": copy.deepcopy(dict(event)),
                "payload_sha256": _digest(event),
            }
        )
    return records


_VIBE_ARGUMENT_ALIASES = {
    "filePath": "file_path",
    "oldString": "old_string",
    "newString": "new_string",
    "replaceAll": "replace_all",
    "maxMatches": "max_matches",
}


def _canonical_arguments(value: Any) -> Any:
    """Convert Vibe wire aliases needed by the provider-neutral replay schema."""

    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    arguments = copy.deepcopy(dict(value))
    for provider_name, canonical_name in _VIBE_ARGUMENT_ALIASES.items():
        if provider_name not in arguments:
            continue
        arguments.setdefault(canonical_name, arguments[provider_name])
        arguments.pop(provider_name, None)
    return arguments


def _replay_kind(
    effect_kind: str,
    name: str,
    arguments: Any,
    status: str,
) -> tuple[str, bool]:
    """Return the canonical replay kind and exact-payload claim."""

    kind = effect_kind.lower()
    tool = name.lower()
    args = arguments if isinstance(arguments, Mapping) else {}

    if kind == "shell" or (kind == "tool" and tool in {"bash", "shell"}):
        exact = isinstance(args.get("command"), str)
        if status in {"completed", "failed"}:
            return "bash", exact
        return "unsupported_nonexecuted_shell", exact

    if kind == "file_edit" or (
        kind == "tool" and tool in {"edit", "multiedit", "str_replace"}
    ):
        exact = all(
            isinstance(args.get(key), str)
            for key in ("file_path", "old_string", "new_string")
        )
        return ("edit" if status == "completed" else "failed_mutation"), exact

    if kind == "file_write" or (
        kind == "tool" and tool in {"write", "write_file"}
    ):
        exact = all(
            isinstance(args.get(key), str) for key in ("file_path", "content")
        )
        return ("write" if status == "completed" else "failed_mutation"), exact

    if kind in {
        "file_search",
        "file_read",
        "todo",
        "user_question",
        "web_search",
        "web_fetch",
        "skill",
    } or (kind == "tool" and tool in {"grep", "read", "todo", "skill"}):
        return "non_mutating", True

    if kind == "subagent":
        return "unsupported_subagent", False
    if kind == "worktree":
        return "unsupported_worktree", False
    return "unsupported_tool", False


def mistral_vibe_action(
    event: Mapping[str, Any],
    *,
    action_index: int,
    completed_sequence: int,
) -> dict[str, Any] | None:
    """Normalize one terminal Vibe public effect without inventing fields."""

    if str(event.get("type") or "") != "effect":
        return None
    detail = event.get("detail")
    state = event.get("state")
    if not isinstance(detail, Mapping) or not isinstance(state, Mapping):
        return None

    effect_kind = str(detail.get("kind") or "tool")
    name = str(
        detail.get("toolName")
        or detail.get("tool_name")
        or event.get("title")
        or effect_kind
    )
    provider_arguments = copy.deepcopy(detail.get("input"))
    arguments = _canonical_arguments(provider_arguments)
    status = str(state.get("status") or "unknown")
    replay_kind, exact_mutation_payload = _replay_kind(
        effect_kind, name, arguments, status
    )
    result = copy.deepcopy(state.get("output"))
    if result is None and state.get("error") is not None:
        result = copy.deepcopy(state.get("error"))
    if result is None and state.get("outputText", state.get("output_text")):
        result = state.get("outputText", state.get("output_text"))

    session_id = _value(event, "session_id", "sessionId")
    turn_id = _value(event, "turn_id", "turnId")
    related_entry_id = _value(event, "related_entry_id", "relatedEntryId")
    child_session_id = detail.get(
        "childSessionId", detail.get("child_session_id")
    )
    return {
        "action_index": action_index,
        "tool_call_id": str(event.get("id") or ""),
        "name": name,
        "arguments": arguments,
        "provider_arguments": provider_arguments,
        "result": result,
        "result_block": copy.deepcopy(dict(state)),
        "status": status,
        "is_error": status == "failed",
        "agent_id": ROOT_AGENT_ID,
        "parent_agent_id": None,
        "parent_tool_use_id": None,
        "started_sequence": None,
        "completed_sequence": completed_sequence,
        "ordering_basis": "native completed-public-history receipt order",
        "replay_kind": replay_kind,
        "provider_entry_id": str(event.get("id") or ""),
        "provider_session_id": str(session_id) if session_id else None,
        "turn_id": str(turn_id) if turn_id else None,
        "related_entry_id": str(related_entry_id) if related_entry_id else None,
        "effect_kind": effect_kind,
        "title": str(event.get("title") or ""),
        "duration_ms": state.get("durationMs", state.get("duration_ms")),
        "contains_exact_mutation_payload": exact_mutation_payload,
        "child_session_id": str(child_session_id) if child_session_id else None,
        "collaboration_event": effect_kind.lower() == "subagent",
    }


def standardize_mistral_vibe(
    events: list[dict[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the canonical body for Vibe 2.24 streaming history."""

    messages: list[dict[str, Any]] = []
    callbacks: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    pending_tools: list[dict[str, Any]] = []
    semantic_issues: list[dict[str, Any]] = []
    duplicate_entry_ids: list[str] = []
    seen_entry_ids: set[str] = set()
    session_id = source.get("session_id")
    observed_session_ids: set[str] = set()
    child_agents: dict[str, dict[str, Any]] = {}

    for sequence, event in enumerate(events, 1):
        entry_type = str(event.get("type") or "")
        entry_id = str(event.get("id") or "")
        event_session_id = _value(event, "session_id", "sessionId")
        if event_session_id:
            observed_session_ids.add(str(event_session_id))
            if session_id is None:
                session_id = str(event_session_id)
        generation_status = str(
            _value(event, "generation_status", "generationStatus") or ""
        )

        if entry_id:
            if entry_id in seen_entry_ids:
                duplicate_entry_ids.append(entry_id)
                continue
            seen_entry_ids.add(entry_id)

        event_turn_id = _value(event, "turn_id", "turnId")
        common = {
            "sequence": sequence,
            "message_id": entry_id or None,
            "session_id": str(event_session_id) if event_session_id else None,
            "turn_id": str(event_turn_id) if event_turn_id else None,
            "generation_status": generation_status,
            "related_entry_id": _value(
                event, "related_entry_id", "relatedEntryId"
            ),
        }
        if generation_status != "completed":
            semantic_issues.append(
                {
                    "entry_id": entry_id,
                    "sequence": sequence,
                    "issue": "nonterminal_public_history_entry_in_stream",
                    "generation_status": generation_status,
                }
            )

        if entry_type == "message":
            messages.append(
                {
                    **common,
                    "entry_type": "message",
                    "role": str(event.get("role") or "unknown"),
                    "content": copy.deepcopy(event.get("content") or []),
                    "source": event.get("source"),
                }
            )
            continue
        if entry_type == "reasoning":
            messages.append(
                {
                    **common,
                    "entry_type": "reasoning",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "reasoning",
                            "text": str(event.get("text") or ""),
                            "summary": copy.deepcopy(event.get("summary") or []),
                        }
                    ],
                }
            )
            continue
        if entry_type == "callback":
            callbacks.append(
                {
                    **common,
                    "callback_id": event.get(
                        "callbackId", event.get("callback_id")
                    ),
                    "title": event.get("title"),
                    "detail": copy.deepcopy(event.get("detail")),
                    "state": copy.deepcopy(event.get("state")),
                }
            )
            continue
        if entry_type != "effect":
            semantic_issues.append(
                {
                    "entry_id": entry_id,
                    "sequence": sequence,
                    "issue": "unsupported_public_stream_record",
                    "entry_type": entry_type,
                }
            )
            continue

        state = event.get("state")
        state_status = (
            str(state.get("status") or "")
            if isinstance(state, Mapping)
            else ""
        )
        if generation_status != "completed" or state_status in {
            "pending",
            "running",
            "blocked",
            "",
        }:
            detail = event.get("detail")
            pending_tools.append(
                {
                    "tool_call_id": entry_id,
                    "name": (
                        detail.get("toolName", detail.get("tool_name"))
                        if isinstance(detail, Mapping)
                        else event.get("title")
                    ),
                    "generation_status": generation_status,
                    "status": state_status,
                    "sequence": sequence,
                }
            )
            continue

        action = mistral_vibe_action(
            event,
            action_index=len(actions) + 1,
            completed_sequence=sequence,
        )
        if action is None:
            semantic_issues.append(
                {
                    "entry_id": entry_id,
                    "sequence": sequence,
                    "issue": "malformed_completed_effect",
                }
            )
            continue
        actions.append(action)
        child_session_id = action.get("child_session_id")
        if child_session_id:
            child_agents[str(child_session_id)] = {
                "agent_id": str(child_session_id),
                "parent_agent_id": ROOT_AGENT_ID,
                "spawn_sequence": sequence,
                "provider": "mistral-vibe",
                "actions_forwarded": False,
            }

    if len(observed_session_ids) > 1:
        semantic_issues.append(
            {
                "issue": "multiple_root_session_ids_in_public_history",
                "session_ids": sorted(observed_session_ids),
            }
        )

    timed_out = bool(source.get("timed_out"))
    returncode = source.get("returncode")
    native_exact = bool(source.get("native_stream_exact"))
    subagent_actions = [
        action for action in actions if action.get("collaboration_event") is True
    ]
    lineage_issues = [
        {
            "agent_id": action.get("child_session_id"),
            "tool_call_id": action.get("tool_call_id"),
            "issue": "subagent_internal_actions_not_forwarded_by_programmatic_stream",
        }
        for action in subagent_actions
    ]
    all_actions_failed = bool(actions) and all(
        action.get("is_error") is True
        or action.get("status") in {"failed", "cancelled", "skipped"}
        for action in actions
    )
    provider_terminal_success = bool(
        not timed_out and returncode == 0 and not all_actions_failed
    )
    if timed_out and not pending_tools:
        semantic_issues.append(
            {
                "issue": "pending_effects_unobservable_after_timeout",
                "detail": (
                    "Vibe streaming mode emits only completed public-history entries"
                ),
            }
        )

    tool_trace_exact = bool(
        native_exact
        and provider_terminal_success
        and not pending_tools
        and not duplicate_entry_ids
        and not semantic_issues
    )
    mutation_issues = [
        {
            "action_index": action.get("action_index"),
            "tool_call_id": action.get("tool_call_id"),
            "replay_kind": action.get("replay_kind"),
            "issue": "mutation_payload_or_descendant_actions_not_exact",
        }
        for action in actions
        if action.get("replay_kind")
        not in {"non_mutating", "bash", "edit", "write"}
        or (
            action.get("replay_kind") in {"bash", "edit", "write"}
            and action.get("contains_exact_mutation_payload") is not True
        )
    ]
    mutation_trace_exact = bool(
        tool_trace_exact and not mutation_issues and not subagent_actions
    )
    provider_errors = [
        {
            "tool_call_id": action.get("tool_call_id"),
            "name": action.get("name"),
            "status": action.get("status"),
            "result": copy.deepcopy(action.get("result")),
        }
        for action in actions
        if action.get("is_error") is True
    ]
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
        "native_terminal_record": False,
        "process_exit_is_authoritative": True,
        "errors": provider_errors,
    }
    provider_quality = (
        "ordered_complete_public_history"
        if mutation_trace_exact
        else "ordered_complete_root_effects_partial_descendants"
        if tool_trace_exact and subagent_actions
        else "ordered_complete_root_effects_unsupported_mutations"
        if tool_trace_exact
        else "ordered_partial_public_history"
    )
    return {
        "provider": "mistral-vibe",
        "session_id": str(session_id) if session_id else None,
        "events": _event_records(events),
        "messages": messages,
        "callbacks": callbacks,
        "agents": [
            {
                "agent_id": ROOT_AGENT_ID,
                "parent_agent_id": None,
                "spawn_sequence": None,
            },
            *child_agents.values(),
        ],
        "actions": actions,
        "terminal": terminal,
        "pending_tools": pending_tools,
        "quality": {
            "native_stream_exact": native_exact,
            "tool_trace_exact": tool_trace_exact,
            "root_effect_trace_exact": tool_trace_exact,
            "mutation_trace_exact": mutation_trace_exact,
            "subagent_lineage_complete": not subagent_actions,
            "completed_action_order_exact": tool_trace_exact,
            "filesystem_reconstruction_exact": None,
            "lineage_issues": lineage_issues,
            "semantic_issues": semantic_issues,
            "mutation_issues": mutation_issues,
            "duplicate_entry_ids": duplicate_entry_ids,
            "provider_quality": provider_quality,
            "provider_terminal_success": provider_terminal_success,
            "provider_errors": provider_errors,
            "all_actions_failed": all_actions_failed,
            "provider_public_history_schema": "vibe-2.24-public-history-entry",
        },
    }


__all__ = [
    "is_mistral_vibe_event",
    "mistral_vibe_action",
    "standardize_mistral_vibe",
]
