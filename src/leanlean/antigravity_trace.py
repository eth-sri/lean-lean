"""Normalize Antigravity CLI headless ``stream-json`` events."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any




def is_antigravity_recovered_submission(result: Mapping[str, Any]) -> bool:
    """Recognize a native terminal error emitted after an explicit submission.

    Antigravity can retain a previous transient provider error as its terminal
    status even after the agent recovers, verifies its work, and emits
    ``<submit>``.  The explicit marker is sufficient to preserve the result as
    a submission because the patch remains subject to the benchmark's
    independent build and signature verification.
    """

    response = result.get("response")
    error = result.get("error")
    return bool(
        str(result.get("status") or "").upper() == "ERROR"
        and isinstance(response, str)
        and "<submit>" in response.lower()
        and isinstance(error, str)
        and bool(error.strip())
    )


def antigravity_terminal_success(result: Mapping[str, Any]) -> bool:
    """Return whether a native Antigravity result is usable as a submission."""

    return bool(
        (
            str(result.get("status") or "").upper() == "SUCCESS"
            and not result.get("error")
        )
        or is_antigravity_recovered_submission(result)
    )


def is_antigravity_event(event: Mapping[str, Any]) -> bool:
    """Return whether an object belongs to Antigravity's headless stream."""

    kind = event.get("event")
    if kind == "init":
        return isinstance(event.get("init"), Mapping)
    if kind == "step_update":
        return isinstance(event.get("step_update"), Mapping)
    if kind == "result":
        return isinstance(event.get("result"), Mapping)
    return False


def antigravity_action(
    event: Mapping[str, Any], *, action_index: int, completed_sequence: int
) -> dict[str, Any] | None:
    """Convert one completed Antigravity tool step into an action record."""

    if event.get("event") != "step_update":
        return None
    step = event.get("step_update")
    if not isinstance(step, Mapping):
        return None
    if step.get("step_type") != "tool" or step.get("state") != "DONE":
        return None
    info = step.get("tool_info")
    info = info if isinstance(info, Mapping) else {}
    name = str(step.get("tool_name") or info.get("name") or "unknown")
    error = copy.deepcopy(info.get("error"))
    normalized = name.lower().replace("-", "_")
    if normalized in {"run_command", "shell", "bash"}:
        replay_kind = "bash"
    elif normalized in {"write_to_file", "write_file", "replace_file_content"}:
        replay_kind = "unsupported_file_change"
    else:
        replay_kind = "non_mutating"
    return {
        "action_index": action_index,
        "tool_call_id": f"antigravity-step-{step.get('step_index', completed_sequence)}",
        "name": name,
        "arguments": copy.deepcopy(info.get("parameters")),
        "result": copy.deepcopy(info.get("output")),
        "error": error,
        "is_error": error is not None,
        "status": "failed" if error is not None else "completed",
        "agent_id": "root",
        "parent_agent_id": None,
        "parent_tool_use_id": None,
        "started_sequence": None,
        "completed_sequence": completed_sequence,
        "ordering_basis": "native completed step receipt order",
        "replay_kind": replay_kind,
    }


def standardize_antigravity(
    events: list[dict[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the provider-neutral trajectory while retaining every raw event."""

    actions: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    terminal: dict[str, Any] = {}
    session_id = source.get("session_id")
    subagent_steps: list[int] = []
    event_records: list[dict[str, Any]] = []
    for sequence, event in enumerate(events, 1):
        event_records.append(
            {
                "sequence": sequence,
                "type": str(event.get("event") or "event"),
                "agent_id": "root",
                "parent_tool_use_id": None,
                "payload": copy.deepcopy(event),
            }
        )
        kind = event.get("event")
        payload = event.get(kind) if isinstance(kind, str) else None
        if isinstance(payload, Mapping) and not session_id:
            session_id = payload.get("conversation_id")
        if kind == "step_update" and isinstance(payload, Mapping):
            if payload.get("step_type") == "agent_response" and payload.get("text_delta"):
                messages.append(
                    {
                        "sequence": sequence,
                        "role": "assistant",
                        "content": payload.get("text_delta"),
                    }
                )
            if isinstance(payload.get("subagent_info"), Mapping):
                subagent_steps.append(int(payload.get("step_index") or 0))
            action = antigravity_action(
                event,
                action_index=len(actions) + 1,
                completed_sequence=sequence,
            )
            if action is not None:
                actions.append(action)
        elif kind == "result" and isinstance(payload, Mapping):
            terminal = copy.deepcopy(dict(payload))

    timed_out = bool(source.get("timed_out"))
    returncode = source.get("returncode")
    recovered_submission = is_antigravity_recovered_submission(terminal)
    success = bool(
        not timed_out
        and antigravity_terminal_success(terminal)
        and (returncode == 0 or recovered_submission)
    )
    if recovered_submission:
        terminal["native_status"] = terminal.get("status")
        terminal["recovered_submission"] = True
        terminal["success"] = True
    native_exact = bool(source.get("native_stream_exact"))
    return {
        "provider": "antigravity-cli",
        "session_id": session_id,
        "events": event_records,
        "messages": messages,
        "agents": [
            {
                "agent_id": "root",
                "parent_agent_id": None,
                "spawn_sequence": None,
            }
        ],
        "actions": actions,
        "terminal": terminal
        or {
            "status": "timed_out" if timed_out else "failed",
            "returncode": returncode,
        },
        "pending_tools": [],
        "quality": {
            "native_stream_exact": native_exact,
            "tool_trace_exact": native_exact,
            "mutation_trace_exact": False,
            "subagent_lineage_complete": not subagent_steps,
            "completed_action_order_exact": native_exact,
            "filesystem_reconstruction_exact": None,
            "lineage_issues": [
                {
                    "step_index": index,
                    "issue": "subagent_inner_actions_not_forwarded_by_antigravity_stream",
                }
                for index in subagent_steps
            ],
            "provider_terminal_success": success,
            "provider_errors": [terminal.get("error")]
            if terminal.get("error") and not recovered_submission
            else [],
            "all_actions_failed": bool(actions)
            and all(action.get("is_error") is True for action in actions),
        },
    }


__all__ = [
    "antigravity_action",
    "antigravity_terminal_success",
    "is_antigravity_event",
    "is_antigravity_recovered_submission",
    "standardize_antigravity",
]
