"""Reconstruct completed Kimi Code tool actions from native stream-json."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        arguments = copy.deepcopy(dict(value))
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_arguments": value}
        arguments = copy.deepcopy(dict(decoded)) if isinstance(decoded, Mapping) else {
            "raw_arguments": value
        }
    else:
        arguments = {}
    path = arguments.get("path")
    if isinstance(path, str) and "file_path" not in arguments:
        arguments["file_path"] = path
    return arguments


class KimiTraceBuilder:
    """Join Kimi assistant tool calls to tool results in receipt order."""

    def __init__(self) -> None:
        self.sequence = 0
        self.action_index = 0
        self.pending: dict[str, dict[str, Any]] = {}
        self.completed_ids: set[str] = set()
        self.duplicate_tool_call_ids: list[str] = []
        self.orphan_tool_result_ids: list[str] = []

    @property
    def pending_tool_uses(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(action)
            for action in sorted(
                self.pending.values(),
                key=lambda action: int(action["started_sequence"]),
            )
        ]

    def ingest(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        self.sequence += 1
        role = str(event.get("role") or "")
        if role == "assistant":
            for offset, raw_call in enumerate(event.get("tool_calls") or []):
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, Mapping):
                    continue
                tool_id = str(
                    raw_call.get("id") or f"kimi-tool-{self.sequence}-{offset + 1}"
                )
                if tool_id in self.pending or tool_id in self.completed_ids:
                    self.duplicate_tool_call_ids.append(tool_id)
                    continue
                name = str(function.get("name") or "unknown")
                if name.lower() == "shell":
                    name = "Bash"
                self.pending[tool_id] = {
                    "tool_use_id": tool_id,
                    "name": name,
                    "input": _tool_arguments(function.get("arguments")),
                    "agent_id": "root",
                    "parent_tool_use_id": None,
                    "started_sequence": self.sequence,
                }
            return []

        if role != "tool":
            return []
        tool_id = str(event.get("tool_call_id") or "")
        pending = self.pending.pop(tool_id, None)
        if pending is None:
            if tool_id not in self.completed_ids:
                self.orphan_tool_result_ids.append(tool_id)
            return []
        self.completed_ids.add(tool_id)
        self.action_index += 1
        status = str(event.get("status") or "").lower()
        event_type = str(event.get("type") or "").lower()
        is_error = bool(event.get("is_error")) or status in {"error", "failed"} or (
            event_type == "error" or event_type.endswith(".error")
        )
        return [
            {
                **pending,
                "action_index": self.action_index,
                "result": copy.deepcopy(event.get("content")),
                "result_block": copy.deepcopy(dict(event)),
                "status": "failed" if is_error else "completed",
                "is_error": is_error,
                "completed_sequence": self.sequence,
            }
        ]


__all__ = ["KimiTraceBuilder"]
