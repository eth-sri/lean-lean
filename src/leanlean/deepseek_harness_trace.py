"""Trace reconstruction for DeepSeek Harness SDK session-event JSONL."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any


DSH_EVENT_TYPES = {
    "turn/start",
    "turn/end",
    "step/start",
    "step/end",
    "user/message",
    "assistant/chunk",
    "assistant/message",
    "tool/call",
    "tool/result",
    "request/header",
    "request/context",
    "session/end-seed",
}


def unwrap_dsh_event(record: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(session_id, event)`` from a runner envelope or raw event."""

    if record.get("type") == "dsh.session_event":
        event = record.get("event")
        return (
            str(record.get("session_id")) if record.get("session_id") else None,
            copy.deepcopy(dict(event)) if isinstance(event, Mapping) else None,
        )
    if record.get("type") in DSH_EVENT_TYPES and isinstance(record.get("data"), Mapping):
        return None, copy.deepcopy(dict(record))
    return None, None


def is_dsh_stream_record(record: Mapping[str, Any]) -> bool:
    if record.get("type") in {"dsh.session_event", "dsh.run_result", "dsh.error"}:
        return True
    return record.get("type") in DSH_EVENT_TYPES and isinstance(record.get("data"), Mapping)


def _tool_result_id(message: Mapping[str, Any]) -> str:
    direct = message.get("toolCallId") or message.get("tool_call_id")
    if direct:
        return str(direct)
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool-result":
                continue
            call_id = block.get("toolCallId") or block.get("tool_call_id")
            if call_id:
                return str(call_id)
    return ""


def _tool_result_content(message: Mapping[str, Any]) -> Any:
    content = message.get("content")
    if not isinstance(content, list):
        return copy.deepcopy(content)
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "tool-result":
            return copy.deepcopy(block.get("content"))
    return copy.deepcopy(content)


def _tool_result_error(message: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    if data.get("error"):
        return True
    content = message.get("content")
    return bool(
        isinstance(content, list)
        and any(
            isinstance(block, Mapping)
            and block.get("type") == "tool-result"
            and block.get("isError") is True
            for block in content
        )
    )


def dsh_response_from_usage(model: str, usage: Mapping[str, Any]) -> dict[str, Any]:
    """Convert DSH's disjoint usage counters into OpenAI-style accounting."""

    def count(name: str) -> int:
        value = usage.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    uncached = count("inputTokens")
    cache_read = count("cacheReadTokens")
    cache_write = count("cacheWriteTokens")
    output = count("outputTokens")
    reasoning = count("reasoningTokens")
    billed_input = uncached + cache_read + cache_write
    prompt_details: dict[str, int] = {}
    # An absent DSH counter is unknown, not evidence that the provider
    # reported zero. Preserve that distinction so incomplete cache evidence
    # blocks authoritative cost publication.
    if "cacheReadTokens" in usage:
        prompt_details["cached_tokens"] = cache_read
    if "cacheWriteTokens" in usage:
        prompt_details["cache_creation_tokens"] = cache_write
    return {
        "model": model or "deepseek-v4-flash",
        "usage": {
            "prompt_tokens": billed_input,
            "completion_tokens": output,
            "total_tokens": billed_input + output,
            "prompt_tokens_details": prompt_details,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class DeepSeekHarnessTraceBuilder:
    """Pair DSH tool events and retain terminal/session accounting facts."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.model = "deepseek-v4-flash"
        self.messages: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.pending: dict[str, dict[str, Any]] = {}
        self.duplicate_tool_call_ids: list[str] = []
        self.orphan_tool_result_ids: list[str] = []
        self.errors: list[dict[str, Any]] = []
        self.finish_reason: str | None = None
        self.final_response = ""
        self._sequence = 0

    def ingest(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        self._sequence += 1
        sequence = self._sequence
        record_type = str(record.get("type") or "")
        if record_type == "dsh.run_result":
            if record.get("session_id"):
                self.session_id = str(record["session_id"])
            self.finish_reason = str(record.get("finish_reason") or "") or None
            self.final_response = str(record.get("final_response") or "")
            return []
        if record_type == "dsh.error":
            self.errors.append(copy.deepcopy(dict(record)))
            return []

        session_id, event = unwrap_dsh_event(record)
        if event is None:
            return []
        if session_id:
            self.session_id = session_id
        event_type = str(event.get("type") or "")
        data = event.get("data")
        if not isinstance(data, Mapping):
            return []

        if event_type == "request/header":
            header = data.get("header")
            config = header.get("config") if isinstance(header, Mapping) else None
            if isinstance(config, Mapping) and config.get("model"):
                self.model = str(config["model"])
        elif event_type == "user/message":
            self.messages.append(
                {
                    "role": "user",
                    "content": copy.deepcopy(data.get("content")),
                    "dsh_event": event,
                }
            )
        elif event_type == "assistant/message":
            message = data.get("message")
            if isinstance(message, Mapping):
                normalized = copy.deepcopy(dict(message))
                normalized.setdefault("role", "assistant")
                normalized["dsh_event"] = event
                self.messages.append(normalized)
            usage = data.get("usage")
            if isinstance(usage, Mapping):
                self.responses.append(dsh_response_from_usage(self.model, usage))
        elif event_type == "tool/call":
            call_id = str(data.get("callId") or "")
            if not call_id:
                return []
            if call_id in self.pending:
                self.duplicate_tool_call_ids.append(call_id)
            self.pending[call_id] = {
                "tool_call_id": call_id,
                "name": str(data.get("name") or "unknown"),
                "arguments": data.get("arguments", "{}"),
                "turn": data.get("turn"),
                "step": data.get("step"),
                "started_sequence": sequence,
            }
        elif event_type == "tool/result":
            message = data.get("message")
            if not isinstance(message, Mapping):
                message = {}
            call_id = _tool_result_id(message)
            pending = self.pending.pop(call_id, None) if call_id else None
            if pending is None:
                self.orphan_tool_result_ids.append(call_id or f"sequence-{sequence}")
                pending = {
                    "tool_call_id": call_id,
                    "name": "unknown",
                    "arguments": "{}",
                    "turn": data.get("turn"),
                    "step": data.get("step"),
                    "started_sequence": None,
                }
            is_error = _tool_result_error(message, data)
            action = {
                **pending,
                "action_index": len(self.actions) + 1,
                "result": _tool_result_content(message),
                "result_block": copy.deepcopy(dict(message)),
                "status": "failed" if is_error else "succeeded",
                "is_error": is_error,
                "agent_id": self.session_id or "root",
                "completed_sequence": sequence,
            }
            self.actions.append(action)
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": copy.deepcopy(message.get("content")),
                    "dsh_event": event,
                }
            )
            return [copy.deepcopy(action)]
        elif event_type == "turn/end":
            reason = data.get("reason")
            if isinstance(reason, Mapping):
                self.finish_reason = str(reason.get("kind") or "") or None
                if self.finish_reason == "error":
                    error = reason.get("error")
                    self.errors.append(
                        copy.deepcopy(dict(error))
                        if isinstance(error, Mapping)
                        else {"message": str(error or "DSH turn failed")}
                    )
        return []

    def to_trace(self, *, native_stream_exact: bool = False) -> dict[str, Any]:
        success = self.finish_reason in {"completed", "max-tokens"} and not self.errors
        terminal: dict[str, Any] = {
            "success": success,
            "finish_reason": self.finish_reason,
            "errors": copy.deepcopy(self.errors),
        }
        if self.errors:
            latest = self.errors[-1]
            terminal["failure_reason"] = str(
                latest.get("message") or latest.get("error") or latest
            )
            status = latest.get("status")
            if isinstance(status, int):
                terminal["api_error_status"] = status
        return {
            "format": "deepseek-harness-sdk-jsonl-v1",
            "provider": "deepseek_harness",
            "session_id": self.session_id,
            "model": self.model,
            "messages": copy.deepcopy(self.messages),
            "actions": copy.deepcopy(self.actions),
            "responses": copy.deepcopy(self.responses),
            "pending_tools": [copy.deepcopy(item) for item in self.pending.values()],
            "terminal": terminal,
            "final_response": self.final_response,
            "quality": {
                "native_stream_exact": native_stream_exact,
                "tool_trace_exact": bool(
                    native_stream_exact
                    and not self.pending
                    and not self.duplicate_tool_call_ids
                    and not self.orphan_tool_result_ids
                ),
                "mutation_trace_exact": False,
                "mutation_trace_note": (
                    "DSH actions are exact, but standardized action replay is not enabled"
                ),
                "subagent_lineage_complete": False,
                "completed_action_order_exact": native_stream_exact,
                "duplicate_tool_call_ids": list(self.duplicate_tool_call_ids),
                "orphan_tool_result_ids": list(self.orphan_tool_result_ids),
            },
        }


def reconstruct_deepseek_harness_trace(
    records: list[dict[str, Any]], *, native_stream_exact: bool = False
) -> dict[str, Any]:
    builder = DeepSeekHarnessTraceBuilder()
    for record in records:
        builder.ingest(record)
    return builder.to_trace(native_stream_exact=native_stream_exact)


__all__ = [
    "DSH_EVENT_TYPES",
    "DeepSeekHarnessTraceBuilder",
    "dsh_response_from_usage",
    "is_dsh_stream_record",
    "reconstruct_deepseek_harness_trace",
    "unwrap_dsh_event",
]
