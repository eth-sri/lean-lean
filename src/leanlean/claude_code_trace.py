"""Reconstruct Claude Code stream-json messages, tools, and subagent lineage."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


TRACE_FORMAT = "claude-code-stream-trace-v1"
ROOT_AGENT_ID = "root"
_SUBAGENT_TOOL_NAMES = {"agent", "task"}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", errors="surrogateescape")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class ClaudeTraceBuilder:
    """Incrementally reconstruct one Claude Code ``stream-json`` transcript.

    The source stream remains authoritative for semantic ordering. Exact source
    states are attached later by the playback recorder after completed tool
    results, because a tool transcript alone cannot prove filesystem effects.
    """

    sequence: int = 0
    raw_event_count: int = 0
    duplicate_event_count: int = 0
    event_counts: Counter[str] = field(default_factory=Counter)
    stream_events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    result_event: dict[str, Any] | None = None
    _seen_semantic_events: dict[str, int] = field(default_factory=dict)
    _tool_uses: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ensure_agent(ROOT_AGENT_ID)

    @staticmethod
    def _agent_id(parent_tool_use_id: Any) -> str:
        return str(parent_tool_use_id) if parent_tool_use_id else ROOT_AGENT_ID

    def _ensure_agent(self, agent_id: str) -> dict[str, Any]:
        if agent_id not in self.agents:
            self.agents[agent_id] = {
                "agent_id": agent_id,
                "parent_agent_id": None if agent_id == ROOT_AGENT_ID else ROOT_AGENT_ID,
                "spawn_tool_use_id": None if agent_id == ROOT_AGENT_ID else agent_id,
                "spawn_sequence": None,
                "tool_name": None,
                "input": None,
                "message_sequences": [],
                "tool_use_ids": [],
            }
        return self.agents[agent_id]

    def _register_subagent(
        self,
        *,
        tool_use_id: str,
        parent_agent_id: str,
        sequence: int,
        name: str,
        arguments: Any,
    ) -> None:
        agent = self._ensure_agent(tool_use_id)
        agent.update(
            {
                "parent_agent_id": parent_agent_id,
                "spawn_tool_use_id": tool_use_id,
                "spawn_sequence": sequence,
                "tool_name": name,
                "input": copy.deepcopy(arguments),
            }
        )

    def ingest(self, event: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Ingest one decoded event and return newly normalized objects."""

        self.raw_event_count += 1
        self.sequence += 1
        sequence = self.sequence
        event_type = str(event.get("type") or "unknown")
        self.event_counts[event_type] += 1
        raw = copy.deepcopy(event)
        raw["trace_sequence"] = sequence
        self.stream_events.append(raw)

        if event_type == "result":
            self.result_event = copy.deepcopy(event)
            return {"messages": [], "completed_actions": []}

        message = event.get("message")
        if event_type not in {"assistant", "user"} or not isinstance(message, dict):
            return {"messages": [], "completed_actions": []}

        parent_tool_use_id = event.get("parent_tool_use_id")
        agent_id = self._agent_id(parent_tool_use_id)
        agent = self._ensure_agent(agent_id)
        semantic_basis = {
            "type": event_type,
            "request_id": event.get("request_id"),
            "parent_tool_use_id": parent_tool_use_id,
            "message": message,
        }
        digest = _canonical_digest(semantic_basis)
        if digest in self._seen_semantic_events:
            self.duplicate_event_count += 1
            return {"messages": [], "completed_actions": []}
        self._seen_semantic_events[digest] = sequence

        normalized = copy.deepcopy(message)
        normalized["trace_sequence"] = sequence
        normalized["trace_agent_id"] = agent_id
        if event.get("request_id") is not None:
            normalized["request_id"] = event.get("request_id")
        if parent_tool_use_id is not None:
            normalized["parent_tool_use_id"] = parent_tool_use_id
        self.messages.append(normalized)
        agent["message_sequences"].append(sequence)

        completed: list[dict[str, Any]] = []
        content = message.get("content") or []
        if not isinstance(content, list):
            return {"messages": [normalized], "completed_actions": completed}

        if event_type == "assistant":
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_use_id = str(block.get("id") or "")
                if not tool_use_id or tool_use_id in self._tool_uses:
                    continue
                name = str(block.get("name") or "unknown")
                invocation = {
                    "action_index": None,
                    "tool_use_id": tool_use_id,
                    "name": name,
                    "input": copy.deepcopy(block.get("input")),
                    "agent_id": agent_id,
                    "parent_tool_use_id": parent_tool_use_id,
                    "request_id": event.get("request_id"),
                    "started_sequence": sequence,
                    "completed_sequence": None,
                    "status": "pending",
                    "is_error": None,
                    "result": None,
                }
                self._tool_uses[tool_use_id] = invocation
                agent["tool_use_ids"].append(tool_use_id)
                if name.lower() in _SUBAGENT_TOOL_NAMES:
                    self._register_subagent(
                        tool_use_id=tool_use_id,
                        parent_agent_id=agent_id,
                        sequence=sequence,
                        name=name,
                        arguments=block.get("input"),
                    )

        if event_type == "user":
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = str(block.get("tool_use_id") or "")
                invocation = self._tool_uses.get(tool_use_id)
                if invocation is None:
                    invocation = {
                        "action_index": None,
                        "tool_use_id": tool_use_id,
                        "name": "unknown",
                        "input": None,
                        "agent_id": agent_id,
                        "parent_tool_use_id": parent_tool_use_id,
                        "request_id": event.get("request_id"),
                        "started_sequence": None,
                        "completed_sequence": None,
                        "status": "orphan_result",
                        "orphan_result": True,
                        "is_error": None,
                        "result": None,
                    }
                    self._tool_uses[tool_use_id] = invocation
                if invocation.get("completed_sequence") is not None:
                    continue
                invocation["action_index"] = len(self.actions) + 1
                invocation["completed_sequence"] = sequence
                if not invocation.get("orphan_result"):
                    invocation["status"] = (
                        "failed" if block.get("is_error") else "completed"
                    )
                invocation["is_error"] = bool(block.get("is_error"))
                invocation["result"] = copy.deepcopy(block.get("content"))
                invocation["result_block"] = copy.deepcopy(block)
                self.actions.append(invocation)
                completed.append(invocation)

        return {"messages": [normalized], "completed_actions": completed}

    @property
    def pending_tool_uses(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(action)
            for action in self._tool_uses.values()
            if action.get("completed_sequence") is None
        ]

    def to_trace(self, *, include_messages: bool = True) -> dict[str, Any]:
        pending = self.pending_tool_uses
        orphan_count = sum(bool(action.get("orphan_result")) for action in self.actions)
        quality = (
            "ordered_complete_tool_trace"
            if not pending and not orphan_count and self.result_event is not None
            else "ordered_partial_tool_trace"
        )
        terminal: dict[str, Any] = {}
        if self.result_event is not None:
            terminal = {
                key: copy.deepcopy(self.result_event.get(key))
                for key in (
                    "subtype",
                    "is_error",
                    "duration_ms",
                    "duration_api_ms",
                    "num_turns",
                    "session_id",
                    "total_cost_usd",
                    "modelUsage",
                )
                if key in self.result_event
            }
        trace = {
            "format": TRACE_FORMAT,
            "quality": quality,
            "ordering_basis": "Claude Code stream-json receipt order",
            "subagent_basis": (
                "parent_tool_use_id emitted by --forward-subagent-text"
            ),
            "filesystem_exactness": (
                "tool semantics only; exact source states are attached by playback"
            ),
            "raw_event_count": self.raw_event_count,
            "semantic_message_count": len(self.messages),
            "duplicate_event_count": self.duplicate_event_count,
            "event_counts": dict(sorted(self.event_counts.items())),
            "completed_tool_count": len(self.actions),
            "pending_tool_count": len(pending),
            "orphan_result_count": orphan_count,
            "agents": [copy.deepcopy(self.agents[key]) for key in sorted(self.agents)],
            "actions": copy.deepcopy(self.actions),
            "pending_tools": pending,
            "terminal": terminal,
            "contains_full_tool_inputs": True,
            "contains_tool_results": True,
            "contains_decoded_stream_events": True,
            "sensitive": True,
        }
        if include_messages:
            trace["messages"] = copy.deepcopy(self.messages)
            trace["stream_events"] = copy.deepcopy(self.stream_events)
        return trace


def reconstruct_claude_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    builder = ClaudeTraceBuilder()
    for event in events:
        builder.ingest(event)
    return builder.to_trace()
