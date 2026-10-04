"""Compact, generator-agnostic action traces for run-level visualizations."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any


ACTION_CATEGORIES = (
    "Read",
    "Edit",
    "Search",
    "Lake",
    "Build",
    "Git",
    "Measure",
    "Verify",
    "Other",
)

_EXEC_PREFIX = r"(?:^|[;&|()\s'\"`])"
_EXEC_SUFFIX = r"(?=\s|$|[;&|()'\"`])"


def _has_executable(command: str, *names: str) -> bool:
    alternatives = "|".join(re.escape(name) for name in names)
    return re.search(
        rf"{_EXEC_PREFIX}(?:{alternatives}){_EXEC_SUFFIX}", command, re.IGNORECASE
    ) is not None


_COMMAND_PREFIX = (
    r"""(?:^|[;&|()]\s*|(?:-lc|-c)\s+['"]\s*)"""
    r"(?:[A-Za-z_][A-Za-z0-9_]*=[^\s;&|()]+\s+)*"
)


def _calls_executable(command: str, name: str) -> bool:
    """Whether a command invokes a named executable, optionally by path."""

    return re.search(
        rf"""{_COMMAND_PREFIX}(?:[^\s;&|()'"`]+/)?{re.escape(name)}{_EXEC_SUFFIX}""",
        command,
        re.IGNORECASE,
    ) is not None


def _calls_python_script(command: str, name: str) -> bool:
    """Whether a command invokes a named script through Python."""

    return re.search(
        rf"""{_COMMAND_PREFIX}(?:[^\s;&|()'"`]+/)?python(?:3(?:\.\d+)?)?\s+"""
        rf"""(?:[^\s;&|()'"`]+/)?{re.escape(name)}{_EXEC_SUFFIX}""",
        command,
        re.IGNORECASE,
    ) is not None


def classify_shell_command(command: str) -> str:
    """Classify one shell-tool turn by its dominant human-level operation."""

    command = command or ""
    lower = command.lower()

    if _calls_executable(lower, "lean_verify"):
        return "Verify"
    if _calls_executable(lower, "proof_length.py") or _calls_python_script(
        lower, "proof_length.py"
    ):
        return "Measure"

    # Explicit command families win for compound commands.
    if _calls_executable(lower, "git"):
        return "Git"
    if re.search(
        rf"""{_COMMAND_PREFIX}(?:[^\s;&|()'"`]+/)?lake\s+build{_EXEC_SUFFIX}""",
        lower,
        re.IGNORECASE,
    ):
        return "Build"
    if _calls_executable(lower, "lake"):
        return "Lake"

    # Mutations win for other compound edit-and-check commands: the durable
    # effect of the turn is the edit.
    shell_edit = (
        "apply_patch" in lower
        or re.search(r"\bsed\b[^;&|\n]*\s-i(?:\s|['\"]|$)", lower)
        or re.search(r"\bperl\b[^;&|\n]*\s-[a-z]*i[a-z]*(?:\s|['\"]|$)", lower)
        or re.search(r"\.(?:write_text|write_bytes)\s*\(", lower)
        or re.search(r"(?:^|[^0-9])>\s*[^;&|\n]*\.lean(?=\s|$|['\"])", lower)
        or re.search(r"\btee\b[^;&|\n]*\.lean(?=\s|$|['\"])", lower)
        or _has_executable(lower, "cp", "mv", "rm", "touch", "mkdir")
    )
    if shell_edit:
        return "Edit"

    if (
        _has_executable(lower, "wc", "du")
        or "numstat" in lower
        or "shortstat" in lower
        or "diffstat" in lower
        or "word_count" in lower
        or "count_words" in lower
        or re.search(r"\bcount(?:ing)?\b[^;&|\n]*\bwords?\b", lower)
    ):
        return "Measure"

    if _has_executable(lower, "rg", "grep", "find", "fd"):
        return "Search"

    if (
        _has_executable(lower, "cat", "head", "tail", "less", "nl")
        or re.search(rf"{_EXEC_PREFIX}sed\s+-n(?=\s|$)", lower)
    ):
        return "Read"

    return "Other"


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {"command": value}
        return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}
    return {}


def classify_tool_call(name: str | None, arguments: Any = None) -> str:
    """Classify a Codex, Claude Code, or OpenAI-style tool call."""

    tool = (name or "").lower()
    args = _arguments(arguments)

    if tool in {"read", "read_file", "view_file"}:
        return "Read"
    if tool in {"grep", "glob", "search", "search_files", "find_files"}:
        return "Search"
    if tool in {
        "edit",
        "multiedit",
        "write",
        "write_file",
        "apply_patch",
        "file_change",
    }:
        return "Edit"
    if tool.startswith("git"):
        return "Git"
    if tool == "lean_verify":
        return "Verify"
    if tool in {"proof_length", "proof_length.py"}:
        return "Measure"
    if tool.startswith("lean_") or "lean-lsp" in tool or tool.startswith("mcp__lean"):
        return "Search" if "search" in tool else "Other"

    command = args.get("command") or args.get("cmd") or ""
    if isinstance(command, str) and command:
        return classify_shell_command(command)
    return "Other"


def _codex_item_action(item: Mapping[str, Any]) -> str | None:
    item_type = item.get("type")
    if item_type == "file_change":
        return "Edit"
    if item_type == "command_execution":
        command = item.get("command")
        return classify_shell_command(command if isinstance(command, str) else "")
    if item_type in {"mcp_tool_call", "tool_call"}:
        return classify_tool_call(item.get("name"), item.get("arguments") or item.get("input"))
    return None


def _canonical_action_sequence(
    trajectory: Mapping[str, Any],
) -> list[str] | None:
    """Prefer canonical actions, then decode retained Vibe public effects."""

    canonical = trajectory.get("actions")
    if isinstance(canonical, list):
        return [
            classify_tool_call(
                str(action.get("name") or ""),
                action.get("arguments", action.get("input")),
            )
            for action in canonical
            if isinstance(action, Mapping)
        ]

    sources: list[Mapping[str, Any]] = [trajectory]
    provider_traces = trajectory.get("provider_traces")
    if isinstance(provider_traces, list):
        sources.extend(
            trace for trace in provider_traces if isinstance(trace, Mapping)
        )

    recognized = False
    sequence: list[str] = []
    for source in sources:
        canonical = source.get("actions")
        if isinstance(canonical, list):
            recognized = True
            for action in canonical:
                if not isinstance(action, Mapping):
                    continue
                sequence.append(
                    classify_tool_call(
                        str(action.get("name") or ""),
                        action.get("arguments", action.get("input")),
                    )
                )
            continue

        stream_events = source.get("stream_events")
        if (
            str(source.get("provider") or "").replace("-", "_")
            != "mistral_vibe"
            or not isinstance(stream_events, list)
        ):
            continue
        recognized = True
        for event in stream_events:
            if not isinstance(event, Mapping) or event.get("type") != "effect":
                continue
            generation_status = event.get(
                "generationStatus", event.get("generation_status")
            )
            state = event.get("state")
            state_status = (
                str(state.get("status") or "")
                if isinstance(state, Mapping)
                else ""
            )
            if generation_status != "completed" or state_status in {
                "",
                "pending",
                "running",
                "blocked",
            }:
                continue
            detail = event.get("detail")
            if not isinstance(detail, Mapping):
                continue
            sequence.append(
                classify_tool_call(
                    str(
                        detail.get("toolName")
                        or detail.get("tool_name")
                        or event.get("title")
                        or ""
                    ),
                    detail.get("input"),
                )
            )
    return sequence if recognized else None


def extract_action_sequence(trajectory: Mapping[str, Any]) -> list[str]:
    """Extract one ordered action category per tool turn without duplicates."""

    canonical = _canonical_action_sequence(trajectory)
    if canonical is not None:
        return canonical

    actions: list[str] = []
    seen_ids: set[str] = set()

    for message in trajectory.get("messages") or []:
        if not isinstance(message, Mapping):
            continue

        # Codex subscription traces retain the same completed item on the
        # assistant call and tool response.  Only the assistant-side item is an
        # action; item IDs provide an additional guard against duplicated data.
        item = message.get("codex_item")
        if message.get("role") == "assistant" and isinstance(item, Mapping):
            item_id = str(item.get("id") or "")
            if item_id and item_id in seen_ids:
                continue
            category = _codex_item_action(item)
            if category is not None:
                actions.append(category)
                if item_id:
                    seen_ids.add(item_id)
                continue

        # Claude Code stores its ordered calls as tool_use content blocks.
        content = message.get("content")
        if isinstance(content, list):
            found_block = False
            for block in content:
                if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                    continue
                block_id = str(block.get("id") or "")
                if block_id and block_id in seen_ids:
                    continue
                actions.append(classify_tool_call(block.get("name"), block.get("input")))
                if block_id:
                    seen_ids.add(block_id)
                found_block = True
            if found_block:
                continue

        # OpenAI-compatible tool_calls, including older Codex trajectories.
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                call_id = str(call.get("id") or "")
                if call_id and call_id in seen_ids:
                    continue
                function = call.get("function") or {}
                if not isinstance(function, Mapping):
                    function = {}
                actions.append(
                    classify_tool_call(function.get("name"), function.get("arguments"))
                )
                if call_id:
                    seen_ids.add(call_id)
            continue

        if message.get("attempt_message_type") == "tool_call":
            call_id = str(message.get("tool_call_id") or message.get("id") or "")
            if call_id and call_id in seen_ids:
                continue
            actions.append(classify_tool_call(message.get("name"), message.get("arguments")))
            if call_id:
                seen_ids.add(call_id)

    return actions


def summarize_action_sequences(sequences: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """Build a compact run summary and retain each repository's sequence."""

    normalized = {instance: list(sequence) for instance, sequence in sequences.items()}
    counts = {category: 0 for category in ACTION_CATEGORIES}
    max_turn = max((len(sequence) for sequence in normalized.values()), default=0)
    by_turn = [[0 for _ in ACTION_CATEGORIES] for _ in range(max_turn)]
    category_index = {category: index for index, category in enumerate(ACTION_CATEGORIES)}

    for sequence in normalized.values():
        for turn, category in enumerate(sequence):
            if category not in category_index:
                category = "Other"
            index = category_index[category]
            counts[category] += 1
            by_turn[turn][index] += 1

    return {
        "categories": list(ACTION_CATEGORIES),
        "counts": counts,
        "total": sum(counts.values()),
        "max_turn": max_turn,
        "by_turn": by_turn,
        "instances": {
            instance: {"sequence": sequence, "total": len(sequence)}
            for instance, sequence in sorted(normalized.items())
        },
    }
