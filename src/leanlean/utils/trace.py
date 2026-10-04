"""Helpers to load and organize LeanLean trace files."""
from leanlean.utils import json_utils as json
from dataclasses import dataclass
from markdown_it import MarkdownIt
from pathlib import Path
import shlex
from typing import Any
import re
import textwrap

from configs.model_constants import get_model_prices
from leanlean.utils.diff_file import DiffFile
from leanlean.utils.others import retry
from leanlean import Model


TOOL_INFER_PROMPT = textwrap.dedent(
    """ 
    You are labeling a tool call with a single intent category.

    Goal: choose a category name that is:
    - Right-sized granularity: more specific than "execute command" but not tied to exact args.
    - Reusable: should apply to many future tool calls.
    - Clean: DO NOT include file paths, flags, quoted strings, IDs, repo names, or counts.
    - Format: 2–5 words, lowercase, verb + object (e.g., "run tests", "search codebase"). Avoid too generic names like "run scripts". In this case specify what the script does (e.g., "compile code").
    You will also need to explain which tool is being used (e.g. pytest, rg, ...) in a dedicated field.

    You will be given:
    - tool_call: the command or structured tool invocation
    - tool_output: optional output text

    Existing categories (use one if it fits):
    {existing_tool_names}

    Decision rules:
    1) If one existing category fits, use it exactly.
    2) If none fit, create ONE new category that:
    - is not tool-specific (avoid "pytest", "kubectl", "terraform", etc.)
    - would likely match 5+ future tool calls
    3) If the tool call does multiple things, pick the PRIMARY intent as the category.
    (Mention secondary intents in reasoning.)

    Return JSON ONLY:
    {{
        "tool_name": "<category>",
        "tool_used": "<specific tool or executable being invoked>",
        "reasoning": "<1–3 sentences: why this is the primary intent; include key clues from call/output; mention secondary intents if any>"
    }}

    Tool call:
    ```
    {tool_call}
    ```

    Tool output (if any):
    ```
    {tool_output}
    ```
    """
).strip()


@dataclass(frozen=True)
class SystemComponent:
    index: int
    message: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None


@dataclass(frozen=True)
class UserComponent:
    index: int
    message: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None

@dataclass(frozen=True)
class AssistantComponent:
    index: int
    message: str
    tool_calls: list[dict[str, Any]]
    raw_content: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None


@dataclass(frozen=False)
class ToolComponent:
    index: int
    tool_name: str
    tool_call: str
    tool_output: str
    exit_code: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_cost: float | None = None

    def _extract_exit_code(self, generator: str) -> int | None:
        text = self.tool_output
        error_code = 0

        if generator == "claude_code":
            m = re.search(r'(?i)\bexit\s*code\s*[:=]?\s*(\d+)\b', text)
            error_code = int(m.group(1)) if m else 0

        elif generator == "qwen_code":
            m = re.search(r'(?im)^\s*exit\s*code\s*[:=]?\s*(\d+)\s*$', text)
            error_code = int(m.group(1)) if m else 0

        elif generator == "codex":
            error_code = self.exit_code if self.exit_code is not None else 0

        return error_code 

    def extract_exit_code(self) -> int | None:
        error_code = 0

        if self.tool_name == "Bash":
            error_code = self._extract_exit_code("claude_code")

        elif self.tool_name == "run_shell_command":
            error_code = self._extract_exit_code("qwen_code")

        elif self.tool_name == "shell":
            error_code = self._extract_exit_code("codex")

        return error_code 

    def is_error(self) -> bool:
        if self.tool_name in {"Bash", "run_shell_command", "shell"}:
            if "Exit code 1" in self.tool_output or "Exit Code: 1" in self.tool_output:
                return True
        return False


    @retry(5) # retry up to 5 times
    def get_name_by_llm(self, model: Model, existing_tool_names: set[str]) -> str:
        """Let an LLM try to infer a better tool name from the tool call
        """
        if self.tool_name not in {"Bash", "run_shell_command", "shell"}:
            return self.tool_name

        simplified_existing_names = {name.replace("Bash:", "") for name in existing_tool_names if name}
        simplified_existing_names = {name.split("::")[0] for name in simplified_existing_names if name}

        tool_call = (self.tool_call or "").strip()
        tool_output = (self.tool_output or "").strip()
        prompt = TOOL_INFER_PROMPT.format(
            existing_tool_names=", ".join(
                f'"{name}"' for name in sorted(simplified_existing_names) if name
            ),
            tool_call=tool_call or "<empty>",
            tool_output=tool_output or "<empty>",
        )
        response = model.query([{"role": "user", "content": prompt}])
        payload = json.loads(response) # In case of parsing error, we retry
    
        inferred_name = payload.get("tool_name")
        if not inferred_name or not isinstance(inferred_name, str):
            return self.tool_name
        inferred_name = inferred_name.strip()

        tool_used = payload.get("tool_used", "").strip()

        inferred_name = f"Bash:{inferred_name}::{tool_used}"

        return inferred_name or self.tool_name

    

@dataclass
class TokenUsage:

    # Tokens count
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0

    completion_tokens: int = 0
    reasoning_tokens: int = 0

    total_tokens: int = 0

    # Cache read/write counts
    cache_creation_tokens: int = 0

    def add(self, usage: dict[str, int]) -> None:
        def _coerce_int(value: int | None) -> int:
            return 0 if value is None else value

        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("input_tokens", 0)
        prompt_tokens = _coerce_int(prompt_tokens)

        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens", 0)
        completion_tokens = _coerce_int(completion_tokens)

        total_tokens = _coerce_int(usage.get("total_tokens", 0))
        
        try:
            prompt_details = usage.get("prompt_tokens_details")
            if prompt_details is None:
                prompt_details = usage.get("input_tokens_details", {})
            cached_prompt_tokens = _coerce_int(prompt_details.get("cached_tokens", 0))
            cache_creation_tokens = _coerce_int(prompt_details.get("cache_creation_tokens", 0))
        except AttributeError:
            cached_prompt_tokens = 0
            cache_creation_tokens = 0

        try:
            reasoning_tokens = _coerce_int(
                usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            )
        except AttributeError:
            reasoning_tokens = 0

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        self.cached_prompt_tokens += cached_prompt_tokens
        self.cache_creation_tokens += cache_creation_tokens
        self.reasoning_tokens += reasoning_tokens

    def price(self, input_token_price: float, output_token_price: float, cache_token_price: float = 0, cache_creation_token_price: float = 0) -> float:

        # prompt_tokens is the *total* input and already includes both the
        # cache-read (cached) and cache-creation tokens, so both must be
        # subtracted to get the fresh, uncached input that's billed at the base
        # input rate. Cache-read and cache-creation are then billed separately
        # at their own rates. Subtracting only cached_prompt_tokens (the old
        # behaviour) double-charged every cache-write token at input_price.
        fresh_input = self.prompt_tokens - self.cached_prompt_tokens - self.cache_creation_tokens
        input_cost = fresh_input * input_token_price
        output_cost = self.completion_tokens * output_token_price
        cache_cost = self.cached_prompt_tokens * cache_token_price
        cache_creation_cost = self.cache_creation_tokens * cache_creation_token_price

        return (input_cost + output_cost + cache_cost + cache_creation_cost) / 1_000_000 # Prices are per million tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }


@dataclass(frozen=True)
class UsageInfo:
    response_index: int
    prompt_tokens: int
    completion_tokens: int
    total_cost: float | None


class Trace:
    """Load a LeanLean trace and expose useful components."""

    _MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True, "linkify": True})

    def __init__(self, data: dict[str, Any], *, path: Path | None = None) -> None:
        self.path = path
        self.raw = data
        self.info: dict[str, Any] = data.get("info") or {}
        raw_messages = list(data.get("messages") or [])
        if self._looks_like_codex_messages(raw_messages):
            self.messages = self._normalize_codex_messages(raw_messages)
        else:
            self.messages = raw_messages
        self.responses: list[dict[str, Any]] = list(data.get("responses") or [])
        self.has_attempt_events = any(
            isinstance(message, dict) and message.get("type") == "attempt_event"
            for message in self.messages
        )
        self.trajectory_format = data.get("trajectory_format")
        self.instance_id = data.get("instance_id")
        self.metadata = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "info",
                "messages",
                "responses",
                "trajectory_format",
                "instance_id",
            }
        }
        self.messages_by_role = self._group_by_role(self.messages)
        self.tool_calls = self._collect_tool_calls(self.messages)
        self.tool_uses = self._collect_tool_uses(self.messages)
        if self.tool_uses:
            self.messages_by_role.setdefault("tool_use", []).extend(self.tool_uses)
        self.tool_results = self._collect_tool_results(self.messages)
        self._inject_tool_role(self.tool_results, self.messages_by_role)
        (
            self._usage_by_message_index,
            self._usage_by_tool_call_id,
        ) = self._collect_usage_mappings(self.messages, self.responses)
        self.system_components = self._build_system_components(
            self.messages, self._usage_by_message_index
        )
        self.user_components = self._build_user_components(
            self.messages, self._usage_by_message_index
        )
        self.assistant_components = self._build_assistant_components(
            self.messages, self._usage_by_message_index
        )
        self.tool_components = self._build_tool_components(
            self.tool_uses, self._usage_by_tool_call_id
        )
        self.steps = self._build_steps(
            self.messages,
            self.tool_uses,
            self._usage_by_message_index,
            self._usage_by_tool_call_id,
        )
        self._steps = self.steps

    def get_first_read_file(self, file_names: list[str]) -> dict[str, Any] | None:
        """Compute the number of tokens/steps/cost until the first read of any of the given files."""

        file_name_set = set(file_names)
        cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0

        for idx, step in enumerate(self.steps):

            prompt_tokens += step.prompt_tokens or 0
            completion_tokens += step.completion_tokens or 0
            cost += step.total_cost or 0.0

            if isinstance(step, ToolComponent):
                tool_call_text = step.tool_call or ""
                for file_name in file_name_set:
                    if file_name in tool_call_text:
                        return {
                            "number_steps": idx,
                            "cost": cost,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        }

        return None


    @classmethod
    def from_path(cls, path: str | Path) -> "Trace":
        trace_path = Path(path)
        return cls.from_text(trace_path.read_text(), path=trace_path)

    @classmethod
    def from_text(cls, text: str, *, path: Path | None = None) -> "Trace":
        data = cls._parse_text(text)
        return cls(data, path=path)

    def visualize(self) -> str:
        return self._render_steps_html(
            self._steps,
            self.info,
            has_attempt_events=self.has_attempt_events,
        )

    @staticmethod
    def _parse_text(text: str) -> dict[str, Any]:
        payload = text.strip()
        if not payload:
            raise ValueError("Trace is empty.")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Trace must be a JSON object.")
        return data

    @staticmethod
    def _looks_like_codex_messages(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            if isinstance(message, dict) and message.get("type") in {
                "reasoning",
                "function_call",
                "function_call_output",
            }:
                return True
        return False

    @classmethod
    def _normalize_codex_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            msg_type = message.get("type")
            if msg_type == "reasoning":
                updated = dict(message)
                updated.setdefault("role", "assistant")
                if updated.get("content") is None:
                    updated["content"] = "reasoning"
                normalized.append(updated)
            else:
                normalized.append(message)
        return normalized

    @staticmethod
    def _coerce_token_count(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return 0

    @classmethod
    def _extract_usage_counts(cls, usage: dict[str, Any]) -> tuple[int, int, int, int]:
        prompt_tokens = cls._coerce_token_count(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        completion_tokens = cls._coerce_token_count(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        cached_prompt_tokens = 0
        cache_creation_tokens = 0

        details = usage.get("prompt_tokens_details")
        if details is None:
            details = usage.get("input_tokens_details")
        if isinstance(details, dict):
            cached_prompt_tokens = cls._coerce_token_count(details.get("cached_tokens"))
            cache_creation_tokens = cls._coerce_token_count(
                details.get("cache_creation_tokens")
            )

        cached_prompt_tokens = max(
            cached_prompt_tokens,
            cls._coerce_token_count(usage.get("cache_read_input_tokens")),
        )
        cache_creation_tokens = max(
            cache_creation_tokens,
            cls._coerce_token_count(usage.get("cache_creation_input_tokens")),
        )

        min_prompt_tokens = cached_prompt_tokens + cache_creation_tokens
        if prompt_tokens < min_prompt_tokens:
            prompt_tokens = min_prompt_tokens

        return (
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
            cache_creation_tokens,
        )

    @classmethod
    def _usage_info_from_response(
        cls,
        response: dict[str, Any],
        response_index: int,
    ) -> UsageInfo:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return UsageInfo(
                response_index=response_index,
                prompt_tokens=0,
                completion_tokens=0,
                total_cost=None,
            )

        (
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
            cache_creation_tokens,
        ) = cls._extract_usage_counts(usage)

        model = response.get("model") or "unknown_model"
        prices = get_model_prices(str(model))
        total_cost = None
        if prices is not None:
            input_price, output_price, cache_price, cache_creation_price = prices
            noncached_prompt_tokens = max(prompt_tokens - cached_prompt_tokens, 0)
            input_cost = noncached_prompt_tokens * input_price
            output_cost = completion_tokens * output_price
            cache_cost = cached_prompt_tokens * cache_price
            cache_creation_cost = cache_creation_tokens * cache_creation_price
            total_cost = (
                input_cost + output_cost + cache_cost + cache_creation_cost
            ) / 1_000_000

        return UsageInfo(
            response_index=response_index,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=total_cost,
        )

    @staticmethod
    def _tool_call_id_from_payload(payload: dict[str, Any]) -> str | None:
        if not isinstance(payload, dict):
            return None
        tool_id = (
            payload.get("id")
            or payload.get("call_id")
            or payload.get("tool_call_id")
            or payload.get("tool_use_id")
        )
        if tool_id is None:
            return None
        return str(tool_id)

    @classmethod
    def _collect_response_tool_call_ids(cls, response: dict[str, Any]) -> list[str]:
        tool_call_ids: list[str] = []

        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"function_call", "tool_call"}:
                    call_id = cls._tool_call_id_from_payload(item)
                    if call_id:
                        tool_call_ids.append(call_id)

        choices = response.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                tool_calls = message.get("tool_calls") or []
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        call_id = cls._tool_call_id_from_payload(call)
                        if call_id:
                            tool_call_ids.append(call_id)
                function_call = message.get("function_call")
                if isinstance(function_call, dict):
                    call_id = cls._tool_call_id_from_payload(function_call)
                    if call_id:
                        tool_call_ids.append(call_id)
                for item in cls._iter_content_items(message):
                    if item.get("type") == "tool_use":
                        call_id = cls._tool_call_id_from_payload(item)
                        if call_id:
                            tool_call_ids.append(call_id)

        return tool_call_ids

    @classmethod
    def _map_usage_to_assistant_messages(
        cls,
        messages: list[dict[str, Any]],
        usage_entries: list[UsageInfo],
        usage_by_tool_call_id: dict[str, UsageInfo],
    ) -> dict[int, UsageInfo]:
        usage_by_message_index: dict[int, UsageInfo] = {}
        used_response_indices: set[int] = set()
        assistant_indices: list[int] = []

        for index, message in enumerate(messages):
            tool_calls: list[dict[str, Any]] = []
            if message.get("type") == "reasoning":
                assistant_indices.append(index)
                tool_calls = cls._codex_tool_calls_after(messages, index)
            elif message.get("role") == "assistant":
                assistant_indices.append(index)
                tool_calls = cls._extract_tool_calls(message)
            else:
                continue

            for call in tool_calls:
                call_id = cls._tool_call_id_from_payload(call)
                if not call_id:
                    continue
                usage_info = usage_by_tool_call_id.get(call_id)
                if usage_info and usage_info.response_index not in used_response_indices:
                    usage_by_message_index[index] = usage_info
                    used_response_indices.add(usage_info.response_index)
                    break

        remaining_entries = [
            entry
            for entry in usage_entries
            if entry.response_index not in used_response_indices
        ]
        remaining_indices = [
            index
            for index in assistant_indices
            if index not in usage_by_message_index
        ]
        for index, usage_info in zip(remaining_indices, remaining_entries):
            usage_by_message_index[index] = usage_info
            used_response_indices.add(usage_info.response_index)

        return usage_by_message_index

    @classmethod
    def _collect_usage_mappings(
        cls,
        messages: list[dict[str, Any]],
        responses: list[dict[str, Any]],
    ) -> tuple[dict[int, UsageInfo], dict[str, UsageInfo]]:
        usage_entries: list[UsageInfo] = []
        usage_by_tool_call_id: dict[str, UsageInfo] = {}

        for response_index, response in enumerate(responses):
            if not isinstance(response, dict):
                usage_entries.append(
                    UsageInfo(
                        response_index=response_index,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_cost=None,
                    )
                )
                continue
            usage_info = cls._usage_info_from_response(response, response_index)
            usage_entries.append(usage_info)
            for tool_call_id in cls._collect_response_tool_call_ids(response):
                if tool_call_id and tool_call_id not in usage_by_tool_call_id:
                    usage_by_tool_call_id[tool_call_id] = usage_info

        usage_by_message_index = cls._map_usage_to_assistant_messages(
            messages, usage_entries, usage_by_tool_call_id
        )

        return usage_by_message_index, usage_by_tool_call_id

    @classmethod
    def _codex_tool_calls_after(
        cls,
        messages: list[dict[str, Any]],
        start_index: int,
    ) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        for message in messages[start_index + 1 :]:
            msg_type = message.get("type")
            role = message.get("role")
            if msg_type in {"reasoning", "message"} or role in {"user", "assistant"}:
                break
            if msg_type == "function_call":
                tool_calls.append(message)
        return tool_calls

    @staticmethod
    def _get_last_user_index(messages: list[dict[str, Any]]) -> int | None:
        if any(
            isinstance(message, dict) and message.get("type") == "attempt_event"
            for message in messages
        ):
            return None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return None

    @staticmethod
    def _clean_diff_text(text: str) -> str:
        if not text:
            return ""
        try:
            return DiffFile.from_text(text).get_cleaned_diff()
        except Exception:
            return text

    @staticmethod
    def _group_by_role(
        messages: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for message in messages:
            role = message.get("role") or "unknown"
            grouped.setdefault(role, []).append(message)
        return grouped

    @staticmethod
    def _collect_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for message in messages:
            if message.get("type") == "function_call":
                calls.append(message)
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if isinstance(call, dict):
                    calls.append(call)
                else:
                    calls.append({"raw": call})
        return calls

    @classmethod
    def _collect_tool_uses(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        uses_by_id: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def ensure(tool_id: str | None) -> dict[str, Any] | None:
            if not tool_id:
                return None
            if tool_id not in uses_by_id:
                uses_by_id[tool_id] = {
                    "id": tool_id,
                    "tool_name": None,
                    "tool_call": None,
                    "tool_output": [],
                }
                order.append(tool_id)
            return uses_by_id[tool_id]

        for message in messages:
            msg_type = message.get("type")
            if msg_type == "function_call":
                entry = ensure(message.get("call_id") or message.get("id"))
                if entry:
                    tool_name = message.get("name")
                    if tool_name and not entry["tool_name"]:
                        entry["tool_name"] = tool_name
                    if entry["tool_call"] is None:
                        entry["tool_call"] = message

            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                entry = ensure(call.get("id"))
                if not entry:
                    continue
                tool_name = cls._tool_name_from_call(call)
                if tool_name and not entry["tool_name"]:
                    entry["tool_name"] = tool_name
                if entry["tool_call"] is None:
                    entry["tool_call"] = call

            for item in cls._iter_content_items(message):
                if item.get("type") == "tool_use":
                    entry = ensure(item.get("id") or item.get("tool_use_id"))
                    if not entry:
                        continue
                    tool_name = item.get("name")
                    if tool_name and not entry["tool_name"]:
                        entry["tool_name"] = tool_name
                    if entry["tool_call"] is None:
                        entry["tool_call"] = item

            if msg_type == "function_call_output":
                entry = ensure(
                    message.get("call_id")
                    or message.get("tool_call_id")
                    or message.get("tool_use_id")
                )
                if entry is not None:
                    entry["tool_output"].append(message)
            elif message.get("role") == "tool":
                entry = ensure(message.get("tool_call_id") or message.get("tool_use_id"))
                if entry is not None:
                    entry["tool_output"].append(message)

            for item in cls._iter_content_items(message):
                if item.get("type") == "tool_result":
                    entry = ensure(
                        item.get("tool_use_id") or item.get("tool_call_id") or item.get("id")
                    )
                    if entry is not None:
                        entry["tool_output"].append(item)

        return [uses_by_id[tool_id] for tool_id in order]

    @staticmethod
    def _iter_content_items(message: dict[str, Any]):
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    yield item

    @staticmethod
    def _tool_name_from_call(call: dict[str, Any]) -> str | None:
        name = call.get("name")
        if isinstance(name, str) and name:
            return name
        function = call.get("function")
        if isinstance(function, dict):
            func_name = function.get("name")
            if isinstance(func_name, str) and func_name:
                return func_name
        return None

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        text_types = {"text", "input_text", "output_text"}
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in text_types:
                        parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            if content.get("type") in text_types:
                return str(content.get("text", ""))
            return Trace._stringify_value(content)
        return str(content)

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=True)
        except TypeError:
            return str(value)

    @staticmethod
    def _unescape_newlines(text: str) -> str:
        if "\\n" not in text:
            return text
        return text.replace("\\r\\n", "\n").replace("\\n", "\n")

    @staticmethod
    def _extract_output_text(text: str) -> str:
        try:
            payload = json.loads(text)
        except Exception:
            return text
        if isinstance(payload, dict) and isinstance(payload.get("output"), str):
            return payload["output"]
        return text

    @staticmethod
    def _coerce_exit_code(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
        return None

    @classmethod
    def _extract_exit_code_from_metadata(cls, payload: Any) -> int | None:
        if not isinstance(payload, dict):
            return None
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return cls._coerce_exit_code(metadata.get("exit_code"))

    @classmethod
    def _extract_exit_code_from_json_text(cls, text: str) -> int | None:
        if not text:
            return None
        candidate = text.strip()
        if not candidate.startswith("{"):
            return None
        try:
            payload = json.loads(candidate)
        except Exception:
            return None
        if isinstance(payload, dict):
            return cls._extract_exit_code_from_metadata(payload)
        return None

    @classmethod
    def _extract_exit_code_from_output_value(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, dict):
            exit_code = cls._extract_exit_code_from_metadata(value)
            if exit_code is not None:
                return exit_code
            if "output" in value:
                exit_code = cls._extract_exit_code_from_output_value(value.get("output"))
                if exit_code is not None:
                    return exit_code
            if "content" in value:
                exit_code = cls._extract_exit_code_from_output_value(value.get("content"))
                if exit_code is not None:
                    return exit_code
            return None
        if isinstance(value, list):
            for item in value:
                exit_code = cls._extract_exit_code_from_output_value(item)
                if exit_code is not None:
                    return exit_code
            return None
        if isinstance(value, str):
            return cls._extract_exit_code_from_json_text(value)
        return None

    @classmethod
    def _extract_exit_code_from_tool_call(cls, tool_call: Any) -> int | None:
        if not isinstance(tool_call, dict):
            return None
        exit_code = cls._extract_exit_code_from_metadata(tool_call)
        if exit_code is not None:
            return exit_code
        for key in ("arguments", "input"):
            if key in tool_call:
                exit_code = cls._extract_exit_code_from_output_value(tool_call.get(key))
                if exit_code is not None:
                    return exit_code
        function = tool_call.get("function")
        if isinstance(function, dict):
            exit_code = cls._extract_exit_code_from_metadata(function)
            if exit_code is not None:
                return exit_code
            exit_code = cls._extract_exit_code_from_output_value(function.get("arguments"))
            if exit_code is not None:
                return exit_code
        return None

    @classmethod
    def _extract_exit_code_from_tool_output(
        cls,
        tool_output: list[dict[str, Any]],
    ) -> int | None:
        for item in tool_output:
            exit_code = cls._extract_exit_code_from_output_value(item)
            if exit_code is not None:
                return exit_code
        return None

    @classmethod
    def _extract_exit_code_from_tool_use(cls, tool_use: dict[str, Any]) -> int | None:
        exit_code = cls._extract_exit_code_from_tool_call(tool_use.get("tool_call"))
        if exit_code is not None:
            return exit_code
        tool_output = tool_use.get("tool_output") or []
        return cls._extract_exit_code_from_tool_output(tool_output)

    @classmethod
    def _extract_tool_calls(cls, message: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tool_calls = message.get("tool_calls")
        tool_calls = list(raw_tool_calls) if isinstance(raw_tool_calls, list) else []
        for item in cls._iter_content_items(message):
            if item.get("type") == "tool_use":
                tool_calls.append(item)
        return tool_calls

    @classmethod
    def _tool_component_from_use(
        cls,
        index: int,
        tool_use: dict[str, Any],
        usage_by_tool_call_id: dict[str, UsageInfo] | None = None,
    ) -> ToolComponent:
        tool_name = cls._stringify_value(tool_use.get("tool_name"))
        tool_call = cls._stringify_value(tool_use.get("tool_call"))
        tool_output = cls._stringify_tool_output(tool_use.get("tool_output") or [])
        exit_code = cls._extract_exit_code_from_tool_use(tool_use)
        tool_id = tool_use.get("id")
        usage_info = (
            usage_by_tool_call_id.get(tool_id)
            if usage_by_tool_call_id and tool_id
            else None
        )
        return ToolComponent(
            index=index,
            tool_name=tool_name,
            tool_call=tool_call,
            tool_output=tool_output,
            exit_code=exit_code,
            prompt_tokens=usage_info.prompt_tokens if usage_info else None,
            completion_tokens=usage_info.completion_tokens if usage_info else None,
            total_cost=usage_info.total_cost if usage_info else None,
        )

    @classmethod
    def _stringify_tool_output(cls, tool_output: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in tool_output:
            if isinstance(item, dict):
                if "content" in item:
                    text = cls._stringify_value(item.get("content"))
                elif "output" in item:
                    output_value = item.get("output")
                    if isinstance(output_value, str):
                        text = output_value
                    else:
                        text = cls._stringify_value(output_value)
                else:
                    text = cls._stringify_value(item)
            else:
                text = cls._stringify_value(item)
            text = cls._extract_output_text(text)
            text = cls._unescape_newlines(text)
            parts.append(text)
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @classmethod
    def _render_markdown_html(cls, text: str) -> str:
        if not text:
            return ""
        return cls._MARKDOWN.render(text)

    @classmethod
    def _fmt_code_block(cls, text: str) -> str:
        return f'<pre class="code-block">{cls._escape_html(text or "")}</pre>'

    @classmethod
    def _render_preview_details(
        cls,
        details_class: str,
        summary_class: str,
        full_class: str,
        title_class: str,
        preview_html: str,
        preview_lines: int,
        full_html: str,
        title: str,
    ) -> str:
        return (
            f'<details class="{details_class}">'
            f'<summary class="{summary_class}">'
            f'<span class="{title_class}">{cls._escape_html(title)}</span>'
            f'<div class="preview-clamp" style="--preview-lines: {preview_lines};">{preview_html}</div>'
            f"</summary>"
            f'<div class="{full_class}" onclick="this.closest(\'details\').open=false;">'
            f'<span class="{title_class}">{cls._escape_html(title)}</span>{full_html}</div>'
            f"</details>"
        )

    @staticmethod
    def _parse_json_like(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in "{[":
            return value
        try:
            return json.loads(stripped)
        except Exception:
            return value

    @classmethod
    def _command_to_text(cls, value: Any) -> str | None:
        if isinstance(value, str):
            command = value.strip()
            return command or None
        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, (str, int, float, bool)):
                    parts.append(shlex.quote(str(part)))
                else:
                    return cls._stringify_value(value)
            command = " ".join(parts).strip()
            return command or None
        return None

    @classmethod
    def _find_command_candidate(cls, value: Any) -> Any | None:
        if isinstance(value, dict):
            for key in ("command", "cmd", "argv", "args"):
                if key in value:
                    return value.get(key)
            for nested in value.values():
                found = cls._find_command_candidate(nested)
                if found is not None:
                    return found
            return None
        if isinstance(value, list):
            for item in value:
                found = cls._find_command_candidate(item)
                if found is not None:
                    return found
        return None

    @classmethod
    def _tool_call_type_name(cls, tool_call: dict[str, Any]) -> str:
        call_type = cls._stringify_value(tool_call.get("type")).strip()
        name = cls._stringify_value(tool_call.get("name")).strip()
        function = tool_call.get("function")
        if isinstance(function, dict):
            function_name = cls._stringify_value(function.get("name")).strip()
            if function_name:
                name = function_name
        if not call_type and (name or isinstance(function, dict)):
            call_type = "tool_call"
        if call_type and name:
            return f"{call_type}/{name}"
        if name:
            return name
        if call_type:
            return call_type
        return "unknown"

    @classmethod
    def _tool_call_arguments_text(cls, tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function")
        candidates: list[Any] = [tool_call.get("arguments"), tool_call.get("input")]
        if isinstance(function, dict):
            candidates.extend([function.get("arguments"), function.get("input")])
        for candidate in candidates:
            if candidate is None:
                continue
            parsed = cls._parse_json_like(candidate)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, indent=2, ensure_ascii=True)
            text = cls._stringify_value(parsed).strip()
            if text:
                return text
        return ""

    @classmethod
    def _extract_tool_call_command(cls, tool_call: dict[str, Any]) -> str | None:
        function = tool_call.get("function")
        candidates: list[Any] = [tool_call.get("arguments"), tool_call.get("input")]
        if isinstance(function, dict):
            candidates.extend([function.get("arguments"), function.get("input")])
        for candidate in candidates:
            parsed = cls._parse_json_like(candidate)
            command = cls._command_to_text(parsed)
            if command:
                return command
            command_candidate = cls._find_command_candidate(parsed)
            command = cls._command_to_text(command_candidate)
            if command:
                return command
        return None

    @classmethod
    def _render_tool_call_card(cls, tool_call: Any) -> str:
        if not isinstance(tool_call, dict):
            return cls._fmt_code_block(cls._stringify_value(tool_call))

        sections: list[str] = ['<div class="section-title">type/name</div>']
        sections.append(cls._fmt_code_block(cls._tool_call_type_name(tool_call)))

        command = cls._extract_tool_call_command(tool_call)
        if command:
            sections.append('<div class="section-title">command</div>')
            sections.append(cls._fmt_code_block(command))
        else:
            arguments = cls._tool_call_arguments_text(tool_call)
            if arguments:
                sections.append('<div class="section-title">arguments</div>')
                sections.append(cls._fmt_code_block(arguments))

        return f'<div class="tool-call-item">{"".join(sections)}</div>'

    @classmethod
    def _bubble_html(cls, role: str, title: str, body_html: str, align: str) -> str:
        role_styles = {
            "system": {"fg": "#111827", "bg": "#FDE68A"},
            "user": {"fg": "#FFFFFF", "bg": "#1D4ED8"},
            "assistant": {"fg": "#FFFFFF", "bg": "#374151"},
            "tool": {"fg": "#FFFFFF", "bg": "#6D28D9"},
            "default": {"fg": "#FFFFFF", "bg": "#111827"},
        }
        sty = role_styles.get(role, role_styles["default"])
        align_class = {
            "left": "bubble-left",
            "right": "bubble-right",
            "center": "bubble-center",
        }.get(align, "bubble-left")
        return f"""
    <div class="row {align_class}">
      <div class="bubble bubble-{role}" style="background:{sty['bg']}; color:{sty['fg']};">
        <div class="role-label" style="background:{sty['fg']}; color:{sty['bg']}; border-color:{sty['fg']};">{cls._escape_html(title)}</div>
        <div class="content">{body_html}</div>
      </div>
    </div>
    """

    @classmethod
    def _standalone_block_html(cls, body_html: str, align: str) -> str:
        align_class = {
            "left": "bubble-left",
            "right": "bubble-right",
            "center": "bubble-center",
        }.get(align, "bubble-left")
        return f"""
    <div class="row {align_class}">
      <div class="standalone-block">{body_html}</div>
    </div>
    """

    @classmethod
    def _render_assistant_content_blocks(cls, content: Any) -> tuple[list[str], str]:
        """Render assistant content into standalone thinking blocks plus body HTML."""
        if content is None:
            return [], ""
        if isinstance(content, str):
            if not content.strip():
                return [], ""
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
            if isinstance(parsed, (dict, list)):
                return cls._render_assistant_content_blocks(parsed)
            return [], f'<div class="markdown-body">{cls._render_markdown_html(content)}</div>'
        if isinstance(content, list):
            thinking_blocks: list[str] = []
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(cls._escape_html(str(item)))
                    continue
                item_type = item.get("type")
                if item_type == "thinking":
                    thinking_text = str(item.get("thinking", ""))
                    if thinking_text.strip():
                        thinking_html = (
                            '<div class="thinking-content markdown-body">'
                            f"{cls._render_markdown_html(thinking_text)}"
                            "</div>"
                        )
                        thinking_blocks.append(
                            cls._render_preview_details(
                                "tool-result-details",
                                "tool-result-summary",
                                "tool-result-full",
                                "tool-result-title",
                                thinking_html,
                                5,
                                thinking_html,
                                "Thinking",
                            )
                        )
                elif item_type in {"text", "input_text", "output_text"}:
                    text = str(item.get("text", ""))
                    if text.strip():
                        parts.append(
                            f'<div class="markdown-body">{cls._render_markdown_html(text)}</div>'
                        )
                elif item_type in {"tool_use", "tool_result"}:
                    continue
                else:
                    parts.append(
                        f'<pre class="code-block">{cls._escape_html(json.dumps(item, ensure_ascii=True))}</pre>'
                    )
            return thinking_blocks, "\n".join(part for part in parts if part)
        return [], cls._escape_html(str(content))

    @classmethod
    def _render_tool_component(cls, component: ToolComponent) -> str:
        sections: list[str] = []
        if component.tool_name:
            sections.append('<div class="section-title">tool_name</div>')
            sections.append(cls._fmt_code_block(component.tool_name))
        if component.tool_call:
            sections.append('<div class="section-title">tool_call</div>')
            parsed_tool_call = cls._parse_json_like(component.tool_call)
            if isinstance(parsed_tool_call, dict):
                sections.append(cls._render_tool_call_card(parsed_tool_call))
            else:
                sections.append(cls._fmt_code_block(component.tool_call))
        if component.tool_output:
            sections.append(cls._render_tool_output_section(component.tool_output))
        if not sections:
            sections.append("<span class='muted'>(no tool data)</span>")
        return "\n".join(sections)

    @classmethod
    def _render_tool_output_section(cls, tool_output: str) -> str:
        return (
            '<details class="tool-output-details">'
            '<summary class="tool-output-summary">'
            '<span class="disclosure-arrow" aria-hidden="true"></span>'
            '<span class="section-title section-title-inline">tool_output</span>'
            "</summary>"
            f"{cls._fmt_code_block(tool_output)}"
            "</details>"
        )

    @classmethod
    def _render_submission_gate_attempt(cls, attempt: dict[str, Any], attempt_index: int) -> str:
        status = "passed" if attempt.get("passed") else "failed"
        returncode = attempt.get("returncode")
        command = cls._stringify_value(attempt.get("command")).strip()
        output = cls._stringify_value(attempt.get("output"))

        sections: list[str] = []
        sections.append('<div class="section-title">attempt</div>')
        sections.append(cls._fmt_code_block(f"{attempt_index}: {status} (returncode={returncode})"))

        if command:
            sections.append('<div class="section-title">command</div>')
            sections.append(cls._fmt_code_block(command))

        if output.strip():
            sections.append('<div class="section-title">compiler/runtime output</div>')
            sections.append(cls._fmt_code_block(output))

        return f'<div class="gate-attempt">{"".join(sections)}</div>'

    @classmethod
    def _render_submission_gate_section(cls, info: dict[str, Any]) -> str:
        if not isinstance(info, dict):
            return ""
        gate = info.get("submission_gate")
        if not isinstance(gate, dict):
            return ""
        attempts = gate.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return ""

        header_lines = [
            f"enabled={bool(gate.get('enabled', False))}",
            f"max_retries={gate.get('max_retries')}",
            f"recorded_attempts={len(attempts)}",
        ]
        parts: list[str] = []
        parts.append('<div class="section-title">submission gate summary</div>')
        parts.append(cls._fmt_code_block("\n".join(header_lines)))

        for idx, attempt in enumerate(attempts, start=1):
            if isinstance(attempt, dict):
                parts.append(cls._render_submission_gate_attempt(attempt, idx))
            else:
                parts.append('<div class="section-title">attempt</div>')
                parts.append(cls._fmt_code_block(cls._stringify_value(attempt)))

        body_html = "".join(parts)
        return cls._bubble_html("tool", "SUBMISSION GATE", body_html, "center")

    @classmethod
    def _render_steps_html(
        cls,
        steps: list[SystemComponent | UserComponent | AssistantComponent | ToolComponent],
        info: dict[str, Any] | None = None,
        has_attempt_events: bool = False,
    ) -> str:
        html_fragments: list[str] = []
        tool_output_count = 0
        if info and not has_attempt_events:
            gate_html = cls._render_submission_gate_section(info)
            if gate_html:
                html_fragments.append(gate_html)
        for step in steps:
            if isinstance(step, SystemComponent):
                body_html = (
                    cls._escape_html(step.message)
                    if step.message
                    else "<span class='muted'>(no text)</span>"
                )
                html_fragments.append(
                    cls._bubble_html("system", "SYSTEM", body_html, "center")
                )
            elif isinstance(step, UserComponent):
                body_html = (
                    cls._escape_html(step.message)
                    if step.message
                    else "<span class='muted'>(no text)</span>"
                )
                html_fragments.append(cls._bubble_html("user", "USER", body_html, "right"))
            elif isinstance(step, AssistantComponent):
                thinking_blocks, content_html = cls._render_assistant_content_blocks(
                    step.raw_content if step.raw_content is not None else step.message
                )
                for thinking_html in thinking_blocks:
                    html_fragments.append(cls._standalone_block_html(thinking_html, "left"))

                parts: list[str] = []
                if content_html:
                    parts.append(content_html)
                elif step.message:
                    parts.append(cls._escape_html(step.message))
                else:
                    parts.append("<span class='muted'>(no text)</span>")
                if step.tool_calls:
                    parts.append('<div class="section-title">tool_calls</div>')
                    parts.append(
                        "".join(cls._render_tool_call_card(call) for call in step.tool_calls)
                    )
                body_html = "\n".join(parts)
                html_fragments.append(
                    cls._bubble_html("assistant", "ASSISTANT", body_html, "left")
                )
            elif isinstance(step, ToolComponent):
                if step.tool_output:
                    tool_output_count += 1
                body_html = cls._render_tool_component(step)
                html_fragments.append(cls._bubble_html("tool", "TOOL", body_html, "center"))
            else:
                html_fragments.append(
                    cls._bubble_html(
                        "default",
                        "UNKNOWN",
                        cls._fmt_code_block(cls._stringify_value(step)),
                        "center",
                    )
                )

        css = """
    <style>
      .trace-shell {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        line-height: 1.5;
        font-size: 15px;
        color: #111827;
      }
      .trace-container {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 8px 8px 12px 8px;
        background: #ffffff;
      }
      .row { display: flex; margin: 10px 0; }
      .bubble {
        max-width: 75%;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        word-break: break-word;
      }
      .bubble-left   { justify-content: flex-start; }
      .bubble-right  { justify-content: flex-end; }
      .bubble-center { justify-content: center; }
      .standalone-block {
        max-width: 75%;
      }
      .standalone-block .tool-result-details,
      .standalone-block .tool-result-full {
        color: #FFFFFF;
        background: #4B5563;
        border-color: rgba(255,255,255,0.18);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12);
        word-break: break-word;
        position: relative;
      }
      .standalone-block .tool-result-details > summary,
      .standalone-block .tool-result-full {
        background: #4B5563;
      }
      .role-label {
        display: inline-block;
        font-weight: 700;
        font-size: 12px;
        padding: 2px 8px;
        border: 1px solid;
        border-radius: 999px;
        margin-bottom: 8px;
      }
      .content { white-space: pre-wrap; }
      .markdown-body { white-space: normal; }
      .markdown-body > :first-child { margin-top: 0; }
      .markdown-body > :last-child { margin-bottom: 0; }
      .markdown-body p,
      .markdown-body ul,
      .markdown-body ol,
      .markdown-body blockquote,
      .markdown-body pre,
      .markdown-body table,
      .markdown-body h1,
      .markdown-body h2,
      .markdown-body h3,
      .markdown-body h4,
      .markdown-body h5,
      .markdown-body h6 {
        margin: 4px 0 6px 0;
      }
      .markdown-body ul,
      .markdown-body ol {
        padding-left: 1.5em;
      }
      .markdown-body li + li {
        margin-top: 2px;
      }
      .markdown-body code {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: rgba(255,255,255,0.14);
        padding: 0 3px;
        border-radius: 3px;
      }
      .markdown-body pre {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 5px;
        padding: 5px 6px;
        overflow-x: auto;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }
      .markdown-body pre code {
        background: transparent;
        padding: 0;
        border-radius: 0;
      }
      .markdown-body blockquote {
        border-left: 3px solid rgba(255,255,255,0.4);
        padding-left: 8px;
        opacity: 0.95;
      }
      .markdown-body a {
        color: #ffffff;
        text-decoration: underline;
      }
      .code-block {
        margin: 8px 0 12px 0;
        padding: 10px 12px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 10px;
        overflow-x: auto;
        white-space: pre;
        color: #FFFFFF;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 14px;
      }
      .section-title {
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 6px;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 0.02em;
        opacity: 0.95;
      }
      .tool-call-item {
        margin: 8px 0 12px 0;
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.25);
      }
      .tool-call-item .section-title:first-child {
        margin-top: 0;
      }
      .trace-controls {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 12px;
      }
      .trace-control-button {
        appearance: none;
        border: 1px solid #d1d5db;
        border-radius: 999px;
        background: #f9fafb;
        color: #111827;
        font: inherit;
        font-size: 13px;
        font-weight: 700;
        padding: 8px 12px;
        cursor: pointer;
        transition: background 0.15s ease, border-color 0.15s ease;
      }
      .trace-control-button:hover {
        background: #f3f4f6;
        border-color: #9ca3af;
      }
      .tool-output-details {
        margin: 8px 0 12px 0;
      }
      .tool-output-summary {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
        list-style: none;
        user-select: none;
      }
      .tool-output-summary::-webkit-details-marker {
        display: none;
      }
      .tool-output-summary + .code-block {
        margin-top: 4px;
      }
      .preview-clamp {
        position: relative;
      }
      .preview-clamp > .code-block,
      .preview-clamp > .thinking-content {
        margin: 0;
        position: relative;
        overflow: hidden;
        max-height: calc(var(--preview-lines, 5) * 1.45em + 22px);
        border-radius: 0;
      }
      .preview-clamp.has-overflow > .code-block::after,
      .preview-clamp.has-overflow > .thinking-content::after {
        content: "↓";
        position: absolute;
        left: 50%;
        bottom: 4px;
        transform: translateX(-50%);
        color: #ffffff;
        font-size: 23px;
        font-weight: 900;
        line-height: 1;
        text-shadow: 0 1px 3px rgba(0,0,0,0.6);
        pointer-events: none;
        z-index: 1;
      }
      .tool-result-details:not([open]) .tool-result-summary .thinking-content::after {
        content: "↓";
        position: absolute;
        left: 50%;
        bottom: 4px;
        transform: translateX(-50%);
        color: #ffffff;
        font-size: 23px;
        font-weight: 900;
        line-height: 1;
        text-shadow: 0 1px 3px rgba(0,0,0,0.6);
        pointer-events: none;
        z-index: 1;
      }
      .thinking-content {
        margin: 0;
        padding: 4px 6px;
        background: rgba(0,0,0,0.18);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        font-size: 14px;
        color: rgba(255,255,255,0.75);
        line-height: 1.45;
      }
      .section-title-inline {
        margin: 0;
      }
      .disclosure-arrow {
        width: 0;
        height: 0;
        border-top: 5px solid transparent;
        border-bottom: 5px solid transparent;
        border-left: 7px solid currentColor;
        transition: transform 0.15s ease;
      }
      .tool-output-details[open] .disclosure-arrow {
        transform: rotate(90deg);
      }
      .tool-result-details, .tool-result-full {
        margin: 0 0 2px 0;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 4px;
        overflow: visible;
        position: relative;
      }
      .tool-result-title {
        position: absolute;
        top: -10px;
        right: 12px;
        z-index: 2;
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        padding: 0.5px 3px;
        border: 1px solid rgba(255,255,255,0.65);
        border-radius: 499.5px;
        background: #ffffff;
        color: #6D28D9;
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }
      .tool-result-details > summary {
        position: relative;
        padding: 5px;
        cursor: pointer;
        background: rgba(255,255,255,0.10);
        list-style: none;
        user-select: none;
      }
      .tool-result-details > summary::-webkit-details-marker {
        display: none;
      }
      .tool-result-summary .thinking-content {
        margin: 0;
        padding: 4px 5px;
        border: none;
        border-radius: 0;
        background: transparent;
      }
      .standalone-block .tool-result-details[open] {
        border-color: rgba(255,255,255,0.18);
        background: #4B5563;
      }
      .tool-result-details[open] {
        border-color: transparent;
        background: transparent;
      }
      .tool-result-details[open] > summary {
        display: none;
      }
      .tool-result-details[open] .tool-result-full {
        cursor: pointer;
      }
      .tool-result-full .thinking-content {
        margin: 0;
        border-radius: 0 0 4px 4px;
        border: none;
      }
      .tool-result-details[open] .tool-result-full .thinking-content {
        border-top: 1px solid rgba(255,255,255,0.15);
        border-radius: 4px;
      }
      .gate-attempt {
        margin: 8px 0 12px 0;
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.30);
      }
      .gate-attempt .section-title:first-child {
        margin-top: 0;
      }
      .muted { color: #F3F4F6; font-style: italic; }
    </style>
    """
        body = "".join(html_fragments)
        controls = ""
        script = ""
        if tool_output_count:
            controls = (
                '<div class="trace-controls">'
                '<button type="button" class="trace-control-button" data-expand-all-tool-outputs>'
                "Expand all tool outputs"
                "</button>"
                "</div>"
            )
            script = """
    <script>
      (() => {
        const details = Array.from(document.querySelectorAll('.tool-output-details'));
        const button = document.querySelector('[data-expand-all-tool-outputs]');
        if (!details.length || !button) {
          return;
        }

        const syncButtonLabel = () => {
          const allOpen = details.every((item) => item.open);
          button.textContent = allOpen
            ? 'Collapse all tool outputs'
            : 'Expand all tool outputs';
        };

        button.addEventListener('click', () => {
          const shouldOpen = details.some((item) => !item.open);
          for (const item of details) {
            item.open = shouldOpen;
          }
          syncButtonLabel();
        });

        for (const item of details) {
          item.addEventListener('toggle', syncButtonLabel);
        }

        syncButtonLabel();
      })();
      (() => {
        const updatePreviewOverflow = () => {
          for (const clamp of document.querySelectorAll('.preview-clamp')) {
            const block = clamp.firstElementChild;
            if (!block) {
              continue;
            }
            clamp.classList.toggle('has-overflow', block.scrollHeight > block.clientHeight + 1);
          }
        };

        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', updatePreviewOverflow, { once: true });
        } else {
          updatePreviewOverflow();
        }

        window.addEventListener('resize', updatePreviewOverflow);
      })();
    </script>
    """
        return (
            f'{css}<div class="trace-shell">{controls}<div class="trace-container">{body}</div></div>{script}'
        )

    @classmethod
    def _collect_tool_results(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for message in messages:
            if message.get("type") == "function_call_output":
                results.append(message)
            elif message.get("role") == "tool":
                results.append(message)
            for item in cls._iter_content_items(message):
                if item.get("type") == "tool_result":
                    results.append(item)
        return results

    @classmethod
    def _inject_tool_role(
        cls,
        tool_results: list[dict[str, Any]],
        by_role: dict[str, list[dict[str, Any]]],
    ) -> None:
        if not tool_results:
            return
        bucket = by_role.setdefault("tool", [])
        for result in tool_results:
            if result.get("role") == "tool":
                continue
            wrapped = dict(result)
            wrapped.setdefault("role", "tool")
            bucket.append(wrapped)

    @classmethod
    def _build_user_components(
        cls,
        messages: list[dict[str, Any]],
        usage_by_message_index: dict[int, UsageInfo] | None = None,
    ) -> list[UserComponent]:
        last_user_index = cls._get_last_user_index(messages)
        components: list[UserComponent] = []
        for index, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            message_text = cls._content_to_text(message.get("content")).strip()
            if index == last_user_index:
                message_text = cls._clean_diff_text(message_text).strip()
            if not message_text:
                continue
            usage_info = (
                usage_by_message_index.get(index) if usage_by_message_index else None
            )
            components.append(
                UserComponent(
                    index=index,
                    message=message_text,
                    prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                    completion_tokens=usage_info.completion_tokens if usage_info else None,
                    total_cost=usage_info.total_cost if usage_info else None,
                )
            )
        return components

    @classmethod
    def _build_system_components(
        cls,
        messages: list[dict[str, Any]],
        usage_by_message_index: dict[int, UsageInfo] | None = None,
    ) -> list[SystemComponent]:
        components: list[SystemComponent] = []
        for index, message in enumerate(messages):
            if message.get("role") != "system":
                continue
            message_text = cls._content_to_text(message.get("content")).strip()
            if not message_text:
                continue
            usage_info = (
                usage_by_message_index.get(index) if usage_by_message_index else None
            )
            components.append(
                SystemComponent(
                    index=index,
                    message=message_text,
                    prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                    completion_tokens=usage_info.completion_tokens if usage_info else None,
                    total_cost=usage_info.total_cost if usage_info else None,
                )
            )
        return components

    @classmethod
    def _build_assistant_components(
        cls,
        messages: list[dict[str, Any]],
        usage_by_message_index: dict[int, UsageInfo] | None = None,
    ) -> list[AssistantComponent]:
        components: list[AssistantComponent] = []
        for index, message in enumerate(messages):
            if message.get("type") == "reasoning":
                message_text = cls._content_to_text(message.get("content")).strip()
                if not message_text:
                    message_text = "reasoning"
                tool_calls = cls._codex_tool_calls_after(messages, index)
            else:
                if message.get("role") != "assistant":
                    continue
                message_text = cls._content_to_text(message.get("content")).strip()
                tool_calls = cls._extract_tool_calls(message)
            if not message_text and not tool_calls:
                continue
            usage_info = (
                usage_by_message_index.get(index) if usage_by_message_index else None
            )
            components.append(
                AssistantComponent(
                    index=index,
                    message=message_text,
                    tool_calls=tool_calls,
                    raw_content=message.get("content"),
                    prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                    completion_tokens=usage_info.completion_tokens if usage_info else None,
                    total_cost=usage_info.total_cost if usage_info else None,
                )
            )
        return components

    @staticmethod
    def _build_tool_components(
        tool_uses: list[dict[str, Any]],
        usage_by_tool_call_id: dict[str, UsageInfo] | None = None,
    ) -> list[ToolComponent]:
        components: list[ToolComponent] = []
        for index, tool_use in enumerate(tool_uses):
            tool_id = tool_use.get("id")
            if not tool_id:
                continue
            components.append(
                Trace._tool_component_from_use(
                    index, tool_use, usage_by_tool_call_id
                )
            )
        return components

    @classmethod
    def _build_steps(
        cls,
        messages: list[dict[str, Any]],
        tool_uses: list[dict[str, Any]],
        usage_by_message_index: dict[int, UsageInfo] | None = None,
        usage_by_tool_call_id: dict[str, UsageInfo] | None = None,
    ) -> list[SystemComponent | UserComponent | AssistantComponent | ToolComponent]:
        steps: list[SystemComponent | UserComponent | AssistantComponent | ToolComponent] = []
        last_user_index = cls._get_last_user_index(messages)
        tool_uses_by_id = {
            tool_use.get("id"): tool_use for tool_use in tool_uses if tool_use.get("id")
        }
        added_tool_ids: set[str] = set()

        def add_tool_component(
            tool_id: str | None, tool_result: dict[str, Any] | None
        ) -> None:
            if tool_id and tool_id in added_tool_ids:
                return
            tool_use = tool_uses_by_id.get(tool_id)
            if tool_use:
                steps.append(
                    cls._tool_component_from_use(
                        len(steps), tool_use, usage_by_tool_call_id
                    )
                )
            else:
                tool_output = cls._stringify_tool_output([tool_result] if tool_result else [])
                usage_info = (
                    usage_by_tool_call_id.get(tool_id)
                    if usage_by_tool_call_id and tool_id
                    else None
                )
                steps.append(
                    ToolComponent(
                        index=len(steps),
                        tool_name="",
                        tool_call="",
                        tool_output=tool_output,
                        prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                        completion_tokens=usage_info.completion_tokens if usage_info else None,
                        total_cost=usage_info.total_cost if usage_info else None,
                    )
                )
            if tool_id:
                added_tool_ids.add(tool_id)

        for index, message in enumerate(messages):
            msg_type = message.get("type")
            if msg_type == "reasoning":
                message_text = cls._content_to_text(message.get("content")).strip()
                if not message_text:
                    message_text = "reasoning"
                tool_calls = cls._codex_tool_calls_after(messages, index)
                if not message_text and not tool_calls:
                    continue
                usage_info = (
                    usage_by_message_index.get(index) if usage_by_message_index else None
                )
                steps.append(
                    AssistantComponent(
                        index=len(steps),
                        message=message_text,
                        tool_calls=tool_calls,
                        raw_content=message.get("content"),
                        prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                        completion_tokens=usage_info.completion_tokens if usage_info else None,
                        total_cost=usage_info.total_cost if usage_info else None,
                    )
                )
                continue

            if msg_type == "function_call_output":
                tool_id = (
                    message.get("call_id")
                    or message.get("tool_use_id")
                    or message.get("tool_call_id")
                )
                add_tool_component(tool_id, message)
                continue

            role = message.get("role")
            if role == "system":
                message_text = cls._content_to_text(message.get("content")).strip()
                if not message_text:
                    continue
                usage_info = (
                    usage_by_message_index.get(index) if usage_by_message_index else None
                )
                steps.append(
                    SystemComponent(
                        index=len(steps),
                        message=message_text,
                        prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                        completion_tokens=usage_info.completion_tokens if usage_info else None,
                        total_cost=usage_info.total_cost if usage_info else None,
                    )
                )
                continue
            if role == "assistant":
                message_text = cls._content_to_text(message.get("content")).strip()
                tool_calls = cls._extract_tool_calls(message)
                if not message_text and not tool_calls:
                    continue
                usage_info = (
                    usage_by_message_index.get(index) if usage_by_message_index else None
                )
                steps.append(
                    AssistantComponent(
                        index=len(steps),
                        message=message_text,
                        tool_calls=tool_calls,
                        raw_content=message.get("content"),
                        prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                        completion_tokens=usage_info.completion_tokens if usage_info else None,
                        total_cost=usage_info.total_cost if usage_info else None,
                    )
                )
                continue

            if role == "user":
                usage_info = (
                    usage_by_message_index.get(index) if usage_by_message_index else None
                )
                content = message.get("content")
                if isinstance(content, list):
                    buffer: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            if buffer:
                                steps.append(
                                    UserComponent(
                                        index=len(steps),
                                        message="\n".join(buffer).strip(),
                                        prompt_tokens=usage_info.prompt_tokens
                                        if usage_info
                                        else None,
                                        completion_tokens=usage_info.completion_tokens
                                        if usage_info
                                        else None,
                                        total_cost=usage_info.total_cost if usage_info else None,
                                    )
                                )
                                buffer = []
                            tool_id = (
                                item.get("tool_use_id")
                                or item.get("tool_call_id")
                                or item.get("id")
                            )
                            add_tool_component(tool_id, item)
                        else:
                            text = cls._content_to_text(item).strip()
                            if text:
                                buffer.append(text)
                    if buffer:
                        message_text = "\n".join(buffer).strip()
                        if index == last_user_index:
                            message_text = cls._clean_diff_text(message_text).strip()
                        steps.append(
                            UserComponent(
                                index=len(steps),
                                message=message_text,
                                prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                                completion_tokens=usage_info.completion_tokens
                                if usage_info
                                else None,
                                total_cost=usage_info.total_cost if usage_info else None,
                            )
                        )
                else:
                    message_text = cls._content_to_text(content).strip()
                    if index == last_user_index:
                        message_text = cls._clean_diff_text(message_text).strip()
                    if message_text:
                        steps.append(
                            UserComponent(
                                index=len(steps),
                                message=message_text,
                                prompt_tokens=usage_info.prompt_tokens if usage_info else None,
                                completion_tokens=usage_info.completion_tokens
                                if usage_info
                                else None,
                                total_cost=usage_info.total_cost if usage_info else None,
                            )
                        )
                continue

            if role == "tool":
                tool_id = message.get("tool_call_id") or message.get("tool_use_id")
                add_tool_component(tool_id, message)
                continue

        return steps

    def get_token_usage(self) -> dict[str, TokenUsage]:

        # A trace can have multiple sub-agents which have their own 
        usage_by_model = {}

        
        for response in self.responses:

            model = response.get("model") or "unknown_model"
            model = model.split("/")[-1]  # Keep only the model name part
            if model not in usage_by_model:
                usage_by_model[model] = TokenUsage()
            usage = usage_by_model[model]

            token_info = response.get("usage", {})
            usage.add(token_info)

        return usage_by_model

    def get_cost(self) -> float:
        total_cost = 0.0
        usage_by_model = self.get_token_usage()
        for model, usage in usage_by_model.items():
            prices = get_model_prices(model)
            if prices is None:
                continue
            input_price, output_price, cache_price, cache_creation_price = prices
            model_cost = usage.price(
                input_token_price=input_price,
                output_token_price=output_price,
                cache_token_price=cache_price,
                cache_creation_token_price=cache_creation_price,
            )
            total_cost += model_cost
        return total_cost

    def get_tool_usage(self, advanced: bool = False) -> dict[str, int]:
        tool_usage: dict[str, int] = {}
        for tool_component in self.tool_components:
            tool_name = (
                tool_component.get_advanced_name() if advanced else tool_component.tool_name
            )
            if not tool_name:
                tool_name = "unknown_tool"
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage
        
    def get_all_tool_names(self) -> set[str]:
        tool_usage = self.get_tool_usage()
        return set(tool_usage.keys())

    def get_error_tool_components(self) -> list[ToolComponent]:
        error_components = []
        for component in self.tool_components:
            if component.is_error():
                error_components.append(component)
        return error_components
