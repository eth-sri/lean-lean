"""Replay Mistral thinking the way Mistral Vibe's native backend does.

Vibe's native ``mistral`` backend (vibe/core/llm/backend/mistral.py, 2.25.0)
sends ``reasoning_effort`` and, for every earlier assistant turn that carried
``reasoning_content``, replays it as a ``thinking`` chunk ahead of the text:

    {"role": "assistant",
     "content": [{"type": "thinking", "thinking": [{"type": "text", "text": R}]},
                 {"type": "text", "text": C}],          # text chunk only if C
     "tool_calls": [...]}

It does this only when the model's thinking level is not "off", which is also
exactly when it sends ``reasoning_effort``.

Behind our gateway Vibe runs its generic OpenAI client, which sends the earlier
thinking as ``reasoning_content``. LiteLLM 1.96.2's MistralConfig strips that
field and flattens list content to text, so the model never sees its own earlier
thinking. (``reasoning_effort`` itself is dropped for every non-"magistral"
model unless the deployment sets ``allowed_openai_params``.) This patch restores
the native wire format for requests that carry ``reasoning_effort``; requests
without it, such as Vibe's compaction model with thinking off, are untouched.
"""

from __future__ import annotations

import inspect
from typing import Any

_TAG = "_leanlean_mistral_thinking"
_EMPTY = "\x00leanlean-empty-assistant\x00"
_installed = False


def _wants_thinking(optional_params: dict) -> bool:
    effort = optional_params.get("reasoning_effort")
    return effort not in (None, "", "off")


def tag_reasoning(messages: list[dict]) -> list[dict]:
    """Move each assistant ``reasoning_content`` into a private tag LiteLLM keeps."""

    tagged = []
    for message in messages:
        reasoning = message.get("reasoning_content") if message.get("role") == "assistant" else None
        if not reasoning:
            tagged.append(message)
            continue
        copy = {k: v for k, v in message.items() if k not in ("reasoning_content", "thinking_blocks")}
        copy[_TAG] = reasoning
        if not copy.get("content") and not copy.get("tool_calls"):
            # LiteLLM drops assistant messages it considers empty; native Vibe
            # still sends the thinking chunk.
            copy["content"] = _EMPTY
        tagged.append(copy)
    return tagged


def restore_thinking(data: dict) -> dict:
    """Rebuild tagged assistant messages as native thinking + text chunks."""

    for message in data.get("messages") or []:
        if _TAG not in message:
            continue
        reasoning = message.pop(_TAG)
        content = message.get("content")
        if content == _EMPTY:
            content = None
        chunks: list[dict[str, Any]] = [
            {"type": "thinking", "thinking": [{"type": "text", "text": reasoning}]}
        ]
        if content:
            chunks.append({"type": "text", "text": content})
        message["content"] = chunks
    return data


def install_mistral_reasoning_replay() -> None:
    """Wrap MistralConfig's sync and async request transforms (idempotent)."""

    global _installed
    if _installed:
        return
    from litellm.llms.mistral.chat.transformation import MistralConfig

    sync_original = MistralConfig.transform_request
    async_original = MistralConfig.async_transform_request

    def transform_request(self, model, messages, optional_params, litellm_params, headers):
        if not _wants_thinking(optional_params):
            return sync_original(self, model, messages, optional_params, litellm_params, headers)
        data = sync_original(self, model, tag_reasoning(messages), optional_params, litellm_params, headers)
        return restore_thinking(data)

    async def async_transform_request(self, model, messages, optional_params, litellm_params, headers):
        if not _wants_thinking(optional_params):
            return await async_original(self, model, messages, optional_params, litellm_params, headers)
        data = async_original(self, model, tag_reasoning(messages), optional_params, litellm_params, headers)
        if inspect.isawaitable(data):
            data = await data
        return restore_thinking(data)

    MistralConfig.transform_request = transform_request
    MistralConfig.async_transform_request = async_transform_request
    _installed = True
