"""Forward Codex's Responses requests to the ChatGPT backend as Codex sent them.

LiteLLM's ``chatgpt`` provider rebuilds every request for clients that are not
Codex: it keeps only an allowlist of body fields (dropping ``prompt_cache_key``,
``text``, ``parallel_tool_calls``, ...), prepends its own copy of the generic
Codex instructions, and sends a fresh ``session_id`` header per call. Behind
the real Codex CLI that changes what the model sees and breaks prompt-cache
affinity, so most of each session's history is billed as uncached input.

With this installed the proxy sends the client's body unchanged and uses the
client's ``prompt_cache_key`` (Codex's session id) as the ``session_id`` header.
"""

from __future__ import annotations

import os

from leanlean.subscription_auth import EXPLICIT_AUTH_ENV


def install_native_chatgpt_passthrough() -> None:
    if os.environ.get(EXPLICIT_AUTH_ENV) != "1":
        return
    from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    def transform_responses_api_request(
        self, model, input, response_api_optional_request_params, litellm_params, headers
    ):
        request = OpenAIResponsesAPIConfig.transform_responses_api_request(
            self,
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        # LiteLLM keeps Codex's client_metadata in litellm_params, not the body.
        client_metadata = dict(litellm_params).get("client_metadata")
        if isinstance(client_metadata, dict) and "client_metadata" not in request:
            request["client_metadata"] = client_metadata
        cache_key = request.get("prompt_cache_key")
        if cache_key:
            headers["session_id"] = str(cache_key)
        return request

    ChatGPTResponsesAPIConfig.transform_responses_api_request = transform_responses_api_request
