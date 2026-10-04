"""Keep provider usage evidence before LiteLLM's streaming normalization."""

from __future__ import annotations


def _litellm_usage_payload(raw: dict) -> dict:
    """Return provider usage in the scalar shape LiteLLM can finalize.

    Bifrost's OpenAI-compatible stream reports ``usage.cost`` as
    ``{"total_cost": ...}``.  LiteLLM accepts that value while parsing the
    chunk, but later calls ``float(usage.cost)`` while closing the stream.  Keep
    the exact provider payload in the forensic metadata and give LiteLLM the
    equivalent scalar value it expects.
    """

    normalized = dict(raw)
    cost = normalized.get("cost")
    if isinstance(cost, dict):
        total_cost = cost.get("total_cost")
        if isinstance(total_cost, (int, float)) and not isinstance(
            total_cost, bool
        ):
            normalized["cost"] = total_cost
        else:
            normalized.pop("cost", None)
    return normalized


def install_stream_usage_capture() -> None:
    from litellm import Usage
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    original = CustomStreamWrapper.handle_openai_chat_completion_chunk

    def handle(self, chunk):
        parsed = original(self, chunk)
        stream = getattr(self, "completion_stream", None)
        raw_id = getattr(stream, "_leanlean_raw_stream_id", None)
        if raw_id:
            self.logging_obj.model_call_details["leanlean_raw_stream_id"] = raw_id
        usage = parsed.get("usage")
        if usage is not None:
            raw = usage if isinstance(usage, dict) else usage.model_dump()
            # The pinned LiteLLM dictionary branch otherwise retains only the
            # three totals, dropping nested cache and reasoning details.
            parsed["usage"] = Usage(**_litellm_usage_payload(raw))
            self.logging_obj.model_call_details["leanlean_provider_usage"] = dict(raw)
        return parsed

    handle._leanlean_usage_capture = True
    if not getattr(original, "_leanlean_usage_capture", False):
        CustomStreamWrapper.handle_openai_chat_completion_chunk = handle

    original_creator = CustomStreamWrapper.chunk_creator
    if getattr(original_creator, "_leanlean_provider_usage", False):
        return

    def create_chunk(self, chunk):
        result = original_creator(self, chunk)
        if result is None or getattr(result, "usage", None) is None:
            return result

        details = getattr(
            getattr(self, "logging_obj", None), "model_call_details", {}
        )
        provider_usage = details.get("leanlean_provider_usage")
        if provider_usage is not None:
            # chunk_creator may overwrite our parsed Usage with an SDK
            # CompletionUsage. Restore the provider payload captured before
            # that overwrite so cache and reasoning counters remain intact.
            raw = (
                provider_usage
                if isinstance(provider_usage, dict)
                else provider_usage.model_dump()
            )
            result.usage = Usage(**_litellm_usage_payload(raw))
        elif self.model.rsplit("/", 1)[-1] == "glm-5.3-flash":
            # chunk_creator overwrites our parsed Usage with the SDK's
            # CompletionUsage when usage accompanies a finish/content chunk.
            # LiteLLM's aggregator expects dict-style membership/get methods;
            # SDK BaseModels fail those checks, triggering local token counts
            # and losing cache details. Convert AFTER that overwrite, before
            # the chunk is stored, streamed, or passed to accounting callbacks.
            usage = result.usage
            raw = usage if isinstance(usage, dict) else usage.model_dump()
            result.usage = Usage(**_litellm_usage_payload(raw))
        return result

    create_chunk._leanlean_provider_usage = True
    CustomStreamWrapper.chunk_creator = create_chunk
