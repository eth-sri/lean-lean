"""Provider-independent token cost accounting."""

from typing import Any
import math

from configs.model_constants import get_model_prices
from leanlean.utils.trace import TokenUsage


def _usage_by_model_from_responses(responses: list[Any]) -> dict[str, TokenUsage]:
    usage_by_model: dict[str, TokenUsage] = {}
    for response in responses:
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if not isinstance(usage, dict):
            continue
        model_name = str(response.get("model") or "unknown_model").split("/")[-1]
        usage_by_model.setdefault(model_name, TokenUsage()).add(usage)
    return usage_by_model



def cache_write_cost_adjustment(usage: dict, prices: tuple) -> float:
    """Correct the configured cache-write rate when native TTL counts exist."""
    partition = (usage.get("cache_creation") or
                 (usage.get("prompt_tokens_details") or {}).get("cache_creation_token_details"))
    if not isinstance(partition, dict):
        return 0.0
    short = partition.get("ephemeral_5m_input_tokens")
    long = partition.get("ephemeral_1h_input_tokens")
    if type(short) is not int or type(long) is not int or min(short, long) < 0:
        return 0.0
    details = usage.get("prompt_tokens_details") or {}
    total = usage.get("cache_creation_input_tokens", details.get("cache_creation_tokens", 0))
    if short + long != total:
        return 0.0
    input_rate, _, _, configured_write_rate = prices
    return (short * input_rate * 1.25 + long * input_rate * 2
            - total * configured_write_rate) / 1_000_000


def cost_from_responses(responses: list[Any]) -> float:
    """Compute total USD cost from normalized response dictionaries."""

    total_cost = 0.0
    token_priced_responses = []
    for response in responses:
        evidence = response.get("native_cost") if isinstance(response, dict) else None
        amount = evidence.get("usd") if isinstance(evidence, dict) else None
        if (isinstance(evidence, dict)
                and evidence.get("basis") == "claude_code_list_price"
                and isinstance(amount, (int, float)) and not isinstance(amount, bool)
                and math.isfinite(amount) and amount >= 0):
            # Claude's terminal per-model list cost retains mixed 5m/1h cache
            # pricing that the aggregate cacheCreationInputTokens loses.
            total_cost += amount
        else:
            token_priced_responses.append(response)
            if isinstance(response, dict):
                prices = get_model_prices(str(response.get("model") or ""))
                usage = response.get("usage")
                if prices and isinstance(usage, dict):
                    total_cost += cache_write_cost_adjustment(usage, prices)
    for model_name, usage in _usage_by_model_from_responses(token_priced_responses).items():
        prices = get_model_prices(model_name)
        if prices is None:
            continue
        input_price, output_price, cache_price, cache_creation_price = prices
        total_cost += usage.price(
            input_token_price=input_price,
            output_token_price=output_price,
            cache_token_price=cache_price,
            cache_creation_token_price=cache_creation_price,
        )
    return total_cost


def cost_accounting_from_responses(responses: list[Any]) -> dict[str, Any]:
    """Qualify benchmark estimates; absent cache evidence is not a cache miss."""
    missing_cache = inconsistent_reasoning = missing_usage = 0
    models = set()
    for response in responses:
        if not isinstance(response, dict):
            missing_usage += 1
            continue
        models.add(str(response.get("model") or "unknown_model").split("/")[-1])
        usage = response.get("usage")
        if not isinstance(usage, dict) or not usage:
            missing_usage += 1
            continue
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        missing_cache += details.get("cached_tokens") is None
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        inconsistent_reasoning += reasoning > completion
    rates = {model: get_model_prices(model) for model in sorted(models)}
    return {
        "basis": "benchmark_rate_estimate_not_invoice",
        "estimated_usd": cost_from_responses(responses),
        "billed_usd": None,
        "native_list_cost_calls": sum(
            isinstance(response, dict) and isinstance(response.get("native_cost"), dict)
            and response["native_cost"].get("basis") == "claude_code_list_price"
            for response in responses
        ),
        "calls": len(responses),
        "calls_missing_usage": missing_usage,
        "calls_missing_cache_details": missing_cache,
        "calls_reasoning_exceeds_completion": inconsistent_reasoning,
        "models_missing_prices": [model for model, prices in rates.items() if prices is None],
        "rates_per_million_input_output_cache_write": rates,
        "usage_status": "incomplete_or_inconsistent" if missing_cache or inconsistent_reasoning or missing_usage or any(p is None for p in rates.values()) else "reported_usage",
    }


def require_complete_cost_accounting(responses: list[Any]) -> dict[str, Any]:
    """Reject estimates whose provider usage or pricing is not auditable."""

    evidence = cost_accounting_from_responses(responses)
    if evidence["usage_status"] != "reported_usage":
        raise ValueError(
            "cost accounting is incomplete or inconsistent: "
            f"missing_usage={evidence['calls_missing_usage']}, "
            f"missing_cache_details={evidence['calls_missing_cache_details']}, "
            "reasoning_exceeds_completion="
            f"{evidence['calls_reasoning_exceeds_completion']}, "
            f"models_missing_prices={evidence['models_missing_prices']}"
        )
    return evidence


