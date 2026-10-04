import os
import re
from copy import deepcopy

##### API KEYS #####

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BITFROST_API_KEY = os.getenv("BITFROST_API_KEY")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
Z_AI_API_KEY = os.getenv("Z_AI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Long-lived Claude Code subscription OAuth token (from `claude setup-token`).
# Passed to Claude Code, which forwards it through the local LiteLLM gateway to
# Anthropic. The gateway authenticates separately with x-litellm-api-key.
CLAUDE_CODE_OAUTH_TOKEN = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")


#### Costs

# Dated provider list-price references used for reproducible API-equivalent
# accounting. Leanstral 1.5 is currently free; its zero price is intentional,
# while token counts are still retained in every response/trajectory.
MISTRAL_API_PRICING_DATE = "2026-08-28"
MISTRAL_API_PRICING_SOURCE = "https://docs.mistral.ai/models/leanstral-1-5"
Z_AI_API_PRICING_DATE = "2026-09-08"
Z_AI_API_PRICING_SOURCE = "https://docs.z.ai/guides/overview/pricing"
OPENAI_ASTRA_PRICING_DATE = "2026-09-08"
OPENAI_ASTRA_PRICING_SOURCE = (
    "https://developers.openai.com/api/docs/models/gpt-6-astra"
)

GEMINI_FLASH_LITE_PRICING_DATE = "2026-09-17"
GEMINI_FLASH_LITE_PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"


# input, output, cache (read), cache_creation
# NOTE: Claude models use the 1-HOUR cache-creation rate (2x base input), not
# the 5-minute rate (1.25x). Claude Code (both the subscription path and the
# proxy/API path) creates 1h ephemeral cache exclusively, so this is the rate
# that actually gets billed. Using the 5m rate undercounts cache-creation cost
# by ~40% (e.g. opus 6.25 -> 10.0) and the overall run cost by ~30%.
MODEL_PRICES = {
    "muse-spark-1.3": (1.25, 4.25, 0.15, 0.0),
    "muse-spark-1.3-contributor": (0.10, 0.20, 0.002, 0.0),
    "muse-spark-1.2": (1.25, 4.25, 0.15, 0.0),
    "muse-spark-1.2-contributor": (0.10, 0.20, 0.002, 0.0),
    "muse-spark-1.1": (1.25, 4.25, 0.15, 0.0),
    "Qwen3-Coder-30B-A3B-Instruct-FP8": (0.1, 0.3, 0.0, 0.0),
    "Qwen3.5-9B": (0.100, 0.145, 0.0, 0.0),  # SiliconFlow (via OpenRouter provider list)
    "Qwen3.6-35B-A3B": (0.20, 1.60, 0.0, 0.0),  # SiliconFlow (via OpenRouter provider list)
    "qwen3.5-397b-a17b": (0.39, 2.34, 0.0, 0.0),
    "Qwen3.5-397B-A17B": (0.60, 3.60, 0.0, 0.0),
    "qwen3.5-122b-a10b": (0.26, 2.08, 0.0, 0.0),
    "qwen3.5-35b-a3b": (0.1625, 1.30, 0.0, 0.0),
    "gemini-3.1-pro-preview": (2.0, 12.0, 0.0, 0.0),
    "gemini-3-flash-preview": (0.5, 3.0, 0.0, 0.0),
    # Standard text rates, pinned to GEMINI_FLASH_LITE_PRICING_DATE.
    # Output includes thinking. Cache storage is billed per token-hour and
    # is not represented by this token-only benchmark estimate.
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01, 0.0),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025, 0.0),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03, 0.0),
    # Introductory standard rates through 2026-12-31.
    "gemini-3.8-flash": (0.75, 3.75, 0.075, 0.0),
    "glm-5": (1.0, 3.2, 0.2, 0.0),
    "glm-5.3": (1.4, 4.4, 0.26, 0.0),
    # Stable list rates; do not bake Z.ai's temporary launch discount into
    # reproducible benchmark accounting.
    "glm-5.3-flash": (0.15, 0.50, 0.03, 0.0),
    "labs-leanstral-1-5": (0.0, 0.0, 0.0, 0.0),
    # Mistral Small 4 is Vibe's native Lean compaction model. No cached-input
    # rate is published, so cache reads are billed at the input rate.
    "mistral-small-latest": (0.15, 0.6, 0.15, 0.0),
    "mistral-small-2603": (0.15, 0.6, 0.15, 0.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0, 0.10, 2.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0, 0.3, 6.0),
    "claude-sonnet-4-6": (3.0, 15.0, 0.3, 6.0),
    "claude-sonnet-5": (3.0, 15.0, 0.30, 6.0),
    "claude-opus-4-5-20251101": (5.0, 25.0, 0.5, 10.0),
    "claude-opus-4-6": (5.0, 25.0, 0.5, 10.0),
    "claude-opus-4-8": (5.0, 25.0, 0.5, 10.0),
    "claude-opus-5": (5.0, 25.0, 0.5, 10.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 8.0),
    "claude-fable-5": (10.0, 50.0, 1.0, 12.5),
    "claude-fable-5-1": (10.0, 50.0, 0.25, 20.0),
    "gpt-5-codex": (1.25, 10.0, 0.125, 0.0),
    "gpt-5.1-codex-mini": (0.25, 2.0, 0.025, 0.0),
    "gpt-5-mini-2025-08-07": (0.25, 2.0, 0.02, 0.0),
    "gpt-5.2-codex": (1.75, 14.0, 0.175, 0.0),
    "gpt-5.3-codex": (1.75, 14.0, 0.175, 0.0),
    "gpt-5.3-codex-mini": (0.25, 2.0, 0.025, 0.0),
    "gpt-5.4": (2.50, 15.0, 0.25, 0.0),
    "gpt-5.4-mini": (0.75, 4.5, 0.075, 0.0),
    "gpt-5.4-nano-2026-03-17": (0.2,1.25,0.02,0.0),
    "gpt-5.5": (5.00, 30.0, 0.50, 0.0),
    # GPT-5.6 cache reads are 10% of input; explicit cache writes are 1.25x.
    "gpt-5.6-sol": (4.00, 20.0, 0.40, 5.00),
    "gpt-5.6-terra": (2.50, 15.0, 0.25, 3.125),
    "gpt-5.6-luna": (0.20, 1.20, 0.02, 0.25),
    # Standard rates from the official model page. The benchmark's 200k
    # auto-compaction threshold stays below Astra's 272k long-context surcharge.
    "gpt-6-astra": (10.00, 50.0, 1.00, 12.50),
    # Standard short-context rates, verified 2026-09-23:
    # https://developers.openai.com/api/docs/models/gpt-6-sol
    "gpt-6-sol": (2.00, 10.0, 0.20, 2.50),
    # https://developers.openai.com/api/docs/models/gpt-6.1-sol (2026-10-01)
    "gpt-6.1-sol": (2.00, 10.0, 0.10, 2.50),
    # https://developers.openai.com/api/docs/models/gpt-6-luna (2026-09-24)
    "gpt-6-luna": (0.10, 0.50, 0.01, 0.125),
}


_MODEL_PRICE_ALIASES = {
    "gpt-5.1-codex": "gpt-5-codex",
    "gpt-5.2-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5.6": "gpt-5.6-sol",
}


def get_model_prices(model_name: str | None) -> tuple[float, float, float, float] | None:
    """Resolve model price by exact name, alias, and common snapshot naming forms."""
    if not model_name:
        return None

    normalized = str(model_name).split("/")[-1]
    prices = MODEL_PRICES.get(normalized)
    if prices is not None:
        return prices

    alias = _MODEL_PRICE_ALIASES.get(normalized)
    if alias:
        prices = MODEL_PRICES.get(alias)
        if prices is not None:
            return prices

    if normalized.endswith("-latest"):
        prices = MODEL_PRICES.get(normalized[:-7])
        if prices is not None:
            return prices

    snapshot_match = re.match(r"^(.*)-\d{4}-\d{2}-\d{2}$", normalized)
    if snapshot_match:
        base = snapshot_match.group(1)
        prices = MODEL_PRICES.get(base)
        if prices is not None:
            return prices
        alias = _MODEL_PRICE_ALIASES.get(base)
        if alias:
            return MODEL_PRICES.get(alias)

    return None


def get_litellm_model_info(model_name: str) -> dict[str, float] | None:
    """Return a LiteLLM ``model_info`` dict built from our own price table.

    Prices in MODEL_PRICES are stored as $/1M tokens; LiteLLM expects $/token.
    Returns None for models not in our table (LiteLLM will skip cost tracking).
    """
    prices = get_model_prices(model_name)
    if prices is None:
        return None
    inp, out, cache_read, cache_creation = prices
    return {
        "input_cost_per_token": inp / 1_000_000,
        "output_cost_per_token": out / 1_000_000,
        "cache_read_input_token_cost": cache_read / 1_000_000,
        "cache_creation_input_token_cost": cache_creation / 1_000_000,
    }


###################### MODELS CONFIGURATION ######################

#############
# ANTHROPIC #
#############

# We use our bitfrost interface now

# MODEL_OPUS = {
#     "model_name": "anthropic/claude-opus-4-5-20251101",
#     "api_base": "https://llm-proxy.example.org/litellm",
#     "model_kwargs": {
#         "drop_params": True,
#         "temperature": 0.0,
#         "api_key": BITFROST_API_KEY,
#     },
# }

MODEL_OPUS = {
    "model_name": "anthropic/claude-opus-4-5-20251101",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}
MODEL_OPUS_4_6 = {
    "model_name": "anthropic/claude-opus-4-6",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}

MODEL_OPUS_4_8 = {
    "model_name": "anthropic/claude-opus-4-8",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}

MODEL_OPUS_4_8_LOW = {**MODEL_OPUS_4_8, "model_kwargs": {**MODEL_OPUS_4_8["model_kwargs"], "reasoning_effort": "low"}}
MODEL_OPUS_4_8_MEDIUM = {**MODEL_OPUS_4_8, "model_kwargs": {**MODEL_OPUS_4_8["model_kwargs"], "reasoning_effort": "medium"}}
MODEL_OPUS_4_8_HIGH = {**MODEL_OPUS_4_8, "model_kwargs": {**MODEL_OPUS_4_8["model_kwargs"], "reasoning_effort": "high"}}

MODEL_OPUS_5 = {
    "model_name": "anthropic/claude-opus-5",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
        "reasoning_effort": "high",
    },
}
MODEL_OPUS_5_5 = {
    **MODEL_OPUS_5,
    "model_name": "anthropic/claude-opus-5-5",
    "model_kwargs": {**MODEL_OPUS_5["model_kwargs"]},
}
MODEL_OPUS_5_LOW = deepcopy(MODEL_OPUS_5)
MODEL_OPUS_5_LOW["model_kwargs"]["reasoning_effort"] = "low"
MODEL_OPUS_5_MEDIUM = deepcopy(MODEL_OPUS_5)
MODEL_OPUS_5_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_OPUS_5_HIGH = deepcopy(MODEL_OPUS_5)

MODEL_SONNET = {
    "model_name": "anthropic/claude-sonnet-4-5-20250929",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}
MODEL_SONNET_4_6 = {
    "model_name": "anthropic/claude-sonnet-4-6",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}

MODEL_SONNET_5 = {
    "model_name": "anthropic/claude-sonnet-5",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}

MODEL_HAIKU = {
    "model_name": "anthropic/claude-haiku-4-5-20251001",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
    },
}

MODEL_FABLE_5 = {
    "model_name": "anthropic/claude-fable-5",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
        "reasoning_effort": "high",
    },
}

MODEL_FABLE_5_1 = {
    "model_name": "anthropic/claude-fable-5-1",
    "model_kwargs": {
        "drop_params": True,
        "api_key": ANTHROPIC_API_KEY,
        "reasoning_effort": "high",
    },
}


##########
# OPENAI #
##########

MODEL_GPT_5_MINI_HIGH = {
    "model_name": "openai/gpt-5-mini-2025-08-07",
    "api_base": "https://llm-proxy.example.org/litellm",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 0.0,
        "api_key": OPENAI_API_KEY,
        "reasoning_effort": "high",
    },
}
MODEL_GPT_5_MINI_MEDIUM = deepcopy(MODEL_GPT_5_MINI_HIGH)
MODEL_GPT_5_MINI_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT_5_NANO_HIGH = deepcopy(MODEL_GPT_5_MINI_HIGH)
MODEL_GPT_5_NANO_HIGH["model_name"] = "openrouter/openai/gpt-5-nano-2025-08-07"
MODEL_GPT_5_NANO_MEDIUM = deepcopy(MODEL_GPT_5_NANO_HIGH)
MODEL_GPT_5_NANO_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT_5_MEDIUM = {
    "model_name": "openai/gpt-5-2025-08-07",
    "model_kwargs": {
        "drop_params": True,
        "api_key": OPENAI_API_KEY,
        "reasoning_effort": "medium",
    },
}
MODEL_GPT_5_HIGH = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT_5_HIGH["model_kwargs"]["reasoning_effort"] = "high"

MODEL_GPT5_CODEX = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_CODEX["model_name"] = "openai/gpt-5-codex"

MODEL_GPT5_1_CODEX_MINI = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_2_CODEX_MINI = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_3_CODEX_MINI = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_1_CODEX_MINI["model_name"] = "openai/gpt-5.1-codex-mini"
MODEL_GPT5_2_CODEX_MINI["model_name"] = "openai/gpt-5.2-codex-mini"
MODEL_GPT5_3_CODEX_MINI["model_name"] = "openai/gpt-5.3-codex-mini"

MODEL_GPT5_4_MINI = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_4_MINI["model_name"] = "openai/gpt-5.4-mini"
MODEL_GPT5_4_MINI_LOW = deepcopy(MODEL_GPT5_4_MINI)
MODEL_GPT5_4_MINI_LOW["model_kwargs"]["reasoning_effort"] = "low"
MODEL_GPT5_4_MINI_MEDIUM = deepcopy(MODEL_GPT5_4_MINI)
MODEL_GPT5_4_MINI_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT5_4_MINI_HIGH = deepcopy(MODEL_GPT5_4_MINI)
MODEL_GPT5_4_MINI_HIGH["model_kwargs"]["reasoning_effort"] = "high"
MODEL_GPT5_4_MINI_XHIGH = deepcopy(MODEL_GPT5_4_MINI)
MODEL_GPT5_4_MINI_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT5_2_CODEX = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_2_CODEX["model_name"] = "openai/gpt-5.2-codex"
MODEL_GPT5_3_CODEX = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_3_CODEX["model_name"] = "openai/gpt-5.3-codex"
MODEL_GPT5_3_CODEX_MEDIUM = deepcopy(MODEL_GPT5_3_CODEX)
MODEL_GPT5_3_CODEX_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT5_3_CODEX_HIGH = deepcopy(MODEL_GPT5_3_CODEX)
MODEL_GPT5_3_CODEX_HIGH["model_kwargs"]["reasoning_effort"] = "high"
MODEL_GPT5_3_CODEX_XHIGH = deepcopy(MODEL_GPT5_3_CODEX)
MODEL_GPT5_3_CODEX_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"
MODEL_GPT5_4 = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_4["model_name"] = "openai/gpt-5.4"

MODEL_GPT5_4_LOW = deepcopy(MODEL_GPT5_4)
MODEL_GPT5_4_LOW["model_kwargs"]["reasoning_effort"] = "low"
MODEL_GPT5_4_MEDIUM = deepcopy(MODEL_GPT5_4)
MODEL_GPT5_4_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT5_4_HIGH = deepcopy(MODEL_GPT5_4)
MODEL_GPT5_4_HIGH["model_kwargs"]["reasoning_effort"] = "high"
MODEL_GPT5_4_XHIGH = deepcopy(MODEL_GPT5_4)
MODEL_GPT5_4_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT5_5 = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_5["model_name"] = "openai/gpt-5.5"
MODEL_GPT5_5_LOW = deepcopy(MODEL_GPT5_5)
MODEL_GPT5_5_LOW["model_kwargs"]["reasoning_effort"] = "low"
MODEL_GPT5_5_MEDIUM = deepcopy(MODEL_GPT5_5)
MODEL_GPT5_5_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT5_5_HIGH = deepcopy(MODEL_GPT5_5)
MODEL_GPT5_5_HIGH["model_kwargs"]["reasoning_effort"] = "high"
MODEL_GPT5_5_XHIGH = deepcopy(MODEL_GPT5_5)
MODEL_GPT5_5_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT5_6_SOL = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_6_SOL["model_name"] = "openai/gpt-5.6-sol"
MODEL_GPT5_6_SOL_MEDIUM = deepcopy(MODEL_GPT5_6_SOL)
MODEL_GPT5_6_SOL_HIGH = deepcopy(MODEL_GPT5_6_SOL)
MODEL_GPT5_6_SOL_HIGH["model_kwargs"]["reasoning_effort"] = "high"
MODEL_GPT5_6_SOL_XHIGH = deepcopy(MODEL_GPT5_6_SOL)
MODEL_GPT5_6_SOL_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT5_6_TERRA = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_6_TERRA["model_name"] = "openai/gpt-5.6-terra"
MODEL_GPT5_6_LUNA = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_6_LUNA["model_name"] = "openai/gpt-5.6-luna"

MODEL_GPT6_ASTRA = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT6_ASTRA["model_name"] = "openai/gpt-6-astra"
MODEL_GPT6_ASTRA_XHIGH = deepcopy(MODEL_GPT6_ASTRA)
MODEL_GPT6_ASTRA_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT6_SOL = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT6_SOL["model_name"] = "openai/gpt-6-sol"
MODEL_GPT6_SOL_XHIGH = deepcopy(MODEL_GPT6_SOL)
MODEL_GPT6_SOL_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT6_1_SOL = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT6_1_SOL["model_name"] = "openai/gpt-6.1-sol"
MODEL_GPT6_1_SOL_XHIGH = deepcopy(MODEL_GPT6_1_SOL)
MODEL_GPT6_1_SOL_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT6_LUNA = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT6_LUNA["model_name"] = "openai/gpt-6-luna"
MODEL_GPT6_LUNA_XHIGH = deepcopy(MODEL_GPT6_LUNA)
MODEL_GPT6_LUNA_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"

MODEL_GPT5_4_NANO = deepcopy(MODEL_GPT_5_MEDIUM)
MODEL_GPT5_4_NANO["model_name"] = "openai/gpt-5.4-nano-2026-03-17"
MODEL_GPT5_4_NANO_LOW = deepcopy(MODEL_GPT5_4_NANO)
MODEL_GPT5_4_NANO_LOW["model_kwargs"]["reasoning_effort"] = "low"
MODEL_GPT5_4_NANO_MEDIUM = deepcopy(MODEL_GPT5_4_NANO)
MODEL_GPT5_4_NANO_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"
MODEL_GPT5_4_NANO_HIGH = deepcopy(MODEL_GPT5_4_NANO)
MODEL_GPT5_4_NANO_HIGH["model_kwargs"]["reasoning_effort"] = "high"
MODEL_GPT5_4_NANO_XHIGH = deepcopy(MODEL_GPT5_4_NANO)
MODEL_GPT5_4_NANO_XHIGH["model_kwargs"]["reasoning_effort"] = "xhigh"


###########
# GPT OSS #
###########

MODEL_GPT_OSS_120B_HIGH = {
    "model_name": "hosted_vllm/openai/gpt-oss-120b",
    "api_base": "http://localhost:4001/v1",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 0.7,
        "reasoning_effort": "high",
    },
}

MODEL_GPT_OSS_120B_MEDIUM = deepcopy(MODEL_GPT_OSS_120B_HIGH)
MODEL_GPT_OSS_120B_MEDIUM["model_kwargs"]["reasoning_effort"] = "medium"

MODEL_GPTOSS_20B_HIGH = deepcopy(MODEL_GPT_OSS_120B_HIGH)
MODEL_GPTOSS_20B_HIGH["model_name"] = "hosted_vllm/openai/gpt-oss-20b"

# MODEL_GPTOSS_20B_HIGH = {
#     "model_name": "hosted_vllm/openai/gpt-oss-20b",
#     "api_base": "http://localhost:4000/v1",
#     "model_kwargs": {
#         "drop_params": True,
#         "temperature": 0.0,
#         "api_key": "anything",
#         "stream": False,
#     },
# }


##################
# Mistral MODELS #
##################

MODEL_DEVSTRAL_1_1 = {
    "model_name": "openrouter/mistralai/devstral-small",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 0.0,
        "api_key": OPENROUTER_API_KEY,
        "reasoning": {
            "effort": "high",
            "exclude": False,
        },
        "provider": {
            "order": ["mistral", "deepinfra"],
            "allow_fallbacks": False,
            "only": ["mistral", "deepinfra"],
        },
    },
}

# Leanstral's free Labs API is owned by the local LiteLLM process. Mistral Vibe
# receives only a fresh gateway key and cannot reach api.mistral.ai directly.
MODEL_LEANSTRAL_1_5 = {
    "model_name": "mistral/labs-leanstral-1-5",
    "api_base": "https://api.mistral.ai/v1",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 1.0,
        "reasoning_effort": "high",
        # LiteLLM's Mistral adapter drops reasoning_effort for every
        # non-"magistral" model (while still reporting it in
        # hidden_params.optional_params), so it must be allowed explicitly or
        # Leanstral runs without thinking.
        "allowed_openai_params": ["reasoning_effort"],
        "api_key": "os.environ/MISTRAL_API_KEY",
    },
}

# Vibe 2.25.0's built-in Lean agent compacts with this model (temperature 0.2,
# thinking off); the gateway serves it next to Leanstral.
MISTRAL_VIBE_COMPACTION_MODEL = "mistral-small-latest"

###############
# QWEN MODELS #
###############

if False:
    MODEL_QWEN3_30B_CODER = {
        "model_name": "openrouter/qwen/qwen3-coder-30b-a3b-instruct",
        "model_kwargs": {
            "drop_params": True,
            "temperature": 0.7,
            "api_key": OPENROUTER_API_KEY,
            "provider": {
                "order": ["nebius/fp8"],
                "allow_fallbacks": False,
                "only": ["nebius/fp8"],
            },
        },
    }

    MODEL_QWEN3_480B_CODER = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_480B_CODER["model_name"] = "openrouter/qwen/qwen3-coder"
    MODEL_QWEN3_5_397B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_5_397B["model_name"] = "openrouter/qwen/qwen3.5-397b-a17b"

else:
    MODEL_QWEN3_30B_CODER = {
        "model_name": "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "api_base": "http://localhost:4000/v1",
        "model_kwargs": {
            "drop_params": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "api_key": "anything",
            "stream": False,
            "max_completion_tokens": 4096,
        },
    }
    MODEL_QWEN3_480B_CODER = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_480B_CODER["model_name"] = (
        "hosted_vllm/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
    )
    MODEL_QWEN3_5_397B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_5_397B["model_name"] = "together_ai/Qwen/Qwen3.5-397B-A17B"
    MODEL_QWEN3_5_397B["model_kwargs"]["api_key"] = TOGETHER_API_KEY
    MODEL_QWEN3_5_397B.pop("api_base", None)
    MODEL_QWEN3_5_397B["model_kwargs"].pop("provider", None)
    MODEL_QWEN3_32B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_32B["model_name"] = "openrouter/qwen/qwen3-32b"
    MODEL_QWEN3_32B["model_kwargs"]["api_key"] = OPENROUTER_API_KEY
    MODEL_QWEN3_32B.pop("api_base", None)
    MODEL_QWEN3_32B["model_kwargs"]["provider"] = {
        "order": ["together", "nebius"],
        "allow_fallbacks": False,
        "only": ["together", "nebius"],
    }
    MODEL_QWEN3_5_122B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_5_122B["model_name"] = "openrouter/qwen/qwen3.5-122b-a10b"
    MODEL_QWEN3_5_122B["model_kwargs"]["api_key"] = OPENROUTER_API_KEY
    MODEL_QWEN3_5_122B.pop("api_base", None)
    MODEL_QWEN3_5_122B["model_kwargs"]["provider"] = {
        "order": ["together", "alibaba"],
        "allow_fallbacks": False,
        "only": ["together", "alibaba"],
    }
    MODEL_QWEN3_5_35B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_5_35B["model_name"] = "openrouter/qwen/qwen3.5-35b-a3b"
    MODEL_QWEN3_5_35B["model_kwargs"]["api_key"] = OPENROUTER_API_KEY
    MODEL_QWEN3_5_35B.pop("api_base", None)
    MODEL_QWEN3_5_35B["model_kwargs"]["provider"] = {
        "order": ["together", "alibaba"],
        "allow_fallbacks": False,
        "only": ["together", "alibaba"],
    }
    # Qwen3.5-9B and Qwen3.6-35B-A3B served locally via vLLM on distinct ports
    # from the 30B (4000). The serve scripts use Qwen's official qwen3_coder
    # tool parser and qwen3 reasoning parser.
    MODEL_QWEN3_5_9B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_5_9B["model_name"] = "hosted_vllm/Qwen/Qwen3.5-9B"
    MODEL_QWEN3_5_9B["api_base"] = "http://localhost:4002/v1"
    MODEL_QWEN3_5_9B["model_kwargs"].update({
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "max_completion_tokens": 32768,
    })
    MODEL_QWEN3_5_9B["model_kwargs"]["extra_body"] = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    MODEL_QWEN3_6_35B = deepcopy(MODEL_QWEN3_30B_CODER)
    MODEL_QWEN3_6_35B["model_name"] = "hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8"
    MODEL_QWEN3_6_35B["api_base"] = "http://localhost:4003/v1"
    MODEL_QWEN3_6_35B["model_kwargs"].update({
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "max_completion_tokens": 32768,
    })
    MODEL_QWEN3_6_35B["model_kwargs"]["extra_body"] = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }


################
# GEMINI MODELS #
################

MODEL_GEMINI_3_PRO = {
    "model_name": "gemini/gemini-3.1-pro-preview",
    "model_kwargs": {
        "drop_params": True,
        "api_key": GEMINI_API_KEY,
    },
}

MODEL_GEMINI_3_8_FLASH = {
    "model_name": "gemini/gemini-3.8-flash",
    "model_kwargs": {
        "drop_params": True,
        "reasoning_effort": "high",
        "api_key": "os.environ/GEMINI_API_KEY",
    },
}

###############
# Z-AI MODELS #
###############

MODEL_GLM5 = {
    "model_name": "anthropic/glm-5",
    "api_base": "https://api.z.ai/api/anthropic",
    "model_kwargs": {
        "drop_params": True,
        "api_key": "os.environ/Z_AI_API_KEY",
    },
}

# OpenAI-compatible Coding Plan endpoint, suitable for ZCode's custom-provider
# mode and future headless ZCode support. ZCode's direct account sign-in cannot
# be intercepted by LiteLLM, so reproducible proxy-backed runs use API-key mode.
MODEL_GLM5_3 = {
    "model_name": "openai/glm-5.3",
    "api_base": "https://api.z.ai/api/coding/paas/v4",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 1.0,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "api_key": "os.environ/Z_AI_API_KEY",
    },
}

# OpenCode-compatible Coding Plan endpoint. GLM-5.3-Flash defaults to max
# reasoning; temperature/top_p match the published model generation defaults.
MODEL_GLM5_3_FLASH = {
    "model_name": "openai/glm-5.3-flash",
    # Bifrost catalog ID; keep the local alias stable for OpenCode.
    "upstream_model": "openai/openrouter/z-ai/glm-5.3-flash",
    "api_base": "https://llm-proxy.example.org/v1",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
        "api_key": "os.environ/PROXY_API_KEY",
    },
}

MODEL_GLM_32B = {
    "model_name": "openrouter/z-ai/glm-4-32b",
    "model_kwargs": {
        "drop_params": True,
        "temperature": 0.0,
        "api_key": OPENROUTER_API_KEY,
    },
}


# Muse Spark 1.3 from Meta's API, through the same LiteLLM proxy as every other model.
# Muse streams nothing while it reasons, and a 128k-token response takes about 10 min,
# so the upstream timeout must outlast LiteLLM's 600 s default and the CLI's 1200 s
# stream-idle limit.
MODEL_MUSE_SPARK_1_3 = {
    "model_name": "openai/muse-spark-1.3",
    "model_class": "litellm_server",
    "api_base": "https://api.meta.ai/v1",
    "model_kwargs": {
        "api_key": "os.environ/META_API_KEY",
        "reasoning_effort": "high",
        "timeout": 1320,
        "stream_timeout": 1320,
    },
}


# Meta's Contributor catalogue ID for the same model: lower pricing, with prompts and
# outputs eligible for Meta training. Same direct route and timeouts.
MODEL_MUSE_SPARK_1_3_CONTRIBUTOR = {
    **MODEL_MUSE_SPARK_1_3,
    "model_name": "openai/muse-spark-1.3-contributor",
}



ALL_MODEL_CONFIGS = {
    "muse-spark-1.3": MODEL_MUSE_SPARK_1_3,
    "muse-spark-1.3-contributor": MODEL_MUSE_SPARK_1_3_CONTRIBUTOR,
    "gpt-5-mini-high": MODEL_GPT_5_MINI_HIGH,
    "gpt-5-mini-medium": MODEL_GPT_5_MINI_MEDIUM,
    "gpt-5-nano-high": MODEL_GPT_5_NANO_HIGH,
    "gpt-5-nano-medium": MODEL_GPT_5_NANO_MEDIUM,
    "gpt-5-medium": MODEL_GPT_5_MEDIUM,
    "gpt-5-high": MODEL_GPT_5_HIGH,
    "gpt-oss-120b-high": MODEL_GPT_OSS_120B_HIGH,
    "gpt-oss-120b-medium": MODEL_GPT_OSS_120B_MEDIUM,
    "gpt-oss-20b-high": MODEL_GPTOSS_20B_HIGH,
    "devstral-small": MODEL_DEVSTRAL_1_1,
    "leanstral-1.5": MODEL_LEANSTRAL_1_5,
    "glm-5": MODEL_GLM5,
    "glm-5.3": MODEL_GLM5_3,
    "glm-5.3-flash": MODEL_GLM5_3_FLASH,
    "glm-32b": MODEL_GLM_32B,
    "gpt-5-codex": MODEL_GPT5_CODEX,
    "qwen3-30b-coder": MODEL_QWEN3_30B_CODER,
    "qwen3-480b-coder": MODEL_QWEN3_480B_CODER,
    "qwen3.5-397b": MODEL_QWEN3_5_397B,
    "qwen3-32b": MODEL_QWEN3_32B,
    "qwen3.5-122b": MODEL_QWEN3_5_122B,
    "qwen3.5-35b": MODEL_QWEN3_5_35B,
    "qwen3.5-9b": MODEL_QWEN3_5_9B,
    "qwen3.6-35b": MODEL_QWEN3_6_35B,
    "gemini-3-pro": MODEL_GEMINI_3_PRO,
    "gemini-3.8-flash": MODEL_GEMINI_3_8_FLASH,
    "opus-4-5": MODEL_OPUS,
    "sonnet-4-5": MODEL_SONNET,
    "opus-4-6": MODEL_OPUS_4_6,
    "opus-4-8": MODEL_OPUS_4_8,
    "opus-4-8-low": MODEL_OPUS_4_8_LOW,
    "opus-4-8-medium": MODEL_OPUS_4_8_MEDIUM,
    "opus-4-8-high": MODEL_OPUS_4_8_HIGH,
    "opus-5": MODEL_OPUS_5,
    "opus-5.5": MODEL_OPUS_5_5,
    "opus-5-low": MODEL_OPUS_5_LOW,
    "opus-5-medium": MODEL_OPUS_5_MEDIUM,
    "opus-5-high": MODEL_OPUS_5_HIGH,
    "fable-5": MODEL_FABLE_5,
    "fable-5.1": MODEL_FABLE_5_1,
    "sonnet-4-6": MODEL_SONNET_4_6,
    "sonnet-5": MODEL_SONNET_5,
    "haiku": MODEL_HAIKU,
    "gpt-5.1-codex-mini": MODEL_GPT5_1_CODEX_MINI,
    "gpt-5.2-codex": MODEL_GPT5_2_CODEX,
    "gpt-5.2-codex-mini": MODEL_GPT5_2_CODEX_MINI,
    "gpt-5.3-codex": MODEL_GPT5_3_CODEX,
    "gpt-5.3-codex-medium": MODEL_GPT5_3_CODEX_MEDIUM,
    "gpt-5.3-codex-high": MODEL_GPT5_3_CODEX_HIGH,
    "gpt-5.3-codex-xhigh": MODEL_GPT5_3_CODEX_XHIGH,
    "gpt-5.3-codex-mini": MODEL_GPT5_3_CODEX_MINI,
    "gpt-5.4-mini": MODEL_GPT5_4_MINI,
    "gpt-5.4-mini-low": MODEL_GPT5_4_MINI_LOW,
    "gpt-5.4-mini-medium": MODEL_GPT5_4_MINI_MEDIUM,
    "gpt-5.4-mini-high": MODEL_GPT5_4_MINI_HIGH,
    "gpt-5.4-mini-xhigh": MODEL_GPT5_4_MINI_XHIGH,
    "gpt-5.4": MODEL_GPT5_4,
    "gpt-5.4-low": MODEL_GPT5_4_LOW,
    "gpt-5.4-medium": MODEL_GPT5_4_MEDIUM,
    "gpt-5.4-high": MODEL_GPT5_4_HIGH,
    "gpt-5.4-xhigh": MODEL_GPT5_4_XHIGH,
    "gpt-5.5": MODEL_GPT5_5,
    "gpt-5.5-low": MODEL_GPT5_5_LOW,
    "gpt-5.5-medium": MODEL_GPT5_5_MEDIUM,
    "gpt-5.5-high": MODEL_GPT5_5_HIGH,
    "gpt-5.5-xhigh": MODEL_GPT5_5_XHIGH,
    "gpt-5.6": MODEL_GPT5_6_SOL,
    "gpt-5.6-sol": MODEL_GPT5_6_SOL,
    "gpt-5.6-sol-medium": MODEL_GPT5_6_SOL_MEDIUM,
    "gpt-5.6-sol-high": MODEL_GPT5_6_SOL_HIGH,
    "gpt-5.6-sol-xhigh": MODEL_GPT5_6_SOL_XHIGH,
    "gpt-5.6-terra": MODEL_GPT5_6_TERRA,
    "gpt-5.6-luna": MODEL_GPT5_6_LUNA,
    "gpt-6-astra": MODEL_GPT6_ASTRA,
    "gpt-6-astra-xhigh": MODEL_GPT6_ASTRA_XHIGH,
    "gpt-6-sol": MODEL_GPT6_SOL,
    "gpt-6-sol-xhigh": MODEL_GPT6_SOL_XHIGH,
    "gpt-6.1-sol": MODEL_GPT6_1_SOL,
    "gpt-6.1-sol-xhigh": MODEL_GPT6_1_SOL_XHIGH,
    "gpt-6-luna": MODEL_GPT6_LUNA,
    "gpt-6-luna-xhigh": MODEL_GPT6_LUNA_XHIGH,
    "gpt-5.4-nano": MODEL_GPT5_4_NANO,
    "gpt-5.4-nano-low": MODEL_GPT5_4_NANO_LOW,
    "gpt-5.4-nano-medium": MODEL_GPT5_4_NANO_MEDIUM,
    "gpt-5.4-nano-high": MODEL_GPT5_4_NANO_HIGH,
    "gpt-5.4-nano-xhigh": MODEL_GPT5_4_NANO_XHIGH,
}
