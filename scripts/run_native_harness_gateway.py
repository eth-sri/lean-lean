#!/usr/bin/env python3
"""Run a host LiteLLM gateway for an interactive first-party model harness."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
from typing import Any

from configs import ALL_MODEL_CONFIGS
from configs.model_constants import get_model_prices
from leanlean.model import get_model


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def usage_totals(trace_path: Path) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    if not trace_path.is_file():
        return totals
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            trace = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(trace, dict) or trace.get("event") != "success":
            continue
        response = trace.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            continue
        input_tokens = _int(usage.get("prompt_tokens", usage.get("input_tokens")))
        output_tokens = _int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        cached_tokens = (
            _int(prompt_details.get("cached_tokens"))
            if isinstance(prompt_details, dict)
            else _int(usage.get("cache_read_input_tokens"))
        )
        reasoning_tokens = (
            _int(completion_details.get("reasoning_tokens"))
            if isinstance(completion_details, dict)
            else _int(usage.get("reasoning_tokens"))
        )
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["cached_input_tokens"] += cached_tokens
        totals["reasoning_output_tokens"] += reasoning_tokens
        totals["total_tokens"] += _int(
            usage.get("total_tokens", input_tokens + output_tokens)
        )
    return totals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expose one registered model through a host-only LiteLLM gateway "
            "for an interactive native harness such as ZCode."
        )
    )
    parser.add_argument("--model", default="glm-5.3", choices=sorted(ALL_MODEL_CONFIGS))
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("logs/native_harness_gateway"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or (
        f"{args.model}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise SystemExit("--run-id must contain only letters, digits, dot, dash, underscore")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = deepcopy(ALL_MODEL_CONFIGS[args.model])
    config.update(
        {
            "model_class": "litellm_server",
            "host": "127.0.0.1",
            "port": args.port,
            "log_dir": str(output_dir / "server"),
        }
    )
    model = get_model(config=config)
    started_at = _utc_now()
    trace_source: Path | None = None
    try:
        model.serve()
        trace_source = Path(model.get_traces_path())
        bare_model = str(model.config.model_name).split("/")[-1]
        print()
        print("Native harness gateway is ready")
        print(f"  Base URL: {model.proxy_url}")
        print(f"  API key:  {model.get_api_key()}")
        print(f"  Model:    {bare_model}")
        print(f"  Run ID:   {run_id}")
        print("Configure ZCode as a custom OpenAI-compatible provider.")
        print("Press Ctrl-C to save the gateway trace and summary.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    finally:
        cost_usd = model.get_cost()
        call_count = model.n_calls
        trace_destination = output_dir / "litellm-model-calls.jsonl"
        if trace_source is not None and trace_source.is_file():
            shutil.copyfile(trace_source, trace_destination)
        totals = usage_totals(trace_destination)
        summary = {
            "format": "native-harness-gateway-run-v1",
            "run_id": run_id,
            "registry_model": args.model,
            "gateway_model": model.config.model_name,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "call_count": call_count,
            "token_usage": totals,
            "api_equivalent_cost_usd": cost_usd,
            "prices_usd_per_million_tokens": get_model_prices(
                model.config.model_name
            ),
            "model_call_trace": (
                trace_destination.name if trace_destination.is_file() else None
            ),
            "native_harness_trace_available": False,
            "contains_persisted_gateway_key": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        model.stop()
        print(f"Saved native-harness gateway artifacts to {output_dir}")


if __name__ == "__main__":
    main()
