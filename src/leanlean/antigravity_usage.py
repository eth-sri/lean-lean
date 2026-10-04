"""Recorded Antigravity token usage, with explicit source-boundary attribution.

Token counts are measured. Prices are benchmark rates, not an invoice; a model
inherited from the session is explicitly distinguished from a per-step model.
Never interpolate costs from elapsed time.
"""
from __future__ import annotations

import bisect
import copy
from typing import Any

from configs.model_constants import get_model_prices

TOKEN_KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens")


def checked_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("missing Antigravity usage")
    result = {key: value.get(key) for key in TOKEN_KEYS}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in result.values()):
        raise ValueError("incomplete Antigravity usage")
    if result["thinking_tokens"] > result["output_tokens"]:
        raise ValueError("Antigravity thinking exceeds output")
    if "total_tokens" in value and value["total_tokens"] != result["input_tokens"] + result["output_tokens"]:
        raise ValueError("inconsistent Antigravity total_tokens")
    return result


def normalized_response(model: str, usage: dict) -> dict:
    u = checked_usage(usage)
    if get_model_prices(model) is None:
        raise ValueError(f"missing Antigravity model prices: {model}")
    prompt = u["input_tokens"] + u["cache_read_tokens"]
    return {"model": model, "usage": {
        "prompt_tokens": prompt, "completion_tokens": u["output_tokens"],
        "total_tokens": prompt + u["output_tokens"],
        "prompt_tokens_details": {"cached_tokens": u["cache_read_tokens"]},
        "completion_tokens_details": {"reasoning_tokens": u["thinking_tokens"]},
    }}


def usage_cost(model: str, usage: dict) -> float:
    u = checked_usage(usage)
    rates = get_model_prices(model)
    if rates is None:
        raise ValueError(f"missing Antigravity model prices: {model}")
    return (u["input_tokens"] * rates[0] + u["output_tokens"] * rates[1]
            + u["cache_read_tokens"] * rates[2]) / 1_000_000


class AntigravityUsageJournal:
    def __init__(self, model: str | None = None):
        self.model = model
        self.sequence = 0
        self.steps: dict[tuple[str, int], dict] = {}
        self.responses: list[dict] = []
        self.total = dict.fromkeys(TOKEN_KEYS, 0)
        self.cost = 0.0
        self.sequences = [0]
        self.boundaries = [self._boundary()]
        self.tools: dict[str, dict] = {}
        self.terminal_usage: dict | None = None
        self.inferred_model_steps = 0

    def _boundary(self) -> dict:
        return {"cost_usd": round(self.cost, 8), "native_usage": dict(self.total),
                "native_event_sequence": self.sequence}

    def ingest(self, event: dict) -> None:
        self.sequence += 1
        kind = event.get("event")
        if kind == "init":
            self.model = event.get("init", {}).get("model") or self.model
        elif kind == "step_update":
            step = event.get("step_update", {})
            if isinstance(step.get("usage"), dict):
                if step.get("state") != "DONE":
                    raise ValueError("Antigravity usage on unfinished step")
                identity = (step.get("conversation_id"), step.get("step_index"))
                if not identity[0] or not isinstance(identity[1], int):
                    raise ValueError("Antigravity usage has no stable step identity")
                usage = checked_usage(step["usage"])
                model = step.get("model") or self.model
                if not model:
                    raise ValueError("Antigravity usage has no model identity")
                record = {"model": model, "usage": usage}
                if identity in self.steps:
                    if self.steps[identity] != record:
                        raise ValueError("conflicting duplicate Antigravity step usage")
                else:
                    response = normalized_response(model, usage)
                    self.steps[identity] = record
                    self.responses.append(response)
                    for key in TOKEN_KEYS:
                        self.total[key] += usage[key]
                    self.cost += usage_cost(model, usage)
                    self.inferred_model_steps += not bool(step.get("model"))
                    self.sequences.append(self.sequence)
                    self.boundaries.append(self._boundary())
            if step.get("step_type") == "tool" and step.get("state") == "DONE":
                tool = f"antigravity-step-{step.get('step_index')}"
                boundary = self._boundary()
                if tool in self.tools and self.tools[tool]["native_usage"] != boundary["native_usage"]:
                    raise ValueError("ambiguous repeated Antigravity tool boundary")
                self.tools[tool] = boundary
        elif kind == "result":
            value = event.get("result", {}).get("usage")
            if value is not None:
                self.terminal_usage = checked_usage(value)

    def validate(self) -> None:
        if not self.steps and self.terminal_usage is None:
            raise ValueError("no recorded Antigravity usage; cost is unknown, not zero")
        if self.steps and self.terminal_usage is not None and self.total != self.terminal_usage:
            raise ValueError("Antigravity step usage does not reconcile with terminal usage")

    def at(self, sequence: int) -> dict:
        if not isinstance(sequence, int) or sequence < 0 or sequence > self.sequence:
            raise ValueError("invalid native usage boundary sequence")
        return copy.deepcopy(self.boundaries[bisect.bisect_right(self.sequences, sequence) - 1])

    def evidence(self) -> dict:
        self.validate()
        usage = self.total if self.steps else self.terminal_usage
        final = self.cost if self.steps else usage_cost(self.model, usage)
        return {"source": "antigravity_native_step_usage", "usage": dict(usage),
                "recorded_cost_usd": round(final, 8), "completed_usage_steps": len(self.steps),
                "terminal_usage_reconciled": bool(self.steps and self.terminal_usage is not None),
                "usage_complete": self.terminal_usage is not None,
                "model_attribution": "session_model_inferred" if self.inferred_model_steps or not self.steps else "per_step_model",
                "session_model": self.model, "auxiliary_model_coverage_verified": False,
                "benchmark_estimate_not_invoice": True, "cost_prorated": False}


def apply_recorded_costs(playback: dict, journal: AntigravityUsageJournal) -> dict:
    """Mutate playback only after validating usage. Unknown boundaries stay null.

    Historical tool IDs locate received model usage at tool completion. They do
    not prove an asynchronous source snapshot was taken at that exact instant.
    New native_event_sequence fields locate recorded usage at source capture.
    """
    evidence = journal.evidence()
    final = evidence["recorded_cost_usd"]
    observations = playback.get("checkpoint_observations") or []
    by_edit = {o["edit_index"]: o for o in observations if o.get("edit_index") is not None}

    def update(record, observation=False):
        field = "cumulative_cost_usd" if observation else "cost_usd"
        kind = record.get("kind")
        sequence = record.get("native_event_sequence")
        source = by_edit.get(record.get("edit_index"), {}) if not observation else {}
        if sequence is None:
            sequence = source.get("native_event_sequence")
        boundary = None
        exact = False
        basis = "unknown_source_boundary"
        if kind == "baseline":
            boundary = journal.at(0); exact = True; basis = "baseline"
        elif kind == "final":
            boundary = {"cost_usd": final, "native_usage": evidence["usage"]}
            exact = evidence["usage_complete"]; basis = "recorded_terminal" if exact else "captured_steps_only"
        elif sequence is not None and journal.steps:
            boundary = journal.at(sequence); exact = True; basis = "recorded_source_sequence"
        elif journal.steps:
            tool = record.get("tool_use_id") or source.get("tool_use_id")
            boundary = journal.tools.get(tool)
            if boundary:
                basis = "native_tool_event_inferred_source_boundary"
        record[field] = boundary["cost_usd"] if boundary else None
        record["native_usage"] = copy.deepcopy(boundary["native_usage"]) if boundary else None
        record["cost_boundary_exact"] = exact
        record["cost_join_basis"] = basis
        record["model_attribution"] = evidence["model_attribution"]
        record.pop("marginal_cost_since_previous_edit_usd", None)

    for observation in observations:
        update(observation, True)
    points = playback.get("points") or []
    previous_cost = 0.0
    previous_usage = dict.fromkeys(TOKEN_KEYS, 0)
    for point in points:
        update(point)
        value = point["cost_usd"]
        usage = point["native_usage"]
        point["marginal_cost_since_previous_edit_usd"] = (
            round(value - previous_cost, 8) if value is not None and previous_cost is not None else None)
        point["marginal_native_usage"] = (
            {k: usage[k] - previous_usage[k] for k in TOKEN_KEYS}
            if usage is not None and previous_usage is not None else None)
        previous_cost, previous_usage = value, usage
    # These are bounds on received native usage between linked events, not a
    # guessed cost or an upper bound on unreported/in-flight provider charges.
    for records, field in ((points, "cost_usd"), (observations, "cumulative_cost_usd")):
        lower = 0.0
        for record in records:
            if record[field] is not None:
                lower = record[field]
            record["recorded_cost_lower_bound_usd"] = lower
        upper = final
        for record in reversed(records):
            if record[field] is not None:
                upper = record[field]
            record["recorded_cost_upper_bound_usd"] = upper
            if record["recorded_cost_lower_bound_usd"] > upper + 1e-7:
                raise ValueError("nonmonotone Antigravity source boundary attribution")
    playback.update(final_cost_usd=final, authoritative_final_cost_usd=final if evidence["usage_complete"] else None,
                    captured_completed_turn_cost_usd=final, cost_prorated=False,
                    cost_basis="recorded native token usage at benchmark rates; no time proration",
                    native_cost_evidence=evidence, final_cost_complete=evidence["usage_complete"],
                    unattributed_terminal_cost_usd=0.0 if evidence["usage_complete"] else None,
                    every_edit_checkpoint_costed=all(p["cost_boundary_exact"] for p in points if p.get("kind") not in {"baseline", "final"}))
    playback["gateway_cost_evidence"] = None
    playback["cost_timeline_complete"] = all(p["cost_usd"] is not None for p in points)
    return evidence
