"""Compact run-level indexes for LeanLean evaluation reports.

The evaluator's per-repository ``report.json`` files intentionally retain full
signature snapshots and diagnostics.  Those files can be tens of megabytes and
must not be read en masse by dashboards.  This module derives one small
``report_summary.json`` at the run root for aggregate views; a detailed report
is then loaded only for the repository the user opens.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_action_summary import build_run_action_summary, read_run_action_summary


SCHEMA_VERSION = 2
SUMMARY_FILENAME = "report_summary.json"
_ROUND_DIR = re.compile(r"round_(\d+)$")
_SORRY = re.compile(r"(?<![A-Za-z_])(?:sorry|admit)(?![A-Za-z_])")

_SCALAR_METRICS = (
    "lean_file_count",
    "baseline_words",
    "post_words",
    "compression_ratio",
    "words_saved",
    "baseline_lean_tokens",
    "post_lean_tokens",
    "lean_token_ratio",
    "lean_tokens_saved",
    "baseline_decl_count",
    "post_decl_count",
    "build_time_seconds",
    "baseline_heartbeats",
    "heartbeats",
    "heartbeat_ratio",
    "heartbeats_saved",
    "agent_commits",
    "files_modified",
    "files_read",
)


def _valid_number(value: Any, *, positive: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return value > 0 if positive else value >= 0


def _source_metrics_complete(metrics: dict[str, Any]) -> bool:
    return (
        _valid_number(metrics.get("baseline_words"), positive=True)
        and _valid_number(metrics.get("post_words"))
        and _valid_number(metrics.get("words_saved"))
        and _valid_number(metrics.get("baseline_lean_tokens"), positive=True)
        and _valid_number(metrics.get("post_lean_tokens"))
        and _valid_number(metrics.get("lean_tokens_saved"))
    )


def _round_number(run_dir: Path, report_path: Path) -> int:
    relative = report_path.relative_to(run_dir)
    for part in relative.parts:
        match = _ROUND_DIR.fullmatch(part)
        if match:
            return int(match.group(1))
    return 0


def _count_patch_files(patch: str) -> int:
    return len(re.findall(r"^diff --git ", patch, re.MULTILINE))


def _count_sorries(patch: str) -> int:
    net = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            sign = 1
        elif line.startswith("-"):
            sign = -1
        else:
            continue
        net += sign * len(_SORRY.findall(line[1:].split("--", 1)[0]))
    return max(net, 0)


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _round_predictions(run_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for path in run_dir.glob("preds_round_*.json"):
        match = re.fullmatch(r"preds_round_(\d+)\.json", path.name)
        if match:
            result[int(match.group(1))] = _read_predictions(path)
    if not result:
        result[0] = _read_predictions(run_dir / "preds.json")
    return result


def _generation_costs(run_dir: Path, instance_id: str) -> dict[int, float]:
    costs: dict[int, float] = {}
    for path in (run_dir / instance_id).glob("round_*_gen_metrics.json"):
        match = re.fullmatch(r"round_(\d+)_gen_metrics\.json", path.name)
        if not match:
            continue
        try:
            value = json.loads(path.read_text()).get("llm_cost")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, (int, float)):
            costs[int(match.group(1))] = float(value)
    return costs


def compact_report(
    instance_id: str,
    report: dict[str, Any],
    *,
    report_path: str,
    round_number: int,
    patch: str,
    round_cost: float,
) -> dict[str, Any]:
    """Return the small, aggregate-safe portion of one detailed report."""

    metrics = report.get("metrics") or {}
    signatures = metrics.get("signatures") or {}
    build = report.get("build_result") or {}
    apply_patch = report.get("apply_patch_result") or {}
    lake_builds = metrics.get("lake_builds") or {}
    compact_metrics = {key: metrics.get(key) for key in _SCALAR_METRICS}
    for key in _SCALAR_METRICS:
        value = compact_metrics.get(key)
        if isinstance(value, (int, float)) and value < 0:
            compact_metrics[key] = None
    source_metrics_complete = _source_metrics_complete(compact_metrics)
    return {
        "instance_id": instance_id,
        "round": round_number,
        "report_path": report_path,
        "resolved": report.get("resolved") is True,
        "empty_patch": report.get("empty_patch") is True,
        "apply_patch_ok": apply_patch.get("returncode") == 0,
        "build_passed": build.get("passed") is True,
        "build_ran": build.get("ran"),
        "signatures_preserved": signatures.get("preserved") is True,
        "signature_counts": {
            "root": signatures.get("root_count"),
            "root_removed": len(signatures.get("root_removed") or []),
            "root_changed": len(signatures.get("root_changed") or []),
            "total_removed": len(signatures.get("total_removed") or []),
            "total_changed": len(signatures.get("total_changed") or []),
        },
        "n_sorry": _count_sorries(patch),
        "patch_files": _count_patch_files(patch),
        "round_cost": round_cost,
        "lake_builds": {
            "attempts": lake_builds.get("attempts", 0),
            "successes": lake_builds.get("successes", 0),
            "failures": lake_builds.get("failures", 0),
        },
        "source_metrics_complete": source_metrics_complete,
        "metrics": compact_metrics,
    }


def build_run_report_summary(run_dir: Path) -> dict[str, Any]:
    """Build a compact index from every detailed report below ``run_dir``."""

    run_dir = run_dir.resolve()
    predictions = _round_predictions(run_dir)
    final_predictions = _read_predictions(run_dir / "preds.json")
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    report_mtimes: list[int] = []

    for report_file in sorted(run_dir.rglob("report.json")):
        try:
            payload = json.loads(report_file.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or not payload:
            continue
        instance_id, report = next(iter(payload.items()))
        if not isinstance(report, dict):
            continue
        round_number = _round_number(run_dir, report_file)
        patch = (
            predictions.get(round_number, {})
            .get(instance_id, {})
            .get("model_patch", "")
        )
        costs = _generation_costs(run_dir, instance_id)
        compact = compact_report(
            instance_id,
            report,
            report_path=str(report_file.relative_to(run_dir)),
            round_number=round_number,
            patch=patch,
            round_cost=costs.get(round_number, 0.0),
        )
        grouped.setdefault(instance_id, {})[round_number] = compact
        report_mtimes.append(report_file.stat().st_mtime_ns)

    instances: dict[str, dict[str, Any]] = {}
    for instance_id, rounds in sorted(grouped.items()):
        latest_round = max(rounds)
        latest = dict(rounds[latest_round])
        final_patch = (
            final_predictions.get(instance_id, {}).get("model_patch", "")
        )
        costs = _generation_costs(run_dir, instance_id)
        latest["n_sorry"] = _count_sorries(final_patch)
        latest["patch_files"] = _count_patch_files(final_patch)
        latest["cost_total"] = sum(costs.values())
        latest["lake_builds_total"] = {
            key: sum((entry.get("lake_builds") or {}).get(key, 0) for entry in rounds.values())
            for key in ("attempts", "successes", "failures")
        }
        instances[instance_id] = {
            "latest_round": latest_round,
            "latest": latest,
            "rounds": {str(number): rounds[number] for number in sorted(rounds)},
        }

    actions = read_run_action_summary(run_dir) or build_run_action_summary(run_dir)

    latest_values = [entry["latest"] for entry in instances.values()]
    prediction_ids = set(final_predictions)
    if not prediction_ids:
        prediction_ids = {
            instance_id
            for round_predictions in predictions.values()
            for instance_id in round_predictions
        }
    report_ids = set(instances)
    missing_reports = sorted(prediction_ids - report_ids)
    unexpected_reports = sorted(report_ids - prediction_ids)
    evaluation_complete = (
        bool(prediction_ids) and not missing_reports and not unexpected_reports
    )

    source_metric_values = [
        entry for entry in latest_values if entry["source_metrics_complete"]
    ]
    all_source_metrics_complete = bool(latest_values) and (
        len(source_metric_values) == len(latest_values)
    )
    totals = {
        "repos": len(instances),
        "predictions": len(prediction_ids),
        "resolved": sum(entry["resolved"] for entry in latest_values),
        "build_passed": sum(entry["build_passed"] for entry in latest_values),
        "signatures_preserved": sum(
            entry["signatures_preserved"] for entry in latest_values
        ),
        "repos_with_sorry": sum(entry["n_sorry"] > 0 for entry in latest_values),
        "cost_total": sum(entry["cost_total"] for entry in latest_values),
        "source_metrics_complete": all_source_metrics_complete,
        "evaluation_complete": evaluation_complete,
        "source_metric_repos": len(source_metric_values),
        "baseline_words": (
            sum(entry["metrics"]["baseline_words"] for entry in source_metric_values)
            if all_source_metrics_complete else None
        ),
        "words_saved": (
            sum(entry["metrics"]["words_saved"] for entry in source_metric_values)
            if all_source_metrics_complete else None
        ),
        "baseline_lean_tokens": (
            sum(
                entry["metrics"]["baseline_lean_tokens"]
                for entry in source_metric_values
            )
            if all_source_metrics_complete else None
        ),
        "lean_tokens_saved": (
            sum(entry["metrics"]["lean_tokens_saved"] for entry in source_metric_values)
            if all_source_metrics_complete else None
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "leanlean_run_report_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "report_count": len(report_mtimes),
            "max_report_mtime_ns": max(report_mtimes, default=None),
        },
        "totals": totals,
        "completeness": {
            "complete": evaluation_complete and all_source_metrics_complete,
            "prediction_repos": len(prediction_ids),
            "report_repos": len(report_ids),
            "missing_reports": missing_reports,
            "unexpected_reports": unexpected_reports,
            "source_metrics_complete": all_source_metrics_complete,
        },
        "actions": actions,
        "instances": instances,
    }


def write_run_report_summary(run_dir: Path) -> Path:
    """Atomically write ``report_summary.json`` for a run."""

    run_dir = run_dir.resolve()
    summary = build_run_report_summary(run_dir)
    destination = run_dir / SUMMARY_FILENAME
    if not summary["instances"] and destination.is_file():
        try:
            existing = json.loads(destination.read_text())
        except (OSError, ValueError):
            existing = None
        if (
            isinstance(existing, dict)
            and isinstance(existing.get("instances"), dict)
            and existing["instances"]
            and (existing.get("source") or {}).get("basis")
            == "authoritative postprocessed submitted endpoints"
        ):
            # Palomar postprocessing publishes compact records from derived
            # playback files rather than legacy report.json sidecars. A generic
            # reindex must not erase those authoritative evaluation results.
            existing["generated_at"] = datetime.now(timezone.utc).isoformat()
            existing["actions"] = summary.get("actions")
            summary = existing
    temporary = run_dir / f".{SUMMARY_FILENAME}.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return destination


def read_run_report_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / SUMMARY_FILENAME
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("instances"), dict)
    ):
        return None
    return value
