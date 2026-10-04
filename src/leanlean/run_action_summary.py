"""Build the lightweight action report emitted when an agent run finishes."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_trace import extract_action_sequence, summarize_action_sequences


ACTION_SUMMARY_FILENAME = "action_summary.json"
ACTION_SUMMARY_SCHEMA_VERSION = 4
_ROUND_DIR = re.compile(r"round_(\d+)$")


def _lean_verify_duration_samples(
    usage: dict[str, Any], instance_id: str
) -> list[dict[str, Any]]:
    """Return one compact observation for each captured Lean Verify call."""

    events = usage.get("events")
    if not isinstance(events, list):
        events = []
    started = {
        str(event.get("invocation_id"))
        for event in events
        if isinstance(event, dict)
        and event.get("tool") == "lean_verify"
        and event.get("event") == "started"
        and event.get("invocation_id")
    }
    finished: set[str] = set()
    samples: list[dict[str, Any]] = []
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("tool") != "lean_verify"
            or event.get("event") != "finished"
        ):
            continue
        invocation_id = str(event.get("invocation_id") or "")
        if invocation_id:
            finished.add(invocation_id)
        exit_code = event.get("exit_code")
        duration_ms = event.get("duration_ms")
        samples.append(
            {
                "instance_id": instance_id,
                "invocation_id": invocation_id or None,
                "duration_ms": duration_ms if isinstance(duration_ms, int) else None,
                "exit_code": exit_code if isinstance(exit_code, int) else None,
                "status": "succeeded" if exit_code == 0 else "failed",
                "phase": "agent",
            }
        )
    samples.extend(
        {
            "instance_id": instance_id,
            "invocation_id": invocation_id,
            "duration_ms": None,
            "exit_code": None,
            "status": "incomplete",
            "phase": "agent",
        }
        for invocation_id in sorted(started - finished)
    )
    setup_events = usage.get("setup_events")
    if isinstance(setup_events, list):
        samples.extend(
            {
                "instance_id": instance_id,
                "invocation_id": str(event.get("invocation_id") or "setup-prewarm"),
                "duration_ms": event.get("duration_ms"),
                "exit_code": event.get("exit_code"),
                "status": str(event.get("status") or "unknown"),
                "phase": "setup_prewarm",
            }
            for event in setup_events
            if isinstance(event, dict)
            and event.get("tool") == "lean_verify"
            and isinstance(event.get("duration_ms"), int)
        )
    return samples


def _trajectory_paths_by_instance(run_dir: Path) -> dict[str, list[Path]]:
    """Prefer per-round trajectories over their legacy/root duplicates."""

    rounds: dict[str, list[tuple[int, Path]]] = {}
    for path in run_dir.glob("round_*/*/*.traj.json"):
        match = _ROUND_DIR.fullmatch(path.parent.parent.name)
        if match:
            rounds.setdefault(path.parent.name, []).append((int(match.group(1)), path))

    legacy = {
        path.parent.name: path
        for path in run_dir.glob("*/*.traj.json")
        if not _ROUND_DIR.fullmatch(path.parent.name)
    }
    instance_ids = sorted(set(rounds) | set(legacy))
    return {
        instance_id: (
            [path for _round, path in sorted(rounds[instance_id])]
            if instance_id in rounds
            else [legacy[instance_id]]
        )
        for instance_id in instance_ids
    }


def build_run_action_summary(run_dir: Path) -> dict[str, Any]:
    """Extract compact ordered actions from every trajectory in a run."""

    run_dir = run_dir.resolve()
    paths_by_instance = _trajectory_paths_by_instance(run_dir)
    sequences: dict[str, list[str]] = {}
    trace_mtimes: list[int] = []
    trace_count = 0
    for instance_id, paths in paths_by_instance.items():
        sequence: list[str] = []
        for path in paths:
            try:
                trajectory = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(trajectory, dict):
                continue
            sequence.extend(extract_action_sequence(trajectory))
            trace_mtimes.append(path.stat().st_mtime_ns)
            trace_count += 1
        if sequence:
            sequences[instance_id] = sequence

    tool_instances: dict[str, dict[str, Any]] = {}
    tool_totals: dict[str, dict[str, int]] = {}
    tool_usage_mtimes: list[int] = []
    lean_verify_samples: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/agent_tool_usage.json")):
        try:
            usage = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        tools = usage.get("tools") if isinstance(usage, dict) else None
        if not isinstance(tools, dict):
            continue
        instance_id = path.parent.name
        tool_instances[instance_id] = tools
        lean_verify_samples.extend(
            _lean_verify_duration_samples(usage, instance_id)
        )
        tool_usage_mtimes.append(path.stat().st_mtime_ns)
        for tool, counts in tools.items():
            if not isinstance(counts, dict):
                continue
            total = tool_totals.setdefault(
                str(tool),
                {
                    "calls": 0,
                    "completed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "incomplete": 0,
                    "duration_ms": 0,
                },
            )
            for field in total:
                value = counts.get(field, 0)
                if isinstance(value, int):
                    total[field] += value

    return {
        "schema_version": ACTION_SUMMARY_SCHEMA_VERSION,
        "kind": "leanlean_run_action_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "trace_count": trace_count,
            "max_trace_mtime_ns": max(trace_mtimes, default=None),
            "tool_usage_file_count": len(tool_instances),
            "max_tool_usage_mtime_ns": max(tool_usage_mtimes, default=None),
        },
        "benchmark_tools": {
            "totals": tool_totals,
            "instances": tool_instances,
            "duration_samples": {"lean_verify": lean_verify_samples},
        },
        **summarize_action_sequences(sequences),
    }


def write_run_action_summary(run_dir: Path) -> Path:
    """Atomically write the compact action report for ``run_dir``."""

    run_dir = run_dir.resolve()
    summary = build_run_action_summary(run_dir)
    destination = run_dir / ACTION_SUMMARY_FILENAME
    temporary = run_dir / f".{ACTION_SUMMARY_FILENAME}.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return destination


def read_run_action_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / ACTION_SUMMARY_FILENAME
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != ACTION_SUMMARY_SCHEMA_VERSION
        or not isinstance(summary.get("instances"), dict)
    ):
        return None
    return summary
