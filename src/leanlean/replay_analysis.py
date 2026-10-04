"""Compact, paper-ready analysis artifacts from exact offline replay results."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EDIT_BUILD_ANALYSIS_FORMAT = "leanlean-edit-build-analysis-v1"
_REPLAY_RESULT_FORMAT = "code-harness-standardized-replay-result-v1"
_CSV_FIELDS = (
    "instance_id",
    "provider",
    "session_id",
    "edit_index",
    "action_index",
    "action_name",
    "action_kind",
    "agent_id",
    "tool_call_id",
    "provider_call_id",
    "lean_tokens",
    "lean_tokens_saved",
    "lean_token_compression_pct",
    "build_status",
    "build_passed",
    "build_returncode",
    "build_timed_out",
    "build_duration_seconds",
    "build_source_tree_unchanged",
    "cost_usd",
    "marginal_cost_since_previous_edit_usd",
    "source_archive_sha256",
    "incremental_diff_sha256",
    "changed_files",
)


def _build_status(point: Mapping[str, Any]) -> str:
    if int(point.get("edit_index") or 0) == 0:
        return "baseline"
    build = point.get("build")
    if not isinstance(build, Mapping):
        return "not_built"
    return "passed" if build.get("passed") is True else "failed"


def replay_edit_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project a full replay result onto one compact row per edit boundary."""

    if result.get("format") != _REPLAY_RESULT_FORMAT:
        raise ValueError("expected a standardized replay result")
    points = result.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("standardized replay result has no points")

    rows: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        # An unattributed terminal change is retained in the full replay result
        # as an exactness failure, but it is not an observed edit boundary and
        # must not silently become a paper datapoint.
        if point.get("kind") == "unattributed_action_endpoint_change":
            continue
        build = point.get("build")
        build = build if isinstance(build, Mapping) else {}
        incremental = point.get("incremental_diff_artifact")
        incremental = incremental if isinstance(incremental, Mapping) else {}
        changed_files = point.get("changed_files", point.get("files"))
        if not isinstance(changed_files, list):
            changed_files = []
        rows.append(
            {
                "instance_id": str(result.get("instance_id") or ""),
                "provider": str(result.get("provider") or ""),
                "session_id": str(result.get("session_id") or ""),
                "edit_index": int(point.get("edit_index") or 0),
                "action_index": int(point.get("action_index") or 0),
                "action_name": str(point.get("action_name") or ""),
                "action_kind": str(point.get("kind") or ""),
                "agent_id": str(point.get("agent_id") or ""),
                "tool_call_id": str(point.get("tool_call_id") or ""),
                "provider_call_id": str(point.get("provider_call_id") or ""),
                "lean_tokens": int(point.get("lean_tokens") or 0),
                "lean_tokens_saved": int(point.get("lean_tokens_saved") or 0),
                "lean_token_compression_pct": float(
                    point.get("lean_token_compression_pct") or 0.0
                ),
                "build_status": _build_status(point),
                "build_passed": (
                    build.get("passed") if "passed" in build else None
                ),
                "build_returncode": build.get("returncode"),
                "build_timed_out": (
                    bool(build.get("timed_out")) if build else None
                ),
                "build_duration_seconds": (
                    float(build.get("duration_seconds") or 0.0)
                    if build
                    else None
                ),
                "build_source_tree_unchanged": build.get(
                    "source_tree_unchanged"
                ),
                "cost_usd": point.get("cost_usd"),
                "marginal_cost_since_previous_edit_usd": point.get(
                    "marginal_cost_since_previous_edit_usd"
                ),
                "source_archive_sha256": str(
                    point.get("source_archive_sha256") or ""
                ),
                "incremental_diff_sha256": str(incremental.get("sha256") or ""),
                "changed_files": [str(path) for path in changed_files],
            }
        )

    if not rows or rows[0]["edit_index"] != 0:
        raise ValueError("standardized replay points have no baseline")
    edit_indexes = [row["edit_index"] for row in rows[1:]]
    if edit_indexes != list(range(1, len(edit_indexes) + 1)):
        raise ValueError("standardized replay edit indexes are not contiguous")
    return rows


def build_edit_build_analysis(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact analysis document without weakening replay claims."""

    rows = replay_edit_rows(result)
    edits = rows[1:]
    status_counts = {
        status: sum(row["build_status"] == status for row in edits)
        for status in ("passed", "failed", "not_built")
    }
    return {
        "format": EDIT_BUILD_ANALYSIS_FORMAT,
        "instance_id": result.get("instance_id"),
        "provider": result.get("provider"),
        "session_id": result.get("session_id"),
        "source_replay_format": result.get("format"),
        "standardized_session_replay_exact": result.get(
            "standardized_session_replay_exact"
        )
        is True,
        "trajectory_action_replay_exact": result.get(
            "trajectory_action_replay_exact"
        )
        is True,
        "subagent_lineage_complete": result.get("subagent_lineage_complete")
        is True,
        "metric_basis": result.get("metric_basis"),
        "build_basis": result.get("build_basis"),
        "build_cache_basis": result.get("build_cache_basis"),
        "edit_boundary": (
            "one successful provider mutation action that changes the scoped "
            "source tree"
        ),
        "summary": {
            "edit_count": len(edits),
            "build_attempt_count": status_counts["passed"]
            + status_counts["failed"],
            "build_pass_count": status_counts["passed"],
            "build_failure_count": status_counts["failed"],
            "not_built_count": status_counts["not_built"],
            "final_lean_token_compression_pct": rows[-1][
                "lean_token_compression_pct"
            ],
        },
        "points": rows,
    }


def compression_build_vega_spec(
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a self-contained Vega-Lite graph specification."""

    if analysis.get("format") != EDIT_BUILD_ANALYSIS_FORMAT:
        raise ValueError("expected an edit/build analysis document")
    values = []
    for row in analysis.get("points") or []:
        if not isinstance(row, Mapping):
            continue
        values.append(
            {
                "edit": int(row.get("edit_index") or 0),
                "compression_pct": float(
                    row.get("lean_token_compression_pct") or 0.0
                ),
                "build": str(row.get("build_status") or "not_built"),
                "returncode": row.get("build_returncode"),
                "duration_seconds": row.get("build_duration_seconds"),
                "agent_id": str(row.get("agent_id") or ""),
                "action": str(
                    row.get("action_name") or row.get("action_kind") or ""
                ),
                "files": ", ".join(
                    str(path) for path in row.get("changed_files") or []
                ),
            }
        )
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": (
            "Lean-token compression after each exactly replayed source mutation, "
            "colored by the isolated checkpoint build result."
        ),
        "title": (
            "Compression and build status — "
            f"{analysis.get('instance_id') or 'trajectory'}"
        ),
        "width": "container",
        "height": 360,
        "data": {"values": values},
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "color": "#5f6b7a",
                    "strokeWidth": 1.5,
                },
                "encoding": {
                    "x": {
                        "field": "edit",
                        "type": "quantitative",
                        "title": "Replayed edit index",
                    },
                    "y": {
                        "field": "compression_pct",
                        "type": "quantitative",
                        "title": "Lean-token compression (%)",
                    },
                    "order": {"field": "edit", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.edit > 0"}],
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 70,
                    "stroke": "white",
                    "strokeWidth": 0.6,
                },
                "encoding": {
                    "x": {"field": "edit", "type": "quantitative"},
                    "y": {"field": "compression_pct", "type": "quantitative"},
                    "color": {
                        "field": "build",
                        "type": "nominal",
                        "title": "Checkpoint build",
                        "scale": {
                            "domain": ["passed", "failed", "not_built"],
                            "range": ["#2ca02c", "#d62728", "#9aa0a6"],
                        },
                    },
                    "tooltip": [
                        {
                            "field": "edit",
                            "type": "quantitative",
                            "title": "Edit",
                        },
                        {
                            "field": "compression_pct",
                            "type": "quantitative",
                            "title": "Compression (%)",
                            "format": ".4f",
                        },
                        {"field": "build", "type": "nominal", "title": "Build"},
                        {
                            "field": "returncode",
                            "type": "quantitative",
                            "title": "Return code",
                        },
                        {
                            "field": "duration_seconds",
                            "type": "quantitative",
                            "title": "Build time (s)",
                            "format": ".3f",
                        },
                        {
                            "field": "agent_id",
                            "type": "nominal",
                            "title": "Agent",
                        },
                        {"field": "action", "type": "nominal", "title": "Action"},
                        {"field": "files", "type": "nominal", "title": "Files"},
                    ],
                },
            },
        ],
        "resolve": {"scale": {"color": "independent"}},
    }


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def write_edit_build_analysis(
    result: Mapping[str, Any], destination: str | os.PathLike[str]
) -> dict[str, dict[str, Any]]:
    """Write compact JSON, CSV, and Vega-Lite artifacts atomically."""

    root = Path(destination).expanduser().resolve(strict=False)
    analysis = build_edit_build_analysis(result)
    paths = {
        "analysis": root / "edit-build-analysis.json",
        "points_csv": root / "edit-build-points.csv",
        "chart": root / "compression-build-chart.vl.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite replay analysis artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    _write_atomic(paths["analysis"], _canonical_json(analysis))
    _write_atomic(
        paths["chart"],
        _canonical_json(compression_build_vega_spec(analysis)),
    )

    csv_path = paths["points_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for row in analysis["points"]:
                csv_row = {field: row.get(field) for field in _CSV_FIELDS}
                csv_row["changed_files"] = json.dumps(
                    csv_row["changed_files"], ensure_ascii=False
                )
                writer.writerow(csv_row)
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)

    return {
        name: {
            **_artifact(path),
            "format": {
                "analysis": EDIT_BUILD_ANALYSIS_FORMAT,
                "points_csv": "text/csv",
                "chart": "vega-lite-v5",
            }[name],
        }
        for name, path in paths.items()
    }


__all__ = [
    "EDIT_BUILD_ANALYSIS_FORMAT",
    "build_edit_build_analysis",
    "compression_build_vega_spec",
    "replay_edit_rows",
    "write_edit_build_analysis",
]
