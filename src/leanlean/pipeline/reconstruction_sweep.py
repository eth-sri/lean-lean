"""Validate and compare size-stratified theorem reconstruction experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from leanlean.pipeline.reconstruction import (
    REPO_ROOT,
    _atomic_text,
    _load_manifest,
    _mapping,
    _path,
    _text,
)


KIND = "leanlean_theorem_reconstruction_sweep"
SCHEMA_VERSION = 1


def load_sweep_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    sweep = dict(_mapping(value, "sweep manifest"))
    expected = {
        "kind",
        "schema_version",
        "run_id",
        "repository",
        "model",
        "reasoning_effort",
        "generator",
        "rounds",
        "runs",
        "execution",
        "container",
        "timeout",
        "evaluation",
        "outputs",
    }
    if set(sweep) != expected:
        raise ValueError(
            "sweep manifest fields differ: "
            f"missing={sorted(expected - set(sweep))}, "
            f"unknown={sorted(set(sweep) - expected)}"
        )
    if sweep.get("kind") != KIND or sweep.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported theorem reconstruction sweep manifest")
    if sweep.get("model") != "gpt-5.6-sol" or sweep.get("reasoning_effort") != "xhigh":
        raise ValueError("sweep must pin gpt-5.6-sol at xhigh")
    if sweep.get("generator") != "codex_sub" or sweep.get("rounds") != 1:
        raise ValueError("sweep requires codex_sub and one round")
    execution = _mapping(sweep["execution"], "execution")
    execution_fields = {
        "order", "max_concurrent_runs", "reuse_published_preprocessing"
    }
    order = execution.get("order")
    max_concurrent = execution.get("max_concurrent_runs")
    if (
        not {"order", "max_concurrent_runs"}.issubset(execution)
        or set(execution) - execution_fields
        or order
        not in {
            "preprocess_all_then_run_sequentially",
            "preprocess_all_then_run_parallel_by_stage",
        }
        or isinstance(max_concurrent, bool)
        or not isinstance(max_concurrent, int)
        or max_concurrent < 1
        or (
            order == "preprocess_all_then_run_sequentially"
            and max_concurrent != 1
        )
        or not isinstance(
            execution.get("reuse_published_preprocessing", False), bool
        )
    ):
        raise ValueError("sweep execution policy is invalid")
    rows = sweep.get("runs")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("sweep.runs must contain at least two size bands")
    seen_names: set[str] = set()
    seen_manifests: set[Path] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"runs[{index}]")
        if set(row) != {
            "name",
            "manifest",
            "minimum_removed_fraction",
            "maximum_removed_fraction",
        }:
            raise ValueError(f"runs[{index}] must pin a manifest and size band")
        name = _text(row.get("name"), f"runs[{index}].name")
        manifest_path = _path(row.get("manifest"), f"runs[{index}].manifest")
        if name in seen_names or manifest_path in seen_manifests:
            raise ValueError("sweep run names and manifests must be unique")
        seen_names.add(name)
        seen_manifests.add(manifest_path)
        lower = row.get("minimum_removed_fraction")
        upper = row.get("maximum_removed_fraction")
        if (
            isinstance(lower, bool)
            or not isinstance(lower, (int, float))
            or isinstance(upper, bool)
            or not isinstance(upper, (int, float))
            or not 0 <= lower <= upper <= 1
        ):
            raise ValueError(f"runs[{index}] has an invalid size band")
        manifest = _load_manifest(manifest_path)
        if _mapping(manifest["preprocessing"], "preprocessing").get("repository") != sweep["repository"]:
            raise ValueError("every sweep run must use the pinned repository")
        for field in (
            "model", "reasoning_effort", "generator", "rounds", "timeout"
        ):
            if manifest.get(field) != sweep.get(field):
                raise ValueError(f"sweep run disagrees on {field}")
        run_container = _mapping(manifest["container"], "run container")
        sweep_container = _mapping(sweep["container"], "sweep container")
        if set(run_container) != set(sweep_container):
            raise ValueError("sweep and run container fields differ")
        for field in set(sweep_container) - {"max_total_memory"}:
            if run_container[field] != sweep_container[field]:
                raise ValueError(f"sweep run disagrees on container.{field}")
    _path(_mapping(sweep["outputs"], "outputs").get("summary"), "outputs.summary")
    return sweep


def resolved_runs(sweep: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Path, dict[str, Any]]]:
    result = []
    for raw in sweep["runs"]:
        row = _mapping(raw, "sweep run")
        manifest_path = _path(row.get("manifest"), "sweep run manifest")
        result.append((row, manifest_path, _load_manifest(manifest_path)))
    return result


def validate_size_bands(sweep: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    last_fraction = -1.0
    for band, _, manifest in resolved_runs(sweep):
        output = _mapping(manifest["outputs"], "run outputs")
        report_path = _path(output.get("ablation_report"), "run ablation report")
        report = _mapping(json.loads(report_path.read_text()), "run ablation report")
        fraction = report.get("removed_fraction")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise ValueError("run ablation report has no removed_fraction")
        lower = float(band["minimum_removed_fraction"])
        upper = float(band["maximum_removed_fraction"])
        if report.get("passed") is not True or not lower <= fraction <= upper:
            raise RuntimeError(
                f"{band['name']} ablation {fraction:.2%} is outside "
                f"[{lower:.2%}, {upper:.2%}]"
            )
        if fraction <= last_fraction:
            raise RuntimeError("sweep ablation sizes are not strictly increasing")
        last_fraction = fraction
        rows.append(
            {
                "name": band["name"],
                "theorem": _mapping(manifest["theorem"], "theorem")["declaration"],
                "removed_lean_tokens": report["removed_lean_tokens"],
                "removed_fraction": fraction,
                "band": [lower, upper],
            }
        )
    return {"passed": True, "runs": rows}


def compare_sweep(sweep: Mapping[str, Any]) -> dict[str, Any]:
    size_validation = validate_size_bands(sweep)
    rows = []
    for band, _, manifest in resolved_runs(sweep):
        outputs = _mapping(manifest["outputs"], "run outputs")
        ablation = _mapping(
            json.loads(_path(outputs["ablation_report"], "ablation report").read_text()),
            "ablation report",
        )
        paired = _mapping(
            json.loads(_path(outputs["summary"], "reconstruction summary").read_text()),
            "reconstruction summary",
        )
        arms = _mapping(paired.get("arms"), "reconstruction arms")
        rows.append(
            {
                "name": band["name"],
                "run_id": manifest["run_id"],
                "theorem": paired["theorem"],
                "removed_lean_tokens": ablation["removed_lean_tokens"],
                "removed_fraction": ablation["removed_fraction"],
                "stripped": arms["stripped"],
                "compressed": arms["compressed"],
                "paired_delta_compressed_minus_stripped": paired[
                    "paired_delta_compressed_minus_stripped"
                ],
            }
        )
    result = {
        "kind": "leanlean_theorem_reconstruction_sweep_summary",
        "schema_version": 1,
        "run_id": sweep["run_id"],
        "repository": sweep["repository"],
        "model": sweep["model"],
        "reasoning_effort": sweep["reasoning_effort"],
        "size_validation": size_validation,
        "runs": rows,
    }
    output_path = _path(
        _mapping(sweep["outputs"], "outputs").get("summary"), "outputs.summary"
    )
    _atomic_text(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
