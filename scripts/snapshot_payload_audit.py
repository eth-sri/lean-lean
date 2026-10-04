#!/usr/bin/env python3
"""Materialize redacted request and native-instruction payload audit snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _calls(value: Any):
    if isinstance(value, dict):
        envelope = value.get("request_envelope")
        if isinstance(envelope, dict):
            yield envelope, value.get("request")
        for child in value.values():
            yield from _calls(child)
    elif isinstance(value, list):
        for child in value:
            yield from _calls(child)


def _load_records(path: Path) -> list[Any]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [json.loads(path.read_text(encoding="utf-8"))]


def _native_instruction_fragments(
    envelope: dict[str, Any], request: Any
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    extras = envelope.get("extras")
    if isinstance(extras, dict):
        for key in ("system", "systemInstruction", "instructions"):
            if key in extras:
                fragments.append(
                    {
                        "source": f"request_envelope.extras.{key}",
                        "value": extras[key],
                    }
                )
    if isinstance(request, list):
        for index, item in enumerate(request):
            if isinstance(item, dict) and item.get("role") in {
                "system",
                "developer",
            }:
                fragments.append(
                    {
                        "source": f"request.{index}",
                        "value": item,
                    }
                )
    return fragments


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8", errors="surrogateescape")


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return str(tool.get("name") or "")


def _payload_bloat_audit(
    envelope: dict[str, Any], request: Any
) -> dict[str, Any]:
    """Measure the behavior-bearing request surface without heuristic tokens."""

    extras = envelope.get("extras")
    extras = dict(extras) if isinstance(extras, dict) else {}
    input_field = envelope.get("input_field")
    reconstructed = dict(extras)
    if isinstance(input_field, str):
        reconstructed[input_field] = request
    tools = extras.get("tools")
    tools = tools if isinstance(tools, list) else []
    fragments = _native_instruction_fragments(envelope, request)
    return {
        "request_bytes": len(_canonical_json_bytes(reconstructed)),
        "native_instruction_bytes": len(_canonical_json_bytes(fragments)),
        "tool_schema_bytes": len(_canonical_json_bytes(tools)),
        "tool_count": len(tools),
        "tool_names": [_tool_name(tool) for tool in tools],
    }


def _enforce_payload_guard(
    audit: dict[str, Any], guard: Any, *, config: str
) -> None:
    if not isinstance(guard, dict):
        raise RuntimeError(f"{config}: payload_guard must be a mapping")
    expected = {
        "exact_tools",
        "max_request_bytes",
        "max_native_instruction_bytes",
        "max_tool_schema_bytes",
    }
    if set(guard) != expected:
        raise RuntimeError(f"{config}: payload_guard must have exactly {sorted(expected)}")
    if audit["tool_names"] != guard["exact_tools"]:
        raise RuntimeError(
            f"{config}: tool payload changed: {audit['tool_names']!r}"
        )
    for metric in (
        "request_bytes",
        "native_instruction_bytes",
        "tool_schema_bytes",
    ):
        maximum = guard[f"max_{metric}"]
        if not isinstance(maximum, int) or maximum < 1:
            raise RuntimeError(f"{config}: max_{metric} must be a positive integer")
        if audit[metric] > maximum:
            raise RuntimeError(
                f"{config}: {metric} grew to {audit[metric]} bytes "
                f"(limit {maximum})"
            )


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8", errors="surrogateescape")
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.resolve().parents[2]
    for entry in manifest["models"]:
        source = root / entry["source"]
        candidates = [
            (envelope, request)
            for record in _load_records(source)
            for envelope, request in _calls(record)
            if envelope.get("extras", {}).get("model") == entry["request_model"]
            and envelope.get("sha256") == entry["request_sha256"]
        ]
        if not candidates:
            raise RuntimeError(
                f"{entry['config']}: no matching {entry['request_model']} request "
                f"in {source}"
            )
        envelope, request = candidates[-1]
        if "source_capture" in entry:
            source_metadata = {"source_capture": entry["source_capture"]}
        else:
            source_metadata = {"source_run_manifest": entry["run_manifest"]}
        output = root / entry["snapshot"]
        _write_json(
            output,
            {
                "kind": "leanlean_native_request_snapshot",
                "schema_version": 1,
                "model_config": entry["config"],
                "harness_version": entry["harness_version"],
                **source_metadata,
                "request_envelope": envelope,
            },
        )

        fragments = _native_instruction_fragments(envelope, request)
        if not fragments:
            raise RuntimeError(
                f"{entry['config']}: no native system/developer instructions found"
            )
        payload_audit = _payload_bloat_audit(envelope, request)
        if "payload_guard" in entry:
            _enforce_payload_guard(
                payload_audit,
                entry["payload_guard"],
                config=entry["config"],
            )
        instructions_sha = _json_sha256(fragments)
        expected_sha = entry.get("native_instructions_sha256")
        if expected_sha is not None and instructions_sha != expected_sha:
            raise RuntimeError(
                f"{entry['config']}: native instruction SHA-256 changed"
            )
        prompt_output = root / entry["prompt_snapshot"]
        _write_json(
            prompt_output,
            {
                "kind": "leanlean_native_instruction_snapshot",
                "schema_version": 1,
                "model_config": entry["config"],
                "harness_version": entry["harness_version"],
                "request_sha256": entry["request_sha256"],
                **source_metadata,
                "native_instructions_sha256": instructions_sha,
                "payload_audit": payload_audit,
                "fragments": fragments,
            },
        )
        print(output.relative_to(root))
        print(f"{prompt_output.relative_to(root)} {instructions_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
