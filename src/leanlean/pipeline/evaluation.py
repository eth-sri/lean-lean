"""Resolve concise evaluation configs into immutable experiment manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from leanlean.dataset_bundle import load_dataset
from leanlean.preprocessing.repositories import file_sha256
from leanlean.preprocessing.standardized_repositories import (
    source_tree_sha256,
)
from leanlean.environment_archives import validated_archive
from leanlean.shared_cache import artifact_tree_sha256
from leanlean.subscription_queue import (
    ANTHROPIC_USAGE_ENDPOINT,
    ANTHROPIC_USAGE_WINDOWS,
    capacity_probe_for_harness,
)


CONFIG_KIND = "leanlean_evaluation"
DATASET_CONFIG_KIND = "leanlean_evaluation_dataset"
MODEL_CONFIG_KIND = "leanlean_evaluation_model"
MANIFEST_KIND = "leanlean_evaluation_run"
RUN_ARTIFACT_KIND = "leanlean_run"
SCHEMA_VERSION = 1

LEANLEAN_20260914_DATASET = "datasets/leanlean_20260914/dataset.yaml"
LEANLEAN_20260914_CANONICAL_CONFIG = "configs/dataset/leanlean_20260914.yaml"
LEANLEAN_20260914_PROMPT_SHA256 = (
    "9e8b045d494936aba649587e30cff40f80fdaaad254b759a7c1f9e9af008e111"
)

SUPPORTED_HARNESSES = {
    "muse_code_native",
    "muse_code_core",
    "codex_sub",
    "codex_sub_mcp",
    "claude_code_sub",
    "claude_code_glm",
    "mistral_vibe_lean",
    "antigravity_cli",
}
CHECKPOINT_HARNESSES = {
    "muse_code_native",
    "muse_code_core",
    "codex_sub",
    "codex_sub_mcp",
    "claude_code_sub",
    "claude_code_glm",
    "mistral_vibe_lean",
    "antigravity_cli",
}
REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max", "ultra"}
SUBSCRIPTION_HARNESSES = {
    "codex_sub",
    "codex_sub_mcp",
    "claude_code_sub",
}

IMPLEMENTATION_FILES = (
    "src/leanlean/model/litellm_wrapper/accounting.py",
    "src/leanlean/model/litellm_wrapper/stream_usage.py",
    "src/leanlean/model/litellm_wrapper/raw_provider_stream.py",
    "src/leanlean/model/litellm_wrapper/checkpoint_costs.py",
    "src/leanlean/model/litellm_wrapper/litellm_server.py",
    "src/leanlean/model/litellm_wrapper/request_rewrite.py",
    "src/leanlean/model/litellm_wrapper/mistral_reasoning.py",
    "src/leanlean/model/litellm_wrapper/litellm_logger.py",
    "src/leanlean/model/costs.py",
    "src/leanlean/utils/trace.py",
    "src/leanlean/subscription_auth.py",
    "src/leanlean/holdout_verifier.py",
    "src/leanlean/holdout_comparator.py",
    "src/leanlean/pipeline/reconstruction_inputs.py",
    "src/leanlean/pipeline/reconstruction_resume.py",
    "src/leanlean/pipeline/reconstruction_scoring.py",
    "src/leanlean/pipeline/reconstruction_repairs.py",
    "src/leanlean/preprocessing/cache_isolation.py",
    "eval.py",
    "eval.sh",
    "resume_eval.sh",
    "scripts/eval.py",
    "scripts/resume_eval.py",
    "scripts/run_codex_sub_yaml.py",
    "scripts/reconstruct.py",
    "scripts/run_theorem_reconstruction_experiment.py",
    "scripts/run_theorem_reconstruction_sweep.py",
    "scripts/install_codex_standalone.sh",
    "scripts/install_claude_standalone.sh",
    "scripts/install_mistral_vibe_standalone.sh",
    "configs/harnesses/agent-products-20260913.yaml",
    "scripts/snapshot_payload_audit.py",
    "scripts/install_antigravity_standalone.sh",
    "scripts/install_muse_standalone.py",
    "src/leanlean/generators/muse_code.py",
    "scripts/generate.py",
    "src/leanlean/run_admission.py",
    "scripts/lean_verify",
    "scripts/build_model_relay_image.sh",
    "docker/evaluation/Dockerfile",
    "docker/evaluation/Dockerfile.shared-cache",
    "docker/leanlean/Dockerfile",
    "src/configs/benchmark_constants.py",
    "src/leanlean/evaluation_images.py",
    "src/leanlean/shared_cache.py",
    "src/leanlean/environment_archives.py",
    "src/leanlean/capture_disposition.py",
    "src/leanlean/environments/docker.py",
    "src/leanlean/environments/resource_view.py",
    "src/configs/generator_constants.py",
    "src/leanlean/pipeline/reconstruction.py",
    "src/leanlean/preprocessing/theorem_holdout.py",
    "src/leanlean/pipeline/reconstruction_sweep.py",
    "src/configs/model_constants.py",
    "src/leanlean/pipeline/evaluation.py",
    "src/leanlean/dataset_bundle.py",
    "src/leanlean/benchmarks/leanlean.py",
    "src/leanlean/preprocessing/olean.py",
    "src/leanlean/metrics/tokens.py",
    "src/leanlean/palomar_comparator.py",
    "src/leanlean/action_trace.py",
    "src/leanlean/run_action_summary.py",
    "src/leanlean/generators/cli_agent.py",
    "src/leanlean/host_side_checkpointing.py",
    "src/leanlean/playback.py",
    "src/leanlean/standardized_trace.py",
    "src/leanlean/mistral_vibe_trace.py",
    "src/leanlean/deepseek_harness_trace.py",
    "src/leanlean/antigravity_trace.py",
    "src/leanlean/kimi_code_trace.py",
    "src/leanlean/subscription_queue.py",
)


@dataclass(frozen=True)
class ResolvedEvaluation:
    config_path: Path
    config: Mapping[str, Any]
    dataset_path: Path
    dataset: Mapping[str, Any]
    manifest: dict[str, Any]
    engine_manifest: dict[str, Any]
    manifest_path: Path
    engine_manifest_path: Path
    run_dir: Path
    run_artifact_path: Path
    config_snapshot_path: Path
    output_dir: Path
    tmux_session: str
    config_snapshot_content: str | None = None
    repo_root: Path | None = None


@dataclass(frozen=True)
class PreparedEvaluation:
    manifest_path: Path
    manifest: Mapping[str, Any]
    engine_manifest_path: Path
    run_artifact_path: Path
    output_dir: Path
    tmux_session: str


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_keys(
    row: Mapping[str, Any],
    expected: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    unknown = set(row) - expected
    missing = expected - set(optional or ()) - set(row)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError(f"{name} has " + "; ".join(details))


def _text(row: Mapping[str, Any], key: str, name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value


def _integer(row: Mapping[str, Any], key: str, name: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{name}.{key} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.{key} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name}.{key} must be a positive integer")
    return result


def _nonnegative_integer(row: Mapping[str, Any], key: str, name: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{name}.{key} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.{key} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{name}.{key} must be a non-negative integer")
    return result


def _number(row: Mapping[str, Any], key: str, name: str) -> float:
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{name}.{key} must be non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}.{key} must be non-negative") from error
    if result < 0:
        raise ValueError(f"{name}.{key} must be non-negative")
    return result


def _repo_path(root: Path, value: str, name: str) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{name} must stay inside the repository")
    return path


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _pinned_reconciliation(
    root: Path, record: Any, instance_id: str
) -> dict[str, Any]:
    """Validate the two seed patches that build the task's refactor branches.

    `main` becomes the checked-out starting state and `other-refactor` the
    alternative; both are rebuilt inside the image because the materialization
    Dockerfile discards any `.git` shipped in the source tree.
    """

    seeds = _mapping(record, f"{instance_id}.reconciliation")
    pinned: dict[str, Any] = {}
    for role in ("main", "other"):
        seed = _mapping(seeds.get(role), f"{instance_id}.reconciliation.{role}")
        patch_path = _repo_path(
            root,
            _text(seed, "path", f"{instance_id}.reconciliation.{role}"),
            f"{instance_id}.reconciliation.{role}.path",
        )
        digest = _text(seed, "sha256", f"{instance_id}.reconciliation.{role}")
        if not patch_path.is_relative_to((root / "datasets").resolve()):
            raise ValueError(
                f"{instance_id}: reconciliation {role} patch must live under datasets/"
            )
        if not patch_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{instance_id}: reconciliation {role} patch is missing")
        if file_sha256(patch_path) != digest:
            raise ValueError(f"{instance_id}: reconciliation {role} patch drift")
        pinned[role] = {
            "path": _relative(root, patch_path),
            "sha256": digest,
            "source_run": str(seed.get("source_run") or ""),
            "source_model": str(seed.get("source_model") or ""),
        }
    if pinned["main"]["sha256"] == pinned["other"]["sha256"]:
        raise ValueError(f"{instance_id}: both reconciliation branches use one patch")
    return pinned


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return _relative(root, path)
    except ValueError:
        return str(path.resolve())


def _dump(value: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(value), sort_keys=False, width=100)


def _load_yaml_mapping(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    return _mapping(value, name)


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8", errors="surrogateescape")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_native_system_prompt(
    value: Any,
    *,
    root: Path,
    model_path: Path,
    harness_version: str,
    request_sha256: str,
) -> dict[str, str]:
    reference = _mapping(value, "model config.system_prompt")
    _exact_keys(
        reference,
        {"mode", "snapshot", "sha256"},
        "model config.system_prompt",
    )
    if _text(reference, "mode", "model config.system_prompt") != "native":
        raise ValueError("model config.system_prompt.mode must be 'native'")
    instructions_sha = _text(reference, "sha256", "model config.system_prompt")
    if not re.fullmatch(r"[0-9a-f]{64}", instructions_sha):
        raise ValueError("model config.system_prompt.sha256 must be SHA-256")

    snapshot_path = _repo_path(
        root,
        _text(reference, "snapshot", "model config.system_prompt"),
        "model config.system_prompt.snapshot",
    )
    if not snapshot_path.is_file():
        raise ValueError(
            f"model config.system_prompt.snapshot is missing: {snapshot_path}"
        )
    snapshot = _load_yaml_mapping(
        snapshot_path,
        "native instruction snapshot",
    )
    _exact_keys(
        snapshot,
        {
            "kind",
            "schema_version",
            "model_config",
            "harness_version",
            "request_sha256",
            "source_run_manifest",
            "source_capture",
            "native_instructions_sha256",
            "payload_audit",
            "fragments",
        },
        "native instruction snapshot",
        optional={"source_run_manifest", "source_capture", "payload_audit"},
    )
    provenance = {"source_run_manifest", "source_capture"} & set(snapshot)
    if len(provenance) != 1:
        raise ValueError(
            f"{snapshot_path}: native instruction snapshot must carry exactly one "
            "of source_run_manifest, source_capture"
        )
    if (
        snapshot.get("kind") != "leanlean_native_instruction_snapshot"
        or snapshot.get("schema_version") != 1
    ):
        raise ValueError(f"{snapshot_path}: unsupported native instruction snapshot")
    if snapshot.get("model_config") != _relative(root, model_path):
        raise ValueError(f"{snapshot_path}: model config reference mismatch")
    if snapshot.get("harness_version") != harness_version:
        raise ValueError(f"{snapshot_path}: harness version mismatch")
    if snapshot.get("request_sha256") != request_sha256:
        raise ValueError(f"{snapshot_path}: request SHA-256 mismatch")
    if snapshot.get("native_instructions_sha256") != instructions_sha:
        raise ValueError(f"{snapshot_path}: native instruction SHA-256 mismatch")
    fragments = snapshot.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError(f"{snapshot_path}: native instruction fragments are missing")
    if _json_sha256(fragments) != instructions_sha:
        raise ValueError(f"{snapshot_path}: native instruction content changed")
    return {
        "mode": "native",
        "snapshot": _relative(root, snapshot_path),
        "sha256": instructions_sha,
    }


def _named_config_path(
    reference: str | Path,
    *,
    repo_root: Path,
    directory: str,
    name: str,
) -> tuple[Path, str]:
    root = repo_root.resolve()
    base = (root / "configs" / directory).resolve()
    key = str(reference)
    parts = key.split("/")
    if (
        not key
        or key.endswith((".yaml", ".yml"))
        or parts[0] == "configs"
        or any(
            part in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part) is None
            for part in parts
        )
    ):
        raise ValueError(
            f"{name} must be an extensionless config name under configs/{directory}/, "
            "not a filesystem path"
        )
    path = (base / f"{key}.yaml").resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"{name} config must live under configs/{directory}/")
    if not path.is_file():
        raise ValueError(f"{name} config not found: {path}")
    key = str(path.relative_to(base).with_suffix(""))
    return path, key


def _derived_run_id(dataset_key: str, model_key: str) -> str:
    raw = f"{dataset_key}__{model_key}".replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", raw).strip("-.")
    if not safe:
        raise ValueError("dataset and model config paths produce an empty run key")
    if len(safe) <= 160:
        return safe
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{safe[:143]}-{digest}"


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _whole_gib(value: str, name: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)g", value.lower())
    if not match:
        raise ValueError(f"{name} must be a whole-GiB value such as 64g")
    return int(match.group(1))


def _select_repositories(
    rows: list[Mapping[str, Any]], requested: Any
) -> list[Mapping[str, Any]]:
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("dataset repository IDs are empty or duplicated")
    if requested == "all":
        return rows
    if not isinstance(requested, list) or not requested:
        raise ValueError("repositories must be 'all' or an ordered ID list")
    if (any(not isinstance(name, str) or name not in ids for name in requested)
            or len(requested) != len(set(requested))):
        raise ValueError("repositories are unknown or duplicated")
    by_id = dict(zip(ids, rows, strict=True))
    return [by_id[name] for name in requested]


# Pinned harness versions a model config may select instead of the generator's
# default host_tool_version (the runner installs the config's version). Opus 5.5
# needs Claude Code 2.1.280; GPT-6.1 Sol runs on Codex 0.160.0, the first pinned
# release whose bundled catalog has gpt-6.1-sol.
HARNESS_VERSION_OVERRIDES = {
    "claude_code_sub": frozenset({"2.1.280"}),
    "codex_sub": frozenset({"0.160.0"}),
}

# Claude Code 2.1.280's --tools names that a run may enable (web, scheduling,
# cloud, worktree and UI-only tools are excluded).
CLAUDE_CODE_SUBAGENT_TOOLS = frozenset({"Task", "Workflow", "SendMessage", "ListAgents"})
CLAUDE_CODE_NATIVE_TOOLS = CLAUDE_CODE_SUBAGENT_TOOLS | frozenset({
    "Bash", "Edit", "Read", "Write", "Monitor", "ToolSearch",
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop",
})


def _codex_model_catalog(value: Any, *, root: Path, harness: str) -> dict[str, str]:
    """Validate a pinned Codex model catalog passed as model_catalog_json."""

    catalog = _mapping(value, "codex_model_catalog")
    _exact_keys(catalog, {"path", "sha256"}, "codex_model_catalog")
    if harness != "codex_sub":
        raise ValueError("codex_model_catalog requires the codex_sub harness")
    path = _text(catalog, "path", "codex_model_catalog")
    digest = _text(catalog, "sha256", "codex_model_catalog")
    source = (root / path).resolve()
    if not source.is_file() or not source.is_relative_to(root.resolve()):
        raise ValueError(f"codex_model_catalog is not a repository file: {path}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path}: codex_model_catalog sha256 mismatch")
    return {"path": path, "sha256": digest}


def _resolve_agent_contract(value: Any, *, context: str) -> dict[str, Any]:
    contract = _mapping(value, context)
    _exact_keys(
        contract,
        {
            "system_prompt",
            "prompt",
            "allowed_skills",
            "mcp_servers",
            "enable_subagents",
            "enable_lean_verify",
            "enable_proof_length",
            "native_tools",
        },
        context,
        optional={"native_tools"},
    )
    resolved = {
        "system_prompt": _text(contract, "system_prompt", context),
        "prompt": _text(contract, "prompt", context),
    }
    if resolved["system_prompt"] != "native":
        raise ValueError(
            f"{context}.system_prompt must be 'native'; custom prompt "
            "replacement is not part of the agent-product benchmark"
        )
    for key in ("allowed_skills", "mcp_servers"):
        entries = contract.get(key)
        if (
            not isinstance(entries, list)
            or any(not isinstance(item, str) or not item.strip() for item in entries)
            or len(entries) != len(set(entries))
        ):
            raise ValueError(f"{context}.{key} must be an array of unique strings")
        resolved[key] = list(entries)
    for key in (
        "enable_subagents",
        "enable_lean_verify",
        "enable_proof_length",
    ):
        enabled = contract.get(key)
        if not isinstance(enabled, bool):
            raise ValueError(f"{context}.{key} must be true or false")
        resolved[key] = enabled
    if resolved["allowed_skills"]:
        raise ValueError(
            f"{context}.allowed_skills must remain empty until skills are pinned"
        )
    if resolved["mcp_servers"]:
        raise ValueError(
            f"{context}.mcp_servers must remain empty until MCP servers are pinned"
        )
    prompt = resolved["prompt"]
    for key, marker in (
        ("enable_lean_verify", "lean_verify"),
        ("enable_proof_length", "proof_length.py"),
    ):
        if marker in prompt and not resolved[key]:
            raise ValueError(
                f"{context}.prompt names unavailable tool {marker}: "
                f"{context}.{key} is false"
            )
    if "native_tools" in contract:
        # Claude Code's --tools list; absent means the harness default
        # (Bash,Edit,Read,Write). Web tools stay off: the task is offline.
        tools = contract["native_tools"]
        if (
            not isinstance(tools, list)
            or not tools
            or any(not isinstance(tool, str) for tool in tools)
            or len(tools) != len(set(tools))
        ):
            raise ValueError(f"{context}.native_tools must be a non-empty array of unique strings")
        unknown = sorted(set(tools) - CLAUDE_CODE_NATIVE_TOOLS)
        if unknown:
            raise ValueError(f"{context}.native_tools has unsupported tools: {unknown}")
        if bool(set(tools) & CLAUDE_CODE_SUBAGENT_TOOLS) != resolved["enable_subagents"]:
            raise ValueError(
                f"{context}.native_tools subagent tools require enable_subagents: true, "
                "and enable_subagents: true requires them"
            )
        resolved["native_tools"] = list(tools)
    return resolved


def _validate_model_harness(model: str, harness: str) -> None:
    if harness == "claude_code_glm" and model != "glm-5.3-flash":
        raise ValueError("claude_code_glm requires model glm-5.3-flash")
    if (harness in {"muse_code_native", "muse_code_core"}) != model.startswith("muse-spark-"):
        raise ValueError("Muse Spark models run only with the muse_code_native or muse_code_core harness")
    if harness in {"codex_sub", "codex_sub_mcp"} and not model.startswith("gpt-"):
        raise ValueError(f"harness {harness} requires a GPT model")
    if harness == "claude_code_sub" and model not in {
        "haiku",
        "sonnet-5",
        "opus-5",
        "opus-5.5",
        "opus-5-low",
        "opus-5-medium",
        "opus-5-high",
        "fable-5.1",
    }:
        raise ValueError(
            "harness claude_code_sub requires a Claude subscription model"
        )
    if harness == "mistral_vibe_lean" and model != "leanstral-1.5":
        raise ValueError(
            "harness mistral_vibe_lean requires model leanstral-1.5"
        )
    if harness == "antigravity_cli" and model != "gemini-3.8-flash":
        raise ValueError(
            "harness antigravity_cli requires model gemini-3.8-flash"
        )


def _resolve_named_pricing(
    value: Any, *, model: str, root: Path
) -> dict[str, Any]:
    """Pin the exact token rates used by benchmark cost accounting."""

    pricing = _mapping(value, "model config.pricing")
    _exact_keys(
        pricing,
        {
            "basis",
            "observed_at",
            "source",
            "currency",
            "unit",
            "reasoning_tokens_billed_as",
            "rates",
        },
        "model config.pricing",
    )
    if (
        _text(pricing, "basis", "model config.pricing")
        != "benchmark_cost_accounting_snapshot"
    ):
        raise ValueError(
            "model config.pricing.basis must be benchmark_cost_accounting_snapshot"
        )
    observed_at = _text(pricing, "observed_at", "model config.pricing")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", observed_at):
        raise ValueError("model config.pricing.observed_at must be YYYY-MM-DD")
    source_path = _repo_path(
        root,
        _text(pricing, "source", "model config.pricing"),
        "model config.pricing.source",
    )
    if not source_path.is_file():
        raise ValueError(f"model config.pricing.source is missing: {source_path}")
    if _text(pricing, "currency", "model config.pricing") != "USD":
        raise ValueError("model config.pricing.currency must be USD")
    if _text(pricing, "unit", "model config.pricing") != "per_million_tokens":
        raise ValueError(
            "model config.pricing.unit must be per_million_tokens"
        )
    if (
        _text(
            pricing,
            "reasoning_tokens_billed_as",
            "model config.pricing",
        )
        != "output"
    ):
        raise ValueError(
            "model config.pricing.reasoning_tokens_billed_as must be output"
        )
    rates = _mapping(pricing.get("rates"), "model config.pricing.rates")
    rate_names = ("input", "output", "cache_read", "cache_creation")
    _exact_keys(rates, set(rate_names), "model config.pricing.rates")
    resolved_rates = {
        name: _number(rates, name, "model config.pricing.rates")
        for name in rate_names
    }

    from configs.model_constants import ALL_MODEL_CONFIGS, get_model_prices

    runtime = ALL_MODEL_CONFIGS.get(model)
    if not isinstance(runtime, Mapping):
        raise ValueError(f"model config.pricing cannot resolve model {model!r}")
    runtime_model = _text(runtime, "model_name", "runtime model config")
    expected = get_model_prices(runtime_model)
    actual = tuple(resolved_rates[name] for name in rate_names)
    matches_runtime = expected is not None and all(
        math.isclose(found, float(wanted), rel_tol=0.0, abs_tol=1e-12)
        for found, wanted in zip(actual, expected, strict=True)
    )
    if not matches_runtime:
        raise ValueError(
            "model config.pricing.rates do not match runtime cost accounting "
            f"for {runtime_model}: expected {expected}, found {actual}"
        )
    return {
        "basis": "benchmark_cost_accounting_snapshot",
        "observed_at": observed_at,
        "source": _relative(root, source_path),
        "currency": "USD",
        "unit": "per_million_tokens",
        "reasoning_tokens_billed_as": "output",
        "rates": resolved_rates,
    }


def _resolve_named_access(
    value: Any, *, harness: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate authored access metadata and derive the queue policy."""

    access = _mapping(value, "model config.access")
    mode = _text(access, "mode", "model config.access")
    if mode == "api":
        _exact_keys(access, {"mode"}, "model config.access")
        if harness in SUBSCRIPTION_HARNESSES:
            raise ValueError(
                f"model config.access.mode api is incompatible with {harness}"
            )
        limits = {"five_hour": False, "weekly": False}
    elif mode == "subscription":
        _exact_keys(
            access,
            {"mode", "limits", "credential_env"},
            "model config.access",
            optional={"credential_env"},
        )
        if harness not in SUBSCRIPTION_HARNESSES:
            raise ValueError(
                "model config.access.mode subscription requires a "
                "subscription harness"
            )
        authored_limits = _mapping(
            access.get("limits"), "model config.access.limits"
        )
        _exact_keys(
            authored_limits,
            {"five_hour", "weekly"},
            "model config.access.limits",
        )
        limits = {}
        for name in ("five_hour", "weekly"):
            enabled = authored_limits.get(name)
            if not isinstance(enabled, bool):
                raise ValueError(
                    f"model config.access.limits.{name} must be true or false"
                )
            limits[name] = enabled
        if not any(limits.values()):
            raise ValueError(
                "subscription access must declare at least one quota limit"
            )
        credential_env = access.get("credential_env")
        if credential_env is not None:
            from configs.generator_constants import ALL_GENERATOR_CONFIGS
            from leanlean.subscription_auth import validate_subscription_key_env

            if ALL_GENERATOR_CONFIGS.get(harness, {}).get(
                "subscription_transport"
            ) != "chatgpt":
                raise ValueError(
                    "model config.access.credential_env requires the ChatGPT "
                    "subscription transport"
                )
            try:
                credential_env = validate_subscription_key_env(credential_env)
            except ValueError as error:
                raise ValueError(
                    f"model config.access.credential_env is invalid: {error}"
                ) from error
    else:
        raise ValueError("model config.access.mode must be api or subscription")

    capacity_probe = capacity_probe_for_harness(harness)
    declared_windows = [
        name
        for name, enabled in (
            ("five_hour", limits["five_hour"]),
            ("seven_day", limits["weekly"]),
        )
        if enabled
    ]
    monitoring_strategy = (
        "none"
        if mode == "api"
        else (
            "managed_usage_endpoint_and_provider_events"
            if capacity_probe != "fixed_cooldown"
            else "provider_events"
        )
    )
    resolved_access = {
        "mode": mode,
        **({"limits": limits} if mode == "subscription" else {}),
        **(
            {"credential_env": credential_env}
            if mode == "subscription" and credential_env is not None
            else {}
        ),
        "monitoring": {
            "enabled": mode == "subscription",
            "strategy": monitoring_strategy,
            "capacity_probe": (
                capacity_probe if mode == "subscription" else None
            ),
            "windows": declared_windows,
        },
    }
    subscription_policy = {
        "quota_queue": mode == "subscription",
        "retry_backoff_seconds": 300,
        "fallback_cooldown_seconds": 18000,
    }
    return resolved_access, subscription_policy


def _materialization_build_target(
    materialization: Mapping[str, Any], artifact: Mapping[str, Any], instance_id: str
) -> str:
    """The Lake target an image build warms; ``repository`` uses the repository's own."""
    target = str(materialization.get("build_target") or "")
    if target != "repository":
        return target
    published = artifact.get("materialization")
    target = str(published.get("build_target") or "") if isinstance(published, Mapping) else ""
    if not target.startswith("+"):
        raise ValueError(f"{instance_id}: no published build target for image_materialization")
    return target


def _source_image_policy(value: Any) -> dict[str, Any] | None:
    """Explicit opt-in to build a published source tree, preserving its dataset."""
    if value is None:
        return None
    policy = _mapping(value, "image_materialization")
    _exact_keys(policy, {"mode", "backend", "build_target", "cache", "image_persistence",
                         "minimum_free_disk_gib"}, "image_materialization")
    if policy["mode"] != "build_before_agent" or policy["backend"] != "dockerfile_network_v1":
        raise ValueError("image_materialization requires isolated build_before_agent / dockerfile_network_v1")
    if policy["cache"] != "disabled":
        raise ValueError("source image assembly requires cache: disabled")
    if policy["image_persistence"] not in {"persist", "ephemeral"}:
        raise ValueError("image_materialization.image_persistence must be persist or ephemeral")
    _text(policy, "build_target", "image_materialization")
    _integer(policy, "minimum_free_disk_gib", "image_materialization")
    return dict(policy)


def resolve_config(
    path: Path,
    *,
    repo_root: Path,
    _config: Mapping[str, Any] | None = None,
    _source_configs: list[Mapping[str, Any]] | None = None,
    _config_snapshot_content: str | None = None,
) -> ResolvedEvaluation:
    """Expand one small authored config into a fully pinned run definition."""

    root = repo_root.resolve()
    path = path.resolve()
    if _config is None:
        if not path.is_relative_to((root / "configs/evaluation").resolve()):
            raise ValueError("evaluation configs must live under configs/evaluation/")
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, Mapping):
            raise ValueError(f"{path}: expected a YAML mapping")
    else:
        config = _config
    expected = {
        "kind",
        "schema_version",
        "dataset",
        "repositories",
        "shuffle",
        "variant",
        "run_id",
        "model",
        "reasoning_effort",
        "harness",
        "rounds",
        "max_api_retries",
        "parallelism",
        "container",
        "monitoring",
        "harness_version",
        "generation_parameters",
        "pricing",
        "subscription",
        "task_metadata_enabled",
        "agent",
        "prompt_ablation",
        "image_materialization",
        "codex_model_catalog",
    }
    _exact_keys(
        config,
        expected,
        "config",
        optional={
            "prompt_ablation",
            "image_materialization",
            "codex_model_catalog",
            "subscription",
            "task_metadata_enabled",
            "shuffle",
            "variant",
            "agent",
            "harness_version",
            "generation_parameters",
            "pricing",
            "max_api_retries",
        },
    )
    if config.get("kind") != CONFIG_KIND or config.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported evaluation config")

    run_id = _text(config, "run_id", "config")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id contains unsupported characters")
    model = _text(config, "model", "config")
    effort = _text(config, "reasoning_effort", "config")
    harness = _text(config, "harness", "config")
    if effort not in REASONING_EFFORTS:
        raise ValueError("reasoning_effort is unsupported")
    if harness not in SUPPORTED_HARNESSES:
        raise ValueError("harness is unsupported")
    harness_version = config.get("harness_version")
    codex_model_catalog = (
        _codex_model_catalog(config["codex_model_catalog"], root=root, harness=harness)
        if "codex_model_catalog" in config
        else None
    )
    generation_parameters = config.get("generation_parameters")
    if generation_parameters is not None and harness_version is None:
        raise ValueError("generation_parameters requires harness_version")
    if harness_version is not None:
        if not isinstance(harness_version, str) or not harness_version:
            raise ValueError("harness_version must be a non-empty string")
    if generation_parameters is not None:
        generation_parameters = _mapping(
            generation_parameters, "generation_parameters"
        )
        _exact_keys(
            generation_parameters,
            {
                "policy",
                "observed_at",
                "request_sha256",
                "request",
                "provider_forwarded",
                "retry_policy",
            },
            "generation_parameters",
            optional={"provider_forwarded", "retry_policy"},
        )
        if generation_parameters.get("policy") != "native_harness_defaults":
            raise ValueError(
                "generation_parameters.policy must preserve native defaults"
            )
        observed_at = _text(
            generation_parameters, "observed_at", "generation_parameters"
        )
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", observed_at):
            raise ValueError("generation_parameters.observed_at must be YYYY-MM-DD")
        request_sha = _text(
            generation_parameters, "request_sha256", "generation_parameters"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", request_sha):
            raise ValueError("generation_parameters.request_sha256 must be SHA-256")
        _mapping(
            generation_parameters.get("request"),
            "generation_parameters.request",
        )
        if "provider_forwarded" in generation_parameters:
            _mapping(
                generation_parameters["provider_forwarded"],
                "generation_parameters.provider_forwarded",
            )
        if "retry_policy" in generation_parameters:
            _mapping(
                generation_parameters["retry_policy"],
                "generation_parameters.retry_policy",
            )
    pricing = config.get("pricing")
    if pricing is not None:
        pricing = _mapping(pricing, "pricing")
    _validate_model_harness(model, harness)
    if config.get("rounds") != 1:
        raise ValueError("evaluation schema v1 requires rounds: 1")
    shuffle = config.get("shuffle", False)
    if not isinstance(shuffle, bool):
        raise ValueError("config.shuffle must be true or false")
    max_api_retries = (
        _nonnegative_integer(config, "max_api_retries", "config")
        if "max_api_retries" in config
        else 10
    )
    prompt_ablation = config.get("prompt_ablation")
    if prompt_ablation is not None:
        _validate_prompt_ablation_label(prompt_ablation, "config.prompt_ablation")
        if config.get("agent") is None:
            raise ValueError("config.prompt_ablation requires config.agent")
    agent = None
    if config.get("agent") is not None:
        agent = _resolve_agent_contract(config["agent"], context="config.agent")
        if "native_tools" in agent and harness != "claude_code_sub":
            raise ValueError("config.agent.native_tools is only supported for claude_code_sub")
        enable_lean_verify = agent["enable_lean_verify"]
        enable_proof_length = agent["enable_proof_length"]
        task_metadata_enabled = enable_lean_verify or enable_proof_length
        if harness.endswith("_mcp"):
            raise ValueError(
                "an empty config.agent.mcp_servers requires a non-MCP harness"
            )
    else:
        task_metadata_enabled = config.get("task_metadata_enabled", True)
        if not isinstance(task_metadata_enabled, bool):
            raise ValueError("task_metadata_enabled must be true or false")
        enable_lean_verify = task_metadata_enabled
        enable_proof_length = task_metadata_enabled

    dataset_path = _repo_path(
        root, _text(config, "dataset", "config"), "dataset"
    )
    if not dataset_path.is_relative_to((root / "datasets").resolve()):
        raise ValueError("dataset must live under datasets/")
    requested_repositories = config.get("repositories")
    if (isinstance(requested_repositories, list) and requested_repositories
            and all(isinstance(item, str) for item in requested_repositories)):
        # Router jobs pin one repository; verify that selection without
        # repeatedly hashing every unselected repository in the full bundle.
        dataset = load_dataset(dataset_path, repository_ids=requested_repositories)
    else:
        dataset = load_dataset(dataset_path)
    variant = str(config.get("variant") or dataset.get("default_variant", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", variant):
        raise ValueError("evaluation variant is empty or contains unsupported characters")
    engine_repo_variant = (
        variant if variant in {"raw", "stripped", "optimized"} else "stripped"
    )
    rows = dataset.get("repositories")
    if not isinstance(rows, list) or len(rows) != dataset.get("repository_count"):
        raise ValueError("dataset repository_count does not match repositories")
    selected = _select_repositories(rows, config.get("repositories"))
    repository_task_prompts: dict[str, str] = {}
    repository_reproof_targets: dict[str, list[str]] = {}
    for row in selected:
        instance_id = _text(row, "id", "dataset.repositories[]")
        reproof_value = row.get("reproof")
        if reproof_value is None:
            continue
        reproof = _mapping(reproof_value, f"{instance_id}.reproof")
        declarations = reproof.get("target_declarations")
        if (
            not isinstance(declarations, list)
            or not declarations
            or any(not isinstance(name, str) or not name for name in declarations)
            or len(declarations) != len(set(declarations))
        ):
            raise ValueError(f"{instance_id}: invalid reproof.target_declarations")
        repository_reproof_targets[instance_id] = list(declarations)
        if agent is not None and "{target_declarations}" in agent["prompt"]:
            bullets = "\n".join(f"- `{name}`" for name in declarations)
            repository_task_prompts[instance_id] = agent["prompt"].replace(
                "{target_declarations}", bullets
            )
    if repository_reproof_targets and set(repository_reproof_targets) != {
        _text(row, "id", "dataset.repositories[]") for row in selected
    }:
        raise ValueError("reproof metadata must cover every selected repository")
    comparator_agent_validation = enable_lean_verify and all(
        isinstance(row.get("variants"), Mapping)
        and isinstance(row["variants"].get(variant), Mapping)
        and isinstance(
            row["variants"][variant].get("task_metadata"), Mapping
        )
        and row["variants"][variant]["task_metadata"].get(
            "validation_command"
        )
        == "lean_verify"
        and row["variants"][variant]["task_metadata"].get(
            "validation_engine"
        )
        == "leanprover/comparator"
        for row in selected
    )
    agent_validation = (
        {
            "mode": "registered_palomar_comparator",
            "visible_contracts": ["Challenge.lean", "comparator.json"],
            "command": "lean_verify",
            "hidden_results_yaml": False,
            "pre_agent_green_verify": not bool(repository_reproof_targets),
            "prewarm_artifacts": (
                "not_run_for_reproof_scaffold"
                if repository_reproof_targets
                else "retained_in_agent_container"
            ),
            "usage_telemetry": [
                *(["proof_length"] if enable_proof_length else []),
                "lean_verify",
            ],
            "staging": {
                "base_image": "dataset_declared_immutable_image",
                "source": "dataset_repository_comparator_directory",
                "timing": "after_container_start_before_agent",
                "files": [
                    "Challenge.lean",
                    "comparator.json",
                    *(["proof_length.py"] if enable_proof_length else []),
                ],
            },
        }
        if comparator_agent_validation
        else None
    )

    protected_record = _mapping(
        dataset.get("repository_database"),
        "dataset.repository_database",
    )
    _exact_keys(
        protected_record,
        {
            "path",
            "sha256",
            "definition_sha256",
            "dataset_id",
            "dataset_version",
        },
        "dataset.repository_database",
        optional={"dataset_id", "dataset_version"},
    )
    protected_results_database = {
        "path": _text(
            protected_record, "path", "dataset.repository_database"
        ),
        "sha256": _text(
            protected_record, "sha256", "dataset.repository_database"
        ),
        "definition_sha256": _text(
            protected_record,
            "definition_sha256",
            "dataset.repository_database",
        ),
        "dataset_id": str(
            protected_record.get("dataset_id")
            or _text(dataset, "id", "dataset")
        ),
        "dataset_version": str(
            protected_record.get("dataset_version")
            or _text(dataset, "version", "dataset")
        ),
    }

    parallelism = _mapping(config.get("parallelism"), "parallelism")
    _exact_keys(
        parallelism,
        {"workers", "start_interval_seconds"},
        "parallelism",
    )
    # Workers are a concurrency ceiling. Partial preparation and repository
    # filters can leave fewer jobs than the configured capacity.
    workers = min(_integer(parallelism, "workers", "parallelism"), len(selected))
    start_interval = _number(
        parallelism, "start_interval_seconds", "parallelism"
    )

    container = _mapping(config.get("container"), "container")
    container_keys = {
        "cpus",
        "build_jobs",
        "memory",
        "max_total_memory",
        "pids_limit",
        "timeout",
        "cgroup_parent",
        "network_policy",
    }
    _exact_keys(container, container_keys | {"resource_visibility", "kill_grace"},
                "container", optional={"resource_visibility", "kill_grace"})
    resource_visibility = container.get("resource_visibility", "host")
    if resource_visibility not in {"host", "container_v1"}:
        raise ValueError("unsupported container.resource_visibility")
    cpus = _integer(container, "cpus", "container")
    build_jobs = _integer(container, "build_jobs", "container")
    if resource_visibility == "container_v1" and build_jobs != cpus:
        raise ValueError("container_v1 requires matching CPU and build-thread counts")
    pids_limit = _integer(container, "pids_limit", "container")
    memory = _text(container, "memory", "container").lower()
    max_total_memory = _text(
        container, "max_total_memory", "container"
    ).lower()
    memory_gib = _whole_gib(memory, "container.memory")
    max_memory_gib = _whole_gib(
        max_total_memory, "container.max_total_memory"
    )
    if memory_gib * workers > max_memory_gib:
        raise ValueError(
            "parallel workers exceed container.max_total_memory: "
            f"{workers} x {memory_gib}g > {max_memory_gib}g"
        )
    timeout = _text(container, "timeout", "container")
    if not re.fullmatch(r"[1-9][0-9]*[smhd]", timeout):
        raise ValueError("container.timeout must look like 30m, 2h, or 1d")
    # How long the container's `sleep` outlives the agent's exec timeout. The
    # agent clock starts after in-container setup, so slow setup eats into it.
    kill_grace = container.get("kill_grace")
    if kill_grace is not None and not (
        isinstance(kill_grace, str) and re.fullmatch(r"[1-9][0-9]*[smh]", kill_grace)
    ):
        raise ValueError("container.kill_grace must look like 30m or 90m")
    if container.get("network_policy") != "model_proxy_only":
        raise ValueError(
            "evaluation requires network_policy: model_proxy_only"
        )

    monitoring = _mapping(config.get("monitoring"), "monitoring")
    monitoring_keys = {
        "snapshots",
        "every_n_edits",
        "reconciliation_interval_seconds",
    }
    _exact_keys(monitoring, monitoring_keys, "monitoring")
    snapshots = monitoring.get("snapshots")
    if not isinstance(snapshots, bool):
        raise ValueError("monitoring.snapshots must be true or false")
    every_n_edits = _integer(monitoring, "every_n_edits", "monitoring")
    reconciliation = _integer(
        monitoring, "reconciliation_interval_seconds", "monitoring"
    )
    if snapshots and harness not in CHECKPOINT_HARNESSES:
        raise ValueError(
            f"host-side snapshots are not supported by {harness}"
        )

    subscription_defaults = {
        "quota_queue": True,
        "retry_backoff_seconds": 300,
        "fallback_cooldown_seconds": 18000,
    }
    authored_subscription = config.get("subscription")
    if authored_subscription is None:
        subscription = subscription_defaults
    else:
        subscription = _mapping(authored_subscription, "subscription")
        _exact_keys(
            subscription,
            set(subscription_defaults),
            "subscription",
        )
    quota_queue = subscription.get("quota_queue")
    if not isinstance(quota_queue, bool):
        raise ValueError("subscription.quota_queue must be true or false")
    retry_backoff_seconds = _integer(
        subscription, "retry_backoff_seconds", "subscription"
    )
    fallback_cooldown_seconds = _integer(
        subscription, "fallback_cooldown_seconds", "subscription"
    )
    if fallback_cooldown_seconds < retry_backoff_seconds:
        raise ValueError(
            "subscription.fallback_cooldown_seconds must be at least "
            "subscription.retry_backoff_seconds"
        )
    capacity_probe = capacity_probe_for_harness(harness)
    capacity_endpoints = {
        "anthropic_oauth_usage": ANTHROPIC_USAGE_ENDPOINT,
    }
    capacity_windows = {
        "anthropic_oauth_usage": ANTHROPIC_USAGE_WINDOWS,
    }
    resolved_subscription = {
        "quota_queue": quota_queue,
        "retry_backoff_seconds": retry_backoff_seconds,
        "fallback_cooldown_seconds": fallback_cooldown_seconds,
        "capacity_probe": capacity_probe,
        "capacity_endpoint": capacity_endpoints.get(capacity_probe),
        "capacity_windows": list(capacity_windows.get(capacity_probe, ())),
        "max_utilization_percent": 100,
    }

    repo_images: dict[str, dict[str, Any]] = {}
    pinned_repositories = []
    image_resolutions: set[str] = set()
    source_image_policy = _source_image_policy(config.get("image_materialization"))
    materialization_backends: set[str] = set()
    safe_run = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
    for row in selected:
        instance_id = _text(row, "id", "dataset.repositories[]")
        variants = _mapping(
            row.get("variants"), f"{instance_id}.variants"
        )
        artifact = _mapping(
            variants.get(variant), f"{instance_id}.{variant}"
        )
        materialization = source_image_policy or artifact.get("materialization")
        if isinstance(materialization, Mapping) and materialization.get("unavailable"):
            raise ValueError(
                f"{instance_id}: {materialization['unavailable']}; evaluate with a "
                "dataset config that sets image_materialization (for example "
                "configs/dataset/leanlean_20260914-hf.yaml)"
            )
        if isinstance(materialization, Mapping):
            if materialization.get("mode") != "build_before_agent":
                raise ValueError(
                    f"{instance_id}: unsupported image materialization mode"
                )
            source_tree = _repo_path(
                root,
                _text(artifact, "cache_tree", f"{instance_id}.{variant}"),
                f"{instance_id}.{variant}.cache_tree",
            )
            allowed_source_roots = (
                (root / "prod_repos").resolve(),
                (root / "datasets").resolve(),
            )
            if not any(
                source_tree.is_relative_to(candidate)
                for candidate in allowed_source_roots
            ) or not source_tree.is_dir():
                raise ValueError(
                    f"{instance_id}: {variant} source tree is missing"
                )
            tree_sha = _text(
                artifact, "tree_sha256", f"{instance_id}.{variant}"
            )
            if (
                not re.fullmatch(r"[0-9a-f]{64}", tree_sha)
                or source_tree_sha256(source_tree) != tree_sha
            ):
                raise ValueError(
                    f"{instance_id}: {variant} source tree drift"
                )
            persist = (
                materialization.get("image_persistence") == "persist"
            )
            engine_pin = {
                "tag": (
                    f"leanlean-{instance_id}-{engine_repo_variant}:"
                    f"eval-{safe_run}"
                ),
                "source_tree": _relative(root, source_tree),
                "tree_sha256": tree_sha,
                "build_target": _materialization_build_target(
                    materialization, artifact, instance_id
                ),
                "build_jobs": build_jobs,
                "build_timeout_seconds": 14400,
                "persist": persist,
                "source_isolation": "standalone_v2",
            }
            backend = str(
                materialization.get("backend") or "dockerfile_network_v1"
            )
            materialization_backends.add(backend)
            if source_image_policy is not None:
                engine_pin.update(backend=backend, repo_variant=engine_repo_variant,
                                  cache=source_image_policy["cache"],
                                  minimum_free_disk_gib=source_image_policy["minimum_free_disk_gib"])
            if backend == "shared_environment_v1":
                environment = _mapping(
                    materialization.get("shared_environment"),
                    f"{instance_id}.shared_environment",
                )
                environment_id = _text(
                    environment, "id", f"{instance_id}.shared_environment"
                )
                environment_image = _text(
                    environment, "image", f"{instance_id}.shared_environment"
                )
                environment_image_id = _text(
                    environment, "image_id", f"{instance_id}.shared_environment"
                )
                if not re.fullmatch(
                    r"sha256:[0-9a-f]{64}", environment_image_id
                ):
                    raise ValueError(
                        f"{instance_id}: invalid shared environment image ID"
                    )
                archive_record = environment.get("archive")
                archive = None
                if isinstance(archive_record, Mapping):
                    archive_path = validated_archive(environment, repo_root=root)
                    archive = {
                        "format": str(archive_record["format"]),
                        "path": _relative(root, archive_path),
                        "sha256": str(archive_record["sha256"]),
                        "bytes": int(archive_record.get("bytes", archive_path.stat().st_size)),
                    }
                warm = _mapping(
                    materialization.get("warm_build_cache"),
                    f"{instance_id}.warm_build_cache",
                )
                warm_path = _repo_path(
                    root,
                    _text(warm, "path", f"{instance_id}.warm_build_cache"),
                    f"{instance_id}.warm_build_cache.path",
                )
                warm_sha = _text(
                    warm, "sha256", f"{instance_id}.warm_build_cache"
                )
                if (
                    not warm_path.is_relative_to((root / "datasets").resolve())
                    or not warm_path.is_dir()
                    or not re.fullmatch(r"[0-9a-f]{64}", warm_sha)
                    or artifact_tree_sha256(warm_path) != warm_sha
                ):
                    raise ValueError(f"{instance_id}: warm build cache drift")
                engine_pin.update({
                    "backend": backend,
                    "shared_environment": {
                        "id": environment_id,
                        "image": environment_image,
                        "image_id": environment_image_id,
                        **({"archive": archive} if archive is not None else {}),
                    },
                    "warm_build_cache": {
                        "path": _relative(root, warm_path),
                        "sha256": warm_sha,
                    },
                    "source_isolation": "standalone_v3",
                })
                branch_seeding = materialization.get("reconciliation")
                if branch_seeding is not None:
                    engine_pin["reconciliation"] = _pinned_reconciliation(
                        root, branch_seeding, instance_id
                    )
            elif backend != "dockerfile_network_v1":
                raise ValueError(
                    f"{instance_id}: unsupported materialization backend {backend!r}"
                )
            repo_images[instance_id] = engine_pin
            pinned = {
                "id": instance_id,
                "source_tree": engine_pin["source_tree"],
                "tree_sha256": tree_sha,
            }
            if backend == "shared_environment_v1":
                pinned.update({
                    "environment_id": engine_pin["shared_environment"]["id"],
                    "environment_image_id": engine_pin["shared_environment"]["image_id"],
                    "warm_build_cache_sha256": engine_pin["warm_build_cache"]["sha256"],
                })
            pinned["materialization"] = copy.deepcopy(engine_pin)
            pinned_repositories.append(pinned)
            image_resolutions.add("build_before_agent")
            continue

        image_tag = _text(artifact, "image", f"{instance_id}.{variant}")
        image_id = _text(artifact, "image_id", f"{instance_id}.{variant}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError(f"{instance_id}: invalid immutable image ID")
        engine_pin = {
            "tag": image_tag,
            "image_id": image_id,
        }
        pinned_repository = {
            "id": instance_id,
            "image": image_tag,
            "image_id": image_id,
        }
        image_manifest = artifact.get("image_manifest")
        if image_manifest is not None:
            image_manifest_path = _repo_path(
                root,
                _text(artifact, "image_manifest", f"{instance_id}.{variant}"),
                f"{instance_id}.{variant}.image_manifest",
            )
            if (
                not image_manifest_path.is_relative_to(
                    (root / "datasets").resolve()
                )
                or not image_manifest_path.is_file()
            ):
                raise ValueError(
                    f"{instance_id}: image manifest is missing or outside datasets/"
                )
            image_manifest_sha256 = _text(
                artifact,
                "image_manifest_sha256",
                f"{instance_id}.{variant}",
            )
            if file_sha256(image_manifest_path) != image_manifest_sha256:
                raise ValueError(f"{instance_id}: Docker image manifest drift")
            manifest_pin = {
                "path": _relative(root, image_manifest_path),
                "sha256": image_manifest_sha256,
            }
            engine_pin["manifest"] = manifest_pin
            pinned_repository["image_manifest"] = manifest_pin
        source = _mapping(row.get("source"), f"{instance_id}.source")
        if source.get("monorepo_member") is True:
            engine_pin["source_isolation"] = "standalone_v2"
        repo_images[instance_id] = engine_pin
        pinned_repositories.append(pinned_repository)
        image_resolutions.add("pinned_id")

    if len(image_resolutions) != 1:
        raise ValueError(
            "selected repositories mix persistent images and source trees"
        )
    image_resolution = next(iter(image_resolutions))
    output_dir = (
        root
        / "output/leanlean/leanlean/fix"
        / harness
        / model
        / f"run_{run_id}"
    )
    run_dir = root / "runs/evaluation" / run_id
    manifest_path = root / "experiments/evaluation" / f"{run_id}.yaml"
    engine_manifest_path = run_dir / "engine.yaml"
    run_artifact_path = run_dir / "run.yaml"
    # tmux reads "." and ":" in a target as window/pane separators.
    tmux_session = re.sub(r"[.:]", "-", f"eval-{run_id}")

    playback = {
        "enabled": snapshots,
        "mode": (
            "host_side_checkpointing" if snapshots else "external_replay"
        ),
        "capture_repository_scope": snapshots,
        "every_n_edits": every_n_edits,
        "capture_snapshots": snapshots,
        "reconciliation_interval_seconds": reconciliation,
        "capture_timeout_seconds": 900,
        "build_each_checkpoint": False,
        "build_timeout_seconds": 1800,
        "build_output_chars": 12000,
        "reject_asynchronous_actions": False,
    }
    engine_manifest = {
        "model": model,
        "reasoning_effort": effort,
        "generator": harness,
        "plan_type": "no_plan",
        "task_variant": "fix",
        "benchmark": "leanlean",
        "dataset_name": "leanlean",
        "repos": [row["id"] for row in pinned_repositories],
        "repo_variant": engine_repo_variant,
        "repo_images": repo_images,
        "protected_results_database": protected_results_database,
        "image_resolution": image_resolution,
        "refactor_rounds": 0,
        "shuffle": shuffle,
        "workers": workers,
        "instance_start_interval_seconds": start_interval,
        "progress": "rich",
        "build_jobs": build_jobs,
        "container_cpus": cpus,
        "container_memory": memory,
        "container_cgroup_parent": _text(
            container, "cgroup_parent", "container"
        ),
        "container_pids_limit": pids_limit,
        "container_timeout": timeout,
        "network_policy": "model_proxy_only",
        "discard_cache_after_eval": image_resolution == "build_before_agent",
        "task_metadata_enabled": task_metadata_enabled,
        "max_api_retries": max_api_retries,
        "enable_lean_verify": enable_lean_verify,
        "enable_proof_length": enable_proof_length,
        "run_id": run_id,
        "playback": playback,
        "trajectory": {
            "capture_native_rollout": snapshots
            and harness in {
                "claude_code_sub",
                "codex_sub",
                "codex_sub_mcp",
                "muse_code_native",
                "muse_code_core",
            },
            "native_rollout_timeout_seconds": 180,
        },
    }
    if resource_visibility != "host":
        engine_manifest["container_resource_visibility"] = resource_visibility
    if kill_grace is not None:
        engine_manifest["container_grace_seconds"] = int(kill_grace[:-1]) * {
            "s": 1, "m": 60, "h": 3600
        }[kill_grace[-1]]
    if harness_version is not None:
        engine_manifest["harness_version"] = harness_version
    if codex_model_catalog is not None:
        engine_manifest["codex_model_catalog"] = dict(codex_model_catalog)
    if generation_parameters is not None:
        engine_manifest["generation_parameters"] = dict(generation_parameters)
    if pricing is not None:
        engine_manifest["pricing"] = dict(pricing)
    if agent is not None:
        engine_manifest["agent"] = agent
    if repository_task_prompts:
        engine_manifest["repository_task_prompts"] = repository_task_prompts
    if repository_reproof_targets:
        engine_manifest["repository_reproof_targets"] = repository_reproof_targets
    if agent_validation is not None:
        engine_manifest["agent_validation"] = agent_validation
    engine_content = _dump(engine_manifest)
    implementation = {
        relative: file_sha256(root / relative)
        for relative in IMPLEMENTATION_FILES
    }
    manifest = {
        "kind": MANIFEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": {
            "manifest": _relative(root, dataset_path),
            "sha256": file_sha256(dataset_path),
            "id": dataset.get("id"),
            "version": dataset.get("version"),
            "variant": variant,
            "preprocessing_run_id": _mapping(
                dataset.get("preprocessing"), "dataset.preprocessing"
            ).get("run_id"),
        },
        "repositories": pinned_repositories,
        "model": model,
        "reasoning_effort": effort,
        "harness": harness,
        **(
            {"harness_version": harness_version, **({"generation_parameters": dict(generation_parameters)} if generation_parameters is not None else {})}
            if harness_version is not None else {}
        ),
        **({"pricing": dict(pricing)} if pricing is not None else {}),
        **(
            {"codex_model_catalog": dict(codex_model_catalog)}
            if codex_model_catalog is not None else {}
        ),
        "rounds": 1,
        "shuffle": shuffle,
        "max_api_retries": max_api_retries,
        "task_metadata_enabled": task_metadata_enabled,
        "enable_lean_verify": enable_lean_verify,
        "enable_proof_length": enable_proof_length,
        "parallelism": {
            "workers": workers,
            "start_interval_seconds": start_interval,
        },
        "image_materialization": {
            "mode": image_resolution,
            "timing": "per_repository_before_agent_container",
            "concurrency": workers,
            "build_timeout_seconds": (
                14400 if image_resolution == "build_before_agent" else None
            ),
            "network_policy": (
                "none"
                if materialization_backends == {"shared_environment_v1"}
                else "github_and_pinned_lake_dependency_fetch_only"
                if image_resolution == "build_before_agent"
                else "not_applicable"
            ),
            "backend": (
                next(iter(materialization_backends))
                if len(materialization_backends) == 1
                else "mixed"
                if materialization_backends
                else "not_applicable"
            ),
            "counted_in_agent_lifetime": False,
            "persist_images": any(
                bool(row.get("persist")) for row in repo_images.values()
            ),
        },
        "container": {
            "cpus": cpus,
            "build_jobs": build_jobs,
            "memory": memory,
            "max_total_memory": max_total_memory,
            "requested_total_memory": f"{memory_gib * workers}g",
            **({"resource_visibility": resource_visibility} if resource_visibility != "host" else {}),
            **({"kill_grace": kill_grace} if kill_grace is not None else {}),
            "pids_limit": pids_limit,
            "timeout": timeout,
            "agent_lifetime": timeout,
            "agent_lifetime_starts_after": "image_materialization",
            "cgroup_parent": engine_manifest["container_cgroup_parent"],
            "network_policy": "model_proxy_only",
        },
        "monitoring": {
            "snapshots": snapshots,
            "every_n_edits": every_n_edits,
            "reconciliation_interval_seconds": reconciliation,
            "build_checkpoints_during_run": False,
            "checkpoint_build_stage": "postprocess --replay",
        },
        "subscription": resolved_subscription,
        "engine": {
            "manifest": _relative(root, engine_manifest_path),
            "sha256": _content_sha256(engine_content),
        },
        "outputs": {
            "run_artifact": _relative(root, run_artifact_path),
            "output_directory": _relative(root, output_dir),
            "tmux_session": tmux_session,
        },
        "implementation": {"files": implementation},
    }
    if agent is not None:
        manifest["agent"] = agent
    if prompt_ablation is not None:
        manifest["prompt_ablation"] = prompt_ablation
    if _source_configs is None:
        manifest["source_config"] = {
            "path": _relative(root, path),
            "sha256": file_sha256(path),
        }
    else:
        manifest["source_configs"] = [dict(item) for item in _source_configs]
    if agent_validation is not None:
        manifest["agent_validation"] = agent_validation
    return ResolvedEvaluation(
        path,
        config,
        dataset_path,
        dataset,
        manifest,
        engine_manifest,
        manifest_path,
        engine_manifest_path,
        run_dir,
        run_artifact_path,
        run_dir / "config.snapshot.yaml",
        output_dir,
        tmux_session,
        _config_snapshot_content,
        root,
    )


def _resolve_named_parallelism(
    dataset_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve evaluation concurrency exclusively from the model config."""
    if "parallelism" in dataset_config:
        raise ValueError("parallelism belongs in the model config, not the dataset config")
    parallelism = _mapping(model_config.get("parallelism"), "model config.parallelism")
    _exact_keys(parallelism, {"workers", "start_interval_seconds"}, "model config.parallelism")
    workers = parallelism["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("model config.parallelism.workers must be a positive integer")
    spacing = _number(parallelism, "start_interval_seconds", "model config.parallelism")
    if not math.isfinite(spacing):
        raise ValueError("model config.parallelism.start_interval_seconds must be finite")
    return {"workers": workers, "start_interval_seconds": spacing}


def _validate_prompt_ablation_label(value: Any, context: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(f"{context} must be a non-empty [A-Za-z0-9_.-] label")


def _validate_leanlean_20260914_prompt_lock(
    dataset_config: Mapping[str, Any], *, repo_root: Path
) -> None:
    """Reject accidental prompt variants from the canonical final benchmark.

    A deliberate prompt ablation opts out with a top-level ``prompt_ablation``
    label, which is carried into the run manifest and postprocessing report.
    """

    if dataset_config.get("dataset") != LEANLEAN_20260914_DATASET:
        return
    if dataset_config.get("prompt_ablation") is not None:
        _validate_prompt_ablation_label(
            dataset_config["prompt_ablation"], "dataset config.prompt_ablation"
        )
        return
    canonical_path = repo_root / LEANLEAN_20260914_CANONICAL_CONFIG
    canonical = _load_yaml_mapping(canonical_path, "canonical LeanLean dataset config")
    canonical_agent = _mapping(canonical.get("agent"), "canonical dataset config.agent")
    canonical_prompt = _text(canonical_agent, "prompt", "canonical dataset config.agent")
    canonical_sha256 = hashlib.sha256(canonical_prompt.encode("utf-8")).hexdigest()
    if canonical_sha256 != LEANLEAN_20260914_PROMPT_SHA256:
        raise ValueError(
            "canonical LeanLean 2026-09-14 prompt drifted; explicit user approval "
            "and a prompt-lock update are required before launch"
        )
    agent = _mapping(dataset_config.get("agent"), "dataset config.agent")
    prompt = _text(agent, "prompt", "dataset config.agent")
    actual_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual_sha256 != LEANLEAN_20260914_PROMPT_SHA256:
        raise ValueError(
            "LeanLean 2026-09-14 prompt mismatch: expected canonical SHA-256 "
            f"{LEANLEAN_20260914_PROMPT_SHA256}, got {actual_sha256}; "
            "do not launch this run"
        )


def resolve_named_config(
    dataset_reference: str | Path,
    model_reference: str | Path,
    *,
    repo_root: Path,
    dataset_snapshot: Mapping[str, str] | None = None,
) -> ResolvedEvaluation:
    """Compose task and model configs identified by their config paths."""

    root = repo_root.resolve()
    dataset_path, dataset_key = _named_config_path(
        dataset_reference,
        repo_root=root,
        directory="dataset",
        name="dataset",
    )
    model_path, model_key = _named_config_path(
        model_reference,
        repo_root=root,
        directory="models",
        name="model",
    )
    if len(Path(model_key).parts) < 2:
        raise ValueError("model must be named as <provider>/<model>")

    dataset_content_path = dataset_path
    if dataset_snapshot is not None:
        dataset_content_path = _repo_path(root, dataset_snapshot["path"], "frozen dataset config")
        if not dataset_content_path.is_relative_to(root / "experiments"):
            raise ValueError("frozen dataset config must live under experiments/")
        if file_sha256(dataset_content_path) != dataset_snapshot["sha256"]:
            raise ValueError("frozen dataset config hash drift")
    dataset_config = _load_yaml_mapping(dataset_content_path, "dataset evaluation config")
    dataset_keys = {
        "kind",
        "schema_version",
        "dataset",
        "repositories",
        "shuffle",
        "variant",
        "rounds",
        "max_api_retries",
        "container",
        "monitoring",
        "agent",
        "postprocessing",
        "prompt_ablation",
        "image_materialization",
    }
    _exact_keys(
        dataset_config,
        dataset_keys,
        "dataset config",
        optional={
            "postprocessing",
            "shuffle",
            "variant",
            "max_api_retries",
            "prompt_ablation",
            "image_materialization",
        },
    )
    if (
        dataset_config.get("kind") != DATASET_CONFIG_KIND
        or dataset_config.get("schema_version") != 1
    ):
        raise ValueError(f"{dataset_path}: unsupported dataset evaluation config")
    _validate_leanlean_20260914_prompt_lock(dataset_config, repo_root=root)

    model_config = _load_yaml_mapping(model_path, "model evaluation config")
    _exact_keys(
        model_config,
        {
            "kind",
            "schema_version",
            "model",
            "harness",
            "harness_version",
            "system_prompt",
            "reasoning_effort",
            "generation_parameters",
            "pricing",
            "access",
            "parallelism",
            "codex_model_catalog",
            "native_tools",
        },
        "model config",
        optional={"codex_model_catalog", "native_tools"},
    )
    if (
        model_config.get("kind") != MODEL_CONFIG_KIND
        or model_config.get("schema_version") != 1
    ):
        raise ValueError(f"{model_path}: unsupported model evaluation config")

    parallelism = _resolve_named_parallelism(dataset_config, model_config)
    harness = _text(model_config, "harness", "model config")
    from configs.generator_constants import ALL_GENERATOR_CONFIGS

    harness_version = _text(model_config, "harness_version", "model config")
    pinned_version = str(
        ALL_GENERATOR_CONFIGS.get(harness, {}).get("host_tool_version") or ""
    )
    if (
        harness_version != pinned_version
        and harness_version not in HARNESS_VERSION_OVERRIDES.get(harness, ())
    ):
        raise ValueError(
            f"model config harness_version {harness_version!r} does not match "
            f"{harness} host_tool_version {pinned_version!r}"
        )
    generation_parameters = _mapping(
        model_config.get("generation_parameters"),
        "model config.generation_parameters",
    )
    system_prompt = _resolve_native_system_prompt(
        model_config.get("system_prompt"),
        root=root,
        model_path=model_path,
        harness_version=harness_version,
        request_sha256=_text(
            generation_parameters,
            "request_sha256",
            "model config.generation_parameters",
        ),
    )
    pricing = _resolve_named_pricing(
        model_config.get("pricing"),
        model=_text(model_config, "model", "model config"),
        root=root,
    )
    resolved_access, subscription_policy = _resolve_named_access(
        model_config.get("access"), harness=harness
    )

    postprocessing = dataset_config.get("postprocessing")
    if postprocessing is not None:
        postprocessing = _mapping(postprocessing, "dataset config.postprocessing")
        _exact_keys(
            postprocessing,
            {"replay", "workers", "threads_per_repository"},
            "dataset config.postprocessing",
        )
        if not isinstance(postprocessing.get("replay"), bool):
            raise ValueError("dataset config.postprocessing.replay must be true or false")
        _integer(postprocessing, "workers", "dataset config.postprocessing")
        _integer(
            postprocessing,
            "threads_per_repository",
            "dataset config.postprocessing",
        )

    composed = {
        key: value
        for key, value in dataset_config.items()
        if key not in {"kind", "schema_version", "postprocessing"}
    }
    composed.update(
        {
            "kind": CONFIG_KIND,
            "schema_version": 1,
            "run_id": _derived_run_id(dataset_key, model_key),
            "model": model_config["model"],
            "harness": harness,
            "harness_version": harness_version,
            "reasoning_effort": model_config["reasoning_effort"],
            "generation_parameters": dict(generation_parameters),
            "pricing": dict(pricing),
            "subscription": subscription_policy,
            "parallelism": parallelism,
        }
    )
    if "codex_model_catalog" in model_config:
        composed["codex_model_catalog"] = _codex_model_catalog(
            model_config["codex_model_catalog"], root=root, harness=harness
        )
    if "native_tools" in model_config:
        # The model's Claude Code tool list joins the dataset's agent contract,
        # where resolve_config validates it.
        agent = _mapping(composed.get("agent"), "dataset config.agent")
        if "native_tools" in agent:
            raise ValueError("native_tools is set in both the dataset and the model config")
        composed["agent"] = {**agent, "native_tools": model_config["native_tools"]}
    source_configs = [
        {
            "role": "dataset",
            "key": dataset_key,
            "path": _relative(root, dataset_path),
            "sha256": file_sha256(dataset_content_path),
        },
        {
            "role": "model",
            "key": model_key,
            "path": _relative(root, model_path),
            "sha256": file_sha256(model_path),
        },
    ]
    snapshot = {
        "kind": "leanlean_evaluation_composition",
        "schema_version": 1,
        "dataset_config": dict(source_configs[0]),
        "model_config": dict(source_configs[1]),
        "system_prompt": dict(system_prompt),
        "pricing": dict(pricing),
        "access": resolved_access,
        "resolved": composed,
    }
    if dataset_snapshot is not None:
        snapshot["frozen_dataset_config"] = dict(dataset_snapshot)
    resolved = resolve_config(
        dataset_path,
        repo_root=root,
        _config=composed,
        _source_configs=source_configs,
        _config_snapshot_content=_dump(snapshot),
    )

    # Keep the composed snapshot aligned with the realized concurrency after
    # repository selection; source_configs still pin the authored ceiling.
    snapshot["resolved"]["parallelism"] = dict(resolved.manifest["parallelism"])

    logical_path = Path(dataset_key) / Path(model_key)
    run_dir = root / "runs/evaluation" / logical_path
    manifest_path = root / "experiments/evaluation" / logical_path / "manifest.yaml"
    output_dir = root / "output/evaluation" / logical_path
    engine_manifest_path = run_dir / "engine.yaml"
    run_artifact_path = run_dir / "run.yaml"
    engine_manifest = dict(resolved.engine_manifest)
    engine_manifest["output_directory"] = _relative(root, output_dir)
    manifest = dict(resolved.manifest)
    if (
        dataset_config.get("dataset") == LEANLEAN_20260914_DATASET
        and dataset_config.get("prompt_ablation") is None
    ):
        manifest["agent_prompt_sha256"] = LEANLEAN_20260914_PROMPT_SHA256
    manifest["system_prompt"] = dict(system_prompt)
    manifest["access"] = resolved_access
    manifest["subscription"] = {
        **manifest["subscription"],
        "capacity_windows": list(resolved_access["monitoring"]["windows"]),
    }
    manifest["engine"] = {
        "manifest": _relative(root, engine_manifest_path),
        "sha256": _content_sha256(_dump(engine_manifest)),
    }
    manifest["outputs"] = {
        "run_artifact": _relative(root, run_artifact_path),
        "output_directory": _relative(root, output_dir),
        "tmux_session": resolved.tmux_session,
    }
    return replace(
        resolved,
        manifest=manifest,
        engine_manifest=engine_manifest,
        manifest_path=manifest_path,
        engine_manifest_path=engine_manifest_path,
        run_dir=run_dir,
        run_artifact_path=run_artifact_path,
        config_snapshot_path=run_dir / "config.snapshot.yaml",
        config_snapshot_content=_dump(snapshot),
        output_dir=output_dir,
    )


def validate_named_run_identity(
    resolved: ResolvedEvaluation, *, repo_root: Path
) -> str | None:
    """Validate that an existing logical path still pins the same configs."""

    if not resolved.run_artifact_path.is_file():
        return None
    artifact = _load_yaml_mapping(
        resolved.run_artifact_path, "evaluation run artifact"
    )
    experiment = _mapping(artifact.get("experiment"), "run.experiment")
    manifest_path = _repo_path(
        repo_root.resolve(),
        _text(experiment, "manifest", "run.experiment"),
        "run.experiment.manifest",
    )
    manifest = _load_yaml_mapping(manifest_path, "evaluation manifest")
    if manifest.get("source_configs") != resolved.manifest.get("source_configs"):
        raise RuntimeError(
            "this dataset/model config path already identifies a run, but one "
            "of the configs changed; create a new dataset config path for the "
            "ablation"
        )
    status = artifact.get("status")
    return str(status) if status is not None else ""


def load_named_postprocessing_config(
    dataset_reference: str | Path,
    *,
    repo_root: Path,
) -> Mapping[str, Any]:
    """Load the postprocessing resources owned by a dataset config."""

    path, _ = _named_config_path(
        dataset_reference,
        repo_root=repo_root,
        directory="dataset",
        name="dataset",
    )
    config = _load_yaml_mapping(path, "dataset evaluation config")
    if (
        config.get("kind") != DATASET_CONFIG_KIND
        or config.get("schema_version") != 1
    ):
        raise ValueError(f"{path}: unsupported dataset evaluation config")
    value = config.get("postprocessing")
    if value is None:
        return {"replay": False, "workers": None, "threads_per_repository": None}
    row = _mapping(value, "dataset config.postprocessing")
    _exact_keys(
        row,
        {"replay", "workers", "threads_per_repository"},
        "dataset config.postprocessing",
    )
    if not isinstance(row.get("replay"), bool):
        raise ValueError("dataset config.postprocessing.replay must be true or false")
    return {
        "replay": row["replay"],
        "workers": _integer(row, "workers", "dataset config.postprocessing"),
        "threads_per_repository": _integer(
            row,
            "threads_per_repository",
            "dataset config.postprocessing",
        ),
    }


def _write_exact(path: Path, content: str, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(
            f"run ID {run_id!r} already has a different {path.name}; "
            "choose a new run_id"
        )
    path.write_text(content)


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_dump(value))
    temporary.replace(path)


def write_run_definition(resolved: ResolvedEvaluation) -> None:
    """Persist the exact experiment and initial run handoff artifact."""

    run_id = str(resolved.manifest["run_id"])
    manifest_content = _dump(resolved.manifest)
    engine_content = _dump(resolved.engine_manifest)
    if (
        not resolved.run_artifact_path.exists()
        and resolved.output_dir.exists()
        and any(resolved.output_dir.iterdir())
    ):
        raise RuntimeError(
            f"run ID {run_id!r} already has output without a pinned run artifact"
        )
    _write_exact(resolved.manifest_path, manifest_content, run_id)
    _write_exact(resolved.engine_manifest_path, engine_content, run_id)
    _write_exact(
        resolved.config_snapshot_path,
        resolved.config_snapshot_content or resolved.config_path.read_text(),
        run_id,
    )

    experiment_sha = _content_sha256(manifest_content)
    if resolved.run_artifact_path.exists():
        existing = (
            yaml.safe_load(resolved.run_artifact_path.read_text()) or {}
        )
        if (
            not isinstance(existing, Mapping)
            or _mapping(
                existing.get("experiment"), "run.experiment"
            ).get("sha256")
            != experiment_sha
        ):
            raise RuntimeError(
                f"run ID {run_id!r} already has a different run artifact"
            )
        return

    output = str(resolved.manifest["outputs"]["output_directory"])
    artifact = {
        "kind": RUN_ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "configured",
        "experiment": {
            "manifest": _relative_or_absolute(
                resolved.repo_root or resolved.config_path.parents[2],
                resolved.manifest_path,
            ),
            "sha256": experiment_sha,
        },
        "dataset": dict(resolved.manifest["dataset"]),
        "execution": {
            "model": resolved.manifest["model"],
            "reasoning_effort": resolved.manifest["reasoning_effort"],
            "harness": resolved.manifest["harness"],
            "harness_version": resolved.manifest.get("harness_version"),
            "generation_parameters": resolved.manifest.get("generation_parameters"),
            "pricing": resolved.manifest.get("pricing"),
            "repositories": len(resolved.manifest["repositories"]),
        },
        "subscription": dict(resolved.manifest["subscription"]),
        **(
            {"access": dict(resolved.manifest["access"])}
            if "access" in resolved.manifest
            else {}
        ),
        "artifacts": {
            "output_directory": output,
            "predictions": f"{output}/preds.json",
            "subscription_status": f"{output}/subscription_status.json",
            "proxy_forensic_trace": (
                f"{output}/proxy_traces/*.forensic.jsonl"
            ),
            "proxy_accounting_trace": (
                f"{output}/proxy_traces/*.accounting.jsonl"
            ),
            "monitoring": f"{output}/*/playback/capture_*/playback.json",
            "standardized_traces": (
                f"{output}/*/playback/capture_*/standardized-trace.json"
            ),
            "checkpoint_archives": (
                f"{output}/*/playback/capture_*/*.tar.gz"
            ),
        },
    }
    _atomic_yaml(resolved.run_artifact_path, artifact)


def load_run_manifest(path: Path, *, repo_root: Path) -> PreparedEvaluation:
    """Load a generated manifest and fail if any pinned input has drifted."""

    root = repo_root.resolve()
    path = path.resolve()
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping")
    if (
        manifest.get("kind") != MANIFEST_KIND
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"{path}: unsupported evaluation run manifest")
    dataset = _mapping(manifest.get("dataset"), "manifest.dataset")
    dataset_path = _repo_path(
        root,
        _text(dataset, "manifest", "manifest.dataset"),
        "dataset",
    )
    if file_sha256(dataset_path) != dataset.get("sha256"):
        raise ValueError(
            f"{path}: dataset artifact drifted after resolution"
        )
    engine = _mapping(manifest.get("engine"), "manifest.engine")
    engine_path = _repo_path(
        root,
        _text(engine, "manifest", "manifest.engine"),
        "engine.manifest",
    )
    if file_sha256(engine_path) != engine.get("sha256"):
        raise ValueError(f"{path}: generated engine manifest drifted")
    implementation = _mapping(
        _mapping(
            manifest.get("implementation"),
            "manifest.implementation",
        ).get("files"),
        "manifest.implementation.files",
    )
    drifted = [
        relative
        for relative, expected in implementation.items()
        if file_sha256(root / relative) != expected
    ]
    if drifted:
        raise ValueError(
            f"{path}: evaluation implementation drifted: "
            + ", ".join(drifted)
        )
    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    run_artifact_path = _repo_path(
        root,
        _text(outputs, "run_artifact", "manifest.outputs"),
        "outputs.run_artifact",
    )
    run_artifact = yaml.safe_load(run_artifact_path.read_text())
    if (
        not isinstance(run_artifact, Mapping)
        or run_artifact.get("kind") != RUN_ARTIFACT_KIND
        or run_artifact.get("run_id") != manifest.get("run_id")
    ):
        raise ValueError(f"{run_artifact_path}: invalid run artifact")
    experiment = _mapping(
        run_artifact.get("experiment"), "run.experiment"
    )
    if (
        experiment.get("manifest") != _relative(root, path)
        or experiment.get("sha256") != file_sha256(path)
    ):
        raise ValueError(
            f"{path}: run artifact does not pin this experiment manifest"
        )
    return PreparedEvaluation(
        path,
        manifest,
        engine_path,
        run_artifact_path,
        _repo_path(
            root,
            _text(outputs, "output_directory", "manifest.outputs"),
            "outputs.output_directory",
        ),
        _text(outputs, "tmux_session", "manifest.outputs"),
    )


def update_run_status(
    path: Path,
    status: str,
    *,
    exit_code: int | None = None,
    error: str | None = None,
    attempt: int | None = None,
    quota_wait: Mapping[str, Any] | None = None,
) -> None:
    """Atomically update only the lifecycle portion of a run artifact."""

    allowed = {
        "configured",
        "running",
        "complete",
        "quota_exhausted",
        "waiting_for_quota",
        "failed",
    }
    if status not in allowed:
        raise ValueError(f"unsupported run status {status!r}")
    artifact = yaml.safe_load(path.read_text())
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("kind") != RUN_ARTIFACT_KIND
    ):
        raise ValueError(f"{path}: invalid run artifact")
    updated = dict(artifact)
    updated["status"] = status
    lifecycle = dict(
        _mapping(updated.get("lifecycle", {}), "run.lifecycle")
    )
    if status == "running":
        started_at = _utc_now()
        lifecycle.setdefault("started_at", started_at)
        lifecycle["last_attempt_started_at"] = started_at
        lifecycle.pop("finished_at", None)
    elif status == "waiting_for_quota":
        lifecycle["quota_wait_count"] = int(
            lifecycle.get("quota_wait_count", 0)
        ) + 1
        lifecycle["last_quota_at"] = _utc_now()
        lifecycle.pop("finished_at", None)
    elif status in {"complete", "quota_exhausted", "failed"}:
        lifecycle["finished_at"] = _utc_now()
    if exit_code is not None:
        lifecycle["exit_code"] = int(exit_code)
    if attempt is not None:
        lifecycle["attempt_count"] = int(attempt)
    if quota_wait is not None:
        lifecycle["quota_wait"] = dict(quota_wait)
    elif status == "complete":
        lifecycle.pop("quota_wait", None)
    if error:
        lifecycle["error"] = error[:2000]
    else:
        lifecycle.pop("error", None)
    updated["lifecycle"] = lifecycle
    _atomic_yaml(path, updated)
