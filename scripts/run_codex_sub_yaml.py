#!/usr/bin/env python3
"""Launch a subscription-backed agent experiment from a complete YAML manifest."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from leanlean.identifiers import same_identifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import generate as runner  # noqa: E402

from leanlean.capture_disposition import capture_is_scoring
from leanlean.palomar_comparator import resolve_palomar_contract
from leanlean.repo_variants import repo_image_tag, repo_variant_from_spec
from leanlean.evaluation_images import (
    EPHEMERAL_IMAGES_ENV,
    MATERIALIZATION_ENV,
    RECEIPTS_ENV,
)
from leanlean.environment_archives import validated_archive
from leanlean.preprocessing.repositories import (
    file_sha256,
    load_repository_database,
)

from leanlean.preprocessing.standardized_repositories import (
    load_standardized_repository_database,
    source_tree_sha256,
)
from leanlean.shared_cache import artifact_tree_sha256
from leanlean.subscription_status import (
    PROVIDER_FAILURE,
    QUOTA_EXIT_CODE,
    QUOTA_FAILURE,
    find_subscription_failures,
)
REQUIRED = {"model", "reasoning_effort", "generator", "run_id", "repos"}
SUPPORTED_GENERATORS = {
    "muse_code_native",
    "muse_code_core",
    "codex_sub",
    "codex_sub_mcp",
    "codex_app_server_sub",
    "claude_code_sub",
    "claude_code_glm",
    "mistral_vibe_lean",
    "antigravity_cli",
}
FORWARDED_DEFAULTS = {
    "workers": 3,
    "instance_start_interval_seconds": 0,
    "container_timeout": "2h",
    "build_jobs": 12,
    "container_cpus": 12,
    "container_memory": "64g",
    "container_cgroup_parent": "lean.slice",
    "image_resolution": "current_tag",
    "discard_cache_after_eval": False,
    "container_pids_limit": 4096,
    "task_metadata_enabled": True,
    "enable_lean_verify": True,
    "enable_proof_length": True,
    "max_api_retries": 10,
    "shuffle": False,
}
FIXED_SETTINGS = {
    "plan_type": "no_plan",
    "task_variant": "fix",
    "benchmark": "leanlean",
    "dataset_name": "leanlean",
    "refactor_rounds": 0,
    "progress": "rich",
    "network_policy": "model_proxy_only",
}
ALLOWED = (
    REQUIRED
    | FORWARDED_DEFAULTS.keys()
    | FIXED_SETTINGS.keys()
    | {
        "repo_variant",
        "repo_images",
        "use_prod",
        "prod_images",
        "repository_database",
        "protected_results_database",
        "playback",
        "harness_version",
        "codex_model_catalog",
        "generation_parameters",
        "pricing",
        "trajectory",
        "agent_instructions",
        "agent",
        "agent_validation",
        "repository_task_prompts",
        "repository_reproof_targets",
        "container_resource_visibility",
        "container_grace_seconds",
        "output_directory",
    }
)


def _theorem_reconstruction_verifier(
    repository,
    *,
    build_jobs: int,
) -> dict[str, object] | None:
    provenance = repository.protected_provenance
    if provenance.get("kind") != "leanlean_theorem_reconstruction_v1":
        return None
    declarations = sorted(repository.protected_declarations)
    if len(declarations) != 1:
        raise ValueError(
            f"{repository.instance_id}: reconstruction requires one protected theorem"
        )
    required = {"kind", "module", "target_file", "build_target", "permitted_axioms"}
    if (set(provenance) - {"edit_policy"} != required
            or provenance.get("edit_policy", "target_file_only") not in {"target_file_only", "repository_wide"}):
        raise ValueError(
            f"{repository.instance_id}: invalid reconstruction provenance fields"
        )
    permitted = provenance.get("permitted_axioms")
    if (
        not isinstance(permitted, list)
        or any(not isinstance(name, str) or not name for name in permitted)
        or len(permitted) != len(set(permitted))
    ):
        raise ValueError(
            f"{repository.instance_id}: invalid reconstruction axiom policy"
        )
    return {
        "schema": "leanlean_theorem_reconstruction_verifier_v1",
        "module": str(provenance["module"]),
        "declaration": declarations[0],
        "target_file": str(provenance["target_file"]),
        "build_target": str(provenance["build_target"]),
        "build_jobs": build_jobs,
        "permitted_axioms": list(permitted),
        "edit_policy": provenance.get("edit_policy", "target_file_only"),
    }


def _resolve_repository_database(path: Path, spec: dict) -> None:
    """Resolve one pinned repository snapshot into ordinary repo image pins."""

    record = spec.get("repository_database")
    if record is None:
        return
    if not isinstance(record, dict):
        raise ValueError(f"{path}: repository_database must be a mapping")
    required = {
        "path", "sha256", "definition_sha256", "dataset_id",
        "dataset_version",
    }
    if set(record) != required:
        raise ValueError(
            f"{path}: repository_database must contain exactly "
            f"{', '.join(sorted(required))}"
        )
    if spec.get("repo_images") is not None or spec.get("prod_images") is not None:
        raise ValueError(
            f"{path}: repository_database is the sole image-pin source; "
            "remove repo_images/prod_images"
        )
    database = load_repository_database(
        Path(record["path"]), repo_root=REPO_ROOT
    )
    checks = {
        "sha256": database.sha256 == record["sha256"],
        "definition_sha256": (
            database.definition_sha256 == record["definition_sha256"]
        ),
        "dataset_id": database.dataset_id == record["dataset_id"],
        "dataset_version": database.dataset_version == record["dataset_version"],
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: repository database drift: {checks}")
    expected_repos = [
        repository.instance_id for repository in database.repositories
    ]
    if spec["repos"] != expected_repos:
        raise ValueError(
            f"{path}: repos must exactly match the repository database order"
        )
    repo_images: dict[str, dict[str, str]] = {}
    repository_contracts: dict[str, dict[str, object]] = {}
    repository_verifiers: dict[str, dict[str, object]] = {}
    for repository in database.repositories:
        image = repository.variants.get(spec["repo_variant"])
        if image is None:
            raise ValueError(
                f"{path}: {repository.instance_id} has no pinned "
                f"{spec['repo_variant']} variant"
            )
        entry = {"tag": str(image["tag"]), "image_id": str(image["image_id"])}
        if repository.needs_raw_preparation:
            entry["source_isolation"] = "standalone_v2"
        repo_images[repository.instance_id] = entry
        source = repository.protected_provenance.get("kind")
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"{path}: {repository.instance_id} has invalid protected provenance"
            )
        repository_contracts[repository.instance_id] = {
            "declarations": sorted(repository.protected_declarations),
            "source": source,
        }
        if source in {"palomar_registry_main_results", "leanlean_theorem_holdout_v1"}:
            repository_verifiers[repository.instance_id] = resolve_palomar_contract(
                repo_root=REPO_ROOT,
                database_row={
                    "instance_id": repository.instance_id,
                    "protected": {
                        "declarations": sorted(
                            repository.protected_declarations
                        ),
                        "provenance": dict(repository.protected_provenance),
                    },
                },
            )
        elif source == "leanlean_theorem_reconstruction_v1":
            verifier = _theorem_reconstruction_verifier(
                repository, build_jobs=int(spec["build_jobs"])
            )
            assert verifier is not None
            repository_verifiers[repository.instance_id] = verifier
    spec["repo_images"] = repo_images

    spec["_repository_contracts"] = repository_contracts
    spec["_repository_verifiers"] = repository_verifiers


def _resolve_protected_results_database(path: Path, spec: dict) -> None:
    """Resolve protected names without changing separately pinned images."""

    record = spec.get("protected_results_database")
    if record is None:
        return
    if not isinstance(record, dict):
        raise ValueError(
            f"{path}: protected_results_database must be a mapping"
        )
    required = {
        "path", "sha256", "definition_sha256", "dataset_id",
        "dataset_version",
    }
    if set(record) != required:
        raise ValueError(
            f"{path}: protected_results_database must contain exactly "
            f"{', '.join(sorted(required))}"
        )
    database_path = Path(record["path"])
    resolved = (
        database_path
        if database_path.is_absolute()
        else REPO_ROOT / database_path
    )
    try:
        kind = json.loads(resolved.read_text()).get("kind")
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(
            f"{path}: could not read protected results database"
        ) from error
    if same_identifier(kind, "leanlean_standardized_repository_database"):
        database = load_standardized_repository_database(
            database_path, repo_root=REPO_ROOT
        )
    elif kind == "leanlean_repository_database":
        database = load_repository_database(
            database_path, repo_root=REPO_ROOT
        )
    else:
        raise ValueError(f"{path}: unsupported protected results database kind")
    checks = {
        "sha256": database.sha256 == record["sha256"],
        "definition_sha256": (
            database.definition_sha256 == record["definition_sha256"]
        ),
        "dataset_id": database.dataset_id == record["dataset_id"],
        "dataset_version": database.dataset_version == record["dataset_version"],
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: protected results database drift: {checks}")
    repositories = database.by_id()
    missing = [name for name in spec["repos"] if name not in repositories]
    if missing:
        raise ValueError(
            f"{path}: protected results database is missing: {', '.join(missing)}"
        )
    contracts: dict[str, dict[str, object]] = {}
    verifiers: dict[str, dict[str, object]] = {}
    repository_entries: list[dict[str, object]] = []
    task_prompts = spec.get("repository_task_prompts")
    if task_prompts is not None:
        if (
            not isinstance(task_prompts, dict)
            or set(task_prompts) != set(spec["repos"])
            or any(not isinstance(value, str) or not value.strip() for value in task_prompts.values())
        ):
            raise ValueError(f"{path}: repository_task_prompts must cover every repository")
    reproof_targets = spec.get("repository_reproof_targets")
    if reproof_targets is not None:
        if (
            not isinstance(reproof_targets, dict)
            or set(reproof_targets) != set(spec["repos"])
            or any(
                not isinstance(names, list)
                or not names
                or len(names) != len(set(names))
                or any(not isinstance(name, str) or not name for name in names)
                for names in reproof_targets.values()
            )
        ):
            raise ValueError(
                f"{path}: repository_reproof_targets must cover every repository "
                "with unique declaration names"
            )
    for instance_id in spec["repos"]:
        repository = repositories[instance_id]
        source = repository.protected_provenance.get("kind")
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"{path}: {instance_id} has invalid protected provenance"
            )
        contracts[instance_id] = {
            "declarations": sorted(repository.protected_declarations),
            "source": source,
        }
        if source in {"palomar_registry_main_results", "leanlean_theorem_holdout_v1"}:
            provenance = dict(repository.protected_provenance)
            if reproof_targets is not None:
                provenance["reconstruction_targets"] = list(
                    reproof_targets[instance_id]
                )
            verifiers[instance_id] = resolve_palomar_contract(
                repo_root=REPO_ROOT,
                database_row={
                    "instance_id": instance_id,
                    "protected": {
                        "declarations": sorted(
                            repository.protected_declarations
                        ),
                        "provenance": provenance,
                    },
                },
            )
        elif source == "leanlean_theorem_reconstruction_v1":
            verifier = _theorem_reconstruction_verifier(
                repository, build_jobs=int(spec["build_jobs"])
            )
            assert verifier is not None
            verifiers[instance_id] = verifier
        family = str(getattr(repository, "family", "standalone"))
        entry = {
            "instance_id": instance_id,
            "repo_url": repository.repository_url,
            "commit": repository.commit,
            "exclude_dirs": list(repository.exclude_dirs),
            "target_dir": repository.target_dir,
            "build_target": repository.build_target,
            "filesystem_isolated": (
                family != "leanpool_member"
                or spec["repo_variant"] != "raw"
            ),
        }
        if task_prompts is not None:
            entry["task_prompt"] = task_prompts[instance_id]
        repository_entries.append(entry)
    existing = spec.get("_repository_contracts")
    if existing is not None and existing != contracts:
        raise ValueError(f"{path}: protected result contract sources disagree")
    spec["_repository_contracts"] = contracts
    existing_verifiers = spec.get("_repository_verifiers")
    if existing_verifiers is not None and existing_verifiers != verifiers:
        raise ValueError(f"{path}: registered verifier contract sources disagree")
    spec["_repository_verifiers"] = verifiers
    spec["_repository_entries"] = repository_entries


def _validate_repo_images(path: Path, spec: dict) -> None:
    repo_images = spec.get("repo_images")
    legacy_images = spec.get("prod_images")
    if repo_images is not None and legacy_images is not None:
        raise ValueError(f"{path}: use repo_images or prod_images, not both")

    field = "repo_images"
    if repo_images is None and legacy_images is not None:
        if spec["repo_variant"] != "optimized":
            raise ValueError(
                f"{path}: legacy prod_images is valid only for optimized runs"
            )
        repo_images = legacy_images
        field = "prod_images"
    if repo_images is None:
        return
    if not isinstance(repo_images, dict) or set(repo_images) != set(spec["repos"]):
        raise ValueError(
            f"{path}: {field} must contain exactly: {', '.join(spec['repos'])}"
        )

    image_resolution = spec.get("image_resolution", "current_tag")
    if image_resolution not in {
        "current_tag",
        "pinned_id",
        "build_before_agent",
    }:
        raise ValueError(
            f"{path}: image_resolution must be current_tag, pinned_id, "
            "or build_before_agent"
        )

    for repo in spec["repos"]:
        entry = repo_images[repo]
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {field}.{repo} must be a mapping")
        requires_standalone = (
            repo.startswith("leanpool__")
            and spec["repo_variant"] != "raw"
        )

        if image_resolution == "build_before_agent":
            safe_run = re.sub(
                r"[^A-Za-z0-9_.-]", "-", str(spec["run_id"])
            )
            expected_tag = (
                f"leanlean-{repo}-{spec['repo_variant']}:"
                f"eval-{safe_run}"
            )
            base_required = {
                "tag",
                "source_tree",
                "tree_sha256",
                "build_target",
                "build_jobs",
                "build_timeout_seconds",
                "persist",
                "source_isolation",
            }
            backend = entry.get("backend", "dockerfile_network_v1")
            required = set(base_required)
            if backend == "shared_environment_v1":
                required.update(
                    {"backend", "shared_environment", "warm_build_cache"}
                )
            elif backend != "dockerfile_network_v1":
                raise ValueError(
                    f"{path}: unsupported materialization backend for {repo}: "
                    f"{backend!r}"
                )
            # Reconciliation repositories carry two extra seed patches that the
            # image replays into `main` and `other-refactor`.
            if "reconciliation" in entry:
                required.add("reconciliation")
            if backend == "dockerfile_network_v1":
                optional_build_fields = {"backend", "repo_variant", "cache", "minimum_free_disk_gib"}
                required.update(optional_build_fields & set(entry))
                if entry.get("cache", "disabled") != "disabled":
                    raise ValueError(f"{path}: isolated source build cache must be disabled")
                if entry.get("repo_variant", spec["repo_variant"]) != spec["repo_variant"]:
                    raise ValueError(f"{path}: materialization repository variant drift")
                free_gib = entry.get("minimum_free_disk_gib", 1)
                if isinstance(free_gib, bool) or not isinstance(free_gib, int) or free_gib < 1:
                    raise ValueError(f"{path}: invalid minimum_free_disk_gib")
            if set(entry) != required:
                raise ValueError(
                    f"{path}: {field}.{repo} materialization fields drift"
                )
            source_tree = (REPO_ROOT / str(entry["source_tree"])).resolve()
            expected_tree = str(entry["tree_sha256"])
            checks = {
                "source_backed_variant": spec["repo_variant"]
                in {"stripped", "optimized"},
                "tag": entry["tag"] == expected_tag,
                "source_tree_scope": any(
                    source_tree.is_relative_to(candidate)
                    for candidate in (
                        (REPO_ROOT / "prod_repos").resolve(),
                        (REPO_ROOT / "datasets").resolve(),
                    )
                ),
                "source_tree_exists": source_tree.is_dir(),
                "tree_sha256": bool(
                    re.fullmatch(r"[0-9a-f]{64}", expected_tree)
                ),
                "build_jobs": (
                    isinstance(entry["build_jobs"], int)
                    and not isinstance(entry["build_jobs"], bool)
                    and entry["build_jobs"] > 0
                ),
                "build_timeout_seconds": (
                    isinstance(entry["build_timeout_seconds"], int)
                    and not isinstance(entry["build_timeout_seconds"], bool)
                    and entry["build_timeout_seconds"] > 0
                ),
                "persist": isinstance(entry["persist"], bool),
                "source_isolation": entry["source_isolation"]
                == (
                    "standalone_v3"
                    if backend == "shared_environment_v1"
                    else "standalone_v2"
                ),
            }
            if checks["source_tree_scope"] and checks["source_tree_exists"]:
                checks["source_tree_digest"] = (
                    source_tree_sha256(source_tree) == expected_tree
                )
            if backend == "shared_environment_v1":
                environment = entry["shared_environment"]
                warm = entry["warm_build_cache"]
                shared_valid = (
                    isinstance(environment, dict)
                    and set(environment)
                    == {"id", "image", "image_id", "archive"}
                    and isinstance(environment.get("id"), str)
                    and bool(environment["id"])
                    and isinstance(environment.get("image"), str)
                    and bool(environment["image"])
                    and bool(
                        re.fullmatch(
                            r"sha256:[0-9a-f]{64}",
                            str(environment.get("image_id") or ""),
                        )
                    )
                    and isinstance(warm, dict)
                    and set(warm) == {"path", "sha256"}
                )
                if shared_valid:
                    try:
                        archive_path = validated_archive(
                            environment, repo_root=REPO_ROOT
                        )
                        archive = environment["archive"]
                        warm_path = (REPO_ROOT / str(warm["path"])).resolve()
                        warm_sha = str(warm["sha256"])
                        shared_valid = (
                            isinstance(archive.get("bytes"), int)
                            and not isinstance(archive["bytes"], bool)
                            and archive["bytes"] == archive_path.stat().st_size
                            and warm_path.is_relative_to(
                                (REPO_ROOT / "datasets").resolve()
                            )
                            and warm_path.is_dir()
                            and bool(re.fullmatch(r"[0-9a-f]{64}", warm_sha))
                            and artifact_tree_sha256(warm_path) == warm_sha
                        )
                    except (OSError, ValueError):
                        shared_valid = False
                checks["shared_environment_artifacts"] = shared_valid
            if not all(checks.values()):
                raise ValueError(
                    f"{path}: source-tree materialization drift for {repo}: "
                    f"{checks}"
                )
            continue

        expected_tag = (
            str(entry.get("tag") or "")
            if image_resolution == "pinned_id"
            else (
                f"leanlean-{repo}-prod:latest"
                if field == "prod_images"
                else repo_image_tag(repo, spec["repo_variant"])
            )
        )
        isolation_claim = entry.get("source_isolation")
        isolation_claim_valid = isolation_claim == "standalone_v2" or (
            image_resolution == "pinned_id"
            and isolation_claim == "standalone_v1"
        )
        tag_valid = bool(expected_tag) and (
            image_resolution == "pinned_id"
            or entry.get("tag") == expected_tag
        )
        manifest_pin = entry.get("manifest")
        if manifest_pin is not None:
            if not isinstance(manifest_pin, dict) or set(manifest_pin) != {
                "path",
                "sha256",
            }:
                raise ValueError(
                    f"{path}: {field}.{repo}.manifest must pin path and sha256"
                )
            manifest_path = (REPO_ROOT / str(manifest_pin["path"])).resolve()
            if (
                not manifest_path.is_relative_to((REPO_ROOT / "datasets").resolve())
                or not manifest_path.is_file()
                or file_sha256(manifest_path) != manifest_pin["sha256"]
            ):
                raise ValueError(f"{path}: Docker image manifest drift for {repo}")
            image_manifest = json.loads(manifest_path.read_text())
            manifest_checks = {
                "kind": image_manifest.get("kind")
                == "leanlean_docker_image",
                "schema_version": image_manifest.get("schema_version") == 1,
                "repository": image_manifest.get("repository") == repo,
                "variant": image_manifest.get("variant") == spec["repo_variant"],
                "tag": image_manifest.get("tag") == entry.get("tag"),
                "image_id": image_manifest.get("image_id")
                == entry.get("image_id"),
            }
            if not all(manifest_checks.values()):
                raise ValueError(
                    f"{path}: Docker image manifest contract drift for {repo}: "
                    f"{manifest_checks}"
                )
        if (
            not tag_valid
            or not entry.get("image_id")
            or (requires_standalone and not isolation_claim_valid)
        ):
            requirement = (
                ", image_id, and source_isolation=standalone_v2 "
                "(or standalone_v1 with image_resolution=pinned_id)"
                if requires_standalone
                else " and its image_id"
            )
            raise ValueError(
                f"{path}: {field}.{repo} must pin tag {expected_tag!r}"
                f"{requirement}"
            )
        inspected = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                (
                    "{{.Id}}\t{{index .Config.Labels "
                    "\"org.openai.leanlean.source_isolation\"}}"
                ),
                entry["image_id"]
                if image_resolution == "pinned_id"
                else expected_tag,
            ],
            capture_output=True,
            text=True,
        )
        inspected_fields = inspected.stdout.strip().split("\t", maxsplit=1)
        actual_id = inspected_fields[0] if inspected_fields else ""
        actual_isolation = (
            inspected_fields[1] if len(inspected_fields) == 2 else ""
        )
        if inspected.returncode != 0 or actual_id != entry["image_id"]:
            raise ValueError(
                f"{path}: {spec['repo_variant']} image pin mismatch for {repo}: "
                f"expected {entry['image_id']}, found {actual_id or 'missing'}"
            )
        if (
            requires_standalone
            and isolation_claim == "standalone_v2"
            and actual_isolation != "standalone_v2"
        ):
            raise ValueError(
                f"{path}: {repo} image is not physically standalone_v2 "
                f"(found {actual_isolation or 'unlabeled'})"
            )
def load_manifest(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    missing = REQUIRED - raw.keys()
    if missing:
        raise ValueError(
            f"{path}: missing required field(s): {', '.join(sorted(missing))}"
        )
    unknown = raw.keys() - ALLOWED
    if unknown:
        raise ValueError(
            f"{path}: unknown field(s): {', '.join(sorted(unknown))}"
        )

    spec = {**FORWARDED_DEFAULTS, **FIXED_SETTINGS, **raw}
    if spec["generator"] not in SUPPORTED_GENERATORS:
        raise ValueError(
            f"{path}: generator={spec['generator']!r} is unsupported by the "
            "subscription runner; expected codex_sub, codex_sub_mcp, "
            "codex_app_server_sub, claude_code_sub, claude_code_glm, "
            "muse_code_native, muse_code_core, mistral_vibe_lean, or antigravity_cli"
        )
    if spec["generator"] == "antigravity_cli" and spec["model"] != (
        "gemini-3.8-flash"
    ):
        raise ValueError(
            f"{path}: generator=antigravity_cli requires gemini-3.8-flash"
        )
    if spec["model"].startswith("muse-spark-"):
        from leanlean.generators.muse_code import validate_reasoning_effort
        validate_reasoning_effort(spec["reasoning_effort"], spec["model"])
    if spec["generator"] in {"muse_code_native", "muse_code_core"} and not spec["model"].startswith("muse-spark-"):
        raise ValueError(f"{spec['generator']} requires a Muse Spark model")
    for key, expected in FIXED_SETTINGS.items():
        if spec[key] != expected:
            raise ValueError(
                f"{path}: {key}={spec[key]!r} is unsupported by the current "
                f"subscription runner; expected {expected!r}"
            )

    if not isinstance(spec["shuffle"], bool):
        raise ValueError(f"{path}: shuffle must be true or false")

    for key in ("playback", "trajectory"):
        value = spec.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"{path}: {key} must be a mapping")
        spec[key] = value

    agent_validation = spec.get("agent_validation")
    if agent_validation is not None:
        expected_agent_validation = {
            "mode": "registered_palomar_comparator",
            "visible_contracts": ["Challenge.lean", "comparator.json"],
            "command": "lean_verify",
            "hidden_results_yaml": False,
            "pre_agent_green_verify": True,
            "prewarm_artifacts": "retained_in_agent_container",
            "usage_telemetry": [
                *(["proof_length"] if spec["enable_proof_length"] else []),
                "lean_verify",
            ],
            "staging": {
                "base_image": "dataset_declared_immutable_image",
                "source": "dataset_repository_comparator_directory",
                "timing": "after_container_start_before_agent",
                "files": [
                    "Challenge.lean",
                    "comparator.json",
                    *(
                        ["proof_length.py"]
                        if spec["enable_proof_length"]
                        else []
                    ),
                ],
            },
        }
        if agent_validation != expected_agent_validation:
            raise ValueError(
                f"{path}: unsupported agent_validation contract"
            )
        if not spec["enable_lean_verify"]:
            raise ValueError(
                f"{path}: agent_validation requires enable_lean_verify=true"
            )

    agent = spec.get("agent")
    if agent is not None:
        expected_keys = {
            "system_prompt",
            "prompt",
            "allowed_skills",
            "mcp_servers",
            "enable_subagents",
            "enable_lean_verify",
            "enable_proof_length",
        }
        if not isinstance(agent, dict) or set(agent) - {"native_tools"} != expected_keys:
            raise ValueError(f"{path}: unsupported agent contract")
        if "native_tools" in agent and spec["generator"] != "claude_code_sub":
            raise ValueError(f"{path}: agent.native_tools is only supported for claude_code_sub")
        if agent["system_prompt"] != "native":
            raise ValueError(f"{path}: agent.system_prompt must be 'native'")
        if not isinstance(agent["prompt"], str) or not agent["prompt"].strip():
            raise ValueError(f"{path}: agent.prompt must be a non-empty string")
        if agent["allowed_skills"] != [] or agent["mcp_servers"] != []:
            raise ValueError(
                f"{path}: skills and MCP servers must remain disabled"
            )
        for key in (
            "enable_subagents",
            "enable_lean_verify",
            "enable_proof_length",
        ):
            if not isinstance(agent[key], bool):
                raise ValueError(f"{path}: agent.{key} must be true or false")
        if (
            spec["enable_lean_verify"] != agent["enable_lean_verify"]
            or spec["enable_proof_length"] != agent["enable_proof_length"]
        ):
            raise ValueError(f"{path}: agent tool toggles disagree with manifest")
        if spec["task_metadata_enabled"] != (
            spec["enable_lean_verify"] or spec["enable_proof_length"]
        ):
            raise ValueError(f"{path}: inconsistent task metadata toggles")

    playback_mode = str(spec["playback"].get("mode", "external_replay"))
    supported_playback_modes = {
        "external_replay",
        "host_side_checkpointing",
        "in_container",
        "passive_action_replay",
        "standardized_replay",
    }
    if playback_mode not in supported_playback_modes:
        raise ValueError(
            f"{path}: playback.mode={playback_mode!r} is unsupported"
        )
    if playback_mode == "passive_action_replay":
        if spec["generator"] != "claude_code_sub":
            raise ValueError(
                f"{path}: passive_action_replay requires generator=claude_code_sub"
            )
        if str(spec.get("agent_instructions") or "").strip():
            raise ValueError(
                f"{path}: passive_action_replay forbids agent_instructions so the "
                "benchmark prompt remains untouched"
            )
        if spec["playback"].get("capture_snapshots", True) is not True:
            raise ValueError(
                f"{path}: passive_action_replay requires capture_snapshots=true"
            )
    if playback_mode == "host_side_checkpointing":
        if spec["generator"] == "codex_app_server_sub":
            raise ValueError(
                f"{path}: host_side_checkpointing requires a streaming CLI "
                "generator with host overlay access; the model and provider "
                "are unrestricted"
            )
        if spec["playback"].get("capture_snapshots", True) is not True:
            raise ValueError(
                f"{path}: host_side_checkpointing requires capture_snapshots=true"
            )
        if spec["playback"].get("capture_repository_scope") is not True:
            raise ValueError(
                f"{path}: host_side_checkpointing requires "
                "capture_repository_scope=true"
            )
        interval = float(
            spec["playback"].get("reconciliation_interval_seconds", 30)
        )
        if interval <= 0:
            raise ValueError(
                f"{path}: host_side_checkpointing requires a positive "
                "reconciliation_interval_seconds"
            )
        if (
            spec["generator"] in {"codex_sub", "codex_sub_mcp"}
            and spec["trajectory"].get("capture_native_rollout") is not True
        ):
            raise ValueError(
                f"{path}: Codex host_side_checkpointing requires "
                "trajectory.capture_native_rollout=true for exact completed-turn "
                "cost boundaries"
            )
    if playback_mode == "standardized_replay":
        if spec["generator"] not in {"claude_code_sub", "codex_sub"}:
            raise ValueError(
                f"{path}: standardized_replay requires claude_code_sub or codex_sub"
            )
        if str(spec.get("agent_instructions") or "").strip():
            raise ValueError(
                f"{path}: standardized_replay forbids agent_instructions so the "
                "benchmark prompt remains untouched"
            )
        if spec["playback"].get("capture_snapshots", True) is not True:
            raise ValueError(
                f"{path}: standardized_replay requires capture_snapshots=true"
            )

    repos = spec["repos"]
    if isinstance(repos, str):
        repos = [repos]
    if not isinstance(repos, list) or not repos or not all(
        isinstance(repo, str) and repo for repo in repos
    ):
        raise ValueError(f"{path}: repos must be a non-empty string list")
    spec["repos"] = repos

    spec["repo_variant"] = repo_variant_from_spec(spec)
    if not isinstance(spec["task_metadata_enabled"], bool):
        raise ValueError(
            f"{path}: task_metadata_enabled must be true or false"
        )
    for key in ("enable_lean_verify", "enable_proof_length"):
        if not isinstance(spec[key], bool):
            raise ValueError(f"{path}: {key} must be true or false")
    if spec.get("container_resource_visibility", "host") not in {"host", "container_v1"}:
        raise ValueError(f"{path}: unsupported container_resource_visibility")
    if spec.get("container_resource_visibility") == "container_v1":
        if int(spec["build_jobs"]) != int(spec["container_cpus"]) or int(spec["build_jobs"]) < 1:
            raise ValueError(f"{path}: container_v1 requires matching positive CPU and thread counts")
    if int(spec["container_pids_limit"]) < 1:
        raise ValueError(f"{path}: container_pids_limit must be positive")
    if float(spec["instance_start_interval_seconds"]) < 0:
        raise ValueError(
            f"{path}: instance_start_interval_seconds must be non-negative"
        )
    if isinstance(spec["max_api_retries"], bool):
        raise ValueError(f"{path}: max_api_retries must be a non-negative integer")
    try:
        max_api_retries = int(spec["max_api_retries"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: max_api_retries must be a non-negative integer"
        ) from exc
    if max_api_retries < 0:
        raise ValueError(f"{path}: max_api_retries must be a non-negative integer")
    spec["max_api_retries"] = max_api_retries

    generator_config = runner.ALL_GENERATOR_CONFIGS[spec["generator"]]
    if not generator_config.get("requires_model_proxy", True):
        raise ValueError(
            f"{path}: generator={spec['generator']!r} bypasses the model-only "
            "relay and is disabled by the subscription runner"
        )
    bootstrap = "\n".join(
        str(command)
        for key in ("install_commands", "post_install_commands")
        for command in generator_config.get(key, [])
    ).lower()
    network_bootstrap_markers = ("curl ", "wget ", "git clone", "apt-get ", "npm ")
    if any(marker in bootstrap for marker in network_bootstrap_markers):
        raise ValueError(
            f"{path}: generator={spec['generator']!r} needs network bootstrap "
            "inside the task container; prepackage its tools before enabling it"
        )
    approved_bundle = {
        "codex_sub": "codex_standalone",
        "codex_sub_mcp": "codex_standalone",
        "claude_code_sub": "claude_standalone",
        "claude_code_glm": "claude_standalone",
        "mistral_vibe_lean": "mistral_vibe_standalone",
        "antigravity_cli": "antigravity_standalone",
        "muse_code_native": "muse_standalone",
        "muse_code_core": "muse_standalone",
    }.get(spec["generator"])
    if generator_config.get("host_tool_bundle") != approved_bundle:
        raise ValueError(
            f"{path}: generator={spec['generator']!r} has no approved offline "
            "host tool bundle"
        )
    _validate_capture_backend(path, spec)
    _resolve_repository_database(path, spec)
    _resolve_protected_results_database(path, spec)
    if (
        (spec.get("agent_validation") or {}).get("mode")
        == "registered_palomar_comparator"
        and set(spec.get("_repository_verifiers", {})) != set(spec["repos"])
    ):
        raise ValueError(
            f"{path}: every repository requires a registered Palomar Comparator"
        )
    _validate_repo_images(path, spec)
    _run_directory(spec)
    return spec


def _validate_capture_backend(path: Path, spec: dict) -> None:
    playback = spec["playback"]
    backend = str(playback.get("capture_backend", "leanlean"))
    if backend not in {"leanlean", "trace_utils"}:
        raise ValueError(
            f"{path}: playback.capture_backend must be leanlean or trace_utils"
        )
    if backend != "trace_utils":
        return
    revision = str(playback.get("trace_utils_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            f"{path}: trace_utils capture requires a pinned 40-character "
            "playback.trace_utils_revision"
        )
    try:
        module = importlib.import_module("harness_wrapper.native_capture")
    except ImportError as exc:
        raise ValueError(
            f"{path}: trace_utils capture requires harness-wrapper on PYTHONPATH"
        ) from exc
    module_path = Path(module.__file__).resolve()
    repository = subprocess.run(
        ["git", "-C", str(module_path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if repository.returncode != 0:
        raise ValueError(
            f"{path}: harness-wrapper is not loaded from a pinned Git checkout"
        )
    actual = subprocess.run(
        ["git", "-C", repository.stdout.strip(), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if actual.returncode != 0 or actual.stdout.strip() != revision:
        raise ValueError(
            f"{path}: trace-utils revision mismatch: expected {revision}, "
            f"found {actual.stdout.strip() or 'unavailable'}"
        )


def _filter_spec(repos: list[str]) -> str:
    return "^(?:" + "|".join(re.escape(repo) for repo in repos) + ")$"


def _run_directory(spec: dict) -> Path:
    configured = spec.get("output_directory")
    if configured is not None:
        path = Path(str(configured))
        path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
        output_root = (REPO_ROOT / "output").resolve()
        if not path.is_relative_to(output_root):
            raise ValueError("output_directory must stay under output/")
        return path
    return (
        REPO_ROOT
        / "output"
        / spec["benchmark"]
        / spec["dataset_name"].replace("/", "_")
        / spec["task_variant"]
        / spec["generator"]
        / spec["model"]
        / f"run_{spec['run_id']}"
    )


def _write_subscription_status(spec: dict) -> int:
    """Persist and return the provider-aware outcome of a completed run."""

    run_dir = _run_directory(spec)
    failures = find_subscription_failures(run_dir, spec["repos"])
    kinds = {failure.kind for failure in failures}
    if PROVIDER_FAILURE in kinds:
        status = PROVIDER_FAILURE
        exit_code = 1
    elif QUOTA_FAILURE in kinds:
        status = QUOTA_FAILURE
        exit_code = QUOTA_EXIT_CODE
    else:
        status = "complete"
        exit_code = 0
    payload = {
        "format": "leanlean-subscription-status-v1",
        "run_id": spec["run_id"],
        "status": status,
        "exit_code": exit_code,
        "failures": [
            {
                "instance_id": failure.instance_id,
                "kind": failure.kind,
                "detail": failure.detail,
            }
            for failure in failures
        ],
    }
    (run_dir / "subscription_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    if status == QUOTA_FAILURE:
        print(
            "Subscription quota was exhausted for: "
            + ", ".join(failure.instance_id for failure in failures)
        )
    elif status == PROVIDER_FAILURE:
        print(
            "Subscription provider failed for: "
            + ", ".join(failure.instance_id for failure in failures)
        )
    return exit_code


def _assert_complete_generation(spec: dict) -> None:
    """Require predictions, plus captures only when monitoring is enabled."""

    run_dir = _run_directory(spec)
    try:
        predictions = json.loads((run_dir / "preds.json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"evaluation produced no valid predictions at {run_dir}"
        ) from exc
    if not isinstance(predictions, dict):
        raise RuntimeError(f"predictions at {run_dir} are not a mapping")

    expected = set(spec["repos"])
    predicted = set(predictions)
    missing_predictions = sorted(expected - predicted)
    unexpected_predictions = sorted(predicted - expected)
    malformed_predictions = sorted(
        instance_id
        for instance_id in expected & predicted
        if not isinstance(predictions[instance_id], dict)
        or not isinstance(predictions[instance_id].get("model_patch"), str)
    )

    captures: dict[str, list[Path]] = {}
    malformed_captures: list[str] = []
    capture_required = spec.get("playback", {}).get("enabled", True)
    capture_paths = sorted(run_dir.glob("*/playback/capture_*/playback.json")) if capture_required else []
    for path in capture_paths:
        if not capture_is_scoring(path):
            continue
        try:
            playback = json.loads(path.read_text())
        except (OSError, ValueError):
            malformed_captures.append(str(path.relative_to(run_dir)))
            continue
        instance_id = str(playback.get("instance_id") or path.parents[2].name)
        captures.setdefault(instance_id, []).append(path)
        points = playback.get("points")
        if (
            not isinstance(points, list)
            or not points
            or not isinstance(points[0], dict)
            or points[0].get("edit_index") != 0
        ):
            malformed_captures.append(str(path.relative_to(run_dir)))
            continue
        for point in points:
            if not isinstance(point, dict):
                malformed_captures.append(str(path.relative_to(run_dir)))
                break
            snapshot = point.get("snapshot_path")
            digest = point.get("source_archive_sha256")
            if not isinstance(snapshot, str) or not isinstance(digest, str):
                malformed_captures.append(str(path.relative_to(run_dir)))
                break
            archive = path.parent / Path(snapshot).name
            if not archive.is_file():
                malformed_captures.append(str(path.relative_to(run_dir)))
                break
        trace = path.parent / "standardized-trace.json"
        if not trace.is_file():
            malformed_captures.append(str(path.relative_to(run_dir)))

    captured = set(captures)
    missing_captures = sorted(expected - captured) if capture_required else []
    unexpected_captures = sorted(captured - expected)
    duplicate_captures = sorted(
        instance_id
        for instance_id in expected & captured
        if len(captures[instance_id]) != 1
    )
    malformed_captures = sorted(set(malformed_captures))
    problems: list[str] = []
    for label, values in (
        ("missing predictions", missing_predictions),
        ("unexpected predictions", unexpected_predictions),
        ("malformed predictions", malformed_predictions),
        ("missing monitored captures", missing_captures),
        ("unexpected monitored captures", unexpected_captures),
        ("duplicate monitored captures", duplicate_captures),
        ("malformed monitored captures", malformed_captures),
    ):
        if values:
            problems.append(f"{label}: {', '.join(values)}")
    if problems:
        raise RuntimeError(
            "generation completeness gate failed (" + "; ".join(problems) + ")"
        )


def _discard_run_cache_images(spec: dict) -> list[str]:
    """Remove only transient cache/base tags belonging to this exact run."""
    run_tag = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        f"{spec['generator']}_{spec['model']}_run_{spec['run_id']}",
    )
    repositories = {
        f"leanlean-{repo}-{suffix}"
        for repo in spec["repos"]
        for suffix in ("cache", "base")
    }
    listed = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}\t{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    references: list[str] = []
    for line in listed.stdout.splitlines():
        try:
            repository, tag = line.split("\t", 1)
        except ValueError:
            continue
        if repository in repositories and (
            tag == run_tag or tag.startswith(run_tag + "_")
        ):
            references.append(f"{repository}:{tag}")
    removed: list[str] = []
    failures: list[str] = []
    for reference in sorted(set(references)):
        result = subprocess.run(
            ["docker", "rmi", reference],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            removed.append(reference)
        else:
            failures.append(f"{reference}: {result.stderr.strip()}")
    if failures:
        raise RuntimeError(
            "Could not discard run cache image(s): " + "; ".join(failures)
        )
    return removed


def _model_config_for_spec(spec: dict) -> dict:
    """Resolve one isolated model config with explicit per-run reasoning."""

    try:
        model_config = deepcopy(runner.ALL_MODEL_CONFIGS[spec["model"]])
    except KeyError as exc:
        raise ValueError(f"unknown model {spec['model']!r}") from exc
    model_kwargs = model_config.setdefault("model_kwargs", {})
    effort = spec["reasoning_effort"]
    model_kwargs["reasoning_effort"] = effort
    if spec["generator"] in {"muse_code_native", "muse_code_core"} and effort == "max":
        # Muse Code sends xhigh for --reasoning-effort max; LiteLLM restores max.
        model_config["reasoning_effort_map"] = {"xhigh": "max"}
    return model_config


def run_experiment(spec: dict) -> int:
    """Generate predictions and monitored source checkpoints.

    The former ``run_codex_sub_experiment.py`` duplicated this orchestration
    and was removed during cleanup. Keep the strict manifest validation in
    this wrapper, but leave all compilation and scoring to postprocessing.
    """

    model_config = _model_config_for_spec(spec)
    generator_config = runner.ALL_GENERATOR_CONFIGS[spec["generator"]]
    original_host_tool_version = generator_config.get("host_tool_version")
    if spec.get("harness_version"):
        generator_config["host_tool_version"] = spec["harness_version"]
    original_codex_model_catalog = generator_config.get("codex_model_catalog")
    if spec.get("codex_model_catalog"):
        if spec["generator"] != "codex_sub":
            raise ValueError("codex_model_catalog requires the codex_sub generator")
        catalog = (REPO_ROOT / spec["codex_model_catalog"]["path"]).resolve()
        digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
        if digest != spec["codex_model_catalog"]["sha256"]:
            raise ValueError(f"{catalog}: codex_model_catalog sha256 mismatch")
        generator_config["codex_model_catalog"] = str(catalog)
    original_instance_template = generator_config.get(
        "instance_template", "{{task}}"
    )
    agent_instructions = str(spec.get("agent_instructions") or "").strip()
    if agent_instructions:
        generator_config["instance_template"] = (
            f"{original_instance_template}\n\n{agent_instructions}"
        )
    filter_spec = _filter_spec(spec["repos"])

    try:
        runner.main(
            plan_type=spec["plan_type"],
            exec_model=spec["model"],
            exec_model_config=model_config,
            generator=spec["generator"],
            task_variant=spec["task_variant"],
            run_id=spec["run_id"],
            run_output_directory=str(_run_directory(spec)),
            dataset_name=spec["dataset_name"],
            benchmark=spec["benchmark"],
            filter_spec=filter_spec,
            slice_spec=f"0:{len(spec['repos'])}",
            shuffle=spec["shuffle"],
            workers=int(spec["workers"]),
            instance_start_interval_seconds=float(
                spec["instance_start_interval_seconds"]
            ),
            progress=spec["progress"],
            plan_args={"refactor_rounds": spec["refactor_rounds"]},
            build_jobs=int(spec["build_jobs"]),
            container_cpus=int(spec["container_cpus"]),
            container_memory=str(spec["container_memory"]),
            container_cgroup_parent=str(spec["container_cgroup_parent"]),
            container_timeout=str(spec["container_timeout"]),
            network_policy=str(spec["network_policy"]),
            task_metadata_enabled=spec["task_metadata_enabled"],
            enable_lean_verify=spec["enable_lean_verify"],
            enable_proof_length=spec["enable_proof_length"],
            agent=spec.get("agent"),
            container_pids_limit=int(spec["container_pids_limit"]),
            container_resource_visibility=str(spec.get("container_resource_visibility", "host")),
            container_grace_seconds=int(spec.get("container_grace_seconds", 1800)),
            max_api_retries=int(spec["max_api_retries"]),
            repository_contracts=spec.get(
                "_repository_contracts", {}
            ),
            repository_verifiers=spec.get(
                "_repository_verifiers", {}
            ),
            repository_entries=spec.get(
                "_repository_entries", []
            ),
            playback=spec["playback"],
            trajectory=spec["trajectory"],
        )
    finally:
        generator_config["host_tool_version"] = original_host_tool_version
        generator_config["instance_template"] = original_instance_template
        if original_codex_model_catalog is None:
            generator_config.pop("codex_model_catalog", None)
        else:
            generator_config["codex_model_catalog"] = original_codex_model_catalog

    if find_subscription_failures(_run_directory(spec), spec["repos"]):
        return _write_subscription_status(spec)

    _assert_complete_generation(spec)
    if spec["discard_cache_after_eval"]:
        removed = _discard_run_cache_images(spec)
        print(f"Discarded {len(removed)} run-specific cache image(s).")
    return _write_subscription_status(spec)



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved command without launching",
    )
    args = parser.parse_args()

    try:
        spec = load_manifest(args.manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    variant_labels = {
        "raw": "raw upstream",
        "stripped": "tree-shaken only",
        "optimized": "tree-shaken plus simple-proof normalization",
    }
    print(f"Manifest: {args.manifest}")
    print(
        f"Repositories: {variant_labels[spec['repo_variant']]} "
        f"({', '.join(spec['repos'])})"
    )
    print(
        "Resolved: "
        f"model={spec['model']} effort={spec['reasoning_effort']} "
        f"generator={spec['generator']} workers={spec['workers']} "
        f"start_interval={spec['instance_start_interval_seconds']}s "
        f"refactor_rounds={spec['refactor_rounds']} "
        f"timeout={spec['container_timeout']} "
        f"network={spec['network_policy']} "
        f"pids={spec['container_pids_limit']} "
        f"max_api_retries={spec['max_api_retries']} run_id={spec['run_id']}"
    )
    if args.dry_run:
        return 0

    os.environ["LEANLEAN_REPO_VARIANT"] = spec["repo_variant"]
    os.environ["LEANLEAN_USE_PROD"] = (
        "0" if spec["repo_variant"] == "raw" else "1"
    )
    override_env = "LEANLEAN_REPO_IMAGE_OVERRIDES"
    resolution = spec["image_resolution"]
    if resolution == "pinned_id":
        os.environ[override_env] = json.dumps(
            {
                repo: spec["repo_images"][repo]["image_id"]
                for repo in spec["repos"]
            },
            sort_keys=True,
        )
        os.environ.pop(MATERIALIZATION_ENV, None)
        os.environ.pop(RECEIPTS_ENV, None)
        os.environ.pop(EPHEMERAL_IMAGES_ENV, None)
    elif resolution == "build_before_agent":
        materializations = {
            repo: spec["repo_images"][repo] for repo in spec["repos"]
        }
        os.environ[override_env] = json.dumps(
            {
                repo: materializations[repo]["tag"]
                for repo in spec["repos"]
            },
            sort_keys=True,
        )
        os.environ[MATERIALIZATION_ENV] = json.dumps(
            materializations, sort_keys=True
        )
        os.environ[RECEIPTS_ENV] = str(
            _run_directory(spec) / "image_materialization"
        )
        os.environ[EPHEMERAL_IMAGES_ENV] = (
            "0"
            if any(row["persist"] for row in materializations.values())
            else "1"
        )
    else:
        os.environ.pop(override_env, None)
        os.environ.pop(MATERIALIZATION_ENV, None)
        os.environ.pop(RECEIPTS_ENV, None)
        os.environ.pop(EPHEMERAL_IMAGES_ENV, None)
    return run_experiment(spec)


if __name__ == "__main__":
    raise SystemExit(main())
