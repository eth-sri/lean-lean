"""Prepare and summarize paired held-out-theorem reconstruction evaluations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from leanlean.evaluation_images import (
    ensure_materialized_image_from_record,
    materialized_image_id,
    retire_image_tags,
)

from leanlean.dataset_bundle import load_dataset
from leanlean.preprocessing.repositories import (
    file_sha256,
    repository_definition_sha256,
)
from leanlean.preprocessing.theorem_holdout import (
    theorem_command_with_sorry as _render_theorem_with_sorry,
    theorem_statement_prefix as _statement_prefix,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
KIND = "leanlean_theorem_reconstruction"
SCHEMA_VERSION = 1
PERMITTED_AXIOMS = ["Classical.choice", "Quot.sound", "propext"]
console = Console()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path(value: Any, name: str) -> Path:
    raw = Path(_text(value, name))
    path = raw if raw.is_absolute() else REPO_ROOT / raw
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{name} must stay inside the repository")
    return path


def _run(command: list[str], *, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            + (result.stdout + result.stderr)[-6000:]
        )
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"output drift at {path}; choose a new run_id")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    manifest = dict(_mapping(value, "manifest"))
    expected = {
        "kind",
        "schema_version",
        "run_id",
        "preprocessing",
        "compression",
        "theorem",
        "arms",
        "model",
        "reasoning_effort",
        "generator",
        "rounds",
        "parallelism",
        "container",
        "timeout",
        "evaluation",
        "outputs",
    }
    if set(manifest) != expected:
        raise ValueError(
            "reconstruction manifest fields differ: "
            f"missing={sorted(expected - set(manifest))}, "
            f"unknown={sorted(set(manifest) - expected)}"
        )
    if manifest.get("kind") != KIND or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported theorem reconstruction manifest")
    if manifest.get("model") != "gpt-5.6-sol":
        raise ValueError("this run must pin model: gpt-5.6-sol")
    if manifest.get("reasoning_effort") != "xhigh":
        raise ValueError("this run must pin reasoning_effort: xhigh")
    if manifest.get("generator") != "codex_sub" or manifest.get("rounds") != 1:
        raise ValueError("reconstruction requires codex_sub and one round")
    container = _mapping(manifest["container"], "container")
    required_container = {
        "cpus",
        "build_jobs",
        "memory",
        "max_total_memory",
        "pids_limit",
        "network_policy",
        "cgroup_parent",
    }
    if set(container) != required_container:
        raise ValueError("container must pin every reconstruction resource")
    if container.get("network_policy") != "model_proxy_only":
        raise ValueError("model containers require network_policy: model_proxy_only")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(manifest["run_id"])):
        raise ValueError("run_id contains unsupported characters")
    arms = _mapping(manifest["arms"], "arms")
    if set(arms) != {"stripped", "compressed"}:
        raise ValueError("arms must define stripped and compressed")
    arm_ids = [_text(value, f"arms.{name}") for name, value in arms.items()]
    if len(set(arm_ids)) != 2:
        raise ValueError("arm repository IDs must be distinct")
    theorem = _mapping(manifest["theorem"], "theorem")
    selection = _mapping(theorem.get("selection"), "theorem.selection")
    required_selection = {
        "policy",
        "baseline_dataset",
        "predicted_removed_lean_tokens",
        "predicted_removed_fraction",
        "minimum_removed_lean_tokens",
        "minimum_removed_fraction",
    }
    if set(selection) != required_selection:
        raise ValueError("theorem.selection must pin prediction and size gate")
    _text(selection.get("policy"), "theorem.selection.policy")
    _path(selection.get("baseline_dataset"), "theorem.selection.baseline_dataset")
    for field in (
        "predicted_removed_lean_tokens",
        "predicted_removed_fraction",
        "minimum_removed_lean_tokens",
        "minimum_removed_fraction",
    ):
        value = selection.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ValueError(f"theorem.selection.{field} must be nonnegative")
    _path(
        _mapping(manifest["outputs"], "outputs").get("ablation_report"),
        "outputs.ablation_report",
    )
    if manifest["evaluation"].get("reproof_edit_policy", "target_file_only") not in {"target_file_only", "repository_wide"}:
        raise ValueError("invalid reconstruction reproof_edit_policy")
    return manifest


_DECLARATION_HEADER = re.compile(
    r"(?m)^[ \t]*(?:theorem|lemma|def|opaque|abbrev|instance)\s+"
    r"([^\s({:\[]+)"
)


def _attached_doc_start(source: str, declaration_start: int) -> int:
    """Include the doc comment immediately attached to a declaration."""

    prefix = source[:declaration_start]
    content_end = len(prefix.rstrip())
    if not prefix[:content_end].endswith("-/"):
        return declaration_start
    comment_start = prefix.rfind("/--", 0, content_end)
    if comment_start < 0:
        return declaration_start
    comment_end = prefix.find("-/", comment_start, content_end) + 2
    if comment_end != content_end or prefix[comment_end:].strip():
        return declaration_start
    return comment_start


def _reinsert_theorem_at_original_location(
    *,
    original_source: str,
    current_source: str,
    holdout: Mapping[str, Any],
) -> str:
    """Restore a held-out theorem before the same following declaration."""

    source_command = _text(holdout.get("source_command"), "holdout.source_command")
    source_command_with_sorry = _text(
        holdout.get("source_command_with_sorry"),
        "holdout.source_command_with_sorry",
    )
    if source_command_with_sorry != _render_theorem_with_sorry(source_command):
        raise ValueError("held-out theorem sorry artifact does not match its source")
    start = holdout.get("start_line")
    end = holdout.get("end_line")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 1
        or end < start
    ):
        raise ValueError("held-out theorem has an invalid original source range")
    original_lines = original_source.splitlines(keepends=True)
    if "".join(original_lines[start - 1:end]) != source_command:
        raise ValueError("held-out theorem source no longer matches its raw file")

    declaration = _text(holdout.get("declaration"), "holdout.declaration")
    short_name = declaration.rsplit(".", 1)[-1]
    if re.search(
        rf"(?m)^[ \t]*(?:theorem|lemma)\s+{re.escape(short_name)}(?=[\s({{:\[])",
        current_source,
    ):
        raise ValueError("held-out theorem is already present in reconstruction arm")

    suffix = "".join(original_lines[end:])
    next_match = _DECLARATION_HEADER.search(suffix)
    if next_match is not None:
        next_name = next_match.group(1)
        # Compression may replace a theorem command with a named command
        # macro. Recognize only macros declared to take a leading identifier.
        from leanlean.metrics.tokens import remove_lean_comments
        code = remove_lean_comments(current_source)
        command_syntax = re.compile(
            r'(?m)^[ \t]*syntax[ \t]+"([A-Za-z_][A-Za-z_0-9]*)[ \t]*"'
            r'[ \t]+ident\b(?:[^\n]|\n[ \t]+)*?\s*:\s*command\b'
        )
        heads = ["theorem", "lemma", "def", "opaque", "abbrev", "instance"]
        heads.extend(match.group(1) for match in command_syntax.finditer(code))
        current_pattern = re.compile(
            rf"(?m)^[ \t]*(?:{'|'.join(map(re.escape, heads))})\s+"
            rf"{re.escape(next_name)}(?=[\s({{:\[])"
        )
        anchor = f"following declaration anchor {next_name!r}"
    else:
        closing = re.search(
            r"(?m)^[ \t]*end(?:[ \t]+[^\s]+)?[ \t]*$", suffix
        )
        if closing is None:
            raise ValueError(
                "held-out theorem has no following declaration or namespace anchor"
            )
        closing_command = closing.group(0).strip()
        current_pattern = re.compile(
            rf"(?m)^[ \t]*{re.escape(closing_command)}[ \t]*$"
        )
        anchor = f"closing namespace anchor {closing_command!r}"
    from leanlean.metrics.tokens import remove_lean_comments
    current_matches = list(current_pattern.finditer(
        remove_lean_comments(current_source, mask_strings=True)
    ))
    if len(current_matches) != 1:
        raise ValueError(f"{anchor} did not resolve exactly once")
    insertion = _attached_doc_start(current_source, current_matches[0].start())
    theorem_source = source_command_with_sorry.rstrip() + "\n\n"
    return current_source[:insertion] + theorem_source + current_source[insertion:]


def _ensure_module_import(source: str, module: str) -> str:
    """Add an import when module pruning removed an isolated theorem module."""

    pattern = re.compile(rf"(?m)^[ \t]*import[ \t]+{re.escape(module)}[ \t]*$")
    if pattern.search(source):
        return source
    return f"import {module}\n" + source


def _compression_patch(manifest: Mapping[str, Any], repository_id: str) -> str:
    from leanlean.pipeline.reconstruction_inputs import generation_artifacts
    _, predictions = generation_artifacts(manifest, "compression")
    row = _mapping(predictions.get(repository_id), "compression prediction")
    patch = row.get("model_patch")
    if not isinstance(patch, str):
        raise ValueError("compression prediction has no model_patch")
    from leanlean.pipeline.reconstruction_repairs import repair_compression_patch
    return repair_compression_patch(patch, manifest.get("compression", {}).get("build_config_repair"))


def _holdout_record(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    preprocessing = _mapping(manifest["preprocessing"], "preprocessing")
    dataset_path = _path(preprocessing.get("dataset"), "preprocessing.dataset")
    dataset = load_dataset(dataset_path)
    repository_id = _text(preprocessing.get("repository"), "preprocessing.repository")
    rows = {
        str(row.get("id")): row
        for row in dataset.get("repositories", [])
        if isinstance(row, Mapping)
    }
    row = dict(_mapping(rows.get(repository_id), "preprocessing repository"))
    report_path = _path(
        _mapping(row.get("preprocessing"), "repository.preprocessing").get("report"),
        "preprocessing report",
    )
    report = _mapping(json.loads(report_path.read_text()), "preprocessing report")
    declaration = _text(
        _mapping(manifest["theorem"], "theorem").get("declaration"),
        "theorem.declaration",
    )
    matches = [
        item
        for item in report.get("removed_theorems", [])
        if isinstance(item, Mapping) and item.get("declaration") == declaration
    ]
    if len(matches) != 1 or matches[0].get("removed_from_stripped_tree") is not True:
        raise ValueError("preprocessing did not certify the requested removed theorem")
    return dataset, row, matches[0]


@contextlib.contextmanager
def _materialized_source_image(
    manifest: Mapping[str, Any],
    repository_id: str,
    base: Mapping[str, Any],
) -> Iterator[str]:
    pinned = base.get("image_id")
    if isinstance(pinned, str):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", pinned):
            raise ValueError("source stripped image_id is invalid")
        yield pinned
        return

    materialization = _mapping(
        base.get("materialization"), "source stripped materialization"
    )
    if (
        materialization.get("mode") != "build_before_agent"
        or materialization.get("backend") != "shared_environment_v1"
    ):
        raise ValueError("unsupported reconstruction source materialization")
    source_tree = _path(base.get("cache_tree"), "source stripped cache tree")
    warm = dict(
        _mapping(
            materialization.get("warm_build_cache"),
            "source stripped warm_build_cache",
        )
    )
    warm_path = _path(warm.get("path"), "source stripped warm build path")
    warm["path"] = warm_path.relative_to(REPO_ROOT).as_posix()

    environment = dict(
        _mapping(
            materialization.get("shared_environment"),
            "source stripped shared_environment",
        )
    )
    archive = dict(
        _mapping(
            environment.get("archive"),
            "source stripped shared environment archive",
        )
    )
    archive_path = _path(
        archive.get("path"), "source stripped shared environment archive path"
    )
    archive["path"] = archive_path.relative_to(REPO_ROOT).as_posix()
    environment["archive"] = archive

    resources = _mapping(manifest["container"], "container")
    safe_run = re.sub(r"[^A-Za-z0-9_.-]", "-", str(manifest["run_id"]))
    tag = (
        f"leanlean-{repository_id}-stripped:"
        f"reconstruct-{safe_run}"
    )
    record = {
        "tag": tag,
        "source_tree": source_tree.relative_to(REPO_ROOT).as_posix(),
        "tree_sha256": _text(
            base.get("tree_sha256"), "source stripped tree_sha256"
        ),
        "build_target": _text(
            materialization.get("build_target"),
            "source stripped build_target",
        ),
        "build_jobs": int(resources["build_jobs"]),
        "build_timeout_seconds": int(manifest["timeout"]),
        "persist": False,
        "source_isolation": "standalone_v3",
        "backend": "shared_environment_v1",
        "shared_environment": environment,
        "warm_build_cache": warm,
    }
    try:
        ensure_materialized_image_from_record(repository_id, record)
        yield materialized_image_id(tag)
    finally:
        retire_image_tags(tag)



def _docker_args(
    manifest: Mapping[str, Any], container_name: str, image: str,
    *, labels: Mapping[str, str] | None = None,
) -> list[str]:
    resources = _mapping(manifest["container"], "container")
    return [
        "docker",
        "create",
        "--name",
        container_name,
        "--network=none",
        "--pids-limit",
        str(resources["pids_limit"]),
        "--cpus",
        str(resources["cpus"]),
        "--memory",
        str(resources["memory"]),
        "--memory-swap",
        str(resources["memory"]),
        "--cgroup-parent",
        str(resources["cgroup_parent"]),
        *[argument for key, value in (labels or {}).items() for argument in ("--label", f"{key}={value}")],
        image,
        "sleep",
        "infinity",
    ]


def _copy_into_container(container: str, source: Path, destination: str) -> None:
    """Copy generated bytes without propagating the host UID into Docker."""

    target = PurePosixPath(destination)
    if not target.is_absolute() or ".." in target.parts or not target.name:
        raise ValueError("container destination must be an absolute file path")
    data = source.read_bytes()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        member = tarfile.TarInfo(target.name)
        member.size = len(data)
        member.mode = 0o644
        member.uid = member.gid = 0
        bundle.addfile(member, io.BytesIO(data))
    copied = subprocess.run(
        ["docker", "exec", "-i", container, "tar", "--no-same-owner",
         "-xf", "-", "-C", str(target.parent)],
        input=archive.getvalue(), capture_output=True, timeout=120, check=False,
    )
    if copied.returncode != 0:
        raise RuntimeError(
            f"could not copy {destination} into reconstruction container: "
            + copied.stderr.decode(errors="replace")
        )


def _materialize_arm(
    *,
    manifest: Mapping[str, Any],
    arm: str,
    instance_id: str,
    base_image: str,
    compression_patch: str,
    original_source: str,
    entry_module: str,
    holdout: Mapping[str, Any],
) -> dict[str, str]:
    safe_run = re.sub(r"[^A-Za-z0-9_.-]", "-", str(manifest["run_id"]))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", instance_id)
    from leanlean.environments.container_identity import container_identity
    container_name, monitoring_labels = container_identity(base_image, {
        "run_id": manifest["run_id"], "role": "reconstruction",
        "model": manifest.get("model"), "instance_id": instance_id,
    })
    image = f"leanlean-{safe_id}-stripped:{safe_run}"
    # Names are only for monitoring; execution and cleanup use Docker's ID.
    container_name = _run(_docker_args(
        manifest, container_name, base_image, labels=monitoring_labels,
    )).stdout.strip()
    try:
        _run(["docker", "start", container_name])
        theorem = _mapping(manifest["theorem"], "theorem")
        target_file = _text(theorem.get("target_file"), "theorem.target_file")
        build_target = _text(theorem.get("build_target"), "theorem.build_target")
        with tempfile.TemporaryDirectory(prefix="leanlean-reconstruct-") as raw_tmp:
            temporary = Path(raw_tmp)
            if arm == "compressed" and compression_patch.strip():
                patch_path = temporary / "compression.diff"
                patch_path.write_text(compression_patch)
                _copy_into_container(container_name, patch_path, "/tmp/compression.diff")
                _run([
                    "docker", "exec", "-w", "/testbed", container_name,
                    "git", "apply", "--whitespace=nowarn", "/tmp/compression.diff",
                ])
            target_in_container = f"/testbed/{target_file}"
            target_exists = subprocess.run(
                ["docker", "exec", container_name, "test", "-f", target_in_container],
                capture_output=True,
            ).returncode == 0
            if target_exists:
                current_path = temporary / "current.lean"
                _run([
                    "docker", "cp", f"{container_name}:{target_in_container}",
                    str(current_path),
                ])
                restored_source = _reinsert_theorem_at_original_location(
                    original_source=original_source,
                    current_source=current_path.read_text(),
                    holdout=holdout,
                )
            else:
                restored_source = _text(
                    holdout.get("source_file_with_sorry"),
                    "holdout.source_file_with_sorry",
                )
                sorry_command = _text(
                    holdout.get("source_command_with_sorry"),
                    "holdout.source_command_with_sorry",
                )
                if restored_source.count(sorry_command) != 1:
                    raise ValueError(
                        "held-out source-file artifact does not contain exactly "
                        "one serialized theorem"
                    )
                target_parent = str(Path(target_in_container).parent)
                _run(["docker", "exec", container_name, "mkdir", "-p", target_parent])
                theorem_module = _text(theorem.get("module"), "theorem.module")
                if theorem_module != entry_module:
                    entry_file = entry_module.replace(".", "/") + ".lean"
                    entry_path = temporary / "entry.lean"
                    _run([
                        "docker", "cp", f"{container_name}:/testbed/{entry_file}",
                        str(entry_path),
                    ])
                    imported = _ensure_module_import(
                        entry_path.read_text(), theorem_module
                    )
                    entry_path.write_text(imported)
                    _copy_into_container(container_name, entry_path, f"/testbed/{entry_file}")
            restored_path = temporary / "restored.lean"
            restored_path.write_text(restored_source)
            _copy_into_container(container_name, restored_path, target_in_container)
            from leanlean.preprocessing.cache_isolation import render_container_cache_cleanup
            _run([
                "docker", "exec", container_name, "python3", "-c",
                render_container_cache_cleanup(clear_project=True),
            ])

            _run(["docker", "exec", "-w", "/testbed", container_name, "git", "add", "-A"])
            _run([
                "docker", "exec", "-w", "/testbed",
                "-e", "GIT_AUTHOR_NAME=LeanLean",
                "-e", "GIT_AUTHOR_EMAIL=leanlean@localhost",
                "-e", "GIT_COMMITTER_NAME=LeanLean",
                "-e", "GIT_COMMITTER_EMAIL=leanlean@localhost",
                container_name, "git", "commit", "-m", f"prepare {arm} reconstruction arm",
            ])
            commit = _run([
                "docker", "exec", "-w", "/testbed", container_name,
                "git", "rev-parse", "HEAD",
            ]).stdout.strip()
            init_path = temporary / "init_commit"
            init_path.write_text(commit + "\n")
            _copy_into_container(container_name, init_path, "/.init_commit")

        resources = _mapping(manifest["container"], "container")
        _run([
            "docker", "exec", "-w", "/testbed",
            "-e", f"LEAN_NUM_THREADS={resources['build_jobs']}",
            container_name, "lake", "build", build_target,
        ], timeout=int(manifest["timeout"]))

        labels = [
            f"LABEL org.openai.leanlean.instance_id={instance_id}",
            "LABEL org.openai.leanlean.repo_variant=stripped",
            f"LABEL org.openai.leanlean.run_id={manifest['run_id']}",
            f"LABEL org.openai.leanlean.reconstruction_arm={arm}",
        ]
        command = ["docker", "commit"]
        for label in labels:
            command.extend(["--change", label])
        command.extend([container_name, image])
        _run(command)
        image_id = _run([
            "docker", "image", "inspect", "--format", "{{.Id}}", image
        ]).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise RuntimeError("materialized arm has no immutable image ID")
        return {"tag": image, "image_id": image_id}
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def _write_dataset(
    *,
    manifest: Mapping[str, Any],
    source_dataset: Mapping[str, Any],
    source_row: Mapping[str, Any],
    images: Mapping[str, Mapping[str, str]],
) -> None:
    outputs = _mapping(manifest["outputs"], "outputs")
    dataset_path = _path(outputs.get("dataset"), "outputs.dataset")
    database_path = _path(outputs.get("repository_database"), "outputs.repository_database")
    source_database_path = _path(
        _mapping(source_dataset.get("repository_database"), "source repository database").get("path"),
        "source repository database path",
    )
    source_database = _mapping(
        json.loads(source_database_path.read_text()), "source repository database"
    )
    source_id = _text(
        _mapping(manifest["preprocessing"], "preprocessing").get("repository"),
        "preprocessing.repository",
    )
    source_rows = {
        str(row.get("instance_id")): row
        for row in source_database["repositories"]
        if isinstance(row, Mapping)
    }
    template = source_rows[source_id]
    theorem = _mapping(manifest["theorem"], "theorem")
    declaration = _text(theorem.get("declaration"), "theorem.declaration")
    arms = _mapping(manifest["arms"], "arms")
    rows = []
    database_rows = []
    for arm in ("stripped", "compressed"):
        instance_id = str(arms[arm])
        db_row = json.loads(json.dumps(template))
        db_row["instance_id"] = instance_id
        db_row["source"]["raw_image"] = {
            "tag": images[arm]["tag"],
            "image_id": images[arm]["image_id"],
        }
        db_row["scopes"].update(
            entry_module=str(theorem["module"]),
            target_dir=str(theorem["metric_scope"]),
            build_target=str(theorem["build_target"]),
        )
        db_row["protected"] = {
            "declarations": [declaration],
            "provenance": {
                "kind": "leanlean_theorem_reconstruction_v1",
                "module": str(theorem["module"]),
                "target_file": str(theorem["target_file"]),
                "build_target": str(theorem["build_target"]),
                "permitted_axioms": list(PERMITTED_AXIOMS),
                "edit_policy": manifest["evaluation"].get("reproof_edit_policy", "target_file_only"),
            },
        }
        database_rows.append(db_row)
        rows.append(
            {
                "id": instance_id,
                "source": {
                    "ablation_arm": arm,
                    "parent_repository": source_id,
                },
                "variants": {
                    "stripped": {
                        "format": "theorem_reconstruction_repository",
                        "image": images[arm]["tag"],
                        "image_id": images[arm]["image_id"],
                    }
                },
            }
        )
    database_payload = {
        "kind": source_database["kind"],
        "schema_version": source_database["schema_version"],
        "dataset_id": str(manifest["run_id"]),
        "dataset_version": "v1",
        "repositories": database_rows,
    }
    database_content = json.dumps(
        database_payload, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    _atomic_text(database_path, database_content)
    definition_sha = repository_definition_sha256(database_payload)
    dataset = {
        "kind": "leanlean_dataset",
        "schema_version": 1,
        "id": str(manifest["run_id"]),
        "version": "v1",
        "default_variant": "stripped",
        "repository_count": 2,
        "repository_database": {
            "path": database_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(database_path),
            "definition_sha256": definition_sha,
            "dataset_id": str(manifest["run_id"]),
            "dataset_version": "v1",
        },
        "preprocessing": {
            "run_id": str(manifest["run_id"]),
            "source_dataset": str(
                _mapping(manifest["preprocessing"], "preprocessing")["dataset"]
            ),
        },
        "repositories": rows,
    }
    _atomic_text(
        dataset_path,
        yaml.safe_dump(dataset, sort_keys=False, width=100),
    )


def _variant_lean_tokens(row: Mapping[str, Any], variant: str, name: str) -> int:
    variants = _mapping(row.get("variants"), f"{name}.variants")
    record = _mapping(variants.get(variant), f"{name}.variants.{variant}")
    value = record.get("lean_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}.variants.{variant}.lean_tokens must be nonnegative")
    return value


def _measure_ablation(
    manifest: Mapping[str, Any], source_row: Mapping[str, Any]
) -> dict[str, Any]:
    theorem = _mapping(manifest["theorem"], "theorem")
    selection = _mapping(theorem.get("selection"), "theorem.selection")
    baseline_path = _path(
        selection.get("baseline_dataset"), "theorem.selection.baseline_dataset"
    )
    baseline_tokens = source_row.get("source", {}).get("baseline_stripped_lean_tokens")
    if isinstance(baseline_tokens, int) and not isinstance(baseline_tokens, bool) and baseline_tokens >= 0:
        baseline = {"repositories": [{"id": source_row["source"]["parent_repository"],
                    "variants": {"stripped": {"lean_tokens": baseline_tokens}}}]}
    else:
        baseline = load_dataset(baseline_path, verify_trees=False)
    repository_id = _text(
        _mapping(manifest["preprocessing"], "preprocessing").get("repository"),
        "preprocessing.repository",
    )
    baseline_rows = {
        str(row.get("id")): row
        for row in baseline.get("repositories", [])
        if isinstance(row, Mapping)
    }
    baseline_row = _mapping(
        baseline_rows.get(source_row.get("source", {}).get("parent_repository", repository_id)), "baseline repository"
    )
    baseline_tokens = _variant_lean_tokens(
        baseline_row, "stripped", "baseline repository"
    )
    ablated_tokens = _variant_lean_tokens(
        source_row, "stripped", "ablated repository"
    )
    removed_tokens = baseline_tokens - ablated_tokens
    removed_fraction = removed_tokens / baseline_tokens if baseline_tokens else 0.0
    minimum_tokens = int(selection["minimum_removed_lean_tokens"])
    minimum_fraction = float(selection["minimum_removed_fraction"])
    passed = removed_tokens >= minimum_tokens and removed_fraction >= minimum_fraction
    return {
        "kind": "leanlean_theorem_ablation_size_gate",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "repository": repository_id,
        "reproof_target": theorem["declaration"],
        "selection_policy": selection["policy"],
        "baseline_dataset": baseline_path.relative_to(REPO_ROOT).as_posix(),
        "baseline_stripped_lean_tokens": baseline_tokens,
        "ablated_stripped_lean_tokens": ablated_tokens,
        "removed_lean_tokens": removed_tokens,
        "removed_fraction": removed_fraction,
        "predicted_removed_lean_tokens": selection["predicted_removed_lean_tokens"],
        "predicted_removed_fraction": selection["predicted_removed_fraction"],
        "minimum_removed_lean_tokens": minimum_tokens,
        "minimum_removed_fraction": minimum_fraction,
        "passed": passed,
    }


def validate_ablation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the actual holdout reduction before spending model compute."""

    _, source_row, _ = _holdout_record(manifest)
    report = _measure_ablation(manifest, source_row)
    report_path = _path(
        _mapping(manifest["outputs"], "outputs").get("ablation_report"),
        "outputs.ablation_report",
    )
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["passed"] is not True:
        raise RuntimeError(
            "theorem ablation is too small: "
            f"{report['removed_lean_tokens']} Lean tokens "
            f"({report['removed_fraction']:.1%})"
        )
    return report


def prepare(
    manifest: Mapping[str, Any], *, progress: Progress | None = None
) -> dict[str, Any]:
    """Prepare both arms, borrowing a caller's dashboard for concurrent cases.

    A borrowed Progress is already running and remains owned by the caller.
    Starting per-worker live displays on the shared console races in Rich.
    """
    outputs = _mapping(manifest["outputs"], "outputs")
    receipt_path = _path(outputs.get("receipt"), "outputs.receipt")
    dataset_path = _path(outputs.get("dataset"), "outputs.dataset")
    if receipt_path.is_file() and dataset_path.is_file():
        existing = _mapping(
            json.loads(receipt_path.read_text()), "preparation receipt"
        )
        images = _mapping(existing.get("images"), "preparation receipt images")
        if existing.get("project_cache_policy") != "clean_project_preserve_dependencies":
            raise ValueError("prepared images used project caches; choose a new run_id for a clean build")
        valid = existing.get("run_id") == manifest["run_id"]
        for record in images.values():
            if not isinstance(record, Mapping):
                valid = False
                break
            inspected = subprocess.run(
                [
                    "docker", "image", "inspect", "--format", "{{.Id}}",
                    str(record.get("tag") or ""),
                ],
                capture_output=True,
                text=True,
            )
            valid = (
                valid
                and inspected.returncode == 0
                and inspected.stdout.strip() == record.get("image_id")
            )
        if valid and set(images) == {"stripped", "compressed"}:
            return dict(existing)
    source_dataset, source_row, holdout = _holdout_record(manifest)
    ablation = validate_ablation(manifest)
    repository_id = _text(
        _mapping(manifest["preprocessing"], "preprocessing").get("repository"),
        "preprocessing.repository",
    )
    patch = _compression_patch(manifest, repository_id)
    theorem = _mapping(manifest["theorem"], "theorem")
    target_file = _text(theorem.get("target_file"), "theorem.target_file")
    if target_file != holdout.get("source_path"):
        raise ValueError("reconstruction target_file is not the original theorem file")
    if theorem.get("module") != holdout.get("module"):
        raise ValueError("reconstruction module is not the original theorem module")
    target_path = Path(target_file)
    if target_path.is_absolute() or ".." in target_path.parts:
        raise ValueError("theorem.target_file must be a safe repository path")
    raw_variant = _mapping(
        _mapping(source_row["variants"], "source variants").get("raw"),
        "source raw variant",
    )
    raw_tree = _path(raw_variant.get("cache_tree"), "source raw cache tree")
    original_path = (raw_tree / target_path).resolve()
    if not original_path.is_relative_to(raw_tree) or not original_path.is_file():
        raise ValueError("original theorem source file is missing")
    original_source = original_path.read_text()
    base = _mapping(
        _mapping(source_row["variants"], "source variants").get("stripped"),
        "source stripped variant",
    )
    entry_module = _text(
        _mapping(source_row.get("scopes", source_row.get("scope")), "source scopes").get("entry_module"),
        "source scopes.entry_module",
    )
    arms = _mapping(manifest["arms"], "arms")
    images: dict[str, dict[str, str]] = {}
    with _materialized_source_image(
        manifest, repository_id, base
    ) as base_image:
        display = contextlib.nullcontext(progress) if progress is not None else Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )
        with display as progress:
            for arm in ("stripped", "compressed"):
                task = progress.add_task(
                    f"{repository_id}: materializing {arm} reconstruction arm", total=1
                )
                images[arm] = _materialize_arm(
                    manifest=manifest,
                    arm=arm,
                    instance_id=str(arms[arm]),
                    base_image=base_image,
                    compression_patch=patch,
                    original_source=original_source,
                    entry_module=entry_module,
                    holdout=holdout,
                )
                progress.update(task, completed=1)
                progress.stop_task(task)
    _write_dataset(
        manifest=manifest,
        source_dataset=source_dataset,
        source_row=source_row,
        images=images,
    )
    receipt = {
        "kind": "leanlean_theorem_reconstruction_preparation",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "theorem": dict(theorem),
        "source_command_sha256": _sha256_text(str(holdout["source_command"])),
        "source_command_with_sorry_sha256": _sha256_text(
            str(holdout["source_command_with_sorry"])
        ),
        "source_file_with_sorry_sha256": _sha256_text(
            str(holdout["source_file_with_sorry"])
        ),
        "compression_patch_sha256": _sha256_text(patch),
        "build_config_repair": manifest.get("compression", {}).get("build_config_repair"),
        "project_cache_policy": "clean_project_preserve_dependencies",
        "ablation": ablation,
        "images": images,
        "prepared_at_unix": time.time(),
    }
    receipt_path = _path(
        _mapping(manifest["outputs"], "outputs").get("receipt"),
        "outputs.receipt",
    )
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _patch_added_lines(patch: str, target_file: str) -> int:
    active = False
    count = 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            active = line[6:] == target_file
        elif active and line.startswith("+") and not line.startswith("+++"):
            count += 1
    return count


def summarize(manifest: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _mapping(manifest["evaluation"], "evaluation")
    run_path = _path(evaluation.get("run_artifact"), "evaluation.run_artifact")
    run = _mapping(yaml.safe_load(run_path.read_text()), "evaluation run")
    if run.get("status") != "complete":
        raise RuntimeError(f"reconstruction evaluation is not complete: {run.get('status')}")
    output = _path(
        _mapping(run.get("artifacts"), "evaluation artifacts").get("output_directory"),
        "evaluation output",
    )
    predictions_path = _path(
        _mapping(run.get("artifacts"), "evaluation artifacts").get("predictions"),
        "evaluation predictions",
    )
    predictions = _mapping(json.loads(predictions_path.read_text()), "predictions")
    theorem = _mapping(manifest["theorem"], "theorem")
    target_file = str(theorem["target_file"])
    arms = _mapping(manifest["arms"], "arms")
    rows: dict[str, dict[str, Any]] = {}
    for arm in ("stripped", "compressed"):
        instance_id = str(arms[arm])
        report_path = output / "round_0" / instance_id / "report.json"
        report = _mapping(json.loads(report_path.read_text()), "arm report")
        result = _mapping(report.get(instance_id), "arm result")
        generation_path = output / instance_id / "round_0_gen_metrics.json"
        generation = _mapping(
            json.loads(generation_path.read_text()), "generation metrics"
        )
        prediction = _mapping(predictions.get(instance_id), "arm prediction")
        patch = str(prediction.get("model_patch") or "")
        metrics = _mapping(result.get("metrics"), "arm metrics")
        verification = _mapping(
            _mapping(result.get("build_result"), "build result").get(
                "reconstruction_verification", {}
            ),
            "reconstruction verification",
        )
        rows[arm] = {
            "instance_id": instance_id,
            "resolved": result.get("resolved") is True,
            "wall_time_seconds": generation.get("wall_time_seconds"),
            "added_lean_tokens": (
                metrics["post_lean_tokens"] - metrics["baseline_lean_tokens"]
                if all(isinstance(metrics.get(key), (int, float)) for key in ("post_lean_tokens", "baseline_lean_tokens"))
                else None
            ),
            "post_lean_tokens": metrics.get("post_lean_tokens"),
            "baseline_lean_tokens": metrics.get("baseline_lean_tokens"),
            "patch_added_lines": _patch_added_lines(patch, target_file),
            "build_time_seconds": _mapping(
                result.get("build_result"), "build result"
            ).get("build_time_seconds"),
            "verification_time_seconds": verification.get("time_seconds"),
            "verification_passed": verification.get("passed") is True,
            "model_patch_sha256": _sha256_text(patch),
        }
    stripped = rows["stripped"]
    compressed = rows["compressed"]
    summary = {
        "kind": "leanlean_theorem_reconstruction_summary",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "theorem": str(theorem["declaration"]),
        "token_metric": "post_lean_tokens_minus_baseline_lean_tokens",
        "token_scope": str(theorem.get("metric_scope") or "whole_repository"),
        "both_resolved": all(row["resolved"] for row in rows.values()),
        "arms": rows,
        "paired_delta_compressed_minus_stripped": {
            key: (
                compressed[key] - stripped[key]
                if isinstance(compressed.get(key), (int, float))
                and isinstance(stripped.get(key), (int, float))
                else None
            )
            for key in (
                "wall_time_seconds",
                "added_lean_tokens",
                "patch_added_lines",
                "build_time_seconds",
                "verification_time_seconds",
            )
        },
    }
    summary_path = _path(
        _mapping(manifest["outputs"], "outputs").get("summary"),
        "outputs.summary",
    )
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("action", choices=("validate", "prepare", "score-compression", "score", "summarize"))
    args = parser.parse_args(argv)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else (REPO_ROOT / args.manifest).resolve()
    )
    manifest = _load_manifest(manifest_path)
    if args.action == "validate":
        report = validate_ablation(manifest)
        console.print_json(data=report)
    elif args.action in ("score", "score-compression"):
        from leanlean.pipeline.reconstruction_scoring import score
        console.print_json(data=score(manifest, compression=args.action == "score-compression"))
    elif args.action == "prepare":
        receipt = prepare(manifest)
        console.print(
            f"[green]prepared[/green] {len(receipt['images'])} reconstruction arms"
        )
    else:
        summary = summarize(manifest)
        console.print_json(data=summary)
    return 0
