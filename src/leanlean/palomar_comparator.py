"""Resolve and run the frozen Palomar Comparator contract in postprocessing.

The benchmark images deliberately omit the statement oracle.  This module
recovers the exact registered Challenge and Comparator configuration from the
frozen Palomar source archive, pins the verifier revisions recorded by the
registry, and exposes only hashes and paths to the postprocessing manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from leanlean.identifiers import same_identifier


VALIDATION_SCHEMA = "leanlean_palomar_comparator_v1"
COMPARATOR_BUNDLE_SCHEMA = "leanlean_palomar_comparator_bundle_v1"
COMPARATOR_PROVENANCE_KINDS = frozenset({
    "palomar_registry_main_results",
    "leanlean_comparator_main_results",
    "leanlean_theorem_holdout_v1",
})
COMPARATOR_BUNDLE_MANIFEST = "manifest.json"
TOOL_BUNDLE = Path(".cache/palomar-comparator/bundle")
# Where each pinned tool lives under TOOL_BUNDLE. The commits themselves come
# from each repository's verification contract; configs/comparator-tools.yaml
# lists the ones the benchmark uses and scripts/build_comparator_bundle.py
# builds them into this layout.
TOOL_LAYOUT = {
    "landrun": "bin/landrun",
    "nanoda": "bin/nanoda_bin",
    "comparator": "comparator/{commit}/comparator",
    "lean4export": "lean4export/{commit}/lean4export",
}
BUNDLE_BUILDER = "scripts/build_comparator_bundle.py"
LANDRUN_WRAPPER = Path("scripts/palomar_landrun_wrapper.sh")
AGENT_VERIFY_WRAPPER = Path("scripts/lean_verify")
TOOL_DESTINATIONS = {
    "comparator": "/usr/local/bin/palomar-comparator",
    "lean4export": "/usr/local/bin/palomar-lean4export",
    "landrun": "/usr/local/bin/palomar-landrun",
    "nanoda": "/usr/local/bin/palomar-nanoda",
    "landrun_wrapper": "/usr/local/bin/palomar-landrun-wrapper",
}
# Comparator revisions differ on whether these fields may be omitted. Older
# revisions supply the documented default; later ones derive a strict
# `FromJson` and abort with "Bool expected" before any build runs. Applying the
# documented default to the runtime copy only -- never to the registered copy
# the repository ships -- keeps every revision on identical semantics.
REGISTERED_CONFIG_DEFAULTS: Mapping[str, Any] = {"enable_nanoda": False}
# The nanoda kernel re-check recurses deeply on its main thread, which
# overflows the 8 MiB Linux default on large developments.
COMPARATOR_STACK_KIB = 32768
# Printed by every pinned Comparator revision as its final line on acceptance.
COMPARATOR_SUCCESS_MARKER = "Your solution is okay!"
# Worker threads nanoda spawns do not inherit the process stack ulimit.
COMPARATOR_RUST_MIN_STACK = 67108864


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_config_copy(original: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return the runtime Comparator config and the defaults it had to add.

    The registered configuration is never rewritten. When the repository author
    omitted a field that the pinned Comparator revision requires, the runtime
    copy carries the documented default so both revisions verify identically.
    """

    config = _mapping(json.loads(original), "registered Comparator config")
    applied = {
        key: value
        for key, value in REGISTERED_CONFIG_DEFAULTS.items()
        if key not in config
    }
    if not applied:
        return original, {}
    runtime = json.dumps({**config, **applied}, indent=2) + "\n"
    return runtime.encode(), applied


def _root_path(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root: {value!r}") from error
    return resolved


def _relative_to_project(repository_path: str, project_root: str) -> str:
    path = PurePosixPath(repository_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe registered repository path: {repository_path!r}")
    if not project_root:
        return path.as_posix()
    root = PurePosixPath(project_root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"registered path {repository_path!r} is outside project {project_root!r}"
        ) from error


def _archive_member(archive: Path, repository_path: str) -> bytes:
    target = PurePosixPath(repository_path)
    if target.is_absolute() or not target.parts or ".." in target.parts:
        raise ValueError(f"unsafe archive member path: {repository_path!r}")
    matches: list[tarfile.TarInfo] = []
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:])
            if relative == target:
                matches.append(member)
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError(
                f"{archive}: expected one regular member {repository_path!r}, "
                f"found {len(matches)}"
            )
        stream = source.extractfile(matches[0])
        if stream is None:
            raise ValueError(f"{archive}: could not read {repository_path!r}")
        return stream.read()


def _pinned_file(
    root: Path,
    record: Mapping[str, Any],
    label: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Resolve one explicit repository-local artifact and verify its digest."""

    path = _root_path(root, _text(record.get("path"), f"{label}.path"), label)
    pinned_sha256 = _text(record.get("sha256"), f"{label}.sha256")
    if expected_sha256 is not None and pinned_sha256 != expected_sha256:
        raise ValueError(f"{label} pin differs from the frozen mapping")
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != pinned_sha256:
        raise ValueError(
            f"{label} drifted: expected {pinned_sha256}, found {actual_sha256}"
        )
    return path


def _bundle_file(
    bundle: Path, record: Mapping[str, Any], label: str
) -> tuple[Path, bytes]:
    relative = PurePosixPath(_text(record.get("path"), f"{label}.path"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe {label} bundle path")
    path = (bundle / Path(*relative.parts)).resolve()
    if not path.is_relative_to(bundle.resolve()) or not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    expected = _text(record.get("sha256"), f"{label}.sha256")
    content = path.read_bytes()
    actual = _sha256_bytes(content)
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected}, found {actual}")
    return path, content


def load_comparator_bundle(
    *, repo_root: Path, bundle_record: Mapping[str, Any], instance_id: str
) -> tuple[Path, Mapping[str, Any], bytes, bytes, bytes]:
    """Load and hash-check one self-contained comparator directory."""

    root = repo_root.resolve()
    expected_fields = {"path", "manifest", "manifest_sha256"}
    if set(bundle_record) != expected_fields:
        raise ValueError(
            f"{instance_id}: comparator bundle fields must be "
            f"{sorted(expected_fields)}"
        )
    bundle = _root_path(
        root,
        _text(bundle_record.get("path"), "comparator.path"),
        "comparator bundle",
    )
    if not bundle.is_dir():
        raise FileNotFoundError(f"missing comparator bundle: {bundle}")
    manifest_name = _text(bundle_record.get("manifest"), "comparator.manifest")
    if manifest_name != COMPARATOR_BUNDLE_MANIFEST:
        raise ValueError(
            f"{instance_id}: comparator manifest must be "
            f"{COMPARATOR_BUNDLE_MANIFEST!r}"
        )
    manifest_path = bundle / manifest_name
    expected_manifest = _text(
        bundle_record.get("manifest_sha256"), "comparator.manifest_sha256"
    )
    if not manifest_path.is_file() or _sha256_file(manifest_path) != expected_manifest:
        raise ValueError(f"{instance_id}: comparator manifest drifted")
    manifest = _mapping(
        json.loads(manifest_path.read_text()), "Palomar comparator manifest"
    )
    if (
        not same_identifier(manifest.get("schema"), COMPARATOR_BUNDLE_SCHEMA)
        or manifest.get("instance_id") != instance_id
    ):
        raise ValueError(f"{instance_id}: invalid comparator bundle identity")
    challenge_record = _mapping(manifest.get("challenge"), "manifest.challenge")
    configuration_record = _mapping(
        manifest.get("configuration"), "manifest.configuration"
    )
    registry_record = _mapping(
        manifest.get("registry_record"), "manifest.registry_record"
    )
    _, challenge = _bundle_file(bundle, challenge_record, "Challenge")
    _, configuration = _bundle_file(
        bundle, configuration_record, "Comparator configuration"
    )
    _, registry = _bundle_file(bundle, registry_record, "Palomar registry record")
    return bundle, manifest, challenge, configuration, registry


def copy_comparator_bundle(
    *,
    repo_root: Path,
    bundle_record: Mapping[str, Any],
    instance_id: str,
    destination: Path,
) -> dict[str, str]:
    """Copy a verified bundle to a durable repository artifact directory."""

    root = repo_root.resolve()
    _, manifest, challenge, configuration, registry = load_comparator_bundle(
        repo_root=root,
        bundle_record=bundle_record,
        instance_id=instance_id,
    )
    destination = destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"{instance_id}: comparator destination escapes repository")
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode()
        + b"\n"
    )
    files = {
        COMPARATOR_BUNDLE_MANIFEST: manifest_bytes,
        _text(
            _mapping(manifest["challenge"], "manifest.challenge").get("path"),
            "manifest.challenge.path",
        ): challenge,
        _text(
            _mapping(
                manifest["configuration"], "manifest.configuration"
            ).get("path"),
            "manifest.configuration.path",
        ): configuration,
        _text(
            _mapping(manifest["registry_record"], "manifest.registry_record").get(
                "path"
            ),
            "manifest.registry_record.path",
        ): registry,
    }
    if destination.exists():
        actual = {
            path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual != files:
            raise ValueError(f"{instance_id}: durable comparator bundle drifted")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".comparator-", dir=destination.parent)
        )
        try:
            for relative, content in files.items():
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            staging.replace(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {
        "path": destination.relative_to(root).as_posix(),
        "manifest": COMPARATOR_BUNDLE_MANIFEST,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


def tool_bundle_path(tool: str, commit: str) -> Path:
    """Repository-relative path of one pinned verifier tool in the bundle."""

    return TOOL_BUNDLE / TOOL_LAYOUT[tool].format(commit=commit)


def _tool(root: Path, relative: Path, commit: str) -> dict[str, str]:
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing Palomar verifier tool {path}; build the pinned bundle with "
            f"`uv run python {BUNDLE_BUILDER}`"
        )
    return {
        "commit": commit,
        "path": str(path.relative_to(root.resolve())),
        "sha256": _sha256_file(path),
    }


def _resolve_bundled_contract(
    *,
    root: Path,
    instance_id: str,
    protected: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_record = _mapping(
        provenance.get("comparator"), "protected.provenance.comparator"
    )
    bundle, manifest, challenge_bytes, config_bytes, record_bytes = (
        load_comparator_bundle(
            repo_root=root,
            bundle_record=bundle_record,
            instance_id=instance_id,
        )
    )
    source_pin = _mapping(manifest.get("source"), "manifest.source")
    challenge_pin = _mapping(manifest.get("challenge"), "manifest.challenge")
    config_pin = _mapping(manifest.get("configuration"), "manifest.configuration")
    solution_pin = _mapping(manifest.get("solution"), "manifest.solution")
    record_pin = _mapping(manifest.get("registry_record"), "manifest.registry_record")
    palomar_id = _text(manifest.get("palomar_id"), "manifest.palomar_id")
    if provenance.get("palomar_ids") != [palomar_id]:
        raise ValueError(f"{instance_id}: Palomar result provenance drift")

    try:
        record = _mapping(
            json.loads(record_bytes), "bundled Palomar registry record"
        )
        config = _mapping(
            json.loads(config_bytes), "bundled Palomar Comparator configuration"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{instance_id}: invalid bundled Comparator evidence") from error
    verification = _mapping(record.get("verification"), "record.verification")
    formalization = _mapping(record.get("formalization"), "record.formalization")
    source = _mapping(record.get("source"), "record.source")
    repository = _text(source.get("repository"), "record.source.repository")
    commit = _text(source.get("commit"), "record.source.commit")
    project_root = str(source_pin.get("project_root") or "").strip("/")
    if (
        source_pin.get("repository") != repository
        or source_pin.get("commit") != commit
        or verification.get("challenge_sha256") != challenge_pin.get("sha256")
        or _sha256_bytes(challenge_bytes) != challenge_pin.get("sha256")
        or _sha256_bytes(config_bytes) != config_pin.get("sha256")
    ):
        raise ValueError(f"{instance_id}: bundled Palomar source provenance drift")

    challenge_path = _text(
        challenge_pin.get("registered_path"), "manifest.challenge.registered_path"
    )
    config_path = _text(
        config_pin.get("registered_path"),
        "manifest.configuration.registered_path",
    )
    solution_path = _text(
        solution_pin.get("registered_path"), "manifest.solution.registered_path"
    )
    challenge_source = _relative_to_project(challenge_path, project_root)
    config_source = _relative_to_project(config_path, project_root)
    solution_source = _relative_to_project(solution_path, project_root)
    if (
        challenge_source != challenge_pin.get("source_path")
        or config_source != config_pin.get("source_path")
        or solution_source != solution_pin.get("source_path")
    ):
        raise ValueError(f"{instance_id}: bundled injection path drift")

    theorem_names = list(config.get("theorem_names") or [])
    definition_names = list(config.get("definition_names") or [])
    permitted_axioms = list(config.get("permitted_axioms") or [])
    expected_declarations = sorted(theorem_names + definition_names)
    if expected_declarations != sorted(protected.get("declarations") or []):
        raise ValueError(f"{instance_id}: Comparator names differ from protected names")
    comparisons = {
        "challenge_module": challenge_pin.get("module"),
        "solution_module": solution_pin.get("module"),
        "theorem_names": formalization.get("theorem_names"),
        "definition_names": formalization.get("definition_names") or [],
        "permitted_axioms": formalization.get("permitted_axioms"),
    }
    for key, expected in comparisons.items():
        if config.get(key, [] if key == "definition_names" else None) != expected:
            raise ValueError(f"{instance_id}: bundled Comparator field {key} drifted")
    lean_toolchain = _text(
        manifest.get("lean_toolchain"), "manifest.lean_toolchain"
    )
    if formalization.get("lean_toolchain") != lean_toolchain:
        raise ValueError(f"{instance_id}: bundled Lean toolchain drifted")

    comparator_commit = _text(
        verification.get("comparator_commit"), "verification.comparator_commit"
    )
    lean4export_commit = _text(
        verification.get("lean4export_commit"), "verification.lean4export_commit"
    )
    landrun_commit = _text(
        verification.get("landrun_commit"), "verification.landrun_commit"
    )
    nanoda_commit = _text(
        verification.get("nanoda_commit"), "verification.nanoda_commit"
    )
    tools = {
        "comparator": _tool(
            root,
            tool_bundle_path("comparator", comparator_commit),
            comparator_commit,
        ),
        "lean4export": _tool(
            root,
            tool_bundle_path("lean4export", lean4export_commit),
            lean4export_commit,
        ),
        "landrun": _tool(
            root, tool_bundle_path("landrun", landrun_commit), landrun_commit
        ),
        "nanoda": _tool(
            root, tool_bundle_path("nanoda", nanoda_commit), nanoda_commit
        ),
        "landrun_wrapper": _tool(root, LANDRUN_WRAPPER, landrun_commit),
    }
    runtime_bytes, runtime_defaults = runtime_config_copy(config_bytes)
    configuration: dict[str, Any] = {
        "bundle_path": _text(config_pin.get("path"), "manifest.configuration.path"),
        "source_path": config_source,
        "sha256": _sha256_bytes(config_bytes),
        "runtime_copy": "byte_for_byte_registered_config",
        "enable_nanoda": bool(config.get("enable_nanoda", False)),
        "external_kernels": sorted(
            str(name) for name in (config.get("external_kernels") or {})
        ),
    }
    if runtime_defaults:
        configuration["runtime_copy"] = (
            "registered_config_with_documented_defaults"
        )
        configuration["runtime_defaults_applied"] = dict(runtime_defaults)
        configuration["runtime_sha256"] = _sha256_bytes(runtime_bytes)
    contract: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "instance_id": instance_id,
        "palomar_id": palomar_id,
        "comparator": dict(bundle_record),
        "registry_record": {
            "bundle_path": _text(record_pin.get("path"), "registry_record.path"),
            "sha256": _sha256_bytes(record_bytes),
            "version": record.get("version"),
        },
        "source_archive": {
            "sha256": _text(
                source_pin.get("archive_sha256"), "manifest.source.archive_sha256"
            ),
            "repository": repository,
            "commit": commit,
            "project_root": project_root or None,
            "required_at_runtime": False,
        },
        "challenge": {
            "bundle_path": _text(challenge_pin.get("path"), "challenge.path"),
            "registered_path": challenge_path,
            "source_path": challenge_source,
            "module": config["challenge_module"],
            "sha256": _sha256_bytes(challenge_bytes),
        },
        "configuration": configuration,
        "solution": {
            "registered_path": solution_path,
            "source_path": solution_source,
            "module": config["solution_module"],
        },
        "theorem_names": theorem_names,
        "definition_names": definition_names,
        "permitted_axioms": permitted_axioms,
        "lean_toolchain": lean_toolchain,
        "tools": tools,
    }
    reconstruction_targets = provenance.get("reconstruction_targets")
    if reconstruction_targets is not None:
        if (not isinstance(reconstruction_targets, list) or not reconstruction_targets
                or len(set(reconstruction_targets)) != len(reconstruction_targets)
                or not set(reconstruction_targets) <= set(theorem_names)):
            raise ValueError("invalid reconstruction targets in comparator provenance")
        contract["reconstruction_targets"] = list(reconstruction_targets)
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def resolve_palomar_contract(
    *, repo_root: Path, database_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve one standardized Palomar row to immutable verifier evidence."""

    root = repo_root.resolve()
    instance_id = _text(database_row.get("instance_id"), "database.instance_id")
    protected = _mapping(database_row.get("protected"), "database.protected")
    provenance = _mapping(protected.get("provenance"), "protected.provenance")
    if provenance.get("kind") == "leanlean_theorem_holdout_v1":
        from leanlean.holdout_comparator import resolve_holdout_contract
        return resolve_holdout_contract(repo_root=root, database_row=database_row)
    if provenance.get("kind") not in COMPARATOR_PROVENANCE_KINDS:
        raise ValueError(f"{instance_id}: no supported Comparator provenance")

    if provenance.get("comparator") is not None:
        return _resolve_bundled_contract(
            root=root,
            instance_id=instance_id,
            protected=protected,
            provenance=provenance,
        )

    mapping_path = _root_path(
        root, _text(provenance.get("registry"), "protected.provenance.registry"), "registry"
    )
    mapping_sha256 = _text(
        provenance.get("registry_sha256"), "protected.provenance.registry_sha256"
    )
    if _sha256_file(mapping_path) != mapping_sha256:
        raise ValueError(f"{instance_id}: frozen Palomar mapping drifted")
    mapping = _mapping(json.loads(mapping_path.read_text()), "Palomar mapping")

    targets = [
        _mapping(row, "Palomar target")
        for row in mapping.get("targets") or []
        if isinstance(row, Mapping) and row.get("benchmark_id") == instance_id
    ]
    results = [
        _mapping(row, "Palomar result")
        for row in mapping.get("results") or []
        if isinstance(row, Mapping) and row.get("benchmark_id") == instance_id
    ]
    if len(targets) != 1 or len(results) != 1:
        raise ValueError(
            f"{instance_id}: expected one frozen Palomar target/result, found "
            f"{len(targets)}/{len(results)}"
        )
    target, result = targets[0], results[0]
    palomar_id = _text(result.get("palomar_id"), "Palomar result id")
    expected_ids = provenance.get("palomar_ids")
    if expected_ids != [palomar_id]:
        raise ValueError(f"{instance_id}: Palomar result provenance drift")

    record_name = Path(_text(result.get("record_path"), "record_path")).name
    record_sha256 = _text(result.get("record_sha256"), "record_sha256")
    record_pin = _mapping(
        provenance.get("registry_record"),
        "protected.provenance.registry_record",
    )
    record_path = _pinned_file(
        root,
        record_pin,
        "Palomar registry record",
        expected_sha256=record_sha256,
    )
    if record_path.name != record_name:
        raise ValueError(f"{instance_id}: Palomar registry record path drift")
    record = _mapping(json.loads(record_path.read_text()), "Palomar registry record")
    verification = _mapping(record.get("verification"), "record.verification")
    formalization = _mapping(record.get("formalization"), "record.formalization")
    source = _mapping(record.get("source"), "record.source")
    repository = _text(source.get("repository"), "record.source.repository")
    commit = _text(source.get("commit"), "record.source.commit")
    if repository != target.get("repository") or commit != target.get("commit"):
        raise ValueError(f"{instance_id}: source archive provenance drift")

    archive_pin = _mapping(
        provenance.get("source_archive"),
        "protected.provenance.source_archive",
    )
    if (
        archive_pin.get("repository") != repository
        or archive_pin.get("commit") != commit
    ):
        raise ValueError(f"{instance_id}: source archive pin provenance drift")
    archive_path = _pinned_file(root, archive_pin, "Palomar source archive")

    project_root = str(target.get("project_path") or "").strip("/")
    challenge_path = _text(
        target.get("registered_challenge_path"), "target.registered_challenge_path"
    )
    config_path = _text(
        target.get("registered_comparator_config_path"),
        "target.registered_comparator_config_path",
    )
    solution_path = _text(
        target.get("registered_solution_path"), "target.registered_solution_path"
    )
    challenge_source = _relative_to_project(challenge_path, project_root)
    config_source = _relative_to_project(config_path, project_root)
    solution_source = _relative_to_project(solution_path, project_root)

    challenge_sha256 = _text(
        verification.get("challenge_sha256"), "verification.challenge_sha256"
    )
    challenge_bytes = _archive_member(archive_path, challenge_path)
    config_bytes = _archive_member(archive_path, config_path)
    if _sha256_bytes(challenge_bytes) != challenge_sha256:
        raise ValueError(f"{instance_id}: source archive Challenge drift")

    try:
        config = _mapping(json.loads(config_bytes), "registered Comparator config")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{instance_id}: invalid registered Comparator config") from error
    theorem_names = list(config.get("theorem_names") or [])
    definition_names = list(config.get("definition_names") or [])
    permitted_axioms = list(config.get("permitted_axioms") or [])
    expected_declarations = sorted(theorem_names + definition_names)
    actual_declarations = sorted(protected.get("declarations") or [])
    if expected_declarations != actual_declarations:
        raise ValueError(f"{instance_id}: Comparator names differ from protected names")
    comparisons = {
        "challenge_module": target.get("registered_challenge_module"),
        "solution_module": target.get("registered_solution_module"),
        "theorem_names": formalization.get("theorem_names"),
        "definition_names": formalization.get("definition_names") or [],
        "permitted_axioms": formalization.get("permitted_axioms"),
    }
    for key, expected in comparisons.items():
        if config.get(key, [] if key == "definition_names" else None) != expected:
            raise ValueError(f"{instance_id}: registered Comparator field {key} drifted")
    if formalization.get("lean_toolchain") != result.get("lean_toolchain"):
        raise ValueError(f"{instance_id}: registered Lean toolchain drifted")

    comparator_commit = _text(
        verification.get("comparator_commit"), "verification.comparator_commit"
    )
    lean4export_commit = _text(
        verification.get("lean4export_commit"), "verification.lean4export_commit"
    )
    landrun_commit = _text(
        verification.get("landrun_commit"), "verification.landrun_commit"
    )
    nanoda_commit = _text(
        verification.get("nanoda_commit"), "verification.nanoda_commit"
    )
    tools = {
        "comparator": _tool(
            root,
            tool_bundle_path("comparator", comparator_commit),
            comparator_commit,
        ),
        "lean4export": _tool(
            root,
            tool_bundle_path("lean4export", lean4export_commit),
            lean4export_commit,
        ),
        "landrun": _tool(
            root, tool_bundle_path("landrun", landrun_commit), landrun_commit
        ),
        "nanoda": _tool(
            root, tool_bundle_path("nanoda", nanoda_commit), nanoda_commit
        ),
        "landrun_wrapper": _tool(root, LANDRUN_WRAPPER, landrun_commit),
    }
    runtime_bytes, runtime_defaults = runtime_config_copy(config_bytes)
    configuration: dict[str, Any] = {
        "archive_path": config_path,
        "source_path": config_source,
        "sha256": _sha256_bytes(config_bytes),
        "runtime_copy": "byte_for_byte_registered_config",
        "enable_nanoda": bool(config.get("enable_nanoda", False)),
        "external_kernels": sorted(
            str(name) for name in (config.get("external_kernels") or {})
        ),
    }
    if runtime_defaults:
        # Recorded only when the registered config actually omitted a field, so
        # repositories that need no default keep a byte-identical contract.
        configuration["runtime_copy"] = "registered_config_with_documented_defaults"
        configuration["runtime_defaults_applied"] = dict(runtime_defaults)
        configuration["runtime_sha256"] = _sha256_bytes(runtime_bytes)
    contract: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "instance_id": instance_id,
        "palomar_id": palomar_id,
        "registry_mapping": {
            "path": str(mapping_path.relative_to(root)),
            "sha256": mapping_sha256,
        },
        "registry_record": {
            "path": str(record_path.relative_to(root)),
            "sha256": record_sha256,
            "version": record.get("version"),
        },
        "source_archive": {
            "path": str(archive_path.relative_to(root)),
            "sha256": _sha256_file(archive_path),
            "repository": repository,
            "commit": commit,
            "project_root": project_root or None,
        },
        "challenge": {
            "archive_path": challenge_path,
            "source_path": challenge_source,
            "module": config["challenge_module"],
            "sha256": _sha256_bytes(challenge_bytes),
        },
        "configuration": configuration,
        "solution": {
            "registered_path": solution_path,
            "source_path": solution_source,
            "module": config["solution_module"],
        },
        "theorem_names": theorem_names,
        "definition_names": definition_names,
        "permitted_axioms": permitted_axioms,
        "lean_toolchain": formalization.get("lean_toolchain"),
        "tools": tools,
    }
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def load_palomar_evidence(
    *, repo_root: Path, contract: Mapping[str, Any]
) -> tuple[bytes, bytes, bytes]:
    """Load the trusted Challenge and byte-identical registered config copies."""

    root = repo_root.resolve()
    challenge_pin = _mapping(contract.get("challenge"), "challenge")
    config_pin = _mapping(contract.get("configuration"), "configuration")
    comparator = contract.get("comparator")
    holdout = contract.get("holdout")
    if isinstance(holdout, Mapping):
        from leanlean.holdout_comparator import derive_evidence
        if holdout.get("schema") != "leanlean_comparator_holdout_v1":
            raise ValueError("unsupported comparator holdout")
        parent_challenge, parent_config, _ = load_palomar_evidence(
            repo_root=root, contract=_mapping(holdout.get("parent"), "holdout.parent")
        )
        challenge, original_config = derive_evidence(
            parent_challenge, parent_config, list(holdout["removed_theorems"])
        )
        closure = holdout.get("challenge_closure")
        if closure is not None:
            from leanlean.holdout_comparator import CHALLENGE_CLOSURE_SCHEMA
            if (
                not isinstance(closure, Mapping)
                or closure.get("schema") != CHALLENGE_CLOSURE_SCHEMA
                or closure.get("input_sha256") != _sha256_bytes(challenge)
            ):
                raise ValueError("holdout Challenge closure contract drifted")
            artifact = _root_path(
                root,
                _text(closure.get("path"), "challenge_closure.path"),
                "holdout Challenge closure",
            )
            challenge = artifact.read_bytes()
            if closure.get("output_sha256") != _sha256_bytes(challenge):
                raise ValueError("holdout Challenge closure artifact drifted")
        if json.loads(original_config)["theorem_names"] != contract.get("theorem_names"):
            raise ValueError("holdout comparator theorem selection drifted")
    elif isinstance(comparator, Mapping):
        _, manifest, challenge, original_config, _ = load_comparator_bundle(
            repo_root=root,
            bundle_record=comparator,
            instance_id=_text(contract.get("instance_id"), "contract.instance_id"),
        )
        manifest_challenge = _mapping(
            manifest.get("challenge"), "manifest.challenge"
        )
        manifest_config = _mapping(
            manifest.get("configuration"), "manifest.configuration"
        )
        if (
            challenge_pin.get("bundle_path") != manifest_challenge.get("path")
            or config_pin.get("bundle_path") != manifest_config.get("path")
        ):
            raise ValueError("registered Palomar bundle path drifted")
    else:
        archive_pin = _mapping(contract.get("source_archive"), "source_archive")
        archive = _root_path(
            root,
            _text(archive_pin.get("path"), "source_archive.path"),
            "source archive",
        )
        if _sha256_file(archive) != archive_pin.get("sha256"):
            raise ValueError("frozen Palomar source archive drifted")
        challenge = _archive_member(
            archive,
            _text(challenge_pin.get("archive_path"), "challenge.archive_path"),
        )
        original_config = _archive_member(
            archive,
            _text(config_pin.get("archive_path"), "configuration.archive_path"),
        )
    if _sha256_bytes(challenge) != challenge_pin.get("sha256"):
        raise ValueError("registered Palomar Challenge drifted")
    if _sha256_bytes(original_config) != config_pin.get("sha256"):
        raise ValueError("registered Palomar Comparator config drifted")
    runtime_config, applied = runtime_config_copy(original_config)
    expected_policy = (
        "registered_config_with_documented_defaults"
        if applied
        else "byte_for_byte_registered_config"
    )
    if config_pin.get("runtime_copy") != expected_policy:
        raise ValueError("registered Palomar Comparator runtime policy drifted")
    if applied and (
        config_pin.get("runtime_defaults_applied") != applied
        or _sha256_bytes(runtime_config) != config_pin.get("runtime_sha256")
    ):
        raise ValueError("registered Palomar Comparator runtime copy drifted")
    return challenge, original_config, runtime_config


def resolve_tool_path(
    *, repo_root: Path, tool: Mapping[str, Any], label: str
) -> Path:
    root = repo_root.resolve()
    path = _root_path(root, _text(tool.get("path"), f"{label}.path"), label)
    if not path.is_file() or _sha256_file(path) != tool.get("sha256"):
        raise ValueError(f"pinned Palomar tool drifted: {label}")
    return path


def install_palomar_tools(
    environment: Any, *, repo_root: Path, contract: Mapping[str, Any]
) -> None:
    """Inject the contract-pinned verifier binaries into one isolated container."""

    copy = getattr(environment, "copy_host_executable", None)
    if not callable(copy):
        raise RuntimeError("Palomar Comparator tool injection requires Docker")
    tools = _mapping(contract.get("tools"), "Palomar Comparator tools")
    for label, destination in TOOL_DESTINATIONS.items():
        pin = _mapping(tools.get(label), f"Palomar tool {label}")
        source = resolve_tool_path(repo_root=repo_root, tool=pin, label=label)
        copy(source, destination)


def install_agent_verify_wrapper(environment: Any, *, repo_root: Path) -> None:
    """Inject the stable, no-argument agent verification command."""

    copy = getattr(environment, "copy_host_executable", None)
    if not callable(copy):
        raise RuntimeError("Palomar Comparator wrapper injection requires Docker")
    root = repo_root.resolve()
    wrapper = (root / AGENT_VERIFY_WRAPPER).resolve()
    if not wrapper.is_file() or not wrapper.is_relative_to(root):
        raise FileNotFoundError(f"missing agent verification wrapper: {wrapper}")
    copy(wrapper, "/usr/local/bin/lean_verify")
