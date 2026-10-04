"""Palomar registry inputs for preprocessing: provenance checks, archives, Comparator bundles.

Each benchmark repository is one registry result. These helpers verify that
result against the hash-pinned candidate mapping, download the upstream archive
at the pinned commit and freeze the evaluation-only Comparator inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from leanlean.palomar_comparator import (
    COMPARATOR_BUNDLE_MANIFEST,
    COMPARATOR_BUNDLE_SCHEMA,
)
from scripts.generate_palomar_candidate_mapping import (
    _archive_bytes,
    _validate_comparator,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_SAFE_PATH = re.compile(r"[A-Za-z0-9_./-]+")
_SAFE_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _root_path(value: str, name: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"{name} must remain inside the repository")
    return resolved


def _run(
    command: list[str],
    *,
    timeout: int | None = None,
    log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if log is None:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n{tail}"
        )
    return result


def _archive_name(repository: str, commit: str) -> str:
    owner, name = repository.split("/", 1)
    return f"{owner}--{name}--{commit}.tar.gz"


def _ensure_archive(
    repository: str,
    commit: str,
    archive_dir: Path,
    *,
    timeout: int,
    mirrors: tuple[str, ...] = (),
) -> Path:
    """Download the pinned commit, from Palomar's preserved fork if upstream lost it."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / _archive_name(repository, commit)
    if destination.is_file():
        try:
            with tarfile.open(destination, "r:gz") as archive:
                next(iter(archive))
            return destination
        except (tarfile.TarError, StopIteration):
            destination.unlink()
    with tempfile.NamedTemporaryFile(
        dir=archive_dir, prefix=destination.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        for source in (repository, *mirrors):
            url = f"https://codeload.github.com/{source}/tar.gz/{commit}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "leanlean-palomar/1"}
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    with temporary_path.open("wb") as output:
                        shutil.copyfileobj(response, output)
                break
            except urllib.error.HTTPError as error:
                if error.code != 404 or source == (repository, *mirrors)[-1]:
                    raise
        with tarfile.open(temporary_path, "r:gz") as archive:
            next(iter(archive))
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _ensure_registry_record(
    result: Mapping[str, Any],
    *,
    registry_url: str,
    record_dir: Path,
    timeout: int,
) -> Path:
    """Materialize one mapping-pinned Palomar record inside this source run."""

    record_path = PurePosixPath(
        _text(result.get("record_path"), "result.record_path")
    )
    if record_path.is_absolute() or ".." in record_path.parts:
        raise ValueError(f"unsafe Palomar record path: {record_path}")
    expected_sha256 = _text(
        result.get("record_sha256"), "result.record_sha256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("result.record_sha256 must be a lowercase SHA-256")
    record_dir.mkdir(parents=True, exist_ok=True)
    destination = record_dir / record_path.name
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        return destination

    url = urllib.parse.urljoin(registry_url, record_path.as_posix())
    with tempfile.NamedTemporaryFile(
        dir=record_dir, prefix=destination.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "leanlean-palomar/1"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Palomar record hash drift: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _archive_project_files(
    archive_path: Path,
    *,
    project_root: str,
    excluded_paths: set[str],
) -> tuple[dict[str, str], str]:
    sources: dict[str, str] = {}
    toolchain = ""
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            repository_path = PurePosixPath(*parts[1:]).as_posix()
            if project_root:
                prefix = project_root.rstrip("/") + "/"
                if not repository_path.startswith(prefix):
                    continue
                relative = repository_path[len(prefix) :]
            else:
                relative = repository_path
            if repository_path in excluded_paths:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"could not read {member.name!r}")
            if relative == "lean-toolchain":
                toolchain = stream.read().decode("utf-8", errors="replace").strip()
            elif relative.endswith(".lean") and ".lake" not in PurePosixPath(relative).parts:
                sources[relative] = stream.read().decode(
                    "utf-8", errors="replace"
                )
    if not toolchain:
        raise ValueError(f"{archive_path}: project has no lean-toolchain")
    if not sources:
        raise ValueError(f"{archive_path}: project has no Lean source")
    return sources, toolchain


def _relative_exclusions(project_root: str, paths: tuple[str, ...]) -> tuple[str, ...]:
    relative: list[str] = []
    prefix = project_root.rstrip("/")
    for path in paths:
        path = path.strip("/")
        if prefix:
            expected = prefix + "/"
            if not path.startswith(expected):
                raise ValueError(
                    f"excluded path {path!r} is outside project root {project_root!r}"
                )
            path = path[len(expected) :]
        if not path or not _SAFE_PATH.fullmatch(path) or any(
            part in {".", ".."} for part in PurePosixPath(path).parts
        ):
            raise ValueError(f"unsafe excluded path {path!r}")
        relative.append(path)
    return tuple(sorted(set(relative)))


def _verified_sha256(value: Any, content: bytes, name: str) -> str:
    expected = _text(value, name)
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{name} must be a SHA-256")
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"{name} does not match the pinned source archive")
    return actual


@dataclass(frozen=True)
class PreparedTarget:
    instance_id: str
    repository: str
    commit: str
    project_root: str
    toolchain: str
    toolchain_url: str
    toolchain_sha256: str
    registry_toolchains: tuple[str, ...]
    repository_roles: tuple[str, ...]
    solution_source: str
    registered_solution_path: str
    solution_sha256: str
    entry_module: str
    challenge_source: str
    registered_challenge_path: str
    challenge_sha256: str
    challenge_module: str
    comparator_config_source: str
    registered_comparator_config_path: str
    comparator_config_sha256: str
    excluded_paths: tuple[str, ...]
    protected_declarations: tuple[str, ...]
    original_registered_declarations: tuple[str, ...]
    palomar_ids: tuple[str, ...]
    image_tag: str
    registry_record_path: Path
    registry_record_sha256: str
    archive_path: Path
    archive_sha256: str


def _write_comparator_bundle(
    target: PreparedTarget, destination: Path
) -> dict[str, str]:
    """Freeze every evaluation-only Palomar input in one comparator directory."""

    root = REPO_ROOT.resolve()
    destination = destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"{target.instance_id}: comparator bundle escaped repository")
    challenge = _archive_bytes(
        target.archive_path, target.registered_challenge_path
    )
    configuration = _archive_bytes(
        target.archive_path, target.registered_comparator_config_path
    )
    registry_record = target.registry_record_path.read_bytes()
    expected = {
        "Challenge.lean": (challenge, target.challenge_sha256),
        "comparator.json": (configuration, target.comparator_config_sha256),
        "registry-record.json": (
            registry_record,
            target.registry_record_sha256,
        ),
    }
    for name, (content, sha256) in expected.items():
        if hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError(f"{target.instance_id}: {name} changed before bundling")
    manifest = {
        "schema": COMPARATOR_BUNDLE_SCHEMA,
        "instance_id": target.instance_id,
        "palomar_id": target.palomar_ids[0],
        "lean_toolchain": target.toolchain,
        "source": {
            "repository": target.repository,
            "commit": target.commit,
            "project_root": target.project_root or None,
            "archive_sha256": target.archive_sha256,
        },
        "registry_record": {
            "path": "registry-record.json",
            "sha256": target.registry_record_sha256,
        },
        "challenge": {
            "path": "Challenge.lean",
            "sha256": target.challenge_sha256,
            "registered_path": target.registered_challenge_path,
            "source_path": target.challenge_source,
            "module": target.challenge_module,
        },
        "configuration": {
            "path": "comparator.json",
            "sha256": target.comparator_config_sha256,
            "registered_path": target.registered_comparator_config_path,
            "source_path": target.comparator_config_source,
        },
        "solution": {
            "registered_path": target.registered_solution_path,
            "source_path": target.solution_source,
            "module": target.entry_module,
            "sha256": target.solution_sha256,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode()
        + b"\n"
    )
    files = {
        "Challenge.lean": challenge,
        "comparator.json": configuration,
        "registry-record.json": registry_record,
        COMPARATOR_BUNDLE_MANIFEST: manifest_bytes,
    }
    if destination.exists():
        actual = {
            path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual != files:
            raise ValueError(
                f"{target.instance_id}: existing comparator bundle drifted"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".comparator-", dir=destination.parent)
        )
        try:
            for relative, content in files.items():
                (staging / relative).write_bytes(content)
            staging.replace(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return {
        "path": destination.relative_to(root).as_posix(),
        "manifest": COMPARATOR_BUNDLE_MANIFEST,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _prepare_target(
    target: Mapping[str, Any],
    results_by_id: Mapping[str, Mapping[str, Any]],
    *,
    toolchain_assets: Mapping[str, Any],
    registry_url: str,
    record_dir: Path,
    record_timeout: int,
    archive_dir: Path,
    archive_timeout: int,
    image_tag_prefix: str,
    mapping_sha256: str,
    source_run_id: str,
) -> PreparedTarget:
    repository = _text(target.get("repository"), "target.repository")
    commit = _text(target.get("commit"), "target.commit")
    project_root = str(target.get("project_path") or "").strip("/")
    if project_root and not _SAFE_PATH.fullmatch(project_root):
        raise ValueError(f"unsafe project root {project_root!r}")

    registered_solution_path = _text(
        target.get("registered_solution_path"),
        "target.registered_solution_path",
    ).strip("/")
    solution_source = _text(
        target.get("solution_source"), "target.solution_source"
    ).strip("/")
    expected_solution_sources = _relative_exclusions(
        project_root, (registered_solution_path,)
    )
    if expected_solution_sources != (solution_source,):
        raise ValueError(
            f"{repository} at {commit}: registered Solution path/project drift"
        )

    registered_challenge_path = _text(
        target.get("registered_challenge_path"),
        "target.registered_challenge_path",
    ).strip("/")
    challenge_source = _text(
        target.get("challenge_source"), "target.challenge_source"
    ).strip("/")
    expected_challenge_sources = _relative_exclusions(
        project_root, (registered_challenge_path,)
    )
    if expected_challenge_sources != (challenge_source,):
        raise ValueError(
            f"{repository} at {commit}: registered Challenge path/project drift"
        )

    registered_comparator_config_path = _text(
        target.get("registered_comparator_config_path"),
        "target.registered_comparator_config_path",
    ).strip("/")
    comparator_sources = _relative_exclusions(
        project_root, (registered_comparator_config_path,)
    )
    comparator_config_source = comparator_sources[0]

    excluded_repository_paths = tuple(
        str(path).strip("/") for path in target.get("excluded_challenge_paths", [])
    )
    if excluded_repository_paths != (registered_challenge_path,):
        raise ValueError(
            f"{repository} at {commit}: isolation must remove exactly the "
            "registered Challenge path"
        )
    excluded_paths = tuple(
        sorted({challenge_source, comparator_config_source})
    )
    palomar_ids = tuple(sorted(target.get("palomar_ids", [])))
    if len(palomar_ids) != 1:
        raise ValueError(
            f"{repository} at {commit}: expected exactly one Palomar result"
        )
    result_rows = [results_by_id[identifier] for identifier in palomar_ids]
    result = result_rows[0]
    registry_record_path = _ensure_registry_record(
        result,
        registry_url=registry_url,
        record_dir=record_dir,
        timeout=record_timeout,
    )
    preserved = tuple(
        str(row["fork_repository"])
        for row in json.loads(registry_record_path.read_text())
        .get("preservation", {})
        .get("repositories", [])
        if row.get("source_repository") == repository and row.get("commit") == commit
    )
    archive_path = _ensure_archive(
        repository, commit, archive_dir, timeout=archive_timeout, mirrors=preserved
    )

    solution_bytes = _archive_bytes(archive_path, registered_solution_path)
    challenge_bytes = _archive_bytes(archive_path, registered_challenge_path)
    comparator_bytes = _archive_bytes(
        archive_path, registered_comparator_config_path
    )
    solution_sha256 = _verified_sha256(
        target.get("registered_solution_sha256"),
        solution_bytes,
        "target.registered_solution_sha256",
    )
    challenge_sha256 = _verified_sha256(
        target.get("registered_challenge_sha256"),
        challenge_bytes,
        "target.registered_challenge_sha256",
    )
    comparator_config_sha256 = _verified_sha256(
        target.get("registered_comparator_config_sha256"),
        comparator_bytes,
        "target.registered_comparator_config_sha256",
    )
    comparator = json.loads(comparator_bytes.decode("utf-8"))
    if not isinstance(comparator, dict):
        raise ValueError(
            f"{archive_path}:{registered_comparator_config_path}: "
            "expected JSON object"
        )
    (
        comparator_solution_module,
        comparator_challenge_module,
        comparator_theorem_names,
        comparator_definition_names,
        comparator_permitted_axioms,
    ) = _validate_comparator(
        comparator,
        label=f"{archive_path}:{registered_comparator_config_path}",
    )

    sources, toolchain = _archive_project_files(
        archive_path,
        project_root=project_root,
        excluded_paths=set(excluded_repository_paths),
    )
    if (
        not solution_source.endswith(".lean")
        or solution_source not in sources
        or solution_source in excluded_paths
    ):
        raise ValueError(
            f"{repository} at {commit}: registered Solution source is absent "
            f"after Challenge isolation: {solution_source!r}"
        )
    if challenge_source in sources or challenge_source not in excluded_paths:
        raise ValueError(
            f"{repository} at {commit}: registered Challenge was not isolated"
        )
    entry_module = _text(
        target.get("registered_solution_module"),
        "target.registered_solution_module",
    )
    if not _SAFE_MODULE.fullmatch(entry_module):
        raise ValueError(f"unsafe registered Solution module {entry_module!r}")
    challenge_module = _text(
        target.get("registered_challenge_module"),
        "target.registered_challenge_module",
    )
    if not _SAFE_MODULE.fullmatch(challenge_module):
        raise ValueError(f"unsafe registered Challenge module {challenge_module!r}")
    toolchain_asset = _mapping(
        toolchain_assets.get(toolchain), f"toolchains.{toolchain}"
    )
    toolchain_url = _text(
        toolchain_asset.get("url"), f"toolchains.{toolchain}.url"
    )
    toolchain_sha256 = _text(
        toolchain_asset.get("sha256"), f"toolchains.{toolchain}.sha256"
    )
    original = tuple(
        sorted(
            set(target.get("registered_theorem_names", []))
            | set(target.get("registered_definition_names", []))
        )
    )
    protected = original
    registry_record_sha256 = _text(
        result.get("record_sha256"), "result.record_sha256"
    )
    if (
        result.get("benchmark_id") != target.get("benchmark_id")
        or result.get("solution_path") != registered_solution_path
        or result.get("solution_sha256") != solution_sha256
        or result.get("solution_source") != solution_source
        or result.get("solution_module") != entry_module
        or result.get("challenge_path") != registered_challenge_path
        or result.get("challenge_sha256") != challenge_sha256
        or result.get("challenge_module") != challenge_module
        or result.get("comparator_config_path")
        != registered_comparator_config_path
        or result.get("comparator_config_sha256")
        != comparator_config_sha256
        or result.get("target_repository") != repository
        or result.get("target_commit") != commit
    ):
        raise ValueError(
            f"{repository} at {commit}: result/target registry provenance drift"
        )
    if (
        comparator_solution_module != entry_module
        or comparator_challenge_module != challenge_module
        or comparator_theorem_names != result.get("theorem_names")
        or comparator_definition_names != result.get("definition_names")
        or comparator_permitted_axioms != result.get("permitted_axioms")
        or sorted(comparator_theorem_names)
        != sorted(target.get("registered_theorem_names", []))
        or sorted(comparator_definition_names)
        != sorted(target.get("registered_definition_names", []))
        or sorted(comparator_permitted_axioms)
        != sorted(target.get("registered_permitted_axioms", []))
    ):
        raise ValueError(
            f"{repository} at {commit}: registered Comparator contract drift"
        )
    registered_toolchains = {str(row["lean_toolchain"]) for row in result_rows}
    repository_roles = {str(row["repository_role"]) for row in result_rows}
    if registered_toolchains != {toolchain}:
        raise ValueError(
            f"{repository} at {commit}: registry/source toolchain mismatch: "
            f"{sorted(registered_toolchains)} != {toolchain!r}"
        )
    instance_id = _text(target.get("benchmark_id"), "target.benchmark_id")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", source_run_id)
    image_tag = f"{image_tag_prefix}-{instance_id}:{safe_run_id}"
    return PreparedTarget(
        instance_id=instance_id,
        repository=repository,
        commit=commit,
        project_root=project_root,
        toolchain=toolchain,
        toolchain_url=toolchain_url,
        toolchain_sha256=toolchain_sha256,
        registry_toolchains=tuple(sorted(registered_toolchains)),
        repository_roles=tuple(sorted(repository_roles)),
        solution_source=solution_source,
        registered_solution_path=registered_solution_path,
        solution_sha256=solution_sha256,
        entry_module=entry_module,
        challenge_source=challenge_source,
        registered_challenge_path=registered_challenge_path,
        challenge_sha256=challenge_sha256,
        challenge_module=challenge_module,
        comparator_config_source=comparator_config_source,
        registered_comparator_config_path=registered_comparator_config_path,
        comparator_config_sha256=comparator_config_sha256,
        excluded_paths=excluded_paths,
        protected_declarations=tuple(protected),
        original_registered_declarations=original,
        palomar_ids=palomar_ids,
        image_tag=image_tag,
        registry_record_path=registry_record_path,
        registry_record_sha256=registry_record_sha256,
        archive_path=archive_path,
        archive_sha256=file_sha256(archive_path),
    )


def localize_lake_manifest(source: Path) -> dict[str, Any]:
    """Override pinned Git packages with paths into the shared environment.

    Keep the published manifest byte-for-byte intact: mathlib's cache client
    needs its Git pins to compute the right binary-cache hashes.  Lake applies
    ``.lake/package-overrides.json`` after reading that manifest, so builds use
    the already supplied package trees without inspecting or cloning Git repos.
    """

    manifest_path = source / "lake-manifest.json"
    payload = json.loads(manifest_path.read_text())
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise ValueError(f"{manifest_path}: packages must be a list")
    overrides: list[dict[str, Any]] = []
    for index, raw in enumerate(packages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{manifest_path}: package {index} must be a mapping")
        package = dict(raw)
        if package.get("type") == "git":
            name = _text(
                package.get("name"), f"{manifest_path}.packages[{index}].name"
            )
            if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
                raise ValueError(f"{manifest_path}: unsafe package name {name!r}")
            package = {
                key: value
                for key, value in package.items()
                if key not in {"url", "rev", "inputRev", "subDir"}
            }
            package.update(
                {
                    "type": "path",
                    "dir": f".lake/packages/{name}",
                }
            )
        elif package.get("type") != "path":
            raise ValueError(
                f"{manifest_path}: unsupported package source type "
                f"{package.get('type')!r}"
            )
        overrides.append(package)
    override_path = source / ".lake" / "package-overrides.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(
        json.dumps(
            {"schemaVersion": payload.get("version", "1.2.0"), "packages": overrides},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return {
        "mode": "lake_package_overrides_v1",
        "pinned_git_packages": sum(
            package.get("type") == "git"
            for package in packages
            if isinstance(package, Mapping)
        ),
        "manifest_sha256": file_sha256(manifest_path),
        "overrides_sha256": file_sha256(override_path),
    }
