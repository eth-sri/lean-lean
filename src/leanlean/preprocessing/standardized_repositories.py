"""Source-agnostic input contract for deterministic Lean preprocessing.

Every source adapter ends here.  A standardized repository is a standalone,
source-only Lean tree with known entry points, reviewed protected declarations,
an immutable raw image, and evidence that the raw tree clean-builds.  No source
family survives this boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from leanlean.identifiers import same_identifier
from leanlean.preprocessing.repositories import (
    file_sha256,
    repository_definition_sha256,
)
from leanlean.shared_cache import artifact_tree_sha256


STANDARDIZED_DATABASE_KIND = "leanlean_standardized_repository_database"
STANDARDIZED_DATABASE_SCHEMA_VERSION = 2
STANDARDIZATION_CONTRACT = "leanlean_standardized_raw_repository_v1"
# Rows preprocessed by lean-strip pin the upstream archive it consumed instead
# of a separately isolated raw tree.
LEAN_STRIP_SOURCE_CONTRACT = "leanlean_upstream_archive_lean_strip_v1"
REPOSITORY_DOCUMENT_BASENAMES = frozenset(
    {"copying", "licence", "license", "notice", "readme"}
)


@dataclass(frozen=True)
class StandardizedRepositorySpec:
    """One build-verified repository after source-specific standardization."""

    instance_id: str
    repository_url: str
    commit: str
    toolchain: str
    raw_image_tag: str | None
    raw_image_id: str | None
    entry_module: str
    target_dir: str
    build_target: str
    exclude_dirs: tuple[str, ...]
    protected_declarations: frozenset[str]
    protected_provenance: Mapping[str, Any]
    source_adapter: str
    raw_repository: Mapping[str, Any]
    standardization: Mapping[str, Any]

    @property
    def is_standardized(self) -> bool:
        return (
            same_identifier(self.standardization.get("contract"), STANDARDIZATION_CONTRACT)
            and self.standardization.get("clean_build_passed") is True
            and bool(self.standardization.get("source_run_id"))
            and bool(self.source_adapter)
        )

    @property
    def needs_raw_preparation(self) -> bool:
        return self.raw_image_tag is None

    @property
    def preprocessing_image_tag(self) -> str:
        if self.raw_image_tag is None:
            raise ValueError(
                f"{self.instance_id}: raw image must be materialized from its dataset"
            )
        return self.raw_image_tag


@dataclass(frozen=True)
class StandardizedRepositoryDatabase:
    path: Path
    sha256: str
    definition_sha256: str
    dataset_id: str
    dataset_version: str
    repositories: tuple[StandardizedRepositorySpec, ...]
    kind: str = STANDARDIZED_DATABASE_KIND
    schema_version: int = STANDARDIZED_DATABASE_SCHEMA_VERSION

    def by_id(self) -> dict[str, StandardizedRepositorySpec]:
        return {row.instance_id: row for row in self.repositories}


def is_repository_document(relative: Path) -> bool:
    """Return whether a source file is documentation clutter, never Lean."""

    name = relative.name.casefold()
    if name.endswith(".lean"):
        return False
    return name.endswith(".md") or name.split(".", 1)[0] in (
        REPOSITORY_DOCUMENT_BASENAMES
    )


def repository_document_paths(root: Path) -> tuple[str, ...]:
    """List documentation files removed at the repository-isolation boundary."""

    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and is_repository_document(path.relative_to(root))
    )


def remove_repository_documents(root: Path) -> tuple[str, ...]:
    """Remove documentation files and now-empty directories deterministically."""

    removed = repository_document_paths(root)
    for relative in removed:
        (root / relative).unlink()
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def _source_files(root: Path, *, exclude_documents: bool) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", ".lake"}:
            continue
        if exclude_documents and is_repository_document(relative):
            continue
        files.append(path)
    return files


def _source_tree_sha256(root: Path, *, exclude_documents: bool) -> str:
    """Hash a source tree, excluding build/Git state and optional documents."""

    digest = hashlib.sha256()
    for path in _source_files(root, exclude_documents=exclude_documents):
        relative = path.relative_to(root)
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(path.stat().st_mode.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash a standardized source tree, excluding build and Git state."""

    return _source_tree_sha256(root, exclude_documents=False)


def isolated_source_tree_sha256(root: Path) -> str:
    """Hash the source tree produced after repository-document cleanup."""

    return _source_tree_sha256(root, exclude_documents=True)


def source_tree_stats(root: Path) -> dict[str, int]:
    """Return source-only file counts and bytes for a materialized tree."""

    files = _source_files(root, exclude_documents=False)
    return {
        "file_count": len(files),
        "lean_file_count": sum(path.suffix.casefold() == ".lean" for path in files),
        "source_bytes": sum(path.stat().st_size for path in files),
    }


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _validate_provenance_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    context: str,
) -> None:
    relative = record.get(path_key)
    expected = record.get(sha_key)
    if relative is None and expected is None:
        return
    if not isinstance(relative, str) or not relative or not isinstance(expected, str):
        raise ValueError(f"{context} must pin both {path_key} and {sha_key}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"{context}.{path_key} is missing or outside the repo")
    actual = file_sha256(resolved)
    if actual != expected:
        raise ValueError(
            f"{context}.{path_key} drift: expected {expected}, found {actual}"
        )


def _load_protected_names(
    root: Path, protected: Mapping[str, Any], *, context: str
) -> tuple[frozenset[str], Mapping[str, Any]]:
    inline = protected.get("declarations")
    file_record = protected.get("file")
    if (inline is None) == (file_record is None):
        raise ValueError(
            f"{context}.protected must define exactly one of declarations or file"
        )
    provenance = protected.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError(f"{context}.protected.provenance must be a mapping")
    if provenance.get("comparator") is None:
        _validate_provenance_file(
            root,
            provenance,
            path_key="registry",
            sha_key="registry_sha256",
            context=f"{context}.protected.provenance",
        )
    _validate_provenance_file(
        root,
        provenance,
        path_key="report",
        sha_key="report_sha256",
        context=f"{context}.protected.provenance",
    )
    resolution = provenance.get("resolution_evidence", {})
    if not isinstance(resolution, Mapping):
        raise ValueError(f"{context}.protected.provenance.resolution_evidence")
    _validate_provenance_file(
        root,
        resolution,
        path_key="report",
        sha_key="sha256",
        context=f"{context}.protected.provenance.resolution_evidence",
    )

    if inline is not None:
        names = inline
    else:
        if not isinstance(file_record, Mapping):
            raise ValueError(f"{context}.protected.file must be a mapping")
        relative = _require_string(file_record, "path", f"{context}.protected.file")
        expected = _require_string(file_record, "sha256", f"{context}.protected.file")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"{context}: protected declaration file is missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"{context}: protected declaration file drift: "
                f"expected {expected}, found {actual}"
            )
        payload = json.loads(path.read_text())
        names = [
            *payload.get("signature_theorems", []),
            *payload.get("protected_definitions", []),
        ]
        count = file_record.get("count")
        if count is not None and int(count) != len(set(names)):
            raise ValueError(
                f"{context}: protected declaration count does not match {path}"
            )
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(f"{context}.protected declarations must be non-empty names")
    if len(names) != len(set(names)):
        raise ValueError(f"{context}.protected declarations contain duplicates")
    return frozenset(names), provenance


def _parse_repository(
    root: Path,
    row: Mapping[str, Any],
    index: int,
    *,
    schema_version: int,
) -> StandardizedRepositorySpec:
    context = f"repositories[{index}]"
    expected_fields = {"instance_id", "source", "scopes", "protected"}
    if schema_version == 1:
        expected_fields.add("variants")
    if set(row) != expected_fields:
        raise ValueError(
            f"{context}: fields must be {sorted(expected_fields)}, "
            f"found {sorted(row)}"
        )
    source = row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{context}.source must be a mapping")
    if source.get("prepared_image") is not None:
        raise ValueError(f"{context}: standardized rows cannot request preparation")

    raw_image = source.get("raw_image")
    if raw_image is None:
        raw_tag = None
        raw_id = None
    else:
        if not isinstance(raw_image, Mapping):
            raise ValueError(f"{context}.source.raw_image must be a mapping")
        raw_tag = _require_string(raw_image, "tag", f"{context}.source.raw_image")
        raw_id = _require_string(raw_image, "image_id", f"{context}.source.raw_image")
        if not raw_id.startswith("sha256:"):
            raise ValueError(f"{context}.source.raw_image.image_id must be immutable")

    standardization = source.get("standardization")
    if not isinstance(standardization, Mapping):
        raise ValueError(f"{context}.source.standardization must be a mapping")
    if standardization.get("contract") == LEAN_STRIP_SOURCE_CONTRACT:
        if "raw_repository" in source:
            raise ValueError(f"{context}: lean-strip rows pin the upstream archive, not a raw tree")
        archive = standardization.get("source_archive")
        if not isinstance(archive, Mapping) or not isinstance(standardization.get("lean_strip"), Mapping):
            raise ValueError(f"{context}: lean-strip rows must pin source_archive and lean_strip")
        _require_string(archive, "sha256", f"{context}.source.standardization.source_archive")
        raw_repository: Mapping[str, Any] = {}
    else:
        raw_repository = _parse_raw_repository(root, source, context)

    if not any(same_identifier(standardization.get("contract"), contract)
               for contract in (STANDARDIZATION_CONTRACT, LEAN_STRIP_SOURCE_CONTRACT)):
        raise ValueError(f"{context}: unsupported standardized raw contract")
    if standardization.get("clean_build_passed") is not True:
        raise ValueError(f"{context}: standardized raw build was not verified")
    adapter = _require_string(
        standardization, "adapter", f"{context}.source.standardization"
    )
    _require_string(
        standardization, "source_run_id", f"{context}.source.standardization"
    )

    scopes = row.get("scopes", {})
    if not isinstance(scopes, Mapping):
        raise ValueError(f"{context}.scopes must be a mapping")
    exclude_dirs = scopes.get("exclude_dirs", [])
    if not isinstance(exclude_dirs, list) or not all(
        isinstance(value, str) for value in exclude_dirs
    ):
        raise ValueError(f"{context}.scopes.exclude_dirs must be a string list")
    protected = row.get("protected")
    if not isinstance(protected, Mapping):
        raise ValueError(f"{context}.protected must be a mapping")
    names, provenance = _load_protected_names(root, protected, context=context)
    if schema_version == 1 and not isinstance(row.get("variants"), Mapping):
        raise ValueError(f"{context}.variants must be a mapping")
    return StandardizedRepositorySpec(
        instance_id=_require_string(row, "instance_id", context),
        repository_url=_require_string(source, "repository_url", f"{context}.source"),
        commit=_require_string(source, "commit", f"{context}.source"),
        toolchain=_require_string(source, "toolchain", f"{context}.source"),
        raw_image_tag=raw_tag,
        raw_image_id=raw_id,
        entry_module=str(scopes.get("entry_module", "")),
        target_dir=str(scopes.get("target_dir", "")),
        build_target=str(scopes.get("build_target", "")),
        exclude_dirs=tuple(exclude_dirs),
        protected_declarations=names,
        protected_provenance=provenance,
        source_adapter=adapter,
        raw_repository=dict(raw_repository),
        standardization=dict(standardization),
    )


def _parse_raw_repository(
    root: Path, source: Mapping[str, Any], context: str
) -> Mapping[str, Any]:
    raw_repository = source.get("raw_repository")
    if not isinstance(raw_repository, Mapping):
        raise ValueError(f"{context}.source.raw_repository must be a mapping")
    relative = _require_string(
        raw_repository, "path", f"{context}.source.raw_repository"
    )
    raw_path = (root / relative).resolve()
    allowed_raw_roots = (
        (root / "prod_repos").resolve(),
        (root / "datasets").resolve(),
    )
    if (
        not any(raw_path.is_relative_to(allowed) for allowed in allowed_raw_roots)
        or not raw_path.is_dir()
    ):
        raise ValueError(f"{context}: standardized raw tree is missing or unsafe")
    digest = _require_string(
        raw_repository, "tree_sha256", f"{context}.source.raw_repository"
    )
    try:
        valid_digest = len(digest) == 64 and int(digest, 16) >= 0
    except ValueError:
        valid_digest = False
    if not valid_digest:
        raise ValueError(f"{context}: invalid standardized raw tree digest")
    for key in ("file_count", "lean_file_count", "source_bytes"):
        value = raw_repository.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{context}.source.raw_repository.{key} is invalid")
    build_cache = raw_repository.get("build_cache")
    if build_cache is not None:
        if not isinstance(build_cache, Mapping):
            raise ValueError(f"{context}.source.raw_repository.build_cache must be a mapping")
        cache_relative = _require_string(
            build_cache, "path", f"{context}.source.raw_repository.build_cache"
        )
        cache_path = (root / cache_relative).resolve()
        if cache_path == raw_path / ".lake" and not cache_path.exists():
            # A build cache is local build state; the published bundle omits it.
            return {**raw_repository, "build_cache": {**build_cache, "unavailable": True}}
        if (
            cache_path != raw_path / ".lake"
            or not cache_path.is_dir()
            or (cache_path / "packages").exists()
        ):
            raise ValueError(
                f"{context}: raw build cache must be raw/.lake without packages"
            )
        cache_digest = _require_string(
            build_cache,
            "tree_sha256",
            f"{context}.source.raw_repository.build_cache",
        )
        if artifact_tree_sha256(cache_path) != cache_digest:
            raise ValueError(f"{context}: raw build cache drift")
        if build_cache.get("packages_persisted") is not False:
            raise ValueError(f"{context}: dependency packages cannot be persisted per repo")
    return raw_repository


def load_standardized_repository_database(
    path: Path, *, repo_root: Path
) -> StandardizedRepositoryDatabase:
    resolved = path if path.is_absolute() else repo_root / path
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved}: expected a JSON object")
    if not same_identifier(payload.get("kind"), STANDARDIZED_DATABASE_KIND):
        raise ValueError(f"{resolved}: unexpected standardized database kind")
    schema_version = payload.get("schema_version")
    if schema_version not in {1, STANDARDIZED_DATABASE_SCHEMA_VERSION}:
        raise ValueError(
            f"{resolved}: unsupported schema_version {schema_version!r}"
        )
    expected_fields = {
        "kind",
        "schema_version",
        "dataset_id",
        "dataset_version",
        "repositories",
    }
    if schema_version == 1:
        expected_fields.add("source_presets")
    if set(payload) != expected_fields:
        raise ValueError(
            f"{resolved}: fields must be {sorted(expected_fields)}, "
            f"found {sorted(payload)}"
        )
    dataset_id = _require_string(payload, "dataset_id", str(resolved))
    dataset_version = _require_string(payload, "dataset_version", str(resolved))
    rows = payload.get("repositories")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{resolved}: repositories must be a non-empty list")
    repositories = tuple(
        _parse_repository(
            repo_root,
            row,
            index,
            schema_version=int(schema_version),
        )
        for index, row in enumerate(rows)
        if isinstance(row, Mapping)
    )
    if len(repositories) != len(rows):
        raise ValueError(f"{resolved}: every repository row must be an object")
    ids = [row.instance_id for row in repositories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{resolved}: duplicate instance_id")
    return StandardizedRepositoryDatabase(
        path=resolved,
        sha256=file_sha256(resolved),
        definition_sha256=repository_definition_sha256(payload),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        repositories=repositories,
        schema_version=int(schema_version),
    )
