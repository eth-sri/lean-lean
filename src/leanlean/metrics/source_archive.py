"""Host-side compression metrics for deterministic source archives.

Benchmark containers are intentionally networkless and are not required to
ship Python. Source archives provide a stable boundary: containers only need
``tar`` while all parsing and counting happens in the trusted host process.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .tokens import count_lean_tokens_in_source, remove_lean_comments


@dataclass(frozen=True)
class SourceMetrics:
    words: int
    lean_tokens: int
    lean_file_count: int


def count_lean_words_in_source(source: str) -> int:
    """Count non-comment, non-import whitespace-delimited Lean words."""

    total = 0
    for line in remove_lean_comments(source).splitlines():
        if line.lstrip().startswith("import "):
            continue
        total += len(line.split())
    return total


def _normalized_path(name: str) -> str:
    return PurePosixPath(name).as_posix().removeprefix("./").strip("/")


def _path_in_scope(
    path: str,
    *,
    exclude_dirs: tuple[str, ...],
    exclude_files: frozenset[str],
    include_prefix: str,
) -> bool:
    if not path.endswith(".lean") or ".lake" in PurePosixPath(path).parts:
        return False
    if any(path == item or path.startswith(item + "/") for item in exclude_dirs):
        return False
    if path in exclude_files:
        return False
    if include_prefix and not (
        path == include_prefix + ".lean" or path.startswith(include_prefix + "/")
    ):
        return False
    return True


def measure_source_archive(
    archive_path: str | Path,
    *,
    exclude_dirs: list[str] | tuple[str, ...] | None = None,
    exclude_files: list[str] | tuple[str, ...] | None = None,
    include_prefix: str = "",
) -> SourceMetrics:
    """Measure one deterministic ``.tar.gz`` source snapshot on the host."""

    archive_path = Path(archive_path)
    normalized_excludes = tuple(
        item.strip().strip("/") for item in (exclude_dirs or ()) if item.strip()
    )
    normalized_excluded_files = frozenset(
        item.strip().strip("/")
        for item in (exclude_files or ())
        if item.strip()
    )
    include_prefix = include_prefix.strip().strip("/")
    words = 0
    lean_tokens = 0
    lean_file_count = 0

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = _normalized_path(member.name)
            if not _path_in_scope(
                path,
                exclude_dirs=normalized_excludes,
                exclude_files=normalized_excluded_files,
                include_prefix=include_prefix,
            ):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(
                    f"could not read source member {member.name!r} from {archive_path}"
                )
            source = stream.read().decode("utf-8", errors="replace")
            words += count_lean_words_in_source(source)
            # The established Lean-token and Lean-file metrics exclude the
            # project configuration file. The historical word metric did not.
            if PurePosixPath(path).name != "lakefile.lean":
                lean_tokens += count_lean_tokens_in_source(source)
                lean_file_count += 1

    return SourceMetrics(
        words=words,
        lean_tokens=lean_tokens,
        lean_file_count=lean_file_count,
    )


__all__ = [
    "SourceMetrics",
    "count_lean_words_in_source",
    "measure_source_archive",
]
