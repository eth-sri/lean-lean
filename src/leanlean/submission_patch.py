"""Tolerate no-op deletions when replaying a submitted patch onto its baseline.

A submitted patch is `git diff` taken inside the agent container.  If the agent
briefly materializes files from another commit (for example through
`git worktree add`) the extraction can emit a deletion for a path that never
existed in the baseline the patch is replayed onto.  `git apply` rejects such a
hunk ("No such file or directory") and the whole submission becomes
unreplayable, even though deleting an absent file leaves the tree exactly as it
was.

`drop_absent_deletions` removes only those hunks: whole-file deletions whose
target path is absent from the baseline.  Every other hunk, including deletions
of files that do exist, is kept byte-for-byte, so the reconstructed endpoint is
identical to what applying the patch would mean.  Callers use it only as a
fallback after a strict apply fails, and record the dropped paths.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_SECTION_START = "diff --git "
_HEADER = re.compile(r"^diff --git a/(?P<a>.+) b/(?P<b>.+)$")


def _sections(patch: str) -> list[str]:
    lines = patch.splitlines(keepends=True)
    sections: list[list[str]] = []
    for line in lines:
        if line.startswith(_SECTION_START) or not sections:
            sections.append([])
        sections[-1].append(line)
    return ["".join(section) for section in sections]


def _deleted_path(section: str) -> str | None:
    """Return the path of a whole-file deletion section, else None."""

    lines = section.splitlines()
    if not lines or not lines[0].startswith(_SECTION_START):
        return None
    if not any(line.startswith("deleted file mode ") for line in lines[1:6]):
        return None
    for line in lines:
        if line.startswith("--- a/"):
            return line[len("--- a/"):]
    match = _HEADER.match(lines[0])
    if match and match.group("a") == match.group("b"):
        return match.group("a")
    return None


def drop_absent_deletions(
    patch: str, exists: Callable[[str], bool]
) -> tuple[str, list[str]]:
    """Return the patch without deletions of absent paths, plus those paths."""

    kept: list[str] = []
    dropped: list[str] = []
    for section in _sections(patch):
        path = _deleted_path(section)
        if path is not None and not exists(path):
            dropped.append(path)
            continue
        kept.append(section)
    return "".join(kept), dropped
