"""Source-only inventory of a repository tree for reconstruction experiments.

It never infers build dependencies from paths mentioned in comments.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_METADATA = frozenset({
    "lakefile.lean", "lakefile.toml", "lake-manifest.json", "lean-toolchain", ".gitignore",
})


def source_inventory(root: Path) -> tuple[list[Path], list[str]]:
    """Ignore build/Git state; retain Lean and explicit root build metadata."""
    kept, excluded = [], []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in {".git", ".lake"})
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(f"source-only input contains a symlink: {relative}")
            if path.suffix == ".lean" or (len(relative.parts) == 1 and name in ROOT_METADATA):
                kept.append(relative)
            else:
                excluded.append(relative.as_posix())
    return sorted(kept), sorted(excluded)
