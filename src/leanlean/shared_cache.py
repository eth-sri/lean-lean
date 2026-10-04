"""Content-addressed artifacts for evaluation-time Lean image assembly."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


def artifact_tree_sha256(root: Path) -> str:
    """Hash paths, modes, symlink targets, and bytes in an artifact tree."""

    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"artifact tree is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode()
        metadata = path.lstat()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(stat.S_IMODE(metadata.st_mode).to_bytes(4, "big"))
        if path.is_symlink():
            target = os.readlink(path).encode()
            digest.update(b"L")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif path.is_dir():
            digest.update(b"D")
        elif path.is_file():
            digest.update(b"F")
            digest.update(metadata.st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise ValueError(f"unsupported artifact path type: {path}")
    return digest.hexdigest()
