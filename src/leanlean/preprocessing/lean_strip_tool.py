"""The pinned lean-strip preprocessing engine.

Preprocessing runs exactly one lean-strip release: the git dependency pinned in
pyproject.toml and uv.lock. Every run resolves the installed package, refuses
any other version or commit, and copies that package together with the
interpreter running it into the network-less strip container.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

LEAN_STRIP_REPOSITORY = "https://github.com/eth-sri/lean-strip"
LEAN_STRIP_VERSION = "1.0.0"
LEAN_STRIP_COMMIT = "cbca97147bc4763dd2a5b27a923d3a122d3af39e"

CONTAINER_ROOT = "/opt/lean-strip"
CONTAINER_PYTHON_PREFIX = f"{CONTAINER_ROOT}/python"
CONTAINER_SITE = f"{CONTAINER_ROOT}/site"


@dataclass(frozen=True)
class LeanStripTool:
    repository: str
    version: str
    commit: str
    package_dir: Path
    dist_info_dir: Path
    package_tree_sha256: str
    python_prefix: Path
    python_version: str

    @property
    def container_python(self) -> str:
        major, minor = sys.version_info[:2]
        return f"{CONTAINER_PYTHON_PREFIX}/bin/python{major}.{minor}"

    def record(self) -> dict[str, str]:
        """The provenance recorded in every preprocessing report."""

        value = asdict(self)
        value.pop("package_dir")
        value.pop("dist_info_dir")
        value.pop("python_prefix")
        return value


def package_tree_sha256(package_dir: Path) -> str:
    """Hash the installed package sources, independent of bytecode caches."""

    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix()
        digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def resolve_lean_strip_tool() -> LeanStripTool:
    """Return the installed lean-strip, or fail unless it is the pinned release."""

    try:
        distribution = importlib.metadata.distribution("lean-strip")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("lean-strip is not installed; run `uv sync`") from error
    if distribution.version != LEAN_STRIP_VERSION:
        raise RuntimeError(
            f"lean-strip {distribution.version} is installed; preprocessing is "
            f"pinned to {LEAN_STRIP_VERSION} (run `uv sync`)"
        )
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    vcs = direct_url.get("vcs_info") or {}
    if (
        direct_url.get("url") != LEAN_STRIP_REPOSITORY
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != LEAN_STRIP_COMMIT
    ):
        raise RuntimeError(
            f"lean-strip must be installed from {LEAN_STRIP_REPOSITORY} at "
            f"{LEAN_STRIP_COMMIT}; found {direct_url or 'an unpinned install'}"
        )
    import lean_strip

    package_dir = Path(lean_strip.__file__).resolve().parent
    # The CLI reads its own version from this metadata.
    dist_info_dir = Path(distribution.locate_file(f"lean_strip-{distribution.version}.dist-info"))
    if not (dist_info_dir / "METADATA").is_file():
        raise RuntimeError(f"lean-strip package metadata is missing: {dist_info_dir}")
    prefix = Path(sys.base_prefix).resolve()
    major, minor = sys.version_info[:2]
    if not (prefix / "bin" / f"python{major}.{minor}").is_file() or str(prefix) in {
        "/usr",
        "/usr/local",
    }:
        raise RuntimeError(
            "the strip container needs a relocatable interpreter; create the "
            "environment with a uv-managed Python (`uv python install 3.12`)"
        )
    return LeanStripTool(
        repository=LEAN_STRIP_REPOSITORY,
        version=distribution.version,
        commit=LEAN_STRIP_COMMIT,
        package_dir=package_dir,
        dist_info_dir=dist_info_dir,
        package_tree_sha256=package_tree_sha256(package_dir),
        python_prefix=prefix,
        python_version=".".join(map(str, sys.version_info[:3])),
    )
