"""lean-strip's workspace interface over a running Docker container.

lean-strip reads `.ilean` files and project sources through a small workspace
interface; `DockerWorkspace` provides it for a running `DockerEnvironment`, whose
`/testbed` and `/tmp` paths lean-strip already uses. The paper's dependency
graphs (scripts/analysis/extract_paper_graphs.py) read `.ilean` files and
project sources through it.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable, Iterator

from lean_strip._pipeline.preprocessing.ilean import module_to_ilean_relpath

__all__ = ["DockerWorkspace", "list_project_lean_files", "snapshot_sources"]


class DockerWorkspace:
    """lean-strip's workspace interface over a running `DockerEnvironment`."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def iter_ilean_documents(
        self, modules: Iterable[str] | None = None
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield project `.ilean` files, sorted, from one container tar stream."""

        requested = sorted(set(modules or []))
        if requested and len(requested) < 512:
            if any(not module or "/" in module for module in requested):
                raise ValueError(f"invalid Lean module name in {requested!r}")
            predicates = " -o ".join(
                f"-path {shlex.quote(f'.lake/build/lib/lean/{module_to_ilean_relpath(module)}')}"
                for module in requested
            )
            find_command = f"find .lake/build/lib -type f \\( {predicates} \\) -print0"
        else:
            find_command = "find .lake/build/lib -type f -name '*.ilean' -print0"
        producer = subprocess.Popen(
            ["docker", "exec", self._env.container_id, "bash", "-lc",
             f"cd /testbed && {find_command} | LC_ALL=C sort -z | tar --null -T - -cf -"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert producer.stdout is not None
        with tarfile.open(fileobj=producer.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".ilean"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    document = json.loads(extracted.read())
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{member.name}: invalid .ilean JSON") from error
                if not isinstance(document, dict):
                    raise ValueError(f"{member.name}: .ilean root is not an object")
                yield member.name, document
        stderr = producer.stderr.read().decode("utf-8", "replace") if producer.stderr else ""
        if producer.wait() != 0:
            raise RuntimeError(f"failed to collect .ilean archive: {stderr[-2000:]}")


def list_project_lean_files(env: Any) -> list[str]:
    """Absolute `/testbed` paths of the project's Lean sources (not `lakefile.lean`)."""

    result = env.execute(
        "find /testbed -name '*.lean' -not -path '*/.lake/*' -not -name 'lakefile.lean' | LC_ALL=C sort"
    )
    if result.get("returncode", 1) != 0:
        raise RuntimeError("could not list project Lean sources: " + result.get("output", "")[-4000:])
    paths = [line.strip() for line in result.get("output", "").splitlines() if line.strip()]
    unexpected = [path for path in paths if not re.fullmatch(r"/testbed/.+\.lean", path)]
    if unexpected:
        raise RuntimeError(f"project Lean source listing contained non-path output: {unexpected[:10]!r}")
    return paths


def snapshot_sources(env: Any, paths: list[str]) -> dict[str, str]:
    """Read the given `/testbed` files out of the container in one tar stream."""

    if not paths:
        return {}
    relative = [path.removeprefix("/testbed/") for path in paths]
    sources: dict[str, str] = {}
    with tempfile.TemporaryFile() as names, tempfile.TemporaryFile() as errors:
        # File lists of large repositories exceed ARG_MAX; feed them on stdin.
        names.write(("\n".join(relative) + "\n").encode())
        names.seek(0)
        producer = subprocess.Popen(
            ["docker", "exec", "-i", env.container_id, "tar", "-C", "/testbed", "-cf", "-", "-T", "-"],
            stdin=names, stdout=subprocess.PIPE, stderr=errors,
        )
        assert producer.stdout is not None
        with tarfile.open(fileobj=producer.stdout, mode="r|") as archive:
            for member in archive:
                extracted = archive.extractfile(member) if member.isfile() else None
                if extracted is not None:
                    sources[f"/testbed/{member.name.lstrip('./')}"] = extracted.read().decode("utf-8", "replace")
        returncode = producer.wait()
        errors.seek(0)
        if returncode != 0:
            raise RuntimeError("could not snapshot project Lean sources: "
                               + errors.read().decode("utf-8", "replace")[-4000:])
    return sources

