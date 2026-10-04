#!/usr/bin/env python3
"""Measure one Lean source file through Lake's exact frontend setup.

The source is temporarily extended *at the end* with a ``#eval`` marker that
prints ``IO.getNumHeartbeats``.  Appending preserves all existing source
positions and generated declaration names.  Passing the module's Lake
``.setup.json`` preserves imported plugins and artifacts, which a standalone
``Lean.Elab.runFrontend`` call does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path


METHOD_VERSION = "exact_lake_setup_sync_v2"
MARKER = "__LEANLEAN_TOTAL_HEARTBEATS__="
MARKER_COMMAND = (
    "\n#eval (show IO Unit from do\n"
    "  let heartbeats ← IO.getNumHeartbeats\n"
    f'  IO.println ("{MARKER}" ++ toString heartbeats))\n'
)
MARKER_RE = re.compile(re.escape(MARKER) + r"(\d+)")


def module_stem(source: Path, project: Path) -> Path:
    """Return the Lake module path for common source-root layouts."""

    relative = source.resolve().relative_to(project.resolve())
    parts = relative.with_suffix("").parts
    for source_root in ("src", "test", "tests"):
        if parts and parts[0] == source_root:
            return Path(*parts[1:])
    return Path(*parts)


def setup_path(source: Path, project: Path) -> Path:
    direct = project / ".lake" / "build" / "ir" / module_stem(source, project)
    direct = direct.with_suffix(".setup.json")
    if direct.is_file():
        return direct

    source_parts = module_stem(source, project).parts
    candidates = list((project / ".lake" / "build" / "ir").rglob("*.setup.json"))
    suffix_matches: list[tuple[int, Path]] = []
    for candidate in candidates:
        rel = candidate.relative_to(project / ".lake" / "build" / "ir")
        candidate_parts = Path(str(rel)[: -len(".setup.json")]).parts
        shared = 0
        for left, right in zip(reversed(source_parts), reversed(candidate_parts)):
            if left != right:
                break
            shared += 1
        if shared:
            suffix_matches.append((shared, candidate))
    if not suffix_matches:
        raise FileNotFoundError(f"no Lake setup artifact found for {source}")
    suffix_matches.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    best_score, best = suffix_matches[0]
    if len(suffix_matches) > 1 and suffix_matches[1][0] == best_score:
        raise RuntimeError(f"ambiguous Lake setup artifact for {source}")
    return best


def import_header(source_text: str) -> str:
    """Extract the leading Lean header, ignoring import-like prose in comments.

    Walk only header tokens (module/prelude/[public] [meta] import [all] Name).
    Preserve the old control bytes for ordinary one-line import headers, but
    never cut through a trailing block comment or include a body declaration.
    """
    length = len(source_text)

    def skip_trivia(position: int) -> int:
        while position < length:
            if source_text[position].isspace():
                position += 1
            elif source_text.startswith('--', position):
                newline = source_text.find('\n', position)
                position = length if newline < 0 else newline + 1
            elif source_text.startswith('/-', position):
                depth = 1
                position += 2
                while depth and position < length:
                    if source_text.startswith('/-', position):
                        depth += 1
                        position += 2
                    elif source_text.startswith('-/', position):
                        depth -= 1
                        position += 2
                    else:
                        position += 1
                if depth:
                    raise ValueError('unterminated comment in Lean header')
            else:
                break
        return position

    identifier = re.compile(r"(?:[^\W\d][\w']*|«[^»]*»)(?:\.(?:[^\W\d][\w']*|«[^»]*»))*")

    def token(position: int) -> tuple[str, int, int]:
        position = skip_trivia(position)
        match = identifier.match(source_text, position)
        if match:
            return match.group(), position, match.end()
        return '', position, position

    position = 0
    end = 0
    word, _, stop = token(position)
    if word == 'module':
        position = end = stop
    word, _, stop = token(position)
    if word == 'prelude':
        position = end = stop
    while True:
        word, _, stop = token(position)
        if word == 'public':
            word, _, stop = token(stop)
        if word == 'meta':
            word, _, stop = token(stop)
        if word != 'import':
            break
        word, _, stop = token(stop)
        if word == 'all':
            word, _, stop = token(stop)
        if not word:
            raise ValueError('missing module name in Lean import')
        position = end = stop
    if end == 0:
        return ''
    # Keep ordinary trailing whitespace and line comments byte-for-byte. If a
    # block comment crosses the newline, omit it rather than truncate it.
    newline = source_text.find('\n', end)
    line_end = length if newline < 0 else newline + 1
    trailing = source_text[end:line_end]
    if not trailing.strip() or trailing.lstrip().startswith('--'):
        return source_text[:line_end]
    if trailing.lstrip().startswith('/-'):
        # A complete same-line comment is safe only if no command follows it.
        prefix = len(source_text[:end].splitlines())
        after = skip_trivia(end)
        if after <= line_end and len(source_text[:after].splitlines()) == prefix:
            return source_text[:line_end]
    return source_text[:end] + '\n'


def run_lean(
    project: Path,
    source: Path,
    setup: Path,
    timeout_seconds: int,
) -> tuple[int, str]:
    # Lake setup options override -D options. Enforce the measurement policy in
    # a disposable setup too, without changing the build's original artifact.
    metadata = json.loads(setup.read_text())
    metadata.setdefault("options", {}).update({"Elab.async": False, "maxHeartbeats": 0})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".setup.json", delete=False) as handle:
        json.dump(metadata, handle)
        measurement_setup = Path(handle.name)
    command = [
        "lake",
        "env",
        "lean",
        "-j1",
        "-DElab.async=false",
        "-DmaxHeartbeats=0",
        str(source),
        "--setup",
        str(measurement_setup),
        "--json",
    ]
    try:
        process = subprocess.Popen(
            command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
    finally:
        measurement_setup.unlink(missing_ok=True)
    output = stdout + stderr
    if process.returncode != 0:
        raise RuntimeError(
            f"Lean exited with {process.returncode} for {source}\n{output[-8000:]}"
        )
    matches = MARKER_RE.findall(output)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one heartbeat marker for {source}, found {len(matches)}\n"
            f"{output[-8000:]}"
        )
    return int(matches[0]), output


def measure_file(
    project: Path,
    source: Path,
    timeout_seconds: int,
    measure_imports: bool,
) -> dict[str, object]:
    project = project.resolve()
    source = source.resolve()
    setup = setup_path(source, project)
    original = source.read_bytes()
    if re.search(rb"\bset_option\s+Elab\.async\s+true\b", original):
        raise ValueError(f"source overrides synchronous heartbeat policy: {source}")
    started = time.monotonic()
    # Never rewrite the image's original source. Under Docker overlay storage,
    # restoring a lower-layer file still leaves a full upper-layer copy behind;
    # a repository-wide sweep can therefore exhaust disk. A disposable sibling
    # preserves the source directory and Lake setup while being fully removed.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".leanlean-heartbeat-",
        suffix=".lean",
        dir=source.parent,
        delete=False,
    ) as handle:
        handle.write(original)
        handle.write(MARKER_COMMAND.encode())
        instrumented_path = Path(handle.name)
    try:
        total, _ = run_lean(project, instrumented_path, setup, timeout_seconds)
    finally:
        instrumented_path.unlink(missing_ok=True)

    imports = None
    if measure_imports:
        header = import_header(original.decode("utf-8"))
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="leanlean-header-",
            suffix=".lean",
            dir="/tmp",
            delete=False,
        ) as handle:
            handle.write(header)
            handle.write(MARKER_COMMAND)
            header_path = Path(handle.name)
        try:
            imports, _ = run_lean(project, header_path, setup, timeout_seconds)
        finally:
            header_path.unlink(missing_ok=True)

    if imports is not None and total < imports:
        raise ValueError(f"negative body heartbeat count for {source}: {total} - {imports}")
    return {
        "method": METHOD_VERSION,
        "unit": "raw_heartbeats",
        "user_heartbeat_divisor": 1000,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "setup_sha256": hashlib.sha256(setup.read_bytes()).hexdigest(),
        "lean_threads": 1,
        "elab_async": False,
        "source": str(source),
        "setup": str(setup),
        "total_heartbeats": total,
        "import_heartbeats": imports,
        "body_heartbeats": total - imports if imports is not None else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def discover_built_files(project: Path, exclude_files: list[str]) -> dict[str, object]:
    """Enumerate live project modules after a clean build, excluding fixtures.

    Unbuilt source is reported explicitly: compilation work is defined by the
    declared build target, not by standalone elaboration of unused files.
    """
    files, unbuilt = [], []
    for root, directories, names in os.walk(project):
        directories[:] = sorted(d for d in directories if d not in {".lake", ".git"})
        for name in sorted(names):
            path = Path(root) / name
            relative = path.relative_to(project).as_posix()
            if not name.endswith(".lean") or name == "lakefile.lean" or relative in exclude_files:
                continue
            setup = project / ".lake/build/ir" / module_stem(path, project).with_suffix(".setup.json")
            (files if setup.is_file() else unbuilt).append(relative)
    return {"files": sorted(files), "unbuilt_sources": sorted(unbuilt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--list-built", action="store_true")
    parser.add_argument("--exclude-file", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--measure-imports", action="store_true")
    args = parser.parse_args()
    if args.list_built:
        print(json.dumps(discover_built_files(args.project, args.exclude_file)))
        return 0
    if args.source is None:
        parser.error("--source is required unless --list-built is used")
    try:
        result = measure_file(
            args.project,
            args.source,
            args.timeout_seconds,
            args.measure_imports,
        )
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
