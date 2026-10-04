"""Remove orphan project artifacts from disposable evaluation build contexts."""

from __future__ import annotations

import os
from pathlib import Path


ARTIFACT_SUFFIXES = (
    ".olean", ".ilean", ".ir", ".trace", ".c", ".o", ".bc", ".setup.json",
)


def source_module_candidates(source: Path) -> set[tuple[str, ...]]:
    """Support root and nested Lake srcDir layouts without importing old caches."""
    modules = set()
    for directory, dirs, files in os.walk(source):
        dirs[:] = [name for name in dirs if name not in {".lake", ".git"}]
        for name in files:
            if not name.endswith(".lean"):
                continue
            parts = (Path(directory) / name).relative_to(source).with_suffix("").parts
            # A source directory prefix is not part of a Lean module name.
            for index in range(len(parts)):
                modules.add(parts[index:])
    return modules


def orphan_cache_files(source: Path, cache: Path) -> list[Path]:
    modules = source_module_candidates(source)
    orphaned = []
    for path in cache.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(cache)
        parts = relative.parts
        if parts[:2] == ("lib", "lean"):
            module_path = parts[2:]
        elif parts[:1] == ("ir",):
            module_path = parts[1:]
        else:
            # Executables and aggregate libraries can contain old project data.
            orphaned.append(path)
            continue
        basename = module_path[-1] if module_path else ""
        matched = []
        for suffix in ARTIFACT_SUFFIXES:
            position = basename.find(suffix)
            if position >= 0:
                matched.append(basename[:position])
        if not matched or not any((*module_path[:-1], name) in modules for name in matched):
            orphaned.append(path)
    return sorted(orphaned)


def sanitize_disposable_build_context(context: Path) -> dict:
    """Only mutate fresh copies under the materializer's temporary directory."""
    if not context.name.startswith("leanlean-materialize-"):
        raise ValueError("cache sanitization requires a disposable materialization context")
    source, cache = context / "source", context / "warm-build"
    if not source.is_dir() or not cache.is_dir() or source.is_symlink() or cache.is_symlink():
        raise ValueError("invalid materialization source/cache directories")
    removed = []
    for path in orphan_cache_files(source, cache):
        if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError("cache artifact escaped the disposable context")
        removed.append(path.relative_to(cache).as_posix())
        path.unlink()
    return {"policy": "drop_orphan_project_cache_before_agent", "removed_files": removed}


def render_container_cache_cleanup(
    *, clear_project: bool = False, clear_config: bool = False
) -> str:
    """Trusted cleanup code for a fresh, disposable reconstruction container."""
    import inspect
    if clear_config:
        if not clear_project:
            raise ValueError("clearing Lake configuration requires clearing project artifacts")
        return (
            "import json, shutil\nfrom pathlib import Path\n"
            "lake = Path('/testbed') / '.lake'\n"
            "if lake.is_symlink(): raise ValueError('symlink Lake directory')\n"
            "for cache in lake.iterdir() if lake.exists() else []:\n"
            "    if cache.name == 'packages': continue\n"
            "    if cache.is_symlink() or not cache.is_dir(): cache.unlink()\n"
            "    else: shutil.rmtree(cache)\n"
            "print(json.dumps({'project_cache_policy': "
            "'clean_project_and_config_preserve_dependencies_v1'}))\n"
        )
    if clear_project:
        return (
            "import json, shutil\nfrom pathlib import Path\n"
            "cache = Path('/testbed') / '.lake/build'\n"
            "if cache.is_symlink(): raise ValueError('symlink cache')\n"
            "shutil.rmtree(cache, ignore_errors=False) if cache.exists() else None\n"
            "print(json.dumps({'project_cache_policy': 'clean_project_preserve_dependencies'}))\n"
        )
    return (
        "import os, json\nfrom pathlib import Path\n"
        + "ARTIFACT_SUFFIXES = " + repr(ARTIFACT_SUFFIXES) + "\n"
        + inspect.getsource(source_module_candidates) + "\n"
        + inspect.getsource(orphan_cache_files) + "\n"
        + "source = Path('/testbed')\ncache = source / '.lake/build'\n"
        + "if cache.is_symlink(): raise ValueError('symlink cache')\n"
        + "removed = []\n"
        + "for path in orphan_cache_files(source, cache):\n"
        + "    if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):\n"
        + "        raise ValueError('cache artifact escaped disposable container')\n"
        + "    removed.append(str(path.relative_to(cache)))\n"
        + "    path.unlink()\n"
        + "print(json.dumps({'orphan_cache_artifacts_removed': removed}))\n"
    )
