#!/usr/bin/env python3
"""Freeze the current Palomar registry, records, archives, and target mapping.

The command is intentionally separate from source normalization. It downloads
the public registry projection, every exact result record, and the immutable
registry source archive containing each registered solution. It then produces
the candidate mapping consumed by preprocessing (configs/preprocessing/).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.generate_palomar_candidate_mapping import (  # noqa: E402
    _build_mapping,
    _write_markdown,
)


DEFAULT_REGISTRY_URL = "https://data.palomar-registry.org/recent.json"
console = Console()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_snapshot_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            "snapshot id must contain only letters, digits, dot, dash, underscore"
        )
    return value


def _download(url: str, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "leanlean-palomar-sync/1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def _record_destination(record_dir: Path, projection: dict[str, Any]) -> Path:
    return record_dir / Path(str(projection["path"])).name


def _fetch_record(
    projection: dict[str, Any],
    *,
    registry_url: str,
    record_dir: Path,
    timeout: int,
) -> Path:
    destination = _record_destination(record_dir, projection)
    base = registry_url.rsplit("/", 1)[0] + "/"
    _download(
        urllib.parse.urljoin(base, str(projection["path"])),
        destination,
        timeout=timeout,
    )
    record = json.loads(destination.read_text(encoding="utf-8"))
    if (
        record.get("id") != projection.get("id")
        or record.get("version") != projection.get("version")
    ):
        raise ValueError(f"record/projection mismatch for {projection.get('id')}")
    return destination


def _target_source(record: dict[str, Any]) -> tuple[str, str]:
    source = record["source"]
    return str(source["repository"]), str(source["commit"])


def _archive_destination(
    archive_dir: Path, repository: str, commit: str
) -> Path:
    owner, name = repository.split("/", 1)
    return archive_dir / f"{owner}--{name}--{commit}.tar.gz"


def _archive_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with tarfile.open(path, "r:gz") as archive:
            next(iter(archive))
    except (OSError, StopIteration, tarfile.TarError):
        return False
    return True


def _fetch_archive(
    source: tuple[str, str],
    *,
    archive_dir: Path,
    timeout: int,
) -> Path:
    repository, commit = source
    destination = _archive_destination(archive_dir, repository, commit)
    if _archive_is_valid(destination):
        return destination
    destination.unlink(missing_ok=True)
    owner, name = repository.split("/", 1)
    _download(
        f"https://codeload.github.com/{owner}/{name}/tar.gz/{commit}",
        destination,
        timeout=timeout,
    )
    if not _archive_is_valid(destination):
        destination.unlink(missing_ok=True)
        raise ValueError(f"invalid source archive for {repository} at {commit}")
    return destination


def _run_parallel(
    description: str,
    items: list[Any],
    worker: Any,
    *,
    workers: int,
) -> list[Any]:
    completed: list[Any] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}", markup=False),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task(description, total=len(items), status="starting")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(worker, item): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                completed.append(future.result())
                progress.update(task, advance=1, status=str(item)[:80])
    return completed


def _launch_tmux(snapshot_id: str) -> None:
    session = f"palomar-sync-{snapshot_id.lower()}"
    if subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        capture_output=True,
    ).returncode == 0:
        console.print(f"[yellow]already running[/yellow]: {session}")
        console.print(f"tmux attach -t {shlex.quote(session)}")
        return
    arguments = list(sys.argv[1:])
    if "--snapshot-id" not in arguments:
        arguments.extend(["--snapshot-id", snapshot_id])
    arguments.append("--inside-tmux")
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *arguments])
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(REPO_ROOT),
            command,
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-window-option", "-t", session, "remain-on-exit", "on"],
        check=True,
    )
    console.print(f"[green]launched[/green] Palomar sync: {session}")
    console.print(f"tmux attach -t {shlex.quote(session)}")


def sync(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_id = _safe_snapshot_id(args.snapshot_id)
    run_dir = REPO_ROOT / "runs/palomar/registry_sync" / snapshot_id
    recent_path = run_dir / "recent.json"
    record_dir = run_dir / "records"
    archive_dir = (
        Path(args.archive_cache).resolve()
        if args.archive_cache
        else run_dir / "archives"
    )
    mapping_path = (
        REPO_ROOT / "data/palomar" / f"candidate_mapping_{snapshot_id}.json"
    )
    markdown_path = (
        REPO_ROOT / "docs" / f"palomar_candidate_mapping_{snapshot_id}.md"
    )
    manifest_path = (
        REPO_ROOT
        / "experiments/preprocessing"
        / f"palomar_registry_sync_{snapshot_id}.yaml"
    )
    report_path = run_dir / "report.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    _download(args.registry_url, recent_path, timeout=args.record_timeout)
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    projections = recent.get("entries")
    if not isinstance(projections, list) or not projections:
        raise ValueError("registry response has no entries")
    if len({(row.get("id"), row.get("version")) for row in projections}) != len(
        projections
    ):
        raise ValueError("registry response contains duplicate id/version pairs")

    manifest = {
        "kind": "palomar_registry_sync",
        "schema_version": 1,
        "run_id": f"palomar_registry_sync_{snapshot_id}",
        "registry": {
            "url": args.registry_url,
            "snapshot": str(recent_path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(recent_path),
            "entries": len(projections),
        },
        "workers": {"downloads": args.workers},
        "timeouts": {
            "record_seconds": args.record_timeout,
            "archive_seconds": args.archive_timeout,
        },
        "outputs": {
            "directory": str(run_dir.relative_to(REPO_ROOT)),
            "records": str(record_dir.relative_to(REPO_ROOT)),
            "archive_cache": str(archive_dir),
            "mapping": str(mapping_path.relative_to(REPO_ROOT)),
            "markdown": str(markdown_path.relative_to(REPO_ROOT)),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, width=100)
    )

    record_paths = _run_parallel(
        "Downloading Palomar records",
        projections,
        lambda projection: _fetch_record(
            projection,
            registry_url=args.registry_url,
            record_dir=record_dir,
            timeout=args.record_timeout,
        ),
        workers=args.workers,
    )
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in record_paths
    ]
    sources = sorted({_target_source(record) for record in records})
    archives = _run_parallel(
        "Downloading pinned source archives",
        sources,
        lambda source: _fetch_archive(
            source,
            archive_dir=archive_dir,
            timeout=args.archive_timeout,
        ),
        workers=args.workers,
    )

    mapping = _build_mapping(
        recent_path,
        record_dir,
        archive_dir,
        snapshot_id=snapshot_id,
    )
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(mapping, markdown_path, mapping_path)
    report = {
        "kind": "palomar_registry_sync_report",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "passed": True,
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "registry_sha256": file_sha256(recent_path),
        "records": len(record_paths),
        "target_archives": len(archives),
        "mapping": str(mapping_path.relative_to(REPO_ROOT)),
        "mapping_sha256": file_sha256(mapping_path),
        "summary": mapping["summary"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    console.print_json(data=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-url", default=DEFAULT_REGISTRY_URL)
    parser.add_argument(
        "--snapshot-id",
        default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--archive-cache", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--record-timeout", type=int, default=90)
    parser.add_argument("--archive-timeout", type=int, default=900)
    parser.add_argument(
        "--inside-tmux", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--no-tmux", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not os.environ.get("TMUX") and not args.inside_tmux and not args.no_tmux:
        _launch_tmux(_safe_snapshot_id(args.snapshot_id))
        return
    sync(args)


if __name__ == "__main__":
    main()
