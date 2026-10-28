#!/usr/bin/env python3
"""Pack a finished dataset/model pair into a results archive, or verify one.

    pack_results.py DATASET --model PROVIDER/MODEL --into ARCHIVE
    pack_results.py --pair PAIR --into ARCHIVE --endpoints-only
    pack_results.py --verify ARCHIVE

Packing copies results/<dataset>/<provider>/<model>/ to
ARCHIVE/results/<dataset>/<provider>/<model>/ in compressed form (see
leanlean.packing), then verifies the copy. The pair must be done: the
evaluation complete, both postprocessing stages complete and a result.json
in every repository. --endpoints-only permits packing a completed endpoint
evaluation without requiring heartbeat measurements. An existing packed pair
is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from leanlean.packing import pack_repository, pack_run_level, verify  # noqa: E402
from leanlean.pipeline.evaluation import resolve_named_config  # noqa: E402

console = Console()


def _status(path: Path) -> str | None:
    try:
        return (yaml.safe_load(path.read_text()) or {}).get("status")
    except (OSError, yaml.YAMLError):
        return None


def _check_done(pair: Path, repositories: list[str], *, endpoints_only: bool = False) -> list[str]:
    problems = []
    if _status(pair / "run/run.yaml") != "complete":
        problems.append("evaluation is not complete")
    for stage in (("endpoints",) if endpoints_only else ("endpoints", "heartbeats")):
        if _status(pair / "postprocess" / stage / "run.yaml") != "complete":
            problems.append(f"postprocess stage {stage} is not complete")
    missing = [repo for repo in repositories if not (pair / repo / "result.json").is_file()]
    if missing:
        problems.append(f"{len(missing)} repositories have no result.json")
    return problems


def _repositories_from_pair(pair: Path, repo_root: Path) -> list[str]:
    """Read the pinned finished run without recomposing generation configs."""
    pair.relative_to(repo_root / "results")
    run = yaml.safe_load((pair / "run/run.yaml").read_text())
    manifest_path = pair / "run/manifest.yaml"
    pin = run["experiment"]
    if (repo_root / pin["manifest"]).resolve() != manifest_path:
        raise ValueError("run pins a different evaluation manifest")
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != pin["sha256"]:
        raise ValueError("evaluation manifest digest mismatch")
    manifest = yaml.safe_load(content)
    if run.get("kind") != "leanlean_run" or manifest.get("kind") != "leanlean_evaluation_run":
        raise ValueError("unsupported evaluation run artifact")
    repositories = [str(row["id"]) for row in manifest["repositories"]]
    if not repositories or len(set(repositories)) != len(repositories):
        raise ValueError("evaluation manifest has no repositories or duplicate repositories")
    if any(not repo or Path(repo).name != repo or repo in {".", ".."} for repo in repositories):
        raise ValueError("unsafe repository identifier")
    return repositories


def _report(root: Path, jobs: int = 16) -> int:
    directories, errors = verify(root, jobs)
    for error in errors[:50]:
        console.print(f"[red]{error}[/red]")
    console.print(f"verified {directories} directories, {len(errors)} errors")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", metavar="DATASET")
    parser.add_argument("--model", metavar="PROVIDER/MODEL")
    parser.add_argument("--pair", type=Path, help="existing results pair; use its pinned run instead of generation configs")
    parser.add_argument("--into", type=Path, metavar="ARCHIVE", help="results archive to add the pair to")
    parser.add_argument("--verify", type=Path, metavar="ARCHIVE", help="verify a results archive (or one packed pair)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--endpoints-only", action="store_true", help="pack completed endpoint results without requiring heartbeat measurements")
    args = parser.parse_args()

    if args.verify is not None:
        return _report(args.verify.expanduser().resolve(), args.jobs)
    if args.pair is not None:
        if args.dataset or args.model or not args.into:
            parser.error("--pair requires --into and cannot be combined with DATASET or --model")
        pair = args.pair.expanduser().resolve()
        repositories = _repositories_from_pair(pair, REPO_ROOT)
    else:
        if not (args.dataset and args.model and args.into):
            parser.error("DATASET, --model and --into are required (or --pair PAIR --into ARCHIVE)")
        resolved = resolve_named_config(args.dataset, args.model, repo_root=REPO_ROOT)
        pair = resolved.output_dir
        repositories = [str(row["id"]) for row in resolved.manifest["repositories"]]
    problems = _check_done(pair, repositories, endpoints_only=args.endpoints_only)
    if problems:
        for problem in problems:
            console.print(f"[red]not done:[/red] {problem}")
        return 1
    relative = pair.relative_to(REPO_ROOT)
    destination = args.into.expanduser().resolve() / relative
    if destination.exists():
        console.print(f"[yellow]already packed[/yellow]: {destination}")
        return 0
    partial = destination.with_name(destination.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    console.print(f"packing {len(repositories)} repositories: {pair} -> {destination}")
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = [pool.submit(pack_repository, pair / repo, partial / repo) for repo in repositories]
        jobs.append(pool.submit(pack_run_level, pair, partial, set(repositories)))
        files = sum(job.result() for job in jobs)
    status = _report(partial, args.jobs)
    if status:
        console.print(f"[red]verification failed; the partial copy is left at {partial}[/red]")
        return status
    partial.rename(destination)
    console.print(f"[green]packed[/green] {files} files into {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
