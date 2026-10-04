#!/usr/bin/env python3
"""Run a size-stratified theorem reconstruction sweep from one YAML manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from leanlean.pipeline.reconstruction_sweep import (  # noqa: E402
    compare_sweep,
    load_sweep_manifest,
    resolved_runs,
    validate_size_bands,
)


console = Console()


def _session_name(run_id: str) -> str:
    base = "theorem-ablation-sweep-" + re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
    return base[:80]


def _run(label: str, command: list[str]) -> None:
    console.rule(label)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _run_parallel(
    stage: str,
    jobs: list[tuple[str, list[str]]],
    *,
    max_workers: int,
    log_dir: Path,
) -> None:
    if not jobs or max_workers < 1:
        raise ValueError(f"{stage}: invalid parallel job count")
    log_dir.mkdir(parents=True, exist_ok=True)

    def execute(label: str, command: list[str]) -> tuple[int, Path]:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "-", label)
        log_path = log_dir / f"{safe}.log"
        with log_path.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        return result.returncode, log_path

    console.rule(stage)
    failures: list[tuple[str, int, Path]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for label, command in jobs:
                task = progress.add_task(label, total=None)
                future = executor.submit(execute, label, command)
                futures[future] = (label, task)
            for future in as_completed(futures):
                label, task = futures[future]
                returncode, log_path = future.result()
                description = (
                    f"[green]{label} complete[/green]"
                    if returncode == 0
                    else f"[red]{label} failed[/red]"
                )
                progress.update(
                    task, total=1, completed=1, description=description
                )
                if returncode != 0:
                    failures.append((label, returncode, log_path))
    if failures:
        for label, returncode, log_path in failures:
            console.rule(f"{label} failed ({returncode})")
            console.print(log_path.read_text()[-6000:])
        raise SystemExit(failures[0][1])



def _launch(path: Path, session: str) -> None:
    if subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"], capture_output=True
    ).returncode == 0:
        console.print(f"[yellow]already exists[/yellow]: {session}")
        console.print(f"tmux attach -t {shlex.quote(session)}")
        return
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), str(path), "--inside-tmux"]
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT), command],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True
    )
    console.print(f"[green]launched[/green] {session}")
    console.print(f"tmux attach -t {shlex.quote(session)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--inside-tmux", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    path = args.manifest.resolve() if args.manifest.is_absolute() else (REPO_ROOT / args.manifest).resolve()
    sweep = load_sweep_manifest(path)
    session = _session_name(str(sweep["run_id"]))
    if not os.environ.get("TMUX") and not args.inside_tmux:
        _launch(path, session)
        return 0
    if not os.environ.get("TMUX"):
        raise RuntimeError("--inside-tmux is reserved for the tmux launcher")

    runs = resolved_runs(sweep)
    for band, manifest_path, manifest in runs:
        preprocessing = manifest["preprocessing"]
        config = str(preprocessing["config"])
        dataset = Path(str(preprocessing["dataset"]))
        dataset_path = dataset if dataset.is_absolute() else REPO_ROOT / dataset
        reuse = bool(
            sweep["execution"].get("reuse_published_preprocessing", False)
        )
        if reuse and dataset_path.is_file():
            console.rule(f"reuse published {band['name']} ablation")
            console.print(dataset_path)
        else:
            _run(
                f"preprocess {band['name']} ablation",
                [str(REPO_ROOT / "preprocess.sh"), config],
            )
        _run(
            f"measure {band['name']} ablation",
            ["uv", "run", "python", "scripts/reconstruct.py", str(manifest_path), "validate"],
        )

    console.rule("validate small/medium/large size stratification")
    console.print_json(data=validate_size_bands(sweep))

    max_workers = int(sweep["execution"]["max_concurrent_runs"])
    log_root = (
        REPO_ROOT
        / "runs"
        / "reconstruction"
        / str(sweep["run_id"])
        / "orchestrator"
    )
    compression_jobs = []
    compression_scoring_jobs = []
    preparation_jobs = []
    reconstruction_jobs = []
    scoring_jobs = []
    summary_jobs = []
    for band, manifest_path, manifest in runs:
        name = str(band["name"])
        compression = manifest["compression"]
        evaluation = manifest["evaluation"]
        if compression.get("mode") != "reuse_submitted_generation":
            compression_jobs.append(
                (
                    f"compress {name}",
                    [
                        "uv",
                        "run",
                        "python",
                        "eval.py",
                        str(compression["dataset_config"]),
                        "--model",
                        str(compression["model_config"]),
                    ],
                )
            )
        preparation_jobs.append(
            (
                f"materialize {name}",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/reconstruct.py",
                    str(manifest_path),
                    "prepare",
                ],
            )
        )
        reconstruction_jobs.append(
            (
                f"reprove {name} A/B arms",
                [
                    "uv",
                    "run",
                    "python",
                    "eval.py",
                    str(evaluation["dataset_config"]),
                    "--model",
                    str(evaluation["model_config"]),
                ],
            )
        )
        for action, jobs in (("score-compression", compression_scoring_jobs), ("score", scoring_jobs)):
            jobs.append((f"{action} {name}", [
                "uv", "run", "python", "scripts/reconstruct.py", str(manifest_path), action,
            ]))
        summary_jobs.append(
            (
                f"summarize {name}",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/reconstruct.py",
                    str(manifest_path),
                    "summarize",
                ],
            )
        )

    if compression_jobs:
        _run_parallel(
            "compress all ablations in parallel",
            compression_jobs,
            max_workers=max_workers,
            log_dir=log_root / "compression",
        )
    else:
        console.print("Reusing pinned submitted compression patches; no model generation.")
    _run_parallel(
        "verify all compressions in parallel",
        compression_scoring_jobs,
        max_workers=max_workers,
        log_dir=log_root / "compression-scoring",
    )
    _run_parallel(
        "materialize all reconstruction pairs in parallel",
        preparation_jobs,
        max_workers=max_workers,
        log_dir=log_root / "preparation",
    )
    _run_parallel(
        "reprove all A/B pairs in parallel",
        reconstruction_jobs,
        max_workers=max_workers,
        log_dir=log_root / "reconstruction",
    )
    _run_parallel(
        "independently verify all reconstruction pairs",
        scoring_jobs,
        max_workers=max_workers,
        log_dir=log_root / "scoring",
    )
    _run_parallel(
        "summarize all pairs in parallel",
        summary_jobs,
        max_workers=max_workers,
        log_dir=log_root / "summary",
    )

    console.rule("compare reconstruction outcomes by ablation size")
    console.print_json(data=compare_sweep(sweep))
    console.print("[green]theorem reconstruction size sweep complete[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
