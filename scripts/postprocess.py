#!/usr/bin/env python3
"""Build and score final submissions, optionally replaying every checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from leanlean.pipeline.evaluation import (  # noqa: E402
    load_named_postprocessing_config,
    resolve_named_config,
    validate_named_run_identity,
)
from leanlean.pipeline.postprocessing import (  # noqa: E402
    PreparedPostprocessing,
    finalize_run,
    resolve_run,
    update_run_status,
    write_run_definition,
)


console = Console()

ENDPOINTS_STAGE = {
    "replay": False,
    "endpoints_only": True,
    "measure_heartbeats": False,
    "warm_baseline_build": True,
    "save_lake_build": True,
}
HEARTBEATS_STAGE = {
    "replay": False,
    "endpoints_only": False,
    "measure_heartbeats": True,
    "warm_baseline_build": False,
    "save_lake_build": False,
}
REPLAY_STAGE = {
    "replay": True,
    "endpoints_only": False,
    "measure_heartbeats": False,
    "warm_baseline_build": False,
    "save_lake_build": False,
}


def _run_status(path: Path) -> str | None:
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return str(status) if isinstance(status, str) else None


def _launch_tmux(arguments: list[str], session: str) -> None:
    """Run this same postprocess command inside a named tmux session."""
    attach = f"tmux attach -t {shlex.quote(session)}"
    command = shlex.join(["bash", str(REPO_ROOT / "postprocess.sh"), *arguments])
    existing = subprocess.run(
        ["tmux", "has-session", "-t", session], capture_output=True
    )
    if existing.returncode == 0:
        panes = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True,
            text=True,
        )
        pane_dead = panes.stdout.splitlines() if panes.returncode == 0 else []
        if not pane_dead or not all(value == "1" for value in pane_dead):
            console.print(f"[yellow]already running[/yellow]: {session}")
            console.print(attach)
            return
        subprocess.run(
            ["tmux", "respawn-pane", "-k", "-t", session, "-c", str(REPO_ROOT), command],
            check=True,
        )
        console.print(f"[green]resumed[/green] dead tmux session {session}")
        console.print(attach)
        return
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT), command],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", session, "remain-on-exit", "on"],
        check=False,
    )
    console.print(f"[green]launched[/green] in tmux session {session}")
    console.print(attach)


def _stage_done(prepared: PreparedPostprocessing) -> bool:
    """The stage's current definition is this one and it completed."""
    try:
        payload = yaml.safe_load(prepared.run_artifact_path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("status") == "complete"
        and payload.get("postprocess_id") == prepared.manifest["postprocess_id"]
    )


def _execute(prepared: PreparedPostprocessing) -> int:
    if _run_status(prepared.run_artifact_path) == "complete":
        console.print(f"[green]already complete[/green]: {prepared.run_artifact_path}")
        return 0
    update_run_status(prepared.run_artifact_path, "running")

    try:
        captures = list(prepared.manifest["captures"])
        total = (
            sum(max(1, len(capture.get("archives") or [])) for capture in captures)
            if prepared.replay
            else len(captures)
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} builds"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("postprocessing builds", total=max(1, total))
            completed: dict[str, int] = {}
            lock = threading.Lock()

            def advance(instance_id: str, current: int, _count: int) -> None:
                with lock:
                    previous = completed.get(instance_id, 0)
                    if current > previous:
                        increment = current - previous
                        projected = progress.tasks[task].completed + increment
                        if projected > progress.tasks[task].total:
                            progress.update(task, total=projected)
                        progress.advance(task, increment)
                        completed[instance_id] = current
                    progress.update(
                        task, description=f"postprocessing builds · {instance_id}"
                    )

            summary = finalize_run(prepared, progress=advance)
    except Exception as error:
        update_run_status(
            prepared.run_artifact_path, "failed", error=str(error), exit_code=1
        )
        raise
    update_run_status(prepared.run_artifact_path, "complete", exit_code=0)
    headline = summary["headline"]
    builds = summary["checkpoint_builds"]
    console.print(
        f"[green]complete[/green]: {prepared.run_artifact_path}\n"
        f"headline word compression: {headline['value']:.3f}% across "
        f"{headline['denominator']} repositories\n"
        f"final builds: {builds['final_passed']} passed / "
        f"{builds['final_submissions'] - builds['final_passed']} failed\n"
        f"edit checkpoint builds: {builds['passed']} passed / "
        f"{builds['failed']} failed"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        metavar="DATASET",
        help="extensionless dataset name under configs/dataset/",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PROVIDER/MODEL",
        help="extensionless <provider>/<model> name under configs/models/",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="also build every captured edit checkpoint",
    )
    parser.add_argument("--endpoints-only", action="store_true",
                        help="verify submitted endpoints only, including timed-out runs")
    parser.add_argument("--accept-budget-exhausted", action="store_true",
                        help="with --endpoints-only, verify and score saved endpoints of recorded full-budget timeouts")
    parser.add_argument(
        "--heartbeats",
        action="store_true",
        help="measure submitted-final heartbeats in the same clean-build container",
    )
    parser.add_argument(
        "--allow-relocated-dataset",
        action="store_true",
        help="rebase only verified materialization paths to a relocated pinned bundle",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="repository build workers (defaults to the evaluation worker count)",
    )
    parser.add_argument(
        "--threads-per-repo",
        type=int,
        help="Lean build threads/CPUs per repository (defaults to evaluation)",
    )
    parser.add_argument(
        "--cgroup-parent",
        help="override the recorded Docker cgroup parent for this postprocessing run",
    )
    parser.add_argument(
        "--container-memory",
        help="override the recorded memory limit for each build container (for example 50g)",
    )
    parser.add_argument(
        "--warm-stripped-baseline-build",
        action="store_true",
        help="preserve the stripped baseline .lake/build and build submitted endpoints incrementally",
    )
    parser.add_argument(
        "--save-lake-build",
        action="store_true",
        help="export each successfully built submitted endpoint's .lake/build archive",
    )
    parser.add_argument(
        "--clean-replay",
        action="store_true",
        help="with --replay, rebuild the project from clean at every checkpoint instead of "
        "keeping .lake/build between a repository's checkpoints (the default)",
    )
    parser.add_argument(
        "--skip-checkpoint-lean-verify",
        action="store_true",
        help="with --replay, run lean_verify only on the submitted endpoint",
    )
    parser.add_argument(
        "--build-only-replay",
        action="store_true",
        help="with --replay, build checkpoints and final endpoint without lean_verify",
    )
    parser.add_argument(
        "--checkpoint-index-file",
        type=Path,
        help="with --replay, build only the edit checkpoints listed per repository "
        "(see scripts/select_checkpoint_windows.py)",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write the immutable postprocessing manifest without launching tmux",
    )
    args = parser.parse_args()

    if args.validate_only and args.prepare_only:
        parser.error("--validate-only and --prepare-only are mutually exclusive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.threads_per_repo is not None and args.threads_per_repo < 1:
        parser.error("--threads-per-repo must be at least 1")

    evaluation = resolve_named_config(args.dataset, args.model, repo_root=REPO_ROOT)
    if not evaluation.run_artifact_path.is_file():
        parser.error("evaluation does not exist; run eval.sh with this dataset/model first")
    validate_named_run_identity(evaluation, repo_root=REPO_ROOT)
    postprocessing = load_named_postprocessing_config(args.dataset, repo_root=REPO_ROOT)
    reference = evaluation.run_artifact_path
    workers = args.workers if args.workers is not None else postprocessing["workers"]
    threads_per_repository = (
        args.threads_per_repo
        if args.threads_per_repo is not None
        else postprocessing["threads_per_repository"]
    )
    # Default to endpoints, heartbeats and any configured replay;
    # explicit flags select individual stages.
    stages = [
        {
            "replay": args.replay,
            "endpoints_only": args.endpoints_only,
            "measure_heartbeats": args.heartbeats,
            "warm_baseline_build": args.warm_stripped_baseline_build,
            "save_lake_build": args.save_lake_build,
        }
    ]
    if not any(stages[0].values()):
        stages = [ENDPOINTS_STAGE, HEARTBEATS_STAGE]
        if postprocessing["replay"]:
            stages.append(REPLAY_STAGE)

    checkpoint_selection = None
    if args.checkpoint_index_file is not None:
        index_path = args.checkpoint_index_file.expanduser()
        if not index_path.is_absolute():
            index_path = REPO_ROOT / index_path
        index_bytes = index_path.resolve().read_bytes()
        index_document = json.loads(index_bytes)
        checkpoint_selection = {
            "source": str(index_path.resolve().relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
            "policy": index_document.get("policy"),
            "window_minutes": index_document.get("window_minutes"),
            "edit_indices": index_document["repositories"],
        }

    any_replay = any(stage["replay"] for stage in stages)
    if args.clean_replay and not any_replay:
        parser.error("--clean-replay requires --replay")
    if (args.skip_checkpoint_lean_verify or args.build_only_replay) and (
        not any_replay or args.clean_replay
    ):
        parser.error("--skip-checkpoint-lean-verify and --build-only-replay need the default incremental --replay")

    def resolve(stage: dict) -> PreparedPostprocessing:
        replay = stage["replay"]
        return resolve_run(
            reference,
            repo_root=REPO_ROOT,
            replay=replay,
            endpoints_only=stage["endpoints_only"],
            accept_budget_exhausted=args.accept_budget_exhausted,
            workers=workers,
            threads_per_repository=threads_per_repository,
            measure_heartbeats=stage["measure_heartbeats"],
            allow_relocated_dataset=args.allow_relocated_dataset,
            cgroup_parent=args.cgroup_parent,
            container_memory=args.container_memory,
            warm_baseline_build=stage["warm_baseline_build"],
            save_lake_build=stage["save_lake_build"],
            # Incremental by default: each checkpoint recompiles only what its edit touched.
            incremental_replay=replay and not args.clean_replay,
            skip_checkpoint_lean_verify=replay and args.skip_checkpoint_lean_verify,
            build_only_replay=replay and args.build_only_replay,
            checkpoint_selection=checkpoint_selection if replay else None,
        )

    prepared_stages = [resolve(stage) for stage in stages]
    first = prepared_stages[0]
    console.print(
        f"pair: {first.output_dir}\n"
        f"identity: {first.manifest['model']} / {first.manifest['reasoning_effort']} / "
        f"{first.manifest['generator']}\n"
        f"repositories: {len(first.manifest['repositories'])}\n"
        f"workers: {first.manifest['parallelism']['workers']} (one repo each)\n"
        f"threads per repo: {first.manifest['parallelism']['threads_per_repository']}"
    )
    for prepared in prepared_stages:
        state = "done" if _stage_done(prepared) else "to run"
        console.print(f"stage {prepared.manifest_path.parent.name}: {state}")
    if args.validate_only:
        console.print("[green]validated[/green]")
        return 0
    if args.prepare_only:
        for prepared in prepared_stages:
            write_run_definition(prepared, repo_root=REPO_ROOT)
            console.print(f"[green]prepared[/green]: {prepared.manifest_path}")
        return 0
    if all(_stage_done(prepared) for prepared in prepared_stages):
        console.print(f"[green]already done[/green]: {first.output_dir}")
        return 0

    if not os.environ.get("TMUX"):
        session = re.sub(r"[.:]", "-", f"postprocess-{first.source_run['run_id']}")
        _launch_tmux(sys.argv[1:], session)
        return 0
    for index, stage in enumerate(stages):
        # Resolve again right before running: a stage may depend on the
        # outputs of the one before it.
        prepared = resolve(stage) if index else first
        if _stage_done(prepared):
            console.print(f"[green]already complete[/green]: {prepared.run_artifact_path}")
            continue
        write_run_definition(prepared, repo_root=REPO_ROOT)
        _execute(prepared)
    console.print(f"[green]done[/green]: {first.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
