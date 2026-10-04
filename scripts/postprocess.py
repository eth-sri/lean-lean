#!/usr/bin/env python3
"""Build and score final submissions, optionally replaying every checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    load_run_manifest,
    resolve_run,
    update_run_status,
    write_run_definition,
)


console = Console()


def _run_status(path: Path) -> str | None:
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return str(status) if isinstance(status, str) else None


def _launch_tmux(manifest: Path, session: str) -> None:
    attach = f"tmux attach -t {shlex.quote(session)}"
    command = shlex.join(
        [
            "bash",
            str(REPO_ROOT / "postprocess.sh"),
            "--run-manifest",
            str(manifest.resolve()),
            "--inside-tmux",
        ]
    )
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
        "runs",
        nargs="*",
        metavar="RUN",
        help="explicit ordered evaluation run IDs or run.yaml paths",
    )
    parser.add_argument(
        "--model", help="model config name as <provider>/<model>"
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="also build every captured edit checkpoint",
    )
    parser.add_argument("--endpoints-only", action="store_true",
                        help="verify submitted endpoints only, including timed-out runs")
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
    parser.add_argument(
        "--allow-missing-subset",
        action="store_true",
        help="record and skip requested subset repositories unavailable in the source run",
    )
    parser.add_argument(
        "--repositories-from",
        type=Path,
        help="process only the ordered repository list from this YAML config",
    )
    parser.add_argument(
        "--ignore-incomplete",
        action="store_true",
        help="postprocess only valid completed repositories from a failed run",
    )
    parser.add_argument(
        "--prefire",
        action="store_true",
        help=(
            "provisionally replay salvaged failed captures to warm validation "
            "caches; strict combined postprocessing must follow"
        ),
    )
    parser.add_argument("--run-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inside-tmux", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_manifest is not None:
        if (
            args.runs
            or args.model is not None
            or args.replay
            or args.endpoints_only
            or args.workers is not None
            or args.threads_per_repo is not None
            or args.cgroup_parent is not None
            or args.container_memory is not None
            or args.warm_stripped_baseline_build
            or args.save_lake_build
            or args.clean_replay
            or args.skip_checkpoint_lean_verify
            or args.build_only_replay
            or args.checkpoint_index_file is not None
            or args.heartbeats
            or args.allow_relocated_dataset
            or args.prefire
            or args.ignore_incomplete
            or args.prepare_only
            or args.repositories_from is not None
            or args.allow_missing_subset
            or not args.inside_tmux
        ):
            parser.error("--run-manifest is reserved for the tmux launcher")
        if not os.environ.get("TMUX"):
            parser.error("--inside-tmux requires a tmux session")
        return _execute(load_run_manifest(args.run_manifest, repo_root=REPO_ROOT))
    if not args.runs:
        parser.error("at least one dataset config, run ID, or run.yaml path is required")
    if args.model is not None and len(args.runs) != 1:
        parser.error("--model requires exactly one dataset config name")
    if args.allow_missing_subset and args.repositories_from is None:
        parser.error("--allow-missing-subset requires --repositories-from")
    if args.validate_only and args.prepare_only:
        parser.error("--validate-only and --prepare-only are mutually exclusive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.threads_per_repo is not None and args.threads_per_repo < 1:
        parser.error("--threads-per-repo must be at least 1")

    references = list(args.runs)
    replay = args.replay
    workers = args.workers
    threads_per_repository = args.threads_per_repo
    if args.model is not None:
        evaluation = resolve_named_config(
            args.runs[0], args.model, repo_root=REPO_ROOT
        )
        if not evaluation.run_artifact_path.is_file():
            parser.error(
                "evaluation does not exist; run eval.py with this dataset/model first"
            )
        validate_named_run_identity(evaluation, repo_root=REPO_ROOT)
        postprocessing = load_named_postprocessing_config(
            args.runs[0], repo_root=REPO_ROOT
        )
        references = [str(evaluation.run_artifact_path)]
        # The config's replay default does not apply to an endpoints-only pass.
        replay = replay or (bool(postprocessing["replay"]) and not args.endpoints_only)
        workers = workers if workers is not None else postprocessing["workers"]
        threads_per_repository = (
            threads_per_repository
            if threads_per_repository is not None
            else postprocessing["threads_per_repository"]
        )

    repository_subset = None
    repository_subset_source = None
    if args.repositories_from is not None:
        subset_path = args.repositories_from.expanduser()
        if not subset_path.is_absolute():
            subset_path = REPO_ROOT / subset_path
        subset_path = subset_path.resolve()
        try:
            subset_document = yaml.safe_load(subset_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            parser.error(f"could not read --repositories-from {subset_path}: {error}")
        if not isinstance(subset_document, dict):
            parser.error("--repositories-from must contain a YAML mapping")
        repository_subset = subset_document.get("repositories")
        if not isinstance(repository_subset, list) or not repository_subset or not all(
            isinstance(value, str) and value for value in repository_subset
        ):
            parser.error("--repositories-from repositories must be a non-empty string list")
        prompt = (subset_document.get("agent") or {}).get("prompt")
        repository_subset_source = {
            "config": str(subset_path.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
            "canonical_prompt_sha256": (
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if isinstance(prompt, str)
                else None
            ),
        }

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

    if args.clean_replay and not replay:
        parser.error("--clean-replay requires --replay")
    if (args.skip_checkpoint_lean_verify or args.build_only_replay) and (not replay or args.clean_replay):
        parser.error("--skip-checkpoint-lean-verify and --build-only-replay need the default incremental --replay")
    prepared = resolve_run(
        references,
        repo_root=REPO_ROOT,
        replay=replay,
        endpoints_only=args.endpoints_only,
        workers=workers,
        threads_per_repository=threads_per_repository,
        measure_heartbeats=args.heartbeats,
        allow_relocated_dataset=args.allow_relocated_dataset,
        prefire=args.prefire,
        ignore_incomplete=args.ignore_incomplete,
        repository_subset=repository_subset,
        allow_missing_subset=args.allow_missing_subset,
        repository_subset_source=repository_subset_source,
        cgroup_parent=args.cgroup_parent,
        container_memory=args.container_memory,
        warm_baseline_build=args.warm_stripped_baseline_build,
        save_lake_build=args.save_lake_build,
        # Incremental by default: each checkpoint recompiles only what its edit touched.
        incremental_replay=replay and not args.clean_replay,
        skip_checkpoint_lean_verify=args.skip_checkpoint_lean_verify,
        build_only_replay=args.build_only_replay,
        checkpoint_selection=checkpoint_selection,
    )
    combination = prepared.manifest["run_combination"]
    source_lines = "\n".join(
        f"  {item['run_id']}" for item in combination["inputs"]
    )
    console.print(
        f"source runs (explicit order):\n{source_lines}\n"
        f"identity: {prepared.manifest['model']} / "
        f"{prepared.manifest['reasoning_effort']} / "
        f"{prepared.manifest['generator']}\n"
        f"repositories: {len(prepared.manifest['repositories'])}\n"
        f"mode: {'finals + every checkpoint' if replay else 'finals'}\n"
        f"heartbeats: {'reuse clean build' if args.heartbeats else 'disabled'}\n"
        f"relocated dataset: {'explicitly allowed' if args.allow_relocated_dataset else 'no'}\n"
        f"prefire: {'provisional cache warming' if args.prefire else 'no'}\n"
        f"workers: {prepared.manifest['parallelism']['workers']} (one repo each)\n"
        f"threads per repo: {prepared.manifest['parallelism']['threads_per_repository']}"
        f"\nwarm stripped baseline build: {'yes' if args.warm_stripped_baseline_build else 'no'}"
        f"\nsave .lake/build: {'yes' if args.save_lake_build else 'no'}"
    )
    if args.validate_only:
        console.print("[green]validated[/green]")
        return 0
    write_run_definition(prepared, repo_root=REPO_ROOT)
    if args.prepare_only:
        console.print(f"[green]prepared[/green]: {prepared.manifest_path}")
        return 0
    if _run_status(prepared.run_artifact_path) == "complete":
        console.print(f"[green]already complete[/green]: {prepared.run_artifact_path}")
        return 0

    if not os.environ.get("TMUX"):
        _launch_tmux(prepared.manifest_path, prepared.tmux_session)
        return 0
    return _execute(prepared)


if __name__ == "__main__":
    raise SystemExit(main())
