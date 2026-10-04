#!/usr/bin/env python3
"""Run preprocessing, compression, and paired theorem reconstruction from YAML."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from leanlean.pipeline.reconstruction import _load_manifest  # noqa: E402


console = Console()


def _relative_command_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{name} escapes the repository")
    return path.relative_to(REPO_ROOT).as_posix()


def _session_name(run_id: str) -> str:
    base = "theorem-ablation-" + re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)
    return base[:80]


def _launch(manifest_path: Path, session: str) -> None:
    if subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        capture_output=True,
    ).returncode == 0:
        console.print(f"[yellow]already running[/yellow]: {session}")
        console.print(f"tmux attach -t {shlex.quote(session)}")
        return
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(manifest_path),
            "--inside-tmux",
        ]
    )
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
        ["tmux", "set-option", "-t", session, "remain-on-exit", "on"],
        check=True,
    )
    console.print(f"[green]launched[/green] {session}")
    console.print(f"tmux attach -t {shlex.quote(session)}")


def _run_stage(label: str, command: list[str]) -> None:
    console.rule(label)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--inside-tmux", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else (REPO_ROOT / args.manifest).resolve()
    )
    manifest = _load_manifest(manifest_path)
    session = _session_name(str(manifest["run_id"]))
    if not os.environ.get("TMUX") and not args.inside_tmux:
        _launch(manifest_path, session)
        return 0
    if not os.environ.get("TMUX"):
        raise RuntimeError("--inside-tmux is reserved for the tmux launcher")

    preprocessing = manifest["preprocessing"]
    compression = manifest["compression"]
    evaluation = manifest["evaluation"]
    preprocess_config = _relative_command_path(
        preprocessing.get("config"), "preprocessing.config"
    )
    compression_config = str(compression.get("dataset_config") or "")
    compression_model = str(compression.get("model_config") or "")
    proof_config = str(evaluation.get("dataset_config") or "")
    proof_model = str(evaluation.get("model_config") or "")
    if not all((compression_config, compression_model, proof_config, proof_model)):
        raise ValueError("compression/evaluation must name dataset and model configs")

    _run_stage(
        "1/6 preprocess with protected-main-theorem holdout",
        [str(REPO_ROOT / "preprocess.sh"), preprocess_config],
    )
    _run_stage(
        "2/6 measure and validate the ablation size",
        [
            "uv", "run", "python", "scripts/reconstruct.py",
            str(manifest_path), "validate",
        ],
    )
    _run_stage(
        "3/6 compress the stripped repository",
        [
            "uv", "run", "python", "eval.py",
            compression_config, "--model", compression_model,
        ],
    )
    _run_stage(
        "verify compression before using it as the treatment",
        ["uv", "run", "python", "scripts/reconstruct.py", str(manifest_path), "score-compression"],
    )
    _run_stage(
        "4/6 materialize stripped and compressed reconstruction arms",
        [
            "uv", "run", "python", "scripts/reconstruct.py",
            str(manifest_path), "prepare",
        ],
    )
    _run_stage(
        "5/6 reprove the same main theorem in both arms",
        [
            "uv", "run", "python", "eval.py",
            proof_config, "--model", proof_model,
        ],
    )
    _run_stage(
        "independently verify both reconstruction arms",
        ["uv", "run", "python", "scripts/reconstruct.py", str(manifest_path), "score"],
    )
    _run_stage(
        "6/6 compute paired code-size and timing deltas",
        [
            "uv", "run", "python", "scripts/reconstruct.py",
            str(manifest_path), "summarize",
        ],
    )
    console.print("[green]theorem reconstruction experiment complete[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
