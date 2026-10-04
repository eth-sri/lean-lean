#!/usr/bin/env python3
"""Run a paired reproving ablation sequentially from its authored YAML manifest."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict:
    path = path.resolve()
    if not path.is_relative_to(ROOT / "experiments"):
        raise ValueError("manifest must live under experiments/")
    spec = yaml.safe_load(path.read_text())
    if spec.get("kind") != "leanlean_reproving_ablation" or spec.get("schema_version") != 1:
        raise ValueError("unexpected paired experiment manifest")
    if [arm.get("name") for arm in spec.get("arms", [])] != ["baseline", "post-compression"]:
        raise ValueError("paired experiment must contain baseline then post-compression")
    if spec.get("execution", {}).get("launch_mode") != "sequential":
        raise ValueError("paired experiment must run sequentially within its memory ceiling")
    if spec.get("execution", {}).get("runner") != "eval.sh":
        raise ValueError("paired experiment must use eval.sh")
    if spec.get("execution", {}).get("workers") not in {2, 6, 12}:
        raise ValueError("paired experiment must use a supported model worker count (2, 6, or 12)")
    container = spec.get("container", {})
    if container.get("network_policy") != "model_proxy_only" or container.get("pids_limit") != 4096:
        raise ValueError("container isolation policy drift")
    if container.get("cpus_per_worker") != 8 or container.get("memory_per_worker") != "64g":
        raise ValueError("worker resource policy drift")
    agent = spec.get("agent", {})
    if agent.get("enable_lean_verify") is not True or agent.get("enable_proof_length") is not False:
        raise ValueError("verifier policy drift")
    if agent.get("enable_subagents") is not False or agent.get("allowed_skills") != [] or agent.get("mcp_servers") != []:
        raise ValueError("agent isolation policy drift")
    monitoring = spec.get("monitoring", {})
    if monitoring.get("snapshots") is not True or monitoring.get("every_n_edits") != 1:
        raise ValueError("monitoring policy drift")
    if monitoring.get("retain_native_traces") is not True or monitoring.get("retain_native_rollouts") is not True:
        raise ValueError("native trace retention policy drift")
    for arm in spec["arms"]:
        argv = shlex.split(arm["command"])
        if argv[:2] != ["bash", "eval.sh"] or "--model" not in argv:
            raise ValueError(f"{arm['name']}: command must invoke eval.sh with an explicit model")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--inside-tmux", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    spec = load_manifest(manifest)
    if args.validate_only:
        print(f"validated {spec['run_id']}: 2 arms, 30 attempts, {spec['execution']['workers']} workers, Lean Verify enabled")
        return 0
    session = spec["execution"]["tmux_session"]
    if not args.inside_tmux:
        if subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0:
            raise RuntimeError(f"tmux session already exists: {session}")
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), str(manifest), "--inside-tmux"])
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), command], check=True)
        subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
        print("tmux attach -t " + session)
        return 0
    if not os.environ.get("TMUX"):
        raise RuntimeError("paired run must execute inside tmux")
    state_path = ROOT / "runs" / "evaluation" / spec["run_id"] / "run.json"
    state = {"kind": "leanlean_reproving_ablation_run", "run_id": spec["run_id"],
             "manifest": str(manifest.relative_to(ROOT)), "status": "running",
             "started_at": time.time(), "arms": {arm["name"]: {"status": "queued"} for arm in spec["arms"]}}
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(state_path, state)
        for arm in spec["arms"]:
            name = arm["name"]
            state["arms"][name] = {"status": "running", "started_at": time.time(), "output": arm["output"]}
            atomic_json(state_path, state)
            result = subprocess.run(shlex.split(arm["command"]), cwd=ROOT)
            state["arms"][name].update(status="complete" if result.returncode == 0 else "failed",
                                       finished_at=time.time(), exit_code=result.returncode)
            atomic_json(state_path, state)
            if result.returncode != 0:
                state.update(status="failed", finished_at=time.time(), failed_arm=name)
                atomic_json(state_path, state)
                return result.returncode
        state.update(status="complete", finished_at=time.time())
        atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
