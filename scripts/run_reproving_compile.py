#!/usr/bin/env python3
"""Compile submitted reproving patches offline and compare with pinned prior builds."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

import yaml
from rich.live import Live
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from leanlean.environments.docker import DockerEnvironment
from leanlean.benchmarks.leanlean import _install_palomar_task_metadata
from leanlean.palomar_comparator import resolve_palomar_contract
from leanlean.pipeline.postprocessing import _clear_verification_cache
from scripts.reproof_offline_dependencies import install as install_offline_dependencies

KIND = "leanlean_reproving_compile_comparison"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(path: Path) -> tuple[dict, dict]:
    path = path.resolve()
    if not path.is_relative_to(ROOT / "experiments"):
        raise ValueError("manifest must live under experiments/")
    spec = yaml.safe_load(path.read_text())
    if spec.get("kind") != KIND or spec.get("schema_version") != 1:
        raise ValueError("unexpected compile manifest")
    if spec["container"]["network_policy"] != "none":
        raise ValueError("compilation must be offline")
    if spec["evaluation"] != {
        "project_build_command": "LEAN_NUM_THREADS=8 lake build",
        "project_build_required": False,
        "lean_verify": True,
        "proof_length": False,
        "clear_project_build_cache": True,
        "success_criterion": "lean_verify",
        "offline_dependency_policy": "pinned_local_package_paths",
    }:
        raise ValueError("compile-only evaluation policy drift")
    monitoring = spec["monitoring"]
    if monitoring.get("snapshots") is not True or monitoring.get("every_n_edits") != 1:
        raise ValueError("source snapshots must remain enabled")
    predictions_path = ROOT / spec["predictions"]["path"]
    if digest(predictions_path) != spec["predictions"]["sha256"]:
        raise ValueError("prediction pin drift")
    predictions = json.loads(predictions_path.read_text())
    contract_path = ROOT / spec["contract_database"]["path"]
    if digest(contract_path) != spec["contract_database"]["sha256"]:
        raise ValueError("protected contract database drift")
    contract_database = json.loads(contract_path.read_text())
    contract_rows = {row["instance_id"]: row for row in contract_database["repositories"]}
    ids = [row["id"] for row in spec["repositories"]]
    if len(ids) != len(set(ids)) or set(ids) != set(predictions):
        raise ValueError("repository/prediction set mismatch")
    if not set(ids).issubset(contract_rows):
        raise ValueError("protected contract set mismatch")
    contract_rows = {iid: contract_rows[iid] for iid in ids}
    for row in spec["repositories"]:
        receipt = ROOT / row["materialization_receipt"]["path"]
        old = ROOT / row["old_result"]["path"]
        if digest(receipt) != row["materialization_receipt"]["sha256"]:
            raise ValueError(f"{row['id']}: materialization receipt drift")
        if digest(old) != row["old_result"]["sha256"]:
            raise ValueError(f"{row['id']}: old result drift")
        receipt_data = json.loads(receipt.read_text())
        if receipt_data["instance_id"] != row["id"] or receipt_data["image_id"] != row["image_id"]:
            raise ValueError(f"{row['id']}: image receipt mismatch")
        old_data = json.loads(old.read_text())[row["old_result"]["key"]]
        if bool(old_data.get("repo_build_passed")) != bool(row["old_result"]["build_passed"]):
            raise ValueError(f"{row['id']}: old result mismatch")
    workers = int(spec["parallelism"]["workers"])
    if workers < 1 or workers > len(ids):
        raise ValueError("invalid worker count")
    return spec, predictions, contract_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--inside-tmux", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    spec, predictions, contract_rows = load(manifest)
    if args.validate_only:
        print(f"validated {len(spec['repositories'])} compile-only submissions")
        return 0
    if not args.inside_tmux:
        session = spec["outputs"]["tmux_session"]
        if subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0:
            raise RuntimeError(f"tmux session already exists: {session}")
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), str(manifest), "--inside-tmux"])
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), command], check=True)
        subprocess.run(["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True)
        print("tmux attach -t " + session)
        return 0
    if not os.environ.get("TMUX"):
        raise RuntimeError("compile run must execute inside tmux")

    output = ROOT / spec["outputs"]["directory"]
    state_path = ROOT / spec["outputs"]["run_artifact"]
    summary_path = ROOT / spec["outputs"]["summary"]
    output.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_sha = digest(manifest)
    mutex = threading.Lock()
    state = {
        "kind": KIND,
        "status": "running",
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": manifest_sha,
        "started_at": time.time(),
        "repositories": {row["id"]: {"status": "queued"} for row in spec["repositories"]},
    }

    def update(iid: str, status: str, **extra: object) -> None:
        with mutex:
            state["repositories"][iid].update(status=status, **extra)
            state["updated_at"] = time.time()
            atomic_json(state_path, state)

    def worker(row: dict) -> dict:
        iid = row["id"]
        destination = output / iid
        destination.mkdir(parents=True, exist_ok=True)
        patch = predictions[iid]["model_patch"]
        (destination / "submitted.patch").write_text(patch)
        update(iid, "starting offline container")
        resources = spec["container"]
        env = DockerEnvironment(
            image=row["image_id"], cwd="/testbed", env={}, forward_env=[],
            timeout=int(resources["timeout_seconds"]),
            run_args=[
                f"--cpus={resources['cpus']}",
                f"--memory={resources['memory']}",
                f"--memory-swap={resources['memory']}",
                "--ulimit=stack=33554432:33554432",
                "--ulimit=nofile=1048576:1048576",
                f"--cgroup-parent={resources['cgroup_parent']}",
            ],
            network_policy="none", container_timeout=resources["container_timeout"],
            container_pids_limit=int(resources["pids_limit"]),
            monitoring_metadata={
                "run_id": spec["run_id"], "role": "reproof_compilation",
                "model": spec["source_model"], "instance_id": iid,
                "manifest": str(manifest.relative_to(ROOT)),
            },
        )
        try:
            update(iid, "capturing baseline source")
            env.export_source_archive(destination / "snapshot_000_baseline.sources.tar.gz", ["."], timeout=300)
            env.write_file_bytes("/tmp/submitted.patch", patch.encode())
            applied = env.execute("git apply --whitespace=nowarn /tmp/submitted.patch", timeout=120)
            (destination / "apply.log").write_text(applied.get("output", ""))
            if applied.get("returncode"):
                result = {
                    "id": iid, "old_build_passed": row["old_result"]["build_passed"],
                    "new_build_passed": False, "stage": "apply_patch",
                    "returncode": applied.get("returncode"), "timed_out": False,
                }
                atomic_json(destination / "result.json", result)
                update(iid, "patch apply failed", result=result)
                return result
            update(iid, "capturing patched source")
            env.export_source_archive(destination / "snapshot_001_patched_terminal.sources.tar.gz", ["."], timeout=300)
            update(iid, "normalizing pinned offline dependencies")
            normalization = install_offline_dependencies(env)
            atomic_json(destination / "dependency-normalization.json", normalization)
            env.execute("rm -rf .lake/build", timeout=300)
            update(iid, "compiling")
            started = time.monotonic()
            execution = env.execute(spec["evaluation"]["project_build_command"], timeout=int(resources["timeout_seconds"]))
            elapsed = round(time.monotonic() - started, 3)
            (destination / "build.log").write_text(execution.get("output", ""))
            passed = execution.get("returncode") == 0 and not execution.get("timed_out", False)
            verify_passed = False
            verify_returncode = None
            verify_timed_out = False
            contract = resolve_palomar_contract(repo_root=ROOT, database_row=contract_rows[iid])
            if "sorryAx" in contract["permitted_axioms"]:
                raise ValueError(f"{iid}: final verification must reject sorryAx")
            update(iid, "installing offline verifier")
            _clear_verification_cache(env, timeout=int(resources["timeout_seconds"]))
            _install_palomar_task_metadata(
                env, contract=contract, exclude_dirs=[], include_prefix="",
                build_jobs=int(resources["cpus"]), prewarm=False,
                enable_lean_verify=True, enable_proof_length=False,
            )
            update(iid, "lean_verify")
            verification = env.execute(
                "LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify",
                timeout=int(resources["timeout_seconds"]),
            )
            (destination / "lean_verify.log").write_text(verification.get("output", ""))
            verify_returncode = verification.get("returncode")
            verify_timed_out = bool(verification.get("timed_out", False))
            verify_passed = verify_returncode == 0 and not verify_timed_out
            result = {
                "id": iid,
                "old_build_passed": bool(row["old_result"]["build_passed"]),
                "old_resolved": bool(row["old_result"]["resolved"]),
                "old_build_seconds": row["old_result"].get("build_seconds"),
                "new_build_passed": passed,
                "new_build_seconds": elapsed,
                "lean_verify_passed": verify_passed,
                "lean_verify_returncode": verify_returncode,
                "lean_verify_timed_out": verify_timed_out,
                "resolved": verify_passed,
                "contract_sha256": contract["sha256"],
                "permitted_axioms": contract["permitted_axioms"],
                "stage": "lake_build",
                "returncode": execution.get("returncode"),
                "timed_out": bool(execution.get("timed_out", False)),
                "image_id": row["image_id"],
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            }
            atomic_json(destination / "result.json", result)
            update(iid, "passed" if result["resolved"] else "failed", result=result)
            return result
        except Exception as error:
            result = {"id": iid, "old_build_passed": bool(row["old_result"]["build_passed"]),
                      "new_build_passed": False, "stage": "infrastructure",
                      "error": f"{type(error).__name__}: {error}"}
            atomic_json(destination / "result.json", result)
            update(iid, "infrastructure error", result=result)
            return result
        finally:
            env.cleanup()

    def dashboard() -> Table:
        table = Table("Repository", "Compilation", title=spec["run_id"])
        with mutex:
            for iid, row in state["repositories"].items():
                table.add_row(iid, row["status"])
        return table

    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(state_path, state)
        results = []
        with ThreadPoolExecutor(max_workers=int(spec["parallelism"]["workers"])) as pool, Live(dashboard(), refresh_per_second=2) as live:
            pending = {pool.submit(worker, row): row["id"] for row in spec["repositories"]}
            while pending:
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    results.append(future.result())
                live.update(dashboard())
        results.sort(key=lambda row: row["id"])
        old_pass = sum(bool(row.get("old_resolved", row["old_build_passed"])) for row in results)
        new_pass = sum(bool(row["new_build_passed"]) for row in results)
        new_verified = sum(bool(row.get("resolved")) for row in results)
        summary = {
            "kind": KIND,
            "run_id": spec["run_id"],
            "comparison": {"old_verified_pass": old_pass, "new_project_build_pass": new_pass, "new_verified_pass": new_verified, "total": len(results), "project_build_delta_vs_old": new_pass - old_pass, "verified_delta": new_verified - old_pass},
            "results": results,
            "lean_verify": True,
            "proof_length": False,
        }
        atomic_json(summary_path, summary)
        state.update(status="complete", finished_at=time.time(), comparison=summary["comparison"])
        atomic_json(state_path, state)
        print(json.dumps(summary["comparison"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
