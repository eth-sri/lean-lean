#!/usr/bin/env python3
"""Clean-build and original-Comparator preflight for the uncompressed arm."""
import argparse
import concurrent.futures
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import yaml
from rich.live import Live
from rich.table import Table
from scripts.package_valid_compressed_holdouts import _write_json, checked, pin
from leanlean.preprocessing.palomar_sources import localize_lake_manifest
from scripts.holdout_complete_challenge_materialization import verify_scaffold_signatures
from leanlean.pipeline.reconstruction import _copy_into_container
from leanlean.preprocessing.cache_isolation import render_container_cache_cleanup
from leanlean.environments.docker import DockerEnvironment
from leanlean.palomar_comparator import resolve_palomar_contract

DATASET = ROOT / "datasets/compressed-sorried-holdouts-20260912-r1"
MANIFEST = ROOT / "experiments/datasets/uncompressed-stage2-preflight-20260912-r1.yaml"


def plan(path):
    if path.exists():
        raise ValueError("manifest exists")
    source = yaml.safe_load((ROOT / "experiments/datasets/uncompressed-sorried-holdouts-20260912-r1.yaml").read_text())
    cases = []
    for case in source["cases"]:
        report = json.loads(checked(case["report"]).read_text())
        directory = DATASET / "repos" / case["id"] / "uncompressed"
        cases.append({
            "id": case["id"], "repository": case["repository"],
            "image_id": report["stripped_output_image_id"],
            "source_report": case["report"],
            "archive": pin(directory / "monitoring/uncompressed.sources.tar.gz"),
            "build_target": report["build_target"],
        })
    manifest = {
        "kind": "leanlean_stage2_dataset_preflight", "schema_version": 1,
        "run_id": "uncompressed-stage2-preflight-20260912-r1",
        "model": "none", "reasoning_effort": "none",
        "generator": "clean_source_build_original_comparator_preflight",
        "repository_variant": "stripped", "rounds": 1, "workers": 4,
        "container": {"cpus": 4, "memory": "32g", "pids_limit": 4096,
                      "network_policy": "model_proxy_only", "cgroup_parent": "lean.slice"},
        "timeout_seconds": 43200,
        "evaluation": {"clean_project_cache": True, "original_comparator": True,
                       "sorryAx": "interface_preflight_only", "launch_reproof": False},
        "monitoring": {"snapshots": True, "every_n_edits": 1, "native_traces": True,
                       "native_rollout_retention": True, "model_rollouts": "not_applicable",
                       "baseline": "baseline.sources.tar.gz", "terminal": "terminal.sources.tar.gz"},
        "dataset": pin(DATASET / "uncompressed.yaml"),
        "database": pin(DATASET / "repository-database.uncompressed.json"),
        "output": "output/validation/uncompressed-stage2-preflight-20260912-r1",
        "tmux_session": "validate-uncompressed-stage2-20260912-r1", "cases": cases,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def run_case(case, manifest, manifest_path, output, database, states):
    key = case["id"]
    directory = output / key
    directory.mkdir(parents=True, exist_ok=True)
    archive = checked(case["archive"])
    shutil.copy2(archive, directory / "baseline.sources.tar.gz")
    env = None
    states[key] = {"status": "running", "phase": "restoring source"}
    result = {"passed": False, "clean_build_passed": False, "signature_comparator_passed": False}
    try:
        resources = manifest["container"]
        timeout = manifest["timeout_seconds"]
        env = DockerEnvironment(
            image=case["image_id"], cwd="/testbed", timeout=timeout,
            container_timeout=f"{timeout * 2}s", network_policy=resources["network_policy"],
            container_pids_limit=resources["pids_limit"],
            monitoring_metadata={"run_id": manifest["run_id"], "role": "validation",
                                 "model": "none", "instance_id": key, "manifest": str(manifest_path)},
            run_args=["--rm", f"--cpus={resources['cpus']}", f"--memory={resources['memory']}",
                      f"--memory-swap={resources['memory']}", "--ulimit=nofile=1048576:1048576",
                      "--ulimit=stack=33554432:33554432", f"--cgroup-parent={resources['cgroup_parent']}"],
        )
        env.restore_source_archive(archive, ["."], timeout=900)
        cleanup = render_container_cache_cleanup(clear_project=True, clear_config=True)
        cleaned = env.execute("python3 -c " + shlex.quote(cleanup), timeout=900)
        if cleaned["returncode"] != 0:
            raise RuntimeError("cache cleanup failed")
        source = DATASET / "repos" / key / "uncompressed/source"
        with tempfile.TemporaryDirectory(prefix="stage2-offline-packages-") as temp:
            temp_path = Path(temp)
            shutil.copy2(source / "lake-manifest.json", temp_path / "lake-manifest.json")
            localize_lake_manifest(temp_path)
            _copy_into_container(env.container_id, temp_path / ".lake/package-overrides.json",
                                 "/testbed/.lake/package-overrides.json")
        states[key] = {"status": "running", "phase": "clean source build"}
        with (directory / "build.log").open("w") as log:
            built = subprocess.run(
                ["docker", "exec", "-w", "/testbed", "-e", f"LEAN_NUM_THREADS={resources['cpus']}",
                 env.container_id, "lake", "build", *shlex.split(case["build_target"] or "")],
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
            )
        if built.returncode:
            raise RuntimeError(f"clean build failed ({built.returncode}); see build.log")
        result["clean_build_passed"] = True
        states[key] = {"status": "running", "phase": "original Comparator"}
        contract = resolve_palomar_contract(repo_root=ROOT, database_row=database[key])
        challenge = "/testbed/" + contract["challenge"]["source_path"]
        env.execute("rm -f " + shlex.quote(challenge), timeout=60)
        with tempfile.TemporaryDirectory(prefix="stage2-comparator-") as temp:
            verify_scaffold_signatures(env.container_id, {
                "preflight_contract": contract, "container": {"build_jobs": resources["cpus"]},
                "timeout": timeout, "direct_comparator_env": True,
                "preflight_log": str(directory / "comparator.log"),
            }, Path(temp))
        result.update(passed=True, signature_comparator_passed=True)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        (directory / "error.log").write_text(result["error"])
    finally:
        shutil.copy2(archive, directory / "terminal.sources.tar.gz")
        if env is not None:
            env.cleanup()
    result.update(status="passed" if result["passed"] else "failed", phase="finished",
                  source_archive=case["archive"])
    _write_json(directory / "result.json", result)
    _write_json(DATASET / "repos" / key / "uncompressed/preflight.json", result)
    return key, result


def table(states):
    view = Table(title="Uncompressed Stage 2: clean build + original Comparator")
    for column in ("Case", "Status", "Phase"):
        view.add_column(column)
    for key, state in states.items():
        view.add_row(key, state["status"], state["phase"])
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=MANIFEST)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    if args.plan:
        plan(path)
        return
    manifest = yaml.safe_load(path.read_text())
    monitor = manifest["monitoring"]
    assert monitor["snapshots"] is True and monitor["every_n_edits"] == 1 and monitor["native_traces"] is True
    assert manifest["model"] == "none" and manifest["evaluation"]["launch_reproof"] is False
    checked(manifest["dataset"])
    database = {r["instance_id"]: r for r in json.loads(checked(manifest["database"]).read_text())["repositories"]}
    for case in manifest["cases"]:
        checked(case["archive"])
        subprocess.run(["docker", "image", "inspect", case["image_id"]], check=True, capture_output=True)
    if args.validate_only:
        print(f"Validated {len(manifest['cases'])} Stage 2 preflights with snapshots and native logs")
        return
    output = ROOT / manifest["output"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output / "manifest.snapshot.yaml")
    states = {c["id"]: {"status": "queued", "phase": "queued"} for c in manifest["cases"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=manifest["workers"]) as pool, Live(table(states), refresh_per_second=1) as live:
        pending = {pool.submit(run_case, c, manifest, path, output, database, states) for c in manifest["cases"]}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                key, result = future.result()
                states[key] = result
            _write_json(output / "status.json", {"run_id": manifest["run_id"], "monitoring": monitor, "cases": states})
            live.update(table(states))
    _write_json(output / "summary.json", {"cases": states, "passed": all(r["passed"] for r in states.values())})


if __name__ == "__main__":
    main()
