#!/usr/bin/env python3
"""Clean-build compressed endpoints after appending only exact sorried holdouts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.live import Live
from rich.table import Table

from leanlean.host_side_checkpointing import _write_source_archive
from leanlean.pipeline import postprocessing as pp
from leanlean.playback import _archive_source_files


ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def endpoint(instance_id: str, *, include_green_checkpoint: bool) -> tuple[Path, str, int | None]:
    base = ROOT / "output/postprocessing/leanlean-refactoring-ablation" / instance_id
    summary_path = base / "summary.json"
    if summary_path.exists():
        summary = load(summary_path)
        state = summary.get("selected_state", {})
        if summary.get("status") != "complete" or not (
            state.get("build_passed") is True and state.get("signatures_preserved") is True
        ):
            raise RuntimeError("submitted endpoint is not build-and-signature green")
        evidence = Path(summary["evidence"])
        archive = evidence.parent / "submitted.sources.tar.gz"
        if not archive.exists():
            raise FileNotFoundError(archive)
        return archive, "submitted_final", state.get("edit_index")

    if not include_green_checkpoint:
        raise RuntimeError("no completed submitted endpoint")
    candidates = sorted(base.glob("capture_*/postprocessed-playback.json"))
    if not candidates:
        raise RuntimeError("no postprocessed playback")
    evidence = candidates[-1]
    data = load(evidence)
    points = data.get("points") or data.get("checkpoints") or []
    green = [
        point for point in points
        if point.get("build", {}).get("passed") is True
        and point.get("signature_validation", {}).get("preserved") is True
    ]
    if not green:
        raise RuntimeError("no build-and-signature green checkpoint")
    point = green[-1]
    edit = int(point["edit_index"])
    archive = evidence.parent / f"checkpoint_{edit:05d}.sources.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(archive)
    return archive, "latest_green_checkpoint", edit


def split_imports(source: str) -> tuple[list[str], str]:
    imports: list[str] = []
    body: list[str] = []
    for line in source.splitlines(keepends=True):
        if re.match(r"^\s*import\s+\S", line):
            imports.append(line.rstrip("\r\n"))
        else:
            body.append(line)
    return imports, "".join(body).strip() + "\n"


def deduplicate_universes(context: str, existing: str) -> str:
    """Avoid redeclaring universe names already present in Solution.lean."""
    declared: set[str] = set()
    for match in re.finditer(r"(?m)^\s*universes?\s+([^\n]+)$", existing):
        declared.update(re.findall(r"[A-Za-z_][A-Za-z0-9_']*", match.group(1)))
    output: list[str] = []
    for line in context.splitlines(keepends=True):
        match = re.match(r"^(\s*)universes?\s+([^\n]+)(\r?\n)?$", line)
        if not match:
            output.append(line)
            continue
        names = re.findall(r"[A-Za-z_][A-Za-z0-9_']*", match.group(2))
        missing = [name for name in names if name not in declared]
        if missing:
            output.append(f"{match.group(1)}universe {' '.join(missing)}\n")
            declared.update(missing)
    return "".join(output)


def prepare(instance_id: str, manifest: dict[str, Any], out: Path) -> dict[str, Any]:
    repo_manifest_path = ROOT / manifest["repository_manifests"] / instance_id / "manifest.yaml"
    repo_manifest = load(repo_manifest_path)
    repository = repo_manifest["repository"]
    contract = repository["palomar_comparator"]["holdout"]
    solution = contract["parent"]["solution"]
    destination = solution["source_path"]
    holdouts_path = ROOT / manifest["dataset_directory"] / "repos" / instance_id / "holdouts.json"
    holdouts = load(holdouts_path)
    baseline, policy, edit = endpoint(
        instance_id,
        include_green_checkpoint=manifest["evaluation"]["allow_latest_green_checkpoint"],
    )
    files = _archive_source_files(baseline)
    if destination not in files:
        raise RuntimeError(f"registered solution source missing: {destination}")

    original = files[destination].decode("utf-8")
    imports: list[str] = []
    bodies: list[str] = []
    exact_sorried: list[str] = []
    original_proofs: list[str] = []
    for holdout in holdouts:
        new_imports, body = split_imports(holdout["source_file_with_sorry"])
        imports.extend(new_imports)
        bodies.append(body)
        exact_sorried.append(holdout["source_command_with_sorry"])
        original_proofs.append(holdout["source_command"])

    unique_imports = list(dict.fromkeys(imports))
    bodies = [deduplicate_universes(body, original) for body in bodies]
    import_policy = manifest["evaluation"].get("source_import_policy", "hoist_source_imports")
    if import_policy == "hoist_source_imports":
        prefix = "\n".join(unique_imports) + "\n\n"
    elif import_policy == "existing_solution_environment":
        prefix = ""
    else:
        raise ValueError(f"unsupported source_import_policy: {import_policy}")
    augmented = prefix + original.rstrip() + "\n\n" + "\n".join(bodies)
    files[destination] = augmented.encode("utf-8")
    if any(command in augmented for command in original_proofs):
        raise RuntimeError("an original proof body survived in the augmented solution")
    if any(command not in augmented for command in exact_sorried):
        raise RuntimeError("exact sorried holdout command missing from augmented solution")

    case_out = out / instance_id
    case_out.mkdir(parents=True, exist_ok=True)
    augmented_archive = case_out / "augmented.sources.tar.gz"
    archive_meta = _write_source_archive(augmented_archive, files)
    baseline_record = {
        "path": str(baseline.relative_to(ROOT)),
        "sha256": sha256(baseline),
        "selection_policy": policy,
        "edit_index": edit,
        "destination": destination,
        "solution_module": solution["module"],
        "holdouts": [h["declaration"] for h in holdouts],
        "original_proof_absent": True,
        "restored_supporting_declarations": 0,
        "source_import_policy": import_policy,
    }
    (case_out / "baseline.json").write_text(json.dumps(baseline_record, indent=2) + "\n")
    return {
        "repo_manifest": repo_manifest,
        "repo_manifest_path": repo_manifest_path,
        "repository": repository,
        "archive": augmented_archive,
        "archive_meta": archive_meta,
        "baseline": baseline_record,
    }


def run_one(instance_id: str, manifest: dict[str, Any], out: Path, statuses: dict[str, str]) -> dict[str, Any]:
    started = now()
    t0 = time.monotonic()
    case_out = out / instance_id
    try:
        with LOCK:
            statuses[instance_id] = "preparing exact interface"
        prepared = prepare(instance_id, manifest, out)
        repository = prepared["repository"]
        container = dict(manifest["container"])
        container["build_timeout_seconds"] = manifest["timeout_seconds"]
        with LOCK:
            statuses[instance_id] = "materializing offline image"
        with pp._materialized_repository_image(repository) as (resolved, retirement):
            env = pp._build_environment(resolved, container, 1)
            try:
                with LOCK:
                    statuses[instance_id] = "restoring augmented snapshot"
                env.restore_source_archive(prepared["archive"], ["."], timeout=900)
                pp._clear_verification_cache(env, timeout=manifest["timeout_seconds"])
                with LOCK:
                    statuses[instance_id] = "clean-building"
                build = env.execute(repository["build_command"], timeout=manifest["timeout_seconds"])
            finally:
                env.cleanup()
        passed = build.get("returncode") == 0 and not build.get("timed_out")
        output = str(build.get("output") or "")
        (case_out / "build.log").write_text(output)
        result = {
            "schema": "leanlean_holdout_interface_eligibility_result_v1",
            "instance_id": instance_id,
            "eligible": passed,
            "status": "eligible" if passed else "ineligible_build_failed",
            "started_at": started,
            "completed_at": now(),
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "baseline": prepared["baseline"],
            "augmented_archive": {
                "path": str(prepared["archive"].relative_to(ROOT)),
                "sha256": sha256(prepared["archive"]),
                **prepared["archive_meta"],
            },
            "build_command": repository["build_command"],
            "build_returncode": build.get("returncode"),
            "build_timed_out": bool(build.get("timed_out")),
            "original_proof_absent": True,
            "restored_supporting_declarations": 0,
        }
    except Exception as exc:
        result = {
            "schema": "leanlean_holdout_interface_eligibility_result_v1",
            "instance_id": instance_id,
            "eligible": False,
            "status": "infrastructure_error",
            "started_at": started,
            "completed_at": now(),
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }
        case_out.mkdir(parents=True, exist_ok=True)
    (case_out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with LOCK:
        statuses[instance_id] = result["status"]
    return result


def table(statuses: dict[str, str], order: list[str]) -> Table:
    value = Table(title="Exact sorried-holdout interface eligibility")
    value.add_column("Repository")
    value.add_column("Status")
    for instance_id in order:
        value.add_row(instance_id, statuses[instance_id])
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load(manifest_path)
    monitoring = manifest.get("monitoring", {})
    if not (monitoring.get("snapshots") is True and monitoring.get("every_n_edits") == 1 and monitoring.get("native_traces") is True):
        raise ValueError("manifest must retain snapshots and native traces")
    if manifest.get("model_requests") != 0 or manifest.get("model") != "none":
        raise ValueError("this deterministic eligibility experiment must make no model calls")
    if manifest["container"].get("network_policy") != "none":
        raise ValueError("build-only containers must use network_policy: none")

    order = list(manifest["repositories"])
    out = ROOT / manifest["output_directory"]
    out.mkdir(parents=True, exist_ok=True)
    statuses = {instance_id: "queued" for instance_id in order}
    events = out / "events.jsonl"
    results: list[dict[str, Any]] = []
    console = Console()
    with Live(table(statuses, order), console=console, refresh_per_second=2) as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(manifest["workers"])) as pool:
            future_map = {pool.submit(run_one, iid, manifest, out, statuses): iid for iid in order}
            while future_map:
                done, _ = concurrent.futures.wait(future_map, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
                live.update(table(statuses, order))
                for future in done:
                    iid = future_map.pop(future)
                    result = future.result()
                    results.append(result)
                    with events.open("a") as stream:
                        stream.write(json.dumps({"at": now(), "instance_id": iid, "status": result["status"]}) + "\n")
    results.sort(key=lambda item: order.index(item["instance_id"]))
    summary = {
        "schema": "leanlean_holdout_interface_eligibility_summary_v1",
        "run_id": manifest["run_id"],
        "completed_at": now(),
        "counts": {
            "total": len(results),
            "eligible": sum(r["status"] == "eligible" for r in results),
            "ineligible_build_failed": sum(r["status"] == "ineligible_build_failed" for r in results),
            "infrastructure_error": sum(r["status"] == "infrastructure_error" for r in results),
        },
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["counts"], sort_keys=True))
    return 1 if summary["counts"]["infrastructure_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
