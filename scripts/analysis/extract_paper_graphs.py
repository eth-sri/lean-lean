#!/usr/bin/env python3
"""Dependency graphs of evaluated endpoints, with the paper's frozen extractor.

The paper's graphs come from one Lean program, frozen on 2026-09-07 and kept
byte for byte in preprocessing-extractor-same-import-scope.lean beside this
script. It imports the repository's entry module and reads the project's
compiled declarations and their direct dependencies (no proof erasure); the
.ilean and source-text layers are the paper's preprocessing ones. Every graph is read from a
built tree:

  before  the stripped baseline image as it is; one graph per repository,
          shared by every model (`baseline_graphs` in the manifest)
  after   each verified submission: its submitted sources plus the .lake/build
          postprocessing retained for them (postprocess.py --save-lake-build);
          without a retained build the project is compiled once in the container

Failed submissions get no graph; they save nothing. The output is what
classify_compression.py reads:

  repositories/<id>/{before,after}.json
  repositories/<id>/{before,after}/original-sources.tar.gz
  repositories/<id>/classification-record.json
  classification-inputs.csv

Usage:
    python scripts/analysis/extract_paper_graphs.py experiments/analysis/<run>.yaml

The manifest (experiments/analysis/demo-luna-paper-graphs.yaml is one) names the run, its
`model` and `dataset`, the `evaluation` (`experiment`: the resolved evaluation
manifest; `output_directory`), the `repositories`, `baseline_graphs`,
`resources`, `timeouts`, `monitoring` and `outputs.directory`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lean_strip._pipeline.preprocessing.ilean import union_ilean_reference_edges  # noqa: E402
from leanlean.benchmarks.leanlean import _discover_modules  # noqa: E402
from leanlean.environments import get_environment  # noqa: E402
from leanlean.evaluation_images import ensure_materialized_image_from_record, materialized_image_id  # noqa: E402
from leanlean.host_side_checkpointing import _write_source_archive  # noqa: E402
from leanlean.preprocessing import olean, strip  # noqa: E402
from leanlean.preprocessing.graph_artifact import build_graph_artifact, write_graph_artifact  # noqa: E402
from leanlean.preprocessing.lean_strip_graph import (  # noqa: E402
    DockerWorkspace, list_project_lean_files, snapshot_sources,
)
from scripts.analysis.graph_container_monitoring import cgroup_sample, inspect  # noqa: E402
from scripts.analysis.quantify_compression_categories import (  # noqa: E402
    SourceIndex, challenge_source_path, load_protected,
)

EXTRACTOR = Path(__file__).with_name("preprocessing-extractor-same-import-scope.lean")
EXTRACTOR_SHA256 = "5dcde2a59b32212a68da1436685a605f1b7445e258b07ca5c4c2a0348705cabb"
LOG = logging.getLogger("paper-graphs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def static_layers(env: Any, declarations: list[dict], dependencies: Mapping[str, set[str]],
                  originals: Mapping[str, str], paths: Mapping[str, str], modules: set[str]) -> tuple[dict, dict]:
    """The three evidence layers, each found independently; an edge records every layer that found it.

    The paper's preprocessing code: lean-strip's .ilean reader, and the
    source-text scan of leanlean.preprocessing.strip (lean-strip's own scan has
    since changed)."""
    kernel = {(a, b) for a, targets in dependencies.items() for b in targets}
    ilean_deps, ilean_report = union_ilean_reference_edges(DockerWorkspace(env), declarations, {}, modules=modules)
    if ilean_report["unsupported_versions"] or ilean_report["modules_missing"]:
        raise RuntimeError(f".ilean evidence is incomplete: unsupported_versions="
                           f"{ilean_report['unsupported_versions']}, missing_modules={ilean_report['modules_missing']}")
    ilean = {(a, b) for a, targets in ilean_deps.items() for b in targets}
    source_deps = strip.augment_deps_with_source(declarations, {}, dict(originals), dict(paths))
    source = {(a, b) for a, targets in source_deps.items() for b in targets}
    ilean_report.update(already_present_edges=len(ilean & kernel), added_edges=len(ilean - kernel))
    evidence = {
        "schema": "static_dependency_evidence_v2", "composition": "set_union",
        "attribution": "multi_label_per_edge", "kernel_edges": len(kernel), "ilean": ilean_report,
        "ilean_edges": len(ilean), "source_text_edges": len(source), "ilean_edges_added": len(ilean - kernel),
        "source_text_edges_added": len(source - kernel - ilean), "total_edges": len(kernel | ilean | source),
    }
    return evidence, {"kernel": kernel, "ilean": ilean, "source_text": source}


def submitted_endpoint_passed(item: Mapping[str, Any]) -> bool:
    # A timed-out run may score an earlier green checkpoint; only the submitted
    # endpoint's own verification decides whether it gets a graph.
    return all(item.get(key) is True for key in (
        'build_passed', 'signatures_preserved', 'semantic_integrity_preserved',
        'evaluation_infrastructure_ok', 'benchmark_fixtures_preserved',
    ))


def resolve_jobs(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One baseline job per repository without a graph, one per verified submission."""
    dataset = ROOT / manifest["dataset"]
    experiment = yaml.safe_load((ROOT / manifest["evaluation"]["experiment"]).read_text())
    if (experiment["model"], experiment["dataset"]["manifest"]) != (manifest["model"], manifest["dataset"]):
        raise ValueError("model or dataset differs from the evaluation run")
    _, scopes, _ = load_protected(dataset)
    database = json.loads((dataset.parent / "repository-database.json").read_text())
    entry = {row["instance_id"]: row["scopes"]["entry_module"] for row in database["repositories"]}
    targets = {row["instance_id"]: row["scopes"]["build_target"] for row in database["repositories"]}
    images = {str(row["id"]): row for row in experiment["repositories"]}
    output = ROOT / manifest["evaluation"]["output_directory"]
    report = json.loads((output / "report_summary.json").read_text())["instances"]
    baselines = ROOT / manifest["baseline_graphs"]
    jobs, endpoints = [], []
    for repo in manifest["repositories"]:
        latest = report[repo]["latest"]
        metrics = latest["metrics"]
        scope = dict(scopes[repo], excluded_files=[])
        challenge = challenge_source_path(dataset.parent / "repos", repo)
        if challenge:
            scope["excluded_files"].append(challenge)
        # Evaluations that built their images from the published trees record how (build_before_agent);
        # a locally preprocessed dataset pins the image preprocessing committed.
        row = images[repo]
        image = row.get("materialization") or dict(tag=row["image"], image_id=row["image_id"],
                                                    build_target=targets[repo])
        base = dict(repository=repo, entry_module=entry[repo], materialization=image)
        endpoint = dict(repository=repo, status="passed" if submitted_endpoint_passed(latest) else "failed",
                        baseline_tokens=int(metrics["baseline_lean_tokens"]),
                        post_tokens=int(metrics["post_lean_tokens"]), metric_scope=scope)
        endpoints.append(endpoint)
        if not (baselines / repo / "before.json").is_file():
            jobs.append(dict(base, phase="before", destination=baselines / repo))
        pair = ROOT / manifest["outputs"]["directory"] / "repositories" / repo
        if endpoint["status"] == "passed" and not (pair / "after.json").is_file():
            capture, = (output / repo / "playback").glob("capture_*")
            playback = json.loads((capture / "postprocessed-playback.json").read_text())
            builds = [p["lake_build_artifact"] for p in playback["points"]
                      if (p.get("lake_build_artifact") or {}).get("status") == "complete"]
            jobs.append(dict(base, phase="after", destination=pair,
                             submitted_archive=capture / "submitted.sources.tar.gz",
                             lake_build=ROOT / builds[-1]["path"] if builds else None,
                             lake_build_sha256=builds[-1]["archive_sha256"] if builds else None))
    return jobs, endpoints


def extract(job: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    repo, phase, destination = job["repository"], job["phase"], Path(job["destination"])
    resources, timeouts = manifest["resources"], manifest["timeouts"]
    record = job["materialization"]
    if "tree_sha256" in record:
        ensure_materialized_image_from_record(repo, record)
    image, image_id = str(record["tag"]), materialized_image_id(str(record["tag"]))
    if "tree_sha256" not in record and image_id != record["image_id"]:
        raise RuntimeError(f"{repo}: {image} is not the evaluated image {record['image_id']}")
    run_args = [f"--ulimit=nofile={resources['nofile_limit']}:{resources['nofile_limit']}",
                f"--cpus={resources['cpus_per_worker']}", f"--env=LEAN_NUM_THREADS={resources['cpus_per_worker']}",
                f"--memory={resources['memory_per_worker']}", f"--memory-swap={resources['memory_per_worker']}"]
    if resources.get("cgroup_parent"):
        run_args.append(f"--cgroup-parent={resources['cgroup_parent']}")
    env = get_environment(dict(
        image=image, cwd="/testbed", timeout=int(timeouts["graph_seconds"]),
        container_timeout=f"{timeouts['container_seconds']}s", run_args=["--rm", *run_args],
        network_policy=resources["network_policy"], container_pids_limit=int(resources["pids_limit"]),
        monitoring_metadata=dict(run_id=manifest["run_id"], role="paper-graphs", model=manifest["model"],
                                 instance_id=repo, worker_id=f"{phase}-{repo}", manifest=manifest["manifest_path"]),
    ), default_type="docker")
    logs = destination / "logs" / phase
    stop = threading.Event()
    monitor = None
    try:
        details = inspect(env.container_id)
        write_json(logs / "container-start.json", details)
        try:
            group = Path("/sys/fs/cgroup") / next(
                line.split(":", 2)[2] for line in Path(f"/proc/{details['State']['Pid']}/cgroup").read_text()
                .splitlines() if line.startswith("0::")).lstrip("/")
        except (KeyError, OSError, StopIteration):
            group = None   # samples then carry only their time

        def sample() -> None:
            with (logs / "resource-samples.jsonl").open("w", buffering=1) as stream:
                while True:
                    stream.write(json.dumps(cgroup_sample(group)) + "\n")
                    if stop.wait(10):
                        break
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        if phase == "after":
            env.restore_source_archive(job["submitted_archive"], ["."], timeout=int(timeouts["restore_seconds"]))
            if env.execute("rm -rf /testbed/.lake/build", timeout=int(timeouts["restore_seconds"])).get("returncode"):
                raise RuntimeError(f"{repo}: could not clear the baseline build")
            if job["lake_build"] is None:
                # No retained build: compile the submission's project once (dependencies stay built).
                build = env.execute(f"cd /testbed && lake build {record['build_target']}",
                                    timeout=int(timeouts["build_seconds"]))
                (logs / "build.log").write_text(str(build.get("output") or ""))
                if build.get("returncode") or build.get("timed_out"):
                    raise RuntimeError(f"{repo}: submitted sources did not build; see {logs / 'build.log'}")
            else:
                if sha256(job["lake_build"]) != job["lake_build_sha256"]:
                    raise ValueError(f"{repo}: retained .lake/build changed")
                with Path(job["lake_build"]).open("rb") as archive:
                    restored = subprocess.run(
                        ["docker", "exec", "-i", env.container_id, "tar", "--no-same-owner", "-C", "/testbed",
                         "-xzf", "-"], stdin=archive, capture_output=True, timeout=int(timeouts["restore_seconds"]))
                if restored.returncode:
                    raise RuntimeError(f"{repo}: could not restore the retained build: {restored.stderr[-2000:]!r}")
        originals = snapshot_sources(env, list_project_lean_files(env))
        modules = _discover_modules(env)
        with (logs / "native-output.log").open("w", buffering=1) as native:
            stream = env.execute_stream

            def capture(command, *, timeout, on_output, capture_output):
                def tee(line):
                    native.write(line)
                    on_output(line)
                return stream(command, timeout=timeout, on_output=tee, capture_output=capture_output)
            env.execute_stream = capture
            declarations, dependencies, _ = olean.read_olean_declarations(
                env, modules=modules, import_roots=[job["entry_module"]],
                timeout=int(timeouts["graph_seconds"]), strict=True)
        if not declarations:
            raise RuntimeError(f"{repo}: empty declaration graph")
        write_json(logs / "kernel-extraction.json", dict(modules=modules, declarations=declarations, dependencies={
            name: sorted(targets) for name, targets in dependencies.items()}))
        paths = strip.resolve_module_paths({d["module"] for d in declarations}, list(originals))
        missing = {d["module"] for d in declarations} - set(paths)
        if missing:
            raise RuntimeError(f"{repo}: no source for modules {sorted(missing)}")
        evidence, layers = static_layers(env, declarations, dependencies, originals, paths,
                                         set(modules) | {d["module"] for d in declarations})
        graph = build_graph_artifact(
            repository=repo, source_image=image, source_image_id=image_id, declarations=declarations,
            kernel_edges=set(layers["kernel"]), ilean_edges=set(layers["ilean"]),
            source_text_edges=set(layers["source_text"]), reports={}, originals=originals, module_paths=paths)
        graph.update(static_dependency_evidence=evidence, entry_module=job["entry_module"],
                     extraction_method=dict(path=str(EXTRACTOR.relative_to(ROOT)), sha256=EXTRACTOR_SHA256))
        write_graph_artifact(destination / f"{phase}.json", graph)
        _write_source_archive(destination / phase / "original-sources.tar.gz",
                              {path.removeprefix("/testbed/"): text.encode("utf-8", errors="surrogateescape")
                               for path, text in originals.items()})
        return dict(repository=repo, phase=phase, status="complete", counts=graph["counts"])
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=15)
        try:
            write_json(logs / "container-final.json", inspect(env.container_id))
        finally:
            env.cleanup()


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def write_inputs(manifest: Mapping[str, Any], endpoints: list[dict[str, Any]]) -> Path:
    """Pair each verified submission with its baseline; check both against the scored tokens."""
    output = ROOT / manifest["outputs"]["directory"]
    rows = []
    for endpoint in endpoints:
        repo = endpoint["repository"]
        row = dict(model=manifest["model"], repository=repo, status=endpoint["status"],
                   baseline_tokens=endpoint["baseline_tokens"], post_tokens=endpoint["post_tokens"],
                   graph_directory="", record_path="", program_cache="")
        if endpoint["status"] == "passed":
            pair = output / "repositories" / repo
            baseline = ROOT / manifest["baseline_graphs"] / repo
            link(baseline / "before.json", pair / "before.json")
            link(baseline / "before" / "original-sources.tar.gz", pair / "before" / "original-sources.tar.gz")
            scope = endpoint["metric_scope"]
            totals = []
            for phase in ("before", "after"):
                sources = SourceIndex.from_archive(pair / phase / "original-sources.tar.gz")
                sources.restrict(exclude_files=scope["excluded_files"], exclude_dirs=scope["exclude_dirs"],
                                 include_prefix=scope["include_prefix"])
                totals.append(sources.total_tokens())
            if totals != [endpoint["baseline_tokens"], endpoint["post_tokens"]]:
                raise ValueError(f"{repo}: graph sources ({totals}) disagree with the scored tokens")
            write_json(pair / "classification-record.json", dict(metric_scope=scope, reconciliation=dict(
                scored_baseline_tokens=endpoint["baseline_tokens"], scored_post_tokens=endpoint["post_tokens"],
                scored_tokens_saved=endpoint["baseline_tokens"] - endpoint["post_tokens"])))
            row.update(graph_directory=str(pair.relative_to(ROOT)),
                       record_path=str((pair / "classification-record.json").relative_to(ROOT)),
                       program_cache=str((output / "program_sites").relative_to(ROOT)))
        rows.append(row)
    index = output / "classification-inputs.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return index


def run(manifest_path: Path) -> Path:
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("kind") != "leanlean_paper_graphs" or manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: not a paper-graph manifest")
    monitoring = manifest["monitoring"]
    if monitoring.get("snapshots") is not True or monitoring.get("every_n_edits") != 1:
        raise ValueError("monitoring.snapshots and every_n_edits: 1 are required")
    if sha256(EXTRACTOR) != EXTRACTOR_SHA256:
        raise ValueError(f"{EXTRACTOR} differs from the paper's frozen extractor")
    # The pinned reader runs whatever program this names; use the paper's.
    olean._DUMP_KEEP_LEAN = EXTRACTOR.read_text()
    manifest["manifest_path"] = str(manifest_path.resolve().relative_to(ROOT))
    output = ROOT / manifest["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved-manifest.json", manifest)
    jobs, endpoints = resolve_jobs(manifest)
    results, failures = [], []
    columns = (SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
               TimeElapsedColumn())
    with Progress(*columns) as progress, ThreadPoolExecutor(int(manifest["resources"]["workers"])) as pool:
        task = progress.add_task("Paper dependency graphs", total=len(jobs))
        futures = {pool.submit(extract, job, manifest): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - recorded, then the run fails
                LOG.exception("%s %s failed", job["phase"], job["repository"])
                failures.append(dict(repository=job["repository"], phase=job["phase"], error=str(error)))
            write_json(output / "progress.json", dict(complete=len(results), failed=len(failures), total=len(jobs),
                                                      failures=failures))
            progress.advance(task)
    if failures:
        raise RuntimeError(f"{len(failures)} graph extractions failed; see {output / 'progress.json'}")
    index = write_inputs(manifest, endpoints)
    print(index.relative_to(ROOT))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parser.parse_args().manifest)


if __name__ == "__main__":
    main()
