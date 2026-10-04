"""Preprocessing: strip each benchmark repository with the pinned lean-strip.

For every repository of the configuration, in order:

1. Verify its registry provenance and download the hash-pinned upstream archive.
2. Build a source image: the Lean toolchain, the project exactly as archived,
   and its fetched Lake dependencies (network allowed only here).
3. Run lean-strip on that project in a container without network access. It
   builds the project, isolates the solution/challenge import closure, strips
   it and checks the challenge.
   With --leanlean-benchmark it writes the published benchmark's layout: its
   lakefile roots and non-Lean files, without the challenge and Comparator
   configuration (evaluation-only inputs kept in the Comparator bundle).
4. Publish the stripped tree, warm-built and committed as the repository's image.
5. Run the official Comparator against that image.

The published dataset has the layout evaluation and postprocessing load.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from leanlean.metrics.tokens import measure_repository_lean_tokens  # noqa: E402
from leanlean.preprocessing.final_verification import (  # noqa: E402
    comparator_environment,
    image_id,
    image_labels,
    select_declared_verifier,
)
from lean_strip._pipeline.preprocessing.dependency_graph import METHOD  # noqa: E402
from lean_strip._pipeline.preprocessing.graph_artifact import build_graph_artifact  # noqa: E402

from leanlean.preprocessing.graph_artifact import graph_sha256  # noqa: E402
from leanlean.preprocessing.lean_strip_tool import (  # noqa: E402
    CONTAINER_PYTHON_PREFIX,
    CONTAINER_ROOT,
    CONTAINER_SITE,
    LeanStripTool,
    resolve_lean_strip_tool,
)
from leanlean.preprocessing.standardized_repositories import (  # noqa: E402
    LEAN_STRIP_SOURCE_CONTRACT,
    STANDARDIZED_DATABASE_KIND,
    STANDARDIZED_DATABASE_SCHEMA_VERSION,
    file_sha256,
    load_standardized_repository_database,
    source_tree_sha256,
    source_tree_stats,
)
from leanlean.preprocessing.palomar_sources import (  # noqa: E402
    PreparedTarget,
    _prepare_target,
    _write_comparator_bundle,
)

CONFIG_KIND = "leanlean_preprocessing"
CONFIG_SCHEMA_VERSION = 3
REPORT_KIND = "leanlean_lean_strip_preprocessing_report"
DATASET_KIND = "leanlean_dataset"
DATASET_SCHEMA_VERSION = 2
IMAGE_MANIFEST_KIND = "leanlean_docker_image"
DOCKERFILE = REPO_ROOT / "docker/preprocessing/Dockerfile"
LABEL_PREFIX = "org.openai.leanlean"
CONFIG_KEYS = {
    "kind", "schema_version", "run_id", "output", "dataset", "registry",
    "toolchains", "repositories", "lean_strip", "workers", "container",
    "timeouts", "keep_source_images", "lakefile_root_repairs", "asset_rule_before_cleanup",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _repo_path(value: Any, name: str) -> Path:
    path = (REPO_ROOT / _text(value, name)).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{name} escapes the repository")
    return path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _docker(*args: str, timeout: float | None = None, check: bool = True,
            input: bytes | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=input)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise RuntimeError(f"docker {args[0]} failed: {detail[-2000:]}")
    return result


@dataclass(frozen=True)
class Config:
    path: Path
    raw: Mapping[str, Any]
    run_id: str
    output: Path
    work: Path
    mapping_path: Path
    mapping_sha256: str
    mapping: Mapping[str, Any]
    jobs: int
    strip_timeout: int
    workers: int
    container: Mapping[str, Any]
    timeouts: Mapping[str, Any]
    keep_source_images: bool
    lakefile_repairs: frozenset[str]
    asset_rule_before_cleanup: frozenset[str]

    @property
    def safe_run(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", self.run_id)


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text())
    raw = _mapping(raw, str(path))
    if set(raw) - CONFIG_KEYS or {"kind", "schema_version", "run_id", "output"} - set(raw):
        raise ValueError(f"{path}: keys must be a subset of {sorted(CONFIG_KEYS)}")
    if raw.get("kind") != CONFIG_KIND or raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"{path}: expected kind {CONFIG_KIND} schema_version {CONFIG_SCHEMA_VERSION}")
    run_id = _text(raw["run_id"], "run_id")
    output = _repo_path(raw["output"], "output")
    if output.parent != (REPO_ROOT / "datasets").resolve() or output.name != run_id:
        raise ValueError("output must be datasets/<run_id>")
    registry = _mapping(raw.get("registry"), "registry")
    mapping_path = _repo_path(registry.get("mapping"), "registry.mapping")
    mapping_sha256 = _text(registry.get("sha256"), "registry.sha256")
    if file_sha256(mapping_path) != mapping_sha256:
        raise ValueError("registry mapping hash drift")
    mapping = json.loads(mapping_path.read_text())
    if mapping.get("kind") != "palomar_candidate_mapping" or mapping.get("schema_version") != 3:
        raise ValueError("registry.mapping must be a schema-3 Palomar candidate mapping")
    repositories = raw.get("repositories")
    if (not isinstance(repositories, list) or not repositories
            or len(set(repositories)) != len(repositories)):
        raise ValueError("repositories must be a non-empty unique list")
    targets = {target["benchmark_id"]: target for target in mapping["targets"]}
    unknown = sorted(set(repositories) - set(targets))
    if unknown:
        raise ValueError(f"repositories absent from the registry mapping: {unknown}")
    for toolchain, asset in _mapping(raw.get("toolchains"), "toolchains").items():
        version = str(toolchain).removeprefix("leanprover/lean4:v")
        expected = ("https://github.com/leanprover/lean4/releases/download/"
                    f"v{version}/lean-{version}-linux.tar.zst")
        if _mapping(asset, toolchain).get("url") != expected or not re.fullmatch(
            r"[0-9a-f]{64}", str(asset.get("sha256"))
        ):
            raise ValueError(f"toolchains.{toolchain}: unexpected asset pin")
    lean_strip = _mapping(raw.get("lean_strip"), "lean_strip")
    if set(lean_strip) != {"version", "commit", "jobs", "timeout_seconds"}:
        raise ValueError("lean_strip must pin version, commit, jobs and timeout_seconds")
    repairs = raw.get("lakefile_root_repairs", [])
    if not isinstance(repairs, list) or not set(repairs) <= set(repositories):
        raise ValueError("lakefile_root_repairs must list configured repositories")
    before_cleanup = raw.get("asset_rule_before_cleanup", [])
    if not isinstance(before_cleanup, list) or not set(before_cleanup) <= set(repositories):
        raise ValueError("asset_rule_before_cleanup must list configured repositories")
    container = _mapping(raw.get("container"), "container")
    if not {"cpus", "memory", "pids_limit", "nofile_limit"} <= set(container):
        raise ValueError("container must set cpus, memory, pids_limit and nofile_limit")
    return Config(
        path=path.resolve(),
        raw=raw,
        run_id=run_id,
        output=output,
        work=(REPO_ROOT / "runs/preprocessing" / run_id).resolve(),
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha256,
        mapping={**mapping, "targets": [targets[item] for item in repositories]},
        jobs=int(lean_strip["jobs"]),
        strip_timeout=int(lean_strip["timeout_seconds"]),
        workers=int(raw.get("workers", 1)),
        container=container,
        timeouts=_mapping(raw.get("timeouts"), "timeouts"),
        keep_source_images=bool(raw.get("keep_source_images", False)),
        lakefile_repairs=frozenset(repairs),
        asset_rule_before_cleanup=frozenset(before_cleanup),
    )


def check_tool_pin(config: Config, tool: LeanStripTool) -> None:
    pin = config.raw["lean_strip"]
    if pin["version"] != tool.version or pin["commit"] != tool.commit:
        raise RuntimeError(
            f"{config.path}: pins lean-strip {pin['version']} at {pin['commit']}, "
            f"but the installed engine is {tool.version} at {tool.commit}"
        )


def prepare_targets(config: Config) -> list[PreparedTarget]:
    registry = config.raw["registry"]
    archive_cache = registry.get("archive_cache")
    archive_dir = _repo_path(archive_cache, "registry.archive_cache") if archive_cache else config.work / "archives"
    results = {row["palomar_id"]: row for row in config.mapping["results"]}
    return [
        _prepare_target(
            target,
            results,
            toolchain_assets=config.raw["toolchains"],
            registry_url=_text(registry.get("url"), "registry.url"),
            record_dir=config.work / "records",
            record_timeout=int(config.timeouts.get("record_download_seconds", 90)),
            archive_dir=archive_dir,
            archive_timeout=int(config.timeouts.get("archive_download_seconds", 180)),
            image_tag_prefix="leanlean-source",
            mapping_sha256=config.mapping_sha256,
            source_run_id=config.run_id,
        )
        for target in config.mapping["targets"]
    ]


def _archive_member(target: PreparedTarget) -> tuple[str, int]:
    with tarfile.open(target.archive_path) as bundle:
        top = bundle.getnames()[0].split("/")[0]
    member = PurePosixPath(top, target.project_root) if target.project_root else PurePosixPath(top)
    return member.as_posix(), len(member.parts)


def _relative_to_project(target: PreparedTarget, registered: str) -> str:
    return PurePosixPath(registered).relative_to(target.project_root or ".").as_posix()


def build_source_image(config: Config, target: PreparedTarget) -> dict[str, str]:
    tag = f"leanlean-{target.instance_id}-source:{config.safe_run}"
    member, strip_components = _archive_member(target)
    log = config.work / "logs" / f"{target.instance_id}.source-image.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="leanlean-source-") as context:
        shutil.copyfile(target.archive_path, Path(context) / "source.tar.gz")
        shutil.copyfile(DOCKERFILE, Path(context) / "Dockerfile")
        arguments = {
            "LEAN_TOOLCHAIN_URL": target.toolchain_url,
            "LEAN_TOOLCHAIN_SHA256": target.toolchain_sha256,
            "ARCHIVE_MEMBER": member,
            "ARCHIVE_STRIP_COMPONENTS": str(strip_components),
            "REPO_COMMIT": target.commit,
            "INSTANCE_ID": target.instance_id,
            "SOURCE_RUN_ID": config.run_id,
        }
        command = ["docker", "build", "--pull=false", "-t", tag,
                   f"--cpu-quota={int(config.container['cpus']) * 100000}",
                   f"--memory={config.container['memory']}"]
        for key, value in arguments.items():
            command += ["--build-arg", f"{key}={value}"]
        with log.open("wb") as handle:
            result = subprocess.run(command + [context], stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=int(config.timeouts.get("source_image_seconds", 14400)))
    if result.returncode != 0:
        raise RuntimeError(f"{target.instance_id}: source image build failed (see {_relative(log)})")
    return {"tag": tag, "image_id": image_id(tag)}


def _container_run_args(config: Config, name: str, image: str, instance_id: str, role: str) -> list[str]:
    container = config.container
    nofile = int(container["nofile_limit"])
    args = [
        "run", "-d", "--name", name, "--network", "none", "-w", "/testbed",
        f"--cpus={container['cpus']}", f"--memory={container['memory']}",
        f"--memory-swap={container['memory']}", f"--pids-limit={container['pids_limit']}",
        f"--ulimit=nofile={nofile}:{nofile}",
        "--label", f"{LABEL_PREFIX}.run_id={config.run_id}",
        "--label", f"{LABEL_PREFIX}.instance_id={instance_id}",
        "--label", f"{LABEL_PREFIX}.role={role}",
    ]
    if container.get("cgroup_parent"):
        args.append(f"--cgroup-parent={container['cgroup_parent']}")
    return args + [image, "sleep", "infinity"]


def _exec(name: str, script: str, *, timeout: float | None = None) -> subprocess.CompletedProcess:
    return _docker("exec", name, "bash", "-c", script, timeout=timeout, check=False)


def _copy_out(name: str, source: str, destination: Path, *, exclude: tuple[str, ...] = (),
              timeout: int = 1800, attempts: int = 3) -> None:
    """Copy a container directory out through a tar file.

    A `docker exec` client occasionally never returns after its command exits,
    so each attempt is bounded and retried before anything is extracted.
    """

    destination.mkdir(parents=True, exist_ok=True)
    command = ["docker", "exec", name, "tar", "-C", source,
               *(f"--exclude=./{path}" for path in exclude), "-cf", "-", "."]
    for attempt in range(1, attempts + 1):
        with tempfile.TemporaryFile() as archive:
            try:
                copied = subprocess.run(command, stdout=archive, stderr=subprocess.PIPE, timeout=timeout)
            except subprocess.TimeoutExpired:
                if attempt == attempts:
                    raise RuntimeError(f"copying {source} out of {name} timed out {attempts} times")
                continue
            if copied.returncode != 0:
                raise RuntimeError(f"could not copy {source} out of {name}: "
                                   f"{copied.stderr.decode(errors='replace')[-1000:]}")
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:") as bundle:
                bundle.extractall(destination, filter="tar")
            return


def _copy_in(name: str, source: Path, destination: str) -> None:
    """Copy a host directory into the container, owned by the container user.

    `docker cp` keeps host ownership, which a rootless daemon cannot map.
    """

    _docker("exec", name, "mkdir", "-p", destination)
    pack = subprocess.Popen(["tar", "-C", str(source), "--exclude=__pycache__", "-cf", "-", "."],
                            stdout=subprocess.PIPE)
    unpack = subprocess.run(["docker", "exec", "-i", name, "tar", "-xf", "-", "--no-same-owner",
                             "-C", destination], stdin=pack.stdout, capture_output=True)
    pack.stdout.close()
    if pack.wait() != 0 or unpack.returncode != 0:
        raise RuntimeError(f"could not copy {source} into {name}: "
                           f"{unpack.stderr.decode(errors='replace')[-1000:]}")


def _stage_tool(name: str, tool: LeanStripTool) -> None:
    _copy_in(name, tool.python_prefix, CONTAINER_PYTHON_PREFIX)
    _copy_in(name, tool.package_dir, f"{CONTAINER_SITE}/lean_strip")
    _copy_in(name, tool.dist_info_dir, f"{CONTAINER_SITE}/{tool.dist_info_dir.name}")
    probe = _exec(name, f"PYTHONPATH={CONTAINER_SITE} {tool.container_python} -m lean_strip --version")
    if probe.returncode != 0 or tool.version not in probe.stdout.decode():
        raise RuntimeError(f"lean-strip does not run in the strip container: "
                           f"{(probe.stdout + probe.stderr).decode(errors='replace')[-1000:]}")


def strip_repository(config: Config, tool: LeanStripTool, target: PreparedTarget,
                     source: Mapping[str, str], repository_dir: Path) -> dict[str, Any]:
    """Run lean-strip offline; commit and export the stripped repository."""

    name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"leanlean-strip-{config.safe_run}-{target.instance_id}")[:120]
    _docker("rm", "-f", name, check=False)
    _docker(*_container_run_args(config, name, source["tag"], target.instance_id, "lean-strip"))
    evidence = repository_dir / "lean-strip"
    comparator_rel = _relative_to_project(target, target.registered_comparator_config_path)
    try:
        _stage_tool(name, tool)
        started = time.time()
        layout = ["--leanlean-benchmark"]
        if target.instance_id in config.lakefile_repairs:
            layout.append("--lake-root-repair")
        if target.instance_id in config.asset_rule_before_cleanup:
            layout += ["--asset-rule", "before-cleanup"]
        run = _exec(
            name,
            f"cd /testbed && PYTHONPATH={CONTAINER_SITE} {tool.container_python} -m lean_strip . "
            f"--comparator '{comparator_rel}' {' '.join(layout)} --jobs {config.jobs} "
            f"--timeout {config.strip_timeout}",
            timeout=int(config.timeouts.get("strip_seconds", 43200)),
        )
        seconds = round(time.time() - started, 1)
        (evidence).mkdir(parents=True, exist_ok=True)
        (evidence / "stdout.txt").write_bytes(run.stdout + run.stderr)
        state = evidence / "state"
        # Keep the report, log and diagnostics; not the backups or scratch workspace.
        _copy_out(name, "/testbed/.lean-strip", state, exclude=("original", "work", "tmp"))
        report = json.loads((state / "report.json").read_text()) if (state / "report.json").is_file() else {}
        status = report.get("status")
        if run.returncode != 0 or status not in {"stripped", "discarded"}:
            raise RuntimeError(
                f"{target.instance_id}: lean-strip {status or 'failed'} "
                f"(stage {report.get('stage')}): {report.get('error') or 'see lean-strip/stdout.txt'}"
            )
        result: dict[str, Any] = {"status": status, "report": report, "seconds": seconds}
        if status == "discarded":
            return result
        # lean-strip wrote the benchmark layout; drop its state and the upstream history.
        pruned = _exec(name, " && ".join([
            "cd /testbed",
            "rm -rf .lean-strip .git",
            "find . -mindepth 1 -depth -type d -empty -not -path './.lake' -not -path './.lake/*' -delete",
            "printf '.lake/\\n' > .gitignore",
        ]))
        if pruned.returncode != 0:
            raise RuntimeError(f"{target.instance_id}: could not finalize the stripped tree: "
                               f"{pruned.stderr.decode(errors='replace')[-1000:]}")
        committed = _exec(name, " && ".join([
            "cd /testbed",
            "git init -q",
            "git config user.name 'A U Thor'",
            "git config user.email 'author@example.com'",
            "git add .",
            "git commit -qm init",
            "git rev-parse HEAD > /.init_commit",
        ]))
        if committed.returncode != 0:
            raise RuntimeError(f"{target.instance_id}: could not commit the stripped tree: "
                               f"{committed.stderr.decode(errors='replace')[-1000:]}")
        # lean-strip builds the stripped tree in its scratch workspace; build the
        # published tree too, so the image carries a warm build of exactly it.
        warm = _exec(name, f"cd /testbed && LEAN_NUM_THREADS={config.jobs} lake build +{target.entry_module}",
                     timeout=int(config.timeouts.get("strip_seconds", 43200)))
        (evidence / "warm-build.log").write_bytes(warm.stdout + warm.stderr)
        if warm.returncode != 0:
            raise RuntimeError(f"{target.instance_id}: the published stripped tree does not build "
                               "(see lean-strip/warm-build.log)")
        _exec(name, f"rm -rf {CONTAINER_ROOT}")
        tag = f"leanlean-{target.instance_id}-stripped:{config.safe_run}"
        labels = {
            "instance_id": target.instance_id,
            "repo_variant": "stripped",
            "run_id": config.run_id,
            "source_commit": target.commit,
            "lean_strip_version": tool.version,
            "lean_strip_commit": tool.commit,
        }
        commit_args = ["commit"]
        for key, value in labels.items():
            commit_args += ["--change", f"LABEL {LABEL_PREFIX}.{key}={value}"]
        _docker(*commit_args, "--change", 'CMD ["sleep", "infinity"]', name, tag)
        stripped = repository_dir / "stripped"
        shutil.rmtree(stripped, ignore_errors=True)
        _copy_out(name, "/testbed", stripped, exclude=(".lake", ".git"))
        result.update({"image": {"tag": tag, "image_id": image_id(tag)}, "lean_strip_flags": layout})
        return result
    finally:
        _docker("rm", "-f", name, check=False)


def database_row(config: Config, target: PreparedTarget, comparator: Mapping[str, str],
                 tool: LeanStripTool) -> dict[str, Any]:
    archive = {"path": _relative(target.archive_path), "sha256": target.archive_sha256,
               "repository": target.repository, "commit": target.commit}
    registered = {
        "registered_solution_path": target.registered_solution_path,
        "registered_solution_sha256": target.solution_sha256,
        "solution_source": target.solution_source,
        "registered_challenge_path": target.registered_challenge_path,
        "registered_challenge_sha256": target.challenge_sha256,
        "registered_challenge_module": target.challenge_module,
        "registered_comparator_config_path": target.registered_comparator_config_path,
        "registered_comparator_config_sha256": target.comparator_config_sha256,
    }
    return {
        "instance_id": target.instance_id,
        "source": {
            "repository_url": f"https://github.com/{target.repository}",
            "commit": target.commit,
            "toolchain": target.toolchain,
            "standardization": {
                "contract": LEAN_STRIP_SOURCE_CONTRACT,
                "adapter": "palomar_registry",
                "clean_build_passed": True,
                "source_run_id": config.run_id,
                "source_manifest": _relative(config.path),
                "project_root": target.project_root or None,
                "entry_module": target.entry_module,
                "source_archive": archive,
                "lean_strip": tool.record(),
                "comparator": dict(comparator),
                **registered,
            },
        },
        "scopes": {"entry_module": target.entry_module, "target_dir": "",
                   "build_target": f"+{target.entry_module}", "exclude_dirs": []},
        "protected": {
            "declarations": list(target.protected_declarations),
            "provenance": {
                "kind": "palomar_registry_main_results",
                "policy": "registry_exact_names_without_signature_scout",
                "registry": _relative(config.mapping_path),
                "registry_sha256": config.mapping_sha256,
                "registry_record": {"path": _relative(target.registry_record_path),
                                    "sha256": target.registry_record_sha256},
                "source_archive": archive,
                "comparator": dict(comparator),
                "source_manifest": _relative(config.path),
                "palomar_ids": list(target.palomar_ids),
                "repository_roles": list(target.repository_roles),
                "registry_toolchains": list(target.registry_toolchains),
                "source_toolchain": target.toolchain,
                "source_toolchain_url": target.toolchain_url,
                "source_toolchain_sha256": target.toolchain_sha256,
                "solution_module": target.entry_module,
                **registered,
                "original_registered_declarations": list(target.original_registered_declarations),
                "override": None,
                "resolution_policy": "registry_solution_path_and_comparator_module_exact_fail_closed",
                "admission_audit": "Lean.collectAxioms_reject_sorryAx",
            },
        },
    }


def verify_with_comparator(config: Config, row: Mapping[str, Any], image: Mapping[str, str]) -> dict[str, Any]:
    timeout = int(config.timeouts.get("comparator_seconds", 21600))
    labels = image_labels(image["tag"])
    expected = {f"{LABEL_PREFIX}.instance_id": row["instance_id"],
                f"{LABEL_PREFIX}.repo_variant": "stripped",
                f"{LABEL_PREFIX}.run_id": config.run_id}
    label_checks = {key: labels.get(key) == value for key, value in expected.items()}
    verifier = select_declared_verifier(row)
    if verifier is None:
        raise ValueError(f"{row['instance_id']}: no Comparator contract declared")
    environment = comparator_environment(image["tag"], config.container, timeout)
    try:
        declared = verifier.verify(environment, row, build_jobs=int(config.container["cpus"]),
                                   timeout_seconds=timeout)
    finally:
        environment.cleanup()
    checks = {
        "stripped_image": image["image_id"].startswith("sha256:"),
        "stripped_image_labels": all(label_checks.values()),
        "declared_source_verifier": declared.get("passed") is True,
    }
    passed = all(checks.values())
    return {"passed": passed, "status": "passed" if passed else "failed", "checks": checks,
            "image": image["tag"], "image_id": image["image_id"],
            "image_label_checks": label_checks, "declared_verifier": declared}


def publish_graph(repository_dir: Path, instance_id: str, source: Mapping[str, str]) -> dict[str, Any]:
    """Relabel lean-strip's graph with the repository identity; the structure is unchanged.

    lean-strip discards a repository whose entry module has no project
    declarations before collecting a graph; it is published with an empty one.
    """

    path = repository_dir / "lean-strip/state/diagnostics/dependency-graph.json"
    if path.is_file():
        graph = json.loads(path.read_text())
    else:
        graph = build_graph_artifact(
            repository=instance_id, source_image=source["tag"], source_image_id=source["image_id"],
            declarations=[], kernel_edges=set(), ilean_edges=set(), source_text_edges=set(),
            reports={}, originals={}, module_paths={},
        )
        graph["graph_method"] = METHOD
    graph.update({"repository": instance_id, "source_image": source["tag"],
                  "source_image_id": source["image_id"]})
    graph["sha256"] = graph_sha256(graph)
    _atomic_json(repository_dir / "dependency-graph.json", graph)
    return {"schema": graph["schema"], "path": "dependency-graph.json",
            "sha256": graph["sha256"], "counts": graph["counts"]}


def _metric(phase: Mapping[str, Any], key: str) -> dict[str, Any]:
    values = phase[key]
    return {"raw": values["before"], "stripped": values["after"], "reduction": values["reduction"]}


def process_repository(config: Config, tool: LeanStripTool, target: PreparedTarget) -> dict[str, Any]:
    instance_id = target.instance_id
    checkpoint = config.work / "state" / f"{instance_id}.json"
    if checkpoint.is_file():
        previous = json.loads(checkpoint.read_text())
        # Reuse only results produced by the pinned engine.
        if previous.get("passed") is True and previous.get("lean_strip") == tool.record():
            return previous
    repository_dir = config.output / "repos" / instance_id
    shutil.rmtree(repository_dir, ignore_errors=True)
    repository_dir.mkdir(parents=True)
    comparator = _write_comparator_bundle(target, repository_dir / "comparator")
    row = database_row(config, target, comparator, tool)
    source = build_source_image(config, target)
    try:
        stripped = strip_repository(config, tool, target, source, repository_dir)
    finally:
        if not config.keep_source_images:
            _docker("image", "rm", source["tag"], check=False)
    engine = stripped["report"]
    graph = publish_graph(repository_dir, instance_id, source)
    report: dict[str, Any] = {
        "kind": REPORT_KIND,
        "schema_version": 1,
        "instance_id": instance_id,
        "run_id": config.run_id,
        "lean_strip": tool.record(),
        "lean_strip_report": engine,
        "lean_strip_seconds": stripped["seconds"],
        "source_image": source,
        "source_repository_url": row["source"]["repository_url"],
        "source_commit": target.commit,
        "source_toolchain": target.toolchain,
        "protected_provenance": row["protected"]["provenance"],
        "dependency_graph": graph,
        "discarded": stripped["status"] == "discarded",
    }
    if report["discarded"]:
        report["total_decls"] = 0
        _atomic_json(repository_dir / "prod-strip-report.json", report)
        result = {"passed": True, "discarded": True, "row": row, "lean_strip": tool.record()}
        _atomic_json(checkpoint, result)
        return result

    image = stripped["image"]
    verification = verify_with_comparator(config, row, image)
    if not verification["passed"]:
        _atomic_json(repository_dir / "final-verification.json", verification)
        raise RuntimeError(f"{instance_id}: the Comparator rejected the stripped repository")
    stripped_dir = repository_dir / "stripped"
    tree = {"tree_sha256": source_tree_sha256(stripped_dir), **source_tree_stats(stripped_dir)}
    tokens = measure_repository_lean_tokens(stripped_dir, exclude_dirs=[], include_prefix="")
    phase = engine["phases"]["stripping"]
    if tokens != phase["tokens"]["after"]:
        raise RuntimeError(f"{instance_id}: published tree has {tokens} Lean tokens, "
                           f"lean-strip reported {phase['tokens']['after']}")
    image_manifest = {
        "kind": IMAGE_MANIFEST_KIND, "schema_version": 1, "repository": instance_id,
        "variant": "stripped", "tag": image["tag"], "image_id": image["image_id"],
        "source_run_id": config.run_id, "labels": image_labels(image["tag"]),
    }
    _atomic_json(repository_dir / "docker-image.json", image_manifest)
    declared = verification["declared_verifier"]
    # The engine report carries the benchmark's declaration accounting; the CLI's
    # summary view leaves out generated helpers such as `.elim` and `.noConfusion`.
    strip_report = engine["engine_report"]
    report.update({
        "total_decls": strip_report["total_decls"],
        "keep_decls": strip_report["keep_decls"],
        "dropped_decls": strip_report["dropped_decls"],
        "lean_strip_flags": stripped["lean_strip_flags"],
        "final_verification": verification,
        "published_repository": {
            "id": instance_id,
            "source": {"repository": row["source"]["repository_url"], "commit": target.commit,
                       "toolchain": target.toolchain, "adapter": "palomar_registry"},
            "scope": dict(row["scopes"]),
            "protected": {
                "count": len(target.protected_declarations),
                "declarations_sha256": hashlib.sha256(
                    "\n".join(sorted(target.protected_declarations)).encode()).hexdigest(),
                "provenance": "source_database",
            },
            "variants": {
                "stripped": {
                    "format": "stripped_source_repository",
                    "cache_tree": "stripped",
                    **tree,
                    "lean_tokens": tokens,
                    "image": image["tag"],
                    "image_id": image["image_id"],
                    "image_manifest": "docker-image.json",
                    "image_manifest_sha256": file_sha256(repository_dir / "docker-image.json"),
                    "final_verification": verification,
                    "task_metadata": {
                        "validation_command": declared["command"],
                        "validation_engine": declared["engine"],
                        "pre_agent_validation": {
                            "command": declared["command"], "required_exit_code": 0,
                            "preprocessing_gate_passed": True,
                            "contract_sha256": declared.get("contract_sha256"),
                        },
                    },
                },
            },
            "preprocessing": {
                "version": f"lean-strip {tool.version}",
                "engine": tool.record(),
                "grind_source_form": "original_grind_calls",
                "report": "prod-strip-report.json",
                "source_run_id": config.run_id,
                "dependency_graph": graph,
            },
            "metrics": {
                # "raw" is the solution/challenge import closure lean-strip isolated,
                # the baseline of the benchmark's preprocessing statistics.
                "lean_tokens": _metric(phase, "tokens"),
                "words": _metric(phase, "words"),
                "declarations": {"raw": strip_report["total_decls"], "stripped": strip_report["keep_decls"],
                                 "dropped": len(strip_report["dropped_decls"])},
                "preprocessing_seconds": stripped["seconds"],
            },
        },
    })
    _atomic_json(repository_dir / "prod-strip-report.json", report)
    result = {"passed": True, "discarded": False, "row": row, "lean_strip": tool.record()}
    _atomic_json(checkpoint, result)
    return result


def publish(config: Config, targets: list[PreparedTarget], results: Mapping[str, Mapping[str, Any]]) -> Path:
    dataset = _mapping(config.raw.get("dataset"), "dataset")
    database = {
        "kind": STANDARDIZED_DATABASE_KIND,
        "schema_version": STANDARDIZED_DATABASE_SCHEMA_VERSION,
        "dataset_id": _text(dataset.get("id"), "dataset.id"),
        "dataset_version": _text(dataset.get("version"), "dataset.version"),
        "repositories": [results[target.instance_id]["row"] for target in targets],
    }
    database_path = config.output / "repository-database.json"
    _atomic_json(database_path, database)
    loaded = load_standardized_repository_database(database_path, repo_root=REPO_ROOT)
    shutil.copyfile(config.path, config.output / "config.snapshot.yaml")

    def entry(instance_id: str) -> dict[str, str]:
        report = f"repos/{instance_id}/prod-strip-report.json"
        return {"id": instance_id, "report": report,
                "report_sha256": file_sha256(config.output / report)}

    kept = [t.instance_id for t in targets if not results[t.instance_id]["discarded"]]
    discarded = [t.instance_id for t in targets if results[t.instance_id]["discarded"]]
    manifest = {
        "kind": DATASET_KIND,
        "schema_version": DATASET_SCHEMA_VERSION,
        "id": database["dataset_id"],
        "version": database["dataset_version"],
        "default_variant": "stripped",
        "repository_count": len(kept),
        "repositories": [entry(item) for item in kept],
        "discarded_repositories": [entry(item) for item in discarded],
        "repository_database": {"path": "repository-database.json", "sha256": loaded.sha256,
                                "definition_sha256": loaded.definition_sha256},
        "preprocessing": {
            "run_id": config.run_id,
            "engine": resolve_lean_strip_tool().record(),
            "config": {"path": "config.snapshot.yaml",
                       "sha256": file_sha256(config.output / "config.snapshot.yaml")},
        },
    }
    dataset_path = config.output / "dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    from leanlean.dataset_bundle import load_dataset

    load_dataset(dataset_path)
    return dataset_path


def run(config_path: Path, *, only: list[str] | None = None) -> bool:
    config = load_config(config_path)
    if (config.output / ".leanlean-materialized.json").exists():
        raise SystemExit(f"{_relative(config.output)} holds the Hugging Face release "
                         "(scripts/fetch_dataset.py); move it aside before preprocessing into it")
    tool = resolve_lean_strip_tool()
    check_tool_pin(config, tool)
    print(f"{config.run_id}: lean-strip {tool.version} ({tool.commit[:12]}), "
          f"{len(config.mapping['targets'])} repositories -> {_relative(config.output)}", flush=True)
    targets = prepare_targets(config)
    selected = [t for t in targets if not only or t.instance_id in only]
    results: dict[str, Mapping[str, Any]] = {}
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(process_repository, config, tool, t): t for t in selected}
        for future in concurrent.futures.as_completed(futures):
            instance_id = futures[future].instance_id
            try:
                results[instance_id] = future.result()
                verdict = "discarded" if results[instance_id]["discarded"] else "stripped"
            except Exception as error:  # keep the other repositories going
                failures[instance_id] = f"{type(error).__name__}: {error}"
                verdict = f"FAILED {failures[instance_id]}"
            print(f"[{time.strftime('%H:%M:%S')}] {instance_id}: {verdict}", flush=True)
    _atomic_json(config.work / "report.json",
                 {"run_id": config.run_id, "lean_strip": tool.record(),
                  "succeeded": sorted(results), "failed": failures})
    if failures:
        print(f"{len(failures)} repositories failed; rerun to retry them", flush=True)
        return False
    if not only:
        dataset = publish(config, targets, results)
        print(f"published {_relative(dataset)}", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--only", action="append", default=[],
                        help="process only this repository (repeatable); does not publish")
    args = parser.parse_args(argv)
    return 0 if run(args.config, only=args.only) else 1


if __name__ == "__main__":
    raise SystemExit(main())
