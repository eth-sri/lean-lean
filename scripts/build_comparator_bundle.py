#!/usr/bin/env python3
"""Build the pinned Palomar Comparator tool bundle from public sources.

Preprocessing's final check, the agent's `lean_verify` and postprocessing's
verification all run the Comparator with the Landrun, nanoda and lean4export
binaries named by each repository's verification contract. This script builds
every commit listed in configs/comparator-tools.yaml and installs it where
`leanlean.palomar_comparator` looks for it:

    .cache/palomar-comparator/bundle/bin/landrun
    .cache/palomar-comparator/bundle/bin/nanoda_bin
    .cache/palomar-comparator/bundle/comparator/<commit>/comparator
    .cache/palomar-comparator/bundle/lean4export/<commit>/lean4export

Each source is fetched from its upstream at the pinned commit and built in a
Docker container from a digest-pinned Go, Rust or Lean (elan) toolchain. Tools
already present are skipped, so rerunning only builds what is missing. Every
installed binary is recorded (commit, toolchain, image, sha256) in
<bundle>/tools-manifest.json.

Usage:
    uv run python scripts/build_comparator_bundle.py
    uv run python scripts/build_comparator_bundle.py --only comparator@68a06410 --only landrun
    uv run python scripts/build_comparator_bundle.py --list

Requires git and Docker on the host, and network access for the fetch, the
toolchains, Lake's lean4export dependency, Go modules and crates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.palomar_comparator import TOOL_BUNDLE, TOOL_LAYOUT as LAYOUT  # noqa: E402

CONFIG = ROOT / "configs/comparator-tools.yaml"
CONFIG_SCHEMA = "leanlean_comparator_tools_v1"
MANIFEST_NAME = "tools-manifest.json"
MANIFEST_SCHEMA = "leanlean_comparator_tool_bundle_v1"
BUILDERS = ("go", "rust", "lean")


@dataclass(frozen=True)
class Pin:
    tool: str
    repository: str
    builder: str
    commit: str
    toolchain: str
    sha256: str | None
    used_by: Any

    @property
    def install_path(self) -> str:
        return LAYOUT[self.tool].format(commit=self.commit)

    @property
    def label(self) -> str:
        return f"{self.tool}@{self.commit[:8]}"


def load_config(path: Path) -> tuple[dict[str, Any], list[Pin]]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise SystemExit(f"{path}: expected schema {CONFIG_SCHEMA}")
    images = config.get("images") or {}
    pins: list[Pin] = []
    for tool, spec in (config.get("tools") or {}).items():
        if tool not in LAYOUT:
            raise SystemExit(f"{path}: unknown tool {tool!r}")
        builder = spec.get("builder")
        if builder not in BUILDERS or not images.get(builder):
            raise SystemExit(f"{path}: {tool} has no usable builder image")
        commits = spec.get("commits") or {}
        if tool in ("landrun", "nanoda") and len(commits) != 1:
            raise SystemExit(f"{path}: {tool} installs at one path; pin exactly one commit")
        for commit, entry in commits.items():
            commit = str(commit)
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise SystemExit(f"{path}: {tool} commit {commit!r} is not a full SHA-1")
            pins.append(Pin(
                tool=tool,
                repository=str(spec["repository"]),
                builder=builder,
                commit=commit,
                toolchain=str(entry["toolchain"]),
                sha256=entry.get("sha256"),
                used_by=entry.get("used_by"),
            ))
    return config, pins


def select(pins: list[Pin], only: list[str]) -> list[Pin]:
    if not only:
        return pins
    chosen: list[Pin] = []
    for request in only:
        tool, _, prefix = request.partition("@")
        matches = [
            pin for pin in pins
            if pin.tool == tool and pin.commit.startswith(prefix)
        ]
        if not matches:
            raise SystemExit(f"--only {request}: no pinned tool matches")
        chosen.extend(pin for pin in matches if pin not in chosen)
    return chosen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def remove_tree(path: Path) -> None:
    """Remove a build tree, including the read-only directories Lake leaves."""

    def make_writable(function, target, _error):
        Path(target).parent.chmod(0o755)
        Path(target).chmod(0o755)
        function(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable)
    else:
        shutil.rmtree(path, onerror=make_writable)


def run(command: list[str], *, log: Path | None = None, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if log is not None:
        with log.open("a") as stream:
            stream.write(f"$ {shlex.join(command)}\n{result.stdout}\n")
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        where = f" (log: {log})" if log is not None else ""
        raise RuntimeError(f"command failed{where}: {shlex.join(command)}\n{tail}")
    return result.stdout


class Builder:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.args = args
        self.config = config
        self.bundle: Path = args.bundle
        self.work: Path = args.work_dir
        self.images: dict[str, str] = config["images"]
        self.manifest_lock = threading.Lock()
        self.rootless = "rootless" in run(
            ["docker", "info", "--format", "{{json .SecurityOptions}}"]
        )

    # -- sources --------------------------------------------------------

    def fetch(self, pin: Pin) -> Path:
        """Fetch one commit into work/src/<tool>-<commit> (a clean checkout)."""

        source = self.work / "src" / f"{pin.tool}-{pin.commit}"
        if source.is_dir() and self._head(source) == pin.commit:
            return source
        source.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{source.name}.", dir=source.parent))
        try:
            run(["git", "init", "-q", str(staging)])
            run(["git", "-C", str(staging), "fetch", "-q", "--depth=1",
                 pin.repository, pin.commit])
            run(["git", "-C", str(staging), "-c", "advice.detachedHead=false",
                 "checkout", "-q", "FETCH_HEAD"])
            if self._head(staging) != pin.commit:
                raise RuntimeError(f"{pin.label}: fetched the wrong commit")
            if source.exists():
                shutil.rmtree(source)
            staging.rename(source)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return source

    @staticmethod
    def _head(path: Path) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def elan_archive(self) -> Path:
        spec = self.config["elan"]
        target = self.work / "downloads" / f"elan-{spec['version']}.tar.gz"
        if target.is_file() and sha256_file(target) == spec["sha256"]:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            partial.write_bytes(response.read())
        if sha256_file(partial) != spec["sha256"]:
            partial.unlink()
            raise RuntimeError(f"elan {spec['version']} archive hash mismatch")
        partial.replace(target)
        return target

    # -- containers -----------------------------------------------------

    def docker(
        self,
        *,
        image: str,
        name: str,
        script: str,
        mounts: dict[Path, str],
        env: dict[str, str],
        log: Path,
    ) -> str:
        command = [
            "docker", "run", "--rm",
            "--name", f"leanlean-toolbuild-{name}-{uuid.uuid4().hex[:6]}",
            "--label", "org.leanlean.role=comparator-tool-build",
            "--label", f"org.leanlean.tool={name}",
            "--cpus", str(self.args.cpus),
            "--memory", self.args.memory,
            "--pids-limit", str(self.args.pids_limit),
        ]
        if self.args.cgroup_parent:
            command += ["--cgroup-parent", self.args.cgroup_parent]
        for host, container in mounts.items():
            command += ["-v", f"{host}:{container}"]
        for key, value in env.items():
            command += ["-e", f"{key}={value}"]
        # Hand every output back to the caller, even after a failed build:
        # archives Lake or Cargo unpack keep foreign owners otherwise. Under
        # rootless Docker the container's root is the calling user.
        owner = "0:0" if self.rootless else f"{os.getuid()}:{os.getgid()}"
        targets = " ".join(sorted(set(mounts.values())))
        script = f"trap 'chown -R {owner} {targets}' EXIT\n" + script
        command += [image, "bash", "-euo", "pipefail", "-c", script]
        return run(command, log=log)

    def install_elan_toolchains(self, pins: list[Pin]) -> None:
        toolchains = sorted({pin.toolchain for pin in pins if pin.builder == "lean"})
        if not toolchains:
            return
        elan_home = self.work / "elan"
        elan_home.mkdir(parents=True, exist_ok=True)
        archive = self.elan_archive()
        installed = {
            path.name for path in (elan_home / "toolchains").glob("*")
        } if (elan_home / "toolchains").is_dir() else set()
        missing = [
            tc for tc in toolchains
            if tc.replace("/", "--").replace(":", "---") not in installed
        ]
        if not missing and (elan_home / "bin/elan").is_file():
            return
        log = self.work / "logs" / "elan.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "if [ ! -x /elan/bin/elan ]; then\n"
            "  tar -xzf /downloads/elan.tar.gz -C /tmp\n"
            "  /tmp/elan-init -y --no-modify-path --default-toolchain none\n"
            "fi\n"
        ) + "".join(f"elan toolchain install {shlex.quote(tc)}\n" for tc in missing)
        print(f"installing elan toolchains: {', '.join(missing) or '(elan only)'}", flush=True)
        self.docker(
            image=self.images["lean"],
            name="elan",
            script=script,
            mounts={elan_home.resolve(): "/elan", archive.resolve(): "/downloads/elan.tar.gz"},
            env={"ELAN_HOME": "/elan", "PATH": "/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
            log=log,
        )

    # -- builds ---------------------------------------------------------

    def build(self, pin: Pin) -> dict[str, Any]:
        source = self.fetch(pin)
        name = f"{pin.tool}-{pin.commit[:8]}"
        build = self.work / "build" / f"{pin.tool}-{pin.commit}"
        out = build / "out"
        tree = build / "src"
        if build.exists():
            remove_tree(build)
        shutil.copytree(source, tree, ignore=shutil.ignore_patterns(".git"), symlinks=True)
        out.mkdir(parents=True)
        log = self.work / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("")
        mounts = {tree.resolve(): "/src", out.resolve(): "/out"}
        image = self.images[pin.builder]
        caches = self.work / "cache"
        if pin.builder == "go":
            (caches / "go-mod").mkdir(parents=True, exist_ok=True)
            mounts[(caches / "go-mod").resolve()] = "/go/pkg/mod"
            env = {"CGO_ENABLED": "0", "GOTOOLCHAIN": "local", "GOFLAGS": "-mod=readonly -modcacherw"}
            script = (
                "cd /src\n"
                "go build -trimpath -buildvcs=false -ldflags=-buildid= -o /out/landrun ./cmd/landrun\n"
                "go version > /out/toolchain-version\n"
            )
            binary = out / "landrun"
        elif pin.builder == "rust":
            (caches / "cargo-registry").mkdir(parents=True, exist_ok=True)
            mounts[(caches / "cargo-registry").resolve()] = "/usr/local/cargo/registry"
            env = {"CARGO_TARGET_DIR": "/out"}
            script = (
                "cd /src\n"
                "cargo build --release --locked\n"
                "rustc --version > /out/toolchain-version\n"
            )
            binary = out / "release/nanoda_bin"
        else:
            declared = (tree / "lean-toolchain").read_text().strip()
            if declared != pin.toolchain:
                raise RuntimeError(
                    f"{pin.label}: checkout declares {declared}, pinned {pin.toolchain}"
                )
            mounts[(self.work / "elan").resolve()] = "/elan"
            env = {
                "ELAN_HOME": "/elan",
                "ELAN_TOOLCHAIN": pin.toolchain,
                "PATH": "/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            }
            script = (
                "cd /src\n"
                # --no-cache: compile dependencies too rather than download
                # Reservoir build archives.
                f"lake build --no-cache {pin.tool}\n"
                "lean --version > /out/toolchain-version\n"
                f"cp .lake/build/bin/{pin.tool} /out/{pin.tool}\n"
            )
            binary = out / pin.tool
        started = time.monotonic()
        print(f"building {pin.label} ({pin.toolchain}) ...", flush=True)
        self.docker(image=image, name=name, script=script, mounts=mounts, env=env, log=log)
        seconds = round(time.monotonic() - started, 1)
        if not binary.is_file():
            raise RuntimeError(f"{pin.label}: build produced no {binary.name} (log: {log})")
        destination = self.bundle / pin.install_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex[:8]}")
        shutil.copyfile(binary, staging)
        staging.chmod(0o755)
        staging.replace(destination)
        record = {
            "tool": pin.tool,
            "repository": pin.repository,
            "commit": pin.commit,
            "toolchain": pin.toolchain,
            "toolchain_version": (out / "toolchain-version").read_text().strip(),
            "image": image,
            "sha256": sha256_file(destination),
            "size": destination.stat().st_size,
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "build_seconds": seconds,
            "pinned_sha256": pin.sha256,
        }
        record["matches_pin"] = (
            record["sha256"] == pin.sha256 if pin.sha256 else None
        )
        if record["matches_pin"] is False:
            print(f"WARNING {pin.label}: sha256 {record['sha256']} differs from the "
                  f"pinned {pin.sha256}; the build is not bit-identical", flush=True)
        if not self.args.keep_build:
            remove_tree(build)
        match = {True: "matches pin", False: "DIFFERS from pin", None: "no pin"}[
            record["matches_pin"]
        ]
        print(f"installed {pin.install_path} sha256={record['sha256'][:16]} "
              f"({match}, {seconds}s)", flush=True)
        return record

    # -- manifest -------------------------------------------------------

    def manifest_path(self) -> Path:
        return self.bundle / MANIFEST_NAME

    def load_manifest(self) -> dict[str, Any]:
        path = self.manifest_path()
        if path.is_file():
            manifest = json.loads(path.read_text())
            if manifest.get("schema") == MANIFEST_SCHEMA:
                return manifest
        return {"schema": MANIFEST_SCHEMA, "tools": {}}

    def record(self, pin: Pin, entry: dict[str, Any]) -> None:
        with self.manifest_lock:
            manifest = self.load_manifest()
            manifest["config"] = os.path.relpath(self.args.config, ROOT)
            manifest["config_sha256"] = sha256_file(self.args.config)
            manifest["tools"][pin.install_path] = entry
            manifest["tools"] = dict(sorted(manifest["tools"].items()))
            path = self.manifest_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
            staging.write_text(json.dumps(manifest, indent=2) + "\n")
            staging.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--bundle", type=Path, default=ROOT / TOOL_BUNDLE,
                        help="install root (default: %(default)s)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="sources, toolchains, caches and logs "
                             "(default: <bundle>/../build-tools)")
    parser.add_argument("--only", action="append", default=[],
                        metavar="TOOL[@COMMIT_PREFIX]",
                        help="build only these pins (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild tools that are already installed")
    parser.add_argument("--list", action="store_true",
                        help="print the pins and their install state, then exit")
    parser.add_argument("--jobs", type=int, default=1,
                        help="concurrent builds (default: %(default)s)")
    parser.add_argument("--cpus", type=float, default=8, help="per container")
    parser.add_argument("--memory", default="16g", help="per container")
    parser.add_argument("--pids-limit", type=int, default=4096, help="per container")
    parser.add_argument("--cgroup-parent", default=None,
                        help="Docker --cgroup-parent for the build containers")
    parser.add_argument("--keep-build", action="store_true",
                        help="keep each build tree under the work directory")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.bundle = args.bundle.resolve()
    args.work_dir = (args.work_dir or args.bundle.parent / "build-tools").resolve()

    config, pins = load_config(args.config)
    pins = select(pins, args.only)
    if args.list:
        for pin in pins:
            present = (args.bundle / pin.install_path).is_file()
            print(f"{'present' if present else 'missing':8} {pin.label:22} "
                  f"{pin.toolchain:30} {pin.install_path}")
        return 0

    pending = [pin for pin in pins if args.force or not (args.bundle / pin.install_path).is_file()]
    for pin in pins:
        if pin not in pending:
            print(f"present  {pin.install_path}", flush=True)
    if not pending:
        print(f"bundle complete: {args.bundle}")
        return 0

    builder = Builder(args, config)
    builder.install_elan_toolchains(pending)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(builder.build, pin): pin for pin in pending}
        for future in concurrent.futures.as_completed(futures):
            pin = futures[future]
            try:
                builder.record(pin, future.result())
            except Exception as error:  # noqa: BLE001 - report every failed pin
                failures.append(pin.label)
                print(f"FAILED {pin.label}: {error}", file=sys.stderr, flush=True)
    if failures:
        print(f"{len(failures)} build(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"bundle complete: {args.bundle} (manifest: {builder.manifest_path()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
