"""Read-only, cgroup-backed resource reporting for new evaluation containers.

No prompt, command wrappers, or files in the agent's repository are involved.
Only an allowlist of proc/sys resource files is projected into the container.
Docker's cgroup memory limit and CPU-time quota remain the enforcement boundary.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading


MODE = "container_v1"
TARGETS = (
    "/proc/meminfo", "/proc/cpuinfo", "/proc/stat",
    "/sys/devices/system/cpu/online",
    "/sys/devices/system/cpu/present",
    "/sys/devices/system/cpu/possible",
    "/sys/devices/system/cpu/offline",
)


def limits(run_args: list[str], threads: int) -> tuple[int, int]:
    def option(name):
        values = [a.split("=", 1)[1] for a in run_args if a.startswith(name + "=")]
        if len(values) != 1:
            raise ValueError(f"resource visibility requires exactly one {name}= value")
        return values[0]

    cpu_text = option("--cpus")
    if not re.fullmatch(r"[1-9][0-9]*", cpu_text) or int(cpu_text) != threads:
        raise ValueError("container_v1 requires equal, positive integer CPUs and build threads")
    memory = option("--memory")
    match = re.fullmatch(r"([1-9][0-9]*)([bkmg]?)", memory.lower())
    if not match:
        raise ValueError("unsupported container memory size for resource visibility")
    memory_bytes = int(match[1]) * 1024 ** {"": 0, "b": 0, "k": 1, "m": 2, "g": 3}[match[2]]
    if option("--memory-swap") != memory:
        raise ValueError("container_v1 currently requires swap disabled")
    if any(a.startswith("--cpuset") for a in run_args):
        raise ValueError("resource visibility manages CPU affinity; remove explicit cpuset overrides")
    return int(cpu_text), memory_bytes


def select_cpus(available: set[int], count: int, identity: str) -> list[int]:
    cpus = sorted(available)
    if count < 1 or count > len(cpus):
        raise ValueError("requested CPU view exceeds the launcher's allowed CPU affinity")
    # Spread workers over the host. These sets are not exclusive reservations.
    start = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) % len(cpus)
    return sorted((cpus + cpus)[start:start + count])


def cpuinfo_view(text: str, selected: list[int]) -> str:
    blocks = []
    for block in text.strip().split("\n\n"):
        match = re.search(r"^processor\s*:\s*(\d+)\s*$", block, re.M)
        if match and int(match[1]) in selected:
            blocks.append(block)
    if len(blocks) != len(selected):
        raise ValueError("cannot project CPU information for the selected affinity")
    return "\n\n".join(blocks) + "\n\n"


def stat_view(text: str, selected: list[int]) -> str:
    rows = [line for line in text.splitlines()
            if line.split()[0] in {f"cpu{cpu}" for cpu in selected}]
    if len(rows) != len(selected):
        raise ValueError("cannot project CPU counters for selected affinity")
    counters = [list(map(int, row.split()[1:])) for row in rows]
    total = "cpu  " + " ".join(str(sum(column)) for column in zip(*counters))
    rest = [line for line in text.splitlines() if not line.startswith("cpu")]
    return "\n".join([total, *rows, *rest]) + "\n"


def meminfo_view(limit: int, current: int, stat: dict[str, int]) -> str:
    current = max(0, min(limit, current))
    free = limit - current
    reclaimable = min(current, max(0, stat.get("inactive_file", 0)))
    values = {
        "MemTotal": limit, "MemFree": free, "MemAvailable": free + reclaimable,
        "Buffers": 0, "Cached": stat.get("file", 0), "SwapCached": 0,
        "Active": stat.get("active_anon", 0) + stat.get("active_file", 0),
        "Inactive": stat.get("inactive_anon", 0) + stat.get("inactive_file", 0),
        "Active(anon)": stat.get("active_anon", 0), "Inactive(anon)": stat.get("inactive_anon", 0),
        "Active(file)": stat.get("active_file", 0), "Inactive(file)": stat.get("inactive_file", 0),
        "Unevictable": stat.get("unevictable", 0), "Mlocked": 0,
        "SwapTotal": 0, "SwapFree": 0, "Dirty": stat.get("file_dirty", 0),
        "Writeback": stat.get("file_writeback", 0), "AnonPages": stat.get("anon", 0),
        "Mapped": stat.get("file_mapped", 0), "Shmem": stat.get("shmem", 0),
        "KReclaimable": stat.get("slab_reclaimable", 0), "Slab": stat.get("slab", 0),
        "SReclaimable": stat.get("slab_reclaimable", 0), "SUnreclaim": stat.get("slab_unreclaimable", 0),
        "KernelStack": stat.get("kernel_stack", 0), "PageTables": stat.get("pagetables", 0),
        "CommitLimit": limit, "Committed_AS": stat.get("anon", 0),
    }
    return "".join(f"{key + ':':<20}{max(0, min(limit, value)) // 1024:>20} kB\n"
                   for key, value in values.items())


class ResourceView:
    def __init__(self, run_args: list[str], threads: int, identity: str):
        count, self.memory_limit = limits(run_args, threads)
        self.cpus = select_cpus(set(os.sched_getaffinity(0)), count, identity)
        self.threads = threads
        self._directory = tempfile.TemporaryDirectory(prefix="leanlean-resource-view-")
        self.root = Path(self._directory.name)
        self._stop = threading.Event()
        self._thread = None
        self.cgroup = None
        try:
            cpu_list = ",".join(map(str, self.cpus)) + "\n"
            initial = {
                "/proc/meminfo": meminfo_view(self.memory_limit, 0, {}),
                "/proc/cpuinfo": cpuinfo_view(Path("/proc/cpuinfo").read_text(), self.cpus),
                "/proc/stat": stat_view(Path("/proc/stat").read_text(), self.cpus),
                **{target: cpu_list for target in TARGETS[3:6]},
                "/sys/devices/system/cpu/offline": "\n",
            }
            for target, content in initial.items():
                path = self.file(target)
                path.write_text(content)
                path.chmod(0o444)
        except BaseException:
            self.close()
            raise

    def file(self, target: str) -> Path:
        if target not in TARGETS:
            raise ValueError("resource-view mount target is not allowlisted")
        return self.root / target.strip("/").replace("/", "_")

    def docker_args(self) -> list[str]:
        args = ["--cpuset-cpus=" + ",".join(map(str, self.cpus)),
                f"--env=LEAN_NUM_THREADS={self.threads}", f"--env=OMP_NUM_THREADS={self.threads}"]
        for target in TARGETS:
            args.extend(["--mount", f"type=bind,src={self.file(target)},dst={target},readonly"])
        return args

    def activate(self, executable: str, container_id: str, logger) -> None:
        pid = int(subprocess.check_output(
            [executable, "inspect", "--format", "{{.State.Pid}}", container_id],
            text=True, timeout=30).strip())
        relative = next(line.split("::", 1)[1] for line in
                        (Path("/proc") / str(pid) / "cgroup").read_text().splitlines()
                        if line.startswith("0::"))
        self.cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
        if int((self.cgroup / "memory.max").read_text()) != self.memory_limit:
            raise RuntimeError("projected memory limit differs from the enforced cgroup limit")
        quota, period = map(int, (self.cgroup / "cpu.max").read_text().split())
        if quota != period * self.threads:
            raise RuntimeError("projected CPU allocation differs from the enforced cgroup quota")
        self.refresh()
        if set(os.sched_getaffinity(pid)) != set(self.cpus):
            logger.info("Container cpuset not delegated; agent execs use inherited taskset affinity")

        def update():
            while not self._stop.wait(1):
                try:
                    self.refresh()
                except OSError:
                    if not self.cgroup.exists():
                        return
                    logger.exception("Could not refresh the container resource view")

        self._thread = threading.Thread(target=update, name="container-resource-view", daemon=True)
        self._thread.start()

    def refresh(self) -> None:
        current = int((self.cgroup / "memory.current").read_text())
        stat = dict((key, int(value)) for key, value in
                    (line.split() for line in (self.cgroup / "memory.stat").read_text().splitlines()))
        # Keep the mounted inode; replacing it would leave Docker viewing stale data.
        # Fixed-width meminfo values also avoid an empty/truncated interval.
        for target, content in {
            "/proc/meminfo": meminfo_view(self.memory_limit, current, stat),
            "/proc/stat": stat_view(Path("/proc/stat").read_text(), self.cpus),
        }.items():
            path = self.file(target)
            path.chmod(0o644)
            with path.open("r+") as stream:
                stream.write(content)
                stream.truncate()
            path.chmod(0o444)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._directory.cleanup()
