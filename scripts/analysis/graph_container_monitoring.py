"""Container diagnostics shared by graph analysis and macro measurement."""
import json
import subprocess
import time
from pathlib import Path


def inspect(container: str) -> dict:
    proc = subprocess.run(["docker", "inspect", container], capture_output=True, text=True, timeout=15)
    if proc.returncode:
        return {"inspect_error": proc.stderr}
    obj = json.loads(proc.stdout)[0]
    return {key: obj[key] for key in ("Id", "Name", "State", "HostConfig")}


def cgroup_sample(cgroup: Path | None) -> dict:
    sample = {"time": time.time()}
    if cgroup is not None:
        for name in ("memory.current", "memory.peak", "memory.max", "memory.events", "memory.events.local", "cpu.stat"):
            try:
                sample[name] = (cgroup / name).read_text().strip()
            except OSError as exc:
                sample[name] = {"error": str(exc)}
    return sample
