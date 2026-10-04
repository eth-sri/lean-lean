"""The declared source verifier (the official Comparator) for stripped repositories."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Protocol

from leanlean.benchmarks.leanlean import (
    _install_palomar_task_metadata,
)
from leanlean.environments.docker import DockerEnvironment
from leanlean.palomar_comparator import (
    COMPARATOR_STACK_KIB,
    COMPARATOR_PROVENANCE_KINDS,
    COMPARATOR_SUCCESS_MARKER,
    resolve_palomar_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class DeclaredVerifier(Protocol):
    name: str

    def applies(self, row: Mapping[str, Any]) -> bool:
        ...

    def verify(
        self,
        environment: DockerEnvironment,
        row: Mapping[str, Any],
        *,
        build_jobs: int,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        ...


class PalomarComparatorVerifier:
    """Run the official frozen Comparator when the source declares that contract."""

    name = "leanprover/comparator"

    def applies(self, row: Mapping[str, Any]) -> bool:
        protected = row.get("protected")
        provenance = (
            protected.get("provenance")
            if isinstance(protected, Mapping)
            else None
        )
        return (
            isinstance(provenance, Mapping)
            and provenance.get("kind") in COMPARATOR_PROVENANCE_KINDS
        )

    def verify(
        self,
        environment: DockerEnvironment,
        row: Mapping[str, Any],
        *,
        build_jobs: int,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        contract = resolve_palomar_contract(
            repo_root=REPO_ROOT, database_row=row
        )
        scopes = row.get("scopes")
        if not isinstance(scopes, Mapping):
            raise ValueError("standardized repository scopes are missing")
        protected = row.get("protected")
        if not isinstance(protected, Mapping):
            raise ValueError("standardized protected declarations are missing")
        declarations = protected.get("declarations")
        if not isinstance(declarations, list):
            raise ValueError("inline protected declarations are required")
        _install_palomar_task_metadata(
            environment,
            contract=contract,
            exclude_dirs=list(scopes.get("exclude_dirs") or []),
            include_prefix=str(scopes.get("target_dir") or "."),
            build_jobs=build_jobs,
            prewarm=False,
        )
        result = environment.execute(
            "cd /testbed && "
            "LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify",
            # The environment lifetime is already pinned from
            # manifest.timeouts.container_seconds. Reuse that sole source of
            # truth instead of passing a second numeric exec timeout that can
            # drift or be clamped independently.
            timeout=False,
        )
        output = str(result.get("output") or "")
        return {
            "applicable": True,
            "engine": self.name,
            "command": "lean_verify",
            "contract_schema": contract["schema"],
            "contract_sha256": contract["sha256"],
            "invocation_policy": "single_final_comparator",
            "invocations": 1,
            "timeout_policy": "pinned_container_lifetime",
            "timeout_seconds": timeout_seconds,
            "returncode": int(result.get("returncode", 1)),
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "output_tail": output[-4000:],
            # Exit 0 alone is not acceptance: `docker exec` reports 0 for a
            # Comparator that died mid-run.
            "passed": result.get("returncode") == 0 and COMPARATOR_SUCCESS_MARKER in output,
        }


DECLARED_VERIFIERS: tuple[DeclaredVerifier, ...] = (
    PalomarComparatorVerifier(),
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def select_declared_verifier(
    row: Mapping[str, Any],
) -> DeclaredVerifier | None:
    matches = [verifier for verifier in DECLARED_VERIFIERS if verifier.applies(row)]
    if len(matches) > 1:
        raise ValueError("repository declares multiple final verifiers")
    return matches[0] if matches else None


def comparator_environment(
    image: str, container: Mapping[str, Any], timeout_seconds: int
) -> DockerEnvironment:
    nofile = int(container["nofile_limit"])
    # The Comparator's nanoda kernel re-check overflows the 8 MiB default main
    # stack on large developments. Take the stack from the same manifest field
    # the preprocessing container uses, never below what nanoda needs.
    stack_bytes = (
        max(int(container.get("lean_thread_stack_kib") or 0), COMPARATOR_STACK_KIB)
        * 1024
    )
    run_args = [
        "--rm",
        f"--cpus={container['cpus']}",
        f"--memory={container['memory']}",
        f"--memory-swap={container['memory']}",
        f"--ulimit=nofile={nofile}:{nofile}",
        f"--ulimit=stack={stack_bytes}:{stack_bytes}",
        # Lake otherwise sizes its job pool to the host's cores, not the container's.
        f"--env=LEAN_NUM_THREADS={container['cpus']}",
    ]
    cgroup_parent = str(container.get("cgroup_parent") or "")
    if cgroup_parent:
        run_args.append(f"--cgroup-parent={cgroup_parent}")
    return DockerEnvironment(
        image=image,
        cwd="/testbed",
        timeout=timeout_seconds,
        container_timeout=f"{timeout_seconds}s",
        container_pids_limit=int(container["pids_limit"]),
        network_mode="none",
        network_policy="none",
        run_args=run_args,
    )


def image_id(tag: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"required image is missing: {tag}")
    return result.stdout.strip()


def image_labels(image: str) -> dict[str, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        labels = json.loads(result.stdout)
    except ValueError:
        return {}
    return labels if isinstance(labels, dict) else {}
