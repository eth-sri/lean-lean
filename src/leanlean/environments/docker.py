# Forked from: https://github.com/SWE-agent/mini-swe-agent

import errno
import gzip
import hashlib
import importlib
import json
import logging
import os
import shlex
import time
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import posixpath
import socket
from urllib.parse import urlsplit, urlunsplit

from leanlean import Environment
from leanlean.environments.container_identity import container_identity


class BaseDockerEnvironment:
    """Base class for Docker environments that provides apply_patch script setup."""

    def __init__(self):
        self.logger = logging.getLogger("leanlean.environment")

    def _get_apply_patch_script_path(self) -> Path | None:
        """Get the path to the apply_patch script."""
        script_path = Path(__file__).parent / "extra" / "apply_patch.py"
        if not script_path.exists():
            self.logger.warning(f"apply_patch script not found at {script_path}")
            return None
        return script_path

    def _setup_apply_patch_script(self):
        """Copy the apply_patch script to the container and make it available in PATH.
        Must be implemented by subclasses based on their container access method.
        """
        raise NotImplementedError("Subclasses must implement _setup_apply_patch_script")


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/project/testbed"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    monitoring_metadata: dict[str, str] = field(default_factory=dict)
    """Optional run_id, role, model, instance_id and worker_id Docker labels."""

    resource_visibility: str = "host"
    """Opt-in cgroup-backed proc/sys view; legacy runs retain host reporting."""
    resource_threads: int = 0

    network_mode: str = "none"
    """Initial Docker network mode. Safe policies require ``none``."""

    network_policy: str = "none"
    """Network policy: ``none`` or ``model_proxy_only``.

    ``model_proxy_only`` starts the task with no network and later permits the
    generator to attach a private, internal Docker network containing only a
    fixed-target relay to the local model gateway. It never gives the task a
    default route or access to the Docker host network.
    """

    model_relay_image: str = "leanlean-model-relay:latest"
    """Locally built, scratch-based fixed-target TCP relay image."""

    container_pids_limit: int = 4096
    """Hard process-count ceiling for agent and build subprocesses."""

    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""

    container_grace_seconds: int = 1800
    """Extra time the container's `sleep` outlives the agent's exec timeout.

    The agent is launched with an exec timeout of `container_timeout_seconds()`.
    If the container's own `sleep` used the same duration, the two would race:
    on a timed-out run the container (`--rm`) could be removed at the very
    moment the exec returns, destroying the agent's work before we can extract
    the diff or commit a snapshot. Keeping the container alive a bit longer
    guarantees there's a live container to capture from after a timeout/kill.
    """

    refactor_rounds: int = 0
    """Number of refactor rounds that reuse this single container after the
    initial generation pass. Each round runs in the same container and gets its
    own exec timeout of `container_timeout_seconds()`, so the container's `sleep`
    must outlive ALL of them: total lifetime scales as (refactor_rounds + 1) *
    container_timeout_seconds() + grace. 0 == single pass, no refactor rounds.
    Without this the `sleep` only covered one round and the container was removed
    mid-run on later rounds, corrupting their captured diffs.
    """

    def __post_init__(self) -> None:
        policies = {"none", "model_proxy_only"}
        if self.network_policy not in policies:
            raise ValueError(
                f"unsupported Docker network policy {self.network_policy!r}; "
                f"expected one of {sorted(policies)}"
            )
        if self.network_mode != "none":
            raise ValueError(
                f"network_policy={self.network_policy!r} requires network_mode='none'"
            )
        if self.container_pids_limit < 1:
            raise ValueError("container_pids_limit must be positive")
        self._validate_run_args()
        if self.resource_visibility not in {"host", "container_v1"}:
            raise ValueError("unsupported resource_visibility")
        if self.resource_visibility == "container_v1":
            from leanlean.environments.resource_view import limits
            limits(self.run_args, self.resource_threads)

    def _validate_run_args(self) -> None:
        """Reject arguments that could undo the environment's isolation."""

        forbidden_exact = {"-P", "-p", "-v"}
        forbidden_prefixes = (
            "--network",
            "--net",
            "--privileged",
            "--pid",
            "--ipc",
            "--uts",
            "--userns",
            "--cgroupns",
            "--cap-add",
            "--device",
            "--mount",
            "--volume",
            "--volumes-from",
            "--security-opt",
            "--add-host",
            "--publish",
            "--expose",
            "--link",
            "--runtime",
            "--gpus",
            "-p=",
            "-v=",
        )
        for arg in self.run_args:
            if arg in forbidden_exact or arg.startswith(forbidden_prefixes):
                raise ValueError(
                    f"unsafe Docker run argument under managed isolation: {arg!r}"
                )

    def container_timeout_seconds(self) -> int:
        """Parse container_timeout string (e.g. '2h', '300m', '18000s') to seconds."""
        s = self.container_timeout.strip()
        if s.endswith("h"):
            return int(s[:-1]) * 3600
        if s.endswith("m"):
            return int(s[:-1]) * 60
        if s.endswith("s"):
            return int(s[:-1])
        return int(s)

    def container_sleep_seconds(self) -> int:
        """How long the container's `sleep` should run: one exec timeout per
        round (initial pass + refactor rounds) plus grace, so the container
        outlives every round that reuses it rather than just the first."""
        return (
            self.container_timeout_seconds() * (max(0, self.refactor_rounds) + 1)
            + self.container_grace_seconds
        )


class DockerEnvironment(BaseDockerEnvironment, Environment):
    def __init__(
        self, *, config_class: type = DockerEnvironmentConfig, logger=None, **kwargs
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self._container_path: str | None = None
        BaseDockerEnvironment.__init__(self)
        self.logger = logger or logging.getLogger("leanlean.environment")
        self.container_id: str | None = None
        self._model_network_name: str | None = None
        self._model_relay_id: str | None = None
        self._model_gateway_target: str | None = None
        self._cleanup_lock = threading.Lock()
        self._resource_view = None
        self.config = config_class(**kwargs)
        try:
            self._start_container()
        except BaseException:
            self.cleanup()
            raise

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name, monitoring_labels = container_identity(
            self.config.image, getattr(self.config, "monitoring_metadata", {})
        )
        resource_args = []
        if self.config.resource_visibility == "container_v1":
            from leanlean.environments.resource_view import ResourceView
            self._resource_view = ResourceView(
                self.config.run_args, self.config.resource_threads, container_name
            )
            resource_args = self._resource_view.docker_args()
            self.config.env.update(
                LEAN_NUM_THREADS=str(self.config.resource_threads),
                OMP_NUM_THREADS=str(self.config.resource_threads),
            )
        label_args = [
            f"--label={key}={value}" for key, value in sorted(monitoring_labels.items())
        ]
        cmd = [
            self.config.executable,
            "run",
            "-d",
            # "--add-host=host.docker.internal:172.17.0.1",
            f"--network={self.config.network_mode}",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--pids-limit={self.config.container_pids_limit}",
            *self.config.run_args,
            *resource_args,
            *label_args,
            self.config.image,
            "sleep",
            # Outlive the agent's exec timeout by a grace buffer so a timed-out
            # run leaves a live container to extract the diff / commit from.
            str(self.config.container_sleep_seconds()),
        ]
        self.logger.info(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60 * 60,  # docker pull might take a while
            check=True,
        )
        self.logger.info(
            f"Started container {container_name} with ID {result.stdout.strip()}"
        )
        self.container_id = result.stdout.strip()
        if self._resource_view is not None:
            self._resource_view.activate(self.config.executable, self.container_id, self.logger)

        # copy /testbed to /projects/main
        if self.config.cwd == "/project/testbed":
            output = self.execute(
                "mkdir -p /project && mv /testbed /project", timeout=False
            )
            self.logger.info(f"Moved /testbed to /project/testbed: {output['output']}")
        self._capture_container_environment()
        # Copy apply_patch script to container and make it executable
        # self._setup_apply_patch_script()

    def copy_host_executable(self, source: str | Path, destination: str) -> None:
        """Copy one trusted host executable into the network-disabled container."""

        if not self.container_id:
            raise RuntimeError("cannot copy a tool before the container starts")
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"host tool does not exist: {source_path}")
        if not destination.startswith("/usr/local/bin/") or "/../" in destination:
            raise ValueError(f"unsafe container tool destination: {destination!r}")
        quoted_destination = shlex.quote(destination)
        # Stream bytes through docker exec instead of ``docker cp``. On hosts
        # using user-namespace remapping, docker cp tries to preserve the host's
        # high numeric UID and can fail before writing the file.
        with source_path.open("rb") as source_stream:
            copied = subprocess.run(
                [
                    self.config.executable,
                    "exec",
                    "-i",
                    self.container_id,
                    "sh",
                    "-c",
                    f"umask 022; cat > {quoted_destination}; "
                    f"chmod 0555 {quoted_destination}",
                ],
                stdin=source_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        if copied.returncode != 0:
            output = copied.stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"could not inject {source_path.name} into eval container: {output}"
            )

    def copy_host_directory(self, source: str | Path, destination: str) -> None:
        """Copy one trusted host directory into the network-disabled container."""

        if not self.container_id:
            raise RuntimeError("cannot copy a tool bundle before the container starts")
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"host tool bundle does not exist: {source_path}")
        destination_path = posixpath.normpath(destination)
        if (
            not destination_path.startswith("/opt/leanlean/")
            or destination_path != destination
        ):
            raise ValueError(
                f"unsafe container tool-bundle destination: {destination!r}"
            )

        prepared = subprocess.run(
            [
                self.config.executable,
                "exec",
                self.container_id,
                "install",
                "-d",
                "-m",
                "0755",
                destination_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        if prepared.returncode != 0:
            output = prepared.stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"could not prepare tool-bundle destination {destination}: {output}"
            )

        archive = subprocess.Popen(
            ["tar", "-C", str(source_path), "-cf", "-", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        try:
            copied = subprocess.run(
                [
                    self.config.executable,
                    "exec",
                    "-i",
                    self.container_id,
                    "tar",
                    "-C",
                    destination_path,
                    "--no-same-owner",
                    "-xf",
                    "-",
                ],
                stdin=archive.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        except Exception:
            archive.terminate()
            try:
                archive.wait(timeout=30)
            except subprocess.TimeoutExpired:
                archive.kill()
                archive.wait(timeout=30)
            raise
        finally:
            archive.stdout.close()

        archive_error = (
            archive.stderr.read().decode("utf-8", errors="replace").strip()
            if archive.stderr is not None
            else ""
        )
        archive_returncode = archive.wait(timeout=30)
        if archive_returncode != 0 or copied.returncode != 0:
            copy_output = copied.stdout.decode("utf-8", errors="replace").strip()
            detail = "; ".join(part for part in (archive_error, copy_output) if part)
            raise RuntimeError(
                f"could not inject {source_path.name} into eval container: {detail}"
            )

    @staticmethod
    def _local_ipv4_addresses() -> set[str]:
        """Return host IPv4 addresses eligible as fixed relay targets."""

        addresses: set[str] = set()
        for hostname in {socket.gethostname(), socket.getfqdn()}:
            try:
                for result in socket.getaddrinfo(
                    hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM
                ):
                    addresses.add(result[4][0])
            except OSError:
                continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                addresses.add(probe.getsockname()[0])
        except OSError:
            pass
        return {address for address in addresses if not address.startswith("127.")}

    def attach_model_gateway(self, base_url: str) -> str:
        """Attach a private fixed-target relay and return its container URL."""

        if self.config.network_policy != "model_proxy_only":
            raise RuntimeError(
                "model access requires network_policy='model_proxy_only'; "
                f"found {self.config.network_policy!r}"
            )
        if not self.container_id:
            raise RuntimeError("cannot attach a model gateway before container start")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(f"unsafe local model gateway URL: {base_url!r}")
        target_port = parsed.port
        if target_port is None or not (1 <= target_port <= 65535):
            raise ValueError(
                f"model gateway URL must contain a valid port: {base_url!r}"
            )
        target_host = parsed.hostname
        if target_host not in self._local_ipv4_addresses():
            raise ValueError(
                f"model gateway target {target_host!r} is not a local host IPv4 address"
            )
        target = f"{target_host}:{target_port}"
        if self._model_gateway_target is not None:
            if self._model_gateway_target != target:
                raise RuntimeError("an eval container cannot attach two model gateways")
            return urlunsplit(
                (parsed.scheme, "model-gateway:18080", parsed.path, parsed.query, "")
            )

        image_check = subprocess.run(
            [self.config.executable, "image", "inspect", self.config.model_relay_image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if image_check.returncode != 0:
            raise RuntimeError(
                f"required relay image {self.config.model_relay_image!r} is missing; "
                "run scripts/build_model_relay_image.sh"
            )

        suffix = uuid.uuid4().hex[:12]
        network_name = f"leanlean-model-{suffix}"
        relay_name, relay_labels = container_identity(
            self.config.image,
            {**getattr(self.config, "monitoring_metadata", {}), "role": "relay"},
        )
        relay_id: str | None = None
        try:
            subprocess.run(
                [
                    self.config.executable,
                    "network",
                    "create",
                    "--internal",
                    "--driver=bridge",
                    "--label=org.openai.leanlean.network_policy=model_proxy_only",
                    network_name,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            created = subprocess.run(
                [
                    self.config.executable,
                    "create",
                    "--name",
                    relay_name,
                    "--network=bridge",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges:true",
                    "--pids-limit=64",
                    "--memory=32m",
                    "--cpus=0.25",
                    *[f"--label={key}={value}" for key, value in sorted(relay_labels.items())],
                    self.config.model_relay_image,
                    target_host,
                    str(target_port),
                    "18080",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            relay_id = created.stdout.strip()
            subprocess.run(
                [
                    self.config.executable,
                    "network",
                    "connect",
                    "--alias",
                    "model-gateway",
                    network_name,
                    relay_id,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            subprocess.run(
                [self.config.executable, "start", relay_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            # Docker does not permit a container to retain the special
            # ``none`` network while joining another one. Disconnecting it first
            # is still fail-closed: until the internal connect succeeds, the
            # task has no network namespace peer at all.
            subprocess.run(
                [
                    self.config.executable,
                    "network",
                    "disconnect",
                    "none",
                    self.container_id,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            subprocess.run(
                [
                    self.config.executable,
                    "network",
                    "connect",
                    network_name,
                    self.container_id,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if relay_id:
                subprocess.run(
                    [self.config.executable, "rm", "-f", relay_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            subprocess.run(
                [
                    self.config.executable,
                    "network",
                    "disconnect",
                    "--force",
                    network_name,
                    self.container_id,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [self.config.executable, "network", "rm", network_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            output = getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or ""
            raise RuntimeError(
                "could not attach the model-only Docker network: " + str(output).strip()
            ) from exc

        self._model_network_name = network_name
        self._model_relay_id = relay_id
        self._model_gateway_target = target
        self.logger.info(
            "Attached model-only network %s via fixed relay to %s",
            network_name,
            target,
        )
        return urlunsplit(
            (parsed.scheme, "model-gateway:18080", parsed.path, parsed.query, "")
        )

    def _setup_apply_patch_script(self):
        """Copy the apply_patch script to the container and make it available in PATH."""
        if not self.container_id:
            return

        script_path = self._get_apply_patch_script_path()
        if not script_path:
            return

        try:
            # Copy script to /usr/local/bin in the container
            copy_cmd = [
                self.config.executable,
                "cp",
                str(script_path),
                f"{self.container_id}:/usr/local/bin/apply_patch",
            ]
            subprocess.run(copy_cmd, check=True, capture_output=True)

            # Make it executable
            chmod_cmd = [
                self.config.executable,
                "exec",
                self.container_id,
                "chmod",
                "+x",
                "/usr/local/bin/apply_patch",
            ]
            subprocess.run(chmod_cmd, check=True, capture_output=True)

            self.logger.debug("Successfully installed apply_patch script in container")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to install apply_patch script: {e}")

    def _capture_container_environment(self) -> None:
        """Capture useful environment settings from the container."""
        if not self.container_id:
            return
        try:
            cmd = [
                self.config.executable,
                "exec",
                self.container_id,
                "env",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            for line in result.stdout.splitlines():
                if line.startswith("PATH="):
                    self._container_path = line[len("PATH=") :]
                    break
        except subprocess.CalledProcessError as exc:
            self.logger.debug(f"Failed to capture container environment: {exc}")

    def _prepare_shell_command(self, command: str) -> str:
        shell_command = command
        if self._container_path:
            safe_path = self._container_path.replace('"', r"\"")
            shell_command = f'export PATH="{safe_path}:$PATH"; {command}'
        return shell_command

    def _build_exec_command(
        self,
        cwd: str,
        *,
        use_stdin: bool,
        shell_command: str | None = None,
    ) -> list[str]:
        cmd = [self.config.executable, "exec"]
        if use_stdin:
            cmd.append("-i")
        cmd.extend(["-w", cwd])
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.append(self.container_id)
        view = getattr(self, "_resource_view", None)
        if view is not None:
            # Rootless Docker may lack cpuset delegation. Set affinity for every
            # agent exec; descendants inherit it. CPU-time quotas remain enforced.
            cmd.extend(["taskset", "--cpu-list", ",".join(map(str, view.cpus))])
        if use_stdin:
            cmd.extend(["bash", "-l", "-s"])
        else:
            if shell_command is None:
                raise ValueError(
                    "shell_command must be provided when use_stdin is False"
                )
            cmd.extend(["bash", "-lc", shell_command])
        return cmd

    def _should_use_stdin(self, cmd: list[str]) -> bool:
        arg_max: int | None = None
        if hasattr(os, "sysconf"):
            try:
                arg_max = os.sysconf("SC_ARG_MAX")
            except (AttributeError, OSError, ValueError):
                arg_max = None
        if not arg_max:
            arg_max = 2 * 1024 * 1024
        cmd_length = sum(len(arg) + 1 for arg in cmd)
        env_length = 0
        for key, value in os.environ.items():
            env_length += len(os.fsencode(key)) + len(os.fsencode(value)) + 2
        return cmd_length + env_length >= arg_max - 4096

    def _stdin_payload(self, shell_command: str) -> str:
        if shell_command.endswith("\n"):
            return shell_command
        return f"{shell_command}\n"

    def execute(self, command: str, cwd: str = "", timeout=True) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict.

        `timeout` is either a flag or an explicit number of seconds:
          - True  -> use config.timeout (per-exec default)
          - False -> use container_timeout_seconds() (the full container lifetime)
          - int/float -> use that many seconds directly
        """
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        if timeout is True:
            exec_timeout = self.config.timeout
        elif timeout is False:
            exec_timeout = self.config.container_timeout_seconds()
        else:
            exec_timeout = timeout

        shell_command = self._prepare_shell_command(command)
        cmd = self._build_exec_command(
            cwd, use_stdin=False, shell_command=shell_command
        )
        use_stdin = self._should_use_stdin(cmd)
        if use_stdin:
            cmd = self._build_exec_command(cwd, use_stdin=True)
            exec_input = self._stdin_payload(shell_command)
            self.logger.debug(
                "Executing command in container %s via stdin: %s",
                self.container_id,
                shlex.join(cmd),
            )
        else:
            exec_input = None
            self.logger.debug(
                "Executing command in container %s: %s",
                self.container_id,
                shlex.join(cmd),
            )
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=exec_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                input=exec_input,
            )
        except OSError as exc:
            if exc.errno != errno.E2BIG or use_stdin:
                raise
            cmd = self._build_exec_command(cwd, use_stdin=True)
            exec_input = self._stdin_payload(shell_command)
            self.logger.warning(
                "Command too long for exec args in container %s; retrying via stdin",
                self.container_id,
            )
            self.logger.debug(
                "Executing command in container %s via stdin: %s",
                self.container_id,
                shlex.join(cmd),
            )
            result = subprocess.run(
                cmd,
                text=True,
                timeout=exec_timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                input=exec_input,
            )
        self.logger.debug(
            f"Command output in container {self.container_id}: {result.stdout}"
        )

        return {"output": result.stdout, "returncode": result.returncode}

    def execute_stream(
        self,
        command: str,
        cwd: str = "",
        timeout=True,
        on_output: Callable[[str], None] | None = None,
        capture_output: bool = True,
    ) -> dict[str, Any]:
        """Execute a command in the Docker container and stream its output.

        Set ``capture_output=False`` for machine-readable streams whose caller
        consumes every line in ``on_output``. This keeps memory bounded for
        multi-gigabyte preprocessing output while preserving the historical
        buffered behavior by default.
        """
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        if timeout is True:
            exec_timeout = self.config.timeout
        elif timeout is False:
            exec_timeout = self.config.container_timeout_seconds()
        else:
            exec_timeout = timeout

        shell_command = self._prepare_shell_command(command)
        cmd = self._build_exec_command(
            cwd, use_stdin=False, shell_command=shell_command
        )
        use_stdin = self._should_use_stdin(cmd)
        if use_stdin:
            cmd = self._build_exec_command(cwd, use_stdin=True)
            stdin_payload = self._stdin_payload(shell_command)
            self.logger.debug(
                "Executing command in container %s via stdin: %s",
                self.container_id,
                shlex.join(cmd),
            )
        else:
            stdin_payload = None
            self.logger.debug(
                "Executing command in container %s: %s",
                self.container_id,
                shlex.join(cmd),
            )
        try:
            process = subprocess.Popen(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                stdin=subprocess.PIPE if use_stdin else None,
            )
        except OSError as exc:
            if exc.errno != errno.E2BIG or use_stdin:
                raise
            use_stdin = True
            cmd = self._build_exec_command(cwd, use_stdin=True)
            stdin_payload = self._stdin_payload(shell_command)
            self.logger.warning(
                "Command too long for exec args in container %s; retrying via stdin",
                self.container_id,
            )
            self.logger.debug(
                "Executing command in container %s via stdin: %s",
                self.container_id,
                shlex.join(cmd),
            )
            process = subprocess.Popen(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                stdin=subprocess.PIPE,
            )
        output_lines: list[str] = []
        timed_out = False

        def reader() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                if capture_output:
                    output_lines.append(line)
                if on_output:
                    try:
                        on_output(line)
                    except Exception:
                        self.logger.exception(
                            "Streaming output callback failed; continuing to drain stdout."
                        )

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        if use_stdin and process.stdin:
            try:
                process.stdin.write(stdin_payload)
                process.stdin.close()
            except Exception as exc:
                self.logger.debug("Failed to send stdin payload: %s", exc)
        try:
            process.wait(timeout=exec_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.logger.warning("Command timed out in container %s", self.container_id)
            process.kill()
            process.wait(timeout=5)
        thread.join(timeout=1)
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass

        return {
            "output": "".join(output_lines),
            "returncode": process.returncode,
            "timed_out": timed_out,
        }

    def execute_native_capture(
        self,
        command: str,
        *,
        capture_dir: str | Path,
        capture_backend: str = "leanlean",
        harness: str,
        model: str,
        forwards_subagent_stream: bool,
        native_edit_payloads_exact: bool | None = None,
        edit_payload_evidence: str | None = None,
        cwd: str = "",
        timeout: bool | int | float = False,
    ) -> dict[str, Any]:
        """Run once while copying exact stdout/stderr bytes to host artifacts.

        There is deliberately no line callback or semantic parser. Provider
        decoding starts only after this method has reaped the child and
        published ``capture.json``.
        """

        cwd = cwd or self.config.cwd
        if not self.container_id:
            raise RuntimeError("native capture requires a running container")
        if timeout is True:
            exec_timeout = self.config.timeout
        elif timeout is False:
            exec_timeout = self.config.container_timeout_seconds()
        else:
            exec_timeout = float(timeout)

        if capture_backend not in {"leanlean", "trace_utils"}:
            raise ValueError(
                "capture_backend must be 'leanlean' or 'trace_utils'"
            )
        directory = Path(capture_dir).expanduser().resolve(strict=False)
        started_at = datetime.now(timezone.utc)
        writer: Any | None = None
        if capture_backend == "trace_utils":
            try:
                native_capture = importlib.import_module(
                    "harness_wrapper.native_capture"
                )
                writer_type = native_capture.NativeCaptureWriter
            except (AttributeError, ImportError) as exc:
                raise RuntimeError(
                    "trace_utils capture requires harness-wrapper on PYTHONPATH"
                ) from exc
            writer = writer_type(
                root=directory.parent,
                capture_id=directory.name,
                harness=harness,
                model=model,
                forwards_subagent_stream=forwards_subagent_stream,
                native_edit_payloads_exact=(
                    harness == "claude-code"
                    if native_edit_payloads_exact is None
                    else native_edit_payloads_exact
                ),
                edit_payload_evidence=(
                    edit_payload_evidence
                    or (
                        "stream-json tool_use inputs retain exact "
                        "Edit/Write/Bash arguments"
                        if harness == "claude-code"
                        else (
                            "codex exec --json file_change items retain path/kind "
                            "but not diff; exact edits require the supplemental "
                            "native rollout tree"
                        )
                    )
                ),
                started_at=started_at,
            )
            stdout_path = writer.stdout_path
            stderr_path = writer.stderr_path
        else:
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(directory, 0o700)
            stdout_path = directory / "native.stdout.jsonl"
            stderr_path = directory / "native.stderr.log"

        shell_command = self._prepare_shell_command(command)
        cmd = self._build_exec_command(
            cwd, use_stdin=False, shell_command=shell_command
        )
        use_stdin = self._should_use_stdin(cmd)
        stdin_payload: bytes | None = None
        if use_stdin:
            cmd = self._build_exec_command(cwd, use_stdin=True)
            stdin_payload = self._stdin_payload(shell_command).encode("utf-8")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if use_stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        stdout_digest = hashlib.sha256()
        stderr_digest = hashlib.sha256()
        stream_bytes = {"stdout": 0, "stderr": 0}
        drain_errors: list[tuple[str, BaseException]] = []

        def drain(stream: Any, path: Path, digest: Any, label: str) -> None:
            try:
                if writer is not None:
                    append = (
                        writer.append_stdout
                        if label == "stdout"
                        else writer.append_stderr
                    )
                    while chunk := stream.read(1024 * 1024):
                        append(chunk)
                    return
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb", buffering=1024 * 1024) as output:
                    while chunk := stream.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        stream_bytes[label] += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException as exc:  # fail closed across drain threads
                drain_errors.append((label, exc))

        assert process.stdout is not None and process.stderr is not None
        stdout_thread = threading.Thread(
            target=drain,
            args=(process.stdout, stdout_path, stdout_digest, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=drain,
            args=(process.stderr, stderr_path, stderr_digest, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        if stdin_payload is not None and process.stdin is not None:
            process.stdin.write(stdin_payload)
            process.stdin.close()
        timed_out = False
        try:
            process.wait(timeout=exec_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=5)
        stdout_thread.join()
        stderr_thread.join()
        if drain_errors:
            details = "; ".join(f"{name}: {error}" for name, error in drain_errors)
            if writer is not None:
                writer.finish(
                    returncode=process.returncode,
                    session_id=None,
                    normalization_complete=False,
                    error=details,
                )
            raise RuntimeError(f"native stream capture failed: {details}")
        if writer is not None:
            capture_result = writer.finish(
                returncode=process.returncode,
                session_id=None,
                normalization_complete=False,
            )
            manifest_path = capture_result.manifest_path
            stdout_path = capture_result.stdout_path
            stderr_path = capture_result.stderr_path
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        if native_edit_payloads_exact is None:
            native_edit_payloads_exact = harness == "claude-code"
        if edit_payload_evidence is None:
            edit_payload_evidence = (
                "stream-json tool_use inputs retain exact Edit/Write/Bash arguments"
                if native_edit_payloads_exact
                else (
                    "codex exec --json file_change items retain path/kind but not "
                    "diff; exact edits require the supplemental native rollout tree"
                )
            )
        if writer is None:
            manifest = {
                "format": "code-harness-native-capture-v1",
                "capture_id": directory.name,
                "harness": harness,
                "model": model,
                "session_id": None,
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": process.returncode,
                "timed_out": timed_out,
                "capture_mode": "post_exit",
                "normalization_phase": "post_exit",
                "normalization_complete": False,
                "native_stream_parsed_during_agent_execution": False,
                "forwards_subagent_stream": forwards_subagent_stream,
                "native_edit_payloads_exact": native_edit_payloads_exact,
                "edit_payload_evidence": edit_payload_evidence,
                "stdout": {
                    "path": stdout_path.name,
                    "format": "provider-native-jsonl-v1",
                    "sha256": stdout_digest.hexdigest(),
                    "bytes": stream_bytes["stdout"],
                },
                "stderr": {
                    "path": stderr_path.name,
                    "format": "provider-native-stderr-v1",
                    "sha256": stderr_digest.hexdigest(),
                    "bytes": stream_bytes["stderr"],
                },
            }
            manifest_path = directory / "capture.json"
            temporary = directory / ".capture.json.tmp"
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            with temporary.open("rb") as source:
                os.fsync(source.fileno())
            os.replace(temporary, manifest_path)
        return {
            "output": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "returncode": process.returncode,
            "timed_out": timed_out,
            "capture_backend": capture_backend,
            "native_capture_manifest": str(manifest_path),
        }

    def quiesce_after_agent_exit(self, *, timeout: int = 15) -> None:
        """Terminate leftover exec processes before terminal source capture.

        The primary streamed CLI has already exited (or reached its configured
        timeout) when this is called. Detached tool children could otherwise
        continue writing while the post-exit archive is streamed. PID 1 is the
        container lifetime ``sleep`` and the helper excludes itself; every
        other process belongs to the completed agent execution.
        """

        command = r"""
self=$$
is_zombie() {
  while read -r key value rest; do
    if [ "$key" = "State:" ]; then
      [ "$value" = "Z" ]
      return
    fi
  done < "$1/status"
  return 1
}
for proc in /proc/[0-9]*; do
  pid=${proc#/proc/}
  case "$pid" in 1|"$self") continue ;; esac
  is_zombie "$proc" && continue
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 2
for proc in /proc/[0-9]*; do
  pid=${proc#/proc/}
  case "$pid" in 1|"$self") continue ;; esac
  is_zombie "$proc" && continue
  kill -KILL "$pid" 2>/dev/null || true
done
for proc in /proc/[0-9]*; do
  pid=${proc#/proc/}
  case "$pid" in 1|"$self") continue ;; esac
  is_zombie "$proc" && continue
  exit 1
done
""".strip()
        result = self.execute(command, timeout=timeout)
        if result.get("returncode"):
            raise RuntimeError(
                "Could not quiesce live agent descendants before terminal capture: "
                + result.get("output", "")[-4000:]
            )

    def read_file(self, filename: str) -> str:
        """Read a file from the container and return its content.

        Args:
            filename: The path to the file to read.

        Returns:
            The content of the file as a string, or an empty string if the file does not exist.
        """
        result = self.execute(f"cat {shlex.quote(filename)}")
        if result["returncode"] != 0:
            self.logger.warning(
                f"Failed to read file {filename} in container {self.container_id}"
            )
            return ""
        return result["output"]

    def write_file(self, path: str, file_content: str) -> None:
        """Write content to a file in the container.

        Args:
            path: The path to the file to write.
            file_content: The content to write.
        """
        # Resolve path to absolute using posixpath to ensure Linux separators
        if not posixpath.isabs(path):
            container_path = posixpath.join(self.config.cwd, path)
        else:
            container_path = path

        # Create parent directory if needed
        parent_dir = posixpath.dirname(container_path)
        if parent_dir:
            self.execute(f"mkdir -p {shlex.quote(parent_dir)}")

        self.execute(
            f"echo {shlex.quote(file_content)} > {shlex.quote(container_path)}"
        )

    def write_file_bytes(self, path: str, file_content: bytes) -> None:
        """Write exact bytes to a container file over stdin.

        Unlike :meth:`write_file`, this preserves trailing newlines and never
        places file contents in a Docker command argument or log line. Passive
        action replay uses it for Claude ``Edit`` and ``Write`` actions.
        """

        if not isinstance(file_content, bytes):
            raise TypeError("file_content must be bytes")
        if ".." in path.split("/"):
            raise ValueError(f"Unsafe container write path: {path!r}")
        if not posixpath.isabs(path):
            container_path = posixpath.join(self.config.cwd, path)
        else:
            container_path = posixpath.normpath(path)

        parent_dir = posixpath.dirname(container_path)
        if parent_dir:
            created = self.execute(
                f"mkdir -p -- {shlex.quote(parent_dir)}",
                timeout=True,
            )
            if created.get("returncode"):
                raise RuntimeError(
                    "Could not create replay file parent: "
                    + created.get("output", "")[-4000:]
                )

        command = [
            self.config.executable,
            "exec",
            "-i",
            "-w",
            self.config.cwd,
            self.container_id,
            "sh",
            "-c",
            # Claude Code's native Write tool creates ordinary source files
            # under the container's default 022 umask. The archive digest
            # includes mode bits, so replay must reproduce 0644 rather than a
            # private 0600 file for newly created paths.
            f"umask 022 && cat > {shlex.quote(container_path)}",
        ]
        written = subprocess.run(
            command,
            input=file_content,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.config.timeout,
        )
        if written.returncode != 0:
            raise RuntimeError(
                "Could not write replay file: "
                + written.stdout.decode("utf-8", errors="replace")[-4000:]
            )

    def commit(self, image_tag: str) -> bool:
        """Commit the current container state as a Docker image.

        Returns True on success, False on failure.
        """
        if not self.container_id:
            return False
        try:
            cmd = [self.config.executable, "commit", self.container_id, image_tag]
            self.logger.info(
                "Committing container %s as %s", self.container_id, image_tag
            )
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            self.logger.info("Committed image %s", image_tag)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.warning("Failed to commit container: %s", e)
            return False

    @staticmethod
    def _validate_archive_paths(paths: list[str]) -> list[str]:
        """Validate repository-relative source paths used by playback archives."""

        normalized: list[str] = []
        for raw_path in paths:
            path = str(raw_path).strip().rstrip("/") or "."
            if path == ".":
                normalized.append(path)
                continue
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe playback archive path: {raw_path!r}")
            normalized.append(candidate.as_posix())
        if not normalized:
            raise ValueError("Playback archive requires at least one source path")
        if "." in normalized and normalized != ["."]:
            raise ValueError("Whole-repository playback path cannot be combined")
        return normalized

    def _export_source_archive_once(
        self,
        destination: str | Path,
        paths: list[str],
        *,
        timeout: int = 600,
    ) -> dict[str, Any]:
        """Stream deterministic sources to the host without container writes."""

        if not self.container_id:
            raise RuntimeError("Cannot export playback state without a container")
        members = self._validate_archive_paths(paths)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        member_setup = " ".join(
            (
                f"if test -f {shlex.quote(member)} || "
                f"test -L {shlex.quote(member)}; then "
                f"printf '%s\\0' {shlex.quote(member)}; "
                f"elif test -d {shlex.quote(member)}; then "
                f"find {shlex.quote(member)} "
                "\\( -name .git -o -name .lake \\) -prune -o "
                "\\( -type f -o -type l \\) -print0; fi;"
            )
            for member in members
        )
        script = (
            "set -o pipefail; cd /testbed; { "
            + member_setup
            + " } | LC_ALL=C sort -zu | "
            "tar --sort=name --format=posix --mtime='@0' "
            "--owner=0 --group=0 --numeric-owner "
            "--pax-option=delete=atime,delete=ctime "
            "--exclude=.git --exclude='./.git' --exclude='*/.git' "
            "--exclude=.lake --exclude='./.lake' --exclude='*/.lake' "
            "--no-recursion --null -cf - --files-from=-"
        )
        command = [
            self.config.executable,
            "exec",
            self.container_id,
            "bash",
            "-lc",
            script,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        expired = threading.Event()

        def kill_expired_export() -> None:
            expired.set()
            process.kill()

        timer = threading.Timer(timeout, kill_expired_export)
        timer.daemon = True
        timer.start()
        digest = hashlib.sha256()
        source_bytes = 0
        try:
            assert process.stdout is not None
            with destination.open("wb") as raw_output:
                with gzip.GzipFile(
                    fileobj=raw_output,
                    mode="wb",
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    while chunk := process.stdout.read(1024 * 1024):
                        digest.update(chunk)
                        source_bytes += len(chunk)
                        compressed.write(chunk)
            process.wait()
            if expired.is_set():
                raise subprocess.TimeoutExpired(command, timeout)
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            destination.unlink(missing_ok=True)
            raise
        finally:
            timer.cancel()
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr is not None
            else ""
        )
        if process.returncode != 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not export playback source archive: " + stderr[-4000:]
            )
        return {
            "source_archive_sha256": digest.hexdigest(),
            "source_archive_bytes": source_bytes,
            "archive_bytes": destination.stat().st_size,
        }

    def export_source_archive(
        self,
        destination: str | Path,
        paths: list[str],
        *,
        timeout: int = 600,
        retry_transient: bool = True,
    ) -> dict[str, Any]:
        """Export a stable source snapshot, retrying transient tar races.

        Directories become an explicit sorted file/symlink list, so root
        metadata changes cannot invalidate the archive. Replay capture may
        retry an actual file mutation; deterministic preprocessing metrics set
        ``retry_transient=False`` and fail on the first mutation.
        """

        retry_markers = (
            "file changed as we read it",
            "File changed as we read it",
            "file removed before we read it",
            "File removed before we read it",
        )
        attempts = 3 if retry_transient else 1
        for attempt in range(1, attempts + 1):
            try:
                return self._export_source_archive_once(
                    destination, paths, timeout=timeout
                )
            except RuntimeError as error:
                retryable = any(marker in str(error) for marker in retry_markers)
                if not retryable or attempt == attempts:
                    raise
                self.logger.warning(
                    "Source archive changed during export; retrying clean "
                    "snapshot (%d/%d): %s",
                    attempt, attempts, error,
                )
                time.sleep(0.2 * attempt)
        raise AssertionError("unreachable")

    def export_directory_archive(
        self,
        destination: str | Path,
        directory: str,
        *,
        timeout: int = 3600,
        attempts: int = 5,
    ) -> dict[str, Any]:
        """Stream one container directory into a compressed, hashed host archive.

        tar exits 1 on "file changed as we read it" when something in the
        container is still writing under the directory; the archive is then
        discarded and the export retried once writes have had time to settle.
        """

        for attempt in range(1, attempts + 1):
            try:
                return self._export_directory_archive_once(
                    destination, directory, timeout=timeout
                )
            except RuntimeError as error:
                if attempt == attempts or "file changed as we read it" not in str(error):
                    raise
                self.logger.warning(
                    "Directory %s changed during export (attempt %d/%d); retrying",
                    directory, attempt, attempts,
                )
                time.sleep(15 * attempt)
        raise AssertionError("unreachable")

    def _export_directory_archive_once(
        self,
        destination: str | Path,
        directory: str,
        *,
        timeout: int,
    ) -> dict[str, Any]:

        if not self.container_id:
            raise RuntimeError("Cannot export a directory without a live container")
        normalized = posixpath.normpath(directory)
        if (
            not normalized.startswith("/testbed/")
            or normalized == "/testbed/"
            or "/../" in f"{normalized}/"
        ):
            raise ValueError(f"directory export must be inside /testbed: {directory!r}")
        relative = normalized.removeprefix("/testbed/")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            destination.name + ".partial-" + uuid.uuid4().hex
        )
        script = (
            "set -euo pipefail; "
            f"test -d {shlex.quote(relative)}; "
            f"find {shlex.quote(relative)} -type f -print -quit | grep -q .; "
            "tar --sort=name --format=posix --mtime='@0' "
            "--owner=0 --group=0 --numeric-owner "
            "--pax-option=delete=atime,delete=ctime "
            f"-C /testbed -cf - -- {shlex.quote(relative)}"
        )
        command = [
            self.config.executable,
            "exec",
            self.container_id,
            "bash",
            "-lc",
            script,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        expired = threading.Event()

        def kill_expired_export() -> None:
            expired.set()
            process.kill()

        timer = threading.Timer(timeout, kill_expired_export)
        timer.daemon = True
        timer.start()
        tree_digest = hashlib.sha256()
        tree_bytes = 0
        try:
            assert process.stdout is not None
            with temporary.open("wb") as raw_output:
                with gzip.GzipFile(
                    fileobj=raw_output,
                    mode="wb",
                    compresslevel=1,
                    mtime=0,
                ) as compressed:
                    while chunk := process.stdout.read(1024 * 1024):
                        tree_digest.update(chunk)
                        tree_bytes += len(chunk)
                        compressed.write(chunk)
            process.wait()
            if expired.is_set():
                raise subprocess.TimeoutExpired(command, timeout)
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            temporary.unlink(missing_ok=True)
            raise
        finally:
            timer.cancel()
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr is not None
            else ""
        )
        if process.returncode != 0 or tree_bytes == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not export container directory archive: " + stderr[-4000:]
            )
        os.replace(temporary, destination)
        archive_digest = hashlib.sha256()
        with destination.open("rb") as archive:
            while chunk := archive.read(1024 * 1024):
                archive_digest.update(chunk)
        return {
            "archive_sha256": archive_digest.hexdigest(),
            "archive_bytes": destination.stat().st_size,
            "tree_tar_sha256": tree_digest.hexdigest(),
            "tree_tar_bytes": tree_bytes,
        }

    def container_upper_dir(self) -> Path:
        """Return the host-visible writable layer for the running container.

        This is intentionally a host-side Docker inspection. It does not
        execute a process, mount a helper, or write a marker in the task
        container. Live source journaling uses the returned upper directory
        after taking its baseline before the model process starts.
        """

        if not self.container_id:
            raise RuntimeError("Cannot inspect storage without a container")
        inspected = subprocess.run(
            [
                self.config.executable,
                "inspect",
                "--format",
                "{{json .GraphDriver.Data}}",
                self.container_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            text=True,
        )
        if inspected.returncode != 0:
            raise RuntimeError(
                "Could not inspect container storage: "
                + inspected.stderr.strip()[-4000:]
            )
        try:
            graph_data = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Docker returned malformed graph-driver data") from exc
        upper_dir = (
            graph_data.get("UpperDir") if isinstance(graph_data, dict) else None
        )
        if not isinstance(upper_dir, str) or not upper_dir:
            raise RuntimeError(
                "Docker graph driver does not expose a host-visible UpperDir"
            )
        resolved = Path(upper_dir).expanduser().resolve(strict=False)
        if not resolved.is_dir():
            raise RuntimeError(
                f"Container UpperDir is not host-readable: {resolved}"
            )
        return resolved

    def export_codex_rollout(
        self,
        destination: str | Path,
        *,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Stream the newest native Codex rollout to a host gzip artifact.

        Only ``rollout-*.jsonl`` regular files below the fixed temporary
        ``CODEX_HOME/sessions`` directory are eligible.  The auth cache and
        configuration are never copied.  Every non-empty line is parsed as
        JSON before the artifact is accepted, and the first event must be a
        Codex ``session_meta`` record.
        """

        if not self.container_id:
            raise RuntimeError("Cannot export a Codex rollout without a container")
        sessions_root = "/tmp/leanlean-codex-home/sessions"
        discovered = subprocess.run(
            [
                self.config.executable,
                "exec",
                self.container_id,
                "find",
                sessions_root,
                "-type",
                "f",
                "-name",
                "rollout-*.jsonl",
                "-printf",
                "%T@\\t%p\\0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if discovered.returncode != 0:
            error = discovered.stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError("Could not discover native Codex rollout: " + error)

        candidates: list[tuple[float, str]] = []
        for raw_entry in discovered.stdout.split(b"\0"):
            if not raw_entry:
                continue
            try:
                raw_timestamp, raw_path = raw_entry.split(b"\t", 1)
                timestamp = float(raw_timestamp)
                path = raw_path.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError("Malformed native Codex rollout path") from exc
            normalized = posixpath.normpath(path)
            basename = posixpath.basename(normalized)
            if (
                not normalized.startswith(sessions_root + "/")
                or not basename.startswith("rollout-")
                or not basename.endswith(".jsonl")
            ):
                raise RuntimeError("Unsafe native Codex rollout path")
            candidates.append((timestamp, normalized))
        if not candidates:
            raise RuntimeError("Codex produced no persisted native rollout")
        _, rollout_path = max(candidates)

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.executable,
            "exec",
            self.container_id,
            "cat",
            "--",
            rollout_path,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        expired = threading.Event()

        def kill_expired_export() -> None:
            expired.set()
            process.kill()

        timer = threading.Timer(timeout, kill_expired_export)
        timer.daemon = True
        timer.start()
        digest = hashlib.sha256()
        source_bytes = 0
        event_count = 0
        tool_call_count = 0
        apply_patch_count = 0
        session_id = ""
        cli_version = ""
        try:
            assert process.stdout is not None
            with destination.open("wb") as raw_output:
                with gzip.GzipFile(
                    fileobj=raw_output,
                    mode="wb",
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    for line in iter(process.stdout.readline, b""):
                        if line.strip():
                            try:
                                event = json.loads(line)
                            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                raise RuntimeError(
                                    "Malformed JSON in native Codex rollout"
                                ) from exc
                            if event_count == 0:
                                if (
                                    not isinstance(event, dict)
                                    or event.get("type") != "session_meta"
                                ):
                                    raise RuntimeError(
                                        "Native Codex rollout has no session_meta header"
                                    )
                                payload = event.get("payload") or {}
                                if isinstance(payload, dict):
                                    session_id = str(
                                        payload.get("session_id")
                                        or payload.get("id")
                                        or ""
                                    )
                                    cli_version = str(payload.get("cli_version") or "")
                            event_count += 1
                            payload = (
                                event.get("payload")
                                if isinstance(event, dict)
                                else None
                            )
                            if isinstance(payload, dict) and payload.get("type") in {
                                "custom_tool_call",
                                "function_call",
                            }:
                                tool_call_count += 1
                                if payload.get("name") == "apply_patch":
                                    apply_patch_count += 1
                        digest.update(line)
                        source_bytes += len(line)
                        compressed.write(line)
            process.wait()
            if expired.is_set():
                raise subprocess.TimeoutExpired(command, timeout)
            if not event_count:
                raise RuntimeError("Native Codex rollout is empty")
        except BaseException:
            process.kill()
            process.wait(timeout=10)
            destination.unlink(missing_ok=True)
            raise
        finally:
            timer.cancel()
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr is not None
            else ""
        )
        if process.returncode != 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not export native Codex rollout: " + stderr[-4000:]
            )
        return {
            "rollout_sha256": digest.hexdigest(),
            "rollout_bytes": source_bytes,
            "archive_bytes": destination.stat().st_size,
            "event_count": event_count,
            "tool_call_count": tool_call_count,
            "apply_patch_count": apply_patch_count,
            "session_id": session_id,
            "cli_version": cli_version,
            "source_filename": posixpath.basename(rollout_path),
        }

    def export_codex_rollout_bundle(
        self,
        destination: str | Path,
        *,
        fresh_home_proven: bool,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Export every root/descendant rollout from an isolated Codex home."""

        if not fresh_home_proven:
            raise RuntimeError(
                "complete Codex lineage capture requires a proven fresh CODEX_HOME"
            )
        if not self.container_id:
            raise RuntimeError("Cannot export Codex rollouts without a container")
        sessions_root = "/tmp/leanlean-codex-home/sessions"
        discovered = subprocess.run(
            [
                self.config.executable,
                "exec",
                self.container_id,
                "find",
                sessions_root,
                "-type",
                "f",
                "-name",
                "rollout-*.jsonl",
                "-print0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if discovered.returncode != 0:
            error = discovered.stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError("Could not discover Codex rollout tree: " + error)
        paths = []
        for raw_path in discovered.stdout.split(b"\0"):
            if not raw_path:
                continue
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError("Malformed Codex rollout path") from exc
            normalized = posixpath.normpath(path)
            basename = posixpath.basename(normalized)
            if (
                not normalized.startswith(sessions_root + "/")
                or not basename.startswith("rollout-")
                or not basename.endswith(".jsonl")
            ):
                raise RuntimeError("Unsafe Codex rollout path")
            paths.append(normalized)
        if not paths:
            raise RuntimeError("Codex produced no persisted native rollouts")

        directory = Path(destination).expanduser().resolve(strict=False)
        directory.mkdir(parents=True, exist_ok=False)
        streams = []
        seen_threads: set[str] = set()
        seen_thread_payloads: dict[str, tuple[str, str]] = {}
        conflicts: list[str] = []
        for stream_index, rollout_path in enumerate(sorted(paths), 1):
            copied = subprocess.run(
                [
                    self.config.executable,
                    "exec",
                    self.container_id,
                    "cat",
                    "--",
                    rollout_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if copied.returncode != 0:
                error = copied.stderr.decode("utf-8", errors="replace")[-4000:]
                raise RuntimeError("Could not read Codex rollout: " + error)
            payload = copied.stdout
            # Persist every byte stream before parsing or lineage validation.
            # A future schema change must never recreate the old failure mode
            # where validation discarded the very artifact needed to recover.
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            filename = f"{stream_index:04d}-{payload_sha256[:16]}.jsonl.gz"
            archive_path = directory / filename
            with archive_path.open("wb") as raw:
                with gzip.GzipFile(
                    fileobj=raw,
                    mode="wb",
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    compressed.write(payload)
            events = []
            for line in payload.splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Malformed JSON in Codex rollout tree") from exc
                if not isinstance(event, dict):
                    raise RuntimeError("Non-object event in Codex rollout tree")
                events.append(event)
            if not events or events[0].get("type") != "session_meta":
                raise RuntimeError("Codex rollout has no session_meta header")
            header = events[0].get("payload")
            if not isinstance(header, dict):
                raise RuntimeError("Codex rollout session_meta payload is malformed")
            # session_id identifies the logical root session and is shared by
            # descendant Codex threads. id is the unique rollout/thread
            # identity. Treating the former as a stream key rejects real
            # subagents as duplicate roots.
            thread_id = str(header.get("id") or header.get("session_id") or "")
            logical_session_id = str(header.get("session_id") or thread_id)
            parent_id = header.get("parent_thread_id")
            parent_id = str(parent_id) if parent_id else None
            try:
                uuid.UUID(thread_id)
                uuid.UUID(logical_session_id)
            except ValueError as exc:
                raise RuntimeError("Malformed Codex rollout thread/session ID") from exc
            if thread_id in seen_threads:
                prior_sha256, prior_path = seen_thread_payloads[thread_id]
                if payload_sha256 != prior_sha256:
                    conflicts.append(
                        "Conflicting Codex rollouts share one thread ID: "
                        f"{prior_path} and {rollout_path}"
                    )
                else:
                    # Byte-identical aliases add no evidence. Keep one
                    # canonical manifest entry; both raw aliases remain saved.
                    continue
            seen_threads.add(thread_id)
            seen_thread_payloads[thread_id] = (payload_sha256, rollout_path)
            streams.append(
                {
                    "path": filename,
                    "sha256": payload_sha256,
                    "bytes": len(payload),
                    "archive_bytes": archive_path.stat().st_size,
                    "event_count": len(events),
                    # session_id remains a backward-compatible alias for the
                    # per-thread identity used by v1 consumers.
                    "session_id": thread_id,
                    "thread_id": thread_id,
                    "logical_session_id": logical_session_id,
                    "parent_thread_id": parent_id,
                    "source_filename": posixpath.basename(rollout_path),
                }
            )
        roots = [stream for stream in streams if stream["parent_thread_id"] is None]
        missing_parents = [
            stream["thread_id"]
            for stream in streams
            if stream["parent_thread_id"] is not None
            and stream["parent_thread_id"] not in seen_threads
        ]
        logical_session_ids = {
            stream["logical_session_id"] for stream in streams
        }
        validation_errors = list(conflicts)
        if len(roots) != 1 or missing_parents:
            validation_errors.append(
                "Codex rollout files do not form one complete rooted session tree"
            )
        if len(logical_session_ids) != 1:
            validation_errors.append(
                "Codex rollout threads do not share one logical session ID"
            )
        if roots and roots[0]["thread_id"] != roots[0]["logical_session_id"]:
            validation_errors.append(
                "Codex root thread ID differs from its logical session ID"
            )
        incomplete_manifest = {
            "format": "codex-native-rollout-bundle-v1",
            "capture_phase": "post_exit",
            "all_fresh_home_sessions_captured": False,
            "session_count": len(streams),
            "streams": streams,
            "root_candidates": [stream["thread_id"] for stream in roots],
            "missing_parent_thread_ids": missing_parents,
            "capture_errors": validation_errors,
        }
        if validation_errors:
            incomplete_path = directory / "rollouts.incomplete.json"
            incomplete_path.write_text(
                json.dumps(incomplete_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("; ".join(validation_errors))
        manifest = {
            "format": "codex-native-rollout-bundle-v1",
            "capture_phase": "post_exit",
            "all_fresh_home_sessions_captured": True,
            "root_session_id": roots[0]["thread_id"],
            "logical_session_id": roots[0]["logical_session_id"],
            "session_count": len(streams),
            "streams": streams,
        }
        manifest_path = directory / "rollouts.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {**manifest, "manifest_path": str(manifest_path)}

    def restore_source_archive(
        self,
        archive: str | Path,
        paths: list[str],
        *,
        timeout: int = 600,
    ) -> None:
        """Replace a replay container's scoped sources from a host archive."""

        if not self.container_id:
            raise RuntimeError("Cannot restore playback state without a container")
        members = self._validate_archive_paths(paths)
        if members == ["."]:
            clean = (
                "find /testbed -mindepth 1 -maxdepth 1 "
                "! -name .git ! -name .lake -exec rm -rf -- {} +"
            )
        else:
            targets = " ".join(shlex.quote(f"/testbed/{member}") for member in members)
            clean = f"rm -rf -- {targets}"
        cleaned: dict[str, Any] = {"returncode": 1, "output": "not attempted"}
        cleanup_attempts = 10
        for attempt in range(1, cleanup_attempts + 1):
            cleaned = self.execute(clean, timeout=timeout)
            if not cleaned.get("returncode"):
                break
            output = str(cleaned.get("output") or "")
            if "Directory not empty" not in output or attempt == cleanup_attempts:
                raise RuntimeError(
                    "Could not clean replay source scope: " + output[-4000:]
                )
            self.logger.warning(
                "Replay source cleanup raced with a live container process; "
                "quiescing and retrying (%d/%d): %s",
                attempt,
                cleanup_attempts,
                output.strip()[-1000:],
            )
            self.quiesce_after_agent_exit(timeout=min(timeout, 15))
            time.sleep(min(0.5 * (2 ** (attempt - 1)), 5.0))
        if cleaned.get("returncode"):
            raise RuntimeError(
                "Could not clean replay source scope after retries: "
                + str(cleaned.get("output") or "")[-4000:]
            )

        command = [
            self.config.executable,
            "exec",
            "-i",
            self.container_id,
            "tar",
            "--no-same-owner",
            "-C",
            "/testbed",
            "-xzf",
            "-",
        ]
        with Path(archive).open("rb") as source:
            restored = subprocess.run(
                command,
                stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if restored.returncode != 0:
            raise RuntimeError(
                "Could not restore replay source archive: "
                + restored.stdout.decode("utf-8", errors="replace")[-4000:]
            )

    def spawn_replay_environment(self, *, lifetime_seconds: int) -> "DockerEnvironment":
        """Start a disposable container from the same pinned image for replay."""

        return type(self)(
            config_class=type(self.config),
            logger=self.logger,
            image=self.container_image_id(),
            cwd="/testbed",
            env={},
            forward_env=[],
            timeout=self.config.timeout,
            executable=self.config.executable,
            run_args=list(self.config.run_args),
            resource_visibility=self.config.resource_visibility,
            resource_threads=self.config.resource_threads,
            network_mode="none",
            network_policy="none",
            container_pids_limit=self.config.container_pids_limit,
            container_timeout=f"{max(1, int(lifetime_seconds))}s",
            container_grace_seconds=self.config.container_grace_seconds,
            refactor_rounds=0,
        )

    def container_image_id(self) -> str:
        """Return the immutable image ID of the running task container."""

        if not self.container_id:
            raise RuntimeError("cannot resolve an image ID without a container")
        inspected = subprocess.run(
            [
                self.config.executable,
                "inspect",
                "--format",
                "{{.Image}}",
                self.container_id,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        image_id = inspected.stdout.strip()
        if inspected.returncode != 0 or not image_id.startswith("sha256:"):
            error = inspected.stderr.strip()[-4000:]
            raise RuntimeError(
                "could not resolve immutable task image ID: "
                + (error or image_id or "unknown docker inspect failure")
            )
        return image_id

    def cleanup(self):
        """Remove only resources created by this environment, including relay."""

        cleanup_lock = getattr(self, "_cleanup_lock", None)
        if cleanup_lock is None:
            return
        with cleanup_lock:
            container_id = getattr(self, "container_id", None)
            relay_id = getattr(self, "_model_relay_id", None)
            network_name = getattr(self, "_model_network_name", None)
            self.container_id = None
            self._model_relay_id = None
            self._model_network_name = None
            self._model_gateway_target = None
            for resource in (container_id, relay_id):
                if resource:
                    subprocess.run(
                        [self.config.executable, "rm", "-f", resource],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60,
                    )
            view = getattr(self, "_resource_view", None)
            self._resource_view = None
            if view is not None:
                view.close()
            if network_name:
                subprocess.run(
                    [self.config.executable, "network", "rm", network_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

    def reset(self) -> None:
        """Reset the environment by stopping and removing the container, then starting a new one."""
        self.cleanup()
        self.container_id = None
        self._start_container()
