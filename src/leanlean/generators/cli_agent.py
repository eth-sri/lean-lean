from leanlean.utils import json_utils as json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List
import logging

from jinja2 import Template

from configs.model_constants import MISTRAL_VIBE_COMPACTION_MODEL
from leanlean import Generator, Environment, Model
from leanlean.claude_action_playback import ClaudeActionPlaybackRecorder
from leanlean.claude_code_trace import ClaudeTraceBuilder
from leanlean.antigravity_trace import (
    antigravity_terminal_success,
    is_antigravity_event,
    is_antigravity_recovered_submission,
)
from leanlean.model.costs import (
    cost_accounting_from_responses,
    cost_from_responses,
    require_complete_cost_accounting,
)
from leanlean.model.litellm_wrapper.accounting import scoped_records
from leanlean.playback import (
    AGENT_PID_PREFIX,
    ExternalPlaybackRecorder,
    LivePlaybackRecorder,
)
from leanlean.host_side_checkpointing import HostSideCheckpointRecorder
from leanlean.standardized_capture import StandardizedRunRecorder
from leanlean.standardized_replay import ReplayPolicy
from leanlean.standardized_trace import is_codex_nonterminal_advisory


_SUBSCRIPTION_ENV_DIR = "/tmp/leanlean-subscription"
_SUBSCRIPTION_ENV_PATH = f"{_SUBSCRIPTION_ENV_DIR}/env"
_MISTRAL_VIBE_HOME = "/tmp/leanlean-vibe-home"
_ANTIGRAVITY_HOME = "/tmp/leanlean-antigravity-home"

def _record_standardized_replay_failure(
    output_dir: Path, *, phase: str, error: BaseException
) -> dict[str, object]:
    """Persist replay telemetry failure without discarding the final patch."""

    payload: dict[str, object] = {
        "format": "code-harness-standardized-replay-failure-v1",
        "capture_mode": "standardized_replay",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "standardized_session_replay_exact": False,
        "final_patch_preserved_for_evaluation": True,
        "evaluation_continues": True,
    }
    destination = output_dir / "standardized-replay-failure.json"
    temporary = output_dir / ".standardized-replay-failure.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return payload


def _attempt_standardized_replay_telemetry(
    operation, *, output_dir: Path, phase: str, logger
) -> tuple[object | None, dict[str, object] | None]:
    try:
        return operation(), None
    except Exception as error:
        logger.exception(
            "Could not finish standardized replay telemetry; the final patch "
            "will still be evaluated."
        )
        return None, _record_standardized_replay_failure(
            output_dir,
            phase=phase,
            error=error,
        )


@dataclass
class CLIAgentConfig:
    launch_command: str
    install_commands: List[str]
    post_install_commands: List[str]
    post_exec_commands: List[str]
    cli_name: str
    instance_template: str = "{{task}}"
    requires_model_proxy: bool = True
    subscription_transport: str | None = None
    preserve_shared_proxy_traces: bool = False
    gateway_accounting: bool = False
    host_tool_bundle: str | None = None
    host_tool_version: str | None = None
    # Host path of a Codex model catalog passed as model_catalog_json (codex_sub).
    codex_model_catalog: str | None = None
    system_prompt: str = ""
    allowed_skills: List[str] = field(default_factory=list)
    mcp_servers: List[str] = field(default_factory=list)
    enable_subagents: bool = True


def resolve_codex_standalone_tools(version: str) -> tuple[Path, Path, Path, Path]:
    """Resolve the complete pinned offline Codex execution bundle."""

    releases = Path(
        os.environ.get(
            "CODEX_STANDALONE_RELEASES",
            "~/.codex/packages/standalone/releases",
        )
    ).expanduser()
    candidates = sorted(releases.glob(f"{version}-*/bin/codex"))
    # Worktrees can install their pinned release locally without changing the
    # desktop application's independently updated Codex installation.
    if not candidates and "CODEX_STANDALONE_RELEASES" not in os.environ:
        local_releases = Path(__file__).resolve().parents[3] / "data/tool_bundles/codex"
        local_candidates = sorted(local_releases.glob(f"{version}-*/bin/codex"))
        if local_candidates:
            releases, candidates = local_releases, local_candidates
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one local Codex standalone release for {version}, "
            f"found {len(candidates)} below {releases}; install that pinned Codex "
            "version on the host before launching an eval"
        )
    codex = candidates[0].resolve()
    code_mode_host = codex.parent / "codex-code-mode-host"
    bwrap = codex.parent.parent / "codex-resources" / "bwrap"
    rg = codex.parent.parent / "codex-path" / "rg"
    if not all(path.is_file() for path in (codex, code_mode_host, bwrap, rg)):
        raise RuntimeError(
            f"Codex standalone release {codex.parent.parent} is incomplete"
        )
    version_check = subprocess.run(
        [str(codex), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = f"codex-cli {version}"
    if version_check.returncode != 0 or version_check.stdout.strip() != expected:
        raise RuntimeError(
            f"pinned Codex binary mismatch: expected {expected!r}, "
            f"found {version_check.stdout.strip()!r}"
        )
    return codex, code_mode_host, bwrap, rg


def install_codex_standalone_tools(env: Environment, version: str) -> None:
    """Copy pinned static tools into an eval without granting package egress."""

    copy = getattr(env, "copy_host_executable", None)
    if not callable(copy):
        raise RuntimeError("Codex standalone tool injection requires Docker")
    codex, code_mode_host, bwrap, rg = resolve_codex_standalone_tools(version)
    copy(codex, "/usr/local/bin/codex")
    copy(code_mode_host, "/usr/local/bin/codex-code-mode-host")
    copy(bwrap, "/usr/local/bin/bwrap")
    copy(rg, "/usr/local/bin/rg")


def resolve_claude_standalone_tool(version: str) -> Path:
    """Resolve and verify Claude Code's pinned native host executable."""

    releases = Path(
        os.environ.get(
            "CLAUDE_STANDALONE_RELEASES",
            "~/.local/share/claude/versions",
        )
    ).expanduser()
    claude = (releases / version).resolve()
    if not claude.is_file():
        raise RuntimeError(
            f"Claude Code {version} is not installed below {releases}; install "
            "that pinned version on the host before launching an eval"
        )
    version_check = subprocess.run(
        [str(claude), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if version_check.returncode != 0 or not version_check.stdout.startswith(
        f"{version} "
    ):
        raise RuntimeError(
            f"pinned Claude binary mismatch: expected {version!r}, "
            f"found {version_check.stdout.strip()!r}"
        )
    return claude


def install_claude_standalone_tool(env: Environment, version: str) -> None:
    """Copy the pinned Claude Code executable into an isolated eval."""

    copy = getattr(env, "copy_host_executable", None)
    if not callable(copy):
        raise RuntimeError("Claude standalone tool injection requires Docker")
    copy(resolve_claude_standalone_tool(version), "/usr/local/bin/claude")



def resolve_antigravity_standalone_tool(version: str) -> Path:
    """Resolve and verify the checksum-pinned Antigravity CLI executable."""

    machine = os.uname().machine.lower()
    platform = {
        "x86_64": "linux-x64",
        "amd64": "linux-x64",
        "aarch64": "linux-arm64",
        "arm64": "linux-arm64",
    }.get(machine)
    if platform is None:
        raise RuntimeError(
            f"Antigravity CLI does not support host architecture {machine!r}"
        )
    override = os.environ.get("ANTIGRAVITY_STANDALONE_EXECUTABLE")
    releases = Path(
        os.environ.get(
            "ANTIGRAVITY_STANDALONE_RELEASES",
            "~/.local/share/leanlean/antigravity/releases",
        )
    ).expanduser()
    candidates = [
        Path(override).expanduser() if override else None,
        releases / f"{version}-{platform}" / "agy",
    ]
    executable = next(
        (candidate.resolve() for candidate in candidates if candidate and candidate.is_file()),
        None,
    )
    if executable is None:
        raise RuntimeError(
            f"Antigravity CLI {version} is not installed. Run "
            "scripts/install_antigravity_standalone.sh, or set "
            "ANTIGRAVITY_STANDALONE_EXECUTABLE."
        )
    version_check = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = f"{version_check.stdout}\n{version_check.stderr}".strip()
    if version_check.returncode != 0 or output != version:
        raise RuntimeError(
            f"pinned Antigravity CLI mismatch: expected {version!r}, found {output!r}"
        )
    return executable


def install_antigravity_standalone_tool(env: Environment, version: str) -> None:
    """Copy the pinned Antigravity CLI into a network-disabled evaluation."""

    copy = getattr(env, "copy_host_executable", None)
    if not callable(copy):
        raise RuntimeError("Antigravity standalone tool injection requires Docker")
    copy(resolve_antigravity_standalone_tool(version), "/usr/local/bin/agy")



def resolve_mistral_vibe_standalone_bundle(version: str) -> Path:
    """Resolve and verify Mistral Vibe's pinned one-directory release bundle."""

    executable_override = os.environ.get("VIBE_STANDALONE_EXECUTABLE")
    releases = Path(
        os.environ.get(
            "VIBE_STANDALONE_RELEASES",
            "~/.local/share/leanlean/vibe/releases",
        )
    ).expanduser()
    candidates = [
        Path(executable_override).expanduser().parent
        if executable_override
        else None,
        releases / version,
    ]
    release = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate
            and (candidate / "vibe").is_file()
            and (candidate / "_internal").is_dir()
        ),
        None,
    )
    if release is None:
        raise RuntimeError(
            f"Mistral Vibe {version} is not installed as a complete standalone "
            "bundle. Run scripts/install_mistral_vibe_standalone.sh, or set "
            "VIBE_STANDALONE_EXECUTABLE to the pinned bundle's vibe executable."
        )

    version_check = subprocess.run(
        [str(release / "vibe"), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = f"vibe {version}"
    output = f"{version_check.stdout}\n{version_check.stderr}".strip()
    if version_check.returncode != 0 or output != expected:
        raise RuntimeError(
            f"pinned Mistral Vibe mismatch: expected {expected!r}, found {output!r}"
        )
    return release


def install_mistral_vibe_standalone_bundle(
    env: Environment, version: str
) -> None:
    """Copy the pinned Vibe runtime bundle into an isolated eval."""

    copy = getattr(env, "copy_host_directory", None)
    if not callable(copy):
        raise RuntimeError("Mistral Vibe standalone bundle injection requires Docker")
    copy(
        resolve_mistral_vibe_standalone_bundle(version),
        "/opt/leanlean/mistral-vibe",
    )



# Vibe 2.25.0's Lean agent tools minus what Codex's denylist also turns off
# (web search, subagents, skills, asking the user); exit_plan_mode is disabled
# by the Lean agent itself. todo stays, as Codex keeps update_plan.
MISTRAL_VIBE_DISABLED_TOOLS = [
    "exit_plan_mode",
    "web_search",
    "web_fetch",
    "task",
    "skill",
    "ask_user_question",
]


def build_mistral_vibe_profile_files(
    *, base_url: str, model: str
) -> dict[str, str]:
    """Build an isolated Vibe profile that can call only the local gateway."""

    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"invalid Vibe gateway URL: {base_url!r}")
    bare_model = model.split("/")[-1]
    if not bare_model:
        raise ValueError("Vibe requires a non-empty model id")
    quoted_url = json.dumps(base_url)
    quoted_model = json.dumps(bare_model)

    config = f"""enable_telemetry = false
enable_otel = false
enable_update_checks = false
enable_auto_update = false
enable_notifications = false
show_greeting = false
vibe_code_enabled = false
experimental_enable_registry_skills = false
active_model = "leanstral"

[experiments]
enable = false

[session_logging]
enabled = false

[[providers]]
name = "leanlean-litellm"
api_base = {quoted_url}
api_key_env_var = "LITELLM_API_KEY"
api_style = "openai"
backend = "generic"

[[models]]
name = {quoted_model}
provider = "leanlean-litellm"
alias = "leanstral"
thinking = "high"
temperature = 1.0
input_price = 0.0
output_price = 0.0
cached_input_price = 0.0
auto_compact_threshold = 200000

[compaction_model]
name = {json.dumps(MISTRAL_VIBE_COMPACTION_MODEL)}
provider = "leanlean-litellm"
alias = "devstral-compact"
thinking = "off"
temperature = 0.2
input_price = 0.15
output_price = 0.6
cached_input_price = 0.15

[tools.bash]
default_timeout = 1200
"""

    agent = f"""display_name = "LeanLean Lean"
description = "Vibe Lean agent routed through the instance LiteLLM gateway"
safety = "neutral"
system_prompt_id = "lean"
active_model = "leanstral"
allowed_models = ["leanstral"]
disabled_tools = {json.dumps(MISTRAL_VIBE_DISABLED_TOOLS)}
include_commit_signature = false

"""
    return {
        "config.toml": config,
        "agents/leanlean-lean.toml": agent,
    }


def playback_launch_command(launch_command: str, playback_mode: str) -> str:
    """Add pause-control ancestry only for instrumented playback modes."""

    if playback_mode in {
        "passive_action_replay",
        "standardized_replay",
        "host_side_checkpointing",
    }:
        return launch_command
    inner_command = (
        f"printf '{AGENT_PID_PREFIX}%s\\n' \"$$\"; "
        f"exec bash -lc {shlex.quote(launch_command)}"
    )
    return f"setsid --wait bash -lc {shlex.quote(inner_command)}"


def _strip_binary_diff_sections(diff: str) -> str:
    """Drop binary-file sections from a git diff.

    A stray binary file in the working tree (e.g. a `core` crash dump) gets
    swept in by `git add -A` and emitted as a `Binary files ... differ` hunk.
    Because the eval applies patches with `git apply` (no `--binary`), such a
    hunk cannot be applied and `git apply` rejects the *entire* patch
    atomically -- silently discarding every real source edit. No metric ever
    consumes binary files, so dropping these sections is always safe.
    """
    if "diff --git " not in diff:
        return diff
    sections: List[List[str]] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git ") or not sections:
            sections.append([line])
        else:
            sections[-1].append(line)
    kept = [
        "".join(sec)
        for sec in sections
        if not any(
            l.startswith("Binary files ") or l.startswith("GIT binary patch")
            for l in sec
        )
    ]
    return "".join(kept)


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class APIError(TerminatingException):
    """Raised when the traces are empty due to an API error."""


class RetryableAPIError(APIError):
    """A structured provider failure eligible for a clean-baseline retry."""


def _claude_result_failure(result_event: dict | None) -> str:
    """Return a compact failure reason for Claude's zero-exit error result.

    Claude Code can emit a structured ``is_error`` result and still leave the
    process with a successful shell exit code.  The JSON stream is trajectory
    data, never a patch, so callers need an independent terminal-success check.
    """

    if not isinstance(result_event, dict):
        return ""
    terminal_reason = str(result_event.get("terminal_reason") or "")
    subtype = str(result_event.get("subtype") or "")
    failed = (
        bool(result_event.get("is_error"))
        or terminal_reason
        in {
            "api_error",
            "error",
            "failed",
        }
        or subtype in {"error", "failed"}
    )
    if not failed:
        return ""
    details = []
    if terminal_reason:
        details.append(f"terminal_reason={terminal_reason}")
    status = result_event.get("api_error_status")
    if status is not None:
        details.append(f"api_error_status={status}")
    return "Claude Code reported a terminal failure" + (
        f" ({', '.join(details)})" if details else ""
    )


def _claude_result_failure_kind(
    events: list[dict], result_event: dict | None
) -> str:
    """Distinguish retryable subscription exhaustion from other failures."""

    if not _claude_result_failure(result_event):
        return ""
    quota_rejected = any(
        event.get("type") == "rate_limit_event"
        and isinstance(event.get("rate_limit_info"), dict)
        and event["rate_limit_info"].get("status") == "rejected"
        for event in events
    )
    status = result_event.get("api_error_status") if result_event else None
    return "quota" if quota_rejected or status == 429 else "provider"



def _codex_failure_kind(detail: str) -> str:
    """Classify a structured Codex provider failure for queueing or retry."""

    normalized = detail.lower()
    auth_markers = (
        "token_revoked", "invalidated oauth token", "authentication",
        "unauthorized", "invalid api key", "invalid_api_key",
        "status 401", "status: 401", '"status": 401',
        "status 403", "status: 403", '"status": 403',
        "openai_subscription_key", "token_expired",
    )
    if any(marker in normalized for marker in auth_markers):
        return "authentication"
    quota_markers = (
        "status 429",
        "status: 429",
        "http 429",
        "rate limit",
        "usage limit",
        "quota exhausted",
        "quota exceeded",
    )
    return "quota" if any(marker in normalized for marker in quota_markers) else "provider"


def _antigravity_failure_kind(detail: str) -> str:
    """Classify explicit Gemini quota failures without masking other errors."""

    normalized = detail.lower()
    markers = (
        "status 429",
        "status: 429",
        "http 429",
        "rate limit",
        "resource_exhausted",
        "quota exceeded",
        "quota exhausted",
        "out of credits",
    )
    return "quota" if any(marker in normalized for marker in markers) else "provider"



class CLIAgent(Generator):
    """Wraps a CLI-based agent (e.g. qwen-code) to interact with LeanLean."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        **kwargs,
    ):
        self.config = CLIAgentConfig(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("leanlean.cli_agent")

    def _prepare_agent_contract(self) -> None:
        """Enforce native prompts while isolating unpinned extensions."""

        if not self.config.system_prompt:
            return
        if self.config.system_prompt != "native":
            raise RuntimeError(
                "agent-product evaluations require system_prompt='native'"
            )
        if self.config.allowed_skills or self.config.mcp_servers:
            raise RuntimeError(
                "externally supplied skills and MCP servers must remain disabled"
            )
        if self.config.cli_name == "codex_sub":
            self.env.execute(
                "rm -rf /tmp/leanlean-codex-home/skills "
                "$HOME/.agents/skills && "
                "install -d -m 700 /tmp/leanlean-codex-home"
            )

    def _install(self):
        """Install the underlying CLI agent without network bootstrap."""

        if self.config.host_tool_bundle is not None:
            if self.config.host_tool_bundle == "codex_standalone":
                if not self.config.host_tool_version:
                    raise RuntimeError("codex_standalone requires host_tool_version")
                install_codex_standalone_tools(self.env, self.config.host_tool_version)
            elif self.config.host_tool_bundle == "claude_standalone":
                if not self.config.host_tool_version:
                    raise RuntimeError("claude_standalone requires host_tool_version")
                install_claude_standalone_tool(self.env, self.config.host_tool_version)
            elif self.config.host_tool_bundle == "antigravity_standalone":
                if not self.config.host_tool_version:
                    raise RuntimeError(
                        "antigravity_standalone requires host_tool_version"
                    )
                install_antigravity_standalone_tool(
                    self.env, self.config.host_tool_version
                )
            elif self.config.host_tool_bundle == "mistral_vibe_standalone":
                if not self.config.host_tool_version:
                    raise RuntimeError(
                        "mistral_vibe_standalone requires host_tool_version"
                    )
                install_mistral_vibe_standalone_bundle(
                    self.env, self.config.host_tool_version
                )
            else:
                raise RuntimeError(
                    f"unsupported host tool bundle {self.config.host_tool_bundle!r}"
                )

        # Run any remaining offline install commands.
        for cmd in self.config.install_commands:
            self.env.execute(cmd, timeout=False)

        # Run the post-install commands
        for cmd in self.config.post_install_commands:
            self.env.execute(cmd, timeout=False)

    def _capture_codex_native_rollout(self) -> None:
        """Export Codex's richer native rollout before deleting ``CODEX_HOME``."""

        if getattr(self, "_native_rollout_capture_attempted", False):
            return
        self._native_rollout_capture_attempted = True
        self._native_rollout_host_path = None
        output_dir = getattr(self, "_native_rollout_output_dir", None)
        exporter = getattr(self.env, "export_codex_rollout", None)
        if output_dir is None or not callable(exporter):
            self.native_rollout = {
                "format": "codex-native-rollout-jsonl-gzip-v1",
                "status": "capture_failed",
                "error": "native rollout capture requires a Docker output directory",
            }
            self.logger.error(self.native_rollout["error"])
            return

        capture_index = getattr(self, "_native_rollout_capture_index", 0) + 1
        self._native_rollout_capture_index = capture_index
        filename = f"capture_{capture_index:03d}.native-rollout.jsonl.gz"
        directory = Path(output_dir)
        destination = directory / filename
        timeout = int(
            (getattr(self, "trajectory_config", {}) or {}).get(
                "native_rollout_timeout_seconds", 120
            )
        )
        try:
            metadata = exporter(destination, timeout=timeout)
        except Exception as exc:
            destination.unlink(missing_ok=True)
            self.native_rollout = {
                "format": "codex-native-rollout-jsonl-gzip-v1",
                "status": "capture_failed",
                "error": str(exc),
            }
            self.logger.exception("Could not export the native Codex rollout.")
            return

        reference_prefix = str(
            getattr(self, "_native_rollout_reference_prefix", "native_rollout")
        ).rstrip("/")
        self.native_rollout = {
            "format": "codex-native-rollout-jsonl-gzip-v1",
            "status": "complete",
            "artifact_path": f"{reference_prefix}/{filename}",
            "digest_basis": "uncompressed native Codex JSONL bytes",
            "contains_full_tool_inputs": True,
            "contains_exact_filesystem_states": False,
            **metadata,
        }
        self._native_rollout_host_path = destination

    @staticmethod
    def _enable_codex_rollout_persistence(launch_command: str) -> str:
        """Turn a configured ephemeral Codex run into a persisted native session."""

        marker = "--ephemeral "
        if marker not in launch_command:
            raise ValueError(
                "native Codex rollout capture requires launch_command to contain "
                "--ephemeral"
            )
        return launch_command.replace(marker, "", 1)

    def _post_exec(self):
        """Run any post-execution commands."""
        if self.config.subscription_transport in {
            "chatgpt",
            "claude_max",
        }:
            self._cleanup_subscription_gateway_env()
        if self.config.cli_name == "codex_sub":
            self._cleanup_codex_subscription_auth()
        if self.config.cli_name == "mistral_vibe":
            self._cleanup_mistral_vibe_home()
        if self.config.cli_name == "antigravity_cli":
            self._cleanup_antigravity_home()

        for cmd in self.config.post_exec_commands:
            self.env.execute(cmd, timeout=False)

        # Clear the traces
        self.model.delete_traces()

    def _cleanup_codex_subscription_auth(self) -> None:
        """Best-effort removal of the password-equivalent temporary auth cache."""
        try:
            # This must happen before extracting a diff or committing a cached
            # image. The exact, fixed /tmp path is never derived from user data.
            self.env.execute(
                "rm -rf -- /tmp/leanlean-codex-home",
                timeout=True,
            )
        except Exception:
            self.logger.exception("Failed to remove temporary Codex auth cache.")


    def _cleanup_mistral_vibe_home(self) -> None:
        """Remove invocation-local Vibe config and sessions before caching."""

        try:
            self.env.execute(
                f"rm -rf -- {_MISTRAL_VIBE_HOME}",
                timeout=True,
            )
        except Exception:
            self.logger.exception("Failed to remove temporary Mistral Vibe home.")

    def _cleanup_antigravity_home(self) -> None:
        """Remove invocation-local Antigravity config and conversation state."""

        try:
            self.env.execute(
                f"rm -rf -- {_ANTIGRAVITY_HOME}",
                timeout=True,
            )
        except Exception:
            self.logger.exception("Failed to remove temporary Antigravity home.")


    def _prepare_antigravity_home(self) -> None:
        """Create a private API-key profile that is forced through the relay."""

        self._cleanup_antigravity_home()
        config_dir = f"{_ANTIGRAVITY_HOME}/.gemini/antigravity-cli"
        prepared = self.env.execute(
            f"install -d -m 700 {config_dir}",
            timeout=True,
        )
        if prepared.get("returncode", 1) != 0:
            raise APIError("Could not create the temporary Antigravity home.")
        settings = json.dumps(
            {
                "modelProvider": "gemini",
                "enableTelemetry": False,
                "showFeedbackSurvey": False,
                "notifications": False,
                "toolPermission": "always-proceed",
                "artifactReviewPolicy": "always-proceed",
            },
            sort_keys=True,
        )
        container_id = getattr(self.env, "container_id", None)
        env_config = getattr(self.env, "config", None)
        executable = getattr(env_config, "executable", "docker")
        if not container_id:
            self._cleanup_antigravity_home()
            raise APIError("Antigravity routing currently requires Docker.")
        destination = f"{config_dir}/settings.json"
        copied = subprocess.run(
            [
                executable,
                "exec",
                "-i",
                container_id,
                "sh",
                "-c",
                f"umask 077 && cat > {shlex.quote(destination)}",
            ],
            input=(settings + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        if copied.returncode != 0:
            self._cleanup_antigravity_home()
            output = copied.stdout.decode("utf-8", errors="replace").strip()
            raise APIError(f"Could not install the Antigravity profile: {output}")


    def _prepare_mistral_vibe_home(self, openai_args: dict) -> None:
        """Install a private proxy-only Vibe config without upstream secrets."""

        base_url = openai_args.get("base_url")
        model = openai_args.get("model")
        if not isinstance(base_url, str) or not isinstance(model, str):
            raise APIError("Mistral Vibe requires a LiteLLM gateway URL and model.")

        files = build_mistral_vibe_profile_files(base_url=base_url, model=model)
        self._cleanup_mistral_vibe_home()
        prepared = self.env.execute(
            f"install -d -m 700 {_MISTRAL_VIBE_HOME} "
            f"{_MISTRAL_VIBE_HOME}/agents",
            timeout=True,
        )
        if prepared.get("returncode", 1) != 0:
            self._cleanup_mistral_vibe_home()
            raise APIError("Could not create the temporary Mistral Vibe home.")

        container_id = getattr(self.env, "container_id", None)
        env_config = getattr(self.env, "config", None)
        executable = getattr(env_config, "executable", "docker")
        if not container_id:
            self._cleanup_mistral_vibe_home()
            raise APIError("Mistral Vibe routing currently requires Docker.")

        for relative_path, contents in files.items():
            if relative_path not in {
                "config.toml",
                "agents/leanlean-lean.toml",
            }:
                self._cleanup_mistral_vibe_home()
                raise APIError(f"Unexpected Mistral Vibe profile path: {relative_path}")
            destination = f"{_MISTRAL_VIBE_HOME}/{relative_path}"
            copied = subprocess.run(
                [
                    executable,
                    "exec",
                    "-i",
                    container_id,
                    "sh",
                    "-c",
                    f"umask 077 && cat > {shlex.quote(destination)}",
                ],
                input=contents.encode("utf-8"),
                text=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            if copied.returncode != 0:
                self._cleanup_mistral_vibe_home()
                output = copied.stdout.decode("utf-8", errors="replace").strip()
                raise APIError(
                    f"Could not install the Mistral Vibe profile: {output}"
                )

    CODEX_MODEL_CATALOG_PATH = "/tmp/leanlean-codex-home/model_catalog.json"

    def _install_codex_model_catalog(self, launch_command: str) -> str:
        """Stream the pinned model catalog into the Codex home and select it."""

        catalog = Path(str(self.config.codex_model_catalog))
        if not catalog.is_file():
            raise APIError(f"Codex model catalog is missing: {catalog}")
        marker = "--model "
        if marker not in launch_command:
            raise ValueError("Codex launch command has no --model marker")
        container_id = getattr(self.env, "container_id", None)
        executable = getattr(getattr(self.env, "config", None), "executable", "docker")
        if not container_id:
            raise APIError("codex_model_catalog requires a Docker environment.")
        prepared = self.env.execute(
            "install -d -m 700 /tmp/leanlean-codex-home", timeout=True
        )
        if prepared.get("returncode", 1) != 0:
            raise APIError("Could not create the Codex home for the model catalog.")
        with catalog.open("rb") as handle:
            copied = subprocess.run(
                [
                    executable, "exec", "-i", container_id, "sh", "-c",
                    f"umask 077 && cat > {self.CODEX_MODEL_CATALOG_PATH}",
                ],
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        if copied.returncode != 0:
            output = copied.stdout.decode("utf-8", errors="replace").strip()
            raise APIError(f"Could not copy the Codex model catalog: {output}")
        return launch_command.replace(
            marker,
            f"-c model_catalog_json={self.CODEX_MODEL_CATALOG_PATH} {marker}",
            1,
        )

    def _prepare_fresh_codex_home(self) -> None:
        """Create an empty private home for one complete native session tree."""

        self._cleanup_codex_subscription_auth()
        prepared = self.env.execute(
            "install -d -m 700 /tmp/leanlean-codex-home",
            timeout=True,
        )
        if prepared.get("returncode", 1) != 0:
            self._cleanup_codex_subscription_auth()
            raise APIError(
                "Could not create the fresh isolated Codex home required for "
                "complete rollout capture."
            )

    def _cleanup_subscription_gateway_env(self) -> None:
        """Remove the fixed private file used to inject gateway credentials."""
        try:
            self.env.execute(
                f"rm -rf -- {_SUBSCRIPTION_ENV_DIR}",
                timeout=True,
            )
        except Exception:
            self.logger.exception(
                "Failed to remove temporary subscription gateway credentials."
            )

    def _install_subscription_gateway_env(
        self,
        openai_args: dict,
        *,
        oauth_token: str | None = None,
    ) -> None:
        """Stream subscription credentials into a private container env file.

        Values travel over ``docker exec -i`` stdin, never in a shell command,
        argv, log line, or generated proxy config. The sourced file is removed
        before diff extraction and before the container can be cached.
        """

        api_key = openai_args.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise APIError("Subscription routing requires a LiteLLM gateway key.")
        if self.config.subscription_transport == "chatgpt":
            variables = {"LITELLM_API_KEY": api_key}
        elif self.config.subscription_transport == "claude_max":
            if not oauth_token:
                raise APIError(
                    "claude_code_sub requires CLAUDE_CODE_OAUTH_TOKEN to be set "
                    "(generate one on the host with `claude setup-token`)."
                )
            variables = {
                "CLAUDE_CODE_OAUTH_TOKEN": oauth_token,
                "ANTHROPIC_CUSTOM_HEADERS": (f"x-litellm-api-key: Bearer {api_key}"),
            }
        else:
            raise APIError(
                f"Unsupported subscription transport: "
                f"{self.config.subscription_transport!r}"
            )

        container_id = getattr(self.env, "container_id", None)
        env_config = getattr(self.env, "config", None)
        executable = getattr(env_config, "executable", "docker")
        if not container_id:
            raise APIError("Subscription routing currently requires Docker.")

        prepared = self.env.execute(
            f"install -d -m 700 {_SUBSCRIPTION_ENV_DIR}",
            timeout=True,
        )
        if prepared.get("returncode", 1) != 0:
            self._cleanup_subscription_gateway_env()
            raise APIError("Could not create the private subscription directory.")

        payload = "".join(
            f"export {name}={shlex.quote(value)}\n" for name, value in variables.items()
        ).encode("utf-8")
        try:
            copied = subprocess.run(
                [
                    executable,
                    "exec",
                    "-i",
                    container_id,
                    "sh",
                    "-c",
                    f"umask 077 && cat > {_SUBSCRIPTION_ENV_PATH}",
                ],
                input=payload,
                text=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._cleanup_subscription_gateway_env()
            raise APIError(
                f"Could not install private subscription credentials: {exc}"
            ) from exc
        if copied.returncode != 0:
            self._cleanup_subscription_gateway_env()
            output = copied.stdout.decode("utf-8", errors="replace").strip()
            raise APIError(
                f"Could not install private subscription credentials: {output}"
            )
        secured = self.env.execute(
            f"chmod 600 {_SUBSCRIPTION_ENV_PATH}",
            timeout=True,
        )
        if secured.get("returncode", 1) != 0:
            self._cleanup_subscription_gateway_env()
            raise APIError("Could not secure private subscription credentials.")

    def _install_codex_subscription_auth(self) -> None:
        """Stream the host Codex login cache into this run's Docker container.

        The native subscription path deliberately does not inject tokens into a
        shell command or environment variable. Streaming avoids ``docker cp``
        preserving a host UID/GID that may be unmappable in a user-namespaced
        container. CODEX_AUTH_FILE can override the default ~/.codex/auth.json
        when the host uses a non-default Codex home.
        """
        auth_path = Path(
            os.environ.get("CODEX_AUTH_FILE", "~/.codex/auth.json")
        ).expanduser()
        if not auth_path.is_file():
            raise APIError(
                "codex_sub requires a file-based Codex login cache. Run "
                "`codex login`, or set CODEX_AUTH_FILE to its auth.json path."
            )

        container_id = getattr(self.env, "container_id", None)
        env_config = getattr(self.env, "config", None)
        executable = getattr(env_config, "executable", "docker")
        if not container_id:
            raise APIError("codex_sub currently requires a Docker environment.")

        prepared = self.env.execute(
            "install -d -m 700 /tmp/leanlean-codex-home",
            timeout=True,
        )
        if prepared.get("returncode", 1) != 0:
            self._cleanup_codex_subscription_auth()
            raise APIError("Could not create the temporary Codex auth directory.")

        try:
            with auth_path.open("rb") as auth_file:
                copied = subprocess.run(
                    [
                        executable,
                        "exec",
                        "-i",
                        container_id,
                        "sh",
                        "-c",
                        (
                            "umask 077 && cat > "
                            "/tmp/leanlean-codex-home/auth.json"
                        ),
                    ],
                    stdin=auth_file,
                    text=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._cleanup_codex_subscription_auth()
            raise APIError(
                f"Could not copy CODEX_AUTH_FILE into the eval container: {exc}"
            ) from exc
        if copied.returncode != 0:
            self._cleanup_codex_subscription_auth()
            output = copied.stdout.decode("utf-8", errors="replace").strip()
            raise APIError(
                f"Could not copy CODEX_AUTH_FILE into the eval container: {output}"
            )
        secured = self.env.execute(
            "chmod 600 /tmp/leanlean-codex-home/auth.json",
            timeout=True,
        )
        if secured.get("returncode", 1) != 0:
            self._cleanup_codex_subscription_auth()
            raise APIError("Could not secure the temporary Codex auth cache.")

    @staticmethod
    def _append_message_if_new(messages: list[dict], message: dict | None) -> None:
        if not isinstance(message, dict) or not message:
            return
        if messages and messages[-1] == message:
            return
        messages.append(message)

    def _process_gemini_logs(self, traces: list[dict]) -> None:
        messages: list[dict] = []
        for trace in traces:
            request_messages = trace.get("request", [])
            if isinstance(request_messages, list):
                for message in request_messages:
                    self._append_message_if_new(
                        messages, message if isinstance(message, dict) else None
                    )

            response_message = (
                trace.get("response", {}).get("choices", [{}])[0].get("message", {})
            )
            self._append_message_if_new(
                messages,
                response_message if isinstance(response_message, dict) else None,
            )

        self.messages.extend(messages)
        for trace in traces:
            self.model.responses.append(trace["response"])

    @staticmethod
    def _sub_responses_from_model_usage(model_usage) -> list[dict]:
        """Convert the result event's per-model `modelUsage` totals into the
        OpenAI-style response dicts cost_from_responses/TokenUsage expect.

        Anthropic reports `inputTokens` *excluding* cache, with separate
        cache-read/creation counts. The cost path expects `prompt_tokens` to be
        the *total* input including cache (split out via prompt_tokens_details),
        so we fold them together here. This is the same normalization LiteLLM
        does for the proxy path, keeping both paths' `responses` shape identical.
        """

        def _int(value) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        responses: list[dict] = []
        if not isinstance(model_usage, dict):
            return responses
        for model_id, usage in model_usage.items():
            if not isinstance(usage, dict):
                continue
            inp = _int(usage.get("inputTokens"))
            out = _int(usage.get("outputTokens"))
            cache_read = _int(usage.get("cacheReadInputTokens"))
            cache_creation = _int(usage.get("cacheCreationInputTokens"))
            responses.append(
                {
                    "model": model_id,
                    "usage": {
                        "prompt_tokens": inp + cache_read + cache_creation,
                        "completion_tokens": out,
                        "total_tokens": inp + cache_read + cache_creation + out,
                        "prompt_tokens_details": {
                            "cached_tokens": cache_read,
                            "cache_creation_tokens": cache_creation,
                        },
                    },
                }
            )
            reported_cost = usage.get("costUSD")
            if (usage.get("costBasis") == "list"
                    and isinstance(reported_cost, (int, float))
                    and not isinstance(reported_cost, bool)):
                responses[-1]["native_cost"] = {
                    "basis": "claude_code_list_price",
                    "usd": reported_cost,
                    "source": "result.modelUsage.costUSD",
                }
        return responses

    @staticmethod
    def _sub_response_from_turn_usage(message: dict) -> dict | None:
        """Convert a single assistant turn's per-message `usage` block into the
        same OpenAI-style response dict as _sub_responses_from_model_usage.

        Used only for the timeout-recovery fallback: when the run is force-killed
        before the `result` event, the authoritative per-model `modelUsage`
        aggregate never arrives, but each assistant turn already carried its own
        usage (snake_case here, vs the result event's camelCase). Summed across
        the deduped turns this reconstructs a best-effort cost. The shape is
        identical to the result-event path so cost_from_responses prices both the
        same way.
        """
        usage = message.get("usage")
        if not isinstance(usage, dict):
            return None

        def _int(value) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        inp = _int(usage.get("input_tokens"))
        out = _int(usage.get("output_tokens"))
        cache_read = _int(usage.get("cache_read_input_tokens"))
        cache_creation = _int(usage.get("cache_creation_input_tokens"))
        return {
            "model": message.get("model") or "unknown_model",
            "usage": {
                **({"cache_creation": usage["cache_creation"]}
                   if isinstance(usage.get("cache_creation"), dict) else {}),
                "prompt_tokens": inp + cache_read + cache_creation,
                "completion_tokens": out,
                "total_tokens": inp + cache_read + cache_creation + out,
                "prompt_tokens_details": {
                    "cached_tokens": cache_read,
                    "cache_creation_tokens": cache_creation,
                },
            },
        }

    def _process_claude_sub_logs(self, raw_stdout: str) -> None:
        """Parse Claude Code `--output-format stream-json` stdout (subscription
        path) into self.messages (trajectory) and model.responses (cost).

        LiteLLM records model calls, while this stream remains authoritative for
        agent messages, tool ordering, and subagent lineage. Cost is taken from
        the authoritative per-model totals in the `result` event, recomputed
        with our price table, and cross-checked against Claude Code's total.
        """
        events: list[dict] = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue  # stderr / progress noise interleaved on the pipe
            if isinstance(event, dict):
                events.append(event)

        if not events:
            raise APIError(
                "No stream-json events found; check CLAUDE_CODE_OAUTH_TOKEN / auth."
            )

        builder = ClaudeTraceBuilder()
        for event in events:
            builder.ingest(event)
        self.provider_trace = builder.to_trace()
        self.messages.extend(builder.messages)

        result_event = builder.result_event
        self.provider_terminal_error = _claude_result_failure(result_event)
        self.provider_terminal_error_kind = _claude_result_failure_kind(
            events, result_event
        )
        terminal = self.provider_trace.get("terminal")
        if isinstance(terminal, dict):
            terminal["success"] = not bool(self.provider_terminal_error)
            if self.provider_terminal_error:
                terminal["failure_reason"] = self.provider_terminal_error
        if self.provider_terminal_error:
            self.logger.error(self.provider_terminal_error)
        assistant_messages = [
            message
            for message in builder.messages
            if message.get("role") == "assistant"
        ]
        assistant_turns: dict[tuple[str, str], dict] = {}
        usage_conflicts = 0
        for fallback_index, message in enumerate(assistant_messages, 1):
            agent_id = str(message.get("trace_agent_id") or "root")
            turn_id = str(
                message.get("id")
                or message.get("request_id")
                or f"assistant-block-{fallback_index}"
            )
            key = (agent_id, turn_id)
            previous = assistant_turns.get(key)
            if previous is not None and (
                previous.get("model") != message.get("model")
                or previous.get("usage") != message.get("usage")
            ):
                usage_conflicts += 1
            assistant_turns[key] = message
        semantic_turns = list(assistant_turns.values())
        self.provider_trace["assistant_usage_block_count"] = len(assistant_messages)
        self.provider_trace["assistant_turn_usage_count"] = len(semantic_turns)
        self.provider_trace["duplicate_assistant_usage_block_count"] = (
            len(assistant_messages) - len(semantic_turns)
        )
        self.provider_trace["assistant_turn_usage_conflict_count"] = usage_conflicts
        if usage_conflicts:
            self.logger.warning(
                "claude_code_sub: %d repeated assistant message IDs carried "
                "conflicting model usage; timeout cost remains best-effort.",
                usage_conflicts,
            )
        # One response per native assistant message ID. Claude stream-json can
        # emit the same turn once per content block (thinking/text/tool), with
        # the same complete usage block on every emission.
        turn_responses = [
            response
            for message in semantic_turns
            if (response := self._sub_response_from_turn_usage(message)) is not None
        ]
        call_count = len(semantic_turns)

        if self.config.gateway_accounting:
            # Third-party models have no reliable Claude price/modelUsage entry.
            # Keep native messages, but use scoped provider usage for all cost.
            self._account_gateway_responses()
            return

        # Cost: prefer the result event's authoritative per-model modelUsage
        # totals. On a force-kill (container timeout) that event never arrives,
        # so fall back to summing the per-turn usage we did capture — a
        # best-effort estimate (subagent turns / turns missing a request_id can
        # undercount slightly vs the aggregate), but far better than $0.
        if result_event is not None:
            self.model.responses.extend(
                self._sub_responses_from_model_usage(result_event.get("modelUsage"))
            )
            self.model.n_calls = call_count
        else:
            self.model.responses.extend(turn_responses)
            self.model.n_calls = call_count
            self.logger.warning(
                "claude_code_sub: no result event (timed out before completion?); "
                "recovered best-effort cost $%.4f from %d per-turn usage blocks.",
                cost_from_responses(turn_responses),
                len(turn_responses),
            )

        # get_cost() has no trace file to read on this path, so point it at the
        # durable responses list and publish the recomputed cost on the model.
        self.model.cost_from_responses_fallback = True
        recomputed = cost_from_responses(self.model.responses)
        self.model.cost = recomputed

        # Cross-check our recompute against the CLI's own number (warn on drift).
        if result_event is not None:
            reported = result_event.get("total_cost_usd")
            if isinstance(reported, (int, float)) and reported > 0:
                rel = abs(recomputed - reported) / reported
                if rel > 0.05:
                    self.logger.warning(
                        "claude_code_sub cost mismatch: recomputed $%.4f vs CLI "
                        "$%.4f (%.0f%% off) — check MODEL_PRICES.",
                        recomputed,
                        reported,
                        rel * 100,
                    )
                else:
                    self.logger.info(
                        "claude_code_sub cost $%.4f (CLI reported $%.4f).",
                        recomputed,
                        reported,
                    )

    def _gateway_records(self, path):
        return scoped_records(Path(path), getattr(self.model, "accounting_scope_id", None))

    def _account_gateway_responses(
        self, *, native_fallback_response: dict | None = None
    ) -> None:
        records = self._gateway_records(self.model.get_cost_traces_path())
        responses = [record["response"] for record in records
                     if record.get("event") == "success"
                     and isinstance(record.get("response"), dict)]
        if responses:
            try:
                require_complete_cost_accounting(responses)
            except ValueError as error:
                if native_fallback_response is None:
                    raise
                self.provider_trace["gateway_accounting_recovery"] = {
                    "reason": str(error),
                    "gateway_evidence": cost_accounting_from_responses(responses),
                    "recovered_from": "antigravity_terminal_usage",
                    "benchmark_estimate_not_invoice": True,
                }
                responses = [native_fallback_response]
        if not responses and native_fallback_response is not None:
            responses = [native_fallback_response]
        self.model.responses.extend(responses)
        self.model.n_calls = len(responses)
        self.model.cost_from_responses_fallback = True
        self.model.cost = cost_from_responses(self.model.responses)
        self.provider_trace["accounting_scope"] = getattr(self.model, "accounting_scope_id", None)

    def _capture_gateway_trace(self) -> None:
        """Attach sanitized LiteLLM model-call records before trace cleanup."""

        getter = getattr(self.model, "get_traces_path", None)
        if not callable(getter):
            return
        accounting_only = bool(
            getattr(
                getattr(self.model, "config", None),
                "subscription_transport",
                None,
            )
        )
        if accounting_only:
            getter = getattr(self.model, "get_cost_traces_path", None)
            if not callable(getter):
                return
        path = Path(getter())
        if not path.is_file():
            return
        calls: list[dict] = []
        for record in self._gateway_records(path):
            calls.append(
                {
                    key: record.get(key)
                    for key in (
                        "ts",
                        "event",
                        "model",
                        "request",
                        "request_envelope",
                        "response",
                        "exception",
                        "agent_context",
                        "litellm_call_id",
                        "usage_provenance",
                        "provider_usage",
                        "raw_provider_stream_id",
                    )
                    if key in record
                }
            )
        if not calls:
            return
        if not isinstance(getattr(self, "provider_trace", None), dict):
            self.provider_trace = {"format": "agent-provider-trace-v1"}
        self.provider_trace["gateway"] = {
            "format": (
                "litellm-model-accounting-trace-v1"
                if accounting_only
                else "litellm-model-call-trace-v1"
            ),
            "call_count": len(calls),
            "scope": (
                "invocation-scoped compact accounting journal; full requests remain in the run journal"
                if accounting_only and getattr(self.model, "accounting_scope_id", None)
                else "legacy proxy-wide compact accounting journal; not attributable per repository"
                if accounting_only
                else (
                    "invocation-scoped gateway journal"
                    if getattr(self.model, "accounting_scope_id", None)
                    else "legacy proxy-wide journal; NOT attributable per repository"
                )
            ),
            "contains_credentials": False,
            "calls": calls,
        }

    @staticmethod
    def _codex_response_from_usage(model_name: str, usage: dict) -> dict:
        """Normalize a Codex `turn.completed` usage block for cost accounting."""

        def _int(value) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        inp = _int(usage.get("input_tokens"))
        cached = _int(usage.get("cached_input_tokens"))
        cache_write = _int(usage.get("cache_write_input_tokens"))
        out = _int(usage.get("output_tokens"))
        reasoning = _int(usage.get("reasoning_output_tokens"))
        return {
            "model": model_name,
            "usage": {
                # Codex reports cached_input_tokens as a subset of input_tokens.
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": inp + out,
                "prompt_tokens_details": {
                    "cached_tokens": cached,
                    "cache_creation_tokens": cache_write,
                },
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        }

    def _append_codex_item(self, item: dict) -> None:
        """Store a completed Codex item in an OpenAI-compatible trajectory shape."""
        item_type = item.get("type")
        if item_type == "command_execution":
            tool_id = str(item.get("id") or f"codex_command_{len(self.messages)}")
            command = item.get("command", "")
            if not isinstance(command, str):
                command = json.dumps(command)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps({"command": command}),
                            },
                        }
                    ],
                    "codex_item": item,
                }
            )
            output = item.get("aggregated_output", item.get("output", ""))
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": output
                    if isinstance(output, str)
                    else json.dumps(output),
                    "exit_code": item.get("exit_code"),
                    "codex_item": item,
                }
            )
            return

        text = item.get("text")
        if not isinstance(text, str):
            text = item.get("message")
        if not isinstance(text, str):
            text = json.dumps(item)
        self.messages.append(
            {
                "role": "assistant",
                "content": text,
                "codex_item": item,
            }
        )

    def _process_codex_sub_logs(self, raw_stdout: str) -> None:
        """Parse `codex exec --json` JSONL without a LiteLLM trace server."""
        self.provider_terminal_error = ""
        self.provider_terminal_error_kind = ""
        events: list[dict] = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                events.append(event)

        if not events:
            diagnostic = raw_stdout.strip()[-4000:]
            raise APIError(
                "No Codex JSONL events found; check OPENAI_SUBSCRIPTION_KEY and "
                "subscription access."
                + (f" Native CLI output: {diagnostic}" if diagnostic else "")
            )

        model_name = str(self.model.config.model_name).split("/")[-1]
        responses: list[dict] = []
        client_errors: list[str] = []
        api_errors: list[str] = []
        terminal_turn_type = ""
        for event in events:
            event_type = event.get("type")
            if event_type == "item.completed" and isinstance(event.get("item"), dict):
                item = event["item"]
                self._append_codex_item(item)
                if item.get("type") == "error":
                    error = item.get("message") or item
                    detail = error if isinstance(error, str) else json.dumps(error)
                    if not is_codex_nonterminal_advisory(detail):
                        client_errors.append(detail)
            elif event_type == "turn.completed":
                # Codex may emit one or more top-level reconnect errors and then
                # successfully resume the same turn. The last explicit turn
                # terminal is authoritative; historical reconnect telemetry
                # must not turn a completed submission into RetryableAPIError.
                terminal_turn_type = event_type
                if isinstance(event.get("usage"), dict):
                    responses.append(
                        self._codex_response_from_usage(model_name, event["usage"])
                    )
            elif event_type == "turn.failed":
                terminal_turn_type = event_type
                error = event.get("error") or event.get("message") or event
                api_errors.append(
                    error if isinstance(error, str) else json.dumps(error)
                )
            elif event_type == "error":
                error = event.get("error") or event.get("message") or event
                api_errors.append(
                    error if isinstance(error, str) else json.dumps(error)
                )

        recovered_api_errors = (
            list(api_errors) if terminal_turn_type == "turn.completed" else []
        )
        if recovered_api_errors:
            # Keep the recovered interruption observable without presenting it
            # as a terminal provider failure to submission classification.
            if not isinstance(getattr(self, "provider_trace", None), dict):
                self.provider_trace = {"format": "agent-provider-trace-v1"}
            self.provider_trace.setdefault("terminal", {}).update(
                success=True,
                is_error=False,
                recovered_api_error_count=len(recovered_api_errors),
                last_recovered_api_error=recovered_api_errors[-1],
            )
            api_errors = []

        if api_errors:
            detail = next(
                (error for error in api_errors
                 if _codex_failure_kind(error) == "authentication"),
                api_errors[-1],
            )
            self.provider_terminal_error = (
                "Codex subscription provider error: " + detail
            )
            self.provider_terminal_error_kind = _codex_failure_kind(detail)
            if self.provider_terminal_error_kind == "authentication":
                self.provider_terminal_error = (
                    "Codex subscription authentication failed; update "
                    "OPENAI_SUBSCRIPTION_KEY in secret.sh and relaunch."
                )
            if not isinstance(getattr(self, "provider_trace", None), dict):
                self.provider_trace = {"format": "agent-provider-trace-v1"}
            self.provider_trace.setdefault("terminal", {}).update(
                success=False,
                is_error=True,
                failure_kind=self.provider_terminal_error_kind,
                failure_reason=self.provider_terminal_error,
            )
            if self.provider_terminal_error_kind not in {"quota", "authentication"}:
                raise RetryableAPIError(f"Codex subscription API run failed: {detail}")
        elif client_errors:
            # Item-level errors are emitted by the local client/tooling as well
            # as by informational UI features. They remain terminal telemetry,
            # but must not buy the model a second attempt in a mutated repo.
            self.provider_terminal_error = (
                "Codex subscription client error: " + client_errors[-1]
            )

        self.model.responses.extend(responses)
        self.model.n_calls = len(responses)
        self.model.cost_from_responses_fallback = True
        self.model.cost = cost_from_responses(self.model.responses)
        if not responses:
            self.logger.warning(
                "codex_sub: no turn.completed usage event (timed out before completion?); "
                "trajectory items were retained but cost may be undercounted."
            )


    def _process_antigravity_logs(self, raw_stdout: str) -> None:
        """Preserve Antigravity NDJSON and account from the local gateway."""

        events: list[dict] = []
        unparsed_lines: list[str] = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                unparsed_lines.append(line)
                continue
            if isinstance(event, dict) and is_antigravity_event(event):
                events.append(event)
            else:
                unparsed_lines.append(line)

        diagnostic = "\n".join(unparsed_lines)[-4000:]
        if not events:
            raise APIError(
                "No Antigravity stream-json events found; check the pinned "
                "bundle, proxy profile, and Gemini API key."
                + (f" Native CLI output: {diagnostic}" if diagnostic else "")
            )
        self.messages.extend(dict(event) for event in events)
        results = [
            event["result"]
            for event in events
            if event.get("event") == "result"
            and isinstance(event.get("result"), dict)
        ]
        terminal = dict(results[-1]) if results else {}
        status = str(terminal.get("status") or "").upper()
        detail = str(terminal.get("error") or diagnostic or "")
        recovered_submission = is_antigravity_recovered_submission(terminal)
        success = antigravity_terminal_success(terminal)
        if recovered_submission:
            terminal["native_status"] = status
            terminal["recovered_submission"] = True
        if not success:
            if not detail:
                detail = "Antigravity CLI did not emit a successful terminal result"
            self.provider_terminal_error = detail
            self.provider_terminal_error_kind = _antigravity_failure_kind(detail)
        terminal["success"] = success
        if not success:
            terminal["failure_reason"] = detail
            terminal["failure_kind"] = self.provider_terminal_error_kind
            if self.provider_terminal_error_kind == "quota":
                terminal["api_error_status"] = 429
        self.provider_trace = {
            "format": "antigravity-cli-stream-json-v1",
            "provider": "antigravity_cli",
            "stream_events": events,
            "raw_event_count": len(events),
            "unparsed_line_count": len(unparsed_lines),
            "terminal": terminal,
        }

        from leanlean.antigravity_usage import AntigravityUsageJournal
        journal = AntigravityUsageJournal(self.model.config.model_name)
        for event in events:
            journal.ingest(event)
        if journal.steps:
            # Completed native usage survives missing terminal events and
            # contains explicit fresh/cache splits. Never turn a timeout into $0.
            evidence = journal.evidence()
            self.provider_trace["native_cost_evidence"] = evidence
            self.model.responses.extend(journal.responses)
            self.model.n_calls = len(journal.responses)
            self.model.cost_from_responses_fallback = True
            self.model.cost = cost_from_responses(self.model.responses)
            return

        native_usage = terminal.get("usage")
        native_fallback_response = None
        if isinstance(native_usage, dict):
            input_tokens = native_usage.get("input_tokens")
            output_tokens = native_usage.get("output_tokens")
            cache_read_tokens = native_usage.get("cache_read_tokens")
            thinking_tokens = native_usage.get("thinking_tokens", 0)
            counts = (
                input_tokens,
                output_tokens,
                cache_read_tokens,
                thinking_tokens,
            )
            if all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in counts
            ):
                # Antigravity reports fresh input and cache reads separately.
                # TokenUsage expects prompt_tokens to include both components.
                prompt_tokens = input_tokens + cache_read_tokens
                native_fallback_response = {
                    "model": self.model.config.model_name,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": cache_read_tokens,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": thinking_tokens,
                        },
                    },
                }

        self._account_gateway_responses(
            native_fallback_response=native_fallback_response
        )

    def _process_mistral_vibe_logs(self, raw_stdout: str) -> None:
        """Preserve Vibe streaming history and use gateway responses for usage."""

        events: list[dict] = []
        unparsed_lines: list[str] = []
        for line in raw_stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                unparsed_lines.append(line)
                continue
            if isinstance(event, dict):
                events.append(event)

        diagnostic = "\n".join(unparsed_lines)[-4000:]
        if not events:
            raise APIError(
                "No Mistral Vibe streaming events found; check the pinned bundle, "
                "proxy profile, and MISTRAL_API_KEY."
                + (f" Native CLI output: {diagnostic}" if diagnostic else "")
            )

        # Vibe's streaming format emits completed public history entries rather
        # than a synthetic terminal record. Keep every entry exactly; the CLI
        # process return code remains the native completion authority.
        self.messages.extend(dict(event) for event in events)
        normalized_diagnostic = diagnostic.lower()
        quota_exhausted = any(
            marker in normalized_diagnostic
            for marker in ("rate limits exceeded", "status: 429", "http 429")
        )
        if quota_exhausted:
            self.provider_terminal_error = (
                "Mistral Vibe reported rate-limit exhaustion"
            )
            self.provider_terminal_error_kind = "quota"
        terminal = {
            "success": None,
            "native_terminal_record": False,
            "process_exit_is_authoritative": True,
        }
        if quota_exhausted:
            terminal.update(
                {
                    "success": False,
                    "failure_reason": self.provider_terminal_error,
                    "failure_kind": "quota",
                    "api_error_status": 429,
                }
            )
        self.provider_trace = {
            "format": "mistral-vibe-streaming-history-v1",
            "provider": "mistral_vibe",
            "stream_events": events,
            "raw_event_count": len(events),
            "unparsed_line_count": len(unparsed_lines),
            "terminal": terminal,
        }

        self._account_gateway_responses()


    def _process_logs(self):
        if self.config.cli_name == "claude_code_sub":
            self._process_claude_sub_logs(getattr(self, "_last_stdout", ""))
            return
        if self.config.cli_name == "codex_sub":
            self._process_codex_sub_logs(getattr(self, "_last_stdout", ""))
            return
        if self.config.cli_name == "mistral_vibe":
            self._process_mistral_vibe_logs(getattr(self, "_last_stdout", ""))
            return
        if self.config.cli_name == "antigravity_cli":
            self._process_antigravity_logs(getattr(self, "_last_stdout", ""))
            return

        traces = [record for record in self._gateway_records(self.model.get_traces_path())
                  if record.get("event") == "success"]
        if len(traces) == 0:
            self.logger.error("No successful traces found. Possible API error.")
            return

        if self.config.cli_name == "gemini_cli":
            self._process_gemini_logs(traces)
            self.model.cost_from_responses_fallback = True
            return

        messages = traces[-1].get("request", [])

        try:
            traces[-1].get("response", {}).get("choices", [{}])[0].get("message", {})
        except AttributeError:
            raise APIError("Malformed response in traces. Possible API error.")

        messages.append(
            traces[-1].get("response", {}).get("choices", [{}])[0].get("message", {})
        )
        self.messages.extend(messages)
        for trace in traces:
            self.model.responses.append(trace["response"])
        self.model.cost_from_responses_fallback = True

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = (
            asdict(self.config)
            | self.env.get_template_vars()
            | self.model.get_template_vars()
        )
        return Template(template).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def _normalize_openai_args(self, kwargs: dict) -> dict:
        """Some models use different urls (e.g. Claude Code)"""

        url = kwargs.get("base_url")

        if self.config.cli_name == "claude_code":
            if url is not None:
                kwargs["base_url"] = url.replace("/v1", "")
        elif self.config.cli_name == "claude_code_sub":
            # Claude Code uses the Anthropic-native gateway endpoint and a bare
            # model alias. Forward effort explicitly so the manifest and CLI
            # invocation cannot diverge.
            if url is not None:
                kwargs["base_url"] = url.removesuffix("/v1")
            model = kwargs.get("model")
            if isinstance(model, str):
                kwargs["model"] = model.split("/")[-1]
            model_kwargs = getattr(self.model.config, "model_kwargs", {}) or {}
            kwargs["reasoning_effort"] = model_kwargs.get("reasoning_effort", "medium")
        elif self.config.cli_name == "codex_sub":
            # The native Codex provider expects a bare OpenAI model id. Carry
            # reasoning effort separately because get_openai_args only exposes
            # proxy connection fields and the model name.
            model = kwargs.get("model")
            if isinstance(model, str):
                kwargs["model"] = model.split("/")[-1]
            model_kwargs = getattr(self.model.config, "model_kwargs", {}) or {}
            kwargs["reasoning_effort"] = model_kwargs.get("reasoning_effort", "medium")
        elif self.config.cli_name == "mistral_vibe":
            model = kwargs.get("model")
            if isinstance(model, str):
                kwargs["model"] = model.split("/")[-1]
        elif self.config.cli_name in {"gemini_cli", "antigravity_cli"}:
            # Google CLIs expect the Gemini-compatible endpoint root, without /v1.
            if url is not None:
                kwargs["base_url"] = url.removesuffix("/v1")
            kwargs["model"] = kwargs.get("model", "").split("/")[-1]
            model_kwargs = getattr(self.model.config, "model_kwargs", {}) or {}
            kwargs["reasoning_effort"] = model_kwargs.get(
                "reasoning_effort", "medium"
            )
        return kwargs

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""

        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.playback: dict = {}
        self.native_rollout: dict = {}
        self.provider_trace: dict = {}
        self.provider_terminal_error = ""
        self.provider_terminal_error_kind = ""
        self._last_exec_timed_out = False
        response_start = len(getattr(self.model, "responses", []))
        recorder = None
        playback_config = getattr(self, "playback_config", {}) or {}
        playback_mode = str(playback_config.get("mode", "external_replay")).strip()
        trajectory_config = getattr(self, "trajectory_config", {}) or {}
        native_rollout_enabled = bool(
            trajectory_config.get("capture_native_rollout", False)
            and self.config.cli_name == "codex_sub"
        )
        self._native_rollout_capture_attempted = False
        self._native_rollout_host_path = None
        capture_dir = None
        snapshot_dir = None
        snapshot_reference_prefix = ""
        if playback_config.get("enabled", False):
            if playback_mode not in {
                "external_replay",
                "in_container",
                "passive_action_replay",
                "standardized_replay",
                "host_side_checkpointing",
            }:
                raise ValueError(
                    "playback.mode must be 'external_replay', 'in_container', "
                    "'passive_action_replay', 'standardized_replay', or "
                    "'host_side_checkpointing'"
                )
            if (
                playback_mode == "passive_action_replay"
                and self.config.cli_name != "claude_code_sub"
            ):
                raise ValueError(
                    "passive_action_replay currently requires claude_code_sub"
                )
            if playback_mode == "standardized_replay" and self.config.cli_name not in {
                "claude_code_sub",
                "codex_sub",
            }:
                raise ValueError(
                    "standardized_replay requires claude_code_sub or codex_sub"
                )
            snapshot_root = getattr(self, "_playback_output_dir", None)
            if snapshot_root is not None:
                capture_index = getattr(self, "_playback_capture_index", 0) + 1
                self._playback_capture_index = capture_index
                capture_name = f"capture_{capture_index:03d}"
                capture_dir = Path(snapshot_root) / capture_name
                if playback_mode in {
                    "external_replay",
                    "passive_action_replay",
                    "standardized_replay",
                    "host_side_checkpointing",
                } or playback_config.get("capture_snapshots", True):
                    snapshot_dir = capture_dir
                snapshot_reference_prefix = f"playback/{capture_name}"
        if (
            playback_config.get("enabled", False)
            and (
                playback_mode == "host_side_checkpointing"
                or self.config.cli_name
                in {
                    "codex_sub",
                    "claude_code_sub",
                }
            )
            and callable(getattr(self.env, "execute_stream", None))
        ):
            try:
                if playback_mode == "standardized_replay":
                    if capture_dir is None:
                        raise RuntimeError(
                            "standardized replay requires an output directory"
                        )
                    target_file = getattr(self, "_target_file", "") or ""
                    target_dir = getattr(self, "_target_dir", "") or ""
                    source_scope = (
                        (target_file,)
                        if target_file
                        else (target_dir, target_dir.rstrip("/") + ".lean")
                        if target_dir
                        else (".",)
                    )
                    env_config = self.env.config
                    image_identity = getattr(self.env, "container_image_id", None)
                    recorder = StandardizedRunRecorder(
                        env=self.env,
                        output_dir=capture_dir,
                        source_scope=source_scope,
                        environment={
                            "image": str(getattr(env_config, "image", "")),
                            "image_id": (
                                image_identity() if callable(image_identity) else None
                            ),
                            "network_policy": str(
                                getattr(env_config, "network_policy", "none")
                            ),
                            "pids_limit": int(
                                getattr(env_config, "container_pids_limit", 0)
                            ),
                            "generator": self.config.cli_name,
                            "cli_version": self.config.host_tool_version,
                            "capture_backend": str(
                                playback_config.get(
                                    "capture_backend", "leanlean"
                                )
                            ),
                            "trace_utils_revision": playback_config.get(
                                "trace_utils_revision"
                            ),
                            "build_command": (
                                getattr(self, "_playback_build_command", "") or ""
                            ),
                        },
                        instance_id=str(
                            getattr(self, "_playback_instance_id", "") or ""
                        ),
                        capture_timeout_seconds=int(
                            playback_config.get("capture_timeout_seconds", 600)
                        ),
                    )
                    recorder.prepare()
                    recorder_type = None
                elif playback_mode == "passive_action_replay":
                    recorder_type = ClaudeActionPlaybackRecorder
                elif playback_mode == "external_replay":
                    recorder_type = ExternalPlaybackRecorder
                elif playback_mode == "host_side_checkpointing":
                    recorder_type = HostSideCheckpointRecorder
                else:
                    recorder_type = LivePlaybackRecorder
                if recorder_type is None:
                    pass
                else:
                    repository_scope = (
                        playback_mode in {"external_replay", "host_side_checkpointing"}
                        and bool(playback_config.get("capture_repository_scope"))
                    )
                    recorder_kwargs = {
                        "env": self.env,
                        "provider": self.config.cli_name,
                        "include_prefix": (
                            ""
                            if repository_scope
                            else getattr(self, "_target_dir", "") or ""
                        ),
                        "target_file": (
                            ""
                            if repository_scope
                            else getattr(self, "_target_file", "") or ""
                        ),
                        "exclude_dirs": tuple(
                            getattr(self, "_playback_exclude_dirs", ()) or ()
                        ),
                        "every_n_edits": int(playback_config.get("every_n_edits", 1)),
                        "snapshot_dir": snapshot_dir,
                        "snapshot_reference_prefix": snapshot_reference_prefix,
                        "instance_id": str(
                            getattr(self, "_playback_instance_id", "") or ""
                        ),
                        "manifest_path": (
                            capture_dir / "playback.json" if capture_dir else None
                        ),
                        "build_command": (
                            getattr(self, "_playback_build_command", "") or ""
                            if playback_config.get("build_each_checkpoint", False)
                            else ""
                        ),
                        "build_timeout_seconds": int(
                            playback_config.get("build_timeout_seconds", 3600)
                        ),
                        "build_output_chars": int(
                            playback_config.get("build_output_chars", 4000)
                        ),
                    }
                    if issubclass(recorder_type, ExternalPlaybackRecorder):
                        recorder_kwargs["capture_timeout_seconds"] = int(
                            playback_config.get("capture_timeout_seconds", 600)
                        )
                        recorder_kwargs["reject_asynchronous_actions"] = bool(
                            playback_config.get("reject_asynchronous_actions", False)
                        )
                    if recorder_type is HostSideCheckpointRecorder:
                        recorder_kwargs["reconciliation_interval_seconds"] = float(
                            playback_config.get(
                                "reconciliation_interval_seconds", 30
                            )
                        )
                    if recorder_type is ClaudeActionPlaybackRecorder:
                        recorder_kwargs["replay_action_timeout_seconds"] = int(
                            playback_config.get("replay_action_timeout_seconds", 600)
                        )
                        recorder_kwargs["bash_replay_policy"] = str(
                            playback_config.get("bash_replay_policy", "all")
                        )
                        recorder_kwargs["terminal_reconciliation"] = str(
                            playback_config.get("terminal_reconciliation", "none")
                        )
                    recorder = recorder_type(**recorder_kwargs)
                    recorder.prepare()
            except Exception:
                self.logger.exception(
                    "Could not initialize compression playback capture."
                )
                if playback_mode in {
                    "passive_action_replay",
                    "standardized_replay",
                    "host_side_checkpointing",
                }:
                    # Starting an unrecorded production run would silently
                    # violate the benchmark contract and cannot be repaired
                    # afterward because its baseline archive is missing.
                    raise
                recorder = None
        prompt = self.render_template(self.config.instance_template)

        if self.config.requires_model_proxy:
            begin_scope = getattr(self.model, "begin_accounting_scope", None)
            if callable(begin_scope):
                begin_scope()

        # Resolve the model endpoint before installation. Proxy-backed agents
        # receive a URL on a private internal network whose only peer is a
        # fixed-target relay to this exact local gateway.
        openai_args = self.model.get_openai_args()
        openai_args = self._normalize_openai_args(openai_args)
        if self.config.requires_model_proxy:
            attach_gateway = getattr(self.env, "attach_model_gateway", None)
            if not callable(attach_gateway):
                raise RuntimeError(
                    "proxy-backed agents require a model-only Docker network"
                )
            openai_args["base_url"] = attach_gateway(openai_args["base_url"])

        # Install pinned host tools after the container starts with no network.
        self._install()
        if self.config.cli_name == "mistral_vibe":
            self._prepare_mistral_vibe_home(openai_args)
        if self.config.cli_name == "antigravity_cli":
            self._prepare_antigravity_home()

        self._prepare_agent_contract()

        # Prepare the command
        oauth_token: str | None = None
        if self.config.cli_name == "claude_code_sub":
            from configs.model_constants import CLAUDE_CODE_OAUTH_TOKEN

            oauth_token = CLAUDE_CODE_OAUTH_TOKEN
        launch_command = self.config.launch_command.format(
            **openai_args, prompt=shlex.quote(prompt)
        )
        standardized_codex = (
            playback_mode == "standardized_replay"
            and self.config.cli_name == "codex_sub"
        )
        if native_rollout_enabled or standardized_codex:
            launch_command = self._enable_codex_rollout_persistence(launch_command)
        if recorder is not None and playback_mode != "standardized_replay":
            launch_command = playback_launch_command(launch_command, playback_mode)
        if self.config.subscription_transport in {
            "chatgpt",
            "claude_max",
        }:
            self._install_subscription_gateway_env(
                openai_args,
                oauth_token=oauth_token,
            )
        if native_rollout_enabled or standardized_codex:
            # Persisted Codex sessions require an existing private home. A
            # fresh CODEX_HOME also makes the exported rollout set exactly
            # this root session and all of its descendants.
            self._prepare_fresh_codex_home()
        if self.config.cli_name == "codex_sub" and self.config.codex_model_catalog:
            launch_command = self._install_codex_model_catalog(launch_command)
        if (
            self.config.cli_name == "codex_sub"
            and self.config.subscription_transport is None
        ):
            # Render first so a bad config cannot strand a copied credential.
            self._install_codex_subscription_auth()

        # Run the CLI. Hitting the container timeout is an expected outcome for
        # long agent runs, not a hard failure: the agent's work is on disk and
        # the litellm proxy has already logged every completed call's usage to
        # the traces file. If we let TimeoutExpired propagate here we skip
        # _process_logs() (below) and lose all cost accounting for the run — the
        # exact bug behind timed-out runs showing no cost. So catch it, fall
        # through to log/diff capture, and surface it as a non-zero exit.
        timed_out = False
        native_capture_manifest: str | None = None
        codex_rollout_manifest: str | None = None
        standardized_capture_error: BaseException | None = None
        try:
            if isinstance(recorder, StandardizedRunRecorder):
                capture = getattr(self.env, "execute_native_capture", None)
                if not callable(capture) or capture_dir is None:
                    raise RuntimeError(
                        "standardized replay requires exact Docker native capture"
                    )
                exec_result = capture(
                    launch_command,
                    capture_dir=capture_dir / "native",
                    capture_backend=str(
                        playback_config.get("capture_backend", "leanlean")
                    ),
                    harness=(
                        "claude-code"
                        if self.config.cli_name == "claude_code_sub"
                        else "codex-cli"
                    ),
                    model=str(openai_args.get("model") or ""),
                    forwards_subagent_stream=(
                        self.config.cli_name == "claude_code_sub"
                    ),
                    timeout=False,
                )
                native_capture_manifest = str(exec_result["native_capture_manifest"])
                timed_out = bool(exec_result.get("timed_out"))
            elif recorder is not None:
                # Passive production capture must not normalize events or write
                # semantic action artifacts while the agent is running. The
                # process runner only drains and buffers native stdout; Claude
                # parsing starts after process exit and terminal-tree capture.
                output_callback = (
                    None
                    if playback_mode == "passive_action_replay"
                    else recorder.ingest_line
                )
                exec_result = self.env.execute_stream(
                    launch_command, timeout=False, on_output=output_callback
                )
                timed_out = bool(exec_result.get("timed_out"))
            else:
                exec_result = self.env.execute(launch_command, timeout=False)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            # For the subscription path, usage/trajectory live on stdout (no
            # proxy traces file), so salvage whatever the agent streamed before
            # the timeout — TimeoutExpired carries the partial PIPE output.
            partial = exc.output or getattr(exc, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            exec_result = {
                "returncode": 124,
                "output": partial or "Container timed out",
            }
            self.logger.warning(
                "CLI agent hit the container timeout; capturing partial logs "
                "(for cost) and the on-disk diff before teardown."
            )
        except Exception:
            # env.execute normally returns non-zero commands instead of raising,
            # but Docker/transport failures can still throw before log parsing.
            if native_rollout_enabled:
                self._capture_codex_native_rollout()
            close_recorder = getattr(recorder, "close", None)
            if callable(close_recorder):
                close_recorder()
            self._post_exec()
            raise

        if recorder is not None and playback_mode in {
            "passive_action_replay",
            "standardized_replay",
        }:
            quiesce = getattr(self.env, "quiesce_after_agent_exit", None)
            if not callable(quiesce):
                self._post_exec()
                raise RuntimeError(
                    f"{playback_mode} requires post-exit process quiescence"
                )
            try:
                quiesce(
                    timeout=int(
                        playback_config.get("post_exit_quiescence_timeout_seconds", 15)
                    )
                )
                if isinstance(recorder, StandardizedRunRecorder):
                    recorder.capture_terminal_source(post_exit_quiesced=True)
                    try:
                        if standardized_codex:
                            exporter = getattr(
                                self.env, "export_codex_rollout_bundle", None
                            )
                            if not callable(exporter) or capture_dir is None:
                                raise RuntimeError(
                                    "standardized Codex replay requires rollout-tree export"
                                )
                            rollout = exporter(
                                capture_dir / "codex-rollouts",
                                fresh_home_proven=True,
                                timeout=int(
                                    trajectory_config.get(
                                        "native_rollout_timeout_seconds", 120
                                    )
                                ),
                            )
                            codex_rollout_manifest = str(rollout["manifest_path"])
                        if native_capture_manifest is None:
                            raise RuntimeError("native capture manifest is missing")
                        recorder.standardize_capture(
                            native_capture_manifest,
                            codex_rollout_manifest=codex_rollout_manifest,
                        )
                    except Exception as exc:
                        standardized_capture_error = exc
                        self.logger.exception(
                            "Could not standardize replay telemetry; the final "
                            "patch will still be extracted and evaluated."
                        )
                else:
                    recorder.post_exit_quiesced = True
            except Exception:
                # Do not stream a terminal archive while a timed-out CLI or a
                # detached tool child may still be writing. Credentials must
                # still be removed before aborting the run.
                self._post_exec()
                raise

        # Stash raw stdout so the subscription parser can read the stream-json
        # events the CLI emitted (the proxy path reads a traces file instead).
        self._last_stdout = exec_result.get("output", "")
        self._last_exec_timed_out = timed_out

        # Process the logs (populates model.responses for cost accounting).
        # Must run before _post_exec(), which deletes the traces. On a timeout
        # don't let a logs hiccup mask the rescued diff — the run already ended.
        try:
            try:
                self._process_logs()
                self._capture_gateway_trace()
            except Exception:
                if not timed_out:
                    raise
                self.logger.exception(
                    "Failed to process logs after timeout; continuing."
                )
        finally:
            if recorder is not None and not isinstance(
                recorder, StandardizedRunRecorder
            ):
                try:
                    mark_provider_exit = getattr(
                        recorder, "mark_provider_exit", None
                    )
                    if callable(mark_provider_exit):
                        mark_provider_exit(
                            timed_out=timed_out,
                            returncode=exec_result.get("returncode"),
                        )
                    if native_rollout_enabled and not standardized_codex:
                        self._capture_codex_native_rollout()
                        attach_costs = getattr(
                            recorder, "attach_codex_rollout_costs", None
                        )
                        if (
                            callable(attach_costs)
                            and self._native_rollout_host_path is not None
                        ):
                            try:
                                attach_costs(self._native_rollout_host_path)
                            except Exception:
                                self.logger.exception(
                                    "Could not join native Codex usage to live "
                                    "source checkpoints; finalizing the source "
                                    "checkpointing with inexact cost boundaries."
                                )
                    responses = list(getattr(self.model, "responses", []))[
                        response_start:
                    ]
                    if self.config.cli_name in {"muse_code", "mistral_vibe", "claude_code_sub"} or self.config.gateway_accounting:
                        attach_gateway = getattr(recorder, "attach_gateway_costs", None)
                        scope_id = getattr(self.model, "accounting_scope_id", None)
                        if callable(attach_gateway) and scope_id:
                            scoped_usage = self._gateway_records(self.model.get_cost_traces_path())
                            if scoped_usage:
                                attach_gateway(scoped_usage, scope_id=scope_id)
                    if isinstance(recorder, ClaudeActionPlaybackRecorder):
                        self.playback = recorder.finish(
                            responses,
                            raw_stdout=self._last_stdout,
                        )
                    else:
                        self.playback = recorder.finish(responses)
                except Exception:
                    self.logger.exception(
                        "Could not finalize compression playback capture."
                    )
            if native_rollout_enabled and not standardized_codex:
                self._capture_codex_native_rollout()
            # Always remove subscription credentials, including on parser/API
            # failures. This runs before diff extraction and image caching.
            self._post_exec()

        # Always extract a patch, even if the CLI exited with an error (e.g.
        # container timeout) or silently bailed out (e.g. codex exec deciding
        # it's done after one round of edits). Any committed or uncommitted
        # work the agent left on disk is still valuable — an empty patch
        # throws it all away.
        diff = self._extract_final_diff()
        if (
            isinstance(recorder, StandardizedRunRecorder)
            and standardized_capture_error is not None
        ):
            self.playback = _record_standardized_replay_failure(
                recorder.output_dir,
                phase="standardization",
                error=standardized_capture_error,
            )
        elif isinstance(recorder, StandardizedRunRecorder):
            bundle, bundle_failure = _attempt_standardized_replay_telemetry(
                lambda: recorder.finish(submitted_patch=diff),
                output_dir=recorder.output_dir,
                phase="bundle",
                logger=self.logger,
            )
            if bundle_failure is not None:
                self.playback = bundle_failure
                return self._submission_result(exec_result, diff)

            replay_result, replay_failure = _attempt_standardized_replay_telemetry(
                lambda: recorder.replay(
                    policy=ReplayPolicy(
                        bash=str(playback_config.get("bash_replay_policy", "all")),
                        terminal_reconciliation=str(
                            playback_config.get("terminal_reconciliation", "none")
                        ),
                        action_timeout_seconds=int(
                            playback_config.get("replay_action_timeout_seconds", 600)
                        ),
                        verify_each_mutation=bool(
                            playback_config.get("verify_each_mutation", False)
                        ),
                        build_after_each_mutation=bool(
                            playback_config.get("build_each_checkpoint", False)
                        ),
                        build_command=(
                            getattr(self, "_playback_build_command", "") or ""
                            if playback_config.get("build_each_checkpoint", False)
                            else ""
                        ),
                        build_timeout_seconds=int(
                            playback_config.get("build_timeout_seconds", 3600)
                        ),
                        build_output_chars=int(
                            playback_config.get("build_output_chars", 4000)
                        ),
                        reject_asynchronous_actions=bool(
                            playback_config.get("reject_asynchronous_actions", False)
                        ),
                    ),
                    final_cost_usd=cost_from_responses(
                        list(getattr(self.model, "responses", []))[response_start:]
                    ),
                    checkpoint_dir=(
                        recorder.output_dir / "replay-checkpoints"
                        if playback_config.get("retain_edit_checkpoints", False)
                        else None
                    ),
                ),
                output_dir=recorder.output_dir,
                phase="replay",
                logger=self.logger,
            )
            if replay_failure is not None:
                self.playback = replay_failure
                return self._submission_result(exec_result, diff)
            assert isinstance(replay_result, dict)
            self.playback = {
                "format": "code-harness-standardized-replay-result-v1",
                "capture_mode": "standardized_replay",
                "bundle_path": str(bundle),
                **replay_result,
            }
            if not replay_result.get("standardized_session_replay_exact"):
                replay_error = RuntimeError(
                    "standardized trajectory did not exactly reconstruct the "
                    "terminal source state"
                )
                self.logger.error(
                    "%s; the final patch will still be evaluated.", replay_error
                )
                self.playback = _record_standardized_replay_failure(
                    recorder.output_dir,
                    phase="replay_verification",
                    error=replay_error,
                )
        elif recorder is not None:
            try:
                if isinstance(recorder, ExternalPlaybackRecorder):
                    self.playback = recorder.replay_submission(diff)
                else:
                    self.playback = recorder.verify_submission(diff)
                if not self.playback.get("final_snapshot_matches_submission"):
                    self.logger.error(
                        "Playback final tree does not reconstruct the submitted "
                        "scoped diff; playback is marked invalid."
                    )
            except Exception:
                self.logger.exception(
                    "Could not replay or verify the final playback submission."
                )

        return self._submission_result(exec_result, diff)

    def _submission_result(self, exec_result: dict, diff: str) -> tuple[str, str]:
        """Return the captured patch even when execution or telemetry failed."""

        execution_failed = exec_result.get("returncode", 0) != 0 or bool(
            self.provider_terminal_error
        )
        if self.provider_terminal_error_kind == "quota":
            if diff.strip():
                self.add_message("user", diff)
            return "QuotaExceeded", diff if diff.strip() else ""
        if self.provider_terminal_error_kind == "authentication":
            if diff.strip():
                self.add_message("user", diff)
            return "AuthenticationFailed", diff if diff.strip() else ""
        if execution_failed and not diff.strip():
            # The result payload is always consumed downstream as a git patch.
            # Diagnostics remain in the native provider trace and run log; raw
            # stream-json or shell errors must never be handed to git apply.
            return "ExecutionFailed", ""

        # A non-empty on-disk diff remains the submission even when telemetry
        # or the provider process failed after making edits.
        self.add_message("user", diff)
        return "Submitted", diff

    def _extract_final_diff(self) -> str:
        """Capture the agent's work as a single patch against the base commit.

        Runs primary extraction (`git diff` vs initial commit after staging
        everything, so both committed and uncommitted work is captured).
        Falls back to `git diff HEAD` (uncommitted-only) if the primary
        extraction yields no output — useful when the reset/rev-list pipeline
        misbehaves in some container state. Logs sizes either way for
        diagnostics.
        """
        # Make sure build artefacts stay ignored before we `git add -A`.
        self.env.execute(
            "cd /testbed && "
            "printf '\\n.venv\\nvenv\\n.env\\nenv\\n*-venv\\n*_venv\\n.lake/\\n' "
            ">> .git/info/exclude",
            timeout=False,
        )

        # File-level mode: scope the captured diff to the single allowed file, so
        # any edits the agent made to other files are dropped (i.e. "reverted").
        # Repo-level mode keeps the whole tree (minus .lake/).
        target_file = getattr(self, "_target_file", "") or ""
        target_dir = getattr(self, "_target_dir", "") or ""
        if target_file:
            pathspec = f"-- {shlex.quote(target_file)}"
        elif target_dir:
            # Subproject mode (LeanPool): scope the diff to the subproject directory
            # and its sibling module file, dropping any edits outside them.
            d = target_dir.rstrip("/")
            pathspec = f"-- {shlex.quote(d + '/')} {shlex.quote(d + '.lean')}"
        else:
            pathspec = "-- ':(exclude).lake/'"

        # Primary: diff the current working tree (committed + uncommitted)
        # against the init commit recorded at image-build time.  Falls back to
        # rev-list for images built before /.init_commit was introduced.
        primary_cmd = (
            "cd /testbed && "
            "git add -A && "
            "base=$(cat /.init_commit 2>/dev/null || git rev-list --max-parents=0 HEAD | tail -1) && "
            f"git diff $base {pathspec}"
        )
        primary_result = self.env.execute(primary_cmd, timeout=False)
        primary_diff = _strip_binary_diff_sections(primary_result.get("output", ""))
        self.logger.info(
            "Final diff extraction (primary): rc=%s, diff_chars=%d",
            primary_result.get("returncode"),
            len(primary_diff),
        )
        if primary_diff.strip():
            return primary_diff

        # Fallback: uncommitted-only diff (scoped to the target file in file-level
        # mode, same as the primary path).
        fb_result = self.env.execute(
            f"cd /testbed && git add -A && git diff --cached HEAD {pathspec}",
            timeout=False,
        )
        fb_diff = _strip_binary_diff_sections(fb_result.get("output", ""))
        self.logger.info(
            "Final diff extraction (fallback): rc=%s, diff_chars=%d",
            fb_result.get("returncode"),
            len(fb_diff),
        )
        if fb_diff.strip():
            return fb_diff

        # Last-ditch: log git status so we at least know what state the
        # working tree was in.
        status_result = self.env.execute(
            "cd /testbed && git status --porcelain=v1 && "
            "echo --- && git log --oneline -5",
            timeout=False,
        )
        self.logger.warning(
            "Final diff extraction produced no output. git status/log:\n%s",
            status_result.get("output", "")[:2000],
        )
        return ""
