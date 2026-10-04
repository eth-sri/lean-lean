"""Pinned Muse Code headless adapter; credentials and state are invocation-local."""
import json
import shlex
from pathlib import Path

from leanlean.generators.cli_agent import CLIAgent, APIError

MUSE_HOME = "/tmp/leanlean-muse"


def resolve_muse_standalone_tool(version: str) -> Path:
    from scripts.install_muse_standalone import VERSION, release_path, verify
    if version != VERSION:
        raise ValueError("Muse version must match the checksum-pinned installer")
    path = release_path()
    if not path.is_file():
        raise RuntimeError("Install the host bundle: python scripts/install_muse_standalone.py")
    verify(path)
    return path


def validate_reasoning_effort(effort: str, model: str) -> None:
    allowed = {"low", "medium", "high", "xhigh"}
    # Meta's Spark 1.3 release adds max; do not silently enable it for older models.
    if model in {"muse-spark-1.3", "muse-spark-1.3-contributor"}:
        allowed.add("max")
    if effort not in allowed:
        raise ValueError(f"Unsupported reasoning effort {effort!r} for {model}")


# Every skill bundled with the pinned Muse Code 1.0.3-R2198.1 (`muse skills list --source all`).
MUSE_BUNDLED_SKILLS = tuple(f"bundled://muse-core/skills/{name}/SKILL.md" for name in (
    "browser-app-delivery", "create-skill", "doctor", "git", "greenfield-project-scaffolding",
    "grill", "grill-and-record", "import", "manage-settings", "plan", "python-env",
    "read-session", "table-fit", "taste",
))


# Native Muse tools beyond the shell/file core that Claude Code (Bash, Edit, Read, Write) and
# Codex (exec_command, write_stdin, apply_patch, view_image) run with in the benchmark.
MUSE_NON_CORE_TOOLS = (
    "search", "read_memory", "add_memory", "edit_memory", "cron_create", "cron_delete",
    "cron_list", "get_goal", "create_goal", "update_goal", "report_progress",
    "snooze_reminder", "write_todos",
)


def muse_native_settings(base_url: str, *, extra_excluded_tools: tuple[str, ...] = ()) -> dict:
    """Shipped Muse defaults under the benchmark-wide rule: no skills, MCP or subagents.

    Subagents cover delegation, the multi-agent `workflow` tool and the reminder
    observers, which are separate model-calling agents beside the main one.
    """
    return {
        "schema_version": 1,
        "endpoint_transport": {"base_url": base_url, "auth": "bearer"},
        "run": {
            "subagent_delegation_mode": "off",
            "reminder_roster": {"preset": "none"},
            "context_slimming": {
                "excluded_tool_names": ["workflow", "read_skill", *extra_excluded_tools],
            },
        },
        "skills": {"activation": {"bundled": {path: "off" for path in MUSE_BUNDLED_SKILLS}}},
        "mcpServers": {},
        "telemetry": {"enabled": False},
    }


class MuseCodeNativeAgent(CLIAgent):
    """Muse Code with shipped defaults; see the muse_code_native generator config."""

    def _install(self):
        self.env.copy_host_executable(
            resolve_muse_standalone_tool(self.config.host_tool_version),
            "/usr/local/bin/muse",
        )

    def _normalize_openai_args(self, kwargs):
        # This shared journal contains earlier repositories too. Remember where
        # this invocation starts so completed costs are attributed exactly once.
        getter = getattr(self.model, "get_cost_traces_path", None)
        trace = getter() if callable(getter) else None
        self._muse_trace_offset = trace.stat().st_size if trace is not None and trace.is_file() else 0
        kwargs["model"] = kwargs["model"].split("/")[-1]
        kwargs["reasoning_effort"] = self.model.config.model_kwargs["reasoning_effort"]
        # CLIAgent replaces base_url with the private container relay after
        # normalization. Keep that mapping, not the pre-attachment host URL.
        self._muse_openai_args = kwargs
        return kwargs

    def _prepare_agent_contract(self):
        super()._prepare_agent_contract()
        if self.config.enable_subagents:
            raise ValueError("The Muse adapter requires enable_subagents: false")
        settings = json.dumps(self._muse_settings())
        result = self.env.execute(
            f"rm -rf -- {MUSE_HOME} && install -d -m 700 {MUSE_HOME}/config/muse {MUSE_HOME}/data "
            f"&& printf %s {shlex.quote(settings)} > {MUSE_HOME}/config/muse/settings.json",
            timeout=True,
        )
        if result.get("returncode", 1) != 0:
            raise APIError("Could not install the isolated Muse profile")

    def _muse_settings(self) -> dict:
        return muse_native_settings(self._muse_openai_args["base_url"])

    def _process_logs(self):
        self.provider_terminal_error = ""
        self.provider_terminal_error_kind = ""
        events = []
        diagnostics = []
        for line in getattr(self, "_last_stdout", "").splitlines():
            try:
                event = json.loads(line)
                if isinstance(event, dict) and "payload_type" in event:
                    events.append(event)
            except ValueError:
                diagnostics.append(line)
        self.messages.extend(events)
        terminal = {"process_exit_is_authoritative": True, "success": False}
        for event in events:
            if event["payload_type"].startswith("run.terminal."):
                payload = event.get("payload", {})
                terminal.update(success=payload.get("terminal") == "completed",
                                reason=payload.get("reason"), state=payload.get("terminal"))
        self.provider_trace = {
            "format": "muse-code-jsonl-v1", "provider": "muse_code",
            "stream_events": events, "raw_event_count": len(events),
            "terminal": terminal, "diagnostics": diagnostics[-30:],
            "standardized_session_replay_exact": False,
        }
        if (getattr(self, "trajectory_config", {}) or {}).get(
            "capture_native_rollout", False
        ):
            self.native_rollout = {
                "format": "muse-code-jsonl-v1",
                "status": "complete",
                "stream_events": events,
                "raw_event_count": len(events),
                "diagnostics": diagnostics[-30:],
            }
        if not terminal["success"]:
            self.provider_terminal_error = str(terminal.get("reason") or
                "Muse Code ended without a successful terminal event")
            self.provider_terminal_error_kind = "provider"
            terminal["failure_reason"] = self.provider_terminal_error
        # A startup rejection produces no billable trace. Preserve its native
        # diagnostics and patch instead of masking the failure with FileNotFound.
        trace_path = self.model.get_cost_traces_path()
        scope_id = getattr(self.model, "accounting_scope_id", None)
        self.provider_trace["accounting_scope"] = scope_id
        if trace_path.is_file():
            seen = {response.get("id") for response in self.model.responses if response.get("id")}
            with trace_path.open("rb") as stream:
                stream.seek(getattr(self, "_muse_trace_offset", 0))
                for line in stream:
                    trace = json.loads(line)
                    # Offsets exclude earlier calls, but concurrent workers also
                    # append after this offset. Only this invocation owns its scope.
                    if (scope_id is not None
                            and (trace.get("agent_context") or {}).get("accounting_scope") != scope_id):
                        continue
                    if trace.get("event") == "success":
                        response = trace["response"]
                        response_id = response.get("id")
                        if response_id and response_id in seen:
                            continue
                        self.model.responses.append(response)
                        if response_id:
                            seen.add(response_id)
        self.model.cost_from_responses_fallback = True

    def _submission_result(self, exec_result, diff):
        terminal = self.provider_trace.setdefault("terminal", {})
        terminal["returncode"] = exec_result.get("returncode")
        if exec_result.get("returncode", 0) != 0:
            terminal["success"] = False
            terminal.setdefault("failure_reason", "Muse Code exited unsuccessfully")
        return super()._submission_result(exec_result, diff)

    def _post_exec(self):
        self.env.execute(f"rm -rf -- {MUSE_HOME}", timeout=True)
        super()._post_exec()


class MuseCodeCoreAgent(MuseCodeNativeAgent):
    """Native Muse limited to the shell/file core; see the muse_code_core generator config."""

    def _muse_settings(self) -> dict:
        return muse_native_settings(
            self._muse_openai_args["base_url"], extra_excluded_tools=MUSE_NON_CORE_TOOLS,
        )
