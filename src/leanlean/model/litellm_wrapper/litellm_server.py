from __future__ import annotations

import atexit
from leanlean.utils import json_utils as json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from openai import OpenAI
import logging
import socket

from leanlean.subscription_auth import (
    EXPLICIT_AUTH_ENV,
    strip_subscription_credentials,
    subscription_auth_record,
)
from leanlean import Model
from leanlean.utils.others import retry
from configs.model_constants import MISTRAL_VIBE_COMPACTION_MODEL, get_litellm_model_info
from leanlean.model.costs import cost_from_responses
from .litellm_logger import LOG_PATH
from .accounting import record_identity, record_scope, scoped_url

_LITELLM_PROXY_START = """
from leanlean.subscription_auth import install_explicit_chatgpt_auth
install_explicit_chatgpt_auth()
from litellm.proxy.proxy_cli import ProxyInitializationHelpers, run_server
from litellm.proxy.proxy_server import app
from leanlean.model.litellm_wrapper.accounting import AccountingScopeMiddleware
app.add_middleware(AccountingScopeMiddleware)
from leanlean.model.litellm_wrapper.stream_usage import install_stream_usage_capture
install_stream_usage_capture()
from leanlean.model.litellm_wrapper.raw_provider_stream import install_raw_provider_stream_capture
install_raw_provider_stream_capture()
from leanlean.model.litellm_wrapper.mistral_reasoning import install_mistral_reasoning_replay
install_mistral_reasoning_replay()
from leanlean.model.litellm_wrapper.request_rewrite import ReasoningEffortRewriteMiddleware
app.add_middleware(ReasoningEffortRewriteMiddleware)

# Muse Code refuses to start without its model catalog; serve the proxy's bare model ids.
@app.get("/muse-code/models")
async def _muse_code_models():
    from litellm.proxy import proxy_server
    ids = sorted({{m["model_name"] for m in (proxy_server.llm_model_list or []) if "/" not in m["model_name"]}})
    return {{"object": "list", "data": [{{"id": i, "object": "model", "owned_by": "meta"}} for i in ids]}}

ProxyInitializationHelpers._get_loop_type = staticmethod(lambda: "asyncio")
run_server.main(args={args!r}, standalone_mode=False)
"""

_PROXY_API_KEYS: dict[tuple[str, int], str] = {}
_PROXY_API_KEYS_LOCK = threading.Lock()


@dataclass
class _TraceCostState:
    """Incremental cost state shared by every client of one proxy trace."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    identity: tuple[int, int] | None = None
    prefix: bytes = b""
    offset: int = 0
    cost: float = 0.0
    calls: int = 0
    scopes: dict[str, tuple[float, int]] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)

    def reset(self, identity: tuple[int, int] | None = None) -> None:
        self.identity = identity
        self.prefix = b""
        self.offset = 0
        self.cost = 0.0
        self.calls = 0
        self.scopes.clear()
        self.seen.clear()


_TRACE_COST_STATES: dict[str, _TraceCostState] = {}
_TRACE_COST_STATES_LOCK = threading.Lock()


def _trace_cost_state(path: Path) -> _TraceCostState:
    key = str(path.resolve())
    with _TRACE_COST_STATES_LOCK:
        return _TRACE_COST_STATES.setdefault(key, _TraceCostState())


def _reset_trace_cost_state(path: Path) -> None:
    state = _trace_cost_state(path)
    with state.lock:
        state.reset()


@dataclass
class LitellmServerConfig:
    model_name: str
    upstream_model: str | None = None
    api_base: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    subscription_transport: str | None = None
    preserve_shared_traces: bool = False
    # Client reasoning efforts rewritten before LiteLLM reads the request (request_rewrite.py).
    reasoning_effort_map: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LitellmServerConfig:
        return LitellmServerConfig(
            model_name=d["model_name"],
            upstream_model=d.get("upstream_model"),
            api_base=d.get("api_base"),
            model_kwargs=d.get("model_kwargs", {}) or {},
            subscription_transport=d.get("subscription_transport"),
            preserve_shared_traces=bool(d.get("preserve_shared_traces", False)),
            reasoning_effort_map=dict(d.get("reasoning_effort_map") or {}),
        )

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _environment_references(value: Any) -> set[str]:
    """Collect LiteLLM os.environ/VAR references from nested config values."""

    if isinstance(value, str) and value.startswith("os.environ/"):
        name = value.removeprefix("os.environ/")
        return {name} if name else set()
    if isinstance(value, dict):
        references: set[str] = set()
        for item in value.values():
            references.update(_environment_references(item))
        return references
    if isinstance(value, (list, tuple)):
        references = set()
        for item in value:
            references.update(_environment_references(item))
        return references
    return set()

def _find_free_port() -> int:
    """Ask the OS for a free TCP port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_primary_ip(fallback: str = "127.0.0.1") -> str:
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return fallback
    finally:
        s.close()


class LitellmServer(Model):
    """
    Simple wrapper that:
      1) Launches a LiteLLM OpenAI-compatible proxy on localhost
      2) Lets you query the configured model directly via litellm
      3) Record the traces independently of the agents used
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        log_dir: str = "logs/litellm_server",
        startup_wait_s: float = 30.0,
        auto_cost_updates: bool = True,
        cost_poll_interval_s: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.config = LitellmServerConfig(**kwargs)
        self.cost: float = 0.0
        self.n_calls: int = 0
        self.responses: list[dict[str, Any]] = []
        self.accounting_scope_id: str | None = None
        # After parsing a completed invocation, durable native/scoped responses
        # outrank the live proxy journal. During an invocation the scope cursor
        # supplies live cost without including any other worker's requests.
        self.cost_from_responses_fallback: bool = False

        self._host = host
        self._bind_host = get_primary_ip() if host == "0.0.0.0" else host
        self._port = port if port != 0 else _find_free_port()
        self._startup_wait_s = startup_wait_s

        self._log_dir = _ensure_dir(log_dir)
        self._server_cfg_path: str | None = None
        self._server_proc: subprocess.Popen | None = None
        self._server_log_fp = None
        self._proxy_url = f"http://{self._bind_host}:{self._port}/v1"
        self._chatgpt_auth_dir: Path | None = None

        self._proxy_registry_key = (self._bind_host, self._port)
        with _PROXY_API_KEYS_LOCK:
            existing_key = _PROXY_API_KEYS.get(self._proxy_registry_key)
            self._owns_proxy_key = existing_key is None
            if existing_key is None:
                existing_key = "".join(
                    random.choices(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        k=64,
                    )
                )
                _PROXY_API_KEYS[self._proxy_registry_key] = existing_key
        self._api_key = existing_key
        self._auto_cost_updates = auto_cost_updates
        self._cost_poll_interval_s = cost_poll_interval_s
        self._cost_lock = threading.Lock()
        self._cost_watcher_stop_event = threading.Event()
        self._cost_watcher_thread: threading.Thread | None = None

        self.logger = logging.getLogger("leanlean.model.litellm_server")

        atexit.register(self.stop)

    def update_port(self, new_port: int) -> None:
        """Update the port used by the server (must be called before serve())."""
        old_registry_key = self._proxy_registry_key
        self._port = new_port
        self._proxy_url = f"http://{self._bind_host}:{self._port}/v1"
        self._proxy_registry_key = (self._bind_host, self._port)
        with _PROXY_API_KEYS_LOCK:
            if (
                self._owns_proxy_key
                and _PROXY_API_KEYS.get(old_registry_key) == self._api_key
            ):
                _PROXY_API_KEYS.pop(old_registry_key, None)
            existing_key = _PROXY_API_KEYS.get(self._proxy_registry_key)
            self._owns_proxy_key = existing_key is None
            if existing_key is None:
                _PROXY_API_KEYS[self._proxy_registry_key] = self._api_key
            else:
                self._api_key = existing_key

    def get_api_key(self) -> str:
        return self._api_key

    def begin_accounting_scope(self) -> str:
        """Start one invocation without truncating the run-wide journal."""
        self.accounting_scope_id = uuid.uuid4().hex
        self.cost_from_responses_fallback = False
        self.config.preserve_shared_traces = True
        self.cost = 0.0
        self.n_calls = 0
        return self.accounting_scope_id

    def get_traces_path(self) -> Path:
        return Path(LOG_PATH) / f"proxy_{self._port}.json"

    def get_cost_traces_path(self) -> Path:
        return Path(LOG_PATH) / f"proxy_{self._port}.cost.jsonl"

    def get_provider_stream_path(self) -> Path:
        return Path(LOG_PATH) / f"proxy_{self._port}.provider-stream.jsonl"

    def _initialize_trace_files(self) -> None:
        """Start one proxy lifetime with fresh forensic and accounting logs."""

        for path in (self.get_traces_path(), self.get_cost_traces_path()):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _reset_trace_cost_state(self.get_cost_traces_path())

    def delete_traces(self) -> None:
        self.logger.debug("Deleting traces for proxy port %s", self._port)
        if (
            self.config.subscription_transport
            or self.config.preserve_shared_traces
        ):
            # Subscription workers share one proxy. A worker must not truncate
            # the run-global forensic or accounting journal while its peers are
            # still active. The owning proxy initializes fresh files in serve()
            # and intentionally preserves them when it stops.
            self.logger.debug(
                "Preserving shared subscription traces for proxy port %s",
                self._port,
            )
            return
        for traces_path in (self.get_traces_path(), self.get_cost_traces_path()):
            try:
                traces_path.unlink()
            except FileNotFoundError:
                pass
        _reset_trace_cost_state(self.get_cost_traces_path())

    def archive_traces(self, directory: str | Path) -> dict[str, Path]:
        """Move stopped proxy journals into a durable run-owned directory."""

        if self._server_proc and self._server_proc.poll() is None:
            raise RuntimeError("cannot archive traces while the proxy is running")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        sources = {
            "forensic": self.get_traces_path(),
            "accounting": self.get_cost_traces_path(),
            "provider-stream": self.get_provider_stream_path(),
        }
        archived: dict[str, Path] = {}
        for label, source in sources.items():
            if not source.is_file():
                continue
            target = destination / (
                f"proxy_{self._port}.{label}.jsonl"
            )
            if target.exists():
                raise FileExistsError(f"proxy trace archive already exists: {target}")
            shutil.move(str(source), str(target))
            archived[label] = target
        return archived

    def _gateway_model_name(self) -> str:
        if self.config.subscription_transport == "chatgpt":
            return "chatgpt/" + self.config.model_name.split("/")[-1]
        if self.config.upstream_model:
            return self.config.upstream_model
        return self.config.model_name

    def _gateway_model_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.config.model_kwargs or {})
        if self.config.subscription_transport in {"chatgpt", "claude_max"}:
            # Upstream credentials come from the private ChatGPT token cache or
            # the forwarded Claude OAuth Authorization header, never config.
            kwargs.pop("api_key", None)
        return kwargs

    def _validate_upstream_environment(self, child_env: dict[str, str]) -> None:
        required = _environment_references(self._gateway_model_kwargs())
        missing = sorted(name for name in required if not child_env.get(name))
        if not missing:
            return
        variables = ", ".join(missing)
        raise RuntimeError(
            f"Model {self.config.model_name!r} requires upstream environment "
            f"variable(s): {variables}. Load them through secret.sh before "
            "starting LiteLLM."
        )

    def _build_proxy_config(self) -> dict[str, Any]:
        """Build a testable LiteLLM config without writing credential files."""

        alias = self.config.model_name
        alias_bare = alias.split("/", 1)[-1] if "/" in alias else None
        upstream_model = self._gateway_model_name()
        model_info = get_litellm_model_info(self.config.model_name)
        model_kwargs = self._gateway_model_kwargs()
        if alias.split("/")[-1] == "glm-5.3-flash":
            # The Anthropic adapter drops unrecognized OpenAI reasoning fields.
            # Preserve GLM's provider-specific controls at the HTTP body boundary.
            model_kwargs["extra_body"] = {
                **model_kwargs.get("extra_body", {}),
                "thinking": model_kwargs.get("thinking", {"type": "enabled"}),
                "reasoning_effort": model_kwargs.get("reasoning_effort", "max"),
            }

        def _model_entry(name: str, upstream: str = upstream_model) -> dict:
            entry: dict[str, Any] = {
                "model_name": name,
                "litellm_params": {
                    "model": upstream,
                    **(
                        {"api_base": self.config.api_base}
                        if self.config.api_base
                        and not self.config.subscription_transport
                        else {}
                    ),
                    **model_kwargs,
                },
            }
            entry_info = dict(model_info or {})
            if self.config.subscription_transport == "chatgpt":
                entry_info["mode"] = "responses"
            if entry_info:
                entry["model_info"] = entry_info
            return entry

        names = [alias]
        if alias_bare and alias_bare != alias:
            names.append(alias_bare)
        model_list = [_model_entry(name) for name in names]

        # Claude Code may select Haiku for a subagent even when the main model
        # is Opus or Sonnet. Register both provider and bare aliases.
        if upstream_model.startswith("anthropic/"):
            haiku = "anthropic/claude-haiku-4-5-20251001"
            haiku_info = get_litellm_model_info(haiku)
            existing = {entry["model_name"] for entry in model_list}
            for haiku_alias in (haiku, haiku.split("/", 1)[-1]):
                if haiku_alias in existing:
                    continue
                haiku_entry: dict[str, Any] = {
                    "model_name": haiku_alias,
                    "litellm_params": {
                        "model": haiku,
                        **model_kwargs,
                    },
                }
                if haiku_info:
                    haiku_entry["model_info"] = haiku_info
                model_list.append(haiku_entry)

        # Mistral Vibe's native Lean agent compacts with Mistral Small at its
        # own temperature and without thinking, so none of Leanstral's
        # deployment parameters apply to it.
        if upstream_model == "mistral/labs-leanstral-1-5":
            compaction = f"mistral/{MISTRAL_VIBE_COMPACTION_MODEL}"
            compaction_entry: dict[str, Any] = {
                "model_name": MISTRAL_VIBE_COMPACTION_MODEL,
                "litellm_params": {
                    "model": compaction,
                    **({"api_base": self.config.api_base} if self.config.api_base else {}),
                    **{
                        key: model_kwargs[key]
                        for key in ("api_key", "drop_params")
                        if key in model_kwargs
                    },
                },
            }
            compaction_info = get_litellm_model_info(compaction)
            if compaction_info:
                compaction_entry["model_info"] = compaction_info
            model_list.append(compaction_entry)

        # Gemini CLI uses this alias internally for loop detection.
        if upstream_model.startswith("gemini/"):
            flash_aliases = [
                "gemini/gemini-3-flash-preview",
                "gemini-3-flash-preview",
            ]
            if upstream_model == "gemini/gemini-3.8-flash":
                flash_aliases.extend(
                    f"gemini-3.8-flash-{effort}"
                    for effort in ("low", "medium", "high")
                )
            for flash_model in flash_aliases:
                model_list.append(
                    {
                        "model_name": flash_model,
                        "litellm_params": {
                            "model": (
                                flash_model
                                if "/" in flash_model
                                else f"gemini/{flash_model}"
                            ),
                            **model_kwargs,
                        },
                    }
                )

        general_settings: dict[str, Any] = {
            "drop_params": True,
            "disable_responses_id_security": True,
            "master_key": "os.environ/LITELLM_MASTER_KEY",
        }
        if self.config.subscription_transport == "claude_max":
            general_settings["forward_client_headers_to_llm_api"] = True

        config = {
            "model_list": model_list,
            "general_settings": general_settings,
            "litellm_settings": {
                "json_logs": True,
                "callbacks": "litellm_logger.file_logger",
            },
        }
        if alias.split("/")[-1] == "glm-5.3-flash":
            # Claude Code speaks /messages, but Z.ai's Coding endpoint only
            # supports chat/completions, not the OpenAI Responses API.
            config["litellm_settings"]["use_chat_completions_url_for_anthropic_messages"] = True
        if self.config.subscription_transport == "chatgpt":
            # Keep upstream 401 errors instead of a synthetic cooldown 429.
            config["router_settings"] = {"disable_cooldowns": True}
        return config

    def _prepare_chatgpt_auth(self, child_env: dict[str, str]) -> None:
        if self.config.subscription_transport != "chatgpt":
            return
        auth_record = subscription_auth_record()
        private_dir = Path(tempfile.mkdtemp(prefix="leanlean-chatgpt-auth-"))
        destination = private_dir / "auth.json"
        destination.write_text(json.dumps(auth_record), encoding="utf-8")
        destination.chmod(0o600)
        private_dir.chmod(0o700)
        self._chatgpt_auth_dir = private_dir
        child_env["CHATGPT_TOKEN_DIR"] = str(private_dir)
        child_env["CHATGPT_AUTH_FILE"] = destination.name
        child_env[EXPLICIT_AUTH_ENV] = "1"
        strip_subscription_credentials(child_env)

    # ------------- public API -------------
    def serve(self) -> None:
        """
        Start a LiteLLM proxy on localhost using a generated config.
        The proxy will expose an OpenAI-compatible API at: http://host:port/v1
        """
        if self._server_proc and self._server_proc.poll() is None:
            return

        self.logger.debug(
            "Model %r via LiteLLM transport=%r",
            self.config.model_name,
            self.config.subscription_transport or "api",
        )
        proxy_cfg = self._build_proxy_config()
        child_env = os.environ.copy()
        # Some developer shells export DEBUG=release for compiled tooling.
        # LiteLLM's Click CLI interprets DEBUG as a boolean flag envvar and crashes on that value.
        if child_env.get("DEBUG") == "release":
            child_env.pop("DEBUG", None)
        child_env["LITELLM_MASTER_KEY"] = self._api_key
        child_env["LEANLEAN_LITELLM_TRACE_ID"] = str(self._port)
        if self.config.reasoning_effort_map:
            child_env["LEANLEAN_REASONING_EFFORT_MAP"] = json.dumps(self.config.reasoning_effort_map)
        self._validate_upstream_environment(child_env)
        self._prepare_chatgpt_auth(child_env)
        self._initialize_trace_files()

        try:
            # Write to a temp file (next to logs for convenience)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", dir=str(self._log_dir)
            )
            json.dump(proxy_cfg, tmp, indent=2)
            tmp.flush()
            tmp.close()
            self._server_cfg_path = tmp.name

            # Copy the callback to the log dir so LiteLLM can import it
            logger_src = Path(__file__).with_name("litellm_logger.py")
            logger_dest = self._log_dir / "litellm_logger.py"
            shutil.copy(logger_src, logger_dest)

            # Open a log file for the proxy process
            log_path = self._log_dir / f"proxy_{self._port}.log"
            self._server_log_fp = open(log_path, "a", buffering=1)

            cli_args = [
                "--config",
                self._server_cfg_path,
                "--host",
                self._host,
                "--port",
                str(self._port),
            ]
            cmd = [sys.executable, "-c", _LITELLM_PROXY_START.format(args=cli_args)]

            self.logger.debug(f"Starting LiteLLM proxy with command: {' '.join(cmd)}")
            self._server_proc = subprocess.Popen(
                cmd,
                stdout=self._server_log_fp,
                stderr=self._server_log_fp,
                close_fds=True,
                env=child_env,
            )
            self.logger.debug(f"LiteLLM proxy started with PID: {self._server_proc.pid}")

            self._wait_until_ready()
            self._start_cost_watcher()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the proxy and clean up temp files."""

        self._stop_cost_watcher()
        self.delete_traces()

        if self._server_proc:
            self.logger.debug(f"Stopping LiteLLM proxy with PID: {self._server_proc.pid}")
        if self._server_proc and self._server_proc.poll() is None:
            try:
                self._server_proc.terminate()
                try:
                    self._server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server_proc.kill()
            finally:
                self._server_proc = None

        if self._server_log_fp:
            try:
                self._server_log_fp.flush()
            finally:
                self._server_log_fp.close()
            self._server_log_fp = None

        if self._server_cfg_path and os.path.exists(self._server_cfg_path):
            try:
                os.remove(self._server_cfg_path)
            except OSError:
                pass
            self._server_cfg_path = None
        if self._chatgpt_auth_dir is not None:
            shutil.rmtree(self._chatgpt_auth_dir, ignore_errors=True)
            self._chatgpt_auth_dir = None
        if self._owns_proxy_key:
            with _PROXY_API_KEYS_LOCK:
                if _PROXY_API_KEYS.get(self._proxy_registry_key) == self._api_key:
                    _PROXY_API_KEYS.pop(self._proxy_registry_key, None)
            self._owns_proxy_key = False

    def get_openai_args(self) -> dict[str, Any]:
        api_key = self.get_api_key()
        base_url = self.proxy_url
        if self.accounting_scope_id:
            base_url = scoped_url(base_url, self.accounting_scope_id)
        model_name = self.config.model_name
        return {"api_key": api_key, "base_url": base_url, "model": model_name}

    @retry(n_attempts=5) # For API errors / timeouts
    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        openai_args = self.get_openai_args()

        model_name = openai_args.pop("model")

        client = OpenAI(**openai_args)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    def get_cost(self) -> float:
        # Completed native/scoped responses outrank a still-live shared journal.
        if self.cost_from_responses_fallback:
            with self._cost_lock:
                self.cost = cost_from_responses(self.responses)
                self.n_calls = len(self.responses)
                return self.cost
        traces_path = self.get_cost_traces_path()
        state = _trace_cost_state(traces_path)
        with state.lock:
            try:
                stream = traces_path.open("rb")
            except FileNotFoundError:
                state.reset()
                if self.cost_from_responses_fallback:
                    total_cost = cost_from_responses(self.responses)
                    call_count = len(self.responses)
                else:
                    total_cost = 0.0
                    call_count = 0
            else:
                with stream:
                    metadata = os.fstat(stream.fileno())
                    identity = (metadata.st_dev, metadata.st_ino)
                    reset_required = (
                        state.identity != identity
                        or metadata.st_size < state.offset
                    )
                    if not reset_required and state.prefix:
                        stream.seek(0)
                        reset_required = stream.read(len(state.prefix)) != state.prefix
                    if reset_required:
                        state.reset(identity)

                    if not state.prefix and metadata.st_size:
                        stream.seek(0)
                        state.prefix = stream.read(min(metadata.st_size, 4096))

                    stream.seek(state.offset)
                    while True:
                        line_start = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            # The proxy appends one JSON record and then its
                            # newline. Leave an in-flight record for the next
                            # poll rather than permanently skipping it.
                            stream.seek(line_start)
                            break
                        state.offset = stream.tell()
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            trace = json.loads(line)
                        except json.JSONDecodeError:
                            self.logger.debug(
                                "Skipping malformed trace line in %s",
                                traces_path,
                            )
                            continue
                        if (
                            not isinstance(trace, dict)
                            or trace.get("event") != "success"
                        ):
                            continue
                        response = trace.get("response")
                        if not isinstance(response, dict):
                            response = {"model": trace.get("model")}
                        identity = record_identity(trace)
                        if identity and identity in state.seen:
                            continue
                        if identity:
                            state.seen.add(identity)
                        cost = cost_from_responses([response])
                        state.cost += cost
                        state.calls += 1
                        scope_id = record_scope(trace)
                        if scope_id:
                            previous_cost, previous_calls = state.scopes.get(scope_id, (0.0, 0))
                            state.scopes[scope_id] = (previous_cost + cost, previous_calls + 1)

                total_cost = state.cost
                call_count = state.calls
                if self.accounting_scope_id:
                    total_cost, call_count = state.scopes.get(self.accounting_scope_id, (0.0, 0))

        with self._cost_lock:
            self.cost = total_cost
            self.n_calls = call_count
        return total_cost

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    # ------------- convenience -------------
    @property
    def proxy_url(self) -> str | None:
        """Returns http://HOST:PORT/v1 if serve() has been called."""
        return self._proxy_url

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_wait_s
        check_host = "127.0.0.1" if self._host == "0.0.0.0" else self._host

        while time.monotonic() < deadline:
            if self._server_proc and self._server_proc.poll() is not None:
                raise RuntimeError(
                    "LiteLLM proxy exited during startup.\n"
                    f"{self._read_proxy_log_tail()}"
                )
            try:
                with socket.create_connection((check_host, self._port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)

        raise RuntimeError(
            f"LiteLLM proxy did not accept connections on {check_host}:{self._port} "
            f"within {self._startup_wait_s:.1f}s.\n{self._read_proxy_log_tail()}"
        )

    def _read_proxy_log_tail(self, max_lines: int = 40) -> str:
        log_path = self._log_dir / f"proxy_{self._port}.log"
        if not log_path.exists():
            return f"Proxy log not found: {log_path}"

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            return f"Could not read proxy log {log_path}: {exc}"

        tail = "".join(lines[-max_lines:]).strip()
        if not tail:
            return f"Proxy log is empty: {log_path}"
        return f"Proxy log tail ({log_path}):\n{tail}"

    def _start_cost_watcher(self) -> None:
        if not self._auto_cost_updates:
            return
        if self._cost_watcher_thread and self._cost_watcher_thread.is_alive():
            return
        self._cost_watcher_stop_event.clear()
        self._cost_watcher_thread = threading.Thread(
            target=self._cost_watcher_loop,
            name="litellm-cost-watcher",
            daemon=True,
        )
        self._cost_watcher_thread.start()

    def _stop_cost_watcher(self) -> None:
        if not self._cost_watcher_thread:
            return
        self._cost_watcher_stop_event.set()
        self._cost_watcher_thread.join(timeout=2)
        self._cost_watcher_thread = None

    def _cost_watcher_loop(self) -> None:
        traces_path = self.get_cost_traces_path()
        last_size = 0
        while not self._cost_watcher_stop_event.is_set():
            try:
                if traces_path.exists():
                    current_size = traces_path.stat().st_size
                    if current_size < last_size:
                        last_size = 0
                    if current_size > last_size:
                        last_size = current_size
                        self.get_cost()
                else:
                    last_size = 0
            except OSError:
                pass
            self._cost_watcher_stop_event.wait(self._cost_poll_interval_s)

    def __del__(self) -> None:
        self.stop()
        
