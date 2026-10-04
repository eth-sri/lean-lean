"""Minimal Codex App Server client for subscription-backed benchmark runs.

The client deliberately retains no raw App Server transcript.  It exposes only
aggregate diff updates, minimal completed filesystem-action payloads,
cumulative token usage, and terminal state to its caller; every other
notification is counted and discarded after processing.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from configs.generator_constants import CODEX_CLI_VERSION


DiffCallback = Callable[[str, str, str], None]
UsageCallback = Callable[[dict[str, Any], str, str], None]
ActionCallback = Callable[[dict[str, Any], str, str], None]


@dataclass
class CodexAppServerResult:
    returncode: int
    timed_out: bool
    terminal_status: str
    terminal_error: str
    thread_id: str
    turn_id: str
    latest_diff: str
    latest_token_usage: dict[str, Any] | None
    diff_update_count: int
    notification_count: int
    discarded_notification_count: int
    method_counts: dict[str, int] = field(default_factory=dict)
    item_type_counts: dict[str, int] = field(default_factory=dict)
    mutation_action_count: int = 0
    stderr_tail: str = ""


class CodexAppServerClient:
    """Run one ephemeral Codex turn through App Server JSONL over Docker exec."""

    def __init__(
        self,
        *,
        env: Any,
        model: str,
        reasoning_effort: str,
        timeout_seconds: int,
        codex_home: str = "/tmp/leanlean-codex-home",
        cli_version: str = CODEX_CLI_VERSION,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.env = env
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.codex_home = codex_home
        self.cli_version = cli_version
        self._popen_factory = popen_factory
        self._messages: queue.Queue[dict[str, Any] | BaseException | None] = (
            queue.Queue()
        )
        self._item_type_counts: Counter[str] = Counter()
        self._stderr: deque[str] = deque(maxlen=200)
        self._method_counts: Counter[str] = Counter()
        self._notification_count = 0
        self._discarded_notification_count = 0
        self._diff_update_count = 0
        self._mutation_action_count = 0
        self._latest_diff = ""
        self._latest_usage: dict[str, Any] | None = None
        self._thread_id = ""
        self._turn_id = ""
        self._terminal_status = ""
        self._terminal_error = ""
        self._on_diff: DiffCallback | None = None
        self._on_usage: UsageCallback | None = None
        self._on_action: ActionCallback | None = None
        self._process: Any = None

    def _command(self) -> list[str]:
        container_id = getattr(self.env, "container_id", None)
        config = getattr(self.env, "config", None)
        if not container_id or config is None:
            raise RuntimeError("Codex App Server requires a Docker environment")
        executable = getattr(config, "executable", "docker")
        cwd = "/testbed"
        cmd = [executable, "exec", "-i", "-w", cwd]
        for key in getattr(config, "forward_env", []) or []:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in (getattr(config, "env", {}) or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        command = (
            ". $HOME/.nvm/nvm.sh && "
            f"CODEX_HOME={self.codex_home} "
            "exec codex app-server --listen stdio://"
        )
        prepare = getattr(self.env, "_prepare_shell_command", None)
        if callable(prepare):
            command = prepare(command)
        cmd.extend([container_id, "bash", "-lc", command])
        return cmd

    def _read_stdout(self) -> None:
        try:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    message = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    self._messages.put(
                        RuntimeError(f"Malformed App Server JSONL: {exc}")
                    )
                    continue
                if isinstance(message, dict):
                    self._messages.put(message)
        except BaseException as exc:
            self._messages.put(exc)
        finally:
            self._messages.put(None)

    def _read_stderr(self) -> None:
        try:
            assert self._process.stderr is not None
            for line in self._process.stderr:
                self._stderr.append(line)
        except BaseException as exc:
            self._stderr.append(f"stderr reader failed: {exc}\n")

    def _send(self, message: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Codex App Server stdin is unavailable")
        self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _next(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("codex app-server", self.timeout_seconds)
        try:
            item = self._messages.get(timeout=remaining)
        except queue.Empty as exc:
            raise subprocess.TimeoutExpired(
                "codex app-server", self.timeout_seconds
            ) from exc
        if item is None:
            stderr = "".join(self._stderr)[-4000:]
            raise RuntimeError(
                "Codex App Server closed before turn completion"
                + (f": {stderr}" if stderr else "")
            )
        if isinstance(item, BaseException):
            raise item
        return item

    @staticmethod
    def _response_error(message: dict[str, Any]) -> str:
        error = message.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or json.dumps(error))
        return str(error or "unknown App Server error")

    def _reply_to_server_request(self, message: dict[str, Any]) -> bool:
        if "id" not in message or "method" not in message:
            return False
        method = str(message.get("method") or "")
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "execCommandApproval",
            "applyPatchApproval",
        }:
            self._send({"id": message["id"], "result": {"decision": "accept"}})
            return True
        self._send(
            {
                "id": message["id"],
                "error": {
                    "code": -32601,
                    "message": f"Unsupported App Server request: {method}",
                },
            }
        )
        return True

    @staticmethod
    def _minimal_mutation_action(item: dict[str, Any]) -> dict[str, Any] | None:
        """Keep replay inputs while dropping command output and conversation data."""
        item_type = item.get("type")
        common = {
            "id": str(item.get("id") or ""),
            "type": item_type,
            "status": str(item.get("status") or ""),
        }
        if item_type == "commandExecution":
            command = item.get("command")
            if not isinstance(command, str) or not command:
                return None
            return {
                **common,
                "command": command,
                "cwd": str(item.get("cwd") or "/testbed"),
                "exitCode": item.get("exitCode"),
                "durationMs": item.get("durationMs"),
            }
        if item_type == "fileChange":
            changes = []
            for change in item.get("changes") or []:
                if not isinstance(change, dict):
                    continue
                diff = change.get("diff")
                if not isinstance(diff, str):
                    continue
                kind = change.get("kind")
                if isinstance(kind, dict):
                    kind = {
                        "type": str(kind.get("type") or ""),
                        "movePath": kind.get("movePath") or kind.get("move_path"),
                    }
                else:
                    kind = str(kind or "")
                changes.append(
                    {
                        "path": str(change.get("path") or ""),
                        "kind": kind,
                        "diff": diff,
                    }
                )
            if not changes:
                return None
            return {**common, "changes": changes}
        return None

    def _handle(self, message: dict[str, Any]) -> None:
        if self._reply_to_server_request(message):
            return
        method = message.get("method")
        if not isinstance(method, str):
            return
        self._notification_count += 1
        self._method_counts[method] += 1
        params = message.get("params")
        if not isinstance(params, dict):
            self._discarded_notification_count += 1
            return

        if method == "turn/diff/updated":
            diff = params.get("diff")
            thread_id = str(params.get("threadId") or "")
            turn_id = str(params.get("turnId") or "")
            if isinstance(diff, str):
                self._latest_diff = diff
                self._diff_update_count += 1
                if self._on_diff is not None:
                    self._on_diff(diff, thread_id, turn_id)
            return

        if method == "thread/tokenUsage/updated":
            usage = params.get("tokenUsage")
            thread_id = str(params.get("threadId") or "")
            turn_id = str(params.get("turnId") or "")
            if isinstance(usage, dict):
                self._latest_usage = usage
                if self._on_usage is not None:
                    self._on_usage(usage, thread_id, turn_id)
            return

        if method == "turn/completed":
            turn = params.get("turn")
            if isinstance(turn, dict):
                self._turn_id = str(turn.get("id") or self._turn_id)
                self._terminal_status = str(turn.get("status") or "")
                error = turn.get("error")
                if error:
                    self._terminal_error = (
                        error if isinstance(error, str) else json.dumps(error)
                    )
            return

        if method == "error":
            error = params.get("error") or params.get("message") or params
            self._terminal_error = (
                error if isinstance(error, str) else json.dumps(error)
            )
        if method == "item/completed":
            item = params.get("item")
            if isinstance(item, dict):
                item_type = item.get("type")
                if isinstance(item_type, str) and item_type:
                    self._item_type_counts[item_type] += 1
                action = self._minimal_mutation_action(item)
                if action is not None:
                    self._mutation_action_count += 1
                    if self._on_action is not None:
                        self._on_action(
                            action,
                            str(params.get("threadId") or ""),
                            str(params.get("turnId") or ""),
                        )
                    return
        self._discarded_notification_count += 1

    def _wait_response(self, request_id: int, deadline: float) -> dict[str, Any]:
        while True:
            message = self._next(deadline)
            if message.get("id") == request_id:
                if message.get("error") is not None:
                    raise RuntimeError(self._response_error(message))
                result = message.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"App Server response {request_id} has no object result"
                    )
                return result
            self._handle(message)

    def _shutdown(self) -> int:
        process = self._process
        if process is None:
            return 1
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass
        try:
            return int(process.wait(timeout=5))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                return int(process.wait(timeout=5))
            except subprocess.TimeoutExpired:
                process.kill()
                return int(process.wait(timeout=5))

    def run(
        self,
        prompt: str,
        *,
        on_diff: DiffCallback | None = None,
        on_usage: UsageCallback | None = None,
        on_action: ActionCallback | None = None,
    ) -> CodexAppServerResult:
        self._on_diff = on_diff
        self._on_usage = on_usage
        self._on_action = on_action
        self._process = self._popen_factory(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        deadline = time.monotonic() + self.timeout_seconds
        timed_out = False
        returncode = 1
        try:
            self._send(
                {
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {
                            "name": "leanlean",
                            "version": "1",
                        },
                        "capabilities": {"experimentalApi": True},
                    },
                }
            )
            self._wait_response(0, deadline)
            self._send({"method": "initialized"})
            self._send(
                {
                    "id": 1,
                    "method": "thread/start",
                    "params": {
                        "model": self.model,
                        "cwd": "/testbed",
                        "approvalPolicy": "never",
                        "sandbox": "workspace-write",
                        "ephemeral": True,
                    },
                }
            )
            started = self._wait_response(1, deadline)
            thread = started.get("thread")
            if not isinstance(thread, dict) or not thread.get("id"):
                raise RuntimeError("thread/start returned no thread id")
            self._thread_id = str(thread["id"])
            self._send(
                {
                    "id": 2,
                    "method": "turn/start",
                    "params": {
                        "threadId": self._thread_id,
                        "input": [{"type": "text", "text": prompt}],
                        "cwd": "/testbed",
                        "model": self.model,
                        "effort": self.reasoning_effort,
                        "approvalPolicy": "never",
                        "sandboxPolicy": {
                            "type": "workspaceWrite",
                            "writableRoots": ["/testbed"],
                            "networkAccess": False,
                        },
                    },
                }
            )
            turn_started = self._wait_response(2, deadline)
            turn = turn_started.get("turn")
            if isinstance(turn, dict):
                self._turn_id = str(turn.get("id") or "")
            while not self._terminal_status:
                self._handle(self._next(deadline))
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminal_status = self._terminal_status or "interrupted"
            self._terminal_error = self._terminal_error or "App Server turn timed out"
        except BaseException as exc:
            self._terminal_status = self._terminal_status or "failed"
            self._terminal_error = self._terminal_error or str(exc)
        finally:
            returncode = self._shutdown()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

        if self._terminal_status == "completed":
            returncode = 0
        elif returncode == 0:
            returncode = 1
        return CodexAppServerResult(
            returncode=returncode,
            timed_out=timed_out,
            terminal_status=self._terminal_status,
            terminal_error=self._terminal_error,
            thread_id=self._thread_id,
            turn_id=self._turn_id,
            latest_diff=self._latest_diff,
            latest_token_usage=self._latest_usage,
            diff_update_count=self._diff_update_count,
            notification_count=self._notification_count,
            discarded_notification_count=self._discarded_notification_count,
            method_counts=dict(sorted(self._method_counts.items())),
            item_type_counts=dict(sorted(self._item_type_counts.items())),
            mutation_action_count=self._mutation_action_count,
            stderr_tail="".join(self._stderr)[-4000:],
        )
