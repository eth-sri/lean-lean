"""Subscription-backed Codex generator using the App Server protocol."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from configs.generator_constants import CODEX_CLI_VERSION
from leanlean.app_server_playback import AppServerDiffPlaybackRecorder
from leanlean.codex_app_server import CodexAppServerClient
from leanlean.generators.cli_agent import CLIAgent
from leanlean.model.costs import cost_from_responses


class CodexAppServerAgent(CLIAgent):
    """Run an ephemeral Codex turn and retain only diff/usage/terminal data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("leanlean.codex_app_server_agent")

    def _post_exec(self) -> None:
        self._cleanup_codex_subscription_auth()
        for cmd in self.config.post_exec_commands:
            self.env.execute(cmd, timeout=False)
        self.model.delete_traces()

    @staticmethod
    def _response_from_app_server_usage(model_name: str, usage: dict[str, Any]) -> dict:
        total = usage.get("total")
        if not isinstance(total, dict):
            total = {}

        def _integer(key: str) -> int:
            value = total.get(key)
            return int(value) if isinstance(value, (int, float)) else 0

        return CLIAgent._codex_response_from_usage(
            model_name,
            {
                "input_tokens": _integer("inputTokens"),
                "cached_input_tokens": _integer("cachedInputTokens"),
                "output_tokens": _integer("outputTokens"),
                "reasoning_output_tokens": _integer("reasoningOutputTokens"),
            },
        )

    def _make_recorder(self) -> AppServerDiffPlaybackRecorder | None:
        playback_config = getattr(self, "playback_config", {}) or {}
        if not playback_config.get("enabled", False):
            return None
        mode = str(playback_config.get("mode", "external_replay")).strip()
        if mode != "external_replay":
            raise ValueError(
                "codex_app_server_sub supports only playback.mode=external_replay"
            )
        snapshot_root = getattr(self, "_playback_output_dir", None)
        if snapshot_root is None:
            raise RuntimeError("App Server playback requires an output directory")
        capture_index = getattr(self, "_playback_capture_index", 0) + 1
        self._playback_capture_index = capture_index
        capture_name = f"capture_{capture_index:03d}"
        capture_dir = Path(snapshot_root) / capture_name
        recorder = AppServerDiffPlaybackRecorder(
            env=self.env,
            provider="codex_app_server_sub",
            include_prefix=getattr(self, "_target_dir", "") or "",
            target_file=getattr(self, "_target_file", "") or "",
            exclude_dirs=tuple(getattr(self, "_playback_exclude_dirs", ()) or ()),
            every_n_edits=int(playback_config.get("every_n_edits", 1)),
            snapshot_dir=capture_dir,
            snapshot_reference_prefix=f"playback/{capture_name}",
            instance_id=str(getattr(self, "_playback_instance_id", "") or ""),
            manifest_path=capture_dir / "playback.json",
            build_command=(
                getattr(self, "_playback_build_command", "") or ""
                if playback_config.get("build_each_checkpoint", False)
                else ""
            ),
            build_timeout_seconds=int(
                playback_config.get("build_timeout_seconds", 3600)
            ),
            build_output_chars=int(playback_config.get("build_output_chars", 4000)),
            capture_timeout_seconds=int(
                playback_config.get("capture_timeout_seconds", 600)
            ),
            capture_mutation_actions=True,
        )
        recorder.prepare()
        return recorder

    def run(self, task: str, **kwargs: Any) -> tuple[str, str]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.playback = {}
        self.native_rollout = {}
        prompt = self.render_template(self.config.instance_template)
        self._install()

        openai_args = self.model.get_openai_args()
        model_name = str(openai_args.get("model") or self.model.config.model_name)
        model_name = model_name.split("/")[-1]
        model_kwargs = getattr(self.model.config, "model_kwargs", {}) or {}
        reasoning_effort = str(model_kwargs.get("reasoning_effort", "medium"))
        response_start = len(getattr(self.model, "responses", []))
        recorder: AppServerDiffPlaybackRecorder | None = None
        latest_response: dict[str, Any] | None = None
        app_result = None
        try:
            recorder = self._make_recorder()
            self._install_codex_subscription_auth()

            def on_diff(diff: str, thread_id: str, turn_id: str) -> None:
                if recorder is not None:
                    recorder.ingest_diff(diff, thread_id, turn_id)

            def on_action(action: dict[str, Any], thread_id: str, turn_id: str) -> None:
                if recorder is not None:
                    recorder.ingest_action(action, thread_id, turn_id)

            def on_usage(
                usage: dict[str, Any], _thread_id: str, _turn_id: str
            ) -> None:
                nonlocal latest_response
                latest_response = self._response_from_app_server_usage(
                    model_name, usage
                )
                if recorder is not None:
                    recorder.update_usage(latest_response)
                self.model.cost = cost_from_responses(
                    [*self.model.responses, latest_response]
                )

            timeout = int(self.env.config.container_timeout_seconds())
            client = CodexAppServerClient(
                env=self.env,
                model=model_name,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout,
            )
            app_result = client.run(
                prompt,
                on_diff=on_diff,
                on_usage=on_usage,
                on_action=on_action,
            )
            if latest_response is not None:
                self.model.responses.append(latest_response)
            self.model.n_calls = len(self.model.responses)
            self.model.cost_from_responses_fallback = True
            self.model.cost = cost_from_responses(self.model.responses)
            if latest_response is None:
                self.logger.warning(
                    "codex_app_server_sub received no cumulative token usage"
                )
        except Exception as exc:
            self.logger.exception("Codex App Server run failed")
            if app_result is None:
                from leanlean.codex_app_server import CodexAppServerResult

                app_result = CodexAppServerResult(
                    returncode=1,
                    timed_out=False,
                    terminal_status="failed",
                    terminal_error=str(exc),
                    thread_id="",
                    turn_id="",
                    latest_diff="",
                    latest_token_usage=None,
                    diff_update_count=0,
                    notification_count=0,
                    discarded_notification_count=0,
                )
        finally:
            if recorder is not None:
                try:
                    responses = list(self.model.responses)[response_start:]
                    self.playback = recorder.finish(responses)
                except Exception:
                    self.logger.exception("Could not finalize App Server playback")
            self._post_exec()

        diff = self._extract_final_diff()
        if recorder is not None:
            self.playback = recorder.replay_submission(diff)
            if not self.playback.get("final_snapshot_matches_submission"):
                self.logger.error(
                    "App Server playback did not exactly reconstruct both the "
                    "terminal source tree and submitted patch"
                )

        summary = {
            "protocol": "codex-app-server-jsonl-v2",
            "cli_version": CODEX_CLI_VERSION,
            "terminal_status": app_result.terminal_status,
            "terminal_error": app_result.terminal_error,
            "thread_id": app_result.thread_id,
            "turn_id": app_result.turn_id,
            "diff_update_count": app_result.diff_update_count,
            "notification_count": app_result.notification_count,
            "discarded_notification_count": app_result.discarded_notification_count,
            "method_counts": app_result.method_counts,
            "item_type_counts": app_result.item_type_counts,
            "mutation_action_count": app_result.mutation_action_count,
            "raw_transcript_retained": False,
        }
        if self.playback:
            self.playback["app_server_transport"] = summary
            if recorder is not None:
                self.playback = recorder.attach_transport_summary(summary)
        self.messages.append(
            {
                "role": "system",
                "type": "codex_app_server_summary",
                "content": (
                    f"terminal_status={app_result.terminal_status}; "
                    f"diff_updates={app_result.diff_update_count}; "
                    "raw_transcript_retained=false"
                ),
                "app_server": summary,
            }
        )
        self.add_message("user", diff)

        if app_result.returncode != 0 and not diff.strip():
            message = app_result.terminal_error or app_result.stderr_tail
            if not message:
                message = "Codex App Server execution failed"
            return "ExecutionFailed", message
        return "Submitted", diff
