"""Post-run replay of Codex App Server aggregate diff checkpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from configs.generator_constants import CODEX_CLI_VERSION

from leanlean.model.costs import cost_from_responses
from leanlean.playback import (
    ExternalPlaybackRecorder,
    _safe_float,
    _split_patch_by_file,
)


@dataclass
class AppServerDiffPlaybackRecorder(ExternalPlaybackRecorder):
    """Replay cumulative turn diffs without live checkpoint commands.

    The agent container is read only before and after the turn. Intermediate
    states come from App Server's aggregate diffs and are measured and built in
    a disposable replay container after the monitored turn has ended.
    """

    app_server_cli_version: str = CODEX_CLI_VERSION
    capture_mutation_actions: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self._latest_diff = ""
        self._latest_diff_sha256 = hashlib.sha256(b"").hexdigest()
        self._last_recorded_diff_sha256 = self._latest_diff_sha256
        self._final_archive_metadata: dict[str, Any] = {}
        self._mutation_actions: list[dict[str, Any]] = []
        self._action_sequence = 0

    def _diff_reference(self, filename: str) -> str:
        if self.snapshot_reference_prefix:
            return f"{self.snapshot_reference_prefix.rstrip('/')}/{filename}"
        return filename

    def _diff_file(self, point: dict[str, Any]) -> Path:
        if self.snapshot_dir is None:
            raise RuntimeError("App Server playback requires an artifact directory")
        return Path(self.snapshot_dir) / Path(str(point["diff_path"])).name

    def _action_reference(self, filename: str) -> str:
        if self.snapshot_reference_prefix:
            return f"{self.snapshot_reference_prefix.rstrip('/')}/{filename}"
        return filename

    def _action_file(self, record: dict[str, Any]) -> Path:
        if self.snapshot_dir is None:
            raise RuntimeError("App Server playback requires an artifact directory")
        return Path(self.snapshot_dir) / Path(str(record["action_path"])).name

    def _read_action(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = gzip.decompress(self._action_file(record).read_bytes())
        action = json.loads(payload.decode("utf-8"))
        if not isinstance(action, dict):
            raise RuntimeError("Malformed App Server mutation action")
        return action

    def update_usage(self, response: dict[str, Any]) -> None:
        """Replace the cumulative usage sample used for checkpoint cost."""
        self._turn_responses = [response]

    def ingest_action(
        self,
        action: dict[str, Any],
        thread_id: str = "",
        turn_id: str = "",
    ) -> None:
        """Durably retain one minimal completed action for post-run replay."""
        if not self.capture_mutation_actions:
            return
        if not self.points:
            self.prepare()
        self._action_sequence += 1
        self.event_count += 1
        payload = json.dumps(
            action, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        filename = f"action_{self._action_sequence:05d}.json.gz"
        destination = Path(self.snapshot_dir) / filename
        with destination.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
                out.write(payload)
        files = []
        if action.get("type") == "fileChange":
            files = sorted(
                {
                    str(change.get("path") or "")
                    for change in action.get("changes") or []
                    if isinstance(change, dict) and change.get("path")
                }
            )
        self._mutation_actions.append(
            {
                "action_index": self._action_sequence,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "cost_usd": round(cost_from_responses(self._turn_responses), 8),
                "type": str(action.get("type") or ""),
                "status": str(action.get("status") or ""),
                "files": files,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "action_path": self._action_reference(filename),
                "action_sha256": hashlib.sha256(payload).hexdigest(),
                "action_bytes": len(payload),
                "replay_status": "pending",
            }
        )
        self._persist_manifest(
            self._playback_result(
                cost_basis="cumulative App Server token usage; final cost pending"
            )
        )

    def ingest_diff(
        self,
        diff: str,
        thread_id: str = "",
        turn_id: str = "",
        *,
        force: bool = False,
    ) -> None:
        if not self.points:
            self.prepare()
        payload = diff.encode("utf-8", errors="surrogateescape")
        digest = hashlib.sha256(payload).hexdigest()
        if not self.capture_mutation_actions:
            self.event_count += 1
        if digest == self._latest_diff_sha256 and not force:
            return
        if digest != self._latest_diff_sha256:
            self.edit_count += 1
        self._latest_diff = diff
        self._latest_diff_sha256 = digest
        if not force and self.edit_count % max(1, self.every_n_edits):
            return
        if digest == self._last_recorded_diff_sha256:
            return

        filename = f"checkpoint_{self.edit_count:05d}.diff.gz"
        destination = Path(self.snapshot_dir) / filename
        with destination.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
                out.write(payload)
        self._last_recorded_diff_sha256 = digest
        reference = self._diff_reference(filename)
        self.points.append(
            {
                "edit_index": self.edit_count,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "cost_usd": round(cost_from_responses(self._turn_responses), 8),
                "lean_tokens": None,
                "lean_tokens_saved": None,
                "files": sorted(_split_patch_by_file(diff)),
                "kind": "app_server_aggregate_diff",
                "exact": True,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "diff_path": reference,
                "snapshot_path": reference,
                "snapshot_format": "cumulative-app-server-unified-diff-gzip-v1",
                "snapshot_sha256": digest,
                "snapshot_bytes": len(payload),
                "replay_status": "pending",
            }
        )
        self._persist_manifest(
            self._playback_result(
                cost_basis="cumulative App Server token usage; final cost pending"
            )
        )

    def _capture_final_archive(self) -> None:
        candidate, metadata, capture_seconds = self._capture_archive_candidate()
        destination = Path(self.snapshot_dir) / "final.sources.tar.gz"
        candidate.replace(destination)
        self._final_archive_metadata = {
            **metadata,
            "snapshot_path": self._archive_reference(destination.name),
            "snapshot_format": "deterministic-source-tar-gzip-v1",
            "snapshot_digest_basis": "canonical uncompressed tar stream",
            "capture_seconds": round(capture_seconds, 3),
        }

    def finish(self, authoritative_responses: list[dict[str, Any]]) -> dict[str, Any]:
        if self._latest_diff_sha256 != self._last_recorded_diff_sha256:
            self.ingest_diff(self._latest_diff, force=True)
        self._capture_final_archive()
        final_cost = cost_from_responses(authoritative_responses)
        timed_records = [*self.points, *self._mutation_actions]
        observed_cost = max(
            (_safe_float(item.get("cost_usd")) for item in timed_records),
            default=0.0,
        )
        cost_basis = "recorded provider usage; terminal total separate; no proration"
        for item in timed_records:
            item["cost_boundary_exact"] = False
            item["cost_join_basis"] = "recorded_partial_usage"
            if observed_cost <= 0 and item.get("kind") != "baseline":
                item["cost_usd"] = None
                item["cost_join_basis"] = "missing_recorded_usage"
        result = self._playback_result(
            cost_basis=cost_basis,
            final_cost_usd=final_cost,
        )
        self._result = result
        self._persist_manifest(result)
        return result

    def _playback_result(
        self, *, cost_basis: str, final_cost_usd: float | None = None
    ) -> dict[str, Any]:
        result = super()._playback_result(
            cost_basis=cost_basis,
            final_cost_usd=final_cost_usd,
        )
        result.update(
            {
                "quality": self._external_quality,
                "capture_mode": (
                    "app_server_mutation_action_replay"
                    if self.capture_mutation_actions
                    else "app_server_aggregate_diff_replay"
                ),
                "metric_basis": (
                    "Lean tokens measured during post-run action replay"
                    if self._external_replayed and self.capture_mutation_actions
                    else "Lean tokens measured during post-run cumulative-diff replay"
                    if self._external_replayed
                    else "pending post-run action/diff replay"
                ),
                "snapshot_basis": (
                    "one pre-turn source archive, minimal App Server mutation "
                    "actions and aggregate diffs, "
                    "and one post-turn verification archive"
                ),
                "build_basis": (
                    "post-run disposable container from the pinned baseline image"
                ),
                "app_server_cli_version": self.app_server_cli_version,
                "retained_app_server_payload": [
                    "turn/diff/updated.diff",
                    "thread/tokenUsage/updated.tokenUsage",
                    "turn/completed terminal state",
                    "item/completed commandExecution command/cwd/status only",
                    "item/completed fileChange changes only",
                ],
                "agent_container_source_reads": "pre_turn_and_post_turn_only",
                "agent_container_source_writes": False,
                "agent_container_git_writes": False,
                "agent_container_builds": False,
                "agent_paused_for_checkpoints": False,
                "replay_cache_retained": False,
                "replay_scope": {
                    "include_prefix": self.include_prefix,
                    "target_file": self.target_file,
                    "exclude_dirs": list(self.exclude_dirs),
                },
                "mutation_action_timeline": {
                    "enabled": self.capture_mutation_actions,
                    "format": "minimal-app-server-mutation-action-gzip-v1",
                    "count": len(self._mutation_actions),
                    "actions": self._mutation_actions,
                },
            }
        )
        if self._final_archive_metadata:
            result["final_source_archive"] = dict(self._final_archive_metadata)
        return result

    def attach_transport_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Persist compact transport counters after replay has completed."""
        self._result["app_server_transport"] = summary
        self._persist_manifest(self._result)
        return self._result

    def _apply_include_args(self) -> str:
        if self.target_file:
            patterns = [self.target_file]
        elif self.include_prefix:
            prefix = self.include_prefix.rstrip("/")
            patterns = [prefix + "/**", prefix + ".lean"]
        else:
            patterns = []
        return " ".join(
            "--include=" + shlex.quote(pattern) for pattern in patterns
        )

    def _restore_and_apply(self, replay_env: Any, diff: str, label: str) -> None:
        replay_env.restore_source_archive(
            self._archive_file(self.points[0]),
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        if not diff:
            return
        patch_path = f"/tmp/leanlean-app-server-{id(self)}-{label}.patch"
        replay_env.write_file(patch_path, diff)
        include_args = self._apply_include_args()
        applied = replay_env.execute(
            "cd /testbed && "
            "git apply --binary --whitespace=nowarn "
            f"{include_args} {shlex.quote(patch_path)}",
            timeout=False,
        )
        if applied.get("returncode"):
            raise RuntimeError(
                f"Could not apply {label} cumulative diff: "
                + applied.get("output", "")[-self.build_output_chars :]
            )

    def _export_replay_digest(self, replay_env: Any, label: str) -> str:
        candidate = Path(self.snapshot_dir) / f".{label}-{id(self)}.tar.gz"
        try:
            metadata = replay_env.export_source_archive(
                candidate,
                self._archive_paths(),
                timeout=self.capture_timeout_seconds,
            )
            return str(metadata["source_archive_sha256"])
        finally:
            candidate.unlink(missing_ok=True)

    def _verify_diff_and_submission(
        self, replay_env: Any, submission: str, *, action_digest: str = ""
    ) -> dict[str, Any]:
        captured = str(self._final_archive_metadata.get("source_archive_sha256") or "")
        action_matches = bool(action_digest) and action_digest == captured
        self._restore_and_apply(replay_env, self._latest_diff, "endpoint")
        app_server_digest = self._export_replay_digest(replay_env, "endpoint-check")
        self._restore_and_apply(replay_env, submission, "submission")
        submitted_digest = self._export_replay_digest(replay_env, "submission-check")
        diff_matches = app_server_digest == captured
        submission_matches = submitted_digest == captured
        matches = (action_matches if action_digest else diff_matches) and submission_matches
        return {
            "checked": True,
            "matches": matches,
            "basis": "exact scoped source archive",
            "action_replay_matches_final": action_matches if action_digest else None,
            "app_server_diff_matches_final": diff_matches,
            "submitted_patch_matches_final": submission_matches,
            "captured_source_archive_sha256": captured,
            "action_replay_source_archive_sha256": action_digest or None,
            "app_server_replay_source_archive_sha256": app_server_digest,
            "submitted_source_archive_sha256": submitted_digest,
            "app_server_diff_sha256": self._latest_diff_sha256,
            "submitted_patch_sha256": hashlib.sha256(
                submission.encode("utf-8", errors="surrogateescape")
            ).hexdigest(),
        }

    @staticmethod
    def _repo_relative_action_path(raw_path: Any) -> str:
        path = PurePosixPath(str(raw_path or ""))
        if path.is_absolute():
            try:
                path = path.relative_to("/testbed")
            except ValueError as exc:
                raise RuntimeError(
                    f"fileChange path is outside /testbed: {raw_path}"
                ) from exc
        if not path.parts or ".." in path.parts:
            raise RuntimeError(f"unsafe fileChange path: {raw_path}")
        return path.as_posix()

    @classmethod
    def _standalone_file_change_diff(cls, change: dict[str, Any]) -> str:
        """Add Git headers to App Server's per-item headerless hunk."""
        diff = str(change.get("diff") or "").rstrip("\n")
        if not diff:
            return ""
        if diff.startswith("diff --git ") or diff.startswith("--- "):
            return diff + "\n"
        path = cls._repo_relative_action_path(change.get("path"))
        kind = change.get("kind")
        if isinstance(kind, dict):
            kind_type = str(kind.get("type") or "").lower()
        else:
            kind_type = str(kind or "").lower()
        if "add" in kind_type or "create" in kind_type:
            before = "/dev/null"
            after = f"b/{path}"
        elif "delete" in kind_type or "remove" in kind_type:
            before = f"a/{path}"
            after = "/dev/null"
        else:
            before = f"a/{path}"
            after = f"b/{path}"
        return (
            f"diff --git a/{path} b/{path}\n"
            f"--- {before}\n"
            f"+++ {after}\n"
            f"{diff}\n"
        )

    def _apply_mutation_action(
        self, replay_env: Any, action: dict[str, Any], label: str
    ) -> dict[str, Any]:
        action_type = action.get("type")
        if action_type == "commandExecution":
            command = action.get("command")
            if not isinstance(command, str) or not command:
                raise RuntimeError(f"{label} has no replayable command")
            cwd = str(action.get("cwd") or "/testbed")
            if cwd != "/testbed" and not cwd.startswith("/testbed/"):
                raise RuntimeError(f"{label} has unsafe replay cwd: {cwd}")
            result = replay_env.execute(
                f"cd {shlex.quote(cwd)} && {command}",
                timeout=self.capture_timeout_seconds,
            )
            return {
                "replay_returncode": result.get("returncode"),
                "observed_returncode": action.get("exitCode"),
            }
        if action_type == "fileChange":
            if action.get("status") != "completed":
                return {"skipped_status": action.get("status")}
            diff = "".join(
                self._standalone_file_change_diff(change)
                for change in action.get("changes") or []
                if isinstance(change, dict) and change.get("diff")
            )
            if not diff:
                return {"empty_file_change": True}
            patch_path = f"/tmp/leanlean-action-{id(self)}-{label}.patch"
            replay_env.write_file(patch_path, diff)
            applied = replay_env.execute(
                "cd /testbed && git apply --binary --whitespace=nowarn "
                f"{self._apply_include_args()} {shlex.quote(patch_path)}",
                timeout=False,
            )
            if applied.get("returncode"):
                raise RuntimeError(
                    f"Could not apply {label} fileChange: "
                    + applied.get("output", "")[-self.build_output_chars :]
                )
            return {"replay_returncode": 0}
        raise RuntimeError(f"Unsupported mutation action type: {action_type}")

    def _replay_mutation_actions(
        self, replay_env: Any, submission: str
    ) -> dict[str, Any]:
        baseline = dict(self.points[0])
        replay_env.restore_source_archive(
            self._archive_file(baseline),
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        baseline_digest = str(baseline.get("source_archive_sha256") or "")
        baseline["lean_tokens"] = self._read_tokens(env=replay_env)
        baseline["replay_status"] = "complete"
        baseline_build = self._run_replay_build(replay_env)
        if baseline_build is not None:
            baseline["build"] = baseline_build
        replay_points = [baseline]
        previous_digest = baseline_digest
        edit_index = 0
        for record in self._mutation_actions:
            action = self._read_action(record)
            record.update(
                self._apply_mutation_action(
                    replay_env, action, f"action-{record['action_index']}"
                )
            )
            digest = self._export_replay_digest(
                replay_env, f"action-{record['action_index']}-check"
            )
            changed = digest != previous_digest
            record["source_changed"] = changed
            record["replay_status"] = "complete"
            if not changed:
                continue
            edit_index += 1
            previous_digest = digest
            point = {
                "edit_index": edit_index,
                "action_index": record["action_index"],
                "elapsed_seconds": record["elapsed_seconds"],
                "cost_usd": record["cost_usd"],
                "lean_tokens": self._read_tokens(env=replay_env),
                "lean_tokens_saved": None,
                "files": record.get("files") or [],
                "kind": f"{record['type']}_action_replay",
                "exact": False,
                "thread_id": record.get("thread_id") or "",
                "turn_id": record.get("turn_id") or "",
                "action_path": record["action_path"],
                "action_sha256": record["action_sha256"],
                "source_archive_sha256": digest,
                "replay_status": "complete",
            }
            build = self._run_replay_build(replay_env)
            if build is not None:
                point["build"] = build
            replay_points.append(point)

        self.points = replay_points
        self.edit_count = edit_index
        baseline_tokens = int(baseline["lean_tokens"])
        self.baseline_lean_tokens = baseline_tokens
        for point in self.points:
            tokens = int(point["lean_tokens"])
            point["lean_tokens_saved"] = baseline_tokens - tokens
            point["lean_token_compression_pct"] = (
                round(100 * (baseline_tokens - tokens) / baseline_tokens, 6)
                if baseline_tokens
                else 0.0
            )
        if self.points and self._mutation_actions:
            terminal = self._mutation_actions[-1]
            self.points[-1]["cost_usd"] = terminal["cost_usd"]
            self.points[-1]["elapsed_seconds"] = terminal["elapsed_seconds"]

        verification = self._verify_diff_and_submission(
            replay_env, submission, action_digest=previous_digest
        )
        for index, point in enumerate(self.points):
            point["exact"] = index == 0
            point["reconstruction"] = "ordered_completed_action_replay"
        if self.points:
            self.points[-1]["exact"] = verification["matches"]
            self.points[-1]["matches_submission"] = verification["matches"]
        self._external_replayed = True
        self._external_quality = (
            "endpoint_verified_app_server_action_replay"
            if verification["matches"]
            else "invalid_final_tree_reconstruction"
        )
        result = self._playback_result(
            cost_basis=str(
                self._result.get("cost_basis", "cumulative App Server token usage")
            ),
            final_cost_usd=_safe_float(self._result.get("final_cost_usd")),
        )
        result["submission_verification"] = verification
        result["final_snapshot_matches_submission"] = verification["matches"]
        self._result = result
        self._persist_manifest(result)
        return result

    def replay_submission(self, patch: str) -> dict[str, Any]:
        """Build and measure every aggregate diff in a disposable container."""
        spawner = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawner):
            raise RuntimeError("App Server playback requires a replay environment factory")
        lifetime = max(
            600,
            len(self.points)
            * (
                self.capture_timeout_seconds
                + (self.build_timeout_seconds if self.build_command else 0)
                + 60
            ),
        )
        replay_env = None
        try:
            replay_env = spawner(lifetime_seconds=lifetime)
            self._prepared = False
            if self.capture_mutation_actions and self._mutation_actions:
                return self._replay_mutation_actions(replay_env, patch)
            for index, point in enumerate(self.points):
                diff = ""
                if index:
                    diff = gzip.decompress(self._diff_file(point).read_bytes()).decode(
                        "utf-8", errors="surrogateescape"
                    )
                self._restore_and_apply(replay_env, diff, f"checkpoint-{index}")
                point["lean_tokens"] = self._read_tokens(env=replay_env)
                point["replay_status"] = "complete"
                build = self._run_replay_build(replay_env)
                if build is not None:
                    point["build"] = build

            baseline_tokens = int(self.points[0]["lean_tokens"])
            self.baseline_lean_tokens = baseline_tokens
            for point in self.points:
                tokens = int(point["lean_tokens"])
                point["lean_tokens_saved"] = baseline_tokens - tokens
                point["lean_token_compression_pct"] = (
                    round(100 * (baseline_tokens - tokens) / baseline_tokens, 6)
                    if baseline_tokens
                    else 0.0
                )

            verification = self._verify_diff_and_submission(replay_env, patch)
            self._external_replayed = True
            self._external_quality = (
                "exact_app_server_diff_replay"
                if verification["matches"]
                else "invalid_final_tree_reconstruction"
            )
            result = self._playback_result(
                cost_basis=str(
                    self._result.get(
                        "cost_basis", "cumulative App Server token usage"
                    )
                ),
                final_cost_usd=_safe_float(self._result.get("final_cost_usd")),
            )
            result["submission_verification"] = verification
            result["final_snapshot_matches_submission"] = verification["matches"]
            if self.points:
                self.points[-1]["matches_submission"] = verification["matches"]
            self._result = result
            self._persist_manifest(result)
            return result
        except Exception as exc:
            self._external_quality = "app_server_diff_replay_failed"
            result = self._result or self._playback_result(
                cost_basis="App Server replay failed"
            )
            result["quality"] = self._external_quality
            result["replay_error"] = str(exc)
            result["final_snapshot_matches_submission"] = False
            self._result = result
            self._persist_manifest(result)
            return result
        finally:
            if replay_env is not None:
                replay_env.cleanup()
