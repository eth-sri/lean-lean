"""Provider-neutral compression playback capture and historical backfill.

New runs can sample the real repository after every completed action.  Older
trajectories do not necessarily contain edit bodies, so :func:`backfill_instance`
reconstructs an explicitly-estimated curve whose endpoint is pinned to the
authoritative evaluation metrics.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import json
import math
import re
import shlex
import subprocess
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from leanlean.claude_code_trace import ClaudeTraceBuilder
from leanlean.deepseek_harness_trace import DeepSeekHarnessTraceBuilder
from leanlean.kimi_code_trace import KimiTraceBuilder
from leanlean.mistral_vibe_trace import mistral_vibe_action
from leanlean.metrics.tokens import count_lean_tokens_in_source
from leanlean.model.costs import cost_from_responses
from leanlean.antigravity_trace import antigravity_action
from leanlean.antigravity_usage import AntigravityUsageJournal, apply_recorded_costs
from leanlean.standardized_trace import detect_trace_provider
from leanlean.submission_patch import drop_absent_deletions


PLAYBACK_FORMAT = "leanlean-playback-v1"
AGENT_PID_PREFIX = "__LEANLEAN_PLAYBACK_AGENT_PID__="
EDIT_TOOL_NAMES = {
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Write",
    "apply_patch",
    "str_replace",
    "str_replace_editor",
}
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(value)
        else default
    )


def _shell_asynchrony_reason(command: str) -> str | None:
    if re.search(r"(?<![\\&<>|])&(?![&>])", command):
        return "background_operator"
    if re.search(r"(?:^|[\s;|()])coproc(?:\s|$)", command):
        return "coproc"
    if re.search(r"(?:^|[\s;|()])disown(?:\s|$)", command):
        return "disown"
    if re.search(r"(?:^|[\s;|()])setsid\s+(?:-[^\s]*f[^\s]*\s+)", command):
        return "detached_setsid"
    return None


def _normalize_path(path: str) -> str:
    path = path.strip()
    for prefix in ("/testbed/", "/project/testbed/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path.lstrip("/")


def _paths_from_tool(name: str, arguments: Any) -> list[str]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            arguments = {}
    if not isinstance(arguments, dict):
        return []

    paths: list[str] = []
    for key in ("file_path", "path", "notebook_path"):
        value = arguments.get(key)
        if isinstance(value, str):
            paths.append(_normalize_path(value))

    patch = arguments.get("patch") or arguments.get("input")
    if isinstance(patch, str) and name == "apply_patch":
        paths.extend(
            _normalize_path(match)
            for match in re.findall(
                r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", patch, re.MULTILINE
            )
        )
    return sorted(set(path for path in paths if path))


def _archive_source_files(path: Path) -> dict[str, bytes]:
    """Read a deterministic source archive into an exact path-to-bytes map."""

    files: dict[str, bytes] = {}
    try:
        archive = tarfile.open(path, mode="r:gz")
    except tarfile.ReadError:
        return {"__scoped_source__": gzip.decompress(path.read_bytes())}
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            name = member.name.removeprefix("./")
            files[name] = stream.read()
    return files


def _incremental_source_diff(
    before: Mapping[str, bytes],
    after: Mapping[str, bytes],
) -> tuple[bytes, list[str], list[str]]:
    """Return a readable byte-stable delta plus its exact changed path set.

    Full checkpoint archives remain authoritative. Text files receive complete
    unified diffs; binary changes are explicitly named and are reconstructed
    from the following checkpoint archive rather than lossy diff text.
    """

    changed_files: list[str] = []
    binary_files: list[str] = []
    chunks: list[str] = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name)
        new = after.get(name)
        if old == new:
            continue
        changed_files.append(name)
        if (old is not None and b"\0" in old) or (new is not None and b"\0" in new):
            binary_files.append(name)
            chunks.append(f"Binary files a/{name} and b/{name} differ\n")
            continue
        old_lines = (
            old.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
            if old is not None
            else []
        )
        new_lines = (
            new.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
            if new is not None
            else []
        )
        chunks.extend(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{name}" if old is not None else "/dev/null",
                tofile=f"b/{name}" if new is not None else "/dev/null",
                lineterm="\n",
            )
        )
    return (
        "".join(chunks).encode("utf-8", errors="surrogateescape"),
        changed_files,
        binary_files,
    )


def extract_edit_events(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return chronological edit boundaries from Codex, Claude, or OpenAI traces."""

    events: list[dict[str, Any]] = []
    pending_claude: dict[str, dict[str, Any]] = {}
    messages = list(trajectory.get("messages") or [])
    phase = 0

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        item = message.get("codex_item")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "agent_message":
                phase += 1
            if item_type == "file_change" and item.get("status") != "failed":
                files = [
                    _normalize_path(str(change.get("path", "")))
                    for change in item.get("changes") or []
                    if isinstance(change, dict) and change.get("path")
                ]
                events.append(
                    {
                        "message_index": message_index,
                        "event_id": str(item.get("id") or f"codex-edit-{message_index}"),
                        "provider": "codex",
                        "kind": "file_change",
                        "files": sorted(set(files)),
                        "phase": phase,
                    }
                )

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") in EDIT_TOOL_NAMES:
                    tool_id = str(block.get("id") or f"claude-edit-{message_index}")
                    pending_claude[tool_id] = {
                        "message_index": message_index,
                        "event_id": tool_id,
                        "provider": "claude",
                        "kind": str(block.get("name")),
                        "files": _paths_from_tool(
                            str(block.get("name")), block.get("input")
                        ),
                        "phase": phase,
                    }
                elif block.get("type") == "tool_result":
                    tool_id = str(block.get("tool_use_id") or "")
                    pending = pending_claude.pop(tool_id, None)
                    if pending is not None and not block.get("is_error", False):
                        pending["message_index"] = message_index
                        events.append(pending)

        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if name not in EDIT_TOOL_NAMES:
                continue
            events.append(
                {
                    "message_index": message_index,
                    "event_id": str(call.get("id") or f"tool-edit-{message_index}"),
                    "provider": "openai",
                    "kind": name,
                    "files": _paths_from_tool(name, function.get("arguments")),
                    "phase": phase,
                }
            )

    # Some normalized Claude trajectories omit tool-result messages.  Retain the
    # invocation as a best-effort boundary rather than silently losing the edit.
    events.extend(pending_claude.values())
    events.sort(key=lambda event: (event["message_index"], event["event_id"]))
    for edit_index, event in enumerate(events, start=1):
        event["edit_index"] = edit_index
    return events


def _split_patch_by_file(patch: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(_DIFF_FILE_RE.finditer(patch))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(patch)
        sections[_normalize_path(match.group(2))] = patch[match.start() : end]
    return sections


def _patch_section_deltas(section: str) -> tuple[int, int]:
    removed: list[str] = []
    added: list[str] = []
    for line in section.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    removed_text = "\n".join(removed)
    added_text = "\n".join(added)
    word_delta = len(removed_text.split()) - len(added_text.split())
    token_delta = count_lean_tokens_in_source(removed_text) - count_lean_tokens_in_source(
        added_text
    )
    return word_delta, token_delta


def _reconcile_file_deltas(
    raw: dict[str, float], exact_total: float, fallback_files: list[str]
) -> dict[str, float]:
    files = sorted(set(raw) | set(fallback_files))
    if not files:
        return {"<unattributed>": exact_total} if exact_total else {}
    values = {path: float(raw.get(path, 0.0)) for path in files}
    residual = exact_total - sum(values.values())
    weights = {path: abs(values[path]) + 1.0 for path in files}
    weight_total = sum(weights.values())
    for path in files:
        values[path] += residual * weights[path] / weight_total
    return values


def _allocate_events(
    events: list[dict[str, Any]], file_deltas: dict[str, float]
) -> list[float]:
    occurrences: dict[str, int] = defaultdict(int)
    for event in events:
        for path in event.get("files") or []:
            occurrences[path] += 1
    unattributed = sum(
        value for path, value in file_deltas.items() if occurrences.get(path, 0) == 0
    )
    allocations: list[float] = []
    for event in events:
        value = 0.0
        for path in event.get("files") or []:
            if occurrences[path]:
                value += file_deltas.get(path, 0.0) / occurrences[path]
        allocations.append(value)
    if allocations:
        allocations[-1] += unattributed
    return allocations


def backfill_instance(
    *,
    instance_id: str,
    trajectory: dict[str, Any],
    patch: str,
    metrics: dict[str, Any],
    total_cost_usd: float,
    wall_time_seconds: float,
) -> dict[str, Any]:
    """Build an estimated edit-level curve for a legacy trajectory.

    Historical Codex ``file_change`` items identify *when* and *which file* was
    edited but omit the patch body.  The final per-file deltas are therefore
    apportioned across that file's edit boundaries.  Baseline and endpoint
    metrics remain exact.
    """

    captured = trajectory.get("playback")
    if isinstance(captured, dict) and captured.get("points"):
        return {"instance_id": instance_id, **captured}

    events = extract_edit_events(trajectory)
    sections = _split_patch_by_file(patch)
    raw_words: dict[str, float] = {}
    raw_tokens: dict[str, float] = {}
    for path, section in sections.items():
        word_delta, token_delta = _patch_section_deltas(section)
        raw_words[path] = float(word_delta)
        raw_tokens[path] = float(token_delta)

    event_files = [path for event in events for path in event.get("files") or []]
    exact_words = _safe_float(metrics.get("words_saved"))
    exact_tokens = _safe_float(metrics.get("lean_tokens_saved"))
    file_words = _reconcile_file_deltas(raw_words, exact_words, event_files)
    file_tokens = _reconcile_file_deltas(raw_tokens, exact_tokens, event_files)
    word_allocations = _allocate_events(events, file_words)
    token_allocations = _allocate_events(events, file_tokens)

    baseline_words = _safe_float(metrics.get("baseline_words"))
    baseline_tokens = _safe_float(metrics.get("baseline_lean_tokens"))
    terminal_index = max(len(trajectory.get("messages") or []) - 1, 1)
    points: list[dict[str, Any]] = [
        {
            "edit_index": 0,
            "message_index": 0,
            "phase": 0,
            "cost_usd": 0.0,
            "elapsed_seconds": 0.0,
            "words_saved": 0.0,
            "lean_tokens_saved": 0.0,
            "word_compression_pct": 0.0,
            "lean_token_compression_pct": 0.0,
            "files": [],
            "exact": True,
        }
    ]
    cumulative_words = 0.0
    cumulative_tokens = 0.0
    for event, word_delta, token_delta in zip(
        events, word_allocations, token_allocations, strict=True
    ):
        cumulative_words += word_delta
        cumulative_tokens += token_delta
        fraction = min(1.0, max(0.0, event["message_index"] / terminal_index))
        points.append(
            {
                "edit_index": event["edit_index"],
                "message_index": event["message_index"],
                "phase": event.get("phase", 0),
                "cost_usd": None,
                "cost_boundary_exact": False,
                "elapsed_seconds": round(wall_time_seconds * fraction, 3),
                "words_saved": round(cumulative_words, 4),
                "lean_tokens_saved": round(cumulative_tokens, 4),
                "word_compression_pct": round(
                    100 * cumulative_words / baseline_words, 6
                )
                if baseline_words
                else 0.0,
                "lean_token_compression_pct": round(
                    100 * cumulative_tokens / baseline_tokens, 6
                )
                if baseline_tokens
                else 0.0,
                "files": event.get("files") or [],
                "exact": False,
            }
        )

    final_point = {
        "edit_index": len(events),
        "message_index": terminal_index,
        "phase": max((event.get("phase", 0) for event in events), default=0),
        "cost_usd": round(total_cost_usd, 8),
        "elapsed_seconds": round(wall_time_seconds, 3),
        "words_saved": exact_words,
        "lean_tokens_saved": exact_tokens,
        "word_compression_pct": round(100 * exact_words / baseline_words, 6)
        if baseline_words
        else 0.0,
        "lean_token_compression_pct": round(100 * exact_tokens / baseline_tokens, 6)
        if baseline_tokens
        else 0.0,
        "files": [],
        "exact": True,
    }
    if points[-1] != final_point:
        points.append(final_point)

    return {
        "instance_id": instance_id,
        "format": PLAYBACK_FORMAT,
        "quality": "historical_estimate",
        "metric_basis": "final per-file diff apportioned across edit boundaries",
        "cost_basis": "retained final total; historical edit usage unavailable",
        "baseline_words": baseline_words,
        "baseline_lean_tokens": baseline_tokens,
        "final_cost_usd": round(total_cost_usd, 8),
        "wall_time_seconds": round(wall_time_seconds, 3),
        "edit_count": len(events),
        "points": points,
    }


def _interpolate(points: list[dict[str, Any]], fraction: float, field: str) -> float:
    if not points:
        return 0.0
    final_cost = _safe_float(points[-1].get("cost_usd"))
    if final_cost <= 0:
        index = min(len(points) - 1, round(fraction * (len(points) - 1)))
        return _safe_float(points[index].get(field))
    target = fraction * final_cost
    previous = points[0]
    for point in points[1:]:
        current_cost = _safe_float(point.get("cost_usd"))
        previous_cost = _safe_float(previous.get("cost_usd"))
        if current_cost >= target:
            width = current_cost - previous_cost
            local = 0.0 if width <= 0 else (target - previous_cost) / width
            return _safe_float(previous.get(field)) + local * (
                _safe_float(point.get(field)) - _safe_float(previous.get(field))
            )
        previous = point
    return _safe_float(points[-1].get(field))


def aggregate_instances(
    instances: list[dict[str, Any]], *, steps: int = 100
) -> list[dict[str, Any]]:
    """Aggregate independent repo curves at equal per-repo spend fractions."""

    if not instances:
        return []
    if any(any(point.get("cost_usd") is None for point in instance.get("points", [])) for instance in instances):
        return []
    total_cost = sum(_safe_float(instance.get("final_cost_usd")) for instance in instances)
    total_wall = max(
        (_safe_float(instance.get("wall_time_seconds")) for instance in instances),
        default=0.0,
    )
    total_edits = sum(int(instance.get("edit_count") or 0) for instance in instances)
    baseline_words = sum(_safe_float(instance.get("baseline_words")) for instance in instances)
    baseline_tokens = sum(
        _safe_float(instance.get("baseline_lean_tokens")) for instance in instances
    )
    aggregate: list[dict[str, Any]] = []
    for step in range(steps + 1):
        fraction = step / steps if steps else 1.0
        words = 0.0
        tokens = 0.0
        repo_word_pct: list[float] = []
        for instance in instances:
            points = instance.get("points") or []
            instance_words = _interpolate(points, fraction, "words_saved")
            instance_tokens = _interpolate(points, fraction, "lean_tokens_saved")
            words += instance_words
            tokens += instance_tokens
            base = _safe_float(instance.get("baseline_words"))
            if base:
                repo_word_pct.append(100 * instance_words / base)
        repo_word_pct.sort()

        def quantile(q: float) -> float:
            if not repo_word_pct:
                return 0.0
            position = q * (len(repo_word_pct) - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return repo_word_pct[lower]
            weight = position - lower
            return repo_word_pct[lower] * (1 - weight) + repo_word_pct[upper] * weight

        aggregate.append(
            {
                "progress_fraction": fraction,
                "cumulative_cost_usd": round(total_cost * fraction, 6),
                "parallel_elapsed_seconds": round(total_wall * fraction, 3),
                "edit_index": round(total_edits * fraction, 2),
                "words_saved": round(words, 3),
                "lean_tokens_saved": round(tokens, 3),
                "word_compression_pct": round(100 * words / baseline_words, 6)
                if baseline_words
                else 0.0,
                "lean_token_compression_pct": round(100 * tokens / baseline_tokens, 6)
                if baseline_tokens
                else 0.0,
                "repo_word_pct_q25": round(quantile(0.25), 6),
                "repo_word_pct_median": round(quantile(0.5), 6),
                "repo_word_pct_q75": round(quantile(0.75), 6),
            }
        )
    return aggregate


@dataclass
class LivePlaybackRecorder:
    """Capture immutable source trees after completed subscription-agent actions."""

    env: Any
    provider: str = "auto"
    include_prefix: str = ""
    target_file: str = ""
    exclude_dirs: tuple[str, ...] = ()
    every_n_edits: int = 1
    snapshot_dir: str | Path | None = None
    snapshot_reference_prefix: str = ""
    instance_id: str = ""
    manifest_path: str | Path | None = None
    metric_reader: Callable[[], int | tuple[int, int]] | None = None
    build_command: str = ""
    build_timeout_seconds: int = 3600
    build_output_chars: int = 4000
    agent_pid_file: str = ""
    agent_pid: int | None = None

    def __post_init__(self) -> None:
        self.started_at = time.monotonic()
        self.points: list[dict[str, Any]] = []
        self.edit_count = 0
        self.event_count = 0
        self.base_commit = ""
        self._last_seen_tree_oid = ""
        self._claude_trace_builder = ClaudeTraceBuilder()
        self._kimi_trace_builder = KimiTraceBuilder()
        self._dsh_trace_builder = DeepSeekHarnessTraceBuilder()
        self._detected_stream_provider: str | None = None
        self._turn_responses: list[dict[str, Any]] = []
        self._antigravity_usage = AntigravityUsageJournal()
        self._claude_turn_responses: dict[tuple[str, str], dict[str, Any]] = {}
        self._prepared = False
        self._build_worktree = f"/tmp/leanlean-playback-build-{id(self)}"
        self._result: dict[str, Any] = {}
        self.baseline_lean_tokens: int | None = None

        self._captured_completed_turn_cost_usd = 0.0
        self._authoritative_final_cost_usd: float | None = None
        self._unattributed_terminal_cost_usd = 0.0
        self._cost_prorated = False
    def _read_tokens(self, tree_oid: str = "", *, env: Any | None = None) -> int:
        if self.metric_reader is not None:
            value = self.metric_reader()
            return int(value[-1] if isinstance(value, tuple) else value)
        from leanlean.benchmarks.leanlean import (
            _LEAN_TOKEN_COUNT_CMD,

            _install_lean_token_count_script,
        )

        capture_env = env or self.env
        if not self._prepared:
            _install_lean_token_count_script(capture_env)
            self._prepared = True
        extra = " ".join(shlex.quote(item) for item in self.exclude_dirs)
        token_env = (
            f"LEAN_TOKEN_INCLUDE={shlex.quote(self.include_prefix)} "
            if self.include_prefix
            else ""
        )
        if tree_oid:
            token_env += f"LEAN_TOKEN_GIT_TREE={shlex.quote(tree_oid)} "
        result = capture_env.execute(
            f"{token_env}{_LEAN_TOKEN_COUNT_CMD} {extra}".strip()
        )
        try:
            return int(result.get("output", "").strip())
        except ValueError:
            return -1

    def _create_tree_snapshot(self) -> str:
        """Write the live working tree into a private Git index and return its OID."""

        if self.metric_reader is not None:
            return ""
        index_path = f"/tmp/.leanlean-playback-{id(self)}.index"
        result = self.env.execute(
            "cd /testbed && "
            f"rm -f {shlex.quote(index_path)} {shlex.quote(index_path + '.lock')} && "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git read-tree "
            f"{shlex.quote(self.base_commit)} && "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git add -A -- . "
            "':(exclude).lake/' && "
            f"GIT_INDEX_FILE={shlex.quote(index_path)} git write-tree; "
            f"rc=$?; rm -f {shlex.quote(index_path)} "
            f"{shlex.quote(index_path + '.lock')}; exit $rc",
            timeout=False,
        )
        output_lines = result.get("output", "").strip().splitlines()
        tree_oid = output_lines[-1] if output_lines else ""
        if result.get("returncode") or not re.fullmatch(r"[0-9a-f]{40,64}", tree_oid):
            raise RuntimeError(
                f"Could not capture playback Git tree: {result.get('output', '')}"
            )
        return tree_oid

    def _snapshot_pathspec(self) -> str:
        if self.target_file:
            return f"-- {shlex.quote(self.target_file)}"
        if self.include_prefix:
            prefix = self.include_prefix.rstrip("/")
            return f"-- {shlex.quote(prefix + '/')} {shlex.quote(prefix + '.lean')}"
        return "-- . ':(exclude).lake/'"

    def _pause_agent(self) -> bool:
        if self.agent_pid is not None:
            result = self.env.execute(
                f"kill -STOP -- -{int(self.agent_pid)}",
                timeout=30,
            )
            return result.get("returncode") == 0
        if not self.agent_pid_file:
            return False
        pid_file = shlex.quote(self.agent_pid_file)
        result = self.env.execute(
            f"if test -s {pid_file}; then "
            f"pid=$(cat {pid_file}); kill -STOP -- \"-$pid\"; "
            "else exit 1; fi",
            timeout=30,
        )
        return result.get("returncode") == 0

    def _resume_agent(self) -> None:
        if self.agent_pid is not None:
            self.env.execute(
                f"kill -CONT -- -{int(self.agent_pid)} 2>/dev/null || true",
                timeout=30,
            )
            return
        if not self.agent_pid_file:
            return
        pid_file = shlex.quote(self.agent_pid_file)
        self.env.execute(
            f"if test -s {pid_file}; then "
            f"pid=$(cat {pid_file}); kill -CONT -- \"-$pid\" 2>/dev/null || true; fi",
            timeout=30,
        )

    def _diff_between(self, base_oid: str, tree_oid: str) -> str:
        result = self.env.execute(
            "cd /testbed && git diff --binary --full-index "
            f"{shlex.quote(base_oid)} {shlex.quote(tree_oid)} "
            f"{self._snapshot_pathspec()}",
            timeout=False,
        )
        if result.get("returncode"):
            raise RuntimeError(
                f"Could not serialize playback Git tree: {result.get('output', '')}"
            )
        return result.get("output", "")

    def _canonical_diff(self, tree_oid: str) -> str:
        return self._diff_between(self.base_commit, tree_oid)

    def _build_tree(self, tree_oid: str) -> dict[str, Any] | None:
        if not self.build_command or not tree_oid or self.metric_reader is not None:
            return None
        worktree = shlex.quote(self._build_worktree)
        author_env = (
            "GIT_AUTHOR_NAME=LeanLean GIT_AUTHOR_EMAIL=playback@localhost "
            "GIT_COMMITTER_NAME=LeanLean GIT_COMMITTER_EMAIL=playback@localhost"
        )
        setup = self.env.execute(
            "cd /testbed && "
            f"commit=$({author_env} git commit-tree {shlex.quote(tree_oid)} "
            f"-p {shlex.quote(self.base_commit)} -m playback) && "
            f"if git -C {worktree} rev-parse --git-dir >/dev/null 2>&1; then "
            f"test ! -L {worktree}/.lake || unlink {worktree}/.lake; "
            f"git -C {worktree} reset --hard \"$commit\" >/dev/null && "
            f"git -C {worktree} clean -fdx >/dev/null; "
            "else "
            f"git worktree add --detach {worktree} \"$commit\" >/dev/null; "
            "fi && "
            f"if test -d /testbed/.lake; then ln -s /testbed/.lake {worktree}/.lake; fi && "
            "printf '%s' \"$commit\"",
            timeout=300,
        )
        commit_lines = setup.get("output", "").strip().splitlines()
        commit_oid = commit_lines[-1] if commit_lines else ""
        if setup.get("returncode") or not re.fullmatch(r"[0-9a-f]{40,64}", commit_oid):
            return {
                "passed": False,
                "returncode": setup.get("returncode", 1),
                "command": self.build_command,
                "duration_seconds": 0.0,
                "output": setup.get("output", "")[-self.build_output_chars :],
                "tree_oid": tree_oid,
                "setup_failed": True,
            }
        started = time.monotonic()
        timed_out = False
        try:
            build = self.env.execute(
                f"cd {worktree} && {self.build_command}",
                timeout=self.build_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial = exc.output or getattr(exc, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            build = {"returncode": 124, "output": partial}
        duration = time.monotonic() - started
        unchanged = self.env.execute(
            f"git -C {worktree} diff --quiet {shlex.quote(commit_oid)} --",
            timeout=60,
        ).get("returncode") == 0
        return {
            "passed": build.get("returncode", 1) == 0,
            "returncode": build.get("returncode"),
            "command": self.build_command,
            "duration_seconds": round(duration, 3),
            "output": (build.get("output", "") or "")[-self.build_output_chars :],
            "tree_oid": tree_oid,
            "timed_out": timed_out,
            "source_tree_unchanged": unchanged,
        }

    def _cleanup_build_worktree(self) -> None:
        if not self.build_command or self.metric_reader is not None:
            return
        worktree = shlex.quote(self._build_worktree)
        self.env.execute(
            f"test ! -L {worktree}/.lake || unlink {worktree}/.lake; "
            f"git -C /testbed worktree remove --force {worktree} 2>/dev/null || true; "
            "git -C /testbed worktree prune",
            timeout=120,
        )

    def _write_snapshot(self, tree_oid: str) -> dict[str, Any]:
        if not tree_oid or self.snapshot_dir is None:
            return {"tree_oid": tree_oid} if tree_oid else {}
        payload = self._canonical_diff(tree_oid).encode(
            "utf-8", errors="surrogateescape"
        )
        directory = Path(self.snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"checkpoint_{self.edit_count:05d}.patch.gz"
        path = directory / filename
        with gzip.open(path, "wb", compresslevel=6) as stream:
            stream.write(payload)
        reference = (
            f"{self.snapshot_reference_prefix.rstrip('/')}/{filename}"
            if self.snapshot_reference_prefix
            else filename
        )
        return {
            "tree_oid": tree_oid,
            "snapshot_path": reference,
            "snapshot_format": "cumulative-git-diff-gzip-v1",
            "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
            "snapshot_bytes": len(payload),
        }

    def _capture_state(
        self, tree_oid: str | None = None
    ) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        tree_oid = tree_oid or self._create_tree_snapshot()
        tokens = self._read_tokens(tree_oid)
        return tokens, self._write_snapshot(tree_oid), self._build_tree(tree_oid)

    def _playback_result(
        self, *, cost_basis: str, final_cost_usd: float | None = None
    ) -> dict[str, Any]:
        elapsed = max(
            _safe_float(self.points[-1].get("elapsed_seconds")) if self.points else 0.0,
            0.0,
        )
        current_cost = (
            _safe_float(self.points[-1].get("cost_usd")) if self.points else 0.0
        )
        result = {
            "instance_id": self.instance_id,
            "format": PLAYBACK_FORMAT,
            "quality": "exact_tree_metrics",
            "provider": self.provider,
            "metric_basis": (
                "Lean tokens measured from immutable Git trees after changed actions"
            ),
            "cost_basis": cost_basis,
            "captured_completed_turn_cost_usd": self._captured_completed_turn_cost_usd,
            "authoritative_final_cost_usd": self._authoritative_final_cost_usd,
            "unattributed_terminal_cost_usd": self._unattributed_terminal_cost_usd,
            "cost_prorated": self._cost_prorated,
            "baseline_lean_tokens": self.baseline_lean_tokens,
            "final_cost_usd": round(
                current_cost if final_cost_usd is None else final_cost_usd, 8
            ),
            "wall_time_seconds": round(elapsed, 3),
            "edit_count": self.edit_count,
            "completed_action_count": self.event_count,
            "snapshot_base_commit": self.base_commit,
            "snapshot_basis": "cumulative binary Git diff from baseline commit",
            "build_command": self.build_command,
            "build_basis": "isolated worktree of each immutable checkpoint tree",
            "points": self.points,
        }
        unpaused_action_captures = len(
            {
                point.get("edit_index")
                for point in self.points
                if point.get("agent_paused") is False
                and point.get("kind") not in {"baseline", "final"}
            }
        )
        result["unpaused_action_capture_count"] = unpaused_action_captures
        result["capture_consistency"] = (
            "best_effort_unpaused"
            if unpaused_action_captures
            else "process_group_paused_at_action_boundaries"
        )
        if unpaused_action_captures:
            result["quality"] = "best_effort_unpaused_action_capture"
        if self._detected_stream_provider == "claude-code":
            trace = self._claude_trace_builder.to_trace(include_messages=False)
            trace["filesystem_changed_action_count"] = sum(
                bool(action.get("playback", {}).get("source_changed"))
                for action in trace["actions"]
            )
            trace["filesystem_checkpointed_action_count"] = sum(
                bool(action.get("playback", {}).get("checkpoint_attached"))
                for action in trace["actions"]
            )
            trace["filesystem_exact_checkpoint_count"] = sum(
                bool(action.get("playback", {}).get("exact"))
                for action in trace["actions"]
            )
            trace["filesystem_exactness"] = (
                "deterministic scoped source archive after each completed tool "
                "result when every_n_edits=1; final submission is digest-verified"
            )
            result["provider_action_trace"] = trace
        elif self._detected_stream_provider == "deepseek-harness":
            result["provider_action_trace"] = self._dsh_trace_builder.to_trace(
                native_stream_exact=True
            )
        return result

    def _persist_manifest(self, result: dict[str, Any]) -> None:
        if self.manifest_path is None:
            return
        path = Path(self.manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(path)

    def prepare(self) -> None:
        if self.metric_reader is None:
            base = self.env.execute(
                "cd /testbed && git rev-parse HEAD", timeout=False
            )
            base_lines = base.get("output", "").strip().splitlines()
            self.base_commit = base_lines[-1] if base_lines else ""
            if base.get("returncode") or not re.fullmatch(
                r"[0-9a-f]{40,64}", self.base_commit
            ):
                raise RuntimeError("Could not resolve playback baseline commit")
        tree_oid = self._create_tree_snapshot()
        tokens, snapshot, build = self._capture_state(tree_oid)
        self._last_seen_tree_oid = tree_oid
        self.baseline_lean_tokens = tokens
        point = {
            "edit_index": 0,
            "elapsed_seconds": 0.0,
            "cost_usd": 0.0,
            "lean_tokens": tokens,
            "lean_tokens_saved": 0,
            "files": [],
            "kind": "baseline",
            "exact": True,
            **snapshot,
        }
        if build is not None:
            point["build"] = build
        self.points.append(point)
        self._persist_manifest(
            self._playback_result(
                cost_basis="live provider usage; final cost pending"
            )
        )

    def _capture(
        self,
        *,
        files: list[str],
        kind: str,
        force: bool = False,
        pause_agent: bool = False,
        action_index: int | None = None,
        tool_use_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        if not self.points:
            self.prepare()
        paused = self._pause_agent() if pause_agent else False
        capture_exact = not (
            pause_agent and self.metric_reader is None
        ) or paused
        try:
            tree_oid = self._create_tree_snapshot()
            metric_mode = self.metric_reader is not None
            tree_changed = metric_mode or tree_oid != self._last_seen_tree_oid
            if not force:
                self.event_count += 1
                if not tree_changed:
                    self._persist_manifest(
                        self._playback_result(
                            cost_basis="live provider usage; final cost pending"
                        )
                    )
                    return
                self.edit_count += 1
                if tree_oid:
                    self._last_seen_tree_oid = tree_oid
                if self.edit_count % max(1, self.every_n_edits):
                    return
            elif not metric_mode:
                if tree_changed:
                    self.edit_count += 1
                    self._last_seen_tree_oid = tree_oid
                elif self.points[-1].get("tree_oid") == tree_oid:
                    terminal = dict(self.points[-1])
                    terminal.update(
                        {
                            "edit_index": self.edit_count,
                            "elapsed_seconds": round(
                                time.monotonic() - self.started_at, 3
                            ),
                            "cost_usd": round(
                                cost_from_responses(self._turn_responses), 8
                            ),
                            "files": sorted(set(files)),
                            "kind": kind,
                        }
                    )
                    self.points.append(terminal)
                    self._persist_manifest(
                        self._playback_result(
                            cost_basis="live provider usage; final cost pending"
                        )
                    )
                    return

            tokens, snapshot, build = self._capture_state(tree_oid)
            point = {
                "edit_index": self.edit_count,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "cost_usd": round(cost_from_responses(self._turn_responses), 8),
                "lean_tokens": tokens,
                "lean_tokens_saved": (
                    self.baseline_lean_tokens - tokens if tokens >= 0 else None
                ),
                "files": sorted(set(files)),
                "kind": kind,
                "exact": capture_exact,
                "action_index": action_index,
                "cost_boundary_exact": self._detected_stream_provider == "claude-code",
                "native_event_sequence": self._antigravity_usage.sequence if self._detected_stream_provider == "antigravity-cli" else None,
                "tool_use_id": tool_use_id,
                "agent_id": agent_id,
                "agent_paused": (
                    paused if pause_agent and self.metric_reader is None else None
                ),
                **snapshot,
            }
            if build is not None:
                point["build"] = build
            self.points.append(point)
            self._persist_manifest(
                self._playback_result(
                    cost_basis="live provider usage; final cost pending"
                )
            )
        finally:
            if paused:
                self._resume_agent()

    @staticmethod
    def _claude_response(message: dict[str, Any]) -> dict[str, Any] | None:
        usage = message.get("usage")
        if not isinstance(usage, dict):
            return None
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
        return {
            "model": message.get("model") or "unknown_model",
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

    def _record_claude_turn_response(self, message: dict[str, Any]) -> None:
        """Keep one usage block per native Claude turn and subagent lane."""

        response = self._claude_response(message)
        if response is None:
            return
        agent_id = str(message.get("trace_agent_id") or "root")
        turn_id = str(
            message.get("id")
            or message.get("request_id")
            or f"assistant-block-{message.get('trace_sequence')}"
        )
        self._claude_turn_responses[(agent_id, turn_id)] = response
        self._turn_responses = list(self._claude_turn_responses.values())

    def ingest_line(self, line: str) -> None:
        stripped = line.strip()
        if stripped.startswith(AGENT_PID_PREFIX):
            try:
                self.agent_pid = int(stripped[len(AGENT_PID_PREFIX) :])
            except ValueError:
                self.agent_pid = None
            return
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            return
        if not isinstance(event, dict):
            return

        detected = detect_trace_provider([event])
        if detected != "generic-jsonl":
            self._detected_stream_provider = detected

        if detected == "antigravity-cli":
            self._antigravity_usage.ingest(event)
            self._turn_responses = self._antigravity_usage.responses
            action = antigravity_action(
                event,
                action_index=self.event_count + 1,
                completed_sequence=0,
            )
            if action is not None:
                risk_recorder = getattr(self, "_record_action_boundary_risk", None)
                if callable(risk_recorder):
                    risk_recorder(action)
                name = str(action.get("name") or "unknown")
                self._capture(
                    files=_paths_from_tool(name, action.get("arguments")),
                    kind=f"antigravity_{name}",
                    pause_agent=True,
                    action_index=int(action["action_index"]),
                    tool_use_id=str(action.get("tool_call_id") or "") or None,
                    agent_id="root",
                )
            return

        if detected == "deepseek-harness":
            for action in self._dsh_trace_builder.ingest(event):
                risk_recorder = getattr(self, "_record_action_boundary_risk", None)
                if callable(risk_recorder):
                    risk_recorder(action)
                name = str(action.get("name") or "unknown")
                self._capture(
                    files=_paths_from_tool(name, action.get("arguments")),
                    kind=f"dsh_{name}",
                    pause_agent=True,
                    action_index=int(action.get("action_index") or 0) or None,
                    tool_use_id=str(action.get("tool_call_id") or "") or None,
                    agent_id=str(action.get("agent_id") or "root"),
                )
            self._turn_responses = list(self._dsh_trace_builder.responses)
            return

        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            item_type = str(item.get("type") or "unknown")
            files = [
                _normalize_path(str(change.get("path", "")))
                for change in item.get("changes") or []
                if isinstance(change, dict) and change.get("path")
            ]
            self._capture(
                files=files,
                kind=item_type,
                pause_agent=True,
                action_index=self.event_count + 1,
                tool_use_id=str(item.get("id") or "") or None,
            )
            return

        if detected == "mistral-vibe" and event.get("type") == "effect":
            state = event.get("state")
            generation_status = event.get(
                "generationStatus", event.get("generation_status")
            )
            state_status = (
                str(state.get("status") or "")
                if isinstance(state, Mapping)
                else ""
            )
            if generation_status == "completed" and state_status not in {
                "",
                "pending",
                "running",
                "blocked",
            }:
                action = mistral_vibe_action(
                    event,
                    action_index=self.event_count + 1,
                    completed_sequence=0,
                )
                if action is not None:
                    risk_recorder = getattr(
                        self, "_record_action_boundary_risk", None
                    )
                    if callable(risk_recorder):
                        risk_recorder(action)
                    name = str(action.get("name") or "unknown")
                    self._capture(
                        files=_paths_from_tool(name, action.get("arguments")),
                        kind=f"mistral_vibe_{action.get('effect_kind')}_{name}",
                        pause_agent=True,
                        action_index=int(action["action_index"]),
                        tool_use_id=str(action.get("tool_call_id") or "") or None,
                        agent_id="root",
                    )
                    return

        ingested = self._claude_trace_builder.ingest(event)
        for normalized in ingested["messages"]:
            if normalized.get("role") != "assistant":
                continue
            self._record_claude_turn_response(normalized)
        for action in ingested["completed_actions"]:
            risk_recorder = getattr(self, "_record_action_boundary_risk", None)
            if callable(risk_recorder):
                risk_recorder(action)
            before_edit_count = self.edit_count
            before_point_count = len(self.points)
            name = str(action.get("name") or "unknown")
            files = _paths_from_tool(name, action.get("input"))
            self._capture(
                files=files,
                kind=f"claude_{name}",
                pause_agent=True,
                action_index=int(action.get("action_index") or 0) or None,
                tool_use_id=str(action.get("tool_use_id") or "") or None,
                agent_id=str(action.get("agent_id") or "") or None,
            )
            source_changed = self.edit_count > before_edit_count
            checkpoint = (
                self.points[-1]
                if source_changed and len(self.points) > before_point_count
                else None
            )
            action["playback"] = {
                "source_changed": source_changed,
                "checkpoint_attached": checkpoint is not None,
                "checkpoint_edit_index": (
                    checkpoint.get("edit_index") if checkpoint else None
                ),
                "snapshot_path": (
                    checkpoint.get("snapshot_path") if checkpoint else None
                ),
                "snapshot_sha256": (
                    checkpoint.get("snapshot_sha256") if checkpoint else None
                ),
                "incremental_diff_path": (
                    checkpoint.get("incremental_diff_path")
                    if checkpoint
                    else None
                ),
                "incremental_diff_sha256": (
                    checkpoint.get("incremental_diff_sha256")
                    if checkpoint
                    else None
                ),
                "cost_usd": checkpoint.get("cost_usd") if checkpoint else None,
                "cost_boundary_exact": bool(
                    checkpoint and checkpoint.get("cost_boundary_exact")
                ),
                "exact": bool(checkpoint and checkpoint.get("exact")),
            }
            self._persist_manifest(
                self._playback_result(
                    cost_basis="live provider usage; final cost pending"
                )
            )

        for action in self._kimi_trace_builder.ingest(event):
            risk_recorder = getattr(self, "_record_action_boundary_risk", None)
            if callable(risk_recorder):
                risk_recorder(action)
            name = str(action.get("name") or "unknown")
            self._capture(
                files=_paths_from_tool(name, action.get("input")),
                kind=f"kimi_{name}",
                pause_agent=True,
                action_index=int(action.get("action_index") or 0) or None,
                tool_use_id=str(action.get("tool_use_id") or "") or None,
                agent_id="root",
            )

    def finish(self, authoritative_responses: list[dict[str, Any]]) -> dict[str, Any]:
        before_terminal_edit_count = self.edit_count
        self._capture(files=[], kind="final", force=True)
        terminal_change = self.edit_count > before_terminal_edit_count
        if hasattr(self, "_terminal_unattributed_source_change"):
            self._terminal_unattributed_source_change = terminal_change

        if self._detected_stream_provider == "antigravity-cli":
            evidence = self._antigravity_usage.evidence()
            final_cost = evidence["recorded_cost_usd"]
            self._captured_completed_turn_cost_usd = final_cost
            self._authoritative_final_cost_usd = final_cost if evidence["usage_complete"] else None
            self._cost_prorated = False
            result = self._playback_result(cost_basis="recorded native token usage; no proration", final_cost_usd=final_cost)
            apply_recorded_costs(result, self._antigravity_usage)
            result["terminal_unattributed_source_change"] = terminal_change
            self._result = result
            self._persist_manifest(result)
            return result

        final_cost = cost_from_responses(authoritative_responses)
        observed_cost = round(cost_from_responses(self._turn_responses), 8)
        self._captured_completed_turn_cost_usd = observed_cost
        self._authoritative_final_cost_usd = round(final_cost, 8)
        if self._detected_stream_provider == "claude-code":
            self._unattributed_terminal_cost_usd = round(
                max(0.0, final_cost - observed_cost),
                8,
            )
            cost_basis = (
                "deduplicated native assistant-turn usage at action boundaries; "
                "authoritative terminal total recorded separately; no proration"
            )
        else:
            # Preserve captured usage without scaling it to an unrelated total.
            # A missing boundary is unknown, never elapsed-time proration.
            self._cost_prorated = False
            cost_basis = "recorded provider usage; unmeasured boundaries unknown; no proration"
            self._unattributed_terminal_cost_usd = round(max(0.0, final_cost - observed_cost), 8)
            if observed_cost <= 0:
                for point in self.points:
                    if point.get("kind") not in {"baseline", "final"}:
                        point["cost_usd"] = None
                        point["cost_boundary_exact"] = False
                        point["cost_join_basis"] = "missing_recorded_usage"

        if self.points:
            self.points[-1]["cost_usd"] = round(final_cost, 8)
            self.points[-1]["cost_boundary_exact"] = True
        previous_edit_cost = 0.0
        for point in self.points:
            if int(point.get("edit_index") or 0) <= 0 or point.get("kind") == "final":
                continue
            cost = point.get("cost_usd")
            point["marginal_cost_since_previous_edit_usd"] = (
                round(cost - previous_edit_cost, 8)
                if cost is not None and previous_edit_cost is not None else None
            )
            previous_edit_cost = cost

        result = self._playback_result(cost_basis=cost_basis, final_cost_usd=final_cost)
        result["terminal_unattributed_source_change"] = terminal_change
        self._result = result
        self._persist_manifest(result)
        return result

    def verify_submission(self, patch: str) -> dict[str, Any]:
        """Verify that the submitted patch reconstructs the final scoped tree."""

        result = self._result or self._playback_result(
            cost_basis="live provider usage; final cost pending"
        )
        captured_tree = str(self.points[-1].get("tree_oid") or "") if self.points else ""
        verification: dict[str, Any] = {
            "checked": False,
            "matches": False,
            "captured_tree_oid": captured_tree,
            "basis": "canonical scoped Git diff from submission base",
        }
        patch_path = f"/tmp/leanlean-playback-submission-{id(self)}.patch"
        index_path = f"/tmp/.leanlean-playback-submission-{id(self)}.index"
        try:
            if self.metric_reader is not None or not captured_tree:
                raise RuntimeError("Submission verification requires captured Git trees")
            writer = getattr(self.env, "write_file", None)
            if not callable(writer):
                raise RuntimeError("Environment does not support submission patch upload")

            base_result = self.env.execute(
                "cd /testbed && "
                "base=$(cat /.init_commit 2>/dev/null || "
                "git rev-list --max-parents=0 HEAD | tail -1) && "
                "printf '%s' \"$base\"",
                timeout=False,
            )
            base_lines = base_result.get("output", "").strip().splitlines()
            submission_base = base_lines[-1] if base_lines else ""
            if base_result.get("returncode") or not re.fullmatch(
                r"[0-9a-f]{40,64}", submission_base
            ):
                raise RuntimeError("Could not resolve submitted patch baseline")

            writer(patch_path, patch)
            index = shlex.quote(index_path)
            patch_file = shlex.quote(patch_path)
            applied = self.env.execute(
                "cd /testbed && "
                f"rm -f {index} {shlex.quote(index_path + '.lock')} && "
                f"GIT_INDEX_FILE={index} git read-tree "
                f"{shlex.quote(submission_base)} && "
                f"{{ test ! -s {patch_file} || "
                f"GIT_INDEX_FILE={index} git apply --cached --binary "
                f"--whitespace=nowarn {patch_file}; }} && "
                f"GIT_INDEX_FILE={index} git write-tree",
                timeout=False,
            )
            tree_lines = applied.get("output", "").strip().splitlines()
            submitted_tree = tree_lines[-1] if tree_lines else ""
            if applied.get("returncode") or not re.fullmatch(
                r"[0-9a-f]{40,64}", submitted_tree
            ):
                raise RuntimeError(
                    "Submitted patch could not be reconstructed: "
                    + applied.get("output", "")[-self.build_output_chars :]
                )

            captured_diff = self._diff_between(submission_base, captured_tree)
            submitted_diff = self._diff_between(submission_base, submitted_tree)
            matches = captured_diff == submitted_diff
            verification.update(
                {
                    "checked": True,
                    "matches": matches,
                    "submission_base_commit": submission_base,
                    "submitted_tree_oid": submitted_tree,
                    "captured_canonical_diff_sha256": hashlib.sha256(
                        captured_diff.encode("utf-8", errors="surrogateescape")
                    ).hexdigest(),
                    "submitted_canonical_diff_sha256": hashlib.sha256(
                        submitted_diff.encode("utf-8", errors="surrogateescape")
                    ).hexdigest(),
                }
            )
            if not matches:
                mismatch = self.env.execute(
                    "cd /testbed && git diff --binary --full-index "
                    f"{shlex.quote(captured_tree)} {shlex.quote(submitted_tree)} "
                    f"{self._snapshot_pathspec()}",
                    timeout=False,
                )
                verification["mismatch_output"] = mismatch.get("output", "")[
                    -self.build_output_chars :
                ]
        except Exception as exc:
            verification["error"] = str(exc)
        finally:
            if self.metric_reader is None:
                try:
                    self.env.execute(
                        f"rm -f {shlex.quote(index_path)} "
                        f"{shlex.quote(index_path + '.lock')} "
                        f"{shlex.quote(patch_path)}",
                        timeout=60,
                    )
                except Exception:
                    pass
                try:
                    self._cleanup_build_worktree()
                except Exception:
                    pass

        result["submission_verification"] = verification
        result["final_snapshot_matches_submission"] = verification["matches"]
        if not verification["checked"]:
            result["quality"] = "unverified_final_tree"
        elif not verification["matches"]:
            result["quality"] = "invalid_final_tree_reconstruction"
        if self.points:
            self.points[-1]["matches_submission"] = verification["matches"]
        self._result = result
        self._persist_manifest(result)
        return result


@dataclass
class ExternalPlaybackRecorder(LivePlaybackRecorder):
    """Capture source archives on the host and replay them after generation."""

    capture_timeout_seconds: int = 600
    reject_asynchronous_actions: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self._last_seen_archive_sha256 = ""
        self._external_replayed = False
        self._external_quality = "external_source_snapshots_pending_replay"
        self._asynchronous_actions: list[dict[str, Any]] = []
        self._terminal_unattributed_source_change = False

    def _archive_paths(self) -> list[str]:
        if self.target_file:
            return [self.target_file]
        if self.include_prefix:
            prefix = self.include_prefix.rstrip("/")
            return [prefix, prefix + ".lean"]
        return ["."]

    def _archive_reference(self, filename: str) -> str:
        if self.snapshot_reference_prefix:
            return f"{self.snapshot_reference_prefix.rstrip('/')}/{filename}"
        return filename

    def _archive_file(self, point: dict[str, Any]) -> Path:
        if self.snapshot_dir is None:
            raise RuntimeError("External playback requires a host snapshot directory")
        return Path(self.snapshot_dir) / Path(str(point["snapshot_path"])).name

    def _capture_archive_candidate(self) -> tuple[Path, dict[str, Any], float]:
        if self.snapshot_dir is None:
            raise RuntimeError("External playback requires capture_snapshots: true")
        exporter = getattr(self.env, "export_source_archive", None)
        if not callable(exporter):
            raise RuntimeError("External playback requires a Docker archive exporter")
        directory = Path(self.snapshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / (
            f".candidate-{id(self)}-{self.event_count}-{time.time_ns()}.tar.gz"
        )
        started = time.monotonic()
        metadata = exporter(
            candidate,
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        return candidate, metadata, time.monotonic() - started

    def _finalize_archive(
        self,
        candidate: Path,
        metadata: dict[str, Any],
        *,
        capture_seconds: float,
    ) -> dict[str, Any]:
        filename = f"checkpoint_{self.edit_count:05d}.sources.tar.gz"
        destination = Path(self.snapshot_dir) / filename
        candidate.replace(destination)
        payload = destination.read_bytes()
        source_files = _archive_source_files(destination)
        return {
            "snapshot_artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "snapshot_artifact_bytes": len(payload),
            "file_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(source_files.items())
            },
            "snapshot_path": self._archive_reference(filename),
            "snapshot_format": "deterministic-source-tar-gzip-v1",
            "snapshot_sha256": metadata["source_archive_sha256"],
            "snapshot_digest_basis": "canonical uncompressed tar stream",
            "snapshot_bytes": metadata["archive_bytes"],
            "source_archive_bytes": metadata["source_archive_bytes"],
            "source_archive_sha256": metadata["source_archive_sha256"],
            "capture_seconds": round(capture_seconds, 3),
            "replay_status": "pending",
        }

    def _write_incremental_diff(
        self,
        previous_archive: Path,
        current_archive: Path,
    ) -> dict[str, Any]:
        before = _archive_source_files(previous_archive)
        after = _archive_source_files(current_archive)
        payload, changed_files, binary_files = _incremental_source_diff(before, after)
        filename = f"checkpoint_{self.edit_count:05d}.incremental.diff"
        destination = Path(self.snapshot_dir) / filename
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite edit diff: {destination}")
        destination.write_bytes(payload)
        return {
            "changed_files": changed_files,
            "binary_changed_files": binary_files,
            "incremental_diff_path": self._archive_reference(filename),
            "incremental_diff_sha256": hashlib.sha256(payload).hexdigest(),
            "incremental_diff_bytes": len(payload),
            "incremental_diff_format": "unified-source-diff-v1",
            "incremental_diff_reconstructs_all_changed_bytes": not binary_files,
        }


    def _record_action_boundary_risk(self, action: Mapping[str, Any]) -> None:
        if str(action.get("name") or "").lower() != "bash":
            return
        arguments = action.get("input", action.get("arguments"))
        command = (
            arguments.get("command")
            if isinstance(arguments, Mapping)
            else None
        )
        reason = (
            _shell_asynchrony_reason(command)
            if isinstance(command, str)
            else None
        )
        if reason:
            self._asynchronous_actions.append(
                {
                    "action_index": action.get("action_index"),
                    "tool_use_id": action.get(
                        "tool_use_id", action.get("tool_call_id")
                    ),
                    "reason": reason,
                }
            )


    def _playback_result(
        self, *, cost_basis: str, final_cost_usd: float | None = None
    ) -> dict[str, Any]:
        result = super()._playback_result(
            cost_basis=cost_basis,
            final_cost_usd=final_cost_usd,
        )
        edit_points = [
            point
            for point in self.points
            if int(point.get("edit_index") or 0) > 0
            and point.get("kind") != "final"
        ]
        archived = [
            point
            for point in edit_points
            if point.get("snapshot_artifact_sha256")
            and point.get("snapshot_artifact_bytes")
        ]
        diffed = [
            point
            for point in edit_points
            if point.get("incremental_diff_sha256")
            and point.get("incremental_diff_bytes")
        ]
        built = [
            point for point in edit_points if isinstance(point.get("build"), dict)
        ]
        passed = [
            point for point in built if point["build"].get("passed") is True
        ]
        costed = [
            point
            for point in edit_points
            if point.get("cost_boundary_exact")
            and isinstance(point.get("cost_usd"), (int, float))
        ]
        boundary_complete = (
            len(edit_points) == self.edit_count
            and all(point.get("action_index") is not None for point in edit_points)
            and all(point.get("exact") is True for point in edit_points)
            and not self._terminal_unattributed_source_change
            and not self._asynchronous_actions
        )
        result.update(
            {
                "quality": self._external_quality,
                "capture_mode": "external_replay",
                "metric_basis": (
                    "Lean tokens measured during post-run replay"
                    if self._external_replayed
                    else "pending post-run replay"
                ),
                "snapshot_base_commit": "",
                "snapshot_basis": (
                    "deterministic scoped source archives streamed to the host"
                ),
                "build_basis": (
                    "post-run disposable container from the pinned baseline image"
                ),
                "agent_container_source_writes": False,
                "agent_container_git_writes": False,
                "agent_container_builds": False,
                "edit_checkpoint_count": len(edit_points),
                "every_edit_checkpoint_archived": (
                    len(archived) == self.edit_count and self.edit_count > 0
                ),
                "every_edit_checkpoint_diffed": (
                    len(diffed) == self.edit_count and self.edit_count > 0
                ),
                "every_edit_checkpoint_built": (
                    len(built) == self.edit_count and self.edit_count > 0
                ),
                "every_edit_checkpoint_costed": (
                    len(costed) == self.edit_count and self.edit_count > 0
                ),
                "checkpoint_build_attempt_count": len(built),
                "checkpoint_build_pass_count": len(passed),
                "checkpoint_build_fail_count": len(built) - len(passed),
                "all_checkpoint_builds_passed": (
                    len(passed) == self.edit_count and self.edit_count > 0
                ),
                "action_boundary_sequence_complete": boundary_complete,
                "terminal_unattributed_source_change": (
                    self._terminal_unattributed_source_change
                ),
                "asynchronous_actions": list(self._asynchronous_actions),
                "reconstruction_basis": (
                    "each changed action has a full byte-authoritative source "
                    "archive; the incremental diff is explanatory"
                ),
            }
        )
        if result["unpaused_action_capture_count"]:
            result["quality"] = "best_effort_unpaused_action_capture"
        return result

    def prepare(self) -> None:
        self.base_commit = ""
        self.baseline_lean_tokens = None
        candidate, metadata, capture_seconds = self._capture_archive_candidate()
        self._last_seen_archive_sha256 = metadata["source_archive_sha256"]
        snapshot = self._finalize_archive(
            candidate,
            metadata,
            capture_seconds=capture_seconds,
        )
        self.points.append(
            {
                "edit_index": 0,
                "elapsed_seconds": 0.0,
                "cost_usd": 0.0,
                "lean_tokens": None,
                "lean_tokens_saved": None,
                "files": [],
                "kind": "baseline",
                "exact": True,
                **snapshot,
            }
        )
        self._persist_manifest(
            self._playback_result(
                cost_basis="live provider usage; final cost pending"
            )
        )

    def _capture(
        self,
        *,
        files: list[str],
        kind: str,
        force: bool = False,
        pause_agent: bool = False,
        action_index: int | None = None,
        tool_use_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        if not self.points:
            self.prepare()
        paused = self._pause_agent() if pause_agent else False
        previous_archive = self._archive_file(self.points[-1])
        capture_exact = not pause_agent or paused
        candidate: Path | None = None
        try:
            candidate, metadata, capture_seconds = self._capture_archive_candidate()
            digest = metadata["source_archive_sha256"]
            changed = digest != self._last_seen_archive_sha256
            if not force:
                self.event_count += 1
                if not changed:
                    candidate.unlink(missing_ok=True)
                    self._persist_manifest(
                        self._playback_result(
                            cost_basis="live provider usage; final cost pending"
                        )
                    )
                    return
                self.edit_count += 1
                self._last_seen_archive_sha256 = digest
                if self.edit_count % max(1, self.every_n_edits):
                    candidate.unlink(missing_ok=True)
                    return
            elif changed:
                self.edit_count += 1
                self._last_seen_archive_sha256 = digest
            elif self.points[-1].get("source_archive_sha256") == digest:
                candidate.unlink(missing_ok=True)
                terminal = dict(self.points[-1])
                terminal.update(
                    {
                        "edit_index": self.edit_count,
                        "elapsed_seconds": round(
                            time.monotonic() - self.started_at, 3
                        ),
                        "cost_usd": round(
                            cost_from_responses(self._turn_responses), 8
                        ),
                        "files": sorted(set(files)),
                        "kind": kind,
                        "action_index": None,
                        "tool_use_id": None,
                        "agent_id": None,
                    }
                )
                self.points.append(terminal)
                self._persist_manifest(
                    self._playback_result(
                        cost_basis="live provider usage; final cost pending"
                    )
                )
                return

            snapshot = self._finalize_archive(
                candidate,
                metadata,
                capture_seconds=capture_seconds,
            )
            candidate = None
            diff = self._write_incremental_diff(
                previous_archive,
                self._archive_file(snapshot),
            )
            self.points.append(
                {
                    "edit_index": self.edit_count,
                    "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                    "cost_usd": round(
                        cost_from_responses(self._turn_responses), 8
                    ),
                    "lean_tokens": None,
                    "lean_tokens_saved": None,
                    "files": sorted(set(files)),
                    "kind": kind,
                    "exact": capture_exact,
                    "action_index": action_index,
                    "tool_use_id": tool_use_id,
                    "agent_id": agent_id,
                    "cost_boundary_exact": (
                        self._detected_stream_provider == "claude-code"
                    ),
                    "agent_paused": paused if pause_agent else None,
                    **diff,
                    **snapshot,
                }
            )
            self._persist_manifest(
                self._playback_result(
                    cost_basis="live provider usage; final cost pending"
                )
            )
        finally:
            if candidate is not None:
                candidate.unlink(missing_ok=True)
            if paused:
                self._resume_agent()

    def _run_replay_build(self, replay_env: Any) -> dict[str, Any] | None:
        if not self.build_command:
            return None
        started = time.monotonic()
        timed_out = False
        try:
            build = replay_env.execute(
                f"cd /testbed && {self.build_command}",
                timeout=self.build_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial = exc.output or getattr(exc, "stdout", None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            build = {"returncode": 124, "output": partial}
        return {
            "passed": build.get("returncode", 1) == 0,
            "returncode": build.get("returncode"),
            "command": self.build_command,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output": (build.get("output", "") or "")[-self.build_output_chars :],
            "timed_out": timed_out,
            "environment": "disposable_replay_container",
        }

    def _verify_external_submission(
        self,
        replay_env: Any,
        patch: str,
    ) -> dict[str, Any]:
        baseline_archive = self._archive_file(self.points[0])
        replay_env.restore_source_archive(
            baseline_archive,
            self._archive_paths(),
            timeout=self.capture_timeout_seconds,
        )
        patch_path = f"/tmp/leanlean-playback-submission-{id(self)}.patch"
        write_file_bytes = getattr(replay_env, "write_file_bytes", None)
        if callable(write_file_bytes):
            write_file_bytes(
                patch_path,
                patch.encode("utf-8", errors="surrogateescape"),
            )
        else:
            replay_env.write_file(patch_path, patch)
        apply_command = (
            "cd /testbed && "
            f"{{ test ! -s {shlex.quote(patch_path)} || "
            f"git apply --binary --whitespace=nowarn {shlex.quote(patch_path)}; }}"
        )
        applied = replay_env.execute(apply_command, timeout=False)
        absent_deletions: list[str] = []
        if applied.get("returncode"):
            # Deleting a path the baseline never had is a no-op that git apply
            # rejects outright; retry once without only those hunks (git apply
            # is atomic on failure, so the restored baseline is untouched).
            baseline_names = set(_archive_source_files(baseline_archive))
            if "__scoped_source__" not in baseline_names:
                filtered, absent_deletions = drop_absent_deletions(
                    patch, lambda path: path in baseline_names
                )
            if absent_deletions:
                payload = filtered.encode("utf-8", errors="surrogateescape")
                if callable(write_file_bytes):
                    write_file_bytes(patch_path, payload)
                else:
                    replay_env.write_file(patch_path, filtered)
                applied = replay_env.execute(apply_command, timeout=False)
        if applied.get("returncode"):
            return {
                "checked": True,
                "matches": False,
                "basis": "exact scoped source archive",
                "error": (
                    "Submitted patch could not be applied during replay: "
                    + applied.get("output", "")[-self.build_output_chars :]
                ),
            }

        verification_archive = Path(self.snapshot_dir) / (
            f".submission-verification-{id(self)}.tar.gz"
        )
        try:
            metadata = replay_env.export_source_archive(
                verification_archive,
                self._archive_paths(),
                timeout=self.capture_timeout_seconds,
            )
            submitted_digest = metadata["source_archive_sha256"]
        finally:
            verification_archive.unlink(missing_ok=True)
        captured_digest = str(
            self.points[-1].get("source_archive_sha256") or ""
        )
        return {
            "checked": True,
            "matches": submitted_digest == captured_digest,
            "basis": "exact scoped source archive",
            "captured_source_archive_sha256": captured_digest,
            "submitted_source_archive_sha256": submitted_digest,
            **(
                {"absent_path_deletions_ignored": absent_deletions}
                if absent_deletions
                else {}
            ),
        }

    def replay_submission(self, patch: str) -> dict[str, Any]:
        """Measure and build every captured state in a disposable container."""

        spawner = getattr(self.env, "spawn_replay_environment", None)
        if not callable(spawner):
            raise RuntimeError("External playback requires a replay environment factory")
        unique_archives = {
            str(point.get("snapshot_path") or "")
            for point in self.points
            if point.get("snapshot_path")
        }
        lifetime = max(
            600,
            len(unique_archives)
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
            measurements: dict[str, tuple[int, dict[str, Any] | None]] = {}
            for point in self.points:
                reference = str(point.get("snapshot_path") or "")
                if reference not in measurements:
                    archive = self._archive_file(point)
                    replay_env.restore_source_archive(
                        archive,
                        self._archive_paths(),
                        timeout=self.capture_timeout_seconds,
                    )
                    tokens = self._read_tokens(env=replay_env)
                    build = self._run_replay_build(replay_env)
                    measurements[reference] = (tokens, build)
                tokens, build = measurements[reference]
                point["lean_tokens"] = tokens
                point["replay_status"] = "complete"
                if build is not None:
                    point["build"] = dict(build)

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

            verification = self._verify_external_submission(replay_env, patch)
            self._external_replayed = True
            if not verification["matches"]:
                self._external_quality = "invalid_final_tree_reconstruction"
            elif self._terminal_unattributed_source_change:
                self._external_quality = "invalid_unattributed_terminal_source_change"
            elif self._asynchronous_actions:
                self._external_quality = "invalid_asynchronous_action_boundaries"
            elif any(
                point.get("agent_paused") is False
                for point in self.points
                if point.get("kind") not in {"baseline", "final"}
            ):
                self._external_quality = "best_effort_unpaused_action_capture"
            else:
                self._external_quality = "exact_external_replay"
            result = self._playback_result(
                cost_basis=str(
                    self._result.get(
                        "cost_basis",
                        "authoritative final cost prorated by elapsed time",
                    )
                ),
                final_cost_usd=_safe_float(
                    self._result.get("final_cost_usd")
                ),
            )
            result["submission_verification"] = verification
            result["final_snapshot_matches_submission"] = verification["matches"]
            if self.points:
                self.points[-1]["matches_submission"] = verification["matches"]
            self._result = result
            self._persist_manifest(result)
            return result
        except Exception as exc:
            self._external_quality = "external_replay_failed"
            result = self._result or self._playback_result(
                cost_basis="external replay failed"
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


def summarize_playback(instances: list[dict[str, Any]], *, steps: int = 100) -> dict[str, Any]:
    qualities = sorted(set(str(instance.get("quality")) for instance in instances))
    aggregate = aggregate_instances(instances, steps=steps)
    valid_instances = [
        instance
        for instance in instances
        if instance.get("build_passed") and instance.get("signatures_preserved")
    ]
    aggregate_valid = aggregate_instances(valid_instances, steps=steps)
    return {
        "format": PLAYBACK_FORMAT,
        "quality": qualities[0] if len(qualities) == 1 else "mixed",
        "repo_count": len(instances),
        "valid_repo_count": len(valid_instances),
        "total_cost_usd": round(
            sum(_safe_float(instance.get("final_cost_usd")) for instance in instances), 6
        ),
        "total_edits": sum(int(instance.get("edit_count") or 0) for instance in instances),
        "baseline_words": round(
            sum(_safe_float(instance.get("baseline_words")) for instance in instances), 3
        ),
        "baseline_lean_tokens": round(
            sum(
                _safe_float(instance.get("baseline_lean_tokens"))
                for instance in instances
            ),
            3,
        ),
        "instances": {instance["instance_id"]: instance for instance in instances},
        "aggregate": aggregate,
        "aggregate_valid": aggregate_valid,
        "aggregate_policy": "equal per-repository spend fraction (parallel portfolio)",
    }
