import concurrent.futures
import math
import os
import re
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
from pathlib import Path
from leanlean.utils import json_utils as json
import logging
import subprocess
import threading
import time
import traceback
from typing import Any, Callable
from copy import deepcopy

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from tqdm import tqdm
import fire

from leanlean import Instance
from leanlean.capture_disposition import (
    DISCARDED_API_FAILURE,
    capture_is_scoring,
    record_capture_disposition,
)

from leanlean.generators.cli_agent import RetryableAPIError
from leanlean.model import get_model
from leanlean.model.costs import (
    cost_from_responses,
    cost_accounting_from_responses,
)
from leanlean.planners import get_planner
from leanlean.generators import get_generator
from leanlean.benchmarks import get_benchmark
from leanlean.run_action_summary import write_run_action_summary
from leanlean.run_admission import admission_held, require_admission
from leanlean.repo_variants import repo_variant_from_env
from leanlean.subscription_status import QUOTA_FAILURE, classify_trajectory
from leanlean.utils.log import add_file_handler, logger
from leanlean.utils.io_utils import save_traj, _compute_run_directory

from configs import (
    ALL_GENERATOR_CONFIGS,
    ALL_MODEL_CONFIGS,
    ALL_PLAN_CONFIGS,
    ALL_BENCHMARK_CONFIGS,
    ALL_TASK_VARIANTS,
    is_plan_training_sequential
)

_OUTPUT_FILE_LOCK = threading.Lock()
_INSTANCE_METRICS_LOCK = threading.Lock()

MAX_RETRIES = 10
COST_UPDATE_INTERVAL_S = 15.0
COMPRESSION_UPDATE_INTERVAL_S = 60.0

# Advisory dollar budget the agent is told it may spend refactoring a repo across
# rounds. It is NOT enforced against the model mid-run (we want to observe whether
# the model respects it), but the refactor loop uses it as a soft backstop: once
# cumulative cost passes the budget, no further rounds are started.
REFACTOR_BUDGET_DEFAULT = 10.0
# Sentinel the agent is asked to emit when it judges no further refactoring is
# worthwhile; seeing it in the agent's final text ends the refactor loop.
SUBMIT_SENTINEL = "<submit>"
_AGENT_TOOL_USAGE_PATH = (
    "/testbed/.git/leanlean-agent-tool-usage.jsonl"
)
_AGENT_TOOL_NAMES = ("proof_length", "lean_verify")


def _agent_tool_usage(env: Any) -> dict[str, Any]:
    """Parse trusted wrapper events stored outside the submitted worktree."""

    raw = env.read_file(_AGENT_TOOL_USAGE_PATH)
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(event, dict)
            and event.get("schema_version") == 1
            and event.get("tool") in _AGENT_TOOL_NAMES
            and event.get("event") in {"started", "finished"}
            and isinstance(event.get("invocation_id"), str)
            and event.get("invocation_id")
            and isinstance(event.get("time_ns"), int)
        ):
            events.append(event)

    tools: dict[str, dict[str, int]] = {}
    for tool in _AGENT_TOOL_NAMES:
        starts = {
            event["invocation_id"]
            for event in events
            if event["tool"] == tool and event["event"] == "started"
        }
        finishes = {
            event["invocation_id"]: event
            for event in events
            if event["tool"] == tool and event["event"] == "finished"
        }
        completed = starts & finishes.keys()
        succeeded = sum(
            int(finishes[invocation].get("exit_code", 1)) == 0
            for invocation in completed
        )
        tools[tool] = {
            "calls": len(starts),
            "completed": len(completed),
            "succeeded": succeeded,
            "failed": len(completed) - succeeded,
            "incomplete": len(starts - finishes.keys()),
            "duration_ms": sum(
                max(0, int(finishes[invocation].get("duration_ms", 0)))
                for invocation in completed
            ),
        }
    raw_setup_events = getattr(env, "_leanlean_setup_tool_usage", [])
    setup_events = [
        dict(event)
        for event in raw_setup_events
        if isinstance(event, dict)
        and event.get("tool") in _AGENT_TOOL_NAMES
        and event.get("phase") == "setup_prewarm"
        and isinstance(event.get("duration_ms"), int)
    ]
    return {
        "schema_version": 1,
        "kind": "leanlean_agent_tool_usage",
        "tools": tools,
        "setup_events": setup_events,
        "events": sorted(
            events,
            key=lambda event: (
                int(event["time_ns"]),
                str(event["invocation_id"]),
                str(event["event"]),
            ),
        ),
    }


def _persist_agent_tool_usage(env: Any, instance_dir: Path) -> dict[str, Any]:
    usage = _agent_tool_usage(env)
    instance_dir.mkdir(parents=True, exist_ok=True)
    destination = instance_dir / "agent_tool_usage.json"
    temporary = instance_dir / ".agent_tool_usage.json.tmp"
    temporary.write_text(json.dumps(usage, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return usage


class InstanceStartGate:
    """Space model-driven instance starts without occupying another worker."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self, instance_id: str) -> None:
        with self._lock:
            now = self._clock()
            delay = max(self._next_start - now, 0.0)
            if delay:
                logger.info(
                    "Quota cadence: waiting %.0fs before starting %s",
                    delay,
                    instance_id,
                )
                self._sleep(delay)
                now = self._clock()
            self._next_start = now + self.interval_seconds


def claude_native_tools_launch(
    launch: str, tools: list[str], *, enable_subagents: bool
) -> str:
    """Swap Claude Code's default --tools list for a run's agent.native_tools."""
    default_tools = "--tools Bash,Edit,Read,Write "
    binary = "/usr/local/bin/claude "
    if launch.count(default_tools) != 1 or launch.count(binary) != 1:
        raise ValueError("Claude launch command has no unique --tools/binary marker")
    launch = launch.replace(default_tools, f"--tools {','.join(tools)} ")
    if enable_subagents:
        # Built-in subagents such as Explore otherwise default to a smaller
        # model; keep every agent on the run's model.
        launch = launch.replace(binary, "CLAUDE_CODE_SUBAGENT_MODEL={model} " + binary)
    return launch


_SECRET_CONFIG_KEYS = frozenset(
    {"api_key", "auth_token", "oauth_token", "access_token", "refresh_token"}
)


def _redact_config_secrets(value: Any) -> Any:
    """Return a logging-safe copy of a nested experiment configuration."""
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if str(key).lower() in _SECRET_CONFIG_KEYS and item
                else _redact_config_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config_secrets(item) for item in value]
    return value


def _refactor_round_note(rounds_done: int) -> str:
    """Framing for a subsequent refactor round. Reports how many rounds of
    refactoring the agent has already performed and asks it to emit the submit
    sentinel if it is done. The dollar budget is still used as an internal
    backstop (see REFACTOR_BUDGET_DEFAULT), but is not disclosed to the agent."""
    return (
        f"\n\nYou have performed {rounds_done} round(s) of refactoring so far. "
        f"If there is no refactoring to be done, output {SUBMIT_SENTINEL} and "
        f"nothing else."
    )


def _message_text(message: Any) -> str:
    """Best-effort concatenation of the textual content of a chat message,
    tolerating both string content and Anthropic/OpenAI block-list content."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _agent_submitted(generator: Any) -> bool:
    """True if the agent emitted the submit sentinel in its final assistant text.

    generator.run() appends the captured diff as a trailing user message, so we
    scan assistant messages (skipping user/tool turns, whose text may echo the
    prompt's own mention of the sentinel)."""
    for message in reversed(getattr(generator, "messages", []) or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        if SUBMIT_SENTINEL in _message_text(message):
            return True
    return False

# Global instance metrics used by the rich dashboard.
INSTANCE_METRICS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_INSTANCE_METRICS_MAX_ROWS: int | None = None

def _format_cost(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.4f}"
    return "-"

def _format_ratio(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"

# Extend these mappings to add new metrics to the rich dashboard.
INSTANCE_METRIC_COLUMNS = ["status", "round", "phase", "cost", "round_cost", "compression", "runtime"]
INSTANCE_METRIC_LABELS = {
    "status": "Status",
    "round": "Round",
    "phase": "Phase",
    "cost": "Total ($)",
    "round_cost": "Round ($)",
    "compression": "Compress",
    "runtime": "Runtime",
}
INSTANCE_METRIC_JUSTIFY = {"cost": "right", "round_cost": "right", "compression": "right", "runtime": "right"}
INSTANCE_METRIC_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "cost": _format_cost,
    "compression": _format_ratio,
}
# Subscription workers share ONE model proxy, so every in-flight worker's
# get_cost() returns the same run-global cumulative rather than its own spend.
# Summing that per worker overstated the run total by a factor of N.
_SHARED_COST_SOURCE = False
_SHARED_COST_PROVIDER = None


def set_shared_cost_source(shared: bool, cost_provider=None) -> None:
    """Record that live cost comes from one proxy shared by every worker."""

    global _SHARED_COST_SOURCE, _SHARED_COST_PROVIDER
    _SHARED_COST_SOURCE = bool(shared)
    _SHARED_COST_PROVIDER = cost_provider


INSTANCE_METRIC_DEFAULTS = {
    "status": "Queued",
    "round": "-",
    "phase": "-",
    "cost": 0.0,
    "compression": None,
    "cost_at_round_start": 0.0,
    "completed_rounds_cost": 0.0,  # cumulative cost from all completed rounds
    "done": False,
    "started_at": None,
    "ended_at": None,
}

def set_instance_metrics_max_rows(max_rows: int | None) -> None:
    global _INSTANCE_METRICS_MAX_ROWS
    _INSTANCE_METRICS_MAX_ROWS = max_rows

def reset_instance_metrics() -> None:
    with _INSTANCE_METRICS_LOCK:
        INSTANCE_METRICS.clear()

def _select_instance_metrics_locked() -> list[tuple[str, dict[str, Any]]]:
    rows = list(INSTANCE_METRICS.items())
    max_rows = _INSTANCE_METRICS_MAX_ROWS
    if max_rows is None or len(rows) <= max_rows:
        return [(instance_id, dict(metrics)) for instance_id, metrics in rows]

    running_ids = [instance_id for instance_id, metrics in rows if not metrics.get("done")]
    if len(running_ids) >= max_rows:
        keep_ids = set(running_ids[-max_rows:])
    else:
        done_ids = [instance_id for instance_id, metrics in rows if metrics.get("done")]
        keep_ids = set(running_ids)
        remaining = max_rows - len(keep_ids)
        if remaining > 0:
            keep_ids.update(done_ids[-remaining:])

    return [
        (instance_id, dict(metrics))
        for instance_id, metrics in rows
        if instance_id in keep_ids
    ]

def _measure_lean_words(instance: Any, env: Any) -> int | None:
    """Best-effort live (non-comment) word count of the .lean sources in the
    container. Returns None for non-leanlean instances or on failure."""
    if not hasattr(instance, "exclude_dirs"):
        return None
    try:
        from leanlean.benchmarks.leanlean import _measure_words
        return _measure_words(
            env,
            exclude_dirs=getattr(instance, "exclude_dirs", None),
            include_prefix=getattr(instance, "_metric_include_prefix", ""),
            exclude_files=getattr(instance, "metric_exclude_files", None),
        )
    except Exception:
        logger.debug("live word count failed", exc_info=True)
        return None

def _update_compression(instance: Any, env: Any, baseline_words: int | None) -> None:
    """Measure current word count and update the live compression ratio."""
    if not baseline_words or baseline_words <= 0:
        return
    post = _measure_lean_words(instance, env)
    # _measure_words returns -1 (not None) when the word count can't be parsed.
    # Without this guard that sentinel leaks into the ratio as round(-1/baseline)
    # ≈ -0.0, which the dashboard renders as a bogus "-0.000" compression.
    if post is None or post < 0:
        return
    update_instance_metrics(
        instance.instance_id, compression=round(post / baseline_words, 4)
    )

def update_instance_metrics(instance_id: str, **updates: Any) -> None:
    with _INSTANCE_METRICS_LOCK:
        metrics = INSTANCE_METRICS.setdefault(instance_id, dict(INSTANCE_METRIC_DEFAULTS))
        metrics.update(updates)

def get_instance_metrics_snapshot() -> list[tuple[str, dict[str, Any]]]:
    with _INSTANCE_METRICS_LOCK:
        return _select_instance_metrics_locked()

def get_instance_metrics_stats() -> tuple[int, float]:
    with _INSTANCE_METRICS_LOCK:
        running = 0
        summed = 0.0
        attributed = 0.0
        shared_cumulative = 0.0
        for metrics in INSTANCE_METRICS.values():
            completed = metrics.get("completed_rounds_cost") or 0.0
            current = metrics.get("cost") or 0.0
            total = completed + current
            summed += total
            if metrics.get("done"):
                # A finished instance has its own attributable cost, taken from
                # its trajectory rather than the shared proxy journal.
                attributed += total
            else:
                running += 1
                shared_cumulative = max(shared_cumulative, total)
        if _SHARED_COST_SOURCE:
            # Each in-flight worker reports the same run-global cumulative, and
            # that figure already includes every finished instance -- so the run
            # total is that cumulative counted once, never the sum.
            if _SHARED_COST_PROVIDER is not None:
                # The owner reads the run journal once; workers read only their scope.
                return running, max(float(_SHARED_COST_PROVIDER()), attributed)
            return running, max(shared_cumulative, attributed)
        return running, summed

def _get_instance_cost(model: Any, generator: Any | None) -> float | None:
    if generator is not None and getattr(generator, "model", None) is not None:
        return getattr(generator.model, "cost", None)
    return getattr(model, "cost", None)

def _current_round_cost(
    instance_id: str,
    model: Any,
    cumulative_or_round_cost: float | None,
) -> float | None:
    """Normalize provider cost into the dashboard's current-round invariant.

    Proxy-backed models expose the live trace's current-round cost. Native
    subscription generators have no trace file, so their fallback get_cost()
    returns the cumulative cost of every durable response. Subtract the amount
    already banked between rounds or the dashboard counts completed work twice.
    """
    if cumulative_or_round_cost is None:
        return None
    if not getattr(model, "cost_from_responses_fallback", False):
        return cumulative_or_round_cost
    with _INSTANCE_METRICS_LOCK:
        metrics = INSTANCE_METRICS.get(instance_id, {})
        completed = metrics.get("completed_rounds_cost") or 0.0
    return max(0.0, cumulative_or_round_cost - completed)


def _refresh_instance_cost(
    instance_id: str,
    model: Any,
    generator: Any | None,
) -> float | None:
    # get_cost() reads the current trace file, which delete_traces() clears at the
    # start of every run(). A no-op round makes zero model calls, so no trace is
    # written and get_cost() correctly returns 0.0 — do NOT guard on the file
    # existing, or we fall back to the stale cached model.cost from a prior round.
    get_cost = getattr(model, "get_cost", None)
    if callable(get_cost):
        try:
            raw_cost = get_cost()
            return _current_round_cost(instance_id, model, raw_cost)
        except Exception:
            pass
    raw_cost = _get_instance_cost(model, generator)
    return _current_round_cost(instance_id, model, raw_cost)

def _cumulative_cost(model: Any) -> float:
    """Total $ cost of every response processed so far. model.responses
    accumulates across all runs and — unlike the trace file — survives the
    delete_traces() that generator.run() performs, so it's the durable source
    of truth for completed-round cost."""
    return cost_from_responses(list(getattr(model, "responses", [])))


def _bank_round_cost(instance_id: str, model: Any, generator: Any | None) -> None:
    """Fold the prior run's cost into the cumulative total before the trace is
    reset.

    Cost lives in a single trace file that delete_traces() wipes at the start of
    every generator.run() (each gate retry, API retry, and refactor round) — so
    reading it back after a run yields 0. Instead we recompute the absolute
    cumulative cost from model.responses (which persists across the reset). Call
    this immediately BEFORE any run(): at that point model.responses holds every
    completed run's responses and none of the in-flight one's, so it equals the
    completed-rounds total. The live "cost" field is zeroed to match the fresh
    trace, keeping the displayed total (completed_rounds_cost + cost) continuous."""
    completed = _cumulative_cost(model)
    with _INSTANCE_METRICS_LOCK:
        m = INSTANCE_METRICS.setdefault(instance_id, dict(INSTANCE_METRIC_DEFAULTS))
        m["completed_rounds_cost"] = completed
        m["cost"] = 0.0
        m["cost_at_round_start"] = 0.0

def _start_metrics_updater(
    instance_id: str,
    instance: Any,
    model: Any,
    get_generator: Callable[[], Any | None],
    get_env: Callable[[], Any | None],
    get_baseline: Callable[[], int | None],
    stop_event: threading.Event,
) -> threading.Thread:
    def _loop() -> None:
        # Refresh cost every tick (cheap, local trace read). Refresh the live
        # compression ratio less often (it's a docker exec word count over the
        # .lean sources) — once a minute is plenty.
        comp_every = max(1, round(COMPRESSION_UPDATE_INTERVAL_S / COST_UPDATE_INTERVAL_S))
        ticks = 0
        while not stop_event.wait(COST_UPDATE_INTERVAL_S):
            cost = _refresh_instance_cost(instance_id, model, get_generator())
            update_instance_metrics(instance_id, cost=cost)
            ticks += 1
            if ticks % comp_every == 0:
                env = get_env()
                if env is not None:
                    _update_compression(instance, env, get_baseline())

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread

def _format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _format_total_cost(value: float) -> str:
    return f"${value:07.2f}"

def _format_runtime(metrics: dict[str, Any]) -> str:
    started_at = metrics.get("started_at")
    if not isinstance(started_at, (int, float)):
        return "-"
    ended_at = metrics.get("ended_at")
    if isinstance(ended_at, (int, float)):
        elapsed = ended_at - started_at
    else:
        elapsed = time.monotonic() - started_at
    if elapsed < 0:
        elapsed = 0
    return _format_elapsed(elapsed)

def _format_metric_value(key: str, metrics: dict[str, Any]) -> str:
    if key == "runtime":
        return _format_runtime(metrics)
    if (
        key in {"cost", "round_cost"}
        and _SHARED_COST_SOURCE
        and not metrics.get("done")
    ):
        # Only the run-global cumulative is observable mid-flight under a shared
        # proxy; printing it here reads as this instance's spend and is wrong.
        return "shared"
    if key == "round_cost":
        # cost resets per round (delete_traces at run start); show current-round cost only
        current = metrics.get("cost") or 0.0
        offset = metrics.get("cost_at_round_start") or 0.0
        return _format_cost(current - offset)
    if key == "cost":
        # True total = completed rounds + current round
        completed = metrics.get("completed_rounds_cost") or 0.0
        current = metrics.get("cost") or 0.0
        return _format_cost(completed + current)
    formatter = INSTANCE_METRIC_FORMATTERS.get(key)
    value = metrics.get(key)
    if formatter:
        return formatter(value)
    if value is None:
        return "-"
    return str(value)

class RichProgressDashboard:
    def __init__(
        self,
        *,
        total: int,
        workers: int,
        dataset_name: str | None = None,
        model_name: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.total = total
        self.completed = 0
        self.workers = workers
        label = " LeanLean"
        if dataset_name:
            label = f"{label} | {dataset_name}"
        if model_name:
            label = f"{label} | model: {model_name}"
        if agent_name:
            label = f"{label} | agent: {agent_name}"
        self.spinner = Spinner("dots", text=label, style="bold cyan")
        self.start_time = time.monotonic()

    def mark_completed(self) -> None:
        self.completed += 1

    def __rich__(self) -> Panel:
        rows = get_instance_metrics_snapshot()
        running, total_cost = get_instance_metrics_stats()
        queued = max(self.total - self.completed - running, 0)
        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        elapsed = _format_elapsed(time.monotonic() - self.start_time)
        header.add_row(
            self.spinner,
            (
                f"done {self.completed}/{self.total} | running {running} | queued {queued} | "
                f"{elapsed} | cost {_format_total_cost(total_cost)}"
            ),
        )

        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Instance", no_wrap=True)
        for key in INSTANCE_METRIC_COLUMNS:
            table.add_column(
                INSTANCE_METRIC_LABELS.get(key, key),
                justify=INSTANCE_METRIC_JUSTIFY.get(key, "left"),
            )

        if rows:
            for instance_id, metrics in rows:
                row = [instance_id]
                for key in INSTANCE_METRIC_COLUMNS:
                    row.append(_format_metric_value(key, metrics))
                table.add_row(*row)
        else:
            empty_row = ["-"] + ["Waiting"] + ["-"] * (len(INSTANCE_METRIC_COLUMNS) - 1)
            table.add_row(*empty_row)

        filler_rows = max(self.workers - len(rows), 0)
        for _ in range(filler_rows):
            filler = ["-"] + ["Idle"] + ["-"] * (len(INSTANCE_METRIC_COLUMNS) - 1)
            table.add_row(*filler)

        return Panel(Group(header, table), padding=(1, 1), border_style="cyan")

def _attach_rich_console(console: Console) -> None:
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.console = console

def update_preds_file(
    output_path: Path, instance_id: str, model_name: str, result: str
):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        temporary = output_path.with_name(
            f".{output_path.name}.tmp-{os.getpid()}"
        )
        try:
            temporary.write_text(json.dumps(output_data, indent=2))
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)


def _prepared_baseline_commit(env: Any) -> str:
    """Return the exact commit every API retry must restart from."""

    result = env.execute("cd /testbed && git rev-parse HEAD", timeout=30)
    output = str(result.get("output") or "").strip().splitlines()
    commit = output[-1].strip() if output else ""
    if result.get("returncode") or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise RuntimeError("could not pin the prepared repository commit for API retries")
    return commit


def _restore_api_retry_baseline(env: Any, baseline_commit: str) -> None:
    """Discard a failed API attempt before granting a replacement attempt."""

    if not re.fullmatch(r"[0-9a-f]{40,64}", baseline_commit):
        raise ValueError("invalid prepared baseline commit")
    result = env.execute(
        "cd /testbed && "
        f"git reset --hard {baseline_commit} && "
        "git clean -fd --exclude='.lake/'",
        timeout=900,
    )
    if result.get("returncode"):
        raise RuntimeError(
            "could not restore the prepared API-retry baseline: "
            + str(result.get("output") or "")[-4000:]
        )
    restored = _prepared_baseline_commit(env)
    if restored != baseline_commit:
        raise RuntimeError(
            f"API retry restored {restored}, expected {baseline_commit}"
        )


def _discard_failed_api_capture(
    generator: Any,
    error: BaseException,
    *,
    retry_number: int,
) -> Path | None:
    """Keep the raw failed capture but exclude it from benchmark scoring."""

    playback_dir = getattr(generator, "_playback_output_dir", None)
    capture_index = int(getattr(generator, "_playback_capture_index", 0) or 0)
    if playback_dir is None or capture_index < 1:
        return None
    capture_name = f"capture_{capture_index:03d}"
    playback_path = Path(playback_dir) / capture_name / "playback.json"
    if not playback_path.is_file():
        return None
    return record_capture_disposition(
        Path(playback_dir),
        capture_name,
        DISCARDED_API_FAILURE,
        reason={
            "kind": "structured_api_failure",
            "error_type": type(error).__name__,
            "error": str(error)[-2000:],
            "retry_number": retry_number,
            "scoring": "excluded_before_clean_baseline_retry",
        },
    )




def _persist_rescued_prediction(
    output_dir: Path,
    instance_id: str,
    model_name: str,
    rescued_diff: str,
) -> None:
    """Make a best-effort final diff eligible for normal round-zero evaluation."""

    for filename in ("preds.json", "preds_round_0.json"):
        update_preds_file(
            output_dir / filename,
            instance_id,
            model_name,
            rescued_diff,
        )


def _instance_has_existing_output(output_dir: Path, instance_id: str) -> bool:
    instance_dir = output_dir / instance_id
    if not instance_dir.is_dir():
        return False
    return (instance_dir / f"{instance_id}.traj.json").exists()


def _get_existing_exit_status(output_dir: Path, instance_id: str) -> str | None:
    traj_path = output_dir / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return None
    try:
        data = json.loads(traj_path.read_text())
    except Exception:
        logger.warning(
            "Failed to read existing trajectory for %s at %s",
            instance_id,
            traj_path,
        )
        return None

    info = data.get("info", {})
    if not isinstance(info, dict):
        return None
    classified = classify_trajectory(data)
    if classified is not None:
        return (
            "QuotaExceeded"
            if classified[0] == QUOTA_FAILURE
            else "ProviderFailed"
        )
    exit_status = info.get("exit_status")
    return exit_status if isinstance(exit_status, str) and exit_status else None


def _instance_has_prediction(output_dir: Path, instance_id: str) -> bool:
    try:
        predictions = json.loads((output_dir / "preds.json").read_text())
    except (OSError, ValueError, TypeError):
        return False
    record = predictions.get(instance_id) if isinstance(predictions, dict) else None
    return isinstance(record, dict) and isinstance(record.get("model_patch"), str)


def _instance_has_scoring_capture(output_dir: Path, instance_id: str) -> bool:
    playback_dir = output_dir / instance_id / "playback"
    for path in sorted(playback_dir.glob("capture_*/playback.json")):
        try:
            if capture_is_scoring(path):
                return True
        except ValueError:
            return False
    return False


def _archive_retryable_instance(
    output_dir: Path,
    instance_id: str,
    *,
    reason: str,
) -> Path | None:
    """Move one non-scoring attempt aside before a clean same-run retry."""

    instance_dir = output_dir / instance_id
    round_dir = output_dir / "round_0" / instance_id
    if not instance_dir.exists() and not round_dir.exists():
        return None
    attempts = output_dir / "retry-attempts"
    index = 1
    while (attempts / f"attempt_{index:03d}" / instance_id).exists():
        index += 1
    destination = attempts / f"attempt_{index:03d}" / instance_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if instance_dir.exists():
        instance_dir.replace(destination)
    else:
        destination.mkdir()
    if round_dir.exists():
        round_dir.replace(destination / "round_0")
    prediction_records: dict[str, Any] = {}
    for filename in ("preds.json", "preds_round_0.json"):
        path = output_dir / filename
        try:
            predictions = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(predictions, dict) and instance_id in predictions:
            prediction_records[filename] = predictions[instance_id]
    (destination / "retry-receipt.json").write_text(
        json.dumps(
            {
                "format": "leanlean-evaluation-retry-attempt-v1",
                "instance_id": instance_id,
                "reason": reason,
                "prediction_records": prediction_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return destination


def _completed_refactor_round(output_dir: Path, instance_id: str) -> int:
    """Highest refactor round already completed for this run/instance.

    Returns the round number N (0 = only the initial generation completed,
    -1 = nothing completed). Used to make refactor rounds *appendable*: re-running
    the same run_id with a larger refactor_rounds fast-forwards past these.
    """
    if not _instance_has_existing_output(output_dir, instance_id):
        # A 0-round run leaves only preds.json (no per-round files / traj may
        # still exist); treat a present final patch as "round 0 done".
        return 0 if _resume_patch(output_dir, instance_id, 0) is not None else -1
    highest = 0
    for p in output_dir.glob("preds_round_*.json"):
        m = re.match(r"preds_round_(\d+)\.json$", p.name)
        if not m:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if instance_id in data:
            highest = max(highest, int(m.group(1)))
    return highest


def _resume_patch(output_dir: Path, instance_id: str, round_num: int) -> str | None:
    """Cumulative baseline->round patch to fast-forward to `round_num`.

    Prefers the per-round preds file; falls back to the run's final preds.json
    (which always holds the latest cumulative diff)."""
    candidates = [output_dir / f"preds_round_{round_num}.json", output_dir / "preds.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        patch = (data.get(instance_id) or {}).get("model_patch")
        if patch:
            return patch
    return None


def _apply_resume_patch(env: Any, patch: str, round_num: int) -> None:
    """Fast-forward a freshly set-up container to a prior round's state by
    applying its cumulative diff and committing it as the new starting point.

    Resets to the pristine init commit first: the stored cumulative diff is taken
    against /.init_commit (so it already contains AGENTS.md and every prior
    change), and applying it onto the planner's freshly-written tree would
    collide. The .lake build cache is untracked and survives the reset/clean."""
    import shlex
    env.execute(
        "cd /testbed && "
        "init=$(cat /.init_commit 2>/dev/null || git rev-list --max-parents=0 HEAD | tail -1) && "
        "git reset --hard \"$init\" && git clean -fd --exclude='.lake/' && "
        f"echo {shlex.quote(patch)} | git apply --whitespace=nowarn - && "
        "git add -A && git reset HEAD .lake/ 2>/dev/null || true && "
        f"git -c user.email='agent@lean' -c user.name='Agent' "
        f"commit -m 'resume-from-round-{round_num}' --allow-empty"
    )


def _deepcopy_jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return deepcopy(value)


def _normalize_message_type(message: dict[str, Any]) -> str:
    role = message.get("role")
    if isinstance(role, str) and role:
        return role
    msg_type = message.get("type")
    if msg_type == "reasoning":
        return "assistant"
    if msg_type == "function_call":
        return "tool_call"
    if msg_type == "function_call_output":
        return "tool_output"
    return "unknown"


def _build_attempt_event_messages(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_index = attempt.get("attempt_index")
        task = attempt.get("task")
        status = attempt.get("exit_status")
        result = attempt.get("result")
        validation = attempt.get("validation")

        events.append(
            {
                "role": "system",
                "type": "attempt_event",
                "content": (
                    f"[attempt {attempt_index}] start\n"
                    f"task_length={len(task or '')}"
                ),
                "attempt_index": attempt_index,
            }
        )

        for message in attempt.get("messages") or []:
            if isinstance(message, dict):
                normalized = _deepcopy_jsonable(message)
                normalized["attempt_index"] = attempt_index
                normalized["attempt_message_type"] = _normalize_message_type(normalized)
                events.append(normalized)

        events.append(
            {
                "role": "system",
                "type": "attempt_event",
                "content": (
                    f"[attempt {attempt_index}] run_finished\n"
                    f"exit_status={status}\n"
                    f"submission_chars={len(result or '')}"
                ),
                "attempt_index": attempt_index,
            }
        )

        if isinstance(validation, dict):
            command = validation.get("command", "")
            returncode = validation.get("returncode")
            passed = bool(validation.get("passed", False))
            output = validation.get("output", "")
            events.append(
                {
                    "role": "system",
                    "type": "attempt_event",
                    "content": (
                        f"[attempt {attempt_index}] submission_gate_validation\n"
                        f"passed={passed}\n"
                        f"returncode={returncode}\n"
                        f"command={command}\n\n"
                        f"{output}"
                    ),
                    "attempt_index": attempt_index,
                    "submission_gate": _deepcopy_jsonable(validation),
                }
            )
    return events


def _build_attempt_event_responses(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_index = attempt.get("attempt_index")
        for response in attempt.get("responses") or []:
            if isinstance(response, dict):
                normalized = _deepcopy_jsonable(response)
                normalized["attempt_index"] = attempt_index
                responses.append(normalized)
    return responses


def _build_attempt_playback(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    captured = []
    for attempt in attempts:
        playback = attempt.get("playback")
        if isinstance(playback, dict) and playback.get("points"):
            captured.append(
                {
                    **_deepcopy_jsonable(playback),
                    "attempt_index": attempt.get("attempt_index"),
                }
            )
    if len(captured) == 1:
        return captured[0]
    if captured:
        return {
            "format": "leanlean-playback-v1",
            "quality": "multiple_attempts",
            "attempts": captured,
        }
    return {}


def _build_attempt_native_rollouts(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for attempt in attempts:
        rollout = attempt.get("native_rollout")
        if isinstance(rollout, dict) and rollout:
            captured.append(
                {
                    **_deepcopy_jsonable(rollout),
                    "attempt_index": attempt.get("attempt_index"),
                }
            )
    return captured


def _build_attempt_provider_traces(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for attempt in attempts:
        trace = attempt.get("provider_trace")
        if isinstance(trace, dict) and trace:
            captured.append(
                {
                    **_deepcopy_jsonable(trace),
                    "attempt_index": attempt.get("attempt_index"),
                }
            )
    return captured


# Runtime parameters substituted into the task prompt just before the agent
# starts. CPU and memory limits come from the same instance settings used for Docker. The wall-clock budget is the container's own exec timeout, so the
# prompt can never claim a limit that has drifted from `container.timeout`, and
# the timestamps are UTC because that is what `date` reports inside the
# container.
CLOCK_PARAMETERS = ("{time_limit}", "{current_time}", "{deadline}")
TASK_PARAMETERS = (*CLOCK_PARAMETERS, "{container_cpus}", "{container_memory}")


def _timeout_seconds(timeout: str) -> int | None:
    """Parse a container timeout ('2h', '300m', '18000s') the way Docker does."""
    text = timeout.strip()
    multiplier = {"h": 3600, "m": 60, "s": 1}.get(text[-1:], 1)
    digits = text[:-1] if multiplier != 1 else text
    return int(digits) * multiplier if digits.isdigit() else None


def _render_task_parameters(task: str, instance: Instance) -> str:
    """Bind time and resource placeholders to the resolved container settings."""
    if "{container_cpus}" in task:
        cpus = getattr(instance, "container_cpus", None)
        if isinstance(cpus, bool) or not isinstance(cpus, (int, float)) or not math.isfinite(cpus) or cpus <= 0:
            raise ValueError("task prompt requires a positive container_cpus limit")
        task = task.replace("{container_cpus}", f"{cpus:g}")
    if "{container_memory}" in task:
        memory = str(getattr(instance, "container_memory", "") or "").strip()
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([bkmgt]?)", memory, re.IGNORECASE)
        if not match or float(match[1]) <= 0:
            raise ValueError("task prompt requires a positive container_memory limit")
        unit = {"": "bytes", "b": "bytes", "k": "KiB", "m": "MiB", "g": "GiB", "t": "TiB"}[match[2].lower()]
        task = task.replace("{container_memory}", f"{match[1]} {unit}")
    if not any(token in task for token in CLOCK_PARAMETERS):
        return task
    timeout = str(getattr(instance, "container_timeout", "") or "")
    seconds = _timeout_seconds(timeout)
    if seconds is None:
        raise ValueError(
            f"task prompt asks for a time budget but container_timeout "
            f"{timeout!r} is unparseable"
        )
    now = datetime.now(timezone.utc)
    stamp = "%Y-%m-%d %H:%M:%S UTC"
    return (
        task.replace("{time_limit}", timeout)
        .replace("{current_time}", now.strftime(stamp))
        .replace("{deadline}", (now + timedelta(seconds=seconds)).strftime(stamp))
    )


def process_instance(
    instance: Instance,
    output_dir: Path,
    config: dict,
    train_plan: bool = False,
    n_tries: int = 1,
    setup_repo: bool = True,
    hard_task: bool = False,
    remove_docs: bool = False,
) -> None:
    """Process a single LeanLean instance."""

    if hard_task:
        instance.harden_task()

    instance_dir = instance.get_dir(output_dir)

    # Persist the task prompt, so each run records what was actually asked of
    # the model (the prompt can change between runs). Clock placeholders are
    # still unbound here; the per-instance copy is rewritten with the rendered
    # text once the agent starts, while the run-level copy keeps this template
    # because instances start at different times.
    try:
        task_variant_template = config.get("task_variant", "{task}")
        prompt_text = task_variant_template.format(task=instance.task)
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "prompt.txt").write_text(prompt_text)
        # Run-level convenience copy (identical across instances for
        # leanlean); first writer wins to avoid concurrent clobbering.
        run_prompt = output_dir / "prompt.txt"
        if not run_prompt.exists():
            run_prompt.write_text(prompt_text)
    except Exception as e:  # never let prompt logging break a run
        logger.warning("Failed to persist task prompt for %s: %s", instance.instance_id, e)

    model = get_model(config=config.get("model", {}))
    planner = get_planner(planner_config=config.get("planner", {}))

    generator = None
    env = None
    exit_status = "NotStarted"
    result = ""
    extra_info = {}
    # Set True once the happy-path persistence (traj + preds + cache commit)
    # has run, so the finally-block safety net doesn't redo it.
    final_capture_done = False

    start_time = time.monotonic()
    update_instance_metrics(
        instance.instance_id,
        status="Starting",
        cost=0.0,
        done=False,
        started_at=start_time,
        ended_at=None,
    )
    # Pre-declared so the background metrics updater's getters are safe to call
    # before the retry loop below assigns them.
    env: Any = None
    baseline_words: int | None = None
    cost_stop_event = threading.Event()
    cost_thread = _start_metrics_updater(
        instance.instance_id,
        instance,
        model,
        lambda: generator,
        lambda: env,
        lambda: baseline_words,
        cost_stop_event,
    )

    try:
        for _ in range(n_tries):
            # Load the environment. Pass the refactor-round count through so the
            # container's lifetime covers every round that reuses it (each round
            # gets its own exec timeout); otherwise the container is removed
            # mid-run on later rounds and their captured diffs are corrupted.
            env_config = dict(config.get("environment", {}))
            env_config["refactor_rounds"] = int(
                config.get("planner", {}).get("refactor_rounds", 0)
            )
            env = instance.setup(env_config=env_config, setup_repo=setup_repo)

            # Exectute the plan (i.e., write AGENTS.md)
            update_instance_metrics(
                instance.instance_id,
                status="Planning",
                cost=_refresh_instance_cost(instance.instance_id, model, generator),
                done=False,
            )
            planner.plan(env=env, model=model, instance=instance)

            if remove_docs:
                instance.remove_docs(env)

            # Reset git history to prevent benchmark-state leakage.
            if hasattr(instance, "_clean_git_history"):
                update_instance_metrics(
                    instance.instance_id,
                    status="Cleaning Git History",
                    cost=_refresh_instance_cost(instance.instance_id, model, generator),
                    done=False,
                )
                instance._clean_git_history(env)

            # Commit all changes so far to have a clean state
            status_result = env.execute("git status --short 2>&1 | head -50 && echo '---' && cat .gitignore 2>/dev/null | grep -E '^/?target' || echo '(target not in .gitignore)' && echo '---' && du -sh * .* 2>/dev/null | sort -rh | head -20", timeout=30)
            logger.info("Pre-git-add state:\n%s", status_result.get("output", ""))
            env.execute("git add . && git reset HEAD .lake/ 2>/dev/null || true")
            env.execute('git commit -m "Prepare environment for execution"')
            round0_retry_baseline_commit = _prepared_baseline_commit(env)

            # Baseline word count for the live compression-ratio display.
            baseline_words = _measure_lean_words(instance, env)

            # Solve the instance itself
            generator = get_generator(
                generator_config=config.get("generator", {}), model=model, env=env
            )
            # File-level mode (leanlean): the agent may only edit one file;
            # the generator scopes its captured diff to it. No-op for repo-level.
            generator._target_file = getattr(instance, "target_file", "")
            # Subproject mode: scope the diff to a directory.
            generator._target_dir = getattr(instance, "target_dir", "")
            generator._playback_output_dir = instance_dir / "playback"
            generator._playback_instance_id = instance.instance_id
            generator._playback_build_command = getattr(instance, "build_command", "") or ""
            # Exact edit-level playback is opt-in and provider-neutral.
            generator.playback_config = config.get("playback", {}) or {}
            # Native Codex rollouts retain full edit tool inputs. Artifacts live
            # outside /testbed and are exported before the temporary auth home
            # is securely removed.
            generator._native_rollout_output_dir = instance_dir / "native_rollout"
            generator._native_rollout_reference_prefix = "native_rollout"
            generator.trajectory_config = config.get("trajectory", {}) or {}
            generator._playback_exclude_dirs = tuple(
                getattr(instance, "exclude_dirs", ()) or ()
            )
            gate_cfg = instance.get_submission_gate() or {}
            gate_enabled = bool(gate_cfg.get("enabled", False))
            gate_max_retries = int(gate_cfg.get("max_retries", 0))
            running_status = (
                f"Running (0/{gate_max_retries})" if gate_enabled else "Running"
            )
            update_instance_metrics(
                instance.instance_id,
                status=running_status,
                round="0",
                phase="Generating",
                cost=_refresh_instance_cost(instance.instance_id, model, generator),
                cost_at_round_start=0.0,  # generator.run() calls delete_traces() at start
                done=False,
            )

            # Bind the clock placeholders now: the container is up and the
            # agent's exec timeout starts with the next run() call, so this is
            # the earliest point where "now" and the deadline are truthful.
            rendered_task = _render_task_parameters(instance.task, instance)
            try:
                (instance_dir / "prompt.txt").write_text(
                    config.get("task_variant", "{task}").format(task=rendered_task)
                )
            except Exception as e:  # never let prompt logging break a run
                logger.warning(
                    "Failed to persist rendered task prompt for %s: %s",
                    instance.instance_id, e,
                )

            # ── appendable refactor rounds ───────────────────────────────────
            # If this run/instance already completed some rounds and more are
            # requested, fast-forward the fresh container to the latest completed
            # round and continue the refactor loop from there (skip re-generating
            # round 0 and the rounds already done).
            base_task = config.get("task_variant", "{task}").format(task=rendered_task)
            refactor_budget = float(
                config.get("planner", {}).get("refactor_budget", REFACTOR_BUDGET_DEFAULT)
            )
            _completed = _completed_refactor_round(output_dir, instance.instance_id)
            _resume_p = _resume_patch(output_dir, instance.instance_id, _completed) if _completed >= 0 else None
            resuming = (
                _completed >= 0 and _resume_p is not None
                and _instance_has_existing_output(output_dir, instance.instance_id)
            )
            start_round = _completed if resuming else 0
            if resuming:
                logger.info(
                    "Fast-forwarding %s to round %d, then continuing",
                    instance.instance_id, _completed,
                )
                _apply_resume_patch(env, _resume_p, _completed)
                result = _resume_p
                exit_status = "Resumed"
                attempt_history = []
                # Backfill preds_round_0.json if the original run had 0 rounds, so
                # per-round eval has the baseline round.
                if _completed == 0 and not (output_dir / "preds_round_0.json").exists():
                    update_preds_file(
                        output_dir / "preds_round_0.json",
                        instance.instance_id, model.config.model_name, _resume_p,
                    )

            retry = not resuming
            n_retries = 0
            max_api_retries = int(config.get("max_api_retries", MAX_RETRIES))
            round0_start_time = time.monotonic()
            while retry:
                require_admission()
                try:
                    gate_attempt = 0
                    gate_feedback = ""
                    gate_history: list[dict[str, Any]] = []
                    attempt_history: list[dict[str, Any]] = []
                    previous_response_count = 0
                    rejected_by_gate = False

                    task_variant_template = config.get("task_variant", "{task}")
                    while True:
                        base_task = task_variant_template.format(task=rendered_task)
                        # Round 0 uses the task prompt as-is; the agent is not
                        # told the dollar budget (it is an internal backstop only).
                        round0_task = base_task
                        if gate_feedback:
                            task = f"{round0_task}\n\n{gate_feedback}"
                        else:
                            task = round0_task
                        attempt_index = len(attempt_history) + 1
                        # Preserve the prior run's cost before run() wipes the trace.
                        # First pass banks any planning cost; later passes bank each
                        # rejected gate attempt so the total survives resubmission.
                        _bank_round_cost(instance.instance_id, model, generator)
                        exit_status, result = generator.run(task)
                        raw_responses = list(getattr(generator.model, "responses", []))
                        if previous_response_count <= len(raw_responses):
                            attempt_responses = raw_responses[previous_response_count:]
                        else:
                            attempt_responses = raw_responses
                        previous_response_count = len(raw_responses)
                        attempt_record: dict[str, Any] = {
                            "attempt_index": attempt_index,
                            "task": task,
                            "exit_status": exit_status,
                            "result": result,
                            "messages": _deepcopy_jsonable(getattr(generator, "messages", [])),
                            "responses": _deepcopy_jsonable(attempt_responses),
                            "playback": _deepcopy_jsonable(getattr(generator, "playback", {})),
                            "native_rollout": _deepcopy_jsonable(
                                getattr(generator, "native_rollout", {})
                            ),
                            "provider_trace": _deepcopy_jsonable(
                                getattr(generator, "provider_trace", {})
                            ),
                            "validation": None,
                        }
                        if not gate_enabled:
                            attempt_history.append(attempt_record)
                            break

                        validation_result = instance.validate_submission(env)
                        attempt_record["validation"] = _deepcopy_jsonable(validation_result)
                        attempt_history.append(attempt_record)
                        gate_history.append(validation_result)
                        if validation_result.get("passed", False):
                            break

                        if gate_attempt >= gate_max_retries:
                            rejected_by_gate = True
                            break

                        gate_attempt += 1
                        update_instance_metrics(
                            instance.instance_id,
                            status=f"Submission gate retry {gate_attempt}/{gate_max_retries}",
                            cost=_refresh_instance_cost(
                                instance.instance_id, model, generator
                            ),
                            done=False,
                        )
                        gate_feedback = instance.build_submission_feedback(validation_result)

                    if gate_history:
                        extra_info["submission_gate"] = {
                            "enabled": gate_enabled,
                            "max_retries": gate_max_retries,
                            "attempts": _deepcopy_jsonable(gate_history),
                        }
                    if attempt_history:
                        extra_info["attempts"] = _deepcopy_jsonable(attempt_history)
                    if rejected_by_gate:
                        exit_status = "RejectedBySubmissionGate"
                        result = ""
                    retry = False
                except RetryableAPIError as api_err:
                    logger.error(f"API error encountered: {api_err}")
                    retry_number = n_retries + 1
                    discarded_capture = _discard_failed_api_capture(
                        generator,
                        api_err,
                        retry_number=retry_number,
                    )
                    if discarded_capture is not None:
                        logger.info(
                            "Excluded failed API capture from scoring via %s",
                            discarded_capture,
                        )
                    # Bank the failed attempt's partial cost before wiping the trace.
                    _bank_round_cost(instance.instance_id, model, generator)
                    model.delete_traces()
                    retry = True
                    n_retries = retry_number
                    retry_status = (
                        "API error; retries disabled"
                        if max_api_retries == 0
                        else f"Retry {n_retries}/{max_api_retries}"
                    )
                    update_instance_metrics(
                        instance.instance_id,
                        status=retry_status,
                        cost=_refresh_instance_cost(
                            instance.instance_id, model, generator
                        ),
                        done=False,
                    )
                    if n_retries >= max_api_retries:
                        raise api_err
                    _restore_api_retry_baseline(
                        env,
                        round0_retry_baseline_commit,
                    )
                    logger.info(
                        f"Retrying... Attempt {n_retries+1}/{max_api_retries}"
                    )

            # Optional "refactor more" rounds: run the agent again on the same
            # container (state preserved) with the current diff as context.
            # After each round: lake build + git commit so per-round progress is trackable.
            # Update the live compression ratio with the round-0 result.
            _update_compression(instance, env, baseline_words)
            # Fold round 0's cost into the cumulative total now that the run is
            # done. generator.run() wiped the trace (so get_cost() reads 0) and
            # the per-round bank only happens *before* each refactor round — with
            # refactor_rounds=0 there is no later bank, so without this the live
            # "Total ($)" stays at $0 from submit until the final update lands.
            _bank_round_cost(instance.instance_id, model, generator)
            refactor_rounds = int(config.get("planner", {}).get("refactor_rounds", 0))

            # Persist round 0 with the exact same artefacts as every refactor round
            # (preds, traj, cumulative + delta patch, gen_metrics) so all runs share
            # one file structure regardless of how many refactor rounds run.
            # Skip when resuming: round 0 (and the fast-forwarded rounds) already exist.
            if not resuming:
                instance_dir.mkdir(parents=True, exist_ok=True)
                round0_wall_time = round(time.monotonic() - round0_start_time, 2)
                round0_responses = [
                    r for a in attempt_history for r in a.get("responses", [])
                ]
                round0_llm_cost = cost_from_responses(round0_responses)
                update_preds_file(
                    output_dir / "preds_round_0.json",
                    instance.instance_id,
                    model.config.model_name,
                    result,
                )
                # Round 0's cumulative diff (baseline → round 0) is the generation
                # result, and its delta is identical (nothing precedes it).
                if result.strip():
                    (instance_dir / "round_0_patch.diff").write_text(result)
                    (instance_dir / "round_0_delta.diff").write_text(result)
                # Save round-0 traj so evaluate can compute per-round stats.
                if attempt_history:
                    save_traj(
                        generator=None,
                        path=output_dir / "round_0" / instance.instance_id / f"{instance.instance_id}.traj.json",
                        exit_status=exit_status,
                        result=result,
                        messages=_build_attempt_event_messages(attempt_history),
                        responses=_build_attempt_event_responses(attempt_history),
                        trajectory_format="leanlean-chronological-attempts-v1",
                        instance_id=instance.instance_id,
                        playback=_build_attempt_playback(attempt_history),
                        native_rollouts=_build_attempt_native_rollouts(
                            attempt_history
                        ),
                        provider_traces=_build_attempt_provider_traces(
                            attempt_history
                        ),
                    )
                # Per-round generation metrics (cost, wall time) — same shape as the
                # refactor-round gen_metrics so the viewer reads round 0 identically.
                gen_metrics = {
                    "round": 0,
                    "llm_cost": round0_llm_cost,
                    "cost_accounting": cost_accounting_from_responses(round0_responses),
                    "wall_time_seconds": round0_wall_time,
                    "exit_status": exit_status,
                }
                (instance_dir / "round_0_gen_metrics.json").write_text(
                    json.dumps(gen_metrics, indent=2)
                )

            # Track response boundary so each refactor round gets its own traj slice.
            # Note: generator.messages is reset to [] on each run(), so no msg slicing needed.
            prev_resp_count = len(getattr(generator.model, "responses", []))

            for refactor_round in range(start_round, refactor_rounds):
                # Soft budget backstop: once cumulative spend passes the advisory
                # budget, stop starting new rounds. The budget is never enforced
                # mid-round — we let the model overshoot within a round and only
                # gate on it here, between rounds.
                spent = _cumulative_cost(model)
                if spent >= refactor_budget:
                    logger.info(
                        "Budget reached for %s: $%.4f spent >= $%.2f budget — "
                        "stopping refactor loop after %d round(s).",
                        instance.instance_id, spent, refactor_budget, refactor_round,
                    )
                    break
                # Bank the previous round's cost before generator.run() wipes the trace.
                _bank_round_cost(instance.instance_id, model, generator)
                update_instance_metrics(
                    instance.instance_id,
                    round=str(refactor_round + 1),
                    phase="Generating",
                )
                # rounds_done counts every round already completed, including
                # round 0 (the initial generation), so the first refactor round
                # reports "1 round performed".
                rounds_done = refactor_round + 1
                refactor_task = base_task + _refactor_round_note(rounds_done)
                logger.info(
                    "Refactor round %d/%d for %s ($%.4f/$%.2f spent)",
                    refactor_round + 1, refactor_rounds, instance.instance_id,
                    spent, refactor_budget,
                )
                round_start_time = time.monotonic()
                exit_status_r, result_r = generator.run(refactor_task)
                # If the agent judged there was nothing left to do, it emits the
                # submit sentinel and makes no edits. Stop before persisting a
                # no-op round; the submit call's cost is folded into the final
                # cumulative total below.
                if _agent_submitted(generator):
                    logger.info(
                        "Agent emitted %s at refactor round %d for %s — stopping.",
                        SUBMIT_SENTINEL, refactor_round + 1, instance.instance_id,
                    )
                    break
                round_wall_time = round(time.monotonic() - round_start_time, 2)
                # Cost of just this round. Don't re-read the trace: generator.run()
                # -> _post_exec() already deleted it, so get_cost() would return 0.
                # model.responses survives and carries per-call usage; this round's
                # responses are everything appended past the prior boundary.
                round_responses = list(getattr(generator.model, "responses", []))[prev_resp_count:]
                round_llm_cost = cost_from_responses(round_responses)
                logger.info(
                    "Refactor round %d/%d done: %s (cost=$%.4f, time=%.1fs)",
                    refactor_round + 1, refactor_rounds, exit_status_r, round_llm_cost, round_wall_time,
                )

                # Commit round state inside container (build happens at evaluate time).
                env.execute(
                    f"cd /testbed && git add -A && git reset HEAD .lake/ 2>/dev/null || true && "
                    f"git -c user.email='agent@lean' -c user.name='Agent' "
                    f"commit -m 'refactor-round-{refactor_round + 1}' --allow-empty"
                )
                logger.info("Refactor round %d committed", refactor_round + 1)
                _update_compression(instance, env, baseline_words)

                # Capture cumulative diff (baseline → this round) and delta diff for per-round evaluation.
                # Exclude .lake/ (binary .olean build artifacts) — binary patches without full index lines fail git apply.
                cumulative_diff = env.execute(
                    "cd /testbed && base=$(cat /.init_commit 2>/dev/null || "
                    "git rev-list --max-parents=0 HEAD | tail -1) && git diff $base HEAD -- ':(exclude).lake/'"
                ).get("output", "")
                delta_diff = env.execute(
                    "cd /testbed && git diff HEAD^ HEAD -- ':(exclude).lake/'"
                ).get("output", "")

                instance_dir.mkdir(parents=True, exist_ok=True)
                if cumulative_diff.strip():
                    (instance_dir / f"round_{refactor_round + 1}_patch.diff").write_text(cumulative_diff)
                    update_preds_file(
                        output_dir / f"preds_round_{refactor_round + 1}.json",
                        instance.instance_id,
                        model.config.model_name,
                        cumulative_diff,
                    )
                if delta_diff.strip():
                    (instance_dir / f"round_{refactor_round + 1}_delta.diff").write_text(delta_diff)

                # Save per-round generation metrics (cost, wall time).
                gen_metrics = {
                    "round": refactor_round + 1,
                    "llm_cost": round_llm_cost,
                    "cost_accounting": cost_accounting_from_responses(round_responses),
                    "wall_time_seconds": round_wall_time,
                    "exit_status": exit_status_r,
                }
                (instance_dir / f"round_{refactor_round + 1}_gen_metrics.json").write_text(
                    json.dumps(gen_metrics, indent=2)
                )

                # Save per-round traj.
                # generator.messages is reset each run(), so use it directly.
                # model.responses accumulates, so slice from boundary.
                curr_resp_count = len(getattr(generator.model, "responses", []))
                round_attempt = {
                    "attempt_index": 1,
                    "task": refactor_task,
                    "exit_status": exit_status_r,
                    "result": result_r,
                    "messages": _deepcopy_jsonable(getattr(generator, "messages", [])),
                    "responses": _deepcopy_jsonable(
                        getattr(generator.model, "responses", [])[prev_resp_count:curr_resp_count]
                    ),
                    "playback": _deepcopy_jsonable(getattr(generator, "playback", {})),
                    "native_rollout": _deepcopy_jsonable(
                        getattr(generator, "native_rollout", {})
                    ),
                    "provider_trace": _deepcopy_jsonable(
                        getattr(generator, "provider_trace", {})
                    ),
                    "validation": None,
                }
                save_traj(
                    generator=None,
                    path=output_dir / f"round_{refactor_round + 1}" / instance.instance_id / f"{instance.instance_id}.traj.json",
                    exit_status=exit_status_r,
                    result=result_r,
                    messages=_build_attempt_event_messages([round_attempt]),
                    responses=_build_attempt_event_responses([round_attempt]),
                    trajectory_format="leanlean-chronological-attempts-v1",
                    instance_id=instance.instance_id,
                    playback=_build_attempt_playback([round_attempt]),
                    native_rollouts=_build_attempt_native_rollouts(
                        [round_attempt]
                    ),
                    provider_traces=_build_attempt_provider_traces(
                        [round_attempt]
                    ),
                )
                prev_resp_count = curr_resp_count

                if result_r.strip():
                    result = result_r  # update to accumulated diff

            if train_plan:
                logger.info(f"Training planner on instance {instance.instance_id}...")
                planner.update_plan(instance=instance, traces=generator.messages, result=result, base_dir=instance_dir, model=model)

            logger.info(f"Instance {instance.instance_id} finished with status {exit_status}")
            tool_usage = _persist_agent_tool_usage(env, instance_dir)
            extra_info["agent_tool_usage"] = tool_usage["tools"]
            attempt_records = extra_info.get("attempts") if isinstance(extra_info.get("attempts"), list) else []
            chronological_messages = _build_attempt_event_messages(attempt_records) if attempt_records else None
            chronological_responses = _build_attempt_event_responses(attempt_records) if attempt_records else None
            save_traj(
                generator=generator,
                path=instance_dir / f"{instance.instance_id}.traj.json",
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                messages=chronological_messages,
                responses=chronological_responses,
                trajectory_format="leanlean-chronological-attempts-v1",
                instance_id=instance.instance_id,
                playback=(
                    _build_attempt_playback(attempt_records)
                    if attempt_records
                    else _deepcopy_jsonable(getattr(generator, "playback", {}))
                ),
                native_rollouts=(
                    _build_attempt_native_rollouts(attempt_records)
                    if attempt_records
                    else (
                        [_deepcopy_jsonable(getattr(generator, "native_rollout", {}))]
                        if getattr(generator, "native_rollout", {})
                        else []
                    )
                ),
                provider_traces=(
                    _build_attempt_provider_traces(attempt_records)
                    if attempt_records
                    else (
                        [_deepcopy_jsonable(getattr(generator, "provider_trace", {}))]
                        if getattr(generator, "provider_trace", {})
                        else []
                    )
                ),
            )
            update_preds_file(
                output_dir / "preds.json", instance.instance_id, model.config.model_name, result
            )

            # Final total from the durable response log. The trace was wiped by
            # the last run()'s _post_exec(), so _get_instance_cost() (cached
            # model.cost, last refreshed off the now-empty trace) would read 0
            # and drop the last round. completed_rounds_cost holds the whole
            # cumulative; cost is zeroed so the displayed total matches it.
            update_instance_metrics(
                instance.instance_id,
                status=exit_status,
                completed_rounds_cost=_cumulative_cost(model),
                cost=0.0,
                done=True,
                ended_at=time.monotonic(),
            )

            # Commit the container's .lake/ build cache for fast evaluate.
            if (
                os.environ.get("LEANLEAN_EPHEMERAL_IMAGES") != "1"
                and hasattr(env, "commit")
                and result.strip()
            ):
                cache_tag = getattr(instance, "_cache_image_tag", f"leanlean-{instance.instance_id}-cache:latest")
                env.commit(cache_tag)

            final_capture_done = True
            del env  # Ensure environment is cleaned up
            env = None

    except Exception as e:
        error_message = str(e)
        if isinstance(e, subprocess.CalledProcessError):
            stdout = (e.stdout or "").strip()
            stderr = (e.stderr or "").strip()
            if stdout:
                logger.error(
                    "Subprocess stdout for instance %s:\n%s",
                    instance.instance_id,
                    stdout,
                )
            if stderr:
                logger.error(
                    "Subprocess stderr for instance %s:\n%s",
                    instance.instance_id,
                    stderr,
                )
            if stdout or stderr:
                details = []
                if stdout:
                    details.append(f"stdout:\n{stdout}")
                if stderr:
                    details.append(f"stderr:\n{stderr}")
                error_message = f"{error_message}\n\n" + "\n\n".join(details)

        logger.error(
            f"Error processing instance {instance.instance_id}: {error_message}",
            exc_info=True,
        )
        exit_status, result = type(e).__name__, error_message
        extra_info = {"traceback": traceback.format_exc()}
        update_instance_metrics(
            instance.instance_id,
            status=exit_status,
            cost=_get_instance_cost(model, generator),
            done=True,
            error=result,
            ended_at=time.monotonic(),
        )
    finally:
        if env is not None:
            try:
                _persist_agent_tool_usage(env, instance_dir)
            except BaseException as usage_error:
                logger.error(
                    "Failed to persist agent tool usage for %s: %s",
                    instance.instance_id,
                    usage_error,
                )
        # Safety net: if the agent run was killed or timed out before the
        # happy-path persistence ran, the container still holds the agent's
        # work on disk. Capture it NOW — extract the diff, write preds, and
        # snapshot the container as the cache image — *before* the container
        # is torn down. A `finally` runs even on KeyboardInterrupt, so this
        # also covers manual kills (which `except Exception` above misses).
        # Without this, a killed run loses its result entirely and forces a
        # lossy replay-reconstruction against a base that may have drifted.
        if env is not None and not final_capture_done:
            try:
                logger.warning(
                    "Instance %s did not finish normally (status=%s); "
                    "running best-effort capture before teardown.",
                    instance.instance_id, exit_status,
                )
                rescued_diff = ""
                if generator is not None and hasattr(generator, "_extract_final_diff"):
                    rescued_diff = generator._extract_final_diff() or ""
                if rescued_diff.strip():
                    instance_dir.mkdir(parents=True, exist_ok=True)
                    _persist_rescued_prediction(
                        output_dir,
                        instance.instance_id,
                        model.config.model_name,
                        rescued_diff,
                    )
                    logger.warning(
                        "Rescued %d-char diff for %s into final and round-0 predictions.",
                        len(rescued_diff), instance.instance_id,
                    )
                    # Snapshot the live container so the exact base + final
                    # state survive even if the base image is later rebuilt.
                    if (
                        os.environ.get("LEANLEAN_EPHEMERAL_IMAGES") != "1"
                        and hasattr(env, "commit")
                    ):
                        cache_tag = getattr(
                            instance, "_cache_image_tag",
                            f"leanlean-{instance.instance_id}-cache:latest",
                        )
                        env.commit(cache_tag)
                else:
                    logger.warning(
                        "Best-effort capture for %s found no diff on disk.",
                        instance.instance_id,
                    )
            except BaseException as cap_err:  # never let the safety net mask the original failure
                logger.error(
                    "Best-effort capture for %s failed: %s",
                    instance.instance_id, cap_err,
                )
        cost_stop_event.set()
        cost_thread.join(timeout=2)
        if env is not None and hasattr(env, "cleanup"):
            env.cleanup()
            env = None
        if hasattr(instance, "retire_materialized_images"):
            try:
                removed = instance.retire_materialized_images()
                if removed:
                    logger.info(
                        "Retired %d materialized image tag(s) for %s",
                        len(removed),
                        instance.instance_id,
                    )
            except BaseException as cleanup_error:
                logger.error(
                    "Failed to retire materialized images for %s: %s",
                    instance.instance_id,
                    cleanup_error,
                )


def generate_baseline(
    output: Path | str,
    workers: int,
    config: dict,
    redo_existing: bool = False,
    override_failed: bool = False,
    train_plan: bool = False,
    setup_repo: bool = True,
    hard_task: bool = False,
    remove_docs: bool = False,
    n_tries: int = 1,
    progress: str = "tqdm",
) -> None:

    if isinstance(output, str):
        output_path = Path(output)
    else:
        output_path = output
    output_path.mkdir(parents=True, exist_ok=True)
    if (
        "LEANLEAN_REPO_VARIANT" in os.environ
        or "LEANLEAN_USE_PROD" in os.environ
    ):
        (output_path / "run_metadata.json").write_text(
            json.dumps(
                {
                    "format": "leanlean-run-metadata-v1",
                    "repo_variant": repo_variant_from_env(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    add_file_handler(output_path / "leanlean.log")
    logger.info(f"Results will be saved to {output_path}")

    progress_mode = (progress or "tqdm").lower()
    if progress_mode not in ("tqdm", "rich"):
        raise ValueError(f"Unknown progress mode: {progress}")
    reset_instance_metrics()
    set_instance_metrics_max_rows(workers if progress_mode == "rich" else None)
    dataset_name = None
    model_name = None
    agent_name = None
    if isinstance(config, dict):
        dataset_name = config.get("benchmark", {}).get("dataset_name")
        model_name = config.get("model", {}).get("model_name")
        agent_name = config.get("generator", {}).get("name")

    # Loading the benchmark
    benchmark = get_benchmark(config["benchmark"])
    instances = benchmark.get_instances()
    requested_rounds = int(config.get("planner", {}).get("refactor_rounds", 0))
    if not redo_existing:
        skipped_existing_instances: list[str] = []
        overridden_failed_instances: list[str] = []
        resumed_instances: list[str] = []
        remaining_instances: list[Instance] = []
        for instance in instances:
            if not _instance_has_existing_output(output_path, instance.instance_id):
                archived = _archive_retryable_instance(
                    output_path,
                    instance.instance_id,
                    reason="missing_trajectory",
                )
                if archived is not None:
                    overridden_failed_instances.append(
                        f"{instance.instance_id} (missing_trajectory)"
                    )
                remaining_instances.append(instance)
                continue

            exit_status = _get_existing_exit_status(
                output_path, instance.instance_id
            )
            retry_reason: str | None = None
            if override_failed and exit_status in {
                "ExecutionFailed",
                "ProviderFailed",
                "AuthenticationFailed",
                "QuotaExceeded",
            }:
                retry_reason = str(exit_status)
            elif not _instance_has_prediction(output_path, instance.instance_id):
                retry_reason = "missing_prediction"
            elif (
                config.get("playback", {}).get("enabled", False)
                and not _instance_has_scoring_capture(
                    output_path, instance.instance_id
                )
            ):
                retry_reason = "missing_scoring_capture"

            if retry_reason is not None:
                _archive_retryable_instance(
                    output_path,
                    instance.instance_id,
                    reason=retry_reason,
                )
                overridden_failed_instances.append(
                    f"{instance.instance_id} ({retry_reason})"
                )
                remaining_instances.append(instance)
                continue

            # Appendable refactor rounds: re-include if more rounds are requested
            # than already completed; process_instance will fast-forward.
            completed = _completed_refactor_round(output_path, instance.instance_id)
            if requested_rounds > completed:
                resumed_instances.append(f"{instance.instance_id} ({completed}->{requested_rounds})")
                remaining_instances.append(instance)
                continue

            skipped_existing_instances.append(instance.instance_id)

        if skipped_existing_instances:
            logger.info(f"Skipping {len(skipped_existing_instances)} existing instances")
        if resumed_instances:
            logger.info("Resuming %d instances for more refactor rounds: %s",
                        len(resumed_instances), ", ".join(resumed_instances))
        if overridden_failed_instances:
            logger.info(
                "Re-running %d incomplete instance(s): %s",
                len(overridden_failed_instances),
                ", ".join(overridden_failed_instances),
            )
        instances = remaining_instances
    logger.info(f"Running on {len(instances)} instances...")

    instance_start_gate = InstanceStartGate(
        float(config.get("instance_start_interval_seconds", 0))
    )

    requires_model_proxy = config["generator"].get("requires_model_proxy", True)
    subscription_transport = config["generator"].get("subscription_transport")
    # Every worker shares this proxy, regardless of API vs subscription auth.
    preserve_shared_proxy_traces = bool(requires_model_proxy)
    if requires_model_proxy and subscription_transport:
        config["model"]["subscription_transport"] = subscription_transport
    if requires_model_proxy and preserve_shared_proxy_traces:
        config["model"]["preserve_shared_traces"] = True
    # One proxy serves every worker on these transports, so its cost journal is
    # the run total -- authoritative for the run, unattributable per instance.
    set_shared_cost_source(
        bool(requires_model_proxy)
        and bool(subscription_transport or preserve_shared_proxy_traces)
    )
    model_server = get_model(config["model"])
    set_shared_cost_source(bool(requires_model_proxy), model_server.get_cost if requires_model_proxy else None)
    planner_class = config["planner"].get("planner_class")
    if not requires_model_proxy and planner_class != "no_plan":
        raise ValueError(
            "Native subscription generators bypass LiteLLM and currently require "
            "plan_type=no_plan; use a proxy-backed generator for model-driven planners."
        )
    if requires_model_proxy:
        model_server.serve()
        config["model"]["port"] = model_server._port
    else:
        logger.info(
            "Generator %s uses native subscription auth; LiteLLM proxy disabled.",
            config["generator"].get("cli_name"),
        )

    def process_futures_tqdm(future_to_id: dict[concurrent.futures.Future, str]) -> None:
        total = len(future_to_id)
        with tqdm(total=total, desc="Processing", unit="task") as pbar:
            for future in concurrent.futures.as_completed(future_to_id):
                try:
                    future.result()
                except concurrent.futures.CancelledError:
                    instance_id = future_to_id[future]
                    update_instance_metrics(instance_id, status="Cancelled", done=True)
                except Exception as e:
                    instance_id = future_to_id[future]
                    logger.error(
                        f"Error in future for instance {instance_id}: {e}",
                        exc_info=True,
                    )
                finally:
                    pbar.update(1)

    def process_futures_rich(future_to_id: dict[concurrent.futures.Future, str]) -> None:
        total = len(future_to_id)
        console = Console()
        _attach_rich_console(console)
        dashboard = RichProgressDashboard(
            total=total,
            workers=workers,
            dataset_name=dataset_name,
            model_name=model_name,
            agent_name=agent_name,
        )
        with Live(
            dashboard,
            console=console,
            refresh_per_second=6,
            transient=False,
            redirect_stdout=True,
            redirect_stderr=True,
        ):
            for future in concurrent.futures.as_completed(future_to_id):
                try:
                    future.result()
                except concurrent.futures.CancelledError:
                    instance_id = future_to_id[future]
                    update_instance_metrics(instance_id, status="Cancelled", done=True)
                except Exception as e:
                    instance_id = future_to_id[future]
                    logger.error(
                        f"Error in future for instance {instance_id}: {e}",
                        exc_info=True,
                    )
                finally:
                    dashboard.mark_completed()

    def process_futures(future_to_id: dict[concurrent.futures.Future, str]) -> None:
        if progress_mode == "rich":
            process_futures_rich(future_to_id)
        else:
            process_futures_tqdm(future_to_id)

    # Tag cache images with a run-specific suffix so concurrent runs on different
    # models never overwrite each other's cache.
    import re as _re
    run_tag = _re.sub(r"[^a-zA-Z0-9_.-]", "_", "_".join(output_path.parts[-3:]))
    for instance in instances:
        if hasattr(instance, "run_tag"):
            instance.run_tag = run_tag

    def process_instance_with_start_gate(instance: Instance) -> None:
        if admission_held():
            update_instance_metrics(instance.instance_id, status="Memory safety hold", done=True)
            return
        instance_start_gate.wait(instance.instance_id)
        if admission_held():
            update_instance_metrics(instance.instance_id, status="Memory safety hold", done=True)
            return
        process_instance(
            instance,
            output_path,
            config,
            train_plan,
            n_tries,
            setup_repo=setup_repo,
            hard_task=hard_task,
            remove_docs=remove_docs,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                process_instance_with_start_gate, instance
            ): instance.instance_id
            for instance in instances
        }

        try:
            process_futures(future_to_id)
        except KeyboardInterrupt:
            logger.info(
                "Cancelling all pending jobs. Press ^C again to exit immediately."
            )
            for f in future_to_id:
                if not f.running() and not f.done():
                    f.cancel()
            remaining = {f: iid for f, iid in future_to_id.items() if not f.done()}
            process_futures(remaining)

    try:
        action_summary_path = write_run_action_summary(output_path)
        logger.info("Wrote compact run action summary to %s", action_summary_path)
    except Exception as exc:
        logger.error("Failed to write compact run action summary: %s", exc)

    model_server.stop()
    if requires_model_proxy and (
        subscription_transport
        or preserve_shared_proxy_traces
    ):
        archived = model_server.archive_traces(output_path / "proxy_traces")
        if archived:
            logger.info(
                "Archived proxy journals: %s",
                ", ".join(
                    f"{label}={trace_path}"
                    for label, trace_path in sorted(archived.items())
                ),
            )



def main(
    plan_type: str,
    exec_model: str,
    generator: str,
    task_variant: str = "fix",
    run_id: str | int = 0,
    port: int = 0,
    plan_model: str | None = None,
    plan_generator: str | None = None,
    output_dir: str = "output",
    run_output_directory: str | None = None,
    dataset_name: str = "leanlean",
    benchmark: str = "leanlean",
    filter_spec: str = "",
    slice_spec: str = "",
    shuffle: bool = False,
    split: str | None = None,
    workers: int = 2,
    instance_start_interval_seconds: float = 0,
    progress: str = "rich",
    override_failed: bool = True,
    plan_args: dict = {},
    train_plan: bool = False,
    continuous_training: bool = False,
    setup_repo: bool = True,
    hard_task: bool = False,
    remove_docs: bool = False,
    exec_model_api_base: str | None = None,
    exec_model_config: dict | None = None,
    debug: bool = False,
    compile_gate: bool|None = None,
    compile_gate_command: str | None = None,
    compile_gate_max_retries: int = 5,
    max_api_retries: int = MAX_RETRIES,
    compile_gate_output_chars: int = 12000,
    build_jobs: int = -1,
    container_cpus: int = -1,
    container_memory: str = "",
    container_cgroup_parent: str = "",
    container_timeout: str = "2h",
    network_policy: str = "model_proxy_only",
    container_pids_limit: int = 4096,
    container_resource_visibility: str = "host",
    container_grace_seconds: int = 1800,
    task_metadata_enabled: bool = True,
    enable_lean_verify: bool = True,
    enable_proof_length: bool = True,
    agent: dict[str, Any] | None = None,
    repository_contracts: dict[str, dict[str, Any]] | None = None,
    repository_verifiers: dict[str, dict[str, Any]] | None = None,
    repository_entries: list[dict[str, Any]] | None = None,
    file_level: bool = False,
    playback: dict | str | None = None,
    trajectory: dict | str | None = None,
) -> None:

    port = 0

    if remove_docs:
        logger.info("Documentation files will be removed from the repositories.")

    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode is ON")
    
    if isinstance(plan_args, str):
        import yaml
        plan_args = yaml.safe_load(plan_args)

    if plan_args is None:
        plan_args = {}

    output_path = Path(output_dir)
    if isinstance(train_plan, str):
        train_plan = train_plan.lower() in ("yes", "true", "t", "1")
    if isinstance(override_failed, str):
        override_failed = override_failed.lower() in ("yes", "true", "t", "1")

    if isinstance(max_api_retries, bool):
        raise ValueError("max_api_retries must be a non-negative integer")
    try:
        max_api_retries = int(max_api_retries)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_api_retries must be a non-negative integer") from exc
    if max_api_retries < 0:
        raise ValueError("max_api_retries must be a non-negative integer")

    import yaml

    optional_configs = {
        "playback": playback,
        "trajectory": trajectory,
        "agent": agent,
    }
    for name, value in optional_configs.items():
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        optional_configs[name] = value

    if train_plan or continuous_training:
        if is_plan_training_sequential(plan_type):
            if workers > 1:
                logger.error(f"Training {plan_type} is not thread-safe; setting workers to 1") 
            workers = 1  # Training plans is not thread-safe

    config = {
        "model": deepcopy(exec_model_config if exec_model_config is not None else ALL_MODEL_CONFIGS[exec_model]),
        "planner": deepcopy(ALL_PLAN_CONFIGS[plan_type]),
        "generator": deepcopy(ALL_GENERATOR_CONFIGS[generator]),
        "benchmark": deepcopy(ALL_BENCHMARK_CONFIGS[benchmark]),
        "task_variant": ALL_TASK_VARIANTS[task_variant],
        "playback": deepcopy(optional_configs["playback"]),
        "trajectory": deepcopy(optional_configs["trajectory"]),
        "instance_start_interval_seconds": float(
            instance_start_interval_seconds
        ),
        "max_api_retries": max_api_retries,
    }

    if "generator_config" in config["planner"]:
        if not config["planner"]["generator_config"]:
            logger.info("Copying generator config to planner config")
            config["planner"]["generator_config"] = config["generator"]
        


    # Default classes if not specified
    if "model_class" not in config["model"]:
        config["model"]["model_class"] = "litellm_server"
    if "planner_class" not in config["planner"]:
        config["planner"]["planner_class"] = "no_plan"
    if "generator_class" not in config["generator"]:
        config["generator"]["generator_class"] = "cli_agent"

    # If litellm_server, set the port
    if config["model"]["model_class"] == "litellm_server":
        config["model"]["port"] = port


    if exec_model_api_base is not None:
        config["model"]["api_base"] = exec_model_api_base

    # Add plan model to planner config
    if plan_model is None:
        plan_model = exec_model
    else:
        plan_model_config = deepcopy(ALL_MODEL_CONFIGS[plan_model])
        if "model_class" not in plan_model_config:
            plan_model_config["model_class"] = "litellm_server"
        config["planner"]["model_config"] = plan_model_config
    config["planner"]["plan_model"] = plan_model


    # Add generator model to planner config
    if plan_generator is None:
        plan_generator = generator
    plan_generator_config = deepcopy(ALL_GENERATOR_CONFIGS[plan_generator])
    if "generator_class" not in plan_generator_config:
        plan_generator_config["generator_class"] = "cli_agent"
    config["planner"]["generator_config"] = plan_generator_config


    if continuous_training:
        config["planner"]["storage_dir"] = f"{config['planner']['storage_dir']}/{dataset_name.replace('/', '_')}/{plan_generator}/run_{run_id}"
    else:
        config["planner"]["storage_dir"] = f"{config['planner']['storage_dir']}/{dataset_name.replace('/', '_')}/{plan_generator}"
    config["planner"].update(plan_args) # Add any additional plan args that we want to be dynamic

    main_dir = (
        Path(run_output_directory)
        if run_output_directory is not None
        else _compute_run_directory(
            output_dir=output_path,
            dataset_name=dataset_name,
            plan_type=plan_type,
            generator=generator,
            exec_model=exec_model,
            run_id=run_id,
            planner_config=config["planner"],
            kind="leanlean",
            train_plan=train_plan,
            continuous_training=continuous_training,
            task_variant=task_variant,
        )
    )

    # Setup benchmark config
    config["benchmark"]["dataset_name"] = dataset_name
    config["benchmark"]["filter_spec"] = filter_spec
    config["benchmark"]["slice_spec"] = slice_spec
    config["benchmark"]["shuffle"] = shuffle
    if compile_gate is not None:
        config["benchmark"]["compile_gate_enabled"] = compile_gate
        if compile_gate_command is not None:
            config["benchmark"]["compile_gate_command"] = compile_gate_command
        config["benchmark"]["compile_gate_max_retries"] = compile_gate_max_retries
        config["benchmark"]["compile_gate_output_chars"] = compile_gate_output_chars
    if benchmark == "leanlean":
        config["benchmark"]["file_level"] = file_level
        config["benchmark"]["task_metadata_enabled"] = task_metadata_enabled
        config["benchmark"]["enable_lean_verify"] = enable_lean_verify
        config["benchmark"]["enable_proof_length"] = enable_proof_length
        if optional_configs["agent"]:
            agent_contract = deepcopy(optional_configs["agent"])
            config["benchmark"]["task_prompt"] = agent_contract["prompt"]
            config["generator"]["system_prompt"] = agent_contract[
                "system_prompt"
            ]
            config["generator"]["allowed_skills"] = agent_contract[
                "allowed_skills"
            ]
            config["generator"]["mcp_servers"] = agent_contract["mcp_servers"]
            config["generator"]["enable_subagents"] = agent_contract[
                "enable_subagents"
            ]
            if generator == "codex_sub":
                marker = "--model {model}"
                controls = (
                    f"-c agents.enabled={str(agent_contract['enable_subagents']).lower()} "
                    f"-c features.multi_agent={str(agent_contract['enable_subagents']).lower()} "
                    "-c include_collaboration_mode_instructions=false "
                    "-c include_apps_instructions=false "
                    "-c features.skip_host_skill_discovery=true "
                    "-c 'skills.config=[{{name=\"imagegen\",enabled=false}},"
                    "{{name=\"openai-docs\",enabled=false}},"
                    "{{name=\"plugin-creator\",enabled=false}},"
                    "{{name=\"skill-creator\",enabled=false}},"
                    "{{name=\"skill-installer\",enabled=false}}]' "
                    "-c features.apps=false -c features.remote_plugin=false "
                    "-c features.plugins=false -c features.recommended_plugins=false "
                    "-c features.skill_search=false "
                    "-c features.skill_mcp_dependency_install=false "
                    "-c features.browser_use=false "
                    "-c features.browser_use_external=false "
                    "-c features.browser_use_full_cdp_access=false "
                    "-c features.computer_use=false "
                    "-c features.image_generation=false "
                    "-c features.goals=false -c features.hooks=false "
                    "-c features.plugin_sharing=false -c features.personality=false "
                    "-c features.tool_suggest=false "
                    "-c features.workspace_dependencies=false "
                    "-c features.enable_mcp_apps=false "
                    "-c features.tool_call_mcp_elicitation=false "
                    "-c features.default_mode_request_user_input=false "
                    "-c features.sleep_tool=false "
                    "-c features.memories=false "
                    "-c features.external_agent_memory_import=false "
                    "-c web_search=disabled "
                )
                launch = config["generator"]["launch_command"]
                if marker not in launch:
                    raise ValueError("Codex launch command has no model marker")
                config["generator"]["launch_command"] = launch.replace(
                    marker, controls + marker, 1
                )
            if generator == "claude_code_sub" and agent_contract.get("native_tools"):
                config["generator"]["launch_command"] = claude_native_tools_launch(
                    config["generator"]["launch_command"],
                    agent_contract["native_tools"],
                    enable_subagents=agent_contract["enable_subagents"],
                )
        config["benchmark"]["repository_contracts"] = (
            repository_contracts or {}
        )
        config["benchmark"]["repository_verifiers"] = (
            repository_verifiers or {}
        )
        config["benchmark"]["repository_entries"] = repository_entries or []
    config["benchmark"]["build_jobs"] = build_jobs
    if container_cpus > 0:
        config["benchmark"]["container_cpus"] = container_cpus
    if container_memory:
        config["benchmark"]["container_memory"] = container_memory
    if container_cgroup_parent:
        config["benchmark"]["container_cgroup_parent"] = container_cgroup_parent
    config["benchmark"]["container_timeout"] = container_timeout
    config["benchmark"]["network_policy"] = network_policy
    config["benchmark"]["container_pids_limit"] = container_pids_limit
    config["benchmark"]["container_resource_visibility"] = container_resource_visibility
    config["benchmark"]["container_grace_seconds"] = container_grace_seconds


    main_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting benchmark with config:\n{_redact_config_secrets(config)}\n"
        f"Output dir: {main_dir}. Setting up repo: {setup_repo}"
    )

    train_plan = train_plan or continuous_training

    generate_baseline(
        output=main_dir,
        workers=workers,
        config=config,
        override_failed=override_failed,
        train_plan=train_plan,
        setup_repo=setup_repo,
        hard_task=hard_task,
        remove_docs=remove_docs,
        progress=progress,
    )

    if 'plan_model_obj' in locals():
        plan_model_obj.stop()  # Stop the plan model server

if __name__ == "__main__":
    fire.Fire(main)
