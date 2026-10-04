"""Provider-aware waiting for immutable subscription evaluation retries."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


ANTHROPIC_USAGE_ENDPOINT = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_USAGE_WINDOWS = ("five_hour", "seven_day")
WINDOW_LABELS = {
    "five_hour": "5h usage",
    "seven_day": "7d usage",
    "seven_day_opus": "7d Opus",
    "seven_day_sonnet": "7d Sonnet",
}


@dataclass(frozen=True)
class QueuePolicy:
    enabled: bool
    retry_backoff_seconds: int
    fallback_cooldown_seconds: int
    capacity_probe: str
    max_utilization_percent: float = 100.0


@dataclass(frozen=True)
class UsageWindow:
    name: str
    utilization: float
    resets_at: str | None


@dataclass(frozen=True)
class UsageStatus:
    windows: tuple[UsageWindow, ...]
    detail: str

    @property
    def available(self) -> bool:
        return bool(self.windows)


@dataclass(frozen=True)
class CapacityDecision:
    usage: UsageStatus
    wait_until: float | None
    reason: str

    @property
    def should_wait(self) -> bool:
        return self.wait_until is not None


def capacity_probe_for_harness(harness: str) -> str:
    if harness == "claude_code_sub":
        return "anthropic_oauth_usage"
    return "fixed_cooldown"


def timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _claude_profile_token(config_dir: str | Path | None = None) -> str:
    """Read Claude's host-only profile token for usage checks.

    ``claude setup-token`` credentials intentionally carry only the inference
    scope. A normal ``claude auth login`` also carries ``user:profile``, which
    is required by the usage endpoint. This token is read only by the host
    queue and is never copied into an agent container or persisted in a run
    artifact.
    """

    config_dir = Path(
        config_dir
        or os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
    )
    try:
        credentials = json.loads((config_dir / ".credentials.json").read_text())
        oauth = credentials["claudeAiOauth"]
        scopes = oauth.get("scopes", ())
        token = oauth.get("accessToken", "")
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    if (
        not isinstance(scopes, list)
        or "user:profile" not in scopes
        or not isinstance(token, str)
    ):
        return ""
    return token


def _fetch_anthropic_usage_payload(
    token: str, timeout_seconds: int
) -> tuple[Any | None, int | None, str | None]:
    request = urllib.request.Request(
        ANTHROPIC_USAGE_ENDPOINT,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "Accept": "application/json",
            "User-Agent": "claude-code/2.1.233",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response), None, None
    except urllib.error.HTTPError as error:
        return None, error.code, None
    except (OSError, ValueError) as error:
        return None, None, type(error).__name__


def fetch_anthropic_usage(timeout_seconds: int = 15) -> UsageStatus:
    inference_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    token_source = os.environ.get("LEANLEAN_CLAUDE_TOKEN_ENV", "")
    # An explicitly selected alternate setup-token must never fall through to
    # the primary Claude profile and gate the alternate run on the wrong
    # account's quota.
    profile_token = (
        _claude_profile_token()
        if token_source in {"", "CLAUDE_CODE_OAUTH_TOKEN"}
        else ""
    )
    candidates = []
    if inference_token:
        candidates.append(inference_token)
    if profile_token and profile_token != inference_token:
        candidates.append(profile_token)
    if not candidates:
        return UsageStatus((), "OAuth token is not loaded")
    payload = None
    last_status = None
    last_error = None
    for token in candidates:
        payload, last_status, last_error = _fetch_anthropic_usage_payload(
            token, timeout_seconds
        )
        if payload is not None:
            break
    if payload is None:
        if last_status == 403:
            return UsageStatus((), "HTTP 403 (token lacks user:profile scope)")
        if last_status is not None:
            return UsageStatus((), f"HTTP {last_status}")
        return UsageStatus(
            (), f"usage check unavailable: {last_error or 'unknown error'}"
        )
    if not isinstance(payload, dict):
        return UsageStatus((), "usage response is not a mapping")

    windows: list[UsageWindow] = []
    for name, value in payload.items():
        if not any(
            name == base or str(name).startswith(base + "_")
            for base in ANTHROPIC_USAGE_WINDOWS
        ):
            continue
        if not isinstance(value, dict) or "utilization" not in value:
            continue
        try:
            utilization = float(value["utilization"])
        except (TypeError, ValueError):
            continue
        reset = value.get("resets_at")
        windows.append(
            UsageWindow(
                str(name),
                utilization,
                str(reset) if reset else None,
            )
        )
    order = {name: index for index, name in enumerate(WINDOW_LABELS)}
    windows.sort(key=lambda window: (order.get(window.name, 100), window.name))
    if not windows:
        return UsageStatus((), "usage response has no quota windows")
    return UsageStatus(tuple(windows), "quota utilization available")


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None



def blocking_windows(
    usage: UsageStatus, max_utilization_percent: float
) -> tuple[UsageWindow, ...]:
    return tuple(
        window
        for window in usage.windows
        if window.utilization >= max_utilization_percent
    )


def _usage_fetcher(
    capacity_probe: str,
) -> Callable[[], UsageStatus] | None:
    return {
        "anthropic_oauth_usage": fetch_anthropic_usage,
    }.get(capacity_probe)


def _reset_epoch(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def capacity_decision(
    policy: QueuePolicy,
    *,
    after_quota_failure: bool = False,
    usage_fetcher: Callable[[], UsageStatus] | None = None,
    now: float | None = None,
) -> CapacityDecision:
    """Check usage once and decide whether this launch must wait."""

    checked_at = time.time() if now is None else now
    fetcher = usage_fetcher or _usage_fetcher(policy.capacity_probe)
    usage = (
        fetcher()
        if fetcher is not None
        else UsageStatus((), "no managed usage endpoint")
    )
    blocked = blocking_windows(usage, policy.max_utilization_percent)
    if blocked:
        resets = [_reset_epoch(window.resets_at) for window in blocked]
        if all(reset is not None and reset > checked_at for reset in resets):
            return CapacityDecision(
                usage,
                max(reset for reset in resets if reset is not None),
                "provider_reset",
            )
        return CapacityDecision(
            usage,
            checked_at + policy.fallback_cooldown_seconds,
            "fallback_cooldown",
        )
    if not after_quota_failure:
        reason = "capacity_available" if usage.available else "unchecked"
        return CapacityDecision(usage, None, reason)
    if usage.available:
        return CapacityDecision(
            usage,
            checked_at + policy.retry_backoff_seconds,
            "retry_backoff",
        )
    return CapacityDecision(
        usage,
        checked_at + policy.fallback_cooldown_seconds,
        "fallback_cooldown",
    )


def _duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _panel(
    *,
    run_id: str,
    harness: str,
    attempt: int,
    usage: UsageStatus,
    next_action: str,
) -> Panel:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Run", run_id)
    table.add_row("Subscription", harness)
    table.add_row("Attempt", str(attempt))
    if usage.available:
        for window in usage.windows:
            detail = f"{window.utilization:.1f}% used"
            if window.resets_at:
                detail += f" · resets {window.resets_at}"
            table.add_row(
                WINDOW_LABELS.get(window.name, window.name.replace("_", " ")),
                detail,
            )
    else:
        table.add_row("Capacity", usage.detail)
    table.add_row("Next", next_action)
    return Panel(table, title="Subscription quota queue", border_style="blue")


def wait_for_capacity(
    decision: CapacityDecision,
    *,
    run_id: str,
    harness: str,
    attempt: int,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Display a local countdown without repeatedly querying usage."""

    if decision.wait_until is None:
        return
    console = Console()
    with Live(console=console, refresh_per_second=1, transient=False) as live:
        while True:
            remaining = decision.wait_until - clock()
            if remaining <= 0:
                live.update(
                    _panel(
                        run_id=run_id,
                        harness=harness,
                        attempt=attempt,
                        usage=decision.usage,
                        next_action="wait complete; checking once before launch",
                    )
                )
                return
            label = {
                "provider_reset": "quota reset",
                "retry_backoff": "retry backoff",
                "fallback_cooldown": "fallback cooldown",
            }.get(decision.reason, "capacity wait")
            live.update(
                _panel(
                    run_id=run_id,
                    harness=harness,
                    attempt=attempt,
                    usage=decision.usage,
                    next_action=f"{label} in {_duration(remaining)}",
                )
            )
            sleeper(min(max(remaining, 0.01), 60))
