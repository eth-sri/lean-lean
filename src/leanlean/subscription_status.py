"""Classify subscription-provider failures recorded in saved trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from leanlean.utils import json_utils as json


QUOTA_EXIT_CODE = 75
QUOTA_FAILURE = "quota_exhausted"
PROVIDER_FAILURE = "provider_error"


@dataclass(frozen=True)
class SubscriptionFailure:
    instance_id: str
    kind: str
    detail: str


def _is_failed_terminal(terminal: dict[str, Any]) -> bool:
    return bool(terminal.get("is_error")) or terminal.get("success") is False


def classify_trajectory(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Return the final provider failure kind and detail, if one was recorded."""

    info = payload.get("info")
    if isinstance(info, dict) and info.get("exit_status") == "AuthenticationFailed":
        return (
            PROVIDER_FAILURE,
            "Authentication failed; update OPENAI_SUBSCRIPTION_KEY in secret.sh "
            "and relaunch.",
        )
    if isinstance(info, dict) and info.get("exit_status") == "QuotaExceeded":
        return QUOTA_FAILURE, "saved exit_status=QuotaExceeded"

    traces = payload.get("provider_traces")
    if not isinstance(traces, list) or not traces:
        return None
    trace = traces[-1]
    if not isinstance(trace, dict):
        return None
    terminal = trace.get("terminal")
    if not isinstance(terminal, dict) or not _is_failed_terminal(terminal):
        return None

    stream_events = trace.get("stream_events")
    quota_rejected = any(
        isinstance(event, dict)
        and event.get("type") == "rate_limit_event"
        and isinstance(event.get("rate_limit_info"), dict)
        and event["rate_limit_info"].get("status") == "rejected"
        for event in (stream_events if isinstance(stream_events, list) else [])
    )
    detail = str(
        terminal.get("failure_reason")
        or terminal.get("terminal_reason")
        or "Subscription provider terminal failure"
    )
    status = terminal.get("api_error_status")
    quota_status = status == 429 or "api_error_status=429" in detail
    if terminal.get("failure_kind") == "quota" or quota_rejected or quota_status:
        return QUOTA_FAILURE, detail
    return PROVIDER_FAILURE, detail


def find_subscription_failures(
    run_dir: Path,
    instance_ids: Iterable[str],
) -> list[SubscriptionFailure]:
    failures: list[SubscriptionFailure] = []
    for instance_id in instance_ids:
        trajectory = run_dir / instance_id / f"{instance_id}.traj.json"
        if not trajectory.is_file():
            continue
        try:
            payload = json.loads(trajectory.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        classified = classify_trajectory(payload)
        if classified is None:
            continue
        kind, detail = classified
        failures.append(
            SubscriptionFailure(
                instance_id=instance_id,
                kind=kind,
                detail=detail,
            )
        )
    return failures
