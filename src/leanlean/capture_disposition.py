"""Fail-closed scoring dispositions for monitored retry captures."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FORMAT = "leanlean-capture-dispositions-v1"
FILENAME = "capture-dispositions.json"
SCHEMA_VERSION = 1
SCORING = "scored"
DISCARDED_API_FAILURE = "discarded_api_failure"
DISCARDED_UNFAIR_RETRY = "discarded_unfair_retry"
ALLOWED_DISPOSITIONS = {
    SCORING,
    DISCARDED_API_FAILURE,
    DISCARDED_UNFAIR_RETRY,
}
_CAPTURE_NAME = re.compile(r"capture_[0-9]{3,}")


def load_capture_dispositions(playback_dir: Path) -> dict[str, dict[str, Any]]:
    """Load and validate the scoring sidecar for one repository."""

    path = playback_dir / FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read capture dispositions {path}: {error}") from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != FORMAT
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("captures"), Mapping)
    ):
        raise ValueError(f"{path}: invalid capture disposition sidecar")
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_entry in payload["captures"].items():
        name = str(raw_name)
        if not _CAPTURE_NAME.fullmatch(name) or not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path}: invalid capture disposition entry {name!r}")
        disposition = raw_entry.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            raise ValueError(
                f"{path}: unsupported disposition {disposition!r} for {name}"
            )
        result[name] = dict(raw_entry)
    return result


def capture_disposition(playback_path: Path) -> str:
    """Return the explicit disposition or the backward-compatible scoring default."""

    capture_dir = playback_path.parent
    entries = load_capture_dispositions(capture_dir.parent)
    entry = entries.get(capture_dir.name)
    return str(entry.get("disposition")) if entry else SCORING


def capture_is_scoring(playback_path: Path) -> bool:
    return capture_disposition(playback_path) == SCORING


def record_capture_disposition(
    playback_dir: Path,
    capture_name: str,
    disposition: str,
    *,
    reason: Mapping[str, Any],
) -> Path:
    """Atomically record why a raw capture is scored or excluded."""

    if not _CAPTURE_NAME.fullmatch(capture_name):
        raise ValueError(f"invalid capture name {capture_name!r}")
    if disposition not in ALLOWED_DISPOSITIONS:
        raise ValueError(f"unsupported capture disposition {disposition!r}")
    path = playback_dir / FILENAME
    entries = load_capture_dispositions(playback_dir)
    entries[capture_name] = {
        "disposition": disposition,
        "reason": dict(reason),
    }
    payload = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "captures": dict(sorted(entries.items())),
    }
    playback_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
