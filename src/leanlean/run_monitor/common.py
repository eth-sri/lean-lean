from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def mapping(value):
    return value if isinstance(value, dict) else {}


def read_record(path: Path, limit=32 * 1024 * 1024):
    try:
        if path.stat().st_size > limit:
            return {}
        content = path.read_text()
        return mapping(json.loads(content) if path.suffix == '.json' else yaml.load(content, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader)))
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def command(args, timeout=15, *, allow_partial=False):
    # Never send command output/errors to the browser: they may contain credentials.
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode and not (allow_partial and result.stdout.strip()):
        raise RuntimeError(f'{Path(args[0]).name} unavailable (exit {result.returncode})')
    return result.stdout


def local_path(root, value):
    if not isinstance(value, str) or not value or '*' in value:
        return None
    p = (root / value).resolve()
    # Run records may reference external roots; do not let records read arbitrary files.
    return p if p.is_relative_to(root) else None
