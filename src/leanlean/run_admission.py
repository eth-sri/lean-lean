"""Host-side, persistent admission hold; never exposed in the agent worktree."""
from __future__ import annotations

import os
from pathlib import Path

ENVIRONMENT_KEY = 'LEANLEAN_ADMISSION_HOLD_FILE'


def admission_held() -> bool:
    value = os.environ.get(ENVIRONMENT_KEY)
    if not value:
        return False
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('admission hold must be an absolute host path')
    # lexists also fails closed on a dangling symlink.
    return os.path.lexists(path)


def require_admission() -> None:
    if admission_held():
        raise RuntimeError('Memory safety hold: no new instance, round, or retry may start')
