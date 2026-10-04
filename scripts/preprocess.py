#!/usr/bin/env python3
"""Preprocess a benchmark dataset with the pinned lean-strip."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from leanlean.pipeline.lean_strip_preprocessing import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
