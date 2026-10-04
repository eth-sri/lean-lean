#!/usr/bin/env python3
"""Public dataset/model postprocessing entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    launcher = Path(__file__).resolve().with_name("postprocess.sh")
    os.execvp("bash", ["bash", str(launcher), *sys.argv[1:]])
