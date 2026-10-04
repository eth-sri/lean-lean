#!/usr/bin/env python3
"""Pick one edit checkpoint per fixed wall-clock window for sparse replay builds.

For every repository capture in an evaluation output directory, the agent's
elapsed time is split into windows of ``--window-minutes``.  The last edit in
each window is kept: it is the source state at the end of that window.  The
result is an index list consumed by ``postprocess.py --checkpoint-index-file``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_edit_indices(points: list[dict], window_seconds: float) -> list[int]:
    last_in_window: dict[int, int] = {}
    for point in points:
        edit_index = int(point.get("edit_index") or 0)
        if edit_index <= 0 or point.get("endpoint_role") == "submitted_patch":
            continue
        window = int(float(point.get("elapsed_seconds") or 0.0) // window_seconds)
        last_in_window[window] = max(last_in_window.get(window, 0), edit_index)
    return sorted(set(last_in_window.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="evaluation output directory")
    parser.add_argument("--window-minutes", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    window_seconds = args.window_minutes * 60
    repositories: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    for repo_dir in sorted(args.output_dir.glob("*/playback")):
        captures = sorted(repo_dir.glob("capture_*/playback.json"))
        if not captures:
            continue
        # The newest capture is the one postprocessing scores.
        points = json.loads(captures[-1].read_text()).get("points") or []
        repository = repo_dir.parent.name
        repositories[repository] = select_edit_indices(points, window_seconds)
        totals[repository] = sum(
            1
            for point in points
            if int(point.get("edit_index") or 0) > 0
            and point.get("endpoint_role") != "submitted_patch"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "leanlean_checkpoint_window_selection_v1",
                "source_output_directory": str(args.output_dir),
                "window_minutes": args.window_minutes,
                "policy": "last_edit_in_each_elapsed_time_window",
                "repositories": repositories,
            },
            indent=2,
        )
        + "\n"
    )
    selected = sum(len(indices) for indices in repositories.values())
    print(
        f"{len(repositories)} repositories: {selected} of {sum(totals.values())} "
        f"edit checkpoints selected ({args.window_minutes:g}-minute windows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
