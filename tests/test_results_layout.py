"""results/<dataset>/<provider>/<model>/: where a pair's files go and how stages are kept."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from leanlean.pipeline.evaluation import pair_directory
from leanlean.pipeline.postprocessing import (
    HEARTBEATS_RESULT,
    _postprocessing_stage_directory,
    _retire_superseded_definition,
    _write_repository_results,
)


def test_pair_directory(tmp_path):
    assert pair_directory(tmp_path, Path("demo/openai/gpt-5.6-luna-xhigh")) == (
        tmp_path / "results/demo/openai/gpt-5.6-luna-xhigh"
    )
    with pytest.raises(ValueError):
        pair_directory(tmp_path, Path("../demo/openai/x"))


def _stage(tmp_path, postprocess_id, status):
    stage = _postprocessing_stage_directory(tmp_path, "endpoints")
    stage.mkdir(parents=True)
    (stage / "manifest.yaml").write_text(yaml.safe_dump({"postprocess_id": postprocess_id}))
    (stage / "run.yaml").write_text(yaml.safe_dump({"postprocess_id": postprocess_id, "status": status}))
    return stage


def test_superseded_definition_moves_to_previous(tmp_path):
    stage = _stage(tmp_path, "old", "complete")
    _retire_superseded_definition(stage, "old")  # same definition: untouched
    assert (stage / "manifest.yaml").exists()
    _retire_superseded_definition(stage, "new")
    assert not (stage / "manifest.yaml").exists()
    assert yaml.safe_load((stage / "previous/old/run.yaml").read_text())["status"] == "complete"


def test_running_definition_is_not_moved(tmp_path):
    stage = _stage(tmp_path, "old", "running")
    with pytest.raises(RuntimeError, match="still running"):
        _retire_superseded_definition(stage, "new")
    assert (stage / "manifest.yaml").exists()


def test_unknown_stage(tmp_path):
    with pytest.raises(ValueError):
        _postprocessing_stage_directory(tmp_path, "revisions")


def _latest(resolved):
    return {
        "resolved": resolved,
        "build_passed": resolved,
        "signatures_preserved": resolved,
        "evaluation_infrastructure_ok": True,
        "empty_patch": False,
        "metrics": {"baseline_lean_tokens": 1000, "post_lean_tokens": 600},
        "cost_total": 1.5,
        "report_path": None,
    }


def test_repository_results(tmp_path):
    repo = tmp_path / "palomar__2026-08-20-000004"
    (repo / "derived").mkdir(parents=True)
    (repo / HEARTBEATS_RESULT).write_text("{}")
    (repo / "round_0_gen_metrics.json").write_text(
        json.dumps({"exit_status": "Submitted", "wall_time_seconds": 120.0})
    )
    prepared = SimpleNamespace(
        output_dir=tmp_path,
        manifest_path=tmp_path / "postprocess/heartbeats/manifest.yaml",
        manifest={"model": "gpt-5.6-luna", "reasoning_effort": "xhigh", "generator": "codex_sub",
                  "dataset": {"id": "demo"}, "postprocess_id": "pid"},
    )
    instances = {
        repo.name: {"latest": _latest(True)},
        "palomar__2026-08-29-000003": {"latest": _latest(False)},
    }
    states = {
        repo.name: {"eligible": True, "lean_token_compression_pct": 40.0, "word_compression_pct": 38.0},
        # A failed submission's state already scores 0.
        "palomar__2026-08-29-000003": {"eligible": False, "lean_token_compression_pct": 0.0},
    }
    _write_repository_results(prepared, instances, states)

    passed = json.loads((repo / "result.json").read_text())
    assert passed["status"] == "passed"
    assert passed["lean_token_compression_pct"] == 40.0
    assert passed["agent_exit_status"] == "Submitted"
    assert passed["agent_seconds"] == 120.0
    assert passed["files"] == {"postprocessed_playback": None, "heartbeats": HEARTBEATS_RESULT, "lake_build": None}
    assert passed["postprocessing"] == {"stage": "heartbeats", "postprocess_id": "pid"}

    failed = json.loads((tmp_path / "palomar__2026-08-29-000003/result.json").read_text())
    assert failed["status"] == "failed"
    assert failed["lean_token_compression_pct"] == 0.0
    assert failed["agent_seconds"] is None
