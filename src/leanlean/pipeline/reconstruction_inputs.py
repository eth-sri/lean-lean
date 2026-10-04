"""Read pinned submitted generations without rewriting their run history."""

import json
from typing import Any, Mapping

import yaml

from leanlean.pipeline.reconstruction import _mapping, _path
from leanlean.preprocessing.repositories import file_sha256


def generation_artifacts(manifest: Mapping[str, Any], stage: str):
    config = manifest[stage]
    run_path = _path(config["run_artifact"], f"{stage}.run_artifact")
    run = _mapping(yaml.safe_load(run_path.read_text()), "run")
    reuse = stage == "compression" and config.get("mode") == "reuse_submitted_generation"
    if run.get("status") != "complete" and not reuse:
        raise RuntimeError(f"{stage} generation is not complete: {run.get('status')}")
    artifacts = _mapping(run["artifacts"], "artifacts")
    output = _path(artifacts["output_directory"], "output directory")
    predictions_path = _path(artifacts["predictions"], "predictions")
    predictions = _mapping(json.loads(predictions_path.read_text()), "predictions")
    if reuse:
        if config.get("predictions_sha256") != file_sha256(predictions_path):
            raise ValueError("saved compression predictions drifted")
        instance_id = manifest["preprocessing"]["repository"]
        if set(predictions) != {instance_id}:
            raise ValueError("saved compression repository set differs")
        if not isinstance(predictions[instance_id].get("model_patch"), str):
            raise ValueError("saved compression has no patch")
        generation = json.loads((output / instance_id / "round_0_gen_metrics.json").read_text())
        if generation.get("exit_status") != "Submitted":
            raise ValueError("saved compression was not submitted successfully")
        from leanlean.subscription_status import classify_trajectory
        trajectory = json.loads((output / instance_id / f"{instance_id}.traj.json").read_text())
        if trajectory.get("info", {}).get("exit_status") != "Submitted" or classify_trajectory(trajectory) is not None:
            raise ValueError("saved compression has a provider failure or incomplete trajectory")
        source = _path(run["dataset"]["manifest"], "source dataset")
        target = _path(manifest["preprocessing"]["dataset"], "expected dataset")
        if source != target:
            if "dataset_equivalence" not in config:
                raise ValueError("saved compression dataset differs")
            from leanlean.pipeline.reconstruction import REPO_ROOT
            from leanlean.pipeline.reconstruction_equivalence import validate_equivalence
            validate_equivalence(REPO_ROOT, config["dataset_equivalence"], source, target, instance_id)
        for field in ("model", "reasoning_effort"):
            if run["execution"].get(field) != manifest[field]:
                raise ValueError(f"saved compression {field} differs")
    return output, predictions
