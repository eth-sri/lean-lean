# Forked from: https://github.com/SWE-agent/mini-swe-generator

import dataclasses
from leanlean.utils import json_utils as json
from pathlib import Path
from typing import Any

from leanlean import Generator
from leanlean.utils.log import logger

def _get_class_name_with_module(obj: Any) -> str:
    """Get the full class name with module path."""
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

def _asdict(obj: Any) -> dict:
    """Convert config objects to dicts."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)  # type: ignore[arg-type]
    return obj  # let's try our luck

def _remove_api_key(d: dict, field: str = "api_key") -> dict:
    """Recursively remove any keys that look like API keys."""
    if isinstance(d, dict):
        return {k: _remove_api_key(v, field) for k, v in d.items() if field not in k.lower()}
    elif isinstance(d, list):
        return [_remove_api_key(item, field) for item in d]
    else:
        return d

def save_traj(
    generator: Generator | None,
    path: Path,
    *,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    messages: list[dict[str, Any]] | None = None,
    responses: list[dict[str, Any]] | None = None,
    trajectory_format: str | None = None,
    **kwargs,
):
    """Save the trajectory of the generator to a file.

    Args:
        generator: The generator to save the trajectory of.
        path: The path to save the trajectory to.
        print_path: Whether to print confirmation of path to the terminal.
        exit_status: The exit status of the generator.
        result: The result/submission of the generator.
        extra_info: Extra information to save (will be merged into the info dict).
        messages: Optional explicit messages to persist instead of generator messages.
        responses: Optional explicit responses to persist instead of generator responses.
        trajectory_format: Optional explicit trajectory format name.
        **kwargs: Additional information to save (will be merged into top level)

    """
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": 0.0,
                "api_calls": 0,
            },
        },
        "messages": [],
        "trajectory_format": "mini-swe-generator-1",
        "responses": [],
    } | kwargs
    if generator is not None:
        data["info"]["model_stats"]["instance_cost"] = generator.model.cost
        data["info"]["model_stats"]["api_calls"] = generator.model.n_calls
        data["messages"] = generator.messages
        data["info"]["config"] = {
            "generator": _asdict(generator.config),
            "model": _asdict(generator.model.config),
            "environment": _asdict(generator.env.config),
            "generator_type": _get_class_name_with_module(generator),
            "model_type": _get_class_name_with_module(generator.model),
            "environment_type": _get_class_name_with_module(generator.env),
        }
        data["responses"] = generator.model.responses
    if messages is not None:
        data["messages"] = messages
    if responses is not None:
        data["responses"] = responses
    if trajectory_format is not None:
        data["trajectory_format"] = trajectory_format
    if extra_info:
        data["info"].update(extra_info)

    #data = _remove_api_key(data)

    logger.debug(f"Saving {path} with data: {data}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _compute_run_directory(
    *,
    output_dir: Path,
    dataset_name: str,
    plan_type: str,
    generator: str,
    exec_model: str,
    run_id: str | int,
    planner_config: dict[str, Any],
    kind: str,
    train_plan: bool = False,
    continuous_training: bool = False,
    task_variant: str | None = None,
) -> Path:
    dataset_segment = dataset_name.replace("/", "_")

    if train_plan:
        assert kind == "leanlean", "Invalid kind for training plan"
        kind = "leanlean_training"

    # task_variant replaces plan_type as the varying dimension when set
    plan_segment = task_variant if task_variant is not None else plan_type
    run_dir = output_dir / kind / dataset_segment / plan_segment

    if train_plan:
        return run_dir / generator / exec_model

    return run_dir / generator / exec_model / f"run_{run_id}"
