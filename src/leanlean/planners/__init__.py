"""Planner implementations for LeanLean."""

import copy
import importlib

from leanlean import Planner

_PLANNER_MAPPING = {
    "no_plan": "leanlean.planners.no_plan.NoPlanPlanner",
}


def get_planner_class(spec: str) -> type[Planner]:
    full_path = _PLANNER_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Unknown planner type: {spec} (resolved to {full_path}, available: {_PLANNER_MAPPING})"
        more = f" ({e})" if str(e) else ""
        raise ValueError(msg + more) from e


def get_planner(planner_config: dict) -> Planner:
    config = copy.deepcopy(planner_config)
    planner_class = config.pop("planner_class", None)
    assert planner_class is not None, "planner_class must be specified in planner_config"
    return get_planner_class(planner_class)(**config)
