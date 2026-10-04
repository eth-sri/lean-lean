"""Generator implementations for LeanLean."""

import copy
import importlib

from leanlean import Generator, Model, Environment

_GENERATOR_MAPPING = {
    "cli_agent": "leanlean.generators.cli_agent.CLIAgent",
}


def get_generator_class(spec: str) -> type[Generator]:
    full_path = _GENERATOR_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown generator type: {spec} (resolved to {full_path}, available: {_GENERATOR_MAPPING})"
        raise ValueError(msg)


def get_generator(generator_config: dict, *, model: Model, env: Environment) -> Generator:
    config = copy.deepcopy(generator_config)
    generator_class = config.pop("generator_class", None)
    assert generator_class is not None, "generator_class must be specified in generator_config"
    return get_generator_class(generator_class)(**config, model=model, env=env)
