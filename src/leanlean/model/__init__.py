from __future__ import annotations

import importlib

from leanlean import Model


def get_model(config: dict | None = None) -> Model:
    """Get an initialized model object from any kind of user input or settings."""
    config = dict(config or {})
    model_class = config.pop("model_class", "")
    return get_model_class(model_class)(**config)

_MODEL_CLASS_MAPPING = {
    "litellm_server": "leanlean.model.litellm_wrapper.litellm_server.LitellmServer",
}

def get_model_class(model_class: str = "") -> type:

    full_path = _MODEL_CLASS_MAPPING.get(model_class, model_class)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Unknown model class: {model_class} (resolved to {full_path}, available: {_MODEL_CLASS_MAPPING})"
        raise ValueError(msg) from e
