"""Semantic repository variants for the LeanLean benchmark."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


REPO_VARIANTS = ("raw", "stripped", "optimized")
REPO_VARIANT_SUFFIX = {
    "raw": "",
    "stripped": "-stripped",
    "optimized": "-optimized",
}
REPO_IMAGE_OVERRIDES_ENV = "LEANLEAN_REPO_IMAGE_OVERRIDES"


def validate_repo_variant(value: Any) -> str:
    """Return a normalized repository variant or raise a useful error."""

    if not isinstance(value, str):
        raise ValueError(
            "repo_variant must be one of: " + ", ".join(REPO_VARIANTS)
        )
    variant = value.strip().lower()
    if variant not in REPO_VARIANT_SUFFIX:
        raise ValueError(
            f"unknown repo_variant {value!r}; expected one of: "
            + ", ".join(REPO_VARIANTS)
        )
    return variant


def repo_variant_from_spec(
    spec: Mapping[str, Any],
    *,
    default: str = "optimized",
) -> str:
    """Resolve the new ternary field with legacy ``use_prod`` compatibility.

    Historical ``use_prod: true`` manifests mean the current production
    baseline, which is now the optimized variant. ``false`` continues to mean
    raw. New manifests should use only ``repo_variant``.
    """

    explicit = spec.get("repo_variant")
    legacy_present = "use_prod" in spec
    if explicit is None:
        if legacy_present:
            legacy = spec["use_prod"]
            if not isinstance(legacy, bool):
                raise ValueError("use_prod must be true or false")
            return "optimized" if legacy else "raw"
        return validate_repo_variant(default)

    variant = validate_repo_variant(explicit)
    if legacy_present:
        legacy = spec["use_prod"]
        if not isinstance(legacy, bool):
            raise ValueError("use_prod must be true or false")
        legacy_variant = "optimized" if legacy else "raw"
        if variant != legacy_variant:
            raise ValueError(
                f"conflicting repository selectors: repo_variant={variant!r} "
                f"but use_prod={legacy!r} maps to {legacy_variant!r}"
            )
    return variant


def repo_variant_from_env(
    env: Mapping[str, str] | None = None,
    *,
    default: str = "optimized",
) -> str:
    """Resolve the repository variant from process environment variables."""

    values = os.environ if env is None else env
    explicit = values.get("LEANLEAN_REPO_VARIANT")
    if explicit is not None:
        return validate_repo_variant(explicit)
    legacy = values.get("LEANLEAN_USE_PROD")
    if legacy is not None:
        return "raw" if legacy == "0" else "optimized"
    return validate_repo_variant(default)


def repo_image_tag(instance_id: str, variant: str) -> str:
    """Return the semantic image tag for one LeanLean repository."""

    normalized = validate_repo_variant(variant)
    return (
        f"leanlean-{instance_id}"
        f"{REPO_VARIANT_SUFFIX[normalized]}:latest"
    )


def repo_image_overrides_from_env(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return per-repository immutable image references for this run."""

    values = os.environ if env is None else env
    raw = values.get(REPO_IMAGE_OVERRIDES_ENV)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {REPO_IMAGE_OVERRIDES_ENV} JSON") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(repo, str)
        and repo
        and isinstance(image, str)
        and image
        for repo, image in parsed.items()
    ):
        raise ValueError(
            f"{REPO_IMAGE_OVERRIDES_ENV} must map repository names to images"
        )
    return parsed
