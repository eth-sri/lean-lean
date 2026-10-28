"""Dataset-owned archives for immutable shared Lean environment images."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ARCHIVE_FORMAT = "docker_image_save_zstd_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_id_if_present(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def validated_archive(environment: Mapping[str, Any], *, repo_root: Path) -> Path:
    archive = environment.get("archive")
    if not isinstance(archive, Mapping):
        raise ValueError("shared environment has no dataset-local archive")
    if archive.get("format") != ARCHIVE_FORMAT:
        raise ValueError("shared environment archive format drift")
    relative = archive.get("path")
    expected = archive.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("shared environment archive path is missing")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("shared environment archive hash is invalid")
    path = (repo_root / relative).resolve()
    if not path.is_relative_to((repo_root / "datasets").resolve()):
        raise ValueError("shared environment archive escaped datasets/")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"shared environment archive drift: {path}")
    return path


def ensure_environment_image(
    environment: Mapping[str, Any], *, repo_root: Path
) -> str:
    """Return the exact image ID, importing its dataset archive if absent."""

    image = str(environment.get("image") or "")
    expected = str(environment.get("image_id") or "")
    if not image or not expected:
        raise ValueError("shared environment image pin is incomplete")
    present = image_id_if_present(image)
    if present:
        if present != expected:
            raise ValueError(f"shared environment image drift: {image}")
        return present
    archive = validated_archive(environment, repo_root=repo_root)
    decoder = subprocess.Popen(
        ["zstd", "--decompress", "--stdout", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    loaded = subprocess.run(
        ["docker", "image", "load"],
        stdin=decoder.stdout,
        capture_output=True,
        text=True,
    )
    decoder.stdout.close()
    decoder_stderr = decoder.stderr.read().decode() if decoder.stderr else ""
    decoder_returncode = decoder.wait()
    if decoder_returncode != 0 or loaded.returncode != 0:
        raise RuntimeError(
            "shared environment archive import failed: "
            + (decoder_stderr + loaded.stderr)[-2000:]
        )
    present = image_id_if_present(image)
    if present != expected:
        raise ValueError(f"imported shared environment image drift: {image}")
    return present
