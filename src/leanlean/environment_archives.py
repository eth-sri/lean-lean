"""Dataset-owned archives for immutable shared Lean environment images."""

from __future__ import annotations

import hashlib
import os
import shutil
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


def export_environment_archive(
    environment: Mapping[str, Any], *, destination: Path, compression_threads: int
) -> dict[str, Any]:
    """Export one pinned image to a compressed, atomically published archive."""

    image = str(environment.get("image") or "")
    expected = str(environment.get("image_id") or "")
    if image_id_if_present(image) != expected:
        raise ValueError(f"shared environment image drift: {image}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    saved = subprocess.Popen(
        ["docker", "image", "save", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert saved.stdout is not None
    compressed = subprocess.run(
        [
            "zstd",
            f"-T{compression_threads}",
            "-3",
            "--force",
            "-o",
            str(temporary),
        ],
        stdin=saved.stdout,
        capture_output=True,
        text=True,
    )
    saved.stdout.close()
    if compressed.returncode != 0:
        saved.terminate()
    saved_returncode = saved.wait()
    saved_stderr = saved.stderr.read().decode() if saved.stderr else ""
    if saved_returncode != 0 or compressed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "shared environment archive export failed: "
            + (saved_stderr + compressed.stderr)[-2000:]
        )
    temporary.replace(destination)
    return {
        "format": ARCHIVE_FORMAT,
        "path": str(destination),
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
    }


def localize_environment_archive(
    environment: Mapping[str, Any], *, destination: Path, repo_root: Path
) -> dict[str, Any]:
    """Place a verified environment archive inside another dataset atomically.

    A hard link avoids duplicating multi-gigabyte archives on the same filesystem;
    unlike a symlink it remains a complete dataset-owned file if the source dataset
    is later removed. Cross-filesystem publication falls back to a byte copy.
    """

    source = validated_archive(environment, repo_root=repo_root)
    archive = environment["archive"]
    expected = str(archive["sha256"])
    destination = destination.resolve()
    if not destination.is_relative_to((repo_root / "datasets").resolve()):
        raise ValueError("localized shared environment escaped datasets/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        pass
    elif destination.is_file() and file_sha256(destination) == expected:
        pass
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        linked = False
        try:
            os.link(source, temporary)
            linked = True
        except OSError:
            shutil.copy2(source, temporary)
        if not linked and file_sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError("localized shared environment archive hash drift")
        temporary.replace(destination)
    return {
        "format": ARCHIVE_FORMAT,
        "path": destination.relative_to(repo_root).as_posix(),
        "sha256": expected,
        "bytes": destination.stat().st_size,
    }
