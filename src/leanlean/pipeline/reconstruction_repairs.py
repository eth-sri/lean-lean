"""Explicit build-config reversions applied after a saved model submission."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


def repair_compression_patch(patch: str, repair: Mapping[str, Any] | None) -> str:
    """Revert only named Lake config files; preserve every Lean-source diff byte."""
    if repair is None:
        return patch
    if set(repair) != {'policy', 'original_patch_sha256', 'revert_files', 'reason'}:
        raise ValueError('build config repair must pin its policy, source patch, files and reason')
    if repair['policy'] != 'revert_build_config_files':
        raise ValueError('unsupported build config repair policy')
    if hashlib.sha256(patch.encode()).hexdigest() != repair['original_patch_sha256']:
        raise ValueError('build config repair source patch drift')
    files = repair['revert_files']
    if (not isinstance(files, list) or not files or len(set(files)) != len(files)
            or not set(files) <= {'lakefile.toml', 'lakefile.lean'}
            or not isinstance(repair['reason'], str) or not repair['reason'].strip()):
        raise ValueError('repair may revert only root Lake build configuration files')
    sections = re.split(r'(?m)(?=^diff --git )', patch)
    kept, removed = [], set()
    for section in sections:
        first = section.splitlines()[0] if section else ''
        match = re.fullmatch(r'diff --git a/(lakefile\.(?:toml|lean)) b/\1', first)
        if match and match[1] in files:
            removed.add(match[1])
        else:
            kept.append(section)
    if removed != set(files):
        raise ValueError('requested build config repair is absent from the source patch')
    return ''.join(kept)
