"""Derive a theorem-holdout Challenge from the immutable parent comparator."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping

from leanlean.metrics.tokens import remove_lean_comments
from leanlean.preprocessing.theorem_holdout import theorem_statement_prefix


def remove_challenge_theorems(source: str, removed: list[str]) -> str:
    """Remove only exact named sorry declarations, preserving other statements."""
    if not removed or len(removed) != len(set(removed)):
        raise ValueError("holdout declarations must be unique and nonempty")
    masked = remove_lean_comments(source, mask_strings=True)
    commands = re.compile(
        r"(?m)^[ \t]*(?:noncomputable[ \t]+)?(?P<kind>namespace|section|end|theorem|lemma)\b"
        r"(?:[ \t]+(?P<name>[^\s({:\[=]+))?"
    )
    namespace = ""
    scopes: list[str] = []
    spans = []
    found = []
    for command in commands.finditer(masked):
        kind, name = command.group("kind", "name")
        if kind in {"namespace", "section"}:
            scopes.append(namespace)
            if kind == "namespace":
                if not name:
                    raise ValueError("unnamed Challenge namespace")
                namespace = ".".join(filter(None, (namespace, name)))
            continue
        if kind == "end":
            if not scopes:
                raise ValueError("unbalanced Challenge scopes")
            namespace = scopes.pop()
            continue
        if not name:
            raise ValueError("unnamed Challenge theorem")
        qualified = name.removeprefix("_root_.") if name.startswith("_root_.") else ".".join(filter(None, (namespace, name)))
        if qualified not in removed:
            continue
        start = command.start()
        prefix = theorem_statement_prefix(source[start:])
        proof_start = start + len(prefix)
        sorry = re.match(r"\s*:=\s*(?:by\s+)?sorry\b", masked[proof_start:])
        if sorry is None:
            raise ValueError(f"{qualified}: Challenge proof is not a simple sorry")
        end = proof_start + sorry.end()
        if masked[end:masked.find("\n", end) if "\n" in masked[end:] else len(masked)].strip():
            raise ValueError(f"{qualified}: unsupported trailing Challenge command")
        before = source[:start].rstrip()
        if before.endswith("-/"):
            doc = before.rfind("/--")
            if doc >= 0 and before.find("-/", doc) == len(before) - 2:
                start = doc
        spans.append((start, end))
        found.append(qualified)
    if sorted(found) != sorted(removed):
        raise ValueError("held-out Challenge theorem did not resolve exactly once")
    for start, end in reversed(spans):
        source = source[:start] + source[end:]
    return source


def derive_evidence(challenge: bytes, configuration: bytes, removed: list[str]):
    config = json.loads(configuration)
    names = config.get("theorem_names", [])
    if not set(removed) <= set(names) or len(names) != len(set(names)):
        raise ValueError("holdout is not a subset of the registered comparator theorems")
    source = remove_challenge_theorems(challenge.decode(), removed)
    config["theorem_names"] = [name for name in names if name not in removed]
    return source.encode(), (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()


CHALLENGE_CLOSURE_SCHEMA = "leanlean_challenge_closure_v1"
_CHALLENGE_WORKDIR = "/tmp/leanlean-holdout-challenge"



def resolve_holdout_contract(*, repo_root, database_row: Mapping[str, Any]):
    from leanlean.palomar_comparator import (
        load_palomar_evidence, resolve_palomar_contract, runtime_config_copy,
    )
    protected = database_row["protected"]
    provenance = protected["provenance"]
    removed = list(provenance["removed_theorems"])
    parent_provenance = provenance["parent"]
    original_names = parent_provenance.get("original_registered_declarations")
    if not isinstance(original_names, list):
        original_names = sorted(set(protected["declarations"]) | set(removed))
    if set(original_names) - set(removed) != set(protected["declarations"]):
        raise ValueError("holdout remaining declarations differ from the parent contract")
    parent = resolve_palomar_contract(repo_root=repo_root, database_row={
        "instance_id": database_row["instance_id"],
        "protected": {"declarations": original_names, "provenance": parent_provenance},
    })
    challenge, config, _ = load_palomar_evidence(repo_root=repo_root, contract=parent)
    challenge, config = derive_evidence(challenge, config, removed)
    result = copy.deepcopy(parent)
    result.pop("sha256")
    holdout = {
        "schema": "leanlean_comparator_holdout_v1",
        "removed_theorems": removed,
        "parent": parent,
    }
    closure = provenance.get("challenge_closure")
    if closure is not None:
        if not isinstance(closure, Mapping) or closure.get("schema") != CHALLENGE_CLOSURE_SCHEMA:
            raise ValueError("invalid holdout Challenge closure provenance")
        roots = closure.get("roots")
        expected_roots = sorted(
            set(json.loads(config).get("theorem_names") or [])
            | set(json.loads(config).get("definition_names") or [])
        )
        if roots != expected_roots:
            raise ValueError("holdout Challenge closure roots drifted")
        if closure.get("input_sha256") != hashlib.sha256(challenge).hexdigest():
            raise ValueError("holdout Challenge closure input drifted")
        artifact = (repo_root / str(closure.get("path") or "")).resolve()
        root = repo_root.resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError("holdout Challenge closure artifact is missing")
        challenge = artifact.read_bytes()
        if hashlib.sha256(challenge).hexdigest() != closure.get("output_sha256"):
            raise ValueError("holdout Challenge closure artifact drifted")
        holdout["challenge_closure"] = copy.deepcopy(dict(closure))
    result["holdout"] = holdout
    result["challenge"]["sha256"] = hashlib.sha256(challenge).hexdigest()
    result["configuration"]["sha256"] = hashlib.sha256(config).hexdigest()
    runtime, defaults = runtime_config_copy(config)
    result["configuration"]["runtime_copy"] = (
        "registered_config_with_documented_defaults" if defaults
        else "byte_for_byte_registered_config"
    )
    if defaults:
        result["configuration"]["runtime_defaults_applied"] = defaults
        result["configuration"]["runtime_sha256"] = hashlib.sha256(runtime).hexdigest()
    result["theorem_names"] = json.loads(config)["theorem_names"]
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result
