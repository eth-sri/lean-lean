#!/usr/bin/env python3
"""Normalize a frozen Palomar registry snapshot into benchmark instances.

The input records and source archives are downloaded separately. This script
does no network access. Each registry result becomes one benchmark instance,
whose entry source is the record's exact ``formalization.solution_path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from leanlean.metrics.source_archive import count_lean_words_in_source


TIER_ORDER = ("compact", "standard", "large", "massive")
SAFE_MODULE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*"
)


COMPARATOR_REQUIRED_KEYS = frozenset(
    {
        "challenge_module",
        "solution_module",
        "theorem_names",
        "permitted_axioms",
    }
)
COMPARATOR_OPTIONAL_KEYS = frozenset(
    {"definition_names", "enable_nanoda"}
)
PALOMAR_PERMITTED_AXIOMS = frozenset(
    {"propext", "Quot.sound", "Classical.choice"}
)


def _name_array(value: Any, label: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        requirement = "a nonempty array" if nonempty else "an array"
        raise ValueError(f"{label} must be {requirement}")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"{label} entries must be nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicate names")
    return value


def _validate_comparator(
    comparator: dict[str, Any], *, label: str
) -> tuple[str, str, list[str], list[str], list[str]]:
    keys = set(comparator)
    missing = sorted(COMPARATOR_REQUIRED_KEYS - keys)
    extra = sorted(keys - COMPARATOR_REQUIRED_KEYS - COMPARATOR_OPTIONAL_KEYS)
    if missing or extra:
        raise ValueError(
            f"{label}: invalid Comparator keys: missing={missing}, extra={extra}"
        )

    solution_module = comparator["solution_module"]
    challenge_module = comparator["challenge_module"]
    if (
        not isinstance(solution_module, str)
        or not SAFE_MODULE.fullmatch(solution_module)
        or not isinstance(challenge_module, str)
        or not SAFE_MODULE.fullmatch(challenge_module)
    ):
        raise ValueError(f"{label}: invalid Comparator modules")

    theorem_names = _name_array(
        comparator["theorem_names"], f"{label}.theorem_names", nonempty=True
    )
    definition_names = _name_array(
        comparator.get("definition_names", []),
        f"{label}.definition_names",
        nonempty=False,
    )
    permitted_axioms = _name_array(
        comparator["permitted_axioms"],
        f"{label}.permitted_axioms",
        nonempty=False,
    )
    unsupported = sorted(set(permitted_axioms) - PALOMAR_PERMITTED_AXIOMS)
    if unsupported:
        raise ValueError(
            f"{label}: unsupported permitted axioms: {unsupported}"
        )
    if "enable_nanoda" in comparator and not isinstance(
        comparator["enable_nanoda"], bool
    ):
        raise ValueError(f"{label}.enable_nanoda must be a boolean")
    return (
        solution_module,
        challenge_module,
        theorem_names,
        definition_names,
        permitted_axioms,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "root"


def _tier(words: int) -> str:
    if words < 10_000:
        return "compact"
    if words < 50_000:
        return "standard"
    if words < 200_000:
        return "large"
    return "massive"


def _archive_path(archive_dir: Path, repository: str, commit: str) -> Path:
    owner, name = repository.split("/", 1)
    return archive_dir / f"{owner}--{name}--{commit}.tar.gz"


def _source_metrics(
    archive: Path,
    *,
    project_path: str,
    excluded_paths: set[str],
    required_paths: set[str],
) -> tuple[int, int, int]:
    words = 0
    lean_bytes = 0
    lean_file_count = 0
    seen_paths: set[str] = set()
    with tarfile.open(archive, "r:gz") as source_archive:
        for member in source_archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            path = PurePosixPath(*parts[1:]).as_posix()
            seen_paths.add(path)
            if not path.endswith(".lean") or ".lake" in PurePosixPath(path).parts:
                continue
            if project_path and not (
                path == project_path + ".lean"
                or path.startswith(project_path + "/")
            ):
                continue
            if path in excluded_paths:
                continue
            stream = source_archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"could not read {path!r} from {archive}")
            source = stream.read().decode("utf-8", errors="replace")
            words += count_lean_words_in_source(source)
            lean_bytes += member.size
            lean_file_count += 1
    missing = sorted(required_paths - seen_paths)
    if missing:
        raise ValueError(
            f"{archive}: registered paths are absent from the pinned source: "
            + ", ".join(missing)
        )
    return words, lean_bytes, lean_file_count


def _registry_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    path = PurePosixPath(value.strip("/"))
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} is unsafe: {value!r}")
    return path.as_posix()


def _project_relative_path(repository_path: str, project_path: str) -> str:
    if not project_path:
        return repository_path
    prefix = project_path.rstrip("/") + "/"
    if not repository_path.startswith(prefix):
        raise ValueError(
            f"registered path {repository_path!r} is outside project "
            f"{project_path!r}"
        )
    return repository_path[len(prefix) :]


def _archive_bytes(archive: Path, repository_path: str) -> bytes:
    matches: list[bytes] = []
    with tarfile.open(archive, "r:gz") as source_archive:
        for member in source_archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            path = PurePosixPath(*parts[1:]).as_posix()
            if path != repository_path:
                continue
            stream = source_archive.extractfile(member)
            if stream is None:
                raise ValueError(
                    f"{archive}: could not read registered path: {repository_path}"
                )
            matches.append(stream.read())
    if not matches:
        raise ValueError(f"{archive}: registered path is absent: {repository_path}")
    if len(matches) != 1:
        raise ValueError(
            f"{archive}: registered path occurs more than once: {repository_path}"
        )
    return matches[0]


def _archive_json(archive: Path, repository_path: str) -> dict[str, Any]:
    payload = json.loads(_archive_bytes(archive, repository_path).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{archive}:{repository_path}: expected JSON object")
    return payload


def _pinned_source_hash(
    record: dict[str, Any],
    *,
    name: str,
    content: bytes,
    label: str,
) -> str:
    verification = record.get("verification")
    if not isinstance(verification, dict):
        raise ValueError(f"{label}.verification must be an object")
    expected = verification.get(name)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label}.verification.{name} must be a SHA-256")
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{label}.verification.{name} does not match the pinned source"
        )
    return actual


def _load_records(
    recent: dict[str, Any], record_dir: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for projection in recent["entries"]:
        path = record_dir / Path(projection["path"]).name
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record["id"] != projection["id"]
            or record["version"] != projection["version"]
        ):
            raise ValueError(f"record/projection mismatch for {projection['id']}")
        records[projection["id"]] = record
    return records


def _build_mapping(
    recent_path: Path,
    record_dir: Path,
    archive_dir: Path,
    *,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    recent = json.loads(recent_path.read_text(encoding="utf-8"))
    records = _load_records(recent, record_dir)
    latest_published_at = max(
        projection["published_at"] for projection in recent["entries"]
    )
    snapshot_date = latest_published_at[:10]
    if snapshot_id and (
        match := re.match(r"(\d{4})(\d{2})(\d{2})", snapshot_id)
    ):
        snapshot_date = "-".join(match.groups())
    targets: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for projection in recent["entries"]:
        record = records[projection["id"]]
        source = record["source"]
        formalization = record["formalization"]
        repository = source["repository"]
        commit = source["commit"]
        project_path = (
            _registry_path(source["project_path"], "source.project_path")
            if source.get("project_path")
            else ""
        )
        solution_path = _registry_path(
            formalization["solution_path"], "formalization.solution_path"
        )
        challenge_path = _registry_path(
            formalization["challenge_path"], "formalization.challenge_path"
        )
        comparator_path = _registry_path(
            formalization["comparator_config_path"],
            "formalization.comparator_config_path",
        )
        solution_source = _project_relative_path(solution_path, project_path)
        challenge_source = _project_relative_path(challenge_path, project_path)
        _project_relative_path(comparator_path, project_path)
        archive = _archive_path(archive_dir, repository, commit)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        solution_bytes = _archive_bytes(archive, solution_path)
        challenge_bytes = _archive_bytes(archive, challenge_path)
        comparator_bytes = _archive_bytes(archive, comparator_path)
        comparator = json.loads(comparator_bytes.decode("utf-8"))
        if not isinstance(comparator, dict):
            raise ValueError(f"{archive}:{comparator_path}: expected JSON object")
        solution_sha256 = _pinned_source_hash(
            record,
            name="solution_sha256",
            content=solution_bytes,
            label=projection["id"],
        )
        challenge_sha256 = _pinned_source_hash(
            record,
            name="challenge_sha256",
            content=challenge_bytes,
            label=projection["id"],
        )
        comparator_sha256 = hashlib.sha256(comparator_bytes).hexdigest()
        (
            solution_module,
            challenge_module,
            theorem_names,
            definition_names,
            permitted_axioms,
        ) = _validate_comparator(
            comparator, label=f"{archive}:{comparator_path}"
        )
        if (
            theorem_names != formalization["theorem_names"]
            or definition_names != formalization.get("definition_names", [])
            or permitted_axioms != formalization["permitted_axioms"]
        ):
            raise ValueError(
                f"{archive}:{comparator_path}: registry/Comparator contract drift"
            )
        words, lean_bytes, lean_file_count = _source_metrics(
            archive,
            project_path=project_path,
            excluded_paths={challenge_path},
            required_paths={solution_path, challenge_path, comparator_path},
        )
        identifier = projection["id"]
        target = {
            "benchmark_id": f"palomar__{_slug(identifier.removeprefix('PALOMAR-'))}",
            "repository": repository,
            "commit": commit,
            "project_path": project_path or None,
            "palomar_ids": [identifier],
            "registered_solution_path": solution_path,
            "registered_solution_sha256": solution_sha256,
            "solution_source": solution_source,
            "registered_solution_module": solution_module,
            "registered_challenge_path": challenge_path,
            "registered_challenge_sha256": challenge_sha256,
            "challenge_source": challenge_source,
            "registered_challenge_module": challenge_module,
            "registered_comparator_config_path": comparator_path,
            "registered_comparator_config_sha256": comparator_sha256,
            "raw_words": words,
            "raw_lean_bytes": lean_bytes,
            "lean_file_count": lean_file_count,
            "size_tier": _tier(words),
            "excluded_challenge_paths": [challenge_path],
            "registered_theorem_names": sorted(formalization["theorem_names"]),
            "registered_definition_names": sorted(definition_names),
            "registered_permitted_axioms": sorted(permitted_axioms),
        }
        targets.append(target)
        record_path = record_dir / Path(projection["path"]).name
        results.append(
            {
                "palomar_id": projection["id"],
                "version": projection["version"],
                "record_path": projection["path"],
                "record_sha256": _sha256(record_path),
                "title": projection["title"],
                "published_at": projection["published_at"],
                "repository_role": record["provenance"]["repository_role"],
                "registry_repository": record["source"]["repository"],
                "registry_commit": record["source"]["commit"],
                "registry_project_path": record["source"].get("project_path"),
                "benchmark_id": target["benchmark_id"],
                "target_repository": target["repository"],
                "target_commit": target["commit"],
                "target_project_path": target["project_path"],
                "raw_words": target["raw_words"],
                "size_tier": target["size_tier"],
                "challenge_path": challenge_path,
                "challenge_sha256": challenge_sha256,
                "solution_path": solution_path,
                "solution_sha256": solution_sha256,
                "solution_source": solution_source,
                "solution_module": solution_module,
                "challenge_module": challenge_module,
                "comparator_config_path": formalization[
                    "comparator_config_path"
                ],
                "comparator_config_sha256": comparator_sha256,
                "lean_toolchain": formalization["lean_toolchain"],
                "theorem_names": formalization["theorem_names"],
                "definition_names": definition_names,
                "permitted_axioms": permitted_axioms,
            }
        )

    result_tiers = Counter(result["size_tier"] for result in results)
    target_tiers = Counter(target["size_tier"] for target in targets)
    return {
        "kind": "palomar_candidate_mapping",
        "schema_version": 3,
        "snapshot": {
            "id": snapshot_id,
            "date": snapshot_date,
            "registry_url": "https://data.palomar-registry.org/recent.json",
            "registry_sha256": _sha256(recent_path),
            "latest_published_at": latest_published_at,
        },
        "tiering": {
            "metric": "raw_words",
            "source_scope": (
                "normalized_isolated_raw_after_removing_palomar_challenge_files"
            ),
            "thresholds": {
                "compact": {"min_inclusive": 0, "max_exclusive": 10_000},
                "standard": {"min_inclusive": 10_000, "max_exclusive": 50_000},
                "large": {"min_inclusive": 50_000, "max_exclusive": 200_000},
                "massive": {"min_inclusive": 200_000, "max_exclusive": None},
            },
        },
        "summary": {
            "registered_results": len(results),
            "registry_repositories": len(
                {result["registry_repository"] for result in results}
            ),
            "thin_wrappers": sum(
                result["repository_role"] == "thin-wrapper" for result in results
            ),
            "target_repositories": len(
                {target["repository"] for target in targets}
            ),
            "benchmark_instances": len(targets),
            "unique_source_snapshots": len(
                {
                    (target["repository"], target["commit"], target["project_path"])
                    for target in targets
                }
            ),
            "theorem_names": sum(len(result["theorem_names"]) for result in results),
            "definition_names": sum(
                len(result["definition_names"]) for result in results
            ),
            "result_tiers": {tier: result_tiers[tier] for tier in TIER_ORDER},
            "target_tiers": {tier: target_tiers[tier] for tier in TIER_ORDER},
            "target_raw_words": sum(target["raw_words"] for target in targets),
        },
        "results": results,
        "targets": targets,
    }


def _github_tree(repository: str, commit: str, project_path: str | None) -> str:
    suffix = f"/{project_path}" if project_path else ""
    return f"https://github.com/{repository}/tree/{commit}{suffix}"


def _write_markdown(mapping: dict[str, Any], output: Path, json_path: Path) -> None:
    summary = mapping["summary"]
    result_tiers = summary["result_tiers"]
    target_tiers = summary["target_tiers"]
    try:
        json_link = json_path.relative_to(output.parent.parent).as_posix()
    except ValueError:
        json_link = json_path.as_posix()
    lines = [
        f"# Palomar candidate mapping ({mapping['snapshot']['date']})",
        "",
        (
            "This is a frozen mapping of the current Palomar result records to "
            f"candidate LeanLean targets. The machine-readable source of "
            f"truth is [`{json_path.as_posix()}`](../{json_link})."
        ),
        "",
        (
            "Each registry result is one benchmark instance. Its exact registered "
            "`formalization.solution_path` is the entry source, even when another "
            "result uses the same repository snapshot. Challenge files remain "
            "external verification evidence and are excluded from the editable "
            "source-size metric."
        ),
        "",
        "## Summary",
        "",
        "| Measure | Count |",
        "|---|---:|",
    ]
    measures = (
        ("Registered results", summary["registered_results"]),
        ("Registry repositories", summary["registry_repositories"]),
        ("Thin wrappers", summary["thin_wrappers"]),
        ("Registry source repositories", summary["target_repositories"]),
        ("Benchmark instances", summary["benchmark_instances"]),
        ("Unique source snapshots", summary["unique_source_snapshots"]),
        ("Registered theorem names", summary["theorem_names"]),
        ("Registered definition names", summary["definition_names"]),
        ("Combined instance raw words", summary["target_raw_words"]),
    )
    lines.extend(f"| {label} | {value:,} |" for label, value in measures)
    lines.extend(
        [
            "",
            "## Tier distribution",
            "",
            "| Unit | Compact | Standard | Large | Massive |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Result records | {result_tiers['compact']} | "
                f"{result_tiers['standard']} | {result_tiers['large']} | "
                f"{result_tiers['massive']} |"
            ),
            (
                f"| Benchmark instances | {target_tiers['compact']} | "
                f"{target_tiers['standard']} | {target_tiers['large']} | "
                f"{target_tiers['massive']} |"
            ),
            "",
            "## Result-to-repository mapping",
            "",
            (
                "| Palomar result | Registry source | Role | Registered solution | "
                "Tier | Raw words | Protected |"
            ),
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for result in mapping["results"]:
        registry_url = _github_tree(
            result["registry_repository"], result["registry_commit"], None
        )
        registry = f"[{result['registry_repository']}]({registry_url})"
        target_label = result["solution_path"]
        target_url = (
            f"https://github.com/{result['target_repository']}/blob/"
            f"{result['target_commit']}/{result['solution_path']}"
        )
        target = f"[`{target_label}`]({target_url})"
        protected = len(result["theorem_names"]) + len(result["definition_names"])
        lines.append(
            f"| {result['palomar_id']} v{result['version']} | {registry} | "
            f"{result['repository_role']} | {target} | {result['size_tier']} | "
            f"{result['raw_words']:,} | {protected} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark-instance policy",
            "",
            (
                "The JSON `targets` array has exactly one row per registry result. "
                "Each row includes the exact repository, commit, optional project "
                "path, registered Solution and Challenge paths, declaration names, "
                "source metrics, and tier. Source archives may be cached by commit, "
                "but evaluation instances are never merged."
            ),
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    mapping = _build_mapping(
        args.recent,
        args.records,
        args.archives,
        snapshot_id=args.snapshot_id,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(mapping, args.output_markdown, args.output_json)
    print(json.dumps(mapping["summary"], indent=2))


if __name__ == "__main__":
    main()
