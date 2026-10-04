"""Independent compile/signature/axiom scoring for theorem A/B runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from leanlean.identifiers import same_identifier

from leanlean.benchmarks.leanlean import LeanLeanInstance
from leanlean.dataset_bundle import load_dataset
from leanlean.preprocessing.repositories import load_repository_database
from leanlean.preprocessing.standardized_repositories import load_standardized_repository_database
from leanlean.pipeline.reconstruction import (
    REPO_ROOT, _atomic_text, _mapping, _materialized_source_image, _path,
)


def _run_artifacts(manifest: Mapping[str, Any], stage: str):
    from leanlean.pipeline.reconstruction_inputs import generation_artifacts
    return generation_artifacts(manifest, stage)


def _repositories(dataset):
    record = dataset["repository_database"]
    path = _path(record["path"], "repository database")
    kind = json.loads(path.read_text())["kind"]
    loader = (
        load_standardized_repository_database
        if same_identifier(kind, "leanlean_standardized_repository_database")
        else load_repository_database
    )
    database = loader(path, repo_root=REPO_ROOT)
    if database.sha256 != record["sha256"] or database.definition_sha256 != record["definition_sha256"]:
        raise ValueError("scoring repository database drift")
    return database.by_id()


def _instance(manifest, repository, image: str) -> LeanLeanInstance:
    resources = manifest["container"]
    provenance = repository.protected_provenance
    verifier = None
    if provenance["kind"] == "leanlean_theorem_holdout_v1":
        from leanlean.palomar_comparator import load_palomar_evidence, resolve_palomar_contract
        verifier = resolve_palomar_contract(repo_root=REPO_ROOT, database_row={
            "instance_id": repository.instance_id,
            "protected": {"declarations": sorted(repository.protected_declarations), "provenance": provenance},
        })
        challenge, config, _ = load_palomar_evidence(repo_root=REPO_ROOT, contract=verifier)
        evidence = _path(manifest["outputs"]["directory"], "outputs.directory") / "compression-comparator"
        _atomic_text(evidence / "Challenge.lean", challenge.decode())
        _atomic_text(evidence / "comparator.json", config.decode())
        _atomic_text(evidence / "contract.json", json.dumps(verifier, indent=2, sort_keys=True) + "\n")
    if provenance["kind"] == "leanlean_theorem_reconstruction_v1":
        declarations = sorted(repository.protected_declarations)
        if len(declarations) != 1:
            raise ValueError("reconstruction needs exactly one protected theorem")
        verifier = {
            "schema": "leanlean_theorem_reconstruction_verifier_v1",
            "module": provenance["module"],
            "declaration": declarations[0],
            "target_file": provenance["target_file"],
            "build_target": provenance["build_target"],
            "build_jobs": int(resources["build_jobs"]),
            "permitted_axioms": list(provenance["permitted_axioms"]),
            "edit_policy": manifest["evaluation"].get(
                "reproof_edit_policy", provenance.get("edit_policy", "target_file_only")
            ),
        }
    instance = LeanLeanInstance(
        instance_id=repository.instance_id,
        repo_url=repository.repository_url,
        commit=repository.commit,
        exclude_dirs=list(repository.exclude_dirs),
        target_dir=repository.target_dir,
        filesystem_isolated=True,
        clean_project_cache=True,
        build_target=repository.build_target,
        required_signatures=sorted(repository.protected_declarations),
        required_signatures_source=provenance["kind"],
        agent_verifier=verifier,
        docker_image=image,
        run_tag=str(manifest["run_id"]) + "-score",
        build_jobs=int(resources["build_jobs"]),
        container_cpus=int(resources["cpus"]),
        container_memory=str(resources["memory"]),
        container_cgroup_parent=str(resources["cgroup_parent"]),
        container_timeout=f"{int(manifest['timeout'])}s",
        network_policy=str(resources["network_policy"]),
        container_pids_limit=int(resources["pids_limit"]),
    )
    if instance.docker_image != image:
        raise ValueError("environment image override conflicts with pinned scoring image")
    return instance


def _score_instance(instance, patch: str, output: Path) -> dict[str, Any]:
    round_dir = output / "round_0"
    ok = instance.compile_rounds(
        [(0, patch, round_dir)], output, 0,
        run_repo_test=True, measure_heartbeats=False,
    )
    if not ok:
        raise RuntimeError(f"scoring infrastructure failed for {instance.instance_id}; see execution.log")
    resolved = instance.analyze_round(round_dir, 0)
    return {"instance_id": instance.instance_id, "resolved": resolved}


def score(manifest: Mapping[str, Any], *, compression: bool = False) -> dict[str, Any]:
    stage = "compression" if compression else "evaluation"
    output, predictions = _run_artifacts(manifest, stage)
    if compression:
        output = _path(manifest["outputs"]["directory"], "outputs.directory") / "compression-score"
    dataset_path = (
        manifest["preprocessing"]["dataset"] if compression
        else manifest["outputs"]["dataset"]
    )
    dataset = load_dataset(_path(dataset_path, "scoring dataset"))
    repositories = _repositories(dataset)
    variants = {row["id"]: row["variants"]["stripped"] for row in dataset["repositories"]}

    def execute(instance_id: str):
        prediction = _mapping(predictions.get(instance_id), "prediction")
        patch = prediction.get("model_patch")
        if not isinstance(patch, str):
            raise ValueError(f"{instance_id}: missing model patch")
        if compression:
            from leanlean.pipeline.reconstruction_repairs import repair_compression_patch
            patch = repair_compression_patch(patch, manifest["compression"].get("build_config_repair"))
        with _materialized_source_image(manifest, instance_id, variants[instance_id]) as image:
            instance = _instance(manifest, repositories[instance_id], image)
            if compression:
                from leanlean.pipeline.reconstruction_resume import verified_compression_result
                verified = verified_compression_result(manifest, instance_id, patch, require_clean_build=True)
                if verified is not None:
                    return verified
            return _score_instance(instance, patch, output)

    ids = (
        [str(manifest["preprocessing"]["repository"])] if compression
        else [str(manifest["arms"][arm]) for arm in ("stripped", "compressed")]
    )
    with ThreadPoolExecutor(max_workers=min(len(ids), int(manifest["parallelism"]["workers"]))) as pool:
        results = list(pool.map(execute, ids))
    # An invalid compression is not a valid treatment. Unsolved reproof arms,
    # however, are legitimate experimental outcomes and must be summarized.
    if compression and not all(row["resolved"] for row in results):
        raise RuntimeError("compression did not preserve its protected results")
    return {"stage": stage, "results": results}
