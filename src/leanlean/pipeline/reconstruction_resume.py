"""Validate explicitly pinned compression checks when resuming preparation."""

import json

from leanlean.pipeline.reconstruction import _path
from leanlean.preprocessing.repositories import file_sha256


def verified_compression_result(manifest, instance_id, patch, *, require_clean_build=False):
    pin = manifest["compression"].get("verification_report_sha256")
    if pin is None:
        return None
    root = _path(manifest["outputs"]["directory"], "outputs.directory")
    report_path = root / "compression-score" / "round_0" / instance_id / "report.json"
    if file_sha256(report_path) != pin:
        raise ValueError("pinned compression verification report drifted")
    result = json.loads(report_path.read_text())[instance_id]
    build = result.get("build_result", {})
    verification = build.get("compression_verification", {})
    if require_clean_build and build.get("project_cache_policy") != "clean_project_preserve_dependencies":
        raise ValueError("pinned verification used project caches; choose a new run_id for a clean build")
    if not (
        result.get("resolved") is True
        and result.get("model_patch") == patch
        and result.get("apply_patch_result", {}).get("returncode") == 0
        and build.get("passed") is True
        and verification.get("passed") is True
        and verification.get("returncode") == 0
        and "Lean default kernel accepts the solution" in verification.get("output", "")
    ):
        raise ValueError("pinned compression verification did not pass for this patch")
    return {"instance_id": instance_id, "resolved": True, "reused_pinned_verification": True}
