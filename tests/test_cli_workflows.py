"""Configuration check for the demo workflow; needs no model or Docker."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_config_resolves_without_docker_or_model_calls():
    if not (ROOT / 'datasets/palomar-compact4/dataset.yaml').is_file():
        pytest.skip('preprocess the demo first: bash preprocess.sh configs/preprocessing/palomar-compact4.yaml')
    from leanlean.pipeline.evaluation import resolve_named_config

    # Resolution only: `eval.sh demo --validate-only` additionally checks the
    # pinned images and harness bundles on this host.
    resolved = resolve_named_config('demo', 'openai/gpt-5.6-luna-xhigh', repo_root=ROOT)
    manifest = resolved.manifest
    assert manifest['monitoring']['snapshots'] is True
    assert manifest['monitoring']['every_n_edits'] == 1
    assert len(manifest['repositories']) == 4
    assert all(row['image_id'].startswith('sha256:') for row in manifest['repositories'])
    assert not list((ROOT / 'datasets/palomar-compact4').rglob('.lake'))
