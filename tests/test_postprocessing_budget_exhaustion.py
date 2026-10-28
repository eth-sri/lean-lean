import json
from types import SimpleNamespace

import pytest

from leanlean.pipeline.postprocessing import _recorded_budget_exhaustion, _selected_state, _trajectory_selection


@pytest.fixture
def attempt(tmp_path):
    repo = tmp_path / 'repo'
    cap = repo / 'playback/capture_001'
    cap.mkdir(parents=True)
    (repo / 'round_0_gen_metrics.json').write_text(json.dumps({'wall_time_seconds': 43201}))
    (cap / 'standardized-trace.json').write_text(json.dumps({'source': {'timed_out': True}}))
    (repo / 'repo.traj.json').write_text(json.dumps({'info': {'exit_status': 'Submitted'}, 'provider_traces': [{'terminal': {'success': False}, 'error': 'execution deadline'}]}))
    return SimpleNamespace(output_dir=tmp_path, source_root=tmp_path, run_id='test', index=0,
                           predictions={'repo': {'model_patch': 'patch'}}, captures=[{'instance_id': 'repo'}],
                           evaluation_manifest={'container': {'timeout': '12h'}})


def test_timeout_is_rejected_by_default_but_explicitly_retained_for_endpoint_verification(attempt):
    assert _trajectory_selection(attempt, 'repo') is None
    selected = _trajectory_selection(attempt, 'repo', accept_budget_exhausted=True)
    assert selected['outcome'] == 'budget_exhausted'


@pytest.mark.parametrize('duration,timed_out', [(43199, True), (43201, False), (float('nan'), True)])
def test_early_failure_or_missing_timeout_evidence_cannot_bypass_submission_guard(attempt, duration, timed_out):
    repo = attempt.output_dir / 'repo'
    (repo / 'round_0_gen_metrics.json').write_text(json.dumps({'wall_time_seconds': duration}))
    (repo / 'playback/capture_001/standardized-trace.json').write_text(json.dumps({'source': {'timed_out': timed_out}}))
    assert not _recorded_budget_exhaustion(attempt, 'repo')
    assert _trajectory_selection(attempt, 'repo', accept_budget_exhausted=True) is None


@pytest.mark.parametrize('build,signatures,expected', [(True, True, 45.), (False, True, 0.), (True, False, 0.)])
def test_budget_exhaustion_scores_only_the_verified_final_endpoint(build, signatures, expected):
    playback = {
        'points': [
            {'edit_index': 1, 'word_compression_pct': 90., 'lean_token_compression_pct': 95.,
             'build': {'passed': True}, 'signature_validation': {'preserved': True}},
            {'edit_index': 7, 'word_compression_pct': 40., 'lean_token_compression_pct': 45.},
        ],
        'build_passed': build, 'signatures_preserved': signatures,
        'selected_state': {'policy': 'budget_exhausted_zero_score', 'eligible': False,
                           'lean_token_compression_pct': 0., 'word_compression_pct': 0.},
    }
    state = _selected_state(playback, timed_out=True, replay=False, budget_exhausted=True)
    assert state['eligible'] == (build and signatures)
    assert state['lean_token_compression_pct'] == expected
    assert state['word_compression_pct'] == (40. if expected else 0.)
    assert state['edit_index'] == 7
    assert state['policy'] == 'submitted_final_state'
    assert state['budget_exhausted'] is True
    assert 'penalty_reason' not in state
