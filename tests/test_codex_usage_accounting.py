import gzip
import json
import logging
from types import SimpleNamespace

import pytest

from leanlean.codex_usage import completed_rollout_response
from leanlean.generators.cli_agent import CLIAgent
from leanlean.model.costs import cost_accounting_from_responses, cost_from_responses, require_complete_cost_accounting


def rollout(tmp_path, *, complete=True, cache=True):
    usage = dict(input_tokens=1000, cached_input_tokens=600, output_tokens=100, reasoning_output_tokens=70)
    if not cache:
        del usage['cached_input_tokens']
    events = [dict(type='session_meta', payload={'id':'one'}),
              dict(type='turn_context', payload={'model':'gpt-6.1-sol'}),
              dict(type='event_msg', payload={'type':'token_count','info':{'total_token_usage':usage}})]
    # The same cumulative counter can be emitted repeatedly; charge it once.
    events.append(events[-1])
    if complete:
        events.append(dict(type='event_msg', payload={'type':'task_complete'}))
    path = tmp_path/'rollout.jsonl.gz'
    with gzip.open(path,'wt') as f:
        f.write('\n'.join(json.dumps(e) for e in events)+'\n')
    return path


def agent(path):
    a=CLIAgent.__new__(CLIAgent)
    a.config=SimpleNamespace(cli_name='codex_sub')
    a.model=SimpleNamespace(config=SimpleNamespace(model_name='openai/gpt-6.1-sol'),responses=[],n_calls=0,cost=0)
    a.logger=logging.getLogger('cost-test')
    a._native_rollout_host_path=path
    return a


def test_cumulative_usage_counts_cache_and_output_once(tmp_path):
    response=completed_rollout_response(rollout(tmp_path),model='gpt-6.1-sol')
    assert cost_from_responses([response]) == pytest.approx(.00186)
    assert response['usage']['completion_tokens']==100  # reasoning already included


def test_missing_stdout_usage_recovered_before_metrics(tmp_path):
    a=agent(rollout(tmp_path))
    a._recover_codex_rollout_usage(0)
    assert a.model.cost==pytest.approx(.00186)
    assert cost_accounting_from_responses(a.model.responses)['usage_status']=='reported_usage'
    assert a.provider_trace['cost_recovery']['recovered_usd']==pytest.approx(.00186)
    a._recover_codex_rollout_usage(0)
    assert len(a.model.responses)==1


def test_previous_round_is_not_replaced_or_recounted(tmp_path):
    a=agent(rollout(tmp_path))
    previous={'model':'gpt-6.1-sol','usage':{'prompt_tokens':0,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':0}}}
    a.model.responses=[previous]
    a._recover_codex_rollout_usage(1)
    assert a.model.responses[0] is previous
    assert a.model.cost==pytest.approx(.00196)


@pytest.mark.parametrize('complete,cache',[(False,True),(True,False)])
def test_partial_or_missing_cache_usage_is_unknown(tmp_path,complete,cache):
    a=agent(rollout(tmp_path,complete=complete,cache=cache))
    a._recover_codex_rollout_usage(0)
    assert not a.model.responses
    evidence=cost_accounting_from_responses(a.model.responses)
    assert evidence['estimated_usd'] is None
    assert evidence['usage_status']=='missing_usage'
    with pytest.raises(ValueError,match='incomplete'):
        require_complete_cost_accounting([])


def test_model_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='model differs'):
        completed_rollout_response(rollout(tmp_path),model='gpt-5.6-sol')


def test_decreasing_counters_are_rejected(tmp_path):
    path=rollout(tmp_path)
    with gzip.open(path,'rt') as f: events=[json.loads(line) for line in f]
    events[3]['payload']['info']['total_token_usage']['input_tokens']=900
    with gzip.open(path,'wt') as f: f.write('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError,match='decreased'):
        completed_rollout_response(path,model='gpt-6.1-sol')


def test_explicit_zero_usage_is_different_from_missing(tmp_path):
    path=rollout(tmp_path)
    with gzip.open(path,'rt') as f: events=[json.loads(line) for line in f]
    for e in events:
        if e['payload'].get('type')=='token_count':
            e['payload']['info']['total_token_usage']={k:0 for k in e['payload']['info']['total_token_usage']}
    with gzip.open(path,'wt') as f: f.write('\n'.join(json.dumps(e) for e in events))
    r=completed_rollout_response(path,model='gpt-6.1-sol')
    assert cost_accounting_from_responses([r])['estimated_usd']==0
    assert cost_accounting_from_responses([r])['usage_status']=='reported_usage'

@pytest.mark.parametrize('host_side',[False,True])
def test_playback_preserves_unknown_final_cost(host_side):
    from threading import RLock
    from leanlean.playback import LivePlaybackRecorder
    from leanlean.host_side_checkpointing import HostSideCheckpointRecorder
    cls=HostSideCheckpointRecorder if host_side else LivePlaybackRecorder
    recorder=cls.__new__(cls)
    recorder.edit_count=0
    recorder.points=[{'kind':'baseline','edit_index':0,'cost_usd':0}, {'kind':'final','edit_index':1,'cost_usd':0}]
    recorder._detected_stream_provider='codex-cli'
    recorder._turn_responses=[]
    recorder._capture=lambda **kwargs:None
    recorder._playback_result=lambda **kwargs:dict(points=recorder.points,**kwargs)
    recorder._persist_manifest=lambda result:None
    if host_side:
        recorder.close=lambda:None
        recorder._checkpoint_lock=RLock()
        recorder._checkpoint_events=[]
        recorder._codex_rollout_costs_exact=False
        recorder._persist_standardized_trace=lambda:None
    result=recorder.finish([])
    assert result['final_cost_usd'] is None
    assert result['authoritative_final_cost_usd'] is None
    assert result['final_cost_complete'] is False
    assert recorder.points[-1]['cost_usd'] is None
    assert recorder.points[0]['cost_usd']==0  # zero before any work is still exact


def test_nonzero_but_incomplete_stdout_total_is_reconciled(tmp_path):
    a=agent(rollout(tmp_path))
    a.model.responses=[{'model':'gpt-6.1-sol','usage':{'prompt_tokens':0,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':0}}}]
    a._recover_codex_rollout_usage(0)
    assert len(a.model.responses)==1
    assert a.model.cost==pytest.approx(.00186)
    assert a.provider_trace['cost_recovery']['previous_stdout_usd']==pytest.approx(.0001)


def test_matching_stdout_total_is_preserved(tmp_path):
    path=rollout(tmp_path);a=agent(path)
    response=completed_rollout_response(path,model='gpt-6.1-sol')
    a.model.responses=[response]
    a._recover_codex_rollout_usage(0)
    assert a.model.responses==[response]
    assert not hasattr(a,'provider_trace')
