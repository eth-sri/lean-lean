"""Regression tests for native error streams and harness deadline handling."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from leanlean.antigravity_trace import antigravity_terminal_success, standardize_antigravity
from leanlean.generators.cli_agent import CLIAgent


@pytest.mark.parametrize('response', ['', ' \n', None, 12, 'Starting\n', 'READY'])
def test_empty_native_success_is_not_a_submission(response):
    terminal = {'status': 'SUCCESS', 'response': response}
    assert not antigravity_terminal_success(terminal)
    trace = standardize_antigravity(
        [{'event': 'result', 'result': terminal}], {'returncode': 0})
    assert not trace['quality']['provider_terminal_success']


@pytest.mark.parametrize('terminal,returncode', [
    ({'status': 'SUCCESS', 'response': 'Verified. <submit>'}, 0),
    ({'status': 'ERROR', 'response': 'Verified. <submit>', 'error': 'Earlier provider error'}, 3),
])
def test_existing_valid_and_recovered_submissions_remain_usable(terminal, returncode):
    assert antigravity_terminal_success(terminal)
    trace = standardize_antigravity(
        [{'event': 'result', 'result': terminal}], {'returncode': returncode})
    assert trace['quality']['provider_terminal_success']


def parse(stdout, *, timed_out=False):
    agent = CLIAgent.__new__(CLIAgent)
    agent.messages = []
    agent.model = SimpleNamespace(config=SimpleNamespace(model_name='gemini/gemini-4-argon'))
    agent._last_exec_timed_out = timed_out
    agent._account_gateway_responses = Mock()
    agent._process_antigravity_logs(stdout)
    agent._account_gateway_responses.assert_called_once()
    return agent


def test_deadline_is_not_misattributed_to_old_background_task_notice():
    notice = 'root agent idle; waiting up to 12h0m0s for 1 background task(s)'
    agent = parse(json.dumps({'event': 'init', 'init': {'model': 'gemini-4-argon'}})
                  + '\n' + notice, timed_out=True)
    assert 'execution deadline' in agent.provider_terminal_error
    assert 'root agent idle' not in agent.provider_terminal_error
    assert agent.provider_trace['native_diagnostic_tail'] == notice
    assert agent.provider_trace['timed_out'] is True
    assert not agent.provider_trace['terminal']['success']


def test_empty_success_becomes_a_provider_failure():
    agent = parse(json.dumps({'event': 'result', 'result': {'status': 'SUCCESS', 'response': ''}}))
    assert 'without the canonical <submit> marker' in agent.provider_terminal_error
    assert not agent.provider_trace['terminal']['success']


def test_deadline_overrides_a_native_success():
    agent = parse(json.dumps({'event': 'result', 'result': {'status': 'SUCCESS', 'response': 'Verified. <submit>'}}),
                  timed_out=True)
    assert not agent.provider_trace['terminal']['success']
