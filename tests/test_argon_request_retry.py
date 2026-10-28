"""No-network fault injection for retries under one native agent request."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leanlean.model.litellm_wrapper.argon_request_retry import (
    ArgonRequestRetryMiddleware,
    normalize_compaction_summary,
)
from leanlean.model.litellm_wrapper.accounting import AccountingScopeMiddleware

PATH = '/v1beta/models/gemini-4-argon-high:streamGenerateContent'
BODY = b'{"contents":[{"role":"user","parts":[{"text":"canonical task"}]}],"generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}'
USAGE = {'promptTokenCount': 1000, 'candidatesTokenCount': 20,
         'thoughtsTokenCount': 30, 'cachedContentTokenCount': 200}


def frame(**payload):
    return b'data: ' + json.dumps(payload).encode() + b'\n\n'


def success(text='complete'):
    return frame(candidates=[{'index': 0, 'content': {'parts': [{'text': text}]}, 'finishReason': 'STOP'}], usageMetadata=USAGE)


async def respond(send, body, status=200):
    await send({'type': 'http.response.start', 'status': status,
                'headers': [(b'content-type', b'text/event-stream')]})
    await send({'type': 'http.response.body', 'body': body, 'more_body': False})


async def exercise(app, *, path=PATH, body=BODY, disconnect_after=None, **options):
    queue = asyncio.Queue()
    await queue.put({'type': 'http.request', 'body': body})
    output, records = [], []
    async def send(message):
        output.append(message)
    middleware = ArgonRequestRetryMiddleware(app, max_retries=5, backoff_seconds=0,
                                            heartbeat_seconds=.01, record=records.append, **options)
    async def disconnect():
        await asyncio.sleep(disconnect_after)
        await queue.put({'type': 'http.disconnect'})
    timer = asyncio.create_task(disconnect()) if disconnect_after is not None else None
    try:
        await middleware({'type': 'http', 'method': 'POST', 'path': path, 'headers': []}, queue.get, send)
    except asyncio.CancelledError:
        if disconnect_after is None:
            raise
    finally:
        if timer:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
    return output, records


def response_bytes(output):
    return b''.join(x.get('body', b'') for x in output)


def test_native_compaction_summary_normalized_once_before_all_retries():
    payload = json.loads(BODY)
    payload['contents'][0]['parts'][0]['text'] = '<CONTEXT_SUMMARY>\nRésumé and unchanged task state.\n</CONTEXT_SUMMARY>\n'
    payload['systemInstruction'] = {'parts': [{'text': 'canonical system prompt'}]}
    original = json.dumps(payload).encode()
    received = []
    async def app(scope, receive, send):
        body = (await receive())['body']
        assert int(dict(scope['headers'])[b'content-length']) == len(body)
        received.append(body)
        await respond(send, success() if len(received) == 2 else frame(error={'code': 500}))
    output, records = asyncio.run(exercise(app, body=original))
    assert received[0] == received[1]
    expected = json.loads(original)
    expected['contents'][0]['parts'][0]['text'] = '\nRésumé and unchanged task state.\n\n'
    assert json.loads(received[0]) == expected
    assert response_bytes(output) == success()
    assert all(r['compaction_delimiters_removed'] for r in records)
    assert records[0]['original_request_sha256'] != records[0]['request_sha256']
    assert normalize_compaction_summary(received[0]) == (received[0], False)


@pytest.mark.parametrize('body', [
    BODY, b'{bad', b'[]', b'{"contents":null}',
    b'{"contents":[{"role":"model","parts":[{"text":"<CONTEXT_SUMMARY>x</CONTEXT_SUMMARY>"}]}]}',
    b'{"contents":[{"role":"user","parts":[{"text":"Discuss <CONTEXT_SUMMARY>x</CONTEXT_SUMMARY>"}]}]}',
    b'{"contents":[{"role":"user","parts":[{"functionResponse":{"response":{"text":"<CONTEXT_SUMMARY>x</CONTEXT_SUMMARY>"}}}]}]}',
])
def test_compaction_normalization_preserves_non_summary_payload_bytes(body):
    assert normalize_compaction_summary(body) == (body, False)


def test_five_retries_then_success_same_request_no_partial_tool_leak():
    calls = []
    partial = frame(candidates=[{'content': {'parts': [{'functionCall': {'name': 'write_file', 'args': {'path': 'DUPLICATE'}}}]}}])
    async def app(scope, receive, send):
        calls.append((await receive())['body'])
        if len(calls) <= 5:
            await respond(send, partial + frame(error={'code': 500}))
        else:
            await respond(send, success())
    output, records = asyncio.run(exercise(app))
    assert calls == [BODY] * 6
    assert response_bytes(output) == success()
    assert b'DUPLICATE' not in response_bytes(output)
    assert [r['delivered'] for r in records] == [False] * 5 + [True]
    assert len({r['litellm_call_id'] for r in records}) == 6
    assert len({r['request_sha256'] for r in records}) == 1


def test_exhaustion_is_explicit_error_and_stops_at_six_attempts():
    async def app(scope, receive, send):
        await respond(send, b'{"error":"temporary"}', 503)
    output, records = asyncio.run(exercise(app))
    assert len(records) == 6
    assert b'failed after 6 attempts (http_503)' in response_bytes(output)
    assert not any(r['delivered'] for r in records)


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_permanent_http_errors_are_forwarded_once(status):
    async def app(scope, receive, send):
        await respond(send, b'bad request', status)
    output, records = asyncio.run(exercise(app))
    assert len(records) == 1
    assert output[0]['status'] == status
    assert response_bytes(output) == b'bad request'


@pytest.mark.parametrize('bad', [b'', b'data: {bad}\n\n', frame(candidates=[{'content': {'parts': [{'text': 'partial'}]}}]), frame(candidates=[{'finishReason': 'MAX_TOKENS'}], usageMetadata=USAGE)])
def test_empty_malformed_truncated_and_output_limit_streams_retry(bad):
    count = 0
    async def app(scope, receive, send):
        nonlocal count
        count += 1
        await respond(send, bad if count == 1 else success())
    output, records = asyncio.run(exercise(app))
    assert count == 2
    assert response_bytes(output) == success()
    assert records[-1]['delivered']


def test_chunked_crlf_and_separate_usage_frame_preserve_exact_bytes():
    wire = success().replace(b'\n', b'\r\n') + frame(usageMetadata=USAGE)
    async def app(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        for byte in wire:
            await send({'type': 'http.response.body', 'body': bytes([byte]), 'more_body': True})
        await send({'type': 'http.response.body', 'body': b''})
    output, records = asyncio.run(exercise(app))
    assert response_bytes(output) == wire
    assert len(records) == 1
    assert records[0]['usage'] == USAGE


def test_silent_upstream_times_out_despite_keepalives_and_cancels_each_attempt():
    cancelled = []
    async def app(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        try:
            while True:
                await send({'type': 'http.response.body', 'body': b': ping\n\n', 'more_body': True})
                await asyncio.sleep(.003)
        finally:
            cancelled.append(True)
    output, records = asyncio.run(exercise(app, idle_seconds=.025))
    assert len(records) == len(cancelled) == 6
    assert all(r['reason'] == 'upstream_idle_timeout' for r in records)
    assert b'failed after 6 attempts' in response_bytes(output)
    assert b'data: {"candidates":[]}' in response_bytes(output)


def test_agent_disconnect_cancels_upstream_without_restarting_budget():
    cancelled = []
    async def app(scope, receive, send):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    output, records = asyncio.run(exercise(app, disconnect_after=.03))
    assert len(cancelled) == len(records) == 1
    assert records[0]['reason'] == 'cancelled_or_exception'
    assert not records[0]['delivered']


def test_other_models_are_untouched():
    async def app(scope, receive, send):
        await respond(send, b'partial unbuffered')
    output, records = asyncio.run(exercise(app, path='/v1beta/models/gemini-3.8-flash:streamGenerateContent'))
    assert response_bytes(output) == b'partial unbuffered'
    assert not records


def test_accounting_scope_survives_all_attempts():
    async def main():
        count = 0
        records = []
        async def app(scope, receive, send):
            nonlocal count
            count += 1
            assert scope['path'] == PATH
            assert (await receive())['body'] == BODY
            await respond(send, success() if count == 2 else frame(error={'code': 500}))
        app = AccountingScopeMiddleware(ArgonRequestRetryMiddleware(app, max_retries=5, backoff_seconds=0, record=records.append))
        queue = asyncio.Queue()
        await queue.put({'type': 'http.request', 'body': BODY})
        async def send(message): pass
        await app({'type': 'http', 'method': 'POST', 'path': '/lc-accounting/' + 'a'*32 + PATH, 'headers': []}, queue.get, send)
        assert [r['agent_context']['accounting_scope'] for r in records] == ['a'*32]*2
    asyncio.run(main())


def test_discarded_cost_is_scoped_added_once_and_missing_usage_is_not_zero(tmp_path):
    from leanlean.generators.cli_agent import CLIAgent
    from leanlean.model.costs import cost_accounting_from_responses
    path = tmp_path/'retries.jsonl'
    records = [{'event': 'argon_request_attempt', 'litellm_call_id': str(i), 'delivered': delivered,
                'usage': usage, 'agent_context': {'accounting_scope': scope}}
               for i, (scope, delivered, usage) in enumerate([
                   ('own', False, USAGE), ('own', False, None), ('own', True, USAGE), ('other', False, USAGE)])]
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    agent = CLIAgent.__new__(CLIAgent)
    agent.model = SimpleNamespace(accounting_scope_id='own', responses=[], get_request_retries_path=lambda: path)
    agent.provider_trace = {}
    agent._account_argon_request_attempts()
    agent._account_argon_request_attempts()
    assert len(agent.model.responses) == 2
    assert agent.model.cost == pytest.approx((800*2 + 200*.1 + 50*10)/1e6)
    assert agent.provider_trace['discarded_request_usage_missing'] == 1
    assert cost_accounting_from_responses(agent.model.responses)['usage_status'] == 'incomplete_or_inconsistent'


def test_sse_auth_error_is_not_retried_or_reclassified_as_transient():
    async def app(scope, receive, send):
        await respond(send, frame(error={'code': 401, 'message': 'unauthorized'}))
    output, records = asyncio.run(exercise(app))
    assert len(records) == 1
    assert b'"code": 401' in response_bytes(output)


def test_disconnect_during_backoff_starts_no_more_requests():
    async def main():
        queue = asyncio.Queue()
        await queue.put({'type': 'http.request', 'body': BODY})
        records = []
        async def app(scope, receive, send):
            await respond(send, frame(error={'code': 500}))
        def record(row):
            records.append(row)
            queue.put_nowait({'type': 'http.disconnect'})
        async def send(message): pass
        middleware = ArgonRequestRetryMiddleware(app, max_retries=5, backoff_seconds=10, record=record)
        await asyncio.wait_for(middleware({'type': 'http', 'method': 'POST', 'path': PATH}, queue.get, send), .5)
        assert len(records) == 1
    asyncio.run(main())


def test_native_step_usage_plus_discarded_attempts_reaches_final_cost(tmp_path):
    from leanlean.generators.cli_agent import CLIAgent
    from leanlean.model.costs import cost_accounting_from_responses
    path = tmp_path/'retries.jsonl'
    path.write_text(json.dumps({'event': 'argon_request_attempt', 'litellm_call_id': 'request:0',
                               'delivered': False, 'usage': USAGE, 'agent_context': {'accounting_scope': 'own'}})+'\n')
    agent = CLIAgent.__new__(CLIAgent)
    agent.model = SimpleNamespace(config=SimpleNamespace(model_name='gemini/gemini-4-argon'),
                                 accounting_scope_id='own', responses=[], get_request_retries_path=lambda: path)
    agent.messages = []
    native = dict(input_tokens=100, output_tokens=20, thinking_tokens=10, cache_read_tokens=0, total_tokens=120)
    events = [{'event': 'init', 'init': {'model': 'gemini-4-argon-high'}},
              {'event': 'step_update', 'step_update': {'conversation_id': 'fixture', 'step_index': 1, 'state': 'DONE', 'usage': native}},
              {'event': 'result', 'result': {'status': 'SUCCESS', 'response': '<submit>', 'usage': native}}]
    agent._process_antigravity_logs('\n'.join(json.dumps(e) for e in events))
    assert agent.provider_trace['terminal']['success']
    assert agent.model.cost == pytest.approx(.0004 + .00212)
    assert len(agent.model.responses) == 2
    evidence = cost_accounting_from_responses(agent.model.responses)
    assert evidence['calls_with_partial_usage'] == 1
    assert evidence['usage_status'] == 'incomplete_or_inconsistent'


def test_no_native_usage_keeps_all_failed_requests_in_accounting(tmp_path):
    from leanlean.generators.cli_agent import CLIAgent
    from leanlean.model.costs import cost_accounting_from_responses
    path = tmp_path/'retries.jsonl'
    path.write_text(json.dumps({'event': 'argon_request_attempt', 'litellm_call_id': 'missing:0',
                               'delivered': False, 'usage': None, 'agent_context': {'accounting_scope': 'own'}})+'\n')
    agent = CLIAgent.__new__(CLIAgent)
    agent.model = SimpleNamespace(config=SimpleNamespace(model_name='gemini/gemini-4-argon'),
                                 accounting_scope_id='own', responses=[], get_request_retries_path=lambda: path)
    agent.messages = []
    agent._process_antigravity_logs(json.dumps({'event': 'init', 'init': {'model': 'gemini-4-argon'}}))
    assert not agent.provider_trace['terminal']['success']
    evidence = cost_accounting_from_responses(agent.model.responses)
    assert evidence['calls_missing_usage'] == 1
    assert evidence['usage_status'] == 'incomplete_or_inconsistent'
