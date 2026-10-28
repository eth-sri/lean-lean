"""Recover final Codex usage from a complete, invocation-local native rollout."""
from __future__ import annotations

import gzip
from pathlib import Path

from leanlean.utils import json_utils as json
from leanlean.model.costs import require_complete_cost_accounting


def completed_rollout_response(path: str | Path, *, model: str) -> dict:
    """Read cumulative counters once; reject partial or ambiguous accounting.

    Repeated token_count events are cumulative, so summing them double charges.
    This recovery path deliberately requires one session and one model. It does
    not infer missing cache usage or turn an interrupted session into a total.
    """
    counters = None
    session_count = 0
    active_model = None
    complete = False
    fields = ('input_tokens', 'cached_input_tokens', 'output_tokens')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            payload = event.get('payload')
            if event.get('type') == 'session_meta':
                session_count += 1
            if not isinstance(payload, dict):
                continue
            if event.get('type') == 'turn_context':
                active_model = payload.get('model')
                if active_model != model:
                    raise ValueError('native Codex model differs from requested cost model')
            if event.get('type') != 'event_msg':
                continue
            kind = payload.get('type')
            if kind in {'task_started', 'turn_aborted', 'token_count'}:
                complete = False
            if kind == 'task_complete':
                complete = True
            if kind != 'token_count':
                continue
            usage = (payload.get('info') or {}).get('total_token_usage')
            if usage is None:
                continue
            if not isinstance(usage, dict) or active_model != model:
                raise ValueError('native Codex usage has no attributable model')
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in fields):
                raise ValueError('native Codex usage is missing valid input, cache, or output counts')
            current = {k: usage[k] for k in fields}
            for k in ('cache_write_input_tokens', 'reasoning_output_tokens'):
                value = usage.get(k, 0)
                if type(value) is not int or value < 0:
                    raise ValueError('invalid native Codex token count')
                current[k] = value
            if counters and any(current[k] < counters[k] for k in current):
                raise ValueError('native Codex cumulative usage counters decreased')
            if current['cached_input_tokens'] + current['cache_write_input_tokens'] > current['input_tokens']:
                raise ValueError('native Codex cached usage exceeds input tokens')
            if current['reasoning_output_tokens'] > current['output_tokens']:
                raise ValueError('native Codex reasoning usage exceeds output tokens')
            counters = current
    if session_count != 1 or not complete or counters is None:
        raise ValueError('native Codex rollout has no complete, single-session usage total')
    response = {
        'model': model,
        'usage': {
            'prompt_tokens': counters['input_tokens'],
            'completion_tokens': counters['output_tokens'],
            'prompt_tokens_details': {
                'cached_tokens': counters['cached_input_tokens'],
                'cache_creation_tokens': counters['cache_write_input_tokens'],
            },
            'completion_tokens_details': {'reasoning_tokens': counters['reasoning_output_tokens']},
        },
        'usage_provenance': 'completed_native_codex_rollout_cumulative_counters',
    }
    require_complete_cost_accounting([response])
    return response
