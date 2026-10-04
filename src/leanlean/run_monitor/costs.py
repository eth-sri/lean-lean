"""Incremental cost estimates from saved accounting events and pinned run rates."""
from __future__ import annotations

import json
from pathlib import Path

from .common import mapping, number


def model_identity(model):
    return str(model or '').split('/')[-1].removeprefix('claude-').replace('.', '-')


def event_cost(event, rates, model):
    if event.get('event') != 'success':
        return None
    response = mapping(event.get('response'))
    event_model = str(response.get('model') or event.get('model') or '')
    if not model or model_identity(event_model) != model_identity(model):
        return None
    usage = mapping(response.get('usage'))
    prompt, output = number(usage.get('prompt_tokens')), number(usage.get('completion_tokens'))
    if prompt is None or output is None:
        return None
    details = mapping(usage.get('prompt_tokens_details'))
    cached = number(details.get('cached_tokens')) or 0
    creation = number(details.get('cache_creation_tokens')) or number(details.get('cache_write_tokens')) or 0
    quantities = {'input': max(0, prompt - cached - creation), 'output': output,
                  'cache_read': cached, 'cache_creation': creation}
    total = 0
    for key, tokens in quantities.items():
        rate = number(rates.get(key))
        if tokens and (rate is None or rate < 0):
            return None
        total += tokens * (rate or 0) / 1_000_000
    return total


class AccountingIndex:
    def __init__(self):
        self.cache = {}

    def read(self, output: Path | None, rates, model, *, live_files=()):
        if output is None or not rates:
            return {'cost': None, 'calls': 0}
        files = sorted(set((output / 'proxy_traces').glob('*.accounting.jsonl')) | set(live_files))
        signature = json.dumps([rates, model], sort_keys=True)
        key = str(output)
        state = self.cache.get(key)
        stats = {}
        for file in files:
            try:
                stats[str(file)] = file.stat()
            except OSError:
                pass
        if state is None or state['signature'] != signature or any(name not in stats or stats[name].st_ino != value[0] or stats[name].st_size < value[1] for name, value in state['offsets'].items()):
            state = {'signature': signature, 'offsets': {}, 'ids': set(), 'cost': 0, 'calls': 0}
            self.cache[key] = state
        for name, st in stats.items():
            offset = state['offsets'].get(name, (st.st_ino, 0))[1]
            try:
                with open(name, 'rb') as stream:
                    stream.seek(offset)
                    while True:
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            break
                        offset = stream.tell()
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        call_id = event.get('litellm_call_id') or event.get('request_id') or mapping(event.get('response')).get('id')
                        # Without a provider request ID, only this physical record is counted.
                        identity = str(call_id) if call_id else f'{name}:{offset}'
                        if identity in state['ids']:
                            continue
                        cost = event_cost(event, rates, model)
                        if cost is not None:
                            state['ids'].add(identity)
                            state['cost'] += cost
                            state['calls'] += 1
                state['offsets'][name] = (st.st_ino, offset)
            except OSError:
                continue
        return {'cost': state['cost'] if state['calls'] else None, 'calls': state['calls'],
                'updated_at': max((st.st_mtime for st in stats.values()), default=None),
                'live': bool(live_files)}
