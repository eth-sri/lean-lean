from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from leanlean.subscription_auth import (
    NAMED_SUBSCRIPTION_KEY_ENVS,
    SUBSCRIPTION_KEY_ENV,
    strip_subscription_credentials,
    subscription_auth_record,
    subscription_provider_name,
)

from .common import mapping, number, stamp
from .claude_quota import ClaudeQuota, secondary_configured


def get_json(url, token, *, bearer=True):
    authorization = f'Bearer {token}' if bearer else token
    request = urllib.request.Request(url, headers={'Authorization': authorization, 'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.load(response)


def safe_failure(error):
    if isinstance(error, urllib.error.HTTPError):
        return f'Provider returned HTTP {error.code}'
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired, queue.Empty)):
        return 'Usage request timed out'
    return f'Usage unavailable ({type(error).__name__})'


def quota_card(name, fetcher):
    usage = fetcher(timeout_seconds=12)
    return {'name': name, 'kind': 'subscription', 'status': 'ok' if usage.available else 'unavailable',
            'detail': usage.detail, 'windows': [{'name': w.name.replace('_', ' '), 'used_percent': w.utilization,
              'resets_at': w.resets_at} for w in usage.windows]}


def normalize_codex(result, *, name='Codex', detail=None, run_provider=None):
    buckets = mapping(result.get('rateLimitsByLimitId'))
    if not buckets and result.get('rateLimits'):
        buckets = {'codex': result['rateLimits']}
    windows = []
    for key, raw in buckets.items():
        bucket = mapping(raw)
        identity = ' '.join(str(value or '') for value in (key, bucket.get('limitId'), bucket.get('limitName'))).lower()
        if 'spark' in identity:
            continue
        for slot in ('primary', 'secondary'):
            value = mapping(bucket.get(slot))
            used = number(value.get('usedPercent'))
            if used is None:
                continue
            minutes = number(value.get('windowDurationMins'))
            duration = ('weekly' if minutes == 10080 else f'{minutes / 60:g}h' if minutes and minutes % 60 == 0 else f'{minutes:g}m' if minutes else slot)
            reset = number(value.get('resetsAt'))
            windows.append({'name': f'{bucket.get("limitName") or key} · {duration}', 'used_percent': used,
                            'resets_at': datetime.fromtimestamp(reset, timezone.utc).isoformat() if reset else None})
    card = {'name': name, 'kind': 'subscription', 'status': 'ok' if windows else 'unavailable',
            'detail': detail or ('Account quota buckets' if windows else 'No quota windows returned for this login'),
            'windows': windows}
    if run_provider:
        card['run_provider'] = run_provider
    return card


def _codex_login_params(record):
    account_id = record.get('account_id')
    if not account_id:
        raise RuntimeError(
            'OPENAI_SUBSCRIPTION_ACCOUNT_ID is missing and the subscription token '
            'does not identify an account'
        )
    return {'type': 'chatgptAuthTokens', 'accessToken': record['access_token'],
            'chatgptAccountId': account_id, 'chatgptPlanType': None}


def stop_process_group(p):
    """Stop the app-server and every proxy child it detached.

    Codex keeps seeding CODEX_HOME from a proxy that outlives a plain
    terminate(); left alive it races the temporary-directory cleanup and
    raises `Directory not empty` on a random card.
    """
    try:
        group = os.getpgid(p.pid)
    except OSError:
        group = None

    def signal_group(number):
        if group is None or group == os.getpgrp():
            return False
        try:
            os.killpg(group, number)
        except OSError:
            return False
        return True

    if not signal_group(signal.SIGTERM):
        p.terminate()
    try:
        p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if not signal_group(signal.SIGKILL):
            p.kill()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    else:
        signal_group(signal.SIGKILL)


def codex_usage(*, name='Codex', auth_record=None, run_provider=None,
                auth_detail=None):
    binary = shutil.which('codex')
    if not binary:
        return {'name': name, 'kind': 'subscription', 'status': 'unavailable',
                'detail': 'Codex CLI is not installed', 'windows': []}
    temporary = (tempfile.TemporaryDirectory(prefix='leanlean-monitor-codex-',
                                             ignore_cleanup_errors=True)
                 if auth_record else None)
    child_env = os.environ.copy()
    strip_subscription_credentials(child_env)
    if temporary:
        codex_home = Path(temporary.name) / 'home'
        codex_home.mkdir(mode=0o700)
        child_env['CODEX_HOME'] = str(codex_home)
    try:
        p = subprocess.Popen([binary, 'app-server', '--listen', 'stdio://'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=child_env,
                             start_new_session=True)
    except Exception:
        if temporary:
            temporary.cleanup()
        raise
    messages = queue.Queue()
    def reader():
        for line in p.stdout:
            try:
                messages.put(json.loads(line))
            except ValueError:
                pass
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    def send(value):
        p.stdin.write(json.dumps(value) + '\n')
        p.stdin.flush()
    deadline = time.monotonic() + 25
    def receive(request_id):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            message = messages.get(timeout=remaining)
            if message.get('id') == request_id:
                if 'error' in message:
                    raise RuntimeError('Account quota request rejected')
                return mapping(message.get('result'))
    try:
        capabilities = {'experimentalApi': True} if auth_record else {}
        send({'id': 1, 'method': 'initialize', 'params': {'clientInfo': {'name': 'leanlean_monitor', 'version': '1.0.0'}, 'capabilities': capabilities}})
        receive(1)
        send({'method': 'initialized', 'params': {}})
        request_id = 2
        if auth_record:
            send({'id': request_id, 'method': 'account/login/start',
                  'params': _codex_login_params(auth_record)})
            receive(request_id)
            request_id += 1
        send({'id': request_id, 'method': 'account/rateLimits/read', 'params': {}})
        detail = (
            'Current Codex login quota buckets'
            if auth_record is None
            else auth_detail or 'Quota buckets for OPENAI_SUBSCRIPTION_KEY from secret.sh'
        )
        return normalize_codex(receive(request_id), name=name, detail=detail,
                               run_provider=run_provider)
    finally:
        stop_process_group(p)
        for close in (p.stdin.close, lambda: thread.join(timeout=2), p.stdout.close):
            try:
                close()
            except OSError:
                pass
        if temporary:
            temporary.cleanup()


def explicit_codex_usage(key_env=SUBSCRIPTION_KEY_ENV):
    name = subscription_provider_name(key_env)
    run_provider = 'Codex' if key_env == SUBSCRIPTION_KEY_ENV else name
    try:
        record = subscription_auth_record(key_env=key_env)
    except RuntimeError as error:
        return {'name': name, 'kind': 'subscription',
                'status': 'unavailable', 'detail': str(error), 'windows': [],
                'run_provider': run_provider}
    return codex_usage(
        name=name,
        auth_record=record,
        run_provider=run_provider,
        auth_detail=f'Quota buckets for {key_env} from secret.sh',
    )


def openrouter_usage():
    key = os.getenv('OPENROUTER_API_KEY')
    card = {'name': 'OpenRouter', 'kind': 'API', 'windows': [], 'metrics': []}
    if not key:
        return {**card, 'status': 'unavailable', 'detail': 'OPENROUTER_API_KEY is not loaded'}
    data = mapping(get_json('https://openrouter.ai/api/v1/key', key).get('data'))
    for field, label in [('limit_remaining', 'Key budget remaining'), ('usage_daily', 'Used today'), ('usage_monthly', 'Used this month'), ('usage', 'Key lifetime usage')]:
        value = number(data.get(field))
        if value is not None:
            card['metrics'].append({'label': label, 'value': value, 'currency': 'USD'})
    card.update(status='ok', detail='Key spending cap is separate from account balance')
    if data.get('limit') is None:
        card['detail'] = 'No key spending cap; account balance may still apply'
    management_key = os.getenv('OPENROUTER_MANAGEMENT_KEY')
    if management_key:
        try:
            balance = mapping(get_json('https://openrouter.ai/api/v1/credits', management_key).get('data'))
            credit, usage = number(balance.get('total_credits')), number(balance.get('total_usage'))
            if credit is not None and usage is not None:
                card['metrics'].insert(0, {'label': 'Account balance', 'value': credit - usage, 'currency': 'USD'})
        except Exception as error:
            card['detail'] += '; account balance: ' + safe_failure(error)
    else:
        card['detail'] += '; account balance needs OPENROUTER_MANAGEMENT_KEY'
    return card



def normalize_glm(payload):
    card = {'name': 'GLM / Z.ai', 'kind': 'subscription', 'windows': []}
    if not isinstance(payload, dict):
        return {**card, 'status': 'unavailable', 'detail': 'GLM returned an unrecognized quota response'}
    if payload.get('success') is False or str(payload.get('code', 200)) not in {'0', '200'}:
        code = str(payload.get('code', ''))
        suffix = f' (code {code})' if code.isdigit() else ''
        return {**card, 'status': 'unavailable', 'detail': 'GLM quota request rejected' + suffix}
    data = mapping(payload.get('data'))
    limits = data.get('limits')
    if not isinstance(limits, list):
        limits = []
    labels = {'TOKENS_LIMIT': 'Token quota', 'TIME_LIMIT': 'MCP quota'}
    for row in limits:
        if not isinstance(row, dict):
            continue
        used = number(row.get('percentage'))
        if used is None:
            current, total = number(row.get('currentValue')), number(row.get('usage'))
            if current is not None and total is not None and total > 0:
                used = current / total * 100
        if used is None:
            continue
        name = labels.get(row.get('type'), 'Plan quota')
        # Numeric unit enums are undocumented; do not invent 5h/weekly labels.
        unit, count = row.get('unit'), number(row.get('number'))
        if isinstance(unit, str) and unit.lower() in {'hour', 'hours', 'day', 'days', 'week', 'weeks', 'month', 'months'} and count:
            name += f' · {count:g} {unit.lower()}'
        reset = row.get('nextResetTime')
        seconds = number(reset)
        if seconds is not None:
            try:
                reset = datetime.fromtimestamp(seconds / 1000 if seconds > 100_000_000_000 else seconds, timezone.utc).isoformat()
            except (ValueError, OSError, OverflowError):
                reset = None
        elif isinstance(reset, str):
            try:
                reset = datetime.fromisoformat(reset.replace('Z', '+00:00')).isoformat()
            except ValueError:
                reset = None
        else:
            reset = None
        card['windows'].append({'name': name, 'used_percent': used, 'resets_at': reset})
    # Distinguish multiple quota windows without guessing their duration.
    for label in set(w['name'] for w in card['windows']):
        matching = [w for w in card['windows'] if w['name'] == label]
        if len(matching) > 1:
            for index, window in enumerate(matching, 1):
                window['name'] += f' {index}'
    return {**card, 'status': 'ok' if card['windows'] else 'unavailable',
            'detail': 'Coding Plan usage; durations and resets shown only when provided' if card['windows'] else 'No Coding Plan quota windows returned for this key'}


def glm_usage():
    key = os.getenv('Z_AI_API_KEY') or os.getenv('ZAI_API_KEY') or os.getenv('GLM_API_KEY')
    if not key:
        return {'name': 'GLM / Z.ai', 'kind': 'subscription', 'status': 'unavailable', 'windows': [],
                'detail': 'Z_AI_API_KEY, ZAI_API_KEY or GLM_API_KEY is not loaded'}
    # Match Z.ai's own usage-query script: raw Authorization token, not a model call.
    return normalize_glm(get_json('https://api.z.ai/api/monitor/usage/quota/limit', key, bearer=False))


class Providers:
    def __init__(self, cache_dir=None):
        self.last_good = {}
        self.claude = ClaudeQuota(cache_dir)
        self.secondary_claude = ClaudeQuota(cache_dir, account='secondary')

    def collect(self, *, force=False):
        functions = [('Codex · current login', lambda: codex_usage(name='Codex · current login')),
                     ('Codex · secret.sh account', explicit_codex_usage)]
        functions.extend(
            (
                subscription_provider_name(key_env),
                lambda key_env=key_env: explicit_codex_usage(key_env),
            )
            for key_env in NAMED_SUBSCRIPTION_KEY_ENVS
        )
        functions.extend([
                     ('Claude', lambda: self.claude.read(force=force)),
                     ('OpenRouter', openrouter_usage), ('GLM / Z.ai', glm_usage)])
        if secondary_configured():
            functions.insert(3, ('Claude · secondary account',
                                 lambda: self.secondary_claude.read(force=force)))
        def collect_one(pair):
            name, function = pair
            try:
                card = function()
                if name.startswith('Claude'):
                    return card
            except Exception as error:
                card = {'name': name, 'status': 'unavailable', 'detail': safe_failure(error), 'windows': []}
            card['checked_at'] = stamp()
            if card['status'] == 'ok':
                card['updated_at'] = card['checked_at']
                self.last_good[name] = dict(card)
            elif name in self.last_good:
                card = {**self.last_good[name], 'status': 'stale', 'detail': card['detail'], 'checked_at': card['checked_at']}
            return card
        with ThreadPoolExecutor(max_workers=min(10, len(functions))) as pool:
            cards = list(pool.map(collect_one, functions))
        for label, envs in [('OpenAI API', ('OPENAI_API_KEY',)), ('Anthropic API', ('ANTHROPIC_API_KEY',)), ('Gemini API', ('GEMINI_API_KEY', 'GOOGLE_API_KEY')), ('Moonshot API', ('MOONSHOT_API_KEY',)), ('Mistral API', ('MISTRAL_API_KEY',)), ('Meta API', ('META_API_KEY',))]:
            if any(os.getenv(env) for env in envs):
                cards.append({'name': label, 'kind': 'API', 'status': 'unsupported', 'windows': [],
                              'detail': 'Key loaded. Balance lookup is not integrated; run costs are shown separately.', 'checked_at': stamp()})
        return {'updated_at': stamp(), 'providers': cards}
