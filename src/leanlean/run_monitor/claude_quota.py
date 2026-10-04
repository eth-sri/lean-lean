"""Conservative, shared Claude quota polling. No inference requests."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from .common import mapping, number


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def retry_seconds(value, now):
    seconds = number(value)
    if seconds is not None:
        return max(0, seconds)
    try:
        return max(0, parsedate_to_datetime(value).timestamp() - now)
    except (TypeError, ValueError, OverflowError):
        return 0


def credentials():
    from leanlean.subscription_queue import _claude_profile_token
    # A profile-scoped login is more suitable than an inference-only setup token.
    return list(dict.fromkeys(t for t in (_claude_profile_token(), os.getenv('CLAUDE_CODE_OAUTH_TOKEN')) if t))


def secondary_credentials():
    from leanlean.subscription_queue import _claude_profile_token
    config = os.getenv(
        'SECOND_CLAUDE_CONFIG_DIR', str(Path.home() / '.claude-secondary')
    )
    profile = _claude_profile_token(config)
    token = os.getenv('SECOND_CLAUDE_CODE_OAUTH_TOKEN') or os.getenv(
        'ALT_CLAUDE_CODE_OAUTH_TOKEN'
    )
    return list(dict.fromkeys(t for t in (profile, token) if t))


def secondary_configured():
    """Whether a secondary account has any usable host credential source."""
    return bool(secondary_credentials())


def auth_failure_detail(code):
    config = Path(os.getenv('CLAUDE_CONFIG_DIR', str(Path.home() / '.claude')))
    try:
        profile = mapping(json.loads((config / '.credentials.json').read_text()).get('claudeAiOauth'))
        expiry = number(profile.get('expiresAt'))
    except (OSError, ValueError, AttributeError):
        expiry = None
    if expiry is not None and expiry / 1000 <= time.time():
        return 'Saved Claude login has expired. Run claude auth login --claudeai on this machine, then restart the dashboard.'
    reason = 'Claude rejected the OAuth credentials (401).' if code == 401 else 'Claude denied access to usage data (403); a user:profile login is required.'
    return reason + ' Run claude auth login --claudeai on this machine, then restart the dashboard.'


def request_usage(token):
    from leanlean.subscription_queue import ANTHROPIC_USAGE_ENDPOINT
    req = urllib.request.Request(ANTHROPIC_USAGE_ENDPOINT, headers={
        'Authorization': f'Bearer {token}', 'anthropic-beta': 'oauth-2025-04-20',
        'anthropic-version': '2023-06-01', 'Accept': 'application/json',
        'User-Agent': 'claude-code/2.1.233',
    })
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.load(response)


def usage_windows(payload):
    payload = mapping(payload)
    windows = []
    limits = payload.get('limits')
    for raw in limits if isinstance(limits, list) else []:
        item = mapping(raw)
        used = number(item.get('percent'))
        if used is None:
            continue
        model = mapping(mapping(item.get('scope')).get('model'))
        model_name = model.get('display_name') or model.get('id')
        kind = item.get('kind')
        label = {'session': '5h', 'weekly_all': 'weekly · all models', 'weekly_scoped': 'weekly'}.get(kind, str(kind or 'Quota').replace('_', ' '))
        if model_name:
            label += ' · ' + str(model_name)
        windows.append({'name': label, 'used_percent': used,
                        'resets_at': item.get('resets_at') if isinstance(item.get('resets_at'), str) else None,
                        'model': model_name})
    if windows:
        return windows
    for name, raw in payload.items():
        value = mapping(raw)
        if not name.startswith(('five_hour', 'seven_day')):
            continue
        used = number(value.get('utilization'))
        if used is not None:
            label = name.replace('_', ' ').replace('seven day', 'weekly').replace('five hour', '5h').replace('fable', 'Fable')
            windows.append({'name': label, 'used_percent': used,
                            'resets_at': value.get('resets_at') if isinstance(value.get('resets_at'), str) else None})
    return windows


class ClaudeQuota:
    def __init__(self, cache_dir: Path | None = None, *, clock=time.time,
                 account='primary'):
        self.cache_dir = cache_dir
        self.clock = clock
        self.memory = {}
        self.account = account

    def _credentials(self):
        return secondary_credentials() if self.account == 'secondary' else credentials()

    def _name(self):
        return 'Claude · secondary account' if self.account == 'secondary' else 'Claude'

    def _auth_failure_detail(self, code):
        if self.account != 'secondary':
            return auth_failure_detail(code)
        if code == 403:
            return ('Secondary setup-token is valid for inference but lacks user:profile scope. '
                    'Run Claude profile login with SECOND_CLAUDE_CONFIG_DIR to show its limits.')
        if code == 401:
            return 'Claude rejected the secondary OAuth credential (401).'
        return 'Secondary Claude usage lookup failed'

    def read(self, *, force=False):
        tokens = self._credentials()
        base = {'name': self._name(), 'kind': 'subscription', 'windows': []}
        if not tokens:
            return {**base, 'status': 'unavailable', 'detail': 'No Claude OAuth credentials loaded'}
        identity = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:24]
        if self.cache_dir is None:
            return self._read(tokens, self.memory.setdefault(identity, {}), base, force=force)
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.cache_dir / f'claude-{identity}.json'
        # One request budget shared by dashboard processes and across restarts.
        with os.fdopen(os.open(path.with_suffix('.lock'), os.O_CREAT | os.O_RDWR, 0o600), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = mapping(json.loads(path.read_text()))
            except (OSError, ValueError):
                state = {}
            result = self._read(tokens, state, base, force=force)
            temporary = path.with_suffix('.tmp')
            with os.fdopen(os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), 'w') as stream:
                json.dump(state, stream)
            temporary.replace(path)
            return result

    def _read(self, tokens, state, base, *, force=False):
        now = self.clock()
        if force and now >= (number(state.get('server_retry_at')) or 0):
            state['next_check'] = 0
        if now < (number(state.get('next_check')) or 0) and state.get('card') and (state.get('schema') == 2 or state['card'].get('status') != 'ok'):
            card = dict(state['card'])
            if card.get('detail') == 'Usage lookup needs valid OAuth credentials with user:profile permission':
                card['detail'] = self._auth_failure_detail(None)
                state['card'] = card
            return card
        error_code, retry_after = None, 0
        payload = None
        for token in tokens:
            try:
                payload = request_usage(token)
                break
            except urllib.error.HTTPError as error:
                error_code = error.code
                if error.code == 429:
                    retry_after = retry_seconds(error.headers.get('Retry-After') if error.headers else None, now)
                # Only authentication/permission failures justify a different token.
                if error.code not in {401, 403}:
                    break
            except (OSError, ValueError):
                error_code = 'network'
                break
        windows = usage_windows(payload)
        if windows:
            next_check = now + 300
            card = {**base, 'windows': windows, 'status': 'ok', 'detail': 'Quota readings cached for five minutes',
                    'checked_at': iso(now), 'updated_at': iso(now), 'next_check_at': iso(next_check)}
            state.update(failures=0, good=card, server_retry_at=None)
        else:
            limited = error_code == 429
            failures = min(10, int(state.get('failures', 0)) + 1) if limited else 0
            delay = max(retry_after, min(900, 60 * 2 ** (failures - 1))) if limited else 300
            next_check = now + delay
            detail = ('Usage endpoint rate-limited (429). This does not establish whether your model quota is exhausted.' if limited else
                      self._auth_failure_detail(error_code) if error_code in {401, 403} else
                      'Claude usage lookup failed' if error_code else 'No quota windows returned')
            good = mapping(state.get('good'))
            card = {**base, **good, 'status': 'stale' if good else ('cooldown' if limited else 'unavailable'),
                    'detail': detail, 'checked_at': iso(now), 'next_check_at': iso(next_check)}
            state['failures'] = failures
            state['server_retry_at'] = now + retry_after if limited and retry_after else None
        state.update(next_check=next_check, card=card, schema=2)
        return dict(card)
