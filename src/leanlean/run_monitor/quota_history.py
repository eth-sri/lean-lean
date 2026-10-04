"""Local quota observations and explicitly approximate run attribution."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from leanlean.subscription_auth import subscription_provider_name

from .common import number


def timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError):
        return None


def provider_for(run):
    if run.get('access_mode') != 'subscription':
        return None
    text = str(run.get('harness', '')).lower()
    run_key = str(run.get('key', '')).lower()
    if 'claude' in text and 'secondary' in run_key:
        return 'Claude · secondary account'
    credential_env = run.get('credential_env')
    if 'codex' in text and isinstance(credential_env, str):
        try:
            return subscription_provider_name(credential_env)
        except ValueError:
            return None
    for marker, provider in [('claude', 'Claude'), ('codex', 'Codex'), ('glm', 'GLM / Z.ai')]:
        if marker in text:
            return provider
    return None


class QuotaHistory:
    def __init__(self, path: Path):
        self.path = path

    def collect(self, cards, runs, *, now=None):
        now = time.time() if now is None else now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=10) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS quota_samples (
                provider TEXT, window TEXT, at REAL, used REAL, reset REAL,
                costs TEXT, drain REAL, allocations TEXT,
                PRIMARY KEY(provider, window, at))''')
            db.execute('BEGIN IMMEDIATE')
            # Readings created before multi-account support used the current
            # login, but run-cost attribution could not know that. Preserve the
            # usage history under its new name while dropping ambiguous shares.
            db.execute(
                "UPDATE quota_samples SET provider=?, costs='{}', allocations='{}' "
                "WHERE provider=?",
                ('Codex · current login', 'Codex'),
            )
            for card in cards:
                if card.get('status') != 'ok' or card.get('kind') != 'subscription':
                    continue
                at = timestamp(card.get('updated_at'))
                if at is None or now - at > 900 or at > now + 60:
                    continue
                provider = card['name']
                run_provider = card.get('run_provider', provider)
                for window in card.get('windows', []):
                    used = number(window.get('used_percent'))
                    if used is None:
                        continue
                    name = window['name']
                    previous = db.execute('SELECT at, used, reset, costs FROM quota_samples WHERE provider=? AND window=? ORDER BY at DESC LIMIT 1', (provider, name)).fetchone()
                    if previous and at <= previous[0]:
                        continue
                    model = str(window.get('model') or '').lower()
                    costs = {r['key']: r['cost_usd'] for r in runs
                             if provider_for(r) == run_provider and number(r.get('cost_usd')) is not None
                             and (not model or model in str(r.get('model', '')).lower())}
                    reset = timestamp(window.get('resets_at'))
                    drain, allocations = None, {}
                    if previous and 0 < at - previous[0] <= 900:
                        old_at, old_used, old_reset, old_costs = previous
                        same_cycle = ((reset is None and old_reset is None) or
                                      (reset is not None and old_reset is not None and abs(reset - old_reset) < 60 and at < old_reset))
                        if same_cycle and used >= old_used:
                            drain = used - old_used
                            before = json.loads(old_costs)
                            weights = {key: cost - before[key] for key, cost in costs.items()
                                       if key in before and cost > before[key]}
                            total = sum(weights.values())
                            if total and drain:
                                allocations = {key: drain * weight / total for key, weight in weights.items()}
                    db.execute('INSERT INTO quota_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                               (provider, name, at, used, reset, json.dumps(costs), drain, json.dumps(allocations)))
            db.execute('DELETE FROM quota_samples WHERE at < ?', (now - 30 * 86400,))
            series, run_totals = {}, {}
            for provider, name, at, used, reset, drain, allocations in db.execute('SELECT provider, window, at, used, reset, drain, allocations FROM quota_samples ORDER BY provider, window, at'):
                key = provider + ' / ' + name
                points = series.setdefault(key, {'provider': provider, 'window': name, 'points': []})['points']
                if at >= now - 86400:
                    points.append({'at': at, 'used': used, 'reset': reset, 'drain': drain})
                for run, share in json.loads(allocations).items():
                    totals = run_totals.setdefault(run, {})
                    totals[key] = totals.get(key, 0) + share
            return {'series': list(series.values()), 'runs': run_totals, 'retention_days': 30,
                    'note': 'Estimates split observed quota increases by each subscription run’s increase in recorded token cost. Other sessions and delayed cost records can skew the split. Resets and gaps over 15 minutes are excluded. Each quota window is separate; do not add them together.'}
