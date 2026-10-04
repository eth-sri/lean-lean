from __future__ import annotations

import os
import time
import ijson
from collections import Counter
from pathlib import Path

from leanlean.capture_disposition import capture_is_scoring

from .costs import AccountingIndex
from .live_costs import live_proxy_journals
from .common import local_path, mapping, number, read_record, stamp

ACTIVE = {'running', 'starting', 'preparing', 'replaying', 'verifying', 'waiting_for_quota', 'waiting_to_retry'}
TERMINAL = {'complete', 'completed', 'succeeded', 'success', 'failed', 'quota_exhausted', 'cancelled', 'complete_with_failures'}
ROOTS = ('evaluation', 'postprocessing', 'preprocessing', 'reconstruction', 'evaluation_queues',
         'evaluation_families', 'postprocessing-queue', 'heartbeats', 'source_preparation', 'scheduled',
         'router')
NAMES = {'run.yaml', 'run.yml', 'run.json', 'state.yaml', 'state.json', 'queue.json', 'progress.json', 'report.json', 'summary.json'}


class RunIndex:
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.worker_cache = {}
        self.accounting = AccountingIndex()
        self.progress_cache = {}

    def records(self):
        found = []
        for name in ROOTS:
            base = self.root / 'runs' / name
            if not base.exists():
                continue
            for parent, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs
                            if not d.startswith(('retired-', 'abandoned-'))
                            and d not in {'logs', 'playback', 'cache', 'work', 'repos', 'engine', '.git', 'repositories'}]
                for filename in files:
                    if filename in NAMES:
                        found.append(Path(parent) / filename)
        return found

    def _read(self, path):
        try:
            st = path.stat()
        except OSError:
            return {}
        sig = (st.st_mtime_ns, st.st_size)
        old = self.cache.get(str(path))
        if old and old[0] == sig:
            return old[1]
        v = read_record(path)
        self.cache[str(path)] = (sig, v)
        return v

    def _workers(self, output, *, active=False):
        # Cache only small projections, never the large prompts/trajectories.
        workers = []
        if output is None or not output.is_dir():
            return workers
        for f in output.glob('*/*.traj.json'):
            try:
                st = f.stat()
            except OSError:
                continue
            key, sig = str(f), (st.st_mtime_ns, st.st_size)
            old = self.worker_cache.get(key)
            if old and old[0] == sig:
                workers.append(dict(old[1]))
                continue
            info, stats = {}, {}
            try:
                # Stop before config/attempt transcripts; keep only scalar accounting.
                with f.open('rb') as stream:
                    for prefix, event, value in ijson.parse(stream):
                        if prefix == 'info.exit_status' and event == 'string':
                            info['exit_status'] = value
                        elif prefix in {'info.model_stats.instance_cost', 'info.model_stats.api_calls'} and event == 'number':
                            stats[prefix.rsplit('.', 1)[-1]] = value
                        elif prefix == 'info.model_stats' and event == 'end_map':
                            break
            except (OSError, ValueError, ijson.JSONError):
                pass
            cost = number(stats.get('instance_cost'))
            worker = {'id': f.parent.name, 'exit_status': info.get('exit_status', 'unknown'),
                      'cost_usd': cost, 'api_calls': number(stats.get('api_calls')), 'updated_at': st.st_mtime}
            self.worker_cache[key] = (sig, worker)
            workers.append(dict(worker))
        by_id = {w['id']: w for w in workers}
        # Live native captures precede final trajectories. Keep only counters;
        # never retain source, tool output, prompts, or duplicate cost estimates.
        captures = {}
        for f in sorted(output.glob('*/playback/capture_*/playback.json')):
            try:
                if capture_is_scoring(f):
                    captures[f.parents[2].name] = f  # newest retained attempt
            except (OSError, ValueError):
                continue
        for instance, f in captures.items():
            try:
                st = f.stat()
                sig = (st.st_mtime_ns, st.st_size)
                key = str(f)
                old = self.progress_cache.get(key)
                if old and old[0] == sig:
                    counters = dict(old[1])
                else:
                    counters = {}
                    with f.open('rb') as stream:
                        for prefix, event, value in ijson.parse(stream):
                            if event != 'number':
                                continue
                            if prefix in {
                                'edit_count', 'completed_action_count', 'capture_failure_count'
                            }:
                                counters[prefix] = int(value)
                            elif prefix == 'wall_time_seconds':
                                counters['elapsed_seconds'] = float(value)
                    self.progress_cache[key] = (sig, dict(counters))
                archives = list(f.parent.glob('*.sources.tar.gz'))
                counters['snapshot_count'] = len(archives)
                activity = st.st_mtime
                native = f.parent / 'native.stdout.jsonl'
                if native.is_file():
                    activity = max(activity, native.stat().st_mtime)
                w = by_id.setdefault(instance, {
                    'id': instance, 'exit_status': 'running' if active else 'unknown',
                    'cost_usd': None, 'api_calls': None, 'updated_at': activity,
                })
                w.update(counters)
                w['updated_at'] = max(w['updated_at'], activity)
            except (OSError, ValueError, ijson.JSONError):
                continue
        return list(by_id.values())

    def _replay_progress(self, manifest):
        stages, built, archived = [], 0, 0
        for capture in manifest.get('captures', []):
            if not isinstance(capture, dict):
                continue
            output = local_path(self.root, capture.get('output_directory'))
            if output is None:
                continue
            playback = capture.get('playback')
            if not isinstance(playback, str):
                continue
            derived = local_path(self.root, str(output / Path(playback).with_name('replayed-playback.json')))
            if derived is not None and not derived.exists():
                derived = local_path(self.root, str(output / Path(playback).with_name('postprocessed-playback.json')))
            if derived is None:
                continue
            count = 0
            updated = None
            try:
                st = derived.stat()
                sig = (st.st_mtime_ns, st.st_size)
                cached = self.progress_cache.get(str(derived))
                if cached and cached[0] == sig:
                    count = cached[1]
                else:
                    with derived.open('rb') as stream:
                        for prefix, event, value in ijson.parse(stream):
                            if prefix == 'checkpoint_build_attempt_count' and event == 'number':
                                count = int(value)
                                break
                    self.progress_cache[str(derived)] = (sig, count)
                updated = st.st_mtime
            except (OSError, ValueError, ijson.JSONError):
                pass
            expected = len(capture.get('archives') or [])
            built += count
            archived += expected
            stages.append({'name': str(capture.get('instance_id', 'unknown')),
                           'stage': f'{count} builds recorded / {expected} archived states',
                           'status': 'replay recorded' if updated else 'pending', 'updated_at': updated})
        return stages, built, archived

    def collect(self):
        rows, errors = [], 0
        paths = self.records()
        for path in paths:
            v = self._read(path)
            if not v or not (v.get('status') or v.get('run_id') or v.get('postprocess_id')):
                continue
            # Prefer the lifecycle record over sibling progress/summary files.
            if path.name not in {'run.yaml', 'run.yml', 'run.json'} and any((path.parent / n).exists() for n in ('run.yaml', 'run.json')):
                continue
            category = path.relative_to(self.root / 'runs').parts[0]
            is_worker_family = str(v.get('kind', '')).endswith('_evaluation_worker_family_state')
            if is_worker_family:
                # Current family artifacts live below runs/evaluation/worker-families,
                # while older ones used runs/evaluation_families. The state kind,
                # not the directory spelling, is the stable discriminator.
                category = 'evaluation_family'
            execution, artifacts = mapping(v.get('execution')), mapping(v.get('artifacts'))
            manifest_value = mapping(v.get('experiment')).get('manifest') or v.get('manifest')
            if not manifest_value and category == 'postprocessing' and path.name == 'queue.json':
                manifest_value = str(self.root / 'experiments' / path.parent.relative_to(self.root / 'runs') / 'manifest.yaml')
            if not manifest_value and category == 'router':
                candidate = self.root / 'experiments/evaluation' / f"{v.get('run_id') or path.parent.name}.yaml"
                if candidate.is_file():
                    manifest_value = str(candidate)
            manifest_path = local_path(self.root, manifest_value)
            manifest = self._read(manifest_path) if manifest_path else {}
            family_model = {}
            family_reference = {}
            if is_worker_family:
                family_model_path = local_path(
                    self.root, mapping(manifest.get('model_config')).get('path'))
                family_model = self._read(family_model_path) if family_model_path else {}
                family_reference_path = local_path(
                    self.root, mapping(manifest.get('canonical_manifest')).get('path'))
                family_reference = self._read(family_reference_path) if family_reference_path else {}
            raw_source = v.get('source_run') or manifest.get('source_run')
            source = mapping(raw_source)
            source_path = local_path(self.root, source.get('artifact') if source else raw_source)
            source_record = self._read(source_path) if source_path else {}
            source_execution = mapping(source_record.get('execution'))
            dataset = (mapping(v.get('dataset')) or mapping(source_record.get('dataset'))
                       or mapping(family_reference.get('dataset')))
            output_value = artifacts.get('output_directory') or manifest.get('output_directory')
            if not output_value and category == 'evaluation':
                output_value = str(Path('output/evaluation') / path.parent.relative_to(self.root / 'runs/evaluation'))
            output = local_path(self.root, output_value)
            workers = self._workers(output, active=v.get('status') in ACTIVE) if category == 'evaluation' else []
            costs = [w['cost_usd'] for w in workers if w['cost_usd'] is not None]
            exits = Counter(w['exit_status'] for w in workers)
            failed = sum(count for s, count in exits.items() if s in {'ExecutionFailed', 'ProviderFailed', 'QuotaExceeded', 'Error', 'Timeout'})
            total = execution.get('repositories')
            if not isinstance(total, int):
                repos = v.get('repositories') or manifest.get('repositories')
                total = len(repos) if isinstance(repos, (list, dict)) else None
            life = mapping(v.get('lifecycle'))
            started = life.get('last_attempt_started_at') or life.get('started_at') or v.get('started_at')
            status = str(v.get('status') or 'unknown')
            access = mapping(v.get('access')) or mapping(source_record.get('access'))
            mode = access.get('mode')
            credential_env = access.get('credential_env')
            router_jobs = mapping(v.get('jobs')) if category == 'router' else {}
            router_statuses = Counter(
                str(item.get('status') or 'unknown')
                for item in router_jobs.values() if isinstance(item, dict)
            )
            if is_worker_family:
                total = len(mapping(v.get('components')))
                started = started or v.get('created_at')
            if category == 'router':
                total = len(router_jobs)
                if router_statuses.get('active') and status == 'configured':
                    # Active ownership remains authoritative while the dispatcher
                    # resolves the next immutable singleton manifest.
                    status = 'running'
                started_values = [
                    attempt.get('started_at')
                    for item in router_jobs.values() if isinstance(item, dict)
                    for attempt in (item.get('attempts') or []) if isinstance(attempt, dict)
                    if isinstance(attempt.get('started_at'), str)
                ]
                if not started and started_values:
                    started = min(started_values)
            settings = mapping(manifest.get('resolved_run_settings'))
            model = (execution.get('model') or source_execution.get('model') or v.get('model')
                     or mapping(manifest.get('execution')).get('model') or manifest.get('model')
                     or settings.get('model') or family_model.get('model'))
            # Small, explicitly selected progress rows; no raw logs or config contents.
            stages = []
            stage_records = mapping(
                v.get('components') if is_worker_family else v.get('jobs') or v.get('repositories'))
            family_components = {
                str(item.get('repository')): item
                for item in (manifest.get('pending') or []) if isinstance(item, dict)
            } if is_worker_family else {}
            for name, item in stage_records.items():
                if not isinstance(item, dict):
                    continue
                item_status = item.get('phase') if is_worker_family else item.get('status')
                if item_status:
                    stage = item.get('stage')
                    stage_started = None
                    stage_finished = None
                    stage_run_key = None
                    if category == 'router':
                        attempts = item.get('attempts')
                        attempt = (attempts[-1] if isinstance(attempts, list) and attempts
                                   and isinstance(attempts[-1], dict) else {})
                        account = str(attempt.get('account') or item.get('last_account') or '')
                        account = account.removeprefix('OPENAI_SUBSCRIPTION_KEY_')
                        if item['status'] == 'pending':
                            stage = f"retry queued · last {account}" if account else 'FIFO pending'
                        elif account:
                            number = attempt.get('number')
                            stage = f"{account} · attempt {number}" if number else account
                        if item['status'] != 'pending':
                            stage_started = attempt.get('started_at')
                            stage_finished = attempt.get('finished_at') or item.get('completed_at')
                            run_artifact = local_path(self.root, attempt.get('run_artifact'))
                            if run_artifact is not None:
                                stage_run_key = str(run_artifact.relative_to(self.root))
                    elif is_worker_family:
                        attempts = item.get('attempts', 0)
                        stage = f'launch attempt {attempts}' if attempts else 'awaiting a free slot'
                        stage_started = item.get('launched_at')
                        component = family_components.get(str(name), {})
                        run_artifact = local_path(self.root, component.get('run_artifact'))
                        if run_artifact is not None:
                            stage_run_key = str(run_artifact.relative_to(self.root))
                    if 'checkpoints_total' in item:
                        stage = f"{item.get('checkpoints_completed', 0)} / {item['checkpoints_total']} checkpoints"
                    stages.append({'name': str(name), 'stage': stage, 'status': item_status,
                                   'started_at': stage_started, 'finished_at': stage_finished,
                                   'run_key': stage_run_key})
                else:
                    for stage, value in item.items():
                        if isinstance(value, dict) and value.get('status'):
                            stages.append({'name': str(name), 'stage': stage, 'status': value['status']})
            replay_stages, replay_built, replay_total = self._replay_progress(manifest) if category == 'postprocessing' and v.get('mode') == 'replay' else ([], None, None)
            if replay_stages:
                stages = replay_stages
            run_id = str(v.get('run_id') or v.get('postprocess_id') or manifest.get('run_id') or path.parent.name)
            display_name = v.get('display_name') or manifest.get('display_name')
            if category == 'router':
                display_name = display_name or f'FIFO router · {run_id}'
            elif is_worker_family:
                display_name = display_name or f'{model or "Evaluation"} refill pool'
            updated = path.stat().st_mtime
            if workers:
                updated = max(updated, max(w['updated_at'] for w in workers))
            completed = (sum(router_statuses[s] for s in ('complete', 'complete_external'))
                         if category == 'router' else replay_built if replay_stages else
                         sum(s['status'] == 'complete' for s in stages) if is_worker_family else
                         (sum(w['exit_status'] not in {'unknown', 'running', 'Running'} for w in workers)
                          if workers else (sum(s['status'] in TERMINAL for s in stages) if stages else None)))
            waiting = (router_statuses.get('pending', 0) if category == 'router'
                       else sum(s['status'] in ({'queued'} if is_worker_family else {'waiting'}) for s in stages))
            active_jobs = (sum(router_statuses[s] for s in ('active', 'active_external'))
                           if category == 'router' else
                           sum(s['status'] in ({'starting', 'running'} if is_worker_family else {'running'})
                               for s in stages))
            rows.append({'key': str(path.relative_to(self.root)), 'id': run_id,
                         'display_name': display_name if isinstance(display_name, str) else None,
                         'category': category,
                         'model': model if isinstance(model, str) else None,
                         'reasoning_effort': execution.get('reasoning_effort') or source_execution.get('reasoning_effort') or settings.get('reasoning_effort') or family_model.get('reasoning_effort'),
                         'harness': execution.get('harness') or source_execution.get('harness') or family_model.get('harness'),
                         'dataset': dataset.get('id'), 'variant': dataset.get('variant') or settings.get('repository_variant'),
                         'status': status, 'started_at': started, 'finished_at': life.get('finished_at') or v.get('finished_at'),
                         'updated_at': updated, 'attempts': life.get('attempt_count'),
                         'manifest': str(manifest_path) if manifest_path else None,
                         'canonical': local_path(self.root, artifacts.get('run_artifact')) in {None, path.resolve()},
                         'output': str(output) if output else None,
                         'session': (f'router-{run_id}' if category == 'router' else
                                     artifacts.get('tmux_session') or v.get('session') or manifest.get('tmux_session')
                                     or mapping(manifest.get('outputs')).get('tmux_session')),
                         'source_run_id': v.get('source_run_id') or source.get('run_id') or source_record.get('run_id'),
                         'total': replay_total if replay_stages else total,
                         'completed': completed,
                         'progress_unit': ('repositories' if category == 'router' else
                                           'builds / archived states' if replay_stages else
                                           'worker outcomes' if category == 'evaluation' else 'jobs'),
                         'submitted': exits.get('Submitted', 0),
                         'failed': (router_statuses.get('failed', 0) if category == 'router' else
                                    sum(s['status'] == 'failed' for s in stages) if is_worker_family else failed),
                         'snapshot_count': sum(w.get('snapshot_count', 0) for w in workers),
                         'capture_failure_count': sum(w.get('capture_failure_count', 0) for w in workers),
                         'waiting': waiting,
                         'active_jobs': active_jobs,
                         'pending_jobs': (router_statuses.get('pending', 0) if category == 'router' else
                                          waiting if is_worker_family else None),
                         'failed_jobs': (router_statuses.get('failed', 0) if category == 'router' else
                                         sum(s['status'] == 'failed' for s in stages) if is_worker_family else None),
                         'worker_limit': manifest.get('maximum_workers') if is_worker_family else None,
                         'cost_usd': sum(costs) if costs else None, 'cost_workers': len(costs),
                         'cost_source': 'worker trajectory totals',
                         'accounted_calls': 0,
                         'cost_basis': 'subscription API equivalent' if mode == 'subscription' else ('recorded API estimate' if mode == 'api' else 'recorded estimate; billing mode unknown'),
                         'access_mode': mode,
                         'credential_env': credential_env if isinstance(credential_env, str) else None,
                         'workers': workers, 'stages': stages,
                         'error': bool(life.get('error') or v.get('error')),
                         'quota_wait': {k: val for k, val in mapping(life.get('quota_wait')).items() if k in {'reason', 'wait_until', 'resets_at'}},
                         })
        # Look up live journals only after exact run/manifest identities exist.
        # Never discover them by model alone: concurrent runs may share a model.
        live_sources = live_proxy_journals(self.root, rows)
        for row in rows:
            if row['category'] != 'evaluation':
                continue
            record = self._read(self.root / row['key'])
            execution = mapping(record.get('execution'))
            rates = mapping(mapping(execution.get('pricing')).get('rates'))
            accounting = self.accounting.read(
                local_path(self.root, row['output']), rates, row['model'],
                live_files=live_sources.get(row['key'], ()),
            )
            row['accounted_calls'] = accounting['calls']
            if accounting['cost'] is not None and (row['status'] in ACTIVE or row['cost_usd'] is None):
                row['cost_usd'] = accounting['cost']
                row['cost_source'] = ('live' if accounting.get('live') else 'archived') + ' proxy accounting at saved run prices; run total counted once'
                if accounting.get('updated_at') is not None:
                    row['updated_at'] = max(row['updated_at'], accounting['updated_at'])
        rows.sort(key=lambda r: (r['status'] in ACTIVE, r['updated_at']), reverse=True)
        return {'updated_at': stamp(), 'runs': rows, 'records_seen': len(paths), 'errors': errors}
