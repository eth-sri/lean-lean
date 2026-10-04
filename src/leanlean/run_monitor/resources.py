from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import psutil

from .common import command, number, stamp
from .runs import ACTIVE
from .storage import collect_storage


def memory_bytes(value):
    match = re.match(r'^\s*([\d.]+)\s*([KMGTPE]?i?B)?', str(value), re.I)
    if not match:
        return None
    unit = (match[2] or 'B').upper()
    scale = {'B': 1, 'KB': 1000, 'MB': 1000**2, 'GB': 1000**3, 'TB': 1000**4,
             'KIB': 1024, 'MIB': 1024**2, 'GIB': 1024**3, 'TIB': 1024**4}
    return float(match[1]) * scale.get(unit, 1)


def descendants_of(pid, parents):
    chain, seen = [], set()
    while pid and pid not in seen:
        seen.add(pid)
        chain.append(pid)
        pid = parents.get(pid)
    return chain


def match_run(args, runs, root):
    # Exact argument paths only: never substring-match a shell script's contents.
    tokens = set()
    for arg in args:
        if isinstance(arg, str) and '\n' not in arg and len(arg) < 1024:
            value = arg.split('=', 1)[-1] if arg.startswith('--') else arg
            tokens.add(value)
            if '/' in value:
                tokens.add(str((root / value).resolve()))
    matches = [r for r in runs if r.get('manifest') in tokens or str(root / r['key']) in tokens]
    current = [r for r in matches if r.get('canonical', True)]
    matches = current or matches
    return matches[0]['key'] if len(matches) == 1 else None


def match_session(name, runs):
    """Match an exact recorded or conventional eval tmux session name."""
    matches = [r for r in runs if name == r.get('session') or (
        r.get('category') == 'evaluation' and name == f"eval-{r['id']}"
    )]
    current = [r for r in matches if r.get('canonical', True)]
    matches = current or matches
    return matches[0]['key'] if len(matches) == 1 else None


class Resources:
    def __init__(self, root):
        self.root = root
        self.peak = {}
        self.process_cpu = {}
        self.last_details = {}
        self.details_at = 0
        self.container_owners = {}
        self.disk_publish = None
        self.storage_last = None

    def machine(self):
        ram, swap = psutil.virtual_memory(), psutil.swap_memory()
        seen, filesystems = set(), []
        paths = [str(self.root), '/', str(Path.home()), '/tmp']
        paths += [p.mountpoint for p in psutil.disk_partitions() if p.fstype not in {'squashfs', 'tmpfs'} and '/docker/' not in p.mountpoint and '/snap/' not in p.mountpoint]
        for path in paths:
            try:
                device = os.stat(path).st_dev
                if device in seen:
                    continue
                seen.add(device)
                u = shutil.disk_usage(path)
                filesystems.append({'path': path, 'total': u.total, 'used': u.used, 'free': u.free})
            except OSError:
                pass
        return {'updated_at': stamp(), 'ram': {'total': ram.total, 'used': ram.total - ram.available,
                'available': ram.available, 'cached': getattr(ram, 'cached', 0), 'percent': ram.percent},
                'swap': {'total': swap.total, 'used': swap.used, 'percent': swap.percent},
                'cpu_percent': psutil.cpu_percent(interval=None), 'cpu_count': os.cpu_count(),
                'load': list(os.getloadavg()), 'filesystems': filesystems,
                'boot_time': psutil.boot_time()}

    def workloads(self, runs):
        now = time.monotonic()
        processes, parents, run_owner = {}, {}, {}
        for p in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'memory_info', 'cpu_times', 'create_time'], ad_value=None):
            info = p.info
            pid = info['pid']
            parents[pid] = info['ppid']
            args = info['cmdline'] or []
            key = match_run(args, runs, self.root)
            if key:
                run_owner[pid] = key
            cpu = info['cpu_times']
            seconds = cpu.user + cpu.system if cpu else 0
            old = self.process_cpu.get(pid)
            percent = max(0, (seconds - old[1]) / (now - old[0]) * 100) if old and old[2] == info['create_time'] and now > old[0] else None
            self.process_cpu[pid] = (now, seconds, info['create_time'])
            processes[pid] = {'pid': pid, 'name': info['name'], 'args': args,
                              'rss': info['memory_info'].rss if info['memory_info'] else 0,
                              'cpu_percent': percent}
        self.process_cpu = {p: v for p, v in self.process_cpu.items() if p in processes}
        chains = {pid: descendants_of(pid, parents) for pid in processes}
        owner = {pid: next((run_owner[a] for a in chain if a in run_owner), None) for pid, chain in chains.items()}
        sessions, session_pids, tmux_error = [], {}, None
        try:
            lines = command(['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_pid}\t#{pane_current_command}\t#{pane_dead}\t#{session_created}'], timeout=6)
            grouped = {}
            for line in lines.splitlines():
                name, pid, current, dead, created = line.split('\t')
                entry = grouped.setdefault(name, {'name': name, 'panes': [], 'created': int(created), 'rss': 0, 'cpu_percent': 0, 'run_keys': []})
                entry['panes'].append({'pid': int(pid), 'command': current, 'dead': dead == '1'})
                session_pids[int(pid)] = name
            for pid, chain in chains.items():
                session = next((session_pids[a] for a in chain if a in session_pids), None)
                if session:
                    entry = grouped[session]
                    entry['rss'] += processes[pid]['rss']
                    entry['cpu_percent'] += processes[pid]['cpu_percent'] or 0
                    if owner[pid] and owner[pid] not in entry['run_keys']:
                        entry['run_keys'].append(owner[pid])
            sessions = list(grouped.values())
        except (OSError, RuntimeError, ValueError, TimeoutError) as e:
            tmux_error = type(e).__name__
        for session in sessions:
            if session['run_keys']:
                continue
            key = match_session(session['name'], runs)
            if not key:
                continue
            session['run_keys'] = [key]
            for pid, chain in chains.items():
                if any(p['pid'] in chain for p in session['panes']) and not owner.get(pid):
                    owner[pid] = key
        # Sessions cover older launchers whose artifacts live outside the run registry.
        # They remain clearly identified as session-discovered work.
        discovered = []
        for session in sessions:
            if session['run_keys']:
                continue
            key = 'session:' + session['name']
            pids = [pid for pid, chain in chains.items() if any(p['pid'] in chain for p in session['panes'])]
            busy = any(processes[pid]['name'] not in {'bash', 'sh', 'zsh', 'fish', 'tmux', 'sleep'} for pid in pids)
            if not busy:
                continue
            session['run_keys'] = [key]
            for pid in pids:
                if not owner.get(pid):
                    owner[pid] = key
            discovered.append({'key': key, 'id': session['name'], 'category': 'session',
                'display_name': None,
                'model': None, 'reasoning_effort': None, 'harness': None, 'dataset': None,
                'variant': None, 'status': 'running', 'started_at': None, 'finished_at': None,
                'updated_at': time.time(), 'attempts': None, 'manifest': None, 'output': None,
                'session': session['name'], 'total': None, 'completed': None, 'submitted': 0,
                'failed': 0, 'cost_usd': None, 'cost_workers': 0, 'cost_basis': 'No cost record linked',
                'access_mode': None, 'workers': [], 'stages': [], 'error': False, 'quota_wait': {},
                'discovered': True})
        runs = list(runs) + discovered
        containers, docker_error = [], None
        try:
            stats_text = command(['docker', 'stats', '--no-stream', '--format', '{{json .}}'], timeout=20)
            stats = [json.loads(line) for line in stats_text.splitlines() if line.strip()]
            ids = [s['ID'] for s in stats]
            if set(ids) != set(self.last_details) or now - self.details_at > 60:
                # Explicit field projection excludes environment variables and command arguments.
                fmt = '{"id":{{json .Id}},"pid":{{json .State.Pid}},"image":{{json .Config.Image}},"labels":{{json .Config.Labels}},"mounts":{{json .Mounts}},"started":{{json .State.StartedAt}}}'
                details = command(['docker', 'inspect', '--format', fmt, *ids], timeout=15, allow_partial=True) if ids else ''
                self.last_details = {d['id'][:12]: d for d in map(json.loads, details.splitlines())}
                self.details_at = now
            for s in stats:
                cid = s['ID']
                d = self.last_details.get(cid[:12], {})
                labels = d.get('labels') or {}
                container_pid = d.get('pid')
                candidates = [r for r in runs if labels.get('org.openai.leanlean.run_id') == r['id']]
                ckey = candidates[0]['key'] if len(candidates) == 1 else None
                evidence = 'run label' if ckey else 'unattributed'
                # docker exec client belongs to the host worker, unlike container PID 1.
                for pid, p in processes.items():
                    args = p['args']
                    if not ckey and p['name'] == 'docker' and 'exec' in args and any(a == cid or a == d.get('id') or a == s['Name'] or (len(a) == 64 and a.startswith(cid)) for a in args):
                        if owner.get(pid):
                            ckey, evidence = owner[pid], 'worker process ancestry'
                            break
                if ckey:
                    self.container_owners[cid] = ckey
                elif self.container_owners.get(cid) in {r['key'] for r in runs}:
                    ckey, evidence = self.container_owners[cid], 'previously observed worker owner'
                if not ckey:
                    # Image tags are useful for generation only; replay can reuse those images.
                    tag = d.get('image', '').partition(':')[2]
                    candidates = [r for r in runs if r['category'] == 'evaluation' and r['id'].replace('__', '_') == tag and r['status'] in ACTIVE]
                    if len(candidates) == 1:
                        ckey, evidence = candidates[0]['key'], 'image tag (inferred)'
                mem, _, limit = s.get('MemUsage', '').partition('/')
                used = memory_bytes(mem)
                self.peak[cid] = max(self.peak.get(cid, 0), used or 0)
                role = 'relay' if 'relay' in s['Name'] else ('eval' if ckey and next(r for r in runs if r['key'] == ckey)['category'] == 'evaluation' else 'worker')
                containers.append({'id': cid, 'name': s['Name'], 'image': d.get('image'), 'pid': container_pid,
                    'role': role, 'run_key': ckey, 'attribution': evidence,
                    'worker': labels.get('org.openai.leanlean.instance_id') or labels.get('org.openai.leanlean.worker_id'),
                    'memory': used, 'memory_limit': memory_bytes(limit), 'observed_peak': self.peak[cid],
                    'cpu_percent': number(s.get('CPUPerc', '').rstrip('%')),
                    'pids': number(s.get('PIDs')), 'block_io': s.get('BlockIO'), 'net_io': s.get('NetIO'),
                    'started_at': d.get('started')})
        except Exception as e:
            docker_error = type(e).__name__
        self.container_owners = {c: r for c, r in self.container_owners.items() if c in {v['id'] for v in containers}}
        container_roots = {c['pid'] for c in containers if c.get('pid')}
        container_processes = {pid for pid, chain in chains.items() if any(a in container_roots for a in chain)}
        totals = {}
        for r in runs:
            owned = [p for pid, p in processes.items() if owner.get(pid) == r['key'] and pid not in container_processes]
            cs = [c for c in containers if c['run_key'] == r['key']]
            totals[r['key']] = {'host_rss': sum(p['rss'] for p in owned), 'container_memory': sum(c['memory'] or 0 for c in cs),
                               'cpu_percent': sum(p['cpu_percent'] or 0 for p in owned) + sum(c['cpu_percent'] or 0 for c in cs),
                               'containers': len(cs), 'host_processes': len(owned),
                               'alive': bool(owned or cs), 'pids': [p['pid'] for p in owned[:20]]}
        top = sorted((p for pid, p in processes.items() if pid not in container_processes), key=lambda p: p['rss'], reverse=True)[:15]
        return {'updated_at': stamp(), 'discovered_runs': discovered, 'containers': containers, 'sessions': sorted(sessions, key=lambda s: s['rss'], reverse=True),
                'runs': totals, 'top_processes': [{k: v for k, v in p.items() if k != 'args'} for p in top],
                'docker_error': docker_error, 'tmux_error': tmux_error,
                'counts': dict(Counter(c['role'] for c in containers)), 'total_container_memory': sum(c['memory'] or 0 for c in containers)}

    def disk(self):
        # Separate slow collector, low priority, one filesystem, no deletion actions.
        try:
            self.storage_last = collect_storage(self.root)
            ownership = self.storage_last
        except Exception as error:
            ownership = {**(self.storage_last or {}), 'stale': True, 'error': type(error).__name__}
        paths = [self.root / name for name in ('output', 'runs', 'data', 'prod_repos', 'datasets', '.cache', '.venv', '.worktrees')]
        paths += [Path.home() / '.cache', Path.home() / '.elan', Path.home() / '.local/share/docker']
        sizes, by_path = [], {}
        for path in paths:
            if not path.exists():
                continue
            try:
                out = command(['nice', '-n', '15', 'du', '-x', '-B1', '--max-depth=' + ('4' if path == self.root / 'output' else '1'), str(path)], timeout=25)
                children = []
                total = None
                for line in out.splitlines():
                    size, name = line.split('\t', 1)
                    by_path[name] = int(size)
                    if name == str(path):
                        total = int(size)
                    elif Path(name).parent == path:
                        children.append({'path': name, 'bytes': int(size)})
                sizes.append({'path': str(path), 'bytes': total, 'children': sorted(children, key=lambda c: c['bytes'], reverse=True)[:12], 'updated_at': stamp()})
            except Exception as e:
                sizes.append({'path': str(path), 'bytes': None, 'error': type(e).__name__})
            if self.disk_publish:
                self.disk_publish({'updated_at': stamp(), 'scanning': True, 'ownership': ownership, 'directories': list(sizes), 'by_path': dict(by_path), 'docker': []})
        docker = []
        try:
            out = command(['docker', 'system', 'df', '--format', '{{json .}}'], timeout=40)
            for line in out.splitlines():
                v = json.loads(line)
                docker.append({k: v.get(k) for k in ('Type', 'TotalCount', 'Active', 'Size', 'Reclaimable')})
        except Exception:
            pass
        return {'updated_at': stamp(), 'scanning': False, 'ownership': ownership, 'directories': sizes, 'by_path': by_path, 'docker': docker,
                'note': 'Allocated disk blocks; directories and Docker can overlap. Shared image layers are not additive. No automatic cleanup.'}
