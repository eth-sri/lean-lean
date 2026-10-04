"""Allocated home-directory storage, partitioned into non-overlapping types."""
from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .common import stamp


STORAGE_TYPES = {
    'datasets': 'Datasets', 'docker': 'Docker', 'outputs': 'Experiment outputs',
    'logs': 'Logs', 'worktrees': 'Worktrees', 'caches': 'Caches',
    'tools': 'Toolchains & tools', 'projects': 'Other projects', 'other': 'Other files',
}


def dataset_type(name):
    if name.startswith('anthropic_') or name.startswith('anthropic-'):
        return 'FLT'
    if name.startswith('leanlean-mini'):
        return 'LeanLean Mini'
    if any(word in name for word in ('holdout', 'groupfill', 'random24')):
        return 'Holdouts & reconstruction'
    if 'ablation' in name or name.startswith(('transfer-', 'ilean-')):
        return 'Ablation & transfer'
    if name.startswith('palomar'):
        return 'Palomar'
    if name.startswith('leanlean_'):
        return 'Full LeanLean'
    return 'Other datasets'


def summarize_storage(root, home, totals, *, partial=False):
    """Partition one du scan; never add independently measured parent/child sizes."""
    root, home = Path(root), Path(home)
    if home not in totals:
        raise ValueError('Home scan did not complete')
    children = defaultdict(list)
    for path in totals:
        if path != home:
            children[path.parent].append(path)
    targets = {
        root / 'datasets': 'datasets', root / 'output': 'outputs', root / 'runs': 'outputs',
        root / 'logs': 'logs', root / '.worktrees': 'worktrees',
        root / '.cache': 'caches', root / '.venv': 'tools',
        home / '.local/share/docker': 'docker', home / '.cache': 'caches',
        home / '.hf': 'caches', home / '.elan': 'tools',
        home / '.vscode-server': 'tools', home / '.codex': 'tools',
    }
    categories = {key: {'id': key, 'label': label, 'bytes': 0, 'items': []}
                  for key, label in STORAGE_TYPES.items()}

    def add(kind, path, size, label=None):
        category = categories[kind]
        category['bytes'] += size
        if size:
            category['items'].append({'path': str(path), 'label': label or path.name, 'bytes': size})

    def visit(path):
        size = totals[path]
        kind = targets.get(path)
        if kind:
            add(kind, path, size)
            return
        if path.parent == home and not path.name.startswith('.') and not root.is_relative_to(path):
            add('projects', path, size)
            return
        if not any(target != path and target.is_relative_to(path) for target in targets):
            add('other', path, size)
            return
        # du already accounts for hard links, direct files, and directory blocks.
        remainder = size - sum(totals[p] for p in children[path])
        add('other', path, max(0, remainder), str(path.relative_to(home)) or 'Home files')
        for child in children[path]:
            visit(child)

    visit(home)
    values = sorted(categories.values(), key=lambda c: c['bytes'], reverse=True)
    for category in values:
        category['items'].sort(key=lambda i: i['bytes'], reverse=True)
    families = defaultdict(lambda: {'bytes': 0, 'items': []})
    for path in children[root / 'datasets']:
        group = families[dataset_type(path.name)]
        group['bytes'] += totals[path]
        group['items'].append({'path': str(path), 'label': path.name, 'bytes': totals[path]})
    dataset_root = root / 'datasets'
    metadata = totals.get(dataset_root, 0) - sum(totals[p] for p in children[dataset_root])
    if metadata > 0:
        families['Dataset metadata']['bytes'] = metadata
    dataset_families = sorted(
        ({'label': label, **v, 'items': sorted(v['items'], key=lambda i: i['bytes'], reverse=True)}
         for label, v in families.items()), key=lambda v: v['bytes'], reverse=True)
    usage = shutil.disk_usage(home)
    mine = totals[home]
    return {'updated_at': stamp(), 'home': str(home), 'bytes': mine, 'partial': partial,
            'share_of_used_percent': mine / usage.used * 100 if usage.used else 0,
            'share_of_capacity_percent': mine / usage.total * 100 if usage.total else 0,
            'filesystem': {'total': usage.total, 'used': usage.used, 'free': usage.free,
                           'other_used': max(0, usage.used - mine),
                           'reserved': max(0, usage.total - usage.used - usage.free)},
            'categories': values, 'dataset_families': dataset_families,
            'note': 'Your share covers this home directory on its filesystem. Shared hard-linked files are counted once; symlinks and other filesystems are not followed. Other used space is unattributed, not a measurement of individual users.'}


def collect_storage(root, home=None):
    home = Path(home or Path.home()).resolve()
    # One scan gives consistent totals across Docker, datasets and sibling projects.
    result = subprocess.run(['nice', '-n', '15', 'du', '-x', '-B1', '--null', '--max-depth=6', str(home)],
                            capture_output=True, timeout=240)
    totals = {}
    for record in result.stdout.split(b'\0'):
        if not record:
            continue
        size, name = record.split(b'\t', 1)
        totals[Path(name.decode(errors='surrogateescape'))] = int(size)
    return summarize_storage(root, home, totals, partial=result.returncode != 0)
