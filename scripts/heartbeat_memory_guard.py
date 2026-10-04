"""Enforce the heartbeat run's aggregate and coordinator resource limits."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess

MAX_MEMORY_BYTES = 200_000_000_000


def memory_bytes(value):
    match = re.fullmatch(r'(\d+)([kmg]?)', str(value).lower())
    if not match:
        raise ValueError(f'unsupported memory size: {value}')
    return int(match[1]) * 1024 ** {'': 0, 'k': 1, 'm': 2, 'g': 3}[match[2]]


def validate_spec(manifest):
    guard = manifest['memory_guard']
    parent = guard['slice']
    if not re.fullmatch(r'[a-zA-Z0-9]+\.slice', parent):
        raise ValueError('heartbeat slice must be a dedicated top-level user slice')
    maximum = int(guard['memory_max_bytes'])
    if not 0 < maximum <= MAX_MEMORY_BYTES:
        raise ValueError('heartbeat aggregate memory must not exceed 200 GB')
    workers = int(manifest['parallelism']['workers'])
    worker_memory = memory_bytes(manifest['container']['memory'])
    coordinator = int(guard['coordinator_memory_bytes'])
    if workers <= 0 or worker_memory <= 0 or coordinator <= 0:
        raise ValueError('worker and coordinator resources must be positive')
    if workers * worker_memory > memory_bytes(manifest['container']['max_total_memory']):
        raise ValueError('worker memory exceeds declared worker aggregate')
    if workers * worker_memory + coordinator > maximum:
        raise ValueError('worker and coordinator memory exceed aggregate ceiling')
    if guard['memory_swap_max_bytes'] != 0:
        raise ValueError('heartbeat swap must remain disabled')
    if not 0 < guard['memory_high_bytes'] <= maximum:
        raise ValueError('invalid memory pressure threshold')
    if manifest['container']['cgroup_parent'] != parent:
        raise ValueError('measurement container escaped heartbeat slice')
    for repository in manifest['repositories']:
        record = repository['materialization']
        if record.get('backend') != 'shared_environment_v1':
            raise ValueError('memory guard requires the shared-environment builder')
        if manifest.get('preparation_mode') != 'restore_in_capped_container':
            raise ValueError('all preparation must run inside capped containers')
    return guard


def cgroup_root(parent):
    relative = subprocess.check_output(
        ['systemctl', '--user', 'show', parent, '-p', 'ControlGroup', '--value'], text=True,
    ).strip()
    if not relative.startswith('/') or relative == '/':
        raise ValueError('heartbeat slice has no kernel cgroup')
    return Path('/sys/fs/cgroup') / relative.lstrip('/')


def verify_limits(manifest, *, require_membership=True):
    guard = validate_spec(manifest)
    root = cgroup_root(guard['slice'])
    maximum = (root / 'memory.max').read_text().strip()
    swap = (root / 'memory.swap.max').read_text().strip()
    if maximum == 'max' or int(maximum) > guard['memory_max_bytes'] or swap != '0':
        raise ValueError('kernel memory/swap limits do not enforce the manifest')
    membership = Path('/proc/self/cgroup').read_text()
    if require_membership and '/' + guard['slice'] + '/' not in membership:
        raise ValueError('coordinator is outside the capped heartbeat slice')
    return root


def launch(manifest, manifest_path, root):
    guard = validate_spec(manifest)
    subprocess.run(['systemctl', '--user', 'start', guard['slice']], check=True)
    subprocess.run([
        'systemctl', '--user', 'set-property', '--runtime', guard['slice'],
        f'MemoryMax={guard["memory_max_bytes"]}', f'MemoryHigh={guard["memory_high_bytes"]}',
        'MemorySwapMax=0', f'TasksMax={guard["pids_limit"]}',
    ], check=True)
    verify_limits(manifest, require_membership=False)
    command = [
        'systemd-run', '--user', '--scope', '--collect',
        '--unit=heartbeat-' + manifest['run_id'], '--slice=' + guard['slice'],
        '-p', f'MemoryMax={guard["coordinator_memory_bytes"]}', '-p', 'MemorySwapMax=0',
        str(root / '.venv/bin/python'), str(root / 'scripts/measure_heartbeats.py'), str(manifest_path),
    ]
    os.execvp(command[0], command)


def verify_environment(environment, manifest):
    """Verify the actual container's kernel ancestry and resource limits."""
    root = verify_limits(manifest)
    row = json.loads(subprocess.check_output(
        ['docker', 'inspect', environment.container_id], text=True,
    ))[0]
    config = row['HostConfig']
    expected = memory_bytes(manifest['container']['memory'])
    if config['Memory'] != expected or config['MemorySwap'] != expected:
        raise ValueError('container memory or swap limit mismatch')
    if config['PidsLimit'] != manifest['container']['pids_limit']:
        raise ValueError('container PID limit mismatch')
    if config['CgroupParent'] != manifest['memory_guard']['slice']:
        raise ValueError('container escaped heartbeat slice')
    pid = row['State']['Pid']
    membership = (Path('/proc') / str(pid) / 'cgroup').read_text().strip().split('::', 1)[1]
    actual = Path('/sys/fs/cgroup') / membership.lstrip('/')
    if not actual.is_relative_to(root):
        raise ValueError('kernel container cgroup escaped heartbeat slice')
    if int((actual / 'memory.max').read_text()) != expected or (actual / 'memory.swap.max').read_text().strip() != '0':
        raise ValueError('kernel container resource enforcement mismatch')
    return {'verified': True, 'aggregate_memory_bytes': manifest['memory_guard']['memory_max_bytes'],
            'worker_memory_bytes': expected, 'swap_bytes': 0, 'cgroup': str(actual),
            'preparation_mode': manifest['preparation_mode']}
