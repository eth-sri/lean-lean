"""Human-readable Docker names and full, non-secret monitoring labels.

Optional metadata never prevents legacy callers from starting a container.
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import yaml

PREFIX = 'org.openai.leanlean.'


def container_identity(image: str, metadata=None, argv=None):
    values = dict(metadata or {})
    args = list(sys.argv[1:] if argv is None else argv)
    manifest = None
    for index, arg in enumerate(args):
        if arg in {'--run-manifest', '--manifest'} and index + 1 < len(args):
            manifest = args[index + 1]
            break
        if arg.startswith(('--run-manifest=', '--manifest=')):
            manifest = arg.split('=', 1)[1]
            break
    if manifest is None:
        manifest = next((a for a in args if a.endswith(('.yaml', '.yml')) and '\n' not in a and len(a) < 1024), None)
    if manifest:
        try:
            path = Path(manifest).resolve()
            if path.stat().st_size < 2 * 1024 * 1024:
                data = yaml.load(path.read_text(), Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
                if isinstance(data, dict):
                    kind = str(data.get('kind', ''))
                    role = next((r for r in ('postprocessing', 'preprocessing', 'reconstruction', 'evaluation') if r in kind), 'worker')
                    values.setdefault('role', role)
                    run_id = data.get('postprocess_id') or data.get('run_id')
                    if run_id:
                        values.setdefault('run_id', str(run_id))
                    values.setdefault('manifest', str(path))
                    execution = data.get('execution') or {}
                    model = execution.get('model') if isinstance(execution, dict) else None
                    model = model or data.get('model')
                    if isinstance(model, str):
                        values.setdefault('model', model)
        except (OSError, ValueError, yaml.YAMLError):
            pass
    instance = re.match(r'leanlean-(.+?)-(?:base|stripped|raw|optimized|cache):', image)
    if instance:
        values.setdefault('instance_id', instance[1])
    allowed = {'run_id', 'role', 'model', 'instance_id', 'worker_id', 'manifest'}
    labels = {PREFIX + k: str(v) for k, v in values.items() if k in allowed and v is not None and str(v)}
    def slug(value, length):
        return re.sub(r'[^a-zA-Z0-9_.-]+', '-', str(value)).strip('-.')[:length] or 'unknown'
    suffix = uuid.uuid4().hex[:8]
    if values.get('run_id'):
        name = 'leanlean-' + '-'.join([slug(values.get('role', 'worker'), 16), slug(values['run_id'], 46), slug(values.get('instance_id') or values.get('worker_id') or 'worker', 34), suffix])
    elif values.get('role') == 'relay':
        name = f'leanlean-relay-{suffix}'
    else:
        name = f'leanlean-{suffix}'
    return name, labels
