"""Locate a running proxy journal by its verified process ancestry, not model name."""
from __future__ import annotations

import ast
from pathlib import Path

import psutil


def proxy_port(args):
    """Parse the launcher's literal argv without executing code or reading secrets."""
    if '-c' not in args:
        return None
    index = args.index('-c') + 1
    if index >= len(args):
        return None
    code = args[index]
    if len(code) > 32768 or 'from litellm.proxy.proxy_cli import' not in code:
        return None
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != 'main' or not isinstance(node.func.value, ast.Name) or node.func.value.id != 'run_server':
                continue
            values = next((kw.value for kw in node.keywords if kw.arg == 'args'), None)
            argv = ast.literal_eval(values)
            if not isinstance(argv, (list, tuple)) or '--port' not in argv:
                continue
            port = int(argv[argv.index('--port') + 1])
            return port if 0 < port < 65536 else None
    except (ValueError, TypeError, SyntaxError, IndexError):
        pass
    return None


def live_proxy_journals(root: Path, runs):
    # Imported lazily because resource discovery also uses the run index.
    from .resources import descendants_of, match_run
    from .runs import ACTIVE

    # Legacy API workers could truncate a shared journal between repos. Those
    # fragments are not whole-run totals; subscription journals are preserved.
    eligible = [r for r in runs if r['category'] == 'evaluation' and r['status'] in ACTIVE
                and r.get('access_mode') == 'subscription']
    if not eligible:
        return {}
    processes = {}
    for process in psutil.process_iter(['pid', 'ppid', 'cmdline', 'create_time'], ad_value=None):
        info = process.info
        processes[info['pid']] = info
    parents = {pid: info['ppid'] for pid, info in processes.items()}
    owners = {pid: match_run(info.get('cmdline') or [], eligible, root) for pid, info in processes.items()}
    found = {}
    for pid, info in processes.items():
        port = proxy_port(info.get('cmdline') or [])
        if port is None:
            continue
        owner = next((owners.get(parent) for parent in descendants_of(pid, parents) if owners.get(parent)), None)
        if owner is None:
            continue
        path = (root / 'logs/litellm_server/traces' / f'proxy_{port}.cost.jsonl').resolve()
        try:
            # Reject stale journals from earlier users of a recycled port.
            if not path.is_relative_to(root.resolve()) or not info.get('create_time') or path.stat().st_mtime < info['create_time']:
                continue
        except OSError:
            continue
        found.setdefault(owner, []).append(path)
    return found
