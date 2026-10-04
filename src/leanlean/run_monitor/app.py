from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .common import stamp
from .providers import Providers
from .quota_history import QuotaHistory
from .resources import Resources
from .runs import RunIndex

STATIC = Path(__file__).with_name('static')


class Monitor:
    def __init__(self, root, *, providers=True):
        self.root = root.resolve()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.limit_refresh = threading.Event()
        self.last_manual_refresh = float("-inf")
        self.index = RunIndex(self.root)
        self.resources = Resources(self.root)
        self.providers = Providers(self.root / '.cache' / 'run-monitor')
        self.quota_history = QuotaHistory(self.root / '.cache' / 'run-monitor' / 'quota-history.sqlite3')
        self.provider_enabled = providers
        self.sections = {name: {'state': 'loading'} for name in ('machine', 'runs', 'workloads', 'disk', 'limits')}
        self.history = deque(maxlen=240)
        self.started = stamp()

    def loop(self, name, interval, fetch):
        while not self.stop.is_set():
            try:
                value = fetch()
                if name == 'workloads' and value.get('docker_error'):
                    with self.lock:
                        previous = self.sections[name]
                    if previous.get('updated_at'):
                        raise RuntimeError('Docker measurements unavailable')
                value['state'] = 'ready'
                with self.lock:
                    self.sections[name] = value
                    if name == 'machine':
                        self.history.append({'at': value['updated_at'], 'ram': value['ram']['used'], 'cpu': value['cpu_percent']})
            except Exception as error:
                with self.lock:
                    old = self.sections[name]
                    self.sections[name] = {**old, 'state': 'stale' if old.get('updated_at') else 'error',
                                           'error': type(error).__name__, 'checked_at': stamp()}
            if name == 'limits':
                self.limit_refresh.wait(interval)
            else:
                self.stop.wait(interval)

    def start(self):
        def publish_disk(value):
            with self.lock:
                self.sections['disk'] = {**value, 'state': 'ready'}
        self.resources.disk_publish = publish_disk
        def workloads():
            with self.lock:
                runs = self.sections['runs'].get('runs', [])
            return self.resources.workloads(runs)
        jobs = [('machine', 3, self.resources.machine), ('runs', 30, self.index.collect),
                ('workloads', 8, workloads), ('disk', 600, self.resources.disk)]
        if self.provider_enabled:
            def limits():
                force = self.limit_refresh.is_set()
                self.limit_refresh.clear()
                value = self.providers.collect(force=force)
                with self.lock:
                    runs = list(self.sections['runs'].get('runs', []))
                try:
                    value['history'] = self.quota_history.collect(value.get('providers', []), runs)
                except Exception as error:
                    value['history'] = {'error': type(error).__name__}
                return value
            jobs.append(('limits', 120, limits))
        else:
            self.sections['limits'] = {'state': 'ready', 'providers': [], 'disabled': True, 'updated_at': stamp()}
        for name, interval, fetch in jobs:
            threading.Thread(target=self.loop, args=(name, interval, fetch), name=f'monitor-{name}', daemon=True).start()

    def refresh_limits(self):
        with self.lock:
            if not self.provider_enabled:
                return 409, 'Provider requests are disabled.'
            if time.monotonic() - self.last_manual_refresh < 30:
                return 429, 'Please wait 30 seconds between refresh requests.'
            self.last_manual_refresh = time.monotonic()
            self.limit_refresh.set()
        return 202, 'Refreshing all providers. Provider retry deadlines still apply.'

    def snapshot(self):
        with self.lock:
            sections = dict(self.sections)
            sections['runs'] = {**sections['runs'], 'runs': sections['runs'].get('runs', []) + sections['workloads'].get('discovered_runs', [])}
            return {'at': stamp(), 'started_at': self.started, 'root': str(self.root),
                    **sections, 'history': list(self.history)}


def handler_for(monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if (host.split(':', 1)[0] not in {'localhost', '127.0.0.1'}
                    or (origin and urlsplit(origin).netloc != host)
                    or self.headers.get('X-Dashboard-Request') != '1'):
                self.send_error(403)
                return
            if urlsplit(self.path).path != '/api/limits/refresh':
                self.send_error(404)
                return
            status, message = monitor.refresh_limits()
            body = json.dumps({'message': message}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            host = self.headers.get('Host', '').split(':', 1)[0]
            origin = self.headers.get('Origin')
            if host not in {'localhost', '127.0.0.1'} or (origin and urlsplit(origin).netloc != self.headers.get('Host')):
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path == '/api/status':
                body = json.dumps(monitor.snapshot(), allow_nan=False).encode()
                content_type = 'application/json'
            elif path == '/health':
                body = b'{"ok":true}'
                content_type = 'application/json'
            else:
                files = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
                if path not in files:
                    self.send_error(404)
                    return
                file = STATIC / files[path]
                body = file.read_bytes()
                content_type = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Local read-only run and capacity dashboard')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-provider-requests', action='store_true', help='Use local machine/run data only')
    args = parser.parse_args()
    monitor = Monitor(args.root, providers=not args.no_provider_requests)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(monitor))
    server.daemon_threads = True
    monitor.start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    try:
        with Live(refresh_per_second=1) as live:
            while True:
                status = monitor.snapshot()
                table = Table('Collector', 'Status', 'Updated', expand=True)
                for name in ('machine', 'workloads', 'runs', 'limits', 'disk'):
                    value = status[name]
                    table.add_row(name, value['state'], str(value.get('updated_at', 'First scan in progress')))
                live.update(Panel(table, title=f'LeanLean Monitor · http://localhost:{args.port}', subtitle='Read-only · Ctrl+C to stop'))
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop.set()
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
