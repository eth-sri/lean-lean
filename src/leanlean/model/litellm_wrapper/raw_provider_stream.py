"""Persist decoded provider SSE frames before SDK/LiteLLM normalization.

Only installed inside the proxy. No request body or authentication headers are
recorded. SSE data is verbatim (not reserialized JSON); wire framing is decoded.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from .accounting import CURRENT_ACCOUNTING_SCOPE

_WRITE_LOCK = threading.Lock()


def journal_path() -> Path:
    from .litellm_logger import LOG_PATH
    trace_id = os.getenv('LEANLEAN_LITELLM_TRACE_ID', '').strip()
    if not (trace_id.isascii() and trace_id.isdigit()):
        trace_id = 'unscoped'
    return Path(LOG_PATH) / f'proxy_{trace_id}.provider-stream.jsonl'


def _append(record: dict, *, durable: bool = False) -> None:
    payload = (json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8')
    path = journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist before forwarding the event. Logging failure is visible to the
    # caller, never silently converted into apparently complete evidence.
    with _WRITE_LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(payload)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError('provider stream journal write made no progress')
                view = view[count:]
            if durable:
                os.fsync(fd)
        finally:
            os.close(fd)


class _Capture:
    def __init__(self, response):
        self.identity = {
            'schema': 'leanlean-provider-sse-v1',
            'stream_id': uuid.uuid4().hex,
            'agent_context': {'accounting_scope': CURRENT_ACCOUNTING_SCOPE.get()},
            'provider_host': response.request.url.host,
            'http_status': response.status_code,
            'provider_request_id': response.headers.get('x-request-id'),
        }
        self.sequence = 0
        self.usage_frames = 0
        self.done = False
        self.closed = False
        self.write('stream_start', durable=True)

    def write(self, event, *, durable=False, **fields):
        _append({**self.identity, 'ts': time.time(), 'event': event, **fields}, durable=durable)

    def observe(self, sse):
        self.sequence += 1
        try:
            payload = json.loads(sse.data)
        except ValueError:
            payload = None
        usage = payload.get('usage') if isinstance(payload, dict) else None
        if usage is not None:
            self.usage_frames += 1
        self.done = sse.data.startswith('[DONE]')
        self.write('provider_sse', durable=usage is not None or self.done,
                   sequence=self.sequence, sse_event=sse.event, sse_id=sse.id,
                   data=sse.data, data_sha256=hashlib.sha256(sse.data.encode()).hexdigest(),
                   provider_response_id=payload.get('id') if isinstance(payload, dict) else None,
                   model=payload.get('model') if isinstance(payload, dict) else None,
                   provider_usage=usage)
        if self.done:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.write('stream_end', durable=True, frame_count=self.sequence,
                   usage_frame_count=self.usage_frames, done_received=self.done,
                   usage_evidence='observed' if self.usage_frames else 'missing',
                   complete=self.done)


def install_raw_provider_stream_capture() -> None:
    """Hook both OpenAI-compatible SDK stream paths without changing events."""
    from openai import Stream, AsyncStream
    original_sync = Stream._iter_events
    if not getattr(original_sync, '_leanlean_raw_capture', False):
        def sync_events(self):
            capture = _Capture(self.response)
            self._leanlean_raw_stream_id = capture.identity['stream_id']
            try:
                for event in original_sync(self):
                    capture.observe(event)
                    yield event
            finally:
                capture.close()
        sync_events._leanlean_raw_capture = True
        Stream._iter_events = sync_events

    original_async = AsyncStream._iter_events
    if not getattr(original_async, '_leanlean_raw_capture', False):
        async def async_events(self):
            capture = _Capture(self.response)
            self._leanlean_raw_stream_id = capture.identity['stream_id']
            try:
                async for event in original_async(self):
                    capture.observe(event)
                    yield event
            finally:
                capture.close()
        async_events._leanlean_raw_capture = True
        AsyncStream._iter_events = async_events
