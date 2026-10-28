"""Buffer complete Argon responses before exposing tool calls; retry up to five times.

Empty GenAI frames keep the client alive while output is buffered. Native
compaction-summary delimiters are removed, preserving their text. Client
disconnects and deadlines cancel upstream work.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from .accounting import CURRENT_ACCOUNTING_SCOPE, scoped_records

ENV = "LEANLEAN_ARGON_REQUEST_RETRIES"
ROUTE = re.compile(r"/models/(?:gemini/)?gemini-4-argon(?:-(?:low|medium|high))?:streamGenerateContent$")
RETRY_HTTP = {408, 429, 500, 502, 503, 504}


def normalize_compaction_summary(body: bytes) -> tuple[bytes, bool]:
    """Avoid Argon's reproducible 429 on AGY's summary wrapper, preserving text."""
    opening, closing = "<CONTEXT_SUMMARY>", "</CONTEXT_SUMMARY>"
    try:
        payload = json.loads(body)
        contents = payload.get("contents", [])
        if not isinstance(contents, list):
            return body, False
        changed = False
        for content in contents:
            if not isinstance(content, dict) or content.get("role") != "user":
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                    continue
                text = part["text"]
                if text.startswith(opening) and text.rstrip().endswith(closing):
                    end = text.rfind(closing)
                    part["text"] = text[len(opening):end] + text[end + len(closing):]
                    changed = True
        if changed:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), True
    except (ValueError, TypeError, AttributeError):
        pass  # Upstream retains authority over malformed requests.
    return body, False


def retry_journal_path(port: str | int | None = None) -> Path:
    from .litellm_logger import LOG_PATH
    value = str(port if port is not None else os.environ.get("LEANLEAN_LITELLM_TRACE_ID", ""))
    if not value.isascii() or not value.isdigit():
        value = "unscoped"
    return Path(LOG_PATH) / f"proxy_{value}.request-retries.jsonl"


def write_record(record: dict) -> None:
    path = retry_journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One host event loop, one bounded metadata record; no prompts or output.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def request_attempt_responses(path: Path, scope_id: str, *, discarded_only: bool = True) -> tuple[list[dict], list[dict]]:
    records = [r for r in scoped_records(path, scope_id) if not discarded_only or not r.get("delivered")]
    responses = []
    for record in records:
        raw = record.get("usage")
        response = {"model": "gemini-4-argon", "retry_attempt_id": record["litellm_call_id"],
                    "usage_incomplete": not record.get("delivered", False)}
        if isinstance(raw, dict) and type(raw.get("promptTokenCount")) is int:
            prompt = raw["promptTokenCount"]
            thoughts = raw.get("thoughtsTokenCount", 0)
            output = raw.get("candidatesTokenCount", 0) + thoughts
            response["usage"] = {
                "prompt_tokens": prompt, "completion_tokens": output,
                "total_tokens": prompt + output,
                "prompt_tokens_details": {"cached_tokens": raw.get("cachedContentTokenCount", 0)},
                "completion_tokens_details": {"reasoning_tokens": thoughts},
            }
        # Missing usage stays missing: never claim a failed request was free.
        responses.append(response)
    return responses, records


class StreamFailure(Exception):
    def __init__(self, reason: str, *, retryable: bool = True, code: int = 502):
        self.code = code
        self.reason = reason
        self.retryable = retryable
        super().__init__(reason)


class Capture:
    def __init__(self, candidates: int):
        self.file = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
        self.start = None
        self.pending = b""
        self.data = []
        self.finished = set()
        self.candidates = candidates
        self.usage = None
        self.bytes = 0
        self.activity = time.monotonic()

    def frame(self):
        if not self.data:
            return
        payload = b"\n".join(self.data)
        self.data.clear()
        if payload == b"[DONE]":
            return
        self.activity = time.monotonic()
        try:
            obj = json.loads(payload)
        except (ValueError, UnicodeError):
            raise StreamFailure("malformed_sse") from None
        if not isinstance(obj, dict):
            raise StreamFailure("malformed_sse")
        if isinstance(obj.get("usageMetadata"), dict):
            self.usage = obj["usageMetadata"]
        if "error" in obj:
            error = obj["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise StreamFailure(
                "provider_sse_error",
                retryable=code is None or str(code) in {str(x) for x in RETRY_HTTP},
                code=int(code) if str(code).isdigit() else 502,
            )
        for candidate in obj.get("candidates", []):
            finish = candidate.get("finishReason")
            if finish in {"MAX_TOKENS", "MALFORMED_FUNCTION_CALL"}:
                raise StreamFailure(str(finish).lower())
            if finish:
                self.finished.add(candidate.get("index", 0))
        # Safety refusals are terminal, not transient transport failures.
        if (obj.get("promptFeedback") or {}).get("blockReason"):
            self.finished.update(range(self.candidates))

    def feed(self, body: bytes):
        self.bytes += len(body)
        if self.bytes > 32 * 1024 * 1024:
            raise StreamFailure("response_too_large", retryable=False)
        self.file.write(body)
        self.pending += body
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                self.frame()
            elif line.startswith(b"data:"):
                self.data.append(line[5:].lstrip(b" "))

    def finish(self):
        if self.pending.startswith(b"data:"):
            self.data.append(self.pending[5:].lstrip(b" ").rstrip(b"\r"))
        self.frame()
        if not set(range(self.candidates)) <= self.finished:
            raise StreamFailure("incomplete_stream")


class ArgonRequestRetryMiddleware:
    def __init__(self, app, *, max_retries=None, idle_seconds=900, heartbeat_seconds=10,
                 backoff_seconds=1, record=write_record):
        self.app = app
        self.max_retries = int(os.environ.get(ENV, "0")) if max_retries is None else max_retries
        if not 0 <= self.max_retries <= 5:
            raise ValueError("Argon request retries must be between zero and five")
        self.idle_seconds = idle_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.backoff_seconds = backoff_seconds
        self.record = record

    async def __call__(self, scope, receive, send):
        if (not self.max_retries or scope["type"] != "http"
                or scope.get("method") != "POST" or not ROUTE.search(scope.get("path", ""))):
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body"):
                break
        body = bytes(body)
        original_request_sha256 = hashlib.sha256(body).hexdigest()
        body, normalized_summary = normalize_compaction_summary(body)
        if normalized_summary:
            scope = dict(scope, headers=[
                (key, value) for key, value in scope.get("headers", [])
                if key.lower() != b"content-length"
            ] + [(b"content-length", str(len(body)).encode())])
        try:
            candidates = int(json.loads(body).get("generationConfig", {}).get("candidateCount", 1))
            if not 1 <= candidates <= 8:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            candidates = 1  # Upstream retains authority over invalid requests.
        disconnected = asyncio.Event()
        committed = False
        request_id = uuid.uuid4().hex

        async def watch_disconnect():
            while True:
                if (await receive())["type"] == "http.disconnect":
                    disconnected.set()
                    return

        async def start():
            nonlocal committed
            if not committed:
                await send({"type": "http.response.start", "status": 200,
                            "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")]})
                committed = True

        async def heartbeat():
            await start()
            await send({"type": "http.response.body", "body": b'data: {"candidates":[]}\n\n', "more_body": True})

        watcher = asyncio.create_task(watch_disconnect())
        disconnect_wait = asyncio.create_task(disconnected.wait())
        try:
            for attempt in range(self.max_retries + 1):
                capture = Capture(candidates)
                delivered = False
                failure = None
                sent_body = False
                async def replay():
                    nonlocal sent_body
                    if not sent_body:
                        sent_body = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    await disconnected.wait()
                    return {"type": "http.disconnect"}

                async def collect(message):
                    if message["type"] == "http.response.start":
                        capture.start = message
                    elif message["type"] == "http.response.body":
                        chunk = message.get("body", b"")
                        if capture.start and capture.start["status"] == 200:
                            capture.feed(chunk)
                        else:
                            capture.file.write(chunk)

                task = asyncio.create_task(self.app(dict(scope), replay, collect))
                try:
                    while not task.done():
                        done, _ = await asyncio.wait({task, disconnect_wait}, timeout=min(self.heartbeat_seconds, self.idle_seconds), return_when=asyncio.FIRST_COMPLETED)
                        if disconnected.is_set():
                            raise asyncio.CancelledError()
                        if task in done:
                            break
                        if time.monotonic() - capture.activity >= self.idle_seconds:
                            raise StreamFailure("upstream_idle_timeout")
                        await heartbeat()
                    await task
                    status = capture.start["status"] if capture.start else 502
                    if status in RETRY_HTTP:
                        raise StreamFailure(f"http_{status}", code=status)
                    if status != 200:
                        if committed:
                            raise StreamFailure(f"http_{status}", retryable=False, code=status)
                        await send(capture.start or {"type": "http.response.start", "status": 502, "headers": []})
                    else:
                        capture.finish()
                        await start()
                    capture.file.seek(0)
                    while chunk := capture.file.read(65536):
                        await send({"type": "http.response.body", "body": chunk, "more_body": True})
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    delivered = True
                    return
                except StreamFailure as error:
                    failure = error
                except (OSError, TimeoutError) as error:
                    failure = StreamFailure(type(error).__name__)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    self.record({"event": "argon_request_attempt", "ts": time.time(),
                                 "litellm_call_id": f"{request_id}:{attempt}", "request_id": request_id,
                                 "agent_context": {"accounting_scope": CURRENT_ACCOUNTING_SCOPE.get()},
                                 "request_sha256": hashlib.sha256(body).hexdigest(),
                                 "original_request_sha256": original_request_sha256,
                                 "compaction_delimiters_removed": normalized_summary,
                                 "attempt": attempt + 1, "max_retries": self.max_retries,
                                 "delivered": delivered, "reason": failure.reason if failure else ("complete" if delivered else "cancelled_or_exception"),
                                 "usage": capture.usage, "usage_evidence": "reported" if capture.usage else "missing",
                                 "http_status": capture.start["status"] if capture.start else None})
                    capture.file.close()
                if not failure.retryable or attempt == self.max_retries:
                    await start()
                    error = {"error": {"code": failure.code, "message": f"Argon request failed after {attempt + 1} attempts ({failure.reason})"}}
                    await send({"type": "http.response.body", "body": b"data: " + json.dumps(error).encode() + b"\n\n", "more_body": False})
                    return
                delay = min(self.backoff_seconds * 2 ** attempt, 30)
                try:
                    await asyncio.wait_for(disconnected.wait(), timeout=delay)
                    return
                except TimeoutError:
                    pass
        finally:
            watcher.cancel()
            disconnect_wait.cancel()
            await asyncio.gather(watcher, disconnect_wait, return_exceptions=True)
