"""Local-only request attribution; never changes the upstream model payload."""

from __future__ import annotations

import re
from contextvars import ContextVar
from urllib.parse import urlsplit, urlunsplit

SCOPE_HEADER = b"x-leanlean-accounting-scope"
SCOPE_PREFIX = "/lc-accounting/"
CURRENT_ACCOUNTING_SCOPE: ContextVar[str | None] = ContextVar("leanlean_accounting_scope", default=None)


def scoped_url(url: str, scope_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", scope_id):
        raise ValueError("invalid accounting scope")
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=SCOPE_PREFIX + scope_id + parts.path))


class AccountingScopeMiddleware:
    """Strip the local routing prefix before LiteLLM authenticates/routes it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        scope_id = None
        if scope["type"] == "http":
            scope = dict(scope)
            headers = [(k, v) for k, v in scope.get("headers", [])
                       if k.lower() != SCOPE_HEADER]
            path = scope.get("path", "")
            if path.startswith(SCOPE_PREFIX):
                scope_id, separator, remainder = path[len(SCOPE_PREFIX):].partition("/")
                if not separator or not re.fullmatch(r"[0-9a-f]{32}", scope_id):
                    await send({"type": "http.response.start", "status": 400, "headers": []})
                    await send({"type": "http.response.body", "body": b"Invalid accounting scope"})
                    return
                scope["path"] = "/" + remainder
                raw_path = scope.get("raw_path", path.encode("utf-8"))
                scope["raw_path"] = raw_path[len(SCOPE_PREFIX) + len(scope_id):]
                headers.append((SCOPE_HEADER, scope_id.encode("ascii")))
            scope["headers"] = headers
        token = CURRENT_ACCOUNTING_SCOPE.set(scope_id)
        try:
            await self.app(scope, receive, send)
        finally:
            CURRENT_ACCOUNTING_SCOPE.reset(token)


def record_scope(record: dict) -> str | None:
    return (record.get("agent_context") or {}).get("accounting_scope")


def record_identity(record: dict) -> str | None:
    """Prefer the gateway attempt ID: retries can share an upstream response ID."""
    call_id = record.get("litellm_call_id")
    if call_id:
        return "call:" + str(call_id)
    response_id = (record.get("response") or {}).get("id")
    return "response:" + str(response_id) if response_id else None


def scoped_records(path, scope_id: str | None):
    """Read once, stream large records, and deduplicate repeated callbacks."""
    import json

    seen = set()
    try:
        stream = path.open(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return
    with stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if scope_id is not None and record_scope(record) != scope_id:
                continue
            identity = record_identity(record)
            # Failure and success callbacks are distinct evidence.
            key = (record.get("event"), identity)
            if identity and key in seen:
                continue
            if identity:
                seen.add(key)
            yield record
