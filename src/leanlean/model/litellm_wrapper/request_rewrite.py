"""Rewrite a native client's request before LiteLLM reads it.

Muse Code 1.0.3 sends ``reasoning.effort: xhigh`` for ``--reasoning-effort max``,
and LiteLLM deployment parameters do not override a client-supplied ``reasoning``
object. The map comes from LEANLEAN_REASONING_EFFORT_MAP (JSON, e.g.
``{"xhigh": "max"}``); efforts not in the map, such as Muse's own low-effort side
requests, pass through unchanged.
"""
import json
import os

ENV_VAR = "LEANLEAN_REASONING_EFFORT_MAP"


class ReasoningEffortRewriteMiddleware:
    def __init__(self, app):
        self.app = app
        self.effort_map = json.loads(os.environ.get(ENV_VAR) or "{}")

    def _rewrite(self, payload) -> bool:
        if not isinstance(payload, dict):
            return False
        changed = False
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") in self.effort_map:
            reasoning["effort"] = self.effort_map[reasoning["effort"]]
            changed = True
        return changed

    async def __call__(self, scope, receive, send):
        if (not self.effort_map
                or scope["type"] != "http" or scope.get("method") != "POST"):
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return await self.app(scope, _replay([message]), send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        body = b"".join(chunks)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
        if self._rewrite(payload):
            body = json.dumps(payload).encode()
            scope = dict(scope, headers=[
                (k, str(len(body)).encode()) if k.lower() == b"content-length" else (k, v)
                for k, v in scope["headers"]
            ])
        return await self.app(scope, _replay([{"type": "http.request", "body": body, "more_body": False}], receive), send)


def _replay(messages, fallback=None):
    async def receive():
        if messages:
            return messages.pop(0)
        return await fallback() if fallback else {"type": "http.disconnect"}
    return receive
