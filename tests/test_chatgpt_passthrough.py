"""The ChatGPT subscription proxy must forward Codex's request as Codex sent it."""
import base64
import glob
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from leanlean.model.litellm_wrapper.litellm_server import LitellmServer

ROOT = Path(__file__).resolve().parents[1]
CODEX_CAPTURES = sorted(
    glob.glob(str(ROOT / "experiments/payload_audit/captures/openai-gpt-6.1-sol-*-codex-0.160.0_*.json"))
)


def _upstream(captured):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            captured.append(({k.lower(): v for k, v in self.headers.items()}, json.loads(body)))
            done = {"type": "response.completed", "response": {
                "id": "resp_test", "object": "response", "created_at": 0, "status": "completed",
                "model": "gpt-6.1-sol", "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}
            sse = f"event: response.completed\ndata: {json.dumps(done)}\n\n".encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.skipif(not CODEX_CAPTURES, reason="no Codex 0.160.0 capture")
def test_codex_request_reaches_upstream_unchanged(tmp_path, monkeypatch):
    captured = []
    upstream = _upstream(captured)
    base = f"http://127.0.0.1:{upstream.server_port}"
    # Both the model's api_base and LiteLLM's ChatGPT default point at the local
    # server, so nothing can reach chatgpt.com.
    monkeypatch.setenv("CHATGPT_API_BASE", base)
    claims = {"exp": int(time.time()) + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}
    token = "x." + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=") + ".y"
    monkeypatch.setenv("OPENAI_SUBSCRIPTION_KEY", token)
    monkeypatch.delenv("OPENAI_SUBSCRIPTION_ACCOUNT_ID", raising=False)

    codex = json.loads(Path(CODEX_CAPTURES[-1]).read_text())["request"]
    second = json.loads(json.dumps(codex))
    second["input"].append({"type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": "next turn"}]})

    proxy = LitellmServer(host="127.0.0.1", log_dir=str(tmp_path), model_name="openai/gpt-6.1-sol",
                          subscription_transport="chatgpt", model_kwargs={"api_base": base})
    proxy.serve()
    try:
        for body in (codex, second):
            request = urllib.request.Request(
                proxy.proxy_url + "/responses", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {proxy.get_api_key()}", "content-type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=60).read()
    finally:
        proxy.stop()
        upstream.shutdown()

    assert len(captured) == 2
    for sent, (headers, received) in zip((codex, second), captured):
        assert received == sent  # no allowlist filtering, no injected instructions
        assert headers["session_id"] == sent["prompt_cache_key"]  # one session across turns
