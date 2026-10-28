#!/usr/bin/env python3
"""Compare AGY directly and through the actual LiteLLM proxy using local fixtures.

No provider credentials or paid calls. Checks semantic JSON/SSE parity, native
usage, and stock core instructions. Does not establish the provider's maximum
context capacity or compaction threshold. Run from the repository root.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import httpx
from leanlean.antigravity_config import ANTIGRAVITY_REQUEST_TOOLS, antigravity_profile_files
from leanlean.antigravity_usage import AntigravityUsageJournal
from leanlean.generators.cli_agent import resolve_antigravity_standalone_tool
from leanlean.model.litellm_wrapper.litellm_server import LitellmServer

MODEL = "gemini-4-argon"
FIXTURE = {
    "candidates": [{"content": {"role": "model", "parts": [
        {"text": "Fixture reasoning.", "thought": True, "thoughtSignature": "cGFyaXR5"},
        {"text": "READY"},
    ]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 200,
                      "candidatesTokenCount": 20, "thoughtsTokenCount": 10,
                      "totalTokenCount": 1030},
    "modelVersion": MODEL, "responseId": "parity-test",
}


def native_run(root, label, base, key, restricted):
    home, cwd = root / label / "home", root / label / "workspace"
    home.mkdir(parents=True)
    cwd.mkdir()
    (cwd / ".git").mkdir()
    for name, content in antigravity_profile_files("1.4.0").items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    env = {k: v for k, v in os.environ.items() if not k.startswith((
        "AGY_", "ANTIGRAVITY_", "GEMINI_", "GOOGLE_", "CLAUDE", "ANTHROPIC_",
        "OPENAI_", "CODEX_", "MCP_"))}
    env.update(HOME=str(home), GEMINI_API_KEY=key, GOOGLE_GEMINI_BASE_URL=base,
               AGY_CLI_DISABLE_AUTO_UPDATE="true", AGY_CLI_OSC7501="off",
               AGY_LLM_GATEWAY_URL=base, AGY_LLM_GATEWAY_API_KEY=key,
               AGY_LLM_GATEWAY_WIRE_PROTOCOL="genai", AGY_LLM_GATEWAY_MODELS=MODEL)
    command = [str(resolve_antigravity_standalone_tool("1.4.0"))]
    if restricted:
        command += ["--agent", "leanlean-local"]
    command += ["--dangerously-skip-permissions", "--disable-slash-commands",
                "--output-format", "stream-json", "--model", MODEL, "--effort", "high",
                "--print-timeout", "30s", "-p", "Return the word READY and do nothing else."]
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=40)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    (root / label / "stdout.jsonl").write_text(stdout)
    (root / label / "stderr.log").write_text(stderr)
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    terminal = [e["result"] for e in events if e.get("event") == "result"][-1]
    assert process.returncode == 0 and terminal["status"] == "SUCCESS", terminal
    journal = AntigravityUsageJournal()
    for event in events:
        journal.ingest(event)
    journal.validate()
    return terminal, journal


def normalize(body, root, label, conversation):
    text = json.dumps(body, sort_keys=True).replace(str(root / label), "<RUN>")
    text = text.replace(conversation, "<CONVERSATION>")
    return re.sub(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)',
                  "<TIME>", text)


def sections(body):
    prompt = body["systemInstruction"]["parts"][0]["text"]
    return dict(re.findall(r"<([a-z_]+)>\n(.*?)</\1>", prompt, re.S))


def audit(root):
    records = []
    wire = ("data: " + json.dumps(FIXTURE) + "\r\n\r\n").encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records.append({"path": self.path, "body": body})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{upstream.server_port}/v1beta"
    proxy = LitellmServer(
        model_name=f"gemini/{MODEL}", api_base=base,
        model_kwargs={"api_key": "dummy", "reasoning_effort": "high", "drop_params": True},
        host="127.0.0.1", log_dir=str(root / "proxy"), auto_cost_updates=False)
    try:
        proxy.serve()
        args = proxy.get_openai_args()
        proxybase = args["base_url"].removesuffix("/v1")
        capture = ROOT / "experiments/payload_audit/captures/google-gemini-4-argon-high-antigravity-1.4.0_2026-10-16.json"
        request = json.loads(capture.read_text())["request"]
        # Exercise conversation history, tool results, signatures, and a large
        # body as well as the native initial request; no tokenization inference.
        extended = copy.deepcopy(request)
        extended["contents"] += [
            {"role": "model", "parts": [{"functionCall": {"name": "view_file", "args": {
                "AbsolutePath": "/tmp/example"}}, "thoughtSignature": "cGFyaXR5"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "view_file", "response": {
                "output": "context fixture " * 20000}}}]},
        ]
        for body in (request, extended):
            response = httpx.post(proxybase + f"/v1beta/models/{MODEL}-high:streamGenerateContent?alt=sse",
                                  json=body, headers={"x-goog-api-key": args["api_key"]}, timeout=30)
            assert response.status_code == 200, response.text
            actual = copy.deepcopy(records[-1]["body"])
            assert actual.pop("model") == MODEL
            assert actual == body, "LiteLLM changed the native request"
            frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            assert frames == [FIXTURE], "LiteLLM changed native SSE data"
            assert records[-1]["path"] == f"/v1beta/models/{MODEL}:streamGenerateContent?alt=sse"
        runs, mains = {}, {}
        for label, url, key, restricted in [
            ("direct", base.removesuffix("/v1beta"), "dummy", True),
            ("proxied", proxybase, args["api_key"], True),
            ("stock", base.removesuffix("/v1beta"), "dummy", False),
        ]:
            start = len(records)
            terminal, journal = native_run(root, label, url, key, restricted)
            batch = records[start:]
            mains[label] = next(x["body"] for x in batch if x["body"].get("tools"))
            runs[label] = {"terminal": terminal, "upstream_requests": len(batch),
                           "native_usage_steps": len(journal.steps), "cost_usd": journal.cost,
                           "normalized_usage": journal.responses[0]["usage"]}
        forwarded = copy.deepcopy(mains["proxied"])
        forwarded.pop("model")
        assert normalize(mains["direct"], root, "direct", runs["direct"]["terminal"]["conversation_id"]) == normalize(
            forwarded, root, "proxied", runs["proxied"]["terminal"]["conversation_id"])
        for field in ("usage", "response", "status", "num_turns"):
            assert runs["direct"]["terminal"][field] == runs["proxied"]["terminal"][field]
        assert runs["direct"]["cost_usd"] == runs["proxied"]["cost_usd"]
        native, stock = sections(mains["direct"]), sections(mains["stock"])
        for name in ("identity", "artifacts", "guidelines", "communication_style"):
            assert native[name] == stock[name], f"Stock {name} changed"
        transcript = "\n".join(l for l in stock["conversation_transcript"].split("\n")
                               if not l.startswith("- Search subagents"))
        assert native["conversation_transcript"] == transcript.replace("# File Format", "File Format")
        assert not ({"subagents", "skills", "slash_commands"} & native.keys())
        names = {f["name"] for group in mains["direct"]["tools"] for f in group["functionDeclarations"]}
        assert names == ANTIGRAVITY_REQUEST_TOOLS
        assert all(b["generationConfig"] == mains["stock"]["generationConfig"] for b in mains.values())
        report = {
            "antigravity_version": "1.4.0", "litellm_version": importlib.metadata.version("litellm"),
            "network_destination": "loopback_only", "provider_calls": 0,
            "request_fields_preserved": True, "response_frames_preserved": True,
            "request_envelope_change": "Adds model=gemini-4-argon and maps the effort alias to the base model route.",
            "direct_vs_proxied_native_request_equal_except_runtime_paths_ids_time": True,
            "direct_vs_proxied_native_usage_equal": True,
            "stock_core_prompt_sections_preserved": True,
            "generation_config": mains["direct"]["generationConfig"], "tools": sorted(names),
            "runs": runs,
            "limitations": [
                "Restricted profile differs from stock in tool list, capability instructions, section ordering, and custom-agent wrapping/heading formatting.",
                "Same native gateway model catalog on both sides; logged-in subscription backend not tested.",
                "Context capacity and long-conversation compaction threshold not established by this fixture.",
                "Native totals exclude cached input; normalized benchmark totals include it.",
                "Native usage does not include all auxiliary requests; cost is a benchmark estimate, not an invoice.",
                "Duration includes proxy overhead; real provider output and cache hits may vary.",
            ],
        }
        (root / "requests.json").write_text(json.dumps(records, indent=2))
        return report
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="leanlean-argon-parity-") as directory:
        report = audit(Path(directory))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: native requests, response frames, usage, and core prompt parity; report: {args.output}")


if __name__ == "__main__":
    main()
