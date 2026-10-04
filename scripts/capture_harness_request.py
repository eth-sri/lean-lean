#!/usr/bin/env python3
"""Capture a model config's native harness request on loopback and snapshot it.

The pinned Codex or Claude Code binary is pointed at a local server that
records its request and rejects it, so no model is called and nothing leaves
the machine. The capture and the native-instruction snapshot are written under
experiments/payload_audit/; the printed hashes go into the model config
(generation_parameters.request_sha256 and system_prompt.sha256).

Usage:
    uv run python scripts/capture_harness_request.py configs/models/openai/gpt-6.1-sol-xhigh.yaml \
        --request-model gpt-6.1-sol
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.generators.cli_agent import (  # noqa: E402
    resolve_claude_standalone_tool,
    resolve_codex_standalone_tools,
)
from snapshot_payload_audit import (  # noqa: E402
    _json_sha256,
    _native_instruction_fragments,
    _payload_bloat_audit,
)

PROMPT = "Return the word READY and do nothing else."
# The benchmark's Codex controls, as in the captures behind the existing configs.
CODEX_CONTROLS = [
    "agents.enabled=false", "features.multi_agent=false", "features.apps=false",
    "features.remote_plugin=false", "features.plugins=false",
    "features.recommended_plugins=false", "features.skill_search=false",
    "features.skill_mcp_dependency_install=false", "features.browser_use=false",
    "features.browser_use_external=false", "features.browser_use_full_cdp_access=false",
    "features.computer_use=false", "features.image_generation=false", "features.goals=false",
    "features.hooks=false", "features.plugin_sharing=false", "features.personality=false",
    "features.tool_suggest=false", "features.workspace_dependencies=false",
    "features.enable_mcp_apps=false", "features.tool_call_mcp_elicitation=false",
    "web_search=disabled",
]
CLAUDE_TOOLS = "Bash,Edit,Read,Write"
# Variables an enclosing agent session or provider setup may export; they change
# the harness's request (entrypoint, environment block, endpoints).
INHERITED_PREFIXES = ("CLAUDE", "ANTHROPIC_", "AI_AGENT", "MCP_", "CODEX_", "OPENAI_")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8", errors="surrogateescape"
    )


def _clean_environment(**overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(INHERITED_PREFIXES)}
    return {**env, **overrides}


def _loopback(run) -> list[dict[str, Any]]:
    """Run ``run(base_url)`` against a server that records and rejects every request."""

    records: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _handle(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length", "0")))
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                body = raw.decode(errors="replace")
            records.append({"method": self.command, "path": self.path, "body": body})
            data = json.dumps({"type": "error", "error": {
                "type": "capture_complete", "message": "loopback capture complete"}}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_POST = do_GET = _handle

        def log_message(self, *args: Any) -> None:
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            run(f"http://127.0.0.1:{server.server_address[1]}")
        finally:
            server.shutdown()
    return records


def capture_codex(version: str, model: str, effort: str, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    binary = resolve_codex_standalone_tools(version)[0]
    catalog = config.get("codex_model_catalog")

    def run(base_url: str) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as cwd:
            env = _clean_environment(CODEX_HOME=home, LITELLM_API_KEY="capture-only")
            controls = [part for control in CODEX_CONTROLS for part in ("-c", control)]
            if catalog:
                controls += ["-c", f"model_catalog_json={(ROOT / catalog['path']).resolve()}"]
            # Closed stdin: Codex 0.160 otherwise waits on "Reading additional input from stdin".
            subprocess.run([
                str(binary), "exec", "--json", "--ephemeral", "--ignore-user-config",
                "--sandbox", "danger-full-access", "--skip-git-repo-check",
                "-c", "model_provider=litellm", "-c", "model_providers.litellm.name=litellm",
                "-c", f"model_providers.litellm.base_url={base_url}/v1",
                "-c", "model_providers.litellm.env_key=LITELLM_API_KEY",
                "-c", "model_providers.litellm.wire_api=responses",
                "-c", f"model_reasoning_effort={effort}", *controls,
                "--model", model, PROMPT,
            ], cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=60)

    def run_until_captured(base_url: str) -> None:
        # Codex may retry the rejected request with backoff; the first one suffices.
        try:
            run(base_url)
        except subprocess.TimeoutExpired:
            pass

    records = _loopback(run_until_captured)
    mains = [r["body"] for r in records if isinstance(r["body"], dict) and "input" in r["body"]]
    if not mains:
        raise RuntimeError(f"Codex sent no Responses request: {[r['path'] for r in records]}")
    return mains[0], len(records)


def capture_claude(version: str, model: str, effort: str, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    binary = resolve_claude_standalone_tool(version)
    tools = ",".join(config.get("native_tools") or []) or CLAUDE_TOOLS

    def run(base_url: str) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as cwd:
            env = _clean_environment(HOME=home, IS_SANDBOX="1", ANTHROPIC_BASE_URL=base_url,
                                     CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-capture-only")
            subprocess.run([
                str(binary), "--dangerously-skip-permissions", "--safe-mode",
                "--disable-slash-commands", "--no-chrome", "--no-session-persistence",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--tools", tools, "--output-format", "stream-json", "--verbose",
                "--model", model, "--effort", effort, "-p", PROMPT,
            ], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)

    records = _loopback(run)
    # The main request is the one carrying the run's Bash tool.
    mains = [r["body"] for r in records
             if isinstance(r["body"], dict)
             and any(t.get("name") == "Bash" for t in r["body"].get("tools", []))]
    if not mains:
        raise RuntimeError(f"Claude Code sent no main request: {[r['path'] for r in records]}")
    return mains[-1], len(records)


CAPTURERS = {"codex_sub": ("codex", capture_codex), "claude_code_sub": ("claude-code", capture_claude)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="model config under configs/models/")
    parser.add_argument("--request-model", required=True, help="model ID the harness sends")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    harness, version, effort = config["harness"], config["harness_version"], config["reasoning_effort"]
    if harness not in CAPTURERS:
        raise SystemExit(f"no loopback capture for harness {harness!r}")
    tool, capture = CAPTURERS[harness]
    request, request_count = capture(version, args.request_model, effort, config)
    request_sha = hashlib.sha256(_canonical(request)).hexdigest()

    relative_config = config_path.relative_to(ROOT).as_posix()
    slug = f"{config_path.parent.name}-{config_path.stem}"
    capture_rel = f"experiments/payload_audit/captures/{slug}-{tool}-{version}_{args.date}.json"
    prompt_rel = f"experiments/payload_audit/prompts/{slug}.json"
    (ROOT / capture_rel).write_text(json.dumps({
        "kind": "leanlean_loopback_request_capture", "schema_version": 1,
        "captured_at": args.date, "harness": harness, "harness_version": version,
        "model": args.request_model, "reasoning_effort": effort,
        "network_destination": "loopback_only", "model_request_forwarded": False,
        "captured_request_count": request_count, "request_sha256": request_sha, "request": request,
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    # The snapshot splits the request as the proxy traces do: the turn list,
    # and everything else as envelope extras.
    turns_key = "input" if "input" in request else "messages"
    envelope = {"extras": {k: v for k, v in request.items() if k != turns_key}, "sha256": request_sha}
    fragments = _native_instruction_fragments(envelope, request[turns_key])
    if not fragments:
        raise RuntimeError("no native system/developer instructions in the captured request")
    instructions_sha = _json_sha256(fragments)
    (ROOT / prompt_rel).write_text(json.dumps({
        "kind": "leanlean_native_instruction_snapshot", "schema_version": 1,
        "model_config": relative_config, "harness_version": version,
        "request_sha256": request_sha, "source_capture": capture_rel,
        "native_instructions_sha256": instructions_sha,
        "payload_audit": _payload_bloat_audit(envelope, request[turns_key]),
        "fragments": fragments,
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"capture:  {capture_rel}")
    print(f"snapshot: {prompt_rel}")
    print(f"generation_parameters.request_sha256: {request_sha}")
    print(f"system_prompt.sha256: {instructions_sha}")
    print(f"request model: {request.get('model')}; tools: {_payload_bloat_audit(envelope, request[turns_key])['tool_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
