#!/usr/bin/env python3
"""Capture a pinned Muse request against a local rejecting fake API (zero spend)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.generator_constants import ALL_GENERATOR_CONFIGS
from leanlean.generators.muse_code import MUSE_NON_CORE_TOOLS, muse_native_settings, resolve_muse_standalone_tool


def capture(model: str, effort: str, generator: str) -> dict:
    captured = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args): pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [{"id": model, "object": "model", "owned_by": "meta"}]}).encode())
        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer offline-audit":
                raise RuntimeError("Muse did not honor the pinned bearer endpoint")
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Offline capture complete; no upstream model was called"}}')
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="muse-payload-audit-") as directory:
            home = Path(directory)
            settings = home / "config/muse/settings.json"
            settings.parent.mkdir(parents=True)
            base_url = f"http://127.0.0.1:{server.server_port}/v1"
            excluded = MUSE_NON_CORE_TOOLS if generator == "muse_code_core" else ()
            settings.write_text(json.dumps(muse_native_settings(base_url, extra_excluded_tools=excluded)))
            workspace = home / "workspace"
            workspace.mkdir()
            config = ALL_GENERATOR_CONFIGS[generator]
            command = config["launch_command"].format(
                model=model, reasoning_effort=effort, api_key="offline-audit",
                prompt=shlex.quote("Refactor the proofs in this codebase. Output <submit> when finished."),
            ).replace("/tmp/leanlean-muse", str(home)).replace("/testbed", str(workspace))
            parts = shlex.split(command)
            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
            while "=" in parts[0]:
                key, value = parts.pop(0).split("=", 1)
                env[key] = value
            parts[0] = str(resolve_muse_standalone_tool(config["host_tool_version"]))
            try:
                result = subprocess.run(parts, env=env, cwd=workspace, capture_output=True, text=True, timeout=45)
                stderr = result.stderr
            except subprocess.TimeoutExpired as exc:
                # Native defaults retry the rejected request with backoff; the first capture suffices.
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            if not captured:
                raise RuntimeError(f"Muse request was not captured: {stderr[-2000:]}")
            print(f"Muse sent {len(captured)} request(s) against the rejecting API; pinning the first")
            return captured[0]
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    spec = yaml.safe_load(args.manifest.read_text())
    if spec.get("evaluation", {}).get("mode") != "offline_payload_audit":
        raise ValueError("This tool only supports an offline payload audit")
    if spec.get("generator") not in {"muse_code_native", "muse_code_core"}:
        raise ValueError("This tool audits the muse_code_native and muse_code_core harnesses")
    body = capture(spec["model"], spec["reasoning_effort"], spec["generator"])
    out = Path(spec["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2) + "\n")
    print(f"Captured one native request without upstream calls: {out}")
    print("Tools:", [t.get("name") for ns in body["tools"] for t in ns.get("tools", [ns])])


if __name__ == "__main__":
    main()
