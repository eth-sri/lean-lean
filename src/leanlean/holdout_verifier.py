"""Signature and axiom checks for compression after main-theorem holdout."""

from __future__ import annotations

import json
import re


def install_holdout_verifier(env, *, names: set[str], build_target: str, build_jobs: int) -> None:
    # Use the benchmark's canonical signature comparison, without restoring
    # the withheld result or exposing any original proof to the agent.
    from leanlean.benchmarks.leanlean import (
        _collect_lean_signatures,
        _render_dump_sigs_script,
        _task_metadata_execute,
        _write_task_metadata_file,
    )

    module = build_target.lstrip("+")
    if not names or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
        raise ValueError("holdout verification needs protected names and a module build target")
    _task_metadata_execute(
        env, f"cd /testbed && LEAN_NUM_THREADS={build_jobs} lake build {build_target}",
        "build holdout signature baseline",
    )
    signatures = _collect_lean_signatures(env, modules=[module], protected_names=names)
    if not names <= signatures.keys():
        raise RuntimeError("holdout signature baseline is missing protected results")
    config = {
        "module": module,
        "build_target": build_target,
        "build_jobs": build_jobs,
        "signatures": {name: list(signatures[name][:3]) for name in sorted(names)},
    }
    _write_task_metadata_file(
        env, "/usr/local/bin/lean_verify",
        render_holdout_verifier(config, _render_dump_sigs_script(names)),
    )
    _task_metadata_execute(env, "chmod 0755 /usr/local/bin/lean_verify", "install holdout lean_verify")
    _task_metadata_execute(
        env, "cd /testbed && LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify",
        "verify held-out compression baseline",
    )


def render_holdout_verifier(config: dict, signature_source: str) -> str:
    names = "#[" + ", ".join(json.dumps(name) for name in config["signatures"]) + "]"
    audit_source = r'''
import Lean
open Lean
unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let imports := (args.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let ctx : Core.Context := { fileName := "<holdout-axioms>", fileMap := default }
  let state : Core.State := { env := env }
  let targets : Array String := __NAMES__
  let action : CoreM Unit := do
    for rendered in targets do
      let target := rendered.toName
      let axioms ← collectAxioms target
      IO.println s!"AXIOMS\t{rendered}\t{String.intercalate " " (axioms.toList.map (·.toString))}"
  let _ ← action.toIO ctx state
'''.strip().replace("__NAMES__", names)
    return f'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

CONFIG = json.loads({json.dumps(config, sort_keys=True)!r})
SIGNATURE_SOURCE = {signature_source!r}
AXIOM_SOURCE = {audit_source!r}

def run(command):
    result = subprocess.run(command, cwd="/testbed", capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout

started = time.time_ns()
status = 1
try:
    if len(sys.argv) != 1:
        raise SystemExit("usage: lean_verify")
    os.environ["LEAN_NUM_THREADS"] = str(CONFIG["build_jobs"])
    run(["lake", "build", CONFIG["build_target"]])
    with tempfile.TemporaryDirectory(prefix="holdout-verify-") as directory:
        signature_path = Path(directory) / "Signatures.lean"
        signature_path.write_text(SIGNATURE_SOURCE)
        output = run(["lake", "env", "lean", "--run", str(signature_path), CONFIG["module"]])
        actual = {{}}
        for line in output.splitlines():
            parts = line.split("\\t")
            if len(parts) == 5:
                kind, name, type_hash, value_hash, role = parts
                actual[name] = [kind, type_hash, value_hash]
        changed = [name for name, expected in CONFIG["signatures"].items() if actual.get(name) != expected]
        if changed:
            raise SystemExit("missing or changed protected signatures: " + ", ".join(changed))
        audit_path = Path(directory) / "Axioms.lean"
        audit_path.write_text(AXIOM_SOURCE)
        output = run(["lake", "env", "lean", "--run", str(audit_path), CONFIG["module"]])
        checked = set()
        for line in output.splitlines():
            parts = line.split("\\t")
            if len(parts) == 3 and parts[0] == "AXIOMS":
                forbidden = set(parts[2].split()) - {{"propext", "Classical.choice", "Quot.sound"}}
                if forbidden:
                    raise SystemExit("forbidden axioms: " + ", ".join(sorted(forbidden)))
                checked.add(parts[1])
        if checked != set(CONFIG["signatures"]):
            raise SystemExit("incomplete protected-result axiom audit")
    print("protected signatures preserved; standard axioms only")
    status = 0
finally:
    if os.environ.get("LEANLEAN_BENCHMARK_INTERNAL") != "1":
        finished = time.time_ns()
        with Path("/testbed/.git/leanlean-agent-tool-usage.jsonl").open("a") as stream:
            stream.write(json.dumps({{"schema_version": 1, "tool": "lean_verify", "event": "finished", "time_ns": finished, "duration_ms": (finished-started)//1000000, "exit_code": status}}) + "\\n")
sys.exit(status)
'''
