#!/usr/bin/env python3
"""Reconstruct per-checkpoint costs of Claude Code captures from the native stream.

For captures whose gateway journal did not reconcile (so every checkpoint cost is
null), price each native assistant message's usage and accumulate it in stream
order. A checkpoint made by a tool call gets the cost through the message that
issued it; any other checkpoint gets the cost of messages fully received by its
wall timestamp. The priced total must reproduce the authoritative final cost.

Offline: no model or Lean execution. Only replayed-playback.json and
postprocessed-playback.json are rewritten (temp + rename, so hardlinked peers
keep their bytes); originals are backed up and every change is listed in the
audit manifest. Native logs, raw playback.json and source archives are untouched.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from configs.model_constants import MODEL_PRICES  # noqa: E402

TARGETS = ("replayed-playback.json", "postprocessed-playback.json")
TOLERANCE_USD = 1e-3


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def message_cost(usage: dict, model: str) -> float:
    input_rate, output_rate, read_rate, write_1h_rate = MODEL_PRICES[model]
    creation = usage.get("cache_creation") or {}
    write_5m = creation.get("ephemeral_5m_input_tokens")
    write_1h = creation.get("ephemeral_1h_input_tokens")
    if write_5m is None and write_1h is None:
        write_5m, write_1h = 0, usage.get("cache_creation_input_tokens", 0)
    return (usage.get("input_tokens", 0) * input_rate
            + usage.get("output_tokens", 0) * output_rate
            + usage.get("cache_read_input_tokens", 0) * read_rate
            + write_5m * 1.25 * input_rate
            + write_1h * write_1h_rate) / 1_000_000


def native_ledger(stream_path: Path) -> dict:
    """Messages in stream order, deduplicated by id, with final usage."""
    messages: dict[str, dict] = {}
    for line in stream_path.open():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event["message"]
        entry = messages.setdefault(message["id"], {"tools": set(), "usage": {}})
        usage = message.get("usage") or {}
        if usage.get("output_tokens", 0) >= entry["usage"].get("output_tokens", 0):
            entry["usage"] = usage
        entry["model"] = message["model"]
        entry["received"] = timestamp(event["timestamp"])
        entry["tools"].update(block["id"] for block in message.get("content") or []
                              if block.get("type") == "tool_use")
    entries = list(messages.values())
    cumulative, by_tool, total = [], {}, 0.0
    for entry in entries:
        total += message_cost(entry["usage"], entry["model"])
        cumulative.append(total)
        for tool in entry["tools"]:
            by_tool[tool] = total
    # Receipt order can differ slightly from stream order; cost by time is the
    # running maximum over messages received by then.
    order = sorted(range(len(entries)), key=lambda i: entries[i]["received"])
    received = [entries[i]["received"] for i in order]
    running, best = [], 0.0
    for i in order:
        best = max(best, cumulative[i])
        running.append(best)
    return dict(total=total, by_tool=by_tool, received=received, running=running,
                message_count=len(messages))


def observation_cost(observation: dict, ledger: dict, final_cost: float) -> tuple[float, str]:
    if observation.get("kind") == "final":
        return final_cost, "authoritative_final"
    tool = observation.get("tool_use_id")
    if tool in ledger["by_tool"]:
        return ledger["by_tool"][tool], "native_stream_tool_use"
    index = bisect.bisect_right(ledger["received"], timestamp(observation["wall_timestamp"])) - 1
    return (ledger["running"][index] if index >= 0 else 0.0), "native_stream_receipt_timestamp"


def reconstruct(playback: dict, ledger: dict, provenance: dict) -> dict:
    final_cost = float(playback["authoritative_final_cost_usd"])
    if abs(ledger["total"] - final_cost) > TOLERANCE_USD:
        raise ValueError(f"Native stream total {ledger['total']:.6f} does not reconcile with {final_cost:.6f}")
    by_edit: dict[int, tuple[float, str]] = {}
    previous = 0.0
    for observation in sorted(playback["checkpoint_observations"], key=lambda o: o["sequence"]):
        cost, basis = observation_cost(observation, ledger, final_cost)
        if cost < previous - 1e-9:
            raise ValueError(f"Nonmonotone reconstructed cost at observation {observation['sequence']}")
        previous = cost
        observation.update(cumulative_cost_usd=cost, cost_boundary_exact=basis != "native_stream_receipt_timestamp",
                           cost_estimated=False, cost_join_basis=basis)
        if observation.get("edit_index") is not None and observation.get("kind") != "final":
            if observation["edit_index"] in by_edit:
                raise ValueError(f"Duplicate edit observation: {observation['edit_index']}")
            by_edit[observation["edit_index"]] = cost, basis
    previous = 0.0
    for point in playback["points"]:
        if point.get("kind") == "baseline":
            cost, basis = 0.0, "baseline"
        elif point.get("kind") == "final":
            cost, basis = final_cost, "authoritative_final"
        else:
            cost, basis = by_edit[point["edit_index"]]
        point.update(cost_usd=cost, marginal_cost_since_previous_edit_usd=cost - previous,
                     cost_boundary_exact=basis != "native_stream_receipt_timestamp", cost_join_basis=basis)
        previous = cost
    playback.update(every_edit_checkpoint_costed=True, cost_timeline_complete=True,
                    intermediate_costs_estimated=False,
                    replay_cost_status="native_stream_reconstructed",
                    cost_reconstruction=provenance)
    return playback


def atomic_write(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", nargs="+", type=Path, help="playback/capture_* directories")
    parser.add_argument("--audit", type=Path, required=True, help="new audit directory (backups + manifest)")
    parser.add_argument("--apply", action="store_true", help="rewrite the playbacks (default: report only)")
    args = parser.parse_args()
    audit = args.audit.resolve()
    if args.apply and audit.exists():
        raise FileExistsError(audit)
    changes, plans = [], []
    for capture in args.captures:
        capture = capture.resolve()
        stream = capture / "native.stdout.jsonl"
        ledger = native_ledger(stream)
        provenance = dict(script="scripts/analysis/reconstruct_native_claude_checkpoint_costs.py",
                          native_stream=str(stream.relative_to(ROOT)), native_stream_sha256=sha(stream),
                          native_message_count=ledger["message_count"],
                          native_priced_total_usd=ledger["total"],
                          reason="gateway journal did not reconcile; checkpoint costs were null",
                          reconstructed_at=datetime.now(timezone.utc).isoformat())
        for name in TARGETS:
            path = capture / name
            playback = reconstruct(json.loads(path.read_text()), ledger, provenance)
            edits = [p for p in playback["points"] if p.get("kind") not in ("baseline", "final")]
            bases = {}
            for point in edits:
                bases[point["cost_join_basis"]] = bases.get(point["cost_join_basis"], 0) + 1
            print(f"{path.relative_to(ROOT)}: total ${ledger['total']:.4f} vs final "
                  f"${playback['authoritative_final_cost_usd']:.4f}; {len(edits)} edit points {bases}")
            plans.append((path, json.dumps(playback, indent=2, ensure_ascii=False).encode() + b"\n"))
    if not args.apply:
        print("Report only; pass --apply to rewrite.")
        return
    for path, data in plans:
        relative = path.relative_to(ROOT)
        backup = audit / "backup" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        before = sha(path)
        atomic_write(path, data)
        changes.append(dict(path=str(relative), before_sha256=before, after_sha256=sha(path),
                            backup=str(backup.relative_to(ROOT))))
    manifest = dict(kind="native_claude_checkpoint_cost_reconstruction", schema_version=1,
                    created_at=datetime.now(timezone.utc).isoformat(), model_calls=False,
                    tolerance_usd=TOLERANCE_USD, changes=changes)
    (audit / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(audit / "manifest.json")


if __name__ == "__main__":
    main()
