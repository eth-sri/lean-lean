"""Export every agent run's trace, normalised for the trace viewer.

For every (repository, model) run the paper scored:
  site/data/traces/<repo>/<model>.json.gz   the run as one list of steps (gzip)
  site/data/traces/index.json               which runs have a trace, with step counts

Each harness keeps its own native trace format: Claude Code's stream-json,
Codex's rollout, the Antigravity CLI's step stream, Muse's event log and Mistral
Vibe's public history.  One adapter per format turns it into the same steps:

  say      text the model wrote to the user
  think    the model's reasoning, where the harness records its text
  user     a message to the model (the task prompt, a resume nudge)
  compact  a context-compaction summary the harness inserted
  tool     one tool call: native tool name, category, the paper's action, input,
           output, error flag

The paper's replayed checkpoint builds (replay-checkpoint-builds.csv) give the
verified trajectory: the compression of the last checkpoint that built.

Tool calls keep their native names (`run_command`, `exec_command`, `Bash` ...);
the category only picks how the viewer renders the input.  What a call did is
the paper's action class (scripts/analysis/plot_action_classes.py): Build, Verify,
Measure, Read, Search, Edit, Git, Lake, Sleep or Other.  Reasoning is shown
where its text was recorded (Mistral Vibe with reasoning enabled); Claude Code
records only signatures, Codex encrypts it, and the other harnesses report token
counts at most.

Every step carries its time since the run started.  The playback checkpoints
(one per source change) give the Lean-token trajectory, and each change is
attached to the step that made it, with a clipped diff.  Long inputs and outputs
are clipped to their head and tail; nothing else is rewritten.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shlex
import sys
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import unified_diff
from pathlib import Path

from anonymize_paths import anonymize_text
SITE = Path(__file__).resolve().parents[1] / "site"
OUT = SITE / "data/traces"
# Importing the adapters does not require the old frozen paper collection.
SHORT = {}
LEANLEAN_CODE = Path(os.environ.get("LEANLEAN_CODE", Path.home() / "lean-lean"))

# Clipping budgets: (lines, characters), kept as head + tail.
CLIP_OUTPUT = (40, 3000)
CLIP_INPUT = (120, 8000)
CLIP_TEXT = (200, 12000)
CLIP_DIFF = (80, 5000)
OMIT = "… {} more lines …"


# ── helpers ───────────────────────────────────────────────────────────────

def clip(text, budget) -> str:
    """Head and tail of `text`, within (lines, chars); a marker line replaces the middle."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").rstrip("\n")
    max_lines, max_chars = budget
    lines = text.split("\n")
    if len(lines) <= max_lines and len(text) <= max_chars:
        return text
    head_lines, tail_lines = max_lines * 2 // 3, max_lines - max_lines * 2 // 3
    head, tail, used = [], [], 0
    for line in lines[:head_lines]:
        line = line if len(line) <= max_chars // 2 else line[:max_chars // 2] + "…"
        if used + len(line) > max_chars * 2 // 3:
            break
        head.append(line)
        used += len(line) + 1
    for line in reversed(lines[len(head):][-tail_lines:]):
        line = line if len(line) <= max_chars // 4 else line[:max_chars // 4] + "…"
        if used + len(line) > max_chars:
            break
        tail.insert(0, line)
        used += len(line) + 1
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join(head + ([OMIT.format(omitted)] if omitted else []) + tail)


def strip_testbed(path) -> str:
    path = str(path or "")
    return path[len("/testbed/"):] if path.startswith("/testbed/") else path


def lang_of(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {".lean": "lean", ".py": "python", ".sh": "sh", ".json": "json", ".toml": "toml", ".md": "md"}.get(suffix, "text")


def first_line(text: str, limit: int = 160) -> str:
    line = next((l for l in str(text or "").split("\n") if l.strip()), "").strip()
    return line if len(line) <= limit else line[:limit] + "…"


def text_diff(path: str, old: str, new: str) -> str:
    return "".join(unified_diff((old or "").splitlines(True), (new or "").splitlines(True),
                                f"a/{path}", f"b/{path}", n=2)).rstrip("\n")


JS_STRING_PATTERN = r'"((?:[^"\\]|\\.)*)"'
_PAPER = None
APPLY_PATCH = r"tools\.apply_patch\(\s*(?:" + JS_STRING_PATTERN + r")?"


def paper_action(step: dict, text: str, repo: str) -> str | list[str]:
    """The paper's action class for one tool call, by the classifier behind its action figure
    (scripts/analysis/plot_action_classes.py): Build, Verify, Measure, Read, Search, Edit, Git,
    Lake, Sleep or Other.  `text` is the call's full input, `repo` the repository the run
    started from: an Edit must change one of its files, and edits of scratch files are Other.
    File tools go by what they do, whatever a harness calls them; a command by the paper's
    shell rules.  A code-mode script is one step here but, as in the paper, one action per
    command or patch it runs: it gets their classes, in order, as a list.  The paper leaves
    out polling, planning and asking the user; here they are Other."""
    global _PAPER
    if _PAPER is None:
        sys.path.insert(0, str(LEANLEAN_CODE / "scripts/analysis"))
        import plot_action_classes
        _PAPER = plot_action_classes
    files = _PAPER.local_repository_files(repo)
    classify = lambda name, args: _PAPER.classify(name, args, files)[0] or "Other"
    name, category, text = step["n"], step["c"], text or ""
    if category == "read":
        return "Read"
    if category == "search":
        return "Search"
    if category in ("edit", "write"):
        # The paths the tool wrote, as its title lists them (relative to the repository).
        if name == "apply_patch":
            return classify(name, {"patch": text})
        return classify(name, {"changes": [{"path": path} for path in step["ti"].split(", ") if path]})
    if category == "run" and step.get("il") == "js":
        actions = []
        for match in re.finditer(EXEC_COMMAND + "|" + APPLY_PATCH, text):
            if match.group(1) is not None:
                actions.append(classify("bash", {"command": js_string(match.group(1))}))
            else:
                # A patch passed as a literal, or bound to a name first.
                patch = js_string(match.group(2)) if match.group(2) is not None else \
                    "\n".join(js_string(m) for m in re.findall(JS_STRING_PATTERN, text) if "*** " in m)
                actions.append(classify("apply_patch", {"patch": patch}))
        return actions[0] if len(actions) == 1 else actions or "Other"
    if category == "run" and text:
        return classify("bash", {"command": text})
    return classify(name, {})


def classify_steps(steps: list[dict], inputs: dict[int, str], repo: str) -> None:
    """Set every tool call's paper action, from its full input."""
    for index, step in enumerate(steps):
        if step["k"] == "tool":
            step["ac"] = paper_action(step, inputs.get(index, ""), repo)


def iso_seconds(stamp: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def rollout_start(rollout: Path) -> float | None:
    """The time of a Codex rollout's first event, which its steps' times count from."""
    with gzip.open(rollout, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                stamp = json.loads(line).get("timestamp")
            except json.JSONDecodeError:
                continue
            if stamp:
                return iso_seconds(stamp)
    return None


def codex_paper_actions(steps: list[dict], inputs: dict[int, str], run: Path, capture: Path, start: float, repo: str) -> None:
    """Classify Codex code-mode steps by the actions the paper counted for them.

    The paper reads the commands and file changes a run actually made (its standardized
    trace), not the scripts that made them: a script's loop runs commands its text never
    spells out.  Each recorded action goes to the script that ran it: one that spells out
    the same command, else the last script started before the action finished."""
    paper_action({"n": "", "c": "other"}, "", repo)  # loads the paper's classifier
    files = _PAPER.local_repository_files(repo)
    calls = [i for i, st in enumerate(steps) if st["k"] == "tool" and st["n"] in ("exec", "exec_command", "apply_patch")]
    if not calls:
        return
    begin = [start + steps[i]["t"] for i in calls]
    spelled = {}
    for k, i in enumerate(calls):
        text = inputs.get(i, "")
        commands = [js_string(m) for m in re.findall(EXEC_COMMAND, text)] if steps[i].get("il") == "js" else \
            [text] if steps[i]["n"] == "exec_command" else []
        for command in commands:
            spelled.setdefault(command.strip(), []).append(k)
    trace = json.loads((capture / "standardized-trace.json").read_text())
    stamps = {str(a.get("tool_call_id") or a.get("action_index")): (a.get("source_observation") or {}).get("wall_timestamp")
              for a in trace.get("actions", [])}
    assigned, finished = {}, start
    for capture_name, identity, name, args in _PAPER.actions(run):
        if capture_name != capture.name:
            continue
        action = _PAPER.classify(name, args, files)[0]
        if action is None:
            continue
        # An action without a time stamp finished no earlier than the one before it.
        finished = iso_seconds(stamps[identity]) if stamps.get(identity) else finished
        latest = max(bisect_right(begin, finished) - 1, 0)
        command = str(args.get("command") or "")
        wrapped = re.match(r"^/bin/bash -lc (.*)$", command, re.S)
        if wrapped:
            try:
                command = shlex.split(wrapped.group(1))[0]
            except (ValueError, IndexError):
                command = wrapped.group(1)
        owners = spelled.get(command.strip(), [])
        # The last script spelling it out that started before the action finished, else the first after.
        k = max((o for o in owners if o <= latest), default=min(owners, default=latest))
        assigned.setdefault(calls[k], []).append(action)
    if not assigned:
        return
    # A call the paper recorded no action for (a patch that failed to apply, a command that never
    # ran) did nothing it counts: Other.
    for i in calls:
        actions = assigned.get(i, ["Other"])
        steps[i]["ac"] = actions[0] if len(actions) == 1 else actions


def tool(t, name, category, title, input_text="", input_lang="text", output="", error=False, **extra) -> dict:
    # The full input is kept aside (as _in) until the step is classified; only its clip is written.
    step = {"k": "tool", "t": round(t, 1), "n": name, "c": category, "ti": title, "_in": input_text}
    if input_text:
        step["i"] = clip(input_text, CLIP_INPUT)
        step["il"] = input_lang
    if output:
        step["o"] = clip(output, CLIP_OUTPUT)
    if error:
        step["e"] = 1
    step.update({k: v for k, v in extra.items() if v is not None})
    return step


def say(t, text, kind="say") -> dict | None:
    text = str(text or "").strip()
    return {"k": kind, "t": round(t, 1), "x": clip(text, CLIP_TEXT)} if text else None


def is_compaction(text: str) -> bool:
    return text.startswith(("This session is being continued from a previous conversation",
                            "Another language model started to solve this problem"))


def result_text(value) -> str:
    """A tool result as text: strings as they are, content blocks joined, the rest as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(result_text(v) for v in value)
    if isinstance(value, dict):
        if value.get("type") in ("text", "input_text", "output_text"):
            return value.get("text", "")
        if value.get("type") == "image":
            return "[image]"
        if "content" in value and len(value) <= 4:
            return result_text(value["content"])
        if "stdout" in value or "stderr" in value:
            return "\n".join(x for x in (value.get("stdout"), value.get("stderr")) if x) or value.get("output", "")
    return json.dumps(value, ensure_ascii=False, indent=1)


class Clock:
    """Time of a native sequence number, from the actions whose time is observed."""

    def __init__(self, pairs):
        pairs = sorted((s, t) for s, t in pairs if s is not None and t is not None)
        self.seq = [s for s, _ in pairs]
        self.time = [t for _, t in pairs]

    def at(self, sequence) -> float:
        if not self.seq or sequence is None:
            return 0.0
        i = bisect_right(self.seq, sequence) - 1
        return self.time[max(i, 0)] if i >= 0 else self.time[0]


def action_time(action) -> float | None:
    observation = action.get("source_observation") or {}
    return observation.get("elapsed_seconds")


# ── Claude Code (Opus, Fable): stream-json, standardized ───────────────────

def claude_steps(trace: dict, capture: Path) -> list[dict]:
    actions = {a["tool_call_id"]: a for a in trace["actions"]}
    clock = Clock([(a.get("completed_sequence"), action_time(a)) for a in trace["actions"]])
    steps, open_tools = [], {}
    agents = {a["agent_id"]: a for a in trace.get("agents", [])}
    for message in sorted(trace["messages"], key=lambda m: m.get("trace_sequence") or 0):
        t = clock.at(message.get("trace_sequence"))
        agent = message.get("trace_agent_id")
        agent = None if agent in (None, "root") else agent
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            kind = block.get("type")
            if kind == "text":
                text = block.get("text", "")
                step = say(t, text, "compact" if is_compaction(text) else "user" if message.get("role") == "user" else "say")
                if step:
                    if agent:
                        step["a"] = agent
                    steps.append(step)
            elif kind == "thinking" and block.get("thinking"):
                # Present only when Claude Code records the text, not just a signature.
                step = say(t, block["thinking"], "think")
                if agent:
                    step["a"] = agent
                steps.append(step)
            elif kind == "tool_use":
                action = actions.get(block.get("id"), {})
                t_tool = action_time(action) or t
                step = claude_tool(t_tool, block.get("name", ""), block.get("input") or {})
                if agent:
                    step["a"] = agent
                step["id"] = block.get("id")
                open_tools[block.get("id")] = step
                steps.append(step)
            elif kind == "tool_result":
                step = open_tools.get(block.get("tool_use_id"))
                if step is not None:
                    output = result_text(block.get("content"))
                    if output:
                        step["o"] = clip(output, CLIP_OUTPUT)
                    if block.get("is_error"):
                        step["e"] = 1
    for step in steps:
        step.pop("id", None)
    if len(agents) > 1:
        names = {aid: (a.get("input") or {}).get("description") or aid for aid, a in agents.items()}
        for step in steps:
            if "a" in step:
                step["a"] = names.get(step["a"], step["a"])
    return steps


def claude_tool(t, name, args) -> dict:
    path = strip_testbed(args.get("file_path") or args.get("notebook_path") or args.get("path") or "")
    if name == "Bash":
        command = args.get("command", "")
        return tool(t, name, "run", args.get("description") or first_line(command), command, "sh")
    if name in ("Edit", "MultiEdit"):
        edits = args.get("edits") or [args]
        diff = "\n".join(text_diff(path, e.get("old_string", ""), e.get("new_string", "")) for e in edits)
        return tool(t, name, "edit", path, diff, "diff")
    if name == "Write":
        return tool(t, name, "write", path, args.get("content", ""), lang_of(path))
    if name == "Read":
        span = f" (lines {args.get('offset') or 1}–{(args.get('offset') or 1) + args['limit'] - 1})" if args.get("limit") else ""
        return tool(t, name, "read", path + span)
    if name in ("Grep", "Glob"):
        return tool(t, name, "search", f"{args.get('pattern', '')}" + (f" in {strip_testbed(args['path'])}" if args.get("path") else ""))
    if name in ("Task", "Agent"):
        return tool(t, name, "agent", args.get("description") or "Subagent", args.get("prompt", ""), "md")
    if name == "TodoWrite":
        todos = args.get("todos") or []
        return tool(t, name, "plan", f"{len(todos)} todo items",
                    "\n".join(f"[{'x' if d.get('status') == 'completed' else '~' if d.get('status') == 'in_progress' else ' '}] {d.get('content', '')}" for d in todos), "md")
    return tool(t, name, "other", name, json.dumps(args, ensure_ascii=False, indent=1), "json")


# ── Codex (Sol, Luna, Astra): rollout, or the exec stream without one ───────

JS_STRING = JS_STRING_PATTERN
EXEC_COMMAND = r"tools\.exec_command\(\{\s*\"?cmd\"?\s*:\s*" + JS_STRING


def js_string(literal: str) -> str:
    try:
        return json.loads(f'"{literal}"')
    except json.JSONDecodeError:
        return literal


def summary_text(item: dict) -> str:
    """The readable summary of a Responses-API reasoning item; its full text is encrypted."""
    return "\n\n".join(part.get("text", "") for part in item.get("summary") or [] if isinstance(part, dict)).strip()


def codex_output(output: str) -> str:
    """Unwrap the JSON chunks some Codex versions return for a command (`{"chunk_id": …, "output": …}`)."""
    parts, found = [], False
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped.startswith("{") and '"output"' in stripped:
            try:
                chunk = json.loads(stripped)
            except json.JSONDecodeError:
                parts.append(line)
                continue
            found = True
            code = chunk.get("exit_code")
            parts.append(str(chunk.get("output") or "").rstrip("\n") + (f"\n[exit {code}]" if code not in (0, None) else ""))
        elif stripped or not found:
            parts.append(line)
    return "\n".join(parts).strip("\n") if found else output


def codex_script(t, script: str, output: str) -> dict:
    """One code-mode `exec` call: a shell command or a patch when it is just that, else the script."""
    output = codex_output(output)
    commands = [js_string(m) for m in re.findall(EXEC_COMMAND, script)]
    patches = re.findall(r"tools\.apply_patch\(\s*" + JS_STRING, script) or \
        (re.findall(r"(?:const|let|var)\s+\w+\s*=\s*" + JS_STRING, script) if "tools.apply_patch(" in script else [])
    polls = script.count("tools.write_stdin(")
    if patches and not commands:
        patch = "\n".join(js_string(p) for p in patches)
        files = [strip_testbed(f) for f in re.findall(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", patch, re.M)]
        return tool(t, "apply_patch", "edit", ", ".join(dict.fromkeys(files)) or "patch", patch, "diff", output)
    # One command, perhaps followed by a poll of its session: show it as that command.
    if len(commands) == 1 and script.count("tools.") == 1 + polls and polls <= 1:
        return tool(t, "exec_command", "run", first_line(commands[0]), commands[0], "sh", output)
    if not commands and polls:
        return tool(t, "write_stdin", "wait", "Wait for a running command", script, "js", output)
    title = first_line(commands[0]) if commands else first_line(script)
    if len(commands) > 1:
        title = f"{len(commands)} commands: {title}"
    return tool(t, "exec", "run", title, script, "js", output)


def codex_rollout_steps(path: Path) -> list[dict]:
    steps, calls, start = [], {}, None
    from datetime import datetime
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            stamp = event.get("timestamp")
            now = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() if stamp else None
            if start is None and now is not None:
                start = now
            t = (now - start) if now is not None and start is not None else (steps[-1]["t"] if steps else 0.0)
            kind, payload = event.get("type"), event.get("payload") or {}
            ptype = payload.get("type")
            if kind == "compacted":
                step = say(t, payload.get("message", ""), "compact")
                if step:
                    steps.append(step)
            elif kind == "event_msg" and ptype == "item_completed":
                item = payload.get("item") or {}
                if item.get("type") == "UserMessage":
                    text = "\n".join(c.get("text", "") for c in item.get("content") or [] if isinstance(c, dict))
                    step = say(t, text, "user")
                    if step:
                        steps.append(step)
            elif kind == "response_item" and ptype == "message" and payload.get("role") == "assistant":
                text = "\n".join(c.get("text", "") for c in payload.get("content") or [] if isinstance(c, dict))
                step = say(t, text)
                if step:
                    steps.append(step)
            elif kind == "response_item" and ptype == "reasoning":
                step = say(t, summary_text(payload), "think")
                if step:
                    step["s"] = 1
                    steps.append(step)
            elif kind == "response_item" and ptype in ("custom_tool_call", "function_call"):
                calls[payload.get("call_id")] = (t, payload)
            elif kind == "response_item" and ptype in ("custom_tool_call_output", "function_call_output"):
                started = calls.pop(payload.get("call_id"), None)
                if started is None:
                    continue
                t0, call = started
                output = result_text(payload.get("output"))
                failed = output.startswith("Script failed") or "Script error:" in output[:400]
                output = re.sub(r"^Script (?:completed|failed)\nWall time [\d.]+ seconds\nOutput:\n", "", output)
                if call.get("type") == "custom_tool_call":
                    step = codex_script(t0, call.get("input", ""), output)
                else:
                    step = tool(t0, call.get("name", ""), "wait" if call.get("name") == "wait" else "other",
                                call.get("name", ""), call.get("arguments", ""), "json", output)
                if failed:
                    step["e"] = 1
                steps.append(step)
    steps.sort(key=lambda s: s["t"])
    return steps


def codex_stream_steps(trace: dict) -> list[dict]:
    """Codex without a retained rollout: commands, file-change paths and messages from the exec stream."""
    clock = Clock([(a.get("completed_sequence"), action_time(a)) for a in trace["actions"]])
    steps = []
    for message in trace["messages"]:
        step = say(clock.at(message.get("sequence")), message.get("content"))
        if step:
            steps.append(step)
    for action in trace["actions"]:
        t = action_time(action) or clock.at(action.get("completed_sequence"))
        args = action.get("arguments") or {}
        if action["name"] == "FileChange":
            files = [f"{c.get('kind')} {strip_testbed(c.get('path'))}" for c in args.get("changes") or []]
            steps.append(tool(t, "file_change", "edit", ", ".join(strip_testbed(c.get("path")) for c in args.get("changes") or []),
                              "\n".join(files), "text"))
        else:
            command = args.get("command", "")
            inner = re.match(r"^/bin/bash -lc (.*)$", command, re.S)
            shown = inner.group(1).strip("\"'") if inner else command
            code = action.get("observed_returncode")
            steps.append(tool(t, "command_execution", "run", first_line(shown), shown, "sh", result_text(action.get("result")),
                              error=bool(code)))
    steps.sort(key=lambda s: s["t"])
    return steps


# ── Antigravity CLI (Gemini): step stream ──────────────────────────────────

def antigravity_steps(trace: dict, capture: Path) -> list[dict]:
    times = {}
    for action in trace["actions"]:
        match = re.match(r"antigravity-step-(\d+)$", str(action.get("tool_call_id")))
        if match:
            times[int(match.group(1))] = action_time(action)
    clock = Clock(times.items())
    steps = []
    # ACTIVE and DONE carry consecutive text deltas. DONE can contain only the
    # tail (or no text at all), so assemble the entire reply before rendering it.
    replies = defaultdict(list)
    with (capture / "native.stdout.jsonl").open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            update = event.get("step_update") or {}
            index = update.get("step_index")
            reply_key = (update.get("conversation_id"), index)
            if update.get("step_type") == "agent_response" and update.get("text_delta"):
                replies[reply_key].append(update["text_delta"])
            if update.get("state") != "DONE":
                continue
            t = times.get(index) or clock.at(index)
            if update.get("step_type") == "agent_response":
                step = say(t, "".join(replies.pop(reply_key, [])))
                if step:
                    steps.append(step)
            elif update.get("step_type") == "user_input" and update.get("text"):
                step = say(t, update.get("text"), "user")
                if step:
                    steps.append(step)
            elif update.get("step_type") == "tool":
                info = update.get("tool_info") or {}
                steps.append(antigravity_tool(t, info.get("name") or update.get("tool_name", ""), info.get("parameters") or {},
                                              info.get("output"), info.get("error")))
    return steps


def antigravity_tool(t, name, args, output, error) -> dict:
    output = result_text(output)
    if error:
        output = (output + "\n" if output else "") + result_text(error)
    path = strip_testbed(args.get("AbsolutePath") or args.get("TargetFile") or args.get("DirectoryPath") or "")
    if name == "run_command":
        return tool(t, name, "run", first_line(args.get("CommandLine", "")), args.get("CommandLine", ""), "sh", output, bool(error))
    if name == "view_file":
        span = f" (lines {args['StartLine']}–{args.get('EndLine', '')})" if args.get("StartLine") else ""
        return tool(t, name, "read", path + span, output="" if re.fullmatch(r"\d+ lines, \d+ bytes", output.strip()) else output, error=bool(error))
    if name in ("write_to_file", "replace_file_content", "multi_replace_file_content", "sed_file"):
        body = args.get("CodeContent") or args.get("ReplacementContent") or ""
        if args.get("TargetContent") is not None:
            body = text_diff(path, args.get("TargetContent", ""), args.get("ReplacementContent", ""))
        return tool(t, name, "write" if name == "write_to_file" else "edit", path, body,
                    "diff" if args.get("TargetContent") is not None else lang_of(path), output, bool(error))
    if name in ("grep_search", "find_by_name"):
        query = args.get("Query") or args.get("Pattern") or ""
        where = strip_testbed(args.get("SearchPath") or args.get("SearchDirectory") or "")
        return tool(t, name, "search", f"{query}" + (f" in {where}" if where else ""), output=output, error=bool(error))
    if name == "list_dir":
        return tool(t, name, "read", path or "/", output=output, error=bool(error))
    if name in ("manage_task", "command_status", "wait", "wait_5_seconds"):
        what = " ".join(str(v) for v in (args.get("Action"), args.get("TaskId") or args.get("CommandId")) if v)
        return tool(t, name, "wait", what or name, output=output, error=bool(error))
    return tool(t, name, "other", name, json.dumps(args, ensure_ascii=False, indent=1) if args else "", "json", output, bool(error))


# ── Muse: the harness's event log ──────────────────────────────────────────

def uuid7_seconds(value) -> float | None:
    """The Unix time in a UUIDv7 (Muse's call ids: `call_01a0b916…`), or None."""
    digits = re.sub(r"[^0-9a-f]", "", str(value or "").lower().removeprefix("call_"))
    if len(digits) != 32 or digits[12] != "7":
        return None
    return int(digits[:12], 16) / 1000


def muse_steps(capture: Path, start: float | None) -> list[dict]:
    """Muse's `recorded_at` is a counter, not a clock, but its call ids are UUIDv7s
    that carry the time each call was made.  `start` is the run's start (Unix time);
    without it, or without a timed call, times are estimated later (estimate_times)."""
    steps, pending_text = [], []

    def flush(t):
        if pending_text:
            step = say(t, "".join(pending_text))
            if step:
                steps.append(step)
            pending_text.clear()

    last_t = 0.0
    with (capture / "native.stdout.jsonl").open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, payload = event.get("payload_type"), event.get("payload") or {}
            if kind == "run.output.delta":
                pending_text.append(payload.get("text", ""))
            elif kind == "run.terminal.completed":
                # The streamed deltas add up to the reply the terminal event repeats in full.
                if payload.get("text"):
                    pending_text[:] = [payload["text"]]
                flush(last_t)
            elif kind == "turn.input.user":
                flush(last_t)
                step = say(last_t, payload.get("prompt"), "user")
                if step:
                    steps.append(step)
            elif kind == "tool.result":
                stamp = uuid7_seconds(payload.get("call_id"))
                if start is not None and stamp is not None:
                    last_t = max(last_t, stamp - start)
                flush(last_t)
                name = (payload.get("correlation_facts") or {}).get("tool_name") or "bash"
                steps.append(muse_tool(last_t, name, payload))
    flush(last_t)
    return steps


def muse_reasoning(run: Path, start: float | None) -> list[dict]:
    """Reasoning summaries from the model responses the gateway recorded (the run's traj.json),
    timed by each response's creation.  The full reasoning is encrypted."""
    path = next(iter(sorted(run.glob("*.traj.json"))), None)
    if start is None or path is None:
        return []
    try:
        responses = json.loads(path.read_text(encoding="utf-8")).get("responses") or []
    except (OSError, json.JSONDecodeError):
        return []
    steps = []
    for response in responses:
        created = response.get("created_at")
        if not isinstance(response, dict) or created is None:
            continue
        for item in response.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                step = say(max(0.0, created - start), summary_text(item), "think")
                if step:
                    step["s"] = 1
                    steps.append(step)
    return steps


def muse_tool(t, name, payload) -> dict:
    text = payload.get("text") or ""
    outcome = (payload.get("correlation_facts") or {}).get("outcome")
    data = None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict) and "command" in data:
        code = data.get("exit_code")
        output = data.get("output") or ""
        if data.get("execution_state") == "background_running":
            output = output or "(running in the background)"
        return tool(t, name, "run", data.get("description") or first_line(data["command"]), data["command"], "sh", output,
                    error=(code not in (0, None)) or outcome == "failure", x=code)
    facts = payload.get("edit_facts") or {}
    path = strip_testbed(facts.get("path", ""))
    if name == "edit_file":
        diff = text.split("\n", 2)[2] if text.startswith("edited") and text.count("\n") >= 2 else text
        return tool(t, name, "edit", path, diff, "diff", error=outcome == "failure")
    if name == "write_file":
        return tool(t, name, "write", path, output=text, error=outcome == "failure")
    if name == "read_file":
        match = re.match(r"Read text file `([^`]*)`\.?\n?", text)
        return tool(t, name, "read", strip_testbed(match.group(1)) if match else "file",
                    output=text[match.end():] if match else text, error=outcome == "failure")
    if name == "write_todos":
        return tool(t, name, "plan", "Update the todo list", output=text)
    if name == "search":
        return tool(t, name, "search", "search", output=text, error=outcome == "failure")
    return tool(t, name, "other", name, output=text, error=outcome == "failure")


# ── Mistral Vibe (Leanstral): public history, standardized ─────────────────

def vibe_steps(trace: dict) -> list[dict]:
    clock = Clock([(a.get("completed_sequence"), action_time(a)) for a in trace["actions"]])
    steps = []
    for message in trace["messages"]:
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        t = clock.at(message.get("sequence"))
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = "think" if block.get("type") == "reasoning" else "user" if message.get("role") == "user" else "say"
            step = say(t, block.get("text"), kind)
            if step:
                step["_s"] = message.get("sequence") or 0
                steps.append(step)
    for action in trace["actions"]:
        t = action_time(action) or clock.at(action.get("completed_sequence"))
        args = action.get("arguments") or {}
        name = action["name"]
        result = action.get("result")
        output = result_text(result)
        error = bool(action.get("is_error"))
        path = strip_testbed(args.get("file_path") or args.get("path") or "")
        if name == "bash":
            step = tool(t, name, "run", first_line(args.get("command", "")), args.get("command", ""), "sh", output, error)
        elif name == "read_file":
            step = tool(t, name, "read", path, output=output, error=error)
        elif name == "write_file":
            step = tool(t, name, "write", path, args.get("content", ""), lang_of(path), error=error,
                        o=clip(output, CLIP_OUTPUT) if error else None)
        elif name in ("edit", "search_replace"):
            diff = text_diff(path, args.get("old_string") or args.get("search") or "", args.get("new_string") or args.get("replace") or "")
            step = tool(t, name, "edit", path, diff, "diff", error=error, o=clip(output, CLIP_OUTPUT) if error else None)
        elif name == "grep":
            matches = result.get("matches") if isinstance(result, dict) and "matches" in result else output
            step = tool(t, name, "search", f"{args.get('pattern', '')}" + (f" in {path}" if path else ""), output=result_text(matches), error=error)
        else:
            step = tool(t, name, "other", name, json.dumps(args, ensure_ascii=False, indent=1) if args else "", "json", output, error)
        step["_s"] = action.get("started_sequence") or action.get("completed_sequence") or 0
        steps.append(step)
    steps.sort(key=lambda s: s.pop("_s"))
    return steps


# ── source changes and the token trajectory ────────────────────────────────

def diff_stats(text: str) -> tuple[list[str], int, int]:
    """Files a diff touches, and its added and removed lines.  A `--- ` line is a file
    header only when a `+++ ` line follows: otherwise it is a removed Lean comment."""
    files, adds, dels = [], 0, 0
    lines = text.split("\n")
    k = 0
    while k < len(lines):
        line = lines[k]
        git = re.match(r"^diff --git a/(.*) b/(.*)$", line)
        if git:
            files += [git.group(1), git.group(2)]
        elif line.startswith("--- ") and k + 1 < len(lines) and lines[k + 1].startswith("+++ "):
            for header in (line, lines[k + 1]):
                path = header[4:].strip()
                if path != "/dev/null":
                    files.append(path[2:] if path[:2] in ("a/", "b/") else path)
            k += 1
        elif line.startswith("+"):
            adds += 1
        elif line.startswith("-"):
            dels += 1
        k += 1
    return list(dict.fromkeys(files)), adds, dels


def token_metric_path(root=None):
    root = root or LEANLEAN_CODE
    for package in ("leanlean", "leancompression"):
        path = root / "src" / package / "metrics/tokens.py"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Lean token metric not found in {root}")


class TokenCounter:
    """The paper's Lean-token metric at every checkpoint, counted from its source snapshot.

    The playback measured only some checkpoints.  This recounts each snapshot with
    LeanLean's own metric (every .lean file outside .lake except lakefile.lean and the
    benchmark's fixture files such as Challenge.lean, without comments and imports),
    recounting only the files the checkpoint's diff touched."""

    def __init__(self, playback: dict, capture: Path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("website_token_metric", token_metric_path())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.count = module.count_lean_tokens_in_source
        self.counts = {}
        self.root = capture.parent.parent
        fixtures = next((p["benchmark_fixture_integrity"].get("files") or {} for p in playback.get("points", [])
                         if isinstance(p.get("benchmark_fixture_integrity"), dict)), None)
        self.fixtures = {f for f in fixtures if f.endswith(".lean")} if fixtures is not None else None
        self.files: dict[str, int] | None = None

    def scored(self, path: str) -> bool:
        parts = path.split("/")
        if not path.endswith(".lean") or parts[-1] == "lakefile.lean" or ".lake" in parts:
            return False
        if self.fixtures is not None:
            return path not in self.fixtures
        return not parts[-1].endswith("Challenge.lean")

    def read(self, snapshot: str, only: set[str] | None):
        import tarfile
        found = {}
        with tarfile.open(self.root / snapshot) as archive:
            for member in archive:
                path = member.name[2:] if member.name.startswith("./") else member.name
                if member.isfile() and self.scored(path) and (only is None or path in only):
                    source = archive.extractfile(member).read()
                    digest = hashlib.sha256(source).digest()
                    if digest not in self.counts:
                        self.counts[digest] = self.count(source.decode("utf-8", "replace"))
                    found[path] = self.counts[digest]
        return found

    def at(self, point: dict, changed: list[str] | None) -> int | None:
        snapshot = point.get("snapshot_path")
        if not snapshot or not (self.root / snapshot).is_file():
            return None
        if self.files is None or changed is None:
            self.files = self.read(snapshot, None)
        else:
            touched = {f for f in changed if self.scored(f)}
            if touched:
                fresh = self.read(snapshot, touched)
                for path in touched:
                    if path in fresh:
                        self.files[path] = fresh[path]
                    else:
                        self.files.pop(path, None)
        return sum(self.files.values())


class RecordedTokens:
    """Use checkpoint measurements already computed by postprocessing."""
    files = None

    def at(self, point, changed):
        return None


def measure(playback: dict, capture: Path, trace: dict | None, *, recorded=False, strict=False) -> tuple[list[list], list[dict], dict]:
    """The Lean-token trajectory [seconds, tokens, cost] and every source change, with
    its files, diff and the Lean tokens after it.  Counts the playback recorded are
    checked against the recount."""
    by_tool = {}
    if trace:
        for action in trace.get("actions", []):
            observation = action.get("source_observation") or {}
            if observation.get("edit_index") is not None:
                by_tool[observation["edit_index"]] = action_time(action)
    counter = RecordedTokens() if recorded else TokenCounter(playback, capture)
    progress, changes, check = [], [], {"checked": 0, "mismatched": 0}
    submitted = None
    for point in playback.get("points", []):
        if point.get("endpoint_role") == "submitted_patch":
            submitted = point
            continue
        t = point.get("elapsed_seconds") or 0.0
        diff_path = capture.parent.parent / point["incremental_diff_path"] if point.get("incremental_diff_path") else None
        text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path and diff_path.is_file() else None
        files, adds, dels = diff_stats(text) if text else ([], 0, 0)
        baseline = point.get("kind") == "baseline"
        try:
            tokens_after = counter.at(point, None if baseline or text is None else files)
        except Exception:  # an unreadable snapshot: fall back to what the playback recorded
            if strict:
                raise
            tokens_after, counter.files = None, None
        recorded = point.get("lean_tokens")
        if recorded is not None and tokens_after is not None:
            check["checked"] += 1
            check["mismatched"] += recorded != tokens_after
            if strict and recorded != tokens_after:
                raise ValueError(f"Checkpoint token mismatch: {capture}, {point.get('snapshot_path')}: {tokens_after} != {recorded}")
        if tokens_after is None:
            if strict:
                raise ValueError(f"Unmeasured checkpoint: {capture}, {point.get('snapshot_path')}")
            tokens_after = recorded
        if tokens_after is not None and (baseline or files or point.get("kind") == "final" or not progress):
            progress.append([0 if baseline else round(t, 1), tokens_after, 0 if baseline else round(point.get("cost_usd") or 0, 3)])
        if not baseline and (files or adds or dels):
            changes.append({"t": by_tool.get(point.get("edit_index"), t), "files": files, "adds": adds, "dels": dels,
                            "text": text, "tokens": tokens_after, "edit": point.get("edit_index")})
    # The score is measured on the submitted patch, which can differ from the last snapshot the
    # monitor took: end the trajectory on it.
    if submitted is not None and progress:
        try:
            tokens_final = counter.at(submitted, None)
        except Exception:
            if strict:
                raise
            tokens_final = None
        if strict and tokens_final != submitted.get("lean_tokens"):
            raise ValueError(f"Submitted endpoint token mismatch: {capture}: {tokens_final} != {submitted.get('lean_tokens')}")
        tokens_final = tokens_final if tokens_final is not None else submitted.get("lean_tokens")
        if tokens_final is not None and tokens_final != progress[-1][1]:
            progress.append([max(progress[-1][0], round(submitted.get("elapsed_seconds") or 0, 1)), tokens_final,
                             round(submitted.get("cost_usd") or progress[-1][2], 3)])
    return progress, changes, check


def estimate_times(steps: list[dict], changes: list[dict], duration: float) -> None:
    """Times for a log without a clock: each source change is matched, in order, to the next
    edit of one of its files; steps between two matches are spread evenly between their times."""
    anchors, j = [(-1, 0.0)], 0
    for change in changes:
        files = set(change["files"])
        k = next((i for i in range(j, len(steps)) if steps[i]["k"] == "tool"
                  and steps[i]["c"] in ("edit", "write") and steps[i]["ti"] in files), None)
        if k is not None and change["t"] >= anchors[-1][1]:
            anchors.append((k, change["t"]))
            j = k + 1
    anchors.append((len(steps), max(duration, anchors[-1][1])))
    for (i0, t0), (i1, t1) in zip(anchors, anchors[1:]):
        for i in range(max(i0, 0), min(i1, len(steps))):
            steps[i]["t"] = round(t0 if i == i0 else t0 + (t1 - t0) * (i - i0) / (i1 - i0), 1)


def attach_changes(steps: list[dict], changes: list[dict], builds: dict[int, bool]) -> None:
    """Attach each source change, with a clipped diff and whether its checkpoint built, to the step that made it."""
    tool_times = [s["t"] for s in steps if s["k"] == "tool"]
    tool_index = [i for i, s in enumerate(steps) if s["k"] == "tool"]
    if not tool_index:
        return
    for c in changes:
        # The change belongs to the last tool call that finished at or before it was observed.
        j = bisect_right(tool_times, c["t"] + 0.5) - 1
        step = steps[tool_index[max(j, 0)]]
        files, adds, dels, text, tokens_after = c["files"], c["adds"], c["dels"], c["text"], c["tokens"]
        change = step.setdefault("d", {"f": [], "a": 0, "r": 0, "diff": ""})
        change["f"] = list(dict.fromkeys(change["f"] + files))
        change["a"] += adds
        change["r"] += dels
        if tokens_after is not None:
            change["tok"] = tokens_after
        if c["edit"] in builds:
            change["b"] = int(builds[c["edit"]])
        if len(change["diff"]) < CLIP_DIFF[1]:
            change["diff"] = clip((change["diff"] + "\n" if change["diff"] else "") + text, CLIP_DIFF)
    for step in steps:
        if "d" in step:
            step["d"]["f"] = step["d"]["f"][:40]


# ── runs ────────────────────────────────────────────────────────────────────

_BUILDS = None


def replay_builds(model: str, repo: str) -> list[dict]:
    """The paper's replayed checkpoint builds of a run (replay-checkpoint-builds.csv): each
    checkpoint the replay built, whether it built and passed verification, and its compression."""
    global _BUILDS
    if _BUILDS is None:
        from sources import PAPER_DATA
        _BUILDS = {}
        path = PAPER_DATA / "replay-checkpoint-builds.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    _BUILDS.setdefault((row["canonical_model"], row["repository"]), []).append({
                        "edit": int(row["edit_index"]), "t": float(row["elapsed_minutes"]) * 60,
                        "passed": row["build_passed"] == "True", "pct": float(row["compression_pct"])})
    return sorted(_BUILDS.get((model, repo), []), key=lambda b: (b["t"], b["edit"]))


def run_start(playback: dict) -> float | None:
    """The run's start as Unix time: a checkpoint's wall time minus its elapsed seconds."""
    from datetime import datetime
    for observation in playback.get("checkpoint_observations") or []:
        stamp, elapsed = observation.get("wall_timestamp"), observation.get("elapsed_seconds")
        if stamp and elapsed is not None:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() - elapsed
    return None


def capture_dir(run: Path) -> Path | None:
    playback = run / "playback"
    if (playback / "native.stdout.jsonl").is_file():
        return playback
    dispositions = playback / "capture-dispositions.json"
    if dispositions.is_file():
        captures = json.loads(dispositions.read_text()).get("captures", {})
        scored = [name for name, c in captures.items() if c.get("disposition") == "scored"]
        if scored:
            return playback / scored[-1]
    captures = sorted(playback.glob("capture_*")) if playback.is_dir() else []
    return captures[-1] if captures else None


def runs() -> list[dict]:
    from sources import EXTRA_MODELS, HIDDEN_MODELS, MODELS, PAPER_DATA, RESULTS
    SHORT.update({m["key"]: m["short"] for m in MODELS + EXTRA_MODELS if m["key"] not in HIDDEN_MODELS})
    out = []
    with (PAPER_DATA / "benchmark-repository-results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in SHORT and row["selected_directory"]:
                out.append({"repo": row["repository"], "model": row["model"], "status": row["status"],
                            "run": RESULTS / row["selected_directory"]})
    extra_keys = set(SHORT) & {m["key"] for m in EXTRA_MODELS}
    with (PAPER_DATA / "benchmark-extra-leaderboard-repositories.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in extra_keys and row["duration_source"]:
                run = RESULTS / os.path.dirname(row["duration_source"])
                while run.name.startswith(("capture_", "playback")):
                    run = run.parent
                out.append({"repo": row["repository"], "model": row["model"], "status": row["status"], "run": run})
    return out


def export_run(job: dict) -> dict:
    run, capture = job["run"], job.get("capture") or capture_dir(job["run"])
    if capture is None:
        return {**job, "error": "no capture"}
    trace_path = capture / "standardized-trace.json"
    trace = json.loads(trace_path.read_text()) if trace_path.is_file() else None
    provider = (trace or {}).get("provider") or "generic-jsonl"
    playback_path = next((capture / n for n in ("postprocessed-playback.json", "playback.json") if (capture / n).is_file()), None)
    playback = json.loads(playback_path.read_text()) if playback_path else {}
    rollouts = sorted((run / "native_rollout").glob(f"{capture.name}.native-rollout.jsonl.gz")) or \
        sorted((run / "native_rollout").glob("*.native-rollout.jsonl.gz"))

    if provider == "claude-code":
        steps, harness = claude_steps(trace, capture), "Claude Code"
    elif provider == "codex-cli":
        steps = codex_rollout_steps(rollouts[0]) if rollouts else codex_stream_steps(trace)
        harness = "Codex CLI"
    elif provider == "antigravity-cli":
        steps, harness = antigravity_steps(trace, capture), "Antigravity CLI"
    elif provider == "mistral-vibe":
        steps, harness = vibe_steps(trace), "Mistral Vibe"
    else:
        start = run_start(playback)
        steps, harness = muse_steps(capture, start), "Muse"
        # Reasoning comes first among a response's items: on a tie it goes before the tool call.
        thoughts = muse_reasoning(run, start)
        if thoughts:
            steps = sorted(thoughts + steps, key=lambda st: (st["t"], st["k"] != "think"))
    # The task prompt is shown once, above the steps.
    prompt = (run / "prompt.txt").read_text(encoding="utf-8") if (run / "prompt.txt").is_file() else None
    if steps and steps[0]["k"] == "user":
        prompt = prompt or steps[0]["x"]
        steps.pop(0)
    progress, changes, check = measure(playback, capture, trace, recorded=job.get("recorded_tokens", False), strict=job.get("recount_tokens", False))
    duration = playback.get("wall_time_seconds") or (steps[-1]["t"] if steps else 0)
    # Muse calls are timed by their UUIDv7 ids; only a run without them needs estimates.
    estimated = harness == "Muse" and not (start is not None and any(st["t"] > 0 for st in steps))
    if estimated:
        estimate_times(steps, changes, duration)
    builds = replay_builds(job["model"], job["repo"])
    attach_changes(steps, changes, {b["edit"]: b["passed"] for b in builds})
    inputs = {i: st.pop("_in") for i, st in enumerate(steps) if "_in" in st}
    classify_steps(steps, inputs, job["repo"])
    if provider == "codex-cli" and rollouts:
        codex_paper_actions(steps, inputs, run, capture, rollout_start(rollouts[0]), job["repo"])
    notes = {
        "Claude Code": "Claude Code records only a signature for each thinking block, so the model's reasoning is not shown.",
        "Codex CLI": ("Codex stores the model's reasoning encrypted; only the summaries the model wrote are shown. " if any(st["k"] == "think" for st in steps)
                      else "Codex stores the model's reasoning encrypted, so it is not shown. ")
        + "Each tool call is one code-mode script; a script that runs one command or applies one patch is shown as that command or patch.",
        "Antigravity CLI": "The Antigravity CLI reports only the number of thinking tokens, and does not record the content written by file edits; the source changes come from the playback snapshots.",
        "Muse": "Muse's event log records tool results and the model's final replies; step times come from the time stamped into each call's id. "
        + ("Its reasoning is encrypted; the summaries the model wrote come from the gateway's record of each response." if any(st["k"] == "think" for st in steps)
           else "Its reasoning is encrypted, and this run recorded no summaries."),
        "Mistral Vibe": "This run did not record the model's reasoning." if not any(st["k"] == "think" for st in steps) else None,
    }
    if estimated:
        notes["Muse"] = "Muse's event log has no clock: step times are estimated from the source snapshots, each matched to the edit that wrote it."
    if harness == "Codex CLI" and not rollouts:
        notes["Codex CLI"] = "This run kept only Codex's exec stream: commands and messages, without the content of file changes."
    payload = {
        "repo": job["repo"], "model": job["model"], "status": job["status"], "harness": harness,
        "prompt": prompt, "duration": round(duration or 0, 1), "cost": playback.get("final_cost_usd"),
        "baseline": playback.get("baseline_lean_tokens") or (progress[0][1] if progress else None),
        "final": playback.get("post_lean_tokens") or (progress[-1][1] if progress else None),
        "note": notes[harness], "estimated": estimated or None, "progress": progress,
        "builds": [[round(b["t"], 1), int(b["passed"]), round(b["pct"], 3)] for b in builds] or None, "steps": steps,
    }
    if job.get("result") is not None:
        result = job["result"]
        if job.get("recount_tokens") and (not progress or progress[0][1] != result["metrics"]["baseline_lean_tokens"]
                or progress[-1][1] != result["metrics"]["post_lean_tokens"]):
            raise ValueError(f"Trace trajectory disagrees with scored endpoints: {job['repo']}")
        payload.update(cost=result["cost_usd"], duration=round(result["agent_seconds"], 1),
                       baseline=result["metrics"]["baseline_lean_tokens"], final=result["metrics"]["post_lean_tokens"])
        # Audited totals do not establish cumulative cost at each checkpoint.
        if result.get("cost_correction"):
            payload["cost_basis"] = result["cost_correction"]["basis"]
            for point in payload["progress"]:
                point[2] = None
        if job.get("recorded_tokens"):
            payload["note"] = (payload["note"] or "") + " Token counts are the postprocessed checkpoint measurements."
        elif job.get("recount_tokens"):
            payload["note"] = (payload["note"] or "") + " Checkpoint token counts are recomputed from the archived source snapshots."
    written, raw = write_trace(OUT / job["repo"], SHORT[job["model"]], payload)
    tools = [s for s in steps if s["k"] == "tool"]
    return {**job, "steps": len(steps), "tools": len(tools), "changes": sum(1 for s in tools if "d" in s),
            "bytes": written, "raw": raw, "check": check}


# Bodies are packed in chunks of about this many bytes of JSON (≈10–20 kB gzipped).
CHUNK_BYTES = 96_000

# The runs ran under the exporting user's account, and a model that lists its mounts or
# environment prints that user's name: blank it out (the name itself is not written here).
USER_NAME = re.compile(rf"\b{re.escape(Path.home().name)}\b")


def scrub(value):
    """`value` with the local user name replaced by `user`, in every string."""
    if isinstance(value, str):
        return anonymize_text(USER_NAME.sub("user", value))
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


# Credentials an agent printed (`env`, a config file): the value of a secret-named variable,
# and API keys by their shape.  Redacted in everything written, inputs and outputs alike.
SECRETS = [
    (re.compile(r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)[A-Z0-9_]*)=([A-Za-z0-9_\-.+/]{16,})"), r"\1=[redacted]"),
    (re.compile(r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})"), "[redacted]"),
    (re.compile(r"(?i)\b(authorization|x-api-key)(\s*[:=]\s*)(?:bearer\s+)?[A-Za-z0-9._~+/-]{16,}"), r"\1\2[redacted]"),
]


def redact(value):
    """`value` with every string in it redacted, through dicts and lists."""
    if isinstance(value, str):
        for pattern, replacement in SECRETS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


def write_trace(folder: Path, short: str, payload: dict) -> tuple[int, int]:
    """Write a run as a head and a pack, so the viewer can show it from the head alone.

      <short>.json.gz  the head: the run, its timeline, and every step without its body
      <short>.bin      the pack: the bodies (inputs, outputs, diffs, compaction summaries)
                       of consecutive steps, as independently gzipped chunks, back to back

    The head's `pack` lists each chunk's [offset, length] and each step's `p` its chunk, so a
    step's body is one HTTP Range request.  Returns (bytes written, raw JSON bytes)."""
    folder.mkdir(parents=True, exist_ok=True)
    payload.update(scrub(payload))
    chunks, current, size = [], {}, 0
    for index, step in enumerate(payload["steps"]):
        body = {}
        if step["k"] == "tool" and step.get("c") == "run" and step.get("i"):
            step["ln"] = step["i"].count("\n")
        for key in ("i", "o"):
            if key in step:
                body[key] = step.pop(key)
        if step.get("d", {}).get("diff"):
            body["diff"] = step["d"].pop("diff")
        if step["k"] == "compact":
            body["x"] = step.pop("x")
        if not body:
            continue
        encoded = len(json.dumps(body, ensure_ascii=False))
        if current and size + encoded > CHUNK_BYTES:
            chunks.append(current)
            current, size = {}, 0
        current[index] = body
        size += encoded
        step["p"] = len(chunks)
    if current:
        chunks.append(current)
    pack, offsets = bytearray(), []
    for chunk in chunks:
        member = gzip.compress(json.dumps(redact(chunk), separators=(",", ":"), ensure_ascii=False).encode(), compresslevel=9, mtime=0)
        offsets.append([len(pack), len(member)])
        pack += member
    payload["pack"] = offsets
    head = json.dumps(redact(payload), separators=(",", ":"), ensure_ascii=False).encode()
    (folder / f"{short}.json.gz").write_bytes(gzip.compress(head, compresslevel=9, mtime=0))
    (folder / f"{short}.bin").write_bytes(bytes(pack))
    return (folder / f"{short}.json.gz").stat().st_size + len(pack), len(head) + sum(len(json.dumps(c)) for c in chunks)


def repack(folder: Path) -> None:
    """Rewrite every trace as head and pack without re-exporting it (after a change to the layout)."""
    for path in sorted(folder.glob("*/*.json.gz")):
        payload = json.loads(gzip.decompress(path.read_bytes()))
        offsets = payload.pop("pack", None)
        if offsets is not None:
            pack = path.with_name(path.name.removesuffix(".json.gz") + ".bin").read_bytes()
            for offset, length in offsets:
                for index, body in json.loads(gzip.decompress(pack[offset:offset + length])).items():
                    step = payload["steps"][int(index)]
                    step.pop("p", None)
                    if "diff" in body:
                        step["d"]["diff"] = body.pop("diff")
                    step.update(body)
        write_trace(path.parent, path.name.removesuffix(".json.gz"), payload)


def reclassify(folder: Path) -> None:
    """Set the paper's action on every tool call of the exported traces without re-exporting
    them, from the inputs they keep (clipped only past CLIP_INPUT), and for Codex runs from
    the actions the paper recorded."""
    jobs = {(j["repo"], SHORT[j["model"]]): j for j in runs()}
    for path in sorted(folder.glob("*/*.json.gz")):
        payload = json.loads(gzip.decompress(path.read_bytes()))
        pack = path.with_name(path.name.removesuffix(".json.gz") + ".bin").read_bytes()
        inputs = {}
        for offset, length in payload.get("pack") or []:
            for index, body in json.loads(gzip.decompress(pack[offset:offset + length])).items():
                if "i" in body:
                    inputs[int(index)] = body["i"]
        repo, short = path.parent.name, path.name.removesuffix(".json.gz")
        classify_steps(payload["steps"], inputs, repo)
        job = jobs.get((repo, short))
        capture = capture_dir(job["run"]) if job and payload.get("harness") == "Codex CLI" else None
        if capture is not None:
            rollouts = sorted((job["run"] / "native_rollout").glob(f"{capture.name}.native-rollout.jsonl.gz")) or \
                sorted((job["run"] / "native_rollout").glob("*.native-rollout.jsonl.gz"))
            if rollouts:
                codex_paper_actions(payload["steps"], inputs, job["run"], capture, rollout_start(rollouts[0]), repo)
        head = json.dumps(redact(payload), separators=(",", ":"), ensure_ascii=False).encode()
        path.write_bytes(gzip.compress(head, compresslevel=9, mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", help="repository ids")
    parser.add_argument("--models", nargs="*", help="model shorts")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--repack", action="store_true", help="only split existing one-file traces into head and pack")
    parser.add_argument("--reclassify", action="store_true", help="only set the paper's action on existing traces' tool calls")
    options = parser.parse_args()
    if options.repack:
        repack(OUT)
        return
    if options.reclassify:
        reclassify(OUT)
        return
    jobs = [j for j in runs() if (not options.only or j["repo"] in options.only)
            and (not options.models or SHORT[j["model"]] in options.models)]
    index_path = OUT / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() and (options.only or options.models) else {}
    total, mismatched = 0, 0
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        for result in pool.map(export_run, jobs):
            if "error" in result:
                print(f"  {result['repo']} {result['model']}: {result['error']}")
                continue
            index.setdefault(result["repo"], {})[SHORT[result["model"]]] = {
                "steps": result["steps"], "tools": result["tools"], "changes": result["changes"]}
            total += result["bytes"]
            check = result["check"]
            mismatched += check["mismatched"]
            print(f"  {result['repo']} {SHORT[result['model']]:>9}: {result['steps']:5d} steps, "
                  f"{result['bytes'] / 1e3:7.0f} kB ({result['raw'] / 1e6:.1f} MB raw), "
                  f"token recount {check['checked'] - check['mismatched']}/{check['checked']} match", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, separators=(",", ":"), sort_keys=True))
    print(f"traces: {len(jobs)} runs, {total / 1e6:.1f} MB; "
          f"{mismatched} checkpoint token counts differ from the playback's" if mismatched else
          f"traces: {len(jobs)} runs, {total / 1e6:.1f} MB; every recounted checkpoint matches the playback")


if __name__ == "__main__":
    main()
