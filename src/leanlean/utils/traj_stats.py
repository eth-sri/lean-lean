"""Lightweight trajectory statistics — no external dependencies."""

import json
import posixpath
import re


def _parse_cmd(m: dict) -> str:
    args = m.get("arguments", "")
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
        return parsed.get("cmd", "") or parsed.get("command", "") if isinstance(parsed, dict) else str(parsed)
    except Exception:
        return str(args)


def _is_lake_build(cmd: str) -> bool:
    return "lake build" in cmd and not any(
        x in cmd for x in ["lake build lib", "lake build --dry"]
    )


def _codex_lake_builds(messages: list) -> dict:
    """Count lake builds from Codex-format trajectories (attempt_message_type=tool_call)."""
    attempts = 0
    successes = 0
    for i, m in enumerate(messages):
        if m.get("attempt_message_type") != "tool_call":
            continue
        cmd = _parse_cmd(m)
        if not _is_lake_build(cmd):
            continue
        attempts += 1
        # Codex runs lake build async (polls via write_stdin), so scan ahead
        # until we find the exit code line rather than stopping at first output.
        for j in range(i + 1, min(i + 200, len(messages))):
            out = messages[j]
            if out.get("attempt_message_type") != "tool_output":
                continue
            text = str(out.get("output", ""))
            if "Process exited with code 0" in text or "Build completed successfully" in text:
                successes += 1
                break
            if "Process exited with code" in text:
                break
    return {"attempts": attempts, "successes": successes, "failures": attempts - successes}


def _claude_code_lake_builds(traj: dict) -> dict:
    """Count lake builds from Claude Code-format trajectories (tool_use/tool_result content blocks)."""
    # Collect all responses from attempts or top-level
    all_responses: list = []
    for attempt in (traj.get("info") or {}).get("attempts") or []:
        all_responses.extend(attempt.get("responses") or [])
    if not all_responses:
        all_responses = traj.get("responses") or []

    # Collect tool_results keyed by tool_use_id from all message sources
    all_messages: list = list(traj.get("messages") or [])
    for attempt in (traj.get("info") or {}).get("attempts") or []:
        all_messages.extend(attempt.get("messages") or [])
    result_map: dict = {}
    for m in all_messages:
        for block in m.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                uid = block.get("tool_use_id", "")
                if uid:
                    result_map[uid] = str(block.get("content", ""))

    # Deduplicate responses by response id (top-level and attempt-level overlap)
    seen_response_ids: set = set()
    attempts = 0
    successes = 0

    for r in all_responses:
        rid = r.get("id")
        if rid and rid in seen_response_ids:
            continue
        if rid:
            seen_response_ids.add(rid)
        for choice in r.get("choices") or []:
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                args_str = fn.get("arguments", "")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    cmd = (args.get("command") or args.get("cmd") or "") if isinstance(args, dict) else ""
                except Exception:
                    cmd = ""
                if not _is_lake_build(cmd):
                    continue
                attempts += 1
                uid = tc.get("id", "")
                result_text = result_map.get(uid, "")
                if result_text:
                    # Claude Code Bash tool returns exit code inline or omits it on success
                    if "returncode=0" in result_text or "Build completed successfully" in result_text:
                        successes += 1
                    elif result_text.strip() and "returncode=" not in result_text and "error" not in result_text.lower():
                        # No explicit failure marker — treat as success
                        successes += 1

    return {"attempts": attempts, "successes": successes, "failures": attempts - successes}


def count_lake_builds(traj: dict) -> dict:
    """Count lake build invocations and successes from a trajectory."""
    messages = traj.get("messages", [])

    # Detect Codex format: messages have attempt_message_type=tool_call
    has_codex_tool_calls = any(m.get("attempt_message_type") == "tool_call" for m in messages)
    if has_codex_tool_calls:
        return _codex_lake_builds(messages)

    # Claude Code format: tool calls live in responses, results in content blocks
    return _claude_code_lake_builds(traj)


def count_files_modified(patch: str) -> int:
    """Count unique .lean files changed in a diff patch."""
    files = set(re.findall(r"diff --git a/(\S+\.lean)", patch))
    return len(files)


_READ_TOOL_NAMES = {"read", "read_file", "view_file"}
_READ_COMMAND_RE = re.compile(
    r"(?:^|[;&|()\s'\"])(?:cat|head|tail|less|nl|grep|rg|sed|awk|wc)"
    r"(?=\s|$|[;&|()'\"])",
    re.IGNORECASE,
)
_EXPLICIT_LEAN_PATH_RE = re.compile(
    r"(?<![\w.*?{}+\-/])/?(?:[\w.+-]+/)*[\w.+-]+\.lean\b"
)
_BRACED_LEAN_PATH_RE = re.compile(
    r"(?<![\w.*?{}+\-/])(?P<prefix>/?(?:[\w.+-]+/)*)"
    r"\{(?P<names>[\w.+-]+(?:,[\w.+-]+)+)\}\.lean\b"
)


def _tool_arguments(value) -> dict:
    """Return tool arguments as a mapping across persisted trace formats."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {"value": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _normalise_lean_path(path: str) -> str:
    """Canonicalise the absolute and relative spellings used in the testbed."""
    path = path.strip().replace("\\", "/")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/") :]
    while path.startswith("./"):
        path = path[2:]
    return posixpath.normpath(path)


def _is_repository_lean_path(path: str) -> bool:
    """Whether a normalized path belongs to the repository source tree."""
    if path.startswith(("/", "../")):
        return False
    parts = path.split("/")
    return (
        ".lake" not in parts
        and path != "lakefile.lean"
    )


def _explicit_lean_paths(text: str) -> set[str]:
    """Extract concrete Lean paths, excluding glob patterns."""
    paths = {
        _normalise_lean_path(match.group(0))
        for match in _EXPLICIT_LEAN_PATH_RE.finditer(text)
    }
    for match in _BRACED_LEAN_PATH_RE.finditer(text):
        prefix = match.group("prefix")
        paths.update(
            _normalise_lean_path(f"{prefix}{name}.lean")
            for name in match.group("names").split(",")
        )
    return paths


def _record_files_read(name: str | None, raw_arguments, read_files: set[str]) -> None:
    """Add the concrete Lean files read by one structured tool call."""
    args = _tool_arguments(raw_arguments)
    tool_name = (name or "").lower()

    if tool_name in _READ_TOOL_NAMES:
        path = (
            args.get("file_path")
            or args.get("path")
            or args.get("file")
            or args.get("value")
        )
        if isinstance(path, str) and path.endswith(".lean"):
            read_files.add(_normalise_lean_path(path))
        paths = args.get("paths")
        if isinstance(paths, list):
            read_files.update(
                _normalise_lean_path(path)
                for path in paths
                if isinstance(path, str) and path.endswith(".lean")
            )
        return

    command = args.get("command") or args.get("cmd") or ""
    if isinstance(command, str) and _READ_COMMAND_RE.search(command):
        read_files.update(_explicit_lean_paths(command))


def count_files_read(
    traj: dict, *, include_prefix: str = ""
) -> int:
    """Count unique Lean files read, across Codex and Claude trace formats.

    Structured Read calls provide their path directly. Shell calls are counted
    when a content-reading command names a concrete .lean path. Calls duplicated
    in multiple trajectory representations are harmless because the final count
    is over canonical repository-relative paths.
    """
    read_files: set[str] = set()
    for message in traj.get("messages") or []:
        if not isinstance(message, dict):
            continue

        # Claude Code stores native tool calls as assistant content blocks.
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    _record_files_read(
                        block.get("name"), block.get("input"), read_files
                    )

        # OpenAI-compatible calls, including native Codex subscription traces.
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if isinstance(function, dict):
                _record_files_read(
                    function.get("name"), function.get("arguments"), read_files
                )

        # Some raw Codex traces retain only the completed item envelope.
        item = message.get("codex_item")
        if isinstance(item, dict) and item.get("type") == "command_execution":
            _record_files_read(
                "shell", {"command": item.get("command")}, read_files
            )

        # Older Codex/proxy trajectories flatten tool calls into messages.
        if message.get("attempt_message_type") == "tool_call":
            _record_files_read(
                message.get("name"),
                message.get("arguments") or message.get("input"),
                read_files,
            )

    # Older Claude proxy trajectories retain calls under model responses.
    for name, args in _claude_tool_calls(traj):
        _record_files_read(name, args, read_files)

    read_files = {
        path for path in read_files if _is_repository_lean_path(path)
    }
    if include_prefix:
        prefix = include_prefix.strip("/")
        read_files = {
            path for path in read_files
            if path == f"{prefix}.lean" or path.startswith(f"{prefix}/")
        }
    return len(read_files)


def count_git_commits(traj: dict) -> int:
    """Count explicit git commit calls made by the agent."""
    messages = traj.get("messages", [])
    count = 0
    for m in messages:
        if m.get("attempt_message_type") != "tool_call":
            continue
        cmd = _parse_cmd(m)
        if "git commit" in cmd and "--amend" not in cmd:
            count += 1
    return count


def _is_mcp_tool(name: str | None) -> bool:
    """True for lean-lsp-mcp tools (lean_goal, lean_diagnostic_messages, ...).

    The server exposes `lean_`-prefixed tools; codex/Claude may namespace them as
    `lean-lsp__*` or `mcp__lean*`. Match any form so the count is generator-agnostic."""
    if not name:
        return False
    n = name.lower()
    return n.startswith("lean_") or "lean-lsp" in n or n.startswith("mcp__lean")


def count_mcp_calls(traj: dict) -> dict:
    """Count lean-lsp MCP tool calls and their per-tool distribution.

    Returns ``{"total": int, "by_tool": {tool_name: count}}`` (by_tool ordered
    most-frequent first). Handles codex (`messages[].name`) and Claude (tool
    calls under `responses`) trajectory formats."""
    messages = traj.get("messages", [])
    has_codex = any(m.get("attempt_message_type") == "tool_call" for m in messages)
    dist: dict[str, int] = {}
    if has_codex:
        for m in messages:
            if m.get("attempt_message_type") == "tool_call" and _is_mcp_tool(m.get("name")):
                name = m.get("name")
                dist[name] = dist.get(name, 0) + 1
    else:
        for name, _args in _claude_tool_calls(traj):
            if _is_mcp_tool(name):
                dist[name] = dist.get(name, 0) + 1
    by_tool = dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"total": sum(dist.values()), "by_tool": by_tool}


# lean4-skills wrappers are deterministic `lean4-skills-*` binaries put on PATH
# (see configs.agent_augmentation). Unlike MCP tools they aren't distinct tool
# calls — the agent runs them through the shell — so a "skill call" is a shell
# command that invokes one, and a piped command may invoke several.
_SKILLS_WRAPPER_RE = re.compile(r"\blean4-skills-[a-z0-9]+(?:-[a-z0-9]+)*")


def _skills_in_cmd(cmd: str | None) -> list[str]:
    """Return each `lean4-skills-*` wrapper invoked in a shell command."""
    if not cmd:
        return []
    return _SKILLS_WRAPPER_RE.findall(cmd)


def count_skills_calls(traj: dict) -> dict:
    """Count lean4-skills wrapper invocations and their per-skill distribution.

    Returns ``{"total": int, "by_tool": {wrapper_name: count}}`` (by_tool ordered
    most-frequent first). Scans shell command text across codex
    (`messages[].name`) and Claude (tool calls under `responses`) trajectory
    formats, mirroring :func:`count_mcp_calls`."""
    messages = traj.get("messages", [])
    has_codex = any(m.get("attempt_message_type") == "tool_call" for m in messages)
    dist: dict[str, int] = {}
    if has_codex:
        for m in messages:
            if m.get("attempt_message_type") == "tool_call":
                for w in _skills_in_cmd(_parse_cmd(m)):
                    dist[w] = dist.get(w, 0) + 1
    else:
        for _name, args in _claude_tool_calls(traj):
            cmd = args.get("command") or args.get("cmd") or ""
            for w in _skills_in_cmd(cmd if isinstance(cmd, str) else ""):
                dist[w] = dist.get(w, 0) + 1
    by_tool = dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))
    return {"total": sum(dist.values()), "by_tool": by_tool}


# --- Edit-mechanism classification -----------------------------------------
#
# How a .lean file gets rewritten splits into two camps, named the SAME way
# across generators so codex and Claude are comparable:
#   * "edit_tool" — a structured edit whose replaced text is in the call itself,
#     so its net line/word delta is computable. This unifies codex `apply_patch`
#     and Claude `Edit`/`MultiEdit`: they do the same thing, so they share a name.
#   * "mass_edit" — an opaque rewrite through the shell: `sed -i`/`perl -i`, a
#     redirect or `tee` into a repo .lean file, or a python script that writes
#     .lean files. The command text doesn't reveal how many lines/words moved.
# The edit-tool compression is measured directly; the mass-edit share is the
# run-level residual (words_saved minus the edit-tool share), since words_saved
# is cumulative across rounds (see analyze_round).

_EDIT_TOOL_NAMES = {"Edit", "MultiEdit", "str_replace", "str_replace_editor", "replace"}
_SHELL_TOOL_NAMES = {"Bash", "shell", "run_shell_command", "exec_command"}

_SED_INPLACE_RE = re.compile(r"\b(?:g?sed|perl)\b[^|&;]*?\s-[a-zA-Z]*i\b")
# Redirect/tee target ending in .lean; we capture the path to drop /tmp scratch.
_REDIRECT_LEAN_RE = re.compile(r">>?\s*(\S*\.lean)\b")
_TEE_LEAN_RE = re.compile(r"\btee\b[^|&;]*?(\S*\.lean)\b")
_SCRATCH_PATH_RE = re.compile(r"(?:^|/)(?:tmp|dev|var/tmp)/")
_PY_INVOKE_RE = re.compile(r"\bpython[0-9.]*\b")
_PY_WRITE_RE = re.compile(
    r"write_text|\.write\(|open\([^)]*['\"](?:w|a)|os\.rename|os\.replace|shutil\."
)
_LEAN_PATH_RE = re.compile(r"[\w./\-]+\.lean\b")
_APPLY_PATCH_FILE_RE = re.compile(r"\*\*\* (?:Update|Add|Delete) File: (\S+)")


def _lean_basenames(text: str) -> set[str]:
    return {m.split("/")[-1] for m in _LEAN_PATH_RE.findall(text)}


def _patch_text_delta(patch_text: str) -> tuple[int, int]:
    """(lines_removed - added, words_removed - added) for a unified/codex patch.

    Positive => the edit shrank the file. `import` lines are excluded from the
    word count to match the compression metric (which strips imports)."""
    removed_lines = added_lines = removed_words = added_words = 0
    for line in patch_text.splitlines():
        if line[:3] in ("+++", "---") or line.startswith(("***", "@@", "diff ")):
            continue
        if line.startswith("+"):
            body = line[1:]
            added_lines += 1
            if not body.lstrip().startswith("import "):
                added_words += len(body.split())
        elif line.startswith("-"):
            body = line[1:]
            removed_lines += 1
            if not body.lstrip().startswith("import "):
                removed_words += len(body.split())
    return removed_lines - added_lines, removed_words - added_words


def _edit_args_delta(args: dict) -> tuple[int, int]:
    """(lines_removed - added, words_removed - added) for an Edit/MultiEdit call."""
    pairs: list[tuple[str, str]] = []
    if "old_string" in args or "new_string" in args:
        pairs.append((args.get("old_string") or "", args.get("new_string") or ""))
    for edit in args.get("edits") or []:
        if isinstance(edit, dict):
            pairs.append((edit.get("old_string") or "", edit.get("new_string") or ""))
    rl = al = rw = aw = 0
    for old, new in pairs:
        rl += len(old.splitlines())
        al += len(new.splitlines())
        rw += len(old.split())
        aw += len(new.split())
    return rl - al, rw - aw


def _classify_shell_edit(cmd: str, counts: dict, lean_files: set[str]) -> None:
    """Bucket a shell command into a mass-edit mechanism (or ignore it).

    Redirects/tee into a scratch path (/tmp, /dev) are excluded — agents write
    throwaway snippets there to type-check, which isn't editing the repo."""
    if not cmd:
        return
    if _SED_INPLACE_RE.search(cmd) and ".lean" in cmd:
        counts["sed_inplace"] += 1
        lean_files |= _lean_basenames(cmd)
        return
    targets = _REDIRECT_LEAN_RE.findall(cmd) + _TEE_LEAN_RE.findall(cmd)
    repo_targets = [t for t in targets if not _SCRATCH_PATH_RE.search(t)]
    if repo_targets:
        counts["redirect"] += 1
        lean_files |= {t.split("/")[-1] for t in repo_targets}
        return
    if _PY_INVOKE_RE.search(cmd) and _PY_WRITE_RE.search(cmd):
        counts["python_write"] += 1


def _claude_tool_calls(traj: dict):
    """Yield (function_name, args_dict) for Claude-format tool calls (deduped)."""
    all_responses: list = []
    for attempt in (traj.get("info") or {}).get("attempts") or []:
        all_responses.extend(attempt.get("responses") or [])
    if not all_responses:
        all_responses = traj.get("responses") or []
    seen: set = set()
    for r in all_responses:
        rid = r.get("id")
        if rid and rid in seen:
            continue
        if rid:
            seen.add(rid)
        for choice in r.get("choices") or []:
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw = fn.get("arguments", "")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    args = {}
                yield name, (args if isinstance(args, dict) else {})


def classify_edits(traj: dict) -> dict:
    """Split .lean editing into measurable edit-tool calls vs opaque mass edits.

    Handles both codex (`apply_patch` custom_tool_call + `exec_command` shell)
    and Claude (`Edit`/`MultiEdit` + `Bash`) trajectory formats. ``*_compression``
    fields are net line/word reductions measured directly from the edit-tool
    calls; mass-edit compression is left as a run-level residual (see module note).
    """
    # apply_patch and Edit/MultiEdit both feed the unified "edit_tool" bucket so
    # the metric is comparable across codex and Claude.
    counts = {
        "apply_patch": 0,   # codex structured edit  -> edit_tool
        "edit": 0,          # Claude Edit/MultiEdit   -> edit_tool
        "sed_inplace": 0,   # \
        "redirect": 0,      #  } mass_edit
        "python_write": 0,  # /
    }
    edit_line_compression = 0
    edit_word_compression = 0
    lean_files: set[str] = set()

    messages = traj.get("messages", [])
    for m in messages:
        if m.get("name") == "apply_patch":
            patch = m.get("input") or m.get("arguments") or ""
            if isinstance(patch, str) and patch:
                counts["apply_patch"] += 1
                dl, dw = _patch_text_delta(patch)
                edit_line_compression += dl
                edit_word_compression += dw
                for fn in _APPLY_PATCH_FILE_RE.findall(patch):
                    lean_files.add(fn.split("/")[-1])
        elif m.get("attempt_message_type") == "tool_call":
            _classify_shell_edit(_parse_cmd(m), counts, lean_files)

    # Claude format: structured edits + Bash live in responses, not messages.
    for name, args in _claude_tool_calls(traj):
        if name in _EDIT_TOOL_NAMES:
            path = str(args.get("file_path") or args.get("path") or "")
            if not path.endswith(".lean"):
                continue
            counts["edit"] += 1
            lean_files.add(path.split("/")[-1])
            dl, dw = _edit_args_delta(args)
            edit_line_compression += dl
            edit_word_compression += dw
        elif name in _SHELL_TOOL_NAMES:
            cmd = args.get("command") or args.get("cmd") or ""
            _classify_shell_edit(cmd if isinstance(cmd, str) else "", counts, lean_files)

    edit_tool_calls = counts["apply_patch"] + counts["edit"]
    mass_edit_calls = counts["sed_inplace"] + counts["redirect"] + counts["python_write"]
    return {
        # Comparable cross-generator headline: one name for the structured edit
        # tool (codex apply_patch + Claude Edit), one for shell/mass edits.
        "edit_tool_calls": edit_tool_calls,
        "mass_edit_calls": mass_edit_calls,
        "edit_tool_line_compression": edit_line_compression,
        "edit_tool_word_compression": edit_word_compression,
        "lean_files_edited": sorted(lean_files),
        # Per-mechanism detail (which concrete tool/shell idiom did the editing).
        "by_mechanism": {
            "apply_patch": counts["apply_patch"],
            "edit": counts["edit"],
            "sed_inplace": counts["sed_inplace"],
            "redirect": counts["redirect"],
            "python_write": counts["python_write"],
        },
    }
