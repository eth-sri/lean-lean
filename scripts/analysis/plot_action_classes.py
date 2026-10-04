#!/usr/bin/env python3
"""Ten-class, mutually exclusive agent-action counts (the paper's agent-actions figure).

Every tool call in a trajectory's standardized trace (Muse: its native event
stream) gets exactly one class, by priority
Edit > Verify > Measure > Git > Build > Lake > Search > Read > Sleep > Other. Edit is strict and takes precedence: a call
is an Edit only when it writes, moves or deletes a file of the agent's starting
repository, or creates a new Lean module in one of its source directories.
Scratch files (new files at the repository root, under /tmp or in new helper
directories), caches and logs do not count. Bookkeeping, polling, reasoning and
delegation calls are not counted at all.

Usage:
    python scripts/analysis/plot_action_classes.py \\
        output/evaluation/leanlean_20260914/anthropic/opus-5-high [MORE_MODEL_DIRS...] \\
        --dataset datasets/leanlean_20260914 --output-dir output/analysis/agent-actions

Writes action-repositories.csv (counts per trajectory), agent-actions.csv (mean
per trajectory and class, per model), actions.csv (every classified call) and
agent-actions.pdf/png.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
import fnmatch
from functools import lru_cache
import itertools
import json
from pathlib import Path
import posixpath
import re
import shlex
from statistics import mean
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

CATEGORIES = ["Build", "Verify", "Measure", "Read", "Search", "Edit", "Git", "Lake", "Sleep", "Other"]
COLORS = ["#167D8D", "#97C86B", "#4DBBD5", "#4C78A8", "#8F6BB3",
          "#E45756", "#E6B840", "#54A24B", "#D58AAB", "#B7B7BB"]
# The paper's table column order.
TABLE_ORDER = ["Read", "Search", "Edit", "Git", "Lake", "Build", "Measure", "Verify", "Sleep", "Other"]
PRIORITY = ["Edit", "Verify", "Measure", "Git", "Build", "Lake", "Search", "Read", "Sleep", "Other"]
DIRECT = {
    "read": "Read", "read_file": "Read", "view_file": "Read",
    "grep": "Search", "glob": "Search", "grep_search": "Search",
    "find_by_name": "Search", "search": "Search", "search_files": "Search",
    "find_files": "Search", "list_dir": "Read",
    "edit": "Edit", "edit_file": "Edit", "multiedit": "Edit", "write": "Edit", "write_file": "Edit",
    "write_to_file": "Edit", "replace_file_content": "Edit",
    "multi_replace_file_content": "Edit", "apply_patch": "Edit",
    "file_change": "Edit", "filechange": "Edit",
    "lean_verify": "Verify", "proof_length": "Measure", "proof_length.py": "Measure",
}
WRAPPERS = {"time", "timeout", "env", "command", "exec", "nohup", "stdbuf", "nice",
            "sudo", "setsid", "do", "then", "else", "if", "!"}
# Every agent container mounts the repository here.
REPO_ROOT = "/testbed"
SEPARATORS = ";&|()\n"
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?([^\n]*)\n(.*?)\n\1\b", flags=re.S)
PATCH_PATHS = re.compile(r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", flags=re.M)
# Edit targets that only exist at run time (shell variables, computed Python paths).
UNRESOLVED = Counter()


def repository_files(report):
    """Files of the agent's starting repository, from its pinned prod-strip report.

    Lean sources after module pruning, retained assets and Lake metadata. This covers
    the whole stripped tree of every leanlean_20260914 repository (.lake excluded).
    """
    provenance = report["protected_provenance"]
    closure = provenance["import_closure"]
    files = set(closure["retained_local_sources"]) | set(closure.get("retained_local_assets") or [])
    files -= set(report["module_pruning"].get("deleted_files") or [])
    files |= {provenance["solution_source"], closure["lake_metadata"]["path"],
              "lean-toolchain", "lake-manifest.json", ".gitignore"}
    return frozenset(files)

@lru_cache(maxsize=None)
def starting_repository(dataset, repository):
    """The repository's file set, from its prod-strip report or, when the report
    carries no import closure (lean-strip preprocessing and the Hugging Face
    release, scripts/fetch_dataset.py), from its stripped tree."""
    directory = Path(dataset) / "repos" / repository
    report = json.loads((directory / "prod-strip-report.json").read_text())
    if "import_closure" in report["protected_provenance"]:
        return repository_files(report)
    tree = directory / "stripped"
    return frozenset(p.relative_to(tree).as_posix() for p in tree.rglob("*")
                     if p.is_file() and ".lake" not in p.relative_to(tree).parts)


@lru_cache(maxsize=256)
def _directories(repo):
    return frozenset("/".join(f.split("/")[:k]) for f in repo for k in range(1, f.count("/") + 1))


def repository_target(path, cwd, repo):
    """'file', 'dir' or 'root' when a path names part of the starting repository, else None."""
    if not path or "$" in path or "`" in path or (cwd is None and not path.startswith("/")):
        return None
    full = posixpath.normpath(posixpath.join(cwd or "/", path))
    if full == REPO_ROOT:
        return "root"
    if not full.startswith(REPO_ROOT + "/"):
        return None
    relative = full[len(REPO_ROOT) + 1:]
    if any(c in relative for c in "*?["):
        return "file" if any(fnmatch.fnmatchcase(f, relative) for f in repo) else None
    if relative in repo:
        return "file"
    if relative in _directories(repo):
        return "dir"
    # A new module beside existing sources; new root-level files are scratch.
    parent = posixpath.dirname(relative)
    return "file" if relative.endswith(".lean") and parent and parent in _directories(repo) else None


def _runtime_path(path, cwd=REPO_ROOT):
    """A path only known at run time that might still lie inside the repository."""
    if not path or ("$" not in path and "`" not in path):
        return False
    static = re.split(r"[$`]", path)[0]
    if not static.startswith("/") and cwd is not None:
        static = posixpath.join(cwd, static)
    return not static.startswith("/") or static.startswith(REPO_ROOT) or REPO_ROOT.startswith(static)


def python_categories(source, cwd=REPO_ROOT, repo=frozenset()):
    """Recognize explicit file reads/writes in Python, not words in strings."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    found, targets, names, constants, walks = set(), [], {}, [], False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            names[node.targets[0].id] = node.value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" not in node.value \
                and len(node.value) < 512:
            constants.append(node.value)

    def value(expr, depth=0):
        """Statically known path expressions: literals, names bound once, Path() and / joins."""
        if depth > 4:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Name) and expr.id in names:
            return value(names[expr.id], depth + 1)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
            left, right = value(expr.left, depth + 1), value(expr.right, depth + 1)
            return None if left is None or right is None else posixpath.join(left, right)
        if isinstance(expr, ast.Call):
            fn = expr.func
            called = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if called in {"Path", "PurePath", "PosixPath", "str"} and len(expr.args) == 1:
                return value(expr.args[0], depth + 1)
            if called == "join" and isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) \
                    and fn.value.attr == "path":
                parts = [value(a, depth + 1) for a in expr.args]
                return None if not parts or None in parts else posixpath.join(*parts)
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
        owner = fn.value.id if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) else ""
        if name in {"glob", "rglob", "iglob", "walk", "listdir", "scandir"}:
            walks = True
        if owner in {"os", "shutil"}:
            # The removed path, or the destination of a copy, move or rename.
            positions = {"remove": [0], "unlink": [0], "rmtree": [0], "replace": [0, 1], "rename": [0, 1],
                         "move": [0, 1], "copy": [1], "copy2": [1], "copyfile": [1], "copytree": [1]}.get(name)
            if positions:
                targets += [value(node.args[k]) for k in positions if k < len(node.args)]
            continue
        # str.replace alone is not a filesystem modification.
        if name in {"write_text", "write_bytes", "unlink", "rename", "touch"} and isinstance(fn, ast.Attribute):
            targets.append(value(fn.value))
        if name in {"read_text", "read_bytes"}:
            found.add("Read")
        if name == "open":
            mode = next((k.value for k in node.keywords if k.arg == "mode"), None)
            if mode is None and isinstance(fn, ast.Name) and len(node.args) > 1:
                mode = node.args[1]
            if (mode is None and isinstance(fn, ast.Attribute) and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and re.fullmatch(r"[rwaxbt+]+", node.args[0].value)):
                mode = node.args[0]
            writes = isinstance(mode, ast.Constant) and isinstance(mode.value, str) \
                and any(c in mode.value for c in "wax+")
            if not writes:
                found.add("Read")
            elif isinstance(fn, ast.Attribute):
                targets.append(value(fn.value))
            else:
                targets.append(value(node.args[0]) if node.args else None)
    if targets:
        kinds = [repository_target(t, cwd, repo) for t in targets if t is not None]
        if "file" in kinds:
            found.add("Edit")
        elif None in targets:
            # A computed target: count it when the script names a repository file,
            # or walks a repository directory.
            named = [repository_target(c, cwd, repo) for c in constants]
            if "file" in named or (walks and ({"dir", "root"} & set(named))):
                found.add("Edit")
            else:
                UNRESOLVED["python"] += 1
    return found


def redirect_targets(segment):
    """Files written by shell output redirection (>, >>, 1>, 2>, &>)."""
    targets = []
    for k, word in enumerate(segment):
        match = re.fullmatch(r"(?:\d|&)?>>?\|?(.*)", word)
        if not match:
            continue
        # `2>&1` reaches here as `2>` at the end of a segment: & separates words.
        target = match.group(1) or (segment[k + 1] if k + 1 < len(segment) else "")
        if target and not target.startswith("&"):
            targets.append(target)
    return targets


def positional(arguments, consumes=()):
    """Non-option arguments, skipping the values of options in `consumes`."""
    out, skip = [], False
    for word in arguments:
        if skip:
            skip = False
        elif word.startswith("-") and word != "-":
            skip = word in consumes
        elif not re.fullmatch(r"(?:\d|&)?>>?\|?.*", word):
            out.append(word)
    return out


def shell_categories(command, repo=frozenset(), cwd=REPO_ROOT, bodies=None):
    categories = set()
    # Shared with nested `bash -c` scripts, which may carry the outer placeholders.
    bodies = [] if bodies is None else bodies
    # Unwrap a whole-command `bash -lc SCRIPT` (Codex wraps every command) first, so
    # heredoc payloads are read unescaped rather than with the wrapper's \" escapes.
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = []
    if (len(parts) == 3 and parts[0].rsplit("/", 1)[-1] in {"bash", "sh", "zsh"}
            and re.fullmatch(r"-[a-z]*c[a-z]*", parts[1])):
        return shell_categories(parts[2], repo, cwd, bodies)

    def placeholder(match):
        bodies.append(match.group(3))
        return f" __HEREDOC{len(bodies) - 1}__ {match.group(2)}"
    # Heredoc payloads are data, not executed shell; keep the rest of the opening line.
    cleaned = HEREDOC.sub(placeholder, command)
    try:
        lex = shlex.shlex(cleaned, posix=True, punctuation_chars=SEPARATORS)
        lex.whitespace = " \t\r"
        lex.whitespace_split = True
        words = list(lex)
    except ValueError:
        return categories or {"Other"}

    def separator(word):
        return bool(word) and all(c in SEPARATORS for c in word)

    def heredoc(segment):
        return next((bodies[int(m.group(1))] for w in segment
                     if (m := re.fullmatch(r"__HEREDOC(\d+)__", w))), None)

    def written(paths, name):
        for path in paths:
            if _runtime_path(path, cwd):
                UNRESOLVED["shell"] += 1
                continue
            kind = repository_target(path, cwd, repo)
            # Removing or moving a whole repository directory also edits it.
            if kind == "file" or (kind == "dir" and name in {"rm", "mv", "rmdir"}):
                categories.add("Edit")

    start = 0
    for i, word in enumerate(words):
        if separator(word):
            start = i + 1
            continue
        prefix = words[start:i]
        if not all(w in WRAPPERS or w.startswith("-") or "=" in w or
                   re.fullmatch(r"[0-9.]+[smhd]?", w) for w in prefix):
            continue
        name = word.rsplit("/", 1)[-1]
        rest = words[i + 1:]
        segment = list(itertools.takewhile(lambda w: not separator(w), rest))
        written(redirect_targets(segment), name)
        if name == "cd":
            target = segment[0] if segment else "~"
            cwd = (None if "$" in target or "`" in target or target in {"-", "~"}
                   or (cwd is None and not target.startswith("/"))
                   else posixpath.normpath(posixpath.join(cwd or "/", target)))
        elif name in {"bash", "sh", "zsh"}:
            for k in range(min(3, len(rest)-1)):
                if rest[k].startswith("-") and "c" in rest[k]:
                    categories.update(shell_categories(rest[k+1], repo, cwd, bodies))
        elif name == "sleep":
            categories.add("Sleep")
        elif name == "lean_verify":
            categories.add("Verify")
        elif name in {"proof_length", "proof_length.py"}:
            categories.add("Measure")
        elif re.fullmatch(r"python(?:3(?:\.\d+)?)?", name) and rest and rest[0].rsplit("/", 1)[-1] == "proof_length.py":
            categories.add("Measure")
        elif re.fullmatch(r"python(?:3(?:\.\d+)?)?", name) and "-c" in rest[:3]:
            k = rest.index("-c")
            if k + 1 < len(rest):
                categories.update(python_categories(rest[k + 1], cwd, repo))
        elif re.fullmatch(r"python(?:3(?:\.\d+)?)?", name) and heredoc(segment) is not None:
            categories.update(python_categories(heredoc(segment), cwd, repo))
        elif name == "perl" and any(re.match(r"-[a-zA-Z0-9]*i", x) for x in rest[:3]):
            written(positional(segment, consumes={"-e", "-E"})[0 if any(w in segment for w in ("-e", "-E")) else 1:], name)
        elif name == "git":
            categories.add("Git")
        elif name == "lake":
            args = list(rest)
            while args and args[0].startswith("-"):
                args.pop(0)
            categories.add("Build" if args and args[0] == "build" else "Lake")
        elif name in {"rg", "grep", "egrep", "fgrep", "find", "fd"}:
            categories.add("Search")
        elif name in {"cat", "head", "tail", "less", "more", "nl", "ls"}:
            categories.add("Read")
        elif name == "cp":
            paths = positional(segment, consumes={"-t", "--target-directory"})
            target = next((segment[k + 1] for k, w in enumerate(segment[:-1]) if w in {"-t", "--target-directory"}), None)
            written([target] if target else paths[-1:] if len(paths) > 1 else [], name)
        elif name in {"mv", "rm", "rmdir", "touch", "tee", "truncate", "unlink"}:
            written(positional(segment, consumes={"-s", "--size"}), name)
        elif name == "apply_patch":
            patch = heredoc(segment) or (segment[0] if segment else "")
            written(PATCH_PATHS.findall(patch), name)
        elif name == "sed":
            if any(re.match(r"-[a-zA-Z]*i|--in-place", x) for x in segment):
                scripts = any(w in segment for w in ("-e", "-f", "--expression", "--file"))
                written(positional(segment, consumes={"-e", "-f", "--expression", "--file"})[0 if scripts else 1:], name)
            else:
                categories.add("Read")
    return categories or {"Other"}


def edit_tool_targets(args):
    """Paths named by a dedicated edit tool's arguments."""
    paths = [args[k] for k in ("file_path", "TargetFile", "path", "target_file", "filePath")
             if isinstance(args.get(k), str)]
    changes = args.get("changes")
    if isinstance(changes, str):
        try:
            changes = ast.literal_eval(changes)
        except (ValueError, SyntaxError):
            changes = []
    if isinstance(changes, list):
        paths += [c[k] for c in changes if isinstance(c, dict) for k in ("path", "move_path")
                  if isinstance(c.get(k), str)]
    for key in ("patch", "input"):
        if isinstance(args.get(key), str):
            paths += PATCH_PATHS.findall(args[key])
    return paths


def classify(name, args, repo):
    """Category and every matching class of one tool call against its starting repository."""
    lower = name.lower()
    if lower in DIRECT:
        category = DIRECT[lower]
        if category != "Edit":
            return category, {category}
        targets = edit_tool_targets(args)
        if not targets:
            # The trace did not record the path; a dedicated edit tool still edited a file.
            UNRESOLVED["edit_tool"] += 1
            return "Edit", {"Edit"}
        if any(repository_target(t, REPO_ROOT, repo) == "file" for t in targets):
            return "Edit", {"Edit"}
        return "Other", {"Other"}
    command = args.get("command", args.get("cmd", args.get("CommandLine", "")))
    if command:
        cwd = args.get("cwd") if isinstance(args.get("cwd"), str) else REPO_ROOT
        categories = shell_categories(command, repo, cwd)
        return next(c for c in PRIORITY if c in categories), categories
    if lower in {"bash", "run_command", "shell", "run_shell_command"}:
        return "Other", {"Other"}
    # Bookkeeping, polling, reasoning, and delegation are outside the ten
    # repository-operation classes, whose Other definition is shell-only.
    return None, set()


def actions(directory):
    for path in sorted((directory / "playback").glob("capture_*/standardized-trace.json")):
        trace = json.loads(path.read_text())
        if trace.get("provider") == "generic-jsonl":
            seen = set()
            for event in json_lines(path.parent / "native.stdout.jsonl"):
                if event.get("payload_type") != "tool.result":
                    continue
                p = event["payload"]
                identity = p.get("call_id")
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                name = p.get("correlation_facts", {}).get("tool_name", "")
                try:
                    args = json.loads(p.get("text", ""))
                except (ValueError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                # Edit tools report their path only in the result's edit facts.
                edited = (p.get("edit_facts") or {}).get("path")
                if not args and isinstance(edited, str):
                    args = {"path": edited}
                yield path.parent.name, identity, name, args
        else:
            seen = set()
            for action in trace.get("actions", []):
                identity = str(action.get("tool_call_id") or action.get("action_index"))
                if identity in seen:
                    continue
                seen.add(identity)
                args = action.get("arguments") or {}
                yield path.parent.name, identity, action.get("name", ""), args


def json_lines(path):
    if path.exists():
        for line in path.open():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_classes(runs, labels, dataset, output):
    """Class counts for every trajectory under each model directory, and per-model means."""
    rows, audit, summary = [], [], []
    for run, label in zip(runs, labels):
        cohort = []
        for directory in sorted(d for d in run.iterdir() if (d / "playback").is_dir()):
            repo = starting_repository(dataset, directory.name)
            counts = Counter()
            for capture, identity, name, args in actions(directory):
                category, matches = classify(name, args, repo)
                if category is None:
                    continue
                counts[category] += 1
                audit.append(dict(model=run.name, repository=directory.name, capture=capture,
                                  action_id=identity, tool=name, category=category,
                                  matched_categories="|".join(c for c in CATEGORIES if c in matches),
                                  arguments=json.dumps(args, ensure_ascii=False)))
            if not counts:
                print(f"Skipping {directory}: no classified actions", flush=True)
                continue
            cohort.append(dict(model=run.name, repository=directory.name, Actions=sum(counts.values()),
                               **{c: counts[c] for c in CATEGORIES}, source=str(directory)))
        if not cohort:
            raise ValueError(f"No classified trajectories: {run}")
        rows += cohort
        summary.append(dict(model=run.name, label=label, trajectories=len(cohort),
                            **{c: mean(r[c] for r in cohort) for c in TABLE_ORDER},
                            total=mean(r["Actions"] for r in cohort)))
    write_csv(output / "action-repositories.csv", rows)
    write_csv(output / "agent-actions.csv", summary)
    write_csv(output / "actions.csv", audit)
    print("Unresolved edit targets:", dict(UNRESOLVED), flush=True)
    return summary


def render(summary, output):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}):
        # Tight page bounds, with room for model labels and a two-row legend.
        fig, ax = plt.subplots(figsize=(182 / 72, 142 / 72))
        bottom = np.zeros(len(summary))
        for category, color in zip(CATEGORIES, COLORS):
            values = np.array([float(r[category]) for r in summary])
            ax.bar(range(len(summary)), values, bottom=bottom, color=color, label=category, width=.65)
            bottom += values
        ax.set_xticks(range(len(summary)), [r["label"] for r in summary])
        ax.set_ylabel("Actions per trajectory", labelpad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.tick_params(pad=2, length=3)
        ax.set_ylim(0, max(bottom) * 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#777777")
        handles, labels = ax.get_legend_handles_labels()
        # Matplotlib fills columns; display the stack order across each row.
        order = [j for col in range(5) for j in range(col, len(handles), 5)]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower center", bbox_to_anchor=(.5, .01), ncol=5,
                   frameon=False, handlelength=.9, handletextpad=.35,
                   columnspacing=.5, labelspacing=.25, borderpad=0, borderaxespad=0)
        fig.subplots_adjust(left=33.52 / 182, right=180.04 / 182,
                            bottom=38.064 / 142, top=141.024 / 142)
        fig.savefig(output / "agent-actions.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output / "agent-actions.png", dpi=240)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="model directories, output/evaluation/<dataset>/<provider>/<model>")
    parser.add_argument("--labels", nargs="+", help="figure label per model directory (default: its name)")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/leanlean_20260914"),
                        help="dataset whose repos/<id>/ define each starting repository")
    parser.add_argument("--output-dir", type=Path, default=Path("output/analysis/agent-actions"))
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        parser.error("give one label per model directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = collect_classes(args.runs, labels, args.dataset, args.output_dir)
    render(summary, args.output_dir)
    print(json.dumps(summary, indent=2))
