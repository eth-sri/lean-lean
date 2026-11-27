"""Graph/diff serialization shared with the original website exporter.

The caller supplies a validated declaration ledger from the current classifier.
Baseline layout and node indices remain those already published by the website.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
import gzip
import json
import re
import subprocess
import tempfile

DATASET = None


KINDS = ["theorem", "def", "inductive", "ctor", "rec", "axiom", "opaque", "instance", "other", "syntax"]

STATUS = {"unchanged": 0, "rewritten": 1, "structural": 2, "deleted": 3, "dead": 4, "comments": 5}

DEFINING = {"def", "abbrev", "instance", "opaque", "structure", "class", "inductive", "irreducible_def", "alias",
            "deriving", "theorem", "lemma"}

_PREAMBLE = re.compile(r"\s+|@\[[^\]]*\]|(?:private|protected|public|meta|noncomputable|partial|unsafe|nonrec|local|mutual)\b|scoped\b(?:\[[^\]]*\])?"
                       r"|(?:set_option\s+\S+\s+\S+|open\b[^\n]*?|omit\b.*?|include\b.*?)\s+in\b", re.S)

def written_kind(kind: str, text: str) -> str:
    """The kernel kind, except definitions a command other than a definition wrote are "syntax"."""
    from leanlean.metrics.tokens import remove_lean_comments
    if kind not in ("def", "opaque") or not text.strip():
        return kind
    code, pos = remove_lean_comments(text), 0
    while (m := _PREAMBLE.match(code, pos)) and m.end() > pos:
        pos = m.end()
    word = re.match(r"[\w']+", code[pos:])
    return kind if word and word.group(0) in DEFINING else "syntax"

def kind_index(kind: str, text: str) -> int:
    kind = written_kind(kind, text)
    return KINDS.index(kind) if kind in KINDS else KINDS.index("other")

def dump(path: Path, payload) -> None:
    """Write JSON; a `.json.gz` path is gzipped (the browser unpacks it, see util.js `load`)."""
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    path.write_bytes(gzip.compress(text, compresslevel=9, mtime=0) if path.suffix == ".gz" else text)

def owners(nodes: list[dict], path_of) -> tuple[list[int], list[int]]:
    """Map every node to the outermost node whose source range contains it."""
    by_file = defaultdict(list)
    for index, node in enumerate(nodes):
        path = path_of(node["module"])
        if path is not None and node.get("start_line") is not None:
            by_file[path].append(index)
    owner = list(range(len(nodes)))
    for ids in by_file.values():
        # Outermost first; among identical ranges the shortest (parent) name.
        ids.sort(key=lambda i: (nodes[i]["start_line"], -nodes[i]["end_line"],
                                len(nodes[i]["name"]), nodes[i]["name"]))
        current = None
        for index in ids:
            if current is not None and nodes[index]["end_line"] <= nodes[current]["end_line"]:
                owner[index] = current
            else:
                current = index
    roots = sorted({owner[i] for i in range(len(nodes))})
    return owner, roots

NODE_SPACING = 18.0

LANE_SPACING = 34.0

SECTION_GAP = 26.0

LEFT_MARGIN = 150.0

MICRO_OFFSETS = (0.0, -6.0, 6.0, -3.0, 3.0)

def harness_files(repo: str) -> dict[str, str]:
    """The files the harness writes into the agent's container: the challenge and comparator.json.

    Neither counts towards the score; the viewer lists them, greyed out, among all files.
    """
    folder = DATASET / "repos" / repo / "comparator"
    manifest = json.loads((folder / "manifest.json").read_text())
    return {manifest["challenge"]["source_path"]: (folder / manifest["challenge"]["path"]).read_text(encoding="utf-8"),
            "comparator.json": (folder / manifest["configuration"]["path"]).read_text(encoding="utf-8")}

def topological(count: int, edges: list[tuple[int, int]]) -> list[int]:
    """Users before their dependencies (Kahn); declarations on a cycle follow in index order."""
    indegree, out = [0] * count, defaultdict(list)
    for a, b in edges:
        out[a].append(b)
        indegree[b] += 1
    queue = [v for v in range(count) if not indegree[v]]
    order = []
    while queue:
        v = queue.pop()
        order.append(v)
        for w in out[v]:
            indegree[w] -= 1
            if not indegree[w]:
                queue.append(w)
    seen = set(order)
    return order + [v for v in range(count) if v not in seen]

def place_added(added: list[dict], edges: set, base: dict) -> tuple[list, list]:
    """Place the agent's added declarations in the baseline's layered layout.

    A declaration joins new lanes opened just below the dependency layer of its
    lowest user (one lane group deeper for each added declaration in between),
    as the layout would rank it; one nothing uses goes just above its highest
    dependency.  Each lane sits its declarations near the mean x of the
    declarations they touch.  Baseline lanes below an insertion move down to
    make room: `shift` lists [y, dy] steps, every baseline y beyond y moving
    down by dy.  Declarations touching nothing keep their own lanes at the
    bottom.  Sets x and y of every added declaration; returns (lanes, shift).
    """
    positions, lanes, lane_of = base["positions"], base["lanes"], base["lane_of"]
    world_w, world_h = base["world"]
    per_lane = base["per_lane"] or 24
    layer_of_lane, layer = [], -1
    for lane in lanes:
        layer += bool(lane[1])
        layer_of_lane.append(layer)
    last_lane = {layer: index for index, layer in enumerate(layer_of_lane)}
    users, deps, inner = defaultdict(list), defaultdict(list), []
    for (ku, u), (kd, d) in edges:
        if kd == "a":
            users[d].append((ku, u))
        if ku == "a":
            deps[u].append((kd, d))
        if ku == kd == "a":
            inner.append((u, d))
    order = topological(len(added), inner)
    # Slot (layer, depth): the lanes after a baseline layer's last lane, depth-th group of them.
    slot = {}
    for k in order:
        below = [(layer_of_lane[lane_of[v]], 0) if kind == "b" else (slot[v][0], slot[v][1] + 1)
                 for kind, v in users[k] if kind == "b" or v in slot]
        if below:
            slot[k] = max(below)
    for k in reversed(order):
        if k in slot:
            continue
        above = [(max(layer_of_lane[lane_of[v]] - 1, 0), 0) if kind == "b" else (slot[v][0], max(slot[v][1] - 1, 0))
                 for kind, v in deps[k] if kind == "b" or v in slot]
        if above:
            slot[k] = min(above)
    # Where in x: the mean of the baseline declarations touched, else of the added ones placed so far.
    target = {}
    for sweep in (order, list(reversed(order))):
        for k in sweep:
            if k in target:
                continue
            xs = [positions[v][0] if kind == "b" else target[v] for kind, v in users[k] + deps[k]
                  if kind == "b" or v in target]
            if xs:
                target[k] = sum(xs) / len(xs)
    left, right = LEFT_MARGIN, max(LEFT_MARGIN, world_w - 40.0)

    def row(members: list[int], y: float, lane_index: int) -> None:
        # Pack in x order NODE_SPACING apart, as close to the targets as the spacing allows.
        members.sort(key=lambda k: (target.get(k, (left + right) / 2), k))
        want = [target.get(k, (left + right) / 2) for k in members]
        xs = [want[0]]
        for w in want[1:]:
            xs.append(max(w, xs[-1] + NODE_SPACING))
        drift = sum(w - x for w, x in zip(want, xs)) / len(xs)
        drift = min(max(drift, left - xs[0]), max(right - xs[-1], left - xs[0]))
        for slot_index, (k, x) in enumerate(zip(members, xs)):
            added[k]["x"] = round(x + drift, 1)
            added[k]["y"] = round(y + MICRO_OFFSETS[(slot_index + lane_index * 2) % len(MICRO_OFFSETS)], 1)

    groups = defaultdict(list)
    for k, s in slot.items():
        groups[s].append(k)
    inserted = defaultdict(list)  # layer -> lanes of members, in depth order
    for (layer, _depth), members in sorted(groups.items()):
        members.sort(key=lambda k: (target.get(k, 0.0), k))
        inserted[layer] += [members[o:o + per_lane] for o in range(0, len(members), per_lane)]
    out_lanes, shift, dy = [], [], 0.0
    for layer, groups_ in sorted(inserted.items()):
        last_y = lanes[last_lane[layer]][0]
        for index, members in enumerate(groups_, 1):
            y = last_y + dy + index * LANE_SPACING
            row(members, y, len(out_lanes))
            out_lanes.append([round(y, 1), "", "inserted", len(members)])
        dy += len(groups_) * LANE_SPACING
        shift.append([round(last_y + LANE_SPACING / 2, 1), round(dy, 1)])
    rest = [k for k in range(len(added)) if k not in slot]
    rest.sort(key=lambda k: (target.get(k, world_w / 2), k))
    y = world_h + dy + SECTION_GAP
    for index, offset in enumerate(range(0, len(rest), per_lane)):
        row(rest[offset:offset + per_lane], y, len(out_lanes))
        out_lanes.append([round(y, 1), "Added by the agent" if index == 0 else "", "added", len(rest[offset:offset + per_lane])])
        y += LANE_SPACING
    return out_lanes, shift

def git_path(name: str) -> str:
    """A path from a diff header: git appends a tab to names with spaces and C-quotes unusual ones."""
    name = name.rstrip("\t")
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape").encode("latin-1").decode("utf-8")
    return name

def git_diff(before: dict[str, str], after: dict[str, str]) -> dict[str, dict]:
    """Unified hunks per modified file, computed by git (histogram algorithm)."""
    changed = [path for path in before if path in after and before[path] != after[path]]
    result = {}
    if not changed:
        return result
    with tempfile.TemporaryDirectory() as tmp:
        for side, tree in (("a", before), ("b", after)):
            for path in changed:
                target = Path(tmp, side, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(tree[path], encoding="utf-8")
        proc = subprocess.run(["git", "-c", "core.quotePath=false", "diff", "--no-index", "--no-color", "--histogram", "-U3",
                               "--src-prefix=", "--dst-prefix=", "a", "b"],
                              cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace")
    current, hunk = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("diff --git"):
            current = None
        elif line.startswith("+++ "):
            current = git_path(line[4:]).removeprefix("b/")
            result[current] = {"hunks": [], "add": 0, "del": 0}
        elif line.startswith("@@") and current:
            head = line.split("@@")[1].strip()
            old, new = head.split(" ")
            o = [int(v) for v in (old[1:].split(",") + ["1"])[:2]]
            n = [int(v) for v in (new[1:].split(",") + ["1"])[:2]]
            hunk = [o[0], o[1], n[0], n[1], []]
            result[current]["hunks"].append(hunk)
        elif current and hunk is not None and line[:1] in (" ", "+", "-", "\\"):
            if line.startswith("\\"):
                continue
            hunk[4].append(line)
            if line[0] == "+":
                result[current]["add"] += 1
            elif line[0] == "-":
                result[current]["del"] += 1
    return result

def line_tokens(source: str) -> list[list[str]]:
    """The Lean tokens of every line (split on newlines, as git does), by the paper's metric: no comments, no imports."""
    from leanlean.metrics.tokens import _IMPORT_COMMAND_RE, lean_refactor_lexer, remove_lean_comments
    code = remove_lean_comments(source).split("\n")
    masked = remove_lean_comments(source, mask_strings=True).split("\n")
    return [[] if _IMPORT_COMMAND_RE.match(m) else lean_refactor_lexer(c).split() for c, m in zip(code, masked, strict=True)]

def code_tokens(source: str) -> list[str]:
    """The Lean tokens of a source text, without comments."""
    from leanlean.metrics.tokens import lean_refactor_lexer, remove_lean_comments
    return lean_refactor_lexer(remove_lean_comments(source)).split()

def token_changes(path: str, state: str, before: str, after: str, hunks: list, tb: int, ta: int) -> tuple[int, int]:
    """Lean tokens deleted and added in a file.

    A modified file's changed lines are git's, plus any line git keeps whose
    tokens differ (a block comment opened or closed elsewhere); within each run
    of changed lines, the tokens both sides share (their longest matching
    blocks) count as kept.
    """
    if state != "modified":
        return (tb, 0) if state == "deleted" else (0, ta) if state == "added" else (0, 0)
    if not tb and not ta:
        return 0, 0
    old, new = line_tokens(before), line_tokens(after)
    ops, o = [], 0
    for os_, oc, _, _, lines in hunks:
        start = os_ - 1 if oc else os_
        ops += [" "] * (start - o) + [line[0] for line in lines]
        o = start + oc
    ops += [" "] * (len(old) - o) + [" "]
    deleted = added = o = n = 0
    gone, came = [], []
    for op in ops:
        a = old[o] if o < len(old) else []
        b = new[n] if n < len(new) else []
        if op == "-":
            gone += a
            o += 1
        elif op == "+":
            came += b
            n += 1
        elif a != b:
            gone += a
            came += b
            o += 1
            n += 1
        else:
            if gone or came:
                kept = sum(m.size for m in SequenceMatcher(None, gone, came, autojunk=False).get_matching_blocks())
                deleted += len(gone) - kept
                added += len(came) - kept
                gone, came = [], []
            o += 1
            n += 1
    if tb - deleted + added != ta:
        raise SystemExit(f"{path}: token diff does not reproduce the token counts: {tb} - {deleted} + {added} != {ta}")
    return deleted, added

def export_model(repo: str, short: str, key: str, graph_dir: Path, base: dict, expected: dict, out: Path, *, classification: dict) -> dict:
    from scripts.analysis.quantify_compression_categories import SourceIndex
    from scripts.analysis import abbreviations, program_sites
    from leanlean.metrics.tokens import count_lean_tokens_in_source

    record = json.loads((graph_dir / "classification-record.json").read_text())
    line_kinds, kept = {}, {}
    result = deepcopy(classification)
    if (result["baseline_tokens"] != expected["baseline"]
            or result["scored_tokens_saved"] != expected["baseline"] - expected["post"]):
        raise ValueError(f"Classification does not reproduce the scored endpoint: {short}/{repo}")
    ledger = result.pop("declarations")

    before_src = SourceIndex.from_archive(graph_dir / "before/original-sources.tar.gz")
    after_src = SourceIndex.from_archive(graph_dir / "after/original-sources.tar.gz")
    before_files, after_files = dict(before_src._files), dict(after_src._files)  # noqa: SLF001
    excluded = set(record.get("metric_scope", {}).get("excluded_files", []))
    # The harness writes the (unscored) challenge file into the container.
    after_files = {path: text for path, text in after_files.items()
                   if path in before_files or path not in excluded}
    after_data = json.loads((graph_dir / "after.json").read_text())
    after_nodes = after_data["nodes"]
    a_owner, a_roots = owners(after_nodes, after_src.path_of_module)
    after_by_name = {node["name"]: i for i, node in enumerate(after_nodes)}

    def after_text(node) -> tuple[str | None, str]:
        path = after_src.path_of_module(node["module"])
        if path is None:
            return None, ""
        lines = after_files.get(path, "").splitlines(keepends=True)
        return path, "".join(lines[node["start_line"] - 1:node["end_line"]])

    # Baseline nodes: tokens after, and the ledger's verdict.
    count = len(base["tokens"])
    after_tokens = [0] * count
    status = [STATUS["deleted"]] * count
    after_range = [None] * count
    detail = {}
    for name, index in base["names"].items():
        position = after_by_name.get(name)
        if position is None:
            continue
        node = after_nodes[a_owner[position]]
        path, text = after_text(node)
        after_tokens[index] = count_lean_tokens_in_source(text) if path and path not in excluded else 0
        after_range[index] = [path, node["start_line"], node["end_line"]]
        status[index] = STATUS["unchanged"]
    for name, entry in ledger.items():
        kind = entry["status"]
        if kind in ("dead", "deleted"):
            index = base["names"].get(name)
            if index is not None:
                status[index] = STATUS[kind]
        elif kind in ("simplification", "structural"):
            index = base["member"].get(name)
            if index is None or status[index] in (STATUS["dead"], STATUS["deleted"]):
                continue
            if kind == "structural" or status[index] != STATUS["structural"]:
                status[index] = STATUS["rewritten" if kind == "simplification" else "structural"]
            parts = [entry.get("delta", 0), entry.get("syntax", 0), entry.get("automation", 0), entry.get("rest", 0)]
            detail[index] = [a + b for a, b in zip(detail.get(index, [0, 0, 0, 0]), parts)]
    for index in range(count):
        if status[index] == STATUS["unchanged"] and after_tokens[index] != base["tokens"][index]:
            # Text of an alias range changed without a named ledger entry.
            status[index] = STATUS["rewritten"]
    for index in range(count):
        # Only comments or layout changed: the same code tokens, nothing booked.
        if status[index] in (STATUS["rewritten"], STATUS["structural"]) and not any(detail.get(index, ())):
            path, start, end = base["ranges"][index]
            a_path, a_start, a_end = after_range[index]
            before = "".join(base["sources"].get(path, "").splitlines(keepends=True)[start - 1:end])
            after = "".join(after_files.get(a_path, "").splitlines(keepends=True)[a_start - 1:a_end])
            if code_tokens(before) == code_tokens(after):
                status[index] = STATUS["comments"]

    # Kept declarations that changed: what each line of their new text became,
    # as runs [first, last, kind] of after-file lines (kind a, s or r, see step_line_kinds).
    kinds_by_file = defaultdict(list)
    step_spans = defaultdict(dict)  # path -> after line -> spans (see step_line_kinds)
    for index in range(count):
        if status[index] not in (STATUS["rewritten"], STATUS["structural"]) or not after_range[index]:
            continue
        path, start, end = after_range[index]
        if path is None:
            continue
        text = "".join(after_files.get(path, "").splitlines(keepends=True)[start - 1:end])
        found, spans = line_kinds.get(text, ({}, {}))
        for offset, row in spans.items():
            step_spans[path][str(start + offset)] = row
        runs = kinds_by_file[path]
        for offset in range(end - start + 1):
            kind = found.get(offset, "r")
            if runs and runs[-1][2] == kind and runs[-1][1] == start + offset - 1:
                runs[-1][1] += 1
            else:
                runs.append([start + offset, start + offset, kind])

    # Added declarations, placed beside what they use.
    before_names = set(base["member"])
    added = []
    added_index = {}
    for root in a_roots:
        node = after_nodes[root]
        if node["name"] in before_names:
            continue
        path, text = after_text(node)
        entry = ledger.get(node["name"], {})
        added_index[root] = len(added)
        added.append({"name": node["name"], "kind": KINDS[kind_index(node["kind"], text)], "file": path,
                      "start": node["start_line"], "end": node["end_line"],
                      "tokens": count_lean_tokens_in_source(text) if path and path not in excluded else 0,
                      "syntax": entry.get("status") == "syntax", "uses": entry.get("syntax", 0) if entry.get("status") == "added" else 0})
    added_edges = set()
    for user, dependency, _bits in after_data["edges"]:
        u, d = a_owner[user], a_owner[dependency]
        u_name, d_name = after_nodes[u]["name"], after_nodes[d]["name"]
        u_ref = ("a", added_index[u]) if u in added_index else ("b", base["member"].get(u_name))
        d_ref = ("a", added_index[d]) if d in added_index else ("b", base["member"].get(d_name))
        if u_ref[1] is None or d_ref[1] is None or u_ref == d_ref or (u_ref[0] == "b" and d_ref[0] == "b"):
            continue
        added_edges.add((u_ref, d_ref))
    # New declarations: green, except syntax.  An abbreviation (notation, macro,
    # a syntax definition) is syntax throughout; in any other new declaration a
    # line is syntax when it uses a new abbreviation or calls a measured macro.
    from leanlean.metrics.tokens import remove_lean_comments
    site_lines = defaultdict(set)
    for site in kept.get("sites", []):
        site_lines[site["path"]].add(program_sites.line_of(after_files.get(site["path"], ""), site["start"]))
    pricer = kept.get("pricer")
    for entry in added:
        path, start, end = entry["file"], entry["start"], entry["end"]
        if path is None or path in excluded:
            continue
        if ledger.get(entry["name"], {}).get("status") == "syntax":
            lines = set(range(start, end + 1))
        else:
            text = "".join(after_files.get(path, "").splitlines(keepends=True)[start - 1:end])
            visible = pricer.visible(entry["name"]) if pricer else []
            lines = {start + k for k, line in enumerate(remove_lean_comments(text).split("\n"))
                     if visible and abbreviations.Pricer.price_steps([abbreviations.tokens(line)], visible) > 0}
            lines |= {n for n in site_lines[path] if start <= n <= end}
        runs = kinds_by_file[path]
        for n in sorted(lines):
            if runs and runs[-1][2] == "s" and runs[-1][1] == n - 1:
                runs[-1][1] = n
            else:
                runs.append([n, n, "s"])

    added_lanes, shift = place_added(added, added_edges, base)
    offset = count
    encoded_edges = [((offset + u) if ku == "a" else u, (offset + d) if kd == "a" else d)
                     for (ku, u), (kd, d) in sorted(added_edges, key=lambda e: (e[0][0], e[0][1], e[1][0], e[1][1]))]

    # Files: tokens and line stats; after text only where it changed, and of the harness files.
    diff = git_diff(before_files, after_files)
    files = []
    changed_text = {}
    for path in sorted(set(before_files) | set(after_files)):
        in_before, in_after = path in before_files, path in after_files
        tb = count_lean_tokens_in_source(before_files[path]) if in_before and path not in excluded else 0
        ta = count_lean_tokens_in_source(after_files[path]) if in_after and path not in excluded else 0
        if in_before and in_after:
            state = "modified" if before_files[path] != after_files[path] else "unchanged"
        else:
            state = "added" if in_after else "deleted"
        if state in ("modified", "added"):
            changed_text[path] = after_files[path]
        stats = diff.get(path, {})
        tdel, tadd = token_changes(path, state, before_files.get(path, ""), after_files.get(path, ""), stats.get("hunks", []), tb, ta)
        files.append({"path": path, "state": state, "before": tb, "after": ta, "scored": path not in excluded,
                      "add": stats.get("add", after_files.get(path, "").count("\n") if state == "added" else 0),
                      "del": stats.get("del", before_files.get(path, "").count("\n") if state == "deleted" else 0),
                      "tadd": tadd, "tdel": tdel})
    for path, text in harness_files(repo).items():
        if path in before_files:
            raise ValueError(f"harness file collides with a repository file: {repo}/{path}")
        changed_text[path] = text
        files.append({"path": path, "state": "unchanged", "before": 0, "after": 0, "scored": False, "harness": True,
                      "add": 0, "del": 0, "tadd": 0, "tdel": 0})
    files.sort(key=lambda f: f["path"])

    outcome = {k: v for k, v in result.items() if k in (
        "baseline_tokens", "scored_tokens_saved", "syntax_optimization", "automation", "dead_code",
        "structural_change", "proof_rewriting", "automation_structural", "automation_simplification")}
    outcome["detail"] = {k: v for k, v in result["detail"].items() if isinstance(v, (int, float))}
    payload = {
        "repository": repo, "model": key, "outcome": outcome,
        "status": status, "after": after_tokens, "range": after_range,
        "ledger": {str(k): v for k, v in detail.items()},
        "added": added, "added_edges": [v for edge in encoded_edges for v in edge], "added_lanes": added_lanes,
        "shift": shift, "line_kinds": {path: sorted(runs) for path, runs in kinds_by_file.items()},
        "step_spans": step_spans,
        "files": files, "text": changed_text,
    }
    dump(out / f"{short}.json.gz", payload)
    dump(out / f"{short}.diff.json.gz", {path: entry["hunks"] for path, entry in diff.items()})
    return {"model": key, "status": {s: status.count(v) for s, v in STATUS.items()}, "added": len(added)}
