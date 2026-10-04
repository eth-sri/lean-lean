"""Pure (no-Docker) source-deletion logic, used to derive holdout Challenge closures.

Preprocessing itself runs the pinned lean-strip, which carries its own copy of
this logic; the holdout Comparator keeps this version because published holdout
datasets pin the closures it derived.

Given a Lean source file's text and the declaration ranges emitted by the
`.olean` declaration reader (see preprocessing/olean.py), `strip_source` removes
the source spans of every declaration NOT in the keep-set, leaving all
non-declaration scaffolding (imports, `open`, `namespace`/`end`, `variable`,
`notation`/`macro`, free-floating comments) untouched.

Everything here is pure text -> text so it can be unit-tested without a
container.

Key facts about the ranges (verified empirically against the pinned toolchains):
  * Line numbers are 1-based; the command range is [start_line, end_line]
    inclusive.
  * The range INCLUDES a leading docstring (`/-- ... -/`), so deleting a
    dropped decl's span removes its docstring too — no orphans.
  * A standalone `@[attr]` line ABOVE the keyword is NOT part of the range, so
    we extend a deletion upward over such lines.
  * Range-less decls (auto-generated `.rec`/`.ctor`/`.match_n`/...) carry
    start_line is None and are never deleted (they have no source).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

DeclRange = dict[str, Any]

# Lines that decorate the *following* declaration but sit ABOVE the keyword and
# are NOT part of the decl's command range, so a deletion of that decl must
# consume them too — otherwise they dangle and corrupt parsing:
#   * a standalone attribute line: `@[simp]`, `@[simp, norm_cast]`
#   * a command-prefix modifier ending in `in`: `set_option ... in`,
#     `open ... in`, `attribute [...] ... in` (these apply to the next command;
#     left orphaned they swallow the subsequent `end`/decl and desync scoping).
_ATTR_ONLY_RE = re.compile(r"^\s*@\[[^\]]*\]\s*$")
# Command-prefix modifiers that scope to the *next* command via a trailing
# `in` (so they sit above the decl keyword, outside its range). Anchored on the
# keyword so docstring/comment prose merely ending in the word "in" is ignored.
_PREFIX_IN_RE = re.compile(
    r"^\s*(set_option|open|attribute|omit|include|variable|universe|#guard_msgs)\b"
    r"[\s\S]*\bin\s*$"
)
_PREFIX_START_RE = re.compile(
    r"^\s*(set_option|open|attribute|omit|include|variable|universe|#guard_msgs)\b"
)


_LINE_COMMENT_RE = re.compile(r"--.*?$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
# Lean identifier (incl. dotted access and unicode letters used in math names).
# The character class deliberately stops short of U+2190-U+2BFF (arrows,
# mathematical operators, brackets) and of the Latin-1 operators U+00D7/U+00F7.
# Those are Lean *syntax*, not identifier characters, and a class that admits
# them silently glues a delimiter onto the following name: `⟨foo` then
# tokenises as one identifier and resolves to nothing, losing every dependency
# introduced by an anonymous constructor, binder or arrow that abuts a name.
_IDENT_CHARS = (
    "A-Za-z_"
    "\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u218F"  # Latin-1/Greek/letterlike, no × ÷
    "\u2C00-\uFFFF"                              # beyond the symbol blocks
)
_IDENT_RE = re.compile(f"[{_IDENT_CHARS}][{_IDENT_CHARS}\\w.']*")
_LEAN_COMMAND_NAME_PATTERN = r"[A-Za-z_À-￿][\w.À-￿']*"

# Lean 4 mangles a `private` declaration to `_private.<Module>.0.<SourceName>`.
# The module component is a *visibility constraint*: a private declaration is
# inaccessible outside the module that defines it, so it must never enter
# another module's candidate set. The trailing component is what source text
# actually writes.
_PRIVATE_KERNEL_NAME_RE = re.compile(
    r"^_private\.((?:(?:«[^»]*»|[^.«»]+)\.)+?)[0-9]+\.(.+)$"
)


def _private_owner_and_display(name: str) -> tuple[str | None, str]:
    """Split a kernel name into (private-defining module or None, source name)."""

    if not name.startswith("_private."):
        return None, name
    match = _PRIVATE_KERNEL_NAME_RE.match(name)
    if match is None:
        return None, name
    return match.group(1).rstrip("."), match.group(2)



# A binder group such as `(hF : IsSmoothBounded F)` or `{x y : Foo}`. Lean's
# generalised field notation resolves `hF.memLp` through the *type* of `hF`, so
# a scan that ignores binder types cannot see that reference at all. Binders
# written in the declaration's own source are recoverable without elaboration.
_BINDER_GROUP_RE = re.compile(
    r"[(\{\[⦃]\s*([^:()\{\}\[\]⦃⦄]+?)\s*:\s*([^()\{\}\[\]⦃⦄]+?)\s*[)\}\]⦄]"
)


def _binder_head_types(span: str) -> dict[str, str]:
    """Map binder names in one declaration's source to their type's head name."""

    out: dict[str, str] = {}
    for binder_part, type_part in _BINDER_GROUP_RE.findall(span):
        head = _IDENT_RE.match(type_part.strip())
        if head is None:
            continue
        for binder in binder_part.split():
            if _IDENT_RE.fullmatch(binder):
                out.setdefault(binder, head.group(0))
    return out


def _strip_comments(text: str) -> str:
    # Block comments first (handles /-- ... -/ docstrings too), then line comments.
    prev = None
    while prev != text:  # nested block comments
        prev = text
        text = _BLOCK_COMMENT_RE.sub(" ", text)
    return _LINE_COMMENT_RE.sub("", text)


def augment_deps_with_source(
    decls: list[DeclRange],
    deps: dict[str, set[str]],
    originals: dict[str, str],
    mod_path: dict[str, str],
) -> dict[str, set[str]]:
    """Add dependency edges from SOURCE-level identifier references.

    `foldConsts` over the elaborated term misses declarations that a decl needs
    only to *elaborate* (lemmas named in `simp`/`rw`, dot-notation projections,
    `open`ed helpers) because they don't survive into the final term. Those names
    do appear textually in the decl's source, so we add an edge for every source
    identifier that resolves to a *visible* project decl by full name, or
    unambiguously by last component (e.g. `b.fivefold` ->
    `DeGiorgi.JNBall.fivefold` when that's the only visible `*.fivefold`).
    Visibility is the source module plus its transitive project imports.
    Comments/docstrings are stripped first to avoid keeping things merely
    mentioned in prose.
    """
    full_names = {d["name"] for d in decls}
    names_by_module: dict[str, set[str]] = {}
    for decl in decls:
        names_by_module.setdefault(decl["module"], set()).add(decl["name"])

    # Text is only meaningful in the environment in which its source module
    # elaborates. Resolving an unqualified token against *every* project
    # declaration creates false cross-branch edges: an obsolete module merely
    # imported by an umbrella can otherwise keep itself alive because it owns
    # the project's only declaration with that last component. Mirror Lean's
    # module visibility instead: a declaration may name declarations from its
    # own module and its transitive project-import closure.
    source_root = derive_source_root(mod_path)
    import_graph = (
        build_module_import_graph(
            originals,
            source_root,
            known_module_paths=mod_path,
        )[0]
        if source_root is not None
        else {}
    )
    visible_names_by_module: dict[str, set[str]] = {}
    source_lines_by_path: dict[
        str,
        tuple[
            list[str],
            list[str],
            dict[int, str],
            dict[int, tuple[str, ...]],
        ],
    ] = {}
    lookup_indexes_by_module: dict[
        str,
        tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]],
    ] = {}

    def visible_names(module: str) -> set[str]:
        cached = visible_names_by_module.get(module)
        if cached is not None:
            return cached
        visible_modules = _closure(import_graph, {module})
        if not visible_modules:
            # Synthetic unit inputs may omit a source graph. Production inputs
            # always have a derivable graph and never take this test-only path.
            names = set(full_names)
        else:
            names = set()
            for visible_module in visible_modules:
                for name in names_by_module.get(visible_module, ()):
                    owner, _display = _private_owner_and_display(name)
                    if owner is not None and owner != module:
                        # Unreachable private homonym: including it would make a
                        # real same-module reference look ambiguous.
                        continue
                    names.add(name)
        visible_names_by_module[module] = names
        return names

    def lookup_indexes(
        module: str,
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Index last components and every dotted suffix once per module.

        Large generated declarations can contain hundreds of thousands of
        identifier tokens. Scanning the full visible-name set for each token
        is quadratic in practice; these indexes preserve the exact string
        matching policy while making each lookup constant-time.
        """

        cached = lookup_indexes_by_module.get(module)
        if cached is not None:
            return cached
        by_last: dict[str, set[str]] = {}
        by_suffix: dict[str, set[str]] = {}
        by_display: dict[str, set[str]] = {}
        for name in visible_names(module):
            _owner, display = _private_owner_and_display(name)
            by_display.setdefault(display, set()).add(name)
            by_last.setdefault(display.rsplit(".", 1)[-1], set()).add(name)
            for index, char in enumerate(display):
                if char == "." and index + 1 < len(display):
                    by_suffix.setdefault(display[index + 1 :], set()).add(name)
        cached = by_last, by_suffix, by_display
        lookup_indexes_by_module[module] = cached
        return cached

    out = {k: set(v) for k, v in deps.items()}
    for d in decls:
        if d["start_line"] is None:
            continue
        path = mod_path.get(d["module"])
        text = originals.get(path) if path else None
        if not text:
            continue
        cached_lines = source_lines_by_path.get(path)
        if cached_lines is None:
            namespace_contexts, opened_namespaces = (
                _source_namespace_resolution_contexts(text)
            )
            cached_lines = (
                text.split("\n"),
                _mask_comments_preserve_layout(text).split("\n"),
                namespace_contexts,
                opened_namespaces,
            )
            source_lines_by_path[path] = cached_lines
        lines, code_lines, namespace_contexts, opened_namespaces = cached_lines
        # Include a scoped ``attribute [...] target in`` prefix in the
        # following declaration's source evidence. Its target is needed only
        # when that declaration survives, so this creates an edge rather than
        # a global root.
        start = _extend_up(
            code_lines, min(len(code_lines), max(1, int(d["start_line"])))
        )
        span = "\n".join(lines[start - 1: d["end_line"]])
        span = _strip_comments(span)
        edges = out.setdefault(d["name"], set())
        visible = visible_names(d["module"])
        by_last, by_suffix, by_display = lookup_indexes(d["module"])
        declaration_line = min(
            len(code_lines), max(1, int(d["start_line"]))
        )
        lexical_namespace = namespace_contexts.get(declaration_line, "")
        lexical_opens = opened_namespaces.get(declaration_line, ())
        binder_types = _binder_head_types(span)
        for tok in _IDENT_RE.findall(span):
            # Lean resolves an identifier through its enclosing namespaces
            # from innermost outwards and falls back to the root name last, so
            # the bare name must NOT be tried first: inside `namespace N` the
            # token `A.b` denotes `N.A.b` whenever that exists, even when an
            # unrelated root-level `A.b` is also in scope. Trying the bare name
            # first silently attributes the edge to a homonym in another
            # module. `_namespace_name_candidates` already yields Lean's order
            # and ends with the bare token. Lookups go through display names so
            # a `private` declaration, written in source under its plain name,
            # maps back to the kernel name the declaration graph uses.
            # A local binder shadows any global of the same name, so a token
            # that *is* a binder in this declaration never denotes a project
            # declaration: `lemma f (c : ℂ)` must not depend on a global `c`.
            if tok in binder_types:
                continue
            resolved_name: str | None = None
            self_reference = False
            for candidate in _namespace_name_candidates(tok, lexical_namespace):
                hits = by_display.get(candidate, set())
                if not hits:
                    continue
                if hits == {d["name"]}:
                    self_reference = True
                    break
                unambiguous = hits - {d["name"]}
                if len(unambiguous) == 1:
                    resolved_name = next(iter(unambiguous))
                    break
                break
            if self_reference:
                continue
            if resolved_name is None and tok in visible:
                # Namespace-qualified resolution was ambiguous or absent; the
                # token is itself a full declaration name, so use it.
                resolved_name = None if tok == d["name"] else tok
                if resolved_name is None:
                    continue
            if resolved_name is not None:
                edges.add(resolved_name)
                continue
            opened_hits = {
                resolved
                for namespace in lexical_opens
                for resolved in by_display.get(f"{namespace}.{tok}", set())
                if resolved != d["name"]
            }
            if len(opened_hits) == 1:
                edges.add(next(iter(opened_hits)))
                continue
            # A source file inside a namespace may use a partially-qualified
            # name (for example ``Tensorial.numIndices`` for
            # ``TensorSpecies.Tensorial.numIndices``). Prefer its unique
            # visible qualified-suffix match before falling back to the final
            # component, which can be ambiguous.
            # Generalised field notation: `hF.memLp` where `hF` is a binder of
            # type `IsSmoothBounded F` denotes `IsSmoothBounded.memLp`. The
            # receiver's type is written in the declaration's own signature, so
            # this particular type-directed reference needs no elaborator.
            if "." in tok:
                receiver, _, field = tok.partition(".")
                type_head = binder_types.get(receiver)
                if type_head and field:
                    for candidate in _namespace_name_candidates(
                        type_head, lexical_namespace
                    ):
                        type_hits = by_display.get(candidate, set())
                        if not type_hits:
                            continue
                        field_hits: set[str] = set()
                        for raw_type in type_hits:
                            _owner, type_display = _private_owner_and_display(
                                raw_type
                            )
                            field_hits |= by_display.get(
                                f"{type_display}.{field}", set()
                            )
                        field_hits -= {d["name"]}
                        if len(field_hits) == 1:
                            edges.add(next(iter(field_hits)))
                        break
            suffix_cands = by_suffix.get(tok, set()) - {d["name"]}
            if len(suffix_cands) == 1:
                edges.add(next(iter(suffix_cands)))
                continue
            # Lean tactics may print names of range-less generated helpers such
            # as `foo._proof_1_7` or `foo.match_1`. Those constants disappear
            # if the source declaration `foo` is removed, but they are absent
            # from `full_names` because the declaration graph intentionally
            # contains only source-backed ranges. Resolve the longest dotted
            # source-backed prefix so an explicit replay-certified tactic rule
            # keeps the command that recreates its generated helper.
            prefix = tok
            while "." in prefix:
                prefix = prefix.rsplit(".", 1)[0]
                # Resolve the prefix the way Lean resolves any identifier:
                # through the enclosing namespaces first, bare name last.
                prefix_hits: set[str] = set()
                for candidate in _namespace_name_candidates(
                    prefix, lexical_namespace
                ):
                    prefix_hits = by_display.get(candidate, set())
                    if prefix_hits:
                        break
                if prefix_hits:
                    unambiguous = prefix_hits - {d["name"]}
                    if len(unambiguous) == 1:
                        edges.add(next(iter(unambiguous)))
                    break
            else:
                prefix = ""
            if prefix:
                continue
            cands = by_last.get(tok.rsplit(".", 1)[-1])
            if cands and len(cands) == 1:
                c = next(iter(cands))
                if c != d["name"]:
                    edges.add(c)

    # Lean exposes source-backed local declarations such as a `let rec`
    # helper as separate constants whose ranges are strictly nested inside the
    # enclosing command. The local constant can occur in the dependency graph
    # independently, but its source cannot survive without the outer command's
    # header. Make that syntactic ownership explicit in the graph so retaining
    # a local helper always retains every declaration command containing it.
    ranged_by_path: dict[str, list[DeclRange]] = {}
    for decl in decls:
        if decl.get("start_line") is None or decl.get("end_line") is None:
            continue
        path = mod_path.get(decl["module"])
        if path and path in originals:
            ranged_by_path.setdefault(path, []).append(decl)

    def source_interval(
        decl: DeclRange,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (int(decl["start_line"]), int(decl.get("start_col") or 0)),
            (int(decl["end_line"]), int(decl.get("end_col") or 0)),
        )

    for file_decls in ranged_by_path.values():
        intervals = [(decl, *source_interval(decl)) for decl in file_decls]
        for child, child_start, child_end in intervals:
            edges = out.setdefault(child["name"], set())
            for outer, outer_start, outer_end in intervals:
                if outer["name"] == child["name"]:
                    continue
                if (
                    outer_start <= child_start
                    and child_end <= outer_end
                    and (outer_start, outer_end) != (child_start, child_end)
                ):
                    edges.add(outer["name"])
    return out


# Every attributed declaration is semantically live for stripping. Lean
# attributes are open-ended: projects define custom simp sets (`delta0_simps`),
# tactic extensions (`tactic`, `positivity`, `delab`), code generators, and
# other environment registrations. Maintaining an allow-list inevitably misses
# new attributes, so any `@[...]` declaration is a root.
_ATTR_TAG_RE = re.compile(r"@\[[^\]]*\]")
# A standalone `attribute [foo] bar` command applying tags to named decls. The
# first-line regex is complemented by continuation handling below.
_SCOPED_ATTRIBUTE_PREFIX = r"(?:scoped[ \t]*\[[^\]\n]*\][ \t]+)?"
_ATTRIBUTE_CMD_RE = re.compile(
    r"^[ \t]*" + _SCOPED_ATTRIBUTE_PREFIX
    + r"attribute\s*\[[^\]]*\]\s*(.*?)\s*$",
    re.MULTILINE,
)


def _attribute_command_payloads(text: str) -> list[str]:
    """Return source payloads of standalone ``attribute [...]`` commands.

    The payload excludes the attribute list itself and includes indented
    continuation lines. Commands ending in ``in`` are returned as well: they
    still root their named targets, but callers can distinguish them from
    persistent environment registrations with ``_is_prefix_attribute``.
    Scoped registrations such as ``scoped[Order] attribute [instance] x``
    are persistent too and use the same exact-target policy.
    """

    lines = _mask_comments_preserve_layout(text).split("\n")
    payloads: list[str] = []
    i = 0
    while i < len(lines):
        match = _ATTRIBUTE_CMD_RE.match(lines[i])
        if not match:
            i += 1
            continue
        payload = [match.group(1)]
        # Lean commonly formats an attribute command as
        #   attribute [fun_prop]
        #     theorem₁
        #     theorem₂
        # Continue through its indented target lines.
        j = i + 1
        while j < len(lines) and lines[j].strip() and lines[j][:1].isspace():
            payload.append(lines[j])
            j += 1
        payloads.append(" ".join(payload).strip())
        i = max(j, i + 1)
    return payloads


def _is_prefix_attribute(payload: str) -> bool:
    """Whether an attribute command is a ``... in`` next-command prefix."""

    return bool(re.search(r"\bin\s*$", payload))

# Declaration kinds whose deps are NOT name-references and so are invisible to
# the closure: typeclass `instance`s (pulled by synthesis at elaboration time)
# and custom-syntax decls — `macro`/`elab`/`syntax`/`notation`/`macro_rules`
# (a surviving proof invokes the syntax, but the call site names no constant).
# Force-keep any decl whose command header is one of these. The keyword may sit
# behind attributes and modifiers (`@[…] noncomputable scoped instance …`).
_FORCE_KEEP_KINDS = r"instance|macro_rules|macro|elab_rules|elab|syntax|notation"
_KIND_KW_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:public\s+|private\s+|protected\s+|noncomputable\s+|scoped\s+|local\s+|"
    r"partial\s+|unsafe\s+)*"
    rf"(?:{_FORCE_KEEP_KINDS})\b",
    re.MULTILINE)


def detect_automation_tagged(
    decls: list[DeclRange],
    originals: dict[str, str],
    mod_path: dict[str, str],
) -> set[str]:
    """Return every project declaration carrying any Lean attribute.

    This includes inline/above `@[...]` annotations and targets of standalone
    `attribute [...] name` commands, including multiline target lists.
    """
    out: set[str] = set()

    for d in decls:
        if d["start_line"] is None:
            continue
        path = mod_path.get(d["module"])
        if not path or path not in originals:
            continue
        lines = originals[path].split("\n")
        start, end = int(d["start_line"]), min(len(lines), int(d["end_line"]))
        # The decl's own command span (attributes only appear as `@[...]`
        # annotations, never inside tactic blocks, so a body `by simp` is safe).
        # The range includes a leading docstring, so an `@[simp]` between the
        # docstring and keyword is in-span.
        if _ATTR_TAG_RE.search("\n".join(lines[start - 1:end])):
            out.add(d["name"]); continue
        # Plus a standalone, possibly multiline `@[...]` block directly
        # above the declaration range (the range may start at its docstring).
        code_lines = _mask_comments_preserve_layout(originals[path]).split("\n")
        j = start - 1
        while j >= 1 and code_lines[j - 1].strip() == "":
            j -= 1
        if _attribute_block_start(code_lines, j) is not None:
            out.add(d["name"])

    out.update(detect_attribute_command_roots(decls, originals))
    return out


def detect_attribute_command_roots(
    decls: list[DeclRange], originals: dict[str, str]
) -> set[str]:
    """Resolve project declarations named by explicit ``attribute`` commands."""

    name_set = {d["name"] for d in decls}
    by_last: dict[str, list[str]] = {}
    for name in name_set:
        by_last.setdefault(name.rsplit(".", 1)[-1], []).append(name)

    out: set[str] = set()
    for text in originals.values():
        for payload in _attribute_command_payloads(text):
            if _is_prefix_attribute(payload):
                continue
            for tok in _IDENT_RE.findall(payload):
                if tok in name_set:
                    out.add(tok)
                else:
                    candidates = by_last.get(tok.rsplit(".", 1)[-1], [])
                    if "." in tok:
                        candidates = [
                            name for name in candidates if name.endswith("." + tok)
                        ]
                    if len(candidates) == 1:
                        out.add(candidates[0])
    return out


# Environment mutations without an explicit attribute target. Their command
# text must survive tree-shaking, but that does not make every declaration in
# the containing file live. Named ``initialize``/``register_option`` commands
# are ordinary source declarations in Lean's range data and are rooted below;
# their bodies then receive the same kernel/iLean/source dependency edges as
# every other declaration.
_OPAQUE_ENVIRONMENT_COMMAND_RE = re.compile(
    r"^[ \t]*(?:"
    r"declare_aesop_rule_sets?|register_simp_attr|register_grind_attr|"
    r"initialize\b|builtin_initialize\b|register_option\b)",
    re.MULTILINE,
)


def detect_attribute_command_files(originals: dict[str, str]) -> set[str]:
    """Return files with persistent, explicit attribute registrations.

    These files must preserve the command text even if all ordinary
    declarations are removed, but their unrelated declarations do *not* need
    to be rooted. ``attribute [...] target in`` is a next-command prefix and
    is deleted atomically with that command by ``strip_source`` instead.
    """

    return {
        path
        for path, text in originals.items()
        if any(
            not _is_prefix_attribute(payload)
            for payload in _attribute_command_payloads(text)
        )
    }


def detect_opaque_environment_command_files(
    originals: dict[str, str],
) -> set[str]:
    """Return files containing non-attribute environment commands."""

    return {
        path
        for path, text in originals.items()
        if _OPAQUE_ENVIRONMENT_COMMAND_RE.search(
            _mask_comments_preserve_layout(text)
        )
    }


def detect_opaque_environment_command_roots(
    decls: list[DeclRange],
    originals: dict[str, str],
    mod_path: dict[str, str],
) -> set[str]:
    """Root source declarations whose range owns an opaque environment command.

    This is declaration-granular: an ``initialize foo := ...`` command keeps
    ``foo`` and the dependencies of its body, not unrelated lemmas elsewhere
    in the file. Commands without a declaration range are preserved as source
    scaffolding by the source-retention plan.
    """

    out: set[str] = set()
    for decl in decls:
        if decl.get("start_line") is None or decl.get("end_line") is None:
            continue
        path = mod_path.get(decl["module"])
        text = originals.get(path) if path else None
        if text is None:
            continue
        lines = text.split("\n")
        start = max(1, int(decl["start_line"]))
        end = min(len(lines), int(decl["end_line"]))
        span = "\n".join(lines[start - 1:end])
        if _OPAQUE_ENVIRONMENT_COMMAND_RE.search(
            _mask_comments_preserve_layout(span)
        ):
            out.add(decl["name"])
    return out


# A name-based declaration graph has no node for namespaces themselves. If a
# surviving file says `open Project.Namespace`, at least one declaration owned
# by that project-local namespace must remain so Lean can resolve the command.
# Seed the complete matching namespace: this is deterministic and avoids
# iterative guessing about which declaration was intended by a later command.
_OPEN_NAMESPACE_RE = re.compile(
    r"^[ \t]*open(?![ \t]+scoped\b)[ \t]+([^\n]+)", re.MULTILINE
)
_SELECTIVE_OPEN_COMMAND_RE = re.compile(
    r"^(?P<indent>[ \t]*)open[ \t]+(?P<namespace>"
    + _LEAN_COMMAND_NAME_PATTERN
    + r")[ \t]+\((?P<targets>[^()\n]*)\)[ \t]*$"
)
_NAMESPACE_SCOPE_RE = re.compile(
    r"^[ \t]*(?:public[ \t]+|private[ \t]+)?namespace[ \t]+"
    r"([A-Za-z_À-￿][\w.À-￿']*)[ \t]*$"
)
_NON_NAMESPACE_SCOPE_RE = re.compile(
    r"^[ \t]*(?:public[ \t]+|private[ \t]+)?(?:section|mutual)\b"
)
_SCOPE_END_RE = re.compile(r"^[ \t]*end(?:[ \t]+[^\s]+)?[ \t]*$")


def _namespace_name_candidates(token: str, lexical_namespace: str) -> list[str]:
    """Return Lean-style exact namespace candidates, nearest scope first."""

    if token.startswith("_root_."):
        return [token.removeprefix("_root_.")]
    components = [part for part in lexical_namespace.split(".") if part]
    candidates: list[str] = []
    for length in range(len(components), -1, -1):
        prefix = ".".join(components[:length])
        candidate = f"{prefix}.{token}" if prefix else token
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _source_namespace_resolution_contexts(
    file_text: str,
) -> tuple[dict[int, str], dict[int, tuple[str, ...]]]:
    """Return lexical namespace and persistent ``open`` context per line.

    This models command-level namespace/section scopes and never guesses from
    arbitrary dotted substrings. The old substring guess misclassified
    Mathlib's top-level ``Topology`` namespace as project-local merely because
    a project declaration lived below ``ForMathlib.Topology``.
    """

    code_lines = _mask_comments_preserve_layout(file_text).split("\n")
    namespace = ""
    opened: list[str] = []
    frames: list[tuple[str, list[str]]] = []
    namespaces_by_line: dict[int, str] = {}
    opens_by_line: dict[int, tuple[str, ...]] = {}

    for line_no, code_line in enumerate(code_lines, 1):
        namespaces_by_line[line_no] = namespace
        opens_by_line[line_no] = tuple(opened)

        namespace_scope = _NAMESPACE_SCOPE_RE.match(code_line)
        if namespace_scope:
            frames.append((namespace, list(opened)))
            namespace = _namespace_name_candidates(
                namespace_scope.group(1), namespace
            )[0]
            continue
        if _NON_NAMESPACE_SCOPE_RE.match(code_line):
            frames.append((namespace, list(opened)))
            continue
        if _SCOPE_END_RE.match(code_line):
            if frames:
                namespace, opened = frames.pop()
            continue

        open_command = _OPEN_NAMESPACE_RE.match(code_line)
        if not open_command:
            continue
        selective = _SELECTIVE_OPEN_COMMAND_RE.match(code_line)
        tokens = (
            [selective.group("namespace")]
            if selective
            else _IDENT_RE.findall(open_command.group(1))
        )
        if "in" in tokens:
            continue
        for token in tokens:
            for candidate in _namespace_name_candidates(token, namespace):
                if candidate not in opened:
                    opened.append(candidate)

    return namespaces_by_line, opens_by_line


def _resolve_project_namespace_members(
    namespace: str,
    declarations: set[str],
    lexical_namespace: str,
) -> set[str]:
    """Resolve a namespace only by Lean's exact lexical prefixes."""

    for candidate in _namespace_name_candidates(namespace, lexical_namespace):
        prefix = candidate + "."
        members = {name for name in declarations if name.startswith(prefix)}
        if members:
            return members
    return set()


_PRIVATE_DECL_NAME_RE = re.compile(
    r"^_private\.(?:(?:«[^»]*»|[^.«»]+)\.)+?[0-9]+\.(.+)$"
)


def _resolve_namespace_provenance(
    namespace: str,
    project_declarations: set[str],
    imported_namespaces: set[str],
    lexical_namespace: str,
    opened_namespaces: tuple[str, ...] = (),
) -> tuple[set[str], bool]:
    """Resolve the first namespace candidate known to the built environment.

    A namespace may contain both project declarations and imported declarations.
    Such mixed namespaces remain semantically imported even after every project
    extension is stripped, so an `open` for them must survive. A namespace is
    removable only when its resolved candidate has project members and no
    imported declaration provenance.
    """

    first_members: set[str] = set()
    imported = False
    # Private declarations still create and inhabit their source namespace.
    # Match that namespace using the user name, but return the original kernel
    # names so visibility/keep-set intersections retain their exact identities.
    source_names: dict[str, str] = {}
    for name in project_declarations:
        private = _PRIVATE_DECL_NAME_RE.match(name) if name.startswith("_private.") else None
        source_names[name] = private.group(1) if private else name
    candidates = _namespace_name_candidates(namespace, lexical_namespace)
    if not namespace.startswith("_root_."):
        candidates += [f"{opened}.{namespace}" for opened in opened_namespaces]
    for candidate in dict.fromkeys(candidates):
        prefix = candidate + "."
        members = {
            name for name, source_name in source_names.items()
            if source_name.startswith(prefix)
        }
        if members and not first_members:
            first_members = members
        imported = imported or candidate in imported_namespaces
        if members and imported:
            break
    # Source-order information is unavailable here. A later project namespace
    # may share the short name of an imported namespace that already existed at
    # this command (for example a later nested ``namespace PowerSeries`` after
    # ``open PowerSeries``). Never delete that valid imported open merely
    # because the later project namespace is dead. This retains no declaration;
    # it only preserves an otherwise valid source command.
    return first_members, imported



_OPEN_HIDING_RE = re.compile(
    r"^\s*open\s+([^\n]*?)\s+hiding\s+([^\n]*)$", re.MULTILINE
)


def detect_open_hiding_roots(
    decls: list[DeclRange], originals: dict[str, str]
) -> set[str]:
    """Return declarations named by an ``open ... hiding ...`` clause.

    A `hiding` clause names declarations in order to exclude them, so the file
    does not depend on their *meaning* - but it does require them to exist:
    deleting one leaves the clause referring to an unknown constant and the
    module stops elaborating. The clause sits in the file header, outside every
    declaration's source range, so the span-based scan never sees it and no
    edge is produced; these names are therefore rooted directly.
    """

    names = {d["name"] for d in decls}
    by_suffix: dict[str, set[str]] = {}
    for name in names:
        for index, char in enumerate(name):
            if char == "." and index + 1 < len(name):
                by_suffix.setdefault(name[index + 1 :], set()).add(name)
    roots: set[str] = set()
    for text in originals.values():
        for namespaces, hidden in _OPEN_HIDING_RE.findall(_strip_comments(text)):
            opened = [n for n in namespaces.split() if _IDENT_RE.fullmatch(n)]
            for token in hidden.split():
                if not _IDENT_RE.fullmatch(token):
                    continue
                for namespace in opened:
                    candidate = f"{namespace}.{token}"
                    if candidate in names:
                        roots.add(candidate)
                    else:
                        matches = by_suffix.get(candidate, set())
                        if len(matches) == 1:
                            roots.add(next(iter(matches)))
    return roots


def detect_open_namespace_roots(
    decls: list[DeclRange], originals: dict[str, str]
) -> set[str]:
    """Return exact project-local candidates exposed by persistent ``open``.

    These are dependency-resolution candidates, not blanket keep roots. The
    production graph uses lexical source edges to retain only candidates that
    surviving source actually references.
    """

    names = {d["name"] for d in decls}
    roots: set[str] = set()
    for text in originals.values():
        masked = _mask_comments_preserve_layout(text)
        namespace_contexts, _ = _source_namespace_resolution_contexts(text)
        for match in _OPEN_NAMESPACE_RE.finditer(masked):
            selective = _SELECTIVE_OPEN_COMMAND_RE.match(match.group(0))
            tokens = (
                [selective.group("namespace")]
                if selective
                else _IDENT_RE.findall(match.group(1))
            )
            if "in" in tokens:
                continue
            line_no = masked.count("\n", 0, match.start()) + 1
            lexical_namespace = namespace_contexts.get(line_no, "")
            for namespace in tokens:
                roots.update(
                    _resolve_project_namespace_members(
                        namespace, names, lexical_namespace
                    )
                )
    return roots


# When the full subproject environment imports two independent modules that
# define the same fully-qualified constant, Lean records only one module as the
# declaration owner. Detect the unambiguous textual form (a dotted declaration
# name) for diagnostics; module import closure, rather than whole-file rooting,
# decides whether either provider is live.
_QUALIFIED_DECL_HEADER_RE = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\][ \t]*)*"
    r"(?:(?:public|private|protected|noncomputable|local|scoped|unsafe|partial)"
    r"[ \t]+)*(?:def|theorem|lemma|abbrev|opaque|axiom)\s+"
    r"(?:_root_\.)?([A-Za-z_À-￿][\wÀ-￿']*(?:\.[A-Za-z_À-￿][\wÀ-￿']*)+)",
    re.MULTILINE,
)


def detect_duplicate_qualified_declaration_files(
    originals: dict[str, str],
) -> set[str]:
    """Return every file in a duplicate fully-qualified provider collision."""
    providers: dict[str, set[str]] = {}
    for path, text in originals.items():
        masked = _mask_comments_preserve_layout(text)
        for match in _QUALIFIED_DECL_HEADER_RE.finditer(masked):
            providers.setdefault(match.group(1), set()).add(path)
    return {
        path
        for paths in providers.values() if len(paths) > 1
        for path in paths
    }


# Lean's source-range API occasionally attaches a surrounding scope command to
# the first or last declaration in that scope. Record these files for audit;
# ``strip_source`` protects namespace/section commands line-wise and handles
# mutual blocks atomically.
_RANGE_STRUCTURAL_COMMAND_RE = re.compile(
    r"^[ \t]*(?:namespace|section|end|mutual)\b", re.MULTILINE
)


def detect_structural_range_files(
    decls: list[DeclRange],
    originals: dict[str, str],
    mod_path: dict[str, str],
) -> set[str]:
    """Return files where a declaration range overlaps a scope command."""
    code_lines = {
        path: _mask_comments_preserve_layout(text).split("\n")
        for path, text in originals.items()
    }
    out: set[str] = set()
    for decl in decls:
        if decl.get("start_line") is None or decl.get("end_line") is None:
            continue
        path = mod_path.get(decl["module"])
        if not path or path not in code_lines:
            continue
        lines = code_lines[path]
        start = max(1, int(decl["start_line"]))
        end = min(len(lines), int(decl["end_line"]))
        if any(
            _RANGE_STRUCTURAL_COMMAND_RE.match(lines[line_no - 1])
            for line_no in range(start, end + 1)
        ):
            out.add(path)
    return out


def plan_source_retention(
    decls: list[DeclRange],
    originals: dict[str, str],
    mod_path: dict[str, str],
    retain_exact_files: set[str] | None = None,
) -> dict[str, Any]:
    """Return the unified standalone/LeanPool source-retention policy.

    Source-contract files are retained byte-for-byte and seed every declaration
    they own. Environment commands remain in the stripped source and root their
    declaration-level dependencies. Structural-range and duplicate-name
    collisions are reported for auditing, but the line-safe stripper handles
    them without whole-file declaration roots.
    """

    retain_exact_files = set(retain_exact_files or ())
    unknown_exact_files = retain_exact_files - set(originals)
    if unknown_exact_files:
        raise ValueError(
            "normalization reported unknown retained files: "
            + ", ".join(sorted(unknown_exact_files))
        )
    retained_modules = {
        module for module, path in mod_path.items() if path in retain_exact_files
    }
    unmapped_exact_files = retain_exact_files - set(mod_path.values())
    if unmapped_exact_files:
        raise ValueError(
            "cannot seed declarations for retained files: "
            + ", ".join(sorted(unmapped_exact_files))
        )

    attribute_command_files = detect_attribute_command_files(originals)
    opaque_environment_files = detect_opaque_environment_command_files(
        originals
    )
    environment_command_files = (
        attribute_command_files | opaque_environment_files
    )
    attribute_command_roots = detect_attribute_command_roots(decls, originals)
    opaque_environment_roots = detect_opaque_environment_command_roots(
        decls, originals, mod_path
    )
    structural_range_files = detect_structural_range_files(
        decls, originals, mod_path
    )
    duplicate_provider_files = detect_duplicate_qualified_declaration_files(
        originals
    )

    return {
        "retain_exact_files": retain_exact_files,
        "retain_exact_modules": retained_modules,
        "retain_exact_declarations": {
            decl["name"]
            for decl in decls
            if decl["module"] in retained_modules
        },
        "preserve_command_files": environment_command_files,
        "environment_command_roots": (
            attribute_command_roots | opaque_environment_roots
        ),
        "attribute_command_roots": attribute_command_roots,
        "opaque_environment_command_roots": opaque_environment_roots,
        "environment_command_files": environment_command_files,
        "attribute_command_files": attribute_command_files,
        "opaque_environment_command_files": opaque_environment_files,
        "structural_range_files": structural_range_files,
        "duplicate_provider_files": duplicate_provider_files,
    }


# `@[instance] def foo : C := …` — instance declared via attribute on a `def`
# rather than the `instance` keyword (the keyword form is caught by _KIND_KW_RE).
_INSTANCE_ATTR_RE = re.compile(r"@\[[^\]]*\binstance\b[^\]]*\]")


def detect_force_keep_kinds(
    decls: list[DeclRange],
    originals: dict[str, str],
    mod_path: dict[str, str],
) -> set[str]:
    """Return project decls that must be force-kept because their dependents
    reach them WITHOUT a name reference: typeclass `instance`s (resolved by
    synthesis) and custom-syntax decls (`macro`/`elab`/`syntax`/`notation`/
    `macro_rules`, invoked as syntax). Both are invisible to the name-based
    dependency closure, so dropping one can break elaboration without naming a
    missing constant.

    Detected from the declaration's source header (keyword behind any
    attributes/modifiers), plus the `@[instance] def` attribute form.
    NOTE: `deriving`-generated anonymous instances, whose range points at the
    host type's `deriving` clause, are not matched here — escalate to a
    Lean-side instance flag if one surfaces."""
    out: set[str] = set()
    for d in decls:
        if d["start_line"] is None:
            continue
        path = mod_path.get(d["module"])
        if not path or path not in originals:
            continue
        lines = originals[path].split("\n")
        start, end = int(d["start_line"]), min(len(lines), int(d["end_line"]))
        span = "\n".join(lines[start - 1:end])
        if _KIND_KW_RE.search(span) or _INSTANCE_ATTR_RE.search(span):
            out.add(d["name"])
    return out


def collect_implicit_dependency_roots(
    decls: list[DeclRange],
    originals: dict[str, str],
    mod_path: dict[str, str],
) -> dict[str, set[str]]:
    """Return environment-mediated roots used by every production strip.

    These declarations may affect elaboration without appearing as ordinary
    name references in a theorem's kernel term.  Keep the policy here so the
    standalone and LeanPool adapters cannot silently diverge.
    """

    return {
        "attributed": detect_automation_tagged(decls, originals, mod_path),
        "implicit_kinds": detect_force_keep_kinds(decls, originals, mod_path)
        | detect_open_hiding_roots(decls, originals),
        "open_namespace_candidates": detect_open_namespace_roots(
            decls, originals
        ),
    }


def compute_keep_set(
    decls: list[DeclRange],
    deps: dict[str, set[str]],
    seed: set[str],
    implicit_roots: set[str] | None = None,
) -> set[str]:
    """Transitive forward-dependency closure = the set of decls to keep.

    Starts from `seed` (the signature theorems + protected defs), plus every
    `meta` declaration (notation/macro/elab machinery — force-kept because it's
    used implicitly at parse time and never shows up in any term's deps), plus
    names in `implicit_roots` whose use is mediated by the environment rather
    than a term reference. Follows `deps` edges (decl -> the project names it
    references) until the worklist drains.

    Names in either root set that are not project declarations are ignored.
    Semantic acceptance, including axiom policy, belongs exclusively to the
    official comparator and never changes this reachability closure.
    """
    name_set = {d["name"] for d in decls}
    keep: set[str] = set()
    work: list[str] = []

    def _add(n: str) -> None:
        if n in name_set and n not in keep:
            keep.add(n)
            work.append(n)

    for s in seed:
        _add(s)
    for n in implicit_roots or set():
        _add(n)
    for d in decls:
        if d["meta"]:
            _add(d["name"])

    while work:
        n = work.pop()
        for dep in deps.get(n, ()):
            _add(dep)
    return keep


def attach_keep(decls: list[DeclRange], keep: set[str]) -> list[DeclRange]:
    """Return decls with a `keep` bool set from membership in `keep`."""
    return [{**d, "keep": d["name"] in keep} for d in decls]


def module_to_relpath(module: str) -> str:
    """Render a Lean module name as its source path without losing quotes.

    Dots inside guillemet-quoted components are literal filename characters,
    and the guillemets themselves are syntax rather than path characters.
    """

    components: list[str] = []
    component: list[str] = []
    quoted = False
    for char in module:
        if char == "«" and not quoted:
            quoted = True
        elif char == "»" and quoted:
            quoted = False
        elif char == "." and not quoted:
            components.append("".join(component))
            component = []
        else:
            component.append(char)
    if quoted:
        raise ValueError(f"unterminated quoted module component: {module!r}")
    components.append("".join(component))
    return "/".join(components) + ".lean"


def resolve_module_paths(modules, original_paths) -> dict[str, str]:
    """Map each module name to its actual source file path.

    The source root varies by repo: some have files at `/testbed/<Mod>/...`,
    others under `/testbed/src/<Mod>/...`. So we don't assume a layout — we index
    every path by its `/`-separated suffixes and look up the module's
    fully-qualified relpath (e.g. `Langlib/Foo/Bar.lean`), which is specific
    enough to be unambiguous. Modules with no matching file are omitted.
    """
    suffix: dict[str, str] = {}
    for p in original_paths:
        segs = p.split("/")
        for i in range(len(segs)):
            suffix.setdefault("/".join(segs[i:]), p)
    out: dict[str, str] = {}
    for m in modules:
        rel = module_to_relpath(m)
        hit = suffix.get(rel)
        if hit:
            out[m] = hit
    return out


def file_has_kept_decl(decl_ranges: list[DeclRange]) -> bool:
    """True if any decl with a real source range in this file is kept.

    A file with no kept decls is a candidate for deletion by the orchestrator
    (guarded by the rebuild gate).
    """
    return any(r["keep"] and r["start_line"] is not None for r in decl_ranges)


def _mask_comments_preserve_layout(
    text: str, *, mask_strings: bool = False
) -> str:
    """Replace comment contents with spaces while preserving newlines/columns.

    This is deliberately layout-preserving: callers can decide whether the
    lines between a command prefix and its declaration are only comment/blank
    trivia without losing the original declaration line numbers. Lean block
    comments nest, and comment markers inside strings are ordinary characters.
    """
    out: list[str] = []
    i = 0
    nest = 0
    in_line = False
    in_string = False
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_line:
            if char == "\n":
                in_line = False
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue
        if nest:
            if char == "/" and nxt == "-":
                nest += 1
                out.extend((" ", " "))
                i += 2
                continue
            if char == "-" and nxt == "/":
                nest -= 1
                out.extend((" ", " "))
                i += 2
                continue
            out.append("\n" if char == "\n" else " ")
            i += 1
            continue
        if in_string:
            out.append(" " if mask_strings and char != "\n" else char)
            if char == "\\" and nxt:
                out.append(" " if mask_strings and nxt != "\n" else nxt)
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue
        if char == "-" and nxt == "-":
            in_line = True
            out.extend((" ", " "))
            i += 2
            continue
        if char == "/" and nxt == "-":
            nest = 1
            out.extend((" ", " "))
            i += 2
            continue
        if char == '"':
            in_string = True
        out.append(" " if mask_strings and in_string else char)
        i += 1
    return "".join(out)


def _attribute_block_start(code_lines: list[str], end_line: int) -> int | None:
    """Return the start of a standalone (possibly multiline) `@[...]` block."""
    if end_line < 1 or not code_lines[end_line - 1].rstrip().endswith("]"):
        return None
    for line_no in range(end_line, 0, -1):
        line = code_lines[line_no - 1]
        if "@[" in line:
            block = "\n".join(code_lines[line_no - 1:end_line])
            return line_no if _ATTR_ONLY_RE.match(block) else None
        if not line.strip():
            return None
    return None


def _prefix_block_start(code_lines: list[str], end_line: int) -> int | None:
    """Return the first line of a possibly multiline `... in` modifier."""
    if end_line < 1 or not re.search(r"\bin\s*$", code_lines[end_line - 1]):
        return None
    for line_no in range(end_line, 0, -1):
        line = code_lines[line_no - 1]
        if not line.strip():
            return None
        if _PREFIX_START_RE.match(line):
            block = "\n".join(code_lines[line_no - 1:end_line])
            return line_no if _PREFIX_IN_RE.match(block) else None
    return None


def _extend_up(code_lines: list[str], start_line: int) -> int:
    """Extend a deletion start upward over decoration lines that belong to this
    decl (standalone `@[attr]` and `... in` command-prefix modifiers), skipping
    blank/comment-only trivia between them. `code_lines` is the same source
    with comments layout-masked. `start_line` and the return are 1-based."""
    s = start_line
    while True:
        j = s - 1
        while j >= 1 and code_lines[j - 1].strip() == "":
            j -= 1  # skip blank and comment-only lines above
        attr_start = _attribute_block_start(code_lines, j)
        prefix_start = _prefix_block_start(code_lines, j)
        if attr_start is not None:
            s = attr_start
        elif prefix_start is not None:
            s = prefix_start
        else:
            break
    return s


def _extend_guarded_diagnostic_up(
    lines: list[str], code_lines: list[str], diagnostic_line: int
) -> int:
    """Include a ``#guard_msgs`` command's expected-message comment.

    Lean encodes expected diagnostics as a doc/block comment immediately above
    ``#guard_msgs in``. Comment masking turns that comment into blank lines, so
    the ordinary modifier walk can find an earlier ``set_option ... in`` but
    otherwise leaves the expected-message comment behind. Only absorb masked
    trivia when ``#guard_msgs`` is the topmost prefix.
    """

    start = _extend_up(code_lines, diagnostic_line)
    if not code_lines[start - 1].lstrip().startswith("#guard_msgs"):
        return start
    j = start - 1
    while j >= 1 and not lines[j - 1].strip():
        j -= 1
    if j < 1 or not lines[j - 1].rstrip().endswith("-/"):
        return start
    for line_no in range(j, 0, -1):
        stripped = lines[line_no - 1].lstrip()
        if stripped.startswith("/--"):
            return line_no
        if not stripped or code_lines[line_no - 1].strip():
            break
    return start


_STRUCTURAL_COMMAND_RE = re.compile(r"^\s*(?:namespace|section|end|mutual)\b")
_STRUCTURAL_OPEN_RE = re.compile(r"^\s*(namespace|section|mutual)\b")
_STRUCTURAL_END_RE = re.compile(r"^\s*end(?:\s|$)")


def _mutual_blocks(code_lines: list[str]) -> list[tuple[int, int]]:
    """Find balanced top-level `mutual ... end` blocks using masked source."""
    stack: list[tuple[str, int]] = []
    blocks: list[tuple[int, int]] = []
    for line_no, line in enumerate(code_lines, 1):
        opened = _STRUCTURAL_OPEN_RE.match(line)
        if opened:
            stack.append((opened.group(1), line_no))
            continue
        if _STRUCTURAL_END_RE.match(line) and stack:
            kind, start = stack.pop()
            if kind == "mutual":
                blocks.append((start, line_no))
    return blocks


def _collapse_blank_runs(text: str) -> str:
    """Collapse runs of 3+ blank lines (left behind by deletions) to one, and
    trim leading/trailing blank lines. Preserves a single trailing newline."""
    text = re.sub(r"\n[ \t]*\n([ \t]*\n)+", "\n\n", text)
    text = text.strip("\n")
    return text + "\n" if text else ""


def strip_source(file_text: str, decl_ranges: list[DeclRange]) -> str:
    """Return `file_text` with every dropped declaration's source span removed.

    `decl_ranges` is the subset of rows for THIS file. Rows that are kept or
    range-less are left in place; a `keep == False` row's lines are deleted
    (extended upward over standalone attribute lines).

    Declaration ranges can OVERLAP — most importantly, a `structure`/`inductive`
    and its auto-generated `casesOn`/`noConfusion`/projection helpers report the
    SAME (or nested) source range, and those helpers are usually dropped while
    the type itself is kept. So **keep wins per line**: a line is deleted only
    if it lies in a drop range AND no kept declaration covers it. A drop decl
    whose own start line is covered by a kept decl is skipped entirely (it's a
    nested/auto-generated sibling of something we're keeping).
    """
    lines = file_text.split("\n")
    code_lines = _mask_comments_preserve_layout(file_text).split("\n")
    n = len(lines)
    keep_mask = [False] * (n + 2)  # 1-based; index 0 unused
    delete = [False] * (n + 2)
    forced_delete = [False] * (n + 2)

    source_rows = [
        r for r in decl_ranges
        if r.get("start_line") is not None and r.get("end_line") is not None
    ]
    kept_source_rows = [r for r in source_rows if r["keep"]]

    def contains_kept_child(outer: DeclRange) -> bool:
        outer_start = (
            int(outer["start_line"]), int(outer.get("start_col") or 0)
        )
        outer_end = (
            int(outer["end_line"]), int(outer.get("end_col") or 0)
        )
        return any(
            outer_start
            <= (int(child["start_line"]), int(child.get("start_col") or 0))
            and (int(child["end_line"]), int(child.get("end_col") or 0))
            <= outer_end
            for child in kept_source_rows
        )

    # Defense in depth for callers that supply an inconsistent keep-set: a
    # source-backed local declaration (for example `outer.key` from a
    # `let rec key`) cannot be retained while deleting the command that
    # lexically contains it. Protect every containing command atomically.
    protected_rows = [r for r in source_rows if contains_kept_child(r)]
    for r in protected_rows:
        s = max(1, int(r["start_line"]))
        e = min(n, int(r["end_line"]))
        for i in range(s, e + 1):
            keep_mask[i] = True

    # `findDeclarationRanges?` can attach `mutual` and/or its closing `end` to
    # only one member's range. Treat the block atomically: retaining one member
    # retains the complete mutual command; dropping all source-backed members
    # removes its opener and closer together.
    for block_start, block_end in _mutual_blocks(code_lines):
        block_rows = [
            r for r in source_rows
            if int(r["start_line"]) <= block_end
            and int(r["end_line"]) >= block_start
        ]
        if not block_rows:
            continue
        if any(r["keep"] for r in block_rows):
            for i in range(block_start, min(n, block_end) + 1):
                keep_mask[i] = True
        else:
            for i in range(block_start, min(n, block_end) + 1):
                forced_delete[i] = True

    for r in decl_ranges:
        if r["keep"] or r["start_line"] is None:
            continue
        s = max(1, int(r["start_line"]))
        e = min(n, int(r["end_line"]))
        if e < s:
            continue
        if keep_mask[s]:
            # start line is protected by an overlapping kept decl -> this is a
            # nested/auto-generated sibling; leave the whole span alone.
            continue
        s = _extend_up(code_lines, s)
        for i in range(s, e + 1):
            if not keep_mask[i]:
                delete[i] = True

    for i in range(1, n + 1):
        if forced_delete[i] and not keep_mask[i]:
            delete[i] = True
        elif _STRUCTURAL_COMMAND_RE.match(code_lines[i - 1]):
            # Declaration ranges occasionally begin at their surrounding
            # namespace/section opener. Never delete one side of a structural
            # scope; dead mutual blocks were handled atomically above.
            delete[i] = False

    kept = [lines[i - 1] for i in range(1, n + 1) if not delete[i]]
    return _collapse_blank_runs("\n".join(kept))



_DIAGNOSTIC_COMMAND_RE = re.compile(
    r"^[ \t]*#(?:check|print|reduce|synth)\b[^\n]*$"
)
_ANONYMOUS_EXAMPLE_TOKEN_RE = re.compile(r"\bexample\b")
_SIMPLE_OPEN_COMMAND_RE = re.compile(
    r"^(?P<indent>[ \t]*)open[ \t]+(?P<names>"
    + _LEAN_COMMAND_NAME_PATTERN
    + r"(?:[ \t]+"
    + _LEAN_COMMAND_NAME_PATTERN
    + r")*)[ \t]*$"
)
_SIMPLE_EXPORT_COMMAND_RE = re.compile(
    r"^[ \t]*export[ \t]+(?P<namespace>"
    + _LEAN_COMMAND_NAME_PATTERN
    + r")[ \t]+\([^\n]*\)[ \t]*$"
)


_ATTRIBUTE_COMMAND_HEAD_RE = re.compile(
    r"^[ \t]*" + _SCOPED_ATTRIBUTE_PREFIX
    + r"attribute\s*\[[^\]]*\]"
)
_VARIABLE_COMMAND_HEAD_RE = re.compile(r"^[ \t]*variable\b")
_INCLUDE_OMIT_COMMAND_HEAD_RE = re.compile(
    r"^[ \t]*(?P<kind>include|omit)\b"
)


def source_may_contain_anonymous_example(file_text: str) -> bool:
    """Return whether Lean should inspect this file for ``example`` commands.

    This is only a cheap candidate filter. Comments and strings are masked;
    false positives are harmless because Lean's official frontend identifies
    the actual outer command ranges. The transformation never uses a textual
    range guess.
    """

    masked = _mask_comments_preserve_layout(file_text, mask_strings=True)
    return bool(_ANONYMOUS_EXAMPLE_TOKEN_RE.search(masked))


def _continued_command_spans(
    code_lines: list[str], head: re.Pattern[str]
) -> dict[int, int]:
    """Return zero-based inclusive spans for simple multiline commands.

    Lean conventionally indents continuation operands more deeply than the
    command keyword. Restricting continuation lines by relative indentation
    keeps a following, independently-indented command out of the span.
    """

    spans: dict[int, int] = {}
    i = 0
    while i < len(code_lines):
        if not head.match(code_lines[i]):
            i += 1
            continue
        indent = len(code_lines[i]) - len(code_lines[i].lstrip())
        j = i + 1
        while j < len(code_lines) and code_lines[j].strip():
            next_indent = len(code_lines[j]) - len(code_lines[j].lstrip())
            if next_indent <= indent:
                break
            j += 1
        spans[i] = j - 1
        i = j
    return spans


def _remove_text_spans(
    text: str, spans: list[tuple[int, int]]
) -> str:
    out = text
    for start, end in sorted(spans, reverse=True):
        # Remove the redundant separator, preserving newlines and indentation.
        if start > 0 and out[start - 1] in " \t":
            while end < len(out) and out[end] in " \t":
                end += 1
        out = out[:start] + out[end:]
    return out


def _resolved_project_references(
    text: str,
    project_declarations: set[str],
    *,
    local_names: set[str] | None = None,
    lexical_namespace: str = "",
) -> set[str]:
    local_names = local_names or set()
    references: set[str] = set()
    for token in _IDENT_RE.findall(text):
        head = token.removeprefix("_root_.").split(".", 1)[0]
        if head in local_names:
            continue
        resolved = _resolve_project_identifier(
            token,
            project_declarations,
            lexical_namespace,
        )
        if resolved is not None:
            references.add(resolved)
    return references


def _outer_binder_spans(
    masked_command: str, payload_start: int
) -> list[tuple[int, int]]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    for index in range(payload_start, len(masked_command)):
        char = masked_command[index]
        if char in pairs:
            stack.append((char, index))
        elif stack and char == pairs[stack[-1][0]]:
            _, start = stack.pop()
            if not stack:
                spans.append((start, index + 1))
    return spans


def _top_level_colon(masked_binder: str) -> int | None:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for index, char in enumerate(masked_binder):
        if char in pairs:
            stack.append(char)
        elif stack and char == pairs[stack[-1]]:
            stack.pop()
        elif char == ":" and len(stack) == 1:
            return index
    return None


def _binder_names(masked_binder: str) -> set[str]:
    colon = _top_level_colon(masked_binder)
    opener = masked_binder[:1]
    if colon is None and opener == "[":
        # `[Class alpha]` is an anonymous instance binder, not a name list.
        return set()
    end = colon if colon is not None else len(masked_binder) - 1
    return set(_IDENT_RE.findall(masked_binder[1:end]))


def _binder_dependency_text(masked_binder: str) -> str:
    colon = _top_level_colon(masked_binder)
    if colon is not None:
        return masked_binder[colon + 1 : -1]
    if masked_binder.startswith("["):
        return masked_binder[1:-1]
    return ""


def _prune_attribute_command(
    command: str,
    project_declarations: set[str],
    deleted_declarations: set[str],
    lexical_namespace: str,
) -> tuple[str, int]:
    masked = _mask_comments_preserve_layout(command, mask_strings=True)
    head = _ATTRIBUTE_COMMAND_HEAD_RE.match(masked)
    if head is None:
        return command, 0
    payload = masked[head.end() :]
    removals: list[tuple[int, int]] = []
    for match in _IDENT_RE.finditer(payload):
        token = match.group(0)
        if token == "in" and not payload[match.end() :].strip():
            continue
        resolved = _resolve_project_identifier(
            token,
            project_declarations,
            lexical_namespace,
        )
        if resolved in deleted_declarations:
            removals.append(
                (head.end() + match.start(), head.end() + match.end())
            )
    if not removals:
        return command, 0

    pruned = "\n".join(
        line.rstrip()
        for line in _remove_text_spans(command, removals).split("\n")
    )
    pruned_masked = _mask_comments_preserve_layout(pruned, mask_strings=True)
    pruned_head = _ATTRIBUTE_COMMAND_HEAD_RE.match(pruned_masked)
    assert pruned_head is not None
    remaining = [
        token
        for token in _IDENT_RE.findall(pruned_masked[pruned_head.end() :])
        if token != "in"
    ]
    return (pruned if remaining else ""), len(removals)


def _prune_variable_command(
    command: str,
    project_declarations: set[str],
    deleted_declarations: set[str],
    live_variable_names: set[str],
    dead_variable_names: set[str],
    lexical_namespace: str,
) -> tuple[str, set[str], set[str], int]:
    masked = _mask_comments_preserve_layout(command, mask_strings=True)
    head = _VARIABLE_COMMAND_HEAD_RE.match(masked)
    if head is None:
        return command, set(), set(), 0

    removals: list[tuple[int, int]] = []
    removed_names: set[str] = set()
    surviving_names: set[str] = set()
    locally_live = set(live_variable_names)
    for start, end in _outer_binder_spans(masked, head.end()):
        binder = masked[start:end]
        names = _binder_names(binder)
        dependencies = _binder_dependency_text(binder)
        dependency_tokens = set(_IDENT_RE.findall(dependencies))
        references_dead_variable = any(
            token.removeprefix("_root_.").split(".", 1)[0]
            in dead_variable_names
            for token in dependency_tokens
        )
        references = _resolved_project_references(
            dependencies,
            project_declarations,
            local_names=locally_live,
            lexical_namespace=lexical_namespace,
        )
        if references_dead_variable or references & deleted_declarations:
            removals.append((start, end))
            removed_names.update(names)
        else:
            surviving_names.update(names)
            locally_live.update(names)

    if not removals:
        return command, removed_names, surviving_names, 0
    pruned = "\n".join(
        line.rstrip()
        for line in _remove_text_spans(command, removals).split("\n")
    )
    pruned_masked = _mask_comments_preserve_layout(pruned, mask_strings=True)
    pruned_head = _VARIABLE_COMMAND_HEAD_RE.match(pruned_masked)
    assert pruned_head is not None
    if not pruned_masked[pruned_head.end() :].strip():
        pruned = ""
    return pruned, removed_names, surviving_names, len(removals)


def _prune_include_omit_command(
    command: str, dead_variable_names: set[str]
) -> tuple[str, int]:
    masked = _mask_comments_preserve_layout(command, mask_strings=True)
    head = _INCLUDE_OMIT_COMMAND_HEAD_RE.match(masked)
    if head is None:
        return command, 0
    payload = masked[head.end() :]
    removals: list[tuple[int, int]] = []
    for match in _IDENT_RE.finditer(payload):
        token = match.group(0)
        if token in dead_variable_names:
            removals.append(
                (head.end() + match.start(), head.end() + match.end())
            )
    if not removals:
        return command, 0
    pruned = "\n".join(
        line.rstrip()
        for line in _remove_text_spans(command, removals).split("\n")
    )
    pruned_masked = _mask_comments_preserve_layout(pruned, mask_strings=True)
    pruned_head = _INCLUDE_OMIT_COMMAND_HEAD_RE.match(pruned_masked)
    assert pruned_head is not None
    remaining = [
        token
        for token in _IDENT_RE.findall(pruned_masked[pruned_head.end() :])
        if token != "in"
    ]
    return (pruned if remaining else ""), len(removals)




def _resolve_project_identifier(
    token: str,
    declarations: set[str],
    lexical_namespace: str = "",
) -> str | None:
    # Universe applications tokenize `Foo.{u}` as `Foo.` and `u`.
    token = token.removeprefix("_root_.").rstrip(".")
    for candidate in _namespace_name_candidates(token, lexical_namespace):
        if candidate in declarations:
            return candidate
    candidates = {
        name
        for name in declarations
        if name.endswith("." + token)
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def remove_dead_commands(
    file_text: str,
    project_declarations: set[str],
    kept_declarations: set[str],
    imported_namespaces: set[str] | None = None,
    *,
    visible_kept_declarations: set[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Remove non-semantic diagnostics and commands with deleted targets.

    ``#check``, ``#print``, ``#reduce``, and ``#synth`` only query or print
    elaborated information. They do not add declarations or mutate the
    environment, so every production strip removes them regardless of target.
    ``#eval`` is deliberately retained because it may execute arbitrary IO.
    Bare ``open`` entries resolving exactly to an exclusively project-owned
    namespace with no live declarations are pruned. Imported and mixed
    namespaces, scoped opens, prefixes, and ambiguous opens remain because they
    can activate notation or elaboration behavior invisible in the declaration
    graph. ``export`` commands whose project-local target
    disappeared are also removed. Persistent ``attribute`` targets and typed
    ``variable`` binders are pruned when they resolve exactly to deleted project
    declarations. Any associated ``include``/``omit`` operands are pruned with
    the dead binder. Ambiguous identifiers fail closed by remaining in the
    source; the warm build then reports the unresolved case instead of guessing.
    """

    lines = file_text.split("\n")
    code_lines = _mask_comments_preserve_layout(file_text).split("\n")
    namespace_contexts, open_contexts = _source_namespace_resolution_contexts(file_text)
    visible_kept = (
        kept_declarations
        if visible_kept_declarations is None
        else visible_kept_declarations
    )
    deleted_declarations = project_declarations - kept_declarations
    imported_namespaces = set(imported_namespaces or ())
    delete_lines: set[int] = set()
    replacements: dict[int, str] = {}
    stats = {
        "diagnostics_removed": 0,
        "open_namespaces_removed": 0,
        "open_declarations_removed": 0,
        "open_commands_removed": 0,
        "exports_removed": 0,
        "attribute_commands_removed": 0,
        "attribute_targets_removed": 0,
        "variable_commands_removed": 0,
        "variable_binders_removed": 0,
        "include_omit_commands_removed": 0,
        "include_omit_names_removed": 0,
    }

    attribute_spans = _continued_command_spans(
        code_lines, _ATTRIBUTE_COMMAND_HEAD_RE
    )
    variable_spans = _continued_command_spans(
        code_lines, _VARIABLE_COMMAND_HEAD_RE
    )
    include_omit_spans = _continued_command_spans(
        code_lines, _INCLUDE_OMIT_COMMAND_HEAD_RE
    )
    diagnostic_spans = _continued_command_spans(
        code_lines, _DIAGNOSTIC_COMMAND_RE
    )
    continuation_lines = {
        line
        for start, end in (
            list(attribute_spans.items())
            + list(variable_spans.items())
            + list(include_omit_spans.items())
            + list(diagnostic_spans.items())
        )
        for line in range(start + 1, end + 1)
    }
    live_variable_names: set[str] = set()
    dead_variable_names: set[str] = set()

    for line_no, code_line in enumerate(code_lines, 1):
        zero_based = line_no - 1
        if zero_based in continuation_lines:
            continue

        if zero_based in attribute_spans:
            end = attribute_spans[zero_based]
            command = "\n".join(lines[zero_based : end + 1])
            pruned, target_count = _prune_attribute_command(
                command,
                project_declarations,
                deleted_declarations,
                namespace_contexts.get(line_no, ""),
            )
            if target_count:
                stats["attribute_targets_removed"] += target_count
                delete_lines.update(range(line_no + 1, end + 2))
                if pruned:
                    replacements[line_no] = pruned
                else:
                    delete_lines.add(line_no)
                    stats["attribute_commands_removed"] += 1
            continue

        if zero_based in variable_spans:
            end = variable_spans[zero_based]
            command = "\n".join(lines[zero_based : end + 1])
            pruned, removed_names, surviving_names, binder_count = (
                _prune_variable_command(
                    command,
                    project_declarations,
                    deleted_declarations,
                    live_variable_names,
                    dead_variable_names,
                    namespace_contexts.get(line_no, ""),
                )
            )
            dead_variable_names.update(removed_names)
            dead_variable_names.difference_update(surviving_names)
            live_variable_names.difference_update(removed_names)
            live_variable_names.update(surviving_names)
            if binder_count:
                stats["variable_binders_removed"] += binder_count
                delete_lines.update(range(line_no + 1, end + 2))
                if pruned:
                    replacements[line_no] = pruned
                else:
                    delete_lines.add(line_no)
                    stats["variable_commands_removed"] += 1
            continue

        if zero_based in include_omit_spans:
            end = include_omit_spans[zero_based]
            command = "\n".join(lines[zero_based : end + 1])
            pruned, name_count = _prune_include_omit_command(
                command, dead_variable_names
            )
            if name_count:
                stats["include_omit_names_removed"] += name_count
                delete_lines.update(range(line_no + 1, end + 2))
                if pruned:
                    replacements[line_no] = pruned
                else:
                    delete_lines.add(line_no)
                    stats["include_omit_commands_removed"] += 1
            continue
        diagnostic = _DIAGNOSTIC_COMMAND_RE.match(code_line)
        if diagnostic:
            start = _extend_guarded_diagnostic_up(
                lines, code_lines, line_no
            )
            delete_lines.update(range(start, diagnostic_spans[zero_based] + 2))
            stats["diagnostics_removed"] += 1
            continue

        selective_open = _SELECTIVE_OPEN_COMMAND_RE.match(code_line)
        if selective_open:
            namespace = selective_open.group("namespace")
            payload = selective_open.group("targets")
            namespace_candidates = _namespace_name_candidates(
                namespace, namespace_contexts.get(line_no, "")
            )
            removals: list[tuple[int, int]] = []
            remaining = 0
            for match in _IDENT_RE.finditer(payload):
                token = match.group(0)
                resolved = next(
                    (
                        f"{candidate}.{token}"
                        for candidate in namespace_candidates
                        if f"{candidate}.{token}" in project_declarations
                    ),
                    None,
                )
                if resolved in deleted_declarations:
                    removals.append(
                        (
                            selective_open.start("targets") + match.start(),
                            selective_open.start("targets") + match.end(),
                        )
                    )
                else:
                    # Kept project targets and imported/unresolved targets must
                    # remain. Ambiguity therefore fails closed into the warm
                    # certification build rather than guessing.
                    remaining += 1
            if removals:
                stats["open_declarations_removed"] += len(removals)
                if remaining:
                    replacements[line_no] = _remove_text_spans(
                        lines[zero_based], removals
                    ).rstrip()
                else:
                    delete_lines.add(line_no)
                    stats["open_commands_removed"] += 1
            continue

        opened = _SIMPLE_OPEN_COMMAND_RE.match(code_line)
        if opened:
            tokens = opened.group("names").split()
            # ``open scoped`` and ``open ... in`` have different semantics;
            # preserve them unless a complete Lean command node models them.
            if tokens and (tokens[0] == "scoped" or tokens[-1] == "in"):
                continue
            retained: list[str] = []
            removed = 0
            for namespace in tokens:
                members, imported = _resolve_namespace_provenance(
                    namespace,
                    project_declarations,
                    imported_namespaces,
                    namespace_contexts.get(line_no, ""),
                    open_contexts.get(line_no, ()),
                )
                if (
                    members
                    and not imported
                    and not (members & visible_kept)
                ):
                    removed += 1
                else:
                    retained.append(namespace)
            if removed:
                stats["open_namespaces_removed"] += removed
                if retained:
                    replacements[line_no] = (
                        f"{opened.group('indent')}open {' '.join(retained)}"
                    )
                else:
                    delete_lines.add(line_no)
                    stats["open_commands_removed"] += 1
            continue

        exported = _SIMPLE_EXPORT_COMMAND_RE.match(code_line)
        if exported:
            members = _resolve_project_namespace_members(
                exported.group("namespace"),
                project_declarations,
                namespace_contexts.get(line_no, ""),
            )
            if members and not (members & kept_declarations):
                delete_lines.add(line_no)
                stats["exports_removed"] += 1

    if not delete_lines and not replacements:
        return file_text, stats

    rendered = [
        replacements.get(line_no, line)
        for line_no, line in enumerate(lines, 1)
        if line_no not in delete_lines
    ]
    return _collapse_blank_runs("\n".join(rendered)), stats


_MODULE_COMPONENT_PATTERN = r"(?:«[^»]+»|[^\s.]+)"
_IMPORT_RE = re.compile(
    r"^[ \t]*(?:(?:public|private|meta)[ \t]+)*import\s+"
    rf"({_MODULE_COMPONENT_PATTERN}(?:\.{_MODULE_COMPONENT_PATTERN})*)",
    re.MULTILINE,
)


def _source_imports(file_text: str) -> list[tuple[str, int, int]]:
    """Return import names and inclusive zero-based source line spans.

    Lean permits a newline (and comments) between `import` and the module.
    Parse the complete masked source so graph visibility, stubs, and import
    removal all agree on those wrapped commands.
    """
    masked = _mask_comments_preserve_layout(file_text, mask_strings=True)
    return [
        (match.group(1), masked.count("\n", 0, match.start()),
         masked.count("\n", 0, match.end()))
        for match in _IMPORT_RE.finditer(masked)
    ]


def import_only_stub(file_text: str) -> str:
    """Return a valid dead-module stub containing only its import preamble.

    Keeping arbitrary non-declaration scaffolding is unsafe when every source
    declaration was dropped: a surviving `variable (p : Parser ...)`,
    `attribute`, or `open scoped` command may reference one of those deleted
    declarations. Preserve only Lean's file preamble commands (`prelude`,
    `module`, and imports). Public imports require the `module` marker, so it
    is retained verbatim.
    """
    kept: list[str] = []
    lines = file_text.split("\n")
    code_lines = _mask_comments_preserve_layout(
        file_text, mask_strings=True
    ).split("\n")
    import_lines = {
        line for _, start, end in _source_imports(file_text)
        for line in range(start, end + 1)
    }
    for line_no, (line, code_line) in enumerate(zip(lines, code_lines, strict=True)):
        # `module` and `prelude` are header commands only at column zero. A
        # declaration body can legally contain an indented term whose identifier
        # is literally `module`; stripping whitespace here used to copy that term
        # into the stub and produce `unexpected identifier; expected command`.
        # Match against comment/string-masked source as well: prose such as
        # "import of the downstream module" inside a module doc comment is not
        # a Lean import command and must never leak into a generated stub.
        if code_line in {"prelude", "module"} or line_no in import_lines:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def plan_stripped_sources(
    originals: dict[str, str],
    rows: list[DeclRange],
    module_paths: dict[str, str],
    *,
    preserve_command_files: set[str] | None = None,
    retain_exact_files: set[str] | None = None,
    anonymous_example_ranges: dict[str, list[DeclRange]] | None = None,
    all_project_declarations: set[str] | None = None,
    imported_namespaces: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Plan the complete declaration/source-command strip without I/O.

    This is the source transformation used by production preprocessing. Keeping
    it pure makes the exact transformation replayable against pinned dependency
    evidence without starting a build or mutating a working repository.
    """

    preserved_commands = set(preserve_command_files or ())
    retained_files = set(retain_exact_files or ())
    examples = {
        path: list(example_rows)
        for path, example_rows in (anonymous_example_ranges or {}).items()
        if path in originals and path not in retained_files
    }
    by_path: dict[str, list[DeclRange]] = {}
    unresolved_drops = 0
    for row in rows:
        if row["start_line"] is None:
            continue
        path = module_paths.get(row["module"])
        if not path or path not in originals:
            if not row["keep"]:
                unresolved_drops += 1
            continue
        by_path.setdefault(path, []).append(row)
    for path, example_rows in examples.items():
        by_path.setdefault(path, []).extend(example_rows)

    desired = dict(originals)
    declaration_drop_files: set[str] = set()
    stubbed_files: list[str] = []
    command_preserved_files: list[str] = []
    for path, file_rows in by_path.items():
        if path in retained_files or all(row["keep"] for row in file_rows):
            continue
        declaration_drop_files.add(path)
        if file_has_kept_decl(file_rows):
            desired[path] = strip_source(originals[path], file_rows)
        elif path in preserved_commands:
            desired[path] = strip_source(originals[path], file_rows)
            command_preserved_files.append(path.removeprefix("/testbed/"))
        else:
            desired[path] = import_only_stub(originals[path])
            stubbed_files.append(module_to_relpath(file_rows[0]["module"]))

    project_declarations = set(all_project_declarations or ())
    project_declarations.update(str(row["name"]) for row in rows)
    kept_declarations = {
        str(row["name"])
        for row in rows
        if row["keep"]
    }
    dead_command_cleanup = {
        "diagnostics_removed": 0,
        "open_namespaces_removed": 0,
        "open_declarations_removed": 0,
        "open_commands_removed": 0,
        "exports_removed": 0,
        "attribute_commands_removed": 0,
        "attribute_targets_removed": 0,
        "variable_commands_removed": 0,
        "variable_binders_removed": 0,
        "include_omit_commands_removed": 0,
        "include_omit_names_removed": 0,
        "files_changed": 0,
    }
    imported = set(imported_namespaces or ())
    # An identically named namespace in an unrelated module cannot make this
    # file's `open` valid. Scope only command cleanup to its import closure;
    # the declaration keep-set and dependency graph remain unchanged.
    import_graph, resolved_paths = build_module_import_graph(
        originals, "/testbed/", known_module_paths=module_paths
    )
    path_modules = {path: module for module, path in resolved_paths.items()}
    kept_by_module: dict[str, set[str]] = {}
    for row in rows:
        if row["keep"]:
            kept_by_module.setdefault(str(row["module"]), set()).add(str(row["name"]))
    # Range-less/generated declarations may have no source module in this
    # plan. Their visibility is unknown, not evidence that the namespace died.
    unmapped_kept = set().union(*(
        names for module, names in kept_by_module.items()
        if module not in resolved_paths
    ))
    for path, content in desired.items():
        if path in retained_files:
            continue
        module = path_modules.get(path)
        visible_kept = (
            unmapped_kept.union(*(
                kept_by_module.get(imported_module, set())
                for imported_module in _closure(import_graph, {module})
            ))
            if module is not None else kept_declarations
        )
        cleaned, cleanup = remove_dead_commands(
            content,
            project_declarations,
            kept_declarations,
            imported,
            visible_kept_declarations=visible_kept,
        )
        if cleaned != content:
            dead_command_cleanup["files_changed"] += 1
            desired[path] = cleaned
        for key, value in cleanup.items():
            dead_command_cleanup[key] += value

    changed = sum(originals.get(path) != content for path, content in desired.items())
    return desired, {
        "stripped_files": len(declaration_drop_files),
        "source_files_changed": changed,
        "stub_files": sorted(stubbed_files),
        "command_preserved_files": sorted(command_preserved_files),
        "files_written_this_iter": changed,
        "unresolved_drops": unresolved_drops,
        "anonymous_examples_removed": sum(len(rows) for rows in examples.values()),
        "anonymous_example_files": sorted(
            path.removeprefix("/testbed/")
            for path, example_rows in examples.items()
            if example_rows
        ),
        "dead_command_cleanup": dead_command_cleanup,
    }


def remove_imports_of(file_text: str, deleted_modules: set[str]) -> str:
    """Drop `import <module>` lines (incl. `public import`) for any module in
    `deleted_modules`. Used to prune hub files after deleting modules."""
    if not deleted_modules:
        return file_text
    removed_lines = {
        line for module, start, end in _source_imports(file_text)
        if module in deleted_modules
        for line in range(start, end + 1)
    }
    return "\n".join(
        line for line_no, line in enumerate(file_text.split("\n"))
        if line_no not in removed_lines
    )


# ---------------------------------------------------------------------------
# Module-level (import-graph) stripping — used for grind-heavy repos, where
# declaration-level stripping is unsafe (bare `grind` searches the whole
# imported environment, so its dependencies are not nameable). Keeping whole
# modules + their full import-closure preserves each kept module's exact
# build-time environment, so grind cannot break.
# ---------------------------------------------------------------------------

def derive_source_root(mod_path: dict[str, str]) -> str | None:
    """Infer the source-root prefix (e.g. `/testbed/src/`) from any
    module->path pair, so we can name modules for files (incl. import hubs)
    that have no declarations and thus aren't in the decl graph."""
    roots = derive_source_roots(mod_path)
    return roots[0] if roots else None


def derive_source_roots(mod_path: dict[str, str]) -> list[str]:
    """Infer every source-root prefix represented by resolved module paths."""

    roots: set[str] = set()
    for mod, path in mod_path.items():
        tail = module_to_relpath(mod)
        if path.endswith("/" + tail):
            roots.add(path[: -len(tail)])
        elif path.endswith(tail):
            roots.add(path[: -len(tail)])
    return sorted(roots, key=lambda root: (-len(root), root))


def build_module_import_graph(
    originals: dict[str, str],
    source_root: str,
    *,
    known_module_paths: dict[str, str] | None = None,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return the project import graph across every resolved source root.

    graph maps each module to the set of project modules it imports (incl. via
    `public import`). ``known_module_paths`` handles layouts where the entry
    module is at the repository root but library modules live under ``src/``.
    Longest-root matching prevents those files from being named ``src.Foo``.
    Covers import-hub modules that carry no declarations.
    """

    roots = {source_root}
    roots.update(derive_source_roots(known_module_paths or {}))
    imported_by_relpath: dict[str, set[str]] = {}
    imports_by_path = {
        path: {module for module, _, _ in _source_imports(source)}
        for path, source in originals.items()
    }
    for imports in imports_by_path.values():
        for imported in imports:
            imported_by_relpath.setdefault(
                module_to_relpath(imported), set()
            ).add(imported)

    mod_to_path: dict[str, str] = {}
    for path in originals:
        if not path.endswith(".lean"):
            continue
        matching_roots = [root for root in roots if path.startswith(root)]
        if not matching_roots:
            continue
        root = max(matching_roots, key=len)
        relative_path = path[len(root):]
        rendered_candidates = imported_by_relpath.get(relative_path, set())
        mod = (
            next(iter(rendered_candidates))
            if len(rendered_candidates) == 1
            else relative_path[:-len(".lean")].replace("/", ".")
        )
        mod_to_path[mod] = path
    for mod, path in (known_module_paths or {}).items():
        if path in originals:
            mod_to_path = {
                existing_mod: existing_path
                for existing_mod, existing_path in mod_to_path.items()
                if existing_path != path or existing_mod == mod
            }
            mod_to_path[mod] = path
    mods = set(mod_to_path)
    graph: dict[str, set[str]] = {}
    for mod, path in mod_to_path.items():
        graph[mod] = imports_by_path[path] & mods
    return graph, mod_to_path


def _closure(graph: dict[str, set[str]], seed: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(seed)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(graph.get(m, ()))
    return seen


def module_keep_set(
    graph: dict[str, set[str]], sig_modules: set[str]
) -> set[str]:
    """Modules to keep: the downward import-closure of the signature modules —
    everything they transitively need to build. This is import-closed (every
    kept module's imports are also kept), so deleting the complement leaves no
    dangling imports among kept modules. The build root is NOT in this set
    (nothing imports it); the caller regenerates it to import the kept set.

    We deliberately do NOT add upward "ancestors": a dead module typically
    imports shared base modules that ARE in the closure, which would make it a
    false ancestor and keep almost everything."""
    return _closure(graph, sig_modules & set(graph))


def declaration_dependency_closure(
    decls: list[DeclRange],
    deps: dict[str, set[str]],
    roots: set[str],
) -> set[str]:
    """Return the project-declaration closure of ``roots`` only.

    This intentionally differs from :func:`compute_keep_set`: global meta,
    attribute, instance, and syntax roots are *not* added. Module pruning runs
    before those implicit roots are discovered, so an otherwise dead
    module cannot keep itself alive merely by containing ``@[simp]`` lemmas or
    instances. Required elaboration environments remain protected by the
    source import closure of every module reached here, followed by a clean
    build and protected-signature gate in the orchestrator.
    """

    names = {decl["name"] for decl in decls}
    seen: set[str] = set()
    stack = [name for name in roots if name in names]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(dep for dep in deps.get(name, ()) if dep in names)
    return seen


def _ensure_imports(file_text: str, modules: set[str]) -> str:
    """Add missing ordinary imports without otherwise rewriting a hub file."""

    if not modules:
        return file_text
    lines = file_text.splitlines()
    existing: set[str] = set()
    last_import: int | None = None
    for index, line in enumerate(lines):
        match = _IMPORT_RE.match(line)
        if match:
            existing.add(match.group(1))
            last_import = index
    missing = sorted(modules - existing)
    if not missing:
        return file_text

    additions = [f"import {module}" for module in missing]
    if last_import is not None:
        insert_at = last_import + 1
    else:
        # ``module``/``prelude`` must remain the first source command. Legacy
        # files normally already have imports; this branch mainly supports
        # generated or test hubs with none.
        header_commands = [
            index
            for index, line in enumerate(lines)
            if line in {"module", "prelude"}
        ]
        insert_at = max(header_commands, default=-1) + 1
    lines[insert_at:insert_at] = additions
    rendered = "\n".join(lines)
    return rendered + ("\n" if file_text.endswith("\n") else "")


def remap_decl_ranges_after_source_rewrite(
    decls: list[DeclRange],
    originals: dict[str, str],
    rewritten: dict[str, str],
    module_paths: dict[str, str],
) -> list[DeclRange]:
    """Move declaration ranges across line-only source rewrites.

    Module pruning can replace one umbrella import with many explicit imports.
    Declaration graph ranges were collected before that rewrite, so every
    later declaration in the file shifts. Map retained source lines through
    the exact diff and fail closed if either end of a declaration range was
    changed rather than merely moved.
    """

    line_maps: dict[str, dict[int, int]] = {}
    for path, source in rewritten.items():
        original = originals.get(path)
        if original is None or original == source:
            continue
        mapping: dict[int, int] = {}
        matcher = difflib.SequenceMatcher(
            None,
            original.splitlines(),
            source.splitlines(),
            autojunk=False,
        )
        for original_start, rewritten_start, size in matcher.get_matching_blocks():
            for offset in range(size):
                mapping[original_start + offset + 1] = rewritten_start + offset + 1
        line_maps[path] = mapping

    remapped: list[DeclRange] = []
    for decl in decls:
        start = decl.get("start_line")
        end = decl.get("end_line")
        path = module_paths.get(decl["module"])
        mapping = line_maps.get(path or "")
        if start is None or end is None or mapping is None:
            remapped.append(decl)
            continue
        new_start = mapping.get(int(start))
        new_end = mapping.get(int(end))
        if new_start is None or new_end is None:
            raise ValueError(
                "source rewrite changed declaration text for "
                f"{decl['name']} at {path}:{start}-{end}"
            )
        remapped.append(
            {**decl, "start_line": new_start, "end_line": new_end}
        )
    return remapped


def plan_protected_module_prune(
    originals: dict[str, str],
    decls: list[DeclRange],
    deps: dict[str, set[str]],
    protected_names: set[str],
    entry_module: str,
    *,
    extra_keep_modules: set[str] | None = None,
    known_module_paths: dict[str, str] | None = None,
    collapse_entry_import_hubs: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Plan member-local module deletion from protected declaration roots.

    ``originals`` must contain only one standalone member's Lean sources. The
    live declaration closure determines candidate owner modules; their complete
    source-import closure is retained to preserve implicit elaboration inputs.
    The entry module is always kept. When it is a declaration-free umbrella,
    imports of deleted branches are removed and imports of protected owner
    modules are ensured directly, so an obsolete re-export cannot make a whole
    branch live.

    The returned source mapping omits deleted files and includes deterministic
    import rewrites. Applying and certifying the plan is deliberately left to
    the container orchestrator.
    """

    entry_tail = module_to_relpath(entry_module)
    entry_paths = [
        path
        for path in originals
        if path == entry_tail or path.endswith("/" + entry_tail)
    ]
    if len(entry_paths) != 1:
        raise ValueError(
            f"expected exactly one source for entry module {entry_module!r}, "
            f"found {entry_paths}"
        )
    entry_path = entry_paths[0]
    source_root = entry_path[: -len(entry_tail)]
    graph, module_paths = build_module_import_graph(
        originals,
        source_root,
        known_module_paths=known_module_paths,
    )
    if entry_module not in graph:
        raise ValueError(f"entry module {entry_module!r} absent from source graph")

    owner_by_name = {decl["name"]: decl["module"] for decl in decls}
    missing = sorted(protected_names - set(owner_by_name))
    if missing:
        raise ValueError(f"protected declarations have no source owner: {missing}")

    live_declarations = declaration_dependency_closure(
        decls,
        deps,
        protected_names,
    )
    protected_modules = {owner_by_name[name] for name in protected_names}
    live_modules = {owner_by_name[name] for name in live_declarations}
    explicit_keep_modules = set(extra_keep_modules or ())
    unknown_keep_modules = explicit_keep_modules - set(graph)
    if unknown_keep_modules:
        raise ValueError(
            "explicit keep modules absent from source graph: "
            f"{sorted(unknown_keep_modules)}"
        )
    declaration_modules = set(owner_by_name.values())
    elided_entry_import_hubs: set[str] = set()
    graph_for_keep = graph
    if collapse_entry_import_hubs and entry_module in live_modules:
        elided_entry_import_hubs = {
            imported
            for imported in graph.get(entry_module, set())
            if imported not in declaration_modules
            and imported not in explicit_keep_modules
            and bool(graph.get(imported))
        }
        if elided_entry_import_hubs:
            graph_for_keep = {
                module: set(imports) for module, imports in graph.items()
            }
            graph_for_keep[entry_module] -= elided_entry_import_hubs

    keep_modules = module_keep_set(
        graph_for_keep, live_modules | explicit_keep_modules
    )
    keep_modules.add(entry_module)
    deleted_modules = set(graph) - keep_modules

    planned = {
        path: source
        for module, path in module_paths.items()
        if module in keep_modules
        for source in [originals[path]]
    }
    rewritten_modules: set[str] = set()
    for module in sorted(keep_modules):
        path = module_paths.get(module)
        if path is None:
            continue
        original_ended_with_newline = planned[path].endswith("\n")
        updated = remove_imports_of(planned[path], deleted_modules)
        # A declaration-free umbrella is a presentation/build root, not a
        # semantic root. Make its protected exports explicit after removing
        # dead intermediary hubs.
        if module == entry_module and entry_module not in live_modules:
            updated = _ensure_imports(
                updated, protected_modules - {entry_module}
            )
        elif module == entry_module and elided_entry_import_hubs:
            updated = _ensure_imports(updated, live_modules - {entry_module})
        if original_ended_with_newline and updated and not updated.endswith("\n"):
            updated += "\n"
        if updated != planned[path]:
            rewritten_modules.add(module)
            planned[path] = updated

    report: dict[str, Any] = {
        "schema": "protected_root_module_prune_v1",
        "entry_module": entry_module,
        "protected_modules": sorted(protected_modules),
        "explicit_keep_modules": sorted(explicit_keep_modules),
        "protected_declarations": sorted(protected_names),
        "live_declaration_count": len(live_declarations),
        "live_declarations": sorted(live_declarations),
        "live_modules": sorted(live_modules),
        "kept_modules": sorted(keep_modules),
        "deleted_modules": sorted(deleted_modules),
        "deleted_files": sorted(
            module_paths[module] for module in deleted_modules
        ),
        "rewritten_modules": sorted(rewritten_modules),
        "entry_is_semantic_root": entry_module in live_modules,
        "collapse_entry_import_hubs": collapse_entry_import_hubs,
        "elided_entry_import_hubs": sorted(elided_entry_import_hubs),
    }
    return planned, report
