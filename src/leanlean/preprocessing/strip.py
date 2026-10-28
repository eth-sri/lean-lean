"""Pure source-deletion logic for the closures pinned by published holdout datasets.

Preprocessing uses the separately pinned lean-strip implementation.
strip_source removes unretained declaration spans while preserving scaffolding.
Ranges are 1-based and inclusive: leading docstrings belong to the span, but
standalone attributes above it must be removed separately. Declarations without
source ranges are never deleted.
"""

from __future__ import annotations

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



_PRIVATE_DECL_NAME_RE = re.compile(
    r"^_private\.(?:(?:«[^»]*»|[^.«»]+)\.)+?[0-9]+\.(.+)$"
)



_OPEN_HIDING_RE = re.compile(
    r"^\s*open\s+([^\n]*?)\s+hiding\s+([^\n]*)$", re.MULTILINE
)



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



# Lean's source-range API occasionally attaches a surrounding scope command to
# the first or last declaration in that scope. Record these files for audit;
# ``strip_source`` protects namespace/section commands line-wise and handles
# mutual blocks atomically.
_RANGE_STRUCTURAL_COMMAND_RE = re.compile(
    r"^[ \t]*(?:namespace|section|end|mutual)\b", re.MULTILINE
)



# `@[instance] def foo : C := …` — instance declared via attribute on a `def`
# rather than the `instance` keyword (the keyword form is caught by _KIND_KW_RE).
_INSTANCE_ATTR_RE = re.compile(r"@\[[^\]]*\binstance\b[^\]]*\]")



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
