"""Internal .olean reader for the canonical preprocessing dependency graph.

Lean loads compiled declarations and ranges; this is one evidence layer, never
an independently published dependency graph. Use dependency_graph for graphs.
"""
from __future__ import annotations

import logging
import shlex
from typing import Any

logger = logging.getLogger(__name__)

# Lean 4 script that emits the project's declaration dependency graph, so the
# keep-set (transitive forward closure of the seed signatures) can be computed
# in Python (see lean_strip.compute_keep_set — testable, and re-expandable
# during the rebuild-recovery loop). Emits two record kinds:
#
#   DECL\tname\tmodule\tstartLine\tstartCol\tendLine\tendCol\tkind\tmetaFlag
#   DEP\tname\tdep1 dep2 dep3 ...
#
# A decl with no source range (auto-generated .rec/.ctor/.match_n/etc.) emits
# "-" for the four range fields and is ignored for deletion by the caller.
#
# metaFlag is "meta" if the decl is parsing/elaboration machinery (its type
# mentions Lean.ParserDescr / Macro / Syntax — i.e. a `notation`/`macro`/`elab`
# declaration). Such decls are force-kept: they're used implicitly at parse
# time, never appear in any term's foldConsts, and dropping them yields
# non-recoverable "elaboration function ... not implemented" errors.
#
# DEP edges are the project-local names appearing in a decl's type/value Expr
# (via Expr.foldConsts — same primitive as _DUMP_SIGS_LEAN), plus structural
# edges (ctor->inductive, rec->family, inductive->ctors) so seeding any member
# pulls its whole family.
#
# findDeclarationRanges?'s command range INCLUDES the leading docstring, so
# deleting a drop-decl's [start,end] line span removes its docstring too, and
# keeping a decl keeps its docstring — no orphaning. Standalone `@[attr]` lines
# above the keyword are NOT in the range; the caller extends drop-deletions
# upward over them.
_DUMP_KEEP_LEAN = r"""
import Lean
open Lean Meta

-- Generated proof constants (notably ``._proof_N``) are implementation
-- details of elaboration.  If they are traversed directly, their source owner
-- can create a false dependency on an unrelated earlier command.  Erase proof
-- subterms in the temporary analysis expression before collecting constants;
-- source/.ilean dependency layers still retain explicitly named proof lemmas.
private partial def eraseProofTerms (expression : Expr) : MetaM Expr := do
  try
    if ← Meta.isProof expression then
      let type ← eraseProofTerms (← Meta.inferType expression)
      return ← Meta.mkSorry type true
  catch _ =>
    pure ()
  match expression with
  | .forallE name domain body binderInfo =>
      let domain' ← eraseProofTerms domain
      Meta.withLocalDecl name binderInfo domain' fun fvar => do
        let body' ← eraseProofTerms (body.instantiate1 fvar)
        return .forallE name domain' (body'.abstract #[fvar]) binderInfo
  | .lam name domain body binderInfo =>
      let domain' ← eraseProofTerms domain
      Meta.withLocalDecl name binderInfo domain' fun fvar => do
        let body' ← eraseProofTerms (body.instantiate1 fvar)
        return .lam name domain' (body'.abstract #[fvar]) binderInfo
  | .letE name type value body nondep =>
      let type' ← eraseProofTerms type
      let value' ← eraseProofTerms value
      Meta.withLetDecl name type' value' fun fvar => do
        let body' ← eraseProofTerms (body.instantiate1 fvar)
        return .letE name type' value' (body'.abstract #[fvar]) nondep
  | .app function argument =>
      return .app
        (← eraseProofTerms function)
        (← eraseProofTerms argument)
  | .mdata _ body =>
      eraseProofTerms body
  | .proj typeName index body =>
      return .proj typeName index (← eraseProofTerms body)
  | _ =>
      return expression

private def proofInsensitiveConsts
    (ppCtx : PPContext) (expression : Expr) : IO (Array Name) :=
  ppCtx.runMetaM do
    return (← eraseProofTerms expression).foldConsts #[] fun name names =>
      names.push name

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let (moduleArgs, importArgs) ←
    match args with
    | ["--modules-file", path] => do
      let content ← IO.FS.readFile path
      let modules := content.splitOn "\n" |>.filter (fun s => !s.isEmpty)
      pure (modules, modules)
    | ["--modules-file", projectPath, "--roots-file", rootsPath] => do
      let projectContent ← IO.FS.readFile projectPath
      let rootsContent ← IO.FS.readFile rootsPath
      let modules := projectContent.splitOn "\n" |>.filter (fun s => !s.isEmpty)
      let roots := rootsContent.splitOn "\n" |>.filter (fun s => !s.isEmpty)
      pure (modules, roots)
    | _ => pure (args, args)
  let imports := (importArgs.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let loadedModules := env.header.moduleNames
  let projModNames : NameHashSet :=
    moduleArgs.foldl (fun names module =>
      let moduleName := module.toName
      if loadedModules.contains moduleName then names.insert moduleName else names) {}
  let auditOpts := Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  let ppCtx : PPContext := { env := env, opts := auditOpts }

  -- Types that mark a decl as parsing/elaboration machinery (notation/macro).
  let metaConsts : List Name :=
    [`Lean.ParserDescr, `Lean.TrailingParserDescr, `Lean.Macro,
     `Lean.MacroM, `Lean.Syntax, `Lean.TSyntax, `Lean.ParserDescr.Parser]

  let coreCtx : Core.Context := { fileName := "<dump_keep>", fileMap := default }
  let coreState : Core.State := { env := env }
  let action : CoreM (Array (Name × Name × ConstantInfo × String) × NameHashSet × Std.HashSet Name) := do
    -- Namespace commands do not have declaration nodes. Record every namespace
    -- prefix backed by at least one declaration outside the selected project
    -- modules. The source cleanup uses this provenance to distinguish a
    -- project-owned namespace from a project extension of an imported namespace
    -- such as `Topology` or `Function`.
    let mut importedNamespaces : Std.HashSet Name := {}
    for (name, _) in env.constants.toList do
      let isProject := match env.getModuleIdxFor? name with
        | none => false
        | some modIdx =>
          let modName := env.header.moduleNames[modIdx]!
          projModNames.contains modName
      if !isProject then
        let mut namespaceName := name.getPrefix
        while !namespaceName.isAnonymous do
          importedNamespaces := importedNamespaces.insert namespaceName
          namespaceName := namespaceName.getPrefix

    -- Pass 1: nodes = project decls that have a SOURCE RANGE. This is the key
    -- filter: it INCLUDES `private`/internal declarations (which carry a range
    -- and are referenced through dependency chains we must follow), while
    -- excluding compiler-generated decls (`.rec`/`.eq_1`/`.match_n`/...) which
    -- have no range. Filtering on `isInternal` instead (as before) silently
    -- dropped every private helper, severing dependency paths and making the
    -- closure badly incomplete.
    let mut nodes : Array (Name × Name × ConstantInfo × String) := #[]
    let mut projNameSet : NameHashSet := {}
    for (name, cinfo) in env.constants.toList do
      match env.getModuleIdxFor? name with
      | none => pure ()
      | some modIdx =>
        let modName := env.header.moduleNames[modIdx]!
        if projModNames.contains modName then
          match ← findDeclarationRanges? name with
          | some dr =>
            let r := dr.range
            let rangeStr := s!"{r.pos.line}\t{r.pos.column}\t{r.endPos.line}\t{r.endPos.column}"
            nodes := nodes.push (name, modName, cinfo, rangeStr)
            projNameSet := projNameSet.insert name
          | none => pure ()

    return (nodes, projNameSet, importedNamespaces)
  let ((nodes, projNameSet, importedNamespaces), _) ← action.toIO coreCtx coreState

  -- Pass 2: emit DECL + DEP for each node.
  for (name, modName, cinfo, rangeStr) in nodes do
    let kind := match cinfo with
      | .thmInfo _    => "theorem"
      | .defnInfo _   => "def"
      | .axiomInfo _  => "axiom"
      | .opaqueInfo _ => "opaque"
      | .ctorInfo _   => "ctor"
      | .recInfo _    => "rec"
      | .inductInfo _ => "inductive"
      | .quotInfo _   => "quot"
    let isMeta := cinfo.type.foldConsts false (fun n acc => acc || metaConsts.contains n)
    let metaFlag := if isMeta then "meta" else "normal"
    IO.println s!"DECL\t{name}\t{modName}\t{rangeStr}\t{kind}\t{metaFlag}"
    let isProjectConstant (n : Name) : Bool :=
      match env.getModuleIdxFor? n with
      | none => false
      | some modIdx =>
        let ownerModule := env.header.moduleNames[modIdx]!
        projModNames.contains ownerModule
    let rec findSourceOwner? (n : Name) : Option Name :=
      if projNameSet.contains n then
        some n
      else if n.isAnonymous then
        none
      else
        findSourceOwner? n.getPrefix
    let findRefs (names : Array Name) (acc : Array Name) : Array Name :=
      names.foldl (fun acc n =>
        if !isProjectConstant n then
          acc
        else
          match findSourceOwner? n with
          | some owner =>
            if owner != name then acc.push owner else acc
          | none => acc) acc
    let mut refs := findRefs (← proofInsensitiveConsts ppCtx cinfo.type) #[]
    refs ← match cinfo with
      | .defnInfo val   => do
        pure <| findRefs (← proofInsensitiveConsts ppCtx val.value) refs
      | .thmInfo val    => do
        pure <| findRefs (← proofInsensitiveConsts ppCtx val.value) refs
      | .opaqueInfo val => do
        pure <| findRefs (← proofInsensitiveConsts ppCtx val.value) refs
      | _               => pure refs
    refs := match cinfo with
      | .ctorInfo val   => refs.push val.induct
      | .recInfo val    => refs ++ val.all.toArray
      | .inductInfo val => refs ++ val.ctors.toArray
      | _               => refs
    -- One record per edge. Lean names may contain quoted components with
    -- spaces (for example ``«Adic spaces»``), so a whitespace-separated
    -- dependency payload is not a lossless wire format.
    for ref in refs do
      IO.println s!"DEP\t{name}\t{ref}"
  for ns in importedNamespaces do
    IO.println s!"EXTNS\t{ns}"
""".strip()

# Path inside the container where we drop the keep-closure Lean script.
_DUMP_KEEP_PATH = "/tmp/_dump_keep.lean"
_DUMP_KEEP_MODULES_PATH = "/tmp/_dump_keep_modules.txt"

# One emitted declaration: name, module, source range, kind, meta flag.
# start_line/.../end_col are None for range-less auto-generated decls.
DeclRange = dict[str, Any]


def _install_dump_keep_script(env: "Environment") -> None:
    """Copy the keep-graph Lean script into the container."""
    cmd = f"cat > {_DUMP_KEEP_PATH} << 'LEANDUMPKEEP_EOF'\n{_DUMP_KEEP_LEAN}\nLEANDUMPKEEP_EOF"
    env.execute(cmd)


class DeclGraphExtractionError(RuntimeError):
    """Keep the subprocess failure distinct from a genuinely empty graph."""

    def __init__(self, *, returncode, timed_out, diagnostics, declarations):
        self.details = {
            "returncode": returncode, "timed_out": bool(timed_out),
            "diagnostics": diagnostics, "partial_declarations": declarations,
            "failure_kind": "timeout" if timed_out else "killed" if returncode in (137, -9) else "process_error",
        }
        super().__init__(
            f"Lean graph extraction {self.details['failure_kind']} "
            f"(rc={returncode}, partial declarations={declarations}): {diagnostics[-2000:]}"
        )


def read_olean_declarations(
    env: "Environment", modules: list[str] | None = None,
    *, timeout: int = 3600, extra_lean_path: str | None = None,
    import_roots: list[str] | None = None,
    strict: bool = True,
) -> tuple[list[DeclRange], dict[str, set[str]], set[str]]:
    """Run the keep-graph Lean script; return declarations, edges, namespaces.

    `decls` is one DeclRange per project decl (with a `meta` bool). `deps` maps
    each decl name to the set of project-local names it references.
    `imported_namespaces` contains exact namespace prefixes with at least one
    declaration owned outside the selected project modules. The caller computes
    the keep-set in Python (lean_strip.compute_keep_set).

    `modules` restricts the scan to a specific module list (e.g. one LeanPool
    subproject's modules); None = discover all project modules. When
    `import_roots` is supplied, only those roots are imported and `modules` is
    an allowlist of project modules in their transitive import closure.

    Requires `lake build` to have succeeded so .olean files exist.
    Raises DeclGraphExtractionError if the query fails. Historical callers may
    explicitly opt into the permissive empty-result behavior with strict=False.
    """
    _install_dump_keep_script(env)
    if modules is None:
        from leanlean.benchmarks.leanlean import _discover_modules
        modules = _discover_modules(env)
    if not modules:
        logger.warning("No project modules discovered — keep-graph skipped")
        return [], {}, set()
    decls: list[DeclRange] = []
    deps: dict[str, set[str]] = {}
    imported_namespaces: set[str] = set()
    diagnostics: list[str] = []

    def consume_line(line: str) -> None:
        line = line.rstrip("\r\n")
        parts = line.split("\t")
        if parts and parts[0] == "DECL" and len(parts) == 9:
            _, name, module, sl, sc, el, ec, kind, meta = parts

            def _int(x: str) -> int | None:
                return None if x == "-" else int(x)

            try:
                decls.append({
                    "name": name, "module": module,
                    "start_line": _int(sl), "start_col": _int(sc),
                    "end_line": _int(el), "end_col": _int(ec),
                    "kind": kind, "meta": meta == "meta",
                })
            except ValueError:
                diagnostics.append(line)
        elif parts and parts[0] == "DEP" and len(parts) == 3:
            deps.setdefault(parts[1], set()).add(parts[2])
        elif parts and parts[0] == "EXTNS" and len(parts) == 2:
            imported_namespaces.add(parts[1])
        elif line:
            diagnostics.append(line)
        if len(diagnostics) > 200:
            del diagnostics[:-200]

    stream = getattr(env, "execute_stream", None)
    write_bytes = getattr(env, "write_file_bytes", None)
    if import_roots is not None and not callable(write_bytes):
        raise RuntimeError("root-scoped graph extraction requires write_file_bytes")
    if callable(stream) and callable(write_bytes):
        payload = ("\n".join(modules) + "\n").encode("utf-8")
        write_bytes(_DUMP_KEEP_MODULES_PATH, payload)
        if import_roots is None:
            lean_command = (
                f"lean --run {_DUMP_KEEP_PATH} "
                f"--modules-file {_DUMP_KEEP_MODULES_PATH}"
            )
        else:
            roots_path = _DUMP_KEEP_MODULES_PATH + ".roots"
            write_bytes(roots_path, ("\n".join(import_roots) + "\n").encode("utf-8"))
            lean_command = (
                f"lean --run {_DUMP_KEEP_PATH} "
                f"--modules-file {_DUMP_KEEP_MODULES_PATH} "
                f"--roots-file {roots_path}"
            )
        cmd = (
            "cd /testbed && lake env sh -c "
            + shlex.quote(
                (
                    "LEAN_PATH=" + shlex.quote(extra_lean_path) + ":$LEAN_PATH "
                    if extra_lean_path
                    else ""
                )
                + "exec " + lean_command
            )
        )
        result = stream(
            cmd,
            timeout=timeout,
            on_output=consume_line,
            capture_output=False,
        )
    else:
        if import_roots is None:
            mod_args = " ".join(shlex.quote(m) for m in modules)
            lean_command = f"lean --run {_DUMP_KEEP_PATH} {mod_args}"
        else:
            roots_path = _DUMP_KEEP_MODULES_PATH + ".roots"
            write_bytes(roots_path, ("\n".join(import_roots) + "\n").encode("utf-8"))
            lean_command = (
                f"lean --run {_DUMP_KEEP_PATH} --modules-file {_DUMP_KEEP_MODULES_PATH} "
                f"--roots-file {roots_path}"
            )
        cmd = (
            "cd /testbed && lake env sh -c "
            + shlex.quote(
                (
                    "LEAN_PATH=" + shlex.quote(extra_lean_path) + ":$LEAN_PATH "
                    if extra_lean_path
                    else ""
                )
                + "exec " + lean_command
            )
        )
        result = env.execute(cmd, timeout=timeout)
        for line in result.get("output", "").splitlines():
            consume_line(line)

    if result.get("returncode", 1) != 0 or result.get("timed_out"):
        detail = "\n".join(diagnostics)[-4000:] or result.get("output", "")[-4000:]
        if strict:
            raise DeclGraphExtractionError(
                returncode=result.get("returncode"), timed_out=result.get("timed_out"),
                diagnostics=detail, declarations=len(decls),
            )
        logger.warning(
            "Lean dump-keep script failed (rc=%s). Output:\n%s",
            result.get("returncode"), detail,
        )
        return [], {}, set()
    return decls, deps, imported_namespaces
