"""Semantic protected-declaration and axiom contracts for correctness gates."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from typing import Any, Mapping

from leanlean.benchmarks.leanlean import _discover_modules


_SCRIPT_PATH = "/tmp/leanlean-protected-signatures.lean"
_SEMANTIC_CLOSURE_PATH = "/tmp/leanlean-semantic-closure.lean"
_AXIOM_AUDIT_PATH = "/tmp/leanlean-protected-axioms.lean"
_TYPE_SOURCE_PATH = "/tmp/leanlean-protected-type-sources.lean"
_TYPE_CHECK_PATH = "/tmp/leanlean-protected-type-check.lean"
_TYPE_DATA_PATH = "/tmp/leanlean-protected-type-check.tsv"
VALIDATION_SCHEMA = "leanlean_signature_contract_v9"
FORBIDDEN_AXIOMS = frozenset({"sorryAx"})
_SCRIPT_TEMPLATE = r"""
import Lean
open Lean Meta

private def signatureKind : ConstantInfo → String
  | .thmInfo _    => "theorem"
  | .defnInfo _   => "def"
  | .axiomInfo _  => "axiom"
  | .opaqueInfo _ => "opaque"
  | .ctorInfo _   => "ctor"
  | .recInfo _    => "rec"
  | .inductInfo _ => "inductive"
  | .quotInfo _   => "quot"

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

private def canonicalExprHash
    (ppCtx : PPContext) (expression : Expr) : IO String :=
  ppCtx.runMetaM do
    return toString (← eraseProofTerms expression).hash

private def signatureValueHash
    (ppCtx : PPContext) : ConstantInfo → IO String
  | .defnInfo val   => canonicalExprHash ppCtx val.value
  | .opaqueInfo val => canonicalExprHash ppCtx val.value
  | _               => pure "-"

private def findConstantByRenderedName
    (env : Environment) (rendered : String) :
    Except String (Option (Name × ConstantInfo)) :=
  let directName := rendered.toName
  match env.constants.find? directName with
  | some cinfo => .ok (some (directName, cinfo))
  | none =>
    let renderedMatches := env.constants.toList.filter fun (name, _) =>
      name.toString == rendered
    match renderedMatches with
    | [] => .ok none
    | [entry] => .ok (some entry)
    | _ => .error s!"ambiguous rendered declaration name: {rendered}"

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let imports := (args.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let auditOpts := Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  let ppCtx : PPContext := { env := env, opts := auditOpts }
  let projectModules := args.map String.toName
  let protectedNames : Array String := __PROTECTED_NAMES__
  let mut lines : Array String := #[]
  for renderedName in protectedNames do
    match findConstantByRenderedName env renderedName with
    | .error message => throw <| IO.userError message
    | .ok none => pure ()
    | .ok (some (name, cinfo)) =>
      let kind := signatureKind cinfo
      let typeHash ← canonicalExprHash ppCtx cinfo.type
      let valueHash ← signatureValueHash ppCtx cinfo
      let role :=
        match env.getModuleIdxFor? name with
        | none => "imported"
        | some modIdx =>
          let modName := env.header.moduleNames[modIdx]!
          if projectModules.contains modName then "internal" else "imported"
      lines := lines.push s!"{kind}\t{name}\t{typeHash}\t{valueHash}\t{role}"
  lines := lines.qsort (· < ·)
  for line in lines do
    IO.println line
""".strip()


_SEMANTIC_CLOSURE_TEMPLATE = r"""
import Lean
open Lean Meta

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

private def isTheorem : ConstantInfo → Bool
  | .thmInfo _ => true
  | _ => false

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let imports := (args.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let auditOpts := Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  let ppCtx : PPContext := { env := env, opts := auditOpts }
  let projectModules := args.map String.toName
  let requested : Array Name :=
    (__PROTECTED_NAMES__ : Array String).map String.toName

  let mut projectNames : NameHashSet := {}
  for (name, _) in env.constants.toList do
    match env.getModuleIdxFor? name with
    | none => pure ()
    | some modIdx =>
      let modName := env.header.moduleNames[modIdx]!
      if projectModules.contains modName then
        projectNames := projectNames.insert name

  let mut protectedSet : NameHashSet := {}
  let mut pending : Array Name := #[]
  let mut missing : Array Name := #[]
  for name in requested do
    match env.constants.find? name with
    | none => missing := missing.push name
    | some _ =>
      protectedSet := protectedSet.insert name
      pending := pending.push name

  let mut cursor := 0
  while cursor < pending.size do
    let name := pending[cursor]!
    cursor := cursor + 1
    match env.constants.find? name with
    | none => pure ()
    | some cinfo =>
      let mut refs ← proofInsensitiveConsts ppCtx cinfo.type
      match cinfo with
      | .defnInfo val =>
        refs := refs ++ (← proofInsensitiveConsts ppCtx val.value)
      | .opaqueInfo val =>
        refs := refs ++ (← proofInsensitiveConsts ppCtx val.value)
      | .ctorInfo val =>
        refs := refs.push val.induct
      | .recInfo val =>
        refs := refs ++ val.all.toArray
      | .inductInfo val =>
        refs := refs ++ val.ctors.toArray
      | _ => pure ()
      for ref in refs do
        if projectNames.contains ref && !protectedSet.contains ref then
          match env.constants.find? ref with
          | some refInfo =>
            if !isTheorem refInfo then
              protectedSet := protectedSet.insert ref
              pending := pending.push ref
          | none => pure ()

  let missingLines := missing.map (fun name => s!"MISSING\t{name}")
  for line in missingLines.qsort (· < ·) do
    IO.println line
  let lines := (protectedSet.toList.map (fun name => s!"SEMANTIC\t{name}")).toArray
  for line in lines.qsort (· < ·) do
    IO.println line
""".strip()


_AXIOM_AUDIT_TEMPLATE = r"""
import Lean
open Lean

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let imports := (args.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let targets : Array Name :=
    (__PROTECTED_NAMES__ : Array String).map String.toName
  let coreCtx : Core.Context := {
    fileName := "<protected-axiom-audit>"
    fileMap := default
  }
  let coreState : Core.State := { env := env }
  let action : CoreM (Array String) := do
    let env ← getEnv
    let mut lines : Array String := #[]
    for target in targets do
      match env.find? target with
      | none => lines := lines.push s!"MISSING\t{target}"
      | some _ =>
        let axioms ← collectAxioms target
        let names := (axioms.toList.map (·.toString)).toArray.qsort (· < ·)
        lines := lines.push
          s!"AXIOMS\t{target}\t{String.intercalate " " names.toList}"
    pure lines
  let (lines, _) ← action.toIO coreCtx coreState
  for line in lines.qsort (· < ·) do
    IO.println line
""".strip()


_TYPE_SOURCE_TEMPLATE = r"""
import Lean
open Lean Meta

private partial def nameToJson : Name → Json
  | .anonymous => .arr #[.str "a"]
  | .str parentName value => .arr #[.str "s", nameToJson parentName, toJson value]
  | .num parentName value => .arr #[.str "n", nameToJson parentName, toJson value]

private partial def levelToJson : Level → Except String Json
  | .zero => .ok (.arr #[.str "z"])
  | .succ level => do
      let encoded ← levelToJson level
      return .arr #[.str "s", encoded]
  | .max left right => do
      let left' ← levelToJson left
      let right' ← levelToJson right
      return .arr #[.str "m", left', right']
  | .imax left right => do
      let left' ← levelToJson left
      let right' ← levelToJson right
      return .arr #[.str "i", left', right']
  | .param name => .ok (.arr #[.str "p", nameToJson name])
  | .mvar _ => .error "unexpected universe metavariable in constant type"

private def binderInfoToJson : BinderInfo → Json
  | .default => .str "d"
  | .implicit => .str "i"
  | .strictImplicit => .str "s"
  | .instImplicit => .str "c"

private def literalToJson : Literal → Json
  | .natVal value => .arr #[.str "n", toJson value]
  | .strVal value => .arr #[.str "s", toJson value]

private partial def exprToJson : Expr → Except String Json
  | .bvar index => .ok (.arr #[.str "b", toJson index])
  | .fvar _ => .error "unexpected free variable in constant type"
  | .mvar _ => .error "unexpected expression metavariable in constant type"
  | .sort level => do
      let level' ← levelToJson level
      return .arr #[.str "s", level']
  | .const name levels => do
      let levels' ← levels.mapM levelToJson
      return .arr #[.str "c", nameToJson name, .arr levels'.toArray]
  | .app function argument => do
      let function' ← exprToJson function
      let argument' ← exprToJson argument
      return .arr #[.str "a", function', argument']
  | .lam name domain body binderInfo => do
      let domain' ← exprToJson domain
      let body' ← exprToJson body
      return .arr #[.str "l", nameToJson name, domain', body', binderInfoToJson binderInfo]
  | .forallE name domain body binderInfo => do
      let domain' ← exprToJson domain
      let body' ← exprToJson body
      return .arr #[.str "f", nameToJson name, domain', body', binderInfoToJson binderInfo]
  | .letE name type value body nondep => do
      let type' ← exprToJson type
      let value' ← exprToJson value
      let body' ← exprToJson body
      return .arr #[.str "e", nameToJson name, type', value', body', toJson nondep]
  | .lit literal => .ok (.arr #[.str "t", literalToJson literal])
  | .mdata _ body => exprToJson body
  | .proj typeName index body => do
      let body' ← exprToJson body
      return .arr #[.str "p", nameToJson typeName, toJson index, body']

private def canonicalType (cinfo : ConstantInfo) : Expr :=
  let levels := (List.range cinfo.levelParams.length).map fun index =>
    Level.param (Name.mkNum `_leanlean_universe index)
  cinfo.type.instantiateLevelParams cinfo.levelParams levels

private def findConstantByRenderedName
    (env : Environment) (rendered : String) :
    Except String (Option (Name × ConstantInfo)) :=
  let directName := rendered.toName
  match env.constants.find? directName with
  | some cinfo => .ok (some (directName, cinfo))
  | none =>
    let renderedMatches := env.constants.toList.filter fun (name, _) =>
      name.toString == rendered
    match renderedMatches with
    | [] => .ok none
    | [entry] => .ok (some entry)
    | _ => .error s!"ambiguous rendered declaration name: {rendered}"

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let imports := (args.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let protectedNames : Array String := __PROTECTED_NAMES__
  for renderedName in protectedNames do
    match findConstantByRenderedName env renderedName with
    | .error message => throw <| IO.userError message
    | .ok none => IO.println s!"MISSING\t{renderedName}"
    | .ok (some (name, cinfo)) =>
      IO.println s!"BEGIN\t{name}"
      IO.println s!"LEVEL_ARITY\t{cinfo.levelParams.length}"
      match exprToJson (canonicalType cinfo) with
      | .error message => throw <| IO.userError s!"could not serialize {name}: {message}"
      | .ok expression => IO.println expression.compress
      IO.println s!"END\t{name}"
""".strip()


_TYPE_CHECK_TEMPLATE = r"""
import Lean
open Lean Meta

private def requireSize (fields : Array Json) (expected : Nat) (tag : String) :
    Except String Unit :=
  if fields.size == expected then .ok ()
  else .error s!"invalid {tag} field count: expected {expected}, found {fields.size}"

private partial def nameFromJson (json : Json) : Except String Name := do
  let fields ← json.getArr?
  if fields.isEmpty then throw "empty name encoding"
  let tag ← fields[0]!.getStr?
  match tag with
  | "a" =>
      requireSize fields 1 tag
      return Name.anonymous
  | "s" =>
      requireSize fields 3 tag
      return Name.str (← nameFromJson fields[1]!) (← fromJson? fields[2]!)
  | "n" =>
      requireSize fields 3 tag
      return Name.num (← nameFromJson fields[1]!) (← fromJson? fields[2]!)
  | _ => throw s!"unknown name tag: {tag}"

private partial def levelFromJson (json : Json) : Except String Level := do
  let fields ← json.getArr?
  if fields.isEmpty then throw "empty level encoding"
  let tag ← fields[0]!.getStr?
  match tag with
  | "z" =>
      requireSize fields 1 tag
      return .zero
  | "s" =>
      requireSize fields 2 tag
      return .succ (← levelFromJson fields[1]!)
  | "m" =>
      requireSize fields 3 tag
      return .max (← levelFromJson fields[1]!) (← levelFromJson fields[2]!)
  | "i" =>
      requireSize fields 3 tag
      return .imax (← levelFromJson fields[1]!) (← levelFromJson fields[2]!)
  | "p" =>
      requireSize fields 2 tag
      return .param (← nameFromJson fields[1]!)
  | _ => throw s!"unknown level tag: {tag}"

private def binderInfoFromJson (json : Json) : Except String BinderInfo := do
  match ← json.getStr? with
  | "d" => return BinderInfo.default
  | "i" => return BinderInfo.implicit
  | "s" => return BinderInfo.strictImplicit
  | "c" => return BinderInfo.instImplicit
  | tag => throw s!"unknown binder-info tag: {tag}"

private def literalFromJson (json : Json) : Except String Literal := do
  let fields ← json.getArr?
  requireSize fields 2 "literal"
  match ← fields[0]!.getStr? with
  | "n" => return .natVal (← fromJson? fields[1]!)
  | "s" => return .strVal (← fromJson? fields[1]!)
  | tag => throw s!"unknown literal tag: {tag}"

private partial def exprFromJson (json : Json) : Except String Expr := do
  let fields ← json.getArr?
  if fields.isEmpty then throw "empty expression encoding"
  let tag ← fields[0]!.getStr?
  match tag with
  | "b" =>
      requireSize fields 2 tag
      return .bvar (← fromJson? fields[1]!)
  | "s" =>
      requireSize fields 2 tag
      return .sort (← levelFromJson fields[1]!)
  | "c" =>
      requireSize fields 3 tag
      let name ← nameFromJson fields[1]!
      let encodedLevels ← fields[2]!.getArr?
      let levels ← encodedLevels.toList.mapM levelFromJson
      return .const name levels
  | "a" =>
      requireSize fields 3 tag
      return .app (← exprFromJson fields[1]!) (← exprFromJson fields[2]!)
  | "l" =>
      requireSize fields 5 tag
      let name ← nameFromJson fields[1]!
      return .lam name (← exprFromJson fields[2]!) (← exprFromJson fields[3]!)
        (← binderInfoFromJson fields[4]!)
  | "f" =>
      requireSize fields 5 tag
      let name ← nameFromJson fields[1]!
      return .forallE name (← exprFromJson fields[2]!) (← exprFromJson fields[3]!)
        (← binderInfoFromJson fields[4]!)
  | "e" =>
      requireSize fields 6 tag
      let name ← nameFromJson fields[1]!
      let nondep : Bool ← fromJson? fields[5]!
      return .letE name (← exprFromJson fields[2]!) (← exprFromJson fields[3]!)
        (← exprFromJson fields[4]!) nondep
  | "t" =>
      requireSize fields 2 tag
      return .lit (← literalFromJson fields[1]!)
  | "p" =>
      requireSize fields 4 tag
      let typeName ← nameFromJson fields[1]!
      let index : Nat ← fromJson? fields[2]!
      return .proj typeName index (← exprFromJson fields[3]!)
  | _ => throw s!"unknown expression tag: {tag}"

private def canonicalType (cinfo : ConstantInfo) : Expr :=
  let levels := (List.range cinfo.levelParams.length).map fun index =>
    Level.param (Name.mkNum `_leanlean_universe index)
  cinfo.type.instantiateLevelParams cinfo.levelParams levels

private def findConstantByRenderedName
    (env : Environment) (rendered : String) :
    Except String (Option (Name × ConstantInfo)) :=
  let directName := rendered.toName
  match env.constants.find? directName with
  | some cinfo => .ok (some (directName, cinfo))
  | none =>
    let renderedMatches := env.constants.toList.filter fun (name, _) =>
      name.toString == rendered
    match renderedMatches with
    | [] => .ok none
    | [entry] => .ok (some entry)
    | _ => .error s!"ambiguous rendered declaration name: {rendered}"

unsafe def main (_args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let moduleNames : Array String := __MODULE_NAMES__
  let imports := (moduleNames.map fun s => { module := s.toName : Import })
  let env ← importModules imports {} 0
  let opts := Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  let ppCtx : PPContext := { env := env, opts := opts }
  let data ← IO.FS.readFile "__TYPE_DATA_PATH__"
  for line in data.splitOn "\n" do
    if !line.isEmpty then
      match line.splitOn "\t" with
      | [renderedName, levelArityText, encoded] =>
        let some levelArity := levelArityText.toNat?
          | throw <| IO.userError s!"invalid level arity for {renderedName}"
        let baselineJson ←
          match Json.parse encoded with
          | .ok value => pure value
          | .error message =>
              throw <| IO.userError s!"invalid expression JSON for {renderedName}: {message}"
        let baselineType ←
          match exprFromJson baselineJson with
          | .ok value => pure value
          | .error message =>
              throw <| IO.userError s!"invalid expression encoding for {renderedName}: {message}"
        match findConstantByRenderedName env renderedName with
        | .error message => throw <| IO.userError message
        | .ok none => throw <| IO.userError s!"missing protected declaration: {renderedName}"
        | .ok (some (_, cinfo)) =>
          if cinfo.levelParams.length != levelArity then
            throw <| IO.userError s!"universe parameter count changed: {renderedName}"
          let compatible ← ppCtx.runMetaM do
            withTransparency .all do
              isDefEq baselineType (canonicalType cinfo)
          if !compatible then
            throw <| IO.userError s!"protected type is not definitionally equal: {renderedName}"
      | _ => throw <| IO.userError "invalid protected type data row"
""".strip()


@dataclass(frozen=True)
class ProtectedContract:
    """The fixed baseline against which every candidate is certified."""

    requested_names: frozenset[str]
    protected_names: frozenset[str]
    semantic_names: frozenset[str]
    local_axiom_names: frozenset[str]
    signatures: dict[str, tuple[str, ...]]
    axioms: dict[str, tuple[str, ...]]
    type_sources: dict[str, dict[str, Any]]
    sha256: str


def render_protected_signature_script(protected_names: set[str]) -> str:
    """Render the exact-name auditor with deterministic declaration ordering."""

    if not protected_names:
        raise ValueError("protected signature collection requires names")
    lean_names = "#[" + ", ".join(
        json.dumps(name, ensure_ascii=False) for name in sorted(protected_names)
    ) + "]"
    return _SCRIPT_TEMPLATE.replace("__PROTECTED_NAMES__", lean_names)


def collect_protected_signatures(
    env: Any,
    *,
    modules: list[str] | None,
    protected_names: set[str],
) -> dict[str, tuple[str, ...]]:
    """Collect only the explicit scout contract from the built Lean environment."""

    script = render_protected_signature_script(protected_names)
    installed = env.execute(
        f"cat > {_SCRIPT_PATH} << 'LEANPROTECTEDSIGS_EOF'\n"
        f"{script}\nLEANPROTECTEDSIGS_EOF"
    )
    if installed.get("returncode", 1) != 0:
        raise RuntimeError("could not install protected signature auditor")

    importable_modules = _discover_modules(env)
    if modules is None:
        selected_modules = importable_modules
    else:
        importable = set(importable_modules)
        selected_modules = [module for module in modules if module in importable]
    if not selected_modules:
        raise RuntimeError("no built project modules for protected signature audit")

    module_arguments = " ".join(shlex.quote(module) for module in selected_modules)
    result = env.execute(
        f"cd /testbed && lake env lean --run {_SCRIPT_PATH} {module_arguments}"
    )
    if result.get("returncode", 1) != 0:
        raise RuntimeError(
            "protected signature auditor failed: "
            + str(result.get("output") or "")[-4000:]
        )

    signatures: dict[str, tuple[str, ...]] = {}
    for line in str(result.get("output") or "").strip().splitlines():
        parts = line.split("\t", 4)
        if len(parts) == 5:
            kind, name, type_hash, value_hash, role = parts
            signatures[name] = (kind, type_hash, value_hash, role)
    return signatures



def _render_name_array(
    template: str, protected_names: set[str], *, purpose: str
) -> str:
    if not protected_names:
        raise ValueError(f"{purpose} requires names")
    lean_names = "#[" + ", ".join(
        json.dumps(name, ensure_ascii=False) for name in sorted(protected_names)
    ) + "]"
    return template.replace("__PROTECTED_NAMES__", lean_names)


def render_semantic_closure_script(protected_names: set[str]) -> str:
    """Render the proof-insensitive project-local definition closure audit."""

    return _render_name_array(
        _SEMANTIC_CLOSURE_TEMPLATE,
        protected_names,
        purpose="semantic closure collection",
    )


def render_protected_axiom_script(protected_names: set[str]) -> str:
    """Render the exact-target transitive axiom audit."""

    return _render_name_array(
        _AXIOM_AUDIT_TEMPLATE,
        protected_names,
        purpose="protected axiom collection",
    )


def _selected_modules(env: Any, modules: list[str] | None, *, purpose: str) -> list[str]:
    importable_modules = _discover_modules(env)
    if modules is None:
        selected_modules = importable_modules
    else:
        importable = set(importable_modules)
        selected_modules = [module for module in modules if module in importable]
    if not selected_modules:
        raise RuntimeError(f"no built project modules for {purpose}")
    return selected_modules


def _run_lean_auditor(
    env: Any,
    *,
    script: str,
    script_path: str,
    delimiter: str,
    modules: list[str] | None,
    purpose: str,
) -> str:
    installed = env.execute(
        f"cat > {script_path} << '{delimiter}'\n{script}\n{delimiter}"
    )
    if installed.get("returncode", 1) != 0:
        raise RuntimeError(f"could not install {purpose}")

    selected_modules = _selected_modules(env, modules, purpose=purpose)
    module_arguments = " ".join(shlex.quote(module) for module in selected_modules)
    result = env.execute(
        f"cd /testbed && lake env lean --run {script_path} {module_arguments}"
    )
    if result.get("returncode", 1) != 0:
        raise RuntimeError(
            f"{purpose} failed: " + str(result.get("output") or "")[-4000:]
        )
    return str(result.get("output") or "")


def render_protected_type_source_script(protected_names: set[str]) -> str:
    """Render the baseline type-source exporter used for Lean defeq checks."""

    return _render_name_array(
        _TYPE_SOURCE_TEMPLATE,
        protected_names,
        purpose="protected type-source collection",
    )


def collect_protected_type_sources(
    env: Any,
    *,
    modules: list[str] | None,
    protected_names: set[str],
) -> dict[str, dict[str, Any]]:
    """Serialize protected baseline types as canonical-universe Lean Expr trees."""

    output = _run_lean_auditor(
        env,
        script=render_protected_type_source_script(protected_names),
        script_path=_TYPE_SOURCE_PATH,
        delimiter="LEANPROTECTEDTYPES_EOF",
        modules=modules,
        purpose="protected type-source auditor",
    )
    sources: dict[str, dict[str, Any]] = {}
    current: str | None = None
    level_arity: int | None = None
    lines: list[str] = []
    missing: set[str] = set()
    for line in output.splitlines():
        if line.startswith("MISSING\t") and current is None:
            missing.add(line.split("\t", 1)[1])
        elif line.startswith("BEGIN\t") and current is None:
            current = line.split("\t", 1)[1]
            level_arity = None
            lines = []
        elif line.startswith("LEVEL_ARITY\t") and current is not None and not lines:
            level_arity = int(line.split("\t", 1)[1])
        elif current is not None and line == f"END\t{current}":
            source = "\n".join(lines).strip()
            if not source:
                raise RuntimeError(f"empty protected type expression for {current}")
            if level_arity is None:
                raise RuntimeError(f"missing protected type level arity for {current}")
            sources[current] = {
                "level_arity": level_arity,
                "expression": source,
            }
            current = None
            level_arity = None
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        raise RuntimeError(f"unterminated protected type source for {current}")
    absent = protected_names - set(sources)
    if missing or absent:
        raise RuntimeError(
            "type-source audit did not resolve every protected declaration: "
            f"missing={sorted(missing)}, absent={sorted(absent)}"
        )
    return sources


def render_protected_type_check_script(
    modules: list[str],
    checks: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render direct Meta.isDefEq checks in the candidate environment."""

    if not modules:
        raise ValueError("protected type compatibility requires modules")
    if not checks:
        raise ValueError("protected type compatibility requires checks")
    module_names = "#[" + ", ".join(
        json.dumps(module, ensure_ascii=False) for module in modules
    ) + "]"
    return (
        _TYPE_CHECK_TEMPLATE
        .replace("__MODULE_NAMES__", module_names)
        .replace("__TYPE_DATA_PATH__", _TYPE_DATA_PATH)
        + "\n"
    )


def _run_protected_type_check(
    env: Any,
    *,
    modules: list[str] | None,
    checks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected_modules = _selected_modules(
        env, modules, purpose="protected type compatibility"
    )
    script = render_protected_type_check_script(selected_modules, checks)
    rows: list[str] = []
    for name, source in sorted(checks.items()):
        expression = str(source.get("expression") or "").strip()
        if not expression:
            raise ValueError(f"protected type expression is empty for {name}")
        try:
            json.loads(expression)
        except ValueError as error:
            raise ValueError(
                f"protected type expression JSON is invalid for {name}"
            ) from error
        if any(character in expression for character in ("\n", "\r", "\t")):
            raise ValueError(f"protected type expression is not compact for {name}")
        level_arity = int(source.get("level_arity") or 0)
        rows.append(f"{name}\t{level_arity}\t{expression}")
    data = "\n".join(rows) + "\n"
    installed = env.execute(
        f"cat > {_TYPE_CHECK_PATH} << 'LEANPROTECTEDTYPECHECK_EOF'\n"
        f"{script}\nLEANPROTECTEDTYPECHECK_EOF"
    )
    if installed.get("returncode", 1) != 0:
        raise RuntimeError("could not install protected type compatibility check")
    installed_data = env.execute(
        f"cat > {_TYPE_DATA_PATH} << 'LEANPROTECTEDTYPEDATA_EOF'\n"
        f"{data}LEANPROTECTEDTYPEDATA_EOF"
    )
    if installed_data.get("returncode", 1) != 0:
        raise RuntimeError("could not install protected type compatibility data")
    return env.execute(
        f"cd /testbed && lake env lean --run {_TYPE_CHECK_PATH}"
    )


def validate_protected_type_sources(
    env: Any,
    *,
    modules: list[str] | None,
    type_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail baseline establishment unless every serialized Expr reconstructs."""

    result = _run_protected_type_check(
        env, modules=modules, checks=type_sources
    )
    if result.get("returncode", 1) != 0:
        raise RuntimeError(
            "baseline protected type expressions did not reconstruct in Lean: "
            + str(result.get("output") or "")[-4000:]
        )


def compare_protected_type_compatibility(
    env: Any,
    *,
    modules: list[str] | None,
    baseline_signatures: Mapping[str, tuple[str, ...]],
    post_signatures: Mapping[str, tuple[str, ...]],
    type_sources: Mapping[str, Mapping[str, Any]],
    protected_names: set[str],
) -> dict[str, Any]:
    """Use Lean definitional equality when exact protected type hashes differ."""

    missing_baseline = sorted(protected_names - set(baseline_signatures))
    missing_post = sorted(protected_names - set(post_signatures))
    exact: list[str] = []
    definitionally_equal: list[str] = []
    kind_changed: list[str] = []
    incompatible: dict[str, str] = {}
    for name in sorted(
        protected_names & set(baseline_signatures) & set(post_signatures)
    ):
        before = baseline_signatures[name]
        after = post_signatures[name]
        if before[0] != after[0]:
            kind_changed.append(name)
            continue
        if before[1] == after[1]:
            exact.append(name)
            continue
        source = type_sources.get(name)
        if source is None:
            raise RuntimeError(f"baseline type source missing for {name}")
        result = _run_protected_type_check(
            env, modules=modules, checks={name: source}
        )
        if result.get("returncode", 1) == 0:
            definitionally_equal.append(name)
        else:
            incompatible[name] = str(result.get("output") or "")[-4000:]
    preserved = not missing_baseline and not missing_post and not kind_changed and not incompatible
    return {
        "schema": VALIDATION_SCHEMA,
        "method": "Lean_Meta_isDefEq_on_serialized_baseline_Expr",
        "preserved": preserved,
        "missing_baseline": missing_baseline,
        "missing_post": missing_post,
        "kind_changed": kind_changed,
        "exact_type_hash": exact,
        "definitionally_equal": definitionally_equal,
        "incompatible": incompatible,
    }


def collect_semantic_definition_closure(
    env: Any,
    *,
    modules: list[str] | None,
    protected_names: set[str],
) -> set[str]:
    """Expand goals with project-local definitions that determine their meaning.

    Proof-valued subterms are erased before following constants, so proof lemmas
    and tactic infrastructure do not become immutable API. Definition bodies are
    followed recursively; inductive declarations pull in their constructors.
    """

    output = _run_lean_auditor(
        env,
        script=render_semantic_closure_script(protected_names),
        script_path=_SEMANTIC_CLOSURE_PATH,
        delimiter="LEANSEMANTICCLOSURE_EOF",
        modules=modules,
        purpose="semantic closure auditor",
    )
    semantic: set[str] = set()
    missing: set[str] = set()
    for line in output.splitlines():
        if line.startswith("SEMANTIC\t"):
            semantic.add(line.split("\t", 1)[1])
        elif line.startswith("MISSING\t"):
            missing.add(line.split("\t", 1)[1])
    absent = protected_names - semantic
    if missing or absent:
        raise RuntimeError(
            "semantic closure did not resolve every protected declaration: "
            f"missing={sorted(missing)}, absent={sorted(absent)}"
        )
    return semantic


def collect_protected_axioms(
    env: Any,
    *,
    modules: list[str] | None,
    protected_names: set[str],
) -> dict[str, tuple[str, ...]]:
    """Return each exact target's transitive kernel axiom closure."""

    output = _run_lean_auditor(
        env,
        script=render_protected_axiom_script(protected_names),
        script_path=_AXIOM_AUDIT_PATH,
        delimiter="LEANPROTECTEDAXIOMS_EOF",
        modules=modules,
        purpose="protected axiom auditor",
    )
    axioms: dict[str, tuple[str, ...]] = {}
    missing: set[str] = set()
    for line in output.splitlines():
        if line.startswith("MISSING\t"):
            missing.add(line.split("\t", 1)[1])
        elif line.startswith("AXIOMS\t"):
            _, name, axiom_text = line.split("\t", 2)
            axioms[name] = tuple(sorted(axiom_text.split()))
    absent = protected_names - set(axioms)
    if missing or absent:
        raise RuntimeError(
            "axiom audit did not resolve every protected declaration: "
            f"missing={sorted(missing)}, absent={sorted(absent)}"
        )
    return axioms


def compare_protected_axioms(
    baseline: Mapping[str, tuple[str, ...] | list[str]],
    post: Mapping[str, tuple[str, ...] | list[str]],
    *,
    protected_names: set[str],
) -> dict[str, Any]:
    """Require every final axiom closure to be a subset of its baseline closure."""

    missing_baseline = sorted(protected_names - set(baseline))
    missing_post = sorted(protected_names - set(post))
    added: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}
    forbidden: dict[str, list[str]] = {}
    for name in sorted(protected_names & set(baseline) & set(post)):
        before = set(baseline[name])
        after = set(post[name])
        new = sorted(after - before)
        gone = sorted(before - after)
        bad = sorted(after & FORBIDDEN_AXIOMS)
        if new:
            added[name] = new
        if gone:
            removed[name] = gone
        if bad:
            forbidden[name] = bad
    preserved = not missing_baseline and not missing_post and not added and not forbidden
    return {
        "schema": VALIDATION_SCHEMA,
        "method": "Lean.collectAxioms_per_target_strict_subset",
        "preserved": preserved,
        "missing_baseline": missing_baseline,
        "missing_post": missing_post,
        "added": added,
        "removed": removed,
        "forbidden": forbidden,
    }


def protected_contract_sha256(
    *,
    requested_names: set[str],
    protected_names: set[str],
    signatures: Mapping[str, tuple[str, ...]],
    axioms: Mapping[str, tuple[str, ...] | list[str]],
    type_sources: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = {
        "schema": VALIDATION_SCHEMA,
        "requested_names": sorted(requested_names),
        "protected_names": sorted(protected_names),
        "signatures": {
            name: list(signatures[name]) for name in sorted(protected_names)
        },
        "axioms": {
            name: sorted(axioms[name]) for name in sorted(requested_names)
        },
        "type_sources": {
            name: {
                "level_arity": int(type_sources[name].get("level_arity") or 0),
                "expression": str(type_sources[name].get("expression") or ""),
            }
            for name in sorted(set(requested_names) & set(type_sources))
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def establish_protected_contract(
    env: Any,
    *,
    modules: list[str] | None,
    requested_names: set[str],
    validate_serialized_types: bool = True,
) -> ProtectedContract:
    """Build a fail-closed, versioned baseline contract from exact goal names."""

    if not requested_names:
        raise ValueError("protected contract requires names")
    semantic_names = collect_semantic_definition_closure(
        env,
        modules=modules,
        protected_names=requested_names,
    )
    axioms = collect_protected_axioms(
        env,
        modules=modules,
        protected_names=requested_names,
    )
    baseline_forbidden = {
        name: sorted(set(values) & FORBIDDEN_AXIOMS)
        for name, values in axioms.items()
        if set(values) & FORBIDDEN_AXIOMS
    }
    if baseline_forbidden:
        raise RuntimeError(
            "protected baseline depends on forbidden axioms: "
            + json.dumps(baseline_forbidden, sort_keys=True)
        )

    axiom_names = {
        axiom
        for values in axioms.values()
        for axiom in values
        if axiom not in FORBIDDEN_AXIOMS
    }
    candidates = semantic_names | axiom_names
    candidate_signatures = collect_protected_signatures(
        env,
        modules=modules,
        protected_names=candidates,
    )
    type_sources: dict[str, dict[str, Any]] = {}
    if validate_serialized_types:
        type_sources = collect_protected_type_sources(
            env, modules=modules, protected_names=requested_names
        )
        validate_protected_type_sources(
            env, modules=modules, type_sources=type_sources
        )
    local_axiom_names = {
        name
        for name in axiom_names
        if name in candidate_signatures
        and candidate_signatures[name][-1] != "imported"
    }
    protected_names = semantic_names | local_axiom_names
    missing_signatures = protected_names - set(candidate_signatures)
    if missing_signatures:
        raise RuntimeError(
            "baseline signatures missing protected declarations: "
            + ", ".join(sorted(missing_signatures))
        )
    signatures = {
        name: candidate_signatures[name] for name in sorted(protected_names)
    }
    digest = protected_contract_sha256(
        requested_names=requested_names,
        protected_names=protected_names,
        signatures=signatures,
        axioms=axioms,
        type_sources=type_sources,
    )
    return ProtectedContract(
        requested_names=frozenset(requested_names),
        protected_names=frozenset(protected_names),
        semantic_names=frozenset(semantic_names),
        local_axiom_names=frozenset(local_axiom_names),
        signatures=signatures,
        axioms=axioms,
        type_sources=type_sources,
        sha256=digest,
    )
