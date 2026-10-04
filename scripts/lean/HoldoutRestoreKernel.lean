import Lean
open Lean Elab Command

namespace HoldoutRestoration

unsafe def trustedEnvironment : IO Environment := do
  let previous ← searchPathRef.get
  searchPathRef.set (System.FilePath.mk "/opt/holdout-trusted" :: previous)
  try
    importModules #[{ module := `HoldoutTrusted : Import }] {} 0
  finally
    searchPathRef.set previous

partial def restore (trusted : Environment) (name : Name) (visiting : List Name := []) : CommandElabM Unit := do
  if ((← getEnv).find? name).isSome then return
  if visiting.contains name then return
  let some ci := trusted.find? name | throwError "Missing trusted declaration {name}"
  let visiting := name :: visiting
  match ci with
  | .ctorInfo v => restore trusted v.induct visiting
  | .recInfo v => for n in v.all do restore trusted n visiting
  | .inductInfo v =>
    let mut types : List InductiveType := []
    for n in v.all do
      let some (.inductInfo iv) := trusted.find? n | throwError "Missing inductive {n}"
      for ref in iv.type.getUsedConstants do restore trusted ref (v.all ++ visiting)
      let mut ctors : List Constructor := []
      for cn in iv.ctors do
        let some (.ctorInfo cv) := trusted.find? cn | throwError "Missing constructor {cn}"
        for ref in cv.type.getUsedConstants do restore trusted ref (v.all ++ visiting)
        ctors := ctors ++ [{ name := cn, type := cv.type }]
      types := types ++ [{ name := n, type := iv.type, ctors := ctors }]
    liftCoreM <| addDecl (.inductDecl v.levelParams v.numParams types v.isUnsafe)
    if ((Lean.versionString.splitOn ".")[1]!).toNat! >= 34 then
      liftCoreM <| Lean.compileDecls v.all.toArray
  | _ =>
    for ref in ci.type.getUsedConstants do restore trusted ref visiting
    let declaration ← match ci with
      | .defnInfo v =>
        for ref in v.value.getUsedConstants do restore trusted ref visiting
        pure (.defnDecl v)
      | .thmInfo v =>
        if v.value.getUsedConstants.contains ``sorryAx then
          throwError "Refusing to copy a trusted sorry theorem as a dependency: {name}"
        for ref in v.value.getUsedConstants do restore trusted ref visiting
        pure (.thmDecl v)
      | .opaqueInfo v =>
        for ref in v.value.getUsedConstants do restore trusted ref visiting
        pure (.opaqueDecl v)
      | .axiomInfo _ => throwError "Refusing to introduce an axiom: {name}"
      | _ => throwError "Unsupported missing declaration {name}"
    liftCoreM <| addAndCompile declaration (logCompileErrors := false)

syntax (name := reproveTheorem) "reprove_theorem " ident " := " term : command

@[command_elab reproveTheorem] unsafe def elaborateReproof : CommandElab := fun stx => do
  let name := stx[1].getId
  let trusted ← trustedEnvironment
  let some (.thmInfo original) := trusted.find? name
    | throwError "Not a trusted theorem: {name}"
  if ((← getEnv).find? name).isSome then throwError "The theorem already exists: {name}"
  for ref in original.type.getUsedConstants do restore trusted ref
  let value ← liftTermElabM <| Term.withDeclName name <| Term.withLevelNames original.levelParams do
    let value ← Term.elabTermEnsuringType stx[3] original.type
    Term.synthesizeSyntheticMVarsNoPostponing
    let value ← instantiateMVars value
    if value.hasMVar then throwError "Unsolved proof metavariables"
    pure value
  liftCoreM <| addDecl (.thmDecl { original with value := value })

end HoldoutRestoration
