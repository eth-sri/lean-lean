import Lean
open Lean

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
  let projModNames : NameHashSet :=
    moduleArgs.foldl (fun names module => names.insert module.toName) {}

  -- Types that mark a decl as parsing/elaboration machinery (notation/macro).
  let metaConsts : List Name :=
    [`Lean.ParserDescr, `Lean.TrailingParserDescr, `Lean.Macro,
     `Lean.MacroM, `Lean.Syntax, `Lean.TSyntax, `Lean.ParserDescr.Parser]

  let coreCtx : Core.Context := { fileName := "<dump_keep>", fileMap := default }
  let coreState : Core.State := { env := env }
  let action : CoreM Unit := do
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
      let findRefs (e : Expr) (acc : Array Name) : Array Name :=
        e.foldConsts acc fun n acc =>
          if !isProjectConstant n then
            acc
          else
            match findSourceOwner? n with
            | some owner =>
              if owner != name then acc.push owner else acc
            | none => acc
      let mut refs := findRefs cinfo.type #[]
      refs := match cinfo with
        | .defnInfo val   => findRefs val.value refs
        | .thmInfo val    => findRefs val.value refs
        | .opaqueInfo val => findRefs val.value refs
        | _               => refs
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
    pure ()
  let (_, _) ← action.toIO coreCtx coreState