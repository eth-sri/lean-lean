import concurrent.futures
import base64
import hashlib
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import shlex
import re
from dataclasses import dataclass, field
import logging
import random
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from leanlean.identifiers import same_identifier
from leanlean.utils import json_utils as json
import traceback
import textwrap
import os
import tempfile
import time
from leanlean.evaluation_images import (
    ensure_materialized_image,
    materialization_specs_from_env,
    retire_image_tags,
)


from leanlean import Environment, Instance, Benchmark
from leanlean.environments import get_environment
from leanlean.metrics.source_archive import SourceMetrics, measure_source_archive
from leanlean.palomar_comparator import (
    VALIDATION_SCHEMA as PALOMAR_VALIDATION_SCHEMA,
    install_agent_verify_wrapper,
    install_palomar_tools,
    load_palomar_evidence,
)
from leanlean.report_summary import write_run_report_summary
from leanlean.repo_variants import (
    repo_image_overrides_from_env,
    repo_image_tag,
    repo_variant_from_env,
)

from leanlean.timeout_policy import LEAN_VERIFY_TIMEOUT_SECONDS

from leanlean.preprocessing.olean import DeclRange

logger = logging.getLogger("leanlean.leanlean")

DEBUG = os.environ.get("LEANLEAN_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# Metrics helpers — run inside the Docker container via env.execute()
# ---------------------------------------------------------------------------
RECONSTRUCTION_VALIDATION_SCHEMA = "leanlean_theorem_reconstruction_verifier_v1"


# Python script: count whitespace-delimited words across all .lean files,
# EXCLUDING Lean comments (`-- line`, `/- block -/` with nesting, `/-- doc -/`).
# String literals are counted (they are code); comments are stripped.
# Keep this outside /testbed (like _DUMP_SIGS_PATH) so it never gets picked up
# by the `git add -A && git diff` patch extraction, nor collides with that
# patch at apply time during eval.
_WORD_COUNT_PATH = "/tmp/.__count_words.py"
_WORD_COUNT_SCRIPT = r'''
import os, re

def strip_comments(src: str) -> str:
    """Remove Lean line and block comments, preserving string literals."""
    n = len(src); i = 0; nest = 0
    in_line = False; in_str = False
    out = []
    while i < n:
        c = src[i]
        nxt = src[i+1] if i+1 < n else ''
        if in_line:
            if c == '\n':
                in_line = False
                out.append('\n')
            i += 1; continue
        if nest > 0:
            if c == '/' and nxt == '-':
                nest += 1; i += 2; continue
            if c == '-' and nxt == '/':
                nest -= 1; i += 2; continue
            i += 1; continue
        if in_str:
            out.append(c)
            if c == '\\' and nxt:
                out.append(nxt); i += 2; continue
            if c == '"':
                in_str = False
            i += 1; continue
        if c == '-' and nxt == '-':
            in_line = True; i += 2; continue
        if c == '/' and nxt == '-':
            nest = 1; i += 2; continue
        if c == '"':
            in_str = True; out.append(c); i += 1; continue
        out.append(c)
        i += 1
    return ''.join(out)

import sys
exclude = set(sys.argv[1:])  # extra dirs to skip, relative to /testbed
# Optional: restrict the count to a single LeanPool subproject, given as a module
# base path relative to /testbed (e.g. "LeanPool/Biswal"). Scope = that directory's
# subtree PLUS the sibling module file "<base>.lean" — handles both multi-file
# subprojects (dir + module file) and single-file ones (just "<base>.lean", no
# dir). Empty = whole repo.
inc = os.environ.get('WORD_COUNT_INCLUDE', '').strip().strip('/')
total = 0
for root, dirs, files in os.walk('/testbed'):
    if '/.lake/' in root or root.endswith('/.lake'):
        continue
    rel = os.path.relpath(root, '/testbed')
    if any(rel == e or rel.startswith(e + os.sep) for e in exclude):
        dirs[:] = []
        continue
    if inc and rel != '.' and not (
        rel == inc or rel.startswith(inc + os.sep) or inc.startswith(rel + os.sep)
    ):
        dirs[:] = []
        continue
    for f in files:
        if not f.endswith('.lean'):
            continue
        p = os.path.join(root, f)
        if inc:
            frel = os.path.relpath(p, '/testbed')
            if not (frel == inc + '.lean' or frel.startswith(inc + os.sep)):
                continue
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as fp:
                src = fp.read()
        except Exception:
            continue
        # `import` lines don't count toward compression: dropping them isn't a
        # real saving, so exclude them from both baseline and post word counts.
        for line in strip_comments(src).splitlines():
            if line.lstrip().startswith('import '):
                continue
            total += len(line.split())
print(total)
'''.lstrip("\n")

_WORD_COUNT_CMD = f"python3 {_WORD_COUNT_PATH}"

# The full-source Lean-token metric uses the Lean Refactor paper lexer in a
# standalone, unit-testable module. Copy that source into the container just like the word
# counter above; benchmark images do not contain the host leanlean package.
_LEAN_TOKEN_COUNT_PATH = "/tmp/.__count_lean_tokens.py"
_LEAN_TOKEN_COUNT_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "metrics" / "tokens.py"
)
_LEAN_TOKEN_COUNT_CMD = f"python3 {_LEAN_TOKEN_COUNT_PATH}"
# Task metadata lives in the repository working directory so the prompt can
# point at ordinary files. It is excluded through .git/info/exclude so these
# benchmark-owned files can never leak into the submitted patch.
_TASK_PROOF_LENGTH_PATH = "/testbed/proof_length.py"
_TASK_MAIN_RESULTS_PATH = "/testbed/main_results.yaml"
_PALOMAR_VISIBLE_CONFIG_PATH = "/testbed/comparator.json"
_PALOMAR_RUNTIME_CONFIG_PATH = "/tmp/leanlean-palomar-comparator.json"
# The pristine registered config and the documented defaults applied to the
# runtime copy. lean_verify uses them to prove the runtime config is the
# registered config plus those defaults and nothing else.
_PALOMAR_REGISTERED_CONFIG_PATH = (
    "/tmp/leanlean-palomar-comparator-registered.json"
)
_PALOMAR_RUNTIME_DEFAULTS_PATH = (
    "/tmp/leanlean-palomar-comparator-defaults.json"
)
_PALOMAR_CHALLENGE_PATH = "/tmp/leanlean-palomar-challenge.lean"
_PALOMAR_CHALLENGE_SOURCE_PATH = "/tmp/leanlean-palomar-challenge-path"
_PALOMAR_BUILD_JOBS_PATH = "/tmp/leanlean-palomar-build-jobs"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TASK_DOCUMENT_PATHS = (
    ":(icase)README*",
    ":(glob,icase)**/README*",
    ":(icase)LICENSE*",
    ":(glob,icase)**/LICENSE*",
    ":(icase)LICENCE*",
    ":(glob,icase)**/LICENCE*",
    ":(icase)NOTICE*",
    ":(glob,icase)**/NOTICE*",
    ":(exclude,icase)*.lean",
    ":(exclude,glob,icase)**/*.lean",
)


# Lean 4 script that imports project modules, iterates the built environment,
# and dumps every project-local declaration as:
#     kind\tfully.qualified.name\ttype_hash\tvalue_hash\trole
#
# where role is "root" or "internal".
#
# Protected names are also emitted when they resolve to declarations imported
# from dependencies. Their role is "imported". This lets a patch replace a
# byte-for-byte duplicate project declaration with its dependency definition
# without looking like an API removal merely because module provenance changed.
#
# A theorem is "root" if no other project-local declaration references it in
# its type or value Expr.  Root theorems are the mathematical claims the repo
# exists to prove — they must be preserved.  Everything else (internal helpers,
# defs, lemmas used by other proofs) is fair game for refactoring.
#
# Relies on `lake env lean --run` to set up LEAN_PATH correctly. Uses Expr.hash
# after replacing every proof-typed subterm with a typed canonical placeholder
# and discarding Expr metadata. This compares kernel expressions modulo proof
# irrelevance and metadata ignored by definitional equality: deleting an
# earlier declaration may rename Lean-generated `._proof_N` constants or alter
# elaborator-local metadata without changing a protected definition. Raw
# Expr.hash would reject those harmless changes. Theorems compare only their
# statement;
# changing a proof term is the point of this benchmark.
_DUMP_SIGS_LEAN_TEMPLATE = r"""
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

-- The placeholder exists only in the temporary expression being hashed. It
-- is never inserted into the environment or accepted as a proof.
--
-- Open binders while descending so ``Meta.isProof`` sees the local context.
-- ``Meta.transform`` visits loose-bound subexpressions in isolation; inference
-- then fails and silently leaves generated proof constants in the hash.
private partial def eraseProofTerms (expression : Expr) : MetaM Expr := do
  try
    if ← Meta.isProof expression then
      -- The inferred proposition can itself contain proof-valued instance
      -- arguments.  Normalize those too before embedding the proposition in
      -- the placeholder; otherwise generated `._proof_N` names leak back
      -- into the supposedly proof-insensitive hash through the sorry's type.
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
      -- Metadata is ignored by kernel definitional equality and can contain
      -- elaborator-local identifiers that change when unrelated declarations
      -- are removed earlier in a module. It is not part of the API value.
      eraseProofTerms body
  | .proj typeName index body =>
      return .proj typeName index (← eraseProofTerms body)
  | _ =>
      return expression

private def canonicalExprHash (ppCtx : PPContext) (expression : Expr) : IO String :=
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
  -- `runMetaM` snapshots the heartbeat budget when it constructs its CoreM
  -- context. Setting the option inside `canonicalExprHash` is too late: large
  -- definitions can still fail at the default 200k budget while `isProof`
  -- reduces nested instances. The surrounding subprocess/container timeouts
  -- remain the hard operational bound, so make the audit itself unlimited.
  let auditOpts := Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  let ppCtx : PPContext := { env := env, opts := auditOpts }
  let projModNames := args.map String.toName
  let protectedNames : Array String := __PROTECTED_NAMES__

  -- Step 1: Collect project-local declarations and their names.
  let mut projDecls : Array (Name × ConstantInfo) := #[]
  let mut projNameSet : NameHashSet := {}
  for (name, cinfo) in env.constants.toList do
    if name.isInternal then continue
    match env.getModuleIdxFor? name with
    | none => pure ()
    | some modIdx =>
      let modName := env.header.moduleNames[modIdx]!
      if projModNames.contains modName then
        projDecls := projDecls.push (name, cinfo)
        projNameSet := projNameSet.insert name

  -- Step 2: Build the set of "referenced" names — project-local names that
  --   appear in some *other* declaration's type or value Expr.
  let mut referenced : NameHashSet := {}
  for (declName, cinfo) in projDecls do
    let findRefs (e : Expr) (acc : Array Name) : Array Name :=
      e.foldConsts acc fun n acc =>
        if projNameSet.contains n && n != declName then acc.push n else acc
    let refs := findRefs cinfo.type #[]
    let refs := match cinfo with
      | .defnInfo val  => findRefs val.value refs
      | .thmInfo val   => findRefs val.value refs
      | .opaqueInfo val => findRefs val.value refs
      | _              => refs
    for r in refs do
      referenced := referenced.insert r

  -- Step 3: Output.  A theorem that nothing else references is a "root".
  let mut lines : Array String := #[]
  for (name, cinfo) in projDecls do
    let kind := signatureKind cinfo
    let typeHash ← canonicalExprHash ppCtx cinfo.type
    let valueHash ← signatureValueHash ppCtx cinfo
    let role := if kind == "theorem" && !referenced.contains name
                then "root" else "internal"
    lines := lines.push s!"{kind}\t{name}\t{typeHash}\t{valueHash}\t{role}"

  -- Step 4: A protected declaration may legitimately move to an imported
  -- dependency. Emit such declarations too, but do not duplicate local ones.
  for renderedName in protectedNames do
    match findConstantByRenderedName env renderedName with
    | .error message => throw <| IO.userError message
    | .ok none => pure ()
    | .ok (some (name, cinfo)) =>
      if !projNameSet.contains name then
        let kind := signatureKind cinfo
        let typeHash ← canonicalExprHash ppCtx cinfo.type
        let valueHash ← signatureValueHash ppCtx cinfo
        lines := lines.push
          s!"{kind}\t{name}\t{typeHash}\t{valueHash}\timported"

  lines := lines.qsort (· < ·)
  for line in lines do
    IO.println line
""".strip()

# Path inside the container where we drop the Lean script.
_DUMP_SIGS_PATH = "/tmp/_dump_sigs.lean"
# Large repositories can require more than the ordinary per-command budget.
_SIGNATURE_AUDIT_TIMEOUT_SECONDS = LEAN_VERIFY_TIMEOUT_SECONDS


# Discover project module names from the BUILT oleans that still have source.
# A lakefile with `globs := #[.submodules `Foo]` builds `Foo.*` but not the root
# aggregator module `Foo`, so enumerating source files would ask the dump script
# to import a module whose olean was never built — and importModules then aborts,
# losing every signature. Conversely, incremental builds retain stale oleans for
# deleted source modules. Importing one of those can conflict with declarations
# moved into a surviving module. Intersect the build cache with the current
# source tree so the list contains exactly the importable, live project modules.
_DISCOVER_MODULES_CMD = (
    r"sources=$(find /testbed -path '*/.lake' -prune -o -type f -name '*.lean' -print); "
    r"find /testbed/.lake/build/lib -name '*.olean' "
    r"| while IFS= read -r file; do "
    r"rel=${file#/testbed/.lake/build/lib/}; rel=${rel#lean/}; "
    r"rel=${rel%.olean}; "
    "if test -f \"/testbed/${rel}.lean\" "
    "|| printf '%s\\n' \"$sources\" | grep -Fq \"/${rel}.lean\"; then "
    "printf './%s\\n' \"$rel\"; fi; "
    r"done | sort"
)

# List project source files (absolute paths), for the heartbeat re-elaboration.
_DISCOVER_LEAN_FILES_CMD = (
    r"find /testbed -name '*.lean' -not -path '*/.lake/*' "
    r"-not -name 'lakefile.lean' | sort"
)

# Path inside the container where we drop the heartbeat-counting Lean script.
_COUNT_HEARTBEATS_PATH = "/tmp/_count_heartbeats.lean"
_COUNT_HEARTBEATS_FRONTEND_PATH = "/tmp/_count_heartbeats_frontend.lean"
_COUNT_HEARTBEATS_EXACT_PATH = "/tmp/_count_heartbeats_exact.py"
_COUNT_HEARTBEATS_EXACT_SOURCE = (
    Path(__file__).resolve().parents[1] / "heartbeat_cli.py"
)

# A *fair*, cache- and host-independent measure of compilation cost: re-elaborate
# each project source file via the Lean frontend and sum the heartbeat deltas
# (IO.getNumHeartbeats is a deterministic allocation counter — the same quantity
# `maxHeartbeats` is checked against). Unlike wall-clock `lake build` time, this
# is unaffected by olean caching or CPU speed. Requires the project's oleans to
# already be built so each file's imports load from cache.
_COUNT_HEARTBEATS_LEAN = r"""
import Lean
open Lean Elab Frontend

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  enableInitializersExecution
  -- maxHeartbeats 0 = unlimited, so a heavy proof never aborts our count.
  -- Build Options via the long-stable core API (Options.insert / DataValue):
  -- `Options` stopped being defeq to `KVMap` in newer Lean (>= v4.29), so the
  -- old `KVMap.empty.setNat ... : Options` no longer typechecks there.
  let opts : Options := Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  let mut total : Nat := 0
  for path in args do
    let input ← IO.FS.readFile path
    let inputCtx := Parser.mkInputContext input path
    let (header, parserState, messages) ← Parser.parseHeader inputCtx
    let (env, messages) ← processHeader header opts messages inputCtx
    if messages.hasErrors then
      IO.eprintln s!"header/import error while loading {path}"
      return 1
    let commandState := Command.mkState env messages opts
    let h0 ← IO.getNumHeartbeats
    let s ← IO.processCommands inputCtx parserState commandState
    let h1 ← IO.getNumHeartbeats
    for msg in s.commandState.messages.toList do
      if msg.severity == MessageSeverity.error then
        IO.eprintln s!"elaboration error while re-checking {path}:"
        IO.eprintln (← msg.toString)
        return 1
    total := total + (h1 - h0)
  IO.println total
  return 0
""".strip()

# Some source files explicitly refer to generated proof names. The lightweight
# processHeader/processCommands path above can initialize the name generator
# differently from the Lean CLI and reject those otherwise-valid files. This
# fallback uses the official shell frontend, preserving CLI elaboration
# semantics. It includes import-loading allocations for the fallback file, but
# the same deterministic overhead appears in both pre- and post-patch sweeps.
_COUNT_HEARTBEATS_FRONTEND_LEAN = r"""
import Lean
open Lean Elab

unsafe def main (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "expected one source path"
      return 2
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let opts : Options :=
    Options.empty.insert (Name.mkSimple "maxHeartbeats") (DataValue.ofNat 0)
  let input ← IO.FS.readFile path
  let moduleName ← moduleNameOfFileName path none
  let h0 ← IO.getNumHeartbeats
  let env? ← Lean.Elab.runFrontend input opts path moduleName
  let h1 ← IO.getNumHeartbeats
  if env?.isNone then
    return 1
  IO.println (h1 - h0)
  return 0
""".strip()


def _install_word_count_script(env: "Environment") -> None:
    """Install the legacy counter only when Python is already available.

    Source metrics are measured from an exported archive on the host. This
    helper remains for old playback captures, but must never mutate a benchmark
    container or attempt network access.
    """
    python_check = env.execute("command -v python3 >/dev/null 2>&1")
    if python_check.get("returncode", 1) != 0:
        raise RuntimeError(
            "legacy in-container word counting requires preinstalled python3; "
            "runtime package installation is disabled"
        )
    cmd = f"cat > {_WORD_COUNT_PATH} << 'WORDCOUNT_EOF'\n{_WORD_COUNT_SCRIPT}\nWORDCOUNT_EOF"
    env.execute(cmd)


def _metric_source_paths(include_prefix: str) -> list[str]:
    include_prefix = include_prefix.strip().strip("/")
    if not include_prefix:
        return ["."]
    return [include_prefix, include_prefix + ".lean"]


def _measure_source_archive(
    env: "Environment",
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
    exclude_files: list[str] | None = None,
) -> SourceMetrics:
    """Export scoped sources and measure them in the trusted host process."""

    exporter = getattr(env, "export_source_archive", None)
    if not callable(exporter):
        raise RuntimeError(
            "source metrics require an environment with export_source_archive"
        )
    with tempfile.TemporaryDirectory(prefix="leanlean-metrics-") as directory:
        archive_path = Path(directory) / "sources.tar.gz"
        metadata = exporter(
            archive_path,
            _metric_source_paths(include_prefix),
            timeout=600,
            retry_transient=False,
        )
        if not isinstance(metadata, dict) or not metadata.get(
            "source_archive_sha256"
        ):
            raise RuntimeError("source archive export returned no canonical digest")
        return measure_source_archive(
            archive_path,
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
            include_prefix=include_prefix,
        )


def _measure_words(
    env: "Environment",
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
    exclude_files: list[str] | None = None,
) -> int:
    """Count whitespace-delimited words in .lean files (comments excluded).

    `include_prefix` (relative to /testbed, e.g. "LeanPool/Biswal") restricts the
    count to that subtree — used to scope the metric to a single LeanPool
    subproject. Empty = whole repo.
    """
    return _measure_source_archive(
        env, exclude_dirs, include_prefix, exclude_files
    ).words


def _install_lean_token_count_script(env: "Environment") -> None:
    """Install the legacy counter only when Python is already available."""

    python_check = env.execute("command -v python3 >/dev/null 2>&1")
    if python_check.get("returncode", 1) != 0:
        raise RuntimeError(
            "legacy in-container token counting requires preinstalled python3; "
            "runtime package installation is disabled"
        )
    script = _LEAN_TOKEN_COUNT_SOURCE_PATH.read_text(encoding="utf-8")
    cmd = (
        f"cat > {_LEAN_TOKEN_COUNT_PATH} << 'LEANTOKENS_EOF'\n"
        f"{script}\nLEANTOKENS_EOF"
    )
    env.execute(cmd)

def _render_task_proof_length_script(
    exclude_dirs: list[str],
    include_prefix: str,
    exclude_files: list[str] | None = None,
) -> str:
    """Return a self-contained copy of the trusted token lexer."""

    metric_source = _LEAN_TOKEN_COUNT_SOURCE_PATH.read_text(encoding="utf-8")
    main_marker = "\ndef main() -> None:\n"
    library_source, marker, _old_main = metric_source.partition(main_marker)
    if not marker:
        raise RuntimeError("Lean token source has no standalone main function")
    fixed_main = textwrap.dedent(
        f'''\
        def main() -> None:
            import json
            import time

            if len(sys.argv) != 1:
                raise SystemExit("proof_length.py takes no arguments")

            root = Path(__file__).resolve().parent
            log_path = root / ".git" / "leanlean-agent-tool-usage.jsonl"
            record_usage = (
                os.environ.get("LEANLEAN_BENCHMARK_INTERNAL") != "1"
                and log_path.parent.is_dir()
            )
            started_ns = time.time_ns()
            invocation_id = f"{{started_ns}}-{{os.getpid()}}"
            if record_usage:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({{
                        "schema_version": 1,
                        "tool": "proof_length",
                        "event": "started",
                        "invocation_id": invocation_id,
                        "time_ns": started_ns,
                    }}, sort_keys=True) + "\\n")
            status = 1
            try:
                tokens = measure_repository_lean_tokens(
                    root,
                    exclude_dirs={exclude_dirs!r},
                    include_prefix={include_prefix!r},
                    exclude_files={list(exclude_files or [])!r},
                )
                print(tokens)
                status = 0
            finally:
                if record_usage:
                    finished_ns = time.time_ns()
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({{
                            "schema_version": 1,
                            "tool": "proof_length",
                            "event": "finished",
                            "invocation_id": invocation_id,
                            "time_ns": finished_ns,
                            "duration_ms": (finished_ns - started_ns) // 1_000_000,
                            "exit_code": status,
                        }}, sort_keys=True) + "\\n")


        if __name__ == "__main__":
            main()
        '''
    )
    return (
        "#!/usr/bin/env python3\n"
        + library_source.lstrip()
        + "\n"
        + fixed_main
    )


def _render_task_main_results(
    *,
    instance_id: str,
    build_command: str,
    exclude_dirs: list[str],
    include_prefix: str,
    protected_names: set[str] | None,
    protected_source: str,
) -> str:
    """Render dependency-free YAML describing the benchmark's visible contract."""

    scalar = json.dumps
    lines = [
        "schema_version: 1",
        "benchmark: leanlean",
        f"instance_id: {scalar(instance_id)}",
        "metric:",
        '  unit: "non-comment, non-import Lean tokens"',
        '  command: "python3 proof_length.py"',
        f"  include_prefix: {scalar(include_prefix or '.')}",
    ]
    if exclude_dirs:
        lines.append("  exclude_directories:")
        lines.extend(f"    - {scalar(path)}" for path in exclude_dirs)
    else:
        lines.append("  exclude_directories: []")
    lines.extend(
        [
            "preservation:",
            f"  build_command: {scalar(build_command)}",
            f"  source: {scalar(protected_source)}",
        ]
    )
    if protected_names:
        lines.append("  main_results:")
        lines.extend(f"    - {scalar(name)}" for name in sorted(protected_names))
    else:
        lines.append("  main_results: []")
        lines.append(
            '  note: "Main results are evaluator-derived semantic roots for this repository."'
        )
    lines.extend(
        [
            "artifact_policy:",
            "  included_in_submission_patch: false",
            "  included_in_compression_metric: false",
            '  note: "Evaluation independently recomputes the metric and protected signatures."',
        ]
    )
    return "\n".join(lines) + "\n"


def _task_metadata_execute(
    env: "Environment",
    command: str,
    action: str,
    *,
    timeout: int | bool | None = None,
) -> dict:
    result = (
        env.execute(command)
        if timeout is None
        else env.execute(command, timeout=timeout)
    )
    if result.get("returncode", 1) != 0:
        raise RuntimeError(
            f"could not {action}: {result.get('output', '').strip()}"
        )
    return result


def _write_task_metadata_file(
    env: "Environment", path: str, content: str, *, append: bool = False
) -> None:
    encoded = base64.b64encode(content.encode()).decode("ascii")
    redirect = ">>" if append else ">"
    _task_metadata_execute(
        env,
        f"printf %s {shlex.quote(encoded)} | base64 -d {redirect} {shlex.quote(path)}",
        f"write {path}",
    )


def _write_task_metadata_bytes(
    env: "Environment", path: str, content: bytes
) -> None:
    encoded = base64.b64encode(content).decode("ascii")
    _task_metadata_execute(
        env,
        f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}",
        f"write {path}",
    )


def _safe_repository_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty repository path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RuntimeError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _render_hide_task_documents_command(
    repository_root: str = "/testbed",
) -> str:
    pathspecs = " ".join(
        shlex.quote(pathspec) for pathspec in _TASK_DOCUMENT_PATHS
    )
    root = shlex.quote(repository_root)
    return (
        "set -o pipefail && "
        f"cd {root} && git ls-files -z -- {pathspecs} | "
        "while IFS= read -r -d '' path; do "
        "git update-index --skip-worktree -- \"$path\" || exit 1; "
        "rm -f -- \"$path\" || exit 1; "
        "done"
    )


def _hide_task_documents(env: "Environment") -> None:
    """Hide stale prose without recording deletions in the model patch."""

    _task_metadata_execute(
        env,
        _render_hide_task_documents_command(),
        "hide README, license, and notice files",
    )


def _install_task_metadata(
    env: "Environment",
    *,
    instance_id: str,
    build_command: str,
    exclude_dirs: list[str],
    include_prefix: str,
    protected_names: set[str] | None,
    protected_source: str,
) -> None:
    """Expose metric and protected-result guidance without changing Git state."""

    python_check = env.execute("command -v python3 >/dev/null 2>&1")
    if python_check.get("returncode", 1) != 0:
        raise RuntimeError(
            "task metadata requires preinstalled python3; rebuild the selected "
            "LeanLean image from the current Dockerfile"
        )
    for relative in (
        "proof_length.py",
        "main_results.yaml",
    ):
        collision = env.execute(
            "cd /testbed && git ls-files --error-unmatch -- "
            + shlex.quote(relative)
        )
        if collision.get("returncode", 1) == 0:
            raise RuntimeError(
                f"benchmark task-metadata path collides with tracked file: {relative}"
            )

    _task_metadata_execute(
        env,
        "mkdir -p /testbed/.git/info",
        "create task metadata support directory",
    )
    _write_task_metadata_file(
        env,
        "/testbed/.git/info/exclude",
        "\n/proof_length.py\n/main_results.yaml\n",
        append=True,
    )
    _write_task_metadata_file(
        env,
        _TASK_PROOF_LENGTH_PATH,
        _render_task_proof_length_script(exclude_dirs, include_prefix),
    )
    _write_task_metadata_file(
        env,
        _TASK_MAIN_RESULTS_PATH,
        _render_task_main_results(
            instance_id=instance_id,
            build_command=build_command,
            exclude_dirs=exclude_dirs,
            include_prefix=include_prefix,
            protected_names=protected_names,
            protected_source=protected_source,
        ),
    )
    _task_metadata_execute(
        env,
        f"chmod 0755 {_TASK_PROOF_LENGTH_PATH}",
        "make proof_length.py executable",
    )
    smoke = _task_metadata_execute(
        env,
        "cd /testbed && LEANLEAN_BENCHMARK_INTERNAL=1 "
        f"python3 {_TASK_PROOF_LENGTH_PATH}",
        "validate proof_length.py",
    )
    try:
        int(smoke.get("output", "").strip())
    except (TypeError, ValueError) as error:
        raise RuntimeError("proof_length.py did not print an integer") from error


def _install_palomar_task_metadata(
    env: "Environment",
    *,
    contract: dict[str, Any],
    exclude_dirs: list[str],
    include_prefix: str,
    build_jobs: int,
    prewarm: bool = True,
    enable_lean_verify: bool = True,
    enable_proof_length: bool = True,
) -> None:
    """Expose only the enabled registered Comparator commands."""

    if not enable_lean_verify and not enable_proof_length:
        raise RuntimeError("Comparator metadata requested with every tool disabled")
    python_check = env.execute("command -v python3 >/dev/null 2>&1")
    if python_check.get("returncode", 1) != 0:
        raise RuntimeError("Comparator task metadata requires preinstalled python3")
    if not same_identifier(contract.get("schema"), PALOMAR_VALIDATION_SCHEMA):
        raise RuntimeError("unsupported Palomar Comparator contract schema")
    toolchain = env.read_file("/testbed/lean-toolchain").strip()
    if toolchain != contract.get("lean_toolchain"):
        raise RuntimeError(
            "benchmark Lean toolchain differs from registered Palomar toolchain: "
            f"{toolchain!r} != {contract.get('lean_toolchain')!r}"
        )
    challenge_pin = contract.get("challenge")
    if not isinstance(challenge_pin, dict):
        raise RuntimeError("Palomar Comparator contract has no Challenge pin")
    challenge_source = _safe_repository_path(
        challenge_pin.get("source_path"), "registered Challenge path"
    )
    metadata_paths = ["comparator.json", challenge_source]
    if enable_proof_length:
        metadata_paths.insert(0, "proof_length.py")
    for relative in metadata_paths:
        collision = env.execute(
            "cd /testbed && "
            "if git ls-files --error-unmatch -- "
            + shlex.quote(relative)
            + " >/dev/null 2>&1 || test -e "
            + shlex.quote(relative)
            + "; then exit 0; else exit 1; fi"
        )
        if collision.get("returncode", 1) == 0:
            raise RuntimeError(
                f"benchmark Comparator path collides with repository file: {relative}"
            )

    challenge, original_config, runtime_config = load_palomar_evidence(
        repo_root=_REPOSITORY_ROOT, contract=contract
    )
    _task_metadata_execute(
        env,
        "mkdir -p /testbed/.git/info",
        "create Comparator metadata support directory",
    )
    _write_task_metadata_file(
        env,
        "/testbed/.git/info/exclude",
        ("\n/proof_length.py" if enable_proof_length else "")
        + "\n/comparator.json\n/"
        + challenge_source
        + "\n",
        append=True,
    )
    if enable_proof_length:
        _write_task_metadata_file(
            env,
            _TASK_PROOF_LENGTH_PATH,
            _render_task_proof_length_script(
                exclude_dirs,
                include_prefix,
                exclude_files=[challenge_source],
            ),
        )
    runtime_defaults = dict(
        (contract.get("configuration") or {}).get("runtime_defaults_applied") or {}
    )
    _write_task_metadata_bytes(env, _PALOMAR_VISIBLE_CONFIG_PATH, original_config)
    _write_task_metadata_bytes(
        env, _PALOMAR_REGISTERED_CONFIG_PATH, original_config
    )
    _write_task_metadata_bytes(env, _PALOMAR_RUNTIME_CONFIG_PATH, runtime_config)
    _write_task_metadata_file(
        env, _PALOMAR_RUNTIME_DEFAULTS_PATH, json.dumps(runtime_defaults) + "\n"
    )
    _write_task_metadata_bytes(env, _PALOMAR_CHALLENGE_PATH, challenge)
    _task_metadata_execute(
        env,
        "mkdir -p "
        + shlex.quote(
            "/testbed/" + str(PurePosixPath(challenge_source).parent)
        ),
        "create registered Challenge directory",
    )
    _write_task_metadata_bytes(
        env, "/testbed/" + challenge_source, challenge
    )
    _write_task_metadata_file(
        env, _PALOMAR_CHALLENGE_SOURCE_PATH, challenge_source + "\n"
    )
    _write_task_metadata_file(
        env, _PALOMAR_BUILD_JOBS_PATH, str(max(1, build_jobs)) + "\n"
    )
    install_palomar_tools(env, repo_root=_REPOSITORY_ROOT, contract=contract)
    install_agent_verify_wrapper(env, repo_root=_REPOSITORY_ROOT)
    _task_metadata_execute(
        env,
        (
            f"chmod 0755 {_TASK_PROOF_LENGTH_PATH} && "
            if enable_proof_length
            else ""
        )
        + "chmod 0755 /usr/local/bin/lean_verify && "
        f"chmod 0444 {_PALOMAR_VISIBLE_CONFIG_PATH} "
        f"{_PALOMAR_REGISTERED_CONFIG_PATH} {_PALOMAR_RUNTIME_CONFIG_PATH} "
        f"{_PALOMAR_RUNTIME_DEFAULTS_PATH} "
        + shlex.quote("/testbed/" + challenge_source),
        "set Comparator task command permissions",
    )
    if enable_proof_length:
        smoke = _task_metadata_execute(
            env,
            "cd /testbed && LEANLEAN_BENCHMARK_INTERNAL=1 "
            f"python3 {_TASK_PROOF_LENGTH_PATH}",
            "validate proof_length.py",
        )
        try:
            int(smoke.get("output", "").strip())
        except (TypeError, ValueError) as error:
            raise RuntimeError("proof_length.py did not print an integer") from error
    _task_metadata_execute(
        env,
        "command -v lean_verify >/dev/null && "
        # The runtime copy may legitimately carry documented defaults, so the
        # agent-visible config is compared against the pristine registered
        # copy. lean_verify itself proves the runtime copy is that registered
        # config plus those defaults and nothing else.
        f"cmp -s {_PALOMAR_VISIBLE_CONFIG_PATH} {_PALOMAR_REGISTERED_CONFIG_PATH} && "
        f"cmp -s /testbed/{shlex.quote(challenge_source)} {_PALOMAR_CHALLENGE_PATH}",
        "validate registered Comparator command",
    )
    if prewarm and enable_lean_verify:
        started_ns = time.monotonic_ns()
        _task_metadata_execute(
            env,
            "cd /testbed && "
            "LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify",
            "prewarm the registered Comparator pipeline",
            # The benchmark manifest already pins the container lifetime.
            # Comparator work can legitimately exceed one hour, so do not add
            # a second shorter timeout that rejects a valid repository.
            timeout=False,
        )
        duration_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        setattr(
            env,
            "_leanlean_setup_tool_usage",
            [
                {
                    "tool": "lean_verify",
                    "phase": "setup_prewarm",
                    "invocation_id": "setup-prewarm",
                    "duration_ms": duration_ms,
                    "exit_code": 0,
                    "status": "succeeded",
                }
            ],
        )


def _render_reconstruction_verify_script(contract: Mapping[str, Any]) -> str:
    if contract.get("edit_policy", "target_file_only") not in {"target_file_only", "repository_wide"}:
        raise ValueError("invalid reconstruction edit_policy")
    target = str(contract["declaration"])
    target_literal = json.dumps(target, ensure_ascii=False)
    lean_source = r"""
import Lean
open Lean

unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let imports := (args.map fun s => { module := s.toName : Import }).toArray
  let env ← importModules imports {} 0
  let target := __TARGET__.toName
  let coreCtx : Core.Context := {
    fileName := "<theorem-reconstruction-axiom-audit>"
    fileMap := default
  }
  let coreState : Core.State := { env := env }
  let action : CoreM String := do
    let env ← getEnv
    match env.find? target with
    | none => return s!"MISSING\t{target}"
    | some _ =>
      let axioms ← collectAxioms target
      let text := String.intercalate " " (axioms.toList.map (·.toString))
      return s!"AXIOMS\t{target}\t{text}"
  let (line, _) ← action.toIO coreCtx coreState
  IO.println line
""".strip().replace("__TARGET__", target_literal)
    config = json.dumps(dict(contract), sort_keys=True)
    return f"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CONFIG = json.loads({config!r})
LEAN_SOURCE = {lean_source!r}


def run(command):
    return subprocess.run(
        command,
        cwd="/testbed",
        capture_output=True,
        text=True,
        check=False,
    )


started_ns = time.time_ns()
status = 1
try:
    initial_path = Path("/.init_commit")
    initial = (
        initial_path.read_text().strip()
        if initial_path.is_file()
        else run(["git", "rev-list", "--max-parents=0", "HEAD"]).stdout.splitlines()[-1]
    )
    tracked = run(["git", "diff", "--name-only", initial, "--"])
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"])
    changed = sorted(set((tracked.stdout + untracked.stdout).splitlines()))
    unexpected = [name for name in changed if name != CONFIG["target_file"]]
    if CONFIG.get("edit_policy", "target_file_only") == "target_file_only" and unexpected:
        print("unexpected edited files: " + ", ".join(unexpected))
        raise SystemExit(1)

    environment = dict(os.environ)
    environment["LEAN_NUM_THREADS"] = str(CONFIG["build_jobs"])
    build = subprocess.run(
        ["lake", "build", CONFIG["build_target"]],
        cwd="/testbed",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        print(build.stdout + build.stderr)
        raise SystemExit(build.returncode)

    audit_path = Path("/tmp/leanlean-theorem-reconstruction-audit.lean")
    audit_path.write_text(LEAN_SOURCE + "\\n")
    audit = run(
        ["lake", "env", "lean", "--run", str(audit_path), CONFIG["module"]]
    )
    output = audit.stdout + audit.stderr
    if audit.returncode != 0:
        print(output)
        raise SystemExit(audit.returncode)
    prefix = "AXIOMS\\t" + CONFIG["declaration"] + "\\t"
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        print(output)
        raise SystemExit(1)
    axioms = set(lines[0][len(prefix):].split())
    forbidden = sorted(axioms - set(CONFIG["permitted_axioms"]))
    if forbidden:
        print("forbidden axioms: " + ", ".join(forbidden))
        raise SystemExit(1)
    print("reconstruction verified; axioms: " + ", ".join(sorted(axioms)))
    status = 0
finally:
    if os.environ.get("LEANLEAN_BENCHMARK_INTERNAL") != "1":
        path = Path("/testbed/.git/leanlean-agent-tool-usage.jsonl")
        if path.parent.is_dir():
            finished_ns = time.time_ns()
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({{
                    "schema_version": 1,
                    "tool": "lean_verify",
                    "event": "finished",
                    "time_ns": finished_ns,
                    "duration_ms": (finished_ns - started_ns) // 1_000_000,
                    "exit_code": status,
                }}, sort_keys=True) + "\\n")
sys.exit(status)
"""


def _install_reconstruction_task_metadata(
    env: "Environment",
    *,
    contract: Mapping[str, Any],
    exclude_dirs: list[str],
    include_prefix: str,
) -> None:
    expected = {
        "schema",
        "module",
        "declaration",
        "target_file",
        "build_target",
        "build_jobs",
        "permitted_axioms",
    }
    if (set(contract) - {"edit_policy"} != expected
            or contract.get("schema") != RECONSTRUCTION_VALIDATION_SCHEMA
            or contract.get("edit_policy", "target_file_only") not in {"target_file_only", "repository_wide"}):
        raise RuntimeError("invalid theorem reconstruction verifier contract")
    identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")
    for key in ("module", "declaration"):
        if not identifier.fullmatch(str(contract[key])):
            raise RuntimeError(f"invalid reconstruction {key}")
    if not re.fullmatch(
        r"\+?[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*",
        str(contract["build_target"]),
    ):
        raise RuntimeError("invalid reconstruction build_target")
    target_file = _safe_repository_path(
        contract["target_file"], "reconstruction target file"
    )
    if not isinstance(contract["build_jobs"], int) or contract["build_jobs"] < 1:
        raise RuntimeError("invalid reconstruction build_jobs")
    permitted = contract["permitted_axioms"]
    if (
        not isinstance(permitted, list)
        or any(not isinstance(name, str) or not name for name in permitted)
        or len(permitted) != len(set(permitted))
    ):
        raise RuntimeError("invalid reconstruction permitted_axioms")

    _task_metadata_execute(
        env,
        "mkdir -p /testbed/.git/info",
        "create reconstruction task metadata directory",
    )
    _write_task_metadata_file(
        env,
        "/testbed/.git/info/exclude",
        "\n/proof_length.py\n/main_results.yaml\n",
        append=True,
    )
    _write_task_metadata_file(
        env,
        _TASK_PROOF_LENGTH_PATH,
        _render_task_proof_length_script(exclude_dirs, include_prefix),
    )
    _write_task_metadata_file(
        env,
        _TASK_MAIN_RESULTS_PATH,
        _render_task_main_results(
            instance_id=str(contract["declaration"]),
            build_command=f"lake build {contract['build_target']}",
            exclude_dirs=exclude_dirs,
            include_prefix=include_prefix,
            protected_names={str(contract["declaration"])},
            protected_source="theorem_reconstruction_holdout",
        ),
    )
    _write_task_metadata_file(
        env,
        "/usr/local/bin/lean_verify",
        _render_reconstruction_verify_script(contract),
    )
    _task_metadata_execute(
        env,
        "chmod 0755 /testbed/proof_length.py /usr/local/bin/lean_verify",
        "make reconstruction tools executable",
    )
    smoke = _task_metadata_execute(
        env,
        "cd /testbed && LEANLEAN_BENCHMARK_INTERNAL=1 "
        "python3 proof_length.py",
        "validate reconstruction proof_length.py",
    )
    try:
        int(smoke.get("output", "").strip())
    except (TypeError, ValueError) as error:
        raise RuntimeError("proof_length.py did not print an integer") from error
    existence = env.execute(
        "cd /testbed && git ls-files --error-unmatch -- "
        + shlex.quote(target_file)
    )
    if existence.get("returncode", 1) != 0:
        raise RuntimeError("reconstruction target file is not tracked")



def _measure_lean_tokens(
    env: "Environment",
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
    exclude_files: list[str] | None = None,
) -> int:
    """Count source tokens without requiring Python or network in the container."""
    return _measure_source_archive(
        env, exclude_dirs, include_prefix, exclude_files
    ).lean_tokens


def _lean_file_count_find_cmd(
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
    exclude_files: list[str] | None = None,
) -> str:
    """`find` that lists exactly the .lean files feeding the compression metric:
    skip .lake, lakefile.lean, and every exclude_dir (relative to /testbed).
    Mirrors _WORD_COUNT_SCRIPT's file-set scope (it still includes files that
    are all comments/imports — we only scope the *set* of files, not contents).

    `include_prefix` (relative to /testbed) restricts the search root to that
    subtree, matching _measure_words' WORD_COUNT_INCLUDE scoping.
    """
    prunes = " ".join(
        f"-not -path {shlex.quote(f'/testbed/{d}/*')}" for d in (exclude_dirs or [])
    )
    file_excludes = " ".join(
        f"-not -path {shlex.quote('/testbed/' + path.strip().strip('/'))}"
        for path in (exclude_files or [])
        if path.strip()
    )
    if include_prefix:
        # Match the subproject's directory subtree OR its sibling module file, so
        # both multi-file and single-file (no dir) subprojects are counted.
        inc = include_prefix.strip("/")
        scope = (
            f"\\( -path {shlex.quote(f'/testbed/{inc}/*')} "
            f"-o -path {shlex.quote(f'/testbed/{inc}.lean')} \\)"
        )
    else:
        scope = ""
    return (
        f"find /testbed {scope} -name '*.lean' -not -path '*/.lake/*' "
        f"-not -name 'lakefile.lean' {prunes} {file_excludes} | wc -l"
    ).strip()


# Memoize the baseline file count per docker image: it's a property of the
# (prebuilt) repo, independent of the model patch / round / experiment, so we
# only need to probe each image once even when analyze() runs many instances
# in parallel.
_BASELINE_FILE_COUNT_CACHE: dict[tuple[str, str], int | None] = {}


def _baseline_lean_file_count(
    image: str, exclude_dirs: list[str] | None, include_prefix: str = ""
) -> int | None:
    """Count the metric-scoped .lean files in a prebuilt image with a quick
    `find` (no `lake build`). Returns None if docker/the image is unavailable so
    callers can fall back to the value persisted at compile time.

    `include_prefix` scopes the count to a subtree (LeanPool subproject); the
    cache is keyed by it so subprojects sharing one image don't collide."""
    cache_key = (image, include_prefix)
    if cache_key in _BASELINE_FILE_COUNT_CACHE:
        return _BASELINE_FILE_COUNT_CACHE[cache_key]
    count: int | None = None
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm", "--network=none", "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true", image, "bash", "-c",
             _lean_file_count_find_cmd(exclude_dirs, include_prefix=include_prefix)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            count = int(proc.stdout.strip())
        else:
            logger.warning(
                "Could not recompute lean_file_count from %s (rc=%s): %s",
                image, proc.returncode, proc.stderr.strip()[:500],
            )
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        logger.warning("Could not recompute lean_file_count from %s: %s", image, e)
    _BASELINE_FILE_COUNT_CACHE[cache_key] = count
    return count


# Memoize the per-image .lean file listing used to expand a repo into one
# file-level instance per source file (file_level mode). Keyed by (image,
# exclude_dirs) since exclude_dirs scopes the set exactly like the metric.
_REPO_LEAN_FILES_CACHE: dict[tuple[str, tuple[str, ...]], list[str]] = {}


def _list_repo_lean_files(image: str, exclude_dirs: list[str] | None) -> list[str]:
    """List metric-scoped .lean files (paths relative to /testbed) in a prebuilt
    image, with a quick `find` (no `lake build`). Same file-set as the compression
    metric: skips .lake, lakefile.lean, and every exclude_dir. Returns [] (with a
    warning) if docker/the image is unavailable so the caller can fall back."""
    key = (image, tuple(exclude_dirs or []))
    if key in _REPO_LEAN_FILES_CACHE:
        return _REPO_LEAN_FILES_CACHE[key]
    prunes = " ".join(
        f"-not -path {shlex.quote(f'/testbed/{d}/*')}" for d in (exclude_dirs or [])
    )
    find_cmd = (
        "find /testbed -name '*.lean' -not -path '*/.lake/*' "
        f"-not -name 'lakefile.lean' {prunes} | sort"
    ).strip()
    files: list[str] = []
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm", "--network=none", "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true", image, "bash", "-c", find_cmd,
            ],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                path = line.strip()
                if path.startswith("/testbed/"):
                    files.append(path[len("/testbed/"):])
        else:
            logger.warning(
                "Could not list lean files from %s (rc=%s): %s",
                image, proc.returncode, proc.stderr.strip()[:500],
            )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Could not list lean files from %s: %s", image, e)
    _REPO_LEAN_FILES_CACHE[key] = files
    return files


def _install_heartbeat_script(env: "Environment") -> None:
    """Copy exact-Lake and compatibility heartbeat scripts into the container."""
    primary_cmd = (
        f"cat > {_COUNT_HEARTBEATS_PATH} << 'COUNTHB_EOF'\n"
        f"{_COUNT_HEARTBEATS_LEAN}\nCOUNTHB_EOF"
    )
    fallback_cmd = (
        f"cat > {_COUNT_HEARTBEATS_FRONTEND_PATH} << 'COUNTHB_FRONTEND_EOF'\n"
        f"{_COUNT_HEARTBEATS_FRONTEND_LEAN}\nCOUNTHB_FRONTEND_EOF"
    )
    exact_source = _COUNT_HEARTBEATS_EXACT_SOURCE.read_text(encoding="utf-8")
    exact_cmd = (
        f"cat > {_COUNT_HEARTBEATS_EXACT_PATH} << 'COUNTHB_EXACT_EOF'\n"
        f"{exact_source}\nCOUNTHB_EXACT_EOF"
    )
    env.execute(exact_cmd)

    env.execute(primary_cmd)
    env.execute(fallback_cmd)


def _measure_heartbeats(
    env: "Environment",
    exclude_dirs: list[str] | None = None,
    include_prefix: str = "",
    exclude_files: list[str] | None = None,
) -> int:
    """Sum the Lean heartbeats to re-elaborate every in-scope source file.

    The preferred path uses the real Lean CLI plus each module's Lake
    ``.setup.json``, preserving plugins, imported artifacts, and generated-name
    behavior. A header-only invocation is subtracted from the complete process
    count to isolate source-body heartbeats. The in-process frontends remain as
    compatibility fallbacks for repositories without setup artifacts.

    Requires `lake build` to have produced module artifacts. ``include_prefix``
    uses the same subtree-or-sibling-module scope as the word/token metrics.
    Returns -1 if measurement fails.
    """
    _install_heartbeat_script(env)
    listing = env.execute(_DISCOVER_LEAN_FILES_CMD).get("output", "")
    excl = exclude_dirs or []
    excluded_files = {
        path.strip().strip("/") for path in (exclude_files or []) if path.strip()
    }
    files: list[str] = []
    for line in listing.splitlines():
        path = line.strip()
        if not path:
            continue
        rel = path[len("/testbed/"):] if path.startswith("/testbed/") else path
        if any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in excl):
            continue
        if rel in excluded_files:
            continue
        if include_prefix:
            prefix = include_prefix.strip("/")
            if rel != f"{prefix}.lean" and not rel.startswith(f"{prefix}/"):
                continue
        files.append(path)
    if not files:
        logger.warning("No project .lean files found for heartbeat measurement")
        return -1

    # Measure one file per `lean` process and sum. Lean caches imported module
    # data globally and does NOT release it between files, so elaborating all
    # ~95 files in a single process grows monotonically to 100 GB+ (the whole
    # development accumulates in one address space). A fresh process per file
    # reclaims that memory on exit, capping peak RSS at a single file's cost —
    # matching how `lake build` forks a `lean` per module. Heartbeats are
    # additive per file, so the total is identical.
    import time as _time
    total = 0
    fallback_count = 0
    logger.info("Measuring heartbeats for %d source file(s)", len(files))
    for index, path in enumerate(files, start=1):
        started = _time.monotonic()
        exact_cmd = (
            f"cd /testbed && python3 {_COUNT_HEARTBEATS_EXACT_PATH} "
            f"--project /testbed --source {shlex.quote(path)} "
            f"--timeout-seconds {LEAN_VERIFY_TIMEOUT_SECONDS} --measure-imports"
        )
        exact_result = env.execute(exact_cmd, timeout=LEAN_VERIFY_TIMEOUT_SECONDS)
        exact_heartbeats: int | None = None
        exact_total: int | None = None
        import_heartbeats: int | None = None
        if exact_result.get("returncode", 1) == 0:
            try:
                import json as _stdlib_json

                payload = _stdlib_json.loads(
                    exact_result.get("output", "").strip().splitlines()[-1]
                )
                if payload.get("ok"):
                    exact_heartbeats = int(payload["body_heartbeats"])
                    exact_total = int(payload["total_heartbeats"])
                    import_heartbeats = int(payload["import_heartbeats"])
            except (ValueError, TypeError, KeyError, IndexError):
                pass
        if exact_heartbeats is not None:
            total += exact_heartbeats
            logger.info(
                "Heartbeat file %d/%d: %s heartbeats=%d elapsed=%.2fs "
                "running_total=%d mode=exact_lake_setup process_total=%d "
                "imports=%d",
                index, len(files), path, exact_heartbeats,
                _time.monotonic() - started, total, exact_total,
                import_heartbeats,
            )
            continue
        if os.environ.get("LEANLEAN_REQUIRE_EXACT_HEARTBEATS") == "1":
            logger.warning(
                "Exact Lake-setup heartbeat measurement failed for %s (rc=%s); "
                "strict exact mode rejects compatibility fallback. Output:\n%s",
                path,
                exact_result.get("returncode"),
                exact_result.get("output", "")[:2000],
            )
            return -1

        logger.warning(
            "Exact Lake-setup heartbeat measurement failed for %s (rc=%s); "
            "retrying with compatibility frontends. Output:\n%s",
            path,
            exact_result.get("returncode"),
            exact_result.get("output", "")[:2000],
        )
        cmd = (
            f"cd /testbed && lake env lean --run {_COUNT_HEARTBEATS_PATH} "
            f"{shlex.quote(path)}"
        )
        result = env.execute(cmd, timeout=LEAN_VERIFY_TIMEOUT_SECONDS)
        mode = "lightweight"
        file_heartbeats: int | None = None
        if result.get("returncode", 1) == 0:
            try:
                file_heartbeats = int(
                    result.get("output", "").strip().splitlines()[-1].strip()
                )
            except (ValueError, IndexError):
                pass
        if file_heartbeats is None:
            logger.warning(
                "Lightweight heartbeat measurement failed for %s (rc=%s); "
                "retrying with official Lean frontend. Output:\n%s",
                path, result.get("returncode"), result.get("output", "")[:2000],
            )
            fallback_cmd = (
                f"cd /testbed && lake env lean --run "
                f"{_COUNT_HEARTBEATS_FRONTEND_PATH} {shlex.quote(path)}"
            )
            result = env.execute(fallback_cmd, timeout=LEAN_VERIFY_TIMEOUT_SECONDS)
            if result.get("returncode", 1) != 0:
                logger.warning(
                    "Official-frontend heartbeat measurement failed for %s "
                    "(rc=%s). Output:\n%s",
                    path, result.get("returncode"), result.get("output", "")[:2000],
                )
                return -1
            try:
                file_heartbeats = int(
                    result.get("output", "").strip().splitlines()[-1].strip()
                )
            except (ValueError, IndexError):
                logger.warning(
                    "Failed to parse official-frontend heartbeat count for %s: %s",
                    path, result.get("output"),
                )
                return -1
            fallback_count += 1
            mode = "official_frontend_fallback"
        total += file_heartbeats
        logger.info(
            "Heartbeat file %d/%d: %s heartbeats=%d elapsed=%.2fs "
            "running_total=%d mode=%s",
            index, len(files), path, file_heartbeats,
            _time.monotonic() - started, total, mode,
        )
    logger.info(
        "Heartbeat measurement complete: files=%d total=%d fallback_files=%d",
        len(files), total, fallback_count,
    )
    return total


def _instance_data_dir(instance_id: str) -> Path:
    """Per-instance data directory (data/<instance_id>/) holding that repo's
    scout signatures, heartbeat baseline, and strip/resolve reports."""
    return Path(__file__).parents[3] / "data" / instance_id


def _baseline_heartbeats_path(instance_id: str) -> Path:
    """Path to the persisted baseline heartbeat count, alongside the per-repo
    scout signatures in data/<instance_id>/heartbeats.txt."""
    return _instance_data_dir(instance_id) / "heartbeats.txt"


def _load_baseline_heartbeats(instance_id: str) -> int | None:
    """Return the persisted baseline heartbeat count for an instance, or None if
    no data/<instance_id>/heartbeats.txt exists (caller measures it live)."""
    path = _baseline_heartbeats_path(instance_id)
    if os.environ.get("LEANLEAN_REMEASURE_BASELINE_HEARTBEATS") == "1":
        logger.info("Heartbeat baseline cache disabled for %s", instance_id)
        return None

    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError) as e:
        logger.warning("Failed to load baseline heartbeats for %s: %s", instance_id, e)
        return None


def _save_baseline_heartbeats(instance_id: str, heartbeats: int) -> None:
    """Persist a freshly measured baseline heartbeat count for reuse."""
    if heartbeats is None or heartbeats < 0:
        return
    if os.environ.get("LEANLEAN_REMEASURE_BASELINE_HEARTBEATS") == "1":
        return
    path = _baseline_heartbeats_path(instance_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{heartbeats}\n")
        logger.info("Persisted baseline heartbeats for %s: %s -> %s",
                    instance_id, heartbeats, path)
    except OSError as e:
        logger.warning("Failed to persist baseline heartbeats for %s: %s", instance_id, e)


_LEAN_MODULE_COMPONENT_RE = re.compile(
    r"^[A-Za-z_À-￿][\wÀ-￿']*$"
)


def _module_name_from_olean_relpath(relative_path: str) -> str:
    """Render an olean/source relative path as an exact Lean module name."""

    components = relative_path.split("/")
    return ".".join(
        component
        if _LEAN_MODULE_COMPONENT_RE.fullmatch(component)
        else f"«{component}»"
        for component in components
    )


def _discover_modules(env: "Environment") -> list[str]:
    """Return project module names (e.g. ['DeGiorgi.Basic', 'DeGiorgi.Main'])."""
    result = env.execute(_DISCOVER_MODULES_CMD)
    modules = [
        _module_name_from_olean_relpath(row.strip().removeprefix("./"))
        if row.strip().startswith("./")
        else row.strip()
        for row in result.get("output", "").strip().splitlines()
        if row.strip()
    ]
    return modules


def _modules_under_target(env: "Environment", target_dir: str) -> list[str] | None:
    """Return built modules belonging to one directory-scoped instance.

    LeanPool images may retain unrelated compiled modules in ``.lake`` even
    after their source trees have been removed. Those oleans are cache
    artifacts, not part of the current benchmark instance, so signature
    collection must never infer instance membership from the whole build cache.

    ``None`` preserves whole-repository behavior for standalone repositories.
    An empty list is intentionally distinct: it makes signature collection fail
    closed when a scoped instance has no built modules.
    """
    if not target_dir:
        return None
    module_prefix = target_dir.strip("/").replace("/", ".")
    return [
        module
        for module in _discover_modules(env)
        if module == module_prefix or module.startswith(f"{module_prefix}.")
    ]


def _render_dump_sigs_script(protected_names: set[str] | None = None) -> str:
    """Render the Lean signature dumper with exact global names to preserve."""
    names = sorted(protected_names or ())
    lean_names = "#[" + ", ".join(
        json.dumps(name, ensure_ascii=False) for name in names
    ) + "]"
    return _DUMP_SIGS_LEAN_TEMPLATE.replace("__PROTECTED_NAMES__", lean_names)


def _install_dump_sigs_script(
    env: "Environment", protected_names: set[str] | None = None
) -> None:
    """Copy the Lean signature-dumping script into the container."""
    # Write via heredoc to avoid shell escaping issues.
    script = _render_dump_sigs_script(protected_names)
    cmd = f"cat > {_DUMP_SIGS_PATH} << 'LEANDUMPSIGS_EOF'\n{script}\nLEANDUMPSIGS_EOF"
    env.execute(cmd)


# Signature dict: {name: (kind, type_hash, value_hash, role)}.
# value_hash is "-" for theorems and declarations without a value. role is
# "root", "internal", or "imported". _diff_signatures also accepts historical
# three-field entries so old compile_result.json files remain analyzable.
SigEntry = tuple[str, ...]
SigMap = dict[str, SigEntry]


def _collect_lean_signatures(
    env: "Environment",
    modules: list[str] | None = None,
    protected_names: set[str] | None = None,
) -> SigMap:
    """Collect project declarations plus globally resolved protected names.

    `modules` restricts the scan to a specific module list (e.g. one LeanPool
    subproject); None = discover all project modules.

    `protected_names` are looked up in the complete imported environment even
    when their declarations originate in a dependency.

    Requires `lake build` to have succeeded so .olean files exist.
    Falls back to an empty dict if the script fails (with a warning).
    """
    _install_dump_sigs_script(env, protected_names)
    importable_modules = _discover_modules(env)
    if modules is None:
        modules = importable_modules
    else:
        # A scoped clean build removes stale oleans belonging to modules outside
        # the target's import closure. Importing the old baseline list as one
        # batch would then abort the Lean process and erase every signature,
        # making successfully preserved roots look deleted. Restrict the audit
        # to modules that actually have an olean after the current build.
        importable = set(importable_modules)
        missing = [module for module in modules if module not in importable]
        if missing:
            logger.info(
                "Signature audit skipped %d requested modules without current oleans",
                len(missing),
            )
        modules = [module for module in modules if module in importable]
    if not modules:
        logger.warning("No project modules discovered — signature check skipped")
        return {}

    mod_args = " ".join(shlex.quote(m) for m in modules)
    cmd = f"cd /testbed && lake env lean --run {_DUMP_SIGS_PATH} {mod_args}"
    result = env.execute(cmd, timeout=_SIGNATURE_AUDIT_TIMEOUT_SECONDS)

    if result.get("returncode", 1) != 0:
        logger.warning(
            "Lean dump-sigs script failed (rc=%s). Output:\n%s",
            result.get("returncode"),
            result.get("output", "")[:2000],
        )
        return {}

    sigs: SigMap = {}
    for line in result.get("output", "").strip().splitlines():
        parts = line.split("\t", 4)
        if len(parts) == 5:
            kind, name, type_hash, value_hash, role = parts
            sigs[name] = (kind, type_hash, value_hash, role)
    return sigs


_DUMP_ANONYMOUS_EXAMPLES_LEAN = r"""
import Lean
open Lean Elab Frontend

partial def containsExample (stx : Syntax) : Bool :=
  stx.isOfKind ``Parser.Command.example || stx.getArgs.any containsExample

unsafe def main (paths : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  let opts : Options :=
    Options.empty.insert `maxHeartbeats (DataValue.ofNat 0)
  for path in paths do
    enableInitializersExecution
    let input ← IO.FS.readFile path
    let inputCtx := Parser.mkInputContext input path
    let (header, parserState, messages) ← Parser.parseHeader inputCtx
    let (env, messages) ← processHeader header opts messages inputCtx
    if messages.hasErrors then
      IO.eprintln s!"header/import error while parsing {path}"
      for message in messages.toList do
        if message.severity == MessageSeverity.error then
          IO.eprintln (← message.toString)
      return 1
    let commandState := Command.mkState env messages opts
    let state ← IO.processCommands inputCtx parserState commandState
    if state.commandState.messages.hasErrors then
      IO.eprintln s!"elaboration error while parsing {path}"
      for message in state.commandState.messages.toList do
        if message.severity == MessageSeverity.error then
          IO.eprintln (← message.toString)
      return 1
    for command in state.commands do
      if containsExample command then
        let start := inputCtx.fileMap.toPosition (command.getPos?.getD 0)
        let stop := inputCtx.fileMap.toPosition (command.getTailPos?.getD 0)
        IO.println s!"EXAMPLE\t{path}\t{start.line}\t{start.column}\t{stop.line}\t{stop.column}"
  return 0
""".strip()

_DUMP_ANONYMOUS_EXAMPLES_PATH = "/tmp/_dump_anonymous_examples.lean"


def _collect_anonymous_example_ranges(
    env: "Environment", paths: list[str]
) -> dict[str, list[DeclRange]]:
    """Return exact outer-command ranges for anonymous ``example`` commands.

    Callers pass source-level candidate files. The official Lean frontend
    supplies the complete command range, including doc-comment and
    command-modifier wrappers; no textual boundary heuristic is involved.

    Each source is elaborated in a fresh Lean process. Large repositories can
    have many heavyweight example-bearing modules, and retaining every loaded
    environment in one process produces unbounded peak memory even though the
    files are independent. Per-file processes preserve the exact frontend
    result while releasing imported environments between sources.
    """

    requested = sorted(set(paths))
    if not requested:
        return {}
    script = _DUMP_ANONYMOUS_EXAMPLES_LEAN
    env.execute(
        f"cat > {_DUMP_ANONYMOUS_EXAMPLES_PATH} << 'LEANEXAMPLES_EOF'\n"
        f"{script}\nLEANEXAMPLES_EOF"
    )
    ranges: dict[str, list[DeclRange]] = {}
    requested_set = set(requested)
    record_index = 0
    for requested_path in requested:
        result = env.execute(
            "cd /testbed && lake env lean --run "
            f"{_DUMP_ANONYMOUS_EXAMPLES_PATH} "
            f"{shlex.quote(requested_path)}",
            timeout=3600,
        )
        returncode = result.get("returncode", 1)
        output = result.get("output", "")
        if returncode != 0:
            detail = output[-4000:].strip() or "<no process output>"
            raise RuntimeError(
                "Lean anonymous-example range extraction failed for "
                f"{requested_path} (exit {returncode}): {detail}"
            )

        for line in output.splitlines():
            parts = line.split("\t")
            if not parts or parts[0] != "EXAMPLE" or len(parts) != 6:
                continue
            _, path, sl, sc, el, ec = parts
            if path not in requested_set or path != requested_path:
                raise RuntimeError(
                    "anonymous-example extractor returned unexpected path "
                    f"{path!r} while processing {requested_path!r}"
                )
            try:
                row: DeclRange = {
                    "name": f"__anonymous_example__.{record_index}",
                    "module": "",
                    "start_line": int(sl),
                    "start_col": int(sc),
                    "end_line": int(el),
                    "end_col": int(ec),
                    "kind": "example",
                    "meta": False,
                    "keep": False,
                }
            except ValueError as error:
                raise RuntimeError(
                    f"invalid anonymous-example range record: {line!r}"
                ) from error
            ranges.setdefault(path, []).append(row)
            record_index += 1
    return ranges


def _load_scout_signatures(instance_id: str) -> set[str] | None:
    """Load Opus-scouted protected names from data/<instance_id>/signatures.json.

    Returns a set of fully-qualified names that must be preserved, or None if
    no scout file exists (caller falls back to the Lean-computed 'root' role).
    """
    path = _instance_data_dir(instance_id) / "signatures.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        names: set[str] = set(data.get("signature_theorems", []))
        names |= set(data.get("protected_definitions", []))
        return names
    except Exception as e:
        logger.warning("Failed to load scout signatures for %s: %s", instance_id, e)
        return None


def _sig_parts(entry: SigEntry) -> tuple[str, str, str | None, str]:
    """Normalize current and historical signature entries."""
    if len(entry) == 4:
        kind, type_hash, value_hash, role = entry
        return kind, type_hash, value_hash, role
    if len(entry) == 3:
        kind, type_hash, role = entry
        return kind, type_hash, None, role
    raise ValueError(f"invalid signature entry: {entry!r}")


def _project_decl_count(sigs: SigMap) -> int:
    """Count declarations originating in measured project modules."""
    return sum(_sig_parts(entry)[3] != "imported" for entry in sigs.values())


def _protected_names(before: SigMap, scout_names: set[str] | None) -> set[str]:
    """Return the names that the post-build global lookup must resolve."""
    if scout_names is not None:
        return set(scout_names)
    return {
        name for name, entry in before.items()
        if _sig_parts(entry)[3] == "root"
    }


def _resolve_requested_signature_names(
    signatures: SigMap, requested_names: set[str]
) -> dict[str, Any]:
    """Resolve protected names against a baseline signature snapshot.

    Exact names win. A stale or over-qualified name is salvaged only when its
    final component identifies exactly one declaration in the snapshot. As a
    second compatibility pass, underscore/case drift in that final component is
    accepted only when the normalized spelling is also unique. This is
    deliberately stricter than tactic-error recovery, where restoring every
    same-suffix candidate is safe: a preservation gate must never silently
    broaden or guess the protected API.
    """
    available = set(signatures)
    exact = requested_names & available
    by_last: dict[str, set[str]] = {}
    by_normalized_last: dict[str, set[str]] = {}
    for name in available:
        last = name.rsplit(".", 1)[-1]
        by_last.setdefault(last, set()).add(name)
        normalized = re.sub(r"[^a-z0-9]", "", last.casefold())
        by_normalized_last.setdefault(normalized, set()).add(name)

    salvaged: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    unresolved: set[str] = set()
    for requested in sorted(requested_names - exact):
        requested_last = requested.rsplit(".", 1)[-1]
        candidates = by_last.get(requested_last, set())
        if not candidates:
            normalized = re.sub(
                r"[^a-z0-9]", "", requested_last.casefold()
            )
            candidates = by_normalized_last.get(normalized, set())
        if len(candidates) == 1:
            salvaged[requested] = next(iter(candidates))
        elif candidates:
            ambiguous[requested] = sorted(candidates)
        else:
            unresolved.add(requested)

    return {
        "resolved": exact | set(salvaged.values()),
        "exact": exact,
        "salvaged": salvaged,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
    }


def _signature_resolution_summary(
    requested_names: set[str], resolution: dict[str, Any]
) -> dict[str, Any]:
    """Make protected-name resolution explicit and JSON serializable."""
    resolved = set(resolution["resolved"])
    return {
        "requested_count": len(requested_names),
        "resolved_count": len(resolved),
        "complete": len(resolved) == len(requested_names),
        "exact": sorted(resolution["exact"]),
        "salvaged": dict(resolution["salvaged"]),
        "ambiguous": dict(resolution["ambiguous"]),
        "unresolved": sorted(resolution["unresolved"]),
    }


def _signature_change_reasons(before: SigEntry, after: SigEntry) -> list[str]:
    before_kind, before_type, before_value, _ = _sig_parts(before)
    after_kind, after_type, after_value, _ = _sig_parts(after)
    reasons: list[str] = []
    if before_kind != after_kind:
        reasons.append("kind")
    if before_type != after_type:
        reasons.append("type")
    if before_kind in {"def", "opaque"} and before_value is not None:
        if before_value != after_value:
            reasons.append("value")
    return reasons


def _diff_signatures(
    before: SigMap,
    after: SigMap,
    scout_names: set[str] | None = None,
) -> dict[str, Any]:
    """Compare Lean environment signature snapshots.

    If scout_names is provided (from data/<id>/signatures.json), those names
    are treated as the protected set regardless of the Lean-computed role.
    Otherwise falls back to the 'root' role assigned by the Lean dump script
    (unreferenced theorems).

    Returns a dict with:
      - preserved: bool — True if all protected declarations survived unchanged.
      - scout_used: bool — True if the scout file drove the protected set.
      - root_removed: list[str] — protected names gone after the patch.
      - root_changed: list[str] — protected declarations whose identity changed.
      - total_removed: list[str] — all declarations removed (informational).
      - total_changed: list[str] — all declarations changed (informational).
      - added: list[str] — new declarations (informational).
      - moved_to_dependency: list[str] — unchanged declarations now imported.
      - change_reasons: map from changed name to kind/type/value differences.
      - root_count: int — number of protected declarations in the baseline.
    """
    root_removed: list[str] = []
    root_changed: list[str] = []
    total_removed: list[str] = []
    total_changed: list[str] = []
    added: list[str] = []
    moved_to_dependency: list[str] = []
    change_reasons: dict[str, list[str]] = {}
    root_count = 0

    for name, before_entry in sorted(before.items()):
        _, _, _, role = _sig_parts(before_entry)
        if scout_names is not None:
            is_root = name in scout_names
        else:
            is_root = role == "root"
        if is_root:
            root_count += 1
        if name not in after:
            total_removed.append(name)
            if is_root:
                root_removed.append(name)
        else:
            after_entry = after[name]
            reasons = _signature_change_reasons(before_entry, after_entry)
            if reasons:
                total_changed.append(name)
                change_reasons[name] = reasons
                if is_root:
                    root_changed.append(name)
            before_role = _sig_parts(before_entry)[3]
            after_role = _sig_parts(after_entry)[3]
            if not reasons and before_role != "imported" and after_role == "imported":
                moved_to_dependency.append(name)

    for name in sorted(after):
        if name not in before and _sig_parts(after[name])[3] != "imported":
            added.append(name)

    return {
        "preserved": len(root_removed) == 0 and len(root_changed) == 0,
        "scout_used": scout_names is not None,
        "root_removed": root_removed,
        "root_changed": root_changed,
        "total_removed": total_removed,
        "total_changed": total_changed,
        "added": added,
        "moved_to_dependency": moved_to_dependency,
        "change_reasons": change_reasons,
        "root_count": root_count,
    }


def _resolution_passes(build_passed: bool, signature_diff: dict[str, Any]) -> bool:
    """A compiling patch resolves the task only when protected APIs survive."""
    return build_passed and bool(signature_diff.get("preserved", False))


LEANLEAN_TASK = """
Refactor the proofs in this codebase.
Remove all dead code and do real simplification, i.e. use powerful tactics or factor out shared logic to simplify the code while preserving the main results.
Success is measured by three factors:

(1) The repo compiling without errors and relying only on standard axioms.
(2) The benchmark's protected main results compile and remain equivalent. Run the no-argument command `lean_verify` from the repository root to verify the signatures, exit code 0 means the main results are preserved according to the lean comparator.
(3) The scoring metric is the reduction in non-whitespace, non-comment "Lean tokens" across all .lean files, compared to the starting state. Compressing imports does not count towards the compression metric. Run python3 proof_length.py to measure it.

When you believe there is no more refactoring to be done, output <submit> and terminate.
""".strip()

LEANLEAN_TASK_WITHOUT_METADATA = """
Refactor the proofs in this codebase.
Remove all dead code and do real simplification, i.e. use powerful tactics or factor out shared logic to simplify the code while preserving the main results.
Success is measured by three factors:

(1) The repo compiling without errors and relying only on standard axioms.
(2) The benchmark's protected main results compile and remain equivalent.
(3) The scoring metric is the reduction in non-whitespace, non-comment "Lean tokens" across all .lean files, compared to the starting state. Compressing imports does not count towards the compression metric.

When you believe there is no more refactoring to be done, output <submit> and terminate.
""".strip()

# One entry per repo. Each becomes a single benchmark instance.
# Commits are optional; if empty, the default branch HEAD is used.
LEANLEAN_REPOS: list[dict] = [
    {
        "instance_id": "strongpnt",
        "repo_url": "https://github.com/math-inc/strongpnt",
        "commit": "2f5835c322314f55f1026ec2f139d704b7c45c69",
        "exclude_dirs": [],
    },
    {
        "instance_id": "langlib",
        "repo_url": "https://github.com/nielstron/langlib",
        "commit": "4c6de7625d1b9756dfcf844618f1e91343096952",
        "exclude_dirs": ["test"],
    },
    {
        "instance_id": "degiorgi",
        "repo_url": "https://github.com/scottnarmstrong/DeGiorgi",
        "commit": "4c1b3077d3782b24065184df4ba59501b2e56fc7",
        "exclude_dirs": [],
    },
    {
        "instance_id": "physlib",
        "repo_url": "https://github.com/leanprover-community/physlib",
        "commit": "c6e61dce0a80e9b1139af2d81cac4dac886c4c29",
        "exclude_dirs": [],
        "target_dir": "Physlib",
        "build_target": "Physlib",
        "filesystem_isolated": True,
    },
    {
        "instance_id": "quantuminfo",
        "repo_url": "https://github.com/leanprover-community/physlib",
        "commit": "c6e61dce0a80e9b1139af2d81cac4dac886c4c29",
        "exclude_dirs": [],
        "target_dir": "QuantumInfo",
        "build_target": "QuantumInfo",
        "filesystem_isolated": True,
    },
]

def _required_signatures_from_contract(
    entry: dict[str, Any],
    repository_contracts: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    """Prefer the pinned repository-database contract when one is available."""

    entry_names = list(entry.get("required_signatures", []))
    instance_id = str(entry["instance_id"])
    contract = repository_contracts.get(instance_id)
    if contract is None:
        return entry_names, ""
    if not isinstance(contract, dict) or set(contract) != {
        "declarations",
        "source",
    }:
        raise ValueError(
            f"{instance_id}: repository contract must contain declarations and source"
        )
    declarations = contract["declarations"]
    source = contract["source"]
    if (
        not isinstance(declarations, list)
        or any(not isinstance(name, str) or not name for name in declarations)
        or len(declarations) != len(set(declarations))
        or not isinstance(source, str)
        or not source
    ):
        raise ValueError(f"{instance_id}: invalid repository contract")
    names = sorted(declarations)
    return names, source


@dataclass
class LeanLeanInstance(Instance):

    instance_id: str
    repo_url: str
    commit: str = ""
    exclude_dirs: list = field(default_factory=list)
    # File-level mode: when set, the agent is told to edit ONLY this file (path
    # relative to /testbed) and its captured diff is scoped to it (changes to
    # other files are dropped). Empty = repo-level (edit anything).
    target_file: str = ""
    # Subproject mode (LeanPool): when set (path relative to /testbed, e.g.
    # "LeanPool/Biswal"), the agent may only edit files under this directory (and
    # the sibling module file "<dir>.lean"); the diff and the compression metric
    # are scoped to it. Empty = not a subproject instance.
    target_dir: str = ""
    # Derived LeanPool images are real standalone repositories. Keep target_dir
    # for metric/diff scope, but do not pretend prompt text is the isolation
    # boundary: the container filesystem itself enforces it.
    filesystem_isolated: bool = False
    # Reconstruction scoring must re-elaborate submitted project sources.
    # Third-party dependency caches remain available for offline builds.
    clean_project_cache: bool = False
    # Module to build for the pass/fail gate + scoring build (e.g. "LeanPool.Biswal").
    # Empty = build the whole repo (`lake build`). Used to scope LeanPool builds to
    # one subproject so a single instance doesn't rebuild the ~120-project pool.
    build_target: str = ""
    # The repo this instance belongs to, used for the docker image, scout
    # signatures, and heartbeat baseline. Defaults to instance_id (repo-level);
    # file-level instances set it to the parent repo's id.
    base_instance_id: str = ""
    # Canonical API names supplied by a benchmark manifest. LeanPool populates
    # this from upstream projects.yml main_declarations/main_results.
    required_signatures: list[str] = field(default_factory=list)
    required_signatures_source: str = ""
    # Internal registered verifier contract. Its Comparator configuration, not
    # the evaluator's synthesized protected-result YAML, is exposed to agents.
    agent_verifier: dict[str, Any] | None = None
    patch: str = ""  # No golden reference — open-ended compression task.
    # Legacy aggregate plus independently controlled agent-visible commands.
    task_metadata_enabled: bool = True
    enable_lean_verify: bool = True
    enable_proof_length: bool = True
    task_prompt: str = ""
    task: str = ""
    repo: str = ""  # Populated in __post_init__; kept for Instance protocol.
    # Populated in __post_init__ to `leanlean-<instance_id>:latest`.
    # Build the image with scripts/build_leanlean_images.sh before use.
    docker_image: str = ""
    compile_gate_enabled: bool = False
    compile_gate_command: str = "lake build"
    compile_gate_max_retries: int = 1
    compile_gate_output_chars: int = 12000
    build_jobs: int = -1  # -1 = use all available cores
    # Cap CPU cores available to the container (and thus to the agent's
    # in-loop `lake build` calls). -1 = no cap.
    container_cpus: int = 12
    # Hard memory ceiling for the container (docker --memory value, e.g. "100g").
    # Empty = no cap. A safety net so a runaway build can't take down a shared
    # host; the container is OOM-killed instead.
    container_memory: str = "100g"
    # Shared cgroup parent (systemd slice, e.g. "lean.slice") for every container.
    # Empty = default cgroup. When set, all concurrent containers draw from this
    # one slice's MemoryMax, so their TOTAL RSS is capped regardless of worker
    # count -- the per-container `container_memory` still caps each one alone.
    container_cgroup_parent: str = ""
    container_timeout: str = "2h"
    network_policy: str = "model_proxy_only"
    container_pids_limit: int = 4096
    container_resource_visibility: str = "host"
    # Extra seconds the container's `sleep` outlives the agent's exec timeout.
    container_grace_seconds: int = 1800

    def __post_init__(self):
        if not self.base_instance_id:
            self.base_instance_id = self.instance_id
        if not self.task:
            if self.task_prompt:
                self.task = self.task_prompt
                if "{theorem}" in self.task:
                    verifier = self.agent_verifier or {}
                    if verifier.get("schema") != RECONSTRUCTION_VALIDATION_SCHEMA:
                        raise ValueError("theorem prompt requires a reconstruction verifier")
                    self.task = self.task.replace("{theorem}", verifier["declaration"])
            elif self.task_metadata_enabled:
                self.task = LEANLEAN_TASK
            else:
                self.task = LEANLEAN_TASK_WITHOUT_METADATA
        if self.target_file:
            # Minimal file-level scoping instruction (diff is also scoped to this
            # file at extraction time, so other edits are reverted regardless).
            self.task = (
                f"{self.task}\n\n"
                f"You are ONLY allowed to edit the file `{self.target_file}`. "
                f"Do not modify any other file; changes to other files will be reverted."
            )
        if self.target_dir and not self.filesystem_isolated:
            # Raw monorepo subproject scoping: derived images enforce this with
            # a standalone filesystem instead of a prompt-only restriction.
            # directory at extraction time, so edits elsewhere are reverted.
            self.task = (
                f"{self.task}\n\n"
                f"You are ONLY allowed to edit files under the directory "
                f"`{self.target_dir}/` (and its sibling module file "
                f"`{self.target_dir}.lean`). Do not modify any other file; "
                f"changes to other files will be reverted."
            )
        # Scope the in-loop compile gate to the subproject too (default was the
        # whole-repo `lake build`), so the agent's submission gate matches scoring.
        if self.build_target and self.compile_gate_command == "lake build":
            self.compile_gate_command = f"lake build {self.build_target}"
        if not self.repo:
            # e.g. "math-inc/strongpnt" from "https://github.com/math-inc/strongpnt"
            self.repo = self.repo_url.removeprefix("https://github.com/").removesuffix(".git")
        image_override = repo_image_overrides_from_env().get(
            self.base_instance_id
        )
        if image_override:
            self.docker_image = image_override
        elif not self.docker_image:
            # Select the explicit semantic repository baseline. The default is
            # optimized (tree-shaken plus conservative trivial-proof
            # normalization). LEANLEAN_USE_PROD remains a compatibility
            # fallback: true -> optimized, false -> raw.
            repo_variant = repo_variant_from_env()
            # Key the image off the parent repo (base_instance_id), so file-level
            # instances reuse their repo's prebuilt image rather than looking for
            # a (nonexistent) per-file image.
            self.docker_image = repo_image_tag(
                self.base_instance_id,
                repo_variant,
            )

    @property
    def build_command(self) -> str:
        """The `lake build` command for scoring/gating. Scoped to a single module
        (`lake build <build_target>`) for LeanPool subprojects so one instance
        doesn't rebuild the whole ~120-project pool; whole-repo `lake build`
        otherwise."""
        return f"lake build {self.build_target}" if self.build_target else "lake build"

    @property
    def _metric_include_prefix(self) -> str:
        """Subtree (relative to /testbed) the compression metric is scoped to, or
        "" for whole-repo. Set for LeanPool subproject instances via target_dir."""
        return self.target_dir

    @property
    def metric_exclude_files(self) -> list[str]:
        if not isinstance(self.agent_verifier, dict):
            return []
        challenge = self.agent_verifier.get("challenge")
        if not isinstance(challenge, dict):
            return []
        return [
            _safe_repository_path(
                challenge.get("source_path"), "registered Challenge path"
            )
        ]

    def _signature_seed(self) -> tuple[set[str] | None, str]:
        """Return the authoritative protected API and where it came from."""
        if self.required_signatures_source:
            return set(self.required_signatures), self.required_signatures_source
        if self.required_signatures:
            return set(self.required_signatures), "benchmark_manifest"
        scout_names = _load_scout_signatures(self.base_instance_id)
        if scout_names is not None:
            return scout_names, "scout"
        return None, "root_fallback"

    def get_submission_gate(self) -> dict[str, Any]:
        return {
            "enabled": self.compile_gate_enabled,
            "max_retries": max(0, int(self.compile_gate_max_retries)),
        }

    def validate_submission(self, env: Environment) -> dict[str, Any]:
        command = self.compile_gate_command
        exec_result = env.execute(command)
        output = exec_result.get("output", "")
        output = output[: self.compile_gate_output_chars]
        return {
            "passed": exec_result.get("returncode", 1) == 0,
            "command": command,
            "returncode": exec_result.get("returncode"),
            "output": output,
        }

    def build_submission_feedback(self, validation_result: dict[str, Any]) -> str:
        return textwrap.dedent(
            f"""
            Your previous submission was rejected by the compile gate.
            Validation command: `{validation_result.get("command", "")}`
            Return code: {validation_result.get("returncode")}

            `lake build` output:
            ```text
            {validation_result.get("output", "")}
            ```

            Please fix the proofs so `lake build` succeeds, then submit again.
            """
        ).strip()

    def get_dir(self, base_dir: Path) -> Path:
        instance_dir = base_dir / self.instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        return instance_dir

    run_tag: str = "latest"

    @property
    def _run_image_tag(self) -> str:
        """The TAG component (after `:`) shared by this instance's run images.

        Docker repository names must be lowercase and forbid dots, but file-level
        instance_ids carry both (e.g. `StrongPNT_Foo.lean`); the tag component
        permits uppercase/dots/underscores, so the per-file identity lives here
        (the repository name keys off the lowercase repo id). Empty suffix for
        repo-level instances, so their tags are byte-identical to before. Bounded
        to docker's 128-char tag limit (hashed if the readable form overflows,
        which deep langlib paths + a long run_tag can do)."""
        scope_key = self.target_file or self.target_dir
        if not scope_key:
            return self.run_tag
        suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", scope_key)
        tag = f"{self.run_tag}_{suffix}"
        if len(tag) > 128:
            digest = hashlib.sha1(tag.encode()).hexdigest()[:12]
            tag = f"{self.run_tag[:100]}_{digest}"
        return tag

    @property
    def _cache_image_tag(self) -> str:
        """Docker tag for the committed build-cache image."""
        return f"leanlean-{self.base_instance_id}-cache:{self._run_image_tag}"

    @property
    def _pinned_base_tag(self) -> str:
        """Immutable per-run snapshot of the base image.

        The shared `leanlean-<id>:latest` tag is mutable — rebuilding
        images (e.g. changing a pin in build_leanlean_images.sh)
        overwrites it. If that happens between a run and a later
        reconcile/evaluate, the base the run actually used is lost. We snapshot
        `:latest` into this per-run tag at setup time and run against it, so the
        run's exact base survives regardless of what happens to `:latest`.

        Keyed off the parent repo (base_instance_id) for the repository name so
        file-level instances produce valid (lowercase) image names; per-file
        uniqueness lives in the tag suffix.
        """
        return f"leanlean-{self.base_instance_id}-base:{self._run_image_tag}"

    def _pin_base_image(self) -> str:
        """Tag the current `:latest` base under the per-run base tag.

        Returns the resolved image ID of the base for the record. Best-effort:
        if tagging fails we fall back to the mutable tag rather than aborting.
        """
        base_id = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.docker_image],
            capture_output=True, text=True,
        ).stdout.strip()
        tag_result = subprocess.run(
            ["docker", "tag", self.docker_image, self._pinned_base_tag],
            capture_output=True, text=True,
        )
        if tag_result.returncode != 0:
            logger.warning(
                "Failed to pin base image %s -> %s: %s",
                self.docker_image, self._pinned_base_tag, tag_result.stderr.strip(),
            )
            return base_id
        logger.info(
            "Pinned base image %s (id=%s) as %s",
            self.docker_image, base_id or "unknown", self._pinned_base_tag,
        )
        return base_id

    def _has_cache_image(self) -> bool:
        """Check if a committed build-cache image exists locally."""
        result = subprocess.run(
            ["docker", "image", "inspect", self._cache_image_tag],
            capture_output=True,
        )
        return result.returncode == 0

    def _remove_cache_image(self) -> None:
        """Remove the committed build-cache image."""
        subprocess.run(
            ["docker", "rmi", self._cache_image_tag],
            capture_output=True,
        )

    @staticmethod
    def _image_exists(tag: str) -> bool:
        return subprocess.run(
            ["docker", "image", "inspect", tag], capture_output=True
        ).returncode == 0

    def _ensure_prebuilt_image_exists(self) -> None:
        """Fail fast with a clear message if the prebuilt image is missing.

        Historical ``-prod`` configurations retain their raw fallback. Explicit
        semantic ``-stripped`` and ``-optimized`` selections intentionally fail
        rather than silently running a different baseline.
        """
        if self._image_exists(self.docker_image):
            return
        if self.docker_image.endswith("-prod:latest"):
            base = self.docker_image.replace("-prod:latest", ":latest")
            if self._image_exists(base):
                logger.warning(
                    "Legacy prod image %s not found; falling back to base %s. "
                    "Prefer an explicit repo_variant in new experiments.",
                    self.docker_image, base,
                )
                self.docker_image = base
                return
        raise RuntimeError(
            f"Docker image {self.docker_image!r} not found locally.\n"
            f"Stripped images are committed by preprocessing:\n"
            f"    bash preprocess.sh configs/preprocessing/<config>.yaml\n"
            f"or build the raw image with:\n"
            f"    bash scripts/build_leanlean_images.sh "
            f"{self.instance_id}\n"
            f"(or omit the argument to build all instances)."
        )

    def _materialize_before_agent(self) -> None:
        record = materialization_specs_from_env().get(self.base_instance_id)
        if record is None:
            return
        expected_tag = str(record.get("tag") or "")
        if self.docker_image != expected_tag:
            raise RuntimeError(
                f"{self.instance_id}: image override {self.docker_image!r} "
                f"does not match materialization tag {expected_tag!r}"
            )
        ensure_materialized_image(self.base_instance_id)

    def retire_materialized_images(self) -> list[dict[str, Any]]:
        record = materialization_specs_from_env().get(self.base_instance_id)
        if record is None or bool(record.get("persist")):
            return []
        return retire_image_tags(
            self._cache_image_tag,
            self._pinned_base_tag,
            self.docker_image,
        )
    def setup(self, env_config: dict[str, Any], setup_repo: bool = True) -> Environment:
        """Start a container from the prebuilt per-instance image.

        The image (built via scripts/build_leanlean_images.sh) already
        has the repo cloned, elan + the right Lean toolchain installed, the
        mathlib cache warmed, .lake/ populated, and a clean single-commit git
        state. This method just spins up the container from it, so applied
        patches in solve() trigger only an incremental `lake build`.
        """
        del setup_repo  # Prebuilt image already has the repo set up.
        self._materialize_before_agent()
        self._ensure_prebuilt_image_exists()
        # Snapshot the (mutable) :latest base into an immutable per-run tag and
        # run against that snapshot, so the run's exact base survives even if
        # :latest is rebuilt later (which is what corrupted an earlier run).
        self._pin_base_image()
        env_config["image"] = self._pinned_base_tag
        env_config["cwd"] = "/testbed"
        # Incremental lake builds are fast, but the timeout also caps the
        # container lifetime during submission-gate retries.
        env_config["timeout"] = 1800  # 30 minutes
        env_config["container_timeout"] = self.container_timeout
        self._apply_cpu_limit(env_config)
        env = get_environment(env_config, default_type="docker")
        try:
            _hide_task_documents(env)
            if self.task_metadata_enabled:
                if self.agent_verifier is not None and self.enable_lean_verify:
                    verifier_schema = self.agent_verifier.get("schema")
                    if same_identifier(verifier_schema, PALOMAR_VALIDATION_SCHEMA):
                        _install_palomar_task_metadata(
                            env,
                            contract=self.agent_verifier,
                            exclude_dirs=self.exclude_dirs,
                            include_prefix=self._metric_include_prefix,
                            build_jobs=self.build_jobs,
                            prewarm=not bool(self.agent_verifier.get("reconstruction_targets")),
                            enable_lean_verify=self.enable_lean_verify,
                            enable_proof_length=self.enable_proof_length,
                        )
                    elif verifier_schema == RECONSTRUCTION_VALIDATION_SCHEMA:
                        _install_reconstruction_task_metadata(
                            env,
                            contract=self.agent_verifier,
                            exclude_dirs=self.exclude_dirs,
                            include_prefix=self._metric_include_prefix,
                        )
                    else:
                        raise RuntimeError("unsupported agent verifier schema")
                elif self.enable_proof_length:
                    protected_names, protected_source = self._signature_seed()
                    _install_task_metadata(
                        env,
                        instance_id=self.instance_id,
                        build_command=self.build_command,
                        exclude_dirs=self.exclude_dirs,
                        include_prefix=self._metric_include_prefix,
                        protected_names=protected_names,
                        protected_source=protected_source,
                    )
                    if self.enable_lean_verify and protected_source == "leanlean_theorem_holdout_v1":
                        from leanlean.holdout_verifier import install_holdout_verifier
                        install_holdout_verifier(
                            env, names=set(protected_names or ()),
                            build_target=self.build_target, build_jobs=self.build_jobs,
                        )
        except BaseException:
            cleanup = getattr(env, "cleanup", None)
            if callable(cleanup):
                cleanup()
            raise
        return env

    def _apply_cpu_limit(self, env_config: dict[str, Any]) -> None:
        """Cap container resources so the agent's lake builds don't saturate the host.

        Sets a CPU-quota (`--cpus`) and, when configured, a hard memory ceiling
        (`--memory`) so a runaway build is OOM-killed rather than taking the host
        (and everyone else on it) down.
        """
        # Start with no route. Proxy-backed agents may later attach an
        # internal network whose only peer is a fixed-target model relay.
        env_config["network_policy"] = self.network_policy
        env_config["network_mode"] = "none"
        env_config["container_pids_limit"] = self.container_pids_limit
        env_config["container_grace_seconds"] = self.container_grace_seconds
        if self.container_resource_visibility != "host":
            env_config["resource_visibility"] = self.container_resource_visibility
            env_config["resource_threads"] = self.build_jobs
        run_args = list(env_config.get("run_args", ["--rm"]))
        if self.container_cpus is not None and self.container_cpus >= 0:
            if not any(a.startswith("--cpus") for a in run_args):
                run_args.append(f"--cpus={self.container_cpus}")
        if self.container_memory and not any(a.startswith("--memory") for a in run_args):
            run_args.append(f"--memory={self.container_memory}")
            # Disable swap accounting so the cap is on real RAM (memory==memory-swap).
            run_args.append(f"--memory-swap={self.container_memory}")
        if self.container_cgroup_parent and not any(
            a.startswith("--cgroup-parent") for a in run_args
        ):
            # All containers under one slice -> their summed RSS is capped by the
            # slice's MemoryMax (a true host-wide total, not workers*per-container).
            run_args.append(f"--cgroup-parent={self.container_cgroup_parent}")
        env_config["run_args"] = run_args

    def setup_with_cache(self, env_config: dict[str, Any]) -> Environment:
        """Start a container from the build-cache image.

        The cache image was committed by the generate step after the agent ran
        and `lake build` succeeded.  It contains the full .lake/ build artifacts.
        We reset the source tree to the clean baseline so the patch can be
        re-applied independently, but .lake/ (untracked) survives the reset,
        making the subsequent `lake build` near-instant.
        """
        env_config["image"] = self._cache_image_tag
        env_config["cwd"] = "/testbed"
        env_config["timeout"] = 1800
        env_config["container_timeout"] = self.container_timeout
        self._apply_cpu_limit(env_config)
        env = get_environment(env_config, default_type="docker")
        # Reset source to the init commit so the submitted patch (which was
        # diffed against that same commit) applies cleanly.  .lake/ is
        # untracked and survives the checkout for fast incremental builds.
        init = "$(cat /.init_commit 2>/dev/null || git rev-list --max-parents=0 HEAD | tail -1)"
        env.execute(
            f"cd /testbed && "
            f"git reset {init} -- . && "
            f"git checkout -- . && "
            f"git clean -fd --exclude='.lake/'"
        )
        return env

    def _compile_result_path(self, base_dir: Path) -> Path:
        return base_dir / self.instance_id / "compile_result.json"

    @staticmethod
    def _appender(log_file: Path):
        def _append_log(*lines: str) -> None:
            with open(log_file, "a") as f:
                for line in lines:
                    f.write(f"{line}\n")
                f.flush()
        return _append_log

    def _apply_and_build(self, env, model_patch, run_repo_test, append_log):
        """Reset the working tree to the pristine baseline, apply `model_patch`
        (a cumulative baseline->round diff), and run `lake build`.

        `.lake/` is excluded from the reset/clean, so when this runs repeatedly
        in one live container the build is incremental against the previous
        round's oleans. Returns (apply_result, build_result, build_passed).
        """
        empty_patch = not bool(model_patch.strip())
        init = "$(cat /.init_commit 2>/dev/null || git rev-list --max-parents=0 HEAD | tail -1)"
        env.execute(
            f"cd /testbed && git reset {init} -- . && git checkout -- . "
            f"&& git clean -fd --exclude='.lake/'"
        )
        if not empty_patch:
            apply_patch_cmd = (
                f"cd /testbed && "
                f"echo {shlex.quote(model_patch)} | git apply --whitespace=nowarn -"
            )
            apply_result = env.execute(apply_patch_cmd)
            append_log(
                "", "[apply_patch]",
                f"command={apply_patch_cmd}",
                f"returncode={apply_result.get('returncode')}",
                "output:", apply_result.get("output", ""),
            )
        else:
            apply_result = {"returncode": 0, "output": ""}
            append_log("", "[apply_patch]", "skipped=true (empty patch)")

        if run_repo_test:
            if self.clean_project_cache:
                from leanlean.preprocessing.cache_isolation import render_container_cache_cleanup
                cleanup = env.execute(
                    "python3 -c " + shlex.quote(render_container_cache_cleanup(clear_project=True))
                )
                append_log("", "[clean_project_cache]", str(cleanup))
                if cleanup.get("returncode", 1) != 0:
                    raise RuntimeError("project build cache cleanup failed")
            import time as _time
            # NOTE: `lake build` has no `-j`/`--jobs` flag, so we don't cap build
            # parallelism here. The host is protected at the container level: a
            # `--cpus` CPU-time quota plus a hard `--memory` ceiling (see
            # _apply_cpu_limit) that OOM-kills a runaway build instead of taking
            # the shared host down.
            threads_prefix = "" if self.build_jobs < 0 else f"LEAN_NUM_THREADS={self.build_jobs} "
            build_cmd = f"cd /testbed && {threads_prefix}{self.build_command}"
            _build_start = _time.monotonic()
            build_exec = env.execute(build_cmd, timeout=LEAN_VERIFY_TIMEOUT_SECONDS)
            build_time_seconds = round(_time.monotonic() - _build_start, 2)
            build_passed = build_exec.get("returncode", 1) == 0
            build_result = {
                "ran": True,
                "passed": build_passed,
                "returncode": build_exec.get("returncode"),
                "errors": "" if build_passed else build_exec.get("output", ""),
                "build_time_seconds": build_time_seconds,
                **({"project_cache_policy": "clean_project_preserve_dependencies"}
                   if self.clean_project_cache else {}),
            }
            append_log(
                "", "[lake_build]",
                f"command={build_cmd}",
                f"returncode={build_exec.get('returncode')}",
                f"build_time_seconds={build_time_seconds}",
                "output:", build_exec.get("output", ""),
            )
            if (
                build_passed
                and (
                    self.required_signatures_source == "leanlean_theorem_holdout_v1"
                    or (
                        isinstance(self.agent_verifier, dict)
                        and self.agent_verifier.get("schema") == RECONSTRUCTION_VALIDATION_SCHEMA
                    )
                )
            ):
                verify_start = _time.monotonic()
                verify_exec = env.execute(
                    "cd /testbed && "
                    "LEANLEAN_BENCHMARK_INTERNAL=1 lean_verify",
                    timeout=LEAN_VERIFY_TIMEOUT_SECONDS,
                )
                verify_seconds = round(_time.monotonic() - verify_start, 2)
                verify_passed = verify_exec.get("returncode", 1) == 0
                verification_key = (
                    "compression_verification"
                    if self.required_signatures_source == "leanlean_theorem_holdout_v1"
                    else "reconstruction_verification"
                )
                if verification_key == "reconstruction_verification":
                    build_result["reproof_edit_policy"] = self.agent_verifier.get("edit_policy", "target_file_only")
                build_result[verification_key] = {
                    "ran": True,
                    "passed": verify_passed,
                    "returncode": verify_exec.get("returncode"),
                    "time_seconds": verify_seconds,
                    "output": verify_exec.get("output", ""),
                }
                build_passed = build_passed and verify_passed
                build_result["passed"] = build_passed
                if not verify_passed:
                    build_result["errors"] = verify_exec.get("output", "")
                append_log(
                    "",
                    "[theorem_reconstruction_verify]",
                    f"returncode={verify_exec.get('returncode')}",
                    f"time_seconds={verify_seconds}",
                    "output:",
                    verify_exec.get("output", ""),
                )
        else:
            build_passed = True
            build_result = {"ran": False, "passed": True, "returncode": 0, "errors": ""}
            append_log("", "[lake_build]", "skipped=true")
        return apply_result, build_result, build_passed

    def _measure_baseline(self, env, measure_heartbeats=True) -> dict:
        # Raw LeanPool instances share a monorepo image, where a target-scoped
        # heartbeat sweep is not a trustworthy standalone-project measurement.
        # Derived images are physically isolated, so measure their target subtree.
        if self.target_dir and not self.filesystem_isolated:
            measure_heartbeats = False
        include_prefix = self._metric_include_prefix
        baseline_words = _measure_words(
            env,
            exclude_dirs=self.exclude_dirs,
            include_prefix=include_prefix,
            exclude_files=self.metric_exclude_files,
        )
        baseline_lean_tokens = _measure_lean_tokens(
            env,
            exclude_dirs=self.exclude_dirs,
            include_prefix=include_prefix,
            exclude_files=self.metric_exclude_files,
        )
        requested_names, protected_source = self._signature_seed()
        signature_modules = _modules_under_target(env, self.target_dir)
        baseline_sigs = _collect_lean_signatures(
            env,
            modules=signature_modules,
            protected_names=requested_names,
        )
        resolution_summary = None
        if requested_names is not None:
            resolution = _resolve_requested_signature_names(
                baseline_sigs, requested_names
            )
            protected_names = set(resolution["resolved"])
            resolution_summary = _signature_resolution_summary(
                requested_names, resolution
            )
        else:
            protected_names = _protected_names(baseline_sigs, None)
        # The pristine-tree heartbeat count never changes for a pinned repo, so
        # it's persisted per-repo in data/<instance_id>/heartbeats.txt (alongside
        # the scout signatures). Read it instead of re-sweeping every file each
        # run; the first run for an instance measures it live and persists it.
        baseline_heartbeats = _load_baseline_heartbeats(self.base_instance_id)
        if baseline_heartbeats is None and measure_heartbeats:
            logger.info(
                "No persisted baseline heartbeats for %s; measuring once and saving",
                self.base_instance_id,
            )
            baseline_heartbeats = _measure_heartbeats(
                env,
                exclude_dirs=self.exclude_dirs,
                include_prefix=include_prefix,
                exclude_files=self.metric_exclude_files,
            )
            _save_baseline_heartbeats(self.base_instance_id, baseline_heartbeats)
        elif baseline_heartbeats is None:
            # Heartbeat measurement disabled and none cached: use the -1 sentinel
            # (analyze_round maps it to a null heartbeat_ratio). Don't persist it,
            # so a later run with measurement enabled still records the real value.
            baseline_heartbeats = -1
        # Count only the .lean files that actually feed the compression metric
        # (see _lean_file_count_find_cmd: skips .lake, lakefile.lean, exclude_dirs).
        lean_file_count_result = env.execute(
            _lean_file_count_find_cmd(
                self.exclude_dirs,
                include_prefix=include_prefix,
                exclude_files=self.metric_exclude_files,
            )
        )
        try:
            lean_file_count = int(lean_file_count_result.get("output", "0").strip())
        except (ValueError, TypeError):
            lean_file_count = None
        return {
            "baseline_words": baseline_words,
            "baseline_lean_tokens": baseline_lean_tokens,
            "baseline_decl_count": _project_decl_count(baseline_sigs),
            "baseline_sigs": {k: list(v) for k, v in baseline_sigs.items()},
            "protected_names": sorted(protected_names),
            "protected_requested": sorted(requested_names or ()),
            "protected_source": protected_source,
            "protected_resolution": resolution_summary,
            "baseline_heartbeats": baseline_heartbeats,
            "lean_file_count": lean_file_count,
        }

    def _measure_post(
        self, env, run_repo_test, build_passed, protected_names,
        measure_heartbeats=True,
    ) -> dict:
        """Build-dependent post-patch measurements (need the live container)."""
        heartbeat_scope_valid = not self.target_dir or self.filesystem_isolated
        heartbeats = (
            _measure_heartbeats(
                env,
                exclude_dirs=self.exclude_dirs,
                include_prefix=self._metric_include_prefix,
                exclude_files=self.metric_exclude_files,
            )
            if (
                run_repo_test
                and build_passed
                and measure_heartbeats
                and heartbeat_scope_valid
            )
            else -1
        )
        post_words = _measure_words(
            env, exclude_dirs=self.exclude_dirs,
            include_prefix=self._metric_include_prefix,
            exclude_files=self.metric_exclude_files,
        )
        post_lean_tokens = _measure_lean_tokens(
            env,
            exclude_dirs=self.exclude_dirs,
            include_prefix=self._metric_include_prefix,
            exclude_files=self.metric_exclude_files,
        )
        if run_repo_test and build_passed:
            signature_modules = _modules_under_target(env, self.target_dir)
            post_sigs = _collect_lean_signatures(
                env,
                modules=signature_modules,
                protected_names=set(protected_names),
            )
        else:
            post_sigs = {}
        return {
            "heartbeats": heartbeats,
            "post_words": post_words,
            "post_lean_tokens": post_lean_tokens,
            "post_decl_count": _project_decl_count(post_sigs),
            "post_sigs": {k: list(v) for k, v in post_sigs.items()},
        }

    def compile_rounds(
        self,
        rounds: "list[tuple[int, str, Path]]",
        base_dir: Path,
        run_id: int,
        run_repo_test: bool = True,
        measure_heartbeats: bool = True,
    ) -> bool:
        """Phase 1 (docker): build every round for this instance in ONE live
        container started from the base prebuilt image. `.lake/` warms
        round-over-round, so each round only recompiles what changed.

        `rounds` is (round_num, cumulative_patch, round_base_dir) sorted by
        round. Writes a compile_result.json per round (build result, heartbeats,
        post words/sigs, shared baseline). No images are persisted -- the
        container is discarded at the end.
        """
        if not rounds:
            return True
        top_dir = rounds[0][2] / self.instance_id
        top_dir.mkdir(parents=True, exist_ok=True)
        top_log = top_dir / "execution.log"
        top_log.write_text("")
        append_log = self._appender(top_log)
        logger.info(f"[compile] instance {self.instance_id}: {len(rounds)} round(s)")

        env = None
        ok = False
        try:
            # One container walks every round's apply+build in sequence, so its
            # `sleep` must outlive ALL of them (each build gets its own exec
            # timeout). Without scaling, the container is sized for a single
            # round and self-removes mid-eval on later rounds, which surfaces as
            # "No such container" in the apply/build result of every round after
            # it expires. Mirror the generate path (generate.py passes
            # refactor_rounds through for the same reason).
            env = self.setup({"refactor_rounds": max(0, len(rounds) - 1)})
            append_log(f"instance_id={self.instance_id}", f"run_id={run_id}", "setup=ok (base prebuilt)")
            if self.target_dir and run_repo_test:
                threads_prefix = (
                    "" if self.build_jobs < 0 else f"LEAN_NUM_THREADS={self.build_jobs} "
                )
                baseline_build_command = (
                    f"cd /testbed && {threads_prefix}{self.build_command}"
                )
                baseline_build = env.execute(baseline_build_command, timeout=LEAN_VERIFY_TIMEOUT_SECONDS)
                append_log(
                    "", "[baseline_build]",
                    f"command={baseline_build_command}",
                    f"returncode={baseline_build.get('returncode')}",
                    "output:", baseline_build.get("output", ""),
                )
                if baseline_build.get("returncode", 1) != 0:
                    output = baseline_build.get("output", "")[-12000:]
                    raise RuntimeError(f"pristine member build failed:\n{output}")
            baseline = self._measure_baseline(env, measure_heartbeats=measure_heartbeats)
            append_log("", "[baseline_metrics]",
                       f"words={baseline['baseline_words']}",
                       f"lean_tokens={baseline['baseline_lean_tokens']}",
                       f"declarations={baseline['baseline_decl_count']}",
                       f"heartbeats={baseline['baseline_heartbeats']}")
            for round_num, patch, round_base in rounds:
                inst_dir = round_base / self.instance_id
                inst_dir.mkdir(parents=True, exist_ok=True)
                rlog = inst_dir / "execution.log"
                if rlog != top_log:  # don't clobber the baseline log written above
                    rlog.write_text("")
                rappend = self._appender(rlog)
                rappend(f"instance_id={self.instance_id}", f"round={round_num}")
                model_patch = patch or ""
                empty_patch = not bool(model_patch.strip())
                apply_result, build_result, build_passed = self._apply_and_build(
                    env, model_patch, run_repo_test, rappend,
                )
                post = self._measure_post(
                    env, run_repo_test, build_passed, baseline["protected_names"],
                    measure_heartbeats=measure_heartbeats,
                )
                build_result["heartbeats"] = post["heartbeats"]
                compile_result = {
                    "instance_id": self.instance_id,
                    "round": round_num,
                    "model_patch": model_patch,
                    "empty_patch": empty_patch,
                    "run_repo_test": run_repo_test,
                    "build_passed": build_passed,
                    "apply_patch_result_raw": apply_result,
                    "build_result": build_result,
                    "baseline": baseline,
                    "post": post,
                }
                with open(self._compile_result_path(round_base), "w") as f:
                    json.dump(compile_result, f, indent=2)
                rappend("", "[post_metrics]",
                        f"words={post['post_words']}",
                        f"lean_tokens={post['post_lean_tokens']}",
                        f"heartbeats={post['heartbeats']}",
                        f"build_passed={build_passed}")
            ok = True
        except Exception as e:
            logger.error(f"Error during compile of instance {self.instance_id}: {e}")
            logger.error(traceback.format_exc())
            append_log("", "[exception]", str(e), traceback.format_exc())
        finally:
            if env is not None:
                try:
                    env.cleanup()
                except Exception:
                    pass
        return ok

    def analyze_round(self, base_dir: Path, run_id: int) -> bool:
        """Phase 2 (no docker): derive metrics from compile_result.json + the
        trajectory and write report.json. Re-runnable; needs no container.
        """
        instance_dir = base_dir / self.instance_id
        comp_path = self._compile_result_path(base_dir)
        if not comp_path.exists():
            raise FileNotFoundError(
                f"No compile_result.json for {self.instance_id} at {comp_path}; "
                f"run the compile phase first."
            )
        comp = json.loads(comp_path.read_text())
        model_patch = comp.get("model_patch", "")
        empty_patch = comp.get("empty_patch", not bool(model_patch.strip()))
        build_passed = comp.get("build_passed", False)
        apply_result = comp.get("apply_patch_result_raw", {"returncode": None, "output": ""})
        build_result = dict(comp.get("build_result", {}))
        baseline = comp.get("baseline", {})
        post = comp.get("post", {})
        baseline_words = baseline.get("baseline_words", 0)
        baseline_lean_tokens = baseline.get("baseline_lean_tokens")
        baseline_heartbeats = baseline.get("baseline_heartbeats", -1)
        baseline_sigs = {k: tuple(v) for k, v in baseline.get("baseline_sigs", {}).items()}
        post_sigs = {k: tuple(v) for k, v in post.get("post_sigs", {}).items()}
        post_words = post.get("post_words", baseline_words)
        post_lean_tokens = post.get("post_lean_tokens")
        post_heartbeats = post.get("heartbeats", build_result.get("heartbeats", -1))

        source_measurements = {
            "baseline_words": baseline_words,
            "post_words": post_words,
            "baseline_lean_tokens": baseline_lean_tokens,
            "post_lean_tokens": post_lean_tokens,
        }
        invalid_measurements = [
            name
            for name, value in source_measurements.items()
            if not isinstance(value, (int, float))
            or (
                name.startswith("baseline_")
                and value <= 0
            )
            or (
                name.startswith("post_")
                and value < 0
            )
        ]
        if invalid_measurements:
            raise RuntimeError(
                "source compression metrics are missing or invalid: "
                + ", ".join(invalid_measurements)
            )

        requested_names, protected_source = self._signature_seed()
        resolution_summary = None
        if requested_names is not None:
            resolution = _resolve_requested_signature_names(
                baseline_sigs, requested_names
            )
            protected_names = set(resolution["resolved"])
            resolution_summary = _signature_resolution_summary(
                requested_names, resolution
            )
        else:
            protected_names = None
        sig_diff = _diff_signatures(
            baseline_sigs, post_sigs, scout_names=protected_names
        )
        sig_diff["protected_source"] = protected_source
        sig_diff["protected_resolution"] = resolution_summary

        # lean_file_count is a baseline property of the repo image (patch-/round-
        # independent), so recompute it here with a quick `find` against the
        # prebuilt image — no `lake build`. This corrects results compiled before
        # the metric-scope fix (exclude_dirs + lakefile.lean) without a costly
        # recompile. Falls back to the persisted value if docker is unavailable.
        lean_file_count = _baseline_lean_file_count(
            self.docker_image, self.exclude_dirs,
            include_prefix=self._metric_include_prefix,
        )
        if lean_file_count is None:
            lean_file_count = baseline.get("lean_file_count")

        metrics: dict[str, Any] = {
            "lean_file_count": lean_file_count,
            "baseline_words": baseline_words,
            "baseline_lean_tokens": baseline_lean_tokens,
            "baseline_decl_count": baseline.get("baseline_decl_count", len(baseline_sigs)),
            "build_time_seconds": build_result.get("build_time_seconds"),
            "baseline_heartbeats": baseline_heartbeats,
            "heartbeats": post_heartbeats,
            "post_words": post_words,
            "post_lean_tokens": post_lean_tokens,
            "post_decl_count": post.get(
                "post_decl_count", _project_decl_count(post_sigs)
            ),
            "baseline_sigs": {k: list(v) for k, v in baseline_sigs.items()},
            "post_sigs": {k: list(v) for k, v in post_sigs.items()},
            "signatures": sig_diff,
        }
        # post_words is -1 when the build failed (never measured). Guard against
        # it so a failed build doesn't masquerade as ~100% compression
        # (ratio = -1/baseline ≈ 0, words_saved = baseline + 1).
        if baseline_words > 0 and post_words >= 0:
            metrics["compression_ratio"] = round(post_words / baseline_words, 6)
            metrics["words_saved"] = baseline_words - post_words
        else:
            metrics["compression_ratio"] = None
            metrics["words_saved"] = None

        if (
            isinstance(baseline_lean_tokens, (int, float))
            and isinstance(post_lean_tokens, (int, float))
            and baseline_lean_tokens > 0
            and post_lean_tokens >= 0
        ):
            metrics["lean_token_ratio"] = round(
                post_lean_tokens / baseline_lean_tokens, 6
            )
            metrics["lean_tokens_saved"] = (
                baseline_lean_tokens - post_lean_tokens
            )
        else:
            metrics["lean_token_ratio"] = None
            metrics["lean_tokens_saved"] = None

        if baseline_heartbeats > 0 and post_heartbeats >= 0:
            metrics["heartbeat_ratio"] = round(post_heartbeats / baseline_heartbeats, 6)
            metrics["heartbeats_saved"] = baseline_heartbeats - post_heartbeats
        else:
            metrics["heartbeat_ratio"] = None
            metrics["heartbeats_saved"] = None

        traj_path = instance_dir / f"{self.instance_id}.traj.json"
        if traj_path.exists():
            try:
                from leanlean.utils.traj_stats import (
                    count_lake_builds, count_git_commits,
                    count_files_modified, count_files_read,
                    classify_edits, count_mcp_calls,
                )
                traj_data = json.loads(traj_path.read_text())
                metrics["lake_builds"] = count_lake_builds(traj_data)
                metrics["agent_commits"] = count_git_commits(traj_data)
                metrics["files_modified"] = count_files_modified(model_patch)
                metrics["files_read"] = count_files_read(
                    traj_data,
                    include_prefix=self._metric_include_prefix,
                )
                # Edit-mechanism split (edit tool vs sed/mass) + measured
                # edit-tool compression; mass share is a run-level residual.
                metrics["edit_breakdown"] = classify_edits(traj_data)
                # lean-lsp MCP usage: total + per-tool distribution.
                metrics["mcp_calls"] = count_mcp_calls(traj_data)
            except Exception as e:
                logger.warning("Failed to extract trajectory stats: %s", e)

        resolved = _resolution_passes(build_passed, sig_diff)
        res = {
            self.instance_id: {
                "resolved": resolved,
                "repo_build_passed": build_passed,
                "empty_patch": empty_patch,
                "empty_patch_build_failed": empty_patch and (not build_passed),
                "apply_patch_result": {
                    "returncode": apply_result.get("returncode"),
                    "errors": (
                        apply_result.get("output", "")
                        if apply_result.get("returncode", 0) != 0
                        else ""
                    ),
                },
                "build_result": build_result,
                "model_patch": model_patch,
                "metrics": metrics,
            }
        }
        with open(instance_dir / "report.json", "w") as f:
            json.dump(res, f, indent=2)
        return resolved

    def solve(
        self,
        patch_diff: str,
        base_dir: Path,
        run_id: int,
        run_repo_test: bool = True,
    ) -> bool:
        """Single-shot eval (one container, no chaining): compile one cumulative
        patch then derive metrics. Kept for callers that evaluate a single patch
        at a time (e.g. evaluate_instance.py and the non-multiround path).
        """
        self.compile_rounds([(0, patch_diff or "", base_dir)], base_dir, run_id, run_repo_test)
        return self.analyze_round(base_dir, run_id)



@dataclass
class LeanLeanConfig:
    filter_spec: str = ""
    slice_spec: str = ""
    shuffle: bool = False
    dataset_name: str = ""
    compile_gate_enabled: bool = False
    compile_gate_command: str = "lake build"
    compile_gate_max_retries: int = 1
    compile_gate_output_chars: int = 12000
    build_jobs: int = -1
    container_cpus: int = 12
    container_memory: str = "100g"
    container_cgroup_parent: str = ""
    container_timeout: str = "2h"
    task_metadata_enabled: bool = True
    enable_lean_verify: bool = True
    enable_proof_length: bool = True
    task_prompt: str = ""
    repository_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    repository_verifiers: dict[str, dict[str, Any]] = field(default_factory=dict)
    repository_entries: list[dict[str, Any]] = field(default_factory=list)
    network_policy: str = "model_proxy_only"
    container_pids_limit: int = 4096
    container_resource_visibility: str = "host"
    # Extra seconds the container's `sleep` outlives the agent's exec timeout.
    container_grace_seconds: int = 1800
    # When True, expand each repo into one instance per .lean file. Each file
    # instance lets the agent edit only that file (diff scoped to it), so a
    # file-level run can be compared against a repo-level run on the same repo.
    file_level: bool = False


class LeanLeanBenchmark(Benchmark):
    """Benchmark for compressing Lean 4 proofs across fixed target repos.

    Published as "LeanLean" in the paper; named LeanLean in code so the
    class is self-documenting.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = LeanLeanConfig(**kwargs)
        self.instances = self.get_instances()
        self.instance_map = {inst.instance_id: inst for inst in self.instances}

    def get_single_instance(self, instance_id: str) -> LeanLeanInstance:
        return self.instance_map[instance_id]

    def get_instances(self) -> list[LeanLeanInstance]:
        raw = [dict(entry) for entry in LEANLEAN_REPOS]
        known_ids = {str(entry["instance_id"]) for entry in raw}
        required_fields = {
            "instance_id",
            "repo_url",
            "commit",
            "exclude_dirs",
            "target_dir",
            "build_target",
            "filesystem_isolated",
        }
        allowed_fields = required_fields | {"task_prompt"}
        for entry in self.config.repository_entries:
            fields = set(entry) if isinstance(entry, dict) else set()
            if not required_fields <= fields or not fields <= allowed_fields:
                raise ValueError(
                    "repository entry must contain exactly the required fields: "
                    + ", ".join(sorted(required_fields))
                    + "; task_prompt is optional"
                )
            instance_id = entry["instance_id"]
            if (
                not isinstance(instance_id, str)
                or not instance_id
                or not isinstance(entry["repo_url"], str)
                or not entry["repo_url"]
                or not isinstance(entry["commit"], str)
                or not isinstance(entry["exclude_dirs"], list)
                or any(
                    not isinstance(path, str) for path in entry["exclude_dirs"]
                )
                or not isinstance(entry["target_dir"], str)
                or not isinstance(entry["build_target"], str)
                or not isinstance(entry["filesystem_isolated"], bool)
                or (
                    "task_prompt" in entry
                    and (
                        not isinstance(entry["task_prompt"], str)
                        or not entry["task_prompt"].strip()
                    )
                )
            ):
                raise ValueError(f"{instance_id}: invalid repository entry")
            if instance_id not in known_ids:
                raw.append(dict(entry))
                known_ids.add(instance_id)

        filtered = filter_instances(
            raw,
            filter_spec=self.config.filter_spec,
            slice_spec=self.config.slice_spec,
            shuffle=self.config.shuffle,
        )
        repo_instances = [self._make_instance(entry) for entry in filtered]
        if not self.config.file_level:
            return repo_instances
        expanded: list[LeanLeanInstance] = []
        for repo in repo_instances:
            # LeanPool subproject instances are already directory-scoped; don't
            # further expand them per-file.
            if repo.target_dir:
                expanded.append(repo)
            else:
                expanded.extend(self._expand_to_file_instances(repo))
        logger.info(
            "file_level: expanded %d repo(s) into %d file instance(s)",
            len(repo_instances), len(expanded),
        )
        return expanded

    def _expand_to_file_instances(
        self, repo: "LeanLeanInstance"
    ) -> list["LeanLeanInstance"]:
        """Turn one repo instance into one instance per .lean source file. Each
        carries `target_file` and shares the repo's image/config. Falls back to
        the repo instance if no files could be listed."""
        files = _list_repo_lean_files(repo.docker_image, repo.exclude_dirs)
        if not files:
            logger.warning(
                "file_level: no .lean files listed for %s (image %s); keeping the "
                "repo-level instance", repo.instance_id, repo.docker_image,
            )
            return [repo]
        instances: list[LeanLeanInstance] = []
        for rel_path in files:
            slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", rel_path)
            instances.append(LeanLeanInstance(
                instance_id=f"{repo.instance_id}__{slug}",
                base_instance_id=repo.instance_id,
                repo_url=repo.repo_url,
                commit=repo.commit,
                exclude_dirs=list(repo.exclude_dirs),
                task_metadata_enabled=repo.task_metadata_enabled,
                enable_lean_verify=repo.enable_lean_verify,
                enable_proof_length=repo.enable_proof_length,
                task_prompt=repo.task_prompt,
                required_signatures=list(repo.required_signatures),
                required_signatures_source=repo.required_signatures_source,
                agent_verifier=repo.agent_verifier,
                filesystem_isolated=repo.filesystem_isolated,
                target_file=rel_path,
                docker_image=repo.docker_image,
                compile_gate_enabled=repo.compile_gate_enabled,
                compile_gate_command=repo.compile_gate_command,
                compile_gate_max_retries=repo.compile_gate_max_retries,
                compile_gate_output_chars=repo.compile_gate_output_chars,
                build_jobs=repo.build_jobs,
                container_cpus=repo.container_cpus,
                container_memory=repo.container_memory,
                container_cgroup_parent=repo.container_cgroup_parent,
                container_timeout=repo.container_timeout,
                network_policy=repo.network_policy,
                container_pids_limit=repo.container_pids_limit,
                container_resource_visibility=repo.container_resource_visibility,
                container_grace_seconds=repo.container_grace_seconds,
            ))
        return instances

    def _make_instance(self, entry: dict[str, Any]) -> LeanLeanInstance:
        required_signatures, required_signatures_source = (
            _required_signatures_from_contract(
                entry, self.config.repository_contracts
            )
        )
        return LeanLeanInstance(
            instance_id=entry["instance_id"],
            repo_url=entry["repo_url"],
            commit=entry.get("commit", ""),
            exclude_dirs=list(entry.get("exclude_dirs", [])),
            # LeanPool subproject scoping (empty/default for standalone repos).
            base_instance_id=entry.get("base_instance_id", ""),
            required_signatures=required_signatures,
            required_signatures_source=required_signatures_source,
            agent_verifier=self.config.repository_verifiers.get(
                str(entry["instance_id"])
            ),
            target_dir=entry.get("target_dir", ""),
            filesystem_isolated=bool(entry.get("filesystem_isolated", False)),
            task_metadata_enabled=self.config.task_metadata_enabled,
            enable_lean_verify=self.config.enable_lean_verify,
            enable_proof_length=self.config.enable_proof_length,
            task_prompt=str(entry.get("task_prompt") or self.config.task_prompt),
            build_target=entry.get("build_target", ""),
            docker_image=entry.get("docker_image", ""),
            compile_gate_enabled=self.config.compile_gate_enabled,
            compile_gate_command=self.config.compile_gate_command,
            compile_gate_max_retries=self.config.compile_gate_max_retries,
            compile_gate_output_chars=self.config.compile_gate_output_chars,
            build_jobs=self.config.build_jobs,
            container_cpus=self.config.container_cpus,
            container_memory=self.config.container_memory,
            container_cgroup_parent=self.config.container_cgroup_parent,
            container_timeout=self.config.container_timeout,
            network_policy=self.config.network_policy,
            container_pids_limit=self.config.container_pids_limit,
            container_resource_visibility=self.config.container_resource_visibility,
            container_grace_seconds=self.config.container_grace_seconds,
        )

    def solve(
        self,
        patch_diffs: dict[str, dict],
        base_dir: Path,
        run_id: int,
        workers: int = 8,
        run_repo_test: bool = True,
        force_eval: bool = False,
    ) -> dict[str, bool]:

        # Set a run-specific cache tag so concurrent runs with different models
        # never collide on the same cache image name.
        import re as _re
        run_tag = _re.sub(r"[^a-zA-Z0-9_.-]", "_", "_".join(base_dir.parts[-3:]))
        for instance in self.instance_map.values():
            instance.run_tag = run_tag

        preds = {}
        for instance_id, patch_dict in patch_diffs.items():
            preds[instance_id] = {
                "model_patch": patch_dict["model_patch"],
                "instance_id": instance_id,
                "model_name_or_path": "model",
            }

        instance_ids = [iid for iid in self.instance_map.keys() if iid in preds]
        logger.info(f"Evaluating {len(instance_ids)} leanlean instances")

        pending_instance_ids: list[str] = []
        skipped_instance_ids: list[str] = []
        for instance_id in instance_ids:
            report_file = base_dir / instance_id / "report.json"
            if report_file.exists() and not force_eval:
                skipped_instance_ids.append(instance_id)
                continue
            pending_instance_ids.append(instance_id)
        if skipped_instance_ids:
            logger.info(
                "Skipping %d instances with existing report.json",
                len(skipped_instance_ids),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(
                    self.instance_map[instance_id].solve,
                    patch_diffs[instance_id]["model_patch"],
                    base_dir,
                    run_id,
                    run_repo_test,
                ): instance_id
                for instance_id in pending_instance_ids
            }
            self._process_futures(future_to_id, desc="Solving")

        results: dict[str, bool] = {}
        for instance_id in instance_ids:
            report_file = base_dir / instance_id / "report.json"
            try:
                with open(report_file, "r") as f:
                    report_data = json.load(f)
                    results[instance_id] = report_data[instance_id]["resolved"]
            except Exception as e:
                logger.error(f"Error reading report for instance {instance_id}: {e}")
                results[instance_id] = False

        return results

    @staticmethod
    def _process_futures(
        future_to_id: dict[concurrent.futures.Future, str],
        desc: str = "Solving",
    ) -> None:
        total = len(future_to_id)
        if total == 0:
            return
        success_count = 0
        failure_count = 0
        with logging_redirect_tqdm():
            with tqdm(
                total=total,
                desc=desc,
                unit="instance",
                leave=True,
                dynamic_ncols=True,
            ) as pbar:
                pbar.set_postfix_str("Pass 0 / Fail 0", refresh=True)
                for future in concurrent.futures.as_completed(future_to_id):
                    resolved = False
                    try:
                        run_succeeded = future.result()
                        resolved = bool(run_succeeded)
                    except concurrent.futures.CancelledError:
                        logger.warning(
                            "Worker for instance %s was cancelled.",
                            future_to_id[future],
                        )
                    except Exception as exc:
                        logger.error(
                            "Unhandled exception in worker for instance %s: %s",
                            future_to_id[future],
                            exc,
                            exc_info=True,
                        )
                    finally:
                        if resolved:
                            success_count += 1
                        else:
                            failure_count += 1
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"Pass {success_count} / Fail {failure_count}",
                            refresh=True,
                        )

    def compile(
        self,
        round_specs: "list[tuple[int, dict, Path]]",
        base_dir: Path,
        run_id: int,
        workers: int = 8,
        run_repo_test: bool = True,
        force: bool = False,
        measure_heartbeats: bool = True,
    ) -> None:
        """Phase 1 dispatcher: chain-build every round per instance.

        `round_specs` is (round_num, preds_dict, round_base_dir) sorted by round.
        Each instance is processed in a single container that walks its rounds
        (see LeanLeanInstance.compile_rounds). `base_dir` is the run dir,
        used only to derive the per-run image tag.
        """
        run_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", "_".join(base_dir.parts[-3:]))
        for instance in self.instance_map.values():
            instance.run_tag = run_tag

        # Build per-instance round lists: [(round_num, cumulative_patch, round_base)].
        jobs: dict[str, list[tuple[int, str, Path]]] = {}
        for round_num, preds, round_base in round_specs:
            for instance_id, pred in preds.items():
                if instance_id not in self.instance_map:
                    continue
                jobs.setdefault(instance_id, []).append(
                    (round_num, pred.get("model_patch", ""), round_base)
                )
        for rounds in jobs.values():
            rounds.sort(key=lambda r: r[0])

        pending: dict[str, list[tuple[int, str, Path]]] = {}
        for instance_id, rounds in jobs.items():
            if not force and all(
                self.instance_map[instance_id]._compile_result_path(rb).exists()
                for _, _, rb in rounds
            ):
                continue
            pending[instance_id] = rounds
        skipped = len(jobs) - len(pending)
        if skipped:
            logger.info("Skipping %d instances already compiled", skipped)
        logger.info("Compiling %d leanlean instances", len(pending))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(
                    self.instance_map[instance_id].compile_rounds,
                    rounds,
                    base_dir,
                    run_id,
                    run_repo_test,
                    measure_heartbeats,
                ): instance_id
                for instance_id, rounds in pending.items()
            }
            self._process_futures(future_to_id, desc="Compiling")

    def analyze(
        self,
        round_specs: "list[tuple[int, dict, Path]]",
        run_id: int,
        workers: int = 8,
    ) -> dict[str, bool]:
        """Phase 2 dispatcher: derive report.json from compile_result.json for
        every (instance, round). No docker, no `lake build` — re-runnable.
        """
        jobs: list[tuple[str, Path]] = []
        for _round_num, preds, round_base in round_specs:
            for instance_id in preds:
                if instance_id not in self.instance_map:
                    continue
                if not self.instance_map[instance_id]._compile_result_path(round_base).exists():
                    logger.warning(
                        "No compile_result.json for %s at %s — run compile first; skipping",
                        instance_id, round_base,
                    )
                    continue
                jobs.append((instance_id, round_base))
        logger.info("Analyzing %d (instance, round) results", len(jobs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {
                executor.submit(
                    self.instance_map[instance_id].analyze_round,
                    round_base,
                    run_id,
                ): instance_id
                for instance_id, round_base in jobs
            }
            self._process_futures(future_to_id, desc="Analyzing")

        results: dict[str, bool] = {}
        for instance_id, round_base in jobs:
            report_file = round_base / instance_id / "report.json"
            try:
                with open(report_file, "r") as f:
                    report_data = json.load(f)
                    results[instance_id] = report_data[instance_id]["resolved"]
            except Exception as e:
                logger.error(f"Error reading report for instance {instance_id}: {e}")
                results[instance_id] = False

        run_dirs = {
            round_base.parent
            if re.fullmatch(r"round_\d+", round_base.name)
            else round_base
            for _round_num, _preds, round_base in round_specs
        }
        for run_dir in sorted(run_dirs):
            try:
                summary_path = write_run_report_summary(run_dir)
                logger.info("Wrote compact run report summary to %s", summary_path)
            except Exception as e:
                logger.error(
                    "Failed to write compact report summary for %s: %s",
                    run_dir,
                    e,
                )
        return results


def filter_instances(
    instances: list[dict],
    *,
    filter_spec: str,
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict]:
    """Filter and slice a list of LeanLean instance dicts."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    if filter_spec:
        instances = [i for i in instances if re.match(filter_spec, i["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        logger.info(f"Instance slice applied: now {len(instances)} instances")
    return instances
