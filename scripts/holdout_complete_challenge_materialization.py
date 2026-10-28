"""Restore held-out statements from their trusted comparator's Lean declaration graph."""
import json
from pathlib import Path
from leanlean.pipeline.reconstruction import _run, _copy_into_container

EXISTING = r"""import Lean
open Lean
unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := args[0]!.toName : Import }] {} 0
  let names ← IO.FS.readFile args[1]!
  for name in names.splitOn "\n" do
    if (env.find? name.toName).isSome then IO.println name
"""



def verify_scaffold_signatures(container,manifest,tmp):
    """Allow sorry only in a temporary signature preflight, never final verification."""
    from leanlean.palomar_comparator import load_palomar_evidence,resolve_tool_path,TOOL_DESTINATIONS
    root=Path(__file__).resolve().parents[1];contract=manifest['preflight_contract']
    challenge,_,runtime=load_palomar_evidence(repo_root=root,contract=contract)
    config=json.loads(runtime);config['permitted_axioms']=list(config['permitted_axioms'])+['sorryAx']
    challenge_path='/testbed/'+contract['challenge']['source_path']
    _run(['docker','exec',container,'test','!','-e',challenge_path])
    _run(['docker','exec',container,'mkdir','-p',str(Path(challenge_path).parent)])
    host=tmp/'challenge.lean';host.write_bytes(challenge);_copy_into_container(container,host,challenge_path)
    configpath=tmp/'scaffold-comparator.json';configpath.write_text(json.dumps(config)+'\n');_copy_into_container(container,configpath,'/tmp/scaffold-comparator.json')
    try:
        for label,dest in TOOL_DESTINATIONS.items():
            source=resolve_tool_path(repo_root=root,tool=contract['tools'][label],label=label)
            _copy_into_container(container,source,dest);_run(['docker','exec',container,'chmod','0755',dest])
        if manifest.get('direct_comparator_env'):
            shell = (
                "pkg_path=$(find /testbed/.lake/packages -type d -path '*/.lake/build/lib/lean' "
                "-print | sort | paste -sd: -); "
                "export LEAN_PATH=${pkg_path:+$pkg_path:}/testbed/.lake/build/lib/lean; "
                "ulimit -s 32768 && /usr/local/bin/palomar-comparator /tmp/scaffold-comparator.json"
            )
        else:
            shell = 'ulimit -s 32768 && lake env /usr/local/bin/palomar-comparator /tmp/scaffold-comparator.json'
        command=['docker','exec','-w','/testbed','-e',f"LEAN_NUM_THREADS={manifest['container']['build_jobs']}",'-e','RUST_MIN_STACK=67108864','-e','PALOMAR_LANDRUN_BIN=/usr/local/bin/palomar-landrun','-e','COMPARATOR_LANDRUN=/usr/local/bin/palomar-landrun-wrapper','-e','COMPARATOR_LEAN4EXPORT=/usr/local/bin/palomar-lean4export','-e','COMPARATOR_NANODA=/usr/local/bin/palomar-nanoda',container,'bash','-c',shell]
        completed = _run(command,timeout=int(manifest['timeout']))
        if manifest.get('preflight_log'):
            Path(manifest['preflight_log']).write_text(completed.stdout + completed.stderr)
    finally:
        _run(['docker','exec',container,'rm','-f',challenge_path,'/tmp/scaffold-comparator.json'])
