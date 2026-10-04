"""Restore held-out statements from their trusted comparator's Lean declaration graph."""
import hashlib,json,re,subprocess,tempfile
from pathlib import Path
from typing import Mapping,Any
from leanlean.pipeline.reconstruction import _run,_docker_args,_mapping,_text,_copy_into_container
from leanlean.preprocessing.olean import _DUMP_KEEP_LEAN
from leanlean.preprocessing import strip
from leanlean.preprocessing.cache_isolation import render_container_cache_cleanup

EXISTING = r"""import Lean
open Lean
unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := args[0]!.toName : Import }] {} 0
  let names ← IO.FS.readFile args[1]!
  for name in names.splitOn "\n" do
    if (env.find? name.toName).isSome then IO.println name
"""

def graph_from_dump(output):
    nodes=[];deps={}
    for line in output.splitlines():
        a=line.split('\t')
        if a[0]=='DECL' and len(a)==9:
            nodes.append(dict(name=a[1],module=a[2],start_line=int(a[3]),start_col=int(a[4]),end_line=int(a[5]),end_col=int(a[6]),kind=a[7],meta=a[8]=='meta'))
        elif a[0]=='DEP' and len(a)==3:deps.setdefault(a[1],set()).add(a[2])
    if not nodes:raise ValueError('Trusted Challenge declaration extraction returned no nodes')
    return nodes,deps

def select_missing(nodes,deps,targets,present):
    byname={n['name']:n for n in nodes}
    if not set(targets)<=set(byname):raise ValueError('Trusted Challenge is missing a held-out theorem')
    if set(targets)&present:raise ValueError('Held-out theorem unexpectedly exists before restoration')
    needed=set();todo=list(targets)
    while todo:
        name=todo.pop()
        if name in needed or name in present:continue
        needed.add(name);todo.extend(deps.get(name,()))
    for n in nodes:
        if n['name'] in needed and n['kind']=='theorem' and n['name'] not in targets:
            raise ValueError('Statement scaffold would restore an unselected helper theorem: '+n['name'])
    return needed

def _install_statement_support(container, manifest, tmp):
    support = manifest.get('statement_support')
    if support is None:
        return {'module': None, 'description': None}
    root = Path(__file__).resolve().parents[1]
    if set(support) == {'patch', 'policy'}:
        source = support['patch']
        path = (root / source['path']).resolve()
        if (set(source) != {'path', 'sha256'} or not path.is_relative_to(root) or
                hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']):
            raise ValueError('statement support patch pin drift')
        text = path.read_text()
        heldouts = {h['declaration'] for h in manifest['holdouts']}
        if any(re.search(r'^\+.*\b(?:theorem|lemma)\s+' + re.escape(name.rsplit('.', 1)[-1]) + r'\b', text, re.M)
               for name in heldouts):
            raise ValueError('statement support patch must not contain a held-out theorem')
        staged = tmp / 'statement-support.diff'
        staged.write_text(text)
        _copy_into_container(container, staged, '/tmp/statement-support.diff')
        _run(['docker', 'exec', '-w', '/testbed', container, 'git', 'apply', '--check', '/tmp/statement-support.diff'])
        _run(['docker', 'exec', '-w', '/testbed', container, 'git', 'apply', '--whitespace=nowarn', '/tmp/statement-support.diff'])
        return {'module': None, 'description': 'target-type support restored at original source positions from pinned theorem-free patch'}
    if set(support) != {'module', 'source', 'destination', 'policy'}:
        raise ValueError('statement_support has unexpected fields')
    module = support['module']
    destination = support['destination']
    source = support['source']
    if (not isinstance(module, str) or not module or
            not isinstance(destination, str) or not destination.endswith('.lean') or
            destination.startswith('/') or '..' in Path(destination).parts):
        raise ValueError('invalid statement support module or destination')
    path = (root / source['path']).resolve()
    if (set(source) != {'path', 'sha256'} or not path.is_relative_to(root) or
            hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']):
        raise ValueError('statement support source pin drift')
    heldouts = {h['declaration'] for h in manifest['holdouts']}
    text = path.read_text()
    if any(re.search(r'\b(?:theorem|lemma)\s+' + re.escape(name.rsplit('.', 1)[-1]) + r'\b', text)
           for name in heldouts):
        raise ValueError('statement support must not contain a held-out theorem')
    staged = tmp / 'ChallengeSupport.lean'
    staged.write_text(text)
    _copy_into_container(container, staged, '/testbed/' + destination)
    return {'module': module, 'description': 'target-type support re-elaborated from pinned theorem-free source: '+module}

def install_kernel_scaffold(container,manifest,tmp):
    entry=manifest['entry_module'];entry_file=manifest['entry_file'];resources=manifest['container']
    support = _install_statement_support(container, manifest, tmp)
    support_module = support['module']
    _run(['docker','exec','-w','/testbed','-e',f"LEAN_NUM_THREADS={resources['build_jobs']}",container,'lake','build',manifest['theorem']['build_target']],timeout=int(manifest['timeout']))
    trusted=tmp/'HoldoutTrusted.lean';trusted.write_bytes(Path(manifest['challenge_source']).read_bytes())
    _copy_into_container(container,trusted,'/tmp/HoldoutTrusted.lean')
    _run(['docker','exec','-w','/testbed',container,'lake','env','lean','--root=/tmp','-o','/tmp/HoldoutTrusted.olean','/tmp/HoldoutTrusted.lean'],timeout=int(manifest['timeout']))
    _run(['docker','exec',container,'mkdir','-p','/opt/holdout-trusted'])
    _run(['docker','exec',container,'sh','-c','cp /tmp/HoldoutTrusted.olean* /opt/holdout-trusted/'])
    base_module='HoldoutReproofBase';helper_module='HoldoutRestore'
    suffix=entry.replace('.','/')+'.lean'
    if not entry_file.endswith(suffix):raise ValueError('Cannot locate entry source directory')
    srcdir=entry_file[:-len(suffix)].rstrip('/')
    prefix=(srcdir+'/' if srcdir else '')
    _run(['docker','exec',container,'test','!','-e','/testbed/'+prefix+base_module+'.lean'])
    _run(['docker','exec',container,'cp','/testbed/'+entry_file,'/testbed/'+prefix+base_module+'.lean'])
    helper=Path(__file__).resolve().parents[1]/'scripts/lean/HoldoutRestoreKernel.lean'
    _copy_into_container(container,helper,'/testbed/'+prefix+helper_module+'.lean')
    source=''
    if support_module:
        source += 'import '+support_module+'\n'
    source += 'import '+base_module+'\nimport '+helper_module+'\n\n'
    for h in sorted(manifest['holdouts'],key=lambda x:x['start_line']):
        source+='/- Original statement (the command below uses its exact trusted Lean type):\n'+h['source_command_with_sorry'].replace('/-','/ -').replace('-/','- /')+'-/\n'
        source+='reprove_theorem '+h['declaration']+' := by\n  sorry\n\n'
    restored=tmp/'entry.lean';restored.write_text(source);_copy_into_container(container,restored,'/testbed/'+entry_file)
    lake_toml=subprocess.run(['docker','exec',container,'test','-f','/testbed/lakefile.toml'],capture_output=True).returncode==0
    config_name='lakefile.toml' if lake_toml else 'lakefile.lean';config=tmp/config_name
    _run(['docker','cp',f'{container}:/testbed/{config_name}',str(config)])
    addition=''
    for module in (base_module,helper_module):
        if lake_toml:
            addition+='\n[[lean_lib]]\nname = "'+module+'"\nroots = ["'+module+'"]\n'
            if srcdir:addition+='srcDir = '+json.dumps(srcdir)+'\n'
        else:
            addition+='\nlean_lib '+module+' where\n  roots := #[`'+module+']\n'
            if srcdir:addition+='  srcDir := '+json.dumps(srcdir)+'\n'
    config.write_text(config.read_text()+addition);_copy_into_container(container,config,'/testbed/'+config_name)
    result = ['exact trusted kernel dependencies; restored lazily by HoldoutRestore']
    if support['description']:
        result.append(support['description'])
    return result

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

def materialize_arm(
    *,
    manifest: Mapping[str, Any],
    arm: str,
    instance_id: str,
    base_image: str,
    compression_patch: str,
    original_source: str,
    entry_module: str,
    holdout: Mapping[str, Any],
) -> dict[str, str]:
    safe_run = re.sub(r"[^A-Za-z0-9_.-]", "-", str(manifest["run_id"]))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", instance_id)
    from leanlean.environments.container_identity import container_identity
    container_name, monitoring_labels = container_identity(base_image, {
        "run_id": manifest["run_id"], "role": "reconstruction",
        "model": manifest.get("model"), "instance_id": instance_id,
    })
    image = f"leanlean-{safe_id}-stripped:{safe_run}"
    # Names are only for monitoring; execution and cleanup use Docker's ID.
    container_name = _run(_docker_args(
        manifest, container_name, base_image, labels=monitoring_labels,
    )).stdout.strip()
    try:
        _run(["docker", "start", container_name])
        theorem = _mapping(manifest["theorem"], "theorem")
        target_file = _text(theorem.get("target_file"), "theorem.target_file")
        build_target = _text(theorem.get("build_target"), "theorem.build_target")
        with tempfile.TemporaryDirectory(prefix="leanlean-reconstruct-") as raw_tmp:
            temporary = Path(raw_tmp)
            if arm == "compressed" and compression_patch.strip():
                patch_path = temporary / "compression.diff"
                patch_path.write_text(compression_patch)
                _copy_into_container(container_name, patch_path, "/tmp/compression.diff")
                _run([
                    "docker", "exec", "-w", "/testbed", container_name,
                    "git", "apply", "--whitespace=nowarn", "/tmp/compression.diff",
                ])
            statement_definitions = install_kernel_scaffold(container_name, manifest, temporary)
            # Inputs were independently clean-built already. Lake rebuilds the
            # changed entry and new base module; unchanged verified imports keep
            # their cache. No old held-out project proof exists in this image.
            _run(["docker", "exec", "-w", "/testbed", container_name, "git", "add", "-A"])
            _run([
                "docker", "exec", "-w", "/testbed",
                "-e", "GIT_AUTHOR_NAME=LeanLean",
                "-e", "GIT_AUTHOR_EMAIL=leanlean@localhost",
                "-e", "GIT_COMMITTER_NAME=LeanLean",
                "-e", "GIT_COMMITTER_EMAIL=leanlean@localhost",
                container_name, "git", "commit", "-m", f"prepare {arm} reconstruction arm",
            ])
            commit = _run([
                "docker", "exec", "-w", "/testbed", container_name,
                "git", "rev-parse", "HEAD",
            ]).stdout.strip()
            init_path = temporary / "init_commit"
            init_path.write_text(commit + "\n")
            _copy_into_container(container_name, init_path, "/.init_commit")

        resources = _mapping(manifest["container"], "container")
        fresh_build = bool(manifest.get('fresh_scaffold_build'))
        if fresh_build:
            cleanup = render_container_cache_cleanup(clear_project=True, clear_config=True)
            _run(['docker', 'exec', container_name, 'python3', '-c', cleanup])
        _run([
            "docker", "exec", "-w", "/testbed",
            "-e", f"LEAN_NUM_THREADS={resources['build_jobs']}",
            container_name, "lake", "build", build_target,
        ], timeout=int(manifest["timeout"]))

        with tempfile.TemporaryDirectory(prefix="holdout-signatures-") as temp:
            verify_scaffold_signatures(container_name,manifest,Path(temp))

        # Certify that the prepared arm has exactly the selected unresolved
        # main results, including dependency-connected groups.
        audit_source = r"""import Lean
open Lean
unsafe def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := "__MODULE__".toName : Import }] {} 0
  let ctx : Core.Context := { fileName := "<holdout-audit>", fileMap := default }
  let st : Core.State := { env := env }
  for arg in args do
    let target := arg.toName
    let action : CoreM String := do
      if ((← getEnv).find? target).isNone then return "MISSING " ++ arg
      let axioms ← collectAxioms target
      return (if axioms.contains ``sorryAx then "SORRY " else "PROVED ") ++ arg
    let (line, _) ← action.toIO ctx st
    IO.println line
""".replace("__MODULE__", entry_module)
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.lean"
            audit_path.write_text(audit_source)
            _copy_into_container(container_name, audit_path, "/tmp/holdout-audit.lean")
            result = _run(["docker", "exec", "-w", "/testbed", container_name,
                           "lake", "env", "lean", "--run", "/tmp/holdout-audit.lean", *manifest["protected"]])
            lines = result.stdout.splitlines()
            if any(line.startswith("MISSING ") for line in lines):
                raise ValueError("Prepared arm is missing a protected declaration")
            unresolved = {line[6:] for line in lines if line.startswith("SORRY ")}
            audited = {line.split(" ",1)[1] for line in lines if line.startswith(("SORRY ", "PROVED "))}
            if audited != set(manifest["protected"]) or unresolved != {h["declaration"] for h in manifest["holdouts"]}:
                raise ValueError("Prepared arm does not have exactly the selected unresolved results: " + repr(unresolved))
        labels = [
            f"LABEL org.openai.leanlean.instance_id={instance_id}",
            "LABEL org.openai.leanlean.repo_variant=stripped",
            f"LABEL org.openai.leanlean.run_id={manifest['run_id']}",
            f"LABEL org.openai.leanlean.reconstruction_arm={arm}",
        ]
        command = ["docker", "commit"]
        for label in labels:
            command.extend(["--change", label])
        command.extend([container_name, image])
        _run(command)
        image_id = _run([
            "docker", "image", "inspect", "--format", "{{.Id}}", image
        ]).stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise RuntimeError("materialized arm has no immutable image ID")
        return {"tag": image, "image_id": image_id,
                "restored_statement_definitions": statement_definitions,
                "fresh_scaffold_build": fresh_build}
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
