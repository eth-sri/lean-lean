"""Route reproof dependencies to unchanged package sources in the pinned image."""
import copy,json,re,toml


def normalize_config(text,filename,lock):
    packages={p['name']:p for p in lock['packages']}
    paths={name:'/testbed/.lake/packages/'+name for name in packages}
    if filename=='lakefile.toml':
        config=toml.loads(text)
        for req in config.get('require',[]):
            name=req['name']
            if name not in paths:raise ValueError('unavailable pinned package: '+name)
            for key in ('git','rev','version','subDir'):req.pop(key,None)
            req['path']=paths[name]
        return toml.dumps(config)
    def replace(match):
        name=match.group('name').strip('«»')
        if name not in paths:raise ValueError('unavailable pinned package: '+name)
        return 'require '+match.group('name')+' from '+json.dumps(paths[name])
    result=re.sub(r'require\s+(?P<name>[\w«»]+)\s+from\s+git\s+"[^"\n]+"(?:\s*@\s*"[^"\n]+")?',replace,text)
    # Optional dependencies guarded by a disabled `meta if ... then` block are not
    # resolved by Lake. Keep their declaration intact while still rejecting any
    # unmatched active git dependency.
    active_check=re.sub(r'meta\s+if[^\n]*\bthen\s*\n\s*require\s+[^\n]+\bfrom\s+git\s+"[^"\n]+"(?:\s*@\s*"[^"\n]+")?', '', result)
    if re.search(r'require\s+[^\n]+\bfrom\s+git\b',active_check):raise ValueError('unsupported computed dependency declaration')
    return result


def normalize_lock(lock):
    result=copy.deepcopy(lock)
    result['packages']=[{**{k:p[k] for k in ('name','scope','inherited','configFile','manifestFile') if k in p},
                         'type':'path','dir':'/testbed/.lake/packages/'+p['name']} for p in lock['packages']]
    return result


def install(env):
    lock=json.loads(env.read_file('/testbed/lake-manifest.json'))
    filename='lakefile.toml' if env.execute('test -f /testbed/lakefile.toml',timeout=30)['returncode']==0 else 'lakefile.lean'
    original=env.read_file('/testbed/'+filename)
    changed=normalize_config(original,filename,lock)
    env.write_file_bytes('/testbed/'+filename,changed.encode())
    env.write_file_bytes('/testbed/lake-manifest.json',(json.dumps(normalize_lock(lock),indent=2)+'\n').encode())
    return {'policy':'root Lake requirements and lock routed to unchanged pinned local package sources',
            'original_lock':lock,'config_file':filename,'original_config':original,'normalized_config':changed,'package_source_changes':False}
