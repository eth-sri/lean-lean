"""Checkpoint replay never emits stale graphs after failed or deleted sources."""
from argparse import Namespace
import json

import pytest
import yaml

from scripts.analysis import extract_checkpoint_graphs as replay


@pytest.mark.parametrize('build_seconds', [0, 10])
@pytest.mark.parametrize('build_timeout', [False, True])
def test_replay_retains_cache_skips_failed_builds_and_resumes(tmp_path, monkeypatch, build_seconds, build_timeout):
    capture = tmp_path / 'capture'
    capture.mkdir()
    records = []
    for index, files in enumerate(({'A.lean':'a','Old.lean':'x'}, {'A.lean':'bad','Old.lean':'x'}, {'A.lean':'b'})):
        archive = capture / f'{index}.tar.gz'
        archive.write_bytes(str(index).encode())
        records.append(dict(edit_index=index, elapsed_seconds=index*10, kind='edit',
                            file_sha256=files, snapshot_path=archive.name,
                            snapshot_artifact_sha256=replay.paper.sha256(archive)))
    records.append(dict(records[-1], kind='final', elapsed_seconds=31))
    (capture/'playback.json').write_text(json.dumps(dict(instance_id='repo', points=records,
        edit_checkpoint_count=2, wall_time_seconds=31)))
    manifest=tmp_path/'manifest.yaml'
    manifest.write_text(yaml.safe_dump(dict(model='opus', evaluation={'experiment':str(tmp_path/'experiment.yaml')},
                                          dataset=str(tmp_path/'dataset.yaml'))))
    (tmp_path/'experiment.yaml').write_text(yaml.safe_dump(dict(repositories=[dict(id='repo',image='image',image_id='id')])))
    (tmp_path/'repository-database.json').write_text(json.dumps(dict(repositories=[dict(instance_id='repo',scopes=dict(build_target='+A',entry_module='A'))])))
    calls=[]
    class Env:
        def restore_source_archive(self, archive, paths, **kwargs):
            calls.append(('restore',archive.name))
            self.index=int(archive.name.split('.')[0])
        def execute(self, command, **kwargs):
            calls.append(('command',command))
            if command.startswith('lake build'):
                assert kwargs['timeout'] == (build_seconds or None)
                if build_timeout and self.index == 1:
                    raise replay.subprocess.TimeoutExpired(command, kwargs['timeout'], output='unfinished')
            return dict(returncode=int(command.startswith('lake build') and self.index==1),output='build result')
        def cleanup(self):
            calls.append(('cleanup',))
    env=Env()
    monkeypatch.setattr(replay.paper,'materialized_image_id',lambda _: 'id')
    monkeypatch.setattr(replay.paper,'get_environment',lambda *a,**k: env)
    extracted=[]
    def graph(*a):
        extracted.append(env.index)
        return dict(nodes=[],edges=[],counts=dict(nodes=0,edges=0))
    monkeypatch.setattr(replay,'graph_from_workspace',graph)
    args=Namespace(manifest=manifest,capture=capture,output=tmp_path/'out',indices=None,samples=None,
                   inventory=False,cpus=1,build_seconds=build_seconds,graph_seconds=10,save_builds=False)
    replay.run(args)
    timeline=json.loads((args.output/'timeline.json').read_text())
    assert [f['status'] for f in timeline['frames']]==['complete','build_failed','complete','complete']
    assert extracted==[0,2]  # No graph from stale artifacts after failed build.
    assert [c for c in calls if c[0]=='restore']==[('restore','0.tar.gz'),('restore','1.tar.gz'),('restore','2.tar.gz')]
    assert calls.count(('command','rm -rf /testbed/.lake/build'))==1 # Deleted module invalidates cache.
    assert timeline['frames'][2]['build_cache_reset'] is True
    assert 'graph' not in timeline['frames'][1]
    assert timeline['frames'][1]['timed_out'] is build_timeout
    assert calls.count(('cleanup',)) == (2 if build_timeout else 1)
    calls.clear()
    replay.run(args)
    if build_timeout:
        assert [c for c in calls if c[0]=='restore']==[('restore','1.tar.gz')]
    else:
        assert calls==[] # Finished states, including failure, resume without another build.
    graph_path=args.output/timeline['frames'][0]['graph']
    graph_path.write_text('{}')
    with pytest.raises(ValueError,match='cached graph changed'):
        replay.run(args)


def test_parallel_chunks_have_disjoint_states_and_reuse_nearest_build():
    from scripts.analysis.parallel_checkpoint_graphs import partition, combined_frames
    points = [dict(edit_index=i, elapsed_seconds=i, file_sha256={'A.lean': str(i)}) for i in range(12)]
    points.append(dict(points[3], elapsed_seconds=12))  # Repeated source is only built once.
    key = lambda i: replay.source_key(points[i])
    results = {key(0): dict(status='complete', graph_pipeline_revision=replay.paper.GRAPH_PIPELINE_REVISION, lake_build='states/a/lake-build.tar.gz'),
               key(2): dict(status='build_failed', timed_out=False),
               key(4): dict(status='complete', graph_pipeline_revision=replay.paper.GRAPH_PIPELINE_REVISION, lake_build='states/b/lake-build.tar.gz'),
               key(7): dict(status='build_failed', timed_out=True)}
    chunks = partition(points, list(range(len(points))), results, 3)
    work = [i for chunk in chunks for i in chunk if i not in (0,4)]
    assert sorted(work) == [1,3,5,6,7,8,9,10,11]
    assert len({key(i) for i in work}) == len(work)
    assert chunks[0][0] == 0
    assert all(chunk[0] == 4 for chunk in chunks[1:])
    frames = combined_frames(points, list(range(len(points))), results)
    assert [f['point_index'] for f in frames] == list(range(len(points)))
    assert frames[7]['timed_out'] is True
    assert frames[12]['source_key'] == frames[3]['source_key']
    assert frames[1]['status'] == 'pending'


def test_parallel_resume_only_assigns_unfinished_states():
    from scripts.analysis.parallel_checkpoint_graphs import partition
    points = [dict(file_sha256={'A.lean': str(i)}) for i in range(3)]
    results = {replay.source_key(p): dict(status='complete', graph_pipeline_revision=replay.paper.GRAPH_PIPELINE_REVISION) for p in points}
    assert partition(points, [0,1,2], results, 8) == []
    results[replay.source_key(points[1])] = dict(status='extraction_failed')
    assert partition(points, [0,1,2], results, 8) == [[1]]


def test_source_transfer_retries_timeout_and_preserves_exact_source(tmp_path, monkeypatch):
    import io
    import subprocess
    import tarfile
    body = 'theorem example : True := by trivial\n'.encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        entry = tarfile.TarInfo('./A.lean')
        entry.size = len(body)
        archive.addfile(entry, io.BytesIO(body))
    calls = []
    def transfer(command, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 0, stdout=output.getvalue())
    monkeypatch.setattr(replay.subprocess, 'run', transfer)
    env = Namespace(container_id='test')
    assert replay.snapshot_sources_bounded(env, ['/testbed/A.lean']) == {'/testbed/A.lean': body.decode()}
    assert len(calls) == 2 and all(c['timeout'] == 60 for c in calls)
    assert calls[-1]['input'] == b'A.lean\n'
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(replay.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='bounded retries'):
        replay.snapshot_sources_bounded(env, ['/testbed/A.lean'], attempts=2)


def test_ilean_transfer_retries_without_partial_documents(monkeypatch):
    import io
    import subprocess
    import tarfile
    output = io.BytesIO()
    document = {'version': 5, 'references': {}}
    with tarfile.open(fileobj=output, mode='w') as archive:
        body = json.dumps(document).encode()
        entry = tarfile.TarInfo('.lake/build/lib/lean/A.ilean')
        entry.size = len(body)
        archive.addfile(entry, io.BytesIO(body))
    calls = []
    def transfer(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 0, stdout=output.getvalue())
    monkeypatch.setattr(replay.subprocess, 'run', transfer)
    workspace = replay.CheckpointWorkspace(Namespace(container_id='test'))
    assert list(workspace.iter_ilean_documents(['A'])) == [('.lake/build/lib/lean/A.ilean', document)]
    assert len(calls) == 2 and all(kwargs['timeout'] == 60 for _, kwargs in calls)
    assert '.lake/build/lib/lean/A.ilean' in calls[-1][0][-1]
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(replay.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='bounded retries'):
        list(workspace.iter_ilean_documents(['A']))


def test_finite_build_timeout_is_a_retryable_result():
    import subprocess
    class Env:
        def execute(self, command, timeout):
            assert timeout == 10
            raise subprocess.TimeoutExpired(command, timeout, output=b'partial build log')
    result = replay.build_checkpoint(Env(), '+Solution', 10)
    assert result == dict(returncode=None, timed_out=True, output='partial build log')
    with pytest.raises(ValueError, match='nonnegative'):
        replay.build_checkpoint(Env(), '+Solution', -1)


def test_parallel_repair_rebuilds_legacy_graphs_in_disjoint_chunks():
    from scripts.analysis.parallel_checkpoint_graphs import partition, combined_frames
    points = [dict(edit_index=i, file_sha256={'A.lean': str(i)}) for i in range(8)]
    results = {replay.source_key(p): dict(status='complete') for p in points}
    results[replay.source_key(points[0])]['graph_pipeline_revision'] = replay.paper.GRAPH_PIPELINE_REVISION
    chunks = partition(points, list(range(8)), results, 4)
    assert len(chunks) == 4
    assert sorted(i for chunk in chunks for i in chunk) == list(range(1, 8))
    frames = combined_frames(points, list(range(8)), results)
    assert frames[0]['status'] == 'complete'
    assert all(f['status'] == 'pending_graph_repair' for f in frames[1:])


def test_parallel_chunks_start_new_states_before_old_timeouts():
    from scripts.analysis.parallel_checkpoint_graphs import partition
    points = [dict(edit_index=i, file_sha256={'A.lean':str(i)}) for i in range(12)]
    results = {replay.source_key(p):dict(status='build_failed',timed_out=True) for p in points[:4]}
    chunks = partition(points,list(range(12)),results,4)
    assert sorted(i for chunk in chunks for i in chunk) == list(range(12))
    ordered = [i for chunk in chunks for i in chunk]
    assert ordered[:8] == list(range(4,12))
    assert ordered[8:] == list(range(4))


def test_snapshot_sources_preserves_hidden_directory_names(monkeypatch):
    import io
    import tarfile
    from types import SimpleNamespace
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name in ('.golf/Golf.lean', './Solution.lean'):
            content = b'example : True := by trivial\n'
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    monkeypatch.setattr(replay.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=stream.getvalue()))
    paths = ['/testbed/.golf/Golf.lean', '/testbed/Solution.lean']
    assert set(replay.snapshot_sources_bounded(SimpleNamespace(container_id='test'), paths)) == set(paths)


def test_fail_fast_waits_for_quiescence_and_retains_failed_build_evidence():
    calls=[]
    class Env:
        def execute_stream(self,command,timeout,on_output):
            assert command=='lake build +Solution'
            on_output('warning: slow tactic\n')
            on_output('error: A.lean:1:1: unsolved goals\n')
            assert calls==[]
            on_output('error: Lean exited with code 1\n')
            return dict(returncode=143,timed_out=False,output='unsolved goals\nerror: Lean exited with code 1\n')
        def quiesce_after_agent_exit(self,timeout):
            calls.append('quiesced')
    result=replay.build_checkpoint(Env(),'+Solution',7200,fail_fast=True)
    assert calls==['quiesced']
    assert result['returncode']==1 and result['early_compile_failure']
    assert not result['timed_out'] and 'unsolved goals' in result['output']


def test_fail_fast_does_not_publish_failure_before_safe_cleanup():
    class Env:
        def execute_stream(self,command,timeout,on_output):
            on_output('error: Lean exited with code 1\n')
            return dict(returncode=143,timed_out=False,output='error')
        def quiesce_after_agent_exit(self,timeout):
            raise RuntimeError('still running')
    with pytest.raises(RuntimeError,match='compiler descendants'):
        replay.build_checkpoint(Env(),'+Solution',7200,fail_fast=True)


def test_fail_fast_drains_output_while_quiescing_and_retries_exit_race():
    import threading
    drained = threading.Event()
    calls = []
    class Env:
        def execute_stream(self, command, timeout, on_output):
            on_output('error: Lean exited with code 1\n')
            # Cleanup must not block this callback: Docker closes its output
            # reader shortly after the main Lake process exits.
            drained.set()
            return dict(returncode=143, timed_out=False, output='compiler failure')
        def quiesce_after_agent_exit(self, timeout):
            assert drained.wait(2), 'output callback blocked on cleanup'
            calls.append('attempt')
            if len(calls) == 1:
                raise RuntimeError('killed process still visible')
    result = replay.build_checkpoint(Env(), '+Solution', 7200, fail_fast=True)
    assert calls == ['attempt', 'attempt']
    assert result['early_compile_failure'] and result['returncode'] == 1
