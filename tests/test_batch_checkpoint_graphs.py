"""Archived graph replay validates snapshots and skips unrelated tar members."""
import hashlib
import io
import json
import subprocess
import tarfile

import pytest
from scripts.analysis.batch_checkpoint_graphs import unpack_capture


def fixture_capture(tmp_path, compressed=False, bad_digest=False):
    source = tmp_path/'archive'
    source.mkdir()
    snapshot = b'checkpoint bytes'
    playback = json.dumps({'points':[{'snapshot_path':'/old/root/checkpoint.sources.tar',
        'snapshot_artifact_sha256': '0'*64 if bad_digest else hashlib.sha256(snapshot).hexdigest()}]}).encode()
    if compressed:
        (source/'playback.json.zst').write_bytes(subprocess.check_output(['zstd','-c'],input=playback))
    else:
        (source/'playback.json').write_bytes(playback)
    data = io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as tar:
        for name, content in [('checkpoint.sources.tar',snapshot),('../escape',b'bad'),('unused.diff',b'diff')]:
            member=tarfile.TarInfo(name); member.size=len(content)
            tar.addfile(member,io.BytesIO(content))
    (source/'checkpoints.tar.zst').write_bytes(subprocess.check_output(['zstd','-c'],input=data.getvalue()))
    return source, hashlib.sha256(playback).hexdigest(), snapshot


@pytest.mark.parametrize('compressed',[False,True])
def test_archive_unpack_is_confined_and_resumable(tmp_path,compressed):
    source,digest,snapshot=fixture_capture(tmp_path,compressed)
    out=tmp_path/'out'
    unpack_capture(source,out,digest)
    assert (out/'checkpoint.sources.tar').read_bytes()==snapshot
    assert not (tmp_path/'escape').exists()
    assert not (out/'unused.diff').exists()
    (source/'checkpoints.tar.zst').unlink()
    unpack_capture(source,out,digest)  # Valid states don't need decompressing again.


def test_bad_snapshot_is_not_published(tmp_path):
    source,digest,_=fixture_capture(tmp_path,bad_digest=True)
    with pytest.raises(ValueError,match='checkpoint checksum mismatch'):
        unpack_capture(source,tmp_path/'out',digest)
    assert not (tmp_path/'out/checkpoint.sources.tar').exists()


def test_changed_playback_is_rejected(tmp_path):
    source,_,_=fixture_capture(tmp_path)
    with pytest.raises(ValueError,match='archived playback changed'):
        unpack_capture(source,tmp_path/'out','0'*64)


def test_gunzipped_packing_preserves_canonical_digest_and_provenance(tmp_path):
    import gzip
    source,digest,snapshot=fixture_capture(tmp_path)
    point={'snapshot_path':'checkpoint.sources.tar.gz',
           'source_archive_sha256':hashlib.sha256(snapshot).hexdigest(),
           'snapshot_artifact_sha256':'a'*64}
    playback=json.dumps({'points':[point]}).encode()
    (source/'playback.json').write_bytes(playback)
    digest=hashlib.sha256(playback).hexdigest()
    out=tmp_path/'out'
    result=unpack_capture(source,out,digest)
    stored=(out/'checkpoint.sources.tar.gz').read_bytes()
    assert gzip.decompress(stored)==snapshot
    assert result['points'][0]['packed_original_snapshot_artifact_sha256']=='a'*64
    assert result['points'][0]['snapshot_artifact_sha256']==hashlib.sha256(stored).hexdigest()
    assert result['packed_source_playback_sha256']==digest
    (source/'checkpoints.tar.zst').unlink()
    assert unpack_capture(source,out,digest)==result


def test_queued_repair_is_claimed_once_without_changing_scheduled_jobs(tmp_path):
    from scripts.analysis.batch_checkpoint_graphs import take_queued_repair
    jobs = [dict(repository='running'), dict(repository='repair')]
    plan = dict(output=str(tmp_path), jobs=jobs)
    (tmp_path/'queued-repairs.json').write_text(json.dumps(['repair']))
    assert take_queued_repair(plan) == jobs[1]
    assert take_queued_repair(plan) is None
    assert plan['jobs'] == jobs


def test_unknown_queued_repair_is_not_silently_discarded(tmp_path):
    from scripts.analysis.batch_checkpoint_graphs import take_queued_repair
    path = tmp_path/'queued-repairs.json'
    path.write_text(json.dumps(['unknown']))
    with pytest.raises(StopIteration):
        take_queued_repair(dict(output=str(tmp_path), jobs=[]))
    assert json.loads(path.read_text()) == ['unknown']


def test_concurrent_schedulers_run_repository_only_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from scripts.analysis import batch_checkpoint_graphs as batch
    calls = []
    def perform(plan, job):
        calls.append(job['repository'])
        output = tmp_path/'repositories'/job['repository']
        batch.replay.paper.write_json(output/'timeline.json', dict(status='complete',
            frames=[dict(status='build_failed', timed_out=False)]))
        batch.replay.paper.write_json(output/'job.json', dict(status='complete'))
    monkeypatch.setattr(batch, '_run_job', perform)
    plan, job = dict(output=str(tmp_path)), dict(repository='repo')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: batch.run_job(plan, job), range(2)))
    assert calls == ['repo']


def test_small_repo_workers_select_shortest_queued_jobs(tmp_path, monkeypatch):
    from scripts.analysis import batch_checkpoint_graphs as batch
    jobs = [dict(repository='large', distinct_states=101),
            dict(repository='medium', distinct_states=40),
            dict(repository='tiny', distinct_states=7),
            dict(repository='busy', distinct_states=2)]
    batch.replay.paper.write_json(tmp_path/'repositories/busy/job.json', dict(status='running'))
    calls = []
    monkeypatch.setattr(batch.subprocess, 'call', lambda args, **kwargs: calls.append(int(args[-1])) or 0)
    batch.run_small_jobs(dict(output=str(tmp_path), jobs=jobs), tmp_path/'plan.json', 1, 100)
    assert calls == [2, 1]
    assert json.loads((tmp_path/'small-repos-workers.json').read_text())['status'] == 'complete'

@pytest.mark.parametrize('bad_digest', [False, True])
def test_large_tar_padding_cannot_deadlock_decompressor(tmp_path, bad_digest):
    source,digest,_=fixture_capture(tmp_path,bad_digest=bad_digest)
    packed=source/'checkpoints.tar.zst'
    raw=subprocess.check_output(['zstd','-dc',str(packed)])
    # tar readers stop at EOF; the producer still has more than a pipe's
    # capacity to write. Exercise both success and early validation failure.
    packed.write_bytes(subprocess.check_output(['zstd','-c'],input=raw+b'\0'*(1024*1024)))
    import sys
    code='''
import sys
from pathlib import Path
from scripts.analysis.batch_checkpoint_graphs import unpack_capture
try:
    unpack_capture(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
except ValueError as e:
    assert sys.argv[4] == 'True' and 'checkpoint checksum mismatch' in str(e)
else:
    assert sys.argv[4] == 'False'
'''
    subprocess.run([sys.executable,'-c',code,str(source),str(tmp_path/'out'),digest,str(bad_digest)],
                   check=True,timeout=10)


def test_completed_batch_publishes_data_without_dashboard_dependencies(tmp_path, monkeypatch):
    from scripts.analysis import batch_checkpoint_graphs as batch
    output = tmp_path / 'repositories' / 'repo'
    batch.replay.paper.write_json(output / 'timeline.json', dict(
        status='complete', frames=[dict(status='complete',
            graph_pipeline_revision=batch.replay.paper.GRAPH_PIPELINE_REVISION)]))
    def unexpected_process(*args, **kwargs):
        raise AssertionError('Completed graph data must not invoke a renderer or container')
    monkeypatch.setattr(batch.subprocess, 'run', unexpected_process)
    job = dict(repository='repo', points=1)
    plan = dict(output=str(tmp_path), jobs=[job], workers=1)
    batch._run_job(plan, job)
    batch.publish(plan, False)
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'complete'
    assert json.loads((output / 'job.json').read_text())['counts'] == {'complete': 1}
    assert not list(tmp_path.rglob('*.html'))
