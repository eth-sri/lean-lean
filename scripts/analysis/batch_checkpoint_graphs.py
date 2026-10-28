#!/usr/bin/env python3
"""Run archived checkpoint captures through graph replay with bounded repo workers.

The JSON plan pins capture/result hashes and evaluation manifests. Each worker
keeps one incremental workspace; graph replay never invokes lean_verify or
exports Lake builds. Completed repositories and source states are resumable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import yaml
from scripts.analysis import extract_checkpoint_graphs as replay


def unpack_capture(source, destination, expected):
    """Extract only checksum-pinned source snapshots; never unpack tar paths."""
    destination.mkdir(parents=True, exist_ok=True)
    playback_bytes = (source/'playback.json').read_bytes() if (source/'playback.json').exists() else subprocess.check_output(['zstd', '-dc', '--long=31', str(source/'playback.json.zst')])
    if hashlib.sha256(playback_bytes).hexdigest() != expected:
        raise ValueError('archived playback changed')
    playback = json.loads(playback_bytes)
    points = {Path(p['snapshot_path']).name: p for p in playback['points']}
    wanted = {name: point['snapshot_artifact_sha256'] for name, point in points.items()}
    missing = set()
    restored = {}
    for name, sha in wanted.items():
        path = destination/name
        if path.exists():
            digest = replay.paper.sha256(path)
            if digest == sha:
                restored[name] = digest
                continue
            # Packed archives strip the gzip wrapper. A restored deterministic
            # wrapper may differ, but its canonical source tar must match.
            if name.endswith('.gz') and points[name].get('source_archive_sha256'):
                if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == points[name]['source_archive_sha256']:
                    restored[name] = digest
                    continue
        missing.add(name)
    if missing:
        with subprocess.Popen(['zstd', '-dc', '--long=31', str(source/'checkpoints.tar.zst')], stdout=subprocess.PIPE) as process:
            try:
                with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                    for member in archive:
                        name = member.name if member.name in missing else member.name + '.gz'
                        if name not in missing:
                            continue
                        if not member.isfile():
                            raise ValueError(f'non-file checkpoint: {member.name}')
                        with archive.extractfile(member) as stream:
                            content = stream.read()
                        if name != member.name:
                            expected_tar = points[name].get('source_archive_sha256')
                            if not expected_tar or hashlib.sha256(content).hexdigest() != expected_tar:
                                raise ValueError(f'canonical checkpoint checksum mismatch: {member.name}')
                            content = gzip.compress(content, compresslevel=6, mtime=0)
                        elif hashlib.sha256(content).hexdigest() != wanted[name]:
                            raise ValueError(f'checkpoint checksum mismatch: {member.name}')
                        path = destination/name
                        path.with_suffix('.tmp').write_bytes(content)
                        path.with_suffix('.tmp').replace(path)
                        restored[name] = hashlib.sha256(content).hexdigest()
                        missing.remove(name)
                # tarfile stops at the tar EOF marker, before zstd necessarily
                # finishes writing padding. Drain the pipe before waiting, or
                # a full pipe deadlocks the reader and decompressor.
                while process.stdout.read(1024 * 1024):
                    pass
            except BaseException:
                # Popen.__exit__ waits too: an extraction error must first stop
                # the producer so it cannot remain blocked writing to our pipe.
                process.kill()
                raise
            finally:
                process.stdout.close()
            if process.wait() != 0:
                raise RuntimeError('checkpoint decompression failed')
    if missing:
        raise ValueError(f'{len(missing)} missing checkpoints; first: {sorted(missing)[:3]}')
    for point in playback['points']:
        name = Path(point['snapshot_path']).name
        if restored[name] != point['snapshot_artifact_sha256']:
            point['packed_original_snapshot_artifact_sha256'] = point['snapshot_artifact_sha256']
            point['snapshot_artifact_sha256'] = restored[name]
            point['snapshot_artifact_bytes'] = (destination/name).stat().st_size
    playback['packed_source_playback_sha256'] = expected
    replay.paper.write_json(destination/'playback.json', playback)
    return playback


def run_job(plan, job):
    # Both the original batch and supplemental workers can reach a queued repo.
    # Hold ownership through materialization, extraction, publishing and cleanup.
    output = Path(plan['output'])/'repositories'/job['repository']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path, timeline_path = output/'job.json', output/'timeline.json'
        if state_path.exists() and timeline_path.exists():
            state = json.loads(state_path.read_text())
            timeline = json.loads(timeline_path.read_text())
            if (state.get('status') == 'complete' and timeline.get('status') == 'complete'
                    and all(replay.reusable_result(f) for f in timeline.get('frames', []))):
                return
        minimum = plan.get('minimum_free_gb', 0) * 1024**3
        while shutil.disk_usage(output).free < minimum:
            replay.paper.write_json(state_path, dict(repository=job['repository'],
                status='waiting_for_disk', minimum_free_gb=plan['minimum_free_gb']))
            print(f"{job['repository']}: waiting for {plan['minimum_free_gb']} GiB free disk", flush=True)
            time.sleep(60)
        return _run_job(plan, job)


def _run_job(plan, job):
    output = Path(plan['output'])/'repositories'/job['repository']
    output.mkdir(parents=True, exist_ok=True)
    status = dict(repository=job['repository'], status='running', started=time.time())
    replay.paper.write_json(output/'job.json', status)
    image_created = False
    tag = None
    try:
        complete = output/'timeline.json'
        timeline = json.loads(complete.read_text()) if complete.exists() else {}
        finished = timeline.get('status') == 'complete' and all(
            replay.reusable_result(f)
            for f in timeline.get('frames', []))
        if not finished:
            capture = output/'capture'
            unpack_capture(Path(job['capture']), capture, job['playback_sha256'])
            experiment_path = Path(job['experiment'])
            if replay.paper.sha256(experiment_path) != job['experiment_sha256']:
                raise ValueError('evaluation manifest changed')
            experiment = yaml.safe_load(experiment_path.read_text())
            record = next(r for r in experiment['repositories'] if r['id'] == job['repository'])
            if 'materialization' in record:
                materialization = dict(record['materialization'])
                # Owned by this batch only; completed jobs can retire their image.
                tag = 'leanlean-checkpoint-batch:' + hashlib.sha256(
                    (str(output)+job['experiment_sha256']).encode()).hexdigest()[:24]
                materialization['tag'] = tag
                os.environ['LEANLEAN_MATERIALIZATION_RECEIPTS_DIR'] = str(output/'image-receipts')
                replay.paper.ensure_materialized_image_from_record(job['repository'], materialization)
                image_created = True
                record = dict(record, image=tag, image_id=replay.paper.materialized_image_id(tag))
            resolved = output/'experiment.yaml'
            resolved.write_text(yaml.safe_dump(dict(experiment, repositories=[record])))
            manifest = output/'graph-manifest.yaml'
            manifest.write_text(yaml.safe_dump(dict(model=plan['model'], dataset=plan['dataset'],
                evaluation={'experiment':str(resolved)})))
            subprocess.run([sys.executable, str(ROOT/'scripts/analysis/extract_checkpoint_graphs.py'),
                str(manifest), str(capture), '--output', str(output),
                '--workers', str(job.get('replay_workers', 1)),
                '--cpus', str(job.get('cpus_per_worker', plan['cpus_per_worker'])), '--build-seconds', str(job.get('build_seconds', plan['build_seconds'])),
                '--graph-seconds', str(job.get('graph_seconds', plan['graph_seconds'])),
                *(['--fail-fast-builds'] if job.get('fail_fast_builds') else [])], check=True)
            timeline = json.loads(complete.read_text())
        counts = Counter(f['status'] for f in timeline['frames'])
        timeouts = [dict(point_index=f['point_index'], edit_index=f['edit_index'],
                         source_key=f['source_key'], build_seconds=f.get('build_seconds'))
                    for f in timeline['frames'] if f.get('timed_out')]
        replay.paper.write_json(output/'build-timeouts.json', timeouts)
        retryable = any(f['status'] == 'extraction_failed' or f.get('timed_out') for f in timeline['frames'])
        status.update(status='needs_retry' if retryable else 'complete', counts=dict(counts),
                      build_timeouts=len(timeouts), finished=time.time())
        # Keep image/capture on errors for exact resume. Remove only our own
        # temporary inputs after all states have terminal results.
        if not retryable:
            if (output/'capture').exists():
                shutil.rmtree(output/'capture')
            if image_created:
                subprocess.run(['docker','image','rm',tag], check=True)
    except Exception as error:
        status.update(status='error', error=str(error), finished=time.time())
        raise
    finally:
        replay.paper.write_json(output/'job.json', status)


def publish(plan, active):
    output = Path(plan['output'])
    rows = []
    for job in plan['jobs']:
        directory = output/'repositories'/job['repository']
        state_path = directory/'job.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='queued')
        timeline_path = directory/'timeline.json'
        counts = {}
        timeouts = []
        if timeline_path.exists():
            timeline = json.loads(timeline_path.read_text())
            counts = dict(Counter(f['status'] for f in timeline['frames']))
            timeouts = [dict(point_index=f['point_index'], edit_index=f['edit_index'],
                             source_key=f['source_key'], build_seconds=f.get('build_seconds'))
                        for f in timeline['frames'] if f.get('timed_out')]
        rows.append(dict(state, repository=job['repository'], progress=counts, points=job['points'],
                         build_timeouts=len(timeouts), timed_out_checkpoints=timeouts))
    summary = dict(updated=time.time(), status='running' if active else
                   ('complete' if all(r['status']=='complete' for r in rows) else 'needs_attention'),
                   workers=plan['workers'], repositories=rows)
    replay.paper.write_json(output/'status.json', summary)


def take_queued_repair(plan):
    """Give a pending repair the next existing worker slot, without adding workers."""
    output = Path(plan['output'])
    queue_path = output/'queued-repairs.json'
    with (output/'.repair-queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
        if not queue:
            return None
        repository = queue[0]
        job = next(job for job in plan['jobs'] if job['repository'] == repository)
        replay.paper.write_json(queue_path, queue[1:])
        return job


def run_small_jobs(plan, plan_path, workers, max_states):
    """Add capacity for the shortest queued repositories without restarting builds."""
    output = Path(plan['output'])
    with (output/'.small-repos.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = []
        for i, job in enumerate(plan['jobs']):
            state_path = output/'repositories'/job['repository']/'job.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if job['distinct_states'] <= max_states and state.get('status', 'queued') == 'queued':
                selected.append(i)
        selected.sort(key=lambda i: (plan['jobs'][i]['distinct_states'], i))
        status = dict(status='running', workers=workers, max_states=max_states,
                      indices=selected, started=time.time())
        status_path = output/'small-repos-workers.json'
        replay.paper.write_json(status_path, status)
        def launch(index):
            job = plan['jobs'][index]
            folder = output/'repositories'/job['repository']
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Starting small repo {job['repository']}: {job['distinct_states']} source states", flush=True)
            with (folder/'worker.log').open('a', buffering=1) as log:
                return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()),
                    str(plan_path.resolve()), '--job', str(index)], stdout=log, stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            codes = list(pool.map(launch, selected))
        status.update(status='complete' if not any(codes) else 'needs_attention',
                      exit_codes=dict(zip(map(str, selected), codes)), finished=time.time())
        replay.paper.write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--job', type=int)
    parser.add_argument('--repair-job', type=int)
    parser.add_argument('--small-repos-workers', type=int)
    parser.add_argument('--max-states', type=int, default=100)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    if args.small_repos_workers is not None:
        if args.small_repos_workers < 1 or args.max_states < 1:
            parser.error('workers and max-states must be positive')
        return run_small_jobs(plan, args.plan, args.small_repos_workers, args.max_states)
    if args.repair_job is not None:
        return run_job(plan, plan['jobs'][args.repair_job])
    if args.job is not None:
        repair = take_queued_repair(plan)
        if repair is not None:
            folder = output/'repositories'/repair['repository']
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'worker.log').open('a', buffering=1) as log:
                index = plan['jobs'].index(repair)
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                    str(args.plan.resolve()), '--repair-job', str(index)],
                    stdout=log, stderr=subprocess.STDOUT, check=False)
        return run_job(plan, plan['jobs'][args.job])
    with (output/'.batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Previous process statuses are stale after a coordinator restart.
        for job in plan['jobs']:
            state_path = output/'repositories'/job['repository']/'job.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get('status') != 'complete':
                    state.update(status='queued')
                    state.pop('error', None)
                    replay.paper.write_json(state_path, state)
        def launch(index):
            job = plan['jobs'][index]
            folder = output/'repositories'/job['repository']
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'worker.log').open('a', buffering=1) as log:
                return subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()),
                    str(args.plan.resolve()), '--job', str(index)], stdout=log, stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = [pool.submit(launch, i) for i in range(len(plan['jobs']))]
            while not all(f.done() for f in futures):
                publish(plan, True)
                time.sleep(10)
            codes = [f.result() for f in futures]
        publish(plan, False)
        if any(codes):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
