"""Parallel coordinator for extract_checkpoint_graphs.py --workers N.

One process/container/cache per chronological chunk. Workers publish disjoint
source keys to the shared states directory via atomic result.json writes.
Only the coordinator writes the combined chronological timeline.
"""
from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

import yaml
from scripts.analysis import extract_checkpoint_graphs as replay


def terminal(result):
    return replay.reusable_result(result)


def read_results(output):
    return {p.parent.name: json.loads(p.read_text()) for p in (output/'states').glob('*/result.json')}


def partition(points, selected, results, workers):
    """Disjoint source keys, chronological locality, nearest completed build seed."""
    unique = {}
    for i in selected:
        unique.setdefault(replay.source_key(points[i]), i)
    pending = [i for key,i in unique.items() if not terminal(results.get(key))]
    # Start untouched source states before reattempting earlier two-hour timeouts.
    pending.sort(key=lambda i: (bool(results.get(replay.source_key(points[i]), {}).get('timed_out')), i))
    chunks = []
    count = min(workers, len(pending))
    for worker in range(count):
        chunk = pending[len(pending)*worker//count:len(pending)*(worker+1)//count]
        seeds = [i for i in selected if i < chunk[0]
                 and (result := results.get(replay.source_key(points[i])))
                 and terminal(result) and result['status'] == 'complete' and result.get('lake_build')]
        chunks.append(([max(seeds)] if seeds else []) + chunk)
    return chunks


def combined_frames(points, selected, results):
    frames = []
    for i in selected:
        point = points[i]
        key = replay.source_key(point)
        frame = {k: point.get(k) for k in ('edit_index', 'elapsed_seconds', 'kind', 'endpoint_role')}
        frame.update(point_index=i, source_key=key)
        result = results.get(key, {'status': 'pending'})
        frame.update(result)
        if result['status'] == 'complete' and not terminal(result):
            frame['status'] = 'pending_graph_repair'
        frames.append(frame)
    return frames


def interrupt(*_):
    raise KeyboardInterrupt


def worker_main(args, log):
    # SIGTERM from a service stop takes the normal cleanup path as well.
    signal.signal(signal.SIGTERM, interrupt)
    with Path(log).open('a', buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            replay.run(args)


def run_parallel(args):
    output = args.output.resolve()
    capture = args.capture.resolve()
    playback_path = capture/'playback.json'
    playback = json.loads(playback_path.read_text())
    points = playback['points']
    selected = replay.select_points(points, args.samples, args.indices)
    manifest = yaml.safe_load(args.manifest.read_text())
    experiment = yaml.safe_load((replay.ROOT/manifest['evaluation']['experiment']).read_text())
    record = next(r for r in experiment['repositories'] if r['id'] == playback['instance_id'])
    database = json.loads((replay.ROOT/manifest['dataset']).with_name('repository-database.json').read_text())
    scope = next(r['scopes'] for r in database['repositories'] if r['instance_id'] == playback['instance_id'])
    identity = dict(playback_sha256=replay.paper.sha256(playback_path), image_id=record['image_id'],
                    extractor_sha256=replay.paper.EXTRACTOR_SHA256, build_target=scope['build_target'], retain_builds=args.save_builds)
    identity_path = output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('output belongs to a different capture/configuration')
    if replay.paper.sha256(replay.paper.EXTRACTOR) != replay.paper.EXTRACTOR_SHA256:
        raise ValueError('paper extractor checksum mismatch')
    replay.paper.write_json(identity_path, identity)
    (output/'states').mkdir(exist_ok=True)
    results = read_results(output)
    # Validate reused graphs before any worker can consume their build caches.
    for result in results.values():
        if result['status'] == 'complete' and replay.paper.sha256(output/result['graph']) != result['graph_file_sha256']:
            raise ValueError(f'cached graph changed: {result["graph"]}')
    chunks = partition(points, selected, results, args.workers)
    timeline = dict(schema='checkpoint_paper_graph_timeline_v1', status='running', repository=playback['instance_id'],
        model=manifest['model'], capture=str(capture), workers=len(chunks),
        summary=dict(edit_checkpoints=playback['edit_checkpoint_count'], timeline_entries=len(points),
            distinct_source_states=len({replay.source_key(p) for p in points}),
            duration_seconds=playback['wall_time_seconds'], selected_entries=len(selected)),
        timeline_fidelity=playback.get('timeline_fidelity'),
        validation='Build success only; intermediate states are not benchmark-verified proofs.')
    config = dict(workers=len(chunks), cpus_per_worker=args.cpus, build_seconds=args.build_seconds,
                  graph_seconds=args.graph_seconds, selected_points=selected, chunks=chunks)
    replay.paper.write_json(output/'run-config.json', config)
    context = multiprocessing.get_context('spawn')
    processes, jobs = [], []
    def publish():
        timeline['frames'] = combined_frames(points, selected, read_results(output))
        replay.paper.write_json(output/'timeline.json', timeline)
        replay.paper.write_json(output/'parallel-job.json', dict(status=timeline['status'], workers=jobs,
            exit_codes={str(i): p.exitcode for i,p in enumerate(processes)}))
    publish()
    old_term = signal.signal(signal.SIGTERM, interrupt)
    try:
        for i, chunk in enumerate(chunks):
            folder = output/'workers'/f'{i:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            shared = folder/'states'
            if not shared.exists():
                shared.symlink_to(output/'states', target_is_directory=True)
            if shared.resolve() != output/'states':
                raise ValueError(f'worker state directory is not the shared store: {shared}')
            child = copy.copy(args)
            child.output, child.workers, child.point_indices = folder, 1, chunk
            log = folder/'worker.log'
            process = context.Process(target=worker_main, args=(child, str(log)), name=f'checkpoint-{i}')
            process.start()
            processes.append(process)
            jobs.append(dict(worker=i, pid=process.pid, log=str(log), point_indices=chunk))
            print(f'worker {i}: {len(chunk)} selected points, edits {points[chunk[0]]["edit_index"]}–{points[chunk[-1]]["edit_index"]}, pid {process.pid}', flush=True)
        while any(p.is_alive() for p in processes):
            publish()
            time.sleep(5)
        failures = [p.exitcode for p in processes if p.exitcode != 0]
        if failures:
            raise RuntimeError(f'checkpoint workers failed: {failures}; see workers/*/worker.log')
        timeline['status'] = 'complete'
    except BaseException as error:
        timeline.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error', error=str(error))
        raise
    finally:
        for process in processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
        for process in processes:
            process.join(timeout=65)
            if process.is_alive():
                process.kill()
                process.join()
        publish()
        signal.signal(signal.SIGTERM, old_term)
    print(str(output/'timeline.json'), flush=True)
