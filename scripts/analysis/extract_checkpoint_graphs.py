#!/usr/bin/env python3
"""Replay source snapshots in one incremental Lake workspace, saving paper graphs.

Accepts the existing paper-graphs manifest and a capture directory. Results live
in a separate output directory; original captures are never modified. Failed
builds have no graph. Identical source trees share a result. Completed results
are resumable; use a new output directory to retry failures or change timeouts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import logging
import re
import threading
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import yaml
from scripts.analysis import extract_paper_graphs as paper



class CheckpointWorkspace(paper.DockerWorkspace):
    """Use the paper's exact .ilean documents with bounded Docker transport."""

    def iter_ilean_documents(self, modules=None):
        from lean_strip._pipeline.preprocessing.ilean import module_to_ilean_relpath
        requested = sorted(set(modules or []))
        if requested and len(requested) < 512:
            if any(not module or '/' in module for module in requested):
                raise ValueError(f'invalid Lean module name in {requested!r}')
            predicates = ' -o '.join(
                f"-path {shlex.quote('.lake/build/lib/lean/' + module_to_ilean_relpath(module))}"
                for module in requested)
            find = rf'find .lake/build/lib -type f \( {predicates} \) -print0'
        else:
            find = "find .lake/build/lib -type f -name '*.ilean' -print0"
        command = ['docker', 'exec', self._env.container_id, 'bash', '-lc',
                   f'cd /testbed && {find} | LC_ALL=C sort -z | tar --null -T - -cf -']
        for attempt in range(3):
            try:
                result = subprocess.run(command, capture_output=True, check=True, timeout=60)
                documents = []
                with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith('.ilean'):
                            continue
                        document = json.loads(archive.extractfile(member).read())
                        if not isinstance(document, dict):
                            raise ValueError(f'{member.name}: .ilean root is not an object')
                        documents.append((member.name, document))
                yield from documents
                return
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, tarfile.TarError) as error:
                if attempt == 2:
                    raise RuntimeError('.ilean transfer failed after bounded retries') from error
                print(f'.ilean transfer attempt {attempt + 1} failed; retrying', flush=True)


# This assignment is local to replay processes. Static dependency analysis stays
# identical to the paper; only its workspace's archive transport changes.
paper.DockerWorkspace = CheckpointWorkspace


def source_key(point):
    # Include build configuration and all sources, not just Lean files.
    return hashlib.sha256(json.dumps(point['file_sha256'], sort_keys=True).encode()).hexdigest()


def select_points(points, samples, indices):
    if indices is not None:
        wanted = {int(x) for x in indices.split(',')}
        missing = wanted - {p['edit_index'] for p in points}
        if missing:
            raise ValueError(f'unknown edit indices: {sorted(missing)}')
        selected = [i for i, p in enumerate(points) if p['edit_index'] in wanted]
    elif samples is not None:
        if samples < 2:
            raise ValueError('--samples must be at least 2')
        end = max(p['elapsed_seconds'] for p in points)
        selected = sorted({min(range(len(points)), key=lambda i: abs(points[i]['elapsed_seconds'] - end*j/(samples-1))) for j in range(samples)})
        selected = sorted(set(selected + [0, len(points)-1]))
    else:
        selected = list(range(len(points)))
    return selected


def snapshot_sources_bounded(env, paths, timeout=60, attempts=3):
    """Read the same source files as the paper, with bounded Docker transport.

    Some docker exec clients hang after the container tar process has exited.
    communicate(timeout) kills/reaps the client and allows a fresh read retry.
    """
    if not paths:
        return {}
    payload = ('\n'.join(path.removeprefix('/testbed/') for path in paths) + '\n').encode()
    for attempt in range(attempts):
        try:
            result = subprocess.run(['docker', 'exec', '-i', env.container_id,
                'tar', '-C', '/testbed', '-cf', '-', '-T', '-'], input=payload,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
                sources = {f"/testbed/{member.name.removeprefix('./')}": archive.extractfile(member).read().decode('utf-8', 'replace')
                           for member in archive if member.isfile()}
            if set(sources) != set(paths):
                raise RuntimeError('source snapshot does not match requested paths')
            return sources
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, tarfile.TarError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError('source transfer failed after bounded retries') from error
            print(f'source transfer attempt {attempt + 1} failed; retrying', flush=True)


def graph_from_workspace(env, repo, image, image_id, entry, timeout):
    originals = snapshot_sources_bounded(env, paper.list_project_lean_files(env))
    return paper.graph_from_workspace(env, repo, image, image_id, entry, timeout,
                                      originals=originals, incremental=True)


def reusable_result(result):
    """Old successful graphs need an explicit audit or extraction with this pipeline."""
    if not result or result.get('timed_out'):
        return False
    return result['status'] == 'build_failed' or (
        result['status'] == 'complete' and
        result.get('graph_pipeline_revision') == paper.GRAPH_PIPELINE_REVISION)


def build_checkpoint(env, target, seconds, *, fail_fast=False):
    """Zero disables the build deadline; finite timeouts remain retryable states."""
    if seconds < 0:
        raise ValueError('build_seconds must be nonnegative (0 means no deadline)')
    if fail_fast:
        stopped = threading.Event()
        observed = threading.Event()
        errors = []
        def stop_failed_build():
            try:
                # A just-killed process can briefly remain visible in /proc.
                # Retry quiescence before allowing the next source restore.
                for attempt in range(3):
                    try:
                        env.quiesce_after_agent_exit(timeout=20)
                        return
                    except Exception:
                        if attempt == 2:
                            raise
            except Exception as error:
                errors.append(error)
            finally:
                stopped.set()

        def on_output(line):
            # Lake emits this only after a Lean compiler has exited unsuccessfully.
            # Warnings, tactic tracing and progress messages are not failures.
            if not observed.is_set() and re.fullmatch(r'error: Lean exited with code [1-9][0-9]*', line.strip()):
                observed.set()
                # Keep the output reader draining while shutdown waits for
                # compiler descendants; execute_stream closes it after exit.
                threading.Thread(target=stop_failed_build, daemon=True).start()
        result = env.execute_stream('lake build ' + shlex.quote(target),
                                    timeout=seconds or None, on_output=on_output)
        if observed.is_set():
            if not stopped.wait(65):
                raise RuntimeError('Failed build did not finish quiescing')
            if errors:
                raise RuntimeError('Could not stop failed compiler descendants') from errors[0]
            result.update(returncode=1, timed_out=False, early_compile_failure=True)
        return result
    try:
        return env.execute('lake build ' + shlex.quote(target), timeout=seconds or None)
    except subprocess.TimeoutExpired as error:
        output = error.output or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', 'replace')
        return dict(returncode=None, timed_out=True, output=output)


def _run(args):
    manifest = yaml.safe_load(args.manifest.read_text())
    capture = args.capture.resolve()
    playback_path = capture / 'playback.json'
    playback = json.loads(playback_path.read_text())
    points = playback['points']
    repo = playback['instance_id']
    experiment = yaml.safe_load((ROOT / manifest['evaluation']['experiment']).read_text())
    record = next(r for r in experiment['repositories'] if r['id'] == repo)
    database = json.loads((ROOT / manifest['dataset']).with_name('repository-database.json').read_text())
    scope = next(r['scopes'] for r in database['repositories'] if r['instance_id'] == repo)
    selected = getattr(args, 'point_indices', None)
    if selected is None:
        selected = select_points(points, args.samples, args.indices)
    summary = dict(edit_checkpoints=playback['edit_checkpoint_count'], timeline_entries=len(points),
                   distinct_source_states=len({source_key(p) for p in points}),
                   duration_seconds=playback['wall_time_seconds'], selected_entries=len(selected))
    print(json.dumps(summary), flush=True)
    if args.inventory:
        return
    if paper.sha256(paper.EXTRACTOR) != paper.EXTRACTOR_SHA256:
        raise ValueError('paper extractor checksum mismatch')
    paper.olean._DUMP_KEEP_LEAN = paper.EXTRACTOR.read_text()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(playback_sha256=paper.sha256(playback_path), image_id=record['image_id'],
                    extractor_sha256=paper.EXTRACTOR_SHA256, build_target=scope['build_target'],
                    retain_builds=args.save_builds)
    identity_path = output / 'identity.json'
    if identity_path.exists():
        previous_identity = json.loads(identity_path.read_text())
        if any(previous_identity.get(k) != value for k, value in identity.items()):
            raise ValueError('output belongs to a different capture/configuration')
    paper.write_json(identity_path, identity)
    paper.write_json(output/'run-config.json', dict(build_seconds=args.build_seconds,
                     graph_seconds=args.graph_seconds, cpus=args.cpus, selected_points=selected))
    timeline = dict(schema='checkpoint_paper_graph_timeline_v1', status='running', repository=repo, model=manifest['model'],
                    capture=str(capture), summary=summary, timeline_fidelity=playback.get('timeline_fidelity'),
                    validation='Build success only; intermediate states are not benchmark-verified proofs.', frames=[])
    env = None
    previous_files = None
    previous_config = None
    resume_state = None
    try:
        for i in selected:
            point = points[i]
            key = source_key(point)
            destination = output / 'states' / key
            result_path = destination / 'result.json'
            frame = {k: point.get(k) for k in ('edit_index', 'elapsed_seconds', 'kind', 'endpoint_role')}
            frame.update(point_index=i, source_key=key)
            cached = json.loads(result_path.read_text()) if result_path.exists() else None
            if reusable_result(cached):
                result = cached
                if result['status'] == 'complete':
                    graph_path = output / result['graph']
                    if paper.sha256(graph_path) != result['graph_file_sha256']:
                        raise ValueError(f'cached graph changed: {graph_path}')
            else:
                destination.mkdir(parents=True, exist_ok=True)
                archive = capture / Path(point['snapshot_path']).name
                if paper.sha256(archive) != point['snapshot_artifact_sha256']:
                    raise ValueError(f'snapshot checksum mismatch: {archive}')
                resumed_from = None
                if env is None:
                    image_id = paper.materialized_image_id(record['image'])
                    if image_id != record['image_id']:
                        raise ValueError('evaluated image changed')
                    env = paper.get_environment(dict(image=record['image'], cwd='/testbed', timeout=args.build_seconds or 900,
                        container_timeout='172800s', network_policy='none', container_pids_limit=4096,
                        run_args=['--rm', f'--cpus={args.cpus}', f'--env=LEAN_NUM_THREADS={args.cpus}',
                                  '--memory=32g', '--memory-swap=32g'],
                        monitoring_metadata=dict(run_id=output.name, role='checkpoint-paper-graphs',
                            instance_id=repo, model=manifest['model'])), default_type='docker')
                    if resume_state is not None:
                        saved_point, saved_result = resume_state
                        saved_build = output / saved_result['lake_build']
                        if paper.sha256(saved_build) != saved_result['lake_build_sha256']:
                            raise ValueError(f'cached build changed: {saved_build}')
                        cleared = env.execute('rm -rf /testbed/.lake/build', timeout=900)
                        if cleared.get('returncode'):
                            raise RuntimeError('could not clear baseline before restoring saved build')
                        with saved_build.open('rb') as stream:
                            subprocess.run(['docker', 'exec', '-i', env.container_id, 'tar', '--no-same-owner',
                                            '-C', '/testbed', '-xzf', '-'], stdin=stream, capture_output=True,
                                           check=True, timeout=900)
                        previous_files = set(saved_point['file_sha256'])
                        previous_config = {k:v for k,v in saved_point['file_sha256'].items() if k in ('lakefile.lean','lakefile.toml','lake-manifest.json','lean-toolchain')}
                        resumed_from = saved_result['source_key']
                if previous_files is None:
                    previous_files = set(points[0]['file_sha256'])
                    previous_config = {k:v for k,v in points[0]['file_sha256'].items() if k in ('lakefile.lean','lakefile.toml','lake-manifest.json','lean-toolchain')}
                env.restore_source_archive(archive, ['.'], timeout=900)
                # Keep .lake/build between snapshots. Drop it on configuration
                # changes or source deletion so removed modules cannot satisfy imports.
                files = set(point['file_sha256'])
                config = {k:v for k,v in point['file_sha256'].items() if k in ('lakefile.lean','lakefile.toml','lake-manifest.json','lean-toolchain')}
                reset = previous_files is not None and (bool(previous_files - files) or config != previous_config)
                if reset:
                    cleared = env.execute('rm -rf /testbed/.lake/build', timeout=900)
                    if cleared.get('returncode'):
                        raise RuntimeError('could not invalidate removed modules')
                previous_files, previous_config = files, config
                start = time.monotonic()
                print(f"building point {i} / edit {point['edit_index']}", flush=True)
                build = build_checkpoint(env, scope['build_target'], args.build_seconds,
                                         fail_fast=getattr(args, 'fail_fast_builds', False))
                if build.get('timed_out'):
                    print(f'BUILD TIMEOUT: point {i} / edit {point["edit_index"]} reached {args.build_seconds}s; recording and moving to the next checkpoint', flush=True)
                (destination / 'build.log').write_text(str(build.get('output') or ''))
                result = dict(status='build_failed', source_key=key, build_seconds=round(time.monotonic()-start, 3),
                              build_returncode=build.get('returncode'), timed_out=bool(build.get('timed_out')),
                              build_cache_reset=reset, resumed_build_source_key=resumed_from)
                if build.get('early_compile_failure'):
                    result['early_compile_failure'] = True
                if build.get('returncode') == 0 and not build.get('timed_out'):
                    try:
                        start = time.monotonic()
                        graph = graph_from_workspace(env, repo, record['image'], record['image_id'],
                                                     scope['entry_module'], args.graph_seconds)
                        paper.write_graph_artifact(destination / 'graph.json', graph)
                        result.update(status='complete', graph_pipeline_revision=paper.GRAPH_PIPELINE_REVISION, graph=str((destination/'graph.json').relative_to(output)),
                                      graph_file_sha256=paper.sha256(destination/'graph.json'), counts=graph['counts'],
                                      extraction_seconds=round(time.monotonic()-start,3))
                        if args.save_builds:
                            build_path = destination/'lake-build.tar.gz'
                            with build_path.with_suffix('.tmp').open('wb') as stream:
                                subprocess.run(['docker','exec',env.container_id,'tar','-C','/testbed','-czf','-', '.lake/build'],
                                               stdout=stream, check=True, timeout=900)
                            build_path.with_suffix('.tmp').replace(build_path)
                            result['lake_build'] = str(build_path.relative_to(output))
                            result['lake_build_sha256'] = paper.sha256(build_path)
                    except Exception as error:
                        result.update(status='extraction_failed', error=str(error))
                elif result['timed_out']:
                    # Discard the container: a timed-out compiler must not race
                    # with the next source restore.
                    env.cleanup()
                    env = None
                    previous_files = previous_config = None
                paper.write_json(result_path, result)
                print(json.dumps(result), flush=True)
            if result['status'] == 'complete' and result.get('lake_build'):
                resume_state = (point, result)
            frame.update(result)
            timeline['frames'].append(frame)
            paper.write_json(output/'timeline.json', timeline)
    except BaseException as error:
        timeline.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error', error=str(error))
        paper.write_json(output/'timeline.json', timeline)
        raise
    finally:
        if env is not None:
            env.cleanup()
    timeline['status'] = 'complete'
    paper.write_json(output/'timeline.json', timeline)
    print(str(output/'timeline.json'), flush=True)


def run(args):
    if args.inventory:
        return _run(args)
    if args.build_seconds < 0:
        raise ValueError('build_seconds must be nonnegative (0 means no deadline)')
    if getattr(args, 'workers', 1) < 1 or args.cpus < 1:
        raise ValueError('workers and cpus must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.replay.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'another replay is using {args.output}') from error
        if getattr(args, 'workers', 1) > 1:
            from scripts.analysis.parallel_checkpoint_graphs import run_parallel
            return run_parallel(args)
        return _run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--samples', type=int, help='sample approximately evenly in elapsed time')
    selection.add_argument('--indices', help='comma-separated edit indices; omitted selects every checkpoint')
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--save-builds', action='store_true')
    parser.add_argument('--fail-fast-builds', action='store_true',
                        help='Stop on a confirmed Lean compiler failure, retaining the incremental cache')
    parser.add_argument('--build-seconds', type=int, default=7200,
                        help='Per-checkpoint build deadline (default: two hours); 0 disables it')
    parser.add_argument('--graph-seconds', type=int, default=900)
    parser.add_argument('--cpus', type=int, default=4, help='CPU limit per worker')
    parser.add_argument('--workers', type=int, default=1, help='Independent incremental build workspaces')
    logging.basicConfig(level=logging.INFO)
    run(parser.parse_args())
