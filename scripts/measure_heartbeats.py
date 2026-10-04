#!/usr/bin/env python3
"""Compare pinned stripped sources with saved compressed endpoints, from YAML."""
from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts import heartbeat_memory_guard
from leanlean.heartbeat_cli import METHOD_VERSION
from leanlean.environments.docker import DockerEnvironment
from leanlean.pipeline import postprocessing as pp

COUNTER = ROOT / 'src/leanlean/heartbeat_cli.py'
REMOTE_COUNTER = '/tmp/leanlean-heartbeat-counter.py'
KIND = 'leanlean_heartbeat_comparison'
HEARTBEAT_METRICS = ('body_heartbeats', 'total_heartbeats', 'import_heartbeats')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path: Path) -> dict:
    return {'path': str(path.resolve().relative_to(ROOT)), 'sha256': sha(path)}


def checked_file(record: dict) -> Path:
    path = ROOT / record['path']
    if sha(path) != record['sha256']:
        raise ValueError(f'pinned input drift: {path}')
    return path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def prepare(source_manifest: Path, run_id: str) -> Path:
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('run_id must contain only letters, digits, underscores and hyphens')
    source = yaml.safe_load(source_manifest.read_text())
    if source.get('kind') != 'leanlean_postprocessing_run':
        raise ValueError('expected a postprocessing manifest')
    if source.get('repository_variant') != 'stripped':
        raise ValueError('comparison requires a stripped source run')
    directory = ROOT / 'experiments/heartbeats' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.yaml'
    if manifest_path.exists():
        raise ValueError(f'manifest already exists: {manifest_path}')
    repositories = []
    for original in source['repositories']:
        repository = copy.deepcopy(original)
        instance_id = repository['id']
        if not repository.get('palomar_comparator'):
            raise ValueError(f'{instance_id}: this initial runner requires a Palomar contract')
        if repository.get('metric_include_prefix') or repository.get('exclude_dirs'):
            raise ValueError('this initial runner requires standalone, whole-repository scope')
        selection = source['run_combination']['selection'][instance_id]
        output = Path(selection['output_directory']) / instance_id
        candidates = sorted(output.glob('playback/capture_*/postprocessed-playback.json'))
        # Select the capture explicitly pinned by the postprocessing manifest.
        captures = [c for c in source['captures'] if c['instance_id'] == instance_id]
        if len(captures) != 1:
            raise ValueError(f'{instance_id}: expected exactly one selected capture')
        capture_dir = Path(captures[0]['output_directory']) / Path(captures[0]['playback']).parent
        playback_path = capture_dir / 'postprocessed-playback.json'
        if playback_path not in candidates:
            raise ValueError(f'{instance_id}: selected postprocessed capture is missing')
        playback = json.loads(playback_path.read_text())
        # Pin the actual submitted final, never a best intermediate checkpoint.
        final = copy.deepcopy(playback['points'][-1])
        if final.get('kind') != 'final':
            raise ValueError(f'{instance_id}: submitted endpoint is missing')
        archive = pp._archive_for_point(capture_dir, final)
        evidence_path = directory / f'{instance_id}.endpoint.json'
        evidence = {
            'playback': pin(playback_path),
            'point': final,
            'source_archive': pin(archive),
            'source_archive_canonical_sha256': final['source_archive_sha256'],
        }
        atomic_json(evidence_path, evidence)
        repository['endpoint'] = pin(evidence_path)
        repository['expected_toolchain'] = repository['palomar_comparator']['lean_toolchain']
        repository['exclude_files'] = [repository['palomar_comparator']['challenge']['source_path']]
        repository['materialization']['tag'] = f'leanlean-heartbeats:{run_id}-{instance_id}'
        repository['materialization']['build_jobs'] = 8
        repository['materialization']['build_timeout_seconds'] = 3600
        repository['materialization']['persist'] = False
        repositories.append(repository)
    manifest = {
        'kind': KIND, 'schema_version': 1, 'run_id': run_id,
        'model': source['model'], 'reasoning_effort': source['reasoning_effort'],
        'generator': source['generator'], 'generation_enabled': False,
        'rounds': source['rounds'], 'repository_variant': ['stripped', 'optimized'],
        'optimized_definition': 'exact saved submitted final with pinned postprocessing validation reused when current',
        'dataset': pin(ROOT / source['source_run']['dataset']),
        'source_manifest': pin(source_manifest),
        'repositories': repositories,
        # Run small repositories first to surface measurement failures promptly.
        'execution_order': sorted(
            [r['id'] for r in repositories],
            key=lambda rid: next(
                json.loads(checked_file(r['endpoint']).read_text())['point'].get('lean_tokens', 0) or 0
                for r in repositories if r['id'] == rid
            ),
        ),
        'parallelism': {'workers': 2},
        'container': {
            'cpus': 8, 'build_jobs': 8, 'memory': '64g', 'max_total_memory': '128g',
            'pids_limit': 4096, 'cgroup_parent': 'lean.slice',
            'network_policy': 'model_proxy_only', 'network_mode': 'none',
            'model_proxy_attached': False,
            'timeout_seconds': 86400, 'build_timeout_seconds': 3600,
            'build_output_chars': 12000,
        },
        'timeout': {'file_seconds': 3600, 'repository_seconds': 86400},
        'evaluation': {
            'method': METHOD_VERSION, 'unit': 'raw_heartbeats',
            'user_heartbeat_divisor': 1000, 'lean_threads': 1, 'elab_async': False,
            'repetitions': 1,
            'reuse_pinned_postprocessing_validation': True,
            'aggregation': 'mean_of_available_measurements',
            'repeatability': 'report_spread_without_rejecting',
            'variation_warning_relative_range_pct': 5.0,
            'primary_heartbeat_count': 'body_heartbeats',
            'also_record': ['total_heartbeats', 'import_heartbeats', 'clean_build_seconds', 'lean_tokens'],
            'scope': 'all project modules built by the declared build target; exclude registered Challenge',
            'clean_build': 'delete only /testbed/.lake/build before each variant; retain dependency caches',
            'correctness': 'reuse exact pinned postprocessing verdict; run Palomar comparator only when evidence is absent or stale',
            'failures': 'null counts and ratios; keep repository in coverage denominator',
            'final_metric': 'separate token and heartbeat reductions; no new weighted headline score',
        },
        'implementation': [pin(Path(__file__).resolve()), pin(COUNTER), pin(Path(pp.__file__))],
        'outputs': {
            'directory': f'output/heartbeats/{run_id}',
            'tmux_session': f'heartbeats-{run_id}',
        },
    }
    manifest['container']['cgroup_parent'] = 'leanheartbeats.slice'
    manifest['memory_guard'] = {
        'slice': 'leanheartbeats.slice', 'memory_max_bytes': 200000000000,
        'memory_high_bytes': 180000000000, 'memory_swap_max_bytes': 0,
        'coordinator_memory_bytes': 8589934592, 'pids_limit': 12288,
        'require_slice_membership': True, 'build_preflight_seconds': 10,
    }
    manifest['preparation_mode'] = 'restore_in_capped_container'
    from leanlean import evaluation_images
    manifest['implementation'] += [pin(Path(heartbeat_memory_guard.__file__)), pin(Path(evaluation_images.__file__))]
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


def execute(env, command: str, timeout: int, log: Path) -> dict:
    started = time.monotonic()
    result = env.execute(command, timeout=timeout)
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    with log.open('a') as handle:
        handle.write(command + '\n' + str(result.get('output', '')) + '\n')
    if result.get('returncode') != 0:
        raise RuntimeError(f'command failed ({result.get("returncode")}): {command}\n{str(result.get("output", ""))[-3000:]}')
    return result


def make_environment(repository: dict, container: dict) -> DockerEnvironment:
    return DockerEnvironment(
        image=repository['image_id'], cwd='/testbed', env={}, forward_env=[],
        timeout=container['build_timeout_seconds'], network_mode='none',
        network_policy=container['network_policy'], container_pids_limit=container['pids_limit'],
        container_timeout=f'{container["timeout_seconds"]}s', container_grace_seconds=60,
        run_args=['--rm', f'--cpus={container["cpus"]}', f'--memory={container["memory"]}',
                  f'--memory-swap={container["memory"]}', '--ulimit=stack=67108864:67108864',
                  f'--cgroup-parent={container["cgroup_parent"]}'],
    )


def compare_results(baseline: dict, optimized: dict) -> dict:
    valid = baseline.get('status') == 'complete' and optimized.get('status') == 'complete'
    result = {'eligible': valid, 'heartbeat_ratio': None, 'heartbeats_saved': None,
              'heartbeat_reduction_pct': None, 'lean_token_reduction_pct': None,
              'clean_build_ratio': None, 'both_improved': None}
    if not valid:
        return result
    before, after = baseline['body_heartbeats'], optimized['body_heartbeats']
    if before > 0:
        result.update(heartbeat_ratio=after / before, heartbeats_saved=before - after,
                      heartbeat_reduction_pct=100 * (1 - after / before))
    tokens = baseline['lean_tokens']
    if tokens > 0:
        result['lean_token_reduction_pct'] = 100 * (1 - optimized['lean_tokens'] / tokens)
    if baseline['clean_build_seconds'] > 0:
        result['clean_build_ratio'] = optimized['clean_build_seconds'] / baseline['clean_build_seconds']
    if before > 0 and tokens > 0:
        result['both_improved'] = after < before and optimized['lean_tokens'] < tokens
    return result


@contextlib.contextmanager
def prepared_repository(repository):
    """Use the pinned dependency image; all Lean builds run in capped containers.

    No Dockerfile RUN or warm-cache image build is needed: each measurement
    already performs a clean project build from the exact source.
    """
    from leanlean.evaluation_images import _source_tree
    from leanlean.preprocessing.palomar_sources import localize_lake_manifest
    record = repository['materialization']
    source = _source_tree(record, repository['id'])
    shared = record['shared_environment']
    actual = subprocess.check_output(
        ['docker', 'image', 'inspect', '--format', '{{.Id}}', shared['image']], text=True,
    ).strip()
    if actual != shared['image_id']:
        raise ValueError('pinned dependency image unavailable or changed')
    with tempfile.TemporaryDirectory(prefix='heartbeat-source-') as temporary:
        directory = Path(temporary)
        subprocess.run(['cp', '-a', '--reflink=auto', str(source), str(directory / 'source')], check=True)
        localize_lake_manifest(directory / 'source')
        archive = directory / 'source.tar.gz'
        subprocess.run(['tar', '-czf', str(archive), '-C', str(directory / 'source'), '.'], check=True)
        yield dict(repository, image_id=actual, prepared_source_archive=archive)


def aggregate_file_observations(record: dict, warning_pct: float = 5) -> dict:
    """Validate and aggregate every saved observation for one source file."""
    measurements = record.get('measurements')
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f'{record.get("source", "unknown")}: no measurements')
    fingerprints = None
    protocol = None
    protocol_keys = ('method', 'unit', 'user_heartbeat_divisor', 'lean_threads', 'elab_async')
    values = {key: [] for key in HEARTBEAT_METRICS}
    for measurement in measurements:
        if not measurement.get('ok') or measurement.get('method') != METHOD_VERSION:
            raise ValueError(f'{record.get("source", "unknown")}: invalid measurement response')
        current_protocol = tuple(measurement.get(key) for key in protocol_keys)
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise ValueError(f'{record.get("source", "unknown")}: measurement protocols differ')
        current = (measurement.get('source_sha256'), measurement.get('setup_sha256'))
        if not all(isinstance(value, str) and value for value in current):
            raise ValueError(f'{record.get("source", "unknown")}: missing input fingerprint')
        if fingerprints is None:
            fingerprints = current
        elif current != fingerprints:
            raise ValueError(f'{record.get("source", "unknown")}: measurement input fingerprints differ')
        for key in HEARTBEAT_METRICS:
            value = measurement.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{record.get("source", "unknown")}: invalid {key}')
            values[key].append(value)
        if measurement['body_heartbeats'] + measurement['import_heartbeats'] != measurement['total_heartbeats']:
            raise ValueError(f'{record.get("source", "unknown")}: inconsistent heartbeat counts')
    means = {key: statistics.mean(samples) for key, samples in values.items()}
    spreads = {}
    warned = []
    for key, samples in values.items():
        relative = None if len(samples) < 2 or means[key] == 0 else 100 * (max(samples) - min(samples)) / means[key]
        spreads[key] = {'min': min(samples), 'max': max(samples), 'range': max(samples) - min(samples),
                        'relative_range_pct': relative}
        if relative is not None and relative > warning_pct:
            warned.append(key)
    return {
        **record, 'status': 'complete', 'measurement_count': len(measurements),
        'aggregation': 'mean_of_available_measurements', 'means': means, 'spreads': spreads,
        'variability_assessed': len(measurements) >= 2,
        'variation_warning': bool(warned), 'variation_warning_metrics': warned,
        **means,
    }


def resume_observation_cache(previous: dict | None) -> list[dict]:
    """Flatten durable seed/cache and latest partial file records by source."""
    by_source = {}
    for item in (previous or {}).get('resume_cache', []):
        if item.get('source'):
            by_source[item['source']] = copy.deepcopy(item)
    for item in (previous or {}).get('files', []):
        if item.get('source'):
            by_source[item['source']] = copy.deepcopy(item)
    return list(by_source.values())


def reaggregate_result(record: dict, warning_pct: float = 5) -> dict:
    updated = copy.deepcopy(record)
    updated['files'] = [aggregate_file_observations(item, warning_pct) for item in record.get('files', [])]
    for key in HEARTBEAT_METRICS:
        updated[key] = sum(item[key] for item in updated['files'])
    updated['measured_file_count'] = len(updated['files'])
    if updated.get('status') == 'complete' and updated.get('expected_file_count') is not None and updated['expected_file_count'] != len(updated['files']):
        raise ValueError('completed result does not cover its declared file set')
    updated['aggregation'] = 'mean_of_available_measurements'
    updated['variation_warning'] = any(item['variation_warning'] for item in updated['files'])
    return updated


def validated_cached_observation(record: dict, fingerprints: dict, warning_pct: float = 5) -> dict:
    aggregated = aggregate_file_observations(record, warning_pct)
    first = aggregated['measurements'][0]
    if any(first[key] != fingerprints[key] for key in ('source_sha256', 'setup_sha256')):
        raise ValueError('restored source/setup fingerprint mismatch')
    return aggregated


def reusable_endpoint_validation(evidence: dict, repository: dict) -> dict | None:
    """Return trusted postprocessing evidence when it exactly covers this endpoint."""
    point = evidence.get('point', {})
    build = point.get('build', {})
    signatures = point.get('signature_validation', {})
    metric = point.get('authoritative_metric', {})
    official = signatures.get('official_comparator', {})
    if not (
        point.get('kind') == 'final'
        and point.get('exact') is True
        and point.get('matches_submission') is True
        and point.get('source_archive_sha256') == evidence.get('source_archive_canonical_sha256')
        and build.get('passed') is True
        and not build.get('timed_out', False)
        and signatures.get('checked') is True
        and signatures.get('preserved') is True
        and signatures.get('comparator_rejected') is False
        and official.get('verdict') == 'accepted'
        and metric.get('checked') is True
        and metric.get('matched') is True
        and metric.get('lean_tokens') == point.get('lean_tokens')
    ):
        return None
    expected_tools = repository.get('palomar_comparator', {}).get('tools', {})
    actual_tools = official.get('tools', {})
    if any(
        actual_tools.get(name, {}).get('sha256') != expected.get('sha256')
        for name, expected in expected_tools.items()
    ):
        return None
    return {
        'build': copy.deepcopy(build),
        'signature_validation': copy.deepcopy(signatures),
        'authoritative_metric': copy.deepcopy(metric),
        'provenance': 'reused_pinned_postprocessing_endpoint',
    }


def fingerprint_built_files(env, files: list[str], log: Path) -> dict:
    payload = json.dumps(files)
    code = (
        "import hashlib,json,runpy; from pathlib import Path; "
        f"m=runpy.run_path({REMOTE_COUNTER!r}); p=Path('/testbed'); files=json.loads({payload!r}); "
        "print(json.dumps({n:{'source_sha256':hashlib.sha256((p/n).read_bytes()).hexdigest(),"
        "'setup_sha256':hashlib.sha256(m['setup_path'](p/n,p).read_bytes()).hexdigest()} for n in files},sort_keys=True))"
    )
    response = execute(env, 'python3 -c ' + shlex.quote(code), 300, log)
    return json.loads(response['output'].splitlines()[-1])


def measure_built_variant(
    env,
    repository,
    variant,
    manifest,
    directory,
    progress=None,
    task=None,
    previous=None,
    *,
    clean_build_seconds=None,
    validation=None,
    lean_tokens=None,
    source_identity=None,
    result_path=None,
):
    """Measure a variant whose clean build is already live in ``env``.

    Postprocessing calls this immediately after its authoritative clean build,
    before restoring another checkpoint or retiring the repository container.
    This preserves the heartbeat protocol while avoiding a second clean build.
    The caller retains ownership of the environment.
    """
    result = {
        'status': 'running', 'repository': repository['id'], 'variant': variant,
        'method': METHOD_VERSION, 'files': [], 'body_heartbeats': None,
        'total_heartbeats': None, 'import_heartbeats': None,
        'resume_cache': resume_observation_cache(previous),
        'clean_build_reused': True,
        'clean_build_source': 'postprocessing_authoritative_clean_build',
        'clean_build_passed': True,
    }
    if clean_build_seconds is not None:
        result['clean_build_seconds'] = clean_build_seconds
    if source_identity:
        result.update(copy.deepcopy(source_identity))
    if (previous or {}).get('resume_provenance') is not None:
        result['resume_provenance'] = copy.deepcopy(previous['resume_provenance'])
    result_path = Path(result_path) if result_path is not None else directory / f'{variant}.json'
    log = result_path.with_suffix('.log')

    def update(description):
        if progress is not None and task is not None:
            progress.update(task, description=description)

    try:
        env.write_file_bytes(REMOTE_COUNTER, COUNTER.read_bytes())
        listing_command = f'python3 {REMOTE_COUNTER} --project /testbed --list-built'
        for name in repository['exclude_files']:
            listing_command += ' --exclude-file ' + shlex.quote(name)
        listing = json.loads(
            execute(env, listing_command, 120, log)['output'].splitlines()[-1]
        )
        result.update(listing)
        files = result.pop('files')
        result['files'] = []
        result['expected_file_count'] = len(files)
        if not files:
            raise ValueError('no built project modules found')
        if variant == 'optimized':
            if validation is None:
                raise ValueError('integrated optimized heartbeat measurement requires validation')
            if (
                not validation.get('build', {}).get('passed')
                or not validation.get('signature_validation', {}).get('preserved')
            ):
                raise ValueError('compressed endpoint failed official correctness validation')
            result['validation'] = copy.deepcopy(validation)
        if lean_tokens is None:
            raise ValueError('integrated heartbeat measurement requires pinned Lean tokens')
        result['lean_tokens'] = lean_tokens
        atomic_json(result_path, result)
        warning_pct = manifest['evaluation'].get(
            'variation_warning_relative_range_pct', 5
        )
        cached = {item['source']: item for item in result['resume_cache']}
        fingerprints = fingerprint_built_files(env, files, log) if cached else {}
        result['rejected_cached_files'] = [
            {'source': name, 'reason': 'not in rebuilt declared file set', 'record': item}
            for name, item in cached.items() if name not in set(files)
        ]
        for index, name in enumerate(files, 1):
            old = cached.get(name)
            if old:
                try:
                    result['files'].append(
                        validated_cached_observation(old, fingerprints[name], warning_pct)
                    )
                    atomic_json(result_path, result)
                    continue
                except (ValueError, KeyError) as error:
                    result['rejected_cached_files'].append(
                        {'source': name, 'reason': str(error), 'record': old}
                    )
            measurements = []
            repetitions = manifest['evaluation'].get('repetitions', 1)
            for repetition in range(repetitions):
                update(
                    f'{repository["id"]} {variant}: {index}/{len(files)} '
                    f'{name} [{repetition + 1}/{repetitions}]'
                )
                timeout = manifest['timeout']['file_seconds']
                command = (
                    f'python3 {REMOTE_COUNTER} --project /testbed --source '
                    + shlex.quote('/testbed/' + name)
                    + f' --timeout-seconds {timeout} --measure-imports'
                )
                execution = execute(env, command, 2 * timeout + 60, log)
                measurement = json.loads(execution['output'].splitlines()[-1])
                if (
                    not measurement.get('ok')
                    or measurement.get('method') != METHOD_VERSION
                ):
                    raise ValueError(f'{name}: invalid measurement response')
                measurements.append(measurement)
                partial = {
                    'source': name, 'status': 'running',
                    'measurements': copy.deepcopy(measurements),
                }
                if result['files'] and result['files'][-1].get('source') == name:
                    result['files'][-1] = partial
                else:
                    result['files'].append(partial)
                result['resume_cache'] = resume_observation_cache(result)
                atomic_json(result_path, result)
            result['files'][-1] = aggregate_file_observations(
                {'source': name, 'status': 'complete', 'measurements': measurements},
                warning_pct,
            )
            atomic_json(result_path, result)
        result = reaggregate_result(result, warning_pct)
        result.update(status='complete', measured_file_count=len(files))
    except Exception as error:
        result.update(status='failed', error=str(error))
    atomic_json(result_path, result)
    return result
def measure_variant(repository, variant, manifest, directory, progress, task, previous=None):
    result = {'status': 'running', 'repository': repository['id'], 'variant': variant,
              'method': METHOD_VERSION, 'files': [], 'body_heartbeats': None,
              'total_heartbeats': None, 'import_heartbeats': None,
              'resume_cache': resume_observation_cache(previous)}
    if (previous or {}).get('resume_provenance') is not None:
        result['resume_provenance'] = copy.deepcopy(previous['resume_provenance'])
    result_path = directory / f'{variant}.json'
    log = directory / f'{variant}.log'
    env = None
    try:
        env = make_environment(repository, manifest['container'])
        result['resource_guard'] = heartbeat_memory_guard.verify_environment(env, manifest)
        env.restore_source_archive(repository['prepared_source_archive'], ['.'], timeout=900)
        result['image_id'] = repository['image_id']
        if variant == 'optimized':
            evidence = json.loads(checked_file(repository['endpoint']).read_text())
            archive = checked_file(evidence['source_archive'])
            if pp._canonical_archive_sha256(archive) != evidence['source_archive_canonical_sha256']:
                raise ValueError('canonical endpoint archive drift')
            env.restore_source_archive(archive, ['.'], timeout=900)
            correct, actual = pp._verify_restored_source(env, archive, evidence['source_archive_canonical_sha256'], timeout=900)
            if not correct:
                raise ValueError('restored endpoint differs from pinned archive')
            result['source_archive_sha256'] = evidence['source_archive_canonical_sha256']
        else:
            result['source_tree_sha256'] = repository['materialization']['tree_sha256']
        toolchain = env.read_file('/testbed/lean-toolchain').strip()
        if toolchain != repository['expected_toolchain']:
            raise ValueError('Lean toolchain drift')
        result['toolchain'] = toolchain
        progress.update(task, description=f'{repository["id"]} {variant}: clean build')
        cleanup = "from pathlib import Path; import shutil; p=Path('/testbed/.lake/build'); assert not p.is_symlink(); shutil.rmtree(p, ignore_errors=True)"
        execute(env, 'python3 -c ' + shlex.quote(cleanup), 120, log)
        build = execute(env, f'LEAN_NUM_THREADS={manifest["container"]["build_jobs"]} ' + repository['build_command'], manifest['container']['build_timeout_seconds'], log)
        result['clean_build_seconds'] = build['elapsed_seconds']
        result['clean_build_passed'] = True
        env.write_file_bytes(REMOTE_COUNTER, COUNTER.read_bytes())
        listing_command = f'python3 {REMOTE_COUNTER} --project /testbed --list-built'
        for name in repository['exclude_files']:
            listing_command += ' --exclude-file ' + shlex.quote(name)
        listing = json.loads(execute(env, listing_command, 120, log)['output'].splitlines()[-1])
        result.update(listing)
        files = result.pop('files')
        result['files'] = []
        result['expected_file_count'] = len(files)
        if not files:
            raise ValueError('no built project modules found')
        if variant == 'optimized':
            validation = (
                reusable_endpoint_validation(evidence, repository)
                if manifest['evaluation'].get('reuse_pinned_postprocessing_validation', False)
                else None
            )
            if validation is None:
                progress.update(task, description=f'{repository["id"]} optimized: verify main results')
                validation = pp._build_palomar_point(copy.deepcopy(evidence['point']), archive.parent, repository,
                    manifest['container'], repo_root=ROOT, environment=env)
            result['validation'] = validation
            if not validation['build'].get('passed') or not validation['signature_validation'].get('preserved'):
                raise ValueError('compressed endpoint failed official correctness validation')
            result['lean_tokens'] = validation['authoritative_metric']['lean_tokens']
        else:
            from leanlean.benchmarks.leanlean import _measure_lean_tokens
            result['lean_tokens'] = _measure_lean_tokens(env, exclude_files=repository['exclude_files'])
            if result['lean_tokens'] < 0:
                raise ValueError('baseline token measurement failed')
        atomic_json(result_path, result)
        warning_pct = manifest['evaluation'].get('variation_warning_relative_range_pct', 5)
        cached = {item['source']: item for item in result['resume_cache']}
        fingerprints = fingerprint_built_files(env, files, log) if cached else {}
        result['rejected_cached_files'] = [
            {'source': name, 'reason': 'not in rebuilt declared file set', 'record': item}
            for name, item in cached.items() if name not in set(files)
        ]
        for index, name in enumerate(files, 1):
            old = cached.get(name)
            if old:
                try:
                    aggregated = validated_cached_observation(old, fingerprints[name], warning_pct)
                    result['files'].append(aggregated)
                    atomic_json(result_path, result)
                    continue
                except (ValueError, KeyError) as error:
                    result['rejected_cached_files'].append({'source': name, 'reason': str(error), 'record': old})
            measurements = []
            repetitions = manifest['evaluation'].get('repetitions', 1)
            for repetition in range(repetitions):
                progress.update(task, description=f'{repository["id"]} {variant}: {index}/{len(files)} {name} [{repetition + 1}/{repetitions}]')
                timeout = manifest['timeout']['file_seconds']
                command = (f'python3 {REMOTE_COUNTER} --project /testbed --source ' + shlex.quote('/testbed/' + name)
                           + f' --timeout-seconds {timeout} --measure-imports')
                execution = execute(env, command, 2 * timeout + 60, log)
                measurement = json.loads(execution['output'].splitlines()[-1])
                if not measurement.get('ok') or measurement.get('method') != METHOD_VERSION:
                    raise ValueError(f'{name}: invalid measurement response')
                measurements.append(measurement)
                # Persist each successful observation before attempting another.
                partial = {'source': name, 'status': 'running', 'measurements': copy.deepcopy(measurements)}
                if result['files'] and result['files'][-1].get('source') == name:
                    result['files'][-1] = partial
                else:
                    result['files'].append(partial)
                result['resume_cache'] = resume_observation_cache(result)
                atomic_json(result_path, result)
            result['files'][-1] = aggregate_file_observations(
                {'source': name, 'status': 'complete', 'measurements': measurements}, warning_pct)
            atomic_json(result_path, result)
        result = reaggregate_result(result, warning_pct)
        result.update(status='complete', measured_file_count=len(files))
    except Exception as error:
        result.update(status='failed', error=str(error))
    finally:
        if env is not None:
            env.cleanup()
        atomic_json(result_path, result)
    return result


def write_summary(output, manifest, rows):
    ordered = [rows.get(r['id'], {'repository': r['id'], 'status': 'pending'}) for r in manifest['repositories']]
    pairs = [r['comparison'] for r in ordered if r.get('comparison', {}).get('eligible')]
    ratios = [p['heartbeat_ratio'] for p in pairs if p['heartbeat_ratio'] is not None]
    summary = {
        'run_id': manifest['run_id'], 'method': METHOD_VERSION, 'unit': 'raw_heartbeats',
        'repository_count': len(ordered), 'complete_pairs': len(pairs),
        'status': 'complete' if len(pairs) == len(ordered) else 'incomplete',
        'macro_heartbeat_reduction_pct': 100 * (1 - statistics.mean(ratios)) if ratios else None,
        'aggregate_scope': 'available valid pairs only; inspect coverage before comparison',
        'repositories': ordered,
    }
    atomic_json(output / 'summary.json', summary)
    fields = ['repository', 'status', 'stripped_heartbeats', 'optimized_heartbeats', 'heartbeat_ratio',
              'heartbeat_reduction_pct', 'lean_token_reduction_pct', 'clean_build_ratio', 'both_improved']
    with (output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({
                'repository': row['repository'], 'status': row['status'],
                'stripped_heartbeats': row.get('stripped', {}).get('body_heartbeats'),
                'optimized_heartbeats': row.get('optimized', {}).get('body_heartbeats'),
                **{k: row.get('comparison', {}).get(k) for k in fields[4:]},
            })


def run(manifest_path: Path) -> int:
    if not os.environ.get('TMUX'):
        raise RuntimeError('launch heartbeat comparisons in the manifest\'s named tmux session')
    manifest = yaml.safe_load(manifest_path.read_text())
    heartbeat_memory_guard.verify_limits(manifest)
    optimized_only = (
        manifest.get('measurement_policy', {}).get('variants_computed')
        == ['optimized']
    )
    if optimized_only:
        reused_baselines = {
            item['repository']
            for item in manifest.get('reuse_results', [])
            if item.get('variant') == 'stripped'
        }
        missing = [
            repository['id'] for repository in manifest['repositories']
            if repository['id'] not in reused_baselines
        ]
        if missing:
            raise ValueError(
                'optimized-only run lacks pinned baselines for: '
                + ', '.join(missing)
            )
    if manifest.get('kind') != KIND or manifest.get('evaluation', {}).get('method') != METHOD_VERSION:
        raise ValueError('unsupported heartbeat manifest')
    if manifest['container']['network_policy'] != 'model_proxy_only' or manifest['container']['network_mode'] != 'none':
        raise ValueError('isolated network policy is required')
    for record in manifest['implementation']:
        checked_file(record)
    checked_file(manifest['dataset'])
    checked_file(manifest['source_manifest'])
    output = ROOT / manifest['outputs']['directory']
    output.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha(manifest_path)
    receipt = output / 'manifest.json'
    if receipt.exists() and json.loads(receipt.read_text())['sha256'] != manifest_sha:
        raise ValueError('output directory belongs to a different manifest')
    atomic_json(receipt, {'path': str(manifest_path), 'sha256': manifest_sha})
    console = Console()
    rows, lock = {}, threading.Lock()
    console.print('Checking the 200 GB aggregate ceiling, including image preparation…')
    heartbeat_memory_guard.verify_limits(manifest)
    for item in manifest.get('reuse_results', []):
        record = json.loads(checked_file(item['result']).read_text())
        repository = next(r for r in manifest['repositories'] if r['id'] == item['repository'])
        if record.get('method') != METHOD_VERSION:
            raise ValueError('only measurements with the same method may be reused')
        if record.get('repository') != item['repository'] or record.get('variant') != item['variant']:
            raise ValueError('reused measurement identity mismatch')
        if record.get('toolchain') != repository['expected_toolchain']:
            raise ValueError('reused measurement toolchain mismatch')
        if item['variant'] == 'stripped':
            if record.get('source_tree_sha256') != repository['materialization']['tree_sha256']:
                raise ValueError('reused baseline source mismatch')
        else:
            evidence = json.loads(checked_file(repository['endpoint']).read_text())
            if record.get('source_archive_sha256') != evidence['source_archive_canonical_sha256']:
                raise ValueError('reused optimized source mismatch')
        destination = output / item['repository'] / (item['variant'] + '.json')
        if not destination.exists():
            atomic_json(destination, record)
    with Progress(SpinnerColumn(), TextColumn('{task.description}', markup=False), TimeElapsedColumn(), console=console) as progress:
        def worker(repository):
            rid = repository['id']
            directory = output / rid
            directory.mkdir(exist_ok=True)
            task = progress.add_task(
                f'{rid}: materializing pinned stripped source',
                total=1 if optimized_only else 2,
            )
            row = {'repository': rid, 'status': 'running'}
            try:
                with prepared_repository(repository) as resolved:
                    variants = (
                        ['optimized']
                        if optimized_only
                        else ['stripped', 'optimized']
                    )
                    if optimized_only:
                        baseline = directory / 'stripped.json'
                        record = json.loads(baseline.read_text())
                        if record.get('status') != 'complete':
                            raise ValueError('pinned stripped baseline is incomplete')
                        row['stripped'] = reaggregate_result(
                            record,
                            manifest['evaluation'].get(
                                'variation_warning_relative_range_pct', 5
                            ),
                        )
                    for variant in variants:
                        existing = directory / f'{variant}.json'
                        previous = json.loads(existing.read_text()) if existing.exists() else {}
                        if previous.get('status') == 'complete':
                            row[variant] = reaggregate_result(previous, manifest['evaluation'].get('variation_warning_relative_range_pct', 5))
                            atomic_json(existing, row[variant])
                        else:
                            row[variant] = measure_variant(resolved, variant, manifest, directory, progress, task, previous)
                        progress.advance(task)
                        with lock:
                            rows[rid] = row.copy()
                            write_summary(output, manifest, rows)
                row['comparison'] = compare_results(row['stripped'], row['optimized'])
                row['status'] = 'complete' if row['comparison']['eligible'] else 'failed'
            except Exception as error:
                row.update(status='failed', error=str(error))
            atomic_json(directory / 'result.json', row)
            progress.update(task, description=f'{rid}: {row["status"]}')
            with lock:
                rows[rid] = row
                write_summary(output, manifest, rows)
            return row
        repositories = {r['id']: r for r in manifest['repositories']}
        write_summary(output, manifest, rows)
        with ThreadPoolExecutor(max_workers=manifest['parallelism']['workers']) as pool:
            futures = [pool.submit(worker, repositories[rid]) for rid in manifest['execution_order']]
            for future in as_completed(futures):
                future.result()
    console.print(f'Results: {output / "comparison.csv"}')
    return int(any(r['status'] != 'complete' for r in rows.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?')
    parser.add_argument('--prepare', type=Path, metavar='POSTPROCESSING_MANIFEST')
    parser.add_argument('--run-id')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if not args.run_id:
            parser.error('--run-id is required with --prepare')
        print(prepare(args.prepare.resolve(), args.run_id))
        return 0
    if not args.manifest:
        parser.error('provide a manifest, or --prepare with --run-id')
    if args.launch:
        heartbeat_memory_guard.launch(yaml.safe_load(args.manifest.read_text()), args.manifest.resolve(), ROOT)
    return run(args.manifest.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
