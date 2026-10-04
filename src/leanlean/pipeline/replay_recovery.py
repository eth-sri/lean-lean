"""Bounded recovery for verifier infrastructure failures, never proof failures."""
from __future__ import annotations

import copy
import json
import logging
import subprocess
from collections.abc import Mapping

DEFAULT_CHECKPOINT_RETRIES = 2
INTERRUPTED_RETURN_CODES = frozenset({137, 143, -9, -15})


class ReplayInfrastructureError(RuntimeError):
    """The current checkpoint could not be verified within the recovery budget."""


def infrastructure_failed(result):
    build = result.get('build')
    verification = result.get('lean_verify')
    if not isinstance(verification, Mapping):
        verification = result.get('signature_validation')
    if not isinstance(build, Mapping):
        return True
    if (build.get('setup_failed') is True or build.get('infrastructure_failure') is True
            or build.get('returncode') in INTERRUPTED_RETURN_CODES):
        return True
    output = str(build.get('output') or '').lower()
    if build.get('passed') is not True and (
        'too many open files' in output
        or ('[landrun:error]' in output and 'permission denied' in output)
        or ("cannot exec 'remote-https'" in output and 'permission denied' in output)
    ):
        return True
    if build.get('passed') is not True:
        return False
    if (isinstance(verification, Mapping)
            and verification.get('skipped') in {'checkpoint_lean_verify_disabled', 'replay_lean_verify_disabled'}):
        return False
    if not isinstance(verification, Mapping) or verification.get('checked') is not True:
        return True
    verification_output = str(verification.get('output') or '').lower()
    if verification.get('passed') is not True and (
        'too many open files' in verification_output
        or ('[landrun:error]' in verification_output and 'permission denied' in verification_output)
        or ("cannot exec 'remote-https'" in verification_output and 'permission denied' in verification_output)
    ):
        return True
    return verification.get('infrastructure_failure') is True


def environment_diagnostics(environment):
    """Inspect only runtime state, never container environment/authentication."""
    container_id = getattr(environment, 'container_id', None)
    record = {'container_id': container_id}
    if not container_id:
        return record
    try:
        executable = environment.config.executable
        result = subprocess.run(
            [executable, 'inspect', '--format', '{{json .State}}', container_id],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode:
            record['inspection_error'] = result.stderr.strip()[-1000:]
        else:
            state = json.loads(result.stdout)
            record['state'] = {key: state.get(key) for key in (
                'Status', 'Running', 'OOMKilled', 'ExitCode', 'Error', 'StartedAt', 'FinishedAt'
            )}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        record['inspection_error'] = str(error)[-1000:]
    return record


def retry_checkpoint(environment, build, recreate, record_failure, *, retries=DEFAULT_CHECKPOINT_RETRIES):
    """Retry only this checkpoint; each retry uses a new, revalidated environment."""
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
        raise ValueError('checkpoint infrastructure retries must be an integer from 0 to 5')
    current = environment
    try:
        for attempt in range(retries + 1):
            result = build(current)
            if not infrastructure_failed(result):
                return result, current
            result = copy.deepcopy(result)
            result.setdefault('build', {})['infrastructure_failure'] = True
            result['build']['runtime_diagnostics'] = environment_diagnostics(current)
            record_failure(result, attempt + 1)
            if attempt == retries:
                detail = str(result['build'].get('output') or 'verifier infrastructure failed')[-1000:]
                raise ReplayInfrastructureError(
                    f'checkpoint infrastructure failed after {attempt + 1} attempts; '
                    f'progress retained, no proof verdict assigned: {detail}'
                )
            logging.getLogger(__name__).warning(
                'Verifier infrastructure failed; recreating its offline container and '
                'retrying the SAME checkpoint (%s/%s).', attempt + 1, retries,
            )
            current.cleanup()
            current = None
            current = recreate()
    except BaseException:
        # A replacement environment must not leak when recovery is exhausted.
        if current is not None and current is not environment:
            current.cleanup()
        raise
