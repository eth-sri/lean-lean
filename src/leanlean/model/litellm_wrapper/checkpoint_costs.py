"""Join live checkpoints to scoped request usage, with explicit timing estimates."""
from __future__ import annotations
import copy
import math
from leanlean.model.costs import require_complete_cost_accounting
from leanlean.recorded_costs import (
    prepare_requests, request_ledger, apply_request_ledger,
    use_authoritative_terminal_cost,
)


def join_checkpoint_costs(points, observations, records, *, scope_id, expected_total,
                          terminal_authoritative=False):
    if not scope_id:
        raise ValueError('checkpoint cost joins require an explicit invocation scope')
    calls=prepare_requests(records,scopes={scope_id})
    ledger=request_ledger(calls)
    matches=math.isclose(ledger['final_cost_usd'],expected_total,abs_tol=1e-7,rel_tol=1e-9)
    if not matches and not terminal_authoritative:
        raise ValueError('scoped journal does not reconcile with invocation total')
    observed={o['edit_index']:o for o in observations if o.get('edit_index') is not None}
    for point in points:
        if point.get('kind') in {'baseline','final'}:continue
        observation=observed.get(point.get('edit_index'))
        if observation and observation.get('source_archive_sha256') != point.get('source_archive_sha256'):
            raise ValueError('checkpoint observation has a different source digest')
    # Validate before mutating source records. The shared offline/live method
    # uses completion time, falling back to recorded request start time.
    data={'points':copy.deepcopy(points),'checkpoint_observations':copy.deepcopy(observations)}
    apply_request_ledger(data,ledger,final_complete=matches,source_evidence={})
    if terminal_authoritative:use_authoritative_terminal_cost(data,expected_total)
    for original,updated in zip(points,data['points']):original.update(updated)
    for original,updated in zip(observations,data['checkpoint_observations']):original.update(updated)
    return {'source':'invocation_scoped_gateway_journal',
            'timestamp_join_complete':data['cost_timeline_complete'],
            'cost_prorated':False,'call_count':len(calls),
            'cost_accounting':require_complete_cost_accounting([r['response'] for r in calls]),
            'captured_completed_turn_cost_usd':ledger['final_cost_usd'],
            'terminal_reconciled':matches,'terminal_total_authoritative':terminal_authoritative,
            'request_minus_terminal_cost_usd':round(ledger['final_cost_usd']-expected_total,10),
            'request_start_time_fallback_count':ledger['request_start_time_fallback_count'],
            'untimed_request_count':data['untimed_request_count'],
            'untimed_request_cost_usd':data['untimed_request_cost_usd']}
