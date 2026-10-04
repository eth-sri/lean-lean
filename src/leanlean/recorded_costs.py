"""Usage-backed request ledgers for capture repair and replay cost validation."""
from __future__ import annotations
import bisect
import copy
import math
from datetime import datetime
from leanlean.model.costs import cost_from_responses, require_complete_cost_accounting
from leanlean.model.litellm_wrapper.accounting import record_identity, record_scope


def prepare_requests(records, *, scopes=None):
    """Validate ownership and deduplicate identical callbacks, never requests."""
    unique = {}
    for record in records:
        if record.get('event') != 'success':
            continue
        scope = record_scope(record)
        if not scope:
            raise ValueError('usage request has no invocation scope')
        if scopes is not None and scope not in scopes:
            continue
        identity = record_identity(record)
        if not identity:
            raise ValueError('usage request has no identity')
        key = (scope, identity)
        if key in unique:
            if unique[key]['response'] != record['response']:
                raise ValueError('conflicting duplicate request usage')
            continue
        unique[key] = copy.deepcopy(record)
    values = list(unique.values())
    if not values:
        raise ValueError('no scoped completed request usage')
    require_complete_cost_accounting([r['response'] for r in values])
    for record in values:
        usage = record['response']['usage']
        prompt = usage.get('prompt_tokens', usage.get('input_tokens'))
        output = usage.get('completion_tokens', usage.get('output_tokens'))
        details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
        cached = details.get('cached_tokens')
        creation = details.get('cache_creation_tokens', usage.get('cache_creation_input_tokens', 0))
        if any(type(v) is not int or v < 0 for v in (prompt, output, cached, creation)) or cached + creation > prompt:
            raise ValueError('invalid request token partition')
    return values


def request_ledger(records):
    """Build a ledger from measured counts, with per-model cumulative counts."""
    records=copy.deepcopy(records)
    for record in records:
        response=record['response']
        if record.get('ts') is None:
            if response.get('completed_at') is not None:
                record['ts']=response['completed_at'];record['timestamp_basis']='provider_completion'
            elif response.get('created_at') is not None:
                record['ts']=response['created_at'];record['timestamp_basis']='request_start_fallback'
    timestamped = all(isinstance(r.get('ts'), (int, float)) and math.isfinite(r['ts']) for r in records)
    ordered = sorted(records, key=lambda r:r['ts']) if timestamped else records
    cumulative = 0.0
    start_fallback_count = 0
    by_model = {}
    entries = []
    tools = {}
    for record in ordered:
        response = record['response'];usage = response['usage'];model = response['model']
        prompt = usage.get('prompt_tokens', usage.get('input_tokens'))
        output = usage.get('completion_tokens', usage.get('output_tokens'))
        details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
        counts = dict(prompt_tokens=prompt, output_tokens=output, cached_tokens=details['cached_tokens'],
                      cache_creation_tokens=details.get('cache_creation_tokens', usage.get('cache_creation_input_tokens', 0)))
        accumulated = by_model.setdefault(model, dict.fromkeys(counts, 0))
        for k,v in counts.items():accumulated[k]+=v
        cumulative += cost_from_responses([response])
        start_fallback_count += record.get('timestamp_basis') == 'request_start_fallback'
        entry = {'ts':record.get('ts'), 'request_id':record_identity(record), 'accounting_scope':record_scope(record),
                 'cost_usd':round(cumulative, 10), 'usage_by_model':copy.deepcopy(by_model),
                 'tool_call_ids':record.get('tool_call_ids') or [],
                 'timestamp_basis':record.get('timestamp_basis','recorded_completion'),
                 'request_start_time_fallback_count':start_fallback_count}
        entries.append(entry)
        for tool in entry['tool_call_ids']:
            if tool in tools and tools[tool]['request_id'] != entry['request_id']:
                raise ValueError('tool ID attributed to multiple requests')
            tools[tool] = entry
    return {'format':'recorded-request-cost-ledger-v1', 'request_count':len(entries), 'entries':entries,
            'tool_boundaries':tools, 'timestamped':timestamped, 'final_cost_usd':round(cumulative, 10),
            'usage_by_model':by_model, 'request_start_time_fallback_count':start_fallback_count,
            'basis':'recorded request usage; completion time preferred, recorded start time fallback; no proration'}


def timestamp_cost_entries(ledger):
    """Sum only requests with recorded completion times; disclose untimed usage.

    A single untimed request must not erase every checkpoint's known usage.
    Untimed costs remain in the final total and in explicit timeline bounds.
    """
    if ledger['timestamped']:
        return ledger['entries'], 0, 0.0
    timed=[];previous_cost=0.;previous_usage={};untimed_count=0;untimed_cost=0.
    for entry in ledger['entries']:
        delta=entry['cost_usd']-previous_cost
        usage={m:{k:v-previous_usage.get(m,{}).get(k,0) for k,v in counts.items()}
               for m,counts in entry['usage_by_model'].items()}
        previous_cost=entry['cost_usd'];previous_usage=entry['usage_by_model']
        ts=entry.get('ts')
        if isinstance(ts,(int,float)) and not isinstance(ts,bool) and math.isfinite(ts):
            timed.append((ts,delta,usage,entry))
        else:
            untimed_count+=1;untimed_cost+=delta
    running=0.;totals={};entries=[]
    for ts,delta,usage,original in sorted(timed,key=lambda item:item[0]):
        running+=delta
        for model,counts in usage.items():
            total=totals.setdefault(model,dict.fromkeys(counts,0))
            for key,value in counts.items():total[key]+=value
        entries.append(dict(original,cost_usd=round(running,10),usage_by_model=copy.deepcopy(totals)))
    return entries,untimed_count,round(untimed_cost,10)


def apply_request_ledger(playback, ledger, *, final_complete, source_evidence):
    """Use request receipt times or tool IDs; preserve unknowns as unknowns."""
    observations=playback.get('checkpoint_observations') or []
    by_edit={o['edit_index']:o for o in observations if o.get('edit_index') is not None}
    entries=ledger['entries']
    timed_entries,untimed_count,untimed_cost=timestamp_cost_entries(ledger)
    times=[e['ts'] for e in timed_entries]
    zero={'cost_usd':0.0,'usage_by_model':{}}
    def boundary(record, field):
        kind=record.get('kind');entry=None;basis='unknown_source_boundary';exact=False
        if kind=='baseline':entry=zero;basis='baseline';exact=True
        elif kind=='final':entry=entries[-1];basis='completed_requests';exact=final_complete
        else:
            tool=record.get('tool_use_id') or record.get('tool_call_id')
            if tool in ledger['tool_boundaries']:
                entry=ledger['tool_boundaries'][tool];basis='request_tool_identity';exact=True
            else:
                observation=record if 'wall_timestamp' in record else by_edit.get(record.get('edit_index'),{})
                stamp=observation.get('wall_timestamp')
                if times and stamp:
                    timestamp=datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()
                    # Use observation start, not snapshot-export completion: it
                    # excludes requests completed while source export was running.
                    index=bisect.bisect_right(times,timestamp)-1
                    entry=timed_entries[index] if index>=0 else zero;basis='request_receipt_timestamp'
                    record['untimed_request_count']=untimed_count
                    record['untimed_request_cost_usd']=untimed_cost
                    record['recorded_cost_lower_bound_usd']=entry['cost_usd']
                    record['recorded_cost_upper_bound_usd']=round(entry['cost_usd']+untimed_cost,10)
                    record['cost_estimated']=True
                    record['request_start_time_fallback_count']=entry.get('request_start_time_fallback_count',0)
        record[field]=entry['cost_usd'] if entry else None
        record['usage_by_model']=copy.deepcopy(entry['usage_by_model']) if entry else None
        record['cost_join_basis']=basis
        record['cost_boundary_exact']=exact
        record.pop('marginal_cost_since_previous_edit_usd',None)
    for o in observations:boundary(o,'cumulative_cost_usd')
    previous=0.0
    for point in playback.get('points') or []:
        boundary(point,'cost_usd');current=point['cost_usd']
        if current is not None and previous is not None and current < previous-1e-7:
            # A tool ID can refer to a request completed before a newer source
            # observation. Keep timing attribution, never sort source states.
            point['cost_usd']=None;point['usage_by_model']=None;point['cost_boundary_exact']=False
            point['cost_join_basis']='ambiguous_source_order';current=None
        point['marginal_cost_since_previous_edit_usd']=round(current-previous,10) if current is not None and previous is not None else None
        previous=current
    total=ledger['final_cost_usd']
    playback.update(final_cost_usd=total,authoritative_final_cost_usd=total if final_complete else None,
                    captured_completed_turn_cost_usd=total,unattributed_terminal_cost_usd=0.0 if final_complete else None,
                    final_cost_complete=final_complete,cost_prorated=False,cost_basis=ledger['basis'],
                    recorded_cost_evidence=source_evidence,
                    untimed_request_count=untimed_count,untimed_request_cost_usd=untimed_cost,
                    request_start_time_fallback_count=ledger.get('request_start_time_fallback_count',0),
                    cost_timeline_complete=all(p['cost_usd'] is not None for p in playback.get('points',[])))
    playback['every_edit_checkpoint_costed']=all(p.get('cost_boundary_exact') is True for p in playback.get('points',[]) if p.get('kind') not in {'baseline','final'})


def invalidate_unmeasured_costs(playback, reason):
    """Keep independently retained totals, remove unsupported point estimates."""
    for point in playback.get('points',[]):
        point['marginal_cost_since_previous_edit_usd']=None
        if point.get('kind') in {'baseline','final'}:continue
        point['cost_usd']=None;point['cost_boundary_exact']=False
        point['marginal_cost_since_previous_edit_usd']=None
        point['cost_join_basis']=reason
    for o in playback.get('checkpoint_observations',[]):
        o['cumulative_cost_usd']=None;o['cost_boundary_exact']=False
    playback.update(cost_prorated=False,cost_timeline_complete=False,every_edit_checkpoint_costed=False,
                    cost_basis='retained final total; intermediate attribution unavailable; no proration',
                    replay_cost_status='blocked',replay_cost_blocker=reason)


def use_authoritative_terminal_cost(playback, terminal_cost):
    """Keep native terminal totals and explicitly estimated request/edit costs."""
    if isinstance(terminal_cost, bool) or not isinstance(terminal_cost, (int, float)) or not math.isfinite(terminal_cost) or terminal_cost < 0:
        raise ValueError('invalid native terminal cost')
    recorded = playback.get('captured_completed_turn_cost_usd')
    playback.update(
        final_cost_usd=terminal_cost, authoritative_final_cost_usd=terminal_cost,
        final_cost_complete=True, final_cost_basis='native Opus terminal total_cost_usd',
        intermediate_costs_estimated=True, cost_prorated=False,
        cost_basis='native Opus terminal total; request-based edit estimates, unscaled',
        replay_cost_status='authoritative_terminal_with_edit_estimates', replay_cost_blocker=None,
        every_edit_checkpoint_costed=False,
        request_minus_terminal_cost_usd=round(recorded-terminal_cost, 10) if recorded is not None else None,
    )
    for point in playback.get('points') or []:
        if point.get('kind') == 'final':
            point.update(cost_usd=terminal_cost, cost_boundary_exact=True,
                         cost_join_basis='native_terminal_total',
                         marginal_cost_since_previous_edit_usd=None)
        elif point.get('kind') != 'baseline':
            point['cost_boundary_exact']=False
            point['cost_estimated']=point.get('cost_usd') is not None
    for observation in playback.get('checkpoint_observations') or []:
        observation['cost_boundary_exact']=False
        observation['cost_estimated']=observation.get('cumulative_cost_usd') is not None
