#!/usr/bin/env python3
"""Report all saved heartbeat metrics without changing or restarting a run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    'lean_tokens': 'Source tokens',
    'total_heartbeats': 'Full-file heartbeats',
    'body_heartbeats': 'Body heartbeats',
    'import_heartbeats': 'Imports/control heartbeats',
    'clean_build_seconds': 'Clean build seconds',
    'measured_file_count': 'Built project files',
}
DEFINITIONS = {
    'total_heartbeats': 'Sum of full-file counters: synchronous project elaboration plus import loading and counter overhead; excludes rebuilding dependency source.',
    'body_heartbeats': 'Sum of full-file minus matched imports-only control counters; estimates project body work.',
    'import_heartbeats': 'Sum of imports-only control counters; includes all imports and counter overhead, not only Mathlib.',
    'clean_build_seconds': 'One observed project build after removing project artifacts, retaining dependency caches; eight build threads on a shared server.',
    'lean_tokens': 'Lean source tokens; source size does not by itself measure readability or refactoring quality.',
    'measured_file_count': 'Project modules reached by the build target; excludes dependency packages and registered Challenge fixture.',
}


def compare_metric(before, after, eligible):
    ratio = after / before if eligible and before is not None and after is not None and before > 0 else None
    return {'stripped': before, 'optimized': after, 'ratio': ratio,
            'reduction_pct': 100 * (1 - ratio) if ratio is not None else None}


def build_report(summary, manifest):
    if summary['run_id'] != manifest['run_id'] or summary['method'] != manifest['evaluation']['method']:
        raise ValueError('summary does not match the selected run and method')
    source_rows = summary['repositories']
    by_id = {row['repository']: row for row in source_rows}
    if len(by_id) != len(source_rows) or set(by_id) - {r['id'] for r in manifest['repositories']}:
        raise ValueError('unexpected or duplicate repository in summary')
    rows = []
    for repository in manifest['repositories']:
        rid = repository['id']
        row = by_id.get(rid, {})
        variants = [row.get(variant, {}) for variant in ('stripped', 'optimized')]
        eligible = all(v.get('status') == 'complete' for v in variants)
        for variant in variants:
            if variant.get('status') != 'complete':
                continue
            for key in METRICS:
                value = variant.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{rid}: invalid completed {key}')
            if variant['total_heartbeats'] != variant['body_heartbeats'] + variant['import_heartbeats']:
                raise ValueError(f'{rid}: full heartbeats must equal body plus imports/control')
        source_repository = repository.get('palomar_comparator', {}).get('source_archive', {}).get('repository', rid)
        rows.append({
            'repository': rid, 'source_repository': source_repository,
            'status': 'complete' if eligible else row.get('status', 'pending'),
            'eligible': eligible,
            'metrics': {key: compare_metric(*(v.get(key) for v in variants), eligible) for key in METRICS},
        })
    aggregates = {}
    for key in METRICS:
        valid = [row['metrics'][key] for row in rows if row['eligible'] and row['metrics'][key]['ratio'] is not None]
        aggregates[key] = {
            'eligible_pairs': len(valid),
            'macro_reduction_pct': statistics.mean(v['reduction_pct'] for v in valid) if valid else None,
            'scope': 'Equal weight per available valid pair with positive baseline; not the full benchmark until coverage is complete.',
        }
    return {
        'run_id': manifest['run_id'], 'method': summary['method'], 'unit': 'raw_heartbeats',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'repository_count': len(rows), 'complete_pairs': sum(row['eligible'] for row in rows),
        'percentage_convention': 'Positive reduction means less work/size/time; negative means an increase.',
        'definitions': DEFINITIONS, 'aggregates': aggregates, 'repositories': rows,
    }


def csv_text(report):
    output = io.StringIO()
    fields = ['repository', 'source_repository', 'status', 'eligible']
    fields += [f'{key}_{suffix}' for key in METRICS for suffix in ('stripped', 'optimized', 'ratio', 'reduction_pct')]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in report['repositories']:
        flat = {key: row[key] for key in fields[:4]}
        flat.update({f'{key}_{suffix}': value for key, metric in row['metrics'].items() for suffix, value in metric.items()})
        writer.writerow(flat)
    return output.getvalue()


def markdown_text(report):
    def change(metric):
        value = metric['reduction_pct']
        if value is None:
            return '—'
        # Show changes here, while JSON/CSV explicitly expose reduction_pct.
        delta = -value
        return '≈0.0%' if abs(delta) < 0.05 else f'{delta:+.1f}%'

    lines = [
        '# Stripped versus compressed: all metrics', '',
        f'Run: `{report["run_id"]}`. Snapshot: {report["generated_at"]}.', '',
        f'**{report["complete_pairs"]}/{report["repository_count"]} complete pairs.** Changes below are compressed relative to stripped; negative means less. Near-zero changes display as ≈0.0%.', '',
        '| Repository | Status | Tokens | Full heartbeats | Body heartbeats | Imports/control | Build time | Files |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['repositories']:
        metrics = row['metrics']
        before, after = (metrics['measured_file_count'][v] for v in ('stripped', 'optimized'))
        files = f'{before} → {after}' if row['eligible'] else '—'
        cells = [row['source_repository'].split('/')[-1], row['status']]
        cells += [change(metrics[key]) for key in list(METRICS)[:-1]]
        cells.append(files)
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Full heartbeats include import loading, but exclude rebuilding Mathlib/dependency sources. Body counts subtract the matched imports-only control. These estimate synchronous Lean work, not elapsed compilation time. Build times are single observations on a shared server.', '',
              'Readability and refactoring quality are not established by token counts or heartbeats.', '',
              '## Absolute measurements', '',
              'Heartbeat values below are millions of raw heartbeats. Each cell is stripped → compressed.', '',
              '| Repository | Tokens | Full heartbeats (M) | Body heartbeats (M) | Imports/control (M) | Build seconds |',
              '|---|---:|---:|---:|---:|---:|']
    for row in report['repositories']:
        if not row['eligible']:
            continue
        cells = [row['source_repository'].split('/')[-1]]
        for key in list(METRICS)[:-1]:
            metric = row['metrics'][key]
            values = [metric[v] for v in ('stripped', 'optimized')]
            if key.endswith('heartbeats'):
                rendered = [f'{v / 1e6:.3f}' for v in values]
            elif key == 'clean_build_seconds':
                rendered = [f'{v:.3f}' for v in values]
            else:
                rendered = [f'{v:,}' for v in values]
            cells.append(' → '.join(rendered))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    output = ROOT / manifest['outputs']['directory']
    receipt = json.loads((output / 'manifest.json').read_text())
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if receipt['sha256'] != manifest_sha:
        raise ValueError('measurement receipt does not match manifest')
    # The runner replaces summary.json atomically. Read once for a coherent snapshot.
    summary_path = output / 'summary.json'
    summary_bytes = summary_path.read_bytes()
    report = build_report(json.loads(summary_bytes), manifest)
    report['provenance'] = {
        'manifest': {'path': str(args.manifest.resolve()), 'sha256': manifest_sha},
        'summary': {'path': str(summary_path), 'sha256': hashlib.sha256(summary_bytes).hexdigest()},
        'reporter': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    # Preserve the exact input snapshot as the measurement run continues updating.
    artifacts = {'all-metrics.input.json': summary_bytes.decode(),
                 'all-metrics.json': json.dumps(report, indent=2) + '\n',
                 'all-metrics.csv': csv_text(report), 'all-metrics.md': markdown_text(report)}
    for name, content in artifacts.items():
        path = output / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(path)
    print(markdown_text(report))
    print(f'Saved all-metrics.json, all-metrics.csv, and all-metrics.md under {output}')


if __name__ == '__main__':
    main()
