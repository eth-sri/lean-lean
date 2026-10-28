"""Explain visible source edits that have no compression contribution.

This is display metadata only: never alter ledger categories or token totals.
"""
from leanlean.metrics.tokens import remove_lean_comments
from scripts.analysis.abbreviations import tokens
from scripts.analysis.quantify_compression_categories import SourceIndex, load_graph

VERSION = 1


def explain_source_edits(payload, before, after, before_source, after_source):
    """Distinguish presentation edits from real, zero-net-token code edits.

    Nonzero measured strategies always win. Missing source evidence remains
    missing; identical code is not evidence for a proof rewrite.
    """
    strategies = dict(payload.get('node_strategies', {}))
    files = [{}, {}]

    def text(node, source, cache):
        path = source.path_of_module(node['module'])
        if path not in source._files:
            return None
        if path not in cache:
            raw = source._files[path]
            # Strip comments before slicing, preserving block-comment context.
            cache[path] = (raw.splitlines(keepends=True),
                           remove_lean_comments(raw).splitlines(keepends=True))
        start, end = int(node['start_line']) - 1, int(node['end_line'])
        return tuple(''.join(lines[start:end]) for lines in cache[path])

    for name in before.nodes.keys() & after.nodes.keys():
        kind = strategies.get(name)
        if kind not in (None, 'modified'):
            continue
        old = text(before.nodes[name], before_source, files[0])
        new = text(after.nodes[name], after_source, files[1])
        if old is None or new is None:
            continue
        moved = before.nodes[name]['module'] != after.nodes[name]['module']
        if old[0] == new[0] and not moved and kind is None:
            continue
        if tokens(old[1]) == tokens(new[1]):
            strategies[name] = 'presentation'
        elif kind == 'modified':
            # The ledger classified an actual edit, but every net category is 0.
            strategies[name] = 'rewrite'
    return dict(payload, node_strategies=strategies, strategy_annotations_version=VERSION)


def annotate_pair(replay, capture, before, after, scope, payload):
    if payload.get('strategy_annotations_version') == VERSION:
        return payload
    graphs, sources = [], []
    for point in (before, after):
        graphs.append(load_graph(replay / 'states' / point['source_key'] / 'graph.json'))
        source = SourceIndex.from_archive(capture / point['snapshot'])
        source.restrict(exclude_files=scope['excluded_files'], exclude_dirs=scope['exclude_dirs'],
                        include_prefix=scope['include_prefix'])
        sources.append(source)
    return explain_source_edits(payload, *graphs, *sources)
