"""Regression cases for scored text omitted or counted twice by graph spans."""
from scripts.analysis.source_accounting import partition, with_source_only_declarations
from scripts.analysis.quantify_compression_categories import GraphData, SourceIndex


def graph(**nodes):
    return GraphData(nodes, {}, set(), set(), 'test', None)


def node(start, end, module='Main'):
    return dict(module=module, start_line=start, end_line=end, kind='theorem')


def test_nested_fields_and_aliases_only_charge_once():
    src = SourceIndex({'Main.lean': 'structure S where\n  value : Nat\n'})
    g = graph(S=node(1, 2), field=node(2, 2), alias=node(1, 2))
    result = partition(g, src, {'S'})
    assert result.tokens['S'] == src.total_tokens()
    assert result.tokens['field'] == result.tokens['alias'] == 0
    assert result.source_commands == 0
    assert result.overlap_tokens > 0


def test_added_shared_variables_charge_syntax_overhead():
    before = SourceIndex({'Main.lean': 'theorem t (n : Nat) : n = n := rfl\n'})
    after = SourceIndex({'Main.lean': 'variable (n : Nat)\ntheorem t : n = n := rfl\n'})
    b = partition(graph(t=node(1, 1)), before)
    a = partition(graph(t=node(2, 2)), after)
    declaration_savings = b.tokens['t'] - a.tokens['t']
    syntax_overhead = b.source_commands - a.source_commands
    assert syntax_overhead < 0
    assert declaration_savings + syntax_overhead == before.total_tokens() - after.total_tokens()


def test_comment_context_is_preserved_when_slicing():
    src = SourceIndex({'Main.lean': '/- comment\nstill comment -/\ntheorem t : True := by trivial\n'})
    result = partition(graph(t=node(2, 3)), src)
    assert result.tokens['t'] == src.total_tokens()
    assert 'still comment' not in result.texts['t']


def test_source_only_duplicate_is_separate_from_graph_theorem():
    src = SourceIndex({'Main.lean': 'theorem t : True := by trivial\n',
                       'Unused.lean': 'namespace N\ntheorem t : True := by trivial\nend N\n'})
    original = graph(t=node(1, 1))
    augmented = with_source_only_declarations(original, src)
    extra = [name for name in augmented.nodes if name not in original.nodes]
    assert len(extra) == 1
    assert augmented.nodes[extra[0]]['module'] == 'Unused'
    assert augmented.nodes[extra[0]]['end_line'] == 2
    result = partition(augmented, src)
    assert result.tokens[extra[0]] == result.tokens['t']
    assert result.source_commands > 0
    assert sum(result.tokens.values()) + result.source_commands == src.total_tokens()
    assert len(original.nodes) == 1


def test_complete_classifier_accounts_for_recall_and_variable_commands(tmp_path, monkeypatch):
    import io
    import tarfile
    from scripts.analysis import classify_compression as classifier

    sources = {
        'before': 'recall Nat.add_comm (n m : Nat) : n + m = m + n\ntheorem t (n : Nat) : n = n := rfl\n',
        'after': 'variable (n : Nat)\ntheorem t : n = n := rfl\n',
    }
    for phase, text in sources.items():
        (tmp_path / f'{phase}.json').touch()
        directory = tmp_path / phase
        directory.mkdir()
        with tarfile.open(directory / 'original-sources.tar.gz', 'w:gz') as archive:
            data = text.encode()
            member = tarfile.TarInfo('Main.lean')
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(classifier, 'load_graph', lambda path: graph(t=node(2, 2)))
    monkeypatch.setattr(classifier.program_sites, 'measure', lambda *args, **kwargs: {'programs': [], 'sites': [], 'uncovered': [], 'error': ''})
    baseline = SourceIndex({'Main.lean': sources['before']}).total_tokens()
    post = SourceIndex({'Main.lean': sources['after']}).total_tokens()
    record = {'metric_scope': {}, 'reconciliation': {'scored_baseline_tokens': baseline, 'scored_tokens_saved': baseline-post}}
    result = classifier.classify('repo', 'model', tmp_path, record, {}, set())
    assert sum(result[key] for key in classifier.CATEGORY_KEYS) == baseline-post
    assert result['detail']['source_command_savings'] > 0
    assert result['detail']['reconciliation_residual'] == 0
    record['reconciliation']['scored_tokens_saved'] += 1
    import pytest
    with pytest.raises(ValueError, match='scored source totals disagree'):
        classifier.classify('repo', 'model', tmp_path, record, {}, set())


def test_source_only_dotted_filename_keeps_exact_source_and_projection():
    src = SourceIndex({'scratch/Solution.orig.lean': 'theorem t : True := by trivial\n',
                       'scratch/Solution/orig.lean': 'theorem other : True := by trivial\n'})
    augmented = with_source_only_declarations(graph(), src)
    for name, item in augmented.nodes.items():
        path = name.split(':')[1]
        assert src.path_of_module(item['module']) == path
        assert src.slice(item) == src._files[path].strip()
