"""Required results can be regenerated without named source graph nodes."""
from scripts.analysis.classify_compression import used_by_survivors, dead_components


def test_fully_replaced_protected_component_stays_live():
    deleted = {'main', 'lemma', 'helper', 'unrelated'}
    dependencies = {'main': {'lemma'}, 'lemma': {'helper'}}
    assert used_by_survivors({'main'}, deleted, dependencies) == {'main', 'lemma', 'helper'}
    assert dead_components({'main'}, deleted, {('main', 'lemma'), ('lemma', 'helper')}) == {'unrelated'}


def test_unused_branch_sharing_live_helper_stays_dead():
    deleted = {'lemma', 'helper', 'unused'}
    dependencies = {'main': {'lemma'}, 'lemma': {'helper'}, 'unused': {'helper'}}
    assert deleted - used_by_survivors({'main'}, deleted, dependencies) == {'unused'}


def test_surviving_nonprotected_user_keeps_its_old_dependencies_live():
    assert used_by_survivors({'retained'}, {'old'}, {'retained': {'old'}}) == {'old'}


def test_classifier_marks_elaborator_replaced_protected_result_deleted(tmp_path, monkeypatch):
    import io
    import tarfile
    from scripts.analysis import classify_compression as classifier
    from scripts.analysis.quantify_compression_categories import GraphData, SourceIndex

    source = {'before': 'theorem main : True := by trivial\n', 'after': 'emit_required_result\n'}
    for side, text in source.items():
        (tmp_path/f'{side}.json').touch()
        (tmp_path/side).mkdir()
        with tarfile.open(tmp_path/side/'original-sources.tar.gz', 'w:gz') as archive:
            member = tarfile.TarInfo('Main.lean'); member.size = len(text.encode())
            archive.addfile(member, io.BytesIO(text.encode()))
    before = GraphData({'main': dict(module='Main', start_line=1, end_line=1, kind='theorem')}, {}, set(), set(), 'test', None)
    after = GraphData({}, {}, set(), set(), 'test', None)
    monkeypatch.setattr(classifier, 'load_graph', lambda p: before if p.name=='before.json' else after)
    monkeypatch.setattr(classifier.program_sites, 'measure', lambda *a, **kw: dict(programs=[], sites=[], uncovered=[], error=''))
    baseline, post = (SourceIndex({'Main.lean': source[side]}).total_tokens() for side in ('before','after'))
    record = dict(protected_declarations=['main'], reconciliation=dict(scored_baseline_tokens=baseline, scored_tokens_saved=baseline-post))
    result = classifier.classify('repo', 'model', tmp_path, record, {}, set())
    assert result['dead_code'] == 0
    assert result['declarations']['main']['status'] == 'deleted'
    assert result['detail']['structural_removals'] == baseline
    assert sum(result[k] for k in classifier.CATEGORY_KEYS) == baseline-post
