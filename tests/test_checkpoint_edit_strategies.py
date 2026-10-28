from types import SimpleNamespace

from scripts.analysis.checkpoint_edit_strategies import explain_source_edits
from scripts.analysis.quantify_compression_categories import SourceIndex


def test_source_edits_do_not_confuse_no_savings_with_missing_classification():
    old = {
        'format': 'def format :=  1\n',
        'rewrite': 'def rewrite := x\n',
        'comment': '/-- old comment -/\ndef comment := 1\n',
        'move': 'def move := 1\n',
        'pending': 'def pending := x\n',
        'measured': 'def measured :=  1\n',
        'missing': 'def missing := 1\n',
    }
    new = dict(old, format='def format := 1\n', rewrite='def rewrite := y\n',
               comment='/-- new comment -/\ndef comment := 1\n',
               pending='def pending := y\n', measured='def measured := 1\n')
    def graph(values, moved=False):
        return SimpleNamespace(nodes={name: dict(module='moved' if moved and name=='move' else name,
            start_line=1, end_line=len(text.splitlines())) for name,text in values.items()})
    before = SourceIndex({f'{n}.lean': text for n,text in old.items() if n!='missing'})
    after = SourceIndex({('moved' if n=='move' else n)+'.lean': text for n,text in new.items()})
    payload = dict(categories={'other': 0}, node_strategies={
        'format': 'modified', 'rewrite': 'modified', 'measured': 'automation'})
    result = explain_source_edits(payload, graph(old), graph(new, True), before, after)
    assert result['node_strategies'] == dict(format='presentation', rewrite='rewrite',
        comment='presentation', move='presentation', measured='automation')
    assert result['categories'] == payload['categories']
    assert payload['node_strategies']['format'] == 'modified'


def test_comment_masking_uses_whole_file_and_preserves_string_changes():
    graph = SimpleNamespace(nodes={
        'block': dict(module='A', start_line=2, end_line=4),
        'string': dict(module='A', start_line=5, end_line=5)})
    before = SourceIndex({'A.lean': '/- opening\nold\n-/\ndef block := 1\ndef string := "a b"\n'})
    after = SourceIndex({'A.lean': '/- opening\nnew\n-/\ndef block := 1\ndef string := "ab"\n'})
    result = explain_source_edits({'node_strategies': {'string': 'modified'}}, graph, graph, before, after)
    assert result['node_strategies'] == {'block': 'presentation', 'string': 'rewrite'}
