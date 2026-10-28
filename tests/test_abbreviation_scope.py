"""Nested syntax pricing must not expand unrelated, identically spelled symbols."""
from scripts.analysis.abbreviations import build
from scripts.analysis.quantify_compression_categories import GraphData, SourceIndex


def price(files, declarations, edges=()):
    nodes = {name: dict(module=module, start_line=line, end_line=line)
             for name, module, line in declarations}
    before = GraphData({}, {}, set(), set(), 'test', None)
    after = GraphData(nodes, {}, set(edges), set(), 'test', None)
    found, _ = build(before, after, SourceIndex({}), SourceIndex(files), {})
    return {a.names[0]: a for a in found}


def test_nested_local_symbols_do_not_leak_between_files_or_forward():
    files = {'Main.lean': 'local notation "ⓐ" => (1 + 2)\nlocal notation "ⓑ" => (ⓐ + 3)\nlocal notation "ⓐ" => (1 + 2 + 3 + 4)\n',
             'Other.lean': 'local notation "ⓐ" => (1 + 2 + 3 + 4 + 5)\n'}
    result = price(files, [('a', 'Main', 1), ('b', 'Main', 2), ('later', 'Main', 3), ('unrelated', 'Other', 1)])
    assert result['b'].saving == result['b'].base_saving + result['a'].base_saving


def test_nested_import_respects_local_visibility():
    files = {'Main.lean': 'import Other\nlocal notation "ⓑ" => (ⓐ + ⓒ)\n',
             'Other.lean': 'notation "ⓐ" => (1 + 2)\nlocal notation "ⓒ" => (3 + 4 + 5)\n'}
    result = price(files, [('b', 'Main', 2), ('a', 'Other', 1), ('c', 'Other', 2)])
    assert result['b'].saving == result['b'].base_saving + result['a'].base_saving


def test_nested_definition_requires_graph_dependency():
    files = {'Main.lean': 'def M := 1 + 2 + 3\nlocal notation "ⓑ" => (M + 4)\n'}
    declarations = [('Unrelated.M', 'Main', 1), ('b', 'Main', 2)]
    unrelated = price(files, declarations)
    assert unrelated['b'].saving == unrelated['b'].base_saving
    related = price(files, declarations, [('b', 'Unrelated.M')])
    assert related['b'].saving == related['b'].base_saving + related['Unrelated.M'].base_saving
