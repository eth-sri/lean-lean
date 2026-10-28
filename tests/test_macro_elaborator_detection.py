from types import SimpleNamespace
from unittest.mock import patch

from scripts.analysis import expand_macro_programs as macros


def collect(extra):
    declarations = [('Solution', 'syntax "grlab_declarations " ident+ : command'), *extra]
    nodes = {str(i): {'module': module, 'start_line': i + 1, 'end_line': i + 1, 'source': source}
             for i, (module, source) in enumerate(declarations)}
    sources = SimpleNamespace(slice=lambda node: node['source'], path_of_module=lambda module: module + '.lean')
    with patch.object(macros, 'written_declaration_groups', return_value=[(key, []) for key in nodes]):
        return macros.collect_programs(sources, SimpleNamespace(nodes=nodes), set(nodes))


ELAB = 'elab_rules : command\n  | `(grlab_declarations $ids:ident*) => do pure ()'
MACRO = 'macro_rules\n  | `(grlab_declarations $ids:ident*) => do pure default'


def test_elaborator_only_syntax_is_explicitly_uncovered():
    programs, uncovered, _ = collect([('Solution', ELAB)])
    assert not programs
    assert any(row['reason'] == 'elab_rules: not a macro expansion' for row in uncovered)


def test_syntax_with_macro_rule_still_requires_measurement():
    programs, _, _ = collect([('Solution', ELAB), ('Solution', MACRO)])
    assert [program.lead for program in programs] == ['grlab_declarations']


def test_unrelated_elaborator_does_not_suppress_macro_measurement():
    programs, _, _ = collect([('Other', ELAB)])
    assert [program.lead for program in programs] == ['grlab_declarations']
