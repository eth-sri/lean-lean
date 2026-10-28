"""Checkpoint identity and cached analysis work independently of dashboard exports."""
from dataclasses import asdict

import pytest

from scripts.analysis import checkpoint_identity as identity
from scripts.analysis.extract_checkpoint_graphs import source_key
from scripts.analysis.quantify_compression_categories import GraphData, SourceIndex



def graph(nodes,edges=()):
    return GraphData(nodes={n['name']:n for n in nodes},layers={},static_edges=set(edges),grind_edges=set(),schema='test',sha256=None)


def test_production_identity_survives_private_file_move_and_directed_users():
    tracker=identity.Identities()
    def node(module):return dict(name=f'_private.{module}.0.helper',module=module,kind='theorem',start_line=1,end_line=1)
    before,_,_=tracker.align(graph([node('A')]),SourceIndex({'A.lean':'private theorem helper : True := by trivial\n'}))
    after,_,_=tracker.align(graph([node('B')]),SourceIndex({'B.lean':'private theorem helper : True := by trivial\n'}))
    assert before.nodes.keys()==after.nodes.keys()=={'helper'}
    original=graph([dict(name=n) for n in ['main','caller','helper']], [('caller','helper')])
    assert 'helper' not in identity.directed_removal_group(original,graph([dict(name=n) for n in ['main','caller']]))
    assert identity.directed_removal_group(original,graph([dict(name='main')]))=={'caller','helper'}


def test_proof_cache_preserves_results_and_accounts_for_context():
    from scripts.analysis import proof_hunks as ph
    from scripts.analysis.checkpoint_classification_cache import proof_cache
    old='theorem foo (n : Nat) : n = n := by\n  exact rfl\n'
    new='theorem foo (n : Nat) : n = n := by\n  simp\n'
    expected=ph.analyse(old,new,after_file='variable (n : Nat)',definitions={'n'},lemmas={'foo'})
    expected_steps=ph.step_analysis(old,new,expected,{'n'},{'foo'})
    original=ph.analyse
    with proof_cache():
        got=ph.analyse(old,new,after_file='variable (n : Nat)',definitions={'n'},lemmas={'foo'})
        assert [asdict(h) for h in got]==[asdict(h) for h in expected]
        assert asdict(ph.step_analysis(old,new,got,{'n'},{'foo'}))==asdict(expected_steps)
        got.clear()
        assert ph.analyse(old,new,after_file='variable (n : Nat)',definitions={'n'},lemmas={'foo'})
        # Changed file binders must not reuse a previous analysis.
        changed=ph.analyse(old,new,after_file='variable (m : Nat)',definitions={'n'},lemmas={'foo'})
    assert ph.analyse is original
    assert [asdict(h) for h in changed]==[asdict(h) for h in original(old,new,after_file='variable (m : Nat)',definitions={'n'},lemmas={'foo'})]


def test_initially_live_declaration_never_becomes_dead_after_its_users_change():
    original=graph([dict(name=n) for n in ['main','caller','helper']], [('main','caller'),('caller','helper')])
    dead=identity.baseline_dead_nodes(original,['main'])
    after=graph([dict(name='main')])  # main now proves the result independently.
    assert dead==set()
    removed=identity.directed_removal_group(original,after)
    assert identity.removal_status('caller',dead,removed)=='deleted'
    assert identity.removal_status('helper',dead,removed)=='deleted'


def test_original_users_block_deadness_even_when_their_dependency_disappears():
    original=graph([dict(name=n) for n in ['main','caller','helper']], [('caller','helper')])
    dead=identity.baseline_dead_nodes(original,['main'])
    assert dead=={'caller','helper'}
    changed_caller=graph([dict(name=n) for n in ['main','caller']])
    assert identity.removal_status('helper',dead,identity.directed_removal_group(original,changed_caller))=='deleted'
    whole_group_removed=graph([dict(name='main')])
    assert identity.removal_status('helper',dead,identity.directed_removal_group(original,whole_group_removed))=='dead'
    # A remaining dependency does not constitute a remaining user: direction matters.
    helper_remains=graph([dict(name=n) for n in ['main','helper']])
    assert identity.removal_status('caller',dead,identity.directed_removal_group(original,helper_remains))=='dead'


def test_missing_original_roots_and_unindexed_sources_cannot_be_called_dead():
    import pytest
    original=graph([dict(name='main'),dict(name='unindexed',source_only=True)])
    assert identity.baseline_dead_nodes(original,['main'])==set()
    for roots in [[],['missing']]:
        with pytest.raises(ValueError,match='Original protected'):
            identity.baseline_dead_nodes(original,roots)


def test_final_checkpoint_collision_requires_exact_source_identity():
    from scripts.analysis.checkpoint_points import match_checkpoint
    edited=dict(edit_index=31,elapsed_seconds=20,file_sha256={'A.lean':'old'},
                snapshot_path='edit.tar.gz',snapshot_artifact_sha256='old-archive')
    repeated=dict(edited,elapsed_seconds=22)
    submitted=dict(repeated,file_sha256={'A.lean':'new'},snapshot_path='submitted.tar.gz',snapshot_artifact_sha256='new-archive')
    points=[edited,repeated,submitted]
    assert match_checkpoint(points,dict(edit=31,seconds=20))==edited
    assert match_checkpoint(points,dict(edit=31,seconds=22,source_key=source_key(submitted)))==submitted
    assert match_checkpoint(points,dict(edit=31,seconds=22,snapshot='edit.tar.gz'))==repeated
    with pytest.raises(ValueError,match='Ambiguous'):
        match_checkpoint(points,dict(edit=31,seconds=22))
    with pytest.raises(ValueError,match='No captured'):
        match_checkpoint(points,dict(edit=31,seconds=20,source_key=source_key(submitted)))
