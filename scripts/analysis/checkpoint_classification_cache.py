"""Bounded, process-local caches for repeated baseline proof comparisons.

Keys include declaration text, available lemma/definition names, automation
macros, and file binders. Results are copied because hunk consumers may mutate
lists. No classification rule changes, and no cache survives this context.
"""
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import pickle
import sys

from scripts.analysis import proof_hunks


@contextmanager
def proof_cache():
    analyse, step_analysis, variable_binders = proof_hunks.analyse, proof_hunks.step_analysis, proof_hunks.variable_binders
    files = {}
    # Comment masking and tokenization are pure string functions. Their cached
    # values are immutable; share the wrapper with modules importing an alias.
    from leanlean.metrics import tokens as token_module
    aliases=[]
    for name,size in [('remove_lean_comments',8192),('lean_refactor_lexer',65536),('_token_count',65536)]:
        original=getattr(token_module,name)
        cached=lru_cache(maxsize=size)(original)
        for module_name,module in list(sys.modules.items()):
            if module_name.startswith(('leanlean.','scripts.analysis.')) and getattr(module,name,None) is original:
                aliases.append((module,name,original))
                setattr(module,name,cached)

    @lru_cache(maxsize=128)
    def binders(text):
        return frozenset(variable_binders(text))

    @lru_cache(maxsize=2048)
    def analysed(old, new, bound, definitions, lemmas, macros):
        return analyse(old,new,after_file=files[bound],definitions=definitions,lemmas=lemmas)

    def analyse_cached(old,new,*,after_file='',definitions=frozenset(),lemmas=None):
        bound=binders(after_file)
        files[bound]=after_file
        return deepcopy(analysed(old,new,bound,frozenset(definitions),
            frozenset(lemmas) if lemmas is not None else None,frozenset(proof_hunks.EXTRA_AUTOMATION)))

    @lru_cache(maxsize=2048)
    def steps(old,new,found,definitions,lemmas,macros):
        return step_analysis(old,new,pickle.loads(found),definitions,lemmas)

    def steps_cached(old,new,found,definitions=frozenset(),lemmas=None):
        return deepcopy(steps(old,new,pickle.dumps(found),frozenset(definitions),
            frozenset(lemmas) if lemmas is not None else None,frozenset(proof_hunks.EXTRA_AUTOMATION)))

    proof_hunks.analyse=analyse_cached
    proof_hunks.step_analysis=steps_cached
    proof_hunks.variable_binders=lambda text:set(binders(text))
    try:
        yield
    finally:
        proof_hunks.analyse,proof_hunks.step_analysis,proof_hunks.variable_binders=analyse,step_analysis,variable_binders
        for module,name,original in reversed(aliases):
            setattr(module,name,original)
