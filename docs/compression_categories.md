# Compression categories

`scripts/analysis/classify_compression.py` partitions the scored source-token
change into five categories, in the order below. Earlier categories take
precedence. Costs of added code are included, so category totals reconcile with
the net token change. Failed submissions score zero.

1. **Syntax optimization:** savings from notation, macros, inlined definitions
   and supported spelling changes. Macro savings are net of declaration costs.
   Spelling changes must lie on a shortest token-edit path between the old and
   new hunk.
2. **Automation:** the remaining change when manual proof steps become automation
   such as `simp`, `grind` or `omega`, including goal routing.
3. **Dead code:** removed declarations unreachable from surviving or protected
   results through the original directed dependency graph. Sharing a helper with
   live code does not itself make a deleted caller live. The older component-based
   estimate remains available as `dead_code_component` for comparison.
4. **Structural change:** removed live declarations and the remaining change in
   retained declarations whose dependency sets changed. Syntax-abbreviation edges
   and losses already explained by automation do not trigger this category.
   New declarations other than syntax abbreviations contribute their token cost.
5. **Proof rewriting:** the remaining change in retained declarations whose
   dependency sets did not change.

The standalone endpoint classifier matches declarations by name, which does not
establish statement equivalence. Retained declarations are split into changed-line
hunks, whose token deltas partition the declaration's change. For identities
across file moves and physical copies, see [dependency tracking](dependency_tracking.md).

## Abbreviations and macros

`abbreviations.py` prices fixed templates as `tokens(expansion) - tokens(call)`,
counting each parameter once. This is exact for parameters used once. Repeated
parameters are flagged. Nested abbreviations include their own expansion savings.
Syntax without a fixed expansion is counted as machinery cost.

Definition uses require graph dependency evidence. Mentions inside unfolding hint
lists do not count as calls. Notation and template macros require a distinctive
literal absent from the baseline and must be visible at the call site. Supported
elaborator factories are handled by `generated_syntax.py`. Unsupported elaborators
and notation without distinctive keys remain unpriced.

`expand_macro_programs.py` measures programmatic macros with one native Lean
expansion step per call site. It registers syntax in a standalone file without
building the repository, removes hygiene marks before scoring, and leaves Lean's
own list and numeral notation intact. Elaborators with environment effects remain
uncovered. These measurements concern source-token expansion, not proof terms.

## Run

```bash
uv run python scripts/analysis/classify_compression.py \
  output/analysis/<run>/classification-inputs.csv \
  --output-dir output/analysis/<run>/compression-origin
```

The outputs are `classification-repositories.csv`, `diff_classification_macro.csv`
(equal-repository model averages), and `classification.json` with declaration
details. The defaults use `ledger.py` and `proof_hunks.step_automation`.
