# Dependency graphs across edits

Endpoint and checkpoint extraction use the same frozen Lean extractor and three
independent evidence layers: kernel dependencies, `.ilean` references and source
references. An edge can belong to several layers. The combined graph is their
set union. Counts and extraction provenance accompany the graph.

## Extract checkpoints

After evaluation has recorded a playback capture, reuse the endpoint graph
manifest with that capture:

```bash
uv run python scripts/analysis/extract_checkpoint_graphs.py \
  experiments/analysis/demo-luna-paper-graphs.yaml \
  results/demo/openai/gpt-5.6-luna-xhigh/<repository>/playback/capture_001 \
  --output output/analysis/demo-checkpoints/<repository> --workers 2
```

The replay keeps its builds and graphs separate from the original capture.
Each source state has a key derived from all recorded file hashes, including
build configuration. Identical source trees share an extraction result. Failed
builds have no graph. Timeouts and extraction failures remain retryable.
Completed graphs require the current extraction revision before reuse.
Incremental extraction excludes stale compiled modules whose sources were
deleted, and resolves source roots consistently with endpoint extraction.

`timeline.json` records each selected point's source key, status and graph hash.
`states/<source-key>/graph.json` contains a successful extraction. Use `--samples`
for approximately uniform elapsed-time sampling or `--indices` for explicit edit
indices. Omit both to extract every recorded point. `batch_checkpoint_graphs.py`
adds archive verification, per-repository locks and bounded scheduling.

## Align declarations between graphs

`scripts.analysis.checkpoint_identity.Identities` is a reusable Python interface.
Use one tracker for a sequence of snapshots in chronological order:

```python
from scripts.analysis.checkpoint_identity import Identities
from scripts.analysis.quantify_compression_categories import SourceIndex, load_graph

tracker = Identities()
before, before_names, before_ambiguous = tracker.align(
    load_graph(before_graph), SourceIndex.from_archive(before_sources)
)
after, after_names, after_ambiguous = tracker.align(
    load_graph(after_graph), SourceIndex.from_archive(after_sources)
)
```

Here the graph and source arguments are paths. Restrict both source indexes to
the recorded metric scope before comparing scored token totals.

Qualified source names connect declarations across snapshots. Compiler-generated
private prefixes are normalized so a file move does not automatically become a
deletion and addition. Physical copies keep separate IDs and token costs.
Ambiguous same-name compiled declarations retain their raw names and are reported
as ambiguous. Source-only declarations remain visible without claiming that they
were compiled or kernel-verified. This is conservative source identity tracking,
not semantic equivalence or arbitrary rename detection.

## Account for deletions

Edges point from a declaration to its dependencies. Original deadness is computed
from the original protected roots, which must exist in the graph. A missing root
is an error. Removing a previously used declaration does not retroactively make
it originally dead.

`directed_removal_group` follows the original dependencies from surviving
original declarations. A shared helper does not make its deleted caller live.
A retained caller does keep its original dependency chain live. The identity
ledger preserves separate contributions from added copies and checks that
category totals equal the scored token change.

Source identities and dependency layers are evidence about recorded code. They
do not replace the Comparator's independent checks on the protected results.
