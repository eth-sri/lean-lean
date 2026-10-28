# October 2026 source update

This update extends the original public source release while retaining its paper
PDF and frozen paper result tables.

- Add pinned GPT-6.1 Sol effort variants, GPT-6 Luna, Haiku 5.5, Opus 5.5 effort
  variants, optional one-million-token Opus windows, and Gemini 4 Argon.
- Preserve native Codex requests through the subscription proxy and reconcile
  recorded usage against retained native rollouts. Harden model relay inactivity
  handling and Argon's native request, compaction and streaming retries.
- Share endpoint and checkpoint graph extraction, track declarations across
  source moves, retain distinct physical copies, and use directed baseline
  dependency evidence when accounting for deleted code.
- Keep evaluation, postprocessing and derived artifacts together under
  `results/<dataset>/<provider>/<model>/`. Add verified, compressed result
  archives, including packing by an existing pair's pinned manifest and an
  endpoint-only mode for runs without heartbeat measurements.

New multi-round compression, theorem reconstruction, Lean-to-paper and review
experiment workflows and findings are excluded from this source update. Run
outputs, generated workspaces, local dashboards and website exporters are also
excluded. Previously published ablation material remains available.

## Validation

The source suite runs without paid model requests:

```bash
uv run python -m pytest -q
```

Model contract tests validate every committed model config, prompt/catalog hash,
request effort and pricing snapshot. Adapter tests exercise local fixture servers.
Graph and accounting tests use synthetic source/graph cases. The demo integration
test requires a separately preprocessed dataset. Full benchmark evaluation and
Docker/Lean rebuilds are not part of the source-only test suite.
