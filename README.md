# LeanLean: Benchmarking Repository-Scale Lean Proof Compression

Can coding agents make a verified Lean library smaller? LeanLean gives an agent 12 hours on each of 64 real-world Lean repositories to compress the Lean codebase as much as it can while preserving the protected theorems.

The pipeline has three stages, each driven by a committed YAML config:

1. **Preprocessing** fetches the repositories from the [Palomar registry](https://data.palomar-registry.org) and strips them with [lean-strip](https://github.com/eth-sri/lean-strip).
2. **Evaluation** runs an agent on every repository in an isolated container.
3. **Postprocessing** verifies each submission with the Palomar Comparator and computes the paper's metrics.

Long stages run in a named tmux session, and repeating a command resumes it.

## Setup

### Requirements

Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker and tmux.

### Install

```bash
uv sync                                             # Python environment, including lean-strip v1.0.0
uv run python scripts/build_comparator_bundle.py    # pinned Palomar Comparator tools, built from source
```

### Credentials

Put the keys for the models you run in `secret.sh` (ignored by Git). The launch scripts source it.

| Harness | Variable |
| --- | --- |
| Codex | `OPENAI_SUBSCRIPTION_KEY` |
| Claude Code | `CLAUDE_CODE_OAUTH_TOKEN` |
| Antigravity (Gemini) | `GEMINI_API_KEY` |
| Muse Code | `META_API_KEY` |
| Mistral Vibe (Leanstral) | `MISTRAL_API_KEY` |

### Agent harnesses

Each model config pins its harness version, and a launch refuses any other. Install them with `scripts/install_codex_standalone.sh <version>`, `scripts/install_claude_standalone.sh <version>`, `scripts/install_antigravity_standalone.sh <version>`, `scripts/install_muse_standalone.py` or `scripts/install_mistral_vibe_standalone.sh`.

## Quick start

The whole pipeline on four small repositories:

```bash
bash preprocess.sh configs/preprocessing/palomar-compact4.yaml    # -> datasets/palomar-compact4
bash eval.sh demo --model openai/gpt-5.6-luna-xhigh
bash postprocess.sh demo --model openai/gpt-5.6-luna-xhigh
```

Run each command after the previous one has finished. The analyses in [Postprocessing](#3-postprocessing) apply the same way.

## 1. Preprocessing

### Build the benchmark

```bash
bash preprocess.sh configs/preprocessing/leanlean_20260914.yaml    # -> datasets/leanlean_20260914
```

Downloads each repository at its pinned commit, builds it, runs lean-strip offline, and verifies the result with the Comparator. `scripts/verify_lean_strip_reproduction.py` checks reproduction against the published benchmark.

### Or download it

To skip preprocessing, download the published benchmark from Hugging Face ([`eth-sri/lean-lean`](https://huggingface.co/datasets/eth-sri/lean-lean)). Then pass the dataset config `leanlean_20260914-hf` instead of `leanlean_20260914` to `eval.sh` and `postprocess.sh`. Its results go to `results/leanlean_20260914-hf/`. The release has no images, so each repository's image is built once before its agent starts.

```bash
uv run python scripts/fetch_dataset.py    # -> datasets/leanlean_20260914
```

## 2. Evaluation

```bash
bash eval.sh leanlean_20260914 --model openai/gpt-5.6-sol-xhigh
```

Every run has 12 hours, 8 CPUs, 64 GiB and a network restricted to a model proxy, and a snapshot is taken after every edit. `configs/dataset/leanlean_20260914.yaml` holds the task prompt. Results go to `results/leanlean_20260914/<provider>/<model>/`.

### Models

| Config (`--model`) | Harness |
| --- | --- |
| `openai/gpt-5.6-sol-xhigh` | Codex 0.154.0 |
| `openai/gpt-5.6-luna-xhigh` | Codex 0.154.0 |
| `openai/gpt-6.1-sol-xhigh` | Codex 0.160.0 |
| `openai/gpt-6.1-sol-max` | Codex 0.160.0 |
| `openai/gpt-6-luna-max` | Codex 0.160.0 |
| `openai/gpt-6-astra-xhigh` | Codex 0.154.0 |
| `anthropic/opus-5-high` | Claude Code 2.1.269 |
| `anthropic/opus-5.5-high` | Claude Code 2.1.280 |
| `anthropic/haiku-5.5-high` | Claude Code 2.1.294 |
| `anthropic/haiku-5.5-xhigh` | Claude Code 2.1.294 |
| `anthropic/fable-5.1-high` | Claude Code 2.1.269 |
| `google/gemini-3.8-flash-high` | Antigravity 1.1.26 |
| `google/gemini-4-argon-high` | Antigravity 1.4.0 |
| `meta/muse-spark-1.3-max-native-w4` | Muse Code 1.0.3 |
| `mistral/leanstral-1.5` | Mistral Vibe 2.25.0 |

Additional effort variants and one-million-token Claude configs are in `configs/models/`.

## 3. Postprocessing

The commands below use Sol. Substitute any evaluated model.

### Compression score and heartbeats

Each submission is rebuilt and checked with the Comparator, its build is kept, and heartbeats are measured. The score is the reduction in Lean tokens. Failed submissions score 0.

```bash
bash postprocess.sh leanlean_20260914 --model openai/gpt-5.6-sol-xhigh
```

### Budget curves

The last edit of every five-minute window is built incrementally.

```bash
uv run python scripts/select_checkpoint_windows.py results/leanlean_20260914/openai/gpt-5.6-sol-xhigh \
  --window-minutes 5 --output runs/checkpoints/sol-5min.json
bash postprocess.sh leanlean_20260914 --model openai/gpt-5.6-sol-xhigh --replay --skip-checkpoint-lean-verify \
  --checkpoint-index-file runs/checkpoints/sol-5min.json
```

### Compression categories

Dependency graphs of each repository before and after compression, then every saving split into syntax, automation, dead code, structural and rewriting. `experiments/analysis/demo-luna-paper-graphs.yaml` is an example manifest.

```bash
uv run python scripts/analysis/extract_paper_graphs.py experiments/analysis/<run>.yaml
uv run python scripts/analysis/classify_compression.py output/analysis/<run>/classification-inputs.csv \
  --output-dir output/analysis/<run>/compression-origin
```

See [checkpoint tracking](docs/dependency_tracking.md) and
[compression attribution](docs/compression_categories.md) for details.

### Agent actions

Every tool call is classified as Build, Verify, Measure, Read, Search, Edit, Git, Lake, Sleep or Other.

```bash
uv run python scripts/analysis/plot_action_classes.py results/leanlean_20260914/openai/gpt-5.6-sol-xhigh \
  --labels Sol --dataset datasets/leanlean_20260914 --output-dir output/analysis/agent-actions
```

### Pack results

Compress a finished run into an archive, verified file by file:

```bash
uv run python scripts/pack_results.py --pair results/demo/openai/gpt-5.6-luna-xhigh --into ~/lean-lean-results
uv run python scripts/pack_results.py --verify ~/lean-lean-results
```

See [result archives](docs/result_archives.md) for options and extraction.

## 4. Ablations

### Opus/Sol merge

GPT-5.6 Luna merges the Opus 5 and GPT-5.6 Sol submissions on twelve repositories. This ablation ran on the paper's internal dataset bundle (with its prebuilt shared environments) and the paper's Opus 5 and Sol runs. Its configs (`configs/dataset/leanlean_2026092*-langlib-*`, `configs/preprocessing/leanlean_2026092*-*-seeding.yaml`) and scripts (`scripts/assemble_reconciliation_dataset.py`, `scripts/seed_reconciliation_branches.py`) are included for reference.

### Prompt ablation

`configs/dataset/leanlean_20260914_mini-prompt-ablation-compress-4band-r2.yaml` and the two `leanlean_20260914_mini-reduce-size-{with,without}-anticheat-20260922-r1.yaml` configs vary the task prompt on four repositories.

## 5. Paper results

`results/paper/data/*.csv` holds the paper's final numbers. To compute them from your own runs, list the run directories in `configs/paper/leanlean_20260914.yaml` and export:

```bash
uv run python scripts/paper/export.py configs/paper/leanlean_20260914.yaml --output-dir output/paper-export
```

`scripts/paper/render.py` draws the paper's figures and tables from either set of CSVs.

## Citation

```bibtex
@misc{leanlean2026,
  title  = {{LeanLean}: Benchmarking Repository-Scale {Lean} Proof Compression},
  author = {R{\"o}gnvaldsson, K{\'a}ri and M{\"u}ndler-Sasahara, Niels and Dekoninck, Jasper and Vechev, Martin},
  year   = {2026},
  note   = {Preprint},
  url    = {https://files.sri.inf.ethz.ch/website/papers/rognvaldsson2026leanlean.pdf}
}
```

## License

MIT, see [LICENSE](LICENSE).
