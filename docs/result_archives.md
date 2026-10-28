# Export and verify results

One evaluation's artifacts live under
`results/<dataset>/<provider>/<model>/`: repository outputs, pinned run manifests,
postprocessing reports, predictions and proxy traces. Packing writes a compressed
copy under the same path inside an archive directory, leaving the source alone.

## Pack a finished run

```bash
uv run python scripts/pack_results.py demo \
  --model openai/gpt-5.6-luna-xhigh --into ~/lean-lean-results
```

This resolves the named dataset/model configs and requires completed evaluation,
endpoint postprocessing and heartbeat postprocessing, plus a `result.json` for
every repository. Failed benchmark submissions are preserved too: completion of
postprocessing does not mean every submission passed.

To export an existing run without recomposing its generation configs, select its
pair directory directly:

```bash
uv run python scripts/pack_results.py \
  --pair results/demo/openai/gpt-5.6-luna-xhigh --into ~/lean-lean-results
```

The exporter validates the saved evaluation manifest against the hash in
`run/run.yaml`, and takes the repository list from that manifest. Add
`--endpoints-only` when endpoint postprocessing is complete but heartbeats were
not measured. It still requires completed evaluation, endpoint postprocessing and
all repository result records. An existing destination pair is never overwritten.

## Verify the archive

Packing verifies the stored copy before returning. To check it again later:

```bash
uv run python scripts/pack_results.py --verify ~/lean-lean-results --jobs 8
```

Each stored directory has a `.index.jsonl` describing original file names, sizes,
SHA-256 hashes and storage methods. Verification checks stored files and archive
members, reports missing content and validates referenced duplicate copies.
Streaming verification drains tar padding before waiting for decompression to
finish, using bounded reads.

Run manifests and postprocessing state remain readable files. Large text files
become `.zst` files. Run-level directories such as `proxy_traces/` become tar/zstd
archives. A capture's source checkpoints and incremental diffs are grouped into
`checkpoints.tar.zst`.

## Read stored content

Zstandard is required. All streams use a long window:

```bash
zstd -dc --long=31 trajectory.json.zst > trajectory.json
mkdir -p restored
zstd -dc --long=31 checkpoints.tar.zst | tar -x -C restored
```

Checkpoint `.sources.tar.gz` files are stored as their uncompressed inner tar
members, named `.sources.tar`. Their inner tar bytes are preserved and verified.
The original gzip envelope is not reconstructed. Consult each index's `how`,
`stored` and `member` fields when locating a file. These commands inspect or
extract the archive. They do not automatically resume an evaluation.
