"""Exercise sync failures and publication using tiny, self-contained inputs."""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tarfile
import unittest
from unittest.mock import patch

import sync_results as sync
import export_traces as traces


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lean, self.archive, self.data, self.cache = [self.root / name for name in ("lean", "archive", "data", "cache")]
        self.cache.mkdir()
        self.spec = dict(key="test-model", short="test", label="Test", provider="Google", logo="gemini", color="#123456",
                         source="dataset/google/test-high", run="results/dataset/google/test-high", updated_at="2026-10-26", config=dict(harness="Antigravity CLI"))
        self.bench = dict(models=[], repositories=[dict(id="repo", tokens=100, band="small")],
                          results=dict(repo={}), bands=[dict(band="small")], scale={}, run=dict(shared=dict(cpus=8, memory="64g", timeout="12h")), prompt="Canonical prompt")
        self.manifest = self.root / "models.json"
        self.write(self.manifest, dict(schema_version=1, models=[self.spec]))
        fragments = [dict(source="instructions", value="Native instructions")]
        digest = hashlib.sha256(json.dumps(fragments, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        prompt = dict(fragments=fragments, native_instructions_sha256=digest, harness_version="1.4.0")
        self.write(self.lean / "prompts/test.json", prompt)
        self.run = self.lean / self.spec["run"]
        resolved = dict(model="test-model", reasoning_effort="high", harness_version="1.4.0", agent=dict(prompt=self.bench["prompt"]),
                        container=self.bench["run"]["shared"], generation_parameters=dict(request=dict(maxOutputTokens=65536, thinkingConfig=dict(thinkingLevel="HIGH"))))
        self.write(self.run / "run/config.snapshot.yaml", dict(resolved=resolved, pricing=dict(rates=dict(input=2, output=10)),
            system_prompt=dict(mode="native", snapshot="prompts/test.json", sha256=digest)))
        self.row = dict(repository="repo", model="test-model", status="passed", reasoning_effort="high", cost_usd=2.5, agent_seconds=30,
            metrics=dict(baseline_lean_tokens=100, post_lean_tokens=50, lean_tokens_saved=50), files=dict(postprocessed_playback="playback/capture_001/postprocessed-playback.json"))
        self.result_path = self.archive / self.spec["source"] / "repo/result.json"
        self.write(self.result_path, self.row)
        self.write(self.run / "repo/result.json", self.row)
        self.capture = self.run / "repo/playback/capture_001"
        points = []
        for index, (kind, count, elapsed) in enumerate((("baseline", 100, 0), ("edit", 75, 10), ("final", 50, 30))):
            snapshot = f"playback/capture_001/checkpoint_{index}.sources.tar.gz"
            target = self.run / "repo" / snapshot
            target.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target, "w:gz") as archive:
                source = b"x " * count
                member = tarfile.TarInfo("Main.lean")
                member.size = len(source)
                archive.addfile(member, io.BytesIO(source))
            points.append(dict(kind=kind, lean_tokens=count if kind != "edit" else None, snapshot_path=snapshot, elapsed_seconds=elapsed))
        points[1]["incremental_diff_path"] = "playback/capture_001/edit.diff"
        (self.capture / "edit.diff").write_text("--- a/Main.lean\n+++ b/Main.lean\n@@ -1 +1 @@\n-old\n+new\n")
        self.write(self.capture / "postprocessed-playback.json", dict(baseline_lean_tokens=100, post_lean_tokens=50,
            final_cost_usd=2.5, points=points))
        self.write(self.capture / "standardized-trace.json", dict(provider="antigravity-cli", actions=[], messages=[]))
        (self.capture / "native.stdout.jsonl").write_text(json.dumps(dict(event="init", init=dict(tools=["run_command"]))) + "\n" +
            json.dumps(dict(step_update=dict(state="DONE", step_index=1, step_type="agent_response", text_delta="Done"))) + "\n")
        self.write(self.data / "benchmark.json", self.bench)
        self.write(self.data / "leaderboard.json", dict(models=[], results=dict(repo={})))
        self.write(self.data / "system-prompts.json", {})
        self.write(self.data / "traces/index.json", {})
        source = self.data / "repos/repo/source.json.gz"
        source.parent.mkdir(parents=True)
        source.write_bytes(gzip.compress(b'{"Main.lean":[0,1]}'))
        classifier = self.lean / "scripts/analysis/plot_action_classes.py"
        classifier.parent.mkdir(parents=True)
        classifier.write_text('def classify(name, args, files): return ("Other", None)\n')
        metric = self.lean / "src/leanlean/metrics/tokens.py"
        metric.parent.mkdir(parents=True)
        metric.write_text("def count_lean_tokens_in_source(source): return len(source.split())\n")
        self.args = argparse.Namespace(manifest=self.manifest, models=None, data=self.data, cache=self.cache,
            leanlean_root=self.lean, archive_root=self.archive, force=False, dry_run=False, check=False, workers=1)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sync.encoded(value))

    def test_sync_is_repeatable_and_repairs_missing_output(self):
        self.assertEqual(sync.sync(self.args), 0)
        installed = {str(p.relative_to(self.data)): p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        bench = sync.read(self.data / "benchmark.json")
        self.assertEqual(bench["models"][0]["score"], 50)
        self.assertEqual(bench["models"][0]["cost"], 2.5)
        self.assertFalse(bench["results"]["repo"]["test-model"]["explorer"])
        head = json.loads(gzip.decompress((self.data / "traces/repo/test.json.gz").read_bytes()))
        self.assertEqual([point[1] for point in head["progress"]], [100, 75, 50])
        self.args.check = True
        self.assertEqual(sync.sync(self.args), 0)
        self.args.check = False
        with patch.object(sync, "export_trace", side_effect=AssertionError("Unchanged traces should be skipped")):
            self.assertEqual(sync.sync(self.args), 0)
        self.assertEqual(installed, {str(p.relative_to(self.data)): p.read_bytes() for p in self.data.rglob("*") if p.is_file()})
        (self.data / "traces/repo/test.bin").unlink()
        self.args.check = True
        self.assertEqual(sync.sync(self.args), 1)

    def test_preflight_failure_does_not_change_public_data(self):
        before = (self.data / "benchmark.json").read_bytes()
        self.result_path.unlink()
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            sync.sync(self.args)
        self.assertEqual((self.data / "benchmark.json").read_bytes(), before)

    def test_baseline_and_archive_mismatch_are_rejected(self):
        row = copy.deepcopy(self.row)
        row["metrics"]["baseline_lean_tokens"] = 101
        self.write(self.result_path, row)
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            sync.prepare(self.spec, self.bench, self.lean, self.archive)
        row = copy.deepcopy(self.row)
        row["cost_usd"] = 3
        self.write(self.result_path, row)
        with self.assertRaisesRegex(ValueError, "packed and unpacked"):
            sync.prepare(self.spec, self.bench, self.lean, self.archive)

    def test_prompt_hash_is_verified(self):
        path = self.lean / "prompts/test.json"
        snapshot = sync.read(path)
        snapshot["fragments"][0]["value"] = "Changed instructions"
        self.write(path, snapshot)
        with self.assertRaisesRegex(ValueError, "prompt hash mismatch"):
            sync.prepare(self.spec, self.bench, self.lean, self.archive)

    def test_failed_runs_score_zero(self):
        row = dict(self.row, status="failed")
        self.write(self.result_path, row)
        self.write(self.run / "repo/result.json", row)
        item = sync.prepare(self.spec, self.bench, self.lean, self.archive)
        self.assertEqual(item["model"]["score"], 0)
        self.assertEqual(item["model"]["passed"], 0)

    def test_export_failure_leaves_public_data_untouched(self):
        before = {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        (self.capture / "native.stdout.jsonl").write_text(json.dumps(dict(event="init", init=dict(tools=["run_command"]))) + "\n")
        with self.assertRaisesRegex(ValueError, "Trace export failed"):
            sync.sync(self.args)
        self.assertEqual(before, {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()})

    def test_publish_rolls_back_after_install_error(self):
        stage = self.root / "stage"
        self.write(stage / "a.json", dict(new=True))
        self.write(stage / "b.json", dict(new=True))
        self.write(self.data / "a.json", dict(old=True))
        original = (self.data / "a.json").read_bytes()
        replace = sync.os.replace

        def fail_second(source, target):
            if Path(source) == stage / "b.json":
                raise OSError("Disk failure")
            return replace(source, target)

        with patch.object(sync.os, "replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "Disk failure"):
                sync.publish(stage, self.data)
        self.assertEqual((self.data / "a.json").read_bytes(), original)
        self.assertFalse((self.data / "b.json").exists())

    def test_paths_cannot_escape_input_root(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            sync.inside(self.lean, "../elsewhere")

    def test_streamed_replies_keep_every_delta(self):
        updates = [
            dict(state="ACTIVE", step_index=1, step_type="agent_response", text_delta="Waiting "),
            dict(state="ACTIVE", step_index=1, step_type="agent_response", text_delta="for the build "),
            dict(state="DONE", step_index=1, step_type="agent_response", text_delta="to finish.\n"),
            dict(state="ACTIVE", step_index=2, step_type="agent_response", text_delta="Verification complete."),
            dict(state="DONE", step_index=2, step_type="agent_response"),
        ]
        (self.capture / "native.stdout.jsonl").write_text("\n".join(json.dumps(dict(step_update=u)) for u in updates))
        steps = traces.antigravity_steps(dict(actions=[]), self.capture)
        self.assertEqual([s["x"] for s in steps], ["Waiting for the build to finish.", "Verification complete."])

    def test_snapshot_token_mismatch_aborts_sync(self):
        path = self.capture / "postprocessed-playback.json"
        playback = sync.read(path)
        playback["points"][0]["lean_tokens"] = 101
        self.write(path, playback)
        with self.assertRaisesRegex(ValueError, "Checkpoint token mismatch"):
            sync.sync(self.args)
        self.assertEqual(sync.read(self.data / "benchmark.json"), self.bench)

    @unittest.skipUnless(shutil.which("zstd"), "zstd is needed for packed JSON")
    def test_large_packed_heartbeat_json_is_included(self):
        self.bench["repositories"][0]["heartbeats"] = 100
        heartbeat = self.result_path.parent / "derived/heartbeats.json.zst"
        heartbeat.parent.mkdir(parents=True)
        payload = sync.encoded(dict(repository="repo", status="complete", lean_tokens=50, body_heartbeats=50000))
        heartbeat.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=payload, capture_output=True, check=True).stdout)
        item = sync.prepare(self.spec, self.bench, self.lean, self.archive)
        self.assertEqual(item["cells"]["repo"]["heartbeats_after"], 50)
        self.assertEqual(item["model"]["heartbeats_scored"], 1)
        self.assertEqual(item["model"]["heartbeats"], 50)
        self.assertIn(heartbeat, item["inputs"])

    def test_heartbeat_for_another_endpoint_is_rejected(self):
        self.bench["repositories"][0]["heartbeats"] = 100
        heartbeat = self.result_path.parent / "derived/heartbeats.json"
        self.write(heartbeat, dict(repository="repo", status="complete", lean_tokens=60, body_heartbeats=50000))
        with self.assertRaisesRegex(ValueError, "heartbeat endpoint token mismatch"):
            sync.prepare(self.spec, self.bench, self.lean, self.archive)

    def test_existing_model_heartbeats_require_frozen_result_hash(self):
        bench = copy.deepcopy(self.bench)
        bench["models"] = [dict(key="test-model", score=50)]
        bench["repositories"][0]["heartbeats"] = 100
        bench["results"]["repo"]["test-model"] = dict(status="passed", post=50)
        bench["result_sources"] = {"test-model": dict(source="results/" + self.spec["source"], result_sha256=dict(repo="outdated"))}
        heartbeat = self.result_path.parent / "derived/heartbeats.json"
        self.write(heartbeat, dict(repository="repo", status="complete", lean_tokens=50, body_heartbeats=50000))
        self.assertEqual(sync.refresh_heartbeats(bench, self.archive), [])
        self.assertNotIn("heartbeats_after", bench["results"]["repo"]["test-model"])
        bench["result_sources"]["test-model"]["result_sha256"]["repo"] = sync.sha(self.result_path)
        self.assertEqual(sync.refresh_heartbeats(bench, self.archive), [dict(model="test-model", measurements=1)])
        self.assertEqual(bench["models"][0]["score"], 50)
        self.assertEqual(bench["models"][0]["heartbeats_scored"], 1)
        self.assertEqual(bench["results"]["repo"]["test-model"]["heartbeats_after"], 50)
        self.assertEqual(sync.refresh_heartbeats(bench, self.archive), [])


if __name__ == "__main__":
    unittest.main()
