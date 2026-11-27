"""Validate graph reuse and score/source guards without a LeanLean checkout."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import sync_graphs as graphs
from sync_results import encoded, origin_summary


class GraphSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_baseline_keeps_published_indices_aliases_and_lanes(self):
        content = gzip.compress(json.dumps("theorem a : True := by trivial\n").encode())
        (self.root / "source.bin").write_bytes(content)
        (self.root / "source.json.gz").write_bytes(gzip.compress(encoded({"A.lean": [0, len(content)]})))
        graph = dict(nodes=dict(name=["a", "b"], file=[0, 0], start=[1, 2], end=[1, 2], tokens=[9, 7], x=[150, 160], y=[30, 65]),
            files=["A.lean"], excluded=[], aliases={"0": ["a.aux"]}, lanes=[[30, "Results", "kept", 1], [64, "", "kept", 1]], world=[300, 100], per_lane=24)
        base = graphs.baseline(graph, self.root)
        self.assertEqual(base["member"], {"a": 0, "b": 1, "a.aux": 0})
        self.assertEqual(base["lane_of"], [0, 1])
        self.assertEqual(base["positions"], [(150, 30), (160, 65)])
        self.assertEqual(base["sources"]["A.lean"], "theorem a : True := by trivial\n")

    def test_model_origin_uses_all_repositories_and_waits_for_full_coverage(self):
        bench = dict(repositories=[dict(id="a", tokens=100), dict(id="b", tokens=200)],
                     results=dict(a={"model": dict(status="passed")}, b={"model": dict(status="failed")}))
        self.assertIsNone(origin_summary(bench, "model"))
        bench["results"]["a"]["model"]["origin"] = dict(deleted=40, proof_simplification=15, structural_diff=-5)
        origin = origin_summary(bench, "model")
        self.assertEqual(origin["saved"], 25)
        self.assertEqual(origin["deleted"], 20)
        self.assertEqual(origin["proof_rewrite"], 5)

    def graph_fixture(self, repositories):
        folder = self.root / "graphs"
        (folder / "compression-origin").mkdir(parents=True)
        row = dict(model="model", repository="repo", status="passed", baseline_tokens="10", post_tokens="4", graph_directory="graphs/repositories/repo", program_cache="graphs/program_sites")
        with (folder / "classification-inputs.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        with (folder / "compression-origin/classification-repositories.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["repository"])
            writer.writeheader()
            writer.writerow(dict(repository="repo"))
        spec = dict(key="model", short="model", graphs="graphs", source="dataset/provider/model", run="results/dataset/provider/model")
        bench = dict(repositories=repositories, results=dict(repo={"model": dict(status="passed", post=4)}),
                     result_sources={"model": dict(result_sha256=dict(repo="original-result-hash"))})
        return spec, bench

    def test_missing_graph_rows_are_rejected(self):
        spec, bench = self.graph_fixture([dict(id="repo", tokens=10), dict(id="missing", tokens=10)])
        with self.assertRaisesRegex(ValueError, "graph coverage differs"):
            graphs.prepare_jobs(spec, bench, self.root, self.root / "archive", self.root / "data", self.root / "cache")

    def test_modified_result_cannot_publish_a_graph_for_a_frozen_score(self):
        spec, bench = self.graph_fixture([dict(id="repo", tokens=10)])
        result = self.root / "archive" / spec["source"] / "repo/result.json"
        result.parent.mkdir(parents=True)
        result.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "archived result changed"):
            graphs.prepare_jobs(spec, bench, self.root, self.root / "archive", self.root / "data", self.root / "cache")


if __name__ == "__main__":
    unittest.main()
