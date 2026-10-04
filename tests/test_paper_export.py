"""The paper export's scoring rules, on a hand-made run directory."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/paper"))
import export  # noqa: E402


def state(eligible, built=True, signatures=True, verified=None):
    return dict(selected_state=dict(eligible=eligible, build_passed=built, signatures_preserved=signatures,
                                    lean_verify_passed=verified), baseline_lean_tokens=200, post_lean_tokens=50)


def test_failed_submissions_score_zero_and_unresolved_are_left_out():
    assert export.score(state(True)) == (75.0, True)
    assert export.score(state(False)) == (0.0, False)
    assert export.score(state(None, None, None)) == (None, None)
    # The endpoint verifier's own pass, without a separate signature check.
    assert export.score(state(True, signatures=None, verified=True)) == (75.0, True)


def test_a_replay_that_skipped_verification_is_not_a_failure():
    record = dict(state(False, signatures=None, verified=False),
                  points=[dict(kind="final", lean_verify=dict(checked=False, skipped="checkpoint_lean_verify_disabled"))])
    record["selected_state"]["policy"] = "submitted_final_state"
    assert export.score(record) == (None, None)


def run(tmp_path, name, verdicts, patches):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "preds.json").write_text(json.dumps({rid: dict(model_patch=p) for rid, p in patches.items()}))
    instances = {rid: dict(latest=dict(resolved=ok, build_passed=True, signatures_preserved=True, cost_total=2.0,
                                       metrics=dict(baseline_lean_tokens=200, post_lean_tokens=50)))
                 for rid, ok in verdicts.items()}
    (directory / "report_summary.json").write_text(json.dumps(dict(instances=instances)))
    for rid in patches:
        (directory / rid).mkdir()
        (directory / rid / "round_0_gen_metrics.json").write_text(json.dumps(dict(wall_time_seconds=3600)))
    return str(directory)


def test_later_runs_replace_earlier_ones_and_keep_their_verdicts(tmp_path):
    repositories = {r: dict(token_band="≤ 10k", stripped_tokens=200) for r in ("a", "b", "c")}
    first = run(tmp_path, "first", {"a": True, "b": False, "c": True}, {"a": "x", "b": "y", "c": "z"})
    # The rerun replaces b; c was re-postprocessed without a verdict for the same patch.
    second = run(tmp_path, "second", {"b": True}, {"b": "y2", "c": "z"})
    rows = {d["repository"]: d for d in export.endpoints("m", dict(runs=[first, second]), repositories)}
    assert [rows[r]["status"] for r in "abc"] == ["passed", "passed", "passed"]
    assert rows["b"]["directory"].endswith("second/b") and rows["a"]["directory"].endswith("first/a")
    assert rows["c"]["cost_usd"] == 2.0 and rows["c"]["seconds"] == 3600


def test_prompt_arm_means_match_the_committed_figure_data():
    data = ROOT / "results/paper/data"
    committed = {r["model"]: r for r in csv.DictReader((data / "opus-prompt-ablation.csv").open(newline=""))}
    rows = list(csv.DictReader((data / "opus-prompt-ablation-classification-repositories.csv").open(newline="")))
    for arm, expected in committed.items():
        value = export.macro([r for r in rows if r["arm"] == arm])
        assert all(abs(value[k] - float(expected[k])) < 1e-9 for k in (*export.CATEGORIES, "saved", "proof_rewrites"))
