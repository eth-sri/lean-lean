"""The paper's figures and tables render from the committed results/paper CSVs."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/paper"


def test_manifest_pins_every_committed_csv():
    tables = json.loads((RESULTS / "manifest.json").read_text())["tables"]
    assert sorted(tables) == sorted(p.name for p in (RESULTS / "data").glob("*.csv"))
    for name, entry in tables.items():
        assert hashlib.sha256((RESULTS / "data" / name).read_bytes()).hexdigest() == entry["sha256"], name


def test_render_writes_every_paper_figure_and_table(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/paper/render.py"), "--no-tex",
                    "--output-dir", str(tmp_path)], check=True, capture_output=True)
    sys.path.insert(0, str(ROOT / "scripts/paper"))
    from render import FIGURES, TABLES
    for path, *_ in FIGURES.values():
        for fmt in ("pdf", "png"):
            assert (tmp_path / "figures" / f"{path}.{fmt}").stat().st_size > 0
    for name in TABLES:
        assert (tmp_path / "tables" / f"{name}.tex").is_file()
    assert sorted(p.name for p in (tmp_path / "figures/examples").iterdir()) == [
        "confluence-opus-sol.tex", "morphic-numbers-opus-gemini.tex", "tucker-preprocess-opus.tex"]
    # Table cells are the CSV values, rounded only for display.
    leaderboard = (tmp_path / "tables/leaderboard.tex").read_text()
    for row in csv.DictReader((RESULTS / "data/leaderboard.csv").open()):
        assert f"{float(row['compression_pct']):.2f}" in leaderboard
    macros = (tmp_path / "tables/results.tex").read_text()
    assert r"\newcommand{\OpusCompression}{48.30}" in macros
