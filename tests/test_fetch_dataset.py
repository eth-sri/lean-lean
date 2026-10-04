"""Consistency of the records scripts/fetch_dataset.py prepares the Hugging Face release with."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "data/datasets/leanlean_20260914"


def test_overlay_and_file_modes_pin_one_release_of_the_mapped_repositories():
    overlay = json.loads((RECORDS / "overlay.json").read_text())
    modes = json.loads((RECORDS / "file-modes.json").read_text())
    assert (modes["dataset"], modes["revision"]) == (overlay["dataset"], overlay["revision"])
    mapping_path = ROOT / overlay["palomar_mapping"]["path"]
    assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == overlay["palomar_mapping"]["sha256"]
    targets = {row["benchmark_id"] for row in json.loads(mapping_path.read_text())["targets"]}
    repositories = overlay["repositories"]
    assert len(repositories) == 64 and set(repositories) <= targets
    for pins in repositories.values():
        assert pins["shared_environment"] in overlay["shared_environments"]
    assert {path.split("/")[1] for path in modes["modes"] if path.startswith("repos/")} <= set(repositories)
