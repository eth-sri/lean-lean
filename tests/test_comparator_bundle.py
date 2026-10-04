"""The pinned Comparator tool bundle config; needs no Docker or network."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_comparator_bundle.py"


def builder():
    spec = importlib.util.spec_from_file_location("build_comparator_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pins_cover_every_benchmark_repository_once():
    module = builder()
    _, pins = module.load_config(module.CONFIG)
    assert {pin.tool for pin in pins} == set(module.LAYOUT)
    for tool in ("comparator", "lean4export"):
        users = [repo for pin in pins if pin.tool == tool for repo in pin.used_by]
        assert len(users) == len(set(users)) == 64, tool
    for pin in pins:
        if pin.builder == "lean":
            assert pin.toolchain.startswith("leanprover/lean4:v4.")


def test_install_paths_match_the_loader_layout(tmp_path):
    module = builder()
    _, pins = module.load_config(module.CONFIG)
    paths = {pin.install_path for pin in pins}
    assert "bin/landrun" in paths and "bin/nanoda_bin" in paths
    assert ("comparator/68a064109f01c08f47c8edc9f51d6a2bbffaa188/comparator"
            in paths)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list", "--bundle", str(tmp_path),
         "--only", "lean4export@15f6055e"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[:2] == ["missing", "lean4export@15f6055e"]
