"""pack_repository / pack_run_level / verify on a small fake pair."""
import io
import json
import tarfile
import gzip
import hashlib

import pytest
import yaml

from leanlean.packing import INDEX, pack_repository, pack_run_level, verify
from scripts.pack_results import _check_done, _repositories_from_pair


def _sources_tar_gz(path, text):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        data = text.encode()
        info = tarfile.TarInfo("Main.lean")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    path.write_bytes(gzip.compress(buffer.getvalue()))


def _fake_pair(root):
    repo = root / "palomar__2026-08-20-000004"
    capture = repo / "playback/capture_001"
    capture.mkdir(parents=True)
    (repo / "result.json").write_text(json.dumps({"status": "passed"}))
    (repo / "palomar__2026-08-20-000004.traj.json").write_text(json.dumps({"steps": ["x" * 100] * 20000}))  # > 1 MiB
    for index in range(3):
        _sources_tar_gz(capture / f"checkpoint_{index:05d}.sources.tar.gz", f"theorem t{index} : True := trivial\n")
        (capture / f"checkpoint_{index:05d}.incremental.diff").write_text(f"diff {index}\n")
    (capture / "playback.json").write_text("{}")
    (root / "run").mkdir()
    (root / "run/run.yaml").write_text("status: complete\n")
    (root / "proxy_traces").mkdir()
    (root / "proxy_traces/a.accounting.jsonl").write_text('{"cost": 1}\n')
    (root / "preds.json").write_text("{}")
    return repo


def test_pack_and_verify(tmp_path):
    pair, packed = tmp_path / "pair", tmp_path / "packed"
    repo = _fake_pair(pair)
    pack_repository(repo, packed / repo.name)
    pack_run_level(pair, packed, {repo.name})

    stored = {p.relative_to(packed).as_posix() for p in packed.rglob("*") if p.is_file()}
    assert f"{repo.name}/palomar__2026-08-20-000004.traj.json.zst" in stored
    assert f"{repo.name}/playback/capture_001/checkpoints.tar.zst" in stored
    assert not any(name.endswith(".sources.tar.gz") for name in stored)
    assert "proxy_traces.tar.zst" in stored and "run/run.yaml" in stored
    assert verify(packed) == (3, [])  # repository, run level, run/

    records = [json.loads(line) for line in open(packed / repo.name / INDEX)]
    assert {r["how"] for r in records} == {"copy", "zstd", "tar", "gunzip-tar"}

    (packed / "preds.json").write_text('{"changed": true}')
    directories, errors = verify(packed)
    assert len(errors) == 1 and "preds.json" in errors[0]


def test_dup_records_point_at_a_stored_copy(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a/x.txt").write_text("x")
    (tmp_path / "a" / INDEX).write_text(json.dumps({"path": "x.txt", "how": "copy", "stored": "x.txt", "size": 1,
        "sha256": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"}) + "\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / INDEX).write_text(json.dumps({"path": "y.txt", "how": "dup", "same_as": "a/x.txt"}) + "\n")
    assert verify(tmp_path) == (2, [])
    (tmp_path / "a/x.txt").unlink()
    _, errors = verify(tmp_path)
    assert any("copy a/x.txt is missing" in e for e in errors)


def test_endpoint_only_packing_still_requires_all_verified_results(tmp_path):
    repo = _fake_pair(tmp_path)
    endpoint = tmp_path / "postprocess/endpoints"
    endpoint.mkdir(parents=True)
    (endpoint / "run.yaml").write_text("status: complete\n")
    assert _check_done(tmp_path, [repo.name]) == ["postprocess stage heartbeats is not complete"]
    assert _check_done(tmp_path, [repo.name], endpoints_only=True) == []
    (repo / "result.json").unlink()
    assert _check_done(tmp_path, [repo.name], endpoints_only=True) == ["1 repositories have no result.json"]
    (endpoint / "run.yaml").write_text("status: failed\n")
    assert "postprocess stage endpoints is not complete" in _check_done(tmp_path, [repo.name], endpoints_only=True)


def test_existing_pair_requires_its_pinned_manifest(tmp_path):
    pair = tmp_path / "results/demo/provider/model"
    (pair / "run").mkdir(parents=True)
    manifest = pair / "run/manifest.yaml"
    manifest.write_text(yaml.safe_dump({"kind": "leanlean_evaluation_run", "repositories": [{"id": "repo"}]}))
    (pair / "run/run.yaml").write_text(yaml.safe_dump({
        "kind": "leanlean_run", "status": "complete", "experiment": {
            "manifest": str(manifest.relative_to(tmp_path)),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
    }))
    assert _repositories_from_pair(pair, tmp_path) == ["repo"]
    manifest.write_text(manifest.read_text() + "model: changed\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        _repositories_from_pair(pair, tmp_path)
    with pytest.raises(ValueError):
        _repositories_from_pair(tmp_path / "elsewhere", tmp_path)


def test_verify_drains_the_tar_stream(tmp_path):
    # Padding larger than the pipe buffer exposes an undrained-stream deadlock.
    # Use a child process so a regression fails on timeout instead of hanging pytest.
    import hashlib
    import subprocess
    import sys

    data = b"theorem t : True := trivial\n" * 100
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("checkpoint_00000.incremental.diff")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue() + b"\0" * (256 * 1024)
    directory = tmp_path / "repo"
    directory.mkdir()
    subprocess.run(["zstd", "-q", "--long=31", "-o", str(directory / "checkpoints.tar.zst")], input=raw, check=True)
    record = {"how": "tar", "path": "playback/checkpoint_00000.incremental.diff", "stored": "checkpoints.tar.zst",
              "member": "checkpoint_00000.incremental.diff", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    (directory / INDEX).write_text(json.dumps(record) + "\n")
    script = ("import sys; from pathlib import Path; from leanlean.packing import verify_directory; "
              "errors = verify_directory(Path(sys.argv[1])); print(errors); sys.exit(len(errors))")
    result = subprocess.run([sys.executable, "-c", script, str(directory)], timeout=60)
    assert result.returncode == 0
