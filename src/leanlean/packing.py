"""Pack and verify a compressed copy of a finished dataset/model pair.

Large text files use zstd; capture checkpoints and run-level directories use
tar/zstd. Each .index.jsonl records original names, hashes and storage methods.
Checkpoint gzip envelopes are removed; their inner tar bytes are preserved.
See docs/result_archives.md for layout, verification and extraction commands.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

INDEX = ".index.jsonl"
ZSTD = ["zstd", "-q", "-10", "--long=31", "-T4"]
COMPRESS_MIN_BYTES = 1 << 20
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".diff", ".txt", ".patch", ".yaml"}
CHECKPOINT_RE = re.compile(
    r"(checkpoint_\d+\.(sources\.tar\.gz|incremental\.diff)|submitted\.sources\.tar\.gz)$"
)
FILE_BY_FILE = {"run", "postprocess"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Store:
    """Writes stored copies under one directory and records its .index.jsonl."""

    def __init__(self, dest: Path):
        self.dest, self.records = dest, []

    def put(self, src: Path, rel: str) -> None:
        size = src.stat().st_size
        target = self.dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(src)
        if size >= COMPRESS_MIN_BYTES and src.suffix in TEXT_SUFFIXES:
            subprocess.run(ZSTD + ["-o", f"{target}.zst", str(src)], check=True)
            stored, how = f"{rel}.zst", "zstd"
        else:
            shutil.copy2(src, target)
            stored, how = rel, "copy"
        self.records.append({"path": rel, "size": size, "sha256": digest, "stored": stored, "how": how})

    def _archive(self, archive: Path):
        archive.parent.mkdir(parents=True, exist_ok=True)
        out = open(archive, "wb")
        proc = subprocess.Popen(ZSTD + ["-c"], stdin=subprocess.PIPE, stdout=out)
        tar = tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT)
        return out, proc, tar

    @staticmethod
    def _close(archive: Path, out, proc, tar) -> None:
        tar.close()
        proc.stdin.close()
        code = proc.wait()
        out.close()
        if code:
            raise RuntimeError(f"zstd failed for {archive}")

    def pack_dir(self, src: Path, rel: str) -> None:
        """Store a whole directory as <rel>.tar.zst."""
        archive = self.dest / f"{rel}.tar.zst"
        handles = self._archive(archive)
        tar = handles[2]
        for path in sorted(p for p in src.rglob("*") if p.is_file() or p.is_symlink()):
            member = path.relative_to(src).as_posix()
            if path.is_symlink():
                tar.add(path, arcname=member, recursive=False)
                self.records.append({"path": f"{rel}/{member}", "how": "symlink",
                                     "target": os.readlink(path), "stored": f"{rel}.tar.zst", "member": member})
                continue
            info = tar.gettarinfo(str(path), arcname=member)
            info.type = tarfile.REGTYPE  # hardlinked files become full copies
            with open(path, "rb") as stream:
                tar.addfile(info, stream)
            self.records.append({"path": f"{rel}/{member}", "size": info.size, "sha256": sha256_file(path),
                                 "stored": f"{rel}.tar.zst", "member": member, "how": "tar"})
        self._close(archive, *handles)

    def pack_checkpoints(self, capture: Path, rel: str, names: list[str]) -> None:
        """One checkpoints.tar.zst per capture; .tar.gz members are stored gunzipped."""
        prefix = f"{rel}/" if rel else ""
        archive = self.dest / f"{prefix}checkpoints.tar.zst"
        handles = self._archive(archive)
        tar = handles[2]
        for name in sorted(names):
            src = capture / name
            stat = src.stat()
            member = name[:-3] if name.endswith(".gz") else name
            inner = hashlib.sha256()
            with tempfile.SpooledTemporaryFile(max_size=256 << 20) as buffer:
                opener = gzip.open if name.endswith(".gz") else open
                with opener(src, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1 << 22), b""):
                        inner.update(chunk)
                        buffer.write(chunk)
                info = tarfile.TarInfo(member)
                info.size, info.mtime, info.mode = buffer.tell(), int(stat.st_mtime), 0o644
                buffer.seek(0)
                tar.addfile(info, buffer)
            self.records.append({
                "path": f"{prefix}{name}", "size": stat.st_size, "sha256": sha256_file(src),
                "stored": f"{prefix}checkpoints.tar.zst", "member": member, "member_sha256": inner.hexdigest(),
                "how": "gunzip-tar" if name.endswith(".gz") else "tar",
            })
        self._close(archive, *handles)

    def put_tree(self, root: Path) -> None:
        """Store every file below root, packing each capture's checkpoints."""
        checkpoints: dict[Path, list[str]] = defaultdict(list)
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            dirnames.sort()
            rel_dir = here.relative_to(root).as_posix()
            for name in sorted(filenames):
                src = here / name
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                if src.is_symlink():
                    self.records.append({"path": rel, "how": "symlink", "target": os.readlink(src)})
                elif name.endswith(".tmp"):
                    continue  # interrupted atomic writes
                elif CHECKPOINT_RE.match(name):
                    checkpoints[here].append(name)
                else:
                    self.put(src, rel)
        for capture, names in checkpoints.items():
            rel_dir = capture.relative_to(root).as_posix()
            self.pack_checkpoints(capture, "" if rel_dir == "." else rel_dir, names)

    def write_index(self) -> None:
        self.dest.mkdir(parents=True, exist_ok=True)
        with open(self.dest / INDEX, "w") as stream:
            for record in self.records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")


def pack_repository(src: Path, dest: Path) -> int:
    store = Store(dest)
    store.put_tree(src)
    store.write_index()
    return len(store.records)


def pack_run_level(pair: Path, dest: Path, repositories: set[str]) -> int:
    """Everything in the pair directory that is not a repository directory."""
    store = Store(dest)
    for entry in sorted(pair.iterdir(), key=lambda p: p.name):
        if entry.name in repositories or entry.name == INDEX:
            continue
        if entry.is_symlink():
            store.records.append({"path": entry.name, "how": "symlink", "target": os.readlink(entry)})
        elif entry.is_dir() and entry.name in FILE_BY_FILE:
            # Small, readable state: stored file by file with its own index.
            sub = Store(dest / entry.name)
            sub.put_tree(entry)
            sub.write_index()
        elif entry.is_dir():
            store.pack_dir(entry, entry.name)
        elif entry.is_file() and not entry.name.endswith(".tmp"):
            store.put(entry, entry.name)
    store.write_index()
    return len(store.records)


def _sha_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 22), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_directory(directory: Path, root: Path | None = None) -> list[str]:
    """Check every stored file listed in directory/.index.jsonl.

    A "dup" record (written by archive builders) points at an identical copy
    stored elsewhere below root; that copy is checked where it is stored.
    """
    errors: list[str] = []
    archives: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in open(directory / INDEX):
        record = json.loads(line)
        how = record["how"]
        if how == "symlink":
            continue
        if how == "dup":
            if not ((root or directory) / record["same_as"].split("::", 1)[0]).exists():
                errors.append(f"{directory}/{record['path']}: copy {record['same_as']} is missing")
            continue
        stored = directory / record["stored"]
        if not stored.is_file():
            errors.append(f"{directory}/{record['path']}: stored file {record['stored']} is missing")
            continue
        if how in ("tar", "gunzip-tar"):
            archives[record["stored"]][record["member"]] = record
            continue
        if how == "copy":
            with open(stored, "rb") as stream:
                ok = _sha_stream(stream) == record["sha256"]
        else:
            proc = subprocess.Popen(["zstd", "-dc", "--long=31", "-q", str(stored)], stdout=subprocess.PIPE)
            ok = _sha_stream(proc.stdout) == record["sha256"] and proc.wait() == 0
        if not ok:
            errors.append(f"{directory}/{record['path']}: {how} mismatch")
    for stored, members in archives.items():
        proc = subprocess.Popen(["zstd", "-dc", "--long=31", "-q", str(directory / stored)], stdout=subprocess.PIPE)
        seen, hashes = set(), {}
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for info in tar:
                if info.isfile():
                    hashes[info.name] = got = _sha_stream(tar.extractfile(info))
                elif info.islnk():
                    got = hashes.get(info.linkname)
                else:
                    continue
                record = members.get(info.name)
                if record is None:
                    continue
                want = record["member_sha256"] if record["how"] == "gunzip-tar" else record["sha256"]
                if got != want:
                    errors.append(f"{directory}/{record['path']}: {record['how']} mismatch")
                seen.add(info.name)
        # tarfile stops at the end-of-archive marker; drain the zero padding
        # after it, or zstd blocks writing into a full pipe and wait() hangs.
        while proc.stdout.read(1 << 20):
            pass
        if proc.wait():
            errors.append(f"{directory}/{stored}: zstd failed")
        errors += [f"{directory}/{members[m]['path']}: missing from {stored}" for m in set(members) - seen]
    return errors


def verify(root: Path, jobs: int = 16) -> tuple[int, list[str]]:
    """Verify every packed directory below root; returns (directories, errors)."""
    from concurrent.futures import ProcessPoolExecutor

    directories = sorted(path.parent for path in root.rglob(INDEX))
    errors: list[str] = []
    with ProcessPoolExecutor(jobs) as pool:
        for found in pool.map(verify_directory, directories, [root] * len(directories)):
            errors += found
    return len(directories), errors
