#!/usr/bin/env python3
"""Install Meta's checksum-pinned binary on the host; no container downloads."""
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import urllib.request

VERSION = "1.0.3-R2198.1"
ARTIFACTS = {
    "x86_64": ("x86", "75a68f98c437dfd17d264730c5bc72d57e5f1e18d10472a9f53261ffcc091352"),
    "aarch64": ("aarch64", "4ffcf55f5eb0668643f30c5febd90d188b9a2da65858918444d31f6046940120"),
}


def release_path() -> Path:
    root = Path(os.environ.get("MUSE_STANDALONE_RELEASES", str(
        Path(__file__).resolve().parents[1] / "data/tool_bundles/muse"
    )))
    return root / VERSION / "muse"


def verify(path: Path) -> None:
    expected = ARTIFACTS[platform.machine()][1]
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError("Muse binary does not match the pinned official checksum")


def main() -> None:
    if platform.system() != "Linux" or platform.machine() not in ARTIFACTS:
        raise RuntimeError("Muse offline bundle supports Linux x86_64/aarch64")
    target = release_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        arch = ARTIFACTS[platform.machine()][0]
        url = ("https://lookaside.facebook.com/lookaside/muse/download/"
               f"?channel=muse&version={VERSION}&file=muse-{arch}-linux")
        with tempfile.TemporaryDirectory(dir=target.parent) as directory:
            temporary = Path(directory) / "muse"
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            verify(temporary)
            temporary.chmod(0o755)
            temporary.replace(target)
    verify(target)
    subprocess.run([str(target), "--version"], check=True)
    print(f"Verified offline bundle: {target}")


if __name__ == "__main__":
    main()
