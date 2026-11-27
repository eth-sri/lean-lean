"""Stamp site/ asset URLs with a hash of the site's code.

Every `?v=...` on a module import, modulepreload or stylesheet link (and the
entry `<script>`) is rewritten to one content hash of all JS and CSS, so any
edit yields new URLs and a browser can never mix a stale module with fresh
ones. Run after editing site/assets (build.sh runs it too).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
REF = re.compile(r"""((?:\./|assets/(?:js|css)/)[\w.-]+\.(?:js|css))(?:\?v=[\w-]+)?(?=["'])""")


def main() -> None:
    files = sorted((SITE / "assets/js").glob("*.js")) + sorted((SITE / "assets/css").glob("*.css"))
    # Hash the code with its version stamps removed, so stamping is idempotent.
    digest = hashlib.sha256()
    # Inline previews use the module version when loading their SVG artwork.
    artwork = sorted((SITE / "assets").glob("explore-*.svg"))
    for path in [*files, *artwork]:
        digest.update(REF.sub(r"\1", path.read_text(encoding="utf-8")).encode())
    version = digest.hexdigest()[:10]
    pages = [*sorted(SITE.glob("*.html")), *sorted(SITE.glob("*/index.html"))]
    for path in [*pages, *files]:
        text = path.read_text(encoding="utf-8")
        stamped = REF.sub(lambda m: f"{m.group(1)}?v={version}", text)
        if stamped != text:
            path.write_text(stamped, encoding="utf-8")
    print(f"assets stamped v={version}")


if __name__ == "__main__":
    main()
