"""Render site/assets/og.png: the link preview (Open Graph / Twitter card).

Updates the card alt text from the rendered chart and versions its image URL.

A 1200x630 card: the title and the paper's subtitle, the SRI Lab and ETH Zurich logos, and the site's
own frontier chart (score against cost). The chart is not redrawn here: the
script serves site/, loads it in a browser and moves the rendered chart into
the card, so the card always matches the website. Needs network access (Google
Fonts) and Playwright's Firefox. Pages runs this before every deployment:

    uv run --with playwright==1.63.0 python pipeline/build_og.py

(`uv run --with playwright==1.63.0 playwright install firefox` once, if Firefox is missing.)
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import html
import http.server
import re
import threading
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"
WIDTH, HEIGHT = 2400, 1260
# The card is laid out at 840x441 CSS pixels and captured at 2400/840 scale, a
# 1200x630 card at twice the pixel density so it stays sharp on a phone.
SCALE = WIDTH / 840
# The chart draws at 1/ZOOM of the card's width and scales up, so its model
# names and axis labels stay legible while fitting all reasoning variants.
ZOOM = 1.2
# The logos' size in the chart's own pixels (the site's are 18): small marks, so
# the model names carry the card.
MARK = 14

CARD_CSS = """
  body { margin: 0; background: var(--paper); }
  .og-card { box-sizing: border-box; width: 840px; height: 441px; padding: 20px 36px 0 30px;
             display: flex; flex-direction: column; gap: 2px; background: var(--paper); overflow: hidden; }
  .og-card header { display: flex; align-items: center; justify-content: space-between; padding-left: 6px; margin-bottom: 5px; }
  .og-card h1 { margin: 0; font: 500 32px/1 var(--font-display); letter-spacing: -0.01em; color: var(--ink); }
  .og-card header p { margin: 6px 0 0; font: 400 20px/1.2 var(--font-ui); color: var(--ink-2); }
  .og-card .orgs { display: flex; align-items: center; gap: 22px; }
  .og-card .orgs img { height: 19px; width: auto; display: block; }
"""

CARD_JS = """([css, zoom, mark]) => {
  const chart = document.querySelector('.chart.frontier');
  const style = document.createElement('style');
  style.textContent = css;
  const card = document.createElement('div');
  card.className = 'og-card';
  card.innerHTML = '<header><div><h1>LeanLean</h1><p>Benchmarking Repository-Scale Lean Proof Compression</p></div><div class="orgs">'
    + '<img src="assets/sri-logo.svg" alt="SRI Lab"><img src="assets/eth-logo.svg" alt="ETH Zurich"></div></header>';
  document.body.replaceChildren(style, card);
  // The chart fills what the header leaves.
  const room = card.getBoundingClientRect().bottom - card.querySelector('header').getBoundingClientRect().bottom - 8;
  chart.dataset.zoom = zoom;
  chart.dataset.mark = mark;
  chart.dataset.height = Math.floor(room / zoom);
  card.append(chart);
  // The chart redraws for its new width on its ResizeObserver; give it a frame.
  return new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
}"""


def write_metadata(alt: str) -> None:
    """Use the rendered chart so hidden models stay out of the card metadata."""
    index = SITE / "index.html"
    text = index.read_text(encoding="utf-8")
    text = re.sub(r'(<meta property="og:image:alt" content=")[^"]*(")',
                  lambda m: m.group(1) + html.escape(alt) + m.group(2), text)
    version = hashlib.sha256((SITE / "assets/og.png").read_bytes()).hexdigest()[:10]
    text = re.sub(r'(https://leanleanbench.com/assets/og\.png)(?:\?v=[\w-]+)?',
                  lambda m: m.group(1) + "?v=" + version, text)
    index.write_text(text, encoding="utf-8")


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


async def main() -> None:
    from playwright.async_api import async_playwright

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(SITE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch()
            page = await browser.new_page(viewport={"width": 840, "height": 441}, device_scale_factor=SCALE)
            await page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
            await page.wait_for_selector(".chart.frontier svg .pt")
            await page.evaluate("document.fonts.ready")
            await page.evaluate(CARD_JS, [CARD_CSS, ZOOM, MARK])
            await page.wait_for_function("""() => {
                const chart = document.querySelector('.og-card .chart');
                const svg = chart?.querySelector(':scope > svg');
                return svg && +svg.getAttribute('width') === Math.round(chart.clientWidth / chart.dataset.zoom);
            }""")
            await page.wait_for_load_state("networkidle")
            alt = await page.locator(".chart.frontier > svg").get_attribute("aria-label")
            await page.screenshot(path=str(SITE / "assets/og.png"))
            await browser.close()
    finally:
        server.shutdown()
    write_metadata(alt)
    print(f"wrote {SITE / 'assets/og.png'} and its alt text")


if __name__ == "__main__":
    asyncio.run(main())
