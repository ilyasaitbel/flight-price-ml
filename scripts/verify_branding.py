"""Verify the SkyPrice branding in a real browser.

Usage
-----
    python scripts/verify_branding.py [URL]

Checks
------
    * document.title
    * the <link rel="icon"> href (must be the SkyPrice SVG data URI)
    * the OpenGraph / Twitter meta tags
    * the hero mark actually rendered (and has real pixel dimensions)
    * that no "WorkBuddy" string leaked into the visible page

Writes a screenshot to docs/app_home.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

CHROMIUM = Path(
    r"C:\Users\Youcode\AppData\Local\ms-playwright\chromium-1248\chrome-win64\chrome.exe"
)


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8501"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(CHROMIUM) if CHROMIUM.exists() else None
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1330})
        page.goto(url, wait_until="domcontentloaded", timeout=90_000)

        # The first request deserialises a 62 MB joblib artifact, so the app
        # can stay blank for a while. Wait for the hero to actually appear.
        page.wait_for_selector(".hero-mark", timeout=180_000)
        page.wait_for_timeout(1500)

        report = page.evaluate(
            """() => {
                const meta = (sel) => {
                    const el = document.querySelector(sel);
                    return el ? el.getAttribute('content') : null;
                };
                const link = document.querySelector("link[rel~='icon']");
                const mark = document.querySelector('.hero-mark');
                const box = mark ? mark.getBoundingClientRect() : null;
                return {
                    title: document.title,
                    favicon: link ? link.getAttribute('href') : null,
                    og_title: meta("meta[property='og:title']"),
                    og_desc: meta("meta[property='og:description']"),
                    og_image: meta("meta[property='og:image']"),
                    twitter_card: meta("meta[name='twitter:card']"),
                    twitter_image: meta("meta[name='twitter:image']"),
                    hero_mark_found: !!mark,
                    hero_mark_size: box ? [Math.round(box.width), Math.round(box.height)] : null,
                    hero_mark_src_prefix: mark ? mark.getAttribute('src').slice(0, 40) : null,
                    body_has_workbuddy: /workbuddy/i.test(document.body.innerText),
                };
            }"""
        )

        print(json.dumps(report, indent=2, ensure_ascii=False))

        page.screenshot(path=str(DOCS_DIR / "app_home.png"))
        print(f"+ {DOCS_DIR / 'app_home.png'}")

        browser.close()


if __name__ == "__main__":
    main()
