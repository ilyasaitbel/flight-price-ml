"""
Capture screenshots of the running SkyPrice app for the README.

Usage
-----
    python scripts/capture_screenshots.py [url] [outdir]

Requires the app to be running, e.g.
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502"
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "docs").resolve()

CHROMIUM = Path(
    r"C:\Users\Youcode\AppData\Local\ms-playwright\chromium-1248\chrome-win64\chrome.exe"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(CHROMIUM) if CHROMIUM.exists() else None,
            args=["--force-color-profile=srgb", "--hide-scrollbars"],
        )
        page = browser.new_page(
            viewport={"width": 1440, "height": 1100},
            device_scale_factor=2,
        )

        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        # Streamlit renders progressively; wait for the hero heading to appear.
        page.wait_for_selector("text=Flight Price Prediction", timeout=60000)
        page.wait_for_timeout(3500)
        # Streamlit scrolls an inner container, not the document, so a tall
        # viewport is the reliable way to capture the whole app.
        page.set_viewport_size({"width": 1440, "height": 1330})
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT_DIR / "app_home.png"))
        print("captured app_home.png")

        # The form defaults are already a valid, realistic trip - submit it.
        button = page.get_by_role("button", name="Predict Flight Price")
        button.click()
        page.wait_for_selector("text=ESTIMATED FLIGHT PRICE", timeout=90000)
        page.wait_for_timeout(2500)

        page.set_viewport_size({"width": 1440, "height": 1950})
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT_DIR / "app_prediction.png"))
        print("captured app_prediction.png")

        browser.close()


if __name__ == "__main__":
    main()
