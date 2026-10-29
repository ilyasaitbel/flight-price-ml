"""
Build the brand assets for SkyPrice.

Usage
-----
    python scripts/build_brand_assets.py

Outputs (into static/)
----------------------
    favicon.svg      the app mark, copied next to the app so Streamlit can serve it
    favicon.png      512x512 raster fallback
    og_image.png     1200x630 social preview card

``static/`` is the folder Streamlit serves as a static asset directory, so the
files land on ``/static/<name>`` at runtime.
"""

from __future__ import annotations

import base64
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
STATIC_DIR = BASE_DIR / "static"

CHROMIUM = Path(
    r"C:\Users\Youcode\AppData\Local\ms-playwright\chromium-1248\chrome-win64\chrome.exe"
)

MARK = ASSETS_DIR / "favicon.svg"


def mark_data_uri() -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(MARK.read_bytes()).decode()


def build_og_card(page) -> None:
    uri = mark_data_uri()
    page.set_viewport_size({"width": 1200, "height": 630})
    page.set_content(
        f"""
<body style="margin:0;width:1200px;height:630px;background:#0B1120;
             font-family:'Segoe UI',system-ui,sans-serif;color:#F8FAFC;overflow:hidden">
  <div style="position:absolute;inset:0;
       background:radial-gradient(at 12% 8%, rgba(59,130,246,.28) 0px, transparent 55%),
                  radial-gradient(at 88% 92%, rgba(79,70,229,.26) 0px, transparent 55%);"></div>

  <div style="position:relative;padding:92px 78px 0;display:flex;gap:40px;align-items:flex-start">
    <img src="{uri}" width="128" height="128" style="border-radius:29px;flex:none;
         box-shadow:0 20px 48px rgba(0,0,0,.5);">

    <div>
      <div style="font-size:15px;font-weight:700;letter-spacing:3.4px;color:#60A5FA;
           text-transform:uppercase;margin-bottom:12px">AI Flight Analytics</div>
      <div style="font-size:82px;font-weight:800;letter-spacing:-3px;line-height:1.0;
           background:linear-gradient(95deg,#FFFFFF 5%,#7DD3FC 55%,#A5B4FC 100%);
           -webkit-background-clip:text;background-clip:text;color:transparent">
        SkyPrice</div>
      <div style="font-size:26px;color:#94A3B8;margin-top:20px;line-height:1.45;max-width:760px">
        Predicting domestic flight fares with machine learning &mdash; 300,153 real flights,
        one Random Forest, sub-two-second estimates.
      </div>
    </div>
  </div>

  <div style="position:absolute;left:78px;right:78px;bottom:124px;height:1px;
       background:linear-gradient(90deg,#273449 0%,#273449 72%,transparent 100%)"></div>

  <div style="position:absolute;left:78px;bottom:58px;display:flex;gap:14px">
    <span style="background:rgba(17,24,39,.85);border:1px solid #273449;color:#BFDBFE;
          padding:12px 22px;border-radius:12px;font-size:18px;font-weight:600">
      300,153 flights</span>
    <span style="background:rgba(17,24,39,.85);border:1px solid #273449;color:#A7F3D0;
          padding:12px 22px;border-radius:12px;font-size:18px;font-weight:600">
      R&sup2; 0.984</span>
    <span style="background:rgba(17,24,39,.85);border:1px solid #273449;color:#FDE68A;
          padding:12px 22px;border-radius:12px;font-size:18px;font-weight:600">
      MAE &#8377;1,338</span>
  </div>

  <div style="position:absolute;right:78px;bottom:64px;color:#64748B;font-size:16px;
       font-family:ui-monospace,Menlo,Consolas,monospace">
    github.com/ilyasaitbel/flight-price-ml</div>
</body>
""",
        wait_until="load",
    )
    page.wait_for_timeout(900)
    page.screenshot(path=str(STATIC_DIR / "og_image.png"))


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy(MARK, STATIC_DIR / "favicon.svg")
    print("+ static/favicon.svg")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=str(CHROMIUM) if CHROMIUM.exists() else None
        )
        page = browser.new_page(device_scale_factor=1)

        # 512x512 raster fallback.
        page.set_viewport_size({"width": 512, "height": 512})
        page.set_content(
            f'<body style="margin:0"><img src="{mark_data_uri()}" width="512" height="512"></body>',
            wait_until="load",
        )
        page.wait_for_timeout(500)
        page.screenshot(path=str(STATIC_DIR / "favicon.png"))
        print("+ static/favicon.png")

        build_og_card(page)
        print("+ static/og_image.png")

        browser.close()


if __name__ == "__main__":
    main()
