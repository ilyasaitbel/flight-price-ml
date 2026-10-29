"""
Start the Streamlit app for a hosted environment.

Usage
-----
    python run.py

Most containers inject the public port as the ``PORT`` environment variable.
This launcher reads it in Python instead of relying on shell expansion, which
is fragile: a start command written as ``--server.port $PORT`` breaks when
``$PORT`` is empty, because ``--server.port`` then swallows the next flag as
its value and Streamlit exits with an integer parse error.

It also binds to 0.0.0.0 so the platform's reverse proxy can reach the app.
"""

from __future__ import annotations

import os
import subprocess
import sys

DEFAULT_PORT = "8501"


def main() -> None:
    port = (os.environ.get("PORT") or "").strip() or DEFAULT_PORT

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        # Serve ./static at /static/... (favicon + social preview card).
        "--server.enableStaticServing",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    print(f"[run.py] starting Streamlit on 0.0.0.0:{port}", flush=True)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
