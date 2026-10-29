"""
Streamlit entry point.

Streamlit Community Cloud (and most one-command hosts) look for a file called
`streamlit_app.py` in the repository root. The real application lives in
`app/app.py`, so this module simply executes it.

Run locally either way:

    streamlit run streamlit_app.py     # from the repository root
    streamlit run app/app.py           # also works
"""

from pathlib import Path
import runpy

APP_FILE = Path(__file__).resolve().parent / "app" / "app.py"

if not APP_FILE.exists():
    raise FileNotFoundError(f"Application entry point missing: {APP_FILE}")

runpy.run_path(str(APP_FILE), run_name="__main__")
