"""Entry point for running the Streamlit research workbench or verifying setup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import DEFAULT_MODEL_ID, RESEARCH_TITLE, ensure_directories


def launch_dashboard(port: int = 8501, address: str = "127.0.0.1") -> int:
    """Launch the Streamlit research dashboard."""
    ensure_directories()
    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        f"--server.address={address}",
        "--server.headless=true",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    print(f"Starting: {RESEARCH_TITLE}")
    print(f"Experimental Model: {DEFAULT_MODEL_ID}")
    raise SystemExit(launch_dashboard())
