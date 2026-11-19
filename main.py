"""Entry point for running the Streamlit research workbench or verifying setup."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_MODEL_ID, RESEARCH_TITLE, ensure_directories


def launch_dashboard(port: int = 3000, address: str = "0.0.0.0") -> int:
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
    parser = argparse.ArgumentParser(description="Launch the Gemini 4 Hydrogen Streamlit workbench.")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind (default: 3000)")
    parser.add_argument("--address", default="0.0.0.0", help="Address to bind (default: 0.0.0.0)")
    args = parser.parse_args()
    print(f"Starting: {RESEARCH_TITLE}")
    print(f"Experimental Model: {DEFAULT_MODEL_ID}")
    raise SystemExit(launch_dashboard(port=args.port, address=args.address))
