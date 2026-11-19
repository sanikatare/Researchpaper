#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PY_BIN=""
if [[ -x "$APP_DIR/.venv/bin/python" ]] && "$APP_DIR/.venv/bin/python" -c "import streamlit" >/dev/null 2>&1; then
  PY_BIN="$APP_DIR/.venv/bin/python"
elif python3 -c "import streamlit" >/dev/null 2>&1; then
  PY_BIN="python3"
else
  echo "[start_dev] Bootstrapping Python virtual environment in .venv..."
  python3 -m venv --without-pip "$APP_DIR/.venv"
  curl -sSfL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "$APP_DIR/.venv/bin/python" /tmp/get-pip.py
  "$APP_DIR/.venv/bin/pip" install --no-cache-dir -r "$APP_DIR/requirements.txt"
  PY_BIN="$APP_DIR/.venv/bin/python"
fi

exec "$PY_BIN" -m streamlit run app/streamlit_app.py \
  --server.port=3000 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false
