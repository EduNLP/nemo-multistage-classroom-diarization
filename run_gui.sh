#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python not found in PATH." >&2
  exit 1
fi

if ! $PY - <<'PY' >/dev/null 2>&1; then
import importlib; import sys
sys.exit(0 if importlib.util.find_spec('streamlit') else 1)
PY
then
  echo "Installing Streamlit into current environment..."
  $PY -m pip install --upgrade pip
  $PY -m pip install -r requirements-gui.txt
fi

exec $PY -m streamlit run app_streamlit.py

