#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Already in venv: $VIRTUAL_ENV"
else
  # Check python exists
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN not found."
    echo "Install it (and venv support) e.g.:"
    echo "  sudo apt install python3.13 python3.13-venv"
    exit 1
  fi

  # Create venv if missing
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv in: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # Activate venv
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  # Upgrade packaging tools
  python -m pip install --upgrade pip setuptools wheel >/dev/null

  # Install requirements if present
  if [[ -f "$REQ_FILE" ]]; then
    echo "Installing dependencies from $REQ_FILE"
    pip install -r "$REQ_FILE"
  else
    echo "No $REQ_FILE found; skipping dependency install."
  fi
fi

echo "Venv activated: $VENV_DIR"
echo "Python: $(python --version)"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exec $SHELL
fi
