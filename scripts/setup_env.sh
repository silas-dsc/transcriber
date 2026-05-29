#!/usr/bin/env bash
# Set up the environment for development / Claude Code web sessions.
# Installs ffmpeg (if possible) and the core Python dependencies into a venv.
set -euo pipefail

cd "$(dirname "$0")/.."

# ffmpeg is needed by librosa to decode compressed audio (mp3, m4a, ...).
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg..."
  if command -v apt-get >/dev/null 2>&1; then
    (apt-get update && apt-get install -y ffmpeg) || \
      echo "warning: could not install ffmpeg automatically; install it manually."
  fi
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

pip install --quiet --upgrade pip setuptools wheel
# Install the package (core deps) plus the dev extras for running tests.
pip install --quiet -e ".[dev]"

echo "Environment ready. Activate with: . .venv/bin/activate"
echo "Run the test suite with: pytest"
