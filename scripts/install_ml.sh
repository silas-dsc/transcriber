#!/usr/bin/env bash
# Install the high-quality ML back-ends (Demucs separation + Spotify
# basic-pitch transcription) reliably across platforms.
#
# Why this script exists:
#   basic-pitch's package metadata pins `tensorflow-macos<2.15.1` on macOS with
#   Python > 3.11, but no such wheel exists there, so a plain
#   `pip install basic-pitch` (or `pip install ".[pitch]"`) fails to resolve.
#   basic-pitch actually bundles every model format (TF, CoreML, TFLite, ONNX)
#   and only needs ONE runtime. We install it with `--no-deps` plus the ONNX
#   runtime, which works on Apple Silicon and every Python version.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/install_ml.sh           # separation + pitch
#   bash scripts/install_ml.sh pitch     # pitch only
#   bash scripts/install_ml.sh separation
set -euo pipefail

cd "$(dirname "$0")/.."
WHAT="${1:-all}"

PY="$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "Python $PY on $(uname -s) $(uname -m)"

install_separation() {
  echo ">> Installing Demucs + PyTorch (stem separation)..."
  pip install "demucs>=4.0" "torch>=2.0"
}

install_pitch() {
  echo ">> Installing basic-pitch with the ONNX runtime (polyphonic pitch)..."
  # --no-deps skips basic-pitch's broken tensorflow-macos pin; we then supply
  # its real runtime requirements plus onnxruntime as the inference back-end.
  pip install --no-deps "basic-pitch>=0.3"
  pip install \
    onnxruntime \
    "mir_eval>=0.6" \
    "resampy>=0.2.2,<0.4.3" \
    scikit-learn \
    typing_extensions
}

case "$WHAT" in
  all)        install_separation; install_pitch ;;
  separation) install_separation ;;
  pitch)      install_pitch ;;
  *) echo "usage: $0 [all|separation|pitch]" >&2; exit 2 ;;
esac

echo
echo "Done. Verify with:"
echo "  python -c \"import demucs; print('demucs', demucs.__version__)\" 2>/dev/null || true"
echo "  python -c \"import basic_pitch; print('basic-pitch ONNX:', basic_pitch.ONNX_PRESENT)\""
echo
echo "Then transcribe with the high-quality back-ends:"
echo "  transcriber song.wav -o song.musicxml --separation demucs --pitch basic-pitch"
