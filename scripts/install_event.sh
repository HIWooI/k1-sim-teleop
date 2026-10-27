#!/usr/bin/env bash
# One-time setup on the event laptop inside a bundle from scripts/make_event_bundle.sh:
# Python 3.12 venv + packages (internet needed once), then the pre-check without the camera.
#   bash scripts/install_event.sh          (PYTHON=/path/to/python3.12 to choose the interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
[ -f "$ROOT/k1demo.env" ] && . "$ROOT/k1demo.env"
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || { echo "Python 3.12 not found (set PYTHON=...)"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
  # torch only reads .pt / .pkl files here: the CPU build (~0.2 GB) instead of PyPI's CUDA build (~3 GB of nvidia-* wheels)
  .venv/bin/pip install -q "torch==$(sed -n 's/^torch==//p' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/pip install -q -r requirements.txt
if [ "$(uname -s)" = "Linux" ]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
.venv/bin/python -m k1live.preflight --no-camera
echo "next: double-click K1Demo.command (macOS) or bash K1Demo.command (Ubuntu)."
echo "GEM-X (NVIDIA GPU laptops only): see docs/setup.md; without it the service uses MediaPipe."
