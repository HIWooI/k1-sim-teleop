#!/usr/bin/env bash
# SONIC (GEAR-SONIC) whole-body webcam teleop on macOS (CPU only).
#
#   ./run_webcam.sh                      # webcam 0, release model, MuJoCo viewer + camera window
#   ./run_webcam.sh --video clip.mp4     # replay a video file in real time instead of the camera
#   ./run_webcam.sh --variant low_latency --source 1
#
# Two processes, like the official stack (webcam_stream.py -> ZMQ :5556 'pose' v3 -> deploy):
#   1) scripts/sim_teleop.py   (mjpython)  SONIC encoder/decoder + MuJoCo G1 + passive viewer
#   2) scripts/webcam_sender.py (python)   camera -> MediaPipe -> SMPL-24 -> ZMQ publisher
# Any other argument is forwarded to webcam_sender.py (e.g. --crop, --legs lock, --model heavy).
set -euo pipefail
cd "$(dirname "$0")"

VARIANT=release
SIM_ARGS=()
SENDER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --manual-stream) SIM_ARGS+=(--auto-stream -1); shift ;;
    --sim-record) SIM_ARGS+=(--record "$2"); shift 2 ;;
    *) SENDER_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -x .venv/bin/mjpython ]]; then
  echo "missing .venv — see README.md (python3.12 -m venv .venv && pip install -r requirements.txt)" >&2
  exit 1
fi
for f in models/$VARIANT/model_encoder.onnx models/pose_landmarker_full.task; do
  [[ -f "$f" ]] || { echo "missing $f — run: .venv/bin/python scripts/download_models.py --sample" >&2; exit 1; }
done

echo "[run_webcam] starting SONIC sim ($VARIANT) with MuJoCo viewer ..."
.venv/bin/mjpython scripts/sim_teleop.py --variant "$VARIANT" ${SIM_ARGS[@]+"${SIM_ARGS[@]}"} &
SIM_PID=$!
trap 'kill $SIM_PID 2>/dev/null || true' EXIT INT TERM
sleep 2

echo "[run_webcam] starting camera -> MediaPipe -> SMPL sender (q to quit, c to calibrate) ..."
.venv/bin/python scripts/webcam_sender.py ${SENDER_ARGS[@]+"${SENDER_ARGS[@]}"} || true
