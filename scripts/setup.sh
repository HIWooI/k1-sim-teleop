#!/usr/bin/env bash
# One-shot, idempotent setup of sonic_sim on macOS (no sudo, no git-lfs needed).
#  - venv with Homebrew python3.12
#  - shallow clone of NVlabs/GR00T-WholeBodyControl into vendor/ (LFS smudge skipped)
#  - fetch only the LFS files we need over media.githubusercontent.com:
#      G1 MuJoCo meshes (scene_43dof.xml), reference motion CSVs, SMPL rest skeleton pkl
#  - GEAR-SONIC ONNX (release / low_latency / sonic_v1_1) + sample data from Hugging Face
#  - MediaPipe pose_landmarker_{full,heavy}.task
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PYTHON:-/opt/homebrew/bin/python3.12}

[[ -x .venv/bin/python ]] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

mkdir -p vendor models videos logs
if [[ ! -d vendor/GR00T-WholeBodyControl ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git vendor/GR00T-WholeBodyControl
fi
REPO=vendor/GR00T-WholeBodyControl
MEDIA=https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/main

fetch_lfs() {  # $1 = repo-relative path; downloads only if the file is still an LFS pointer
  local f="$REPO/$1"
  if [[ ! -f "$f" ]] || head -c 40 "$f" | grep -q "git-lfs"; then
    curl -sSfL -o "$f.tmp" "$MEDIA/$1" && mv "$f.tmp" "$f"
  fi
}

G1=gear_sonic/data/robot_model/model_data/g1
for m in $(grep -o 'file="[^"]*"' "$REPO/$G1/g1_29dof_with_hand.xml" | sed 's/file="//;s/"//' | sort -u); do
  fetch_lfs "$G1/meshes/$m" &
done
wait
for d in "$REPO"/gear_sonic_deploy/reference/example/*/; do
  for f in "$d"*.csv; do
    fetch_lfs "${f#$REPO/}" &
  done
done
wait
fetch_lfs gear_sonic/data/human/human_joints_info.pkl

.venv/bin/python scripts/download_models.py --variants release low_latency sonic_v1_1 --sample
for v in full heavy; do
  [[ -f models/pose_landmarker_$v.task ]] || curl -sSfL -o models/pose_landmarker_$v.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$v/float16/latest/pose_landmarker_$v.task"
done
echo "setup done: $ROOT"
