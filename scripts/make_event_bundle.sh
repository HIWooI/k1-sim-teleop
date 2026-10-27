#!/usr/bin/env bash
# Build a self-contained folder to copy to the event laptop (macOS or Ubuntu): the code at HEAD, the model files
# git does not carry (K1 policies, K1 SONIC walk4 7000, planner, MediaPipe), the K1 description, the fallback and
# rehearsal videos, and k1demo.env (read by K1Demo.command). Then run scripts/install_event.sh on that laptop.
#   scripts/make_event_bundle.sh [OUT_DIR]        default dist/k1_event_bundle_<date>
# GEM-X (NVIDIA GPU only) is not bundled — its environment is installed separately (docs/setup.md).
# A bundle is for your own event machine only: it carries files that must not be redistributed (the SMPL-X derived
# human_joints_info, event videos) - never publish or share a bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-dist/k1_event_bundle_$(date +%Y%m%d)}
AI_SAPIENS_ROOT=${AI_SAPIENS_ROOT:-third_party/ai_sapiens}
PYBIN=${PYBIN:-.venv/bin/python}; [ -x "$PYBIN" ] || PYBIN=python3   # needs PyYAML
EVENT_DIR=${K1LIVE_EVENT_DIR:-${K1_EVENT_VIDEOS:-videos/event}}   # event media ("{event_dir}" in event.yaml)
[ -e "$OUT" ] && { echo "already exists: $OUT"; exit 2; }
MESHES=sonic_k1/overlay/gear_sonic/data/assets/robot_description/urdf/k1/meshes   # git-ignored (ROBOTIS asset)
[ -d "$MESHES" ] || { echo "no staged K1 meshes ($MESHES): run sonic_k1/scripts/stage_assets.py first"; exit 2; }
mkdir -p "$OUT"
git archive --format=tar HEAD | tar -x -C "$OUT"
copy() { mkdir -p "$OUT/$(dirname "$1")"; cp -R "$1" "$OUT/$1"; }
copy models/k1/k1s_walk4/deploy/k1s_walk4_007000           # hybrid walking (event default)
copy models/k1/k1s_la4/deploy_002000                       # sonic backend (optional)
copy models/planner/planner_sonic.onnx                     # GEAR-SONIC kinematic planner
copy models/pose_landmarker_full.task
copy models/pose_landmarker_heavy.task
copy models/human_joints_info.npz
copy models/config.json
for a in student_asset loco_armdriven_r2; do copy k1/assets/$a/exported/policy.onnx; done
copy vendor/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1   # G1 model: planner output (G1 qpos) -> K1
copy vendor/GR00T-WholeBodyControl/gear_sonic/data/human/human_joints_info.pkl
copy "$MESHES"
copy sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml
mkdir -p "$OUT/third_party/ai_sapiens"
cp -R "$AI_SAPIENS_ROOT/ai_sapiens_description" "$OUT/third_party/ai_sapiens/"
mkdir -p "$OUT/event" "$OUT/videos"
"$PYBIN" - "$OUT" "$EVENT_DIR" <<'PY'
import glob, os, shutil, sys, yaml
out, event_dir = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("k1live/web/event.yaml"))
for v in cfg.get("fallback_videos", []):
    src = v.replace("{event_dir}", event_dir)
    rel = os.path.relpath(src, event_dir)
    os.makedirs(os.path.join(out, "event", os.path.dirname(rel)), exist_ok=True)
    shutil.copy2(src, os.path.join(out, "event", rel)); print("fallback", rel)
PY
cp videos/k1live_rehearsal.mp4 "$OUT/videos/"
cat > "$OUT/k1demo.env" <<'ENV'
# read by K1Demo.command and scripts/install_event.sh ($ROOT = this folder)
export AI_SAPIENS_ROOT="$ROOT/third_party/ai_sapiens"
export K1LIVE_EVENT_DIR="$ROOT/event"
export HF_HUB_OFFLINE=1     # everything is in this folder / the GEM-X folder: never wait for huggingface.co at the venue
# GEM-X (NVIDIA only): GEM-X perception folder (demo_webcam.py, GEM-X/ inside; its GEM-X/.venv is used)
#export K1LIVE_GEMX_PERCEPTION="$ROOT/third_party/gemx_perception"
ENV
du -sh "$OUT"; echo "bundle: $OUT  ->  copy to the event laptop, then: bash scripts/install_event.sh"
