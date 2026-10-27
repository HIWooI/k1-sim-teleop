#!/usr/bin/env bash
# GPU server: k1s_bones2 = k1s_bones it20000 resumed with walk4's feet reward + the GEM-X-like SMPL
# input augmentation (sonic_k1_small_la4_bones2.yaml, overlay mdp/smpl_gemx_aug.py; docs/training.md).
# A resumed run writes next to its checkpoint, so every run gets its own folder holding a copy of model_step_020000.pt.
#
#   bash bones2.sh install                 copy the bones2 overlay files from pkg/ into the SONIC checkout (src/)
#   bash bones2.sh smoke [ITERS]           16 envs, 20000 -> 20000+ITERS (default 50), foreground, own folder
#   bash bones2.sh train [ITERS]           4096 envs, 20000 -> 20000+ITERS (default 5000), detached (train.sh train)
#   bash bones2.sh export RUN STEP...      ONNX export (train.sh export) of RUN/model_step_STEP.pt
set -euo pipefail
W=${SONIC_K1_WORK:-$HOME/sonic_k1}
PKG=$W/pkg
ROOT=$W/src/GR00T-WholeBodyControl
EXP=sonic_k1_small_la4_bones2
SRC_RUN=$W/runs/TRL_K1_Track/k1s_bones-20261015_100757
SET=$PKG/k1_bones_train
OV=$PKG/sonic_k1_train/overlay/gear_sonic
FILES=(envs/manager_env/mdp/smpl_gemx_aug.py config/exp/manager/universal_token/all_modes/$EXP.yaml)

newrun() {   # $1 = name -> run folder with the it20000 checkpoint
  local d=$W/runs/TRL_K1_Track/$1-$(date -u +%Y%m%d_%H%M%S)
  mkdir -p "$d" && cp "$SRC_RUN/model_step_020000.pt" "$d/" && echo "$d"
}

case "${1:?install|smoke|train|export}" in
  install)
    for f in "${FILES[@]}"; do mkdir -p "$(dirname "$ROOT/gear_sonic/$f")"; cp "$OV/$f" "$ROOT/gear_sonic/$f"; echo "  $f"; done ;;
  smoke)
    it=$((20000 + ${2:-50}))
    d=$(newrun k1s_bones2_smoke)
    export PATH="$W/env/bin:$PATH" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 SONIC_STRICT_ROBOT_ORDER=1 WANDB_MODE=disabled
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json LD_PRELOAD=$W/env/lib/libusb-1.0.so.0
    cd "$ROOT"
    python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/$EXP \
      num_envs=16 headless=True use_wandb=false base_dir=$W/runs experiment_name=k1s_bones2_smoke \
      ++algo.config.num_learning_iterations=$it ++callbacks.model_save.save_frequency=25 \
      ++manager_env.commands.motion.motion_lib_cfg.motion_file=$SET/robot \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SET/smpl \
      ++checkpoint=$d/model_step_020000.pt +resume=true "${@:3}" 2>&1 | tee "$W/logs/k1s_bones2_smoke.log" ;;
  train)
    it=$((20000 + ${2:-5000}))
    d=$(newrun k1s_bones2)
    echo "run $d"
    SONIC_K1_MOTIONS=$SET bash $PKG/sonic_k1_train/box/train.sh train k1s_bones2 $EXP 4096 $it \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SET/smpl \
      ++checkpoint=$d/model_step_020000.pt +resume=true "${@:3}" ;;
  export)
    run=$2; shift 2
    for s in "$@"; do
      S6=$(printf %06d "$s")
      [ -f "$run/exported/model_step_${S6}_decoder.onnx" ] || bash $PKG/sonic_k1_train/box/train.sh export "$run/model_step_$S6.pt"
    done ;;
  *) echo "unknown command $1"; exit 2 ;;
esac
