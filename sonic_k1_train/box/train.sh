#!/usr/bin/env bash
# On the GPU server: launch / export a K1 SONIC run inside $SONIC_K1_WORK (default $HOME/sonic_k1; env from setup_env.sh,
# own Isaac Lab v2.3.2 checkout). Training runs detached in tmux, stdout -> $SONIC_K1_WORK/logs.
#
#   train.sh smoke  [NAME [EXP]]                   16 envs, 30 iterations, foreground (extra hydra overrides after EXP)
#   train.sh train  NAME EXP NUM_ENVS ITERS [hydra overrides...]   detached (tmux session k1_NAME)
#   train.sh export CKPT                           ONNX (eval_agent_trl.py +export_onnx_only=true)
#   train.sh eval   CKPT MOTION_DIR [N_ENVS]       im_eval metrics (success rate, mpjpe) on a motion dir
set -euo pipefail
W=${SONIC_K1_WORK:-$HOME/sonic_k1}
E=$W/env
ROOT=$W/src/GR00T-WholeBodyControl
MOTIONS=${SONIC_K1_MOTIONS:-$W/pkg/k1_motions}
RUNS=$W/runs
export PATH="$E/bin:$PATH" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 SONIC_STRICT_ROBOT_ORDER=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json WANDB_MODE=disabled
export LD_PRELOAD=$E/lib/libusb-1.0.so.0  # open3d needs libusb; no system libusb -> conda-forge libusb in the env
mkdir -p "$W/logs" "$RUNS"
cmd="${1:?smoke|train|export|eval}"; shift

gpu_busy() {  # any compute process on the GPU that is not ours -> refuse (the GPU may be shared)
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | while IFS=, read -r pid rest; do
    [ -n "$pid" ] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$W/" && continue  # our own sonic_k1 runs
    echo "$pid,$rest"
  done
}

case "$cmd" in
  smoke)
    name="${1:-smoke}"; exp="${2:-sonic_k1_small}"; shift $(( $# > 2 ? 2 : $# ))
    [ -z "$(gpu_busy)" ] || { echo "GPU busy:"; gpu_busy; exit 3; }
    cd "$ROOT"
    python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/$exp \
      num_envs=16 headless=True use_wandb=false base_dir=$RUNS experiment_name=$name \
      ++algo.config.num_learning_iterations=30 \
      ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTIONS/robot "$@" 2>&1 | tee "$W/logs/${name}.log"
    ;;
  train)
    name="$1"; exp="$2"; n="$3"; it="$4"; shift 4
    [ -z "$(gpu_busy)" ] || { echo "GPU busy:"; gpu_busy; exit 3; }
    log="$W/logs/${name}.log"
    # the tmux server predates this shell (other users' sessions): pass the environment explicitly
    envs="PATH=$PATH OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 SONIC_STRICT_ROBOT_ORDER=1 WANDB_MODE=disabled"
    envs="$envs VK_ICD_FILENAMES=$VK_ICD_FILENAMES LD_PRELOAD=$LD_PRELOAD"
    tmux new-session -d -s "k1_$name" "cd $ROOT && env $envs python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/$exp \
      num_envs=$n headless=True use_wandb=false base_dir=$RUNS experiment_name=$name \
      ++algo.config.num_learning_iterations=$it \
      ++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTIONS/robot $* 2>&1 | tee $log; echo EXIT=\$? >> $log"
    echo "tmux k1_$name, log $log"
    ;;
  export)
    ckpt="$1"; shift
    cd "$ROOT"
    python gear_sonic/eval_agent_trl.py +checkpoint="$ckpt" +headless=True ++num_envs=1 +export_onnx_only=true "$@" 2>&1 \
      | tee "$W/logs/export_$(basename "$(dirname "$ckpt")")_$(basename "$ckpt" .pt).log"
    ;;
  eval)
    ckpt="$1"; mdir="$2"; n="${3:-64}"; shift 3 || shift $#
    cd "$ROOT"
    # tracking/eval (relaxed thresholds) brings back the G1 wrist names for ee_body_pos -> K1 bodies
    python gear_sonic/eval_agent_trl.py +checkpoint="$ckpt" +headless=True ++eval_callbacks=im_eval ++run_eval_loop=False \
      ++num_envs=$n "+manager_env/terminations=tracking/eval" \
      "++manager_env.terminations.ee_body_pos.params.body_names=[left_ankle_roll_link,right_ankle_roll_link,left_wrist_roll_rubber_hand,right_wrist_roll_rubber_hand]" \
      "++manager_env.commands.motion.motion_lib_cfg.motion_file=$mdir" "$@" 2>&1 \
      | tee "$W/logs/eval_$(basename "$(dirname "$ckpt")")_$(basename "$ckpt" .pt)_$(basename "$mdir").log"
    ;;
  *) echo "unknown command $cmd"; exit 2 ;;
esac
