#!/usr/bin/env bash
# Train SONIC on ROBOTIS K1 (23 DOF) from the sonic_k1/ package, then export ONNX.
# Needs linux/install_sonic_training.sh first. Upstream recipe: docs/source/user_guide/new_embodiments.md.
#
#   bash linux/train_sonic_k1.sh prepare          # apply sonic_k1/ to $SONIC_ROOT (idempotent)
#   bash linux/train_sonic_k1.sh smoke            # 16 envs, 5 iterations, headless: does K1 load and step?
#   bash linux/train_sonic_k1.sh train [--gpus N] [--detach]   # full run (accelerate launch when N > 1)
#   bash linux/train_sonic_k1.sh resume [--detach]             # continue the latest run of the same experiment
#   bash linux/train_sonic_k1.sh export [--checkpoint PT]      # eval_agent_trl.py +export_onnx_only=true
#   bash linux/train_sonic_k1.sh status                        # latest run dir, last.pt, exported ONNX
#
# sonic_k1/ contract (bundle root, or --k1-dir):
#   patches/*.patch     git-apply'd in $SONIC_ROOT in name order (skipped when already applied)
#                       e.g. robots/__init__.py, robot_mapping in modular_tracking_env_cfg.py, order_converter.py
#   overlay/**          copied into $SONIC_ROOT as-is: gear_sonic/envs/manager_env/robots/k1.py,
#                       gear_sonic/config/exp/manager/universal_token/all_modes/sonic_k1.yaml,
#                       gear_sonic/data/assets/robot_description/{urdf/k1/,mjcf/k1.xml}, ...
#   data/robot/**.pkl   K1 motion_lib PKLs (MuJoCo joint/body order; loaded recursively)
#   data/smpl/**.pkl    optional SMPL motions with the same keys (needed for the SMPL / webcam encoder)
#   train.env           shell vars: EXP (hydra +exp), NUM_ENVS, ITERS, PROJECT, EXTRA (more hydra overrides)
# Runs land in $SONIC_WORK/k1_runs/<project>/<exp>-<timestamp>/ (last.pt, exported/*.onnx);
# exported ONNX are also copied to $SONIC_WORK/k1_export/<timestamp>/ with SHA256SUMS.
set -euo pipefail
# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CMD="${1:-}"; [ -n "$CMD" ] && shift || true
K1_DIR="$BUNDLE_ROOT/sonic_k1"; GPUS=1; DETACH=0; CKPT=""; RUNS="$SONIC_WORK/k1_runs"
EXP="manager/universal_token/all_modes/sonic_k1"; NUM_ENVS=4096; ITERS=""; PROJECT="TRL_K1_Track"; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --k1-dir) K1_DIR="$2"; shift 2 ;;
    --exp) EXP_CLI="$2"; shift 2 ;;
    --num-envs) NUM_ENVS_CLI="$2"; shift 2 ;;
    --iters) ITERS_CLI="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
# shellcheck source=/dev/null
[ -f "$K1_DIR/train.env" ] && source "$K1_DIR/train.env"
EXP="${EXP_CLI:-$EXP}"; NUM_ENVS="${NUM_ENVS_CLI:-$NUM_ENVS}"; ITERS="${ITERS_CLI:-$ITERS}"
PY="${SONIC_TRAIN_PY:-$SONIC_WORK/env_isaaclab/bin/python}"
[ -x "$PY" ] || die "training venv missing ($PY): bash $LINUX_DIR/install_sonic_training.sh"
[ -d "$SONIC_ROOT/gear_sonic" ] || die "no SONIC checkout at $SONIC_ROOT"
MOTION="$K1_DIR/data/robot"; SMPL="$K1_DIR/data/smpl"
OVR=("++manager_env.commands.motion.motion_lib_cfg.motion_file=$MOTION" "base_dir=$RUNS" "project_name=$PROJECT")
[ -d "$SMPL" ] && OVR+=("++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=$SMPL")
# shellcheck disable=SC2206  # EXTRA is a whitespace-separated list of hydra overrides by contract
[ -n "$EXTRA" ] && OVR+=($EXTRA)

prepare() {
  [ -d "$K1_DIR" ] || die "sonic_k1 package not found at $K1_DIR (--k1-dir)"
  local p n=0
  for p in "$K1_DIR"/patches/*.patch; do
    [ -e "$p" ] || continue
    if git -C "$SONIC_ROOT" apply --reverse --check "$p" 2>/dev/null; then ok "already applied: $(basename "$p")"
    elif git -C "$SONIC_ROOT" apply --check "$p"; then git -C "$SONIC_ROOT" apply "$p"; ok "applied: $(basename "$p")"; n=$((n + 1))
    else die "patch does not apply: $p (SONIC at $(git -C "$SONIC_ROOT" rev-parse --short HEAD); check the base commit)"; fi
  done
  if [ -d "$K1_DIR/overlay" ]; then
    cp -R "$K1_DIR/overlay/." "$SONIC_ROOT/"
    ok "overlay copied: $(find "$K1_DIR/overlay" -type f | wc -l | tr -d ' ') files"
  fi
  local exp_yaml="$SONIC_ROOT/gear_sonic/config/exp/$EXP.yaml"
  [ -f "$exp_yaml" ] || die "experiment config missing: $exp_yaml (EXP=$EXP)"
  [ -n "$(find "$MOTION" -name '*.pkl' 2>/dev/null | head -n1)" ] || die "no K1 motion PKLs under $MOTION"
  ok "exp $EXP, $(find "$MOTION" -name '*.pkl' | wc -l | tr -d ' ') motion PKLs$([ -d "$SMPL" ] && echo ", SMPL $(find "$SMPL" -name '*.pkl' | wc -l | tr -d ' ')")"
  git -C "$SONIC_ROOT" status --short | sed 's/^/    /' | head -n 20
}

launch() {  # launch NAME CMD... -> foreground with tee, or detached in tmux
  local name="$1"; shift
  local log; log="$SONIC_WORK/logs/k1_${name}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$(dirname "$log")" "$RUNS"
  log "$name: $*"
  log "log: $log"
  if [ "$DETACH" = 1 ]; then
    have tmux || die "--detach needs tmux"
    # a tmux server started earlier (other sessions) does not inherit this shell's exports (OMNI_KIT_ACCEPT_EULA,
    # PATH, ...): the first box run sat at the EULA prompt -> re-source the env file inside the session
    tmux new-session -d -s "k1_$name" "{ [ -f $(printf %q "$ENV_FILE") ] && source $(printf %q "$ENV_FILE"); true; } && cd $(printf %q "$SONIC_ROOT") && $(printf '%q ' "$@") 2>&1 | tee $(printf %q "$log")"
    ok "running in tmux session k1_$name (tmux attach -t k1_$name); SSH can disconnect safely"
  else
    ( cd "$SONIC_ROOT" && "$@" 2>&1 | tee "$log" )
  fi
}

TRAIN=()
train_cmd() {  # train_cmd [hydra overrides...] -> TRAIN array
  TRAIN=("$PY")
  [ "$GPUS" -gt 1 ] && TRAIN=("$PY" -m accelerate.commands.launch --num_processes="$GPUS")
  TRAIN+=(gear_sonic/train_agent_trl.py "+exp=$EXP" "$@" "${OVR[@]}")
}

latest_ckpt() { find "$RUNS" -name last.pt -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n1 | cut -d' ' -f2-; }

case "$CMD" in
  prepare) prepare ;;
  smoke)
    prepare
    train_cmd num_envs=16 headless=True ++algo.config.num_learning_iterations=5
    launch smoke "${TRAIN[@]}" ;;
  train)
    prepare
    IT=(headless=True); [ -n "$ITERS" ] && IT+=("++algo.config.num_learning_iterations=$ITERS")
    train_cmd "num_envs=$NUM_ENVS" "${IT[@]}"
    launch train "${TRAIN[@]}" ;;
  resume)
    train_cmd "num_envs=$NUM_ENVS" headless=True +resume=True
    launch resume "${TRAIN[@]}" ;;
  export)
    CKPT="${CKPT:-$(latest_ckpt)}"
    [ -f "$CKPT" ] || die "no checkpoint (looked for the newest last.pt under $RUNS; pass --checkpoint)"
    DETACH=0
    launch export "$PY" gear_sonic/eval_agent_trl.py "+checkpoint=$CKPT" +headless=True ++num_envs=1 +export_onnx_only=true "${OVR[0]}"
    OUT="$SONIC_WORK/k1_export/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
    cp "$(dirname "$CKPT")"/exported/*.onnx "$OUT"/ && cp "$CKPT" "$OUT"/
    ( cd "$OUT" && sha256sum ./* > SHA256SUMS )
    ok "exported -> $OUT"; ls -la "$OUT"
    echo "copy to the Mac:  scp -r $(whoami)@$(hostname):$OUT ./models/k1_sonic/" ;;
  status)
    C="$(latest_ckpt)"
    echo "runs: $RUNS"; echo "latest last.pt: ${C:-none}"
    [ -n "$C" ] && ls -la "$(dirname "$C")" "$(dirname "$C")/exported" 2>/dev/null
    tmux ls 2>/dev/null | grep '^k1_' || true ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
