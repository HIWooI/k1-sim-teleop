#!/usr/bin/env bash
# SONIC training stack on the Linux GPU box: Isaac Sim (pip) + Isaac Lab 2.3.2 + gear_sonic[training].
# Native install for RTX 5090/5080 (CUDA 12.8+, torch cu128). Idempotent, never runs sudo.
# Upstream: docs/source/getting_started/installation_training.md (Python 3.11, Isaac Lab 2.3+).
#
#   bash linux/install_sonic_training.sh [--accept-eula]     # venv -> $SONIC_WORK/env_isaaclab
#   bash linux/install_sonic_training.sh --check-only
#
# Steps: 1 preflight (Ubuntu 22.04+, driver CUDA >= 12.8, apt cmake/build-essential)  2 Python 3.11 venv (uv)
#        3 isaacsim[all,extscache]==$ISAACSIM_VERSION from pypi.nvidia.com  4 torch 2.7.0 / torchvision 0.22.0 cu128
#        5 IsaacLab v2.3.2 clone + ./isaaclab.sh --install  6 pip install -e gear_sonic[training]
#        7 check_environment.py --training
# --accept-eula sets OMNI_KIT_ACCEPT_EULA=YES (NVIDIA Omniverse EULA) for the non-interactive first start;
# without it the first Isaac Sim import asks interactively. Only pass it if you accept the EULA.
set -euo pipefail
# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ISAACLAB_TAG="v2.3.2"; ISAACSIM_VERSION="5.1.0"; TORCH_SPEC=(torch==2.7.0 torchvision==0.22.0)
VENV="$SONIC_WORK/env_isaaclab"; ISAACLAB_ROOT="$SONIC_WORK/IsaacLab"; CHECK_ONLY=0; EULA=0
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --isaaclab-root) ISAACLAB_ROOT="$2"; shift 2 ;;
    --isaaclab-tag) ISAACLAB_TAG="$2"; shift 2 ;;
    --isaacsim-version) ISAACSIM_VERSION="$2"; shift 2 ;;
    --accept-eula) EULA=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
run() { if [ "$CHECK_ONLY" = 1 ]; then echo "    (check-only) would run: $*"; else "$@"; fi; }
export PATH="$HOME/.local/bin:$PATH"
PY="$VENV/bin/python"; PIP=("$PY" -m pip install)
BLOCK=0

step "1 preflight"
# shellcheck source=/dev/null
. /etc/os-release 2>/dev/null || true
[ "$(uname -s)" = Linux ] || die "Linux only"
version_ge "${VERSION_ID:-0}" 22.04 || { err "Isaac Sim 5.x needs Ubuntu 22.04+ (GLIBC 2.35), got ${PRETTY_NAME:-?}"; BLOCK=1; }
DRV_CUDA="$(nvidia_cuda_version)"
if [ -n "$DRV_CUDA" ] && version_ge "$DRV_CUDA" "$MIN_CUDA"; then ok "driver supports CUDA $DRV_CUDA"
else err "driver CUDA ${DRV_CUDA:-none} < $MIN_CUDA (see install_sonic_deploy.sh preflight)"; BLOCK=1; fi
MISSING="$(apt_missing cmake build-essential git git-lfs libusb-1.0-0)"   # libusb: open3d (motion lib) dlopens it
[ -z "$MISSING" ] || { sudo_step "apt-get update && apt-get install -y $MISSING   # isaaclab.sh --install builds extensions"; BLOCK=1; }
[ -d "$SONIC_ROOT/gear_sonic" ] || { err "SONIC checkout missing at $SONIC_ROOT (run install_sonic_deploy.sh step 5)"; BLOCK=1; }
step_done "$([ "$BLOCK" = 0 ] && echo ok || echo fail)"
if [ "$BLOCK" = 1 ]; then flush_sudo_steps; exit 2; fi

step "2 Python 3.11 venv"
have uv || { [ "$CHECK_ONLY" = 1 ] || curl -LsSf https://astral.sh/uv/install.sh | sh; }
if [ -x "$PY" ]; then ok "$VENV ($("$PY" -V))"; step_done skip
else run uv venv --python 3.11 --seed "$VENV"; step_done ok; fi   # --seed: plain pip inside, as the Isaac Lab docs use

step "3 Isaac Sim $ISAACSIM_VERSION (pip)"
if "$PY" -c "import importlib.metadata as m; assert m.version('isaacsim').startswith('$ISAACSIM_VERSION')" 2>/dev/null; then
  ok "isaacsim $ISAACSIM_VERSION installed"; step_done skip
else run "${PIP[@]}" "isaacsim[all,extscache]==$ISAACSIM_VERSION" --extra-index-url https://pypi.nvidia.com; step_done ok; fi

step "4 torch cu128 (sm_120)"
if "$PY" -c "import torch,sys; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
  ok "torch $("$PY" -c 'import torch; print(torch.__version__)') has sm_120"; step_done skip
else run "${PIP[@]}" -U "${TORCH_SPEC[@]}" --index-url https://download.pytorch.org/whl/cu128; step_done ok; fi

step "5 Isaac Lab $ISAACLAB_TAG"
[ -d "$ISAACLAB_ROOT/.git" ] || run git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
if "$PY" -c "import isaaclab" 2>/dev/null; then ok "isaaclab importable"; step_done skip
else
  # shellcheck disable=SC1091
  if ( cd "$ISAACLAB_ROOT" && source "$VENV/bin/activate" && run ./isaaclab.sh --install ); then step_done ok
  else err "isaaclab.sh --install failed"; step_done fail; exit 2; fi
fi
[ "$EULA" = 1 ] && [ "$CHECK_ONLY" = 0 ] && persist_env OMNI_KIT_ACCEPT_EULA YES

step "6 gear_sonic[training]"
if "$PY" -c "import importlib.metadata as m; m.version('gear_sonic'); import hydra, trl, accelerate" 2>/dev/null; then ok "gear_sonic + training deps"; step_done skip
elif ( cd "$SONIC_ROOT" && run "${PIP[@]}" -e "gear_sonic/[training]" ); then step_done ok
else err "pip install -e gear_sonic/[training] failed"; step_done fail; exit 2; fi
# Pins found on the training GPU server (sonic_k1_train; docs/training.md): the latest wandb/datasets
# pull protobuf>=5, click>=8.2, psutil>=7, websockets>=13 (isaacsim-kernel pins the older ones) and pyarrow>=2x
# (needs numpy 2; Isaac Sim needs numpy 1.26). gear_sonic also imports rich + tensordict on the training path.
run "${PIP[@]}" "wandb==0.18.7" "protobuf==4.25.9" "click==8.1.7" "psutil==5.9.8" "websockets==12.0" \
  "pyarrow==19.0.1" "datasets==3.6.0" rich termcolor "tensordict>=0.8,<0.9"
# isaaclab/gear_sonic installs can pull a non-cu128 torch back in
"$PY" -c "import torch,sys; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null \
  || run "${PIP[@]}" --force-reinstall --no-deps "${TORCH_SPEC[@]}" --index-url https://download.pytorch.org/whl/cu128

step "7 check_environment.py --training"
if [ "$CHECK_ONLY" = 0 ]; then
  if ( cd "$SONIC_ROOT" && "$PY" check_environment.py --training ); then step_done ok; else warn "environment check reported problems"; step_done warn; fi
else step_done skip; fi

[ "$CHECK_ONLY" = 1 ] || { persist_env SONIC_TRAIN_PY "$PY"; persist_env ISAACLAB_ROOT "$ISAACLAB_ROOT"; }
step_summary
flush_sudo_steps
echo "Smoke test (G1, upstream): cd $SONIC_ROOT && $PY gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release num_envs=16 headless=True ++algo.config.num_learning_iterations=5"
echo "K1: bash $LINUX_DIR/train_sonic_k1.sh prepare|smoke|train|export"
