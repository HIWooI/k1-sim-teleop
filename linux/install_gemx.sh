#!/usr/bin/env bash
# GEM-X (webcam -> SOMA body) venv for SONIC's live_camera_teleop example. Idempotent, no sudo.
# Follows docs/source/tutorials/live_camera_teleop.md "Step 1", with torch cu128 for RTX 5090/5080 (sm_120)
# instead of the doc's cu126.
#
#   bash linux/install_gemx.sh                       # find/clone GEM-X, build .venv, link soma_assets, pyzmq/scipy
#   bash linux/install_gemx.sh --gemx-root ~/GEM-X                    # reuse an existing checkout
#   bash linux/install_gemx.sh --check-only
#
# Steps: 1 locate/clone (--recursive)  2 uv + .venv (Python 3.12)  3 torch/torchvision cu128 (sm_120 check)
#        4 third_party/soma (-e) + its LFS assets  5 scripts/install_env.sh  6 inputs/soma_assets -> absolute link
#        7 pyzmq scipy  8 import smoke test (gem, SomaLayer, SONIC packer via $SONIC_ROOT)
# The GEM-X checkpoint itself downloads from Hugging Face on the first webcam_stream.py run (not here).
set -euo pipefail
# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CHECK_ONLY=0; TORCH_INDEX="https://download.pytorch.org/whl/cu128"
while [ $# -gt 0 ]; do
  case "$1" in
    --gemx-root) GEMX_ROOT="$2"; GEMX_EXPLICIT=1; shift 2 ;;
    --work) SONIC_WORK="$2"; ENV_FILE="$SONIC_WORK/sonic_env.sh"; shift 2 ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
run() { if [ "$CHECK_ONLY" = 1 ]; then echo "    (check-only) would run: $*"; else "$@"; fi; }
export PATH="$HOME/.local/bin:$PATH"
FAIL=0

# --------------------------------------------------------------------------- 1 locate / clone
step "1 locate or clone GEM-X"
if [ -z "${GEMX_EXPLICIT:-}" ] && [ ! -d "$GEMX_ROOT/gem" ]; then
  for c in "$HOME/GEM-X" "$SONIC_WORK/GEM-X"; do
    [ -d "$c/gem" ] && { GEMX_ROOT="$c"; break; }
  done
fi
if [ -d "$GEMX_ROOT/gem" ]; then
  ok "GEM-X at $GEMX_ROOT ($(git -C "$GEMX_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  [ -d "$GEMX_ROOT/third_party/soma/.git" ] || [ -f "$GEMX_ROOT/third_party/soma/.git" ] \
    || run git -C "$GEMX_ROOT" submodule update --init --recursive
  step_done skip
else
  run git clone --recursive "$GEMX_REPO_URL" "$GEMX_ROOT"
  step_done ok
fi
[ "$CHECK_ONLY" = 1 ] || persist_env GEMX_ROOT "$GEMX_ROOT"
PY="$GEMX_ROOT/.venv/bin/python"

# --------------------------------------------------------------------------- 2 uv + venv
step "2 uv + .venv (Python 3.12)"
if ! have uv; then
  log "installing uv -> ~/.local/bin"
  [ "$CHECK_ONLY" = 1 ] || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [ -x "$PY" ]; then ok ".venv exists ($("$PY" -V 2>&1))"; step_done skip
else ( cd "$GEMX_ROOT" && run uv venv .venv --python 3.12 ); step_done ok; fi
uvpip() { run uv pip install --python "$PY" "$@"; }

# --------------------------------------------------------------------------- 3 torch cu128
step "3 torch with sm_120 support (cu128+)"
TORCH_INFO="$("$PY" - 2>/dev/null <<'EOF' || true
import torch
arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
print(torch.__version__, torch.version.cuda, int(torch.cuda.is_available()), int("sm_120" in arch))
EOF
)"
read -r T_VER T_CUDA T_AVAIL T_SM120 <<<"${TORCH_INFO:-none none 0 0}"
if [ "$T_AVAIL" = 1 ] && [ "$T_SM120" = 1 ]; then ok "torch $T_VER (CUDA $T_CUDA) has sm_120 kernels"; step_done skip
else
  [ "$T_VER" = none ] || warn "torch $T_VER (CUDA $T_CUDA, cuda_available=$T_AVAIL, sm_120=$T_SM120) -> reinstalling from $TORCH_INDEX"
  uvpip torch torchvision --index-url "$TORCH_INDEX"
  step_done ok
fi

# --------------------------------------------------------------------------- 4 soma
step "4 third_party/soma (+ LFS assets)"
if [ -d "$GEMX_ROOT/third_party/soma" ]; then
  SOMA_STAMP="$GEMX_ROOT/.venv/.soma_installed"
  if [ -f "$SOMA_STAMP" ]; then ok "soma installed ($(cat "$SOMA_STAMP"))"
  else uvpip -e "$GEMX_ROOT/third_party/soma" && { [ "$CHECK_ONLY" = 1 ] || date '+%F %T' > "$SOMA_STAMP"; }; fi
  PTR="$(git -C "$GEMX_ROOT/third_party/soma" lfs ls-files 2>/dev/null | awk '$2=="-"' | wc -l)"
  if [ "$PTR" -gt 0 ]; then log "git lfs pull in third_party/soma ($PTR pointer files; ~800 MB)"; ( cd "$GEMX_ROOT/third_party/soma" && run git lfs pull ); fi
  step_done ok
else err "third_party/soma missing (clone was not --recursive?)"; FAIL=1; step_done fail; fi

# --------------------------------------------------------------------------- 5 install_env.sh
step "5 GEM-X scripts/install_env.sh"
STAMP="$GEMX_ROOT/.venv/.install_env_done"
if [ -f "$STAMP" ]; then ok "already ran ($(cat "$STAMP"))"; step_done skip
elif [ -f "$GEMX_ROOT/scripts/install_env.sh" ]; then
  if [ "$CHECK_ONLY" = 1 ]; then echo "    (check-only) would run scripts/install_env.sh"; step_done skip
  else
    # shellcheck disable=SC1091
    if ( cd "$GEMX_ROOT" && source .venv/bin/activate && bash scripts/install_env.sh ); then
      date '+%F %T' > "$STAMP"; step_done ok
    else err "install_env.sh failed"; FAIL=1; step_done fail; fi
    # install_env.sh may pull a default (non-cu128) torch: re-check sm_120
    "$PY" -c "import torch,sys; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null \
      || { warn "install_env.sh replaced torch; reinstalling cu128 build"; uvpip --reinstall torch torchvision --index-url "$TORCH_INDEX"; }
  fi
else warn "scripts/install_env.sh not found"; step_done warn; fi

# --------------------------------------------------------------------------- 6 soma_assets link
step "6 inputs/soma_assets -> absolute link"
LINK="$GEMX_ROOT/inputs/soma_assets"; TARGET="$GEMX_ROOT/third_party/soma/assets"
if [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$TARGET" ] && [ -d "$LINK/" ]; then ok "$LINK -> $TARGET"; step_done skip
else run mkdir -p "$GEMX_ROOT/inputs"; run ln -sfn "$TARGET" "$LINK"; step_done ok; fi   # relative target would dangle

# --------------------------------------------------------------------------- 7 bridge deps
step "7 pyzmq scipy (ZMQ bridge)"
if "$PY" -c "import zmq, scipy" 2>/dev/null; then ok "pyzmq/scipy present"; step_done skip
else uvpip pyzmq scipy; step_done ok; fi

# --------------------------------------------------------------------------- 8 smoke test
step "8 import smoke test"
if [ "$CHECK_ONLY" = 0 ]; then
  if ( cd "$GEMX_ROOT" && SONIC_ROOT="$SONIC_ROOT" "$PY" - <<EOF
import sys, torch
sys.path[:0] = ["$GEMX_ROOT", "$SONIC_ROOT", "$SONIC_ROOT/gear_sonic/examples/live_camera_teleop"]
from gem.utils.soma_utils.soma_layer import SomaLayer   # noqa: F401
import scripts.demo.demo_soma_onnx                      # noqa: F401
from soma_to_smpl import _get_packer
pack = _get_packer("$SONIC_ROOT")
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE", "| packer:", pack.__module__)
EOF
  ); then ok "GEM-X + SONIC bridge import OK"; step_done ok
  else err "import smoke test failed (see traceback above)"; FAIL=1; step_done fail; fi
else step_done skip; fi

step_summary
[ "$FAIL" = 0 ] || exit 2
printf '\n%sGEM-X ready.%s  GEMX_ROOT=%s (saved in %s)\n' "$C_G" "$C_0" "$GEMX_ROOT" "$ENV_FILE"
echo "Camera check:  bash $LINUX_DIR/run_sim_teleop.sh --camera-check    Live:  bash $LINUX_DIR/run_sim_teleop.sh --mode webcam"
