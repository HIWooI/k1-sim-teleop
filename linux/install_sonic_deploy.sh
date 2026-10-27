#!/usr/bin/env bash
# Official SONIC deploy (C++ / TensorRT) + MuJoCo sim venv + models on an Ubuntu x86_64 box with an
# RTX 5090 / 5080 (Blackwell). Idempotent: every step checks first and skips what is already there.
# Never runs sudo: root-only steps are written to $SONIC_WORK/sudo_steps.sh for you to review and run.
#
#   bash linux/install_sonic_deploy.sh                 # full install / re-check (safe to re-run)
#   bash linux/install_sonic_deploy.sh --check-only    # report only, change nothing
#
# Steps (upstream docs: getting_started/{installation_deploy,quickstart,download_models}.md):
#   1 preflight   Ubuntu / x86_64 / driver (CUDA >= 12.8 for sm_120) / CUDA toolkit
#   2 apt deps    what upstream scripts/install_deps.sh would apt-get  -> sudo_steps.sh if missing
#   3 user deps   just -> ~/.local/bin, ONNX Runtime 1.16.3 -> ~/.local/onnxruntime, cppzmq/json headers
#                 (the user-space equivalents of install_deps.sh; no sudo needed)
#   4 TensorRT    10.13 TAR -> ~/TensorRT (extracts a TensorRT-10.13*.tar.gz it finds; download needs an NVIDIA login)
#   5 repo        clone NVlabs/GR00T-WholeBodyControl + git lfs pull
#   6 build       source scripts/setup_env.sh && just build   (target/release/g1_deploy_onnx_ref)
#   7 sim venv    install_scripts/install_mujoco_sim.sh -> .venv_sim (skipped if it already imports)
#   8 models      download_from_hf.py (release + planner), --low-latency, [--sample]
#   9 compat      build linux/compat/zmq_v3_check against the official ZMQ subscriber header
#
# Do NOT run upstream scripts/install_deps.sh on a 50-series box before CUDA 12.8 is installed: when it finds no
# CUDA it apt-installs CUDA 12.4, which cannot run on sm_120. The upstream Docker image (CUDA 12.4.1) has the same problem.
set -euo pipefail
# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CHECK_ONLY=0; WITH_SAMPLE=0; WITH_V11=0; REBUILD=0; SKIP_LFS=0
usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; cat <<EOF

options:
  --work DIR          work dir (default \$SONIC_WORK = $SONIC_WORK)
  --sonic-root DIR    SONIC checkout (default \$SONIC_WORK/GR00T-WholeBodyControl)
  --trt-root DIR      TensorRT root (default \$TensorRT_ROOT = $TensorRT_ROOT)
  --sample            also download sample_data (~4 MB)
  --sonic-v1-1        also download the sonic_v1_1 policy
  --rebuild           just clean && just build
  --skip-lfs          skip 'git lfs pull' (only if you know the LFS files are present)
  --check-only        report what is missing, change nothing
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --work) SONIC_WORK="$2"; ENV_FILE="$SONIC_WORK/sonic_env.sh"; SUDO_FILE="$SONIC_WORK/sudo_steps.sh"; shift 2 ;;
    --sonic-root) SONIC_ROOT="$2"; shift 2 ;;
    --trt-root) TensorRT_ROOT="$2"; shift 2 ;;
    --sample) WITH_SAMPLE=1; shift ;;
    --sonic-v1-1) WITH_V11=1; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --skip-lfs) SKIP_LFS=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done
DEPLOY="$SONIC_ROOT/gear_sonic_deploy"
mkdir -p "$SONIC_WORK"
BLOCKERS=()
blocker() { BLOCKERS+=("$*"); err "$*"; }
run() { if [ "$CHECK_ONLY" = 1 ]; then echo "    (check-only) would run: $*"; else "$@"; fi; }
export PATH="$HOME/.local/bin:$PATH"

# --------------------------------------------------------------------------- 1 preflight
step "1 preflight (OS / GPU / CUDA)"
# shellcheck source=/dev/null
. /etc/os-release 2>/dev/null || true
[ "$(uname -s)" = Linux ] || die "this script is for the Linux box (got $(uname -s))"
[ "$(uname -m)" = x86_64 ] || warn "arch $(uname -m): TensorRT 10.13 requirement is for x86_64 (Jetson needs 10.7)"
case "${VERSION_ID:-}" in 20.04|22.04|24.04) ok "Ubuntu ${VERSION_ID}" ;; *) warn "OS ${PRETTY_NAME:-unknown}: upstream lists Ubuntu 20.04/22.04/24.04" ;; esac
if have nvidia-smi; then
  GPU="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1)"
  DRV_CUDA="$(nvidia_cuda_version)"
  ok "GPU: ${GPU:-?}  (driver supports CUDA ${DRV_CUDA:-?})"
  if [ -n "$DRV_CUDA" ] && ! version_ge "$DRV_CUDA" "$MIN_CUDA"; then
    blocker "driver supports only CUDA $DRV_CUDA; RTX 50xx needs a driver for CUDA >= $MIN_CUDA (R570+)"
    sudo_step "# NVIDIA driver for CUDA >= $MIN_CUDA (Blackwell). Pick the open kernel module flavor recommended by ubuntu-drivers:"
    sudo_step "ubuntu-drivers list --gpgpu; apt-get install -y nvidia-driver-570-open   # then reboot"
  fi
else
  blocker "nvidia-smi not found (NVIDIA driver missing)"
fi
NVCC_V="$(nvcc_version)"
if [ -n "$NVCC_V" ] && version_ge "$NVCC_V" "$MIN_CUDA"; then
  ok "CUDA toolkit $NVCC_V ($(nvcc_path))"
  CUDA_HOME_GUESS="$(dirname "$(dirname "$(nvcc_path)")")"
  [ "$CHECK_ONLY" = 1 ] || persist_env CUDAToolkit_ROOT "$CUDA_HOME_GUESS"
else
  blocker "CUDA toolkit >= $MIN_CUDA not found (nvcc: ${NVCC_V:-none})"
  sudo_step "# CUDA toolkit 12.8 from the NVIDIA apt repo (keyring for ubuntu${VERSION_ID//./})"
  sudo_step "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update"
  sudo_step "apt-get install -y cuda-toolkit-12-8"
fi
step_done "$([ ${#BLOCKERS[@]} -eq 0 ] && echo ok || echo warn)"

# --------------------------------------------------------------------------- 2 apt deps
step "2 system packages (apt, root)"
# exactly what upstream install_deps.sh apt-installs for x86_64 Ubuntu, plus tmux (run_sim_teleop.sh) and v4l-utils (camera check)
APT_PKGS=(build-essential clang cmake git git-lfs pkg-config patchelf zlib1g-dev curl wget
          libyaml-cpp-dev libeigen3-dev libmsgpack-dev libzmq3-dev nlohmann-json3-dev libgtest-dev tmux v4l-utils)
if have dpkg-query; then
  MISSING="$(apt_missing "${APT_PKGS[@]}")"
  if [ -n "$MISSING" ]; then
    warn "missing apt packages: $MISSING"
    sudo_step "apt-get update && apt-get install -y $MISSING"
    step_done warn
  else ok "all ${#APT_PKGS[@]} apt packages present"; step_done ok; fi
else warn "dpkg-query not found; skipping package check"; step_done skip; fi

# --------------------------------------------------------------------------- 3 user-space deps
step "3 user-space deps (just, onnxruntime, cppzmq)"
if have just; then ok "just $(just --version 2>/dev/null | awk '{print $2}')"
else
  JUST_VERSION="1.36.0"
  log "installing just $JUST_VERSION -> ~/.local/bin (upstream install_deps.sh puts it in /usr/local/bin with sudo)"
  run mkdir -p "$HOME/.local/bin"
  if [ "$CHECK_ONLY" = 0 ]; then
    curl -fsSL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
      | tar -xz -C "$HOME/.local/bin" just
  fi
fi
ORT_DIR=""
for p in /opt/onnxruntime /usr/local/onnxruntime /usr/lib/onnxruntime "$HOME/.local/onnxruntime"; do
  [ -d "$p" ] && { ORT_DIR="$p"; break; }
done
if [ -n "$ORT_DIR" ]; then ok "ONNX Runtime at $ORT_DIR"
else
  ORT_V="1.16.3"   # the version upstream install_deps.sh pins
  log "installing ONNX Runtime $ORT_V -> ~/.local/onnxruntime (scripts/setup_env.sh searches there)"
  if [ "$CHECK_ONLY" = 0 ]; then
    tmpd="$(mktemp -d)"
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_V}/onnxruntime-linux-x64-${ORT_V}.tgz" | tar -xz -C "$tmpd"
    mkdir -p "$HOME/.local"; mv "$tmpd"/onnxruntime-linux-x64-* "$HOME/.local/onnxruntime"; rm -rf "$tmpd"
  fi
  ORT_DIR="$HOME/.local/onnxruntime"
fi
# cppzmq (header-only zmq.hpp): apt package on 22.04+ is cppzmq-dev. The deploy CMake does not look at upstream's
# third_party/cppzmq fallback, so without root we drop the headers into ~/.local/include and export CPATH (gcc/clang honor it).
if [ -f /usr/include/zmq.hpp ]; then ok "cppzmq: /usr/include/zmq.hpp"
elif [ -f "$HOME/.local/include/zmq.hpp" ]; then ok "cppzmq: ~/.local/include/zmq.hpp"
else
  log "cppzmq headers -> ~/.local/include (or: sudo apt-get install -y cppzmq-dev)"
  if [ "$CHECK_ONLY" = 0 ]; then
    tmpd="$(mktemp -d)"
    git clone -q --depth 1 https://github.com/zeromq/cppzmq.git "$tmpd/cppzmq"
    mkdir -p "$HOME/.local/include"; cp "$tmpd"/cppzmq/zmq.hpp "$tmpd"/cppzmq/zmq_addon.hpp "$HOME/.local/include/"; rm -rf "$tmpd"
  fi
fi
# shellcheck disable=SC2016
[ -f /usr/include/zmq.hpp ] || [ "$CHECK_ONLY" = 1 ] || persist_line cpath "export CPATH=$HOME/.local/include\${CPATH:+:\$CPATH}"
# setup_env.sh adds TensorRT and CUDA to LD_LIBRARY_PATH; a user-space ONNX Runtime is not symlinked into
# /usr/local/lib (that needs sudo), so make it findable for the run scripts.
# shellcheck disable=SC2016
[ "$CHECK_ONLY" = 1 ] || persist_line ort_ld "export LD_LIBRARY_PATH=$ORT_DIR/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
step_done ok

# --------------------------------------------------------------------------- 4 TensorRT
step "4 TensorRT $TRT_REQUIRED (TAR)"
TRT_V="$(trt_version "$TensorRT_ROOT")"
if [ -z "$TRT_V" ]; then
  TAR="$(find "$HOME/Downloads" "$SONIC_WORK" "$HOME" -maxdepth 1 -name "TensorRT-${TRT_REQUIRED}*.tar.gz" 2>/dev/null | head -n1)"
  if [ -n "$TAR" ] && [ -e "$TensorRT_ROOT" ]; then
    blocker "$TensorRT_ROOT exists but holds no TensorRT; move it away (or pass --trt-root) so $TAR can be extracted there"
    TAR=""
  fi
  if [ -n "$TAR" ]; then
    log "extracting $TAR -> $TensorRT_ROOT (archive ~10 GB unpacked; a few minutes)"
    if [ "$CHECK_ONLY" = 0 ]; then
      tmpd="$(mktemp -d "$SONIC_WORK/trt.XXXX")"
      tar -xzf "$TAR" -C "$tmpd"
      mv "$tmpd"/TensorRT-* "$TensorRT_ROOT" && rmdir "$tmpd"
      TRT_V="$(trt_version "$TensorRT_ROOT")"
    fi
  fi
fi
if [ -z "$TRT_V" ]; then
  blocker "TensorRT not found at $TensorRT_ROOT"
  cat >&2 <<EOF
      Download (NVIDIA developer login): https://developer.nvidia.com/tensorrt/download/10x
        -> "TensorRT 10.13 GA for Linux x86_64 and CUDA 12.0 to 12.9 TAR Package" (TAR, not DEB)
      Put the .tar.gz in ~/Downloads or $SONIC_WORK and re-run this script (it extracts to $TensorRT_ROOT).
EOF
  step_done fail
elif [ "$(echo "$TRT_V" | cut -d. -f1,2)" != "$TRT_REQUIRED" ]; then
  blocker "TensorRT $TRT_V at $TensorRT_ROOT, upstream requires exactly $TRT_REQUIRED on x86_64 (other versions give wrong inference)"
  step_done fail
else
  ok "TensorRT $TRT_V at $TensorRT_ROOT"
  [ "$CHECK_ONLY" = 1 ] || persist_env TensorRT_ROOT "$TensorRT_ROOT"
  step_done ok
fi

# --------------------------------------------------------------------------- 5 repo + LFS
step "5 SONIC repo + LFS"
if [ -d "$SONIC_ROOT/.git" ]; then ok "repo at $SONIC_ROOT ($(git -C "$SONIC_ROOT" rev-parse --short HEAD))"
else run git clone "$SONIC_REPO_URL" "$SONIC_ROOT"; fi
if [ "$SKIP_LFS" = 0 ] && [ -d "$SONIC_ROOT/.git" ]; then
  if have git-lfs || git lfs version >/dev/null 2>&1; then
    run git -C "$SONIC_ROOT" lfs install --local >/dev/null
    run git -C "$SONIC_ROOT" lfs pull
    PTR="$(git -C "$SONIC_ROOT" lfs ls-files 2>/dev/null | awk '$2=="-"' | wc -l)"
    if [ "$PTR" -eq 0 ]; then ok "all LFS objects present"; else warn "$PTR LFS file(s) still pointers"; fi
  else blocker "git-lfs missing (apt package git-lfs)"; fi
fi
[ "$CHECK_ONLY" = 1 ] || persist_env SONIC_ROOT "$SONIC_ROOT"
step_done ok

# --------------------------------------------------------------------------- 6 build
step "6 build g1_deploy_onnx_ref (just build)"
BIN="$DEPLOY/target/release/g1_deploy_onnx_ref"
if [ ${#BLOCKERS[@]} -gt 0 ]; then
  warn "skipping build: ${#BLOCKERS[@]} blocker(s) above"; step_done skip
elif [ -n "$(apt_missing libzmq3-dev nlohmann-json3-dev libyaml-cpp-dev libeigen3-dev libmsgpack-dev cmake clang)" ]; then
  warn "skipping build: apt packages missing (run sudo_steps.sh first)"; step_done skip
elif [ -x "$BIN" ] && [ "$REBUILD" = 0 ]; then
  ok "already built: $BIN (deploy.sh re-runs an incremental 'just build' anyway; --rebuild to force)"; step_done skip
elif [ "$CHECK_ONLY" = 1 ]; then echo "    (check-only) would build in $DEPLOY"; step_done skip
else
  (
    cd "$DEPLOY"
    set +eu   # setup_env.sh is not -u clean
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    # shellcheck source=/dev/null
    source scripts/setup_env.sh
    set -eu
    [ "$REBUILD" = 1 ] && just clean
    just build 2>&1 | tee "$SONIC_WORK/build_deploy.log" | grep -E "error|warning: unused|✅|Built target" || true
  )
  if [ -x "$BIN" ]; then ok "built $BIN"; step_done ok
  else blocker "build failed, see $SONIC_WORK/build_deploy.log"; step_done fail; fi
fi

# --------------------------------------------------------------------------- 7 sim venv
step "7 MuJoCo sim venv (.venv_sim)"
SIMPY="$SONIC_ROOT/.venv_sim/bin/python"
if [ -x "$SIMPY" ] && "$SIMPY" -c "import mujoco, tyro, gear_sonic, unitree_sdk2py" 2>/dev/null; then
  ok ".venv_sim imports mujoco/gear_sonic/unitree_sdk2py"; step_done skip
elif [ ! -d "$SONIC_ROOT" ]; then step_done skip
else
  # upstream script rm -rf's .venv_sim and reinstalls (uv + Python 3.10, ~2-5 min)
  if ( cd "$SONIC_ROOT" && run bash install_scripts/install_mujoco_sim.sh ); then step_done ok
  else blocker "install_mujoco_sim.sh failed"; step_done fail; fi
fi

# --------------------------------------------------------------------------- 8 models
step "8 models (HF nvidia/GEAR-SONIC)"
need_dl() { [ -f "$DEPLOY/$1" ] && [ "$(stat -c %s "$DEPLOY/$1" 2>/dev/null || echo 0)" -gt 100000 ] && return 1; return 0; }
if [ -x "$SIMPY" ] && [ -d "$SONIC_ROOT" ]; then
  "$SIMPY" -c "import huggingface_hub" 2>/dev/null || run "$SIMPY" -m pip install -q huggingface_hub 2>/dev/null \
    || run uv pip install --python "$SIMPY" -q huggingface_hub
  if need_dl policy/release/model_encoder.onnx || need_dl policy/release/model_decoder.onnx || need_dl planner/target_vel/V2/planner_sonic.onnx; then
    ( cd "$SONIC_ROOT" && run "$SIMPY" download_from_hf.py )
  else ok "release policy + planner present"; fi
  if need_dl policy/low_latency/model_encoder.onnx; then ( cd "$SONIC_ROOT" && run "$SIMPY" download_from_hf.py --low-latency --no-planner )
  else ok "low_latency policy present"; fi
  if [ "$WITH_V11" = 1 ] && need_dl policy/sonic_v1_1/model_encoder.onnx; then ( cd "$SONIC_ROOT" && run "$SIMPY" download_from_hf.py --sonic-v1-1 --no-planner ); fi
  if [ "$WITH_SAMPLE" = 1 ] && [ ! -d "$SONIC_ROOT/sample_data" ]; then ( cd "$SONIC_ROOT" && run "$SIMPY" download_from_hf.py --sample ); fi
  step_done ok
else warn "no .venv_sim python yet; models skipped"; step_done skip; fi

# --------------------------------------------------------------------------- 9 compat checker
step "9 ZMQ v3 compat checker (official subscriber header)"
if [ "$CHECK_ONLY" = 0 ] && [ -d "$DEPLOY" ] && [ -z "$(apt_missing libzmq3-dev nlohmann-json3-dev)" ]; then
  if bash "$LINUX_DIR/compat/build_zmq_v3_check.sh" >/dev/null; then ok "built $SONIC_WORK/bin/zmq_v3_check"; step_done ok
  else warn "zmq_v3_check build failed (optional)"; step_done warn; fi
else step_done skip; fi

# --------------------------------------------------------------------------- summary
step_summary
flush_sudo_steps
if [ ${#BLOCKERS[@]} -gt 0 ]; then
  printf '\n%sBLOCKERS (%d):%s\n' "$C_R" "${#BLOCKERS[@]}" "$C_0"; printf '  - %s\n' "${BLOCKERS[@]}"
  echo "Fix them (sudo_steps.sh / TensorRT download) and re-run: bash $0"
  exit 2
fi
printf '\n%sSONIC deploy ready.%s env: source %s\n' "$C_G" "$C_0" "$ENV_FILE"
echo "Next: bash $LINUX_DIR/install_gemx.sh   then   bash $LINUX_DIR/run_sim_teleop.sh --mode replay|webcam|remote"
