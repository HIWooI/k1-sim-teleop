# shellcheck shell=bash
# Shared helpers for linux/*.sh — sourced, never executed.
#
# Layout on the Linux box (override with env vars or the scripts' flags):
#   $SONIC_WORK                    ~/sonic            work dir, env file, logs
#   $SONIC_ROOT                    $SONIC_WORK/GR00T-WholeBodyControl
#   $GEMX_ROOT                     $SONIC_WORK/GEM-X  (an existing checkout in ~/GEM-X is auto-detected)
#   $TensorRT_ROOT                 ~/TensorRT         (TensorRT 10.13 TAR, x86_64)
#   $SONIC_WORK/sonic_env.sh       exports written by the install scripts, sourced by the run scripts
#   $SONIC_WORK/sudo_steps.sh      root-only steps collected by the install scripts (we never run sudo)

LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$LINUX_DIR/.." && pwd)"
export LINUX_DIR BUNDLE_ROOT

: "${SONIC_WORK:=$HOME/sonic}"
ENV_FILE="$SONIC_WORK/sonic_env.sh"
SUDO_FILE="$SONIC_WORK/sudo_steps.sh"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
: "${SONIC_ROOT:=$SONIC_WORK/GR00T-WholeBodyControl}"
: "${GEMX_ROOT:=$SONIC_WORK/GEM-X}"
: "${TensorRT_ROOT:=$HOME/TensorRT}"
export SONIC_WORK SONIC_ROOT GEMX_ROOT TensorRT_ROOT

# shellcheck disable=SC2034  # constants used by the scripts that source this file
SONIC_REPO_URL="https://github.com/NVlabs/GR00T-WholeBodyControl.git"
# shellcheck disable=SC2034
GEMX_REPO_URL="https://github.com/NVlabs/GEM-X.git"
# shellcheck disable=SC2034
MIN_CUDA="12.8"        # RTX 5090/5080 (Blackwell, sm_120) need CUDA 12.8+
# shellcheck disable=SC2034
TRT_REQUIRED="10.13"   # upstream: x86_64 TensorRT 10.13 is *required* (other versions -> wrong inference)

if [ -t 1 ]; then C_B=$'\033[1;34m'; C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[1;31m'; C_0=$'\033[0m'
else C_B=""; C_G=""; C_Y=""; C_R=""; C_0=""; fi

log()  { printf '%s[%s]%s %s\n' "$C_B" "$(date +%H:%M:%S)" "$C_0" "$*"; }
ok()   { printf '%s  ✔%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '%s  !%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
err()  { printf '%s  ✘%s %s\n' "$C_R" "$C_0" "$*" >&2; }
die()  { err "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# version_ge A B  -> true if A >= B (dotted versions)
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

# ---------------------------------------------------------------- timed steps
STEP_NAMES=(); STEP_SECS=(); STEP_STATES=()
_STEP_T0=0; _STEP_NAME=""
step() { _STEP_NAME="$1"; _STEP_T0=$(date +%s); log "▶ $1"; }
step_done() {  # step_done [ok|skip|warn|fail]
  local state="${1:-ok}" dt=$(( $(date +%s) - _STEP_T0 ))
  STEP_NAMES+=("$_STEP_NAME"); STEP_SECS+=("$dt"); STEP_STATES+=("$state")
}
step_summary() {
  local i
  printf '\n%-46s %-6s %8s\n' "step" "state" "elapsed"
  for i in "${!STEP_NAMES[@]}"; do
    printf '%-46s %-6s %7ss\n' "${STEP_NAMES[$i]}" "${STEP_STATES[$i]}" "${STEP_SECS[$i]}"
  done
}

# ---------------------------------------------------------------- sudo steps (collected, never executed)
SUDO_STEPS=()
sudo_step() { SUDO_STEPS+=("$*"); }
apt_missing() {  # apt_missing pkg... -> prints packages that are not installed
  local p
  for p in "$@"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || printf '%s ' "$p"
  done
}
flush_sudo_steps() {
  [ "${#SUDO_STEPS[@]}" -eq 0 ] && { rm -f "$SUDO_FILE"; return 0; }
  mkdir -p "$SONIC_WORK"
  {
    echo "#!/usr/bin/env bash"
    echo "# Root-only steps collected by $(basename "$0") on $(date '+%F %T')."
    echo "# Review, then run once:  sudo bash $SUDO_FILE   — afterwards re-run $(basename "$0")."
    echo "set -euo pipefail"
    printf '%s\n' "${SUDO_STEPS[@]}"
  } > "$SUDO_FILE"
  warn "${#SUDO_STEPS[@]} step(s) need root and were NOT run. Review and run:"
  printf '      sudo bash %s\n' "$SUDO_FILE" >&2
  sed -n '5,$p' "$SUDO_FILE" | sed 's/^/        /' >&2
}

# ---------------------------------------------------------------- persisted env
persist_env() {  # persist_env NAME VALUE  -> upsert "export NAME=VALUE" in $ENV_FILE
  local name="$1" value="$2" tmp
  mkdir -p "$SONIC_WORK"
  touch "$ENV_FILE"
  tmp="$(mktemp)"
  grep -v "^export ${name}=" "$ENV_FILE" > "$tmp" || true
  printf 'export %s=%q\n' "$name" "$value" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
  export "$name=$value"
}
persist_line() {  # persist_line KEY 'raw shell line'  -> upsert a raw line (expanded when sourced) tagged "# KEY"
  local key="$1" line="$2" tmp
  mkdir -p "$SONIC_WORK"
  touch "$ENV_FILE"
  tmp="$(mktemp)"
  grep -v "# ${key}\$" "$ENV_FILE" > "$tmp" || true
  printf '%s  # %s\n' "$line" "$key" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
}

# ---------------------------------------------------------------- GPU / CUDA probes
nvidia_cuda_version() {  # max CUDA version the driver supports (nvidia-smi header), empty if none
  have nvidia-smi || return 0
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1
}
nvcc_path() {
  local c
  if have nvcc; then command -v nvcc; return 0; fi
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-1[2-9]*/bin/nvcc; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 0
}
nvcc_version() {  # e.g. 12.8
  local n; n="$(nvcc_path)"
  [ -n "$n" ] && "$n" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1
  return 0
}
trt_version() {  # TensorRT version under $1 (default $TensorRT_ROOT), e.g. 10.13.0; empty if not found
  local root="${1:-$TensorRT_ROOT}" lib h
  # libnvinfer.so.10.13.0 is the most reliable source (10.x headers may define NV_TENSORRT_MAJOR via TRT_*_ENTERPRISE)
  lib="$(find "$root/lib" "$root/targets/x86_64-linux-gnu/lib" -maxdepth 1 -name 'libnvinfer.so.*.*.*' 2>/dev/null | head -n1)"
  if [ -n "$lib" ]; then echo "${lib##*libnvinfer.so.}"; return 0; fi
  h="$root/include/NvInferVersion.h"
  [ -f "$h" ] || return 0
  awk '$1=="#define" && $3 ~ /^[0-9]+$/ {v[$2]=$3}
       END{ma=v["NV_TENSORRT_MAJOR"]; if(ma=="")ma=v["TRT_MAJOR_ENTERPRISE"];
           mi=v["NV_TENSORRT_MINOR"]; if(mi=="")mi=v["TRT_MINOR_ENTERPRISE"];
           pa=v["NV_TENSORRT_PATCH"]; if(pa=="")pa=v["TRT_PATCH_ENTERPRISE"];
           if(ma!="") printf "%s.%s.%s\n", ma, mi, pa}' "$h"
}

# Pick a display for the MuJoCo / OpenCV windows when started over SSH.
#   prints "DISPLAY XAUTHORITY" or nothing (=> headless)
detect_display() {
  if [ -n "${DISPLAY:-}" ]; then echo "$DISPLAY ${XAUTHORITY:-}"; return 0; fi
  [ -S /tmp/.X11-unix/X0 ] || [ -S /tmp/.X11-unix/X1 ] || return 0
  local d=":0" xa=""
  [ -S /tmp/.X11-unix/X0 ] || d=":1"
  for xa in "$HOME/.Xauthority" /run/user/"$(id -u)"/gdm/Xauthority /run/user/"$(id -u)"/.mutter-Xwaylandauth.*; do
    [ -f "$xa" ] && { echo "$d $xa"; return 0; }
  done
  echo "$d"
}
