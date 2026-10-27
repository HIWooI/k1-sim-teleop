#!/usr/bin/env bash
# Build linux/compat/zmq_v3_check.cpp against the official SONIC header
#   $SONIC_ROOT/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_packed_message_subscriber.hpp
#
#   bash linux/compat/build_zmq_v3_check.sh                # Linux: real libzmq + cppzmq -> $SONIC_WORK/bin/zmq_v3_check
#   bash linux/compat/build_zmq_v3_check.sh --shim -o OUT  # any OS, no libzmq: capture-file replay (ZMQ_SHIM_REPLAY=FILE)
# Needs nlohmann/json headers (apt nlohmann-json3-dev, brew nlohmann-json).
set -euo pipefail
# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SHIM=0; OUT="$SONIC_WORK/bin/zmq_v3_check"
while [ $# -gt 0 ]; do
  case "$1" in
    --shim) SHIM=1; shift ;;
    -o) OUT="$2"; shift 2 ;;
    --sonic-root) SONIC_ROOT="$2"; shift 2 ;;
    *) die "usage: $0 [--shim] [-o OUT] [--sonic-root DIR]" ;;
  esac
done
INC="$SONIC_ROOT/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface"
[ -f "$INC/zmq_packed_message_subscriber.hpp" ] || die "official header not found under $INC (set SONIC_ROOT)"
[ "$(uname -s)" = Darwin ] && SHIM=1

CXX="${CXX:-$(command -v g++ || command -v clang++ || command -v c++)}"
FLAGS=(-std=c++17 -O2 -Wall -I "$INC")
for d in /opt/homebrew/include /usr/local/include "$HOME/.local/include"; do [ -d "$d" ] && FLAGS+=(-I "$d"); done
LIBS=(-pthread)
if [ "$SHIM" = 1 ]; then
  FLAGS=(-I "$LINUX_DIR/compat/zmq_replay_shim" "${FLAGS[@]}")   # shim first so <zmq.hpp> resolves to it
else
  LIBS+=(-lzmq)
fi
mkdir -p "$(dirname "$OUT")"
"$CXX" "${FLAGS[@]}" "$LINUX_DIR/compat/zmq_v3_check.cpp" -o "$OUT" "${LIBS[@]}"
echo "$OUT ($([ "$SHIM" = 1 ] && echo "replay shim" || echo "libzmq"))"
