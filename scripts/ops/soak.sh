#!/usr/bin/env bash
# Event-mode rehearsal on the event PC without a camera: k1live.soak (service + display-like viewers, optional faults)
# on the rehearsal video, inside a systemd user scope (shared PC: CPU / memory caps), with the offline guard on
# (scripts/offline_guard), CPU telemetry (telemetry.py) and GPU telemetry (one `nvidia-smi dmon` process) next to it.
#   bash scripts/ops/soak.sh NAME MINUTES [extra k1live.soak args, e.g. --chaos default]
# env: CPUQ=400% MEM=8G OFFLINE=1 K1LIVE_PORT_OFFSET=1200 SOURCE=videos/k1live_rehearsal.mp4
#      CPU_ONLY=1 = GPU fence (cpu_only.env + nvidia_watch.py, no GPU telemetry): MediaPipe + llvmpipe rendering
# Output: logs/ops/NAME/ (soak.json / soak.md, session/, telemetry.csv, gpu_dmon.log, offline_attempts.log)
set -u
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
NAME=$1; MIN=$2; shift 2
OUT="$ROOT/logs/ops/$NAME"
[ -e "$OUT" ] && { echo "exists: $OUT"; exit 2; }
mkdir -p "$OUT"
if [ "${CPU_ONLY:-0}" = "1" ]; then   # GPU fence: Mesa llvmpipe, no GEM-X, watchdog kills on any NVIDIA use
  . "$ROOT/scripts/ops/cpu_only.env"
  python3 "$ROOT/scripts/ops/nvidia_watch.py" "$K1_FENCE" "$OUT/nvidia_watch.json" --seconds $(( ${MIN%.*} * 60 + 300 )) \
      >"$OUT/nvidia_watch.out" 2>&1 &
fi
[ -f "$ROOT/k1demo.env" ] && . "$ROOT/k1demo.env"
export MUJOCO_GL=${MUJOCO_GL:-egl} K1LIVE_PORT_OFFSET=${K1LIVE_PORT_OFFSET:-1200}
if [ "${OFFLINE:-1}" = "1" ]; then
  export PYTHONPATH="$ROOT/scripts/offline_guard${PYTHONPATH:+:$PYTHONPATH}" K1_OFFLINE_LOG="$OUT/offline_attempts.log"
fi
UNIT="k1event-$NAME-$$"
systemd-run --user --scope -q --unit="$UNIT" -p CPUQuota="${CPUQ:-400%}" -p MemoryMax="${MEM:-8G}" \
    nice -n 10 "$ROOT/.venv/bin/python" -m k1live.soak --minutes "$MIN" --source "$ROOT/${SOURCE:-videos/k1live_rehearsal.mp4}" \
    --out "$OUT" "$@" >"$OUT/soak.out" 2>&1 &
SPID=$!
sleep 1
CG="/sys/fs/cgroup$(systemctl --user show -p ControlGroup --value "$UNIT.scope" 2>/dev/null)"
python3 "$ROOT/scripts/ops/telemetry.py" "$OUT/telemetry.csv" --cgroup "$CG" --match "$ROOT" --gpu-every 0 \
    --seconds $(( ${MIN%.*} * 60 + 120 )) &
TPID=$!
DPID=""
if [ "${CPU_ONLY:-0}" != "1" ]; then   # GPU: one dmon process, local date + time columns
  nvidia-smi dmon -s pucm -d 5 -o DT >"$OUT/gpu_dmon.log" 2>&1 &
  DPID=$!
fi
wait "$SPID"
kill "$TPID" $DPID 2>/dev/null
python3 "$ROOT/scripts/ops/telemetry.py" --summary "$OUT/telemetry.csv" \
    $([ -n "$DPID" ] && echo --dmon "$OUT/gpu_dmon.log") >"$OUT/telemetry.json"
echo "offline attempts: $(grep -c '^[0-9]' "$OUT/offline_attempts.log" 2>/dev/null || echo 0)" | tee "$OUT/offline.txt"
cat "$OUT/soak.md" "$OUT/telemetry.json" "$OUT/nvidia_watch.out" 2>/dev/null
