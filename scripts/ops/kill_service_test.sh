#!/usr/bin/env bash
# The service itself is killed (SIGKILL) under K1Demo.command, as a crash would: the launcher must start a new one,
# the new one must clear the dead one's children (camera / GEM-X / simulation still holding the ports) and get back
# to the live view. Then the GEM-X worker alone is killed (the sender reloads it). No browser, rehearsal video as camera.
#   bash scripts/ops/kill_service_test.sh [OUT_DIR]          (inside the bundle folder; offset 1200 by default)
# CPU_ONLY=1: GPU fence (cpu_only.env + nvidia_watch.py); GEM-X is then unavailable, so the worker kill step is skipped.
set -u
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
OUT=${1:-$ROOT/logs/ops/kill_service}
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
if [ "${CPU_ONLY:-0}" = "1" ]; then   # GPU fence: Mesa llvmpipe, no GEM-X, watchdog kills on any NVIDIA use
  . "$ROOT/scripts/ops/cpu_only.env"
  python3 "$ROOT/scripts/ops/nvidia_watch.py" "$K1_FENCE" "$OUT/nvidia_watch.json" --seconds 600 \
      >"$OUT/nvidia_watch.out" 2>&1 &
fi
export K1LIVE_PORT_OFFSET=${K1LIVE_PORT_OFFSET:-1200}
BASE="http://127.0.0.1:$((8700 + K1LIVE_PORT_OFFSET))"
PYV="$ROOT/.venv/bin/python"
state() { curl -s -m 2 "$BASE/api/state" | "$PYV" -c "import json,sys
try: s=json.load(sys.stdin)
except Exception: print('down'); sys.exit()
p=s.get('perception') or {}
print(s.get('view'), p.get('active'), (s.get('sim') or {}).get('ctrl_hz'))" 2>/dev/null || echo down; }
wait_live() {   # seconds until view == live (max $1 s)
  local t0; t0=$(date +%s.%N)
  for _ in $(seq 1 $(( $1 * 4 ))); do
    case "$(state)" in live\ *) "$PYV" -c "import time;print(round(time.time()-$t0,1))"; return 0;; esac
    sleep 0.25
  done
  echo "timeout"; return 1
}
procs() { pgrep -f "$PYV -m k1live\.(service|sender|sender_gemx|sim_server)|$ROOT/k1live/gemx_worker.py" -a | \
          sed -E "s#$ROOT/##g" | awk '{print $1, $2, $3, $4}'; }
K1DEMO_NO_BROWSER=1 K1DEMO_NO_CAMERA=1 K1DEMO_ARGS="--source $ROOT/videos/k1live_rehearsal.mp4 --start --auto" \
    setsid bash "$ROOT/K1Demo.command" >"$OUT/k1demo.out" 2>&1 </dev/null &
KPID=$!
{
  echo "first start -> live: $(wait_live 120) s   ($(state))"
  sleep 20
  echo "--- before kill:"; procs
  SVC=$(pgrep -f "$PYV -m k1live\.service" | head -1)
  kill -KILL "$SVC"; echo "killed service pid $SVC (SIGKILL) at $(date +%T)"
  sleep 0.5; echo "--- right after kill (orphans):"; procs
  echo "service killed -> live again: $(wait_live 180) s   ($(state))"
  sleep 5; echo "--- after recovery:"; procs
  echo "sender_gemx: $(pgrep -fc "$PYV -m k1live\.sender_gemx")  gemx_worker: $(pgrep -fc "$ROOT/k1live/gemx_worker.py")  sim_server: $(pgrep -fc "$PYV -m k1live\.sim_server")"
  if [ "${CPU_ONLY:-0}" != "1" ]; then
    sleep 20
    W=$(pgrep -f "$ROOT/k1live/gemx_worker.py" | head -1)
    kill -KILL "$W"; echo "killed gemx_worker pid $W (SIGKILL) at $(date +%T)"
    sleep 1
    echo "gemx worker killed -> live again: $(wait_live 120) s   ($(state))"
    sleep 5; echo "--- after worker recovery:"; procs
  fi
} 2>&1 | tee "$OUT/result.txt"
kill -TERM -- -"$KPID" 2>/dev/null
for _ in $(seq 1 60); do curl -s -m 1 "$BASE/api/health" >/dev/null 2>&1 || break; sleep 0.5; done
sleep 3
echo "left after stop: $(procs | wc -l) processes" | tee -a "$OUT/result.txt"
