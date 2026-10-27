#!/usr/bin/env bash
# Official SONIC sim teleop in one tmux session (3 panes), startable over SSH:
#   pane 0  MuJoCo sim       run_sim_loop.py via linux/sim_loop_keys.py (also takes '9' from the terminal)
#   pane 1  C++ deploy       ./deploy.sh [--cp ...] --input-type zmq --zmq-host <host> sim
#   pane 2  pose input       GEM-X webcam_stream.py | soma_pt_to_sonic_v3.py replay | stream monitor (remote sender)
#
#   bash linux/run_sim_teleop.sh --mode webcam [--source 0]          # GEM-X on this box's camera
#   bash linux/run_sim_teleop.sh --mode video --source clip.mp4      # GEM-X on a video file
#   bash linux/run_sim_teleop.sh --mode replay --pt hpe_results.pt   # saved GEM-X result, no camera
#   bash linux/run_sim_teleop.sh --mode remote --zmq-host <mac-ip>   # Mac runs scripts/webcam_sender.py (MediaPipe)
#   bash linux/run_sim_teleop.sh --mode remote --tunnel              # same, through `ssh -R 5556:localhost:5556 box`
#   bash linux/run_sim_teleop.sh --camera-check [--source 0]         # GEM-X --kp-only --show, nothing else
#
# Keys (official order): pane 1 ']' start control -> sim '9' drop robot (MuJoCo window or pane 0) ->
#   stand relaxed and upright in front of the camera (pane 2 shows soma=yes) -> pane 1 ENTER = ZMQ streaming on/off,
#   pane 1 'O' = emergency stop. --auto does the first three (Proceed prompt, ']', '9'); --auto-stream also ENTER.
# tmux: Ctrl-b o = next pane, Ctrl-b d = detach (keeps running), `tmux attach -t sonic_teleop` to come back.
set -euo pipefail
# shellcheck source=linux/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODE=webcam; SOURCE=0; PT=""; ZMQ_HOST=""; TUNNEL=0; POLICY=release; DISPLAY_MODE=auto
AUTO=0; AUTO_STREAM=0; WINDOW=30; SMOOTH=0.8; SESSION=sonic_teleop; KILL=0; ATTACH=1; CAMERA_CHECK=0
usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'

options:
  --policy release|low_latency   low_latency = SMPL 4-frame lookahead policy (lower teleop lag)
  --window N --smooth S          GEM-X rolling window (default 30) / smoothing (default 0.8), tutorial values
  --display auto|headless        auto: $DISPLAY, else the box's local X display :0; headless: no windows
  --auto / --auto-stream         press the start keys for you (sim only)
  --session NAME  --kill         tmux session name / kill an existing session of that name first
  --no-attach                    leave the session detached
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --pt) PT="$2"; shift 2 ;;
    --zmq-host) ZMQ_HOST="$2"; shift 2 ;;
    --tunnel) TUNNEL=1; shift ;;
    --policy) POLICY="$2"; shift 2 ;;
    --window) WINDOW="$2"; shift 2 ;;
    --smooth) SMOOTH="$2"; shift 2 ;;
    --display) DISPLAY_MODE="$2"; shift 2 ;;
    --auto) AUTO=1; shift ;;
    --auto-stream) AUTO=1; AUTO_STREAM=1; shift ;;
    --session) SESSION="$2"; shift 2 ;;
    --kill) KILL=1; shift ;;
    --no-attach) ATTACH=0; shift ;;
    --camera-check) CAMERA_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

DEPLOY="$SONIC_ROOT/gear_sonic_deploy"
SIMPY="$SONIC_ROOT/.venv_sim/bin/python"
GEMXPY="$GEMX_ROOT/.venv/bin/python"
EXAMPLES="$SONIC_ROOT/gear_sonic/examples/live_camera_teleop"

# --------------------------------------------------------------------------- display
DISP=""; XAUTH=""
if [ "$DISPLAY_MODE" = auto ]; then read -r DISP XAUTH <<<"$(detect_display)" || true; fi
ENV_X="true"
if [ -n "$DISP" ]; then ENV_X="export DISPLAY=$DISP${XAUTH:+ XAUTHORITY=$XAUTH}"; fi

# --------------------------------------------------------------------------- camera check only
if [ "$CAMERA_CHECK" = 1 ]; then
  [ -x "$GEMXPY" ] || die "GEM-X venv missing: bash $LINUX_DIR/install_gemx.sh"
  [ -n "$DISP" ] || die "--camera-check needs a display (none found; log in on the box's desktop or use ssh -X)"
  eval "$ENV_X"
  ls /dev/video* 2>/dev/null || warn "no /dev/video* devices"
  cd "$GEMX_ROOT"
  exec env GEMX_ROOT="$GEMX_ROOT" SONIC_ROOT="$SONIC_ROOT" "$GEMXPY" "$EXAMPLES/webcam_stream.py" --source "$SOURCE" --kp-only --show
fi

# --------------------------------------------------------------------------- preflight
have tmux || die "tmux missing (apt package tmux; see \$SONIC_WORK/sudo_steps.sh)"
[ -x "$DEPLOY/deploy.sh" ] || die "SONIC deploy not found at $DEPLOY: bash $LINUX_DIR/install_sonic_deploy.sh"
[ -x "$SIMPY" ] || die ".venv_sim missing: bash $LINUX_DIR/install_sonic_deploy.sh"
case "$POLICY" in
  release) DEPLOY_ARGS="" ;;
  low_latency) DEPLOY_ARGS="--cp policy/low_latency/model --obs-config policy/low_latency/observation_config.yaml"
               [ -f "$DEPLOY/policy/low_latency/model_encoder.onnx" ] || die "low_latency policy not downloaded" ;;
  *) die "--policy release|low_latency" ;;
esac
case "$MODE" in
  webcam|video)
    [ -x "$GEMXPY" ] || die "GEM-X venv missing: bash $LINUX_DIR/install_gemx.sh"
    [ "$MODE" = video ] && [ ! -f "$SOURCE" ] && die "--mode video needs --source <file>"
    ZMQ_HOST="localhost"
    SHOW=""; [ -n "$DISP" ] && SHOW="--show"
    INPUT_CMD="cd $(printf %q "$GEMX_ROOT") && export GEMX_ROOT=$(printf %q "$GEMX_ROOT") SONIC_ROOT=$(printf %q "$SONIC_ROOT") && $ENV_X && .venv/bin/python $(printf %q "$EXAMPLES/webcam_stream.py") --source $(printf %q "$SOURCE") --stream-sonic --window $WINDOW --smooth $SMOOTH $SHOW"
    READY_PAT="soma=yes" ;;
  replay)
    [ -x "$GEMXPY" ] || die "GEM-X venv missing: bash $LINUX_DIR/install_gemx.sh"
    [ -f "$PT" ] || die "--mode replay needs --pt <hpe_results.pt>"
    ZMQ_HOST="localhost"
    INPUT_CMD="cd $(printf %q "$GEMX_ROOT") && export GEMX_ROOT=$(printf %q "$GEMX_ROOT") SONIC_ROOT=$(printf %q "$SONIC_ROOT") && .venv/bin/python $(printf %q "$EXAMPLES/soma_pt_to_sonic_v3.py") --pt $(printf %q "$PT") --fps 30 --loop"
    READY_PAT="pack_pose_message|fallback packer|frame" ;;
  remote)
    if [ "$TUNNEL" = 1 ]; then
      # 127.0.0.1, not "localhost": the deploy trusts timestamp_monotonic only for the literal host "localhost",
      # and through a tunnel that clock is the Mac's -> wrong data-delay stats / absent-stream alarm.
      ZMQ_HOST="127.0.0.1"
      log "tunnel mode: on the Mac run  ssh -N -R 5556:localhost:5556 $(whoami)@<this-box>  (sender binds Mac:5556)"
    fi
    [ -n "$ZMQ_HOST" ] || die "--mode remote needs --zmq-host <sender-ip> or --tunnel"
    [ "$ZMQ_HOST" = localhost ] && warn "--zmq-host localhost with a remote sender: use the IP (or 127.0.0.1 for a tunnel)"
    CHECK_BIN="$SONIC_WORK/bin/zmq_v3_check"
    INPUT_CMD="cd $(printf %q "$LINUX_DIR") && while true; do $(printf %q "$SIMPY") zmq_v3_compat.py listen --host $ZMQ_HOST --seconds 5 | tail -4; [ -x $(printf %q "$CHECK_BIN") ] && $(printf %q "$CHECK_BIN") --host $ZMQ_HOST --count 100 --idle 3 | tail -2; echo '--- (Ctrl-C to stop monitoring; deploy keeps its own connection)'; sleep 10; done"
    READY_PAT="RESULT COMPATIBLE" ;;
  *) die "--mode webcam|video|replay|remote" ;;
esac
SIM_ARGS=""; [ -z "$DISP" ] && SIM_ARGS="--headless" && warn "no X display: sim runs headless (no viewer); '9' still works in pane 0"

# --------------------------------------------------------------------------- tmux session
if tmux has-session -t "$SESSION" 2>/dev/null; then
  [ "$KILL" = 1 ] || die "tmux session '$SESSION' exists: attach with 'tmux attach -t $SESSION' or re-run with --kill"
  tmux kill-session -t "$SESSION"
fi
LOGDIR="$SONIC_WORK/logs/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"
# address panes by their stable ids: tmux renumbers pane *indexes* by position after each split
PANE=()
PANE[0]="$(tmux new-session -d -s "$SESSION" -n teleop -x 240 -y 60 -P -F '#{pane_id}')"   # sim (left top)
PANE[1]="$(tmux split-window -h -t "${PANE[0]}" -P -F '#{pane_id}')"                       # deploy (right)
PANE[2]="$(tmux split-window -v -t "${PANE[0]}" -P -F '#{pane_id}')"                       # input (left bottom)
for p in 0 1 2; do tmux pipe-pane -t "${PANE[$p]}" -o "cat >> $LOGDIR/pane$p.log"; done
send() { tmux send-keys -t "${PANE[$1]}" "$2" Enter; }
key() { tmux send-keys -t "${PANE[$1]}" "$2"; }

ENVSRC="[ -f $(printf %q "$ENV_FILE") ] && source $(printf %q "$ENV_FILE")"
send 0 "$ENVSRC; $ENV_X; cd $(printf %q "$SONIC_ROOT") && .venv_sim/bin/python $(printf %q "$LINUX_DIR/sim_loop_keys.py") $SIM_ARGS"
send 1 "$ENVSRC; cd $(printf %q "$DEPLOY") && ./deploy.sh $DEPLOY_ARGS --input-type zmq --zmq-host $ZMQ_HOST sim"
send 2 "$ENVSRC; $INPUT_CMD"
tmux select-pane -t "${PANE[1]}"
log "tmux session '$SESSION' (mode=$MODE policy=$POLICY zmq-host=$ZMQ_HOST display=${DISP:-headless}); logs: $LOGDIR"

wait_for() {  # wait_for PANE REGEX TIMEOUT_S
  local t_end=$(( $(date +%s) + $3 ))
  while [ "$(date +%s)" -lt "$t_end" ]; do
    tmux capture-pane -p -t "${PANE[$1]}" -S -300 2>/dev/null | grep -Eq "$2" && return 0
    sleep 1
  done
  return 1
}
if [ "$AUTO" = 1 ]; then
  log "auto: waiting for deploy prompt (first run builds: up to ~10 min)"
  wait_for 1 "Proceed with deployment" 900 || die "deploy prompt not seen; check pane 1 / $LOGDIR/pane1.log"
  key 1 Enter
  # first start of a policy builds TensorRT engines (cached as .trt next to the ONNX): minutes, not seconds
  wait_for 1 "Init Done" 900 || die "no 'Init Done' (is the sim in pane 0 running? see $LOGDIR/pane1.log)"
  key 1 "]"
  wait_for 1 "CONTROL state" 20 || warn "did not see the CONTROL transition"
  sleep 2; key 0 "9"; log "auto: ']' sent, band released ('9')"
  if [ "$AUTO_STREAM" = 1 ]; then
    log "auto: waiting for the pose stream ($READY_PAT in pane 2)"
    wait_for 2 "$READY_PAT" 300 || die "input not ready; ENTER not sent"
    sleep 2; key 1 Enter
    if wait_for 1 "ZMQ STREAMING MODE: ENABLED" 10; then ok "streaming enabled"; else warn "streaming not confirmed in pane 1"; fi
  fi
fi
if [ "$ATTACH" = 1 ] && [ -t 0 ] && [ -z "${TMUX:-}" ]; then exec tmux attach -t "$SESSION"; fi
echo "attach: tmux attach -t $SESSION      stop: press O in pane 1, then tmux kill-session -t $SESSION"
