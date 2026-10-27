#!/bin/bash
# K1 실시간 데모 실행기 — Finder에서 더블클릭 (macOS) / Ubuntu: bash K1Demo.command
#   1) 사전 점검 (k1live.preflight)  2) 서비스 시작 (k1live.service, 죽으면 다시 시작)
#   3) 브라우저 창 두 개: 운영 패널, 관객 화면(앱 창)
# 이 창(터미널)을 닫거나 Ctrl-C를 누르면 데모가 종료됩니다. MuJoCo 시뮬레이션만 사용합니다(실제 로봇 없음).
#
# 환경 변수 (선택):
#   K1DEMO_NO_CAMERA=1      사전 점검에서 카메라 검사를 건너뜀
#   K1DEMO_NO_BROWSER=1     브라우저를 열지 않음
#   K1DEMO_ARGS="..."       서비스 추가 인자 (예: "--source videos/rehearsal.mp4" 또는 "--mock")
#   K1DEMO_DISPLAY_POS=X,Y  관객 화면 창 위치 (두 번째 모니터 좌표, 예: 1920,0) — 주면 그 화면에서 전체 화면
#   K1DEMO_FULLSCREEN=1     관객 화면을 바로 전체 화면으로 (K1DEMO_DISPLAY_POS 없이도)
#   K1DEMO_BROWSER=...      Ubuntu: 브라우저 고르기 (기본 auto = Chrome/Chromium/Edge, 없으면 Firefox; firefox; 또는 실행 파일)
#   K1DEMO_CHROME_ARGS=...  Chrome 창 두 개에 붙일 인자 (예: "--disable-gpu" = GPU 없이 그리기)

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
[ -f "$ROOT/k1demo.env" ] && . "$ROOT/k1demo.env"   # bundle: AI_SAPIENS_ROOT, K1LIVE_EVENT_DIR
PY="$ROOT/.venv/bin/python"
OS="$(uname -s)"
PORT=$((8700 + ${K1LIVE_PORT_OFFSET:-0}))
PANEL_URL="http://127.0.0.1:$PORT/"
DISPLAY_URL="http://localhost:$PORT/display"   # another host name = its own browser connection pool

if [ "$OS" = "Linux" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

echo "================================================================"
echo " AI Sapiens K1 — 실시간 따라하기 데모"
echo " 폴더: $ROOT"
echo "================================================================"

if [ ! -x "$PY" ]; then
  echo "[실행기] 파이썬 환경(.venv)이 없습니다: $PY"
  echo "         설치 방법은 docs/setup.md를 보세요."
  read -r -p "Enter를 누르면 창을 닫습니다." _
  exit 1
fi

# ---------------------------------------------------------------- 1) 사전 점검
PRE_ARGS=()
[ "${K1DEMO_NO_CAMERA:-0}" = "1" ] && PRE_ARGS+=(--no-camera)
case " ${K1DEMO_ARGS:-} " in *" --mock "*) PRE_ARGS+=(--mock) ;; esac
echo "[실행기] 사전 점검 중… (30초 정도 걸릴 수 있습니다)"
"$PY" -m k1live.preflight "${PRE_ARGS[@]}"
PRE=$?
if [ $PRE -ne 0 ]; then
  echo
  echo "[실행기] 사전 점검에 실패 항목이 있습니다. 위 표를 확인하세요."
  if [ -t 0 ]; then
    read -r -p "그래도 시작하려면 Enter, 중단하려면 Ctrl-C … " _
  fi
fi

# ---------------------------------------------------------------- 2) 잠자기 방지
KEEPAWAKE=""
if [ "$OS" = "Darwin" ] && command -v caffeinate >/dev/null; then
  caffeinate -dimsu -w $$ &
  KEEPAWAKE=$!
elif command -v systemd-inhibit >/dev/null; then
  systemd-inhibit --what=idle:sleep --who=K1Demo --why="K1 demo running" --mode=block \
    bash -c "while kill -0 $$ 2>/dev/null; do sleep 5; done" 2>/dev/null &
  KEEPAWAKE=$!
fi

# ---------------------------------------------------------------- 3) 브라우저
open_browsers() {
  for _ in $(seq 1 120); do
    curl -s -m 1 "${PANEL_URL}api/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
  [ "${K1DEMO_NO_BROWSER:-0}" = "1" ] && { echo "[실행기] 브라우저 열기 생략 (K1DEMO_NO_BROWSER=1)"; return; }
  # both Chrome windows: no first-run / default-browser dialogs, no "Can't update Chrome" dialog (an un-updated Chrome
  # pops it over the window, e.g. Chrome 151 without updates), no translate bar over the Korean page
  local chrome_args=(--no-first-run --no-default-browser-check "--simulate-outdated-no-au=Tue, 31 Dec 2099 23:59:59 GMT"
                     --disable-features=Translate)
  # shellcheck disable=SC2206
  [ -n "${K1DEMO_CHROME_ARGS:-}" ] && chrome_args+=($K1DEMO_CHROME_ARGS)   # e.g. "--disable-gpu" (software rendering)
  local disp_args=(--app="$DISPLAY_URL" "${chrome_args[@]}"
                   --autoplay-policy=no-user-gesture-required --disable-session-crashed-bubble)
  if [ -n "${K1DEMO_DISPLAY_POS:-}" ]; then
    disp_args+=(--window-position="$K1DEMO_DISPLAY_POS" --start-fullscreen)
  elif [ "${K1DEMO_FULLSCREEN:-0}" = "1" ]; then
    disp_args+=(--start-fullscreen)
  fi
  if [ "$OS" = "Darwin" ]; then
    local chrome=""
    for app in "Google Chrome" "Chromium" "Microsoft Edge"; do
      if [ -d "/Applications/$app.app" ] || [ -d "$HOME/Applications/$app.app" ]; then chrome="$app"; break; fi
    done
    if [ -n "$chrome" ]; then
      local prof="$HOME/Library/Application Support/K1Demo/display-profile"
      mkdir -p "$prof"
      open -na "$chrome" --args "${chrome_args[@]}" --new-window "$PANEL_URL"
      sleep 1
      open -na "$chrome" --args --user-data-dir="$prof" "${disp_args[@]}"
    else
      open -a Safari "$PANEL_URL"
      open -a Safari "$DISPLAY_URL"
    fi
  else
    local chrome="" ff=""
    case "${K1DEMO_BROWSER:-auto}" in
      auto)
        for b in google-chrome google-chrome-stable chromium chromium-browser microsoft-edge; do
          command -v "$b" >/dev/null && { chrome="$b"; break; }
        done
        [ -z "$chrome" ] && command -v firefox >/dev/null && ff=firefox ;;
      firefox) ff=firefox ;;
      *) chrome="$K1DEMO_BROWSER" ;;
    esac
    if [ -n "$chrome" ]; then
      local prof="${XDG_CONFIG_HOME:-$HOME/.config}/k1demo-display"
      mkdir -p "$prof"
      "$chrome" "${chrome_args[@]}" --new-window "$PANEL_URL" >/dev/null 2>&1 &
      sleep 1
      "$chrome" --user-data-dir="$prof" "${disp_args[@]}" >/dev/null 2>&1 &
    elif [ -n "$ff" ]; then
      # Firefox (Ubuntu's snap): two separate instances with their own profiles (no welcome page, autoplay allowed; snap
      # Firefox only writes non-hidden folders of $HOME); kiosk = full screen without toolbars when asked for
      local ffdir="${K1DEMO_FF_DIR:-$HOME/k1demo-firefox}" p
      for p in panel display; do
        mkdir -p "$ffdir/$p"
        [ -f "$ffdir/$p/user.js" ] || cat > "$ffdir/$p/user.js" <<'FF'
user_pref("browser.aboutwelcome.enabled", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("startup.homepage_welcome_url", "");
user_pref("datareporting.policy.dataSubmissionPolicyBypassNotification", true);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.translations.automaticallyPopup", false);
user_pref("media.autoplay.default", 0);
FF
      done
      local kiosk=()
      { [ -n "${K1DEMO_DISPLAY_POS:-}" ] || [ "${K1DEMO_FULLSCREEN:-0}" = "1" ]; } && kiosk=(--kiosk)
      firefox --no-remote --profile "$ffdir/panel" "$PANEL_URL" >/dev/null 2>&1 &
      sleep 2
      firefox --no-remote --profile "$ffdir/display" "${kiosk[@]}" "$DISPLAY_URL" >/dev/null 2>&1 &
    else
      xdg-open "$PANEL_URL" >/dev/null 2>&1 &
      xdg-open "$DISPLAY_URL" >/dev/null 2>&1 &
    fi
  fi
  echo "[실행기] 운영 패널: $PANEL_URL"
  echo "[실행기] 관객 화면: $DISPLAY_URL  (창을 프로젝터 화면으로 옮긴 뒤 F 키 = 전체 화면)"
}

# ---------------------------------------------------------------- 4) 서비스 (죽으면 다시 시작)
STOPPING=0
SVC=""
stop_all() {
  STOPPING=1
  if [ -n "$SVC" ] && kill -0 "$SVC" 2>/dev/null; then
    echo
    echo "[실행기] 종료 중…"
    kill -TERM "$SVC" 2>/dev/null
    for _ in $(seq 1 50); do kill -0 "$SVC" 2>/dev/null || break; sleep 0.2; done
    kill -KILL "$SVC" 2>/dev/null
  fi
  [ -n "$KEEPAWAKE" ] && kill "$KEEPAWAKE" 2>/dev/null
}
trap 'stop_all; exit 0' INT TERM HUP

if curl -s -m 1 "${PANEL_URL}api/health" >/dev/null 2>&1; then
  echo "[실행기] 서비스가 이미 실행 중입니다 — 브라우저만 엽니다."
  open_browsers
  exit 0
fi

open_browsers &
# shellcheck disable=SC2086
while [ $STOPPING -eq 0 ]; do
  "$PY" -m k1live.service ${K1DEMO_ARGS:-} &
  SVC=$!
  wait "$SVC"
  CODE=$?
  [ $STOPPING -eq 1 ] && break
  if [ $CODE -eq 0 ]; then
    echo "[실행기] 서비스가 종료되었습니다."
    break
  fi
  if [ $CODE -eq 2 ]; then
    echo "[실행기] 포트 $PORT 이(가) 이미 사용 중입니다 — 다른 데모 창이 열려 있는지 확인하세요."
    break
  fi
  echo "[실행기] 서비스가 예기치 않게 종료됨 (code $CODE) — 3초 후 다시 시작합니다."
  sleep 3
done
stop_all
if [ -t 0 ] && [ "${K1DEMO_NO_BROWSER:-0}" != "1" ]; then
  read -r -t 30 -p "Enter를 누르면 창을 닫습니다." _
fi
exit 0
