# 운영 매뉴얼

사람이 카메라 앞에서 움직이면 시뮬레이션 속 K1이 따라 하는 데모를 현장에서 돌리는 방법이다. 컴퓨터 한 대, 운영자 한 명을 전제로 한다.
설치는 [setup.md](setup.md), 구조는 [architecture.md](architecture.md).

## 한 장 요약

```
켜기   ① 전원 · 카메라 · 프로젝터(확장 모드) 연결
       ② K1Demo.command 실행 → 점검 표에 '실패'가 없으면 그대로 진행
       ③ 관객 화면 창을 프로젝터로 끌어다 놓고 F (전체 화면)
       ④ 패널 상단 '정상', 카메라 OK, 제어 50 Hz 확인
체험   표시 위에 세움 → "정면 보고 팔 내리고 잠깐 서 계세요" → C(보정) → Space(시작)
키     Space 시작/정지   C 보정   R 리셋   Esc 비상정지   방향키/WASD 걷기   Q·E 회전(누르는 동안)
문제   넘어짐 → 3초 뒤 자동으로 일어남, 아니면 R        관객 화면 멈춤 → 그 창에서 새로고침
       카메라 '없음' → USB 다시 꽂기 / 패널 '입력 소스'   로봇이 혼자 걸음 → C 다시, 급하면 '걷기 끄기'
       이상하면 언제든 Esc → 로봇이 서서 멈춤, 다시 Space
끄기   패널 맨 아래 '서비스 종료' 두 번 또는 터미널 창 닫기
```

## 1. 준비물

| 준비물 | 권장 | 메모 |
|---|---|---|
| 컴퓨터 | macOS(Apple Silicon) 또는 Ubuntu 22.04/24.04, 8코어 이상, RAM 16 GB 이상 | 노트북은 전원 어댑터 필수. GEM-X를 쓰려면 NVIDIA GPU |
| 카메라 | USB 웹캠 1080p 30 fps, 화각 65–78° | **자동 프레이밍·인물 추적 끔**, 삼각대 고정(카메라가 움직이면 사람이 걸은 것으로 본다) |
| 화면 2개 | 운영 패널 / 관객 화면(**확장 모드**) | 미러링만 되면 관객 화면을 전체 화면으로 두고 키보드만 쓴다 |
| 공간 | 체험자: 카메라에서 2.5–3.5 m, 폭 2 m, 앞뒤 1.5 m, 서는 위치(home) 표시 | 머리부터 발끝까지 화면에 들어와야 한다 |
| 조명 | 사람 앞쪽에서 고르게 | 역광이면 인식이 떨어진다 |
| 네트워크 | 필요 없음 | 시작·실행에 인터넷을 쓰지 않는다. 현장에서 네트워크를 끈 채로 한 번 시작해 본다 |

카메라는 사람 가슴 높이(1.2–1.4 m), 정면, 살짝 아래를 보게 둔다. 사람이 여러 명 비치면 가운데 사람을 따라 한다.

## 2. 설치 확인 (전날)

1. `.venv`와 모델·자산([setup.md](setup.md)).
2. 대체 영상·리허설 영상: 기본 폴더 `videos/event`(`K1_EVENT_VIDEOS`, 서비스만 바꿀 때 `K1LIVE_EVENT_DIR`). 대체 영상은 `{event_dir}/fallback/demo_1.mp4`,
   `demo_2.mp4`(예: `scripts/render/render_reel.py`로 만든 사람 | 로봇 녹화)이고, 영상은 저장소에 없다.
3. macOS 카메라 권한: 시스템 설정 › 개인정보 보호 및 보안 › 카메라 › 터미널 허용.
4. 사전 점검: `.venv/bin/python -m k1live.preflight` → 통과/경고/실패 표. **실패**는 그 줄의 내용대로 고친다. 경고는 데모 가능(예: sonic 비활성).
5. 리허설(카메라 없이): `K1DEMO_ARGS="--source <영상>" ./K1Demo.command` 또는 패널 › 설정 › 입력 소스에서 리허설 영상.

Ubuntu에서는 `bash K1Demo.command`로 실행한다.

## 3. 당일 순서

1. 전원, 카메라, 프로젝터 연결. 디스플레이 **확장 모드**.
2. `K1Demo.command` 실행 → 점검 표(30초 이내). '실패'가 있으면 Enter를 눌러야 계속된다.
3. 브라우저 창 두 개: 운영 패널 `http://127.0.0.1:8700/`, 관객 화면 `http://localhost:8700/display`(앱 창) → 프로젝터로 끌어 `F`.
   두 번째 모니터 좌표를 알면 `K1DEMO_DISPLAY_POS=1920,0`으로 자동 배치·전체 화면.
4. 패널 상단 **정상**(초록), 카메라 OK, 제어 ≈ 50 Hz, 관객 화면 '실시간'.
5. 체험자 보정(`C`) → 시작(`Space`). 사람이 바뀌면 다시 `C`.
6. 아무도 없이 20초가 지나면 자동 대기(인사 동작 반복). 새 사람이 서면 '사람 감지됨' 배너.

자동 보정·자동 시작(설정 › 자동)을 켜면 대기 중 사람이 2초 서 있을 때 자동으로 보정·시작한다(기본 꺼짐).

체험자 안내 예: "표시 위에 서 주세요" → "정면을 보고 팔을 내린 채 잠깐 서 계세요"(C) → "팔을 천천히 움직여 보세요"(Space) →
"앞으로 한 걸음 나오면 로봇이 걷고, 제자리로 오면 멈춰요". 빠른 동작보다 천천히 큰 동작이 잘 따라 한다.

## 4. 강연자를 비출 때

카메라가 무대·연단의 강연자를 비추고 로봇이 팔 동작을 따라 하는 경우.

- **걷기 › 보행 모드: 강연자**(`event.yaml` 기본 `loco_profile: presenter`). 몸을 돌려도 로봇은 돌지 않고, 강연자가 옆으로 걸어갈 때만 같이 걷다가
  멈추면 1–2초 안에 선다. 카메라 쪽 앞뒤 걸음은 거의 따라 하지 않는다. 걷는 모습이 필요 없으면 **끄기**.
- **사람 찾을 영역**: 패널의 카메라 화면에서 강연자가 다니는 범위를 드래그(노란 상자). 청중 머리·패널석을 뺀다. 바꾸는 순간 인식이 1–2초 멈추므로
  강연 전에 정한다.
- 카메라: 삼각대 고정, 강연자 정면 ±30°, 렌즈 높이 1.2–1.5 m, 전신(최소 무릎 위)이 화면 높이의 50–80 %. MediaPipe는 사람 키가 화면의 30 % 이하이면
  거의 못 찾으므로(영역을 그리면 회복) 먼 전신에는 GEM-X가 낫다([results.md](results.md) §6).
- 보정(C)은 필요 없다(보행 기준 위치는 0.4초 서 있으면 스스로 잡는다).
- 명령줄로도: `K1LIVE_LOCO_PROFILE=presenter K1LIVE_CROP=0.2,0,0.8,0.9 ./K1Demo.command`, `python -m k1live.service --loco-profile presenter --crop 0.2,0,0.8,0.9`.

## 5. 예비 컴퓨터로 바꾸기

GPU 컴퓨터가 멈추거나 켜지지 않을 때를 대비해, 같은 커밋의 번들을 다른 컴퓨터(예: Mac)에 설치해 둔다. NVIDIA GPU가 없으면 인식은 MediaPipe가 되고
(설정이 GEM-X여도 서비스가 스스로 바꾼다), 로봇 쪽(hybrid, 팔, 걷기 모드)은 같다. MediaPipe는 강연장 거리에서 **사람 찾을 영역**이 필요하므로 설치 때
그 컴퓨터에서 미리 그려 둔다(설정은 그 컴퓨터의 `logs/k1live/settings.json`에 저장). 바꾸는 순서(목표 2분): 프로젝터 HDMI·웹캠 USB를 옮김 →
`K1Demo.command` → 관객 화면 전체 화면 → '정상'·'MediaPipe' 확인 → Space. 리허설 때 한 번 연습해서 시간을 잰다.

## 6. 강연자 앞뒤 걷기 맞추기 (현장 리허설)

강연자 모드의 앞뒤 축은 스톡 영상으로는 값을 정할 수 없었다(카메라 줌·패닝, 멀고 작은 사람). 앞뒤 걷기가 필요할 때만 행사 카메라 위치에서 확인한다.

| 순서 | 강연자 | 볼 것 |
|---|---|---|
| a | 제자리에서 1분 동안 말하며 손짓, 슬라이드 쪽으로 몸 돌리기 | 로봇이 **한 걸음도 걷지 않아야** 한다(필수) |
| b | 옆으로 2 m 걸었다 멈추고 돌아오기 × 2 | 1–2초 안에 같은 방향으로 출발, 멈추면 2초 안에 정지 |
| c | 카메라 쪽으로 1.5–2 m 걸었다 멈추고 돌아가기 × 3 | 앞으로 걷는지(기본값은 대부분 걷지 않는다) |

GEM-X용 후보 값 — `k1live/web/event.yaml`의 `gemx:` 아래에 넣고 서비스를 다시 시작한다:

```yaml
  args: ["--presenter-param", "fwd_start=0.25", "--presenter-param", "fwd_stop=0.20",
         "--presenter-param", "fwd_confirm_s=0.40", "--presenter-param", "fwd_home_follow_s=1.2"]
```

스톡 영상 탐색에서 가장 나았던 값이다(앞뒤 반응 5 → 14/24 구간, 옆걸음 그대로, 고정 카메라 제자리 강연자 오보행 0). 시작 로그에
`presenter walking overrides: {...}`가 찍히면 적용된 것이다. **a에서 한 번이라도 걸으면 지우고 원래 값으로 돌아간다.** 카메라가 높아 내려다보면
`"--gemx-dist", "ground"`(발목이 닿은 바닥 위치로 거리 측정)를 더한다. 사람 키 높이 카메라에서는 쓰지 않는다.

운영자는 언제든 방향키·`WASD`를 누르고 있는 동안 로봇을 걷게 할 수 있다(사람 보행보다 우선).

## 7. 운영 패널

**상태 타일**

| 타일 | 정상 | 의미 |
|---|---|---|
| 카메라 | OK, 30 fps | 입력 소스와 초당 프레임, 지금 쓰는 인식 |
| 사람 | 감지 | 보정 여부, 없으면 몇 초째 없는지 |
| 지연 | ≤ 150 ms (초록) | 카메라 프레임 → 로봇 제어 반영, 최근 30초 p95. 250 ms 넘으면 빨강 |
| 제어 | 48–52 Hz | 제어 주기와 실시간 비율(1.00이 정상) |
| 낙상 · 리셋 | 0 / n | 넘어진 횟수 / 리셋 횟수 |
| 백엔드 | hybrid · 서기(팔 집중) | 지금 상태: 서기 / 걷기(SONIC) / 멈추는 중. 아래에 쓸 수 있는 백엔드 |
| 걷기 | 체험객 / 강연자 / 꺼짐 / 걷는 중 | 보행 모드, 최대 속도 |
| 관객 화면 | 실시간 | '대체 영상'이면 사유 |

**버튼과 키**: `Space` 시작/정지, `C` 보정, `R` 리셋, `Esc` 비상정지(사람을 무시하다가 `Space`로 해제), 방향키/`WASD` 운영자 보행(누르는 동안),
`Q`/`E` 제자리 회전.

**설정**

- 미러(기본 켜짐): 체험자 오른팔 → 로봇 왼팔.
- 백엔드: **hybrid(기본)** / split / sonic(체크포인트가 있을 때). 패널에서 고른 값은 `settings.json`에 저장되어 `event.yaml`보다 우선한다 —
  리허설 때 다른 값을 골랐다면 행사 전에 되돌린다.
- 팔 경로: **개선(기본)** = 뼈 방향 IK + PD 팔 + 12 rad/s 참조 / 기존 = 위치 IK + mimic student 팔(6 rad/s).
- 인식: MediaPipe(CPU) / GEM-X(GPU). GEM-X를 누르면 모델을 불러오는 20–40초 동안 관객 화면은 대체 영상이다. 쓸 수 없으면 자동으로 MediaPipe.
- 입력 소스(카메라 번호 또는 리허설 영상), 대체 영상(자동 / 항상 / 끄기), 대기 전환 시간, 자동 보정·자동 시작.

**관객 화면**: 왼쪽 카메라(골격 표시, 거울처럼 반전), 오른쪽 K1. 문구는 `event.yaml`(`title`, `guide*`, `msg_*`)에서 바꾼다. 카메라 또는 시뮬레이션이
2초 넘게 끊기면 대체 영상, 둘 다 3초 정상이면 실시간. `F` 또는 더블클릭 = 전체 화면.

## 8. 문제 해결

| 증상 | 할 일 |
|---|---|
| 카메라 안 잡힘 | USB 다시 꽂기(20초 안에 자동 재연결), 설정 › 입력 소스에서 다른 번호, macOS 카메라 권한, 카메라를 쓰는 다른 앱 종료 |
| 사람 감지 안 됨 | 전신이 보이게 거리 조절, 역광 피하기, 한 사람만. 강연장 거리면 사람 찾을 영역 또는 GEM-X |
| 로봇이 따라 하지 않음 | `Space`, 비상정지 배너 확인, `C` 다시 |
| 좌우가 반대 | 미러 켜기, 정면을 보는지 확인 |
| 넘어짐 | 3초 안에 자동으로 선다. 자주 넘어지면 최대 속도 0.3, 걷기 끄기 |
| 로봇이 혼자 걷는다 | 체험자가 바뀌었는데 `C`를 안 눌렀을 때 흔함 → 표시 위에 세우고 `C`. 강연자면 걷기 › 강연자, 카메라 고정 확인 |
| 청중·패널석 사람을 따라 한다 | 사람 찾을 영역을 강연자 범위로 |
| 느려짐(지연 빨강, 제어 < 48 Hz) | 전원 연결, 다른 앱·탭 닫기. 그래도 느리면 서비스 종료 후 다시 실행 |
| 관객 화면 멈춤 | 미리보기도 멈췄으면 프로세스 '재시작', 관객 화면만이면 그 창 새로고침 → `F` |
| "GEM-X 인식을 쓸 수 없어 MediaPipe로 실행 중" | 서비스가 스스로 MediaPipe로 계속 돈다. 원인은 배너와 `logs/k1live/<세션>/sender.log` |
| hybrid 버튼 회색 / 시작하자마자 split | 걷기 체크포인트·플래너·스테이징 메시 중 하나가 없음([setup.md](setup.md) §4) |
| hybrid에서 걷기가 늦게 시작 / 멈춘 뒤 잠깐 팔을 내린 채 서 있음 | 정상: 0.2 s 이어진 보행 명령에서 출발, 멈출 때는 발을 모으고(최대 3 s) 사람 팔로 돌아온다 |
| "포트 8700 이미 사용 중" | 이미 실행 중 — 열린 창을 쓰거나 그 터미널을 닫고 다시 실행 |

## 9. 설정 파일 — `k1live/web/event.yaml`

서비스를 다시 시작하면 적용된다. 빠진 키는 `k1live/service.py`의 기본값.

| 키 | 기본 | 뜻 |
|---|---|---|
| `backend` | `hybrid` | 첫 실행 백엔드(hybrid 파일이 없으면 split) |
| `hybrid_variant` | `models/k1/k1s_walk4/deploy/k1s_walk4_007000` | hybrid 걷기 체크포인트(없으면 walk3 5000 기본 경로) |
| `arm_path`, `arm_ik`, `arm_mode`, `arm_vmax`, `arm_amax` | `improved`, `dir`, `pd`, `12`, `150` | 팔 경로 |
| `perception` | `gemx` | 첫 실행 인식(`mediapipe` / `gemx`). GEM-X를 못 쓰면 MediaPipe |
| `gemx` | | `perception_dir`, `python`, `cmd`, `hfov`, `arm_src`, `max_h`, (선택) `args` |
| `loco_profile` | `presenter` | 첫 실행 보행 모드(`visitor` / `presenter`) |
| `crop` | `null` | 사람 찾을 영역 `[x0, y0, x1, y1]`(0–1, 미러 전) |
| `title`, `subtitle_sonic`, `guide`, `guide_no_walk`, `guide_presenter`, `msg_*` | | 관객 화면 문구 |
| `event_dir` | `$K1_EVENT_VIDEOS` 또는 `videos/event` | `{event_dir}` 경로의 기준(`K1LIVE_EVENT_DIR`가 우선) |
| `fallback_videos`, `rehearsal_videos` | `{event_dir}/fallback/demo_{1,2}.mp4`, `{event_dir}/event_*.mp4` 등 | 대체 영상, 리허설 영상 목록(glob) |
| `cameras` | `[0, 1, 2]` | 패널에 보일 카메라 번호 |
| `idle_after_s` / `fallback_after_s` / `live_after_s` / `fall_reset_s` | 20 / 2 / 3 / 6 | 대기 전환, 대체 영상·실시간 전환, 서비스가 리셋을 보내는 시간 |
| `grace_s` / `stale_s` | 45 / 3 | 하위 프로그램 시작 유예 / 상태가 끊기면 재시작하는 시간 |
| `mem_limit_mb` | sender 1500, sim 6000, sender_gemx 10000 | 넘으면 아무도 따라 하지 않을 때 그 프로그램만 재시작 |
| `display_cam_flip`, `cam_fit`, `robot_fit` | false, cover, cover | 관객 화면 표시 |

명령줄(개발·리허설):

```bash
.venv/bin/python -m k1live.service [--source N|영상] [--backend split|hybrid|sonic|sonic_human] [--fallback auto|on|off]
                                   [--start] [--auto] [--mock] [--loco-profile visitor|presenter] [--crop x0,y0,x1,y1]
.venv/bin/python -m k1live.sim_server --backend hybrid --arm-mode pd --arm-vmax 12 --arm-amax 150
.venv/bin/python -m k1live.sender --source 0 --arm-ik dir                  # MediaPipe
.venv/bin/python -m k1live.sender_gemx --source clip.mp4 --loop            # GEM-X
.venv/bin/python -m k1live.rehearse --minutes 5 --backend hybrid --render-client   # 두 프로세스 + 계측
```

## 10. 운영 시험 도구 (`scripts/ops/`, `scripts/offline_guard/`)

| 도구 | 내용 |
|---|---|
| `python -m k1live.soak --minutes N --source <영상> [--chaos default] -- --auto` | 서비스를 실제 하위 프로그램과 함께 띄우고 관객 화면 같은 클라이언트를 붙여 장시간 시험. `--chaos`는 정해진 시각에 강제 종료·멈춤을 넣는다. 결과 `soak.json`/`soak.md`(기준별 통과·실패) |
| `python -m k1live.soak --make-rehearsal <out.mp4> --event-videos <폴더>` | 리허설 영상(사람 동작 + 빈 화면 구간) 만들기 |
| `bash scripts/ops/soak.sh NAME MINUTES [soak 인자]` | 위 soak + CPU·GPU 텔레메트리 + 오프라인 차단을 한 번에(Linux) |
| `bash scripts/ops/kill_service_test.sh [OUT]` | 실행기 아래의 서비스 자체를 SIGKILL → 실행기가 다시 띄우고 남은 자식 프로세스를 정리하는지 |
| `scripts/ops/telemetry.py` | CPU·메모리(선택: GPU) 텔레메트리 CSV와 요약 |
| `. scripts/ops/cpu_only.env` + `scripts/ops/nvidia_watch.py` | NVIDIA GPU를 전혀 쓰지 않는 실행(Mesa llvmpipe 렌더, CUDA 숨김)과, 그 실행이 NVIDIA 라이브러리·장치를 열면 즉시 끝내는 감시. 각 스크립트는 `CPU_ONLY=1`로 이 모드를 쓴다 |
| `PYTHONPATH=$PWD/scripts/offline_guard K1_OFFLINE_LOG=<log> bash K1Demo.command` | 파이썬 프로세스의 외부 네트워크 접속을 막고 시도를 기록한다(네트워크 없이 시작되는지 확인) |

장시간 시험의 통과 기준(`soak.json`의 `verdict`): 예기치 않은 프로세스 실패 0, 장애마다 10초 안에 정상·15초 안에 실시간 화면, 지연 p95 ≤ 150 ms가
99 % 이상, 제어 50 ± 2 Hz(하위 1 % ≥ 48), 실시간 비율 하위 1 % ≥ 0.98, 낙상 0, 메모리 추세 sender·sim ≤ 30 MB/h·service ≤ 5 MB/h,
관객 화면 클라이언트 ≥ 20 fps.

## 11. 행사 번들

```bash
scripts/make_event_bundle.sh [OUT_DIR]      # 코드(HEAD) + git에 없는 모델·자산·대체 영상 + k1demo.env
cd OUT_DIR && bash scripts/install_event.sh # 행사 컴퓨터에서 한 번: .venv + 사전 점검(인터넷 한 번 필요)
```

번들에는 SMPL-X에서 유도한 `human_joints_info.pkl`, 행사용 영상처럼 **재배포할 수 없는 파일**이 들어간다.
행사 컴퓨터로 옮기는 용도로만 쓰고 공개 저장소나 공유 링크에 올리지 않는다. GEM-X 환경은 번들에 넣지 않는다.

## 12. 종료와 로그

패널 맨 아래 **서비스 종료**를 4초 안에 두 번, 또는 터미널에서 `Ctrl-C` / 창 닫기. 로그는 `logs/k1live/<날짜-시각>/`에 남는다
([architecture.md](architecture.md) §5).
