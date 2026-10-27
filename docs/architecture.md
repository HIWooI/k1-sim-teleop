# 구조

사람이 카메라 앞에서 움직이면 MuJoCo 속 ROBOTIS AI Sapiens K1(23 DOF)이 실시간으로 따라 하는 서비스의 구성이다.
시뮬레이션만 다루며 로봇 I/O 코드는 없다. 메시지 계약은 [`k1live/protocol.py`](../k1live/protocol.py)에 있다.

## 1. 프로세스와 포트

```
K1Demo.command ─ preflight ─ service (:8700) ─┬─ sender      카메라/영상 → 인식 → 팔 IK·걷기 의도 → HUMAN (5701), /cam.mjpg (8701)
                                              ├─ sim_server  50 Hz 제어 + MuJoCo 200 Hz → SIM_STATE (5703), /robot.mjpg (8702)
                                              └─ 브라우저    운영 패널 /, 관객 화면 /display  ← CTRL (5702), SENDER_STATE (5704)
```

| 포트 | 방향 | 내용 |
|---|---|---|
| 5701 HUMAN | sender → sim_server | `HumanFrame`(팔 관절 목표, 허리·숙임, 골반 위치, 걷기 명령, 선택: SMPL-24), ~24–30 Hz |
| 5702 CTRL | service → sender, sim_server | `start`·`stop`·`reset`·`calibrate`·`set`·`gesture`·`drive`·`estop`·`quit` (PUB/SUB, 다시 붙으면 설정 전체를 재전송) |
| 5703 SIM_STATE | sim_server → service | 모드, 제어 Hz, 실시간 비율, 낙상·리셋, 백엔드 상태, 10 Hz |
| 5704 SENDER_STATE | sender → service | 카메라·사람·보정·인식 상태, 4 Hz |
| 8700 | service | 운영 패널 `/`, 관객 화면 `/display`, `/api/*` |
| 8701 / 8702 | sender / render | `/cam.mjpg`(골격 표시) / `/robot.mjpg`(로봇), MJPEG, 보는 사람이 있을 때만 인코딩 |

모든 포트는 `127.0.0.1`이며 `K1LIVE_PORT_OFFSET`만큼 옮길 수 있다(한 컴퓨터에서 여러 벌).

## 2. sender — 인식과 사람 → 로봇 목표

| 단계 | MediaPipe 경로 (`k1live/sender.py`) | GEM-X 경로 (`k1live/sender_gemx.py` + `k1live/gemx_worker.py`) |
|---|---|---|
| 프레임 | `FrameSource` 스레드가 항상 최신 프레임 1장만 둔다(늦은 프레임은 버림). 영상 파일은 프레임 i를 시작 + i/fps에 찍힌 것으로 친다(가짜 카메라) | 같음 |
| 인식 | MediaPipe Pose(full, CPU) | 별도 프로세스(GEM-X venv, CUDA): YOLOX 검출 → VitPose SOMA-77 2D → GEM 64프레임 인과 창 → SOMA FK. stdin/stdout 파이프로 프레임·결과를 주고받으며, 같은 파이프를 ssh로 넘겨 다른 GPU 기계에서 돌릴 수도 있다 |
| 사람 좌표 | MediaPipe world/이미지 랜드마크 | `k1live/gemx.py soma_det`: MediaPipe-33 배치로 바꿈(정면 각도 = VitPose 2D, 깊이 = GEM-X 3D) |
| 단일 인물 | `JumpGate`: 골반이 한 프레임에 크게 튀면 버리고, 8프레임 연속이면 받아들이고, 0.5 s 아무도 없으면 잊는다. 운영 패널의 '사람 찾을 영역'(`--crop`)으로 범위를 좁힐 수 있다 | 같음 |
| 실패 처리 | 카메라가 끊겨도 종료하지 않고 2 s마다 다시 연다 | CUDA·GEM-X·체크포인트가 없거나 GEM-X 프로세스가 거듭 죽으면 종료 코드 5 → 서비스가 MediaPipe sender로 바꿔 띄운다 |

이후는 두 경로가 같은 `HumanMapper`(`k1live/human.py`)를 거친다.

- **팔 IK** (`k1live/arm_ik.py`, `--arm-ik dir`, 행사 기본): 위치가 아니라 상완·전완의 **방향**을 맞춘다. 화면 평면 각도는 2D, 깊이 성분만 3D에서
  가져온다. 팔마다 4관절을 닫힌 해로 시작해 관절 한계 안에서 감쇠 Gauss-Newton 몇 번(프레임당 1–2 ms). 두 팔을 곧게 위로 들면 무게중심이 뒤로
  가서 보행 정책이 뒤로 걸어 버리므로, 위로 든 뼈는 최소 30° 앞으로 기울인다(정면 2D 각도는 그대로). 이전 경로(`classic`)는 위치 기반 CPU IK다.
- **미러**(기본 켜짐): 관절 공간 좌우 교환 + roll·yaw·손목 부호 반전. 관객이 거울처럼 본다.
- **허리·숙임·골반 위치**: 어깨선과 골반선의 yaw 차, 몸통 앞 기울기, 보정 위치(home) 대비 이동.
- **보정**: 어깨가 보이고 0.5 s 동안 가만히 있을 때 자동으로 home을 잡는다. CTRL `calibrate`로 다시 잡을 수 있다.
- **SMPL-24 스트림**(`sonic_human` 백엔드용): MediaPipe-33 → `sonic/mp2smpl.py`. GEM-X도 SOMA-77 → MediaPipe-33 → 같은 변환을 쓴다.

### 걷기 의도 (`k1live/locomotion.py: LocoEstimator`)

"발판 조이스틱": home 대비 앞뒤·옆으로 나가면 그 방향 속도, 몸을 돌리면 회전, home으로 돌아오면 정지.

| 신호 | MediaPipe | GEM-X (`k1live/gemx.py GemxLocoEstimator`) |
|---|---|---|
| 거리 | 이미지 랜드마크와 world 랜드마크의 약원근 맞춤 → 거리 = 초점거리 / 축척(숙이거나 몸을 돌려도 덜 속는다) | 카메라 좌표 골반 깊이(`--gemx-dist pelvis`, 기본) 또는 발목 광선 ∩ 바닥면(`ground`, 카메라가 높을 때) |
| 옆 | 골반 이미지 위치를 미터로 환산 | 카메라 좌표 골반 x |
| 회전 | 어깨선·골반선 yaw가 같은 부호일 때 작은 쪽 | 3D 어깨·골반 선 |

축마다 데드존 + 히스테리시스(출발 문턱 > 정지 문턱) + 확인 시간, 문턱 위에서는 거리에 비례한 속도. 사람이 없으면 즉시 0, 3 s 넘게 비면 다음
사람은 0.8 s 서 있을 때 스스로 home을 잡는다. 걷지 않는 축에서는 home이 사람을 천천히(τ 3 s) 따라가 앞으로 조금씩 밀려 오는 사람을 걸음으로
읽지 않는다.

| 프로필 | 쓰임 | 동작 |
|---|---|---|
| `visitor` | 카메라 앞 2.5–3.5 m에 선 체험자 | 한 걸음 나오면 걷고, 몸을 돌리면 돈다 |
| `presenter` (행사 기본) | 무대·연단의 강연자 | 몸을 돌려도 로봇은 돌지 않는다(`turn=False`). home이 걷는 축에서도 따라가(`follow_engaged`, 0.8 s) "걷는 동안만 같이 걷고 멈추면 1–2초 안에 선다". 옆모습일 때는 거리를 믿지 않는다 |

presenter의 앞뒤 축 값은 `sender_gemx --presenter-param KEY=VALUE`(반복 가능)로 현장에서 바꿀 수 있다([runbook.md](runbook.md) §6).

## 3. sim_server — 50 Hz 제어 (`k1live/sim_server.py`)

- **스케줄**: 20 ms 벽시계 틱(`k1live/timing.sleep_until`: 남은 시간의 60 %씩 자고 마지막 0.5 ms만 spin). 한 주기 넘게 늦으면 따라잡지 않고 건너뛴다.
- **팔**: 최신 `HumanFrame`의 팔 목표를 속도·가속 제한(행사 설정 12 rad/s · 150 rad/s², 이산 제동 곡선이라 오버슛 없음)을 거쳐 참조로 만든다.
  사람을 놓치면 1 s 유지 뒤 기본 자세로.
- **보행**: CTRL `drive`(운영자, 0.3 s 유효) > 사람의 걷기 명령. 백엔드별 속도 제한기(`VelocityLimiter`)와 원점 3 m 울타리를 거친다.
  사람 없음·estop·낙상·프레임 끊김이면 0(0.6 m/s에서 0까지 0.14 s).
- **방향 유지(heading hold)**: 회전 명령이 없으면 마지막 방향 기준으로 작은 회전(≤ 0.4 rad/s)을 더해, 팔 동작의 반작용으로 로봇이 제자리에서
  도는 것을 막는다(30 s 손 흔들기에서 132° → 2°).
- **모드**: live(따라하기), idle(따라 하지 않음), 사람이 `idle_after` 동안 없으면 대기 동작(인사·환영·박수 등 `k1/gestures.py`) 반복.
- **낙상**: 골반 < 0.45 m 또는 기울기 > 45° → 1 s 뒤 원점 리스폰. 틱 예외는 기록만 하고 계속, 50번 연속이면 종료(서비스가 재시작).
- **렌더**: 별도 프로세스(`k1live/render.py`)가 공유 메모리의 qpos로 같은 모델을 그린다(1280×720, 30 fps). 스레드로 두면 GIL 때문에 Linux에서
  틱이 수 ms 밀려서 프로세스로 분리했다. 렌더가 죽어도 제어는 계속된다.

## 4. 백엔드 (`k1live/backends/`)

| 백엔드 | 다리 | 팔 | 비고 |
|---|---|---|---|
| `split` | ROBOTIS `loco_armdriven_r2`(속도 명령) | ROBOTIS mimic student(참조 추종) 또는 PD(`--arm-mode pd`) | `k1/sim.py`. 숙임 불가. loco 정책의 방향별 데드존 때문에 속도 제한기가 0이 아닌 명령을 시작 레벨로 바로 올린다 |
| `sonic` | K1 SONIC 전신 정책 + GEAR-SONIC 키네마틱 플래너 | 같은 정책(사람 팔을 참조에 덧입힘) | 숙임·허리 가능. 사람 팔 추종은 split보다 덜 정확 |
| `sonic_human` | K1 SONIC SMPL 인코더(사람 전신 자세를 직접) | 같은 정책 | BONES-SEED 실험. 수락 기준 미달([training.md](training.md) §5) |
| **`hybrid`** (행사 기본) | 서 있을 때 split, 걸을 때 K1 SONIC + 플래너 | 서 있을 때 PD 팔(사람), 걸을 때 플래너 팔 흔들기 | 아래 |

### hybrid (`k1live/backends/hybrid.py`)

```
STAND ──(걷기 명령 0.2 s 지속)──▶ WALK ──(명령 0, 0.5 s 이상 걸은 뒤)──▶ SETTLE ──(발·골반 0.5 s 정지, 최대 3 s)──▶ STAND
다리 = loco(속도 0)                다리·팔 = K1 SONIC + 플래너 참조           다리 = SONIC(플래너 idle: 발 모음)
팔   = 사람 팔(PD)                                                          팔   = 걷기 팔 유지(settle_arms="walk")
```

- **물리는 하나**: split과 같은 MuJoCo 모델·같은 정책 순서라 STAND는 split과 비트 단위로 같다.
- **그림자 실행**: 매 틱 student·loco·SONIC 정책을 모두 같은 로봇 상태로 추론해(플래너는 비동기 스레드) 넘겨받는 쪽의 관측 이력이 항상 살아 있다.
- **넘겨주기**: 관절 묶음마다 두 PD 토크를 smoothstep으로 섞는다(다리 0.4 s, 팔 0.8 s).
- **플래너 명령**: 플래너는 0.2 m/s 아래에서 발을 거의 들지 않으므로 이동 명령은 최소 0.2 m/s로 올리고(`min_step_v`), 제자리 회전은 작은 호로 바꾼다.
- **`settle_arms="walk"`**(기본): 멈추는 동안 팔을 걷기 팔로 유지하고, 다리가 loco로 돌아간 뒤 사람 팔로 0.8 s에 걸쳐 돌아간다. 사람의 '팔 내림'
  자세(손이 엉덩이 안쪽)로 바로 돌아가면 손이 엉덩이 링크를 눌러 SONIC 다리가 균형 걸음을 못 하던 낙상을 막는다([results.md](results.md) §4).
- 걷기 체크포인트는 `event.yaml`의 `hybrid_variant` 또는 `K1LIVE_HYBRID_VARIANT`(행사 설정 `models/k1/k1s_walk4/deploy/k1s_walk4_007000`).
- 메모리: 플래너 ONNX 때문에 sim_server가 split보다 약 2.4 GB 더 쓴다(약 3.1 GB).

### GEAR-SONIC 키네마틱 플래너 (`k1live/planner.py`)

`planner_sonic.onnx`(Hugging Face `nvidia/GEAR-SONIC`)를 C++ 배포 루프와 같은 규칙으로 돌리는 Python 포트다: 11개 입력, 30 → 50 Hz 리샘플,
재계획 규칙, 8프레임 블렌딩, 별도 추론 스레드. 출력(G1 자세)은 `G1ToK1`로 K1 관절로 바꾼다(골반에 G1 허리 roll·pitch를 접고, 발 자세 목표로 다리 IK,
상완 방향 맞춤). 카메라에서 오는 연속 명령에는 작은 변화 허용오차를 둬서 매 틱 재계획하지 않게 했다.

## 5. 서비스 (`k1live/service.py`)

- 하위 프로세스(sender, sim_server)를 띄우고 감시한다: 죽으면 1·2·4·8·15초 간격으로 다시 시작(1분 정상이면 초기화), 상태가 `stale_s`(3 s) 끊기면
  멈춘 것으로 보고 강제 종료 후 재시작, 다시 붙으면 현재 설정 전체를 재전송한다.
- 메모리가 `mem_limit_mb`(프로그램별)를 넘으면 **아무도 따라 하지 않을 때만** 그 프로그램을 다시 시작한다(10분에 한 번까지).
- 카메라 또는 시뮬레이션이 `fallback_after_s`(2 s) 끊기면 관객 화면을 대체 영상으로, 둘 다 `live_after_s`(3 s) 정상이면 실시간으로 돌린다.
- 자동 흐름: 사람이 `idle_after_s`(20 s) 없으면 대기, (자동 보정·시작을 켰을 때) 사람이 2 s 서 있으면 보정 → 시작.
- GEM-X sender가 종료 코드 5로 끝나면 그 자리에서 MediaPipe sender를 띄우고 패널 배너에 이유를 보인다.
- 운영자 설정(카메라, 걷기 모드, 사람 찾을 영역, 최대 속도, 미러, 백엔드, 팔 경로, 인식 등)은 `logs/k1live/settings.json`에 저장되어 `event.yaml`보다 우선한다.
- 세션 로그: `logs/k1live/<YYYYmmdd-HHMMSS>/` — `service.log`, `sender.log`, `sim_server.log`, `sim/`, `events.jsonl`, `metrics.csv`(5 s마다), `config.json`.
- `K1Demo.command`(실행기): 사전 점검 → 서비스(죽으면 3 s 뒤 다시) → 브라우저 창 두 개. 실행 중 잠자기를 막는다(macOS `caffeinate`, Linux `systemd-inhibit`).

## 6. 계측 정의

- **지연**: 그 프레임의 `HumanFrame`을 처음 쓴 제어 틱의 시각 − 캡처 시각(영상은 그 프레임의 예정 시각). 인식·IK·ZMQ·틱 대기를 모두 포함한다.
- **팔 오차**: 로봇 팔 10관절 − 그 순간의 IK 목표(각도 평균). "참조 대비"는 서버가 제한한 참조와 비교한 값.
- **손 위치 오차**: 같은 순간 로봇 손바닥 중심과 IK 목표 자세의 손바닥 중심 거리(몸통 좌표, 두 손 평균).
- **2D 팔 각도 오차**: 로봇 상완·전완을 몸통 정면에 투영한 각도와, 사람 영상에서 독립 검출기로 잰 2D 관절 각도의 차.
