# k1-sim-teleop — 웹캠으로 K1 휴머노이드를 시뮬레이션에서 원격 조종

카메라 앞에서 사람이 팔을 움직이면 MuJoCo 속 **ROBOTIS AI Sapiens K1**(23 DOF)이 실시간으로 따라 하고, 사람이 걸으면
로봇도 걷는 데모입니다. 운영 패널과 관객 화면, 자동 복구, 사전 점검까지 갖춘 현장용 서비스로 만들었습니다.
실제 로봇은 쓰지 않습니다(시뮬레이션만).

- **상체**: 사람 팔 → 뼈 방향 팔 IK → PD 팔 제어(서 있을 때)
- **보행**: 사람의 걷기 의도 → 속도 명령 → NVIDIA GEAR-SONIC 키네마틱 플래너가 걸음 참조를 만들고, GEAR-SONIC 학습 코드로
  K1용으로 학습한 정책이 그 참조를 따라 걷는다 (built with NVIDIA GEAR-SONIC)
- **행사 기본 백엔드 = hybrid**: 서 있을 때는 팔에 집중(보행 정책 `loco_armdriven_r2` 다리 + PD 팔), 걷기 명령이 이어지면 K1 SONIC
  보행 정책으로 넘어가고, 멈추면 다시 돌아온다
- **인식**: MediaPipe Pose(CPU, 어느 노트북에서나 기본 대체 경로) 또는 GEM-X(NVIDIA GPU, 전신 3D·깊이; 외부 폴더 필요)

```
카메라 / 영상 파일(가짜 카메라)
   │
   ▼
sender (k1live/sender.py: MediaPipe  |  k1live/sender_gemx.py + k1live/gemx_worker.py: GEM-X, GPU)
   단일 인물 게이트 → HumanMapper: 팔 IK(k1live/arm_ik.py) · 걷기 의도(k1live/locomotion.py) · 미러
   │ HUMAN (ZMQ :5701, ~30 Hz)                       /cam.mjpg :8701
   ▼
sim_server (k1live/sim_server.py, 50 Hz 제어 · MuJoCo 200 Hz)
   팔 참조 제한 → 보행 속도 제한 · 방향 유지 → 백엔드(split | hybrid | sonic | sonic_human) → 낙상 감지 · 자동 리셋
   │ SIM_STATE (:5703)        render 프로세스 → /robot.mjpg :8702
   ▼
service (k1live/service.py, :8700) ── 운영 패널 /  · 관객 화면 /display · 하위 프로세스 감시·재시작 · 대체 영상
   ▲
K1Demo.command (사전 점검 → 서비스 → 브라우저 창 두 개)
```

자세한 구조는 [docs/architecture.md](docs/architecture.md)를 봅니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `k1live/` | 실시간 서비스: sender, sim_server, 백엔드, 플래너 포트, GEM-X 연결, 운영 패널·관객 화면(`web/`), 사전 점검, 장시간 시험 |
| `k1/` | K1 MuJoCo 시뮬레이션(split 구동), CPU 기구학, 제스처, ROBOTIS K1 자산(`assets/`) |
| `sonic/` | GEAR-SONIC 배포 루프의 Python 포트(G1·K1), SMPL 변환, MediaPipe 포즈 |
| `sonic_k1/` | GEAR-SONIC 학습 코드에 K1 embodiment를 더하는 overlay·patch, K1 자산 스테이징 |
| `sonic_k1_train/` | K1 SONIC 학습(데이터 생성, GPU 서버 스크립트, MuJoCo 평가) |
| `scripts/` | 설치·모델 받기, 평가 도구, 렌더(`render/`), 운영 시험(`ops/`), 행사 번들 |
| `linux/` | Linux + NVIDIA에서 공식 GEAR-SONIC 스택(C++/TensorRT, G1) 설치·실행 래퍼 |
| `tests/` | 단위·통합 시험 |
| `docs/` | 문서 |

## 빠른 시작

**발표자 팔 모방 + SHAPE Dual Stage 화면을 Linux PC 한 대에서 실행**하려면
[발표용 설치·실행 안내](docs/presenter.md)를 사용합니다:
`PYTHON=python3.12 bash scripts/setup_presenter.sh` → `bash run_presenter.sh --source 0`.
이 구성은 서 있는 상태의 팔 모방이며 전신 자세 모방은 아닙니다.

자세한 설치는 [docs/setup.md](docs/setup.md), 운영은 [docs/runbook.md](docs/runbook.md)에 있습니다.

```bash
# 0) 받기 — 필요한 것: git, curl, Python 3.12 (macOS: brew install python@3.12, Ubuntu: sudo apt install python3.12-venv libgles2)
git clone https://github.com/HIWooI/k1-sim-teleop && cd k1-sim-teleop

# 1) 환경 + 공개 모델(GEAR-SONIC ONNX, 플래너, MediaPipe, GR00T 저장소 일부)
scripts/setup.sh                                  # macOS 기준; Linux는 PYTHON=python3.12 scripts/setup.sh
.venv/bin/python -c "from k1live.planner import fetch_planner; fetch_planner()"   # planner_sonic.onnx

# 2) ROBOTIS K1 자산 (공개, Apache-2.0)
git clone https://github.com/ROBOTIS-GIT/ai_sapiens third_party/ai_sapiens
.venv/bin/python sonic_k1/scripts/stage_assets.py

# 3) K1 정책 가중치 (GitHub Release weights-v1, docs/weights.md)
scripts/fetch_weights.sh

# 4) 실행
.venv/bin/python -m k1live.preflight --no-camera  # 사전 점검
./K1Demo.command                                  # macOS 더블클릭 / Ubuntu: bash K1Demo.command
.venv/bin/python -m k1live.service --mock         # 모델 없이 패널·관객 화면만 확인
```

`K1Demo.command`를 실행하면 사전 점검 표가 나온 뒤 브라우저 창 두 개(운영 패널 `http://127.0.0.1:8700/`, 관객 화면
`http://localhost:8700/display`)가 열립니다. 카메라 앞 2.5–3.5 m에 서서 `C`(보정) → `Space`(시작)를 누르면 로봇이 따라 하고, 앞으로
한 걸음 나오면 걷습니다. `Esc`는 비상정지, 터미널 창을 닫으면 종료됩니다. macOS는 처음에 터미널의 카메라 권한을 허용해야 합니다.
웹캠 대신 영상 파일로 해 보려면 `K1DEMO_ARGS="--source <영상>" ./K1Demo.command`. 키·문제 해결은
[docs/runbook.md](docs/runbook.md)의 '한 장 요약'에 있습니다.

가중치 없이 바로 돌려 볼 수 있는 것은 G1(Unitree) GEAR-SONIC Mac 포트입니다: `./run_webcam.sh`(웹캠 → MediaPipe → SMPL → GEAR-SONIC
release 정책 → MuJoCo G1).

## 포함하지 않은 것과 구하는 법

| 항목 | 위치 | 구하는 법 |
|---|---|---|
| K1 SONIC 보행 정책(행사 설정: walk4 7000) | `models/k1/k1s_walk4/deploy/k1s_walk4_007000/` | Release `weights-v1` — `scripts/fetch_weights.sh` ([docs/weights.md](docs/weights.md)). 직접 학습은 [docs/training.md](docs/training.md) |
| K1 정책(`loco_armdriven_r2`, `student_asset`) | `k1/assets/*/exported/policy.onnx` | 같은 Release·스크립트. split·hybrid 백엔드에 필요 |
| K1 메시 | `sonic_k1/overlay/.../urdf/k1/meshes/` | `sonic_k1/scripts/stage_assets.py`가 ROBOTIS ai_sapiens에서 만든다 |
| GEAR-SONIC ONNX·플래너, MediaPipe 모델 | `models/` | `scripts/setup.sh`, `fetch_planner()` (NVIDIA Open Model License / Apache-2.0) |
| SMPL-X 기반 `human_joints_info.pkl` | `vendor/GR00T-WholeBodyControl/...` | `scripts/setup.sh`가 GR00T 저장소에서 받는다(SMPL-X 라이선스, 재배포 금지) |
| GEM-X 실시간 인식 폴더 | `third_party/gemx_perception` 또는 `K1LIVE_GEMX_PERCEPTION` | 별도 준비(`demo_webcam.py`가 있는 GEM-X 실시간 데모 폴더). 없으면 서비스가 MediaPipe로 돈다 |
| 시연·평가 영상 | `videos/` | 포함하지 않음. 아무 전신 영상을 `--source`로 넣으면 된다 |

## 주요 결과

평가는 영상 파일을 카메라처럼 실시간 재생해서 했고, 가능한 것은 행사 인식 경로(GEM-X 실시간 경로)를 기준으로 쟀습니다.
방법과 전체 표는 [docs/results.md](docs/results.md)에 있습니다.

| 항목 | 결과 |
|---|---|
| 낙상 | hybrid 서기↔걷기 전환 265–402회 낙상 0. GEM-X 입력 재현 세트(61.7분)에서 멈출 때 낙상 16 → 0(SETTLE 동안 걷기 팔 유지) |
| 팔 정확도 (GEM-X 입력) | 서 있을 때 서버 팔 참조 대비 1.54°. 로봇 2D 팔 각도 오차 16.3° → 5.3°(팔 IK·PD 팔 개선 전후), 사람 대비 지연 200 → 67 ms |
| 지연 | 캡처 → 제어 반영 p95: MediaPipe 43 ms, GEM-X 83–96 ms (기준 ≤ 150 ms) |
| 걸음 품질 (split → hybrid walk4 7000) | 보폭 0.15 → 0.25 m, 걸음 수 4.3 → 2.2 /s, 발 들림 3.0 → 4.8 cm, 무릎 굽힘 45 → 31° |
| 장시간 | 2시간 장애 주입 시험 통과(장애 8건 자동 복구, 낙상 0, 50 Hz). GEM-X 행사 조건 20분: 지연 p95 96 ms, 50 Hz, 낙상 0 |
| 사람 전신 정책(BONES-SEED 학습, 시도) | 수락 기준 미달 — 팔 13.6° vs split 2.05°. 행사 설정에는 쓰지 않음 |

## 문서

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 프로세스·포트·메시지, 인식, 팔 IK, 걷기 의도, 백엔드, 서비스 |
| [docs/setup.md](docs/setup.md) | 설치, 모델·자산 준비, 환경 변수 |
| [docs/runbook.md](docs/runbook.md) | 운영 매뉴얼: 실행, 패널, 설정 파일, 문제 해결, 운영 시험 도구 |
| [docs/training.md](docs/training.md) | GEAR-SONIC 방식 K1 학습: overlay, 데이터, 보행 학습, BONES 실험 |
| [docs/weights.md](docs/weights.md) | K1 정책 가중치(Release): 받기, 내용, 학습 데이터, 라이선스 |
| [docs/results.md](docs/results.md) | 평가 방법과 결과 |
| [docs/linux_official.md](docs/linux_official.md) | 공식 GEAR-SONIC 스택(Linux, C++/TensorRT) 래퍼 |
| [docs/g1_to_k1_mapping.md](docs/g1_to_k1_mapping.md) | G1(29 DOF) → K1(23 DOF) 관절 매핑 |

시험: `.venv/bin/python -m pytest -q tests sonic_k1/tests` (정책 가중치·K1 자산·시험 영상이 없으면 해당 시험은 건너뛴다).

## 라이선스

이 저장소의 코드는 [Apache License 2.0](LICENSE)입니다. NVIDIA GR00T-WholeBodyControl(GEAR-SONIC)과 ROBOTIS ai_sapiens·cyclo_lab에서
가져오거나 고친 파일, 실행할 때 받는 모델·데이터의 라이선스는 [NOTICE](NOTICE)와 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 봅니다.
NVIDIA, GEAR-SONIC, ROBOTIS, AI Sapiens는 각 소유자의 상표이며, 이 프로젝트는 그들과 제휴하거나 그들의 보증을 받은 것이 아닙니다.
