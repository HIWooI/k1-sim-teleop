# 설치

## 1. 요구 사항

| 항목 | 권장 |
|---|---|
| OS | macOS(Apple Silicon) 또는 Ubuntu 22.04/24.04 x86_64 |
| Python | 3.12 (`requirements.txt`의 버전으로 확인함). Ubuntu는 `sudo apt install python3.12-venv libgles2`(MediaPipe가 `libGLESv2.so.2`를 연다. 데스크톱 Ubuntu에는 보통 이미 있다) |
| CPU·메모리 | 8코어 이상, RAM 16 GB 이상. 데모는 CPU 약 1.5–2.5코어, 메모리 약 2–2.5 GB(split) / 약 4.5 GB(hybrid)를 쓴다 |
| GPU | 필요 없음(MediaPipe 경로). GEM-X 인식을 쓰려면 NVIDIA GPU(VRAM 8 GB 이상, 실측 사용량 6.2 GB) |
| 카메라 | USB 웹캠 1080p 30 fps, 화각 65–78°. 영상 파일을 카메라처럼 쓸 수도 있다(`--source <파일>`) |
| 브라우저 | Chrome(관객 화면 '앱 창'). Firefox도 동작 |

## 2. 파이썬 환경과 공개 모델

```bash
scripts/setup.sh                       # macOS: Homebrew python3.12, Linux: PYTHON=python3.12 scripts/setup.sh
```

`scripts/setup.sh`가 하는 일(멱등):

- `.venv` 생성, `requirements.txt` 설치
- `vendor/GR00T-WholeBodyControl` 얕은 clone(LFS는 필요한 파일만): G1 MJCF·메시, 예제 참조 모션, `human_joints_info.pkl`
- `scripts/download_models.py`: Hugging Face `nvidia/GEAR-SONIC`의 G1 정책 ONNX(release·low_latency·sonic_v1_1), `config.json`, 예제 데이터 → `models/`
- MediaPipe `pose_landmarker_{full,heavy}.task` → `models/`

K1 hybrid·sonic 백엔드에 필요한 플래너는 따로 받는다(774 MB):

```bash
.venv/bin/python -c "from k1live.planner import fetch_planner; fetch_planner()"   # → models/planner/planner_sonic.onnx
```

Linux에서는 렌더가 `MUJOCO_GL`이 없으면 스스로 `egl`을 쓴다. 디스플레이 없는 서버면 `MUJOCO_GL=egl`을 명시한다. `opencv-python` 대신
`opencv-python-headless`도 된다.

## 3. ROBOTIS K1 자산

K1 MJCF와 메시는 ROBOTIS가 공개한 [ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens)(Apache-2.0)에서 온다.

```bash
git clone https://github.com/ROBOTIS-GIT/ai_sapiens third_party/ai_sapiens       # 다른 곳이면 AI_SAPIENS_ROOT=<경로>
.venv/bin/python sonic_k1/scripts/stage_assets.py                               # 학습·hybrid용 URDF·MJCF·메시(m 단위) 스테이징
```

- 시뮬레이션(`k1/sim.py`)은 `$AI_SAPIENS_ROOT/ai_sapiens_description/mujoco/k1/k1.xml`과 `meshes/k1_rev1`을 읽는다.
- `stage_assets.py`는 `sonic_k1/overlay/gear_sonic/data/assets/robot_description/`에 `urdf/k1/k1.urdf`, `urdf/k1/meshes/*.stl`(git 제외),
  `mjcf/k1.xml`을 쓴다. hybrid 백엔드와 학습 둘 다 이 스테이징 결과를 쓴다.

## 4. 정책 가중치 (GitHub Release)

`scripts/fetch_weights.sh`가 아래 표의 처음 세 줄을 Release `weights-v1`에서 받아 제자리에 둔다([weights.md](weights.md)). 나머지는 직접 학습한다.

| 경로 | 쓰는 곳 | 구하는 법 |
|---|---|---|
| `k1/assets/loco_armdriven_r2/exported/policy.onnx` | split·hybrid의 다리(보행 정책) | `scripts/fetch_weights.sh` |
| `k1/assets/student_asset/exported/policy.onnx` | split의 팔(mimic student), hybrid의 그림자 실행 | `scripts/fetch_weights.sh` |
| `models/k1/k1s_walk4/deploy/k1s_walk4_007000/` (`model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml`) | hybrid가 걸을 때(행사 설정) | `scripts/fetch_weights.sh`(직접 학습은 [training.md](training.md)). 다른 폴더면 `event.yaml`의 `hybrid_variant` 또는 `K1LIVE_HYBRID_VARIANT` |
| `models/k1/k1s_la4/deploy_002000/` | `sonic` 백엔드(선택) | 배포하지 않음. [training.md](training.md)대로 학습·export |
| `K1LIVE_HUMAN_VARIANT` 폴더 | `sonic_human` 백엔드(선택, 실험) | 배포하지 않음. 같은 방법(SMPL 인코더 포함 run) |

없을 때의 동작: 사전 점검이 '실패'(K1 정책) 또는 '경고'(hybrid·sonic)를 보이고, hybrid 파일이 없으면 서비스가 split으로 시작한다. 모델 없이
화면·패널만 보려면 `python -m k1live.service --mock`.

## 5. GEM-X 인식 (선택, NVIDIA GPU)

GEM-X 경로는 이 저장소 밖의 **GEM-X 실시간 인식 폴더**를 import해서 쓴다: `demo_webcam.py`(모델 빌드, 키포인트 박스 전파, 인과 창 데이터 구성,
SOMA FK)와 그 안의 `GEM-X/` 체크아웃·venv(torch + CUDA). 이 폴더는 저장소에 포함하지 않는다. GEM-X 자체는
[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)이고, `linux/install_gemx.sh`가 GEM-X 체크아웃과 venv를 만드는 절차를 담고 있다.

| 설정 | 기본 | 뜻 |
|---|---|---|
| `K1LIVE_GEMX_PERCEPTION` / `event.yaml gemx.perception_dir` | `third_party/gemx_perception` | `demo_webcam.py`가 있는 폴더 |
| `event.yaml gemx.python` | `<perception_dir>/GEM-X/.venv/bin/python` | GEM-X venv의 python |
| `event.yaml gemx.cmd` | 없음 | 인식만 다른 GPU 기계에서 돌릴 때의 GEM-X 프로세스 실행 명령(ssh) |
| `GEMX_PY` | 위와 같음 | `scripts/render/dump_gemx_tracks.sh`가 쓰는 GEM-X python |

확인: `<GEM-X python> k1live/gemx_worker.py --check`(CUDA·import·체크포인트), 또는 `python -m k1live.preflight --gemx-full`(모델까지 로딩).
GEM-X를 쓸 수 없으면 서비스는 MediaPipe로 돈다.

## 6. 환경 변수

| 변수 | 뜻 |
|---|---|
| `AI_SAPIENS_ROOT` | ROBOTIS ai_sapiens 체크아웃(기본 `third_party/ai_sapiens`) |
| `K1LIVE_GEMX_PERCEPTION` | GEM-X 실시간 인식 폴더(기본 `third_party/gemx_perception`) |
| `K1LIVE_HYBRID_VARIANT` | hybrid 걷기 체크포인트 폴더 |
| `K1LIVE_HUMAN_VARIANT` | `sonic_human` 체크포인트 폴더 |
| `K1_EVENT_VIDEOS` | 행사 미디어·사람 영상 폴더(기본 `videos/event`): `event.yaml`의 `{event_dir}` 기본값이자 평가·렌더 스크립트의 입력 폴더 |
| `K1LIVE_EVENT_DIR` | 서비스의 `{event_dir}`만 바꿀 때(다른 컴퓨터, 행사 번들) — `K1_EVENT_VIDEOS`보다 우선 |
| `K1LIVE_PORT_OFFSET` | 모든 포트에 더하는 값(한 컴퓨터에서 여러 벌) |
| `K1LIVE_LOCO_PROFILE`, `K1LIVE_CROP` | 그 실행에만 쓰는 걷기 모드·사람 찾을 영역 |
| `K1LIVE_ARM_IK`, `K1LIVE_ARM_MODE`, `K1LIVE_ARM_VMAX`, `K1LIVE_ARM_AMAX` | sender·sim_server 팔 경로 기본값 |
| `K1DEMO_ARGS`, `K1DEMO_DISPLAY_POS`, `K1DEMO_FULLSCREEN`, `K1DEMO_NO_CAMERA`, `K1DEMO_NO_BROWSER`, `K1DEMO_BROWSER`, `K1DEMO_CHROME_ARGS` | 실행기(`K1Demo.command`) 옵션 — 파일 머리말 참고 |
| `SONIC_K1_WORK`, `SONIC_K1_HOST`, `SONIC_K1_MOTIONS` | 학습 서버 작업 폴더·접속 이름·모션 폴더([training.md](training.md)) |

`k1demo.env`가 저장소 폴더에 있으면 실행기와 운영 시험 스크립트가 먼저 읽는다(행사 번들이 만든다).

## 7. 확인

```bash
.venv/bin/python -m k1live.preflight [--no-camera] [--quick] [--gemx-full] [--json]   # 통과/경고/실패 표
.venv/bin/python -m pytest -q tests sonic_k1/tests                                    # 단위·통합 시험
```

정책 가중치나 K1 메시가 없으면 그것을 쓰는 시험은 건너뛰거나 실패한다.
