# 공식 GEAR-SONIC 스택 (Linux + NVIDIA) — `linux/` 래퍼

이 저장소의 `sonic/`은 공식 C++/TensorRT 배포(`gear_sonic_deploy`, Unitree G1)를 Python + onnxruntime으로 옮긴 sim2sim 재현이다. 공식 경로 그대로
돌려야 할 때(실기 연결, 공식 결과와 대조)를 위해 upstream 문서(`docs/source/getting_started/{installation_deploy,installation_training,quickstart,download_models}.md`,
`docs/source/tutorials/{zmq,live_camera_teleop}.md`, `docs/source/user_guide/{training,new_embodiments}.md`)의 절차를 `linux/` 스크립트로 묶었다.

> 확인 범위: 래퍼를 처음 만들 때 확인한 것은 `bash -n`·shellcheck(경고 0), 공통 함수 단위 시험, `train_sonic_k1.sh prepare` 계약 시험, Mac에서 돌린
> ZMQ 호환 시험(§4)이다. 아래 소요 시간은 추정이다. 첫 실행 때 각 스크립트가 찍는 단계별 시간으로 바꿔 적는다.

## 0. 빠른 시작

```bash
# 로컬 → GPU 기계로 번들 복사 (linux/ + 송신기 scripts/, sonic/ + K1 패키지 sonic_k1/)
rsync -a --exclude .venv --exclude vendor --exclude models --exclude videos --exclude logs ./ <user>@<host>:~/sonic/bundle/
# GPU 기계에서, ~/sonic/bundle
bash linux/install_sonic_deploy.sh --check-only     # 무엇이 없는지만 본다
```

| # | 명령 | 하는 일 | 첫 실행 예상 |
|---|---|---|---|
| 1 | `bash linux/install_sonic_deploy.sh --check-only` | OS·드라이버·CUDA·TensorRT·apt 패키지 점검(아무것도 바꾸지 않음) | < 1분 |
| 2 | `sudo bash ~/sonic/sudo_steps.sh` | root가 필요한 것만 모아 둔 파일(1·4단계가 만들었을 때만): apt 패키지, CUDA 12.8 툴킷, (필요 시) 드라이버 | 수–20분 |
| 3 | TensorRT 10.13 TAR를 브라우저로 받아 `~/Downloads`에 | NVIDIA 개발자 로그인이 필요해 자동화하지 않음 | 회선 속도에 따름 |
| 4 | `bash linux/install_sonic_deploy.sh` | just·ONNX Runtime·cppzmq(홈 디렉터리, root 불필요), TensorRT 풀기, GR00T clone + LFS, `just build`, `.venv_sim`, HF 모델, 호환 검사기 빌드 | 20–40분 |
| 5 | `bash linux/install_gemx.sh` | GEM-X clone, Python 3.12 venv, torch cu128(sm_120 확인), soma + LFS, `install_env.sh`, `inputs/soma_assets` 절대경로 링크, import 시험 | 15–40분 |
| 6 | `bash linux/run_sim_teleop.sh --mode replay --pt <hpe_results.pt> --auto` | 카메라 없이 공식 3-pane 구성 확인 | 정책별 첫 기동 때 TensorRT 엔진 생성 수 분 |
| 7 | `bash linux/run_sim_teleop.sh --camera-check` → `--mode webcam` | GPU 기계의 카메라로 GEM-X 라이브 | 첫 실행 때 GEM-X 체크포인트 다운로드 |
| 7' | (로컬) `scripts/webcam_sender.py` + (GPU) `run_sim_teleop.sh --mode remote --zmq-host <송신 기계 IP>` | 다른 기계의 웹캠(MediaPipe) → GPU 기계의 SONIC (§4) | 1분 |

공통: 멱등(된 단계는 건너뜀), **sudo를 직접 실행하지 않음**(root 단계는 `~/sonic/sudo_steps.sh`로 모아 보여 주고 멈춤), 경로는 `~/sonic/sonic_env.sh`에
저장. 기본 경로 `SONIC_WORK=~/sonic`, `SONIC_ROOT=~/sonic/GR00T-WholeBodyControl`, `GEMX_ROOT=~/sonic/GEM-X`, `TensorRT_ROOT=~/TensorRT`
(환경 변수나 `--work/--sonic-root/--gemx-root/--trt-root`로 변경).

**RTX 50 시리즈(Blackwell, sm_120)**: upstream Docker는 CUDA 12.4.1 이미지라 지원하지 않는다 → 네이티브 설치(CUDA 12.8+ 드라이버·툴킷 + TensorRT 10.13 TAR).
upstream은 x86_64에서 TensorRT **10.13**을 요구하므로("다른 버전은 잘못된 추론 결과") `install_sonic_deploy.sh`는 10.13이 아니면 멈춘다. GEM-X·Isaac Lab
쪽 torch도 cu128 빌드여야 하고, 두 설치 스크립트 모두 `torch.cuda.get_arch_list()`에 `sm_120`이 있는지 확인한다. upstream `scripts/install_deps.sh`는
CUDA를 못 찾으면 apt로 12.4를 깔므로 50 시리즈에서는 CUDA 12.8을 먼저 설치하거나 `install_sonic_deploy.sh`를 쓴다.

## 1. 설치 스크립트

### `linux/install_sonic_deploy.sh`

| 단계 | upstream 수동 명령 | 스크립트 |
|---|---|---|
| preflight | — | Ubuntu 20.04/22.04/24.04, x86_64, CUDA ≥ 12.8(드라이버·nvcc). 부족하면 설치 명령을 sudo_steps에 기록 |
| apt | `./scripts/install_deps.sh` (sudo) | 같은 목록 + tmux, v4l-utils 중 없는 것만 sudo_steps로 |
| 홈 디렉터리 설치 | install_deps.sh가 sudo로 하던 것 | `just` → `~/.local/bin`, ONNX Runtime 1.16.3 → `~/.local/onnxruntime`, cppzmq 헤더 → `~/.local/include` |
| TensorRT | TAR 풀기, `TensorRT_ROOT` | `~/Downloads`·`~/sonic`·`~`에서 `TensorRT-10.13*.tar.gz`를 찾아 `~/TensorRT`로, 버전 확인 |
| repo | clone + `git lfs pull` | 없으면 clone, LFS pull, 포인터가 남았는지 확인 |
| build | `source scripts/setup_env.sh && just build` | 바이너리가 있으면 건너뜀(`--rebuild`) |
| sim venv | `install_mujoco_sim.sh` | `.venv_sim`이 import되면 건너뜀 |
| 모델 | `download_from_hf.py` | 파일이 있으면 건너뜀. `--sample`, `--sonic-v1-1` 선택 |
| 호환 검사기 | — | `linux/compat/zmq_v3_check` 빌드(공식 ZMQ subscriber 헤더 사용, §4) |

### `linux/install_gemx.sh`

`git clone --recursive NVlabs/GEM-X` → `uv venv .venv --python 3.12` → torch/torchvision **cu128** → `uv pip install -e third_party/soma` + `git lfs pull`
→ `bash scripts/install_env.sh`(끝난 뒤 torch가 바뀌었으면 cu128로 되돌림) → `inputs/soma_assets` **절대경로** 링크(상대경로면 조용히 끊긴다) →
`pyzmq scipy` → import 시험. 이미 있는 GEM-X 체크아웃은 `--gemx-root`로 지정하면 재사용하고 빠진 것만 채운다.

## 2. 실행 — `linux/run_sim_teleop.sh` (tmux 1세션, 3 pane)

```
┌ pane 0: MuJoCo sim ───────────┬ pane 1: C++ deploy (포커스) ─────────────────────┐
│ linux/sim_loop_keys.py        │ ./deploy.sh [--cp low_latency] --input-type zmq   │
│ (= run_sim_loop.py + 터미널 9) │            --zmq-host <host> sim                  │
├ pane 2: 포즈 입력 ─────────────┤  키: ]  시작 · ENTER 스트리밍 on/off · O 비상정지 │
│ webcam_stream.py / 리플레이 /  │                                                   │
│ 원격 송신 모니터               │                                                   │
└───────────────────────────────┴───────────────────────────────────────────────────┘
```

```bash
bash linux/run_sim_teleop.sh --mode replay --pt <hpe_results.pt>       # 카메라 없이 (GEM-X 결과 리플레이)
bash linux/run_sim_teleop.sh --camera-check --source 0                  # GEM-X --kp-only --show 만
bash linux/run_sim_teleop.sh --mode webcam --source 0                   # 라이브
bash linux/run_sim_teleop.sh --mode video --source <clip.mp4>           # 영상 파일을 GEM-X로
bash linux/run_sim_teleop.sh --mode remote --zmq-host <송신 기계 IP>     # 다른 기계의 송신기 (§4)
bash linux/run_sim_teleop.sh ... --policy low_latency                   # SMPL lookahead 4프레임 정책
bash linux/run_sim_teleop.sh ... --auto          # Proceed → Init Done 후 ']' → '9' 까지 자동 (sim 전용)
bash linux/run_sim_teleop.sh ... --auto-stream   # + 입력이 준비되면 ENTER 까지 (sim 전용)
```

- **순서(공식과 같음)**: pane 1 `Proceed? [Y/n]` Enter → `Init Done` 뒤 `]` → sim에서 `9`(로봇 내리기) → 카메라 앞에 똑바로 서서 pane 2 `soma=yes` 확인 →
  pane 1 `ENTER`(스트리밍) → 처음엔 천천히 팔만. 멈출 때 `ENTER`, 종료 `O`.
- **화면 없이 SSH만 있을 때**: `$DISPLAY`가 없으면 로컬 X(`:0`)를 찾아 MuJoCo 창·GEM-X 미리보기를 그 모니터에 띄우고, 디스플레이가 없으면 sim을
  `--headless`로 돌린다. upstream은 탄성밴드(`9`)를 MuJoCo 창에서만 받으므로 `linux/sim_loop_keys.py`가 pane 0 터미널의 `9`(7/8 길이)를 같은 객체에 넘긴다.
- tmux: `Ctrl-b o` 다음 pane, `Ctrl-b d` 분리, `tmux attach -t sonic_teleop` 재접속. pane 출력은 `~/sonic/logs/<시각>/pane{0,1,2}.log`.
- `deploy.sh`에 `--zmq-port/--zmq-topic`을 붙이지 않는다(모르는 인자를 인터페이스 이름으로 받아 `sim`을 덮어쓸 수 있다는 upstream 경고). 기본 5556/`pose`가
  송신기들과 같다.

## 3. 다른 기계의 웹캠 → Linux SONIC

```
송신 기계 (카메라, MediaPipe CPU)                    GPU 기계
scripts/webcam_sender.py  ── PUB bind tcp://*:5556 ──►  deploy.sh --input-type zmq --zmq-host <송신 기계 IP> sim
  MediaPipe → SMPL-24 → v3                            SUB connect → SMPL 인코더(mode 2) → G1 (MuJoCo / 실기)
```

```bash
# 송신 기계
.venv/bin/python scripts/webcam_sender.py --source 0          # 창에서 c 보정, l 다리, q 종료
# GPU 기계
bash linux/run_sim_teleop.sh --mode remote --zmq-host <송신 기계 IP>    # pane 2가 5초마다 공식 파서로 스트림을 검사
# 방화벽이 막혀 있으면 송신 기계에서 SSH 역터널:
ssh -N -R 5556:localhost:5556 <user>@<GPU 기계>    # GPU 기계에서는 run_sim_teleop.sh --mode remote --tunnel
```

## 4. ZMQ v3 호환성 검증 (Mac에서 실행)

도구: `linux/zmq_v3_compat.py`(공식 C++ 파서를 줄 단위로 옮긴 Python 포트: `ZMQPackedMessageSubscriber::PollOnce` → `DecodeIntoMotionSequence` →
`StreamedMotionMerger::ValidateIncomingData`)와 `linux/compat/zmq_v3_check.cpp`(공식 헤더 `zmq_packed_message_subscriber.hpp`를 그대로 컴파일해 같은
바이트를 디코드; libzmq가 없으면 캡처 재생 shim `zmq_replay_shim/zmq.hpp`로 빌드).

| 시험 | 결과 |
|---|---|
| 바이트 대조: `sonic/stream.py:v3_message` vs 공식 `pack_pose_message`, 50개 | 바이트 동일(우리 쪽만 끝에 선택 필드 `timestamp_monotonic`) |
| 파서 포트 디코드 + 왕복 | 전부 v3로 디코드, 값 일치 |
| 거부 동작(필드 누락, 프레임 수 불일치, 타입·버전·토픽 오류, 잘린 payload 등 9가지) | 9/9 C++와 같은 메시지로 거부 |
| 종단: 실제 `webcam_sender.py --video`를 12초 수신(deploy와 같은 SUB 옵션) | 293개 전부 COMPATIBLE, 30.0 Hz, 간격 p95 42 ms, 150 ms 초과 0, frame_index 갭·역행 0 |
| 공식 C++ 헤더로 같은 캡처 재생 | 293/293 통과 |

와이어 형식: `pose`(4 B) + 헤더 1280 B(JSON, NUL 패딩) + 데이터 804 B = 메시지당 2088 B. 필드 `smpl_joints f32[1,24,3]`, `smpl_pose f32[1,21,3]`,
`joint_pos f32[1,29]`, `joint_vel f32[1,29]`, `body_quat f32[1,4]`(wxyz), `frame_index i64[1]`, `timestamp_monotonic f64[1]`.

```bash
.venv/bin/python linux/zmq_v3_compat.py selftest
.venv/bin/python linux/zmq_v3_compat.py e2e --video <clip.mp4> --port 5599 --save /tmp/e2e.cap
bash linux/compat/build_zmq_v3_check.sh --shim -o /tmp/zmq_v3_check      # 공식 헤더 + 재생 shim (nlohmann-json 필요)
ZMQ_SHIM_REPLAY=/tmp/e2e.cap /tmp/zmq_v3_check --count 100000
```

알고 쓸 것:

- deploy는 `--zmq-host`가 문자열 그대로 `localhost`일 때만 `timestamp_monotonic`을 자기 시계로 믿는다(스트림 지연 통계·150 ms 경보용, 제어에는 안 씀).
  SSH 터널은 `127.0.0.1`로 붙인다(`--tunnel`이 그렇게 한다).
- 공식 merger는 메시지 1개 = 50 Hz 1프레임으로 쌓는다. 30 Hz 송신기도 공식 GEM-X 예제와 같은 방식으로 재생된다.
- 송신기를 다시 시작하면 frame_index가 0으로 돌아가 merger가 리셋된다(로봇이 새 포즈로 튐) → 스트림 OFF 상태에서만 다시 시작한다.
- 골반 규약: 기본 `--pelvis j0`(학습 규약), 공식 GEM-X 예제는 골반 원점(`--pelvis zero`). 둘 다 같은 인코더가 받는다.
- Wi-Fi는 절전 때문에 패킷이 뭉쳐 올 수 있다. 유선 또는 절전 끄기를 권한다.

## 5. 실기 (G1)

시뮬레이터를 반드시 끄고, 겐트리와 비상정지 담당자를 둔다.

```bash
./deploy.sh --input-type zmq --zmq-host localhost real      # 배포가 온보드면 --zmq-host <워크스테이션 IP>
```

팔을 크게 뻗거나 빠르게 휘두르면 정책이 균형을 위해 앞으로 걸어 나간다(upstream 경고). `run_sim_teleop.sh --auto`는 sim 전용이다.

## 6. K1 학습 래퍼 — `linux/install_sonic_training.sh`, `linux/train_sonic_k1.sh`

공식 C++ deploy는 G1 전용(29 DOF)이라 K1에는 학습이 필요하다. 실제로 쓴 학습 절차는 [training.md](training.md)(`sonic_k1_train/`)이고, 아래는 같은
`sonic_k1/` 패키지 계약을 공식 설치 경로에서 돌리는 일반 래퍼다.

- `linux/install_sonic_training.sh`: Python 3.11 venv → `isaacsim[all,extscache]==5.1.0` → torch 2.7.0 cu128 → Isaac Lab **v2.3.2** + `./isaaclab.sh --install`
  → `pip install -e "gear_sonic/[training]"` → `check_environment.py --training`. Isaac Sim 첫 import는 NVIDIA EULA를 묻는다 — 동의하는 경우에만 `--accept-eula`.
- `linux/train_sonic_k1.sh prepare | smoke | train --detach [--gpus N] | resume --detach | export | status` — `sonic_k1/patches`를 이름순으로 적용,
  `overlay/`를 복사, `train.env`(EXP, NUM_ENVS, ITERS, PROJECT, EXTRA)를 읽는다. 결과는 `~/sonic/k1_runs/<PROJECT>/<exp>-<시각>/`, export는
  `~/sonic/k1_export/<시각>/`(+ `SHA256SUMS`).

## 7. 체크리스트

- [ ] `install_sonic_deploy.sh` 블로커 0: CUDA ≥ 12.8, TensorRT 10.13, apt 패키지, `just build` 성공
- [ ] `run_sim_teleop.sh --mode replay … --auto`: `Init Done` → `]` → `9` → 로봇 기립(TensorRT 엔진 캐시 생성)
- [ ] `install_gemx.sh`: torch sm_120, `inputs/soma_assets` 절대경로 링크, import 시험
- [ ] `run_sim_teleop.sh --camera-check`: 전신(발 포함), 2–3 m, 정면 조명, 한 사람
- [ ] 라이브: pane 2 `soma=yes` 확인 후 `ENTER`, 천천히 팔부터
- [ ] (분산) GPU 기계에서 `zmq_v3_check --host <송신 기계 IP>` → `RESULT COMPATIBLE`
- [ ] 행사 전날: 정책별 첫 기동(TensorRT 엔진)과 GEM-X 체크포인트 다운로드를 미리 끝내 둔다
