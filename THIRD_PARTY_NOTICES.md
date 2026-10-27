# 제3자 고지

이 저장소의 코드는 Apache License 2.0([LICENSE](LICENSE))이다. 아래는 이 저장소에 들어 있는 제3자 유래 파일과, 이 저장소에는
없지만 설치·실행·학습 때 받거나 쓰는 것들의 라이선스다. 법적 문구는 원문(영문)을 그대로 옮겼다. 각 라이선스의 정확한 조건은
링크한 원문이 우선한다.

## 1. 이 저장소에 들어 있는 제3자 유래 파일

### NVIDIA GR00T-WholeBodyControl (GEAR-SONIC) — Apache License 2.0

<https://github.com/NVlabs/GR00T-WholeBodyControl> (코드는 Apache-2.0, 모델 가중치는 NVIDIA Open Model License)

| 파일 | 관계 |
|---|---|
| `sonic_k1/overlay/gear_sonic/envs/manager_env/robots/k1.py` | upstream `robots/h2.py`를 본뜬 K1 로봇 설정(수정) |
| `sonic_k1/overlay/gear_sonic/config/**`, `sonic_k1_train/overlay/gear_sonic/config/**` | upstream `sonic_release.yaml` 등을 K1용으로 고친 실험 설정 |
| `sonic_k1/patches/*.patch`, `sonic_k1_train/patches/*.patch` | upstream 소스에 대한 차분(문맥 줄은 upstream 원문) |
| `sonic/{deploy,params,rot,motion,sim,runner,smpl,stream,gemx}.py` | upstream C++ 배포 루프(`g1_deploy_onnx_ref`)·Python 유틸의 Python 포트(수정) |
| `k1live/planner.py` | upstream 키네마틱 플래너 배포 코드의 Python 포트(수정) |
| `linux/zmq_v3_compat.py`, `linux/sim_loop_keys.py`, `linux/compat/zmq_v3_check.cpp` | upstream ZMQ 수신기·시뮬레이션 루프 대조용 포트·래퍼 |

`sonic_k1_train/overlay/gear_sonic/envs/manager_env/mdp/smpl_gemx_aug.py`는 이 저장소에서 새로 쓴 코드다.

### ROBOTIS AI Sapiens — Apache License 2.0

<https://github.com/ROBOTIS-GIT/ai_sapiens>

| 파일 | 관계 |
|---|---|
| `k1/assets/k1.urdf` | `ai_sapiens_description/urdf/k1_rev1` xacro에서 생성된 URDF |
| `k1/assets/k1_config.yaml` | `ai_sapiens_sim2real/config/k1_config.yaml` 일부(`robot_joint_order`) |
| `sonic_k1/overlay/gear_sonic/data/assets/robot_description/urdf/k1/k1.urdf`, `.../mjcf/k1.xml` | 같은 URDF·MJCF를 `sonic_k1/scripts/stage_assets.py`로 고친 것(머리 링크 병합, 메시 경로·단위) |
| (저장소에 없음) K1 메시 STL | `stage_assets.py`가 ai_sapiens에서 만들어 `.gitignore` 대상 폴더에 둔다 |

### ROBOTIS cyclo_lab — Apache License 2.0

<https://github.com/ROBOTIS-GIT/cyclo_lab>: `sonic_k1/overlay/gear_sonic/envs/manager_env/robots/k1.py`의 K1 액추에이터 상수(QC080·QC060
토크·속도·armature)는 cyclo_lab `K1_rev1.py`의 값이다.

### K1 정책 파일 — 작성자의 이전 프로젝트에서 작성·학습

ROBOTIS cyclo_lab 학습 환경을 써서 이 저장소 작성자가 이전 프로젝트에서 만든 파일이다.

- `k1/assets/student_asset/expert_adapter.py`: 작성자가 작성했다(2026-08-13). 정책마다 다른 관절 순서·관측·action scale을 한
  순서로 맞춘다. cyclo_lab 저장소 안에서 작성해 그 저장소의 머리말(Copyright 2026 ROBOTIS CO., LTD., Apache License 2.0)이 붙어 있다.
- `k1/assets/loco_armdriven_r2/params/sim2real.yaml`: 작성자가 cyclo_lab으로 학습한 보행 정책(2026-08-21)을 export할 때 나온
  배포 파라미터(관절 순서, PD 게인, 기본 자세).
- `k1/assets/student_asset/params/sim2real.yaml`: 작성자가 학습한 mimic student 정책(2026-08-14)의 배포 파라미터. 같은 파일이
  ROBOTIS cyclo_mjlab 공개 저장소(Apache-2.0)에도 있다.
- 두 정책의 가중치(`exported/policy.onnx`)는 포함하지 않는다.

## 2. 설치·실행 때 받는 것 (재배포하지 않음)

| 항목 | 받는 곳 | 라이선스 |
|---|---|---|
| GEAR-SONIC 정책 ONNX(release·low_latency·sonic_v1_1), `planner_sonic.onnx`, `config.json` | Hugging Face `nvidia/GEAR-SONIC` (`scripts/download_models.py`, `k1live.planner.fetch_planner`) | NVIDIA Open Model License — "Licensed by NVIDIA Corporation under the NVIDIA Open Model License." |
| GEAR-SONIC 예제 데이터(`--sample`), GR00T 예제 참조 모션 CSV | Hugging Face / GR00T 저장소 LFS (`scripts/setup.sh`) | 원 저장소 조건을 따른다 |
| `human_joints_info.pkl` (SMPL-X 휴지 골격에서 유도) | GR00T 저장소 LFS (`scripts/setup.sh`) | SMPL-X 모델 라이선스(비상업, 재배포 금지) <https://smpl-x.is.tue.mpg.de/modellicense.html> — 이 파일과 이를 담은 행사 번들은 공개하지 않는다 |
| Unitree G1 MJCF·메시 | GR00T 저장소 LFS (`scripts/setup.sh`) | 원 저장소 조건을 따른다 |
| MediaPipe Pose Landmarker 모델(`pose_landmarker_{full,heavy}.task`) | Google 저장소 (`scripts/setup.sh`) | Apache License 2.0 |
| GEM-X 코드·가중치 (선택, GPU) | <https://github.com/NVlabs/GEM-X>, Hugging Face `nvidia/GEM-X` | 코드 Apache-2.0, 가중치 "Licensed by NVIDIA Corporation under the NVIDIA Open Model License." |
| SOMA-X (GEM-X가 씀) | Hugging Face `nvidia/SOMA-X` | Apache License 2.0 |
| ROBOTIS ai_sapiens (K1 MJCF·메시) | <https://github.com/ROBOTIS-GIT/ai_sapiens> | Apache License 2.0 |
| MuJoCo | PyPI | Apache License 2.0 |
| onnxruntime | PyPI | MIT |
| cppzmq | `linux/install_sonic_deploy.sh` | MIT |
| Isaac Lab | `linux/install_sonic_training.sh`, `sonic_k1_train/box/setup_env.sh` | BSD-3-Clause |
| Isaac Sim, TensorRT, CUDA | NVIDIA | 각 NVIDIA 라이선스(설치 때 동의 필요) |
| SMPLSim | GEAR-SONIC 학습 의존성 | BSD-3-Clause |

## 3. 학습 데이터 (가중치를 공개할 때)

이 저장소는 학습된 정책 가중치를 포함하지 않는다. 직접 학습한 가중치를 배포할 때는 학습에 쓴 데이터의 조건을 따른다.

- **BONES-SEED**(사람 전신 실험 `k1s_bones*`, 선택): <https://bones.studio/info/seed-license>. 원 데이터 재배포 금지, 비상업 조건 등.
  이 데이터로 학습한 모델에는 다음 문구를 붙인다: "Training data includes Motion Data by Bones Studio (https://bones.studio)."
- **SMPL-X 유도 파일**(`human_joints_info.pkl`, SMPL 인코더 입력): SMPL-X 모델 라이선스를 따른다.
- 보행 정책 데이터(`sonic_k1_train/data/`로 생성): GEAR-SONIC 플래너 출력(NVIDIA Open Model License), GR00T 예제 모션, ROBOTIS cyclo_lab
  댄스(Apache-2.0), 직접 만든 제스처·합성 동작. 각 출처의 조건을 확인한다.

## 4. 평가에 쓴 영상

평가에 쓴 사람 영상(Pexels 스톡 영상, YouTube Creative Commons 영상 등)은 저장소에 포함하지 않는다.
