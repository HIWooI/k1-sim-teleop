# k1/ — ROBOTIS AI Sapiens K1 MuJoCo 시뮬레이션 (CPU)

K1(23 DOF)을 MuJoCo 물리(200 Hz, 제어 50 Hz)와 onnxruntime CPU로 돌리는 패키지다. 실시간 서비스(`k1live/`)의 split·hybrid 백엔드가 이 시뮬레이션을
쓰고, 오프라인 영상 렌더·제스처도 여기에 있다.

## 출처

| 파일 | 원본 | 비고 |
|---|---|---|
| `sim.py` | 작성자가 이전에 만든 K1 sim2sim(split 구동) | PD·토크/속도 한계·armature·URDF 관절 한계·관측·분할 구동. 씬(배경·조명·오프스크린 렌더), 정책 순서 모터, 표시용 재질만 바꿈 |
| `cpu_kinematics.py` | 작성자의 이전 인식 코드 | CPU 위치 IK(사람 점 → K1 팔) |
| `assets/k1.urdf` | ROBOTIS [ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens) `ai_sapiens_description/urdf/k1_rev1` (Apache-2.0) | xacro 생성본 |
| `assets/k1_config.yaml` | ROBOTIS ai_sapiens `ai_sapiens_sim2real/config/k1_config.yaml` (Apache-2.0) | `robot_joint_order`만 |
| `assets/student_asset/expert_adapter.py` | 작성자의 이전 프로젝트(cyclo_lab 안에서 작성해 ROBOTIS Apache-2.0 머리말이 붙음) | 정책마다 다른 관절 순서·관측·action scale을 한 순서로 맞춘다 |
| `assets/*/params/sim2real.yaml` | 작성자가 cyclo_lab으로 학습한 K1 정책의 배포 파라미터 | export 때 생성된 파일 |
| `assets/*/exported/policy.onnx` | 작성자가 학습한 K1 정책(`student_asset` = mimic student, `loco_armdriven_r2` = 보행) | git에는 없음(`.gitignore`). Release `weights-v1`에서 `scripts/fetch_weights.sh`로 받는다 |
| (참조만) K1 MJCF·메시 | `$AI_SAPIENS_ROOT/ai_sapiens_description/mujoco/k1/k1.xml`, `meshes/k1_rev1` | 기본 `third_party/ai_sapiens` |

## 구동 방식

- **split(기본)**: 팔 10관절 = mimic student(참조 추종) 또는 PD(`arm_mode="pd"`), 다리 12 + 허리 = `loco_armdriven_r2`(속도 명령). mimic에게는 하체가
  참조를 완벽히 따라가는 것처럼 보여 주고, 게인은 그 관절을 지휘하는 정책의 값을 쓴다.
- **mimic(목례만)**: 숙임은 고관절 pitch가 필요한데 split에서는 다리가 loco 소유라 불가능 → `bow`만 전신 mimic(골반·몸통 0.30 rad 숙임).

## 실행

```bash
.venv/bin/python -m k1.render_event                        # 제스처 8종 → videos/ 아래 제스처별 mp4 + json
.venv/bin/python -m k1.run_video --video in.mp4 --name x   # 사람 영상 → 사람 | K1 나란히 (MediaPipe, 오프라인)
.venv/bin/python -m k1.run_video --preset                  # $K1_EVENT_VIDEOS(기본 videos/event)의 미리 정한 클립들
.venv/bin/python -m k1.make_showreel all --check-frames 8  # 렌더 결과를 이어 붙인 릴 (--title-top 으로 제목)
```

## 제스처 (`gestures.py`)

G1 키프레임을 관절 단위로 옮길 수 없어서(K1 팔 5 DOF, 허리 yaw 1 DOF) **손바닥·팔꿈치 목표점(몸통 좌표)을 최소저크 키프레임으로 움직이고 매 프레임
K1 FK로 최소자승 IK**를 푼다(관절 한계는 URDF). 순서: 손 흔들기 → 목례 → 두 팔 환영 → 박수 → 손바닥 안내 → 머리 위 하트 → 만세 → 편히 서기.
서비스의 대기 동작과 운영 패널의 제스처 버튼이 이것을 쓴다.

| 제스처 | 구동 | 낙상 | 골반 최대 기울기 | 팔 추종 오차 평균 / p90 |
|---|---|---|---|---|
| wave_hello | split | 0 | 3.0° | 4.88° / 8.65° |
| bow | 전신 mimic | 0 | 21.7° | 4.53° / 8.72° |
| welcome_open_arms | split | 0 | 2.9° | 7.98° / 12.82° |
| clap | split | 0 | 3.7° | 5.62° / 11.77° |
| point_guide | split | 0 | 2.7° | 5.79° / 11.29° |
| heart | split | 0 | 4.4° | 6.20° / 14.81° |
| raise_both_arms | split | 0 | 3.8° | 8.21° / 18.67° |
| stand_at_ease | split | 0 | 2.9° | 4.82° / 8.31° |

(mimic student 팔 경로 기준. 오차 = 로봇 관절 − 참조, 제스처 구간 전체 평균.)

## 사람 영상 → K1 (`run_video.py`)

MediaPipe Pose(full, One-Euro 필터) → 사람 몸통 좌표로 정규화 → CPU IK(`--method` 기본값 = 위치 IK, `dir` = 방향 IK) → 팔 10관절을 50 Hz로
리샘플·속도 제한 → split 구동. 화면은 1920×1080, 왼쪽 사람(인물 중심 크롭), 오른쪽 K1 정면. JSON에는 팔 추종 오차, 2D 팔 각도 오차(사람 2D 관절 vs 로봇
정면 투영), 팔 가속 RMS가 남는다. 이 모듈의 함수(`mp_targets`, `torso_frame`)는 실시간 sender와 같고 시험이 그것을 확인한다.

GEM-X 입력 모드(`--source`, `soma.py`, `soma_mlp.py`; GEM-X 결과 폴더 `$K1_SOMA_CLIPS/<clip>/`): `soma_ik`(SOMA 77관절 → 같은 CPU IK), `soma_csv`(soma-retargeter K1 CSV, `--drive mimic`이면 전신),
`soma_mlp`(증류 MLP, 가중치 미포함).

## 한계

1. split에서는 로봇이 몸을 돌리거나 숙이지 않는다(허리 yaw·다리는 보행 정책 소유). 목례만 전신 mimic.
2. 손목 비틀기·손가락은 영상에서 오지 않는다(MediaPipe에 전완 회전 정보 없음).
3. 물리는 MuJoCo이며 센서 잡음·통신 지연이 없다. 실기 K1과 다를 수 있다.
