# sonic_k1 — GEAR-SONIC 새 embodiment: ROBOTIS K1 (AI Sapiens k1_rev1, 23 DOF)

upstream GR00T-WholeBodyControl `docs/source/user_guide/new_embodiments.md`(H2 예시)를 따라 GEAR-SONIC 학습 코드에 K1을 더하는 패키지다.
**upstream 저장소를 직접 고치지 않는다.** `overlay/`를 덮어쓰고 `patches/`를 적용한다(`scripts/apply.sh`). 실제 학습 절차는
[`docs/training.md`](../docs/training.md), G1 → K1 관절 매핑은 [`docs/g1_to_k1_mapping.md`](../docs/g1_to_k1_mapping.md).

## 레이아웃

```
overlay/gear_sonic/envs/manager_env/robots/k1.py                  Isaac Lab 로봇 설정 (관절·바디 순서, 액추에이터, 기본자세, action scale)
overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml       모션 라이브러리용 MJCF (stage_assets.py 생성)
overlay/gear_sonic/data/assets/robot_description/urdf/k1/k1.urdf   Isaac Lab 물리용 URDF (stage_assets.py 생성)
overlay/gear_sonic/data/assets/robot_description/urdf/k1/meshes/   STL mm→m 변환본 — git 제외, stage_assets.py로 생성
overlay/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_k1.yaml   실험 설정
patches/0001-register-k1-robot.patch   robots/__init__.py, modular_tracking_env_cfg.py robot_mapping["k1"], order_converter.K1Converter
patches/0002-check-robot-order.patch   시작 시 Isaac 바디 순서와 k1.py 비교 (SONIC_STRICT_ROBOT_ORDER=1이면 중단)
data/robot/*.pkl                       K1 모션 라이브러리 (scripts/k1_csv_to_pkl.py, git 제외)
train.env                              linux/train_sonic_k1.sh 용 변수
scripts/  stage_assets.py · apply.sh · k1_csv_to_pkl.py · replay_pkl.py · g1_to_k1.py · dummy_onnx.py · k1_assets.py
tests/    test_k1_robot.py (순서·FK·게인) · test_k1_deploy.py (Mac 배포 루프)
```

`overlay/`와 `patches/`는 NVIDIA GR00T-WholeBodyControl(Apache-2.0) 파일을 고친 것이고, URDF·MJCF는 ROBOTIS ai_sapiens(Apache-2.0)에서 만든다
([NOTICE](../NOTICE)).

## 사용

```bash
git clone https://github.com/ROBOTIS-GIT/ai_sapiens third_party/ai_sapiens         # 또는 AI_SAPIENS_ROOT=<경로>
.venv/bin/python sonic_k1/scripts/stage_assets.py                                 # URDF·MJCF·메시 스테이징
.venv/bin/python -m pytest -q sonic_k1/tests
sonic_k1/scripts/apply.sh --check <GR00T-WholeBodyControl>                         # 그다음 --check 없이
```

`stage_assets.py`가 ai_sapiens MJCF를 그대로 쓰지 않는 이유: GEAR-SONIC의 모션 라이브러리(`Humanoid_Batch`)는 모든 `<body>`를 골격 노드로 보는데
Isaac Lab URDF 임포터는 고정된 head 관절을 병합하므로 head_link를 torso_link에 병합해 24 바디로 맞춘다. 또 `<freejoint>`가 아니라
`<joint type="free">`여야 하고, 메시는 `scale` 속성 대신 미터 단위로 변환해야 한다.

## 값의 출처

- **관절 순서**: MuJoCo·하드웨어 순서는 `ai_sapiens_sim2real/config/k1_config.yaml`과 같다. Isaac Lab 순서는 URDF를 BFS로 돌고 고정 관절(head)을 병합한
  결과이며, ROBOTIS cyclo_lab K1 환경의 PhysX 순서와 같다. 같은 규칙이 GEAR-SONIC의 `G1_ISAACLAB_JOINTS`도 재현한다(test).
- **액추에이터**: ROBOTIS cyclo_lab `K1_rev1.py` 값. QC080 96.864 Nm · 11.519 rad/s · armature 0.01936542, QC060 47.277 Nm · 20.944 rad/s ·
  armature 0.00564892, 발목은 링크 구동이라 2배. 게인은 GEAR-SONIC 규칙(10 Hz, ζ = 2): kp 76.5/153/44.6/22.3, kd 4.87/9.73/2.84/1.42.
  action scale = 0.25 · effort / kp(K1 student 자산의 `sim2real.yaml`과 같음, test).
- **기본자세**: K1 mimic의 `default_position`(hip −0.3, knee 0.63, ankle −0.33, 어깨 0.2/±0.2, 팔꿈치 0.6). 서 있는 골반 높이 0.765 m, 초기 높이 0.77.

## 검증

| 검사 | 결과 |
|---|---|
| Isaac 순서 규칙 → GEAR-SONIC G1 목록 / cyclo_lab K1 PhysX 순서 | 둘 다 일치 |
| `k1.py` 인덱스 맵 = URDF·MJCF에서 계산한 값, 역순열 | 일치 |
| MJCF FK = URDF FK (무작위 자세 20개, 24 바디) | 1e-6 이내 |
| 게인·action scale·기본자세 vs student `sim2real.yaml` | 0.2 % 이내 |
| `apply.sh` (적용 → 다시 실행) | 깨끗이 적용, 두 번째는 건너뜀 |
| Mac 배포 루프 (`sonic/robots.py`) | G1은 리팩터 전후 비트 동일. K1은 더미 정책으로 INIT → CONTROL → SMPL 스트림 종단 실행 |
| Isaac Lab 학습 | `SONIC_STRICT_ROBOT_ORDER=1`로 스모크·본 학습 모두 순서 검사 통과 |

## 알려진 한계

- SMPL 데이터는 기본 `dummy`이고, VR 3점 오프셋은 추정치다.
- URDF와 MJCF의 머리 고정 위치가 다르다(URDF −0.0185, 0, 0.385 vs MJCF −0.02725, 0, 0.361). URDF 값을 쓴다.
