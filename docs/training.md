# K1 SONIC 학습 — GEAR-SONIC 방식, GPU 1장

NVIDIA GEAR-SONIC(GR00T-WholeBodyControl의 `gear_sonic`)의 학습 코드를 그대로 쓰고, ROBOTIS K1을 새 embodiment로 더해 GPU 1장·소규모 데이터로
학습했다. 목표는 GEAR-SONIC 규모(수만 GPU시간, 수백 시간 모캡)의 재현이 아니라, **같은 학습 방식으로 K1이 서고 제스처를 하고 플래너 걸음을
따라 걷는 정책**을 얻는 것이었다. 결과로 나온 보행 정책(walk4 7000)이 행사 설정 hybrid 백엔드의 걷기를 맡는다.

행사 설정 walk4 7000은 GitHub Release로 배포한다([weights.md](weights.md)). 다른 run의 가중치는 배포하지 않는다(학습 데이터 조건 — §6).

## 1. K1 embodiment (`sonic_k1/`)

upstream `docs/source/user_guide/new_embodiments.md`(H2 예시)를 따른다. upstream 저장소를 직접 고치지 않고 `overlay/`를 덮어쓰고 `patches/`를 적용한다.

| 파일 | 내용 |
|---|---|
| `overlay/gear_sonic/envs/manager_env/robots/k1.py` | Isaac Lab 로봇 설정: 관절·바디 순서, 액추에이터(ROBOTIS cyclo_lab K1_rev1 값), 기본자세, action scale |
| `overlay/.../robot_description/urdf/k1/k1.urdf`, `mjcf/k1.xml` | `scripts/stage_assets.py`가 ROBOTIS ai_sapiens에서 만든 물리용 URDF와 모션 라이브러리용 MJCF(머리 링크 병합, 메시 m 단위) |
| `overlay/.../config/exp/manager/universal_token/all_modes/sonic_k1.yaml` | `sonic_release`를 K1 바디 이름으로 옮긴 실험 설정 |
| `patches/0001-register-k1-robot.patch` | 로봇 등록, `robot_mapping["k1"]`, 관절 순서 변환기 |
| `patches/0002-check-robot-order.patch` | 시작할 때 Isaac 바디 순서와 `k1.py`를 비교(`SONIC_STRICT_ROBOT_ORDER=1`이면 중단) |
| `scripts/apply.sh` | `apply.sh [--check] <GR00T-WholeBodyControl>`: overlay 복사 + patch 적용(멱등) |
| `tests/` | 관절 순서·FK·게인, Mac 배포 루프 |

게인은 GEAR-SONIC 규칙(10 Hz, ζ = 2), action scale = 0.25 · effort / kp로 ROBOTIS K1 student 자산의 값과 0.2 % 안에서 같다. 기본자세는 K1 mimic의
기본자세(서 있는 골반 높이 0.765 m).

## 2. 학습 서버

GPU 서버(실제 사용: RTX PRO 6000 96 GB 1장, Isaac Sim 5.1, Isaac Lab v2.3.2)의 작업 폴더는 `SONIC_K1_WORK`(기본 `$HOME/sonic_k1`)이다.
학습 스크립트는 그 폴더 밖을 건드리지 않는다.

| 스크립트 | 어디서 | 내용 |
|---|---|---|
| `sonic_k1_train/box/setup_env.sh` | 서버 | 학습 환경을 만든 절차 그대로: 이미 있는 Isaac Sim 5.1 conda env(`BASE_ENV`)를 `$SONIC_K1_WORK/env`로 복사 클론, Isaac Lab v2.3.2 체크아웃, `gear_sonic[training]`와 Isaac Sim이 요구하는 고정 버전(numpy 1.26, protobuf 4.25 등), open3d용 libusb |
| `sonic_k1_train/box/sync_to_box.sh <모션 폴더> [이름]` | 로컬 | `sonic_k1/`(스테이징 메시 포함), `sonic_k1_train/`, 만든 모션을 서버로 rsync. 접속 이름은 `SONIC_K1_HOST`(필수) |
| `sonic_k1_train/box/prepare.sh` | 서버 | 서버의 GEAR-SONIC 체크아웃에 `sonic_k1` overlay·patch와 `sonic_k1_train` overlay·평가 patch를 적용 |
| `sonic_k1_train/box/train.sh` | 서버 | `smoke [NAME [EXP]]`(16 env, 30 iter) · `train NAME EXP NUM_ENVS ITERS [hydra override...]`(tmux 분리) · `export CKPT`(ONNX) · `eval CKPT MOTION_DIR`(im_eval). GPU에 다른 compute 프로세스가 있으면 시작하지 않는다 |
| `sonic_k1_train/box/curves.py` | 서버 | 학습 로그 → 곡선 CSV·요약 |

GEAR-SONIC 코드 사본은 upstream `gear_sonic/`을 서버로 rsync하고 `git init`한 것이다(patch를 `git apply`하려고). Isaac Sim 첫 실행은 NVIDIA EULA
동의가 필요하다(동의하는 경우에만 `OMNI_KIT_ACCEPT_EULA=YES`). 서버에서 부딪힌 것들(예: open3d의 libusb, 의존성 버전 충돌, im_eval의 G1 손목
이름)은 `setup_env.sh`와 `sonic_k1_train/patches/`에 반영되어 있다.

## 3. 데이터

| 생성기 | 내용 |
|---|---|
| `sonic_k1_train/data/build_dataset.py --out <dir> --mjcf <스테이징 k1.xml>` | 첫 라이브러리 170클립 27.5분: 제스처 8종(`k1/gestures.py`) × 다리 자세 2 × 속도 3, 합성 서 있기·체중 이동 68클립(다리 IK로 발 고정), GEAR-SONIC 예제 G1 모션 → K1 12클립, ROBOTIS cyclo_lab K1 댄스 4클립, GEM-X 리타깃 댄스 2클립. 보류 14클립(하트·스쿼트) |
| `sonic_k1_train/data/planner_dataset.py --out <dir> --minutes 75 --workers 8` | 플래너 보행 라이브러리: 무작위 명령 스크립트(8–20 s)를 라이브와 같은 명령 경로(`k1live.planner.KinematicPlanner`)에 넣고 G1 출력을 K1로 바꿔 발 미끄럼을 정리, 미러. 실제 사용: 273클립 × 미러 = 128분(정지·전진·후진·옆·대각·회전·걸으며 회전, 0.2–1.0 m/s), 절반에 팔 덧입힘(제스처·사람 팔·무작위 팔) |
| `sonic_k1_train/data/planner_arms.py`, `planner_arms_live.py` | 걷기에 덧입힐 팔: 제스처·사람 영상 팔(오프라인 IK), 라이브 `HumanMapper`와 같은 체인으로 만든 사람 팔 |
| `sonic_k1_train/box/make_walk2_motions.sh` | 보행 세트 + 제스처 84클립 3배(적응 샘플링에서 제스처가 덜 뽑히던 문제) |

모든 클립은 K1 발 접촉구로 매 프레임 접지하고, 미러(`_M`)는 좌우 관절 교환 + roll·yaw 부호 반전으로 만든다. 학습 세트(walk 계열):
170클립 + 플래너 546 PKL = 716 PKL, 약 2.6시간(보류 92 PKL).

## 4. 학습 계보

설정은 `sonic_k1_train/overlay/gear_sonic/config/exp/manager/universal_token/all_modes/`. 구조는 GEAR-SONIC과 같고(universal token: 인코더 g1·teleop·smpl →
FSQ 토큰 → 행동 디코더 + 재구성 보조 디코더, PPO, 적응 샘플링, 도메인 랜덤화, heading 정규화) MLP만 줄였다(ONNX 인코더 6.7 MB / 디코더 7.2 MB).
4096 env, 200 Hz 물리 / 50 Hz 정책.

| run | 설정 | 시작점 | 바꾼 것 | 결과 |
|---|---|---|---|---|
| `k1s_main` | `sonic_k1_small` | 처음부터 | 첫 라이브러리(170클립), 미래 10 × 0.1 s | 3,508 iter / 1 h 54 m: 학습 세트 성공 91.8 %, MPJPE local 18.3 mm. MuJoCo 낙상 0, 제스처 팔 2.4–3.8° |
| `k1s_la4` | `sonic_k1_small_la4` | 처음부터 | 미래 4 × 20 ms = 80 ms (라이브 지연용) | 2,010 iter / 1 h 7 m: 성공 90.6 %. 걷지는 못함 |
| `k1s_walk` | `sonic_k1_small_la4_walk` | la4 2000 warm start | 플래너 보행 세트 | 5500 |
| `k1s_walk2` | `..._walk2` | walk 5000 | 골반 속도 보상, xy 이탈 종료, 제스처 ×3 | |
| `k1s_walk3` | `..._walk3` | walk2 2000 resume | 팔 링크 방향 보상 | 5000: 보행 10종 낙상 0, 최소 속도비 0.80, 제스처 3.56° |
| **`k1s_walk4`** | `..._walk4` | walk3 5000 resume | 두 발목 상대 위치 보상(σ 0.06 m) — 발을 끄는 걸음 대책 | **7000 = 행사 설정**: 낙상 0/10, 속도비 앞 0.83 · 뒤 0.75 · 옆 1.10 · 회전 1.00, 제스처 3.90°, 발 들림·스탠스·미끄럼 최선 |
| `k1s_walk5` | `..._walk5` | walk4 7000 resume | 발 보상 ×3 | 효과가 작아 채택하지 않음 |
| `k1s_bones`, `k1s_bones2` | `..._bones`, `..._bones2` | walk3 6000(SMPL 인코더 제외) | 사람 SMPL 인코더 + BONES-SEED | §5 |

수락 기준(보행): 보행 10종 낙상 0, 제스처 팔 오차 ≤ 4°, 앞·뒤·옆·회전 속도비 ≥ 0.6. walk4 체크포인트 선택은 hybrid 안에서 걸음 품질로 비교했다
([results.md](results.md) §3).

실행 예(서버, `cd $SONIC_K1_WORK`):

```bash
bash pkg/sonic_k1_train/box/prepare.sh
bash pkg/sonic_k1_train/box/train.sh smoke
SONIC_K1_MOTIONS=$PWD/pkg/k1_walk2_motions bash pkg/sonic_k1_train/box/train.sh train k1s_walk4 sonic_k1_small_la4_walk4 4096 8000 \
  ++checkpoint=<walk3 run 폴더 사본>/model_step_005000.pt +resume=true
bash pkg/sonic_k1_train/box/train.sh export runs/TRL_K1_Track/<run>/model_step_007000.pt
```

resume한 run은 체크포인트 옆에 결과를 쓰므로, 체크포인트를 새 폴더에 복사한 뒤 그 사본에서 시작한다. warm start(`++checkpoint=`만)는 가중치·std·critic만
이어받고 옵티마이저·적응 샘플링은 새로 시작한다(모션 세트가 바뀔 때).

### export → 라이브용 deploy 폴더 → 평가

```bash
# export된 ONNX → deploy 폴더(model_encoder/decoder.onnx + observation_config.yaml)
.venv/bin/python sonic_k1_train/mac/eval_k1_mujoco.py --export models/k1/<run>/exported --step 7000 --future 4x1 \
    --variant-dir models/k1/<run>/deploy/<run>_007000 --stand 20
# 보행 10종(라이브와 같은 플래너 명령 경로) + 걸음 품질
.venv/bin/python sonic_k1_train/mac/eval_walk.py --backend sonic --variant-dir models/k1/<run>/deploy/<run>_007000 --json w.json
.venv/bin/python sonic_k1_train/mac/eval_walk.py --backend split --json split.json
.venv/bin/python sonic_k1_train/mac/eval_walk_table.py w.json split.json
# 라이브 hybrid에 쓰기
K1LIVE_HYBRID_VARIANT=models/k1/<run>/deploy/<run>_007000 ./K1Demo.command     # 또는 event.yaml hybrid_variant
```

다른 평가 도구: `eval_walk_clips.py`(데이터셋 클립), `eval_walk_human.py`(걷는 사람 영상을 라이브 체인으로), `eval_walk_video.py`(체크포인트 비교 영상),
`baseline_student.py`(ROBOTIS student 비교), `scripts/eval_hybrid.py`(hybrid 걸음·서기·전환), `scripts/eval_hybrid_live.py`(GEM-X 덤프 라이브 재생).

## 5. 사람 전신 정책 실험 (BONES-SEED, 수락 기준 미달)

hybrid는 "서 있을 때 팔, 걸을 때 플래너 걸음"을 나눠 맡긴다. 이와 달리 GEAR-SONIC의 SMPL 인코더처럼 **사람 전신 자세를 직접** 받아 다리·발걸음까지
따라 하는 정책도 시도했다(`sonic_human` 백엔드).

| 단계 | 도구 |
|---|---|
| 받기 | `sonic_k1_train/box/bones_fetch.py`(Hugging Face `bones-studio/seed`는 라이선스 동의가 필요한 gated 저장소. 토큰은 로컬에만 두고 서버에는 서명 URL만 전달) |
| 고르기·변환 | `sonic_k1_train/box/bones_pipeline.sh` → `data/bones_select.py`(서기·걷기·회전·옆/뒷걸음·제스처), `box/bones_extract.py`, `data/bones_to_k1.py`(G1 리타깃 → K1), `data/bones_human_part.py`·`data/human_smpl.py`(사람 SOMA → MediaPipe식 33점 → SMPL-24, 라이브와 같은 변환) |
| 학습 세트 | `box/make_bones_train.sh`(보행 세트 + 사람 세트 4,743클립 8.8시간 + 라이브 팔 세트), `box/drop_smpl_encoder.py`(warm start용) |
| `k1s_bones` | walk3 6000에서 20000 iter |
| `k1s_bones2` | `box/bones2.sh`: bones 20000 resume, walk4의 발 보상 + **GEM-X식 SMPL 입력 증강**(`sonic_k1_train/overlay/.../mdp/smpl_gemx_aug.py`, 파라미터 `sonic_k1_train/data/gemx_aug/`, 측정·맞춤 `scripts/gemx_aug_stats.py`) |

판정(GEM-X 입력, [results.md](results.md) §8): bones2 25000에서 GEM-X 시험 낙상은 모두 0이 됐지만, **팔 오차 13.6°(split 2.05°)로 7–14배**,
카메라 쪽으로 걸어와 웅크리는 구간은 기울기 44.6°(낙상 규칙 45°)로 경계라 행사 설정에 넣지 않았다. 운영자가 지켜보는 서서 하는 시연에서만 쓸 수 있다
(`K1LIVE_HUMAN_VARIANT`).

## 6. 데이터·가중치 라이선스

- **BONES-SEED**: 원 데이터 재배포 금지, 사용 조건(비상업 등)은 <https://bones.studio/info/seed-license>. 이 데이터로 학습한 모델에는
  "Training data includes Motion Data by Bones Studio (https://bones.studio)." 문구가 필요하다.
- **`human_joints_info.pkl`**(SMPL 인코더 입력의 휴지 골격, SMPL-X 유도): SMPL-X 라이선스(비상업, 재배포 금지). 저장소와 공개 번들에 넣지 않는다.
- 보행 데이터: GEAR-SONIC 플래너 출력(NVIDIA Open Model License), GR00T 예제 모션, ROBOTIS cyclo_lab 댄스(Apache-2.0), 제3자 댄스 영상에서 리타깃한
  클립 2개, 직접 만든 제스처·합성 동작.

walk4 7000은 BONES-SEED와 SMPL-X 파일 없이 학습했고, 위 보행 데이터만으로 학습해 Release로 배포한다([weights.md](weights.md)). BONES로 학습한
`k1s_bones*` 가중치는 배포하지 않는다. 직접 학습한 가중치를 배포하려면 위 조건을 확인한다([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
