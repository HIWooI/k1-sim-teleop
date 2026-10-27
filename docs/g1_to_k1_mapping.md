# SONIC G1 출력(29 DOF) → ROBOTIS K1(23 DOF) 관절 매핑 — B안 사전 조사

B안은 SONIC(G1 정책)이 만든 G1 관절 궤적을 K1 관절 레퍼런스로 바꿔 K1 mimic 정책(`k1/assets/student_asset`, 또는
split 구동: 팔=레퍼런스, 다리+허리=`loco_armdriven_r2`)에 넣는 구성이다. 이 문서는 그 가운데 **관절 매핑** 부분을 다룬다.
코드는 `sonic_k1/scripts/g1_to_k1.py`(`G1ToK1.copy`, `G1ToK1.orient`)에 있고, 수치는 `--eval`로 다시 낼 수 있다.
두 로봇의 MJCF는 G1이 `vendor/.../g1/scene_43dof.xml`, K1이 `sonic_k1`에서 staged한 `mjcf/k1.xml`이다.

## 1. 구조 차이 (MuJoCo로 확인)

- **같은 이름, 같은 순서의 사슬.** 다리는 hip pitch(Y) → roll(X) → yaw(Z) → knee(Y) → ankle pitch(Y) → ankle roll(X)이다.
  팔은 shoulder pitch(Y) → roll(X) → yaw(Z) → elbow(Y) → wrist roll(X)이다. 영점 자세의 전완은 두 로봇 모두 +X 방향(팔꿈치 90°)이다.
- **K1에 없는 축 6개.** 허리 roll·pitch(`waist_roll_joint`, `waist_pitch_joint`)와 양손목 pitch·yaw(`*_wrist_pitch_joint`,
  `*_wrist_yaw_joint`)이다. 허리 3축이 1축으로, 손목 3축이 1축으로 줄어든다.
- **G1 축 기울기.** 같은 각도를 그대로 복사하면 아래 기울기만큼 틀어진다. K1의 관절 축은 모두 순수한 X·Y·Z다.
  - G1의 `*_hip_roll_link`는 Y축으로 −10° 기울어져 있고 `*_knee_link`에서 +10°로 되돌린다. 그래서 hip roll 축은 (0.985, 0, 0.174), hip yaw 축은 (−0.174, 0, 0.985)이다.
  - G1의 `*_shoulder_pitch_link`는 X축으로 ±16° 기울어져 있고 `*_shoulder_roll_link`에서 되돌린다. 그래서 shoulder pitch 축은 (0, 0.961, ±0.276)이다.
  - 두 경우 모두 기울기가 다음 링크에서 상쇄되므로, **영점에서는 두 로봇의 링크 좌표계가 일치한다.** 3장의 orient 매핑은 이 성질을 이용한다.
- **링크 길이.** 위팔은 두 로봇 모두 0.185 m다. 나머지는 다르다.

  | 링크 | G1 | K1 |
  |---|---|---|
  | 엉덩이→무릎 | 0.341 m | 0.367 m |
  | 정강이 | 0.318 m | 0.294 m |
  | 팔꿈치→손목 | 0.184 m | 0.069 m |
  | 손 | 짧음 | 길다(손목 roll 링크 기준 0.03–0.28 m) |

  관절각을 옮기면 손발의 **위치**는 같지 않다. 행사 제스처에서 중요한 것은 방향이다.

## 2. 관절별 매핑표 (MuJoCo/하드웨어 순서)

| G1 (idx) | K1 (idx) | 방법 | G1 범위 [rad] | K1 범위 [rad] | 비고 |
|---|---|---|---|---|---|
| left_hip_pitch (0) | left_hip_pitch (0) | orient (hip 3축 동시) | −2.531 … 2.880 | −2.444 … 2.444 | |
| left_hip_roll (1) | left_hip_roll (1) | orient | −0.524 … 2.967 | −0.611 … 2.531 | G1 축 10° 기울기 |
| left_hip_yaw (2) | left_hip_yaw (2) | orient | −2.758 … 2.758 | −3.142 … 3.142 | G1 축 10° 기울기 |
| left_knee (3) | left_knee (3) | 복사 | −0.087 … 2.880 | **0 … 2.356** | 깊은 스쿼트는 클립된다 |
| left_ankle_pitch (4) | left_ankle_pitch (4) | 복사 | −0.873 … 0.524 | **−0.785** … 0.785 | 런지·스쿼트는 하한에 닿는다 |
| left_ankle_roll (5) | left_ankle_roll (5) | 복사 | −0.262 … 0.262 | −0.524 … 0.524 | |
| right_* 다리 (6–11) | right_* (6–11) | 위와 같음 | 대칭 | 대칭 | |
| waist_yaw (12) | waist_yaw (12) | 복사 | −2.618 … 2.618 | −6.283 … 6.283 | |
| waist_roll (13) | — | **버림.** orient는 어깨로 흡수 | −0.52 … 0.52 | — | 몸통 옆기울기 손실 |
| waist_pitch (14) | — | **버림.** orient는 어깨로 흡수 | −0.52 … 0.52 | — | 몸통 숙임 손실. 목례는 hip pitch가 대신한다 |
| left_shoulder_pitch (15) | left_shoulder_pitch (13) | orient (어깨 3축 동시) | −3.089 … 2.670 | −3.142 … 3.142 | G1 축 16° 기울기 |
| left_shoulder_roll (16) | left_shoulder_roll (14) | orient | −1.588 … 2.252 | **0 … 3.142** | K1 팔은 몸 안쪽(정중선 쪽)으로 못 넘긴다 |
| left_shoulder_yaw (17) | left_shoulder_yaw (15) | orient | −2.618 … 2.618 | −3.142 … 3.142 | |
| left_elbow (18) | left_elbow (16) | 복사 | −1.047 … 2.094 | −0.785 … 2.182 | |
| left_wrist_roll (19) | left_wrist_roll (17) | 복사 | −1.972 … 1.972 | −3.142 … 3.142 | |
| left_wrist_pitch (20) | — | **버림** | −1.614 … 1.614 | — | 손 방향 손실 |
| left_wrist_yaw (21) | — | **버림** | −1.614 … 1.614 | — | 손 방향 손실 |
| right_* 팔 (22–28) | right_* (18–22) | 위와 같음 | 대칭 | 대칭(roll은 **−3.142 … 0**) | |

## 3. 매핑 방법

- **copy.** 같은 이름의 각도를 그대로 복사하고, K1에 없는 6축은 버리고, K1 한계로 클립한다.
- **orient (권장).**
  - 어깨: G1 FK로 얻은 `*_shoulder_yaw_link`의 자세를 K1 몸통 기준(골반·Rz(waist_yaw))으로 표현한다.
    이것을 K1의 pitch·roll·yaw로 YXZ 오일러 분해한다.
    G1 몸통의 roll·pitch는 이 변환에 그대로 들어가므로, **팔은 골반 기준으로 같은 방향을 유지한다.**
  - 엉덩이: G1 `*_knee_link`의 자세에서 무릎각을 뺀 뒤 같은 방식으로 분해한다.
  - 나머지 관절은 복사하고, 마지막에 K1 한계로 클립한다.

## 4. 정량 비교 (`g1_to_k1.py --eval`, 5프레임 간격)

오차는 영점에서 일치하는 링크 좌표계를 골반 기준으로 비교한 각도 차이(°)다. 비교한 링크는 위팔=shoulder_yaw_link,
전완=elbow_link, 허벅지=knee_link, 발=ankle_roll_link, 손=wrist_yaw_link 대 wrist_roll_rubber_hand, 몸통=torso_link다.
행사 제스처 8종은 `sonic/gestures.py`(G1 키프레임)에서, ref는 SONIC 예제 참조 모션(`_M` 미러본은 같은 값이라 생략)에서 왔다.

| clip | copy: 팔 평균 / 최악 | copy: 다리 | orient: 팔 평균 / 최악 | orient: 다리 | 손 (손목 pitch/yaw 손실) | 몸통 (허리 roll/pitch 손실) | K1 한계 도달 (orient, 프레임 %) |
|---|---|---|---|---|---|---|---|
| event:wave_hello | 5.0 / 8.7 | 0.3 | 0.0 / 0.0 | 0.0 | 0.0 | 3.4 | - |
| event:bow | 9.4 / 12.0 | 0.3 | 0.0 / 0.0 | 0.0 | 0.0 | 9.0 | - |
| event:welcome_open_arms | 6.2 / 9.7 | 0.3 | 0.0 / 0.0 | 0.0 | 0.0 | 3.4 | - |
| event:clap | 6.6 / 11.0 | 0.3 | 0.3 / 3.0 | 0.0 | 0.3 | 3.4 | left_shoulder_roll 14%, right_shoulder_roll 14% |
| event:point_guide | 6.7 / 16.8 | 0.3 | 0.0 / 0.0 | 0.0 | 0.0 | 3.4 | - |
| event:raise_both_arms | 15.9 / 30.9 | 0.3 | 5.2 / 14.9 | 0.0 | 5.2 | 3.4 | left_shoulder_roll 43%, right_shoulder_roll 43% |
| event:heart | 18.0 / 31.9 | 0.3 | 0.0 / 0.0 | 0.0 | 0.0 | 3.4 | - |
| event:weight_shift | 3.6 / 3.6 | 0.4 | 0.0 / 0.0 | 0.0 | 0.0 | 3.4 | - |
| ref:dance_in_da_party_001__A464 | 18.3 / 54.4 | 3.2 | 3.3 / 44.2 | 0.0 | 16.9 | 13.0 | right_shoulder_roll 24%, left_shoulder_roll 4%, right_ankle_pitch 1% |
| ref:macarena_001__A545 | 20.1 / 53.4 | 3.0 | 1.4 / 41.1 | 0.0 | 21.4 | 14.6 | right_shoulder_roll 11%, left_shoulder_roll 9%, left_elbow 4% |
| ref:neutral_kick_R_001__A543 | 11.2 / 39.9 | 2.2 | 0.6 / 24.1 | 0.0 | 13.7 | 9.3 | left_elbow 6%, right_elbow 3%, left_shoulder_roll 3% |
| ref:squat_001__A359 | 25.9 / 43.7 | 1.9 | 0.0 / 0.0 | 0.9 | 6.4 | 18.9 | left_ankle_pitch 18%, left_knee 15%, right_knee 15% |
| ref:tired_one_leg_jumping_R_001__A359 | 23.0 / 47.6 | 2.2 | 0.3 / 15.0 | 0.3 | 21.0 | 20.9 | left_knee 12%, left_shoulder_roll 4%, right_elbow 2% |
| ref:walking_quip_360_R_002__A428 | 14.2 / 24.0 | 2.3 | 0.0 / 0.0 | 0.0 | 9.7 | 13.4 | - |

읽는 법은 다음과 같다.

- **행사 제스처는 orient로 거의 완벽하다.**
  - 대부분 팔·다리 오차가 0°다.
  - 예외는 박수와 만세다. 박수는 최악 3°다. 만세는 평균 5°, 최악 15°다.
  - 예외 두 개 모두 **K1 어깨 roll 한계(왼팔 ≥ 0, 오른팔 ≤ 0)** 때문이다. 두 팔을 몸 안쪽으로 모으면 막힌다.
  - copy를 쓰면 하트 18°, 만세 16°, 목례 9°가 틀어진다. G1 어깨·엉덩이 축의 기울기 때문이다.
- **잃는 것은 몸통과 손이다.**
  - 몸통: 제스처 중립자세의 허리 pitch 0.06 rad가 사라져 3.4°, 목례는 9°다. 목례의 숙임은 hip pitch와 루트로 표현해야 한다.
  - 손: 제스처 키프레임은 손목 pitch·yaw를 쓰지 않으므로 손실이 0이다. 춤·런지 ref에서는 손 방향이 6–24° 손실된다.
- **다리.** orient로 0–1°다. 스쿼트·런지는 K1 무릎·발목 한계에 걸린다. 행사의 서 있는 동작과는 무관하다.

## 5. B안 브리지 스케치 (구현하지 않음 — 실제 데모는 K1 전용 학습과 hybrid 백엔드로 갔다)

1. SONIC G1 루프(`scripts/sim_teleop.py`, 웹캠 → SMPL → G1)를 그대로 돌린다.
   제어 틱마다 G1 `q`(29)와 골반 자세를 꺼낸다. 물리 시뮬에서 실제로 실현된 G1 동작이다.
2. `G1ToK1.orient`로 K1 `q_ref`(23)를 만든다. `dq_ref`는 유한차분에 저역통과를 건다.
   K1에는 몸통 기울기가 없으므로, anchor(K1 mimic의 `motion_anchor_ori_b`, torso_link)는 G1 골반 자세·Rz(waist_yaw)로 둔다.
3. K1 mimic student에 넣는다(`k1/assets/student_asset/expert_adapter.py` 관측 조립).
   - 관측은 `motion_command` 46 = q 23 + dq 23, anchor 6, 각속도 3, 관절 위치 23, 관절 속도 23, 이전 행동 23이다.
   - 또는 split 구동으로 팔만 레퍼런스로 두고 다리+허리는 `loco_armdriven_r2`가 균형을 잡게 한다.
   - split 구동이 팔 외란에 훨씬 강하다(구형파·전범위 60 s 통과).
4. 위험 요소.
   - G1 정책의 지연(180 ms)에 K1 정책의 지연이 더해진다.
   - G1이 균형을 잡으려 걸음을 떼면, 그 다리 동작이 K1에 그대로 전해진다. 다리를 고정하거나 split 구동을 쓴다.
   - K1 어깨 roll 한계가 있다. 박수·하트 키프레임은 K1 기준으로 다시 확인해야 한다.
