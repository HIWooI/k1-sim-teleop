# sonic_k1_train — GEAR-SONIC 방식 K1 학습 (GPU 1장, 소규모)

K1 embodiment 패키지(`sonic_k1/`) 위에서 작은 데이터·작은 모델·GPU 1장으로 K1 SONIC 정책을 학습하는 데 필요한 것을 둔다. 과정·수치·결론은
[`docs/training.md`](../docs/training.md).

```
data/k1_kin.py                 K1 운동학(MuJoCo FK): 접지, x-z 평면 미러, 다리 IK, motion_lib PKL 엔트리
data/build_dataset.py          첫 모션 라이브러리 (제스처 / 합성 서 있기 / G1→K1 / cyclo_lab 댄스 / 리타깃 댄스)
data/planner_dataset.py        GEAR-SONIC 플래너 보행 라이브러리 (라이브와 같은 명령 경로, 발 미끄럼 정리, 미러)
data/planner_arms.py           걷기에 덧입힐 팔 (제스처·사람 영상 팔·무작위 팔)
data/planner_arms_live.py      라이브 HumanMapper와 같은 체인으로 만든 사람 팔
data/planner_preview.py        플래너 클립 미리보기 시트
data/human_smpl.py             사람 모캡 → SMPL-24 (라이브와 같은 MediaPipe-33 → mp2smpl 변환)
data/bones_select.py · bones_to_k1.py · bones_human_part.py · bones_preview.py    BONES-SEED → K1 + 사람 SMPL 쌍 (실험)
data/smpl_aug_stats.py         GEM-X 스트림 SMPL 오차 통계 (증강 맞춤)
data/gemx_aug/                 GEM-X식 SMPL 입력 증강 파라미터 (k1s_bones2)
overlay/gear_sonic/...         실험 설정(sonic_k1_small*, _la4, _walk*, _bones*), mdp/smpl_gemx_aug.py
patches/                       서버 GEAR-SONIC 체크아웃용 평가 수정 (im_eval의 G1 손목 이름, 실패 클립 목록) — prepare.sh가 적용
box/setup_env.sh               학습 서버 환경 (Isaac Sim 5.1 env 복사 클론, Isaac Lab v2.3.2, 의존성 고정)
box/sync_to_box.sh             로컬 → 서버: sonic_k1/(+메시), sonic_k1_train/, 만든 모션 (SONIC_K1_HOST)
box/prepare.sh                 서버 GEAR-SONIC 체크아웃에 sonic_k1 + 이 overlay·patch 적용
box/train.sh                   smoke | train(tmux) | export(ONNX) | eval(im_eval)
box/curves.py                  학습 로그 → 곡선 CSV·요약
box/make_walk2_motions.sh      보행 세트 + 제스처 ×3
box/bones_fetch.py · bones_extract.py · bones_pipeline.sh · make_bones_train.sh · drop_smpl_encoder.py · bones2.sh   BONES 실험
mac/eval_k1_mujoco.py          export → deploy 폴더 준비, MuJoCo에서 서 있기·제스처 평가 (라이브 백엔드의 K1Controller도 여기)
mac/eval_walk.py               보행 10종 폐루프(플래너 명령 경로), 걸음 품질
mac/eval_walk_clips.py · eval_walk_human.py · eval_walk_table.py · eval_walk_video.py   평가·표·비교 영상
mac/eval_human.py · smpl_stream_video.py      SMPL 인코더(사람 전신) 평가
mac/baseline_student.py · summarize.py        ROBOTIS student 비교 기준, 평가 표
mac/soma_ref_to_pkl.py         SOMA 참조 → PKL
```

`mac/eval_k1_mujoco.py`의 `K1Controller`는 라이브 백엔드(`k1live/backends/{sonic,sonic_human,hybrid}.py`)가 import하므로 이 폴더는 실행에도 필요하다.

## 순서

```bash
# 로컬
.venv/bin/python sonic_k1_train/data/build_dataset.py --out /tmp/k1_motions \
    --mjcf sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml
.venv/bin/python sonic_k1_train/data/planner_dataset.py --out /tmp/k1_walk_planner --minutes 75 --workers 8
SONIC_K1_HOST=<서버 ssh 이름> sonic_k1_train/box/sync_to_box.sh /tmp/k1_motions
# 서버 (작업 폴더 $SONIC_K1_WORK, 기본 $HOME/sonic_k1; 최초 1회 setup_env.sh)
bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/prepare.sh
bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/train.sh smoke
bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/train.sh train k1s_main sonic_k1_small 4096 20000
bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/train.sh export $SONIC_K1_WORK/runs/TRL_K1_Track/<run>/model_step_XXXXXX.pt
# 로컬
rsync -az <서버>:<SONIC_K1_WORK>/runs/TRL_K1_Track/<run>/exported models/k1/<run>/
.venv/bin/python sonic_k1_train/mac/eval_k1_mujoco.py --export models/k1/<run>/exported --step XXXX \
    --motions /tmp/k1_motions --stand 20 --clips bow wave_hello welcome_open_arms clap point_guide --start ground \
    --json eval_<run>.json
```

`train.sh`는 GPU에 다른 compute 프로세스가 있으면 시작하지 않는다. `--future 4x1`은 80 ms lookahead 변형(`sonic_k1_small_la4` 계열) export용.
`--start ground`(발을 바닥에 두고 시작)를 정책 비교에 권한다.
