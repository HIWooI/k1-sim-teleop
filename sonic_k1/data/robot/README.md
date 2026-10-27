GEAR-SONIC 학습용 K1 모션 라이브러리(`train.env`: `SONIC_MOTION_FILE=data/robot`). PKL은 git에 넣지 않고 soma-retargeter K1 CSV
(`*_retarget_k1_from_bvh.csv`)에서 다시 만든다:

    .venv/bin/python sonic_k1/scripts/k1_csv_to_pkl.py <csv 또는 폴더> --out sonic_k1/data/robot
    .venv/bin/python sonic_k1/scripts/replay_pkl.py sonic_k1/data/robot/<name>.pkl --video videos/<name>_k1.mp4

파일 하나에 모션 하나, 키는 모션 이름: `root_trans_offset` (T,3) m, `pose_aa` (T,24,3) MuJoCo 바디 순서, `dof` (T,23) MuJoCo 순서,
`root_rot` (T,4) xyzw, `smpl_joints` (T,24,3) 0(자리 표시), `fps` 30.
[docs/training.md](../../../docs/training.md)의 학습 run들은 `sonic_k1_train/data/`로 모션 라이브러리를 만든다.
