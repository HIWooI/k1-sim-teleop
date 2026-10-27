"""k1/run_video.py 50 Hz split-mode references (logs/k1_ref_<clip><suffix>.npz) -> K1 motion_lib PKLs for
eval_k1_mujoco.py, so the SONIC K1 policy can be run on exactly the arm reference the showreel's split drive used.

Same construction as build_dataset.s1_gestures: legs moved from the ROBOTIS loco stance to the SONIC K1 stance,
pelvis upright at 0.8 m, grounded on the foot spheres with both feet planted (lock_xy), no mirror.

    .venv/bin/python sonic_k1_train/mac/soma_ref_to_pkl.py logs/k1_ref_pexels_10374292_soma_ik.npz ... \
        --out /tmp/k1_soma_refs --mjcf sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml
    .venv/bin/python sonic_k1_train/mac/eval_k1_mujoco.py --variant-dir models/k1/k1s_main/deploy_003500 \
        --motions /tmp/k1_soma_refs --clips /tmp/k1_soma_refs/robot/soma/<clip>.pkl ... --stand 5 --start ground
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from k1_kin import STANCE_LOCO, STANCE_SONIC, K1Kin, save_motion  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MJCF = REPO / "sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mjcf", type=Path, default=MJCF)
    a = ap.parse_args()
    k = K1Kin(a.mjcf)
    for f in a.refs:
        z = np.load(f, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        assert names == k.names, "k1/sim.py policy order != K1 MuJoCo order"
        if str(z["drive"]) != "split":
            sys.exit(f"{f}: only split-mode (arms) references are converted")
        dof = z["q"].astype(np.float64).copy()
        for s in ("left", "right"):
            for j in ("hip_pitch", "knee", "ankle_pitch"):
                dof[:, k.J[f"{s}_{j}_joint"]] += STANCE_SONIC[j] - STANCE_LOCO[j]
        dof = k.clip(dof)
        T = len(dof)
        quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
        root = k.ground(np.tile([0.0, 0.0, 0.8], (T, 1)), quat, dof, lock_xy=True)
        name = f.stem.replace("k1_ref_", "")
        save_motion(a.out / "robot" / "soma", name, k.entry(root, quat, dof, float(z["fps"])))
        print(f"{f.name} -> {a.out / 'robot/soma' / (name + '.pkl')}  T={T} @ {float(z['fps'])} fps")


if __name__ == "__main__":
    main()
