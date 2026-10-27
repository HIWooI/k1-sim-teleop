"""Teleop tracking metric: limb directions of the simulated G1 vs the commanded SMPL limbs.

Both are expressed in their own root frame (robot pelvis body frame vs SMPL root frame), so the
metric is independent of heading/translation and of the different bone lengths.
"""
import numpy as np

from . import rot

# (name, smpl parent, smpl child, robot parent body, robot child body)
LIMBS = [
    ("L_upper_arm", 16, 18, "left_shoulder_roll_link", "left_elbow_link"),
    ("L_forearm", 18, 20, "left_elbow_link", "left_wrist_roll_link"),
    ("R_upper_arm", 17, 19, "right_shoulder_roll_link", "right_elbow_link"),
    ("R_forearm", 19, 21, "right_elbow_link", "right_wrist_roll_link"),
    ("L_thigh", 1, 4, "left_hip_roll_link", "left_knee_link"),
    ("L_shin", 4, 7, "left_knee_link", "left_ankle_roll_link"),
    ("R_thigh", 2, 5, "right_hip_roll_link", "right_knee_link"),
    ("R_shin", 5, 8, "right_knee_link", "right_ankle_roll_link"),
    ("trunk", 0, 12, "pelvis", "torso_link"),
]


class LimbTracker:
    def __init__(self, model):
        self.ids = [(n, a, b, model.body(pa).id, model.body(pb).id) for n, a, b, pa, pb in LIMBS]
        self.pelvis = model.body("pelvis").id
        self.err = {n: [] for n, *_ in LIMBS}

    def update(self, data, smpl_local):
        q_p = data.xquat[self.pelvis]
        for n, a, b, ia, ib in self.ids:
            v_r = rot.quat_rotate(rot.quat_conj(q_p), data.xpos[ib] - data.xpos[ia])
            v_s = smpl_local[b] - smpl_local[a]
            c = np.dot(v_r, v_s) / (np.linalg.norm(v_r) * np.linalg.norm(v_s) + 1e-9)
            self.err[n].append(np.degrees(np.arccos(np.clip(c, -1, 1))))

    def summary(self, skip=0):
        out = {n: float(np.mean(v[skip:])) for n, v in self.err.items() if len(v) > skip}
        arms = [out[k] for k in out if "arm" in k]
        legs = [out[k] for k in out if k[2:] in ("thigh", "shin")]
        if arms:
            out["arms_mean"] = float(np.mean(arms))
        if legs:
            out["legs_mean"] = float(np.mean(legs))
        return out
