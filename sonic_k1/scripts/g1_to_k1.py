"""G1 (29 DOF, SONIC output) -> K1 (23 DOF) joint mapping for option B (SONIC G1 -> K1 mimic bridge).

Two mappings, evaluated against G1 forward kinematics (docs/g1_to_k1_mapping.md has the table):
  copy    same-name joint copy, waist roll/pitch and wrist pitch/yaw dropped, clipped to K1 limits
  orient  shoulder and hip triples solved so the K1 upper arm / thigh has the same orientation as on G1
          (G1 hip roll/yaw axes are tilted 10 deg about Y and its shoulder pitch axis 16 deg about X; K1 axes
          are pure Y-X-Z), the G1 torso roll/pitch K1 cannot do is folded into both shoulders so the arms keep
          their pelvis-relative direction; elbow/knee/ankle/wrist-roll copied; clipped to K1 limits

    .venv/bin/python sonic_k1/scripts/g1_to_k1.py --eval           # error table on event gestures + SONIC refs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DROPPED = ["waist_roll_joint", "waist_pitch_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
           "right_wrist_pitch_joint", "right_wrist_yaw_joint"]


class G1ToK1:
    def __init__(self):
        import mujoco

        from sonic.robots import get_robot

        self.mj = mujoco
        self.g1, self.k1 = get_robot("g1"), get_robot("k1")
        self.mg, self.mk = self.g1.load_model(), self.k1.load_model()
        self.dg, self.dk = mujoco.MjData(self.mg), mujoco.MjData(self.mk)
        self.gq = np.array([self.mg.jnt_qposadr[self.mg.joint(n).id] for n in self.g1.mujoco_joint_names])
        self.kq = np.array([self.mk.jnt_qposadr[self.mk.joint(n).id] for n in self.k1.mujoco_joint_names])
        self.gi = {n: i for i, n in enumerate(self.g1.mujoco_joint_names)}
        self.ki = {n: i for i, n in enumerate(self.k1.mujoco_joint_names)}
        rng = np.array([self.mk.jnt_range[self.mk.joint(n).id] for n in self.k1.mujoco_joint_names])
        self.lo, self.hi = rng[:, 0], rng[:, 1]

    def _g1_fk(self, q_g1):
        d = self.dg
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.gq] = q_g1
        self.mj.mj_kinematics(self.mg, d)
        return d

    def _k1_fk(self, q_k1):
        d = self.dk
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[self.kq] = q_k1
        self.mj.mj_kinematics(self.mk, d)
        return d

    def _R(self, d, m, body):
        return np.array(d.xmat[m.body(body).id]).reshape(3, 3)

    def copy(self, q_g1, keep_unclipped=False):
        q = np.zeros(self.k1.num_dof)
        for n, i in self.ki.items():
            q[i] = q_g1[self.gi[n]]
        return q if keep_unclipped else np.clip(q, self.lo, self.hi)

    def orient(self, q_g1, keep_unclipped=False):
        q = self.copy(q_g1, keep_unclipped=True)  # knees, elbows, ankles, waist yaw, wrist roll
        q_unclipped = np.array([q_g1[self.gi[n]] for n in self.k1.mujoco_joint_names])
        d = self._g1_fk(q_g1)
        mg = self.mg
        R_pel = self._R(d, mg, "pelvis")
        wy = q_unclipped[self.ki["waist_yaw_joint"]]
        R_k1_torso = R_pel @ Rot.from_euler("z", wy).as_matrix()  # K1 torso: pelvis * Rz(waist_yaw)
        for side in ("left", "right"):
            # arm: G1 shoulder_yaw_link orientation (== K1 frame convention at zero pose) relative to K1 torso
            R = R_k1_torso.T @ self._R(d, mg, f"{side}_shoulder_yaw_link")
            p, r, y = Rot.from_matrix(R).as_euler("YXZ")  # K1: Ry(pitch) Rx(roll) Rz(yaw), intrinsic
            q[self.ki[f"{side}_shoulder_pitch_joint"]] = p
            q[self.ki[f"{side}_shoulder_roll_joint"]] = r
            q[self.ki[f"{side}_shoulder_yaw_joint"]] = y
            # leg: G1 knee_link frame with the knee angle removed (G1 knee_link re-aligns the -10 deg hip tilt)
            kn = q_g1[self.gi[f"{side}_knee_joint"]]
            R = R_pel.T @ self._R(d, mg, f"{side}_knee_link") @ Rot.from_euler("y", -kn).as_matrix()
            p, r, y = Rot.from_matrix(R).as_euler("YXZ")
            q[self.ki[f"{side}_hip_pitch_joint"]] = p
            q[self.ki[f"{side}_hip_roll_joint"]] = r
            q[self.ki[f"{side}_hip_yaw_joint"]] = y
        return q if keep_unclipped else np.clip(q, self.lo, self.hi)

    # ------------------------------------------------------------------ evaluation
    # Link frames that coincide on both robots at the zero pose (G1's tilted hip/shoulder axes are undone by
    # its next body's quat), compared relative to the pelvis as geodesic angles. Positions differ by design
    # (G1 thigh 0.34 / shank 0.32 m vs K1 0.37 / 0.29 m; G1 wrist sits 1 cm below the forearm axis).
    LINKS = {"upper_arm": ("{s}_shoulder_yaw_link",) * 2, "forearm": ("{s}_elbow_link",) * 2,
             "hand": ("{s}_wrist_yaw_link", "{s}_wrist_roll_rubber_hand"),
             "thigh": ("{s}_knee_link",) * 2, "foot": ("{s}_ankle_roll_link",) * 2}

    def link_R(self, d, m, robot_idx):
        out = {}
        R_pel = self._R(d, m, "pelvis")
        out["torso"] = R_pel.T @ self._R(d, m, "torso_link")
        for k, names in self.LINKS.items():
            for s in ("left", "right"):
                out[f"{s[0]}_{k}"] = R_pel.T @ self._R(d, m, names[robot_idx].format(s=s))
        return out

    def errors(self, q_g1, q_k1):
        rg = {k: v.copy() for k, v in self.link_R(self._g1_fk(q_g1), self.mg, 0).items()}
        rk = self.link_R(self._k1_fk(q_k1), self.mk, 1)
        return {k: float(np.degrees(Rot.from_matrix(rg[k].T @ rk[k]).magnitude())) for k in rg}


def clips():
    """(name, (T,29) G1 MuJoCo-order joint angles) — the 8 event gestures + SONIC example references."""
    from sonic import gestures
    from sonic.motion import list_csv_motions, load_csv_motion
    from sonic.params import ISAACLAB_TO_MUJOCO, REFERENCE_DIR

    out = []
    for n in ("wave_hello", "bow", "welcome_open_arms", "clap", "point_guide", "raise_both_arms", "heart",
              "weight_shift"):
        if hasattr(gestures, n):
            m = getattr(gestures, n)()
            out.append((f"event:{n}", m.joint_pos[:, ISAACLAB_TO_MUJOCO]))
    if REFERENCE_DIR.exists():
        for p in list_csv_motions(REFERENCE_DIR):
            m = load_csv_motion(p)
            out.append((f"ref:{p.name}", m.joint_pos[:, ISAACLAB_TO_MUJOCO]))
    return out


def evaluate(stride=5):
    from collections import Counter

    mp = G1ToK1()
    rows = []
    print("| clip | copy: arm mean / worst | copy: leg | orient: arm mean / worst | orient: leg | hand (wrist pitch/yaw lost) "
          "| torso (waist roll/pitch lost) | K1 limits hit (orient, % frames) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, Q in clips():
        Q = Q[::stride]
        res, hits = {}, Counter()
        for method in ("copy", "orient"):
            errs = []
            for q in Q:
                qk = getattr(mp, method)(q, keep_unclipped=True)
                q_clip = np.clip(qk, mp.lo, mp.hi)
                if method == "orient":
                    for i in np.nonzero((qk < mp.lo - 1e-6) | (qk > mp.hi + 1e-6))[0]:
                        hits[mp.k1.mujoco_joint_names[i].replace("_joint", "")] += 1
                errs.append(mp.errors(q, q_clip))
            arms = np.mean([[e[k] for k in e if "arm" in k] for e in errs])
            legs = np.mean([[e[k] for k in e if k[2:] in ("thigh", "foot")] for e in errs])
            worst = np.max([[e[k] for k in e if "arm" in k] for e in errs])
            hand = np.mean([[e[k] for k in e if "hand" in k] for e in errs])
            torso = np.mean([e["torso"] for e in errs])
            res[method] = (arms, legs, worst, hand, torso)
        rows.append((name, res, len(Q), hits))
        hit_s = ", ".join(f"{k} {100 * v / len(Q):.0f}%" for k, v in hits.most_common(3)) or "-"
        c, o = res["copy"], res["orient"]
        print(f"| {name} | {c[0]:.1f} / {c[2]:.1f} | {c[1]:.1f} | {o[0]:.1f} / {o[2]:.1f} | {o[1]:.1f} | {o[3]:.1f} | "
              f"{o[4]:.1f} | {hit_s} |", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    if a.eval:
        evaluate(a.stride)
