"""Build the small K1 SONIC motion library (motion_lib PKLs, MuJoCo joint order) on the Mac.

    .venv/bin/python sonic_k1_train/data/build_dataset.py --out sonic_k1_train/build/k1_motions \
        --mjcf sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml

Sources (BONES-SEED is license-gated -> this library is built from other sources):
  gest_*   K1 event gestures (k1/gestures.py: palm/elbow IK in K1 joint space, 50 Hz), two leg stances
           (ROBOTIS loco stance, SONIC K1 default stance) x three speeds (fps x 0.85 / 1.0 / 1.2)
  cyclo_*  ROBOTIS cyclo_lab K1_rev1 dance1/dance2 (K1-native CSV, 50 Hz; $CYCLO_LAB_ROOT, default third_party/cyclo_lab)
  guap     GEM-X Guap soma-retargeter K1 CSV (deg / cm, 30 Hz; $K1_GUAP_CSV, skipped when absent)
  g1k1_*   SONIC G1 example motions (gear_sonic_deploy/reference/example + sample walk PKL) mapped joint-by-joint
           to K1 (same axes / zero pose / segment lengths; G1 waist roll+pitch folded into the pelvis with the
           hip joints compensated, G1 wrist pitch/yaw dropped, K1 joint limits, re-grounded)
  syn_*    synthetic standing: idle, lateral / sagittal weight shift, shallow squat, pelvis yaw, random smooth arm
           motions, all with feet held by leg IK
Every clip is grounded on the K1 foot contact spheres, then mirrored (_M) unless the source already has its mirror.
Held out of training (generalisation check): gest_heart*, g1k1_squat_001__A359* -> <out>/heldout/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_kin import (ARMS_DEFAULT, STANCE_LOCO, STANCE_SONIC, K1Kin, quat_xyzw_to_wxyz,  # noqa: E402
                    resample, save_motion, smooth_noise)

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor/GR00T-WholeBodyControl"
G1_REF = VENDOR / "gear_sonic_deploy/reference/example"
G1_SAMPLE = REPO / "models/sample_data/robot_filtered"
CYCLO = Path(os.environ.get("CYCLO_LAB_ROOT", REPO / "third_party/cyclo_lab")) / "source/cyclo_lab/data/motions/K1_rev1"
GUAP = Path(os.environ.get("K1_GUAP_CSV", REPO / "third_party/gem-x/Guap_ver2_retarget_k1_from_bvh.csv"))
HELDOUT = ("gest_heart", "g1k1_squat_001__A359")
SKIP_G1 = ("jumping",)  # flight phases: grounding would erase them, K1 is not trained for jumps here

G1_MJ = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
         "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
         "waist_yaw", "waist_roll", "waist_pitch",
         "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
         "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
         "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
         "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
G1_IL_TO_MJ = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
                        12, 16, 20, 22, 24, 26, 28])  # convert_soma_csv_to_motion_lib.MJ_TO_IL: dof_mj = jp_il[:, this]


class Builder:
    def __init__(self, kin: K1Kin, out: Path):
        self.k, self.out, self.manifest = kin, out, []

    def add(self, name, group, root_pos, root_quat_wxyz, dof, fps, lock_xy=False, mirror=True, clip=True):
        k = self.k
        dof = k.clip(dof) if clip else np.asarray(dof)
        root_pos = k.ground(root_pos, root_quat_wxyz, dof, lock_xy=lock_xy)
        variants = [(name, root_pos, root_quat_wxyz, dof)]
        if mirror:
            variants.append((name + "_M",) + k.mirror(root_pos, root_quat_wxyz, dof))
        for n, p, q, d in variants:
            held = n.startswith(HELDOUT)
            sub = self.out / ("heldout" if held else "robot") / group
            save_motion(sub, n, k.entry(p, q, d, fps))
            self.manifest.append({"name": n, "group": group, "frames": len(d), "fps": fps,
                                  "seconds": round(len(d) / fps, 2), "heldout": held,
                                  "pelvis_z": [round(float(p[:, 2].min()), 3), round(float(p[:, 2].max()), 3)]})

    def summary(self):
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=1))
        tab = {}
        for r in self.manifest:
            c = tab.setdefault((r["heldout"], r["group"]), [0, 0.0])
            c[0] += 1
            c[1] += r["seconds"]
        for (held, group), (n, sec) in sorted(tab.items()):
            print(f"{'heldout' if held else 'train':8s} {group:10s} clips {n:4d}  {sec / 60:6.1f} min")
        tr = [r for r in self.manifest if not r["heldout"]]
        print("total train clips", len(tr), "minutes", round(sum(r["seconds"] for r in tr) / 60, 1))


# ---------------------------------------------------------------------------------------------- sources
def restance(kin, dof, src, dst):
    """Move the legs from stance src to stance dst, keeping per-frame deviations (e.g. bow hip flexion)."""
    d = dof.copy()
    for s in ("left", "right"):
        for j in ("hip_pitch", "knee", "ankle_pitch"):
            i = kin.J[f"{s}_{j}_joint"]
            d[:, i] += dst[j] - src[j]
    return d


def s1_gestures(b: Builder, s1_dir: Path):
    sys.path.insert(0, str(s1_dir))
    from k1 import gestures as G, sim as S  # the k1/ package

    sim = S.K1Split(mode="split")
    assert sim.names == b.k.names, "k1/ policy order != K1 MuJoCo order"
    ik = G.ArmIK(sim)
    for gname, f in G.GESTURES.items():
        m = f(ik)
        torso = Rot.from_quat(m.torso_quat)  # xyzw
        yaw = m.q[:, b.k.J["waist_yaw_joint"]]
        pelvis = quat_xyzw_to_wxyz((torso * Rot.from_euler("z", -yaw[:, None])).as_quat())
        for stance_name, stance in (("loco", STANCE_LOCO), ("sonic", STANCE_SONIC)):
            dof = restance(b.k, m.q, STANCE_LOCO, stance)
            root = np.tile([0.0, 0.0, 0.8], (m.T, 1))
            for sp in (0.85, 1.0, 1.2):
                b.add(f"gest_{gname}_{stance_name}_x{sp:.2f}", "gestures", root, pelvis, dof, 50.0 * sp, lock_xy=True)


def cyclo_dances(b: Builder):
    for csv in sorted(CYCLO.glob("*/*.csv")):
        a = np.loadtxt(csv, delimiter=",")
        b.add(f"cyclo_{csv.stem}", "cyclo", a[:, :3], quat_xyzw_to_wxyz(a[:, 3:7]), a[:, 7:30], 50.0)


def guap(b: Builder):
    if not GUAP.exists():
        print(f"guap: {GUAP} not found, skipped")
        return
    a = np.genfromtxt(GUAP, delimiter=",", names=True)
    col = lambda *ns: np.stack([a[n] for n in ns], 1)  # noqa: E731
    pos = col("root_translateX", "root_translateY", "root_translateZ") / 100.0
    eul = col("root_rotateX", "root_rotateY", "root_rotateZ")
    q = quat_xyzw_to_wxyz(Rot.from_euler("xyz", eul, degrees=True).as_quat())
    dof = np.deg2rad(col(*[f"{n}_dof" for n in b.k.names]))
    b.add("guap_ver2", "guap", pos, q, dof, 30.0)


def g1_to_k1(kin: K1Kin, root_pos, root_quat_wxyz, dof_g1_mj):
    g = {n: dof_g1_mj[:, i] for i, n in enumerate(G1_MJ)}
    T = len(root_pos)
    dof = np.zeros((T, kin.n))
    for n in kin.names:
        dof[:, kin.J[n]] = g[n.replace("_joint", "")]
    # K1 has waist yaw only: fold G1 waist roll/pitch into the pelvis, compensate the hips so the thighs keep
    # their world orientation (exact for pitch, first order for roll)
    wr, wp = g["waist_roll"], g["waist_pitch"]
    R = Rot.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]]) * Rot.from_euler("y", wp[:, None]) * Rot.from_euler("x", wr[:, None])
    for s in ("left", "right"):
        dof[:, kin.J[f"{s}_hip_pitch_joint"]] -= wp
        dof[:, kin.J[f"{s}_hip_roll_joint"]] -= wr
    return root_pos, quat_xyzw_to_wxyz(R.as_quat()), dof


def g1_examples(b: Builder):
    import joblib

    for d in sorted(G1_REF.iterdir()):
        if not (d / "joint_pos.csv").exists() or any(s in d.name for s in SKIP_G1):
            continue
        rd = lambda f: np.loadtxt(d / f, delimiter=",", skiprows=1, ndmin=2)  # noqa: E731
        jp, bp, bq = rd("joint_pos.csv"), rd("body_pos.csv"), rd("body_quat.csv")
        T = len(jp)
        bp, bq = bp.reshape(T, -1, 3), bq.reshape(T, -1, 4)
        p, q, dof = g1_to_k1(b.k, bp[:, 0], bq[:, 0], jp[:, G1_IL_TO_MJ])
        has_mirror = (G1_REF / (d.name + "_M")).exists() or d.name.endswith("_M")
        b.add(f"g1k1_{d.name}", "g1k1", p, q, dof, 50.0, mirror=not has_mirror)
    for f in sorted(G1_SAMPLE.rglob("*.pkl")):
        e = next(iter(joblib.load(f).values()))
        p, q, dof = g1_to_k1(b.k, e["root_trans_offset"], quat_xyzw_to_wxyz(e["root_rot"]), e["dof"])
        b.add(f"g1k1_{f.stem}", "g1k1", p, q, dof, float(e["fps"]), mirror=False)  # sample set has its _M


def synthetic(b: Builder, n_arm=24, seed=0):
    """Standing padding with the feet held by leg IK under a moving pelvis, plus random smooth arm motion."""
    k, fps, rng = b.k, 50.0, np.random.default_rng(seed)
    arm_names = [n for n in k.names if any(t in n for t in ("shoulder", "elbow", "wrist"))]
    arm_idx = [k.J[n] for n in arm_names]
    # safe arm box around the default pose (radians): pitch forward/up, roll out, yaw, elbow, wrist
    amp = {"shoulder_pitch": (-1.6, 0.5), "shoulder_roll": (0.0, 0.9), "shoulder_yaw": (-0.6, 0.6),
           "elbow": (-0.2, 1.4), "wrist_roll": (-0.8, 0.8)}

    def arms_random(T, scale):
        q = np.zeros((T, len(arm_idx)))
        z = smooth_noise(rng, T, fps, len(arm_idx), period=(1.2, 3.0))
        for c, n in enumerate(arm_names):
            key = next(a for a in amp if a in n)
            lo, hi = amp[key]
            if n.startswith("right") and key in ("shoulder_roll",):
                lo, hi = -hi, -lo
            centre = ARMS_DEFAULT.get(n, 0.0)
            half = scale * 0.5 * (hi - lo)
            q[:, c] = np.clip(centre + half * z[:, c], lo, hi)
        return q

    def clip_with_pelvis(name, T, pelvis_fn, stance, arms_fn=None, waist_fn=None):
        dof0 = k.pose(stance)
        p0 = np.array([0.0, 0.0, 0.8])
        q0 = np.array([1.0, 0, 0, 0])
        p0[2] = k.ground(p0[None], q0[None], dof0[None])[0, 2]
        feet = k.foot_poses(p0, q0, dof0)
        P, Q, D = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, k.n))
        prev = dof0
        arms = arms_fn(T) if arms_fn else None
        for t in range(T):
            dp, rpy = pelvis_fn(t / fps)
            P[t] = p0 + dp
            Q[t] = Rot.from_euler("xyz", rpy).as_quat()[[3, 0, 1, 2]]
            d = prev.copy()
            if arms is not None:
                d[arm_idx] = arms[t]
            if waist_fn:
                d[k.J["waist_yaw_joint"]] = waist_fn(t / fps)
            prev = D[t] = k.leg_ik(P[t], Q[t], d, feet, q_init=prev)
        b.add(name, "synthetic", P, Q, D, fps, lock_xy=True)

    sin = lambda f, t, ph=0.0: np.sin(2 * np.pi * f * t + ph)  # noqa: E731
    for st_name, st in (("sonic", STANCE_SONIC), ("loco", STANCE_LOCO)):
        clip_with_pelvis(f"syn_stand_idle_{st_name}", 600,
                         lambda t: (np.array([0, 0, 0.004 * sin(0.25, t)]), np.zeros(3)), st)
        clip_with_pelvis(f"syn_shift_lr_{st_name}", 750,
                         lambda t: (np.array([0, 0.035 * sin(0.2, t), -0.01 * (1 - np.cos(4 * np.pi * 0.2 * t)) / 2]),
                                    np.array([0.03 * sin(0.2, t), 0, 0])), st)
        clip_with_pelvis(f"syn_shift_fb_{st_name}", 600,
                         lambda t: (np.array([0.02 * sin(0.25, t), 0, 0]), np.array([0, 0.05 * sin(0.25, t), 0])), st)
        clip_with_pelvis(f"syn_squat_{st_name}", 600,
                         lambda t: (np.array([0, 0, -0.035 * (1 - np.cos(2 * np.pi * 0.15 * t))]), np.zeros(3)), st)
        clip_with_pelvis(f"syn_yaw_{st_name}", 600,
                         lambda t: (np.zeros(3), np.array([0, 0, 0.12 * sin(0.2, t)])), st,
                         waist_fn=lambda t: 0.35 * sin(0.2, t, 1.0))
    for i in range(n_arm):
        st = STANCE_SONIC if i % 2 == 0 else STANCE_LOCO
        scale = 0.5 if i < n_arm // 2 else 0.9
        f1, f2, ph = rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3), rng.uniform(0, 6.28)
        ay, ax, ar = rng.uniform(0, 0.03), rng.uniform(0, 0.015), rng.uniform(0, 0.08)
        clip_with_pelvis(
            f"syn_arms_{i:02d}", 500,
            lambda t, f1=f1, f2=f2, ph=ph, ay=ay, ax=ax, ar=ar: (
                np.array([ax * sin(f2, t), ay * sin(f1, t, ph), -0.01 * (1 - np.cos(2 * np.pi * f2 * t)) / 2]),
                np.array([ay * sin(f1, t, ph), 0.0, ar * sin(f2, t)])),
            st, arms_fn=lambda T, s=scale: arms_random(T, s),
            waist_fn=lambda t, f1=f1: 0.25 * sin(f1, t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--s1", type=Path, default=REPO, help="directory containing the k1/ package (repo root)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of: gestures cyclo guap g1k1 synthetic")
    ap.add_argument("--n-arm", type=int, default=24)
    a = ap.parse_args()
    b = Builder(K1Kin(a.mjcf), a.out)
    steps = {"gestures": lambda: s1_gestures(b, a.s1), "cyclo": lambda: cyclo_dances(b), "guap": lambda: guap(b),
             "g1k1": lambda: g1_examples(b), "synthetic": lambda: synthetic(b, a.n_arm)}
    for name, fn in steps.items():
        if a.only is None or name in a.only:
            fn()
            print(f"[{name}] done, {len(b.manifest)} files so far", flush=True)
    b.summary()


if __name__ == "__main__":
    main()
