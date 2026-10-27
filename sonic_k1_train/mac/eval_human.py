"""Closed-loop MuJoCo check of a K1 SONIC human-encoder (SMPL) checkpoint on converted pairs (held-out BONES-SEED,
GEM-X clips): the person's SMPL stream drives k1live.backends.sonic_human exactly as the live sender would; the K1
reference of the same clip is used only for scoring (and an optional ghost in the video).

    MUJOCO_GL=egl python sonic_k1_train/mac/eval_human.py --variant models/k1/k1s_human/deploy \
        --set <set dir> --heldout 20 --json results/human_eval.json [--video videos/k1_human_eval.mp4]

Per clip: respawn standing, 1 s of the clip's first person frame (the presence gate fades in), the clip at 50 Hz,
1 s holding its last frame. Wrist roll input = the reference's (the live sender sends its IK value).
Metrics (reference taken (n_smpl - 1) ticks back = the delay-mode lag): fall (pelvis < 0.45 m or tilt > 45 deg, the
sim_server rule) and down (pelvis < 0.45 m: a real fall; only that ends the clip early), min pelvis height,
arm / leg joint error (deg), max tilt, root travel ratio (robot path length / reference path length), heading change
error inside the clip (robot vs reference, unwrapped, deg; the lead-in fade turn is reported apart), policy ms per tick.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "sonic_k1_train" / "data"), str(REPO / "sonic_k1_train" / "mac")):
    if p not in sys.path:
        sys.path.insert(0, p)

from k1live import smpl_stream  # noqa: E402
from sonic import rot  # noqa: E402

DOWN_Z = 0.45   # m: the pelvis really on the way down (the sim_server's fall rule also counts tilt > 45 deg, which a
                # deep bow reaches while standing - the clip then keeps running and only a real fall stops it)


def load_pair(set_dir: Path, key: str):
    import joblib

    p = next(set_dir.rglob(f"robot/**/{key}.pkl"))
    e = joblib.load(p)[key]
    s = joblib.load(p.parents[2] / "smpl" / f"{key}.pkl")
    q = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][:, :3])
    J = smpl_stream.world_joints(s["smpl_joints"], q * np.array([1.0, -1.0, -1.0, -1.0]))
    return p, e, J, q


def _yaw(q):
    return float(rot.calc_heading(np.asarray(q, np.float64)))


def run_clip(be, set_dir, key, rec=None, gh=None, every=2):
    from sonic.motion import load_robot_pkl

    p, e, J, Q = load_pair(set_dir, key)
    wr = [be.names.index(f"{s}_wrist_roll_joint") for s in ("left", "right")]
    msgs = [{"present": True, "smpl": smpl_stream.to_msg(J[f], Q[f], e["dof"][f][wr])} for f in range(len(J))]
    be.reset()
    arm = be.arm_default.copy()
    lag = be.n_smpl - 1 if be.smpl_future == "delay" else 0
    ref = np.asarray(e["dof"], np.float64)
    names = be.names
    legs = [i for i, n in enumerate(names) if any(t in n for t in ("hip", "knee", "ankle"))]
    arms = [i for i, n in enumerate(names) if any(t in n for t in ("shoulder", "elbow", "wrist"))]
    seq = [msgs[0]] * 50 + msgs + [msgs[-1]] * 50
    q_log, ref_log, xy, yaw, tilt, ms, frames = [], [], [], [], [], [], []
    fell = down = None
    zmin = 9.0
    ghost_m = load_robot_pkl(p, robot=be.robot) if gh is not None else None
    for k, m in enumerate(seq):
        info = be.step(arm, human=m)
        ms.append(info["policy_ms"])
        f = min(max(k - 50 - lag, 0), len(ref) - 1)
        if 50 + lag <= k < 50 + len(ref) + lag:
            q_log.append(be.q_now())
            ref_log.append(ref[f])
        x, y, yw = be.root_pose()
        xy.append((x, y))
        yaw.append(yw)
        tilt.append(be.tilt_deg())
        zmin = min(zmin, be.pelvis_height())
        if fell is None and be.fallen():
            fell = round(k * be.dt, 2)
        if down is None and be.pelvis_height() < DOWN_Z:
            down = round(k * be.dt, 2)
        if rec is not None and k % every == 0:
            img = rec.grab()
            if gh is not None:
                adh = rot.quat_mul(rot.calc_heading_quat(be.ctrl.heading.init_base_quat),
                                   rot.calc_heading_quat_inv(ghost_m.body_quat[0, 0]))
                g = gh.render(ghost_m, f, adh, np.asarray(xy[0]) + _rot2(yaw[0] - _yaw(ghost_m.body_quat[0, 0])) @
                              (ghost_m.body_pos[f, 0, :2] - ghost_m.body_pos[0, 0, :2]))
                img = np.concatenate([g, img], axis=1)
            frames.append(_caption(img, f"{key}  t={k * be.dt:5.1f}s  ({'FALL' if fell else info['mode']})"))
        if down is not None and k * be.dt > down + 1.0:
            break
    q_log, ref_log = np.array(q_log), np.array(ref_log)
    err = np.degrees(np.abs(q_log - ref_log)) if len(q_log) else np.zeros((1, len(names)))
    rp = np.asarray(e["root_trans_offset"])[:, :2]
    ref_path = float(np.linalg.norm(np.diff(rp, axis=0), axis=1).sum())
    xy = np.array(xy)
    rob_path = float(np.linalg.norm(np.diff(xy[50:], axis=0), axis=1).sum())
    # turning inside the clip (unwrapped, so a 360 deg over-rotation counts): the reference's root heading from its
    # first to its last frame vs the robot's from the clip's first tick (k = 50) to the end. The lead-in turn before
    # that is the presence-gate fade from the idle pose (heading 0) to the clip's absolute heading.
    ref_h = np.unwrap([_yaw(_wxyz(q)) for q in e["root_rot"]])
    ref_turn = float(ref_h[-1] - ref_h[0])
    yu = np.unwrap(yaw)
    k0 = min(50, len(yu) - 1)
    rob_turn = float(yu[-1] - yu[k0])
    return {"key": key, "seconds": round(len(ref) / 50.0, 1), "fell_at": fell, "down_at": down,
            "min_pelvis_z": round(zmin, 3),
            "arms_deg": round(float(err[:, arms].mean()), 2), "legs_deg": round(float(err[:, legs].mean()), 2),
            "arms_p95_deg": round(float(np.percentile(err[:, arms], 95)), 2), "max_tilt_deg": round(max(tilt), 1),
            "ref_path_m": round(ref_path, 2), "robot_path_m": round(rob_path, 2),
            "path_ratio": round(rob_path / ref_path, 2) if ref_path > 0.2 else None,
            "ref_turn_deg": round(float(np.degrees(ref_turn)), 1), "robot_turn_deg": round(float(np.degrees(rob_turn)), 1),
            "turn_err_deg": round(float(np.degrees(abs(rob_turn - ref_turn))), 1),
            "lead_in_turn_deg": round(float(np.degrees(yu[k0] - yu[0])), 1),
            "policy_ms": round(float(np.mean(ms)), 2)}, frames


def _wxyz(q_xyzw):
    q = np.asarray(q_xyzw, np.float64)
    return q[[3, 0, 1, 2]]


def _rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _caption(img, text):
    import cv2

    img = img.copy()
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--keys", nargs="*", default=[])
    ap.add_argument("--heldout", type=int, default=0, help="also take the first N held-out keys")
    ap.add_argument("--future", default="delay", choices=["delay", "hold"])
    ap.add_argument("--json", default="")
    ap.add_argument("--video", default="")
    a = ap.parse_args()
    from k1live.backends.sonic_human import SonicHumanBackend

    keys = list(a.keys)
    if a.heldout:
        keys += sorted(p.stem for p in (a.set / "heldout" / "robot").rglob("*.pkl"))[:a.heldout]
    be = SonicHumanBackend(a.variant, smpl_future=a.future)
    rec = gh = None
    if a.video:
        from eval_k1_mujoco import Ghost

        from sonic.sim import Recorder

        cam = dict(width=480, height=400, distance=2.6, azimuth=160.0, elevation=-10.0)
        rec, gh = Recorder(be.sim, **cam), Ghost(be.sim, be.robot, **cam)
    rows, frames = [], []
    for k in keys:
        r, fr = run_clip(be, a.set, k, rec, gh)
        rows.append(r)
        frames += fr
        print(json.dumps(r), flush=True)
    ok = [r for r in rows if r["fell_at"] is None]
    summ = {"variant": str(a.variant), "clips": len(rows), "falls": len(rows) - len(ok),
            "falls_down": sum(r["down_at"] is not None for r in rows),
            "fall_keys": [r["key"] for r in rows if r["fell_at"] is not None],
            "arms_deg_mean_up": round(float(np.mean([r["arms_deg"] for r in rows if r["down_at"] is None])), 2),
            "legs_deg_mean_up": round(float(np.mean([r["legs_deg"] for r in rows if r["down_at"] is None])), 2),
            "arms_deg_mean": round(float(np.mean([r["arms_deg"] for r in ok])), 2) if ok else None,
            "legs_deg_mean": round(float(np.mean([r["legs_deg"] for r in ok])), 2) if ok else None,
            "rows": rows}
    if a.video and frames:
        import imageio

        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(a.video, frames, fps=25, quality=7, macro_block_size=1)
        summ["video"] = a.video
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
