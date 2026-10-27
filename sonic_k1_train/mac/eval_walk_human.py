"""Walking people (real video) through the live chain, headless: does the robot walk where the person walks, and stay up?

Same per-tick chain as scripts/render_walk_demo.py without the rendering: the clip's MediaPipe
landmarks (scripts/run_video.py cache) -> k1live.human.HumanMapper (arms, waist, bow) + k1live.locomotion.LocoEstimator
(walking intent) -> VelocityLimiter.for_backend -> backend.step at 50 Hz. Backends: split (ROBOTIS loco + student arms),
sonic with a walking checkpoint (--variant <deploy dir whose name contains "walk"> -> planner + walking; the planner's
vx_gain comes from $K1LIVE_PLANNER_VX_GAIN), hybrid (split arms standing, --variant SONIC + planner walking) and
sonic_human (--human-variant: K1 SONIC SMPL encoder; the clip's HumanFrame.smpl - the sender's whole-body stream - drives
legs and steps; the walking intent is not used and only serves as the reference of where the person went).
Needs the k1live tree (run from the repo root).

Metrics per segment and backend: fall, max tilt, time the limited command asks to move, net displacement and turn of
the robot vs the integral of the limited command (body frame), arm error vs the slew-limited target, and the gait
(eval_walk.gait_metrics: stride, cadence, foot lift = swing ankle rise) on the ticks where the robot itself moves.

  K1LIVE_PLANNER_VX_GAIN=1.15 .venv/bin/python sonic_k1_train/mac/eval_walk_human.py \
      --seg toward=<pexels_6764051.mp4>@2-17 --seg turn=<pexels_9558217.mp4>@0-12.4 --seg side=<pexels_3888260.mp4>@6-21.4 \
      --variant ../data/deploy/k1s_walk3_003000 --json logs/walk_human.json
  # GEM-X (SOMA) input: every-frame real-time-worker dumps, live timing (scripts/gemx_feed.py)
  .venv/bin/python sonic_k1_train/mac/eval_walk_human.py --seg ... --gemx-root <dumps> --arm-ik dir --backends split hybrid
  # event arm path, hybrid vs the human encoder
  .venv/bin/python sonic_k1_train/mac/eval_walk_human.py --seg ... --arm-ik dir --backends hybrid sonic_human \
      --variant models/k1/k1s_walk4/deploy/k1s_walk4_007000 --human-variant models/k1/k1s_bones/deploy/<step>_walkhuman
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "sonic_k1_train" / "mac"))

DT = 0.02
ARM_VMAX = 6.0
MOVE_V = 0.08      # m/s, pelvis speed over 0.5 s: the robot itself is walking (gait metrics on these ticks)


def robot_moving(xy, half=12):
    xy = np.asarray(xy, float)
    v = np.zeros(len(xy))
    if len(xy) > 2 * half:
        v[half:-half] = np.linalg.norm(xy[2 * half:] - xy[:-2 * half], axis=1) / (2 * half * DT)
    return v > MOVE_V


def gemx_input(path, gemx_root, retime_ms=34.0):
    """GEM-X instead of MediaPipe: the every-frame real-time-worker dump of the clip (<gemx_root>/<video stem>.npz)
    with live timing (scripts/gemx_feed.py). -> (rows, {video frame: row}) - the sender maps only the frames the worker
    returned, through JumpGate like k1live.sender_gemx."""
    import gemx_feed as GF

    rows = GF.load_rows(GF.find_dump(gemx_root, Path(path).stem)).retime(retime_ms)
    return rows, {int(k): i for i, k in enumerate(rows.k)}


def run(spec, backend, variant, mirror=False, human_variant=None, arm_ik="shape18", gemx_root=None, profile="visitor",
        arm_mode="student", arm_vmax=ARM_VMAX):
    import cv2
    import run_video as g1rv
    from eval_walk import foot_contacts, gait_metrics

    from k1live.backends.base import make_backend
    from k1live.human import HumanMapper
    from k1live.locomotion import LocoEstimator, VelocityLimiter

    name, rest = spec.split("=", 1)
    path, _, rng = rest.partition("@")
    a, _, b = rng.partition("-")
    t0, t1 = float(a or 0), float(b) if b else None
    if gemx_root:
        from k1live.gemx import GemxLocoEstimator
        from k1live.human import JumpGate

        rows, row_at = gemx_input(path, gemx_root)
        vfps, n_vid, wh = rows.fps, rows.n_video, rows.wh
        gate = JumpGate()
        hm = HumanMapper(mirror=mirror, loco=False, arm_ik=arm_ik, perception="gemx")
        est = GemxLocoEstimator(mirror=mirror, profile=profile)
    else:
        W, V, I, vfps = g1rv.extract(path, "full", None, 1, 1280)
        cap = cv2.VideoCapture(path)
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        n_vid = len(W)
        hm = HumanMapper(mirror=mirror, loco=False, arm_ik=arm_ik)
        est = LocoEstimator(mirror=mirror, profile=profile)
    k0 = int(t0 * vfps)
    k1 = n_vid if t1 is None else min(n_vid, int(t1 * vfps))
    kw = {"sonic": {"variant_dir": variant}, "hybrid": {"variant_dir": variant, "arm_mode": arm_mode},
          "split": {"arm_mode": arm_mode}, "sonic_human": {"variant_dir": human_variant}}.get(backend, {})
    be = make_backend(backend, **kw)
    # sonic_human ignores the command; its limiter only builds the reference of where the person walked
    lim = VelocityLimiter.for_backend(getattr(be, "limiter_preset", "sonic" if backend == "sonic_human" else backend))
    arm = be.arm_default.copy()
    last, k_last = None, -1
    v = np.zeros(3)
    v_cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    L = {"lim": [], "xy": [], "yaw": [], "tilt": [], "err": [], "feet": [], "on": [], "knee": [], "rp": [], "z": []}
    m, d = be.model, be.data
    feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
    floor = m.geom("floor").id
    knees = [be.names.index(f"{s}_knee_joint") for s in ("left", "right")]
    fell = None
    n = int((k1 - k0) / vfps / DT)
    for i in range(n):
        t = i * DT
        k = k0 + int(t * vfps)
        if k != k_last and k < k1 and (not gemx_root or k in row_at):   # GEM-X: only the frames the worker returned
            k_last = k
            if gemx_root:
                det = gate(rows.det(row_at[k]))
                if gate.relocked:
                    hm.on_relock()
                    est.forget_home()
            else:
                det = None if np.isnan(W[k, 0, 0]) else {"world": W[k], "vis": V[k], "image": I[k]}
            f = hm.process(det, k / vfps, *wh)
            last = (det, f)
            obs = f.get("obs")
            c = est.update(obs) if obs is not None and f["present"] else None
            v_cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0} if c is None else c
        tgt = be.arm_default if (last is None or last[1]["arm_q"] is None) else np.asarray(last[1]["arm_q"])
        arm = arm + np.clip(tgt - arm, -arm_vmax * DT, arm_vmax * DT)
        v = lim(v_cmd, DT)
        wy = last[1].get("waist_yaw") if last is not None else None
        bw = last[1].get("bow") if last is not None else None
        extra = {}
        if backend == "sonic_human":   # sim_server: human = the followed person frame while tracking, else None
            f_ = last[1] if last is not None else None
            extra["human"] = {"present": True, "smpl": f_["smpl"]} if (f_ is not None and f_["present"]
                                                                       and f_.get("smpl") is not None) else None
        elif backend == "hybrid":
            extra["walk_cmd"] = tuple(v)
        be.step(arm, waist_yaw=wy, bow=bw, loco_cmd=tuple(v), **extra)
        x, y, yaw = be.root_pose()
        L["lim"].append(np.asarray(v, float).copy())
        L["xy"].append([x, y])
        L["yaw"].append(yaw)
        L["tilt"].append(be.tilt_deg())
        L["err"].append(float(np.degrees(np.abs(be.arm_q_now() - arm)).mean()))
        q = be.q_now()
        L["feet"].append(np.array([d.xpos[b] for b in feet]))
        L["on"].append(foot_contacts(m, d, feet, floor))
        L["knee"].append(q[knees])
        w_, x_, y_, z_ = d.xquat[be.pelvis_body]
        L["rp"].append((float(np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_))),
                        float(np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1, 1)))))
        L["z"].append(be.pelvis_height())
        if fell is None and be.fallen():
            fell = round(t, 2)
            break
    walk = getattr(be, "walk", True)
    be.close()
    A = {k: np.asarray(vv, bool if k == "on" else float) for k, vv in L.items()}
    yw = np.unwrap(A["yaw"])
    # integral of the limited command in the world (body-frame velocity rotated by the robot's heading)
    c, s = np.cos(yw), np.sin(yw)
    cmd_w = np.stack([c * A["lim"][:, 0] - s * A["lim"][:, 1], s * A["lim"][:, 0] + c * A["lim"][:, 1]], 1)
    cmd_disp = cmd_w.sum(0) * DT
    disp = A["xy"][-1] - A["xy"][0]
    moving = np.linalg.norm(A["lim"][:, :2], axis=1) > 0.05
    rmov = robot_moving(A["xy"])
    gait = gait_metrics(A, rmov)
    return {"seg": name, "backend": backend, "perception": "gemx" if gemx_root else "mediapipe", "profile": profile,
            "arm_mode": arm_mode, "walk_enabled": bool(walk), "fell": fell is not None, "fell_at": fell,
            "seconds": round(len(A["yaw"]) * DT, 1), "max_tilt_deg": round(float(A["tilt"].max()), 1),
            "moving_frac": round(float(moving.mean()), 2),
            "cmd_disp_m": [round(float(x), 2) for x in cmd_disp], "robot_disp_m": [round(float(x), 2) for x in disp],
            "cmd_turn_deg": round(float(np.degrees(A["lim"][:, 2].sum() * DT)), 1),
            "robot_turn_deg": round(float(np.degrees(yw[-1] - yw[0])), 1),
            "disp_ratio": round(float(np.dot(disp, cmd_disp) / max(np.dot(cmd_disp, cmd_disp), 1e-6)), 2)
            if np.linalg.norm(cmd_disp) > 0.1 else None,
            "arm_err_deg": round(float(A["err"].mean()), 2),
            "robot_moving_s": round(float(rmov.sum() * DT), 1),
            "robot_path_m": round(float(np.linalg.norm(np.diff(A["xy"], axis=0), axis=1).sum()), 2),
            "gait": gait}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", action="append", required=True, help="name=path@t0-t1")
    ap.add_argument("--variant", default="", help="sonic / hybrid walking deploy dir")
    ap.add_argument("--human-variant", default="", help="sonic_human deploy dir (SMPL encoder)")
    ap.add_argument("--backends", nargs="*", default=["split", "sonic"],
                    choices=["split", "sonic", "hybrid", "sonic_human"])
    ap.add_argument("--arm-ik", default="shape18", choices=["shape18", "dir"], help="sender arm IK (event: dir)")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--gemx-root", default="", help="GEM-X instead of MediaPipe: folder of every-frame real-time worker "
                    "dumps <video stem>.npz (live timing, JumpGate, HumanMapper(perception=gemx), GemxLocoEstimator)")
    ap.add_argument("--profile", default="visitor", help="walking-intent profile (E's table: visitor; event: presenter)")
    ap.add_argument("--arm-mode", default="student", choices=["student", "pd"], help="split / hybrid arms (event: pd)")
    ap.add_argument("--arm-vmax", type=float, default=ARM_VMAX, help="arm target speed limit rad/s (event: 12)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = []
    for spec in a.seg:
        for be in a.backends:
            r = run(spec, be, a.variant or None, a.mirror, a.human_variant or None, a.arm_ik, a.gemx_root or None,
                    a.profile, a.arm_mode, a.arm_vmax)
            out.append(r)
            print(json.dumps(r), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
