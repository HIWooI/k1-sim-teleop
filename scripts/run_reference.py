"""M1: SONIC robot-motion encoder (mode 0) tracking a G1 reference motion, headless.

Timeline (mirrors the official sim2sim procedure in docs/tutorials):
  t=0       deploy INIT: 3 s ramp to default angles while hanging on the elastic band
  t=3       ']' start control (policy runs, reference paused at frame 0)
  t=3+drop  '9' release the elastic band (robot drops onto the floor)
  t=...     'T' play the reference motion; it ends -> back to frame 0 (paused), keep standing

Example:
  .venv/bin/python scripts/run_reference.py --motion macarena_001__A545 --video videos/m1_macarena.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic.deploy import SonicController, parse_gain_specs  # noqa: E402
from sonic.motion import list_csv_motions, load_csv_motion, load_robot_pkl, load_smpl_pkl  # noqa: E402
from sonic.params import CONTROL_DT, REFERENCE_DIR  # noqa: E402
from sonic.metrics import LimbTracker  # noqa: E402
from sonic.runner import EpisodeLog, best_lag, control_tick, place_on_ground, tracking_metrics  # noqa: E402
from sonic.sim import G1Sim, Recorder  # noqa: E402


def load_any(spec, pelvis="zero", smooth=0.0, pelvis_pkl="j0"):
    if spec == "standing":
        from sonic.motion import standing_motion

        return standing_motion()
    p = Path(spec)
    if p.suffix == ".bvh":  # GEM-X SOMA export (+ hpe_results.pt next to it)
        from sonic.gemx import load_gemx_clip

        return load_gemx_clip(p, p.parent / "hpe_results.pt", pelvis_mode=pelvis, smooth=smooth)
    if p.suffix == ".pkl":
        return load_robot_pkl(p) if "robot" in str(p) else load_smpl_pkl(p, pelvis_mode=pelvis_pkl)
    if not p.exists():
        p = REFERENCE_DIR / spec
    return load_csv_motion(p)


class Ghost:
    """Kinematic playback of the reference on a second MjData (same scene/camera)."""

    def __init__(self, sim):
        import mujoco

        from sonic.sim import Recorder

        self.mj = mujoco
        self.sim = sim
        self.data = mujoco.MjData(sim.model)
        self.rec = Recorder(sim)
        self.rec.sim = type("S", (), {"data": self.data})()

    def render(self, motion, f, adh, xy):
        from sonic import rot
        from sonic.params import ISAACLAB_TO_MUJOCO

        d = self.data
        d.qpos[:] = 0
        d.qpos[0:2] = xy
        d.qpos[2] = motion.body_pos[f, 0, 2]
        d.qpos[3:7] = rot.quat_mul(adh, motion.body_quat[f, 0])
        d.qpos[self.sim.qadr] = motion.joint_pos[f][ISAACLAB_TO_MUJOCO]
        self.mj.mj_forward(self.sim.model, d)
        self.rec.renderer.update_scene(d, camera=self.rec.cam)
        return self.rec.renderer.render().copy()


def load_side_video(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames, fps


def side_by_side(left, right, text=""):
    import cv2

    h = right.shape[0]
    w = int(left.shape[1] * h / left.shape[0])
    left = cv2.resize(left, (w, h))
    out = np.concatenate([left, right], axis=1)
    if text:
        cv2.putText(out, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compare_metrics(log, ref, lag=0):
    a = log.arrays()
    sel = a["playing"].astype(bool)
    q = a["q_isaac"][sel]
    fr = np.clip(a["frame"][sel] - lag, 0, ref.timesteps - 1)
    err = np.abs(q - ref.joint_pos[fr])
    groups = {"legs": [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], "waist": [2, 5, 8],
              "arms": [11, 12, 15, 16, 19, 20, 21, 22], "wrists": [23, 24, 25, 26, 27, 28]}
    out = {"all_deg": float(np.degrees(err.mean()))}
    for k, idx in groups.items():
        out[f"{k}_deg"] = float(np.degrees(err[:, idx].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="macarena_001__A545",
                    help="folder name under gear_sonic_deploy/reference/example, a path, or a smpl .pkl")
    ap.add_argument("--variant", default="release", choices=["release", "low_latency", "sonic_v1_1"])
    ap.add_argument("--start", default="band", choices=["band", "ground"],
                    help="band: official elastic-band drop; ground: spawn standing on the floor")
    ap.add_argument("--spawn-yaw", type=float, default=0.0, help="initial robot yaw (rad) for --start ground")
    ap.add_argument("--drop-after", type=float, default=2.0, help="s after control start to release band")
    ap.add_argument("--play-after", type=float, default=2.0, help="s after band release to press T")
    ap.add_argument("--min-duration", type=float, default=20.0, help="total CONTROL time (s)")
    ap.add_argument("--loops", type=int, default=1, help="play the motion this many times")
    ap.add_argument("--video", default="")
    ap.add_argument("--ghost", action="store_true", help="render the kinematic reference next to the robot")
    ap.add_argument("--side-video", default="", help="source clip shown left of the robot, synced to playback")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--json", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pelvis", default="zero", choices=["zero", "j0"], help="SMPL pelvis convention for .bvh input")
    ap.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing for .bvh input (soma_to_smpl --smooth)")
    ap.add_argument("--compare", default="", help="robot motion (csv dir / robot pkl) to score SMPL tracking against")
    ap.add_argument("--motor-kp-scale", action="append", default=[], help='e.g. "0-11=1.2" (hardware motor idx)')
    ap.add_argument("--motor-kd-scale", action="append", default=[])
    args = ap.parse_args()
    args.pelvis_set = any(a.startswith("--pelvis") for a in sys.argv)

    if args.list:
        for p in list_csv_motions(REFERENCE_DIR):
            print(p.name)
        return

    motion = load_any(args.motion, args.pelvis, args.smooth, args.pelvis if args.pelvis_set else "j0")
    compare = load_any(args.compare) if args.compare else None
    print(f"motion {motion.name}: {motion.timesteps} frames ({motion.timesteps * CONTROL_DT:.1f}s), "
          f"encode_mode={motion.encode_mode}")

    sim = G1Sim(elastic_band=True)
    if args.start == "ground":
        # stand on the floor; the band holds the pelvis at its standing height during INIT
        place_on_ground(sim, yaw=args.spawn_yaw)
    ctrl = SonicController(args.variant, motor_kp_scale=parse_gain_specs(args.motor_kp_scale),
                          motor_kd_scale=parse_gain_specs(args.motor_kd_scale))
    ctrl.set_motion(motion, play=False)
    rec = Recorder(sim) if args.video else None
    side = load_side_video(args.side_video) if (args.video and args.side_video) else None
    out_frames = []
    ghost_motion = motion if motion.encode_mode == 0 else compare
    ghost = Ghost(sim) if (args.video and args.ghost and ghost_motion is not None) else None
    every = max(1, int(round(1.0 / (args.fps * CONTROL_DT))))

    log = EpisodeLog()
    limbs = LimbTracker(sim.model) if motion.encode_mode == 2 else None
    tick = 0
    t_ctrl0 = None
    t_drop = None
    plays = 0
    min_z_after_drop = 9.0
    min_z_standing = 9.0
    max_tilt = 0.0
    fell_at = None
    wall0 = time.perf_counter()
    while True:
        if ctrl.state == "WAIT_FOR_CONTROL":
            ctrl.start_control()
            t_ctrl0 = sim.time
            print(f"[t={sim.time:5.2f}] start control")
        if t_ctrl0 is not None:
            tc = sim.time - t_ctrl0
            if t_drop is None and tc >= args.drop_after:
                sim.set_band(False)
                t_drop = sim.time
                print(f"[t={sim.time:5.2f}] release elastic band")
            if t_drop is not None and plays < args.loops and not ctrl.play and sim.time - t_drop >= args.play_after:
                if plays == 0 or ctrl.motion_completed:
                    ctrl.motion_completed = False
                    ctrl.play = True
                    plays += 1
                    print(f"[t={sim.time:5.2f}] play #{plays}")
            if t_drop is not None and sim.time - t_drop > 0.5:
                z = sim.pelvis_height()
                tilt = sim.tilt_deg()
                max_tilt = max(max_tilt, tilt)
                min_z_after_drop = min(min_z_after_drop, z)
                if not ctrl.play:
                    min_z_standing = min(min_z_standing, z)
                if (z < 0.3 or tilt > 60) and fell_at is None:
                    fell_at = sim.time
                    print(f"[t={sim.time:5.2f}] FALL detected (pelvis z={z:.3f}, tilt={tilt:.0f}deg)")
            done_playing = plays >= args.loops and not ctrl.play
            if done_playing and tc >= args.min_duration:
                break
            if tc > args.min_duration + motion.timesteps * CONTROL_DT * args.loops + 30:
                break
        f_used, playing = ctrl.current_frame, ctrl.play
        control_tick(ctrl, sim, log)
        if limbs is not None and playing:
            limbs.update(sim.data, motion.smpl_joints[f_used])
        if rec is not None and tick % every == 0:
            img = rec.grab()
            if ghost is not None:
                f = min(ctrl.current_frame, ghost_motion.timesteps - 1)
                adh = ctrl.apply_delta_heading()
                if ghost_motion is not motion:  # align the G1 retarget's heading to the SMPL root's
                    from sonic import rot

                    adh = rot.quat_mul(adh, rot.quat_mul(rot.calc_heading_quat(motion.body_quat[0, 0]),
                                                         rot.calc_heading_quat_inv(ghost_motion.body_quat[0, 0])))
                gimg = ghost.render(ghost_motion, f, adh, sim.data.qpos[0:2])
                label = "G1 retarget (not seen by policy)" if ghost_motion is not motion else "reference (kinematic)"
                img = side_by_side(gimg, img, f"left: {label}  right: SONIC mode {motion.encode_mode}  t={sim.time:4.1f}s")
            if side is not None:
                frames, sfps = side
                k = int(ctrl.current_frame * CONTROL_DT * sfps) if ctrl.play else 0
                img = side_by_side(frames[min(k, len(frames) - 1)], img,
                                   f"t={sim.time:5.1f}s  frame={ctrl.current_frame}  mode={motion.encode_mode}")
            out_frames.append(img)
        tick += 1
    wall = time.perf_counter() - wall0

    lag = best_lag(log)
    met = tracking_metrics(log, 0)
    met_lag = tracking_metrics(log, lag)
    a = log.arrays()
    res = {
        "motion": motion.name, "variant": args.variant, "start": args.start,
        "control_time_s": float(sim.time - t_ctrl0),
        "min_pelvis_z_after_drop": float(min_z_after_drop),
        "min_pelvis_z_while_standing": float(min_z_standing),
        "max_pelvis_tilt_deg": float(max_tilt),
        "fell": fell_at is not None, "fell_at_s": fell_at,
        "tracking_lag0": met, "best_lag_frames": int(lag), "tracking_best_lag": met_lag,
        "policy_tick_ms_mean": float(np.mean(a["tick_ms"])), "policy_tick_ms_p95": float(np.percentile(a["tick_ms"], 95)),
        "wall_s": wall, "realtime_factor": float(sim.time / wall),
    }
    if limbs is not None:
        res["limb_dir_err_deg"] = limbs.summary(skip=25)
    if compare is not None:
        best = min(range(0, 16), key=lambda L: compare_metrics(log, compare, L)["all_deg"])
        res["vs_compare_lag0"] = compare_metrics(log, compare, 0)
        res["vs_compare_best_lag"] = {"lag_frames": best, **compare_metrics(log, compare, best)}
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
    if rec is not None:
        import imageio.v2 as imageio

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, out_frames, fps=args.fps, quality=7, macro_block_size=1)
        print("video ->", args.video, len(out_frames), "frames")


if __name__ == "__main__":
    main()
