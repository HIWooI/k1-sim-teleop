"""K1 event gestures in MuJoCo physics -> videos/k1_event_<name>.mp4 + .json (metrics).

Timeline per clip: split-mode entry (loco standing pose), `settle` s holding the gesture's first reference frame,
the gesture, `post` s holding its last frame. Recorded from `pre` s before the gesture, one fixed camera.

  .venv/bin/python -m k1.render_event                                  # all 8, 1920x1080, no caption
  .venv/bin/python -m k1.render_event --clips wave_hello clap --no-video  # metrics only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from k1.gestures import GESTURES, ArmIK
from k1.sim import FPS, K1Split, Camera

ROOT = Path(__file__).resolve().parents[1]
CAMERA = (162.0, 2.4, 0.76, -6.0)  # azimuth, distance, lookat z, elevation: 18 deg off the front, room for arms up


def run_clip(name, out_dir, video=True, size=(1920, 1080), camera=CAMERA, settle=2.0, pre=0.5, post=0.8, fps=25,
             quality=9, ik=None):
    t_build = time.perf_counter()
    ik = ik or ArmIK(K1Split())
    motion = GESTURES[name](ik)
    sim = K1Split(mode=motion.mode)
    sim.reset(motion.torso_quat[0], q0=motion.q[0])
    n_settle, n_post = int(settle * FPS), int(post * FPS)
    total = n_settle + motion.T + n_post
    every = max(1, int(round(FPS / fps)))
    writer = cam = None
    if video:
        import imageio.v2 as imageio

        out = Path(out_dir) / f"k1_event_{name}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out, fps=fps, quality=quality, macro_block_size=1)
        cam = Camera(sim, w=size[0], h=size[1], azimuth=camera[0], distance=camera[1], lookat_z=camera[2],
                     elevation=camera[3])
    rec_from = n_settle - int(pre * FPS)
    qs, refs = [], []
    fell, min_z, max_tilt, frames = None, 9.0, 0.0, 0
    for k in range(total):
        i = min(max(k - n_settle, 0), motion.T - 1)
        sim.step(motion.q[i], motion.dq[i] if 0 <= k - n_settle < motion.T else np.zeros(23), motion.torso_quat[i])
        if k >= n_settle:
            qs.append(sim.q)
            refs.append(motion.q[i])
        if k > 25:
            z, tilt = sim.pelvis_height(), sim.tilt_deg()
            min_z, max_tilt = min(min_z, z), max(max_tilt, tilt)
            if fell is None and (z < 0.35 or tilt > 60):
                fell = k / FPS
        if writer is not None and k >= rec_from and (k - rec_from) % every == 0:
            writer.append_data(cam.grab())
            frames += 1
    if writer is not None:
        writer.close()
    qs, refs = np.array(qs), np.array(refs)
    err = np.degrees(np.abs(qs - refs))
    arms = ~sim.lower
    # the policy follows the reference with a small delay: also report the best 0-10 frame lag (arms)
    lag_err = []
    for lag in range(0, 11):
        e = np.degrees(np.abs(qs[lag:, arms] - refs[:len(refs) - lag, arms]))
        lag_err.append(e.mean())
    best = int(np.argmin(lag_err))
    res = {
        "clip": name, "robot": "ROBOTIS AI Sapiens K1 (23 DOF), MuJoCo",
        "mode": "split: arms=mimic student, legs+waist=loco_armdriven_r2" if motion.mode == "split"
        else "whole-body mimic student (needs hip pitch)",
        "source": "IK-authored keyframes (k1/gestures.py)", "duration_s": motion.T / FPS,
        "fell": fell is not None, "fell_at_s": fell, "min_pelvis_z": round(min_z, 3), "max_pelvis_tilt_deg": round(max_tilt, 1),
        "tracking_err_deg": {"arms": round(float(err[:, arms].mean()), 2), "arms_p90": round(float(np.percentile(err[:, arms], 90)), 2),
                             "legs_waist": round(float(err[:, sim.lower].mean()), 2),
                             "arms_best_lag": round(float(lag_err[best]), 2), "best_lag_frames": best},
        "per_joint_err_deg": {n.replace("_joint", ""): round(float(v), 1) for n, v in zip(sim.names, err.mean(0))},
        "video": {"size": list(size), "fps": fps, "frames": frames, "settle_s": settle, "pre_s": pre, "post_s": post,
                  "camera": dict(zip(("azimuth", "distance", "lookat_z", "elevation"), camera))} if video else None,
        "wall_s": round(time.perf_counter() - t_build, 1),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"k1_event_{name}.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=list(GESTURES))
    ap.add_argument("--out-dir", default=str(ROOT / "videos"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--camera", default=",".join(str(v) for v in CAMERA), help="azimuth,distance,lookat_z,elevation")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    camera = tuple(float(v) for v in args.camera.split(","))
    ik = ArmIK(K1Split())
    for name in args.clips:
        r = run_clip(name, args.out_dir, video=not args.no_video, size=size, camera=camera, ik=ik)
        t = r["tracking_err_deg"]
        print(f"{name:18s} fell={r['fell']} minZ={r['min_pelvis_z']:.3f} tilt={r['max_pelvis_tilt_deg']:5.1f} "
              f"arms={t['arms']:5.2f} (lag {t['best_lag_frames']}: {t['arms_best_lag']:5.2f}) legs={t['legs_waist']:5.2f} "
              f"[{r['mode'][:5]}] {r['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
