"""Clean presentation clips of calm event gestures (wave, bow, welcome, clap, point, ...).

Robot-motion encoder (mode 0) tracking the hand-authored references in sonic/gestures.py, or any
reference loadable by run_reference.py (--motion). Same deploy loop/timeline as M1 (INIT on the
elastic band, release, settle); only the gesture segment is recorded, front camera, neutral scene,
one small caption.

  .venv/bin/python scripts/render_event.py --clips wave_hello bow welcome_open_arms clap point_guide
  # showreel parts: 1080p, one camera for every clip, no burned-in caption
  .venv/bin/python scripts/render_event.py --clips bow --size 1920x1080 --camera 150,2.9,0.82,-6 --no-caption \
      --out-dir videos/showreel_parts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic.deploy import SonicController  # noqa: E402
from sonic.gestures import CLIPS  # noqa: E402
from sonic.params import CONTROL_DT, ROOT  # noqa: E402
from sonic.runner import EpisodeLog, control_tick, tracking_metrics  # noqa: E402
from sonic.sim import G1Sim, event_scene_path  # noqa: E402

TITLES = {
    "wave_hello": "Waving hello", "bow": "Polite bow", "welcome_open_arms": "Welcome - opening both arms",
    "clap": "Clapping", "point_guide": "Guiding with an open hand", "raise_both_arms": "Raising both arms",
    "heart": "Heart gesture", "weight_shift": "Standing - small weight shift",
}


# per-clip camera (azimuth, distance, lookat height): a bow reads best from the side-front,
# overhead gestures need a taller frame, pointing is shown from the other side
CAMERA = {"bow": (118.0, 2.15, 0.74), "point_guide": (205.0, 2.3, 0.78),
          "raise_both_arms": (160.0, 2.55, 0.88), "heart": (165.0, 2.5, 0.88)}
DEFAULT_CAMERA = (158.0, 2.15, 0.74)


def parse_size(s):
    w, h = (int(v) for v in s.lower().split("x"))
    return w, h


def parse_camera(s):
    """'azimuth,distance,lookat_z[,elevation]' -> tuple (the same camera for every clip)."""
    v = tuple(float(x) for x in s.split(","))
    if len(v) not in (3, 4):
        raise ValueError(f"--camera needs 3 or 4 numbers, got {s!r}")
    return v


class CleanCam:
    def __init__(self, sim, w=1280, h=720, distance=2.15, azimuth=158.0, elevation=-7.0, lookat_z=0.74):
        self.sim = sim
        self.r = mujoco.Renderer(sim.model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.lookat_z = lookat_z
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        self.xy = None

    def grab(self, caption=""):
        p = self.sim.data.xpos[self.sim.pelvis]
        xy = p[:2].copy()
        self.xy = xy if self.xy is None else 0.95 * self.xy + 0.05 * xy  # slow follow, no shake
        self.cam.lookat[:] = [self.xy[0], self.xy[1], self.lookat_z]
        self.r.update_scene(self.sim.data, camera=self.cam, scene_option=self.opt)
        img = self.r.render().copy()
        if caption:
            h = img.shape[0]
            k = h / 720.0
            cv2.putText(img, caption, (int(24 * k), h - int(24 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (60, 60, 70),
                        max(2, int(round(2 * k))), cv2.LINE_AA)
        return img


def run_clip(name, motion, variant, out_dir, fps=25, pre=1.0, post=1.5, size=(1280, 720), camera=None,
             caption=True, quality=8):
    """camera: None = per-clip CAMERA table, else (azimuth, distance, lookat_z[, elevation]) for this clip."""
    sim = G1Sim(event_scene_path(), elastic_band=True)
    ctrl = SonicController(variant)
    ctrl.set_motion(motion, play=False)
    cam_args = CAMERA.get(name, DEFAULT_CAMERA) if camera is None else camera
    az, dist, lz = cam_args[:3]
    elev = cam_args[3] if len(cam_args) > 3 else -7.0
    cam = CleanCam(sim, w=size[0], h=size[1], azimuth=az, distance=dist, lookat_z=lz, elevation=elev)
    every = max(1, int(round(1.0 / (fps * CONTROL_DT))))
    log = EpisodeLog()
    import imageio.v2 as imageio

    out = Path(out_dir) / f"event_{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, quality=quality, macro_block_size=1)  # streamed: 1080p frames are big
    n_frames = 0
    label = f"NVIDIA GEAR-SONIC on Unitree G1 (MuJoCo sim)  |  {TITLES.get(name, name)}" if caption else ""
    t_ctrl0 = t_drop = t_play = t_end = None
    fell, min_z, max_tilt, tick = None, 9.0, 0.0, 0
    while True:
        now = sim.time
        if ctrl.state == "WAIT_FOR_CONTROL":
            ctrl.start_control()
            t_ctrl0 = now
        if t_ctrl0 is not None and t_drop is None and now - t_ctrl0 >= 2.0:
            sim.set_band(False)
            t_drop = now
        if t_drop is not None and t_play is None and now - t_drop >= 2.5:
            ctrl.play = True
            t_play = now
        if t_play is not None and t_end is None and ctrl.motion_completed:
            t_end = now
        control_tick(ctrl, sim, log)
        if t_drop is not None and now - t_drop > 0.5:
            z, tilt = sim.pelvis_height(), sim.tilt_deg()
            min_z, max_tilt = min(min_z, z), max(max_tilt, tilt)
            if (z < 0.3 or tilt > 60) and fell is None:
                fell = now
        recording = t_drop is not None and now - t_drop >= 2.5 - pre
        if recording and tick % every == 0:
            writer.append_data(cam.grab(label))
            n_frames += 1
        tick += 1
        if t_end is not None and now - t_end >= post:
            break
    writer.close()
    met = tracking_metrics(log)
    res = {"clip": name, "variant": variant, "mode": "robot-motion encoder (mode 0)",
           "source": "hand-authored keyframes (sonic/gestures.py)" if name in CLIPS else "reference",
           "duration_s": motion.timesteps * CONTROL_DT, "fell": fell is not None,
           "min_pelvis_z": min_z, "max_pelvis_tilt_deg": max_tilt,
           "tracking_joint_err_deg": {k: round(v, 2) for k, v in met.items() if k.endswith("deg")},
           "video": {"size": list(size), "fps": fps, "frames": n_frames, "pre_s": pre, "post_s": post,
                     "camera": {"azimuth": az, "distance": dist, "lookat_z": lz, "elevation": elev}}}
    (Path(out_dir) / f"event_{name}.json").write_text(json.dumps(res, indent=2))
    return res, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["wave_hello", "bow", "welcome_open_arms", "clap", "point_guide"])
    ap.add_argument("--variant", default="release")
    ap.add_argument("--out-dir", default=str(ROOT / "videos"))
    # display options (defaults reproduce the original 1280x720 per-clip-camera clips)
    ap.add_argument("--size", default="1280x720", help="WxH, e.g. 1920x1080")
    ap.add_argument("--camera", default="", help="azimuth,distance,lookat_z[,elevation] used for EVERY clip "
                    "(default: per-clip camera table)")
    ap.add_argument("--no-caption", action="store_true", help="no burned-in caption (showreel adds its own)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--pre", type=float, default=1.0, help="s recorded before the gesture starts")
    ap.add_argument("--post", type=float, default=1.5, help="s recorded after the gesture ends")
    ap.add_argument("--quality", type=float, default=8, help="imageio/ffmpeg quality 0-10")
    args = ap.parse_args()
    camera = parse_camera(args.camera) if args.camera else None
    for name in args.clips:
        motion = CLIPS[name]()
        res, out = run_clip(name, motion, args.variant, args.out_dir, fps=args.fps, pre=args.pre, post=args.post,
                            size=parse_size(args.size), camera=camera, caption=not args.no_caption,
                            quality=args.quality)
        t = res["tracking_joint_err_deg"]
        print(f"{name:20s} fell={res['fell']} minZ={res['min_pelvis_z']:.2f} tilt={res['max_pelvis_tilt_deg']:.1f} "
              f"joint_err={t.get('all_deg')}deg arms={t.get('arms_deg')} legs={t.get('legs_deg')} -> {out}")


if __name__ == "__main__":
    main()
