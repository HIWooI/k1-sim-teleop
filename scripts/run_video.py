"""M3 offline: video file -> MediaPipe -> SMPL-24 -> streamed SMPL encoder -> G1 (headless).

Exercises exactly the live code path (MediaPipeToSmpl + StreamBuffer cursor/lag + controller),
but in simulated time so it is deterministic and can be recorded side-by-side.

  .venv/bin/python scripts/run_video.py --video videos/gemx_guap/Guap_ver2.mp4 \
      --crop 0.28,0,0.72,1 --out videos/m3_guap_mediapipe.mp4
  # showreel part: 1920x1080, two equal panels (human cropped around the person | robot), no text
  .venv/bin/python scripts/run_video.py --video in.mp4 --num-poses 1 --clean --clean-size 1920x1080 --no-caption \
      --out videos/showreel_parts/teleop_x.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sonic.deploy import SonicController, parse_gain_specs  # noqa: E402
from sonic.metrics import LimbTracker  # noqa: E402
from sonic.motion import standing_motion  # noqa: E402
from sonic.mp2smpl import MediaPipeToSmpl, OneEuro  # noqa: E402
from sonic.params import CONTROL_DT, ROOT, SCENE_XML  # noqa: E402
from sonic.pose_mp import PoseTracker, draw_overlay  # noqa: E402
from sonic.runner import control_tick  # noqa: E402
from sonic.sim import G1Sim, Recorder, event_scene_path  # noqa: E402
from sonic.stream import StreamBuffer, attach, stream_step  # noqa: E402

WINDOW = {"release": 10, "sonic_v1_1": 10, "low_latency": 4}


class FrameReader:
    """Sequential on-demand frame access (no full decode into RAM — some sources are 4K)."""

    def __init__(self, path, height=720, fit=None):
        self.path, self.height, self.fit = str(path), height, fit
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.path)
        self.idx, self.frame = -1, None

    def get(self, k):
        if k < self.idx:
            self._open()
        while self.idx < k:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx += 1
            h = self.height
            if self.fit is not None:
                self.frame = self.fit(f)
            else:
                self.frame = cv2.resize(f, (int(round(f.shape[1] * h / f.shape[0] / 2)) * 2, h),
                                        interpolation=cv2.INTER_AREA)
        return self.frame


def person_span(I, V, min_vis=0.5):
    """Horizontal extent (normalised image x) the tracked person uses over the clip: robust min/max of the
    visible upper-body + hip landmarks (shoulders, elbows, wrists, hands, hips, nose)."""
    idx = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24]
    x = I[:, idx, 0]
    ok = (V[:, idx] > min_vis) & ~np.isnan(x)
    if not ok.any():
        return 0.25, 0.75
    xs = x[ok]
    return float(np.percentile(xs, 1)), float(np.percentile(xs, 99))


class PanelFit:
    """Fit a source frame into a fixed (pw, ph) panel without distortion and without panning: crop a static
    window around the person when the source is wider than the panel, otherwise (or if the person would not fit)
    letterbox on a blurred, lightened copy of the frame."""

    def __init__(self, pw, ph, span, margin=0.06):
        self.pw, self.ph = pw, ph
        self.x0, self.x1 = span
        self.margin = margin

    def __call__(self, f):
        pw, ph = self.pw, self.ph
        h, w = f.shape[:2]
        s = ph / h
        need = (self.x1 - self.x0 + 2 * self.margin) * w * s  # px the person needs at full panel height
        if need > pw:
            s *= pw / need
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if nw >= pw and nh >= ph:
            cx = 0.5 * (self.x0 + self.x1) * nw
            xa = int(np.clip(round(cx - pw / 2), 0, nw - pw))
            ya = (nh - ph) // 2
            return img[ya:ya + ph, xa:xa + pw]
        bg = cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(bg, (0, 0), 30)
        bg = cv2.addWeighted(bg, 0.55, np.full_like(bg, 245), 0.45, 0)
        if nw > pw:  # wider than the panel even after scaling: crop around the person
            cx = 0.5 * (self.x0 + self.x1) * nw
            xa = int(np.clip(round(cx - pw / 2), 0, nw - pw))
            img, nw = img[:, xa:xa + pw], pw
        ya, xa = (ph - nh) // 2, (pw - nw) // 2
        bg[ya:ya + nh, xa:xa + nw] = img
        return bg


def _downscale(f, max_side):
    s = max(f.shape[:2])
    if max_side and s > max_side:
        sc = max_side / s
        f = cv2.resize(f, (int(f.shape[1] * sc), int(f.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return f


def extract(video, model, crop, num_poses, max_side=1280):
    st = Path(video).stat()  # size + mtime: a replaced file with the same name must not reuse old landmarks
    key = hashlib.md5(f"{Path(video).resolve()}|{st.st_size}|{int(st.st_mtime)}|{model}|{crop}|{num_poses}|{max_side}"
                      .encode()).hexdigest()[:10]
    cache = ROOT / "logs" / f"mp_{Path(video).stem}_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        if z["world"].ndim == 3 and len(z["world"]):
            return z["world"], z["vis"], z["image"], float(z["fps"])
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tr = PoseTracker(model, num_poses=num_poses, crop=crop)
    W, V, I = [], [], []
    n = 0
    t0 = time.perf_counter()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        d = tr.process(_downscale(f, max_side), n / fps)
        n += 1
        if d is None:
            W.append(np.full((33, 3), np.nan)); V.append(np.zeros(33)); I.append(np.full((33, 3), np.nan))
        else:
            W.append(d["world"]); V.append(d["vis"]); I.append(d["image"])
    print(f"[mediapipe] {n} frames, {1000 * (time.perf_counter() - t0) / max(n, 1):.1f} ms/frame")
    W, V, I = np.array(W), np.array(V), np.array(I)
    if n == 0:
        raise SystemExit(f"cannot decode any frame from {video} (incomplete download?)")
    if np.isnan(W[:, 0, 0]).all():
        raise SystemExit(f"MediaPipe found no person in {video}")
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, world=W, vis=V, image=I, fps=fps)
    return W, V, I, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--crop", default="", help="x0,y0,x1,y1 normalised crop (multi-person clips)")
    ap.add_argument("--num-poses", type=int, default=2)
    ap.add_argument("--variant", default="release", choices=list(WINDOW))
    ap.add_argument("--pelvis", default="j0", choices=["zero", "j0"])
    ap.add_argument("--root-pitch-share", type=float, default=0.5)
    ap.add_argument("--legs", default="auto", choices=["auto", "track", "lock"])
    ap.add_argument("--arm-depth-scale", type=float, default=1.0,
                    help="<1 shrinks MediaPipe's arm depth (arms reaching forward make SONIC step forward)")
    ap.add_argument("--max-root-tilt", type=float, default=40.0, help="safety cap on commanded root tilt (deg)")
    ap.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff (Hz)")
    ap.add_argument("--beta", type=float, default=0.4, help="One-Euro beta")
    ap.add_argument("--calib-frames", type=int, default=15)
    ap.add_argument("--delay", type=float, default=0.0, help="extra stream delay (s)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--tail", type=float, default=3.0, help="s to keep streaming the last pose")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--quality", type=float, default=7, help="imageio/ffmpeg quality 0-10 of --out")
    ap.add_argument("--clean", action="store_true", help="presentation layout: event scene, front camera, "
                    "human video left (synced to the robot's commanded frame), one small caption")
    ap.add_argument("--clean-size", default="", help="with --clean: output WxH (e.g. 1920x1080) split into two equal "
                    "panels; the human video is cropped/letterboxed around the person (default: legacy 720p layout)")
    ap.add_argument("--no-caption", action="store_true", help="with --clean: no pipeline caption bar")
    ap.add_argument("--skeleton", action="store_true", help="draw the MediaPipe skeleton on the human video")
    ap.add_argument("--max-side", type=int, default=1280, help="downscale frames before MediaPipe")
    ap.add_argument("--max-video-seconds", type=float, default=0.0, help="use only the first N s of the clip")
    ap.add_argument("--motor-kp-scale", action="append", default=[], help='e.g. "0-11=1.2" (hardware motor idx)')
    ap.add_argument("--motor-kd-scale", action="append", default=[])
    args = ap.parse_args()

    crop = tuple(float(x) for x in args.crop.split(",")) if args.crop else None
    W, V, I, vfps = extract(args.video, args.model, crop, args.num_poses, args.max_side)
    if args.max_video_seconds > 0:
        n = int(args.max_video_seconds * vfps)
        W, V, I = W[:n], V[:n], I[:n]
    T = len(W)
    # hold last valid frame where MediaPipe lost the person
    valid = ~np.isnan(W[:, 0, 0])
    first = int(np.argmax(valid))
    for t in range(T):
        if not valid[t]:
            src = t - 1 if t > first else first
            W[t], V[t], I[t] = W[src], V[t] * 0 + V[src], I[src]
    filt = OneEuro(args.min_cutoff, args.beta)
    Wf = np.array([filt(W[t], t / vfps) for t in range(T)])
    conv = MediaPipeToSmpl(root_pitch_share=args.root_pitch_share, pelvis_mode=args.pelvis, legs=args.legs,
                           arm_depth_scale=args.arm_depth_scale, max_root_tilt_deg=args.max_root_tilt)
    conv.calibrate(Wf[first:first + args.calib_frames])
    samples = []
    locked = 0
    for t in range(T):
        o = conv.convert(Wf[t], V[t])
        locked += conv.legs_locked
        samples.append((t / vfps, o["body_quat"], o["smpl_joints"]))
    print(f"[mp2smpl] {T} frames @ {vfps:.0f} fps, lost {int((~valid).sum())}, legs locked {locked}")

    sim = G1Sim(event_scene_path() if args.clean else SCENE_XML, elastic_band=True)
    ctrl = SonicController(args.variant, motor_kp_scale=parse_gain_specs(args.motor_kp_scale),
                          motor_kd_scale=parse_gain_specs(args.motor_kd_scale))
    ctrl.set_motion(standing_motion(), play=False)
    buf = StreamBuffer(window=WINDOW[args.variant], delay=args.delay, max_samples=T * args.loops + 16)
    limbs = LimbTracker(sim.model)
    rec = None
    fit = None
    if args.out:
        if args.clean:
            from render_event import CleanCam, parse_size

            if args.clean_size:
                W_, H_ = parse_size(args.clean_size)
                pw = W_ // 2
                fit = PanelFit(pw, H_, person_span(I, V))
                rec = CleanCam(sim, w=W_ - pw, h=H_, distance=2.35, azimuth=180.0, elevation=-5.0, lookat_z=0.80)
            else:
                rec = CleanCam(sim, w=720, h=720, distance=2.35, azimuth=180.0, elevation=-5.0, lookat_z=0.80)
        else:
            rec = Recorder(sim, azimuth=180.0, elevation=-10.0, distance=2.8)
    reader = FrameReader(args.video, height=720 if args.clean else 480, fit=fit) if args.out else None
    lag = (WINDOW[args.variant] - 1) * CONTROL_DT + args.delay
    every = max(1, int(round(1.0 / (args.fps * CONTROL_DT))))

    t_ctrl0 = t_drop = t_stream = None
    clip_len = T / vfps
    end_time = None
    fell_at = None
    min_z = 9.0
    max_tilt = 0.0
    writer, n_out = None, 0
    if args.out:
        import imageio.v2 as imageio

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # streamed to disk: a 1080p side-by-side clip does not fit comfortably in RAM
        writer = imageio.get_writer(args.out, fps=args.fps, quality=args.quality, macro_block_size=1)
    tick = 0
    wall0 = time.perf_counter()
    while True:
        now = sim.time
        if ctrl.state == "WAIT_FOR_CONTROL":
            ctrl.start_control()
            t_ctrl0 = now
        if t_ctrl0 is not None and t_drop is None and now - t_ctrl0 >= 2.0:
            sim.set_band(False)
            t_drop = now
        if t_drop is not None and t_stream is None and now - t_drop >= 1.0:
            t_stream = now
            end_time = t_stream + clip_len * args.loops + args.tail
            # samples are timestamped on the sim clock: video t=0 <-> t_stream
            for k in range(args.loops):
                for (ts, q, j) in samples:
                    buf.add_sample(t_stream + k * clip_len + ts, q, j)
            attach(ctrl, buf, now)
            print(f"[t={now:5.2f}] streaming ON (window={buf.window})")
        if t_stream is not None:
            stream_step(ctrl, buf, now)
        control_tick(ctrl, sim)
        if t_stream is not None:
            limbs.update(sim.data, ctrl.current_motion.smpl_joints[ctrl.current_frame])
        if t_drop is not None and now - t_drop > 0.5:
            z, tilt = sim.pelvis_height(), sim.tilt_deg()
            min_z, max_tilt = min(min_z, z), max(max_tilt, tilt)
            if (z < 0.3 or tilt > 60) and fell_at is None:
                fell_at = now
                print(f"[t={now:5.2f}] FALL (z={z:.2f}, tilt={tilt:.0f})")
        if rec is not None and args.clean and tick % every == 0 and t_stream is not None and now >= t_stream - 0.5:
            robot = rec.grab()
            tv = max(0.0, (now - t_stream - lag)) % clip_len  # human frame the robot is executing now
            k = min(int(tv * vfps), T - 1)
            hum = reader.get(k)
            if args.skeleton and not np.isnan(I[k, 0, 0]) and fit is None:
                hum = draw_overlay(hum, {"image": I[k], "vis": V[k]})
            img = np.concatenate([cv2.cvtColor(hum, cv2.COLOR_BGR2RGB), robot], axis=1)
            if not args.no_caption:
                h = img.shape[0]
                cv2.rectangle(img, (0, h - 34), (img.shape[1], h), (245, 245, 247), -1)
                cv2.putText(img, "human video  ->  MediaPipe Pose  ->  SMPL  ->  NVIDIA GEAR-SONIC  ->  Unitree G1 (MuJoCo sim)",
                            (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 80), 1, cv2.LINE_AA)
            writer.append_data(img)
            n_out += 1
        elif rec is not None and not args.clean and tick % every == 0:
            img = rec.grab()
            k = int(((now - t_stream) % clip_len) * vfps) if t_stream is not None else 0
            k = min(k, T - 1)
            det = None if np.isnan(I[k, 0, 0]) else {"image": I[k], "vis": V[k]}
            src = draw_overlay(reader.get(k), det, f"MediaPipe {args.model}")
            if src is not None:
                h = img.shape[0]
                src = cv2.resize(src, (int(src.shape[1] * h / src.shape[0]), h))
                img = np.concatenate([cv2.cvtColor(src, cv2.COLOR_BGR2RGB), img], axis=1)
            cv2.putText(img, f"t={now:5.1f}s SONIC {args.variant} smpl-mode {'LEGS LOCKED' if conv.legs_locked else ''}",
                        (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(img)
            n_out += 1
        tick += 1
        if end_time is not None and now >= end_time:
            break
    wall = time.perf_counter() - wall0
    res = {
        "video": str(args.video), "variant": args.variant, "pelvis": args.pelvis, "legs": args.legs,
        "root_pitch_share": args.root_pitch_share, "fell": fell_at is not None, "fell_at": fell_at,
        "min_pelvis_z": min_z, "max_tilt_deg": max_tilt, "lost_frames": int((~valid).sum()),
        "legs_locked_frames": int(locked),
        "limb_dir_err_deg": limbs.summary(skip=25), "stream_latency_ms": (buf.window - 1) * 20 + args.delay * 1000,
        "realtime_factor": sim.time / wall,
    }
    if args.out and args.clean:
        # clean videos start 0.5 s before streaming; the human panel shows the frame the robot executes now,
        # so human-video time t appears at output time t + human_t0_in_video_s
        res["video_out"] = {"path": str(args.out), "fps": args.fps, "frames": n_out, "size": args.clean_size or "legacy",
                            "human_t0_in_video_s": round(0.5 + lag, 3), "human_clip_s": round(clip_len, 3)}
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    if writer is not None:
        writer.close()
        print("video ->", args.out, n_out)


if __name__ == "__main__":
    main()
