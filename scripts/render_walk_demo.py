"""Demo reel: person video | K1 in MuJoCo, the live chain run offline on real clips (heavy, EGL).

Per segment (the robot respawns at the origin): the clip plays at its own frame rate as the fake camera ->
MediaPipe landmarks (scripts/run_video.py cache) -> k1live.human.HumanMapper (arm IK, waist) and
k1live.locomotion.LocoEstimator (walking intent; home = its own auto-home once the person stands still) -> 50 Hz:
arm targets slew-limited to 6 rad/s, walking command through VelocityLimiter.for_backend(backend) -> backend.step().
Mirror is off in the reel (person and robot side by side, both facing the viewer: same side = same arm / direction);
the live default (mirror on) only swaps left/right.

  .venv/bin/python scripts/render_walk_demo.py --out videos/k1_live_walk_split.mp4 \
      --seg "toward=<path>@0-21|카메라 쪽으로 걸어오면 로봇이 앞으로 걷습니다" --seg ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DT = 0.02
ARM_VMAX = 6.0


def cmd_text(v):
    vx, vy, wz = v
    parts = []
    if abs(vx) > 0.02:
        parts.append(f"{'전진' if vx > 0 else '후진'} {abs(vx):.2f} m/s")
    if abs(vy) > 0.02:
        parts.append(f"{'왼쪽' if vy > 0 else '오른쪽'} {abs(vy):.2f} m/s")
    if abs(wz) > 0.02:
        parts.append(f"{'왼쪽' if wz > 0 else '오른쪽'} 회전 {abs(wz):.1f} rad/s")
    return "보행: " + (" · ".join(parts) if parts else "정지")


def segment(spec, backend, size, fps_out, writer, overlays, mirror=False):
    import cv2
    import run_video as g1rv
    from PIL import Image

    import make_showreel as ms
    from k1.sim import Camera
    from k1live.backends.base import make_backend
    from k1live.human import HumanMapper
    from k1live.locomotion import LocoEstimator, VelocityLimiter
    from sonic.pose_mp import draw_overlay

    name, rest = spec.split("=", 1)
    rest, _, caption = rest.partition("|")
    path, _, rng = rest.partition("@")
    a, _, b = rng.partition("-")
    t0, t1 = float(a or 0), float(b) if b else None
    W, V, I, vfps = g1rv.extract(path, "full", None, 1, 1280)
    cap = cv2.VideoCapture(path)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    k0 = int(t0 * vfps)
    k1 = len(W) if t1 is None else min(len(W), int(t1 * vfps))
    W_, H_ = size
    pw = W_ // 2
    be = make_backend(backend)
    lim = VelocityLimiter.for_backend(backend)
    hm = HumanMapper(mirror=mirror, loco=False)
    est = LocoEstimator(mirror=mirror)
    fit = g1rv.PanelFit(pw, H_, g1rv.person_span(I[k0:k1], V[k0:k1]))
    reader = g1rv.FrameReader(path, height=H_)
    if hasattr(be, "sim") and hasattr(be.sim, "lo"):        # split: k1.sim.K1Split presentation camera
        cam = Camera(be.sim, w=W_ - pw, h=H_, azimuth=155.0, distance=2.9, elevation=-10.0, lookat_z=0.62)
    else:
        from sonic.sim import Recorder

        cam = Recorder(be.sim, width=W_ - pw, height=H_, distance=2.9, azimuth=155.0, elevation=-10.0)
    every = max(1, int(round(1.0 / (fps_out * DT))))
    arm = be.arm_default.copy()
    last, k_last = None, -1
    v = np.zeros(3)
    log = {"t": [], "cmd": [], "lim": [], "xy": [], "yaw": [], "tilt": []}
    fell = None
    n = int((k1 - k0) / vfps / DT)
    sc = W_ / 1920.0
    cap_ov = ms.Overlay(ms.pill([(caption, ms.font("semibold", int(34 * sc)), ms.INK)], pad=(int(26 * sc), int(16 * sc)),
                                radius=int(16 * sc), bg=(255, 255, 255, 225)), 0, 0) if caption else None
    if cap_ov is not None:
        cap_ov.x, cap_ov.y = (W_ - cap_ov.rgb.shape[1]) // 2, H_ - cap_ov.rgb.shape[0] - int(40 * sc)
    frames = 0
    for i in range(n):
        t = i * DT
        k = k0 + int(t * vfps)
        if k != k_last and k < k1:
            k_last = k
            det = None if np.isnan(W[k, 0, 0]) else {"world": W[k], "vis": V[k], "image": I[k]}
            f = hm.process(det, k / vfps, *wh)
            last = (det, f)
            obs = f.get("obs")
            c = est.update(obs) if obs is not None and f["present"] else None
            if c is None:
                v_cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            else:
                v_cmd = c
        tgt = be.arm_default if (last is None or last[1]["arm_q"] is None) else np.asarray(last[1]["arm_q"])
        arm = arm + np.clip(tgt - arm, -ARM_VMAX * DT, ARM_VMAX * DT)
        v = lim(v_cmd, DT)
        wy = last[1].get("waist_yaw") if last is not None else None
        bw = last[1].get("bow") if last is not None else None
        be.step(arm, waist_yaw=wy, bow=bw, loco_cmd=tuple(v))
        x, y, yaw = be.root_pose()
        log["t"].append(t)
        log["cmd"].append([v_cmd["vx"], v_cmd["vy"], v_cmd["wz"]])
        log["lim"].append(v.tolist())
        log["xy"].append([x, y])
        log["yaw"].append(yaw)
        log["tilt"].append(be.tilt_deg())
        if fell is None and be.fallen():
            fell = round(t, 2)
        if i % every == 0:
            robot = cam.grab()
            src = reader.get(k)
            if last is not None and last[0] is not None:
                src = draw_overlay(src, last[0])
            hum = cv2.cvtColor(fit(src), cv2.COLOR_BGR2RGB)
            img = np.concatenate([hum, robot], axis=1)
            for ov in overlays:
                ov.blend(img)
            st = ms.Overlay(ms.pill([(cmd_text(v), ms.font("medium", int(28 * sc)), ms.INK)], pad=(int(20 * sc), int(12 * sc)),
                                    radius=int(14 * sc), bar=None, bg=(255, 255, 255, 215)), pw + int(40 * sc), int(110 * sc))
            st.blend(img)
            if cap_ov is not None:
                cap_ov.blend(img)
            writer.append_data(img)
            frames += 1
    be.close()
    L = {k: np.asarray(v_) for k, v_ in log.items()}
    moved = float(np.linalg.norm(L["xy"][-1] - L["xy"][0]))
    return {"segment": name, "clip": path, "t0": t0, "t1": t1, "seconds": round(n * DT, 1), "frames": frames,
            "fell": fell is not None, "fell_at": fell, "max_tilt_deg": round(float(L["tilt"].max()), 1),
            "moved_m": round(moved, 2), "turned_rad": round(float(np.ptp(np.unwrap(L["yaw"]))), 2),
            "walk_frac": round(float((np.abs(L["lim"]).max(1) > 0.02).mean()), 3), "home": est.home}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", action="append", required=True, help="name=path@t0-t1|caption")
    ap.add_argument("--backend", default="split", choices=["split", "sonic"])
    ap.add_argument("--out", default=str(ROOT / "videos" / "k1_live_walk_split.mp4"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--mirror", action="store_true")
    a = ap.parse_args()
    import imageio.v2 as imageio
    from PIL import Image

    import make_showreel as ms

    size = tuple(int(x) for x in a.size.split("x"))
    W_, H_ = size
    sc = W_ / 1920.0
    f26, pad, rad = ms.font("semibold", int(26 * sc)), (int(20 * sc), int(12 * sc)), int(14 * sc)
    lab_h = ms.pill([("사람 (카메라 영상)", f26, ms.INK)], pad=pad, radius=rad, bar=None, bg=(255, 255, 255, 210))
    lab_r = ms.pill([(f"AI Sapiens K1 (MuJoCo 시뮬레이션, {a.backend})", f26, ms.INK)], pad=pad, radius=rad, bar=None,
                    bg=(255, 255, 255, 210))
    ovs = [ms.Overlay(lab_h, int(40 * sc), int(36 * sc)), ms.Overlay(lab_r, W_ // 2 + int(40 * sc), int(36 * sc)),
           ms.Overlay(Image.new("RGBA", (4, H_), (255, 255, 255, 255)), W_ // 2 - 2, 0)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
    t0 = time.perf_counter()
    res = []
    for spec in a.seg:
        r = segment(spec, a.backend, size, a.fps, writer, ovs, mirror=a.mirror)
        print(json.dumps(r, default=float), flush=True)
        res.append(r)
    writer.close()
    out = {"video": a.out, "backend": a.backend, "mirror": a.mirror, "segments": res,
           "wall_s": round(time.perf_counter() - t0, 1)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("video", "wall_s")}))


if __name__ == "__main__":
    main()
