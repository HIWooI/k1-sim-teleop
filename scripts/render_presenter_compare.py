"""Side-by-side reel: presenter video | K1 split | K1 sonic, the same person stream into both (MuJoCo only).

Per segment both robots respawn. The clip plays at its own clock as the fake camera; the person stream is exactly
scripts/eval_presenter.py's (sender-faithful MediaPipe cache -> JumpGate -> HumanMapper + LocoEstimator, walking
profile --loco-profile, person region from eval_presenter.CROPS when the segment asks for it) and each robot is driven
by its own k1live.sim_server.SimServer on the simulated clock (eval_presenter.ServerRun). Mirror is off in the reel
(person and robots side by side, all facing the viewer: same side = same arm); the live default (mirror on) only
swaps left / right. Small labels only.

  .venv/bin/python scripts/render_presenter_compare.py --out videos/k1_presenter_compare.mp4 \
      --seg "p01_stage_mic@0-13|무대 강연 · 손에 마이크" --seg "p07_podium_close@0-13:crop|연단 뒤 (허리 위)" ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DT = 0.02
SKY = np.array([228, 231, 236], np.uint8)


class RobotCam:
    """Offscreen camera on a backend's model, slow pelvis follow (both backends look the same way)."""

    def __init__(self, be, w, h, azimuth=180.0, distance=2.6, elevation=-8.0, lookat_z=0.62):
        import mujoco

        m = be.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
        self.mj, self.be = mujoco, be
        self.r = mujoco.Renderer(m, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.distance, self.cam.elevation = azimuth, distance, elevation
        self.lookat_z = lookat_z
        self.xy = None

    def grab(self):
        p = self.be.data.xpos[self.be.pelvis_body]
        xy = p[:2].copy()
        self.xy = xy if self.xy is None else 0.95 * self.xy + 0.05 * xy
        self.cam.lookat[:] = [self.xy[0], self.xy[1], self.lookat_z]
        self.r.update_scene(self.be.data, camera=self.cam)
        img = self.r.render().copy()
        img[np.all(img < 4, axis=2)] = SKY      # a model without a skybox (sonic's staged MJCF) renders black sky
        return img

    def close(self):
        self.r.close()


def parse_seg(spec):
    """name@t0-t1[:crop]|caption"""
    head, _, caption = spec.partition("|")
    name, _, rest = head.partition("@")
    rng, _, flag = rest.partition(":")
    a, _, b = rng.partition("-")
    return name, float(a or 0.0), (float(b) if b else None), flag == "crop", caption


def segment(spec, args, size, writer, overlays, log_root):
    import cv2

    import eval_presenter as E
    import make_showreel as ms
    import run_video as g1rv
    from sonic.pose_mp import draw_overlay

    name, t0, t1, use_crop, caption = parse_seg(spec)
    clip = {c["name"]: c for c in E.manifest_clips()}[name]
    crop = E.CROPS.get(name) if use_crop else None
    D = E.detections(clip["video"], crop=crop)
    frames = E.person_stream(D, mirror=False, profile=args.loco_profile, t0=t0, t1=t1)
    W_, H_ = size
    pw = W_ // 3
    runs = [E.ServerRun(be, Path(log_root) / f"reel_{name}_{be}",
                        args.walk_variant if be == "sonic_walk" else None) for be in args.backends]
    cams = [RobotCam(r.srv.be, pw, H_) for r in runs]
    src_fps = float(D["src_fps"])
    k_lo = int(np.searchsorted(D["t"], t0 - 1e-9))
    I, V = D["image"][k_lo:k_lo + len(frames)], D["vis"][k_lo:k_lo + len(frames)]
    fit = g1rv.PanelFit(pw, H_, g1rv.person_span(I, V))
    reader = g1rv.FrameReader(clip["video"], height=H_)
    sc = W_ / 1920.0
    cap_ov = None
    if caption:
        cap_ov = ms.Overlay(ms.pill([(caption, ms.font("semibold", int(30 * sc)), ms.INK)],
                                    pad=(int(22 * sc), int(14 * sc)), radius=int(14 * sc), bg=(255, 255, 255, 225)),
                            0, 0)
        cap_ov.x, cap_ov.y = (W_ - cap_ov.rgb.shape[1]) // 2, H_ - cap_ov.rgb.shape[0] - int(36 * sc)
    every = max(1, int(round(1.0 / (args.fps * DT))))
    n = int((frames[-1]["t_capture"] + 0.3) / DT)
    shown = 0
    for i in range(n):
        for r in runs:
            r.step(frames)
        if i % every:
            continue
        k_det = max(0, runs[0].k - 1)              # newest HumanFrame the servers have seen
        tt = t0 + i * DT
        src = reader.get(int(round(tt * src_fps)))
        f = frames[k_det] if frames else None
        if f is not None and f["_found"] and not f["_gated"]:
            kk = k_lo + k_det
            src = draw_overlay(src, {"image": D["image"][kk], "vis": D["vis"][kk]}, color=(80, 220, 120))
        if crop is not None:
            h, w = src.shape[:2]
            cv2.rectangle(src, (int(crop[0] * w), int(crop[1] * h)), (int(crop[2] * w) - 1, int(crop[3] * h) - 1),
                          (40, 200, 255), max(2, w // 400))
        panels = [cv2.cvtColor(fit(src), cv2.COLOR_BGR2RGB)] + [c.grab() for c in cams]
        img = np.concatenate(panels, axis=1)
        for ov in overlays:
            ov.blend(img)
        if cap_ov is not None:
            cap_ov.blend(img)
        writer.append_data(img)
        shown += 1
    out = {"segment": name, "t0": t0, "t1": t1, "crop": crop, "caption": caption, "frames": shown,
           "backends": {}}
    for r, c in zip(runs, cams):
        out["backends"][r.backend] = r.metrics()
        c.close()
        r.close()
    out["perception"] = E.perception_metrics(frames, clip["kind"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", action="append", required=True, help="name@t0-t1[:crop]|caption (eval_presenter MANIFEST)")
    ap.add_argument("--backends", nargs=2, default=["split", "sonic"])
    ap.add_argument("--labels", nargs=2, default=["K1 split (팔 student + 다리 loco)", "K1 sonic (K1 SONIC 전신)"])
    ap.add_argument("--walk-variant", default="")
    ap.add_argument("--loco-profile", default="presenter")
    ap.add_argument("--out", default=str(ROOT / "videos" / "k1_presenter_compare.mp4"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    import imageio.v2 as imageio
    from PIL import Image

    import make_showreel as ms

    size = tuple(int(x) for x in a.size.split("x"))
    W_, H_ = size
    sc = W_ / 1920.0
    pw = W_ // 3
    f22, pad, rad = ms.font("semibold", int(22 * sc)), (int(16 * sc), int(10 * sc)), int(12 * sc)
    labels = ["사람 (카메라 영상)"] + list(a.labels)
    ovs = [ms.Overlay(ms.pill([(lab, f22, ms.INK)], pad=pad, radius=rad, bar=None, bg=(255, 255, 255, 210)),
                      j * pw + int(24 * sc), int(24 * sc)) for j, lab in enumerate(labels)]
    ovs += [ms.Overlay(Image.new("RGBA", (4, H_), (255, 255, 255, 255)), j * pw - 2, 0) for j in (1, 2)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
    t0 = time.perf_counter()
    res = []
    for spec in a.seg:
        r = segment(spec, a, size, writer, ovs, ROOT / "logs" / "presenter" / "server")
        print(json.dumps({k: r[k] for k in ("segment", "frames")} | {
            b: {k: m[k] for k in ("arm_err_deg", "falls", "max_tilt_deg", "person_walk_s", "moved_m")}
            for b, m in r["backends"].items()}, default=float), flush=True)
        res.append(r)
    writer.close()
    out = {"video": a.out, "backends": a.backends, "loco_profile": a.loco_profile, "mirror": False, "segments": res,
           "seconds": round(sum(r["frames"] for r in res) / a.fps, 1), "wall_s": round(time.perf_counter() - t0, 1)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("video", "seconds", "wall_s")}))


if __name__ == "__main__":
    main()
