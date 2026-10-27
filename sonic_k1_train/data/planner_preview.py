"""Kinematic preview of motion_lib PKLs (headless EGL/OSMesa render; run on a Linux host): a contact sheet (one row per
clip, frames evenly spaced, pelvis path drawn as a floor trail) and an optional side-by-side replay video.

    MUJOCO_GL=egl python sonic_k1_train/data/planner_preview.py --dir <out>/robot/planner --n 8 \
        --sheet logs/planner_sheet.png --video videos/k1_planner_kinematic.mp4 --video-clips 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def load(p):
    import joblib

    e = next(iter(joblib.load(p).values()))
    q = np.asarray(e["root_rot"])[:, [3, 0, 1, 2]]
    return np.asarray(e["root_trans_offset"]), q, np.asarray(e["dof"]), float(e["fps"])


class Render:
    def __init__(self, w=320, h=320):
        import mujoco

        from sonic.robots import get_robot

        self.mj = mujoco
        self.r = get_robot("k1")
        self.m = self.r.load_model()
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.r.mujoco_joint_names])
        self.ren = mujoco.Renderer(self.m, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.4, 140.0, -18.0

    def frame(self, p, q, dof, trail=None):
        d = self.d
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = p, q, dof
        self.mj.mj_forward(self.m, d)
        self.cam.lookat[:] = [p[0], p[1], 0.55]
        self.ren.update_scene(d, camera=self.cam)
        if trail is not None:  # pelvis path on the floor
            scn = self.ren.scene
            for a, b in zip(trail[:-1:3], trail[3::3]):
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                     np.array([0.9, 0.3, 0.2, 1.0], np.float32))
                self.mj.mjv_connector(g, self.mj.mjtGeom.mjGEOM_CAPSULE, 0.008, np.r_[a[:2], 0.002], np.r_[b[:2], 0.002])
                scn.ngeom += 1
        return self.ren.render().copy()

    def close(self):
        self.ren.close()


def label(img, text):
    import cv2

    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--video-clips", type=int, default=4)
    a = ap.parse_args()
    files = sorted(p for p in a.dir.rglob("*.pkl") if not p.stem.endswith("_M"))
    rng = np.random.default_rng(a.seed)
    pick = [files[i] for i in sorted(rng.choice(len(files), min(a.n, len(files)), replace=False))]
    R = Render()
    if a.sheet:
        import imageio

        rows = []
        for f in pick:
            p, q, dof, fps = load(f)
            idx = np.linspace(0, len(p) - 1, a.cols).astype(int)
            rows.append(np.concatenate([label(R.frame(p[i], q[i], dof[i], p), f"{f.stem[:30]} t={i / fps:.1f}s" if k == 0 else f"t={i / fps:.1f}s")
                                        for k, i in enumerate(idx)], 1))
        Path(a.sheet).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(a.sheet, np.concatenate(rows, 0))
        print("sheet", a.sheet, [f.stem for f in pick])
    if a.video:
        import imageio

        clips = [load(f) + (f.stem,) for f in pick[: a.video_clips]]
        T = max(len(c[0]) for c in clips)
        imgs = []
        for t in range(0, T, 2):
            tiles = []
            for p, q, dof, fps, name in clips:
                i = min(t, len(p) - 1)
                tiles.append(label(R.frame(p[i], q[i], dof[i], p[: i + 1]), f"{name[:28]} {i / fps:4.1f}s"))
            imgs.append(np.concatenate(tiles, 1))
        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(a.video, imgs, fps=25, quality=7, macro_block_size=1)
        print("video", a.video)
    R.close()


if __name__ == "__main__":
    main()
