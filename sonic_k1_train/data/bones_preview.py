"""Look at converted pairs: the person's SMPL-24 stream (left, skeleton) next to the K1 reference (right, MuJoCo) at
the same instants - contact sheet PNG and/or side-by-side MP4. Offscreen only (MUJOCO_GL=egl on a Linux host).

    MUJOCO_GL=egl python sonic_k1_train/data/bones_preview.py --set <set> --keys k1 k2 --sheet out.png --video out.mp4

Both panels look at the robot's initial heading from the front (the person has the same heading in the capture world,
check.json heading_diff_deg), so left/right in the picture are the same body side in both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from k1live import smpl_stream  # noqa: E402

BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13),
         (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}


def load(set_dir: Path, key: str):
    import joblib

    p = next((set_dir / "robot").rglob(f"{key}.pkl"), None) or next((set_dir / "heldout" / "robot").rglob(f"{key}.pkl"))
    base = p.parents[2]  # <set> or <set>/heldout: holds robot/<group>/ and smpl/
    e = joblib.load(p)[key]
    s = joblib.load(base / "smpl" / f"{key}.pkl")
    return e, s


class Panels:
    def __init__(self, h=360, w=320):
        import mujoco

        from sonic.robots import get_robot

        self.mj = mujoco
        self.spec = get_robot("k1")
        self.m = self.spec.load_model()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = 2.4, -8.0
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.spec.mujoco_joint_names])
        self.h, self.w = h, w

    def robot(self, e, f, yaw0):
        d = self.d
        d.qpos[:] = 0
        d.qpos[:3] = e["root_trans_offset"][f]
        q = e["root_rot"][f]
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[self.qadr] = e["dof"][f]
        self.mj.mj_forward(self.m, d)
        self.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.55]
        self.cam.azimuth = np.degrees(yaw0) + 180.0
        self.r.update_scene(d, camera=self.cam)
        return self.r.render().copy()

    def person(self, s, f, yaw0, scale=0.78):
        import cv2

        img = np.full((self.h, self.w, 3), 245, np.uint8)
        r0 = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][f, :3])
        J = np.asarray(s["smpl_joints"][f], np.float64)  # world-rotated, pelvis at 0
        c, sn = np.cos(-yaw0), np.sin(-yaw0)
        J = J @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]).T
        z0 = -J[:, 2].min()
        px = lambda p: (int(self.w / 2 + p[1] * scale * self.h / 1.6), int(self.h * 0.92 - (p[2] + z0) * scale * self.h / 1.6))  # noqa: E731
        for a, b in BONES:
            col = (40, 90, 220) if b in LEFT else (220, 80, 40)
            cv2.line(img, px(J[a]), px(J[b]), col, 3, cv2.LINE_AA)
        for i, p in enumerate(J):
            cv2.circle(img, px(p), 4, (30, 30, 30), -1, cv2.LINE_AA)
        hd = np.degrees(np.arccos(np.clip(1 - 2 * (r0[1] ** 2 + r0[2] ** 2), -1, 1)))
        cv2.putText(img, f"root tilt {hd:4.1f} deg", (8, self.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        return img

    def close(self):
        self.r.close()


def caption(img, text):
    import cv2

    img = img.copy()
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def yaw_of(e):
    q = e["root_rot"][0]  # xyzw
    return float(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--sheet", type=Path, default=None)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--video-seconds", type=float, default=20.0)
    a = ap.parse_args()
    P = Panels()
    if a.sheet:
        rows = []
        for k in a.keys:
            e, s = load(a.set, k)
            T, y0 = len(e["dof"]), yaw_of(e)
            cells = []
            for f in np.linspace(0, T - 1, a.cols).astype(int):
                pair = np.concatenate([P.person(s, f, y0), P.robot(e, f, y0)[:, :, ::-1]], 1)
                cells.append(caption(pair, f"{k[:30]}  t={f / e['fps']:.1f}s"))
            rows.append(np.concatenate(cells, 1))
        a.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.sheet), np.concatenate(rows, 0))
        print("wrote", a.sheet)
    if a.video:
        import imageio.v2 as imageio

        wr = imageio.get_writer(str(a.video), fps=25, codec="libx264", quality=7, macro_block_size=8)
        for k in a.keys:
            e, s = load(a.set, k)
            y0 = yaw_of(e)
            for f in range(0, min(len(e["dof"]), int(a.video_seconds * e["fps"])), 2):
                pair = np.concatenate([P.person(s, f, y0)[:, :, ::-1], P.robot(e, f, y0)], 1)
                wr.append_data(caption(pair, f"{k}  t={f / e['fps']:.2f}s  (left: person SMPL stream, right: K1)"))
        wr.close()
        print("wrote", a.video)
    P.close()


if __name__ == "__main__":
    main()
