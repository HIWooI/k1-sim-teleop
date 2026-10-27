"""K1 arm IK (k1live/arm_ik.py) and the arm path options (HumanMapper arm_ik, split arm_mode, server/sender flags).

    .venv/bin/python -m unittest tests.test_k1live_arm_ik -v       # pytest tests/test_k1live_arm_ik.py also works

The split-backend tests need the K1 MJCF (AI_SAPIENS_ROOT) and the policy ONNX files; skipped when missing.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k1live.arm_ik import (ARM_DEFAULT, SIDES, DirArmIK, DirRetargeter, image_inplane, overhead_tilt,  # noqa: E402
                           person_dirs)
from k1live.human import HumanMapper, mirror_arms  # noqa: E402

K1_XML = Path(os.environ.get("AI_SAPIENS_ROOT", str(ROOT / "third_party" / "ai_sapiens"))) / \
    "ai_sapiens_description/mujoco/k1/k1.xml"
POLICIES = [ROOT / "k1/assets" / a / "exported/policy.onnx" for a in ("student_asset", "loco_armdriven_r2")]
HAVE_SIM = K1_XML.exists() and all(p.exists() for p in POLICIES)
IMG_W, IMG_H, PX_PER_M = 1280, 720, 300.0


def person(left=(0.0, 0.0), right=(0.0, 0.0), depth_noise=None, rng=None):
    """A person facing the camera, arms in given poses, as a MediaPipe-like detection with a geometrically consistent
    (orthographic) image. Each arm: (frontal angle from hanging down toward the side, deg; forward lean, deg) for the
    upper arm, forearm straight. World = MediaPipe axes (x = person's left, y down, z away from the camera)."""
    w = np.zeros((33, 3))
    w[11], w[12] = [0.18, -0.50, 0.0], [-0.18, -0.50, 0.0]
    w[23], w[24] = [0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]
    w[25], w[26], w[27], w[28] = [0.1, 0.45, 0], [-0.1, 0.45, 0], [0.1, 0.85, 0], [-0.1, 0.85, 0]
    for s, sh, el, wr, (ang, fwd) in ((1, 11, 13, 15, left), (-1, 12, 14, 16, right)):
        a, f = np.radians(ang), np.radians(fwd)
        d = np.array([s * np.sin(a) * np.cos(f), np.cos(a) * np.cos(f), -np.sin(f)])  # forward = toward the camera
        w[el] = w[sh] + 0.28 * d
        w[wr] = w[el] + 0.25 * d
    img = np.zeros((33, 3))
    img[:, 0] = 0.5 + w[:, 0] * PX_PER_M / IMG_W
    img[:, 1] = 0.6 + w[:, 1] * PX_PER_M / IMG_H
    if depth_noise:
        w = w.copy()
        w[:, 2] += (rng or np.random.default_rng(0)).normal(0.0, depth_noise, 33)
    return {"world": w, "vis": np.full(33, 0.95), "image": img}


def img_px(det):
    return det["image"][:, :2] * np.array([IMG_W, IMG_H])


def frontal(d):
    """Frontal-plane angle of a torso-frame direction from up toward the robot's left (k1.run_video.RobotArm2D)."""
    return np.degrees(np.arctan2(d[1], d[2]))


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


class TestChain(unittest.TestCase):
    def test_fk_matches_urdf(self):
        """ArmChain.fk == the URDF tree FK (k1.cpu_kinematics) for shoulder roll link, elbow link, palm."""
        from k1.cpu_kinematics import UrdfKinematics
        from k1.sim import URDF

        robot = UrdfKinematics(URDF)
        ik = DirArmIK()
        rng = np.random.default_rng(1)
        for side in SIDES:
            ch = ik.chain[side]
            names = [f"{side}_{p}_joint" for p in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                                    "wrist_roll")]
            ix = [robot.joint_names.index(n) for n in names]
            for _ in range(20):
                q = np.zeros(robot.ndof)
                q[ix] = rng.uniform(ch.lo, ch.hi)
                pos, _, _ = robot.forward(q)
                sh, el, palm, _ = ch.fk(q[ix[:4]])
                np.testing.assert_allclose(sh, pos[f"{side}_shoulder_roll_link"], atol=1e-9)
                np.testing.assert_allclose(el, pos[f"{side}_elbow_link"], atol=1e-9)
                # the wrist link origin lies on the elbow -> palm line, 6.87 cm from the elbow
                wl = pos[f"{side}_wrist_roll_rubber_hand"]
                u = (palm - el) / np.linalg.norm(palm - el)
                np.testing.assert_allclose(wl, el + u * np.linalg.norm(wl - el), atol=1e-9)

    def test_round_trip_and_limits(self):
        ik = DirArmIK()
        rng = np.random.default_rng(2)
        errs = []
        for side in SIDES:
            lo, hi = ik.lo[side], ik.hi[side]
            for _ in range(150):
                q = rng.uniform(lo, hi)
                u, f = ik.directions(side, q)
                ik.prev[side] = np.clip(q + rng.normal(0, 0.3, 4), lo, hi)
                qs = ik.solve(side, u, f)
                self.assertTrue(np.all(qs >= lo - 1e-9) and np.all(qs <= hi + 1e-9))
                u2, f2 = ik.directions(side, qs)
                errs.append(np.degrees([np.arccos(np.clip(u @ u2, -1, 1)), np.arccos(np.clip(f @ f2, -1, 1))]))
        errs = np.array(errs)
        self.assertLess(np.median(errs), 1.5, errs.mean(0))
        self.assertLess(np.percentile(errs, 95), 6.0)

    def test_elbow_never_hyperextends(self):
        ik = DirArmIK()
        for side in SIDES:
            q = ik.solve(side, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0]))  # straight arm hanging
            self.assertLessEqual(q[3], np.pi / 2 + 1e-9)


class TestPersonDirs(unittest.TestCase):
    def test_frontal_angles(self):
        """Arms hanging / sideways / up / 45 deg: the target and the solved robot arm show the same frontal angle."""
        ret = DirRetargeter()
        for ang in (5.0, 45.0, 90.0, 135.0, 170.0):
            det = person((ang, 0.0), (ang, 0.0))
            for _ in range(5):
                q = ret.solve(det["world"], img_px(det))
            for i, side in enumerate(SIDES):
                u, f = ret.ik.directions(side, q[5 * i:5 * i + 4])
                want = 180.0 - ang if side == "left" else -(180.0 - ang)   # from up, toward the robot's left
                self.assertLess(angdiff(frontal(u), want), 3.0, (side, ang, frontal(u)))
                self.assertLess(angdiff(frontal(f), want), 3.0, (side, ang, frontal(f)))

    def test_forward_arm_uses_depth(self):
        """Arm pointing at the camera (forward 80 deg): pitch forward, not sideways."""
        ret = DirRetargeter()
        det = person((10.0, 80.0), (10.0, 80.0))
        for _ in range(5):
            q = ret.solve(det["world"], img_px(det))
        self.assertLess(q[0], -1.0)   # left shoulder pitch forward/up
        self.assertLess(q[5], -1.0)

    def test_depth_noise_does_not_move_the_frontal_angle(self):
        """MediaPipe's z is the noisy coordinate: with the image in-plane angle the frontal angle stays put."""
        rng = np.random.default_rng(3)
        clean = person((60.0, 10.0), (100.0, 0.0))
        d0 = person_dirs(clean["world"], img_px(clean))
        worst_img, worst_world = 0.0, 0.0
        for _ in range(20):
            noisy = person((60.0, 10.0), (100.0, 0.0), depth_noise=0.05, rng=rng)
            d1 = person_dirs(noisy["world"], img_px(noisy), image_w=1.0)
            d2 = person_dirs(noisy["world"], None, image_w=0.0)
            for s in SIDES:
                for b in (0, 1):
                    worst_img = max(worst_img, angdiff(frontal(d1[s][b]), frontal(d0[s][b])))
                    worst_world = max(worst_world, angdiff(frontal(d2[s][b]), frontal(d0[s][b])))
        self.assertLess(worst_img, 1.0)
        self.assertGreater(worst_world, worst_img)

    def test_image_inplane_definition(self):
        det = person((90.0, 0.0), (0.0, 0.0))
        ang, ln = image_inplane(img_px(det))
        self.assertAlmostEqual(np.degrees(ang[0]), 90.0, delta=0.5)        # left upper arm: toward the person's left
        self.assertAlmostEqual(abs(np.degrees(ang[2])), 180.0, delta=0.5)  # right arm hanging
        self.assertTrue(np.all(ln > 10))

    def test_overhead_tilt(self):
        t = np.tan(np.radians(30))
        for d in ([0, 0, 1.0], [0, 0.5, 0.866], [-0.2, -0.3, 0.93]):
            d = np.array(d) / np.linalg.norm(d)
            o = overhead_tilt(d, t)
            self.assertAlmostEqual(np.linalg.norm(o), 1.0)
            self.assertAlmostEqual(frontal(o), frontal(d), places=6)
            self.assertGreaterEqual(o[0] + 1e-9, t * o[2])
        for d in ([0, 1.0, 0], [0, 0.3, -0.95], [0.6, 0, 0.8]):   # horizontal, down, already forward enough
            d = np.array(d) / np.linalg.norm(d)
            np.testing.assert_allclose(overhead_tilt(d, t), d)


class TestMapper(unittest.TestCase):
    def _run(self, mapper, det, n=10):
        out = None
        for i in range(n):
            out = mapper.process(det, 100.0 + i / 30, IMG_W, IMG_H)
        return np.array(out["arm_q"])

    def test_default_is_shape18(self):
        self.assertEqual(HumanMapper(loco=False, smpl=False).arm_ik, "shape18")
        with self.assertRaises(ValueError):
            HumanMapper(loco=False, smpl=False, arm_ik="nope")

    def test_dir_mirror(self):
        det = person((40.0, 20.0), (120.0, 0.0))
        a = self._run(HumanMapper(mirror=False, loco=False, smpl=False, arm_ik="dir"), det)
        b = self._run(HumanMapper(mirror=True, loco=False, smpl=False, arm_ik="dir"), det)
        np.testing.assert_allclose(mirror_arms(a), b, atol=1e-9)
        np.testing.assert_allclose(a[[4, 9]], ARM_DEFAULT[[4, 9]])   # wrist roll at the relaxed default

    def test_dir_speed(self):
        hm = HumanMapper(mirror=True, loco=False, smpl=False, arm_ik="dir")
        rng = np.random.default_rng(4)
        ms = []
        for i in range(60):
            det = person((rng.uniform(0, 170), rng.uniform(-20, 60)), (rng.uniform(0, 170), rng.uniform(-20, 60)))
            hm.process(det, 100.0 + i / 30, IMG_W, IMG_H)
            ms.append(hm.ik_ms)
        self.assertLess(np.median(ms), 3.0, f"dir IK median {np.median(ms):.2f} ms")


@unittest.skipUnless(HAVE_SIM, "K1 MJCF / policy ONNX missing")
class TestSplitArmPath(unittest.TestCase):
    def test_modes(self):
        from k1live.backends.split import SplitBackend

        with self.assertRaises(ValueError):
            SplitBackend(arm_mode="nope")
        be = SplitBackend()
        self.assertEqual(be.arm_mode, "student")
        self.assertGreater(be.trim_gain, 0)
        pd = SplitBackend(arm_mode="pd")
        self.assertEqual(pd.trim_gain, 0.0)
        np.testing.assert_allclose(pd.sim.kp[pd.arm_idx], pd.sim.loco.kp[pd.arm_idx])

    def test_pd_holds_a_pose_standing(self):
        """PD arms + gravity compensation hold a raised pose within ~2 deg, legs stand (no fall, small tilt)."""
        from k1live.backends.split import SplitBackend

        be = SplitBackend(arm_mode="pd")
        goal = np.array([-1.2, 0.6, 0.2, 0.6, 0.15, -1.2, -0.6, -0.2, 0.6, -0.15])
        for _ in range(150):
            be.step(goal)
        self.assertFalse(be.fallen())
        self.assertLess(be.tilt_deg(), 8.0)
        err = np.degrees(np.abs(be.arm_q_now() - goal))
        self.assertLess(err.mean(), 2.0, err)
        be.close()


class TestFlags(unittest.TestCase):
    def test_server_and_sender_defaults(self):
        """The new options exist and default to the current behaviour (student arms, 6 rad/s / 80 rad/s^2, shape18)."""
        from unittest import mock

        from k1live import sender, sim_server

        env = {k: v for k, v in os.environ.items()
               if k not in ("K1LIVE_ARM_MODE", "K1LIVE_ARM_VMAX", "K1LIVE_ARM_AMAX", "K1LIVE_ARM_IK")}
        with mock.patch.dict(os.environ, env, clear=True):
            import importlib

            importlib.reload(sim_server)
            importlib.reload(sender)
            a = sim_server.parse_args([])
            self.assertEqual((a.arm_mode, a.arm_vmax, a.arm_amax), ("student", sim_server.ARM_VMAX, sim_server.ARM_AMAX))
            b = sim_server.parse_args(["--arm-mode", "pd", "--arm-vmax", "12", "--arm-amax", "150"])
            self.assertEqual((b.arm_mode, b.arm_vmax, b.arm_amax), ("pd", 12.0, 150.0))
            self.assertEqual(sender.parse_args([]).arm_ik, "shape18")
            self.assertEqual(sender.parse_args(["--arm-ik", "dir"]).arm_ik, "dir")


if __name__ == "__main__":
    unittest.main()
