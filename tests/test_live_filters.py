"""Live-teleop stabilisers in sonic/mp2smpl.py and sonic/stream.py (stdlib unittest, no pytest).

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sonic import rot  # noqa: E402
from sonic.mp2smpl import (BoneDeadband, MediaPipeToSmpl, PresenceGate, blend_frames,  # noqa: E402
                           frame_world, make_frame, parents)
from sonic.stream import StreamBuffer, ctrl_message, unpack_ctrl_message  # noqa: E402


def _bone_lengths(frame):
    W = frame_world(frame)
    par = parents()
    return np.array([np.linalg.norm(W[i] - W[par[i]]) for i in range(1, 24)])


def _yawed(frame, yaw):
    q = rot.quat_mul(rot.euler_z_to_quat(yaw), np.asarray(frame["body_quat"], np.float64))
    return make_frame(q, rot.quat_rotate(np.broadcast_to(rot.euler_z_to_quat(yaw), (24, 4)), frame_world(frame)))


def _raised_left_arm(conv):
    """Idle pose with the left arm lifted sideways to shoulder height."""
    W = frame_world(conv.idle_frame()).copy()
    L = conv.len
    W[18] = W[16] + L[18] * np.array([0.0, 1.0, 0.0])
    W[20] = W[18] + L[20] * np.array([0.0, 1.0, 0.0])
    W[22] = W[20] + L[22] * np.array([0.0, 1.0, 0.0])
    return make_frame([1.0, 0.0, 0.0, 0.0], W)


def _cached_landmarks():
    hits = sorted((ROOT / "logs").glob("mp_event_stand_*.npz")) + sorted((ROOT / "logs").glob("mp_event_bow_*.npz"))
    return hits[0] if hits else None


class TestDefaultsUnchanged(unittest.TestCase):
    """With default arguments the converter must reproduce the committed baseline exactly
    (run_video.py results stay comparable)."""

    def test_matches_initial_import(self):
        cache = _cached_landmarks()
        if cache is None:
            self.skipTest("no cached MediaPipe landmarks in logs/")
        try:
            src = subprocess.run(["git", "show", "9603dd7:sonic/mp2smpl.py"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout
        except Exception as e:  # pragma: no cover - git missing
            self.skipTest(f"git unavailable: {e}")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mp2smpl_base.py"
            p.write_text(src)
            spec = importlib.util.spec_from_file_location("sonic._mp2smpl_base", p)
            old = importlib.util.module_from_spec(spec)
            old.__package__ = "sonic"
            spec.loader.exec_module(old)
        z = np.load(cache)
        W, V = z["world"], z["vis"]
        ok = ~np.isnan(W[:, 0, 0])
        W, V = W[ok][:150], V[ok][:150]
        for kw in ({}, {"arm_depth_scale": 0.7, "legs": "auto"}, {"legs": "lock", "pelvis_mode": "zero"}):
            a, b = old.MediaPipeToSmpl(**kw), MediaPipeToSmpl(**kw)
            a.calibrate(W[:15])
            b.calibrate(W[:15])
            V2 = V.copy()
            V2[40:60, 25:29] = 0.2  # exercise the auto leg lock
            for t in range(len(W)):
                oa, ob = a.convert(W[t], V2[t]), b.convert(W[t], V2[t])
                np.testing.assert_allclose(oa["smpl_joints"], ob["smpl_joints"], atol=1e-6)
                np.testing.assert_allclose(oa["body_quat"], ob["body_quat"], atol=1e-7)
                self.assertEqual(a.legs_locked, b.legs_locked)


@pytest.mark.needs_smpl
class TestBlend(unittest.TestCase):
    def setUp(self):
        self.conv = MediaPipeToSmpl()
        self.idle = self.conv.idle_frame()
        self.arm = _raised_left_arm(self.conv)

    def test_endpoints(self):
        self.assertIs(blend_frames(self.idle, self.arm, 0.0), self.idle)
        self.assertIs(blend_frames(self.idle, self.arm, 1.0), self.arm)
        np.testing.assert_allclose(blend_frames(self.idle, self.arm, 1e-9)["smpl_joints"], self.idle["smpl_joints"], atol=1e-5)

    def test_keeps_canonical_skeleton(self):
        ref = _bone_lengths(self.idle)
        b = _yawed(self.arm, 0.6)
        for w in np.linspace(0.05, 0.95, 7):
            np.testing.assert_allclose(_bone_lengths(blend_frames(self.idle, b, w)), ref, atol=1e-5)

    def test_root_slerp(self):
        b = _yawed(self.idle, 1.0)
        m = blend_frames(self.idle, b, 0.5)
        self.assertAlmostEqual(float(rot.calc_heading(np.asarray(m["body_quat"], np.float64))), 0.5, places=4)

    def test_idle_pose_is_calm(self):
        W = frame_world(self.idle)
        self.assertLess(W[20, 2], W[16, 2] - 0.4)  # wrists well below the shoulders
        self.assertLess(W[21, 2], W[17, 2] - 0.4)
        self.assertGreater(W[20, 1] - W[21, 1], W[16, 1] - W[17, 1] - 0.05)  # hands at the sides, not crossed


@pytest.mark.needs_smpl
class TestDeadband(unittest.TestCase):
    def test_still_noise_is_removed_and_motion_passes(self):
        conv = MediaPipeToSmpl()
        idle = conv.idle_frame()
        rng = np.random.default_rng(0)
        db = BoneDeadband(bone_deg=2.0, root_deg=1.0)
        W0 = frame_world(idle)
        outs = []
        par = parents()
        raw, outs = [], []
        for _ in range(100):  # every bone direction wobbles by ~0.6 deg (MediaPipe on a still person)
            W = np.zeros_like(W0)
            W[0] = W0[0]
            for i in range(1, 24):
                R = rot.quat_to_rotmat(rot.aa_to_quat(rng.normal(0, np.radians(0.6), 3)))
                W[i] = W[par[i]] + R @ (W0[i] - W0[par[i]])
            q = rot.aa_to_quat(rng.normal(0, np.radians(0.3), 3))
            f = make_frame(q, W)
            raw.append(f["smpl_joints"])
            outs.append(db(f)["smpl_joints"])
        raw, outs = np.array(raw[5:]), np.array(outs[5:])
        def step_rms(x):  # frame-to-frame motion = what makes the robot shake
            return float(np.sqrt((np.diff(x, axis=0) ** 2).sum(-1).mean()))

        self.assertLess(step_rms(outs), 0.2 * step_rms(raw))
        # the held pose wanders inside the band at most: never further than ~bone_deg from the truth
        self.assertLess(float(np.abs(outs - idle["smpl_joints"]).max()), 0.06)
        # a real movement still goes through, at most bone_deg behind
        arm = _raised_left_arm(conv)
        for w in np.linspace(0, 1, 40):
            out = db(blend_frames(idle, arm, w))
        v_out = frame_world(out)[20] - frame_world(out)[18]
        v_ref = frame_world(arm)[20] - frame_world(arm)[18]
        ang = np.degrees(np.arccos(np.dot(v_out, v_ref) / np.linalg.norm(v_out) / np.linalg.norm(v_ref)))
        self.assertLess(ang, 2.0 + 0.5)

    def test_zero_band_is_passthrough(self):
        conv = MediaPipeToSmpl()
        f = _raised_left_arm(conv)
        self.assertIs(BoneDeadband(0.0, 0.0)(f), f)


@pytest.mark.needs_smpl
class TestPresenceGate(unittest.TestCase):
    def _max_step(self, frames):
        J = np.array([frame_world(f) for f in frames])
        return float(np.abs(np.diff(J, axis=0)).max())

    def test_hold_fade_return(self):
        conv = MediaPipeToSmpl()
        idle, arm = conv.idle_frame(), _raised_left_arm(conv)
        g = PresenceGate(idle, hold_s=0.5, fade_out_s=1.5, fade_in_s=0.8, confirm_s=0.2)
        dt = 1 / 30
        seq, states = [], []
        t = 0.0
        plan = [(arm, 2.0), (None, 0.3), (arm, 1.0), (None, 3.0), (arm, 2.0)]
        for live, dur in plan:
            for _ in range(int(round(dur / dt))):
                seq.append(g(t, live))
                states.append(g.state)
                t += dt
        n0 = int(round(2.0 / dt))
        # starts idle, fades in only after the confirm time
        np.testing.assert_allclose(seq[0]["smpl_joints"], idle["smpl_joints"], atol=1e-6)
        self.assertEqual(states[n0 - 1], "TRACK")
        np.testing.assert_allclose(seq[n0 - 1]["smpl_joints"], arm["smpl_joints"], atol=1e-5)
        # a 0.3 s dropout only holds the pose
        for k in range(n0, n0 + 9):
            self.assertEqual(states[k], "HOLD")
            np.testing.assert_allclose(seq[k]["smpl_joints"], arm["smpl_joints"], atol=1e-5)
        # 3 s gone -> back at the idle pose at the end
        n_gone_end = int(round(6.3 / dt)) - 1
        self.assertEqual(states[n_gone_end], "IDLE")
        np.testing.assert_allclose(seq[n_gone_end]["smpl_joints"], idle["smpl_joints"], atol=1e-5)
        self.assertEqual(states[-1], "TRACK")
        np.testing.assert_allclose(seq[-1]["smpl_joints"], arm["smpl_joints"], atol=1e-5)
        # never a jump: per-frame joint step bounded (a raw idle<->arm switch is ~0.6 m)
        self.assertLess(self._max_step(seq), 0.05)

    def test_reappearing_mid_fade_is_continuous(self):
        conv = MediaPipeToSmpl()
        idle, arm = conv.idle_frame(), _raised_left_arm(conv)
        other = _yawed(arm, 0.4)
        g = PresenceGate(idle)
        seq, t = [], 0.0
        for live, n in [(arm, 60), (None, 30), (other, 60)]:
            for _ in range(n):
                seq.append(g(t, live))
                t += 1 / 30
        self.assertLess(self._max_step(seq), 0.05)


class TestStreamWatchdog(unittest.TestCase):
    @pytest.mark.needs_smpl
    def test_stale_sender_fades_to_idle_and_back(self):
        conv = MediaPipeToSmpl()
        idle, arm = conv.idle_frame(), _raised_left_arm(conv)
        buf = StreamBuffer(window=10, delay=0.0, idle=idle, stale_hold=0.4, stale_fade=1.0, resume_fade=0.5)
        t = 0.0
        while t < 1.0:
            buf.add_sample(t, arm["body_quat"], arm["smpl_joints"])
            t += 1 / 30
        last = t - 1 / 30
        buf.start(last)
        buf.active = True
        out = []
        for tt in np.arange(last, last + 3.0, 0.02):  # sender silent for 3 s
            buf.tick(tt)
            out.append(buf.motion.smpl_joints[-1].copy())
        np.testing.assert_allclose(out[5], arm["smpl_joints"], atol=1e-5)  # within hold
        np.testing.assert_allclose(out[-1], idle["smpl_joints"], atol=1e-5)  # faded out
        self.assertTrue(buf.stale)
        t0 = last + 3.0
        for k in range(60):  # sender back
            buf.add_sample(t0 + k * 0.02, arm["body_quat"], arm["smpl_joints"])
            buf.tick(t0 + k * 0.02)
            out.append(buf.motion.smpl_joints[-1].copy())
        self.assertFalse(buf.stale)
        np.testing.assert_allclose(out[-1], arm["smpl_joints"], atol=1e-5)
        self.assertLess(float(np.abs(np.diff(np.array(out), axis=0)).max()), 0.05)

    @pytest.mark.needs_smpl
    def test_without_idle_holds_last(self):
        conv = MediaPipeToSmpl()
        arm = _raised_left_arm(conv)
        buf = StreamBuffer(window=10)
        buf.add_sample(0.0, arm["body_quat"], arm["smpl_joints"])
        buf.start(0.0)
        buf.tick(5.0)
        np.testing.assert_allclose(buf.motion.smpl_joints[-1], arm["smpl_joints"], atol=1e-6)

    def test_ctrl_roundtrip(self):
        msg = ctrl_message("status", text="TRACK")
        self.assertTrue(msg.startswith(b"ctrl"))
        self.assertEqual(unpack_ctrl_message(msg), {"cmd": "status", "text": "TRACK"})


@pytest.mark.needs_smpl
class TestLegsAndArms(unittest.TestCase):
    def _upright(self):
        """Synthetic upright MediaPipe world landmarks (y down, z away from the camera)."""
        P = np.zeros((33, 3))
        P[[11, 12]] = [[0.18, -0.5, 0], [-0.18, -0.5, 0]]
        P[[13, 14]] = [[0.2, -0.22, 0], [-0.2, -0.22, 0]]
        P[[15, 16]] = [[0.21, 0.02, 0], [-0.21, 0.02, 0]]
        P[[17, 18, 19, 20]] = [[0.22, 0.08, 0.01], [-0.22, 0.08, 0.01], [0.21, 0.09, -0.01], [-0.21, 0.09, -0.01]]
        P[[23, 24]] = [[0.1, 0, 0], [-0.1, 0, 0]]
        P[[25, 26]] = [[0.1, 0.42, -0.02], [-0.1, 0.42, -0.02]]
        P[[27, 28]] = [[0.1, 0.82, 0], [-0.1, 0.82, 0]]
        P[[31, 32]] = [[0.1, 0.88, -0.12], [-0.1, 0.88, -0.12]]
        P[[7, 8]] = [[0.07, -0.62, 0.02], [-0.07, -0.62, 0.02]]
        P[0] = [0, -0.65, -0.08]
        return P

    def test_leg_hysteresis_no_flicker(self):
        P = self._upright()
        conv = MediaPipeToSmpl(legs="auto", leg_blend_s=0.4, leg_unlock_thresh=0.65, leg_lock_frames=3,
                               leg_unlock_frames=10)
        conv.calibrate([P] * 5)
        vis = np.ones(33)
        flips = 0
        prev = conv.legs_locked
        lw = []
        for k in range(120):
            v = vis.copy()
            v[25:29] = 0.45 if k % 2 else 0.55  # visibility hovering around the threshold
            conv.convert(P, v, t=k / 30)
            flips += conv.legs_locked != prev
            prev = conv.legs_locked
            lw.append(conv.leg_w)
        self.assertEqual(flips, 0)
        self.assertFalse(conv.legs_locked)
        lw = []
        for k in range(120, 160):  # legs really gone -> lock, smoothly
            v = vis.copy()
            v[25:29] = 0.1
            conv.convert(P, v, t=k / 30)
            lw.append(conv.leg_w)
        self.assertTrue(conv.legs_locked)
        self.assertAlmostEqual(lw[-1], 1.0)
        self.assertLessEqual(max(np.diff([0.0] + lw)), 1 / 30 / 0.4 + 1e-9)

    def test_arm_forward_cap(self):
        P = self._upright()
        P[13] = [0.18, -0.5, -0.3]  # elbow straight towards the camera
        P[15] = [0.18, -0.5, -0.56]
        conv = MediaPipeToSmpl(max_arm_forward=0.5)
        conv.calibrate([self._upright()] * 5)
        W = conv.convert(P)["joints_W"]
        u = (W[18] - W[16]) / np.linalg.norm(W[18] - W[16])
        self.assertLessEqual(u[0], 0.5 + 1e-6)
        conv0 = MediaPipeToSmpl()
        conv0.calibrate([self._upright()] * 5)
        W0 = conv0.convert(P)["joints_W"]
        self.assertGreater(((W0[18] - W0[16]) / np.linalg.norm(W0[18] - W0[16]))[0], 0.9)


if __name__ == "__main__":
    unittest.main()
