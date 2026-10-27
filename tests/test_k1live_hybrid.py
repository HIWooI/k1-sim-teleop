"""Hybrid backend (k1live/backends/hybrid.py): standing = the split backend bit for bit, walking = K1 SONIC + planner,
hand-over bookkeeping, sim_server wiring. MuJoCo simulation only.

    .venv/bin/python -m unittest tests.test_k1live_hybrid -v      # ~1-2 min on one core

Needs the staged K1 MJCF + meshes (planner retarget), models/k1/k1s_walk3/deploy/k1s_walk3_005000,
models/planner/planner_sonic.onnx, the vendor K1 MJCF (AI_SAPIENS_ROOT) and the K1 policy ONNX files; skipped otherwise.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT = 0.02


def _available():
    try:
        from k1.sim import K1_XML
        from k1live.backends.hybrid import HybridBackend

        policies = [ROOT / "k1/assets" / a / "exported/policy.onnx" for a in ("student_asset", "loco_armdriven_r2")]
        # the standing side is the split backend: vendor K1 MJCF (AI_SAPIENS_ROOT) + the ROBOTIS policies (not in git)
        return HybridBackend.available() and K1_XML.exists() and all(p.exists() for p in policies)
    except Exception:
        return False


AVAILABLE = _available()


def _arms(be, i):
    return np.clip(be.arm_default + 0.35 * np.sin(0.06 * i + np.arange(10)), be.arm_lo, be.arm_hi)


@unittest.skipUnless(AVAILABLE, "hybrid backend files missing (walk3 checkpoint, staged K1 MJCF, planner)")
class TestHybridBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from k1live.backends.hybrid import HybridBackend

        cls.be = HybridBackend(planner_async=False)

    @classmethod
    def tearDownClass(cls):
        cls.be.close()

    def setUp(self):
        self.be.reset()

    def test_stand_equals_split(self):
        """STAND runs the split controllers on the same physics: same trajectory as the split backend (SONIC and
        the planner run in the shadow without touching the robot)."""
        from k1live.backends.split import SplitBackend

        sp = SplitBackend()
        be = self.be
        prev = be.arm_default.copy()
        for i in range(200):
            arm = prev + np.clip(_arms(be, i) - prev, -0.1, 0.1)
            prev = arm
            hold = (0.0, 0.0, 0.05 * np.sin(0.1 * i))         # the server's heading hold only
            a = be.step(arm, loco_cmd=hold, arm_dq=None, walk_cmd=(0.0, 0.0, 0.0))
            b = sp.step(arm, loco_cmd=hold, arm_dq=None)
            self.assertEqual(a["phase"], "stand")
        self.assertLess(float(np.abs(be.data.qpos - sp.data.qpos).max()), 1e-9)
        np.testing.assert_allclose(a["target"], b["target"], atol=1e-9)

    def test_stand_equals_split_pd_arms(self):
        """arm_mode="pd" (the PD arm path): standing = SplitBackend(arm_mode="pd") as well."""
        from k1live.backends.hybrid import HybridBackend
        from k1live.backends.split import SplitBackend

        sp = SplitBackend(arm_mode="pd")
        be = HybridBackend(planner_async=False, arm_mode="pd")
        try:
            prev = be.arm_default.copy()
            for i in range(150):
                arm = prev + np.clip(_arms(be, i) - prev, -0.1, 0.1)
                dq = (arm - prev) / DT
                prev = arm
                a = be.step(arm, loco_cmd=(0.0, 0.0, 0.0), arm_dq=dq, walk_cmd=(0.0, 0.0, 0.0))
                sp.step(arm, loco_cmd=(0.0, 0.0, 0.0), arm_dq=dq)
            self.assertEqual(a["phase"], "stand")
            self.assertLess(float(np.abs(be.data.qpos - sp.data.qpos).max()), 1e-9)
            for i in range(int(4.0 / DT)):   # and it still walks / comes back with PD arms
                c = (0.35, 0.0, 0.0) if i * DT < 1.5 else (0.0, 0.0, 0.0)
                info = be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
                self.assertFalse(be.fallen())
            self.assertIn(info["phase"], ("settle", "stand"))
        finally:
            be.close()

    def test_shadow_last_action_is_applied_target(self):
        """While SONIC does not drive a joint, its last action decodes to the target actually applied there."""
        be = self.be
        for i in range(10):
            info = be.step(_arms(be, i))
        r = be.robot
        q = r.default_angles + be.ctrl.last_action[r.isaaclab_to_mujoco] * r.action_scale
        np.testing.assert_allclose(q, info["target"], atol=1e-9)

    def test_walk_then_settle_back_to_stand(self):
        be = self.be
        x0 = be.root_pose()[0]
        phases, t_walk, t_stand = [], None, None
        tilt = 0.0
        for i in range(int(12.0 / DT)):
            t = i * DT
            c = (0.4, 0.0, 0.0) if 1.0 <= t < 5.0 else (0.0, 0.0, 0.0)
            info = be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
            phases.append(info["phase"])
            tilt = max(tilt, be.tilt_deg())
            if t_walk is None and info["phase"] == "walk":
                t_walk = t
            if t_walk is not None and t_stand is None and t > 5.0 and info["phase"] == "stand":
                t_stand = t
            self.assertFalse(be.fallen(), f"fell at {t:.2f}")
        self.assertAlmostEqual(t_walk, 1.0 + be.enter_s, delta=0.05)
        self.assertGreater(be.root_pose()[0] - x0, 0.6)          # ~0.3 m/s for 4 s
        self.assertIsNotNone(t_stand)
        self.assertLess(t_stand, 5.0 + be.settle_max_s + 0.1)
        self.assertEqual(phases[-1], "stand")
        self.assertEqual(be.wL, 0.0)
        self.assertLess(tilt, 20.0)

    def test_turn_stops_when_released(self):
        """A turn command becomes a stepping arc; once released the robot stops turning (no planner facing lead
        left to catch up with: up to 0.6 rad before the fix)."""
        be = self.be
        yaw_at_release = None
        for i in range(int(7.0 / DT)):
            t = i * DT
            c = (0.0, 0.0, 0.5) if 0.5 <= t < 3.0 else (0.0, 0.0, 0.0)
            be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
            if yaw_at_release is None and t >= 3.0:
                yaw_at_release = be.root_pose()[2]
            self.assertFalse(be.fallen())
        turned = abs(np.degrees(np.angle(np.exp(1j * (be.root_pose()[2] - yaw_at_release)))))
        self.assertGreater(abs(np.degrees(yaw_at_release)), 30.0)     # it did turn
        self.assertLess(turned, 15.0)                                 # and stopped soon after the release

    def test_settle_keeps_walking_arms(self):
        """settle_arms="walk" (default): SETTLE keeps the walking arms (SONIC on the planner arm reference); the arms go
        back to the person only after the legs are on the loco again (docs/results.md: the person's resting arms
        clamped the hands onto the hips while SONIC was stopping)."""
        be = self.be
        seen_settle = back = False
        for i in range(int(9.0 / DT)):
            t = i * DT
            c = (0.3, 0.0, 0.0) if 0.5 <= t < 2.5 else (0.0, 0.0, 0.0)
            info = be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
            if info["phase"] == "settle":
                seen_settle = True
                self.assertEqual(info["wA"], 1.0)
                self.assertEqual(info["wP"], 1.0)
            if seen_settle and info["phase"] == "stand" and info["wL"] > 0.0:
                self.assertEqual(info["wP"], 1.0)        # legs still handing back: arms wait
            if seen_settle and info["phase"] == "stand" and info["wP"] == 0.0:
                back = True
            self.assertFalse(be.fallen())
        self.assertTrue(seen_settle)
        self.assertTrue(back)

    def test_settle_arms_person_is_the_old_behaviour(self):
        from k1live.backends.hybrid import HybridBackend

        be = HybridBackend(planner_async=False, settle_arms="person")
        try:
            wA = []
            for i in range(int(5.0 / DT)):
                c = (0.3, 0.0, 0.0) if 0.5 <= i * DT < 2.5 else (0.0, 0.0, 0.0)
                info = be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
                if info["phase"] == "settle":
                    wA.append(info["wA"])
            self.assertTrue(wA)
            self.assertEqual(wA[-1], 0.0)                # arms already back on the split side in SETTLE
        finally:
            be.close()

    def test_short_blip_does_not_walk(self):
        """A command shorter than enter_s (e.g. a noisy intent frame) never leaves STAND."""
        be = self.be
        for i in range(100):
            c = (0.3, 0.0, 0.0) if 20 <= i < 26 else (0.0, 0.0, 0.0)    # 0.12 s
            info = be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
            self.assertEqual(info["phase"], "stand")

    def test_handover_targets_continuous(self):
        """Joint targets change smoothly through STAND -> WALK -> STAND (no step at the switch ticks)."""
        be = self.be
        prev, jumps = None, []
        for i in range(int(9.0 / DT)):
            t = i * DT
            c = (0.3, 0.0, 0.0) if 0.5 <= t < 3.0 else (0.0, 0.0, 0.0)
            info = be.step(be.arm_default, loco_cmd=c, walk_cmd=c)
            if prev is not None and 0 < info["wL"] < 1:
                jumps.append(float(np.abs(info["target"] - prev).max()))
            prev = info["target"].copy()
        self.assertTrue(jumps)
        self.assertLess(max(jumps), 0.25)    # rad per 20 ms tick during the blends


@unittest.skipUnless(AVAILABLE, "hybrid backend files missing")
class TestHybridB(unittest.TestCase):
    def test_sonic_legs_stand_and_bow(self):
        from k1live.backends.hybrid import HybridBackend

        be = HybridBackend(legs="sonic", planner_async=False)
        try:
            self.assertTrue(be.supports["bow"])
            self.assertEqual(be.wL, 1.0)
            for i in range(int(4.0 / DT)):
                info = be.step(_arms(be, i), bow=0.3 if i > 50 else 0.0, waist_yaw=0.2)
                self.assertFalse(be.fallen())
            self.assertEqual(info["phase"], "stand")
            self.assertEqual(info["wA"], 0.0)
        finally:
            be.close()


@unittest.skipUnless(AVAILABLE, "hybrid backend files missing")
class TestServerWiring(unittest.TestCase):
    def test_available_and_limiter_preset(self):
        from k1live import protocol as P
        from k1live.backends.hybrid import HybridBackend
        from k1live.sim_server import WalkLimiter, available_backends

        self.assertIn("hybrid", P.BACKENDS)
        self.assertIn("hybrid", available_backends())
        wl = WalkLimiter(HybridBackend.limiter_preset)
        out = None
        for _ in range(3):                 # no loco dead band: a small sideways command is not lifted to 0.4 m/s
            out = wl(np.array([0.0, 0.15, 0.0]))
        self.assertLess(out[1], 0.2)


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@unittest.skipUnless(AVAILABLE, "hybrid backend files missing")
class TestServerTicks(unittest.TestCase):
    def test_drive_walks_and_stops(self):
        import zmq

        from k1live import protocol as P
        from k1live.sim_server import SimServer, parse_args

        ctx = zmq.Context()
        srv = None
        try:
            st = ctx.socket(zmq.PUB)
            st.setsockopt(zmq.LINGER, 0)
            st.bind(f"tcp://127.0.0.1:{_free_port()}")
            args = parse_args(["--backend", "hybrid", "--no-render", "--no-attract", "--human-port", str(_free_port()),
                               "--ctrl-port", str(_free_port()), "--log-dir",
                               str(ROOT / "logs" / "k1live" / "test_hybrid")])
            srv = SimServer(args, ctx, st)
            srv.be.kp.stream.async_ = False         # deterministic in the test
            wall = 1000.0
            phases = []
            for i in range(int(8.0 / DT)):
                wall += DT
                if 1.0 <= i * DT < 4.0:
                    srv.on_ctrl(P.CtrlMsg(P.CMD_DRIVE, {"vx": 0.35, "vy": 0.0, "wz": 0.0}), wall)
                srv.tick(wall)
                phases.append(srv.last_info.get("phase"))
                self.assertFalse(srv.fallen)
            self.assertIn("walk", phases)
            self.assertIn("hybrid:", srv.sim_state(wall).note)
            self.assertGreater(srv.be.root_pose()[0], 0.4)
        finally:
            if srv is not None:
                srv.close()
            ctx.destroy(0)


if __name__ == "__main__":
    unittest.main()
