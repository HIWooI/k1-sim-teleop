"""K1 live core: protocol round trip, person -> K1 mapping, limiters, and a headless 30 s end-to-end run
(sender on a video file as a fake camera + sim_server + render, on free ports; MuJoCo simulation only).

    .venv/bin/python -m unittest tests.test_k1live_core -v            # stdlib unittest (pytest also works)
    K1LIVE_E2E_SECONDS=10 .venv/bin/python -m unittest tests.test_k1live_core -v

The end-to-end test needs the event clip (K1LIVE_TEST_VIDEO, default the waving clip under K1_EVENT_VIDEOS), the
K1 MJCF (AI_SAPIENS_ROOT) and the K1 policy ONNX files; it is skipped when any is missing. Never opens a camera.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k1live import protocol as P  # noqa: E402
from k1live.human import ARM_DEFAULT, HumanMapper, PoseObs, mirror_arms, mp_targets, torso_frame  # noqa: E402
from k1live.sim_server import ArmLimiter, leash, limit_cmd  # noqa: E402

EVENT = Path(os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))
VIDEO = Path(os.environ.get("K1LIVE_TEST_VIDEO", EVENT / "raw" / "pexels_10374292.mp4"))
K1_XML = Path(os.environ.get("AI_SAPIENS_ROOT", str(ROOT / "third_party" / "ai_sapiens"))) / \
    "ai_sapiens_description/mujoco/k1/k1.xml"
POLICIES = [ROOT / "k1/assets" / a / "exported/policy.onnx" for a in ("student_asset", "loco_armdriven_r2")]
HAVE_SIM = K1_XML.exists() and all(p.exists() for p in POLICIES)   # K1 MJCF + ROBOTIS policies (not in git)
E2E_S = float(os.environ.get("K1LIVE_E2E_SECONDS", "30"))

# URDF arm limits, ARM_JOINTS order (k1/assets/k1.urdf)
ARM_LO = np.array([-3.1416, 0.0, -3.1416, -0.7854, -3.1416, -3.1416, -3.1416, -3.1416, -0.7854, -3.1416])
ARM_HI = np.array([3.1416, 3.1416, 3.1416, 2.1817, 3.1416, 3.1416, 0.0, 3.1416, 2.1817, 3.1416])


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def synthetic_det(left_up=0.0, right_up=0.0):
    """MediaPipe-like detection of a person facing the camera (world: x = person's left, y down, z away)."""
    w = np.zeros((33, 3))
    w[11], w[12] = [0.18, -0.50, 0.0], [-0.18, -0.50, 0.0]          # shoulders
    w[23], w[24] = [0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]               # hips
    w[25], w[26], w[27], w[28] = [0.1, 0.45, 0], [-0.1, 0.45, 0], [0.1, 0.85, 0], [-0.1, 0.85, 0]
    for s, sh, el, wr, up in ((1, 11, 13, 15, left_up), (-1, 12, 14, 16, right_up)):
        a = np.radians(up)  # 0 = arm hanging, 90 = arm sideways, 180 = straight up (frontal plane)
        d = np.array([s * np.sin(a), np.cos(a), 0.0])
        w[el] = w[sh] + 0.28 * d
        w[wr] = w[el] + 0.25 * d
    img = np.zeros((33, 3))
    img[:, 0] = 0.5 + w[:, 0] * 0.4   # person's left = image right
    img[:, 1] = 0.6 + w[:, 1] * 0.4
    return {"world": w, "vis": np.full(33, 0.95), "image": img}


class TestProtocol(unittest.TestCase):
    def test_round_trip(self):
        msgs = [
            (P.TOPIC_HUMAN, P.HumanFrame(seq=3, t_capture=1.5, present=True, conf=0.9, arm_q=[0.1] * 10,
                                         waist_yaw=0.1, bow=0.0, root={"x": 0.1, "y": 0.0, "yaw": 0.0},
                                         loco={"vx": 0.2, "vy": 0.0, "wz": 0.0}, calibrated=True)),
            (P.TOPIC_CTRL, P.CtrlMsg("set", {"mode": "idle", "max_speed": {"vx": 0.4}})),
            (P.TOPIC_SIM_STATE, P.SimState(t=1.0, mode="live", backend="split", streaming=True, walk_enabled=True,
                                           estop=False, fallen=False, resets=0, pelvis_z=0.77, tilt_deg=1.0,
                                           loco_cmd=[0, 0, 0], human_present=True, human_age_ms=40.0, ctrl_hz=50.0,
                                           rt_factor=1.0, backends=["split"])),
            (P.TOPIC_SENDER_STATE, P.SenderState(t=1.0, cam_ok=True, source="camera:0", fps=30.0, present=True,
                                                 calibrated=True, pose_ms=25.0)),
        ]
        for topic, m in msgs:
            fr = P.encode(topic, m)
            self.assertEqual(fr[0], topic)
            self.assertEqual(P.decode(fr, type(m)), m)

    def test_joint_order(self):
        self.assertEqual(len(P.POLICY_JOINTS), 23)
        self.assertEqual(P.POLICY_JOINTS[13:], P.ARM_JOINTS)


@pytest.mark.needs_smpl
class TestHuman(unittest.TestCase):
    def test_mirror_involution_and_limits(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform(ARM_LO, ARM_HI)
            m = mirror_arms(q)
            np.testing.assert_allclose(mirror_arms(m), q)
            self.assertTrue(np.all(m >= ARM_LO - 1e-9) and np.all(m <= ARM_HI + 1e-9))
        np.testing.assert_allclose(mirror_arms(ARM_DEFAULT), ARM_DEFAULT)

    def test_same_retargeting_as_reel(self):
        try:
            import k1.run_video as rv
        except Exception as e:  # pragma: no cover
            self.skipTest(f"k1.run_video not importable: {e}")
        det = synthetic_det(40, 120)
        a, b = mp_targets(det["world"], det["vis"]), rv.mp_targets(det["world"], det["vis"])
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(torso_frame(a[0]), rv.torso_frame(b[0]))

    def _arm(self, mapper, det, n=12):
        out = None
        for i in range(n):
            out = mapper.process(det, 100.0 + i / 30, 1280, 720)
        return np.array(out["arm_q"]), out

    def test_ik_range_and_mirror(self):
        plain = HumanMapper(mirror=False, loco=False)
        mirr = HumanMapper(mirror=True, loco=False)
        for lu, ru in ((0, 0), (90, 0), (170, 30), (60, 150)):
            det = synthetic_det(lu, ru)
            q, out = self._arm(plain, det)
            self.assertTrue(out["present"])
            self.assertEqual(q.shape, (10,))
            self.assertTrue(np.all(q >= ARM_LO - 1e-6) and np.all(q <= ARM_HI + 1e-6), q)
            qm, _ = self._arm(mirr, det)
            np.testing.assert_allclose(qm, mirror_arms(q), atol=0.05)
        # person's left arm up (no mirror) -> robot left shoulder pitch strongly negative (arm up), right arm down
        q, _ = self._arm(HumanMapper(mirror=False, loco=False), synthetic_det(175, 0), n=30)
        self.assertLess(q[0], -1.5)
        self.assertGreater(q[5], -0.6)
        # sideways (90 deg) -> roll out on that side
        q, _ = self._arm(HumanMapper(mirror=False, loco=False), synthetic_det(90, 0), n=30)
        self.assertGreater(q[1], 1.0)

    def test_calibration_root_and_absent(self):
        m = HumanMapper(mirror=True, loco=False, calib_frames=5)
        for i in range(8):  # walking in (moving across the image): no calibration yet
            det = synthetic_det()
            det["image"] = det["image"].copy()
            det["image"][:, 0] += 0.03 * i
            out = m.process(det, 9.0 + i / 30, 1280, 720)
        self.assertFalse(out["calibrated"])
        det = synthetic_det()
        for i in range(6):
            out = m.process(det, 10.0 + i / 30, 1280, 720)
        self.assertTrue(out["calibrated"])
        self.assertAlmostEqual(out["root"]["y"], 0.0, places=3)
        det2 = synthetic_det()
        det2["image"] = det2["image"].copy()
        det2["image"][:, 0] += 0.05  # person moved to image right = their left -> mirrored: robot's right (y < 0)
        out = m.process(det2, 11.0, 1280, 720)
        self.assertLess(out["root"]["y"], -0.05)
        out = m.process(None, 11.1, 1280, 720)
        self.assertFalse(out["present"])
        self.assertIsNone(out["arm_q"])
        self.assertIsNone(out["loco"])

    def test_jump_gate_heals(self):
        from k1live.human import JumpGate

        g = JumpGate(max_jump=0.12, patience=4)
        far, near = synthetic_det(), synthetic_det()
        far["image"] = far["image"].copy()
        far["image"][:, 1] += 0.45  # a bogus first lock far from where the person really is
        self.assertIsNotNone(g(far))
        got = [g(near) is not None for _ in range(6)]
        self.assertEqual(got, [False, False, False, True, True, True])  # rejected 3x, then re-acquired
        self.assertIsNone(g(None))

    def test_obs_mapping(self):
        det = synthetic_det()
        from k1live.human import obs_from_det

        o = obs_from_det(det, 5.0, 640, 480)
        self.assertEqual(o.lm_img.shape, (33, 4))
        self.assertEqual(o["lm_world"].shape, (33, 4))
        self.assertEqual(o["t"], 5.0)
        self.assertEqual(PoseObs.mean([o, o]).lm_world.shape, (33, 4))


class TestLimiters(unittest.TestCase):
    def test_arm_limiter_bounds(self):
        lim = ArmLimiter(np.zeros(10))
        goal = np.linspace(-1.5, 1.5, 10)
        qs, vs = [lim.q.copy()], [lim.v.copy()]
        for _ in range(150):
            q, v = lim.step(goal)
            qs.append(q)
            vs.append(v)
        qs, vs = np.array(qs), np.array(vs)
        self.assertLessEqual(np.abs(vs).max(), 6.0 + 1e-9)
        self.assertLessEqual(np.abs(np.diff(vs, axis=0)).max() / 0.02, 80.0 + 1e-6)
        np.testing.assert_allclose(qs[-1], goal, atol=1e-3)
        overshoot = np.max(np.where(goal > 0, qs - goal, goal - qs), axis=0)
        self.assertLess(overshoot.max(), 0.02)

    def test_cmd_limits(self):
        cur = np.zeros(3)
        for _ in range(100):
            cur = limit_cmd(np.array([2.0, -1.0, 3.0]), cur, (0.6, 0.3, 0.6))
        np.testing.assert_allclose(cur, [0.6, -0.3, 0.6])
        n = 0
        while np.any(cur != 0) and n < 100:  # stop from full speed
            cur = limit_cmd(np.zeros(3), cur, (0.6, 0.3, 0.6))
            n += 1
        self.assertLessEqual(n * 0.02, 0.3)
        self.assertLessEqual(limit_cmd(np.array([-1.0, 0, 0]), np.array([-0.4, 0, 0]), (0.6, 0.3, 0.6))[0], -0.39)
        self.assertGreaterEqual(limit_cmd(np.array([-1.0, 0, 0]), np.array([-0.4, 0, 0]), (0.6, 0.3, 0.6))[0], -0.4)
        # at the arena edge, facing outward: forward is cut, backward is kept
        np.testing.assert_allclose(leash(np.array([0.5, 0, 0]), 3.5, 0.0, 0.0), [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(leash(np.array([-0.3, 0, 0.2]), 3.5, 0.0, 0.0), [-0.3, 0, 0.2])


def _sub(ctx, port, topic):
    import zmq

    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://127.0.0.1:{port}")
    s.setsockopt(zmq.SUBSCRIBE, topic)
    return s


@unittest.skipUnless(VIDEO.exists() and HAVE_SIM, "event clip, K1 MJCF or policy ONNX missing")
class TestEndToEnd(unittest.TestCase):
    """sender --source <clip> --loop + sim_server, headless, E2E_S seconds."""

    def test_e2e(self):
        import zmq

        ports = {k: free_port() for k in ("human", "ctrl", "sim", "snd", "cam", "robot")}
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        log_dir = ROOT / "logs" / "k1live" / f"test_e2e_{time.strftime('%Y%m%d-%H%M%S')}"
        ctx = zmq.Context()
        ctrl = ctx.socket(zmq.PUB)
        ctrl.setsockopt(zmq.LINGER, 0)
        ctrl.bind(f"tcp://127.0.0.1:{ports['ctrl']}")
        t0 = time.time()
        sim = subprocess.Popen([sys.executable, "-m", "k1live.sim_server", "--backend", "split",
                                "--human-port", str(ports["human"]), "--ctrl-port", str(ports["ctrl"]),
                                "--state-port", str(ports["sim"]), "--robot-port", str(ports["robot"]),
                                "--log-dir", str(log_dir), "--idle-after", "3"], cwd=ROOT, env=env)
        snd = subprocess.Popen([sys.executable, "-m", "k1live.sender", "--source", str(VIDEO), "--loop",
                                "--human-port", str(ports["human"]), "--ctrl-port", str(ports["ctrl"]),
                                "--state-port", str(ports["snd"]), "--http-port", str(ports["cam"]),
                                "--log", str(log_dir / "sender.jsonl")], cwd=ROOT, env=env)
        s_sim, s_snd = _sub(ctx, ports["sim"], P.TOPIC_SIM_STATE), _sub(ctx, ports["snd"], P.TOPIC_SENDER_STATE)
        poller = zmq.Poller()
        poller.register(s_sim, zmq.POLLIN)
        poller.register(s_snd, zmq.POLLIN)
        first = {}
        sims, snds = [], []
        drove = False
        try:
            while time.time() - t0 < E2E_S + 8:
                for s, _ in poller.poll(100):
                    fr = s.recv_multipart()
                    if fr[0] == P.TOPIC_SIM_STATE:
                        m = P.decode(fr, P.SimState)
                        first.setdefault("sim", time.time() - t0)
                        if m.note != "loading":
                            sims.append(m)
                    else:
                        m = P.decode(fr, P.SenderState)
                        first.setdefault("snd", time.time() - t0)
                        snds.append(m)
                run_s = (sims[-1].t - sims[0].t) if len(sims) > 1 else 0.0
                if sims and not drove and run_s > E2E_S * 0.5:  # operator drive for 3 s mid-run
                    for _ in range(30):
                        ctrl.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("drive", {"vx": 0.3, "vy": 0.0, "wz": 0.0})))
                        time.sleep(0.1)
                    drove = True
                if run_s >= E2E_S:
                    break
            cam = urllib.request.urlopen(f"http://127.0.0.1:{ports['cam']}/cam.jpg", timeout=5).read()
            rob = urllib.request.urlopen(f"http://127.0.0.1:{ports['robot']}/robot.jpg", timeout=5).read()
            ctrl.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("set", {"video": str(VIDEO), "loop": True})))
            time.sleep(0.5)
        finally:
            for p in (snd, sim):
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)
            codes = [p.wait(timeout=15) for p in (snd, sim)]
            ctx.destroy(0)
        self.assertEqual(codes, [0, 0])
        self.assertLess(first.get("sim", 99), 3.0)
        self.assertLess(first.get("snd", 99), 3.0)
        self.assertEqual(cam[:2], b"\xff\xd8")
        self.assertEqual(rob[:2], b"\xff\xd8")
        live = [m for m in sims if m.t - sims[0].t > 2.0]
        self.assertGreater(len(live), 5 * E2E_S)  # ~10 Hz state
        hz = np.array([m.ctrl_hz for m in live])
        rt = np.array([m.rt_factor for m in live])
        self.assertGreater(np.median(hz), 48.0)
        self.assertGreater(np.median(rt), 0.97)
        self.assertEqual(sum(m.resets for m in live[-1:]), 0)
        self.assertTrue(any(m.human_present for m in live))
        self.assertTrue(any(abs(m.loco_cmd[0]) > 0.2 for m in live))  # drive reached the robot
        self.assertTrue(any(s.cam_ok and s.present for s in snds))
        summ = json.loads((log_dir / "summary.json").read_text())
        print("\n[e2e]", json.dumps({"first_state_s": first, "hz_median": float(np.median(hz)),
                                     "rt_median": float(np.median(rt)), **summ}), flush=True)
        self.assertEqual(summ["falls"], 0)
        self.assertLess(summ["lat_ms"]["p95"], 250.0)  # loose here; the acceptance numbers are in the rehearsal
        self.assertGreater(summ["lat_ms"]["n"], 10 * E2E_S)


@unittest.skipUnless(HAVE_SIM, "K1 MJCF / policy ONNX missing")
class TestServerLogic(unittest.TestCase):
    """SimServer.tick driven in-process with a synthetic clock and HumanFrames (no sender, no render)."""

    @classmethod
    def setUpClass(cls):
        import zmq

        from k1live.sim_server import SimServer, parse_args

        cls.ctx = zmq.Context()
        st = cls.ctx.socket(zmq.PUB)
        st.bind(f"tcp://127.0.0.1:{free_port()}")
        args = parse_args(["--no-render", "--no-attract", "--human-port", str(free_port()), "--ctrl-port",
                           str(free_port()), "--log-dir", str(ROOT / "logs" / "k1live" / "test_logic")])
        cls.srv = SimServer(args, cls.ctx, st)
        cls.wall = 1000.0

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        cls.ctx.destroy(0)

    def run_for(self, seconds, hf_fn=None):
        out = []
        for _ in range(int(round(seconds / 0.02))):
            self.wall += 0.02
            if hf_fn is not None:
                hf = hf_fn(self.wall)
                if hf is not None:
                    self.srv.hf = hf
            self.srv.tick(self.wall)
            # walking command = the limiter output (srv.cmd also carries the small heading-hold yaw correction)
            out.append((self.wall, self.srv.walk_lim.cur.copy(), self.srv.lim.q.copy(), self.srv.fallen))
        return out

    def hf(self, t, present=True, arm=None, loco=None, seq=[0]):
        seq[0] += 1
        return {"seq": seq[0], "t_capture": t - 0.03, "present": present, "conf": 0.9,
                "arm_q": None if not present else list(arm if arm is not None else ARM_DEFAULT),
                "waist_yaw": 0.0, "bow": 0.0, "loco": loco, "_t_recv": t - 0.01}

    def test_logic(self):
        srv = self.srv
        srv.do_reset("test")
        srv.mode, srv.streaming, srv.walk, srv.estop = "live", True, True, False
        arm = ARM_DEFAULT.copy()
        arm[0] = -1.2  # left arm forward/up
        walk = {"vx": 0.5, "vy": 0.0, "wz": 0.0}
        # a person walking forward: arms follow, command ramps up and stays below the limit
        r = self.run_for(3.0, lambda t: self.hf(t, arm=arm, loco=walk))
        self.assertAlmostEqual(r[-1][1][0], 0.5, places=3)
        self.assertAlmostEqual(r[-1][2][0], -1.2, places=3)
        self.assertFalse(any(x[3] for x in r))
        # the person vanishes (present False): walking command 0 within 0.3 s, arms hold ~1 s, then return
        t_gone = self.wall
        r = self.run_for(3.0, lambda t: self.hf(t, present=False))
        t_zero = next(t for t, c, q, f in r if np.all(c == 0))
        self.assertLessEqual(t_zero - t_gone, 0.3 + 1e-6)
        q_at = {round(t - t_gone, 2): q[0] for t, c, q, f in r}
        self.assertAlmostEqual(q_at[0.9], -1.2, places=3)  # holding
        self.assertGreater(r[-1][2][0], -0.2)               # back toward the relaxed 0.2 at <= 1 rad/s
        # the sender dies (no frames at all): stale after 0.25 s, so also 0 within 0.3 s
        self.run_for(1.0, lambda t: self.hf(t, arm=arm, loco=walk))
        t_gone = self.wall
        r = self.run_for(1.0)
        self.assertLessEqual(next(t for t, c, q, f in r if np.all(c == 0)) - t_gone, 0.3 + 1e-6 + 0.26)
        # operator drive has priority over the person and times out 0.3 s after the last message
        srv.drive, srv.t_drive = np.array([0.0, 0.0, 0.5]), self.wall
        r = self.run_for(0.25, lambda t: self.hf(t, arm=arm, loco=walk))
        self.assertGreater(r[-1][1][2], 0.3)
        self.assertEqual(r[-1][1][0], 0.0)
        r = self.run_for(0.8, lambda t: self.hf(t, arm=arm, loco=walk))
        self.assertEqual(r[-1][1][2], 0.0)
        self.assertGreater(r[-1][1][0], 0.3)
        # estop: walking and following stop, person ignored until start
        srv.on_ctrl(P.CtrlMsg("estop"), self.wall)
        r = self.run_for(1.5, lambda t: self.hf(t, arm=arm, loco=walk))
        self.assertTrue(np.all(r[-1][1] == 0))
        self.assertGreater(r[-1][2][0], -0.2)
        srv.on_ctrl(P.CtrlMsg("start"), self.wall)
        r = self.run_for(1.0, lambda t: self.hf(t, arm=arm, loco=None))
        self.assertAlmostEqual(r[-1][2][0], -1.2, places=2)
        # a fall: knocked over -> fallen -> respawned standing within 3 s, resets counted
        resets = srv.resets
        d = srv.be.data
        d.qpos[2] = 0.25
        d.qpos[3:7] = [np.cos(0.8), np.sin(0.8), 0.0, 0.0]
        r = self.run_for(3.0, lambda t: self.hf(t, arm=arm, loco=None))
        i_fall = next(i for i, x in enumerate(r) if x[3])
        i_up = next(i for i, x in enumerate(r) if i > i_fall and not x[3])
        self.assertLessEqual((i_up - i_fall) * 0.02, 3.0)
        self.assertEqual(srv.resets, resets + 1)
        self.assertFalse(srv.be.fallen())
        # heading hold: waving one arm for 10 s (spins the plain split robot ~40 deg) keeps the heading
        srv.do_reset("test")
        wave = ARM_DEFAULT.copy()
        wave[0], wave[3] = -2.4, 0.6
        yaw0 = srv.be.root_pose()[2]
        for i in range(500):
            w = wave.copy()
            w[1] = 0.3 + 0.35 * np.sin(2 * np.pi * 1.1 * i * 0.02)
            self.run_for(0.02, lambda t: self.hf(t, arm=w, loco=None))
        self.assertLess(abs(np.degrees(srv.be.root_pose()[2] - yaw0)), 8.0)
        # idempotent / partial set
        ms = srv.walk_lim.max_speed
        srv.on_ctrl(P.CtrlMsg("set", {"mode": "live", "walk": True, "backend": "split"}), self.wall)
        self.assertEqual(srv.walk_lim.max_speed, ms)
        self.assertEqual(srv.backend_name, "split")
        self.assertIsNone(srv.pending_backend)
        srv.on_ctrl(P.CtrlMsg("set", {"max_speed": {"vx": 0.3}}), self.wall)
        self.assertAlmostEqual(srv.walk_lim.max_speed, 0.3)
        srv.on_ctrl(P.CtrlMsg("set", {"max_speed": 0.6}), self.wall)
        self.assertAlmostEqual(srv.walk_lim.scale, 1.0)


@unittest.skipUnless(HAVE_SIM, "K1 MJCF / policy ONNX missing")
class TestSplitBackend(unittest.TestCase):
    def test_walk_command_and_trim(self):
        from k1live.backends.split import SplitBackend

        errs = {}
        for gain in (0.0, 2.0):
            be = SplitBackend(trim_gain=gain)
            goal = be.arm_default.copy()
            goal[0], goal[5] = -1.2, -1.2  # both arms forward
            x0 = be.root_pose()[0]
            for k in range(200):
                be.step(goal, arm_dq=np.zeros(10), loco_cmd=(0.4, 0.0, 0.0) if k >= 50 else (0.0, 0.0, 0.0))
                self.assertFalse(be.fallen())
            errs[gain] = float(np.degrees(np.mean(np.abs(be.arm_q_now() - goal))))
            self.assertGreater(be.root_pose()[0] - x0, 0.6)  # walked forward ~3 s at 0.4 m/s
            be.reset()
            self.assertTrue(np.all(be.trim == 0))
            self.assertLess(abs(be.root_pose()[0]), 1e-6)
        self.assertLess(errs[2.0], 0.6 * errs[0.0], errs)
        self.assertLess(errs[2.0], 4.0, errs)


if __name__ == "__main__":
    unittest.main()
