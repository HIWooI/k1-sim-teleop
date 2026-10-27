"""Presenter mode (docs/architecture.md): the "presenter" walking profile, the side-on distance hold, re-homing
when the single-person tracker jumps to someone else, the sender's --crop / --loco-profile. stdlib unittest, numpy
only (synthetic landmarks from tests/test_k1live_walk.py; no MuJoCo, no MediaPipe, no camera):

    .venv/bin/python -m unittest tests.test_k1live_presenter -v
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from k1live.locomotion import LocoEstimator  # noqa: E402
from test_k1live_walk import FPS, first, make_obs, smooth_step  # noqa: E402


def drive(scenario, T, profile, seed=0, **params):
    """Auto-home (as live: no operator calibration), update() every frame -> (t, cmd (N, 3), est)."""
    rng = np.random.default_rng(seed)
    est = LocoEstimator(mirror=True, profile=profile, **params)
    ts, cs = [], []
    for t in np.arange(0.0, T, 1.0 / FPS):
        s = scenario(t)
        if s is None:
            continue
        c = est.update(make_obs(t, rng=rng, **s))
        ts.append(t)
        cs.append([c["vx"], c["vy"], c["wz"]])
    return np.array(ts), np.array(cs), est


def sway(t):
    return 0.01 * math.sin(2 * math.pi * 0.3 * t)


class TestProfiles(unittest.TestCase):
    def test_visitor_is_the_old_default(self):
        self.assertEqual(LocoEstimator().p, LocoEstimator.DEFAULTS)
        self.assertEqual(LocoEstimator(profile="visitor").p, LocoEstimator.DEFAULTS)

    def test_explicit_params_override_the_profile(self):
        est = LocoEstimator(profile="presenter", home_follow_s=1.5, side_start=0.4)
        self.assertEqual(est.p["home_follow_s"], 1.5)
        self.assertEqual(est.p["side_start"], 0.4)
        self.assertFalse(est.p["turn"])
        with self.assertRaises(ValueError):
            LocoEstimator(profile="dancer")
        with self.assertRaises(TypeError):
            LocoEstimator(profile="presenter", nonsense=1)


class TestPresenterTurning(unittest.TestCase):
    """A presenter turns 60 deg to the slides and stays turned 5 s: the visitor joystick spins the robot the whole
    time, the presenter profile never turns it."""

    @staticmethod
    def scen(t):
        yaw = math.radians(60) * (smooth_step(t, 3.0, 3.6) - smooth_step(t, 8.6, 9.2))
        return dict(X=0.1 + sway(t), Z=4.0, yaw=yaw, arm_r=(0.4, 1.2))

    def test_visitor_spins(self):
        t, c, _ = drive(self.scen, 12.0, "visitor")
        self.assertGreater(np.sum(c[:, 2] != 0) / FPS, 4.0)

    def test_presenter_does_not(self):
        for seed in (0, 1, 2):
            t, c, _ = drive(self.scen, 12.0, "presenter", seed=seed)
            self.assertTrue(np.all(c == 0.0), f"seed {seed}: command at t={first(t, np.any(c != 0, 1))}")


class TestPresenterWalkAlong(unittest.TestCase):
    """Strolls 1.2 m to their left at 0.7 m/s, then stays there: the robot walks (mirror: to its right) while the
    presenter walks and stops on its own ~1 s later; the visitor joystick keeps walking until they come back."""

    @staticmethod
    def scen(t):
        return dict(X=0.0 + 1.2 * np.clip((t - 3.0) / (1.2 / 0.7), 0.0, 1.0) + sway(t), Z=4.0, yaw=0.0)

    def test_presenter_walks_while_walking(self):
        t_end = 3.0 + 1.2 / 0.7
        for seed in (0, 1, 2):
            t, c, _ = drive(self.scen, 12.0, "presenter", seed=seed)
            on = first(t, c[:, 1] != 0)
            self.assertIsNotNone(on, f"seed {seed}: never walked")
            self.assertLess(on - 3.0, 1.5)
            self.assertTrue(np.all(c[:, 1] <= 0.0), "mirror: the person's left is the robot's right")
            off = first(t, (t > on) & (c[:, 1] == 0))
            self.assertLess(off - t_end, 2.0)
            self.assertTrue(np.all(c[t > off + 0.05] == 0.0), f"seed {seed}: walked again while standing")
            self.assertTrue(np.all(c[:, [0, 2]] == 0.0))

    def test_visitor_keeps_walking(self):
        t, c, _ = drive(self.scen, 12.0, "visitor")
        self.assertTrue(np.all(c[t > 8.0, 1] != 0.0))


class TestSideOnDistanceHold(unittest.TestCase):
    """MediaPipe's side-view world landmarks make a presenter turned to the board look 0.3-0.6 m closer (real clip
    p11). Simulated as a distance change while the body is turned 60 deg: held, and no step when they face the
    camera again. Without the hold (visitor) it walks forward."""

    @staticmethod
    def scen(t):
        turned = smooth_step(t, 3.0, 3.5) - smooth_step(t, 8.0, 8.5)
        z = 3.5 - 0.6 * smooth_step(t, 3.5, 4.5) + 0.6 * smooth_step(t, 7.0, 8.0)
        return dict(X=0.0 + sway(t), Z=z, yaw=math.radians(60) * turned)

    def test_presenter_holds(self):
        for seed in (0, 1):
            t, c, _ = drive(self.scen, 12.0, "presenter", seed=seed)
            self.assertTrue(np.all(c == 0.0), f"seed {seed}: command at t={first(t, np.any(c != 0, 1))}")

    def test_without_hold_walks(self):
        t, c, _ = drive(self.scen, 12.0, "visitor", turn=False)
        self.assertTrue(np.any(c[:, 0] > 0.0))

    def test_rebase_after_turning_back(self):
        # a presenter-like estimator that would see the 0.6 m jump at the moment they face the camera again
        def scen(t):
            turned = smooth_step(t, 3.0, 3.5) - smooth_step(t, 6.0, 6.3)
            z = 3.5 - 0.6 * smooth_step(t, 3.5, 4.5)
            return dict(X=sway(t), Z=z, yaw=math.radians(60) * turned)

        t, c, est = drive(scen, 10.0, "presenter", follow_engaged=False)
        self.assertTrue(np.all(c == 0.0), f"command at t={first(t, np.any(c != 0, 1))}")
        self.assertLess(abs(est.offset["fwd"]), 0.15)


class TestRelock(unittest.TestCase):
    def test_gate_flags_an_accepted_jump(self):
        from k1live.human import JumpGate
        from test_k1live_core import synthetic_det

        g = JumpGate(max_jump=0.12, patience=4)
        a, b = synthetic_det(), synthetic_det()
        b["image"] = b["image"].copy()
        b["image"][:, 0] += 0.35            # someone else, 0.35 of the frame to the side
        self.assertIsNotNone(g(a))
        self.assertFalse(g.relocked)
        flags = []
        for _ in range(5):
            g(b)
            flags.append(g.relocked)
        self.assertEqual(flags, [False, False, False, True, False])
        g(None)
        self.assertFalse(g.relocked)

    @pytest.mark.needs_smpl
    def test_mapper_rehomes_walking(self):
        from k1live.human import HumanMapper
        from test_k1live_core import synthetic_det

        hm = HumanMapper(loco=True, auto_calibrate=False)
        if not hm.loco.available:
            self.skipTest("no k1live.locomotion")
        est = hm.loco.est
        est.home, est.calibrated = (3.0, 0.0, 0.0, 0.0), True
        hm.process(synthetic_det(), 0.0, 1280, 720)
        hm.on_relock()
        self.assertIsNone(est.home)
        self.assertFalse(est.calibrated)


class TestSenderArgs(unittest.TestCase):
    def test_crop_and_profile(self):
        from k1live.sender import parse_args, parse_crop

        self.assertEqual(parse_crop("0.1,0,0.9,0.7"), (0.1, 0.0, 0.9, 0.7))
        for bad in ("0.5,0,0.4,1", "0,0,1", "a,b,c,d", "0,0,1.2,1", "0.5,0.5,0.55,1"):
            self.assertIsNone(parse_crop(bad), bad)
        self.assertIsNone(parse_crop(""))
        a = parse_args(["--crop", "0.2,0,0.8,0.75", "--loco-profile", "presenter"])
        self.assertEqual(a.crop_box, (0.2, 0.0, 0.8, 0.75))
        self.assertEqual(a.loco_profile, "presenter")
        a = parse_args([])
        self.assertIsNone(a.crop_box)
        self.assertEqual(a.loco_profile, "visitor")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse_args(["--crop", "0.9,0,0.1,1"])

    def test_environment_defaults(self):
        import os
        from unittest import mock

        from k1live.sender import parse_args

        with mock.patch.dict(os.environ, {"K1LIVE_CROP": "0.1,0.0,0.9,0.8", "K1LIVE_LOCO_PROFILE": "presenter"}):
            a = parse_args([])
            self.assertEqual((a.crop_box, a.loco_profile), ((0.1, 0.0, 0.9, 0.8), "presenter"))
            self.assertIsNone(parse_args(["--crop", ""]).crop_box)          # explicit "" = whole frame
        # a bad value from the service's environment must not stop the sender
        with mock.patch.dict(os.environ, {"K1LIVE_CROP": "nonsense", "K1LIVE_LOCO_PROFILE": "dancer"}), \
                contextlib.redirect_stderr(io.StringIO()):
            a = parse_args([])
            self.assertEqual((a.crop_box, a.loco_profile), (None, "visitor"))


class TestLiveSwitch(unittest.TestCase):
    """The operator switches the walking mode / person region while the sender runs (CTRL set)."""

    @pytest.mark.needs_smpl
    def test_mapper_profile_switch(self):
        from k1live.human import HumanMapper

        hm = HumanMapper(loco=True, auto_calibrate=False)
        if not hm.loco.available:
            self.skipTest("no k1live.locomotion")
        self.assertEqual(hm.loco.est.profile, "visitor")
        hm.set_loco_profile("presenter")
        self.assertEqual((hm.loco_profile, hm.loco.est.profile), ("presenter", "presenter"))
        self.assertFalse(hm.loco.est.p["turn"])
        self.assertIsNone(hm.loco.est.home)                 # a fresh estimator homes again on its own

    def test_crop_from_ctrl_list(self):
        from k1live.sender import parse_crop

        self.assertEqual(parse_crop([0.2, 0, 0.8, 0.85]), (0.2, 0.0, 0.8, 0.85))
        for whole in (None, "", []):
            self.assertIsNone(parse_crop(whole))
        for bad in ([0.2, 0, 0.8], ["a", 0, 1, 1], [0, 0, 1, 1.5], {"x": 1}):
            self.assertIsNone(parse_crop(bad), bad)

    def test_sender_applies_set(self):
        """A real sender on a video (fake camera): CTRL set loco_profile / crop shows up in its /state at once; the
        same values again are a no-op; a bad region is ignored with a note."""
        import json
        import os
        import socket
        import subprocess
        import time
        import urllib.request

        import zmq

        from k1live import protocol as P

        ev = Path(os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))
        video = Path(os.environ.get("K1LIVE_TEST_VIDEO", ev / "presenter" / "pexels_6952136.mp4"))
        if not video.exists() or not (ROOT / "models" / "pose_landmarker_full.task").exists():
            self.skipTest("presenter clip or MediaPipe model missing")

        def free():
            with socket.socket() as so:
                so.bind(("127.0.0.1", 0))
                return so.getsockname()[1]

        ports = {k: free() for k in ("human", "state", "ctrl", "http")}
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://127.0.0.1:{ports['ctrl']}")
        proc = subprocess.Popen([sys.executable, "-m", "k1live.sender", "--source", str(video), "--loop",
                                 "--human-port", str(ports["human"]), "--state-port", str(ports["state"]),
                                 "--ctrl-port", str(ports["ctrl"]), "--http-port", str(ports["http"])],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env={k: v for k, v in os.environ.items()
                                     if k not in ("K1LIVE_LOCO_PROFILE", "K1LIVE_CROP")})
        try:
            def state():
                with urllib.request.urlopen(f"http://127.0.0.1:{ports['http']}/state", timeout=2) as r:
                    return json.loads(r.read())

            def wait(pred, timeout=20.0):
                end = time.time() + timeout
                while time.time() < end:
                    try:
                        s = state()
                        if pred(s):
                            return s
                    except OSError:
                        pass
                    time.sleep(0.2)
                return None

            s0 = wait(lambda s: s.get("note") != "loading" and s.get("fps", 0) > 0)
            self.assertIsNotNone(s0, "sender never came up")
            self.assertEqual((s0["loco_profile"], s0["crop"]), ("visitor", None))
            box = [0.0, 0.0, 0.55, 1.0]
            for _ in range(3):   # PUB/SUB: repeat like the service's re-sync would
                pub.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("set", {"loco_profile": "presenter",
                                                                            "crop": box})))
                time.sleep(0.3)
            s1 = wait(lambda s: s["loco_profile"] == "presenter" and s["crop"] == box, 5.0)
            self.assertIsNotNone(s1, f"not applied: {state()}")
            s2 = wait(lambda s: s["present"], 10.0)           # still tracking someone inside the region
            self.assertIsNotNone(s2)
            pub.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("set", {"crop": [0.9, 0, 0.1, 1]})))
            s3 = wait(lambda s: "영역 무시" in s.get("note", ""), 5.0)
            self.assertIsNotNone(s3)
            self.assertEqual(s3["crop"], box)
            pub.send_multipart(P.encode(P.TOPIC_CTRL, P.CtrlMsg("set", {"crop": None, "loco_profile": "visitor"})))
            self.assertIsNotNone(wait(lambda s: s["crop"] is None and s["loco_profile"] == "visitor", 5.0))
        finally:
            proc.terminate()
            out = proc.communicate(timeout=10)[0]
            pub.close(0)
            ctx.term()
        self.assertEqual(out.count("person region"), 2, out[-2000:])   # region on, region off: no repeats
        self.assertEqual(out.count("walking profile presenter"), 1, out[-2000:])


if __name__ == "__main__":
    unittest.main()
