"""Walking intent (k1live/locomotion.py) on synthetic MediaPipe landmarks with known ground truth, plus the
controller-side velocity limiter. stdlib unittest, numpy only (no MuJoCo, no MediaPipe):

    .venv/bin/python -m unittest tests.test_k1live_walk -v

The synthetic person is a 33-point MediaPipe-like skeleton (metres) placed in front of a pinhole camera
(1280x720, 65 deg horizontal FOV, 1.2 m high, level): lm_world = camera-aligned hip-centred points (x right,
y down, z away from the camera, like pose_world_landmarks), lm_img = their projection normalised to the image.
Gaussian noise on both (2 px, 1 cm) plus a slow sway.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k1live.locomotion import LocoEstimator, VelocityLimiter, measure  # noqa: E402

IMG_W, IMG_H, HFOV, CAM_H, HIP_H = 1280, 720, 65.0, 1.2, 0.92
F_PX = 0.5 * IMG_W / math.tan(math.radians(HFOV) / 2)
FPS = 30.0

# person frame: x = the person's left (image right when facing the camera), y down, z away from the camera
# (= behind the person); origin = hip centre
_BASE = {
    0: (0.0, -0.66, -0.09), 1: (0.02, -0.69, -0.08), 2: (0.03, -0.69, -0.08), 3: (0.045, -0.69, -0.07),
    4: (-0.02, -0.69, -0.08), 5: (-0.03, -0.69, -0.08), 6: (-0.045, -0.69, -0.07), 7: (0.075, -0.67, -0.01),
    8: (-0.075, -0.67, -0.01), 9: (0.025, -0.62, -0.08), 10: (-0.025, -0.62, -0.08),
    11: (0.18, -0.50, 0.0), 12: (-0.18, -0.50, 0.0), 23: (0.10, 0.0, 0.0), 24: (-0.10, 0.0, 0.0),
    25: (0.10, 0.44, -0.01), 26: (-0.10, 0.44, -0.01), 27: (0.10, 0.84, 0.03), 28: (-0.10, 0.84, 0.03),
    29: (0.10, 0.88, 0.07), 30: (-0.10, 0.88, 0.07), 31: (0.11, 0.90, -0.10), 32: (-0.11, 0.90, -0.10),
}
UPPER = list(range(0, 23))
ARM_UP, ARM_LO = 0.28, 0.26


def _rot_y(a):
    """Rotation about the vertical (y down): + = the person turns to their own left (front swings to +x)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _rot_x(a):
    """Rotation about the person's left axis: + = bow forward (head toward the camera, -z)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def skeleton(arm_l=(0.0, 0.0), arm_r=(0.0, 0.0), elbow=(0.0, 0.0), bow=0.0, twist=0.0, shift=0.0, crouch=0.0):
    """33 points in the person frame. arm_*: (abduction, forward flexion) rad of the upper arm; elbow: flexion
    (rad) per side; bow: torso pitch about the hips; twist: upper-body yaw relative to the hips; shift: lateral
    weight shift of the hips/torso (m, feet stay); crouch: hips lowered (m, knees bend forward)."""
    P = np.zeros((33, 3))
    for i, p in _BASE.items():
        P[i] = p
    for k, (side, (abd, flex), el) in enumerate((("l", arm_l, elbow[0]), ("r", arm_r, elbow[1]))):
        sh = P[11 if side == "l" else 12]
        sx = 1.0 if side == "l" else -1.0
        # upper arm hanging down (+y), abducted outwards (+-x) and flexed forward (-z)
        d = np.array([sx * math.sin(abd) * math.cos(flex), math.cos(abd) * math.cos(flex), -math.sin(flex)])
        d /= np.linalg.norm(d)
        el_p = sh + ARM_UP * d
        # forearm: upper-arm direction bent forward (-z) by the elbow flexion
        fwd = np.array([0.0, 0.0, -1.0])
        fwd = fwd - d * (fwd @ d)
        fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
        fd = math.cos(el) * d + math.sin(el) * fwd
        wr = el_p + ARM_LO * fd
        ie, iw = (13, 15) if side == "l" else (14, 16)
        P[ie], P[iw] = el_p, wr
        for j, off in zip((17, 19, 21) if side == "l" else (18, 20, 22), (0.06, 0.07, 0.04)):
            P[j] = wr + off * fd + np.array([sx * 0.01, 0.0, 0.0])
    # upper body: twist about the vertical then bow about the hip axis
    P[UPPER] = P[UPPER] @ _rot_y(twist).T
    P[UPPER] = P[UPPER] @ _rot_x(bow).T
    P[[23, 24]] += 0.0
    upper_and_hips = UPPER + [23, 24]
    P[upper_and_hips, 0] += shift
    P[upper_and_hips, 1] += crouch
    P[[25, 26], 0] += 0.5 * shift
    P[[25, 26], 1] += 0.5 * crouch
    P[[25, 26], 2] -= 0.8 * crouch
    return P


def make_obs(t, X, Z, yaw=0.0, rng=None, noise=True, vis=None, **pose):
    """Camera-frame person at lateral X (m, image right +), distance Z (m), body yaw (+ = turned to the
    person's left) -> obs dict for LocoEstimator."""
    P = skeleton(**pose) @ _rot_y(yaw).T          # person frame -> camera axes (facing the camera at yaw 0)
    hip = 0.5 * (P[23] + P[24])
    world = P - hip
    cam = world + np.array([X, CAM_H - HIP_H, Z])
    if noise and rng is not None:
        cam_img = cam + rng.normal(0, 0.004, cam.shape)
    else:
        cam_img = cam
    u = F_PX * cam_img[:, 0] / cam_img[:, 2] + 0.5 * IMG_W
    v = F_PX * cam_img[:, 1] / cam_img[:, 2] + 0.5 * IMG_H
    if noise and rng is not None:
        u = u + rng.normal(0, 2.0, u.shape)
        v = v + rng.normal(0, 2.0, v.shape)
        world = world + rng.normal(0, 0.01, world.shape)
    vv = np.ones(33) if vis is None else vis
    lm_img = np.c_[u / IMG_W, v / IMG_H, cam[:, 2] - cam[:, 2].mean(), vv]
    lm_world = np.c_[world, vv]
    return {"t": float(t), "img_w": IMG_W, "img_h": IMG_H, "lm_img": lm_img, "lm_world": lm_world}


def smooth_step(t, t0, t1):
    u = np.clip((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def run(scenario, T, mirror=True, seed=0, calib_t=1.0, **params):
    """scenario(t) -> dict(X, Z, yaw, pose kwargs). Calibrates like the sender: landmarks averaged over the
    first 0.5 s, then update() every frame. Returns (t, cmd (N,3), truth list)."""
    rng = np.random.default_rng(seed)
    est = LocoEstimator(mirror=mirror, **params)
    ts = np.arange(0.0, T, 1.0 / FPS)
    window = []
    out_t, out_c, truth = [], [], []
    calibrated = False
    for t in ts:
        s = scenario(t)
        if s is None:   # person not tracked: the sender does not call update()
            continue
        obs = make_obs(t, rng=rng, **s)
        if not calibrated:
            window.append(obs)
            if t >= calib_t:
                avg = dict(obs)
                avg["lm_img"] = np.mean([o["lm_img"] for o in window[-15:]], axis=0)
                avg["lm_world"] = np.mean([o["lm_world"] for o in window[-15:]], axis=0)
                est.calibrate(avg)
                calibrated = True
            continue
        c = est.update(obs)
        out_t.append(t)
        out_c.append([c["vx"], c["vy"], c["wz"]])
        truth.append(s)
    return np.array(out_t), np.array(out_c), truth, est


def still(**pose):
    def f(t):
        sway = 0.01 * math.sin(2 * math.pi * 0.3 * t)
        return dict(X=0.05 + sway, Z=2.6, yaw=math.radians(2.0) * math.sin(0.7 * t), **pose)
    return f


def first(t, mask):
    i = np.flatnonzero(mask)
    return float(t[i[0]]) if len(i) else None


class TestMeasure(unittest.TestCase):
    def test_distance_lateral_yaw(self):
        for X, Z, yaw in ((0.0, 2.5, 0.0), (0.4, 3.0, 0.3), (-0.3, 1.8, -0.5)):
            m = measure(make_obs(0.0, X, Z, yaw, noise=False), hfov_deg=HFOV)
            self.assertAlmostEqual(m.z, Z, delta=0.05 * Z)
            self.assertAlmostEqual(m.x, X, delta=0.03)
            self.assertAlmostEqual(m.yaw, yaw, delta=0.02)

    def test_scale_ignores_bow_arms_twist(self):
        ref = measure(make_obs(0.0, 0.0, 2.5, noise=False))
        for pose in (dict(bow=0.6), dict(arm_l=(2.8, 0.0), arm_r=(2.8, 0.0)), dict(twist=0.4),
                     dict(arm_l=(0.3, 1.2), arm_r=(0.3, 1.2), elbow=(1.2, 1.2)), dict(crouch=0.15)):
            m = measure(make_obs(0.0, 0.0, 2.5, noise=False, **pose))
            self.assertAlmostEqual(m.z, ref.z, delta=0.06, msg=str(pose))

    def test_hidden_torso_is_none(self):
        vis = np.ones(33)
        vis[[23, 24]] = 0.1
        self.assertIsNone(measure(make_obs(0.0, 0.0, 2.5, vis=vis, noise=False)))


class TestNoFalseWalk(unittest.TestCase):
    """Gestures in place: the command must stay exactly zero."""

    def check(self, scen, T=20.0, seeds=(0, 1, 2)):
        for seed in seeds:
            t, c, _, _ = run(scen, T, seed=seed)
            self.assertTrue(np.all(c == 0.0), f"seed {seed}: false walk at t={first(t, np.any(c != 0, 1))}")

    def test_standing(self):
        self.check(still())

    def test_bow(self):
        def f(t):
            s = still()(t)
            s["bow"] = 0.7 * float(smooth_step(t % 6.0, 1.5, 3.0) - smooth_step(t % 6.0, 4.0, 5.5))
            return s
        self.check(f)

    def test_arms_overhead_and_wave(self):
        def f(t):
            s = still()(t)
            up = 2.9 * float(smooth_step(t % 8, 1.5, 3.5) - smooth_step(t % 8, 5.5, 7.5))
            s["arm_l"] = (up, 0.0)
            s["arm_r"] = (1.4 + 0.5 * math.sin(2 * math.pi * 1.5 * t), 0.0)
            s["elbow"] = (0.0, 1.0 + 0.5 * math.sin(2 * math.pi * 1.5 * t))
            return s
        self.check(f)

    def test_clap_reach_forward(self):
        def f(t):
            s = still()(t)
            a = 0.25 + 0.15 * math.sin(2 * math.pi * 2.0 * t)
            s["arm_l"] = (-a, 1.3)
            s["arm_r"] = (-a, 1.3)
            s["elbow"] = (0.5, 0.5)
            return s
        self.check(f)

    def test_weight_shift_and_crouch(self):
        def f(t):
            s = still()(t)
            s["shift"] = 0.08 * math.sin(2 * math.pi * 0.25 * t)
            s["crouch"] = 0.12 * float(smooth_step(t % 7, 2, 3) - smooth_step(t % 7, 4.5, 5.5))
            return s
        self.check(f)

    def test_slow_creep_forward(self):
        """Shuffling toward the camera at 5 cm/s for 10 s (0.5 m in total, like the waving clip): no walk."""
        def f(t):
            s = still()(t)
            s["Z"] = 2.6 - 0.05 * max(t - 1.5, 0.0)
            return s
        self.check(f, T=11.5)

    def test_small_drift_and_upper_twist(self):
        def f(t):
            s = still()(t)
            s["Z"] = 2.6 - 0.12 * math.sin(2 * math.pi * 0.05 * t)
            s["X"] = 0.05 + 0.08 * math.sin(2 * math.pi * 0.07 * t)
            s["yaw"] = math.radians(10) * math.sin(2 * math.pi * 0.1 * t)
            return s
        self.check(f, T=30.0)


def excursion(axis, amount, speed, t0=2.0, hold=3.0, Z0=2.6):
    """Walk `amount` along axis (Z: toward the camera = -, X: lateral, yaw) at `speed`, hold, walk back home."""
    dur = abs(amount) / speed
    t1, t2, t3 = t0 + dur, t0 + dur + hold, t0 + 2 * dur + hold

    def off(t):
        if t < t0:
            return 0.0
        if t < t1:
            return amount * (t - t0) / dur
        if t < t2:
            return amount
        if t < t3:
            return amount * (1 - (t - t2) / dur)
        return 0.0

    def f(t):
        s = dict(X=0.0, Z=Z0, yaw=0.0)
        bob = 0.03 * math.sin(2 * math.pi * 1.8 * t) if t0 < t < t3 else 0.0
        if axis == "Z":
            s["Z"] = Z0 - off(t)
            s["yaw"] = math.radians(6) * math.sin(2 * math.pi * 0.9 * t) if t0 < t < t3 else 0.0
        elif axis == "X":
            s["X"] = off(t)
        else:
            s["yaw"] = off(t)
        s["X"] += bob * 0.3
        return s

    return f, off, (t0, t1, t2, t3)


class TestWalkIntent(unittest.TestCase):
    def latencies(self, axis, amount, speed, col, thr, mirror=True):
        f, off, (t0, t1, t2, t3) = excursion(axis, amount, speed)
        t, c, _, _ = run(f, t3 + 4.0, mirror=mirror)
        ts = np.arange(t0, t3, 0.001)
        o = np.array([abs(off(x)) for x in ts])
        t_cross_out = ts[np.argmax(o >= thr)]
        t_on = first(t, np.abs(c[:, col]) > 0)
        on = c[np.abs(c[:, col]) > 0, col]
        t_off = first(t[t > t2], c[t > t2, col] == 0)
        return t_on - t_cross_out, t_off - t3, np.sign(on.mean()), c

    def test_forward(self):
        start, stop, sgn, c = self.latencies("Z", 0.55, 0.8, 0, 0.40)
        self.assertLess(start, 0.5)
        self.assertLess(stop, 1.0)
        self.assertEqual(sgn, 1.0)
        self.assertTrue(np.all(c[:, 1:] == 0), "forward walk leaked into vy/wz")

    def test_backward(self):
        start, stop, sgn, _ = self.latencies("Z", -0.55, 0.6, 0, 0.40)
        self.assertLess(start, 0.5)
        self.assertLess(stop, 1.0)
        self.assertEqual(sgn, -1.0)

    def test_sideways_mirror(self):
        # image right = the person's left; mirrored robot walks to its right (vy < 0)
        start, stop, sgn, _ = self.latencies("X", 0.5, 0.6, 1, 0.40, mirror=True)
        self.assertLess(start, 0.5)
        self.assertLess(stop, 1.0)
        self.assertEqual(sgn, -1.0)
        _, _, sgn2, _ = self.latencies("X", 0.5, 0.6, 1, 0.40, mirror=False)
        self.assertEqual(sgn2, 1.0)

    def test_turn_mirror(self):
        start, stop, sgn, _ = self.latencies("yaw", math.radians(45), math.radians(90), 2, math.radians(30))
        self.assertLess(start, 0.5)
        self.assertLess(stop, 1.0)
        self.assertEqual(sgn, -1.0)   # person turns left -> mirrored robot turns right
        _, _, sgn2, _ = self.latencies("yaw", math.radians(45), math.radians(90), 2, math.radians(30), mirror=False)
        self.assertEqual(sgn2, 1.0)

    def test_full_turn_keeps_direction(self):
        """Turning all the way round: no reversal while the back faces the camera (yaw wraps through 180 deg)."""
        f = lambda t: dict(X=0.0, Z=2.6, yaw=float(np.clip(t - 1.0, 0, None)) * math.radians(60))  # noqa: E731
        t, c, _, _ = run(f, 6.0, mirror=False)
        on = c[(t > 2.0) & (t < 4.6), 2]          # 60 .. 216 deg: across the wrap
        self.assertTrue(np.all(on > 0), on)

    def test_forget_home(self):
        rng = np.random.default_rng(0)
        est = LocoEstimator()
        est.calibrate(make_obs(0.0, 0.0, 2.6, noise=False))
        est.forget_home()
        self.assertFalse(est.calibrated)
        cmds = [est.update(make_obs(t, 0.5, 2.0, rng=rng)) for t in np.arange(0.0, 2.0, 1 / FPS)]
        self.assertTrue(all(c == {"vx": 0.0, "vy": 0.0, "wz": 0.0} for c in cmds))
        self.assertTrue(est.calibrated)
        self.assertAlmostEqual(est.home[0], 2.0, delta=0.15)

    def test_mirror_attribute_toggle(self):
        f, _, _ = excursion("X", 0.5, 0.6)
        rng = np.random.default_rng(0)
        est = LocoEstimator(mirror=True)
        est.calibrate(make_obs(0.0, rng=None, noise=False, **f(0.0)))
        vy = []
        for k, t in enumerate(np.arange(0.0, 5.0, 1 / FPS)):
            if k == 120:
                est.mirror = False
            vy.append(est.update(make_obs(t, rng=rng, **f(t)))["vy"])
        vy = np.array(vy)
        self.assertTrue(np.all(vy[100:120] < 0) and np.all(vy[121:140] > 0))

    def test_lost_person_stops(self):
        f, _, (t0, t1, t2, t3) = excursion("Z", 0.6, 0.8, hold=6.0)
        rng = np.random.default_rng(0)
        est = LocoEstimator()
        est.calibrate(make_obs(0.0, rng=None, noise=False, **f(0.0)))
        t_lost = t1 + 1.0
        res = []
        for t in np.arange(0.0, t2, 1 / FPS):
            obs = make_obs(t, rng=rng, **f(t)) if t < t_lost else None   # update(None) = nobody tracked
            res.append((t, est.update(obs)["vx"]))
        res = np.array(res)
        self.assertGreater(res[(res[:, 0] > t_lost - 0.3) & (res[:, 0] < t_lost), 1].min(), 0)
        self.assertTrue(np.all(res[res[:, 0] >= t_lost + 0.3, 1] == 0))

    def test_gap_restarts_and_rehomes(self):
        """A long absence (another visitor arrives somewhere else) must not walk the robot."""
        rng = np.random.default_rng(0)
        est = LocoEstimator()
        est.calibrate(make_obs(0.0, 0.0, 2.6, noise=False))
        for t in np.arange(0.0, 2.0, 1 / FPS):
            est.update(make_obs(t, 0.0, 2.6, rng=rng))
        cmds = [est.update(make_obs(t, 0.6, 2.0, rng=rng)) for t in np.arange(8.0, 12.0, 1 / FPS)]
        self.assertTrue(all(c == {"vx": 0.0, "vy": 0.0, "wz": 0.0} for c in cmds))
        self.assertTrue(est.calibrated)
        self.assertAlmostEqual(est.home[0], 2.0, delta=0.15)

    def test_speed_map(self):
        """Further from home = faster, capped at the max."""
        rng = np.random.default_rng(0)
        est = LocoEstimator()
        est.calibrate(make_obs(0.0, 0.0, 2.8, noise=False))
        v = {}
        for d in (0.45, 0.65, 1.2):
            est.calibrate(make_obs(0.0, 0.0, 2.8, noise=False))
            for t in np.arange(0.0, 1.5, 1 / FPS):
                c = est.update(make_obs(t, 0.0, 2.8 - d, rng=rng))
            v[d] = c["vx"]
        self.assertLess(v[0.45], v[0.65])
        self.assertAlmostEqual(v[1.2], est.p["vx_max"], delta=1e-9)


class TestVelocityLimiter(unittest.TestCase):
    def test_caps_and_start_levels(self):
        lim = VelocityLimiter()
        out = [lim({"vx": 2.0, "vy": -1.0, "wz": 3.0}, 0.02) for _ in range(200)]
        np.testing.assert_allclose(out[-1], [0.6, -0.5, 1.2])
        # first tick: straight to the policy start levels (no time in the dead band), then the accel limit; vx is
        # already at its start level on that tick, so the turn uses the walking (arc) start level
        np.testing.assert_allclose(out[0], [0.2 + 0.8 * 0.02, -(0.45 + 0.6 * 0.02), 0.5 + 1.5 * 0.02])

    def test_turn_start_levels(self):
        for cmd, first in (({"wz": 0.3}, 0.9), ({"wz": -0.3}, -1.1)):
            lim = VelocityLimiter()
            self.assertAlmostEqual(lim(cmd, 0.02)[2], first)
        lim = VelocityLimiter()
        for _ in range(50):
            lim({"vx": 0.4}, 0.02)
        self.assertAlmostEqual(lim({"vx": 0.4, "wz": 0.3}, 0.02)[2], 0.5)   # arc: walking start level

    def test_small_command_lifted(self):
        lim = VelocityLimiter()
        v = [lim({"vy": 0.1}, 0.02) for _ in range(10)]
        self.assertAlmostEqual(v[-1][1], 0.4)
        v = [lim({"vy": 0.01}, 0.02) for _ in range(5)]   # below eps = stop (from 0.3: drops at once)
        self.assertEqual(v[0][1], 0.0)

    def test_stop_ramps_to_start_level_then_zero(self):
        lim = VelocityLimiter()
        for _ in range(200):
            lim({"vx": 0.6}, 0.02)
        vs = []
        while True:
            v = lim({"vx": 0.0}, 0.02)[0]
            vs.append(v)
            if v == 0.0:
                break
        self.assertTrue(all(a > b for a, b in zip(vs, vs[1:])))
        self.assertAlmostEqual(len(vs) * 0.02, (0.6 - 0.2) / lim.p["dx"] + 0.02, delta=0.03)

    def test_reverse_goes_through_zero(self):
        lim = VelocityLimiter()
        for _ in range(100):
            lim({"wz": 1.0}, 0.02)
        seq = [lim({"wz": -1.0}, 0.02)[2] for _ in range(30)]
        self.assertIn(0.0, seq)
        k = seq.index(0.0)
        self.assertTrue(all(x >= 0 for x in seq[:k]) and all(x < 0 for x in seq[k + 1:]))

    def test_sonic_preset_has_no_start_levels(self):
        lim = VelocityLimiter.for_backend("sonic")
        v = lim({"vx": 0.1, "vy": -0.1, "wz": 0.2}, 0.02)
        np.testing.assert_allclose(v, [0.8 * 0.02, -0.6 * 0.02, 1.5 * 0.02])
        self.assertEqual(VelocityLimiter.for_backend("split").p, VelocityLimiter().p)

    def test_scale_and_back(self):
        lim = VelocityLimiter(scale=0.5)
        for _ in range(200):
            v = lim({"vx": -1.0, "vy": 0.0, "wz": 0.0}, 0.02)
        self.assertAlmostEqual(v[0], -0.35)
        lim = VelocityLimiter(scale=0.5)
        for _ in range(200):
            v = lim({"vx": 0.6}, 0.02)
        self.assertAlmostEqual(v[0], 0.3)


K1_STAGED = ROOT / "sonic_k1" / "overlay" / "gear_sonic" / "data" / "assets" / "robot_description" / "urdf" / "k1" / "meshes"
SONIC_VARIANT = ROOT / "models" / "k1" / "k1s_la4" / "deploy_002000"


def _vendor_k1():
    try:
        from k1.sim import K1_XML

        return K1_XML.exists()
    except Exception:
        return False


@unittest.skipUnless(K1_STAGED.exists() and (SONIC_VARIANT / "model_encoder.onnx").exists() and _vendor_k1(),
                     "staged K1 meshes, the K1 SONIC export or the vendor K1 MJCF (AI_SAPIENS_ROOT) missing")
class TestSonicBackend(unittest.TestCase):
    """k1live/backends/sonic.py through the Backend interface: stands, follows an arm step, waist, interface."""

    def test_stand_gesture_waist_reset(self):
        """Streams a trained event gesture (k1/gestures welcome_open_arms) as the server would: the robot stays up,
        the arms follow (eval_k1_mujoco: 2.6 deg for this checkpoint; here <= 5 deg mean, compared 60 ms later), the waist follows the person's yaw, reset() respawns at the given pose. (A static 0.5 rad shoulder
        lift outside the training gestures is NOT tracked well by this checkpoint - ~7 deg mean / 22 deg max.)"""
        from k1.gestures import GESTURES, ArmIK
        from k1.sim import K1Split
        from k1live.backends.base import make_backend
        from k1live.protocol import ARM_JOINTS, POLICY_JOINTS

        q = GESTURES["welcome_open_arms"](ArmIK(K1Split())).q[:, [POLICY_JOINTS.index(n) for n in ARM_JOINTS]]
        be = make_backend("sonic")
        try:
            self.assertFalse(be.supports["walk"])          # gesture checkpoint: no walking
            self.assertEqual(len(be.arm_default), 10)
            self.assertGreater(be.pelvis_height(), 0.7)
            Q = []
            for k in range(len(q)):
                r = be.step(q[k], loco_cmd=(0.3, 0.0, 0.0))  # ignored (supports walk False)
                Q.append(be.arm_q_now())
                self.assertFalse(be.fallen())
            self.assertEqual(r["target"].shape, (23,))
            err = np.degrees(np.abs(np.array(Q)[3:] - q[:-3])).mean()   # policy lag ~60 ms
            self.assertLess(err, 5.0)
            x0 = be.root_pose()
            for k in range(100):
                be.step(be.arm_default, waist_yaw=0.3, loco_cmd=(0.0, 0.0, 0.0))
            self.assertAlmostEqual(be.q_now()[be.names.index("waist_yaw_joint")], 0.3, delta=0.1)
            self.assertLess(np.hypot(be.root_pose()[0] - x0[0], be.root_pose()[1] - x0[1]), 0.1)
            be.reset(1.0, -0.5, 0.7)
            x, y, yaw = be.root_pose()
            self.assertAlmostEqual(x, 1.0, delta=0.05)
            self.assertAlmostEqual(y, -0.5, delta=0.05)
            self.assertAlmostEqual(yaw, 0.7, delta=0.05)
        finally:
            be.close()


if __name__ == "__main__":
    unittest.main()
