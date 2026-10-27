"""Calm event gestures authored in K1 joint space (23 DOF, policy order).

K1 arms have 5 DOF (shoulder pitch/roll/yaw, elbow, wrist roll) and the waist only yaw, so the G1 keyframes of
sonic/gestures.py do not transfer joint-by-joint. Instead each gesture is written as palm (and elbow) targets in
the torso frame, eased with minimum-jerk keyframes, and solved per 50 Hz frame by bounded least squares on the
K1 kinematics (MuJoCo FK of the same model the physics uses, URDF joint limits). Wrist roll is set directly
(palm orientation). Legs/waist hold the loco_armdriven_r2 standing pose: in split mode they belong to the
locomotion policy anyway. The bow needs hip pitch, which split mode cannot command, so `bow` is the one clip
meant for the whole-body mimic student (mode "mimic"): pelvis/torso pitched forward with the legs kept vertical.

K1 conventions (checked with FK): shoulder_pitch < 0 = arm forward/up; left shoulder_roll > 0 = arm out (right:
< 0, joint range one-sided); elbow 0 = forearm at 90 deg, ~1.5 = straight; torso frame x forward, y left, z up.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

FPS = 50.0
HAND_OFFSET = np.array([0.20, 0.0, 0.0])  # palm centre in the *_wrist_roll_rubber_hand frame (mesh centre ~0.17)
SIDES = ("left", "right")


@dataclass
class Motion:
    name: str
    q: np.ndarray        # (T, 23) policy order
    dq: np.ndarray       # (T, 23)
    torso_quat: np.ndarray  # (T, 4) xyzw, reference torso orientation in the world
    mode: str = "split"  # "split" (arms=mimic, legs+waist=loco) or "mimic" (whole body)

    @property
    def T(self):
        return len(self.q)


def _minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def track(keys, t):
    """keys: [(time, value)], value array-like; min-jerk between keys, held after the last."""
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return np.asarray(keys[0][1], float)
    for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            s = _minjerk((t - t0) / max(t1 - t0, 1e-6))
            return np.asarray(a, float) + s * (np.asarray(b, float) - np.asarray(a, float))
    return np.asarray(keys[-1][1], float)


def envelope(t, t0, t1, ramp=0.5):
    """0 -> 1 -> 0 smooth window on [t0, t1] (for oscillations)."""
    if t <= t0 or t >= t1:
        return 0.0
    return float(min(_minjerk((t - t0) / ramp), _minjerk((t1 - t) / ramp)))


class ArmIK:
    """Bounded least-squares IK of one 5-DOF arm on a kinematic copy of the K1 model (pelvis fixed upright)."""

    def __init__(self, sim):
        self.sim = sim
        m = self.m = sim.model
        self.d = mujoco.MjData(m)
        self.names = sim.names
        self.J = {n.replace("_joint", ""): i for i, n in enumerate(self.names)}
        self.base = np.where(sim.lower, sim.loco.default, sim.loco.default)  # loco standing pose, arms relaxed
        self.idx = {s: [self.J[f"{s}_{j}"] for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
                    for s in SIDES}
        self.wrist = {s: self.J[f"{s}_wrist_roll"] for s in SIDES}
        self.hand = {s: m.body(f"{s}_wrist_roll_rubber_hand").id for s in SIDES}
        self.elbow = {s: m.body(f"{s}_elbow_link").id for s in SIDES}
        self.shoulder = {s: m.body(f"{s}_shoulder_pitch_link").id for s in SIDES}
        self.torso = m.body("torso_link").id
        self.lo, self.hi = sim.lo, sim.hi
        self.q = self.base.copy()
        self._fk(self.q)
        self.sh0 = {s: self._local(self.d.xpos[self.shoulder[s]]) for s in SIDES}

    def _fk(self, q):
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[2], d.qpos[3] = 0.76, 1.0
        d.qpos[self.sim.qadr] = q
        mujoco.mj_kinematics(self.m, d)

    def _local(self, p):
        R = self.d.xmat[self.torso].reshape(3, 3)
        return R.T @ (p - self.d.xpos[self.torso])

    def points(self, q, side):
        self._fk(q)
        R = self.d.xmat[self.hand[side]].reshape(3, 3)
        palm = self.d.xpos[self.hand[side]] + R @ HAND_OFFSET
        return self._local(palm) - self.sh0[side], self._local(self.d.xpos[self.elbow[side]]) - self.sh0[side]

    def solve(self, side, palm, elbow=None, w_elbow=0.6, reg=0.03, q_init=None):
        """Arm joints putting the palm (and elbow) at targets given relative to the shoulder in the torso frame."""
        idx = self.idx[side]
        q = (self.q if q_init is None else q_init).copy()
        x0 = np.clip(q[idx], self.lo[idx] + 1e-4, self.hi[idx] - 1e-4)
        nominal = self.base[idx]

        def res(x):
            qq = q.copy()
            qq[idx] = x
            p, e = self.points(qq, side)
            r = [p - palm]
            if elbow is not None:
                r.append(w_elbow * (e - elbow))
            r.append(reg * (x - nominal))
            return np.concatenate(r)

        sol = least_squares(res, x0, bounds=(self.lo[idx], self.hi[idx]), method="trf", xtol=1e-7, ftol=1e-7,
                            max_nfev=200)
        q[idx] = sol.x
        return q, float(np.linalg.norm(res(sol.x)[:3]))


def _quat_y(angle):
    """xyzw quaternion of a rotation about world y (positive = lean forward)."""
    return np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])


def _finish(name, frames, torso=None, mode="split"):
    q = np.asarray(frames)
    # light smoothing of the IK output (3-tap) then velocities by finite differences, like a CSV reference
    qs = q.copy()
    qs[1:-1] = 0.25 * q[:-2] + 0.5 * q[1:-1] + 0.25 * q[2:]
    dq = np.gradient(qs, 1.0 / FPS, axis=0)
    if torso is None:
        torso = np.tile([0.0, 0.0, 0.0, 1.0], (len(qs), 1))
    return Motion(name, qs, dq, np.asarray(torso), mode)


def _arms_motion(ik, name, duration, spec, wrist=None):
    """spec(t) -> {side: (palm, elbow or None)} (targets rel. shoulder); wrist(t) -> {side: wrist roll}."""
    n = int(round(duration * FPS)) + 1
    frames = []
    q = ik.base.copy()
    for i in range(n):
        t = i / FPS
        tg = spec(t)
        for side in SIDES:
            if side in tg and tg[side] is not None:
                palm, elbow = tg[side]
                q, _ = ik.solve(side, np.asarray(palm), None if elbow is None else np.asarray(elbow), q_init=q)
            else:
                q[ik.idx[side]] = ik.base[ik.idx[side]]
        if wrist is not None:
            for side, v in wrist(t).items():
                q[ik.wrist[side]] = v
        frames.append(q.copy())
    return _finish(name, frames)


# -------------------------------------------------------------------------------------------------- gestures
def mirror(v):
    """Left-arm target (rel. left shoulder) -> right arm (y flips)."""
    return np.array([v[0], -v[1], v[2]])


def rest_targets(ik):
    pl, el = ik.points(ik.base, "left")
    pr, er = ik.points(ik.base, "right")
    return {"left": (pl, el), "right": (pr, er)}


def _blend(keys_l, keys_r=None):
    keys_r = keys_r if keys_r is not None else [(t, (mirror(p), None if e is None else mirror(e))) for t, (p, e) in keys_l]

    def side(keys, t):
        p = track([(k, v[0]) for k, v in keys], t)
        es = [v[1] for _, v in keys]
        e = None if any(x is None for x in es) else track([(k, v[1]) for k, v in keys], t)
        return p, e

    return lambda t: {"left": side(keys_l, t), "right": side(keys_r, t)}


def wave_hello(ik):
    R = rest_targets(ik)
    up_p, up_e = mirror([0.06, 0.20, 0.25]), mirror([0.03, 0.17, 0.02])
    keys_r = [(0.0, R["right"]), (1.0, R["right"]), (2.2, (up_p, up_e)), (5.8, (up_p, up_e)), (7.0, R["right"]),
              (8.0, R["right"])]
    base = _blend([(0.0, R["left"]), (8.0, R["left"])], keys_r)

    def spec(t):
        tg = base(t)
        p, e = tg["right"]
        sway = 0.065 * envelope(t, 2.2, 5.8, 0.4) * np.sin(2 * np.pi * 1.1 * (t - 2.2))
        tg["right"] = (p + np.array([0.0, sway, -0.25 * abs(sway)]), e)
        return tg

    wr = ik.base[ik.wrist["right"]]
    return _arms_motion(ik, "wave_hello", 8.0, spec,
                        wrist=lambda t: {"right": float(track([(0, wr), (1.0, wr), (2.2, -1.2), (5.8, -1.2),
                                                               (7.0, wr)], t))})


def welcome_open_arms(ik):
    R = rest_targets(ik)
    o = (np.array([0.20, 0.30, -0.06]), np.array([0.07, 0.16, -0.12]))
    keys = [(0.0, R["left"]), (1.0, R["left"]), (2.6, o), (4.8, o), (6.3, R["left"]), (7.5, R["left"])]
    wl = ik.base[ik.wrist["left"]]
    wk = [(0, wl), (1.0, wl), (2.6, -1.3), (4.8, -1.3), (6.3, wl)]
    return _arms_motion(ik, "welcome_open_arms", 7.5, _blend(keys),
                        wrist=lambda t: {"left": float(track(wk, t)), "right": -float(track(wk, t))})


def clap(ik):
    R = rest_targets(ik)
    op = (np.array([0.21, 0.015, -0.16]), np.array([0.05, 0.09, -0.17]))
    keys = [(0.0, R["left"]), (1.0, R["left"]), (2.2, op), (6.2, op), (7.3, R["left"]), (8.0, R["left"])]
    base = _blend(keys)

    def spec(t):
        tg = base(t)
        # palms meet near the midline: 2.0 Hz open/close, palm centres stay ~7.8 cm apart (hand meshes ~6.6 cm)
        a = envelope(t, 2.4, 6.0, 0.3)
        c = 0.5 * (1 - np.cos(2 * np.pi * 2.0 * (t - 2.4)))
        dy = a * (0.045 * c)
        p, e = tg["left"]
        tg["left"] = (p + np.array([0.0, -dy, 0.0]), e)
        p, e = tg["right"]
        tg["right"] = (p + np.array([0.0, dy, 0.0]), e)
        return tg

    wl = ik.base[ik.wrist["left"]]
    wk = [(0, wl), (1.0, wl), (2.2, 1.45), (6.2, 1.45), (7.3, wl)]
    return _arms_motion(ik, "clap", 8.0, spec,
                        wrist=lambda t: {"left": float(track(wk, t)), "right": -float(track(wk, t))})


def point_guide(ik):
    R = rest_targets(ik)
    g = (mirror([0.22, 0.30, -0.06]), mirror([0.08, 0.16, -0.10]))
    keys_r = [(0.0, R["right"]), (1.0, R["right"]), (2.4, g), (4.8, g), (6.2, R["right"]), (7.2, R["right"])]
    wr = ik.base[ik.wrist["right"]]
    wk = [(0, wr), (1.0, wr), (2.4, 1.4), (4.8, 1.4), (6.2, wr)]
    return _arms_motion(ik, "point_guide", 7.2, _blend([(0, R["left"]), (7.2, R["left"])], keys_r),
                        wrist=lambda t: {"right": float(track(wk, t))})


def heart(ik):
    R = rest_targets(ik)
    h = (np.array([0.02, -0.03, 0.34]), np.array([0.02, 0.17, 0.13]))
    keys = [(0.0, R["left"]), (1.0, R["left"]), (3.0, h), (5.2, h), (7.0, R["left"]), (8.0, R["left"])]
    return _arms_motion(ik, "heart", 8.0, _blend(keys))


def raise_both_arms(ik):
    R = rest_targets(ik)
    u = (np.array([0.05, 0.10, 0.42]), np.array([0.03, 0.08, 0.18]))
    keys = [(0.0, R["left"]), (1.0, R["left"]), (3.0, u), (4.6, u), (6.6, R["left"]), (7.6, R["left"])]
    return _arms_motion(ik, "raise_both_arms", 7.6, _blend(keys))


def stand_at_ease(ik):
    """Standing, arms relaxed: hands drift to a loose clasp in front of the hips and back (calm idle)."""
    R = rest_targets(ik)
    c = (np.array([0.13, -0.035, -0.30]), np.array([0.03, 0.07, -0.18]))
    keys = [(0.0, R["left"]), (1.0, R["left"]), (3.0, c), (6.5, c), (8.5, R["left"]), (9.0, R["left"])]
    return _arms_motion(ik, "stand_at_ease", 9.0, _blend(keys))


def bow(ik, angle=0.30):
    """Polite bow (whole-body mimic): hands clasped in front, pelvis/torso pitched `angle` rad, legs vertical."""
    R = rest_targets(ik)
    c = (np.array([0.14, -0.035, -0.28]), np.array([0.03, 0.07, -0.17]))
    keys = [(0.0, R["left"]), (1.0, R["left"]), (2.0, c), (6.0, c), (7.0, R["left"]), (8.0, R["left"])]
    m = _arms_motion(ik, "bow", 8.0, _blend(keys))
    pk = [(0.0, 0.0), (2.0, 0.0), (3.4, angle), (4.6, angle), (6.0, 0.0), (8.0, 0.0)]
    pitch = np.array([float(track(pk, i / FPS)) for i in range(m.T)])
    q = m.q.copy()
    for s in SIDES:
        q[:, ik.J[f"{s}_hip_pitch"]] = ik.base[ik.J[f"{s}_hip_pitch"]] - pitch  # thighs stay vertical
    torso = np.stack([_quat_y(a) for a in pitch])
    dq = np.gradient(q, 1.0 / FPS, axis=0)
    return Motion("bow", q, dq, torso, mode="mimic")


# event order: greeting -> bow -> welcome, applause -> guiding -> heart -> hooray -> standing
GESTURES = {f.__name__: f for f in (wave_hello, bow, welcome_open_arms, clap, point_guide, heart, raise_both_arms,
                                    stand_at_ease)}
