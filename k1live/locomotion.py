"""Walking intent: person in front of a monocular camera -> K1 velocity command ("floor joystick").

The person is calibrated at a home spot (standing still, facing the camera). Stepping forward (toward the camera),
back or sideways by more than a threshold walks the robot in that direction, turning the body turns it, coming
back to home stops it. Gestures (bow, raised arms, clapping, weight shifts) must not walk it.

Where the person is, per frame (MediaPipe Pose raw output, no model of our own):

* distance: weak-perspective fit of the world landmarks (metric, camera-aligned, hip-centred) to the image
  landmarks (px): u_i ~ s * X_i + t. s = px per metre at the person's depth (s ~ f / Z) and t = the image
  position of the hip centre. Because the world skeleton foreshortens exactly like the image skeleton, s does
  not change when the person bows (torso shorter in the image), turns (narrower shoulders) or lifts the arms -
  unlike any single image length (shoulder width, torso length). Shoulders + hips always, knees/ankles when
  visible (longer baseline). Z = f_px / s with f_px from the camera's horizontal FOV.
* sideways: X = (t_u - c_u) / s (metres in the camera frame at the person's depth).
* turning: yaw of the world shoulder line and of the hip line, each relative to its home value; the body turn is
  the smaller of the two when they agree in sign (else 0). A real turn rotates both lines (MediaPipe: within ~5 deg
  of each other); twisting the upper body (welcome gesture: shoulders 45 deg, hips 20) or lifting both arms overhead
  (the shoulder landmarks drift: shoulders 29 deg, hips 15) moves mostly the shoulder line.

Each signal is median-of-3 de-glitched and low-passed, compared with home, and passes a dead zone with
hysteresis (start threshold > stop threshold) plus a short confirmation time, then maps linearly to a speed.
Lost person -> zero after `lost_hold_s`. Before any calibration the estimator homes itself automatically once
the person has stood still for `auto_home_s`, and again after a long absence (a new visitor).

Output is in robot terms (K1 body frame, x forward, y left, yaw CCW): with mirror=True a step to the person's
right walks the robot to its left and a turn to the person's left turns the robot to its right (a reflection);
forward/back are not mirrored (person toward the camera -> robot walks toward its viewer).

`VelocityLimiter` is the controller-side shaping (max speed + acceleration, applied at 50 Hz). Defaults come
from docs/results.md.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# MediaPipe Pose landmark ids
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANK, R_ANK = 25, 26, 27, 28
TORSO = (L_SH, R_SH, L_HIP, R_HIP)
LEGS = (L_KNEE, R_KNEE, L_ANK, R_ANK)

ZERO = {"vx": 0.0, "vy": 0.0, "wz": 0.0}


def _get(obs, key, default=None):
    if obs is None:
        return default
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Measurement:
    """Where the person is in this frame (camera frame: x right, z away from the camera)."""
    t: float
    z: float          # distance from the camera, m
    x: float          # lateral position, m (image right positive)
    yaw: float        # shoulder-line yaw, rad (+ = the person turned to their own left, 0 = facing the camera)
    yaw_hip: float    # hip-line yaw, rad
    scale: float      # weak-perspective px per metre
    legs: bool        # knees + ankles used in the fit
    resid_px: float   # RMS fit residual / s (m): sanity of the fit


def measure(obs, hfov_deg=65.0, min_vis=0.5, use_legs=True, max_resid_m=0.12):
    """One MediaPipe result -> Measurement, or None if the torso is not visible enough to trust."""
    if obs is None:
        return None
    img = _get(obs, "lm_img")
    wld = _get(obs, "lm_world")
    if img is None or wld is None:
        return None
    img = np.asarray(img, np.float64)
    wld = np.asarray(wld, np.float64)
    W = float(_get(obs, "img_w", 1.0))
    H = float(_get(obs, "img_h", 1.0))
    vis = img[:, 3] if img.shape[1] > 3 else (wld[:, 3] if wld.shape[1] > 3 else np.ones(len(img)))
    vis = np.nan_to_num(vis, nan=0.0)
    if np.any(vis[list(TORSO)] < min_vis) or not np.all(np.isfinite(img[list(TORSO), :2])):
        return None
    idx = list(TORSO)
    legs = use_legs and bool(np.all(vis[list(LEGS)] >= min_vis)) and np.all(np.isfinite(img[list(LEGS), :2]))
    if legs:
        idx += list(LEGS)
    uv = img[idx, :2] * np.array([W, H])
    xy = wld[idx, :2]
    w = vis[idx]
    wn = w / w.sum()
    mu_uv = wn @ uv
    mu_xy = wn @ xy
    A = xy - mu_xy
    B = uv - mu_uv
    den = float((w[:, None] * A * A).sum())
    if den < 1e-6:
        return None
    s = float((w[:, None] * A * B).sum()) / den
    if not np.isfinite(s) or s <= 1.0:
        return None
    resid = float(np.sqrt((wn[:, None] * (B - s * A) ** 2).sum(axis=1).sum())) / s
    if resid > max_resid_m:
        return None
    t_uv = mu_uv - s * mu_xy  # image position of the world origin (hip centre)
    f = 0.5 * W / math.tan(math.radians(hfov_deg) / 2)
    z = f / s
    x = (t_uv[0] - 0.5 * W) / s
    d = wld[L_SH, :3] - wld[R_SH, :3]
    dh = wld[L_HIP, :3] - wld[R_HIP, :3]
    return Measurement(float(_get(obs, "t", 0.0)), z, x, math.atan2(d[2], d[0]), math.atan2(dh[2], dh[0]), s, legs,
                       resid)


def body_turn(d_sh, d_hip, mode="min"):
    """Shoulder- and hip-line yaw offsets (rad) -> body turn. min: the smaller one if both agree in sign, else 0;
    shoulder / hip: that line only; mean: their average."""
    if mode == "shoulder":
        return d_sh
    if mode == "hip":
        return d_hip
    d_hip = d_sh + _wrap(d_hip - d_sh)   # same side of the +-180 deg wrap as the shoulder line
    if mode == "mean":
        return 0.5 * (d_sh + d_hip)
    if d_sh * d_hip <= 0:
        return 0.0
    return math.copysign(min(abs(d_sh), abs(d_hip)), d_sh)


class _Axis:
    """Dead zone + hysteresis + confirmation + linear speed map for one signed offset."""

    def __init__(self, start, stop, full, v_min, v_max_pos, v_max_neg, confirm_s):
        self.start, self.stop, self.full = start, stop, full
        self.v_min, self.v_pos, self.v_neg = v_min, v_max_pos, v_max_neg
        self.confirm_s = confirm_s
        self.engaged = 0           # -1, 0, +1
        self.pending = 0
        self.t_pending = None

    def reset(self):
        self.engaged, self.pending, self.t_pending = 0, 0, None

    def __call__(self, d, t):
        sgn = 1 if d > 0 else -1
        if self.engaged and (abs(d) < self.stop or sgn != self.engaged):
            self.engaged = 0
        if not self.engaged:
            if abs(d) >= self.start:
                if self.pending != sgn:
                    self.pending, self.t_pending = sgn, t
                if t - self.t_pending >= self.confirm_s - 1e-9:
                    self.engaged = sgn
                    self.pending, self.t_pending = 0, None
            else:
                self.pending, self.t_pending = 0, None
        if not self.engaged:
            return 0.0
        vmax = self.v_pos if self.engaged > 0 else self.v_neg
        if vmax <= 0:
            return 0.0
        u = np.clip((abs(d) - self.stop) / max(self.full - self.stop, 1e-6), 0.0, 1.0)
        return float(self.engaged * (min(self.v_min, vmax) + (vmax - min(self.v_min, vmax)) * u))


class LocoEstimator:
    """Person -> {"vx","vy","wz"} (robot body frame). See the module docstring.

    obs: dict or object with t (s), img_w, img_h, lm_img (33,4: x,y normalised, z, visibility),
    lm_world (33,4: metres, hip-centred, camera-aligned, visibility); None = nobody tracked."""

    DEFAULTS = dict(
        hfov_deg=65.0,          # camera horizontal field of view (laptop webcams 60-78 deg)
        min_vis=0.5,
        use_legs=True,
        yaw_mode="min",         # body turn from the shoulder/hip lines: min | mean | shoulder | hip
        tau_z=0.12, tau_x=0.10, tau_yaw=0.10,   # low-pass time constants, s
        # forward / back (m from home along the camera axis; + = toward the camera)
        fwd_start=0.30, fwd_stop=0.18, fwd_full=0.80, vx_min=0.25, vx_max=0.60, vx_back_max=0.30,
        # sideways (m); VelocityLimiter lifts it to the loco policy's start level (0.4 left / 0.45 right)
        side_start=0.25, side_stop=0.13, side_full=0.60, vy_min=0.40, vy_max=0.45,
        # turning (deg); lifted to 0.9 left / 1.1 right in place, 0.5 while walking forward
        yaw_start=30.0, yaw_stop=15.0, yaw_full=60.0, wz_min=0.60, wz_max=0.90,
        yaw_hold=100.0,         # deg: past this (back to the camera) the engaged turn direction is kept, so the
                                # +-180 deg wrap of a person turning all the way round does not reverse the robot
        confirm_s=0.10,         # offset must stay past the start threshold this long
        lost_hold_s=0.15,       # keep the last command over short MediaPipe drop-outs, then zero
        calib_s=0.5,            # calibrate() without obs: average home over the next calib_s (output 0 meanwhile)
        auto_home=True,         # home automatically when uncalibrated / after a long absence
        auto_home_s=0.8,        # ... once the person stood still this long
        rehome_after_s=3.0,     # absence that makes the next person re-home (0 = never)
        still_z=0.04, still_x=0.03, still_yaw_deg=4.0,   # "standing still": std over auto_home_s
        home_follow_s=3.0,      # home follows the person with this time constant on axes that are not walking:
                                # slow drift (shuffling forward ~5 cm/s) stays under the start threshold
                                # (v * tau = 0.15 m), a deliberate step (>= 0.3 m/s) still crosses it. 0 = off
        follow_engaged=False,   # True: home also follows on walking axes, i.e. the offset is a high-pass of the
                                # position (~ speed x home_follow_s): the robot walks while the person walks and
                                # stops ~1 s after they stop, wherever that is ("walk along" instead of "joystick")
        turn=True,              # False: body turns never turn the robot (wz = 0)
        fwd_max_yaw_deg=0.0,    # > 0: while the shoulder or hip line is turned more than this away from the
                                # camera, the distance is not trusted and held (MediaPipe's side / back view world
                                # landmarks are off: a presenter turning to the board "walks" 0.3-0.6 m). 0 = off
        fwd_confirm_s=None,     # forward / back axis only: its own confirm_s (None = confirm_s)
        fwd_home_follow_s=None,  # forward / back axis only: its own home_follow_s (None = home_follow_s; 0 = off)
    )

    # Named parameter sets (LocoEstimator(profile=...); the sender's --loco-profile). docs/architecture.md
    PROFILES = {
        # a visitor standing 2.5-3.5 m in front of the camera, steps / turns on purpose ("floor joystick")
        "visitor": {},
        # a presenter on a stage / at a podium seen by a fixed camera: turning to the slides, the board or the side
        # of the audience must not spin the robot, a few metres of strolling must not send it walking for good.
        # Turning off, distance held while side-on, walk-along (stops ~1 s after the presenter stops), higher
        # thresholds and a longer confirmation for the noisier far-away measurement; home is taken after 0.4 s of
        # roughly standing (a presenter is rarely still; with walk-along home follows anyway).
        "presenter": dict(turn=False, follow_engaged=True, home_follow_s=0.8, fwd_max_yaw_deg=35.0,
                          fwd_start=0.40, fwd_stop=0.20, side_start=0.30, side_stop=0.15, confirm_s=0.25,
                          auto_home_s=0.4, still_z=0.15, still_x=0.10, still_yaw_deg=15.0),
    }

    def __init__(self, mirror=True, profile="visitor", **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise TypeError(f"unknown LocoEstimator params: {sorted(unknown)}")
        if profile not in self.PROFILES:
            raise ValueError(f"unknown LocoEstimator profile {profile!r} (one of {sorted(self.PROFILES)})")
        self.mirror = bool(mirror)
        self.profile = profile
        self.p = {**self.DEFAULTS, **self.PROFILES[profile], **params}   # explicit params override the profile
        p = self.p
        self.ax_x = _Axis(p["fwd_start"], p["fwd_stop"], p["fwd_full"], p["vx_min"], p["vx_max"], p["vx_back_max"],
                          p["confirm_s"] if p["fwd_confirm_s"] is None else p["fwd_confirm_s"])
        self.ax_y = _Axis(p["side_start"], p["side_stop"], p["side_full"], p["vy_min"], p["vy_max"], p["vy_max"],
                          p["confirm_s"])
        ys, yo, yf = (math.radians(p[k]) for k in ("yaw_start", "yaw_stop", "yaw_full"))
        self.ax_w = _Axis(ys, yo, yf, p["wz_min"], p["wz_max"], p["wz_max"], p["confirm_s"])
        self.home = None              # (z, x, yaw_shoulder, yaw_hip)
        self.calibrated = False       # True after calibrate() or an auto-home
        self.reset()

    # ------------------------------------------------------------------------------------------------------
    def reset(self):
        """Forget the filters and the command (keeps home)."""
        self.f = None                 # filtered (z, x, yaw_shoulder, yaw_hip)
        self.t_f = None
        self.raw = deque(maxlen=3)    # for the median-of-3 de-glitch
        self.cmd = dict(ZERO)
        self.t_seen = None            # last time the person was measured
        self._calib = None            # list of measurements while calibrating
        self._calib_t0 = None
        self._still = deque()
        for a in (self.ax_x, self.ax_y, self.ax_w):
            a.reset()
        self.last = None              # last Measurement
        self._z_held = False          # distance held (fwd_max_yaw_deg)
        self.offset = None            # {"fwd","side","yaw"} camera terms, vs home

    def forget_home(self):
        """Drop home (new visitor, video switch): the next person is homed again once they stand still."""
        self.reset()
        self.home, self.calibrated = None, False

    def calibrate(self, obs=None):
        """Home = the person's position in obs (the sender passes landmarks averaged over its ~0.5 s calibration
        window). Without obs (or if obs is unusable) home is averaged over the next calib_s of update() frames."""
        self.reset()
        m = self._measure(obs) if obs is not None else None
        if m is not None:
            self._set_home([m])
            self.t_seen = self.t_f = m.t
            return True
        self.home, self.calibrated = None, False
        self._calib, self._calib_t0 = [], None
        return False

    def _mp(self):
        p = self.p
        return dict(hfov_deg=p["hfov_deg"], min_vis=p["min_vis"], use_legs=p["use_legs"])

    def _measure(self, obs):
        """Where the person is in one frame (Measurement or None). Subclasses swap the source (k1live.gemx: the 3D
        camera-frame skeleton of GEM-X instead of the weak-perspective fit of MediaPipe landmarks)."""
        return measure(obs, **self._mp())

    def _set_home(self, ms):
        z = float(np.median([m.z for m in ms]))
        x = float(np.median([m.x for m in ms]))
        ys = float(math.atan2(np.mean([math.sin(m.yaw) for m in ms]), np.mean([math.cos(m.yaw) for m in ms])))
        yh = float(math.atan2(np.mean([math.sin(m.yaw_hip) for m in ms]), np.mean([math.cos(m.yaw_hip) for m in ms])))
        self.home = (z, x, ys, yh)
        self.calibrated = True
        self.f = np.array(self.home)

    # ------------------------------------------------------------------------------------------------------
    def _lost(self, t):
        if self.t_seen is None or t is None or t - self.t_seen > self.p["lost_hold_s"]:
            self.cmd = dict(ZERO)
            for a in (self.ax_x, self.ax_y, self.ax_w):
                a.reset()
            self._still.clear()
        if (self.p["auto_home"] and self.p["rehome_after_s"] > 0 and self.t_seen is not None and t is not None
                and t - self.t_seen > self.p["rehome_after_s"] and self._calib is None):
            self.home, self.calibrated = None, False
            self.f, self.raw = None, deque(maxlen=3)
        return dict(self.cmd)

    def _filter(self, m):
        """median-of-3 (single-frame MediaPipe glitches, shoulder-line front/back flips) -> first-order low-pass."""
        p = self.p
        v = np.array([m.z, m.x, m.yaw, m.yaw_hip])
        if self.f is not None:
            v[2:] = self.f[2:] + _wrap(v[2:] - self.f[2:])  # unwrap around the filter state before the median
        self.raw.append(v)
        med = np.median(np.array(self.raw), axis=0) if len(self.raw) == 3 else v
        if self.f is None or self.t_f is None:
            self.f, self.t_f = med.copy(), m.t
            return self.f
        dt = max(m.t - self.t_f, 1e-3)
        a = 1.0 - np.exp(-dt / np.array([p["tau_z"], p["tau_x"], p["tau_yaw"], p["tau_yaw"]]))
        self.f = self.f + a * (med - self.f)   # yaw state stays continuous (unwrapped); offsets wrap it
        self.t_f = m.t
        return self.f

    def _auto_home(self, m):
        p = self.p
        self._still.append(m)
        while self._still and m.t - self._still[0].t > p["auto_home_s"]:
            self._still.popleft()
        if len(self._still) < 3 or m.t - self._still[0].t < 0.9 * p["auto_home_s"]:
            return
        zs = [s.z for s in self._still]
        xs = [s.x for s in self._still]
        ys = [s.yaw for s in self._still]
        yh = [s.yaw_hip for s in self._still]
        if (np.std(zs) < p["still_z"] * max(1.0, np.median(zs) / 2.5) and np.std(xs) < p["still_x"]
                and np.degrees(np.std(np.unwrap(ys))) < p["still_yaw_deg"]
                and np.degrees(np.std(np.unwrap(yh))) < p["still_yaw_deg"]):
            self._set_home(list(self._still))
            self._still.clear()

    def update(self, obs):
        """One perception result -> command dict {"vx","vy","wz"} (robot body frame, before sim limits)."""
        t = _get(obs, "t")
        m = self._measure(obs)
        if m is None:
            return self._lost(t)
        gap = None if self.t_seen is None else m.t - self.t_seen
        if gap is not None and gap > self.p["lost_hold_s"]:
            # the person was not tracked for a while (the sender does not call update() then): start fresh
            self._lost(m.t)
            self.f, self.t_f = None, None
            self.raw.clear()
        self.t_seen = m.t
        self.last = m
        if self._calib is not None:  # calibrating: average, output 0
            if self._calib_t0 is None:
                self._calib_t0 = m.t
            self._calib.append(m)
            self._set_home(self._calib)
            if m.t - self._calib_t0 >= self.p["calib_s"]:
                self._calib = None
                self.raw.clear()
                self.f, self.t_f = np.array(self.home), m.t
            self.cmd = dict(ZERO)
            return dict(self.cmd)
        if self.home is None:
            if self.p["auto_home"]:
                self._auto_home(m)
                if self.home is not None:
                    self.raw.clear()
                    self.f, self.t_f = np.array(self.home), m.t
            self.cmd = dict(ZERO)
            return dict(self.cmd)
        t_prev = self.t_f
        z_prev = None if self.f is None else float(self.f[0])
        z, x, ys, yh = self._filter(m)
        p = self.p
        if p["fwd_max_yaw_deg"] > 0 and z_prev is not None:
            if max(abs(_wrap(ys)), abs(_wrap(yh))) > math.radians(p["fwd_max_yaw_deg"]):
                z = self.f[0] = z_prev       # side-on to the camera: hold the distance (and its filter state)
                self._z_held = True
            elif self._z_held:               # facing again: restart from the raw distance; the jump is not a step
                zr = float(np.median([r[0] for r in self.raw]))
                self.home = (self.home[0] + zr - z_prev,) + tuple(self.home[1:])
                z = self.f[0] = zr
                self._z_held = False
        tau_fwd = p["home_follow_s"] if p["fwd_home_follow_s"] is None else p["fwd_home_follow_s"]
        if (p["home_follow_s"] > 0 or tau_fwd > 0) and t_prev is not None:
            dt = max(m.t - t_prev, 0.0)
            a = 1.0 - math.exp(-dt / p["home_follow_s"]) if p["home_follow_s"] > 0 else 0.0
            a_fwd = 1.0 - math.exp(-dt / tau_fwd) if tau_fwd > 0 else 0.0
            h = list(self.home)
            fe = p["follow_engaged"]
            if fe or not self.ax_x.engaged:
                h[0] += a_fwd * (z - h[0])
            if fe or not self.ax_y.engaged:
                h[1] += a * (x - h[1])
            if fe or not self.ax_w.engaged or not p["turn"]:
                h[2] += a * _wrap(ys - h[2])
                h[3] += a * _wrap(yh - h[3])
            self.home = tuple(h)
        z0, x0, ys0, yh0 = self.home
        fwd = z0 - z                 # + = closer to the camera
        side = x - x0                # + = image right = the person's left
        dyaw = body_turn(_wrap(ys - ys0), _wrap(yh - yh0), self.p["yaw_mode"])  # + = the person turned left
        if self.ax_w.engaged and abs(dyaw) > math.radians(self.p["yaw_hold"]):
            dyaw = math.copysign(abs(dyaw), self.ax_w.engaged)
        self.offset = {"fwd": float(fwd), "side": float(side), "yaw": float(dyaw)}
        vx = self.ax_x(fwd, m.t)
        s = self.ax_y(side, m.t)     # + = toward the person's left
        w = self.ax_w(dyaw, m.t) if p["turn"] else 0.0     # + = the person turned left
        # robot terms: mirror -> person's left = robot's right (y < 0); person turning left -> robot turns right
        sign = -1.0 if self.mirror else 1.0
        self.cmd = {"vx": vx, "vy": sign * s, "wz": sign * w}
        return dict(self.cmd)

    # ------------------------------------------------------------------------------------------------------
    def root(self):
        """HumanFrame.root: pelvis offset from home in robot terms {"x","y","yaw"} (None before home)."""
        if self.offset is None:
            return None
        sign = -1.0 if self.mirror else 1.0
        o = self.offset
        return {"x": o["fwd"], "y": sign * o["side"], "yaw": sign * o["yaw"]}


class VelocityLimiter:
    """Controller-side shaping of a velocity command at the control rate (50 Hz): per-axis caps, start levels,
    acceleration / deceleration limits, and a global speed scale (operator `max_speed`).

    The ROBOTIS loco policy (loco_armdriven_r2) has dead bands that depend on the direction (MuJoCo sweeps from
    standing with the arms following gestures, docs/results.md): forward needs >= 0.2 m/s, back >= 0.2,
    sideways left >= 0.4 / right >= 0.45, turning in place left >= 0.9 / right >= 1.1 rad/s; while walking forward
    (|vx| >= 0.2) a turn of 0.5-0.8 rad/s already works (arcs). Below these it stands or barely moves, so ramping a
    command up from 0 would sit in the dead band. A non-zero command is therefore lifted to at least its start level
    at once and ramps (accel limit) from there; on the way back to 0 it ramps down (decel limit) to the start level
    and then drops to 0. A sign change goes through 0 first. Caps are applied after the lift."""

    DEFAULTS = dict(vx_max=0.60, vx_back_max=0.35, vy_max=0.50, wz_max=1.20,           # caps
                    vx_start=0.20, vx_back_start=0.20,                                 # start levels
                    vy_start_left=0.40, vy_start_right=0.45,
                    wz_start_left=0.90, wz_start_right=1.10, wz_start_walking=0.50,
                    ax=0.8, ay=0.6, aw=1.5,              # acceleration above the start level, m/s^2 / rad/s^2
                    dx=3.0, dy=3.0, dw=6.0,              # deceleration (3 m/s^2: stop 0.23 m from 0.6 m/s)
                    eps=0.02)                            # |command| below this = 0

    def __init__(self, scale=1.0, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise TypeError(f"unknown VelocityLimiter params: {sorted(unknown)}")
        self.p = dict(self.DEFAULTS, **params)
        self.scale = float(scale)
        self.v = np.zeros(3)

    # sonic: the kinematic planner turns any speed >= 0.05 m/s into a proper gait, so there is no dead band to jump
    # over; caps = the planner walking set / the walking acceptance scenarios
    PRESETS = {"split": {},
               "sonic": dict(vx_max=0.60, vx_back_max=0.30, vy_max=0.30, wz_max=0.60, vx_start=0.0, vx_back_start=0.0,
                             vy_start_left=0.0, vy_start_right=0.0, wz_start_left=0.0, wz_start_right=0.0,
                             wz_start_walking=0.0)}

    @classmethod
    def for_backend(cls, name, scale=1.0, **params):
        """Limiter preset of a k1live backend ("split": loco dead bands, "sonic": no start levels)."""
        return cls(scale=scale, **dict(cls.PRESETS.get(name, {}), **params))

    def reset(self):
        self.v[:] = 0.0

    def _start(self, i, sign, walking):
        p = self.p
        if i == 0:
            return p["vx_start"] if sign > 0 else p["vx_back_start"]
        if i == 1:
            return p["vy_start_left"] if sign > 0 else p["vy_start_right"]
        if walking:
            return p["wz_start_walking"]
        return p["wz_start_left"] if sign > 0 else p["wz_start_right"]

    def __call__(self, cmd, dt):
        p = self.p
        if cmd is None:
            cmd = ZERO
        c = np.array([cmd.get("vx", 0.0), cmd.get("vy", 0.0), cmd.get("wz", 0.0)], np.float64)
        c = np.nan_to_num(c) * self.scale
        caps = ((p["vx_back_max"], p["vx_max"]), (p["vy_max"], p["vy_max"]), (p["wz_max"], p["wz_max"]))
        acc = (p["ax"], p["ay"], p["aw"])
        dec = (p["dx"], p["dy"], p["dw"])
        for i in range(3):
            ci, vi = c[i], self.v[i]
            if abs(ci) < p["eps"]:
                ci = 0.0
            walking = i == 2 and abs(self.v[0]) >= p["vx_start"] - 1e-9
            if vi != 0.0 and (ci == 0.0 or ci * vi < 0):       # stopping (or reversing: stop first)
                st = self._start(i, vi, walking)
                mag = abs(vi) - dec[i] * dt
                self.v[i] = 0.0 if mag <= st + 1e-9 else math.copysign(mag, vi)
                continue
            if ci == 0.0:
                self.v[i] = 0.0
                continue
            sg = 1.0 if ci > 0 else -1.0
            st = self._start(i, sg, walking)
            cap = caps[i][0] if sg < 0 else caps[i][1]
            tgt = min(max(abs(ci), st), max(cap, st))
            cur = max(abs(vi), st)                             # from standing: jump to the start level
            if tgt > cur:
                cur = min(tgt, cur + acc[i] * dt)
            else:
                cur = max(tgt, cur - dec[i] * dt)
            self.v[i] = sg * cur
        return self.v.copy()
