"""Hand-authored, event-appropriate G1 reference motions (robot-motion encoder, mode 0).

BONES-SEED (HF bones-studio/seed) has real mocap for greetings/bows/claps, but it is a gated
dataset (license acceptance needed on the HF account). Until then these keyframed clips give
calm gestures: G1 joint keyframes (hardware/MuJoCo joint names) on top of a relaxed standing
pose taken from the SONIC example mocap, eased with minimum-jerk interpolation, root pitch
for the bow. Joint velocities are finite differences, exactly like a CSV reference.

G1 sign conventions (checked with MuJoCo FK):
  shoulder_pitch < 0 : arm forward / up        shoulder_roll  > 0 : left arm out (right: < 0)
  elbow = 0 : 90 deg bent, ~1.5 : straight      hip_pitch < 0 : thigh forward (hip flexion)
  root pitch > 0 : lean forward (bow)
"""
from __future__ import annotations

import numpy as np

from . import rot
from .motion import FPS, MotionSequence
from .params import MUJOCO_JOINT_NAMES, MUJOCO_TO_ISAACLAB

J = {n.replace("_joint", ""): i for i, n in enumerate(MUJOCO_JOINT_NAMES)}

NEUTRAL = {
    "left_hip_pitch": -0.06, "left_hip_roll": 0.03, "left_hip_yaw": 0.0, "left_knee": 0.20,
    "left_ankle_pitch": -0.14, "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.06, "right_hip_roll": -0.03, "right_hip_yaw": 0.0, "right_knee": 0.20,
    "right_ankle_pitch": -0.14, "right_ankle_roll": 0.0,
    "waist_yaw": 0.0, "waist_roll": 0.0, "waist_pitch": 0.06,
    "left_shoulder_pitch": 0.10, "left_shoulder_roll": 0.20, "left_shoulder_yaw": -0.35, "left_elbow": 1.20,
    "left_wrist_roll": 0.0, "left_wrist_pitch": 0.0, "left_wrist_yaw": 0.0,
    "right_shoulder_pitch": 0.10, "right_shoulder_roll": -0.20, "right_shoulder_yaw": 0.35, "right_elbow": 1.20,
    "right_wrist_roll": 0.0, "right_wrist_pitch": 0.0, "right_wrist_yaw": 0.0,
}


def mirror(d):
    """Left<->right copy of a joint dict (roll/yaw signs flip)."""
    out = {}
    for k, v in d.items():
        if k.startswith("left_"):
            k2 = "right_" + k[5:]
        elif k.startswith("right_"):
            k2 = "left_" + k[6:]
        else:
            k2 = k
        flip = any(s in k for s in ("roll", "yaw"))
        out[k2] = -v if flip else v
    return out


def both(d_left):
    """Symmetric pose from a left-arm dict."""
    return {**d_left, **mirror(d_left)}


def _minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def build(name, keys, root_pitch_keys=None, root_z=0.79, fps=FPS):
    """keys: list of (time_s, {joint: abs value}); missing joints keep their previous key value.

    Optional oscillations: a key may contain "_osc": {joint: (amp, freq_hz)} applied until the
    next key with a smooth on/off envelope.
    """
    T = int(round(keys[-1][0] * fps)) + 1
    t = np.arange(T) / fps
    # resolve keyframes to full vectors
    full = []
    cur = dict(NEUTRAL)
    for tk, d in keys:
        cur = {**cur, **{k: v for k, v in d.items() if not k.startswith("_")}}
        full.append((tk, np.array([cur[n.replace("_joint", "")] for n in MUJOCO_JOINT_NAMES]), d.get("_osc", {})))
    q = np.zeros((T, 29))
    for i in range(T):
        ti = t[i]
        k = max(j for j in range(len(full)) if full[j][0] <= ti + 1e-9)
        if k == len(full) - 1:
            q[i] = full[k][1]
            continue
        (t0, a, osc), (t1, b, _) = full[k], full[k + 1]
        s = _minjerk((ti - t0) / max(t1 - t0, 1e-6))
        q[i] = a + s * (b - a)
        if osc:
            env = np.sin(np.pi * np.clip((ti - t0) / (t1 - t0), 0, 1)) ** 0.5
            for jn, (amp, f) in osc.items():
                q[i, J[jn]] += env * amp * np.sin(2 * np.pi * f * (ti - t0))
    pitch = np.zeros(T)
    if root_pitch_keys:
        for i in range(T):
            ti = t[i]
            ks = root_pitch_keys
            k = max(j for j in range(len(ks)) if ks[j][0] <= ti + 1e-9)
            if k == len(ks) - 1:
                pitch[i] = ks[k][1]
            else:
                s = _minjerk((ti - ks[k][0]) / (ks[k + 1][0] - ks[k][0]))
                pitch[i] = ks[k][1] + s * (ks[k + 1][1] - ks[k][1])
    jp = q[:, MUJOCO_TO_ISAACLAB]
    jv = np.gradient(jp, 1.0 / fps, axis=0)
    quat = rot.quat_from_angle_axis(pitch, np.broadcast_to([0.0, 1.0, 0.0], (T, 3)))
    pos = np.zeros((T, 1, 3))
    pos[:, 0, 2] = root_z - 0.12 * np.sin(pitch) ** 2
    return MotionSequence(name, jp, jv, pos, quat[:, None, :], None, [0], 0)


# ---------------------------------------------------------------------------------- clips
# arm poses below were solved with MuJoCo FK + least squares on elbow/wrist targets
R_WAVE_UP = {"right_shoulder_pitch": -0.48, "right_shoulder_roll": -1.30, "right_shoulder_yaw": -1.05,
             "right_elbow": 0.0, "right_wrist_roll": 0.0}
LEFT_OPEN = {"left_shoulder_pitch": -0.55, "left_shoulder_roll": 0.85, "left_shoulder_yaw": 0.25,
             "left_elbow": 1.30, "left_wrist_roll": -0.4}
LEFT_CLAP_CLOSED = {"left_shoulder_pitch": -0.64, "left_shoulder_roll": 0.0, "left_shoulder_yaw": -0.40,
                    "left_elbow": 0.26, "left_wrist_roll": 0.0}
LEFT_CLAP_OPEN = {"left_shoulder_pitch": -0.49, "left_shoulder_roll": 0.17, "left_shoulder_yaw": -0.11,
                  "left_elbow": 0.05}
R_POINT = {"right_shoulder_pitch": -1.05, "right_shoulder_roll": -0.45, "right_shoulder_yaw": 0.2,
           "right_elbow": 1.40, "right_wrist_roll": 0.9, "waist_yaw": -0.15}
LEFT_UP = {"left_shoulder_pitch": -2.55, "left_shoulder_roll": 0.25, "left_shoulder_yaw": 0.0, "left_elbow": 1.35}
LEFT_HEART = {"left_shoulder_pitch": -2.85, "left_shoulder_roll": 0.92, "left_shoulder_yaw": -1.36, "left_elbow": 0.45}


def wave_hello():
    k = [(0.0, {}), (1.0, {}), (2.2, R_WAVE_UP),
         (2.4, {**R_WAVE_UP, "_osc": {"right_shoulder_roll": (0.22, 1.1), "right_elbow": (0.20, 1.1)}}),
         (5.6, R_WAVE_UP), (6.8, {k: NEUTRAL[k] for k in R_WAVE_UP}), (8.0, {})]
    # the 2.4 key both holds the pose and carries the oscillation until 5.6
    return build("wave_hello", k)


def bow():
    arms = both({"left_shoulder_pitch": -0.15, "left_shoulder_roll": 0.12, "left_shoulder_yaw": -0.2,
                 "left_elbow": 1.35})
    down = {**arms, "left_hip_pitch": -0.06 - 0.42, "right_hip_pitch": -0.06 - 0.42, "waist_pitch": 0.20,
            "left_knee": 0.12, "right_knee": 0.12, "left_ankle_pitch": -0.10, "right_ankle_pitch": -0.10}
    k = [(0.0, {}), (1.0, {}), (1.6, arms), (3.0, down), (4.2, down), (5.6, arms), (6.4, {k: NEUTRAL[k] for k in arms}),
         (7.5, {})]
    pitch = [(0.0, 0.0), (1.6, 0.0), (3.0, 0.42), (4.2, 0.42), (5.6, 0.0), (7.5, 0.0)]
    return build("bow", k, root_pitch_keys=pitch)


def welcome_open_arms():
    o = both(LEFT_OPEN)
    k = [(0.0, {}), (1.0, {}), (2.6, o), (4.6, o), (6.2, {k: NEUTRAL[k] for k in o}), (7.5, {})]
    return build("welcome_open_arms", k)


def clap():
    c, o = both(LEFT_CLAP_CLOSED), both(LEFT_CLAP_OPEN)
    k = [(0.0, {}), (1.0, {}), (2.2, o)]
    t = 2.2
    for _ in range(4):
        k += [(t + 0.28, c), (t + 0.62, o)]
        t += 0.62
    k += [(t + 1.2, {kk: NEUTRAL[kk] for kk in o}), (t + 2.2, {})]
    return build("clap", k)


def point_guide():
    k = [(0.0, {}), (1.0, {}), (2.4, R_POINT), (4.6, R_POINT), (6.0, {kk: NEUTRAL[kk] for kk in R_POINT}), (7.2, {})]
    return build("point_guide", k)


def raise_both_arms():
    u = both(LEFT_UP)
    k = [(0.0, {}), (1.0, {}), (3.2, u), (4.4, u), (6.6, {kk: NEUTRAL[kk] for kk in u}), (7.6, {})]
    return build("raise_both_arms", k)


def heart():
    h = both(LEFT_HEART)
    k = [(0.0, {}), (1.0, {}), (3.0, h), (5.0, h), (7.0, {kk: NEUTRAL[kk] for kk in h}), (8.0, {})]
    return build("heart", k)


def weight_shift():
    sway = {"_osc": {"left_hip_roll": (0.07, 0.35), "right_hip_roll": (0.07, 0.35),
                     "left_ankle_roll": (-0.07, 0.35), "right_ankle_roll": (-0.07, 0.35)}}
    k = [(0.0, {}), (1.0, sway), (9.6, {}), (10.5, {})]
    return build("weight_shift", k)


CLIPS = {f.__name__: f for f in (wave_hello, bow, welcome_open_arms, clap, point_guide, raise_both_arms,
                                 heart, weight_shift)}
