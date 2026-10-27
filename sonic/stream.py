# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Streamed SMPL motion (ZMQ protocol v3 consumer side) — 50 Hz resampling + playback cursor.

The official StreamedMotionMerger appends one frame per received message (frame_index
stride 1) and the control loop, once caught up, sits `saved_frame_for_observation_window`
frames before the newest one. A 30 fps camera therefore gets replayed at 50 Hz with
hold/wait stutter. Here we instead resample the timestamped stream to exact 50 Hz control
frames (what the encoder was trained on), and keep the same steady-state cursor:
    frames[cursor .. cursor+window-1]  ==  the `window` newest 50 Hz frames
so the commanded "current" frame lags the newest sample by (window-1) * 20 ms
(180 ms for the release model, 60 ms for low_latency).
Protocol v3 wire format helpers (pack/unpack) are byte-compatible with
gear_sonic/examples/live_camera_teleop/soma_to_smpl.py:_fallback_pack_pose_message.

Watchdog (optional, `idle=`): if the sender goes silent (crash, camera unplugged) the buffer holds
the last pose for `stale_hold` s and then cross-fades to the idle standing pose over `stale_fade` s;
when samples come back it fades from whatever it was playing to the live stream over
`resume_fade` s. A sender that keeps publishing (the presence gate in webcam_sender.py) never
triggers it.

Side channel: topic "ctrl" + JSON ({"cmd": "reset" | "quit" | "status", ...}) on the same PUB
socket — the camera window's r/q keys and the sender's status line reach the sim through it. The
official deploy subscribes to "pose" only and never sees these.
"""
from __future__ import annotations

import bisect
import json
from collections import deque

import numpy as np

from . import rot
from .motion import smpl_motion_from_arrays

HISTORY_FRAMES = 5  # StreamedMotionMerger::HISTORY_FRAMES
HEADER_SIZE = 1280
CTRL_TOPIC = "ctrl"


class StreamBuffer:
    def __init__(self, window=10, delay=0.0, max_samples=256, idle=None, stale_hold=0.4, stale_fade=1.5,
                 resume_fade=0.8, robot=None):
        from .robots import g1

        robot = robot or g1()
        self.wrist_idx = list(robot.wrist_il)  # G1: IsaacLab 23..28 (protocol v3 joint_pos[:, 23:29])
        self.window = window
        self.delay = delay
        self.samples = deque(maxlen=max_samples)  # (t, quat(4), joints(24,3), wrists(len(wrist_idx)))
        n = HISTORY_FRAMES + window
        self.motion = smpl_motion_from_arrays("streamed", np.tile([1.0, 0, 0, 0], (n, 1)), np.zeros((n, 24, 3)),
                                              num_dof=robot.num_dof, wrist_idx=self.wrist_idx)
        self.active = False
        self.last_sample_time = None
        self.idle = idle  # stream frame dict (body_quat, smpl_joints) or None = watchdog off
        self.stale_hold, self.stale_fade, self.resume_fade = stale_hold, stale_fade, resume_fade
        self.stale = False
        self._last_out = None
        self._resume = None  # (t_resume, frame the output was at when samples came back)

    def add_sample(self, t, body_quat, smpl_joints, wrists=None):
        q = np.asarray(body_quat, dtype=np.float64).reshape(4)
        if self.samples and np.dot(self.samples[-1][1], q) < 0:
            q = -q
        nw = len(self.wrist_idx)
        w = np.zeros(nw) if wrists is None else np.asarray(wrists, dtype=np.float64).reshape(-1)[:nw]
        if (self.idle is not None and self.active and self._last_out is not None
                and self.last_sample_time is not None and t - self.last_sample_time > self.stale_hold):
            self._resume = (float(t), self._last_out)
        self.samples.append((float(t), q, np.asarray(smpl_joints, dtype=np.float64).reshape(24, 3), w))
        self.last_sample_time = float(t)

    def _watchdog(self, t, x):
        """Apply hold -> fade-to-idle when the input is stale, and the fade back on resume."""
        from .mp2smpl import blend_frames, frame_world, make_frame, smoothstep

        if self.idle is None or x is None:
            return x
        cur = {"body_quat": x[0], "smpl_joints": x[1]}
        wrists = x[2]
        age = t - self.last_sample_time
        self.stale = age > self.stale_hold
        if self.stale:
            Wl = frame_world(cur)
            Wi = frame_world(self.idle)
            idle = make_frame(self.idle["body_quat"], Wi + (Wl[0] - Wi[0]))  # keep the sender's pelvis convention
            w = smoothstep((age - self.stale_hold) / max(self.stale_fade, 1e-3))
            cur = blend_frames(cur, idle, w)
            wrists = (1 - w) * wrists
        elif self._resume is not None:
            t_r, frm = self._resume
            w = smoothstep((t - t_r) / max(self.resume_fade, 1e-3))
            if w >= 1.0:
                self._resume = None
            else:
                cur = blend_frames(frm, cur, w)
        self._last_out = cur
        return (np.asarray(cur["body_quat"], np.float64), np.asarray(cur["smpl_joints"], np.float64), wrists)

    def sample_at(self, t):
        s = self.samples
        if not s:
            return None
        if t <= s[0][0]:
            return s[0][1:]
        if t >= s[-1][0]:
            return s[-1][1:]
        i = bisect.bisect_right([e[0] for e in s], t)  # s[i-1].t <= t < s[i].t
        a, b = s[i - 1], s[i]
        u = (t - a[0]) / max(b[0] - a[0], 1e-6)
        return (rot.quat_slerp(a[1], b[1], u), (1 - u) * a[2] + u * b[2], (1 - u) * a[3] + u * b[3])

    def start(self, t):
        """ENTER: fill the window with the current pose (no jump), caller re-inits heading."""
        x = self.sample_at(t - self.delay)
        if x is None:
            return False
        self._resume = None
        x = self._watchdog(t - self.delay, x)
        m = self.motion
        m.body_quat[:, 0] = x[0]
        m.smpl_joints[:] = x[1]
        m.joint_pos[:, self.wrist_idx] = x[2]
        self.active = True
        return True

    def tick(self, t):
        """Append the 50 Hz frame for wall/sim time t; returns cursor index."""
        x = self._watchdog(t - self.delay, self.sample_at(t - self.delay))
        m = self.motion
        if x is not None:
            m.body_quat[:-1] = m.body_quat[1:]
            m.smpl_joints[:-1] = m.smpl_joints[1:]
            m.joint_pos[:-1] = m.joint_pos[1:]
            m.body_quat[-1, 0] = x[0]
            m.smpl_joints[-1] = x[1]
            m.joint_pos[-1, self.wrist_idx] = x[2]
        return HISTORY_FRAMES


def attach(ctrl, buf: StreamBuffer, t):
    """Toggle streaming on (like pressing ENTER in the C++ deploy)."""
    if not buf.start(t):
        return False
    ctrl.current_motion = buf.motion
    ctrl.current_frame = HISTORY_FRAMES
    ctrl.play = True
    ctrl.reinitialize_heading = True
    return True


def stream_step(ctrl, buf: StreamBuffer, t):
    """Call once per control tick before ctrl.step()."""
    if buf.active and ctrl.current_motion is buf.motion:
        ctrl.current_frame = buf.tick(t)
        ctrl.play = True


# ---------------------------------------------------------------- protocol v3 wire format
_DT = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64", np.dtype(np.int32): "i32",
       np.dtype(np.int64): "i64", np.dtype(bool): "bool"}
_DT_INV = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": np.bool_}


def pack_pose_message(pose_data: dict, topic="pose", version=3) -> bytes:
    fields, binary = [], []
    for k, v in pose_data.items():
        v = np.ascontiguousarray(v)
        dt = _DT.get(v.dtype, "f32")
        if dt == "f32" and v.dtype != np.float32:
            v = v.astype(np.float32)
        fields.append({"name": k, "dtype": dt, "shape": list(v.shape)})
        binary.append(v.tobytes())
    hdr = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")).encode()
    assert len(hdr) <= HEADER_SIZE
    return topic.encode() + hdr.ljust(HEADER_SIZE, b"\x00") + b"".join(binary)


def unpack_pose_message(msg: bytes, topic="pose"):
    body = msg[len(topic):]
    hdr = json.loads(body[:HEADER_SIZE].rstrip(b"\x00").decode())
    off = HEADER_SIZE
    out = {}
    for f in hdr["fields"]:
        dt = np.dtype(_DT_INV[f["dtype"]])
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        out[f["name"]] = np.frombuffer(body, dtype=dt, count=n, offset=off).reshape(f["shape"])
        off += n * dt.itemsize
    return hdr["v"], out


def ctrl_message(cmd, **kw) -> bytes:
    return CTRL_TOPIC.encode() + json.dumps({"cmd": cmd, **kw}, separators=(",", ":")).encode()


def unpack_ctrl_message(msg: bytes):
    return json.loads(msg[len(CTRL_TOPIC):].decode())


def v3_message(smpl_joints, body_quat, frame_index, t_mono=None, wrists=None):
    jp = np.zeros((1, 29), np.float32)
    if wrists is not None:
        jp[:, 23:29] = wrists
    d = {
        "smpl_joints": np.asarray(smpl_joints, np.float32).reshape(1, 24, 3),
        "smpl_pose": np.zeros((1, 21, 3), np.float32),
        "joint_pos": jp,
        "joint_vel": np.zeros((1, 29), np.float32),
        "body_quat": np.asarray(body_quat, np.float32).reshape(1, 4),
        "frame_index": np.array([frame_index], np.int64),
    }
    if t_mono is not None:
        d["timestamp_monotonic"] = np.array([t_mono], np.float64)
    return pack_pose_message(d)
