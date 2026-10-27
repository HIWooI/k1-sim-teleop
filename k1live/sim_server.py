"""Controller + MuJoCo K1 process: HUMAN -> 50 Hz real-time control -> 200 Hz physics -> /robot.mjpg, SIM_STATE.

    .venv/bin/python -m k1live.sim_server [--backend split|sonic]     # simulation only, no robot I/O anywhere

Control loop (wall clock, 20 ms ticks, k1live/timing.sleep_until; a tick that is late by more than a period is skipped, not caught up, and
counted): drain HUMAN + CTRL -> mode logic -> arm reference (rate-limited toward the person's IK pose) + walking
command (limited) -> backend.step (policies + 4 physics substeps) -> fall check -> qpos to the render process ->
SimState (10 Hz) -> log (1 Hz summary + events, logs/k1live/<YYYYmmdd-HHMMSS>/sim.jsonl).

Arms: the newest HumanFrame.arm_q is the goal; the reference moves toward it at <= 6 rad/s with <= 80 rad/s^2
(braking so it does not overshoot) and the policy gets that reference and its velocity. Person lost: hold the last
pose for 1 s, then return to the relaxed pose at <= 1 rad/s. Walking: CTRL drive (priority, valid 0.3 s after the
last message) or HumanFrame.loco (when walk is on) -> WalkLimiter (k1live.locomotion.VelocityLimiter: the loco
policy's start levels, caps, accel / decel; `max_speed` scales it) -> leash: outward motion is cut at a 3 m circle
around the spawn point -> heading hold (no turn commanded: wz = 2 x yaw error, <= 0.4 rad/s, against the arm-induced yaw drift) -> the backend's
velocity command. estop, a missing person (HumanFrame not
present or older than 0.25 s) or walk off -> 0 (drive still works with walk off, never under estop).

Modes (CTRL set mode): live = follow the person; when nobody has been seen for --idle-after s (default 8) the robot
plays the attract loop (k1/gestures.py greetings in order, bow only on backends that can bow) until someone shows up.
idle = do not follow; stand, attract loop when nobody is there. gesture = CTRL gesture {"name"} plays one clip, then
the previous mode resumes. Fall (pelvis < 0.45 m or tilt > 45 deg): respawn standing after 1 s, resets += 1.

CTRL: start / stop (streaming the person), reset, set {mode, backend, walk, max_speed, idle_after}, gesture, drive,
estop, quit. A set that repeats the current value is a no-op. The first SimState (note "loading") goes out before the
backend loads (< 1 s after start), then 2 Hz until the loop runs at 10 Hz.

Exit codes: 0 = quit / SIGTERM / SIGINT / --max-seconds, 2 = bad arguments, 4 = a port is already in use,
5 = the backend failed to load.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from k1live import protocol as P  # noqa: E402

DT = 0.02
ARM_VMAX = 6.0          # rad/s, reference slew limit (same as the offline reel)
ARM_AMAX = 80.0         # rad/s^2
ARM_RETURN_V = 1.0      # rad/s back to the relaxed pose
HOLD_S = 1.0            # person lost: hold the last arm pose this long
HUMAN_STALE_S = 0.25    # a HumanFrame older than this (capture time) does not drive anything
DRIVE_TIMEOUT_S = 0.3
FALL_RESET_S = 1.0
ARENA_R = 3.0
ACC = np.array([1.0, 1.0, 2.0])
DEC = np.array([4.0, 4.0, 6.0])
ATTRACT_ORDER = ("wave_hello", "bow", "welcome_open_arms", "clap", "point_guide", "heart", "raise_both_arms",
                 "stand_at_ease")
ATTRACT_PAUSE_S = 2.5
GESTURE_CACHE_VERSION = 1
HEADING_WMAX = 0.4     # rad/s, heading-hold correction cap
HAND_OFFSET = np.array([0.20, 0.0, 0.0])  # palm centre in *_wrist_roll_rubber_hand (k1/gestures.HAND_OFFSET)


# ------------------------------------------------------------------------------------------------ gestures
def gesture_cache_path() -> Path:
    src = (ROOT / "k1" / "gestures.py").read_bytes()
    h = hashlib.sha1(src + str(GESTURE_CACHE_VERSION).encode()).hexdigest()[:10]
    return ROOT / "logs" / "k1live" / f"gestures_{h}.npz"


def build_gesture_cache(path: str) -> None:
    """Child process: every k1/gestures.py clip (50 Hz, policy order) -> one npz (IK takes tens of seconds)."""
    try:
        os.nice(10)
    except OSError:
        pass
    from k1.gestures import GESTURES, ArmIK
    from k1.sim import K1Split

    ik = ArmIK(K1Split())
    out = {}
    for name, fn in GESTURES.items():
        m = fn(ik)
        out[f"{name}__q"], out[f"{name}__dq"], out[f"{name}__torso"] = m.q, m.dq, m.torso_quat
        out[f"{name}__mode"] = np.array(m.mode)
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)


class Gestures:
    """Loads the gesture cache, building it in a background process the first time."""

    def __init__(self, arm_idx):
        self.arm_idx = arm_idx
        self.path = gesture_cache_path()
        self.clips = {}
        self.proc = None
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.proc = mp.get_context("spawn").Process(target=build_gesture_cache, args=(str(self.path),),
                                                        daemon=True, name="k1live-gestures")
            self.proc.start()
        self.poll()

    def poll(self):
        if self.clips or not self.path.exists():
            return bool(self.clips)
        z = np.load(self.path)
        for k in z.files:
            if k.endswith("__q"):
                n = k[:-3]
                torso = z[f"{n}__torso"]
                pitch = 2.0 * np.arctan2(torso[:, 1], torso[:, 3])  # rotation about y (bow)
                self.clips[n] = {"q": z[k][:, self.arm_idx], "dq": z[f"{n}__dq"][:, self.arm_idx], "bow": pitch,
                                 "mode": str(z[f"{n}__mode"])}
        return True

    def close(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()


# ------------------------------------------------------------------------------------------------ helpers
class ArmLimiter:
    """Second-order follower: velocity <= vmax, acceleration <= amax, brakes in time (no overshoot)."""

    def __init__(self, q0):
        self.q = np.array(q0, np.float64)
        self.v = np.zeros_like(self.q)

    def reset(self, q0):
        self.q[:] = q0
        self.v[:] = 0.0

    def step(self, goal, dt=DT, vmax=ARM_VMAX, amax=ARM_AMAX):
        err = np.asarray(goal, np.float64) - self.q
        ad = amax * dt
        # fastest speed from which per-tick decrements of amax*dt still stop within |err| (discrete braking curve)
        v_brake = ad * (np.sqrt(0.25 + 2.0 * np.abs(err) / (ad * dt)) - 0.5)
        v_des = np.sign(err) * np.minimum(np.minimum(v_brake, np.abs(err) / dt), vmax)
        self.v = np.clip(v_des, self.v - ad, self.v + ad)
        self.q = self.q + self.v * dt
        return self.q.copy(), self.v.copy()


def limit_cmd(target, cur, max_speed, dt=DT):
    """Speed clip (vx backward <= min(0.4, vx_max)) then accel / decel limits per component."""
    vx, vy, wz = max_speed
    t = np.array([np.clip(target[0], -min(0.4, vx), vx), np.clip(target[1], -vy, vy), np.clip(target[2], -wz, wz)])
    t[np.abs(t) < 0.03] = 0.0
    slowing = (np.abs(t) < np.abs(cur)) | (np.sign(t) * np.sign(cur) < 0)
    lim = np.where(slowing, DEC, ACC) * dt
    return cur + np.clip(t - cur, -lim, lim)


def leash(cmd, x, y, yaw, r_max=ARENA_R):
    """Cut the outward part of the commanded velocity at the arena edge."""
    r = math.hypot(x, y)
    if r < r_max:
        return cmd
    c, s = math.cos(yaw), math.sin(yaw)
    vw = np.array([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1]])
    n = np.array([x, y]) / r
    out = vw @ n
    if out <= 0:
        return cmd
    vw = vw - out * n
    return np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], cmd[2]])


class Hist:
    """Whole-run distribution in constant memory (the server may run all day): fixed bins, percentiles to one bin
    (reported at the bin centre), exact n / mean / max."""

    def __init__(self, hi, step):
        self.step = step
        self.c = np.zeros(int(np.ceil(hi / step)) + 1, np.int64)
        self.n, self.sum, self.max = 0, 0.0, None

    def add(self, vals):
        if not len(vals):
            return
        v = np.asarray(vals, np.float64)
        np.add.at(self.c, np.clip((v / self.step).astype(np.int64), 0, len(self.c) - 1), 1)
        self.n += len(v)
        self.sum += float(v.sum())
        self.max = float(v.max()) if self.max is None else max(self.max, float(v.max()))

    def pct(self, q):
        if not self.n:
            return None
        if q >= 100:
            return round(self.max, 2)
        k = int(np.searchsorted(np.cumsum(self.c), q / 100.0 * self.n))
        return round(min((k + 0.5) * self.step, self.max), 2)

    def mean(self):
        return round(self.sum / self.n, 2) if self.n else None


def pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


class WalkLimiter:
    """Walking command shaping. The VelocityLimiter (k1live.locomotion) when present: per-backend presets with the
    loco policy's direction-specific start levels (split: sideways >= 0.4 m/s, turning in place >= 0.9 rad/s, below
    that the robot does not move), caps vx 0.6 / back 0.35 / vy 0.5 / wz 1.2, accel 0.8 m/s^2, decel 3 m/s^2;
    `max_speed` (forward m/s) scales every command (max_speed / 0.6). Without it: limit_cmd (caps x scale)."""

    def __init__(self, backend_name, max_speed=0.6):
        self.max_speed = float(max_speed)
        self.cur = np.zeros(3)
        self.vl = None
        try:
            from k1live.locomotion import VelocityLimiter

            self.vl = VelocityLimiter.for_backend(backend_name, scale=self.scale)
        except Exception:  # module not there (core-only checkout): simple caps + accel
            self.vl = None

    @property
    def scale(self):
        return self.max_speed / 0.6

    def set_max_speed(self, v):
        self.max_speed = float(np.clip(v, 0.0, 1.0))
        if self.vl is not None:
            self.vl.scale = self.scale

    def reset(self):
        self.cur[:] = 0.0
        if self.vl is not None:
            self.vl.reset()

    def __call__(self, target):
        if self.vl is not None:
            self.cur = np.asarray(self.vl({"vx": target[0], "vy": target[1], "wz": target[2]}, DT), np.float64)
        else:
            self.cur = limit_cmd(target, self.cur, np.array([0.6, 0.3, 0.6]) * self.scale)
        return self.cur.copy()


def sonic_available():
    """sonic is offered only when its files are there (checkpoint + staged K1 MJCF / meshes); constructing it to find
    out would cost ~1.5 s of CPU inside the control process."""
    try:
        import importlib.util

        if importlib.util.find_spec("k1live.backends.sonic") is None:
            return False
        from k1live.backends import sonic as S

        ok = getattr(S.SonicBackend, "available", None)
        if ok is not None:
            return bool(ok())
        from sonic.robots import k1_scene_xml

        k1_scene_xml()  # raises FileNotFoundError without the staged MJCF / meshes
        return Path(S.DEFAULT_VARIANT).exists()
    except Exception:
        return False


def sonic_human_available():
    """sonic_human needs a K1 SONIC checkpoint with an SMPL encoder mode (k1live/backends/sonic_human.py)."""
    try:
        from k1live.backends.sonic_human import SonicHumanBackend

        return SonicHumanBackend.available() and sonic_available()
    except Exception:
        return False


def hybrid_available():
    """hybrid needs the SONIC walking checkpoint, the staged K1 MJCF and the planner ONNX (k1live/backends/hybrid.py)."""
    try:
        from k1live.backends.hybrid import HybridBackend

        return HybridBackend.available()
    except Exception:
        return False


def available_backends():
    return ["split"] + (["sonic"] if sonic_available() else []) + (["sonic_human"] if sonic_human_available() else []) + \
        (["hybrid"] if hybrid_available() else [])


# ------------------------------------------------------------------------------------------------ server
class HandError:
    """Palm position error (cm, torso frame, mean of both hands) of the robot vs its arm goal: the visible part of
    the joint error (shoulder yaw / wrist roll about a straight limb barely move the hand)."""

    def __init__(self, be):
        import mujoco

        self.mj, self.be = mujoco, be
        m = be.model
        self.d = mujoco.MjData(m)
        self.hands = [m.body(f"{s}_wrist_roll_rubber_hand").id for s in ("left", "right")]
        self.torso = m.body("torso_link").id
        self.adr = be.qadr[be.arm_idx]

    def _palms(self):
        d = self.d
        self.mj.mj_kinematics(self.be.model, d)
        R, p = d.xmat[self.torso].reshape(3, 3), d.xpos[self.torso]
        return [R.T @ (d.xpos[h] + d.xmat[h].reshape(3, 3) @ HAND_OFFSET - p) for h in self.hands]

    def __call__(self, goal):
        self.d.qpos[:] = self.be.data.qpos
        a = self._palms()
        self.d.qpos[self.adr] = goal
        g = self._palms()
        return float(np.mean([np.linalg.norm(x - y) for x, y in zip(a, g)]) * 100.0)


class SimServer:
    def __init__(self, args, ctx, st_pub):
        self.args = args
        self.ctx, self.st_pub = ctx, st_pub
        self.hum = ctx.socket(zmq.SUB)
        self.hum.setsockopt(zmq.LINGER, 0)
        self.hum.setsockopt(zmq.RCVHWM, 50)
        self.hum.connect(f"tcp://{P.HOST}:{args.human_port}")
        self.hum.setsockopt(zmq.SUBSCRIBE, P.TOPIC_HUMAN)
        self.ctl = ctx.socket(zmq.SUB)
        self.ctl.setsockopt(zmq.LINGER, 0)
        self.ctl.connect(f"tcp://{P.HOST}:{args.ctrl_port}")
        self.ctl.setsockopt(zmq.SUBSCRIBE, P.TOPIC_CTRL)

        from k1live.backends.base import make_backend

        self.make_backend = make_backend
        self.backend_name = args.backend
        self.be = make_backend(args.backend, **self.backend_kw(args.backend))
        self.backends = available_backends()
        if self.backend_name not in self.backends:
            self.backends.append(self.backend_name)
        self.lim = ArmLimiter(self.be.arm_default)
        self.gestures = Gestures(self.be.arm_idx)
        self.hand_err = self._hand_error()
        self.render = None
        if not args.no_render:
            from k1live.render import RenderLink

            self.render = RenderLink(self.be.render_model_fn, self.be.model.nq, port=args.robot_port,
                                     size=tuple(int(v) for v in args.render_size.split("x")), fps=args.render_fps)

        # operator state
        self.mode = args.mode
        self.streaming = not args.no_autostart
        self.walk = not args.no_walk
        self.estop = False
        self.walk_lim = WalkLimiter(getattr(self.be, "limiter_preset", self.backend_name), args.max_speed)
        self.idle_after = args.idle_after
        # runtime state
        self.hf = None                  # newest HumanFrame (dict)
        self.hf_used_seq = -1
        self.t_start = time.time()
        self.t_person = self.t_start    # wall time a person was last tracked (start: wait idle_after first)
        self.drive, self.t_drive = np.zeros(3), -1e9
        self.cmd = np.zeros(3)
        self.yaw_ref = None             # heading held while no turn is commanded
        self.goal = self.be.arm_default.copy()
        self.play = None                # {"name", "i", "clip", "kind": "ctrl" | "attract"}
        self.attract_i, self.t_attract_next = 0, 0.0
        self.fallen, self.t_fall, self.resets = False, 0.0, 0
        self.tracking = False
        self.arm_err = None
        self.note = ""
        self.pending_backend = None
        self._new_be = None
        self.last_info = {}
        self.stop = False
        self.errors = 0
        self.ctrl_hz, self.rt = 0.0, 0.0
        # metrics
        self.lat_ms, self.age_ms = [], []
        self.tick_ms, self.pol_ms, self.late_ms = [], [], []
        self.err_goal, self.err_ref, self.err_hand = [], [], []
        self.H = {"lat": Hist(2000.0, 0.25), "pol": Hist(100.0, 0.01), "tick": Hist(100.0, 0.01),
                  "err": Hist(180.0, 0.05), "err_ref": Hist(180.0, 0.05), "hand": Hist(100.0, 0.05)}
        self.skipped_sec = 0
        self.skipped = 0
        self.ticks = 0
        self.n_track = 0
        self.n_person_walk = 0          # ticks in this second whose walking came from the person (not drive)
        self.person_walk_ticks = 0
        self.hz_hist = collections.deque()
        self.falls = 0
        self.n_human = 0
        log_dir = Path(args.log_dir) if args.log_dir else ROOT / "logs" / "k1live" / time.strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.log = open(log_dir / "sim.jsonl", "a", buffering=1)
        self.event("start", backend=self.backend_name, args=vars(args), pid=os.getpid())

    # -------------------------------------------------------------------------------------------- logging
    def event(self, ev, **kw):
        self.log.write(json.dumps({"t": round(time.time(), 3), "event": ev, **kw}, default=str) + "\n")

    # -------------------------------------------------------------------------------------------- inputs
    def drain(self, wall):
        while True:
            try:
                fr = self.hum.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                d = json.loads(fr[1].decode())
            except Exception:
                continue
            d["_t_recv"] = wall
            self.hf = d
            self.n_human += 1
        while True:
            try:
                fr = self.ctl.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                m = P.decode(fr, P.CtrlMsg)
            except Exception:
                continue
            self.on_ctrl(m, wall)

    def on_ctrl(self, m, wall):
        a = m.args or {}
        if m.cmd != P.CMD_DRIVE:
            self.event("ctrl", cmd=m.cmd, args=a)
        if m.cmd == P.CMD_START:
            self.streaming, self.estop = True, False
        elif m.cmd == P.CMD_STOP:
            self.streaming = False
        elif m.cmd == P.CMD_RESET:
            self.do_reset("ctrl")
        elif m.cmd == P.CMD_ESTOP:
            self.estop, self.streaming, self.play = True, False, None
            self.drive[:] = 0.0
        elif m.cmd == P.CMD_DRIVE:
            self.drive = np.array([float(a.get("vx", 0.0)), float(a.get("vy", 0.0)), float(a.get("wz", 0.0))])
            self.t_drive = wall
        elif m.cmd == P.CMD_GESTURE:
            self.start_gesture(str(a.get("name", "")), "ctrl")
        elif m.cmd == P.CMD_SET:
            if a.get("mode") in ("live", "idle") and a["mode"] != self.mode:
                self.mode = a["mode"]
                if self.play is not None and self.play["kind"] == "attract":
                    self.play = None
            if "walk" in a:
                self.walk = bool(a["walk"])
            if "max_speed" in a:  # forward m/s (a number, or {"vx": ...}); scales every walking command
                ms = a["max_speed"]
                v = ms.get("vx", self.walk_lim.max_speed) if isinstance(ms, dict) else ms
                if float(v) != self.walk_lim.max_speed:
                    self.walk_lim.set_max_speed(float(v))
            if "idle_after" in a:
                self.idle_after = float(a["idle_after"])
            if a.get("backend") and a["backend"] != self.backend_name and self.pending_backend is None:
                self.switch_backend(str(a["backend"]))
        elif m.cmd == P.CMD_QUIT:
            self.stop = True

    def backend_kw(self, name):
        a = self.args
        if name == "hybrid":
            return {"trim_gain": a.arm_trim, "legs": a.hybrid_legs, "walk_arm_human": a.walk_arm_human,
                    "arm_mode": a.arm_mode}
        return {"waist": a.waist, "trim_gain": a.arm_trim, "arm_mode": a.arm_mode} if name == "split" else {}

    def switch_backend(self, name):
        if name not in P.BACKENDS:
            self.note = f"unknown backend {name}"
            return
        self.pending_backend = name

        def load():
            try:
                be = self.make_backend(name, **self.backend_kw(name))
                be.reset()  # here, not on the control thread (sonic's reset runs a 1.5 s warm-up: a 110 ms stall)
                self._new_be = be
            except Exception as e:  # stay on the current backend
                self._new_be = None
                self.note = f"backend {name} failed: {type(e).__name__}: {e}"
                self.event("backend_failed", name=name, error=str(e))
                self.pending_backend = None

        self._new_be = None
        threading.Thread(target=load, daemon=True).start()

    def apply_backend(self):
        be = self._new_be
        if be is None:
            return
        old = self.be
        self.be, self._new_be, self.backend_name, self.pending_backend = be, None, self.pending_backend, None
        old.close()
        self.lim = ArmLimiter(self.be.arm_default)
        self.gestures = Gestures(self.be.arm_idx)
        self.hand_err = self._hand_error()
        self.cmd[:] = 0.0
        self.walk_lim = WalkLimiter(getattr(self.be, "limiter_preset", self.backend_name), self.walk_lim.max_speed)
        if self.render is not None and self.render.model_fn != self.be.render_model_fn:
            from k1live.render import RenderLink

            self.render.close()
            self.render = RenderLink(self.be.render_model_fn, self.be.model.nq, port=self.args.robot_port)
        if self.backend_name not in self.backends:
            self.backends.append(self.backend_name)
        self.event("backend", name=self.backend_name)

    def _hand_error(self):
        try:
            return HandError(self.be)
        except Exception:  # a backend model without the K1 hand bodies: no hand metric
            return None

    def do_reset(self, why):
        self.be.reset()
        self.lim.reset(self.be.arm_default)
        self.goal = self.be.arm_default.copy()
        self.cmd[:] = 0.0
        self.walk_lim.reset()
        self.yaw_ref = None
        self.fallen = False
        self.resets += 1
        self.play = None if (self.play and self.play["kind"] == "attract") else self.play
        self.event("reset", why=why, resets=self.resets)

    # -------------------------------------------------------------------------------------------- gestures
    def start_gesture(self, name, kind):
        self.gestures.poll()
        clip = self.gestures.clips.get(name)
        if clip is None:
            self.note = f"gesture {name!r} not available" + ("" if self.gestures.clips else " (still building)")
            return False
        if clip["mode"] == "mimic" and not self.be.supports.get("bow"):
            if kind == "ctrl":
                self.note = f"gesture {name!r} needs a backend that can bow"
            return False
        self.play = {"name": name, "i": 0, "clip": clip, "kind": kind}
        self.event("gesture", name=name, kind=kind)
        return True

    def next_attract(self, wall):
        if wall < self.t_attract_next:
            return
        for _ in range(len(ATTRACT_ORDER)):
            name = ATTRACT_ORDER[self.attract_i % len(ATTRACT_ORDER)]
            self.attract_i += 1
            if self.start_gesture(name, "attract"):
                return

    # -------------------------------------------------------------------------------------------- one tick
    def tick(self, wall):
        self.drain(wall)
        if self._new_be is not None:
            self.apply_backend()
        hf = self.hf
        fresh = hf is not None and wall - hf["t_capture"] < HUMAN_STALE_S
        person = bool(fresh and hf.get("present") and hf.get("arm_q") is not None)
        if person:
            self.t_person = wall
        absent = wall - self.t_person
        following = self.mode == "live" and self.streaming and not self.estop
        new_frame = hf is not None and hf["seq"] != self.hf_used_seq
        waist = bow = None
        vmax = ARM_VMAX
        self.tracking = False
        # --- arms --------------------------------------------------------------------------------------
        if self.play is not None and self.play["kind"] == "attract" and (person or self.estop or self.fallen):
            self.play = None  # someone arrived: stop the attract clip (live: follow them)
        if self.play is not None:
            clip, i = self.play["clip"], self.play["i"]
            self.goal = clip["q"][min(i, len(clip["q"]) - 1)]
            if self.be.supports.get("bow"):
                bow = float(clip["bow"][min(i, len(clip["q"]) - 1)])
            self.play["i"] += 1
            if self.play["i"] >= len(clip["q"]):
                self.play = None
                self.t_attract_next = wall + ATTRACT_PAUSE_S
        elif following and person:
            self.goal = np.asarray(hf["arm_q"], np.float64)
            waist, bow = hf.get("waist_yaw"), hf.get("bow")
            self.tracking = True
        elif following and absent < HOLD_S:
            pass  # hold the last pose
        else:
            nobody = absent >= self.idle_after
            if not self.estop and not self.fallen and nobody and self.mode in ("live", "idle") and \
                    self.args.attract and self.gestures.poll():
                self.next_attract(wall)
            self.goal = self.be.arm_default.copy()
            vmax = ARM_RETURN_V
        if vmax == ARM_VMAX and self.play is None:   # following the person: the configured limits (--arm-vmax/-amax)
            ref_q, ref_dq = self.lim.step(self.goal, vmax=self.args.arm_vmax, amax=self.args.arm_amax)
        else:
            ref_q, ref_dq = self.lim.step(self.goal, vmax=vmax if self.play is None else ARM_VMAX)
        # --- walking -------------------------------------------------------------------------------------
        target = np.zeros(3)
        if not self.estop and not self.fallen:
            if wall - self.t_drive < DRIVE_TIMEOUT_S:
                target = self.drive
            elif self.walk and following and person and self.play is None and hf.get("loco"):
                lo = hf["loco"]
                target = np.array([lo.get("vx", 0.0), lo.get("vy", 0.0), lo.get("wz", 0.0)], np.float64)
                if np.any(np.abs(target) > 0.02):
                    self.n_person_walk += 1
        x, y, yaw = self.be.root_pose()
        cmd = self.walk_lim(target)
        walk_cmd = cmd.copy()          # without the heading hold: what a backend with supports["intent"] walks on
        if cmd[2] != 0.0 or self.yaw_ref is None:  # turning on purpose: the heading to hold is wherever it stops
            self.yaw_ref = yaw
        elif self.args.heading_hold > 0 and not self.fallen and self.be.supports.get("walk"):
            # arm motion makes the standing / walking loco robot yaw (waving one arm: 125 deg in 30 s); a small
            # yaw-rate correction below the policy's turn-in-place start level holds the heading within ~2 deg
            err = (self.yaw_ref - yaw + math.pi) % (2 * math.pi) - math.pi
            cmd[2] = float(np.clip(self.args.heading_hold * err, -HEADING_WMAX, HEADING_WMAX))
        self.cmd = leash(cmd, x, y, yaw)
        # --- step -----------------------------------------------------------------------------------------
        extra = {"human": hf if self.tracking else None} if self.be.supports.get("human") else {}
        if self.be.supports.get("intent"):
            extra["walk_cmd"] = leash(walk_cmd, x, y, yaw)
        info = self.be.step(ref_q, waist_yaw=waist, bow=bow, loco_cmd=self.cmd, arm_dq=ref_dq, **extra)
        self.last_info = info
        if info.get("phase") not in (None, "stand"):
            # hybrid walking / settling on SONIC: the heading hold is not applied; hold wherever it ends up standing
            self.yaw_ref = self.be.root_pose()[2]
        if new_frame:
            self.hf_used_seq = hf["seq"]
            lat = (time.time() - hf["t_capture"]) * 1e3  # capture -> first control tick that used it
            if self.tracking:
                self.lat_ms.append(lat)
                self.age_ms.append((hf["_t_recv"] - hf["t_capture"]) * 1e3)
        self.pol_ms.append(info.get("policy_ms", 0.0))
        if self.tracking:
            self.n_track += 1
            q = self.be.arm_q_now()
            eg = float(np.degrees(np.mean(np.abs(q - self.goal))))
            er = float(np.degrees(np.mean(np.abs(q - ref_q))))
            self.err_goal.append(eg)
            self.err_ref.append(er)
            if self.hand_err is not None and self.n_track % 5 == 0:
                self.err_hand.append(self.hand_err(self.goal))
        # --- falls ----------------------------------------------------------------------------------------
        if not self.fallen and self.be.fallen():
            self.fallen, self.t_fall = True, wall
            self.falls += 1
            self.play = None
            self.event("fall", pelvis_z=round(self.be.pelvis_height(), 3), tilt=round(self.be.tilt_deg(), 1),
                       mode=self.mode, cmd=[round(v, 3) for v in self.cmd])
        elif self.fallen and wall - self.t_fall >= FALL_RESET_S:
            self.do_reset("fall")
        if self.render is not None:
            self.render.push(self.be.data.qpos)

    # -------------------------------------------------------------------------------------------- state
    def sim_state(self, wall, note=None):
        hf = self.hf
        mode = "gesture" if (self.play is not None and self.play["kind"] == "ctrl") else self.mode
        be_note = self.be.note() if hasattr(self.be, "note") else ""
        notes = [n for n in (note if note is not None else self.note, be_note,
                             "attract" if self.play is not None and self.play["kind"] == "attract" else "",
                             "" if self.gestures.clips else "gestures building") if n]
        return P.SimState(
            t=wall, mode=mode, backend=self.backend_name, streaming=self.streaming, walk_enabled=self.walk,
            estop=self.estop, fallen=self.fallen, resets=self.resets, pelvis_z=round(self.be.pelvis_height(), 3),
            tilt_deg=round(self.be.tilt_deg(), 1), loco_cmd=[round(float(v), 3) for v in self.cmd],
            human_present=bool(hf is not None and hf.get("present") and wall - hf["t_capture"] < HUMAN_STALE_S),
            human_age_ms=None if hf is None else round((wall - hf["t_capture"]) * 1e3, 1),
            ctrl_hz=round(self.ctrl_hz, 2), rt_factor=round(self.rt, 3),
            gesture=None if self.play is None else self.play["name"], arm_err_deg=self.arm_err,
            note="; ".join(notes), backends=list(self.backends))

    def summary(self, wall):
        """1 Hz log line; also resets the per-second windows."""
        lat, pol = self.lat_ms, self.pol_ms
        self.arm_err = round(float(np.mean(self.err_goal)), 2) if self.err_goal else None
        rec = {"t": round(wall, 3), "event": "sec", "mode": self.mode, "stream": self.streaming,
               "tracking_frac": round(self.n_track / max(len(self.tick_ms), 1), 2), "person": bool(self.hf and self.hf.get("present")),
               "play": None if self.play is None else self.play["name"], "hz": round(self.ctrl_hz, 2),
               "rt": round(self.rt, 4), "skipped": self.skipped, "skipped_sec": self.skipped_sec, "person_walk_ticks": self.n_person_walk, "lat_p50": pct(lat, 50), "lat_p95": pct(lat, 95),
               "lat_max": pct(lat, 100), "age_p50": pct(self.age_ms, 50), "pol_p50": pct(pol, 50),
               "pol_p95": pct(pol, 95), "tick_p95": pct(self.tick_ms, 95), "late_p95": pct(self.late_ms, 95),
               "arm_err_goal": self.arm_err, "arm_err_ref": round(float(np.mean(self.err_ref)), 2)
               if self.err_ref else None, "hand_err_cm": round(float(np.mean(self.err_hand)), 2)
               if self.err_hand else None, "cmd": [round(float(v), 3) for v in self.cmd],
               "pelvis_z": round(self.be.pelvis_height(), 3), "tilt": round(self.be.tilt_deg(), 1),
               "xy_yaw": [round(v, 3) for v in self.be.root_pose()],
               "yaw_err_deg": None if self.yaw_ref is None else round(math.degrees(
                   (self.yaw_ref - self.be.root_pose()[2] + math.pi) % (2 * math.pi) - math.pi), 2), "resets": self.resets, "falls": self.falls,
               "n_human": self.n_human}
        if hasattr(self.be, "note"):
            li = getattr(self, "last_info", None) or {}
            rec["backend_state"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in li.items()
                                    if k in ("phase", "wL", "wA", "wP", "planner_infer_ms")}
            rec["backend_stats"] = dict(getattr(self.be, "stats", {}))
        self.log.write(json.dumps(rec) + "\n")
        for k, v in (("lat", self.lat_ms), ("pol", self.pol_ms), ("tick", self.tick_ms), ("err", self.err_goal),
                     ("err_ref", self.err_ref), ("hand", self.err_hand)):
            self.H[k].add(v)
        self.skipped_sec = 0
        self.lat_ms, self.age_ms, self.pol_ms, self.tick_ms, self.late_ms = [], [], [], [], []
        self.n_track = 0
        self.err_goal, self.err_ref, self.err_hand = [], [], []
        self.person_walk_ticks += self.n_person_walk
        self.n_person_walk = 0
        return rec

    # -------------------------------------------------------------------------------------------- loop
    def run(self):
        a = self.args
        wall_t0 = time.perf_counter()
        t_next = time.perf_counter()
        t_state = t_sec = 0.0
        t_end = time.perf_counter() + a.max_seconds if a.max_seconds else None
        spin = a.spin_ms / 1e3
        from k1live.timing import sleep_until
        while not self.stop and not STOP.is_set():
            # ---- wait for the tick (sleep, then a short spin for precision) ----------------------------
            sleep_until(t_next, spin=spin)
            start = time.perf_counter()
            late = start - t_next
            if late > DT:  # do not catch up: skip the missed ticks
                n = int(late / DT)
                self.skipped += n
                t_next += n * DT
                if not self.skipped_sec or n > 2:  # at most one event per second unless it is a real stall
                    self.event("skip", ticks=n, late_ms=round(late * 1e3, 1))
                self.skipped_sec += n
            self.late_ms.append(max(0.0, (start - t_next) * 1e3))
            t_next += DT
            wall = time.time()
            sim_before = self.be.sim_time
            try:
                self.tick(wall)
                self.errors = 0
            except Exception as e:  # one bad message / tick must not end the show; a persistent fault does
                import traceback

                self.errors += 1
                self.event("error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])
                if self.errors >= 50:
                    raise
            end = time.perf_counter()
            self.tick_ms.append((end - start) * 1e3)
            self.ticks += 1
            # rates over the last second: ticks per wall second, simulated seconds per wall second
            dsim = self.be.sim_time - sim_before
            self.hz_hist.append((end, dsim if dsim >= 0 else DT))  # a reset rewinds sim time: count one tick
            while self.hz_hist and end - self.hz_hist[0][0] > 1.0:
                self.hz_hist.popleft()
            if end - wall_t0 >= 1.0 and len(self.hz_hist) > 1:
                span = max(self.hz_hist[-1][0] - self.hz_hist[0][0], 1e-6)
                self.ctrl_hz = (len(self.hz_hist) - 1) / span
                self.rt = sum(d for _, d in list(self.hz_hist)[1:]) / span
            if end - t_state >= 0.1:
                t_state = end
                self.st_pub.send_multipart(P.encode(P.TOPIC_SIM_STATE, self.sim_state(wall)))
            if end - t_sec >= 1.0:
                if t_sec:
                    self.summary(wall)
                t_sec = end
            if self.note and self.ticks % 250 == 0:
                self.note = ""  # transient notes fade after a few seconds
            if t_end is not None and end >= t_end:
                break
        return 0

    def close(self):
        wall = time.time()
        self.summary(wall)
        res = self.final_stats()
        (self.log_dir / "summary.json").write_text(json.dumps(res, indent=2))
        self.event("exit", **res)
        self.log.close()
        if self.render is not None:
            self.render.close()
        self.gestures.close()
        self.be.close()
        return res

    def final_stats(self):
        H = self.H
        return {"ticks": self.ticks, "skipped": self.skipped, "falls": self.falls, "resets": self.resets,
                "arms": {"mode": self.args.arm_mode, "vmax": self.args.arm_vmax, "amax": self.args.arm_amax},
                "person_walk_s": round(self.person_walk_ticks * DT, 2),
                "lat_ms": {"n": H["lat"].n, "p50": H["lat"].pct(50), "p95": H["lat"].pct(95), "max": H["lat"].pct(100)},
                "policy_ms": {"p50": H["pol"].pct(50), "p95": H["pol"].pct(95), "max": H["pol"].pct(100)},
                "tick_ms": {"p50": H["tick"].pct(50), "p95": H["tick"].pct(95), "max": H["tick"].pct(100)},
                "arm_err_deg": {"vs_ik_goal_mean": H["err"].mean(), "vs_ik_goal_p90": H["err"].pct(90),
                                "vs_reference_mean": H["err_ref"].mean()},
                "hand_err_cm": {"mean": H["hand"].mean(), "p90": H["hand"].pct(90)},
                "wall_s": round(time.time() - self.t_start, 1), "sim_s": round(self.be.sim_time, 2),
                "log_dir": str(self.log_dir)}


STOP = threading.Event()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="split", choices=list(P.BACKENDS))
    ap.add_argument("--mode", default="live", choices=["live", "idle"])
    ap.add_argument("--no-autostart", action="store_true", help="wait for CTRL start before following the person")
    ap.add_argument("--no-walk", action="store_true", help="start with walking from the person off")
    ap.add_argument("--no-attract", dest="attract", action="store_false", help="no gesture loop when nobody is there")
    ap.add_argument("--idle-after", type=float, default=8.0, help="s without a person before the attract loop")
    ap.add_argument("--max-speed", type=float, default=0.6, help="forward m/s; scales all walking commands")
    ap.add_argument("--waist", action="store_true", help="split: drive the waist yaw from the person (off: loco owns it)")
    ap.add_argument("--arm-trim", type=float, default=2.0, help="split / hybrid: outer-loop arm trim gain 1/s (0 = off)")
    ap.add_argument("--arm-mode", default=os.environ.get("K1LIVE_ARM_MODE", "student"), choices=["student", "pd"],
                    help="split / hybrid (standing): arms by the mimic student, or pd = PD straight to the reference with gravity "
                    "compensation (docs/architecture.md; recommended with --arm-vmax 12 --arm-amax 150). "
                    "Default $K1LIVE_ARM_MODE or student")
    ap.add_argument("--arm-vmax", type=float, default=float(os.environ.get("K1LIVE_ARM_VMAX", ARM_VMAX)),
                    help=f"rad/s, arm reference speed limit while following the person (default {ARM_VMAX})")
    ap.add_argument("--arm-amax", type=float, default=float(os.environ.get("K1LIVE_ARM_AMAX", ARM_AMAX)),
                    help=f"rad/s^2, arm reference acceleration limit while following the person (default {ARM_AMAX})")
    ap.add_argument("--hybrid-legs", default="switch", choices=["switch", "sonic"],
                    help="hybrid: legs = ROBOTIS loco while standing + SONIC while walking (switch), or SONIC always")
    ap.add_argument("--walk-arm-human", type=float, default=0.0,
                    help="hybrid: weight of the person's arms over the planner's arm swing while walking (0..1)")
    ap.add_argument("--heading-hold", type=float, default=2.0, help="yaw-rate correction gain 1/s when no turn is "
                    "commanded (0 = off); capped at 0.4 rad/s")
    ap.add_argument("--no-render", action="store_true", help="no render process / no /robot.mjpg")
    ap.add_argument("--render-size", default="1280x720")
    ap.add_argument("--render-fps", type=float, default=30.0)
    ap.add_argument("--human-port", type=int, default=P.HUMAN_PORT)
    ap.add_argument("--ctrl-port", type=int, default=P.CTRL_PORT)
    ap.add_argument("--state-port", type=int, default=P.SIM_STATE_PORT)
    ap.add_argument("--robot-port", type=int, default=P.ROBOT_HTTP_PORT)
    ap.add_argument("--log-dir", default="", help="default logs/k1live/<YYYYmmdd-HHMMSS>/")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--spin-ms", type=float, default=0.5, help="busy-wait this long before a tick (k1live/timing.py)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    ctx = zmq.Context()
    st_pub = ctx.socket(zmq.PUB)
    st_pub.setsockopt(zmq.LINGER, 0)
    try:
        st_pub.bind(f"tcp://{P.HOST}:{args.state_port}")
    except zmq.ZMQError as e:
        print(f"[sim] bind {args.state_port} failed: {e}", file=sys.stderr, flush=True)
        ctx.destroy(0)
        return 4
    loading = threading.Event()

    def loading_state():  # until the control loop publishes itself (sockets are not shared across threads)
        while not loading.is_set():
            st_pub.send_multipart(P.encode(P.TOPIC_SIM_STATE, P.SimState(
                t=P.now(), mode=args.mode, backend=args.backend, streaming=False, walk_enabled=not args.no_walk,
                estop=False, fallen=False, resets=0, pelvis_z=0.0, tilt_deg=0.0, loco_cmd=[0.0, 0.0, 0.0],
                human_present=False, human_age_ms=None, ctrl_hz=0.0, rt_factor=0.0, note="loading")))
            loading.wait(0.5)

    th = threading.Thread(target=loading_state, daemon=True, name="loading")
    th.start()
    print(f"[sim] state on {args.state_port}; loading backend {args.backend}", flush=True)
    try:
        srv = SimServer(args, ctx, st_pub)
    except Exception as e:
        loading.set()
        th.join(1.0)
        import traceback

        traceback.print_exc()
        print(f"[sim] backend {args.backend} failed: {e}", file=sys.stderr, flush=True)
        ctx.destroy(0)
        return 5
    loading.set()
    th.join(1.0)
    print(f"[sim] running: backend {srv.backend_name}, mode {srv.mode}, split arms {args.arm_mode}, arm reference "
          f"{args.arm_vmax:g} rad/s {args.arm_amax:g} rad/s^2, log {srv.log_dir}", flush=True)
    code = 0
    try:
        code = srv.run()
    finally:
        res = srv.close()
        ctx.destroy(0)
        print(f"[sim] exit {code}: {json.dumps(res)}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
