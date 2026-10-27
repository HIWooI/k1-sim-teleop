# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""GEAR-SONIC kinematic planner (planner_sonic.onnx) in Python, plus the G1 -> K1 retarget used by the K1 live demo
(MuJoCo simulation only - nothing here talks to a robot).

Port of the C++ deploy (vendor/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref, upstream b042411):
  * PlannerModel      localmotion_kplanner_{onnx,tensorrt}.hpp - the 11 ONNX inputs; advanced inputs at the deploy
                      defaults (has_specific_target 0, zero waypoints, height -1, seed 1234, allowed_pred_num_tokens
                      = TensorRT mask [0,0,0,1,1,1,0,...] -> 9..11 tokens = 36..44 frames at 30 Hz = 1.2..1.47 s)
  * resample_50hz     LocalMotionPlannerBase::ResampleGeneratedSequence50Hz (30 Hz -> 50 Hz, lerp / slerp,
                      joint velocity = forward difference * 50, last frame copies the previous velocity)
  * context_from      LocalMotionPlannerBase::InitializeContext / UpdateContextFromMotion (4 frames at 30 Hz sampled
                      from the *current planner motion* at gen_frame = current_frame + motion_look_ahead_steps (2))
  * blend_tracks      G1Deploy::CurrentFrameAdvancement (rebase the old motion at current_frame, fade the new one in
                      over 8 frames starting at gen_frame - current_frame, then current_frame = 0 and advance)
  * PlannerStream     G1Deploy::Planner (10 Hz planner tick: replan on mode / facing / height change; for non-static
                      modes also on speed / direction change or every replan interval while moving) + the 50 Hz
                      frame advancement. The robot state is used only at initialisation, exactly like the C++
                      (anchor="plan"); anchor="robot" additionally re-anchors the context root xy + heading on the
                      robot at every replan (closed loop on position/heading, joints still from the plan).
    Additions for continuous (camera) commands, all off when the tolerances are 0: change tolerances on direction /
    facing / speed and a minimum time between replans, so a slowly varying command does not replan every 0.1 s.
    Not ported: the idle re-adapt blend of the lower body toward the measured joints when the idle clip runs out.

  * G1ToK1            G1 qpos (36) -> K1 frame (root pos, root quat wxyz, 23 joints in k1live.protocol.POLICY_JOINTS
                      order): K1 pelvis = G1 pelvis with the G1 waist roll/pitch folded in (K1 torso = G1 torso),
                      lowered by dz relative to the feet (more knee bend, ~ the K1 default stance); feet = G1
                      ankle_roll_link poses (same sole height, K1 stance width);
                      legs by damped-least-squares IK on the K1 MJCF; shoulders by matching the upper-arm orientation
                      in the torso frame (G1 shoulder axes are tilted 16 deg, K1's are not); elbow / wrist roll /
                      waist yaw copied; K1 joint limits; then each frame grounded on the K1 foot contact spheres.
  * KinematicPlanner  what the live service uses (robot="k1" or "g1"): reset(qpos), command(vx, vy, wz | ...),
                      step(state) -> RefWindow (current + future 50 Hz frames in robot world coordinates).

    from k1live.planner import KinematicPlanner
    kp = KinematicPlanner(robot="k1", async_=True)      # async: inference in a worker thread (as the C++ planner thread)
    kp.reset(sim_qpos)                                  # robot qpos (7 + 23): world offset = robot xy + heading
    kp.command(vx=0.4, vy=0.0, wz=0.0)                  # body-frame walking command (K1 m/s, rad/s)
    win = kp.step(sim_qpos)                             # call at 50 Hz; win.q[0] = reference now, win.q[1:] future
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLANNER_ONNX = ROOT / "models" / "planner" / "planner_sonic.onnx"
HF_REPO = "nvidia/GEAR-SONIC"

MODES = {"idle": 0, "slowWalk": 1, "walk": 2, "run": 3, "squat": 4, "kneelTwoLeg": 5, "kneelOneLeg": 6,
         "lyingFacedown": 7, "handCrawling": 8, "idleBoxing": 9, "walkBoxing": 10, "leftJab": 11, "rightJab": 12,
         "randomPunches": 13, "elbowCrawling": 14, "leftHook": 15, "rightHook": 16}
STATIC_MODES = {0, 4, 5, 6, 7, 9}                       # is_static_motion_mode
REPLAN_INTERVAL = {3: 0.1, 8: 0.2, 11: 1.0, 12: 1.0, 13: 1.0, 15: 1.0, 16: 1.0}  # run / crawl / punches, else 1.0
ALLOWED_TOKENS = (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0)      # localmotion_kplanner_tensorrt.hpp (the active backend)
DEFAULT_HEIGHT = 0.788740                               # PlannerConfig::default_height
CONTROL_DT = 0.02
PLANNER_DT = 0.1
BLEND_FRAMES = 8
G1_NQ = 29


def fetch_planner(dst: Path = PLANNER_ONNX) -> Path:
    """Download planner_sonic.onnx (public file, no token) into models/planner/."""
    if dst.exists():
        return dst
    import shutil

    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_REPO, "planner_sonic.onnx", token=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dst)
    return dst


# ------------------------------------------------------------------------------------------------ quaternions (wxyz)
def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, float), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def qyaw(q):
    """Heading (rad) of wxyz quaternions: direction of the rotated x axis in the xy plane."""
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    yaw = np.asarray(yaw, float)
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=-1)


def slerp(q0, q1, t):
    """Vectorised slerp (math_utils quat_slerp_d: shortest path, lerp when nearly parallel)."""
    q0, q1 = np.atleast_2d(np.asarray(q0, float)), np.atleast_2d(np.asarray(q1, float))
    t = np.broadcast_to(np.asarray(t, float).reshape(-1, 1), (len(q0), 1)) if np.ndim(t) else np.full((len(q0), 1), t)
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    near = d > 0.9995
    a = np.where(near, 1 - t, np.sin((1 - t) * th) / np.where(near, 1, s))
    b = np.where(near, t, np.sin(t * th) / np.where(near, 1, s))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------------------------------------ 50 Hz tracks
@dataclass
class Track:
    """A 50 Hz motion (planner_motion_ / MotionSequence with body 0 only): joints in MuJoCo qpos order."""
    pos: np.ndarray    # (T, 3)
    quat: np.ndarray   # (T, 4) wxyz
    q: np.ndarray      # (T, n)
    dq: np.ndarray     # (T, n)
    aux: "Track | None" = None   # same frames for another robot (K1), blended together with this one

    @property
    def T(self) -> int:
        return len(self.q)

    def qpos(self, f) -> np.ndarray:
        f = int(np.clip(f, 0, self.T - 1))
        return np.concatenate([self.pos[f], self.quat[f], self.q[f]])

    def take(self, idx) -> "Track":
        idx = np.clip(np.asarray(idx), 0, self.T - 1)
        return Track(self.pos[idx], self.quat[idx], self.q[idx], self.dq[idx],
                     None if self.aux is None else self.aux.take(idx))


def resample_50hz(qpos30: np.ndarray) -> Track:
    n = len(qpos30)
    T = int(np.floor(n / 30.0 * 50))
    f30 = np.arange(T) / 50.0 * 30.0
    f0 = np.floor(f30).astype(int)
    f1 = np.minimum(f0 + 1, n - 1)
    w1 = (f30 - f0)[:, None]
    lerp = lambda x: (1 - w1) * x[f0] + w1 * x[f1]  # noqa: E731
    q = lerp(qpos30[:, 7:])
    dq = np.empty_like(q)
    dq[:-1] = (q[1:] - q[:-1]) * 50.0
    dq[-1] = dq[-2] if T > 1 else 0.0
    return Track(lerp(qpos30[:, :3]), slerp(qpos30[f0, 3:7], qpos30[f1, 3:7], w1[:, 0]), q, dq)


def context_from(track: Track, gen_frame: int) -> np.ndarray:
    """UpdateContextFromMotion: 4 qpos frames at 30 Hz starting at gen_frame (50 Hz index) of the current motion."""
    t = gen_frame / 50.0 + np.arange(4) / 30.0
    f50 = t * 50.0
    f0 = np.minimum(np.floor(f50).astype(int), track.T - 1)
    f1 = np.minimum(f0 + 1, track.T - 1)
    w1 = np.clip(f50 - f0, 0.0, 1.0)[:, None]
    ctx = np.empty((4, 7 + track.q.shape[1]))
    ctx[:, :3] = (1 - w1) * track.pos[f0] + w1 * track.pos[f1]
    ctx[:, 3:7] = slerp(track.quat[f0], track.quat[f1], w1[:, 0])
    ctx[:, 7:] = (1 - w1) * track.q[f0] + w1 * track.q[f1]
    return ctx


def blend_tracks(old: Track, cur: int, new: Track, fgen: int, nb: int = BLEND_FRAMES):
    """CurrentFrameAdvancement: old rebased at cur, new starting at fgen - cur faded in over nb frames.
    Returns the blended track (frame 0 == old frame cur) or None when the new motion arrived too late."""
    L = fgen - cur + new.T
    if L <= 0:
        return None
    bs = max(0, fgen - cur)
    f = np.arange(L)
    fo = np.clip(f + cur, 0, old.T - 1)
    fn = np.clip(f + cur - fgen, 0, new.T - 1)
    w = np.clip((f - bs) / nb, 0.0, 1.0)[:, None]
    mix = lambda a, b: (1 - w) * a[fo] + w * b[fn]  # noqa: E731
    aux = None
    if old.aux is not None and new.aux is not None:
        o, n = old.aux, new.aux
        aux = Track(mix(o.pos, n.pos), slerp(o.quat[fo], n.quat[fn], w[:, 0]), mix(o.q, n.q), mix(o.dq, n.dq))
    return Track(mix(old.pos, new.pos), slerp(old.quat[fo], new.quat[fn], w[:, 0]), mix(old.q, new.q),
                 mix(old.dq, new.dq), aux)


# ------------------------------------------------------------------------------------------------ ONNX model
class PlannerError(RuntimeError):
    pass


class PlannerModel:
    """planner_sonic.onnx (V2, 11 inputs) on onnxruntime CPU."""

    def __init__(self, path: str | Path = PLANNER_ONNX, threads: int = 4, allowed_tokens=ALLOWED_TOKENS):
        import onnxruntime as ort

        path = Path(path)
        if not path.exists():
            path = fetch_planner(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        need = {"context_mujoco_qpos", "target_vel", "mode", "movement_direction", "facing_direction", "random_seed",
                "height", "has_specific_target", "specific_target_positions", "specific_target_headings",
                "allowed_pred_num_tokens"}
        if names != need:
            raise PlannerError(f"unexpected planner inputs {sorted(names)}")
        self.allowed = np.array([allowed_tokens], np.int64)
        self.last_ms = 0.0

    def __call__(self, context, mode, target_vel=-1.0, movement_direction=(0, 0, 0), facing_direction=(1, 0, 0),
                 height=-1.0, seed=1234) -> np.ndarray:
        inp = {
            "context_mujoco_qpos": np.asarray(context, np.float32).reshape(1, 4, 36),
            "target_vel": np.array([target_vel], np.float32),
            "mode": np.array([mode], np.int64),
            "movement_direction": np.asarray(movement_direction, np.float32).reshape(1, 3),
            "facing_direction": np.asarray(facing_direction, np.float32).reshape(1, 3),
            "random_seed": np.array([seed], np.int64),
            "height": np.array([height], np.float32),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": self.allowed,
        }
        t0 = time.perf_counter()
        q, n = self.sess.run(None, inp)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        n = int(np.asarray(n).reshape(-1)[0])
        out = np.asarray(q[0, :n], np.float64)
        if n <= 1 or n > 64 or not np.isfinite(out).all():  # ResampleGeneratedSequence50Hz checks
            raise PlannerError(f"bad planner output (frames={n})")
        return out


# ------------------------------------------------------------------------------------------------ planner stream (G1)
@dataclass
class Movement:
    """MovementState: mode id, world-frame movement / facing directions, speed (-1 = mode default), height."""
    mode: int = 0
    move: tuple = (0.0, 0.0, 0.0)
    face: tuple = (1.0, 0.0, 0.0)
    speed: float = 0.0
    height: float = -1.0


@dataclass
class Tolerances:
    """0 = exact C++ change detection. Defaults for continuous camera / joystick commands."""
    move_rad: float = 0.12
    face_rad: float = 0.10
    speed: float = 0.05
    min_interval: float = 0.2   # seconds between replans unless the mode changes


EXACT = Tolerances(0.0, 0.0, 0.0, 0.0)


def _ang(a, b):
    a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-5 or nb < 1e-5:
        return 0.0 if (na < 1e-5) == (nb < 1e-5) else np.pi
    return float(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))


class PlannerStream:
    """G1-level planner loop (planner thread + 50 Hz frame advancement), synchronous or with a worker thread.

    tick() is one 50 Hz control tick: it applies a finished planner result (blend), advances the frame, and every
    PLANNER_DT runs the replan decision. frame() / window() read the reference at the current frame."""

    def __init__(self, model: PlannerModel | None = None, look_ahead: int = 2, tol: Tolerances = Tolerances(),
                 async_: bool = False, latency_ticks: int = 0, seed: int = 1234, default_height: float = DEFAULT_HEIGHT):
        self.model = model or PlannerModel()
        self.look_ahead, self.tol, self.seed, self.default_height = look_ahead, tol, seed, default_height
        self.async_ = async_
        self.latency_ticks = latency_ticks          # sync mode: apply each result this many ticks later (sim latency)
        self.pool = ThreadPoolExecutor(1) if async_ else None
        self.lock = threading.Lock()
        self.reanchor = None                        # callable(ctx) -> ctx, set by KinematicPlanner(anchor="robot")
        self.convert = None                         # callable(Track) -> Track for another robot, run with the
                                                    # inference (worker thread in async mode) -> Track.aux
        self.track: Track | None = None
        self.n_replans = 0
        self.errors = 0

    # ---------------------------------------------------------------- lifecycle
    def initialize(self, q_mj: np.ndarray):
        """Initialize(): context = 4 x (origin, default height, identity quat, given joints); IDLE inference."""
        ctx = np.zeros((4, 36))
        ctx[:, 2] = self.default_height
        ctx[:, 3] = 1.0
        ctx[:, 7:] = np.asarray(q_mj, float)
        self.track = resample_50hz(self.model(ctx, 0, -1.0, (0, 0, 0), (1, 0, 0), -1.0, self.seed))
        if self.convert is not None:
            self.track.aux = self.convert(self.track)
        self.cur = 0
        self.ticks = 0
        self.last = Movement(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -1.0, -1.0)  # what Initialize() planned with
        self.want = Movement()
        self.counter = 0.0
        self.since_replan = 1e9
        self.pending = []                           # [(ready_tick | future, fgen, track)]
        self.n_replans = 0

    def set_movement(self, mv: Movement):
        with self.lock:
            self.want = mv

    # ---------------------------------------------------------------- planner tick (10 Hz)
    def _need_replan(self, w: Movement) -> bool:
        L, tol = self.last, self.tol
        self.counter += PLANNER_DT
        self.since_replan += PLANNER_DT
        interval = REPLAN_INTERVAL.get(w.mode, 1.0)
        time_to_replan = self.counter >= interval - 1e-9
        if time_to_replan:
            self.counter = 0.0
        mode_changed = w.mode != L.mode
        face_changed = _ang(w.face, L.face) > tol.face_rad if tol.face_rad > 0 else tuple(w.face) != tuple(L.face)
        height_changed = w.height != L.height
        speed_changed = abs(w.speed - L.speed) > tol.speed if tol.speed > 0 else w.speed != L.speed
        move_changed = _ang(w.move, L.move) > tol.move_rad if tol.move_rad > 0 else tuple(w.move) != tuple(L.move)
        static = w.mode in STATIC_MODES
        need = mode_changed or face_changed or height_changed or (
            not static and (speed_changed or move_changed or (time_to_replan and w.speed != 0)))
        if need and not mode_changed and self.since_replan < tol.min_interval:
            return False
        return need

    def _plan(self, ctx, w: Movement):
        tr = resample_50hz(self.model(ctx, w.mode, w.speed if w.mode not in STATIC_MODES else -1.0,
                                      w.move, w.face, w.height, self.seed))
        if self.convert is not None:
            tr.aux = self.convert(tr)
        return tr

    def planner_tick(self):
        with self.lock:
            w = self.want
        if self.pending and self.async_:            # one inference in flight at a time (planner thread is serial)
            return
        if not self._need_replan(w):
            return
        self.last = w
        self.since_replan = 0.0
        fgen = self.cur + self.look_ahead           # UpdatePlanning: gen_frame_ = current_frame + look_ahead
        ctx = context_from(self.track, fgen)
        if self.reanchor is not None:
            ctx = self.reanchor(ctx)
        self.n_replans += 1
        if self.async_:
            self.pending.append((self.pool.submit(self._plan, ctx, w), fgen, None))
        else:
            try:
                tr = self._plan(ctx, w)
            except PlannerError:
                self.errors += 1
                return
            self.pending.append((self.ticks + self.latency_ticks, fgen, tr))

    # ---------------------------------------------------------------- control tick (50 Hz)
    def tick(self):
        """Apply a finished plan (blend), advance one frame, run the 10 Hz planner tick. Returns True on a blend."""
        blended = False
        keep = []
        for item in self.pending:
            ready, fgen, tr = item
            if self.async_:
                if not ready.done():
                    keep.append(item)
                    continue
                try:
                    tr = ready.result()
                except PlannerError:
                    self.errors += 1
                    continue
            elif ready > self.ticks:
                keep.append(item)
                continue
            nt = blend_tracks(self.track, self.cur, tr, fgen)
            if nt is not None:
                self.track, self.cur, blended = nt, 0, True
        self.pending = keep
        self.cur = min(self.cur + 1, self.track.T - 1)
        self.ticks += 1
        if self.ticks % int(round(PLANNER_DT / CONTROL_DT)) == 0:
            self.planner_tick()
        return blended

    def window(self, n: int) -> Track:
        return self.track.take(self.cur + np.arange(n))

    def heading(self) -> float:
        return float(qyaw(self.track.quat[self.cur]))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------------------------------------ G1 -> K1
class G1ToK1:
    """G1 planner qpos -> K1 reference frame (see module doc). Joint order of the output = K1 MuJoCo order
    (== k1live.protocol.POLICY_JOINTS)."""

    STANCE_DY = 0.006   # K1 feet sit 6 mm further out than G1's at the zero pose (0.125 vs 0.119 m)

    def __init__(self, dz: float = -0.02, ik_iters: int = 12):
        import mujoco

        from sonic.robots import get_robot

        self.mj = mujoco
        self.g1, self.k1 = get_robot("g1"), get_robot("k1")
        self.mg, self.mk = self.g1.load_model(), self.k1.load_model()
        self.dg, self.dk = mujoco.MjData(self.mg), mujoco.MjData(self.mk)
        self.gq = np.array([self.mg.jnt_qposadr[self.mg.joint(n).id] for n in self.g1.mujoco_joint_names])
        self.kq = np.array([self.mk.jnt_qposadr[self.mk.joint(n).id] for n in self.k1.mujoco_joint_names])
        self.kv = np.array([self.mk.jnt_dofadr[self.mk.joint(n).id] for n in self.k1.mujoco_joint_names])
        self.gi = {n: i for i, n in enumerate(self.g1.mujoco_joint_names)}
        self.ki = {n: i for i, n in enumerate(self.k1.mujoco_joint_names)}
        rng = np.array([self.mk.jnt_range[self.mk.joint(n).id] for n in self.k1.mujoco_joint_names])
        self.lo, self.hi = rng[:, 0] + 1e-3, rng[:, 1] - 1e-3
        self.dz, self.iters = dz, ik_iters
        self.bg = {n: self.mg.body(n).id for n in ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
                                                   "left_shoulder_yaw_link", "right_shoulder_yaw_link")}
        self.bk = {n: self.mk.body(n).id for n in ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")}
        self.legs = {s: np.array([self.ki[f"{s}_{j}_joint"] for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                                                                         "ankle_pitch", "ankle_roll")]) for s in ("left", "right")}
        self.copy_idx = [(self.ki[n], self.gi[n]) for n in self.k1.mujoco_joint_names
                         if any(t in n for t in ("elbow", "wrist_roll", "waist_yaw"))]
        self.sole_dz = self._ankle_height(self.mk, self.dk, self.kq) - self._ankle_height(self.mg, self.dg, self.gq)
        self.foot_geoms = [g for g in range(self.mk.ngeom) if self.mk.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and self.mk.body(self.mk.geom_bodyid[g]).name.endswith("ankle_roll_link")]
        self.foot_r = self.mk.geom_size[self.foot_geoms, 0].copy()
        self.prev = None
        self.jac_p = np.zeros((3, self.mk.nv))
        self.jac_r = np.zeros((3, self.mk.nv))

    def _ankle_height(self, m, d, qadr):
        """Height of the ankle_roll_link origin above the lowest foot collision point at the zero pose."""
        mj = self.mj
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[2] = 1.0
        mj.mj_kinematics(m, d)
        zs = []
        for g in range(m.ngeom):
            if (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0) or "left_ankle_roll" not in m.body(m.geom_bodyid[g]).name:
                continue
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX:
                R = d.geom_xmat[g].reshape(3, 3)
                zs.append(d.geom_xpos[g][2] - float(np.abs(R[2]) @ m.geom_size[g]))
            elif m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE:
                zs.append(d.geom_xpos[g][2] - m.geom_size[g][0])
        return float(d.xpos[m.body("left_ankle_roll_link").id][2] - min(zs))

    def reset(self):
        self.prev = None

    @staticmethod
    def _R(d, b):
        return d.xmat[b].reshape(3, 3).copy()

    def convert(self, qpos_g1: np.ndarray) -> np.ndarray:
        """One G1 qpos (36) -> K1 qpos (7 + 23), grounded. Warm-started from the previous call."""
        from scipy.spatial.transform import Rotation as Rot

        mj, mg, dg, mk, dk = self.mj, self.mg, self.dg, self.mk, self.dk
        qg = np.asarray(qpos_g1, float)
        dg.qpos[:] = 0
        dg.qpos[:7] = qg[:7]
        dg.qpos[self.gq] = qg[7:]
        mj.mj_kinematics(mg, dg)
        g = {n: qg[7 + i] for n, i in self.gi.items()}
        # pelvis: K1 torso = G1 torso, K1 has waist yaw only
        R_torso = self._R(dg, self.bg["torso_link"])
        Rz = lambda a: Rot.from_euler("z", a).as_matrix()  # noqa: E731
        R_pel = R_torso @ Rz(-g["waist_yaw_joint"])
        out = np.zeros(7 + self.k1.num_dof)
        out[:3] = qg[:3] + [0.0, 0.0, self.dz]
        out[3:7] = Rot.from_matrix(R_pel).as_quat()[[3, 0, 1, 2]]
        q = self.prev.copy() if self.prev is not None else self.k1.default_angles.copy()
        for ki, gi in self.copy_idx:
            q[ki] = qg[7 + gi]
        # shoulders: upper-arm orientation in the torso frame, K1 shoulder = Ry(pitch) Rx(roll) Rz(yaw)
        for s in ("left", "right"):
            R = R_torso.T @ self._R(dg, self.bg[f"{s}_shoulder_yaw_link"])
            p, r, y = Rot.from_matrix(R).as_euler("YXZ")
            q[self.ki[f"{s}_shoulder_pitch_joint"]] = p
            q[self.ki[f"{s}_shoulder_roll_joint"]] = r
            q[self.ki[f"{s}_shoulder_yaw_joint"]] = y
        # feet targets: G1 ankle_roll_link pose, same sole height, K1 stance width
        yaw = np.arctan2(R_pel[1, 0], R_pel[0, 0])
        lat = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        tgt = {}
        for s, sgn in (("left", 1.0), ("right", -1.0)):
            b = self.bg[f"{s}_ankle_roll_link"]
            tgt[s] = (dg.xpos[b] + [0, 0, self.sole_dz] + sgn * self.STANCE_DY * lat, self._R(dg, b))
        q = self._leg_ik(out[:7], np.clip(q, self.lo, self.hi), tgt)
        out[7:] = q
        self.prev = q.copy()
        # ground: lowest K1 foot contact sphere on z = 0
        dk.qpos[:7] = out[:7]
        dk.qpos[self.kq] = q
        mj.mj_kinematics(mk, dk)
        out[2] -= float((dk.geom_xpos[self.foot_geoms, 2] - self.foot_r).min())
        return out

    def _leg_ik(self, root7, q, tgt, lam=1e-2):
        mj, mk, dk = self.mj, self.mk, self.dk
        dk.qpos[:7] = root7
        for _ in range(self.iters):
            dk.qpos[self.kq] = q
            mj.mj_kinematics(mk, dk)
            mj.mj_comPos(mk, dk)
            worst = 0.0
            for s in ("left", "right"):
                b = self.bk[f"{s}_ankle_roll_link"]
                tp, tR = tgt[s]
                R = dk.xmat[b].reshape(3, 3)
                ep = tp - dk.xpos[b]
                Re = tR @ R.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                mj.mj_jacBody(mk, dk, self.jac_p, self.jac_r, b)
                cols = self.kv[self.legs[s]]
                J = np.vstack([self.jac_p[:, cols], self.jac_r[:, cols]])
                e = np.concatenate([ep, er])
                worst = max(worst, float(np.abs(ep).max()))
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
                q[self.legs[s]] = np.clip(q[self.legs[s]] + dq, self.lo[self.legs[s]], self.hi[self.legs[s]])
            if worst < 2e-4:
                break
        self.ik_err = worst
        return q

    def convert_track(self, tr: Track) -> Track:
        """All frames of a G1 track -> K1 track (IK warm-started frame to frame, from the K1 default pose)."""
        self.prev = None
        out = np.stack([self.convert(tr.qpos(f)) for f in range(tr.T)])
        q = out[:, 7:]
        dq = np.empty_like(q)
        dq[:-1] = (q[1:] - q[:-1]) * 50.0
        dq[-1] = dq[-2] if len(q) > 1 else 0.0
        return Track(out[:, :3], out[:, 3:7], q, dq)


# ------------------------------------------------------------------------------------------------ live API
@dataclass
class RefWindow:
    """Reference at the current 50 Hz frame (index 0) and the next frames, robot world coordinates."""
    pos: np.ndarray        # (H, 3)
    quat: np.ndarray       # (H, 4) wxyz
    q: np.ndarray          # (H, n) robot joint order (K1: k1live.protocol.POLICY_JOINTS)
    dq: np.ndarray         # (H, n)
    mode: int = 0
    speed: float = 0.0
    replanned: bool = False
    planner_ms: float = 0.0
    info: dict = field(default_factory=dict)


def window_motion(win: RefWindow, robot, q=None, name: str = "planner"):
    """RefWindow -> sonic.motion.MotionSequence for sonic.deploy.SonicController (Isaac Lab joint order).

    The window's frame 0 is duplicated in front: set ctrl.current_frame = 1 and ctrl.play = True every tick. The
    controller re-takes its heading reference whenever current_frame == 0 (UpdateHeadingState), which would cancel
    every turn of a window that always starts at frame 0; with frame 1 it is taken once, when the caller sets
    ctrl.reinitialize_heading = True at the planner start (the C++ does this on the first planner motion).
    q: optional (H, n) joints replacing win.q (e.g. the person's arms laid over the walk); velocities are then
    re-derived by forward differences like ResampleGeneratedSequence50Hz."""
    from sonic.motion import MotionSequence

    q = win.q if q is None else np.asarray(q, float)
    jp = np.concatenate([q[:1], q])[:, robot.mujoco_to_isaaclab]
    if q is win.q:
        jv = np.concatenate([win.dq[:1], win.dq])[:, robot.mujoco_to_isaaclab]
    else:
        jv = np.zeros_like(jp)
        jv[:-1] = np.diff(jp, axis=0) / CONTROL_DT
        jv[-1] = jv[-2]
    bp = np.concatenate([win.pos[:1], win.pos])[:, None]
    bq = np.concatenate([win.quat[:1], win.quat])[:, None]
    return MotionSequence(name, jp, jv, bp, bq, None, [0], 0)


class KinematicPlanner:
    """Planner + (for robot="k1") the G1 -> K1 retarget, in the robot's world frame.

    reset(qpos):  robot qpos (7 + n) -> planner initialised standing at the robot's xy and heading
    command(...): body-frame (vx, vy, wz) in robot m/s, rad/s; or explicit world movement_direction /
                  facing_direction / speed / mode (planner semantics, speed in robot m/s)
    step(state):  one 50 Hz tick -> RefWindow (current + `horizon` frames); state = robot qpos, only used with
                  anchor="robot"
    vx_gain:      body-frame commands only: vx is multiplied by vx_gain before it goes to the planner (the mode is still
                  picked from the command). The planner's slowWalk output runs at ~0.85x the requested speed forward /
                  backward but ~1.0x sideways (eval_walk ref vs cmd), so ~1.15 makes the reference match the command;
                  1.0 = the C++ behaviour (and how the training set was made). None -> $K1LIVE_PLANNER_VX_GAIN or 1.0.
    """

    SLOW_MAX = 0.8      # G1 m/s: slowWalk covers 0.1 .. 0.8 (LocomotionMode comment), walk above
    IDLE_SPEED = 0.05

    def __init__(self, robot: str = "k1", model_path: str | Path = PLANNER_ONNX, async_: bool = False,
                 threads: int = 4, horizon: int = 50, anchor: str = "plan", tol: Tolerances = Tolerances(),
                 latency_ticks: int = 0, seed: int = 1234, speed_scale: float = 1.0, dz: float = -0.02,
                 max_face_lead: float = 0.6, model: PlannerModel | None = None, allow_run: bool = False,
                 vx_gain: float | None = None):
        assert robot in ("k1", "g1") and anchor in ("plan", "robot")
        self.robot, self.horizon, self.anchor = robot, horizon, anchor
        self.speed_scale, self.max_face_lead, self.allow_run = speed_scale, max_face_lead, allow_run
        # default from the environment so a live backend can set it without code changes (K1LIVE_PLANNER_VX_GAIN=1.15)
        self.vx_gain = float(os.environ.get("K1LIVE_PLANNER_VX_GAIN", "1.0")) if vx_gain is None else float(vx_gain)
        self.stream = PlannerStream(model or PlannerModel(model_path, threads), tol=tol, async_=async_,
                                    latency_ticks=latency_ticks, seed=seed)
        self.retarget = G1ToK1(dz=dz) if robot == "k1" else None
        if self.retarget is not None:
            self.stream.convert = self.retarget.convert_track
        self.state = None
        if anchor == "robot":
            self.stream.reanchor = self._reanchor
        from sonic.robots import get_robot

        self.g1 = get_robot("g1")

    # ---------------------------------------------------------------- frames
    def _to_world(self, pos, quat):
        c, s = np.cos(self.yaw0), np.sin(self.yaw0)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return pos @ R.T + [self.xy0[0], self.xy0[1], 0.0], qmul(yaw_quat(self.yaw0), quat)

    def _to_plan(self, xy, yaw):
        c, s = np.cos(-self.yaw0), np.sin(-self.yaw0)
        d = np.asarray(xy) - self.xy0
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]), wrap(yaw - self.yaw0)

    def _reanchor(self, ctx):
        """anchor="robot": move the context root (xy + heading) onto the measured robot root, keep the rest."""
        st = self.state
        if st is None:
            return ctx
        xy, yaw = self._to_plan(st[:2], float(qyaw(st[3:7])))
        dyaw = wrap(yaw - qyaw(ctx[0, 3:7]))
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s], [s, c]])
        rel = (ctx[:, :2] - ctx[0, :2]) @ R.T
        ctx = ctx.copy()
        ctx[:, :2] = xy + rel
        ctx[:, 3:7] = qmul(yaw_quat(dyaw), ctx[:, 3:7])
        return ctx

    # ---------------------------------------------------------------- lifecycle
    def reset(self, qpos=None):
        """qpos: robot qpos (7 + n). The planner always starts standing in its own frame at the origin, heading 0
        (InitializeContext); this offset maps it onto the robot's xy and heading."""
        if qpos is None:
            self.xy0, self.yaw0 = np.zeros(2), 0.0
        else:
            qpos = np.asarray(qpos, float)
            self.xy0, self.yaw0 = qpos[:2].copy(), float(qyaw(qpos[3:7]))
        q_g1 = self.g1.default_angles.copy()
        if self.robot == "g1" and qpos is not None:
            q_g1 = qpos[7:7 + G1_NQ].copy()
        self.stream.initialize(q_g1)
        self.face_yaw = 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.explicit = None

    def command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, *, movement_direction=None,
                facing_direction=None, speed=None, mode=None, height: float = -1.0):
        """Body-frame velocity command (robot units), or the planner's own inputs (world frame of the robot)."""
        if movement_direction is not None or facing_direction is not None or mode is not None:
            self.explicit = (movement_direction, facing_direction, speed, mode, height)
            return
        self.explicit = None
        self.cmd = (float(vx), float(vy), float(wz))

    def _movement(self) -> Movement:
        st = self.stream
        if self.explicit is not None:
            md, fd, sp, mo, h = self.explicit
            rot = lambda v: (np.array([[np.cos(-self.yaw0), -np.sin(-self.yaw0)], [np.sin(-self.yaw0), np.cos(-self.yaw0)]]) @ np.asarray(v[:2], float))  # noqa: E731
            md = (0, 0, 0) if md is None else tuple(rot(md)) + (0.0,)
            if fd is None:
                fd = (np.cos(self.face_yaw), np.sin(self.face_yaw), 0.0)
            else:
                fd = tuple(rot(fd)) + (0.0,)
                self.face_yaw = float(np.arctan2(fd[1], fd[0]))
            sp = -1.0 if sp is None else float(sp) / self.speed_scale
            mo = self._auto_mode(sp) if mo is None else (MODES[mo] if isinstance(mo, str) else int(mo))
            return Movement(mo, md, fd, sp, h)
        vx, vy, wz = self.cmd
        ref_yaw = st.heading()
        # facing target integrates wz, held within max_face_lead of the reference heading
        self.face_yaw = float(ref_yaw + np.clip(wrap(self.face_yaw + wz * CONTROL_DT - ref_yaw),
                                                -self.max_face_lead, self.max_face_lead))
        speed = float(np.hypot(vx, vy)) / self.speed_scale
        face = (np.cos(self.face_yaw), np.sin(self.face_yaw), 0.0)
        if speed < self.IDLE_SPEED:
            return Movement(0, (0.0, 0.0, 0.0), face, 0.0, -1.0)
        c, s = np.cos(self.face_yaw), np.sin(self.face_yaw)
        move = ((c * vx - s * vy) / (speed * self.speed_scale), (s * vx + c * vy) / (speed * self.speed_scale), 0.0)
        if self.vx_gain != 1.0:   # longitudinal speed correction (planner under-runs forward / backward)
            mode = self._auto_mode(speed)
            vx = vx * self.vx_gain
            speed = float(np.hypot(vx, vy)) / self.speed_scale
            move = ((c * vx - s * vy) / (speed * self.speed_scale), (s * vx + c * vy) / (speed * self.speed_scale), 0.0)
            return Movement(mode, move, face, speed, -1.0)
        return Movement(self._auto_mode(speed), move, face, speed, -1.0)

    def _auto_mode(self, speed):
        if speed < self.IDLE_SPEED:
            return 0
        if speed <= self.SLOW_MAX:
            return 1
        return 3 if (self.allow_run and speed > 2.5) else 2

    def step(self, state=None) -> RefWindow:
        """One 50 Hz tick. state: robot qpos (optional; used by anchor="robot")."""
        if state is not None:
            self.state = np.asarray(state, float)
        st = self.stream
        st.set_movement(self._movement())
        blended = st.tick()
        w = st.track.take(st.cur + np.arange(self.horizon))
        if self.retarget is not None:   # K1 frames: converted with each plan (worker), blended like the G1 ones
            w = w.aux
        pos, quat = self._to_world(w.pos, w.quat)
        q = w.q if self.robot == "k1" else w.q[:, :G1_NQ]
        return RefWindow(pos, quat, q, w.dq, st.last.mode, st.last.speed * self.speed_scale, blended,
                         st.model.last_ms, {"replans": st.n_replans, "frame": st.cur, "track_len": st.track.T,
                                            "face_yaw": self.face_yaw + self.yaw0})

    def g1_window(self, n=None) -> Track:
        """The underlying G1 planner frames at the current frame (planner frame, for debugging / ghosts)."""
        return self.stream.window(n or self.horizon)

    def close(self):
        self.stream.close()
