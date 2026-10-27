"""Sonic backend (track B, GEAR-SONIC style): the K1 SONIC whole-body policy tracks a reference built every tick from

* lower body + root: the GEAR-SONIC kinematic planner (k1live/planner.py, planner_sonic.onnx -> K1) driven by the
  walking command, exactly as the SONIC VR teleop drives its lower body (planner window = current + future frames),
* arms: the person's arm IK (arm_q, already rate-limited by the server), laid over the planner's arm joints,
* waist yaw: the person's torso twist; bow: the person's forward lean as pelvis pitch with vertical thighs
  (like k1/gestures.bow), only while the planner stands (faded out when walking).

Physics = the K1 port used to evaluate the SONIC checkpoints (sonic.sim.G1Sim with sonic.robots.get_robot("k1"):
staged K1 MJCF, training gains / armature / torque limits, 200 Hz PD, 50 Hz policy), so the numbers match
sonic_k1_train/mac/eval_k1_mujoco.py and eval_walk.py. MuJoCo only - nothing here talks to a robot.

The 80 ms policy (`k1s_la4`: 4 future frames x 20 ms) needs the person's arms 60 ms ahead of "now". `arm_future`:
  delay        frame 0 = the arm target of 3 ticks ago, frames 1..3 = the newer ones: a true future, robot 60 ms later
  extrapolate  frame 0 = the newest target, frames 1..3 = constant-velocity extrapolation (clamped)
  hold         every frame = the newest target (zero future velocity)
docs/results.md compares them: same arm error within 0.1 deg on real person streams, extrapolate has the
lowest lag (best-lag 80-140 ms vs delay 140-180 ms) -> default extrapolate.

    be = SonicBackend()                      # models/k1/k1s_la4/deploy_002000, arm_future="extrapolate"
    be.reset()
    be.step(arm_q, waist_yaw=0.1, bow=None, loco_cmd=(0.3, 0.0, 0.0))
"""
from __future__ import annotations

import dataclasses
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from k1live.backends.base import Backend, arm_indices
from k1live.protocol import ARM_JOINTS, POLICY_JOINTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT = ROOT / "models" / "k1" / "k1s_la4" / "deploy_002000"
SUBSTEPS = 4
WAIST = POLICY_JOINTS.index("waist_yaw_joint")
HIP_PITCH = [POLICY_JOINTS.index(f"{s}_hip_pitch_joint") for s in ("left", "right")]
ARM_VMAX = 6.0          # rad/s, clamp of the extrapolated arm velocity


def build_render_model():
    """Same MjModel as the physics (staged K1 MJCF + floor, training armature), for a render process."""
    from sonic.robots import get_robot

    return get_robot("k1").load_model()


def _quat_pitch(a):
    """wxyz rotation about the body y axis (+ = nose down / bow forward)."""
    return np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])


class SonicBackend(Backend):
    name = "sonic"
    supports = {"arms": True, "waist_yaw": True, "bow": True, "walk": True}
    render_model_fn = "k1live.backends.sonic:build_render_model"

    def __init__(self, variant_dir=DEFAULT_VARIANT, planner: Optional[bool] = None, arm_future: str = "extrapolate",
                 n_future: int = 4, waist_max: float = 0.6, bow_max: float = 0.35, anchor: str = "plan",
                 threads: int = 2, planner_async: bool = False, band_s: float = 1.0, settle_s: float = 0.5,
                 walk: Optional[bool] = None):
        """walk: follow walking commands. Default: only for a walking-trained checkpoint (variant path contains
        "walk" in its run / deploy folder name, e.g. models/k1/k1s_la4_walk/deploy_002000); the gesture checkpoint k1s_la4 marches in place on a walking reference, so it ignores loco_cmd.
        planner: lower body from the kinematic planner. Default = walk. Without it the lower body / root reference is
        the K1 standing pose (what the gesture checkpoint was trained on: with the planner's idle stance its arm error
        is 2-3 deg higher, docs/results.md)."""
        assert arm_future in ("delay", "extrapolate", "hold")
        sys.path.insert(0, str(ROOT / "sonic_k1_train" / "mac"))
        from eval_k1_mujoco import K1Controller  # K1 export observation layout (encoder_index, n-frame aliases)

        from sonic.motion import standing_motion
        from sonic.robots import get_robot
        from sonic.sim import G1Sim

        self.robot = get_robot("k1")
        assert list(self.robot.mujoco_joint_names) == POLICY_JOINTS
        self.sim = G1Sim(robot=self.robot, elastic_band=True)
        self.model, self.data = self.sim.model, self.sim.data
        self.names = list(POLICY_JOINTS)
        self.qadr = self.sim.qadr
        self.arm_idx = arm_indices(self.names)
        self.arm_default = self.robot.default_angles[self.arm_idx].copy()
        jr = np.array([self.model.jnt_range[self.model.joint(n).id] for n in self.names])
        lim = np.array([self.model.jnt_limited[self.model.joint(n).id] for n in self.names]).astype(bool)
        self.q_lo = np.where(lim, jr[:, 0], -np.inf)
        self.q_hi = np.where(lim, jr[:, 1], np.inf)
        self.arm_lo, self.arm_hi = self.q_lo[self.arm_idx].copy(), self.q_hi[self.arm_idx].copy()
        self.pelvis_body = self.sim.pelvis
        self.variant_dir = Path(variant_dir)
        self.ctrl = K1Controller.load(self.variant_dir, self.robot)
        self._standing = standing_motion(50, robot=self.robot)
        want_walk = walk if walk is not None else any("walk" in part for part in self.variant_dir.parts[-2:])
        planner = want_walk if planner is None else planner
        self.kp = None
        if planner:
            from k1live.planner import KinematicPlanner

            self.kp = KinematicPlanner(robot="k1", anchor=anchor, horizon=max(8, n_future + 4), threads=threads,
                                       async_=planner_async)
        self.walk = bool(planner and want_walk)
        self.supports = dict(self.supports, walk=self.walk)
        self.arm_future, self.n_future = arm_future, n_future
        self.waist_max, self.bow_max = waist_max, bow_max
        self.band_s, self.settle_s = band_s, settle_s
        self.last_window = None
        self.reset()

    # ------------------------------------------------------------------------------------------- lifecycle
    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Respawn standing at (x, y, yaw): default pose on the floor, policy history cleared, held on the elastic
        band for band_s, released, settle_s on the standing reference, then the planner starts at the robot's pose.
        The warm-up is simulated here (about 1.5 s of sim time, ~0.1 s of compute)."""
        from sonic.runner import place_on_ground

        sim, ctrl = self.sim, self.ctrl
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        place_on_ground(sim, yaw=yaw)
        self.data.qpos[0:2] = [x, y]
        mujoco.mj_forward(self.model, self.data)
        if sim.band is not None:
            sim.band.point[0:2] = [x, y]
        sim.set_band(True)
        ctrl.reset()
        ctrl.set_motion(self._standing, play=False)
        ctrl.state = "CONTROL"           # already in the default pose on the floor: skip the 3 s INIT ramp
        self._arm_buf = deque(maxlen=self.n_future + 4)
        self._waist_buf = deque(maxlen=self.n_future + 4)
        self._bow_w = 0.0
        self._bow = 0.0
        for k in range(int(round((self.band_s + self.settle_s) / self.dt))):
            if k == int(round(self.band_s / self.dt)):
                sim.set_band(False)
            self._physics_tick()
        if self.kp is not None:
            self.kp.reset(np.concatenate([self.data.qpos[:7], self.data.qpos[self.qadr]]))
            ctrl.reinitialize_heading = True

    def close(self) -> None:
        if self.kp is not None:
            self.kp.close()

    def _physics_tick(self):
        t0 = time.perf_counter()
        q_t, kp, kd = self.ctrl.step(self.sim.low_state())
        ms = (time.perf_counter() - t0) * 1e3
        self.sim.set_command(q_t, kp, kd)
        self.sim.step(SUBSTEPS)
        return q_t, ms

    # ------------------------------------------------------------------------------------------- reference
    def _future(self, buf, n):
        """(n, k) rows for frames 0..n-1 from the newest-last buffer, per self.arm_future."""
        b = np.asarray(buf)
        new = b[-1]
        if self.arm_future == "hold" or len(b) == 1:
            return np.repeat(new[None], n, axis=0)
        if self.arm_future == "extrapolate":
            v = np.clip((b[-1] - b[-2]) / self.dt, -ARM_VMAX, ARM_VMAX)
            return new[None] + np.arange(n)[:, None] * self.dt * v[None]
        idx = np.arange(n) - (n - 1)            # delay: frame n-1 = newest, frame 0 = n-1 ticks old
        idx = np.maximum(len(b) - 1 + idx, 0)
        return b[idx]

    def _window(self, loco_cmd):
        if self.kp is not None:
            self.kp.command(*([float(v) for v in loco_cmd] if self.walk else (0.0, 0.0, 0.0)))
            return self.kp.step(np.concatenate([self.data.qpos[:7], self.data.qpos[self.qadr]]))
        from k1live.planner import RefWindow

        H = self.n_future + 4
        R0 = self._standing
        return RefWindow(np.tile(R0.body_pos[0, 0], (H, 1)), np.tile(R0.body_quat[0, 0], (H, 1)),
                         np.tile(self.robot.default_angles, (H, 1)), np.zeros((H, self.robot.num_dof)))

    def step(self, arm_q: Sequence[float], waist_yaw: Optional[float] = None, bow: Optional[float] = None,
             loco_cmd: Sequence[float] = (0.0, 0.0, 0.0), arm_dq: Optional[Sequence[float]] = None) -> dict:
        from k1live.planner import window_motion

        arm_q = np.clip(np.asarray(arm_q, np.float64), self.arm_lo, self.arm_hi)
        self._arm_buf.append(arm_q)
        self._waist_buf.append(float(np.clip(waist_yaw, -self.waist_max, self.waist_max))
                               if waist_yaw is not None else np.nan)
        w = self._window(loco_cmd)
        H = len(w.q)
        n = min(self.n_future, H)
        q = w.q.copy()
        q[:n, self.arm_idx] = self._future(self._arm_buf, n)
        q[n:, self.arm_idx] = q[n - 1, self.arm_idx]
        wb = np.array(self._waist_buf, float)
        if np.isfinite(wb[-1]):
            wf = self._future(np.where(np.isfinite(wb), wb, wb[-1])[:, None], n)[:, 0]
            q[:n, WAIST] = wf
            q[n:, WAIST] = wf[-1]
        # bow: pelvis pitch with vertical thighs, faded in only while the planner reference stands still
        standing = (w.speed if self.kp is not None else 0.0) < 0.05
        self._bow_w = float(np.clip(self._bow_w + (1.0 if standing else -1.0) * self.dt / 0.5, 0.0, 1.0))
        target_bow = float(np.clip(bow, 0.0, self.bow_max)) if bow is not None else 0.0
        self._bow += float(np.clip(target_bow - self._bow, -0.8 * self.dt, 0.8 * self.dt))   # 0.8 rad/s
        b = self._bow * self._bow_w
        pos, quat = w.pos, w.quat
        if b > 1e-3:
            q[:, HIP_PITCH] -= b
            quat = _qmul_right(quat, _quat_pitch(b))
        q = np.clip(q, self.q_lo, self.q_hi)
        win = dataclasses.replace(w, pos=pos, quat=quat, q=q)
        self.last_window = win
        ctrl = self.ctrl
        ctrl.current_motion = window_motion(win, self.robot, q)   # frame 0 duplicated -> current_frame = 1
        ctrl.current_frame, ctrl.play = 1, True
        q_t, ms = self._physics_tick()
        return {"policy_ms": ms, "planner_ms": float(getattr(w, "planner_ms", 0.0)), "target": q_t,
                "ref_q": q[0].copy()}


def _qmul_right(q, r):
    """q (H, 4) * r (4,) for wxyz quaternions (rotate in the body frame)."""
    q = np.asarray(q, float)
    w1, x1, y1, z1 = np.moveaxis(q, -1, 0)
    w2, x2, y2, z2 = r
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)
