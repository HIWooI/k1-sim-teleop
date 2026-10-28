"""Split backend (track A, no training): arms = ROBOTIS mimic student following the person's arm IK, legs + waist =
ROBOTIS loco policy `loco_armdriven_r2` driven by the walking command (`velocity_commands` = vx, vy, wz).

Wraps k1.sim.K1Split (port of the original sim2sim --split, numerically unchanged). Two live-specific choices:

* torso reference = the robot's own current torso yaw (roll/pitch upright). The mimic student only commands the
  arms here, but it observes the reference-vs-robot torso orientation; with a fixed reference a walking, turning
  robot would show it a growing yaw error it never saw in training.
* arm trim (outer loop): the student holds a constant reference ~6 deg / 6 cm off (even the relaxed default pose:
  left shoulder yaw +14 deg, wrist roll +10 deg). A slow bounded integrator adds (reference - arm) to the reference it
  is given while that reference moves slower than trim_vmax (leaking otherwise): static error 7.3 -> ~1.5 deg, hand
  6.8 -> ~3 cm, a moving goal 7.4 -> ~5 deg, no falls (docs/results.md). trim_gain=0 turns it off.
* waist yaw from the person is off by default (the loco policy owns the waist; overriding its target is outside
  its training). `waist=True` overrides the waist PD target with the person's yaw, clamped to +-waist_max rad.
  Bow (hip pitch) cannot be commanded in split mode and is ignored.
* arm path options (docs/architecture.md): arm_mode="student" (default) feeds the reference to the mimic student;
  arm_lead_s > 0 feeds it the reference extrapolated that far ahead (q + lead * dq, the student lags its reference).
  arm_mode="pd" bypasses the student for the arms: PD straight to the reference with model-based gravity
  compensation (MuJoCo bias forces / kp folded into the target) and velocity feed-forward (kd * dq / kp), arm gains
  pd_gains="loco" (loco_armdriven_r2's own arm gains, 40 / 2) or "student" (22.3 / 1.42), no arm trim unless
  pd_trim. The legs stay with loco_armdriven_r2 (by its name meant for arms driven by something else; measured:
  walking with PD arms has the same falls / tilt / speed as with the student). The student still runs (keeps its
  history warm).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from k1.sim import ASSETS, K1Split, build_model, quat_to_mat
from k1live.backends.base import Backend, arm_indices
from k1live.protocol import ARM_JOINTS, POLICY_JOINTS

WRIST_ROLL = [ARM_JOINTS.index(n) for n in ("left_wrist_roll_joint", "right_wrist_roll_joint")]
ARM_MODES = ("student", "pd")


def build_render_model():
    """Same MjModel as the physics (k1.sim scene), for a render process."""
    return build_model(POLICY_JOINTS, render_scene=os.environ.get("K1LIVE_RENDER_SCENE"))


def _yaw_quat_xyzw(R):
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


class SplitBackend(Backend):
    name = "split"
    supports = {"arms": True, "waist_yaw": False, "bow": False, "walk": True}
    render_model_fn = "k1live.backends.split:build_render_model"

    def __init__(self, waist: bool = False, waist_max: float = 0.35, trim_gain: float = 2.0, trim_max: float = 0.25,
                 trim_vmax: float = 1.0, trim_leak_s: float = 2.0, mimic_asset=ASSETS / "student_asset",
                 loco_asset=ASSETS / "loco_armdriven_r2", arm_mode: str = "student", arm_lead_s: float = 0.0,
                 pd_gains: str = "loco", gravity_comp: bool = True, vel_ff: bool = True, pd_trim: bool = False):
        if arm_mode not in ARM_MODES:
            raise ValueError(f"arm_mode {arm_mode!r} not in {ARM_MODES}")
        self.sim = K1Split(mimic_asset=mimic_asset, loco_asset=loco_asset, mode="split")
        assert list(self.sim.names) == POLICY_JOINTS, self.sim.names
        self.model, self.data = self.sim.model, self.sim.data
        self.names = list(self.sim.names)
        self.qadr = self.sim.qadr
        self.arm_idx = arm_indices(self.names)
        self.waist_idx = self.names.index("waist_yaw_joint")
        # rest arms = the loco policy's default arm pose (what k1/run_video and k1/gestures use as "relaxed")
        self.arm_default = self.sim.loco.default[self.arm_idx].copy()
        self.arm_lo, self.arm_hi = self.sim.lo[self.arm_idx].copy(), self.sim.hi[self.arm_idx].copy()
        self.pelvis_body = self.sim.pelvis_id
        self.waist = waist
        self.waist_max = waist_max
        if waist:
            self.supports = dict(self.supports, waist_yaw=True)
        self._prev_arm = None
        if arm_mode == "pd" and not pd_trim:
            trim_gain = 0.0   # the trim corrects the student's offset; PD + gravity compensation tracks better without
        self.trim_gain, self.trim_max, self.trim_vmax, self.trim_leak_s = trim_gain, trim_max, trim_vmax, trim_leak_s
        self.trim = np.zeros(10)
        self.arm_mode, self.arm_lead_s = arm_mode, float(arm_lead_s)
        self.gravity_comp, self.vel_ff = gravity_comp, vel_ff
        if arm_mode == "pd":
            gains = {"loco": self.sim.loco, "student": self.sim.pol}[pd_gains]
            self.sim.kp[self.arm_idx] = gains.kp[self.arm_idx]
            self.sim.kd[self.arm_idx] = gains.kd[self.arm_idx]
        self.arm_vadr = self.sim.vadr[self.arm_idx]
        self.reset()

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        s = self.sim
        s.reset(ref_torso_quat_xyzw=(0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)), yaw=yaw)
        s.data.qpos[0:2] = [x, y]
        import mujoco

        mujoco.mj_forward(s.model, s.data)
        s.init_align = np.eye(3)  # the torso reference is the robot's own yaw from here on (see module doc)
        self._prev_arm = None
        self.trim = np.zeros(10)

    def step(self, arm_q: Sequence[float], waist_yaw: Optional[float] = None, bow: Optional[float] = None,
             loco_cmd: Sequence[float] = (0.0, 0.0, 0.0), arm_dq: Optional[Sequence[float]] = None) -> dict:
        s = self.sim
        arm_q = np.clip(np.asarray(arm_q, np.float64), self.arm_lo, self.arm_hi)
        if arm_dq is None:
            arm_dq = np.zeros(10) if self._prev_arm is None else (arm_q - self._prev_arm) / self.dt
        self._prev_arm = arm_q.copy()
        if self.trim_gain > 0:
            quasi = np.abs(np.asarray(arm_dq)) < self.trim_vmax
            err = arm_q - self.arm_q_now()
            self.trim = np.where(quasi, self.trim + self.trim_gain * err * self.dt,
                                 self.trim * (1.0 - self.dt / self.trim_leak_s))
            self.trim = np.clip(self.trim, -self.trim_max, self.trim_max)
        rq = s.loco.default.copy()
        rdq = np.zeros(len(rq))
        lead = self.arm_lead_s * np.asarray(arm_dq) if self.arm_mode == "student" else 0.0
        rq[self.arm_idx] = np.clip(arm_q + lead + self.trim, self.arm_lo, self.arm_hi)
        rdq[self.arm_idx] = arm_dq
        override = None
        if self.arm_mode == "pd":
            override = dict(zip(self.arm_idx.tolist(), self._pd_target(rq[self.arm_idx], np.asarray(arm_dq)).tolist()))
        if self.waist and waist_yaw is not None:
            override = dict(override or {})
            override[self.waist_idx] = float(np.clip(waist_yaw, -self.waist_max, self.waist_max))
        rtq = _yaw_quat_xyzw(quat_to_mat(s.data.xquat[s.torso_id]))
        target = s.step(rq, rdq, rtq, cmd=np.asarray(loco_cmd, np.float64), target_override=override)
        return {"policy_ms": s.last_policy_ms, "target": target}

    def _pd_target(self, q_ref, dq_ref):
        """PD target that makes tau = kp (q_ref - q) + kd (dq_ref - dq) + gravity/Coriolis of the arm joints (the
        backend's PD is kp (target - q) - kd dq): the feed-forward terms folded into the target."""
        kp, kd = self.sim.kp[self.arm_idx], self.sim.kd[self.arm_idx]
        t = np.asarray(q_ref, np.float64).copy()
        if self.vel_ff:
            t += kd * dq_ref / kp
        if self.gravity_comp:
            t += self.data.qfrc_bias[self.arm_vadr] / kp
        return t
