"""Hybrid backend: upper body first while standing, a clean SONIC gait while walking (MuJoCo simulation only).

Why: the split gait looks bad (bent knees, wide stance, shuffling, torso twisting) while the split arms follow the
person well, so the lower body hands over to SONIC only while walking and the upper body has priority otherwise.

    STAND  (no walking command)   = the split backend, numerically: arms = ROBOTIS mimic student following the person's
                                    arm IK (+ the same arm trim), legs + waist = ROBOTIS loco_armdriven_r2 at velocity 0
                                    (the server's heading hold only)
    WALK   (walking command held  = K1 SONIC walk3 (whole body) tracking the GEAR-SONIC kinematic planner's gait; arms =
            for >= enter_s)         the planner's arm swing (`walk_arm_human` > 0 lays the person's arms over it)
    SETTLE (command back to 0)    = SONIC on the planner's idle (feet brought together), arms still the walking ones
                                    (settle_arms="walk"); after feet + pelvis are still for settle_s the legs go back to
                                    the loco, then the arms to the person

Every tick runs all three policies (shadow execution) on the same robot state, so each one has a live observation
history when it takes over; a policy whose joints are (partly) driven by another one gets as its `last_action` the
target actually applied, encoded in its own action space (ROBOTIS: (q - offset) / scale, SONIC: (q - default) / scale in
Isaac Lab order). Hand-over = convex blend of the two controllers' PD torques per joint group (legs + waist, arms),
smoothstep over blend_s. The planner runs asynchronously (worker thread, as the C++ planner thread).

legs="sonic" is alternative B: the legs (and waist) are always SONIC (planner idle while standing), the arms always the
student (planner arm swing while walking); STAND then also follows the person's waist yaw and bow (SONIC pelvis pitch).

Physics = the split model (k1.sim: vendor k1.xml, friction 0.8, URDF limits, PD in code with the velocity-limit torque
cut), so STAND is bit-identical to the split backend; SONIC's gains (kp = armature * (2 pi 10)^2) equal the student's.

    be = HybridBackend()                       # models/k1/k1s_walk3/deploy/k1s_walk3_005000
    be.step(arm_q, loco_cmd=(0.3, 0, 0), arm_dq=dq, walk_cmd=(0.3, 0, 0))   # walk_cmd: the command without heading hold
    be.phase                                   # "stand" | "walk" | "settle"
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from k1.sim import ASSETS, K1Split, quat_to_mat
from k1live.backends.base import Backend, arm_indices
from k1live.protocol import POLICY_JOINTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT = Path(os.environ.get("K1LIVE_HYBRID_VARIANT",
                                      ROOT / "models" / "k1" / "k1s_walk3" / "deploy" / "k1s_walk3_005000"))
WAIST = POLICY_JOINTS.index("waist_yaw_joint")
HIP_PITCH = [POLICY_JOINTS.index(f"{s}_hip_pitch_joint") for s in ("left", "right")]
ARM_VMAX = 6.0                 # rad/s, clamp of the person-arm extrapolation into SONIC's future frames
INTENT_V, INTENT_W = 0.03, 0.05   # |v| m/s, |wz| rad/s: a walking command
SETTLE_V = 0.06                # m/s: feet / pelvis slower than this count as still


def _yaw_quat_xyzw(R):
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


def _smooth(u):
    return u * u * (3.0 - 2.0 * u)


def _quat_pitch(a):
    return np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])


class HybridBackend(Backend):
    name = "hybrid"
    supports = {"arms": True, "waist_yaw": False, "bow": False, "walk": True, "intent": True}
    render_model_fn = "k1live.backends.split:build_render_model"   # same MjModel: no render restart on a switch
    limiter_preset = "sonic"     # walking goes to the planner: no loco dead bands to jump over (k1live.locomotion)

    def __init__(self, variant_dir=DEFAULT_VARIANT, legs: str = "switch", walk_arms: str = "sonic",
                 walk_arm_human: float = 0.0, blend_s: float = 0.4, arm_blend_s: float = 0.8, enter_s: float = 0.2,
                 min_walk_s: float = 0.5, settle_s: float = 0.5, settle_max_s: float = 3.0, cmd_at_w: float = 0.0,
                 min_step_v: float = 0.2, turn_step_v: float = 0.2,
                 lock: Optional[str] = None, planner_async: bool = True, planner_threads: int = 2,
                 vx_gain: Optional[float] = None, waist: Optional[bool] = None, waist_max: float = 0.6,
                 bow_max: float = 0.35, trim_gain: float = 2.0, trim_max: float = 0.25, trim_vmax: float = 1.0,
                 trim_leak_s: float = 2.0, n_future: int = 4, warmup_s: float = 0.5, arm_mode: str = "student",
                 pd_gains: str = "loco", gravity_comp: bool = True, vel_ff: bool = True, pd_trim: bool = False,
                 settle_arms: str = "walk",
                 mimic_asset=ASSETS / "student_asset", loco_asset=ASSETS / "loco_armdriven_r2"):
        """legs: "switch" (STAND = loco legs, WALK = SONIC) or "sonic" (alternative B: SONIC legs always).
        walk_arms: arm tracker while walking, "sonic" (whole-body SONIC) or "student". walk_arm_human: weight of the
        person's arms in the walking arm reference (0 = planner arm swing). min_step_v: a translation command is
        lifted to at least this speed (m/s) for the planner, whose gaits below ~0.2 m/s shuffle (foot lift < 5 mm);
        turn_step_v: turning on the spot becomes a stepping arc at this forward speed (the planner's in-place turn is
        an idle-stance pivot with the feet sliding); 0 = planner as is. cmd_at_w: the planner gets the walking
        command once the leg hand-over weight reaches this (0 = at once). lock: "walk" = SONIC + planner only (no
        hand-over; the reference test of SONIC on this physics), "stand" = split only.
        arm_mode: arms while the split side drives them (STAND): "student" (mimic student, default) or "pd" =
        the split PD arm path (docs/architecture.md): PD straight to the reference with gravity compensation and velocity
        feed-forward, gains pd_gains ("loco" 40/2 or "student" 22.3/1.42), no arm trim unless pd_trim; the student
        keeps running in the shadow. Same numbers as SplitBackend(arm_mode="pd") while standing.
        settle_arms: arms while stopping, "walk" (default) = SETTLE keeps the walking arms (planner arm reference on
        the walking arm tracker) and they go back to the person once the legs are on the loco again; "person" = back
        to the person's arms as soon as SETTLE starts (the earlier behaviour: the person's resting pose has the hands just
        inside the hips, so the arms clamped onto the hip links while SONIC was stopping - docs/results.md)."""
        assert legs in ("switch", "sonic") and walk_arms in ("sonic", "student") and lock in (None, "walk", "stand")
        assert arm_mode in ("student", "pd") and pd_gains in ("loco", "student") and settle_arms in ("walk", "person")
        sys.path.insert(0, str(ROOT / "sonic_k1_train" / "mac"))
        from eval_k1_mujoco import K1Controller

        from k1live.planner import KinematicPlanner
        from sonic.motion import standing_motion
        from sonic.robots import get_robot

        # --- physics + ROBOTIS policies (the split backend's K1Split, used through its parts) ---------------------
        self.sim = s = K1Split(mimic_asset=mimic_asset, loco_asset=loco_asset, mode="split")
        assert list(s.names) == POLICY_JOINTS, s.names
        self.model, self.data = s.model, s.data
        self.names = list(s.names)
        self.qadr = s.qadr
        self.arm_idx = arm_indices(self.names)
        self.arm_mask = np.zeros(len(self.names), bool)
        self.arm_mask[self.arm_idx] = True
        self.lower = s.lower.copy()                       # legs + waist
        self.arm_default = s.loco.default[self.arm_idx].copy()
        self.arm_lo, self.arm_hi = s.lo[self.arm_idx].copy(), s.hi[self.arm_idx].copy()
        self.q_lo = np.where(np.isfinite(s.lo), s.lo, -np.inf)
        self.q_hi = np.where(np.isfinite(s.hi), s.hi, np.inf)
        self.pelvis_body = s.pelvis_id
        self.feet = [s.model.body(f"{x}_ankle_roll_link").id for x in ("left", "right")]
        # PD gains of the split side (loco legs + waist, student arms; arm_mode="pd": the PD arm gains)
        self.arm_mode, self.gravity_comp, self.vel_ff = arm_mode, gravity_comp, vel_ff
        self.kp_split, self.kd_split = s.kp.copy(), s.kd.copy()
        if arm_mode == "pd":
            g = {"loco": s.loco, "student": s.pol}[pd_gains]
            self.kp_split[self.arm_idx] = g.kp[self.arm_idx]
            self.kd_split[self.arm_idx] = g.kd[self.arm_idx]
            if not pd_trim:
                trim_gain = 0.0
        self.arm_vadr = s.vadr[self.arm_idx]
        # --- K1 SONIC + planner ---------------------------------------------------------------------------------
        self.robot = get_robot("k1")
        assert list(self.robot.mujoco_joint_names) == POLICY_JOINTS
        self.variant_dir = Path(variant_dir)
        self.ctrl = K1Controller.load(self.variant_dir, self.robot)
        self._standing = standing_motion(50, robot=self.robot)
        self.kp = KinematicPlanner(robot="k1", horizon=max(8, n_future + 4), threads=planner_threads,
                                   async_=planner_async, vx_gain=vx_gain)
        r = self.robot
        self.s_default, self.s_scale = r.default_angles.copy(), r.action_scale.copy()
        self.i2m = r.isaaclab_to_mujoco
        # --- options -----------------------------------------------------------------------------------------
        self.legs, self.walk_arms, self.lock = legs, walk_arms, lock
        self.walk_arm_human = float(np.clip(walk_arm_human, 0.0, 1.0))
        self.blend_s, self.arm_blend_s, self.enter_s = blend_s, arm_blend_s, enter_s
        self.min_walk_s, self.settle_s, self.settle_max_s, self.cmd_at_w = min_walk_s, settle_s, settle_max_s, cmd_at_w
        self.n_future, self.warmup_s = n_future, warmup_s
        self.min_step_v, self.turn_step_v = min_step_v, turn_step_v
        self.settle_arms = settle_arms
        sonic_legs = legs == "sonic" or lock == "walk"
        self.waist = sonic_legs if waist is None else bool(waist and sonic_legs)
        self.waist_max, self.bow_max = waist_max, bow_max
        self.supports = dict(self.supports, waist_yaw=self.waist, bow=sonic_legs)
        self.trim_gain, self.trim_max, self.trim_vmax, self.trim_leak_s = trim_gain, trim_max, trim_vmax, trim_leak_s
        self.last_window = None
        self.stats = {"walk_entries": 0, "stand_entries": 0, "settle_timeouts": 0, "variant": self.variant_dir.name}
        self.reset()

    @staticmethod
    def available(variant_dir=None) -> bool:
        """Files the backend needs: the SONIC deploy dir, the staged K1 MJCF (planner retarget), the planner ONNX."""
        try:
            from k1live.planner import PLANNER_ONNX
            from sonic.robots import k1_scene_xml

            k1_scene_xml()
            v = Path(variant_dir or DEFAULT_VARIANT)
            return (v / "model_encoder.onnx").exists() and (v / "model_decoder.onnx").exists() and PLANNER_ONNX.exists()
        except Exception:
            return False

    # ------------------------------------------------------------------------------------------- lifecycle
    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """legs="switch": the split reset (loco default pose, pelvis 0.79 m, nothing simulated). legs="sonic" / lock
        "walk": SONIC default pose on the floor and warmup_s simulated on the planner's idle (~25 ms of compute)."""
        s = self.sim
        s.reset(ref_torso_quat_xyzw=(0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)), yaw=yaw)
        d = s.data
        sonic_legs = self.legs == "sonic" or self.lock == "walk"
        if sonic_legs:   # SONIC stance (deeper knee bend than the loco's), arms at the split rest pose
            q0 = self.s_default.copy()
            q0[self.arm_idx] = self.arm_default
            d.qpos[s.qadr] = q0
            d.qpos[2] = 1.0
            mujoco.mj_kinematics(s.model, d)
            zmin = min(float(d.geom_xpos[g][2] - s.model.geom_rbound[g]) for g in range(s.model.ngeom)
                       if s.model.geom_bodyid[g] in self.feet and s.model.geom_contype[g] | s.model.geom_conaffinity[g])
            d.qpos[2] = 1.0 - zmin + 0.002
        d.qpos[0:2] = [x, y]
        mujoco.mj_forward(s.model, d)
        s.init_align = np.eye(3)       # torso reference = the robot's own yaw (as the split backend)
        ctrl = self.ctrl
        ctrl.reset()
        ctrl.set_motion(self._standing, play=False)
        ctrl.state = "CONTROL"
        self.kp.reset(np.concatenate([d.qpos[:7], d.qpos[s.qadr]]))
        ctrl.reinitialize_heading = True
        self._prev_arm = None
        self.trim = np.zeros(10)
        self.phase = "walk" if self.lock == "walk" else "stand"
        self.t_phase = 0.0
        self.t_intent = 0.0
        self.t_still = 0.0
        # hand-over state: u = linear ramp 0..1, weight = smoothstep(u)
        full = 1.0 if sonic_legs else 0.0
        self.uL = full                                      # legs + waist: 0 loco, 1 SONIC
        self.uA = 1.0 if (self.lock == "walk") else 0.0     # arms: 0 student, 1 SONIC
        self.uP = 1.0 if (self.lock == "walk" and self.walk_arm_human < 1) else 0.0   # arm reference: 0 person, 1 planner
        self._uP_prev = self.uP
        self._bow = self._bow_w = 0.0
        self._pc_wz = 0.0
        self.last_info = {}
        if sonic_legs and self.warmup_s > 0:
            for _ in range(int(round(self.warmup_s / self.dt))):
                self.step(self.arm_default)

    def close(self) -> None:
        self.kp.close()

    # ------------------------------------------------------------------------------------------- weights
    @property
    def wL(self) -> float:
        return float(_smooth(self.uL))

    @property
    def wA(self) -> float:
        return float(_smooth(self.uA))

    @property
    def wP(self) -> float:
        return float(_smooth(self.uP))

    def _ramp(self, u, target, T):
        if T <= 0:
            return float(target)
        return float(np.clip(u + np.sign(target - u) * self.dt / T, min(u, target), max(u, target)))

    def _feet_still(self):
        m, d = self.model, self.data
        v = np.zeros(6)
        sp = []
        for b in self.feet:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            sp.append(float(np.hypot(v[3], v[4])))
        return max(sp) < SETTLE_V and float(np.hypot(d.qvel[0], d.qvel[1])) < SETTLE_V

    def _update_phase(self, walk_cmd, win_static):
        """Mode logic, one tick (see module doc)."""
        dt = self.dt
        self.t_phase += dt
        intent = bool(abs(walk_cmd[0]) > INTENT_V or abs(walk_cmd[1]) > INTENT_V or abs(walk_cmd[2]) > INTENT_W)
        self.t_intent = self.t_intent + dt if intent else 0.0
        if self.lock == "walk":
            self.phase = "walk"
            return intent
        if self.lock == "stand":
            self.phase = "stand"
            return False
        ph = self.phase
        if ph == "stand" and self.t_intent >= self.enter_s - 1e-9:
            self.phase, self.t_phase = "walk", 0.0
            self.stats["walk_entries"] += 1
        elif ph == "walk" and not intent and self.t_phase >= self.min_walk_s:
            self.phase, self.t_phase, self.t_still = "settle", 0.0, 0.0
        elif ph == "settle":
            if intent:
                self.phase, self.t_phase = "walk", 0.0
            else:
                self.t_still = self.t_still + dt if (win_static and self._feet_still()) else 0.0
                timeout = self.t_phase >= self.settle_max_s
                if self.t_still >= self.settle_s - 1e-9 or timeout:
                    self.stats["settle_timeouts"] += int(timeout and self.t_still < self.settle_s)
                    self.stats["stand_entries"] += 1
                    self.phase, self.t_phase = "stand", 0.0
        return intent

    def _planner_cmd(self, c):
        """Walking command -> planner command: no shuffle speeds (see min_step_v / turn_step_v)."""
        c = np.array(c, np.float64)
        v = float(np.hypot(c[0], c[1]))
        if v > INTENT_V and v < self.min_step_v:
            c[:2] *= self.min_step_v / v
        elif v <= INTENT_V and abs(c[2]) > INTENT_W and self.turn_step_v > 0:
            c[0], c[1] = self.turn_step_v, 0.0
        return c

    # ------------------------------------------------------------------------------------------- one tick
    def step(self, arm_q: Sequence[float], waist_yaw: Optional[float] = None, bow: Optional[float] = None,
             loco_cmd: Sequence[float] = (0.0, 0.0, 0.0), arm_dq: Optional[Sequence[float]] = None,
             walk_cmd: Optional[Sequence[float]] = None) -> dict:
        """walk_cmd: the walking command without the server's heading hold (None = loco_cmd). It decides the mode and
        goes to the planner; the loco (standing) only gets the heading-hold part of loco_cmd."""
        from k1live.planner import RefWindow, window_motion

        t_tick = time.perf_counter()
        s, d, ctrl = self.sim, self.data, self.ctrl
        dt = self.dt
        loco_cmd = np.asarray(loco_cmd, np.float64)
        walk_cmd = loco_cmd.copy() if walk_cmd is None else np.asarray(walk_cmd, np.float64)
        arm_q = np.clip(np.asarray(arm_q, np.float64), self.arm_lo, self.arm_hi)
        if arm_dq is None:
            arm_dq = np.zeros(10) if self._prev_arm is None else (arm_q - self._prev_arm) / dt
        arm_dq = np.asarray(arm_dq, np.float64)
        self._prev_arm = arm_q.copy()

        # ---- planner window (frame 0 = now) -----------------------------------------------------------------
        qpos30 = np.concatenate([d.qpos[:7], d.qpos[s.qadr]])
        walking = self.phase == "walk"
        pc = self._planner_cmd(walk_cmd) if (walking and self.wL >= self.cmd_at_w - 1e-9) else np.zeros(3)
        if abs(pc[2]) < 1e-9 and abs(self._pc_wz) >= 1e-9:
            # turn released: drop the planner's facing lead (up to 0.6 rad ahead of its heading while turning), or
            # the robot keeps turning up to ~35 deg after the person / operator stopped
            self.kp.face_yaw = self.kp.stream.heading()
        self._pc_wz = float(pc[2])
        self.kp.command(*[float(v) for v in pc])
        t0 = time.perf_counter()
        w = self.kp.step(qpos30)
        planner_ms = (time.perf_counter() - t0) * 1e3
        win_static = bool(w.mode == 0 and np.abs(w.q[1] - w.q[0])[self.lower].max() < 2e-3)

        # ---- mode + hand-over weights ----------------------------------------------------------------------
        self._update_phase(walk_cmd, win_static)
        ph = self.phase
        sonic_legs_always = self.legs == "sonic" or self.lock == "walk"
        tL = 1.0 if (sonic_legs_always or ph in ("walk", "settle")) else 0.0
        # arms on the walking side: while walking, and (settle_arms "walk") until the legs are back on the loco
        arms_walking = ph == "walk" or (self.settle_arms == "walk" and (
            ph == "settle" or (ph == "stand" and self.uL > 0.0 and not sonic_legs_always)))
        tA = 1.0 if (self.lock == "walk" or (arms_walking and self.walk_arms == "sonic")) else 0.0
        tP = (1.0 - self.walk_arm_human) if arms_walking else 0.0
        self.uL = self._ramp(self.uL, tL, self.blend_s)
        self.uA = self._ramp(self.uA, tA, self.arm_blend_s)
        self.uP = self._ramp(self.uP, tP, self.arm_blend_s)
        wL, wA, wP = self.wL, self.wA, self.wP
        dwP = (wP - float(_smooth(self._uP_prev))) / dt
        self._uP_prev = self.uP

        # ---- arm reference: person <-> planner arm swing (frame 0 + SONIC's future frames) ------------------
        H = len(w.q)
        n = min(self.n_future, H)
        k = np.arange(n)[:, None]
        human_f = arm_q[None] + k * dt * np.clip(arm_dq, -ARM_VMAX, ARM_VMAX)[None]
        plan_arm = w.q[:n][:, self.arm_idx]
        arm_f = (1.0 - wP) * human_f + wP * plan_arm
        arm_f = np.clip(arm_f, self.arm_lo, self.arm_hi)
        arm_ref = arm_f[0]
        arm_ref_dq = (1.0 - wP) * arm_dq + wP * w.dq[0, self.arm_idx] + dwP * (plan_arm[0] - human_f[0])

        # ---- ROBOTIS student (arms) + loco (legs, standing), as k1.sim.K1Split.step ----------------------------
        t_pol = time.perf_counter()
        if self.trim_gain > 0:   # split backend arm trim (outer loop); only while the student follows the person
            quasi = np.abs(arm_ref_dq) < self.trim_vmax
            err = arm_ref - self.arm_q_now()
            if wA > 0 or wP > 0:
                quasi[:] = False
            self.trim = np.where(quasi, self.trim + self.trim_gain * err * dt,
                                 self.trim * (1.0 - dt / self.trim_leak_s))
            self.trim = np.clip(self.trim, -self.trim_max, self.trim_max)
        rq = s.loco.default.copy()
        rdq = np.zeros(len(rq))
        rq[self.arm_idx] = np.clip(arm_ref + self.trim, self.arm_lo, self.arm_hi)
        rdq[self.arm_idx] = arm_ref_dq
        rtq = _yaw_quat_xyzw(quat_to_mat(d.xquat[s.torso_id]))
        R_wb = quat_to_mat(d.xquat[s.torso_id])
        R_wt = quat_to_mat(np.asarray(rtq)[[3, 0, 1, 2]])
        anchor = (R_wb.T @ (s.init_align @ R_wt))[:, :2].reshape(-1)
        jp, jv = d.qpos[s.qadr].copy(), d.qvel[s.vadr].copy()
        t_student = s.pol.act({
            "motion_command": np.concatenate([rq, rdq]),
            "motion_anchor_ori_b": anchor,
            "base_ang_vel": d.qvel[3:6].copy(),
            "joint_pos_rel": np.where(s.lower, rq, jp) - s.pol.default,
            "joint_vel_rel": np.where(s.lower, rdq, jv),
            "last_action": s.pol.last_action,
        })
        R_root = quat_to_mat(d.qpos[3:7].copy())
        lt = s._lt
        hold = loco_cmd - walk_cmd if walk_cmd is not loco_cmd else np.zeros(3)
        loco_in = np.array([0.0, 0.0, hold[2]]) if self.lock != "stand" else loco_cmd
        t_loco = s.loco.act({
            lt["ang"]: d.qvel[3:6].copy(),
            lt["grav"]: R_root.T @ np.array([0.0, 0.0, -1.0]),
            lt["cmd"]: loco_in,
            lt["jp"]: jp - s.loco.default,
            lt["jv"]: jv,
            lt["act"]: s.loco.last_action,
        })
        t_split = np.where(s.lower, t_loco, t_student)
        if self.arm_mode == "pd":   # SplitBackend._pd_target: feed-forward terms folded into the PD target
            kpa, kda = self.kp_split[self.arm_idx], self.kd_split[self.arm_idx]
            ta = rq[self.arm_idx].copy()
            if self.vel_ff:
                ta += kda * arm_ref_dq / kpa
            if self.gravity_comp:
                ta += d.qfrc_bias[self.arm_vadr] / kpa
            t_split = t_split.copy()
            t_split[self.arm_idx] = ta
        rob_ms = (time.perf_counter() - t_pol) * 1e3

        # ---- SONIC reference (planner window, arms laid over, waist / bow while standing) -> policy -----------
        q = w.q.copy()
        q[:n, self.arm_idx] = arm_f
        q[n:, self.arm_idx] = arm_f[-1]
        stand_w = 1.0 - wP
        if self.waist and waist_yaw is not None:
            wy = float(np.clip(waist_yaw, -self.waist_max, self.waist_max))
            q[:, WAIST] = (1 - stand_w) * q[:, WAIST] + stand_w * wy
        standing = w.speed < 0.05 and ph != "walk"
        self._bow_w = float(np.clip(self._bow_w + (1.0 if standing else -1.0) * dt / 0.5, 0.0, 1.0))
        tb = float(np.clip(bow, 0.0, self.bow_max)) if (bow is not None and self.supports["bow"]) else 0.0
        self._bow += float(np.clip(tb - self._bow, -0.8 * dt, 0.8 * dt))
        b = self._bow * self._bow_w
        quat = w.quat
        if b > 1e-3:
            q[:, HIP_PITCH] -= b
            from k1live.backends.sonic import _qmul_right

            quat = _qmul_right(quat, _quat_pitch(b))
        q = np.clip(q, self.q_lo, self.q_hi)
        win = dataclasses.replace(w, quat=quat, q=q)
        self.last_window = win
        ctrl.current_motion = window_motion(win, self.robot, q)
        ctrl.current_frame, ctrl.play = 1, True
        if wL <= 0.0:          # SONIC not on the legs: keep its heading frame on the robot (no stale yaw error)
            ctrl.reinitialize_heading = True
        t0 = time.perf_counter()
        ls = {"q": jp, "dq": jv, "imu_quat": d.qpos[3:7].copy(), "imu_gyro": d.qvel[3:6].copy()}
        q_sonic, kp_sonic, kd_sonic = ctrl.step(ls)
        sonic_ms = (time.perf_counter() - t0) * 1e3

        # ---- hand-over: convex blend of the two PD controllers per joint group -----------------------------
        wj = np.where(s.lower, wL, wA)
        kp_b = (1 - wj) * self.kp_split + wj * kp_sonic
        kd_b = (1 - wj) * self.kd_split + wj * kd_sonic
        target = ((1 - wj) * self.kp_split * t_split + wj * kp_sonic * q_sonic) / kp_b
        split_only, sonic_only = wj <= 0.0, wj >= 1.0      # exact at the ends (kp t / kp may round by an ulp)
        kp_b = np.where(split_only, self.kp_split, np.where(sonic_only, kp_sonic, kp_b))
        kd_b = np.where(split_only, self.kd_split, np.where(sonic_only, kd_sonic, kd_b))
        target = np.where(split_only, t_split, np.where(sonic_only, q_sonic, target))
        for _ in range(s.decim):
            qd = d.qvel[s.vadr]
            tau = kp_b * (target - d.qpos[s.qadr]) - kd_b * qd
            tau = np.where((qd >= s.vel_lim) & (tau > 0), 0.0, tau)
            tau = np.where((qd <= -s.vel_lim) & (tau < 0), 0.0, tau)
            d.ctrl[:] = np.clip(tau, -s.tau_lim, s.tau_lim)
            mujoco.mj_step(s.model, d)
        s.t += 1

        # ---- shadow bookkeeping: last_action = the applied target in each policy's action space -------------
        if wA > 0:   # student's own joints = arms
            enc = s.pol.adapter.encode_action(target, canonical=False)
            s.pol.last_action = np.where(self.arm_mask, enc, s.pol.last_action)
        if wL > 0:   # loco's own joints = legs + waist
            enc = s.loco.adapter.encode_action(target, canonical=False)
            s.loco.last_action = np.where(s.lower, enc, s.loco.last_action)
        if (wj < 1).any():
            raw_m = (target - self.s_default) / self.s_scale
            la = ctrl.last_action.copy()
            la[self.i2m] = np.where(wj < 1, raw_m, la[self.i2m])
            ctrl.last_action = la
        pol_ms = rob_ms + sonic_ms
        s.last_policy_ms = pol_ms
        self.last_info = {"policy_ms": pol_ms, "planner_ms": planner_ms, "planner_infer_ms": float(w.planner_ms),
                          "target": target, "phase": ph, "wL": wL, "wA": wA, "wP": wP, "ref_q": q[0].copy(),
                          "tick_ms": (time.perf_counter() - t_tick) * 1e3}
        return self.last_info

    def note(self) -> str:
        """Short state for SimState.note (the panel's backend tile shows the phase)."""
        return f"hybrid:{self.phase}"
