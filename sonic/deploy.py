# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Python port of G1Deploy (g1_deploy_onnx_ref.cpp) — encoder + decoder ONNX at 50 Hz.

Faithfully reproduced from the C++:
  * observation registry names/dimensions and gatherer semantics
    (GatherHis*, GatherMotion*MultiFrame, GatherMotionAnchorOrientationMutiFrame,
    GatherEncoderMode, GatherTokenState), including the mode filter in
    GatherEncoderObservations (non-required encoder inputs stay 0)
  * StateLogger::GetLatest(n, dt, newest_first=false): oldest -> newest, zero entries
    (base_quat = 0,0,0,0 -> gravity 0) padded at the oldest end
  * body_q = q[mujoco_to_isaaclab] - default_angles[mujoco_to_isaaclab], last_action = raw
    previous network output (IsaacLab order) logged *before* the new inference
  * heading: UpdateHeadingState / ComputeApplyDeltaHeading
  * action -> q_target = default_angles + action[isaaclab_to_mujoco] * g1_action_scale,
    kp = kps, kd = kds, then apply_motor_gain_scales (optional per-motor factors)
  * robot-parameterised (sonic.robots.RobotSpec): G1 by default (bit-identical to the C++ port),
    K1 or any other embodiment via robot=..., which sets joint orders, gains, action scale, default
    pose and every observation size (num_dof, wrist / lower-body joint lists)
  * INIT: 3 s linear ramp from the current pose to default_angles with kps/kds
  * CurrentFrameAdvancement for reference motions and the "streamed" motion
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from . import rot
from .motion import MotionSequence
from .params import CONTROL_DT, INIT_DURATION, MODELS
from .robots import RobotSpec, g1


# --------------------------------------------------------------------------------------
# observation config (observation_config.hpp semantics: enabled flags, order = file order)
# --------------------------------------------------------------------------------------
@dataclass
class ObsConfig:
    observations: list
    encoder_dimension: int
    encoder_observations: list
    encoder_modes: dict  # mode_id -> (name, [required obs])

    @classmethod
    def load(cls, path):
        cfg = yaml.safe_load(Path(path).read_text())
        obs = [o["name"] for o in cfg["observations"] if o.get("enabled", True)]
        enc = cfg.get("encoder", {}) or {}
        enc_obs = [o["name"] for o in enc.get("encoder_observations", []) if o.get("enabled", True)]
        modes = {m["mode_id"]: (m["name"], list(m["required_observations"]))
                 for m in enc.get("encoder_modes", [])}
        return cls(obs, int(enc.get("dimension", 0)), enc_obs, modes)


@dataclass
class LogEntry:
    base_quat: np.ndarray
    base_ang_vel: np.ndarray
    body_q: np.ndarray  # isaaclab order, minus default
    body_dq: np.ndarray
    last_action: np.ndarray


def _zero_entry(n):
    return LogEntry(np.zeros(4), np.zeros(3), np.zeros(n), np.zeros(n), np.zeros(n))


@dataclass
class HeadingState:
    init_base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    delta_heading: float = 0.0


class SonicController:
    """One instance == one C++ G1Deploy (minus DDS/threads)."""

    def __init__(self, variant: str = "release", models_dir: Path = MODELS,
                 motor_kp_scale: dict | None = None, motor_kd_scale: dict | None = None,
                 threads: int = 2, robot: RobotSpec | None = None):
        self.robot = robot = robot or g1()
        self._zero = _zero_entry(robot.num_dof)
        vdir = Path(models_dir) / variant
        self.cfg = ObsConfig.load(vdir / "observation_config.yaml")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(vdir / "model_encoder.onnx"), so, providers=prov)
        self.decoder = ort.InferenceSession(str(vdir / "model_decoder.onnx"), so, providers=prov)
        self.enc_in = self.encoder.get_inputs()[0].name
        self.dec_in = self.decoder.get_inputs()[0].name
        self.enc_dim = int(self.encoder.get_inputs()[0].shape[1])
        self.dec_dim = int(self.decoder.get_inputs()[0].shape[1])
        self.token_dim = self.cfg.encoder_dimension or 64

        self.registry = self._build_registry()
        self.policy_layout = self._layout(self.cfg.observations, self.dec_dim, "policy")
        self.encoder_layout = self._layout(self.cfg.encoder_observations, self.enc_dim, "encoder")

        self.kp = robot.kps.copy()
        self.kd = robot.kds.copy()
        for k, v in (motor_kp_scale or {}).items():
            self.kp[int(k)] *= float(v)
        for k, v in (motor_kd_scale or {}).items():
            self.kd[int(k)] *= float(v)

        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        self.state = "INIT"
        self.time_ = 0.0
        self.init_q = None
        self.history: list[LogEntry] = []
        self.last_action = np.zeros(self.robot.num_dof)
        self.token = np.zeros(self.token_dim)
        self.current_motion: MotionSequence | None = None
        self.current_frame = 0
        self.play = False
        self.reinitialize_heading = True
        self.heading = HeadingState()
        self.init_ref_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.saved_frame_for_observation_window = 0
        self.last_cmd = None
        self.last_policy_obs = None
        self.last_encoder_obs = None
        self.motion_completed = False

    def set_motion(self, motion: MotionSequence, play: bool = False):
        """Keyboard N/P equivalent: switch motion, frame 0, re-init heading."""
        self.current_motion = motion
        self.current_frame = 0
        self.play = play
        self.reinitialize_heading = True

    def start_control(self):
        """']' key: WAIT_FOR_CONTROL -> CONTROL (history starts empty)."""
        self.state = "CONTROL"

    # ------------------------------------------------------------------ registry
    def _layout(self, names, model_dim, what):
        lay, off = [], 0
        for n in names:
            if n not in self.registry:
                raise KeyError(f"unknown {what} observation '{n}'")
            dim = self.registry[n][0]
            lay.append((n, off, dim))
            off += dim
        if off != model_dim:
            raise ValueError(f"{what} obs dim {off} != model input {model_dim}")
        return lay

    @staticmethod
    def registry_dims(robot: RobotSpec, token_dim: int = 64) -> dict:
        """name -> size of every observation the deploy knows, for `robot` (no models needed)."""
        probe = SonicController.__new__(SonicController)
        probe.robot, probe.token_dim = robot, token_dim
        return {k: v[0] for k, v in probe._build_registry().items()}

    def _build_registry(self):
        R = {}
        tok = self.token_dim if hasattr(self, "token_dim") else 64
        N = self.robot.num_dof
        W = list(self.robot.wrist_il)
        L = list(self.robot.lower_body_il)
        nW, nL = len(W), len(L)

        def jp(n, s, idx=None):
            return lambda: self._motion_joint_pos(n, s, idx)

        def jv(n, s, idx=None):
            return lambda: self._motion_joint_vel(n, s, idx)

        def ori(n, s, mode=0):
            return lambda: self._anchor_orientation(n, s, mode)

        R["token_state"] = (tok, lambda: self.token.copy())
        R["encoder_mode"] = (3, lambda: self._encoder_mode(3))
        R["encoder_mode_4"] = (4, lambda: self._encoder_mode(4))
        R["encoder_index"] = (4, lambda: self._encoder_mode(4))
        # motion joint terms
        for n, s in [(1, 1), (10, 5), (10, 1), (3, 1), (5, 5)]:
            suffix = "" if (n, s) == (1, 1) else f"_{n}frame_step{s}"
            R[f"motion_joint_positions{suffix}"] = (N * n, jp(n, s))
            R[f"motion_joint_velocities{suffix}"] = (N * n, jv(n, s))
        for s in (5, 1):
            R[f"motion_joint_positions_lowerbody_10frame_step{s}"] = (10 * nL, jp(10, s, L))
            R[f"motion_joint_velocities_lowerbody_10frame_step{s}"] = (10 * nL, jv(10, s, L))
        for n in (10, 2, 4):
            R[f"motion_joint_positions_wrists_{n}frame_step1"] = (nW * n, jp(n, 1, W))
        R["motion_joint_velocities_wrists_10frame_step1"] = (10 * nW, jv(10, 1, W))
        # sonic_release aliases
        R["command_multi_future_nonflat"] = (20 * N, lambda: np.concatenate([self._motion_joint_pos(10, 5), self._motion_joint_vel(10, 5)]))
        R["motion_anchor_ori_b_mf_nonflat"] = (60, ori(10, 5))
        R["command_multi_future_lower_body"] = (20 * nL, lambda: np.concatenate([self._motion_joint_pos(10, 5, L), self._motion_joint_vel(10, 5, L)]))
        R["motion_anchor_ori_b"] = (6, ori(1, 1))
        R["smpl_joints_multi_future_local_nonflat"] = (720, lambda: self._smpl_joints(10, 1))
        R["smpl_root_ori_b_multi_future"] = (60, ori(10, 1))
        R["joint_pos_multi_future_wrist_for_smpl"] = (10 * nW, jp(10, 1, W))
        # root z
        R["motion_root_z_position"] = (1, lambda: self._root_z(1, 1))
        for n, s in [(10, 5), (10, 1), (3, 1)]:
            R[f"motion_root_z_position_{n}frame_step{s}"] = (n, (lambda n=n, s=s: self._root_z(n, s)))
        R["smpl_root_z_10frame_step1"] = (10, lambda: self._root_z(10, 1))
        # orientation terms (mode 0 = base, 1 = heading, 2 = refheading)
        R["motion_anchor_orientation"] = (6, ori(1, 1))
        for tag, mode in [("", 0), ("_heading", 1), ("_refheading", 2)]:
            if tag:
                R[f"motion_anchor_orientation{tag}"] = (6, ori(1, 1, mode))
            for n, s in [(10, 5), (10, 1)]:
                R[f"motion_anchor_orientation{tag}_{n}frame_step{s}"] = (6 * n, ori(n, s, mode))
        for n in (10, 2, 4):
            R[f"smpl_anchor_orientation_{n}frame_step1"] = (6 * n, ori(n, 1, 0))
        for tag, mode in [("_heading", 1), ("_refheading", 2)]:
            for n in (10, 2):
                R[f"smpl_anchor_orientation{tag}_{n}frame_step1"] = (6 * n, ori(n, 1, mode))
        # smpl joints
        R["smpl_joints"] = (72, lambda: self._smpl_joints(1, 1))
        for n, s in [(5, 5), (10, 5), (10, 1), (2, 1), (4, 1)]:
            R[f"smpl_joints_{n}frame_step{s}"] = (72 * n, (lambda n=n, s=s: self._smpl_joints(n, s)))
        R["smpl_joints_lower_10frame_step1"] = (270, lambda: self._smpl_joints(10, 1, [0, 1, 2, 4, 5, 7, 8, 10, 11]))
        # VR 3-point (only needed for teleop mode 1 -> not supported in this port)
        for n, d in [("vr_3point_local_target", 9), ("vr_3point_local_target_compliant", 9),
                     ("vr_3point_local_orn_target", 12), ("vr_3point_compliance", 3),
                     ("vr_5point_local_target", 15), ("vr_5point_local_orn_target", 20)]:
            R[n] = (d, self._unsupported(n))
        # history terms
        R["base_angular_velocity"] = (3, lambda: self._his("ang", 1))
        R["body_joint_positions"] = (N, lambda: self._his("q", 1))
        R["body_joint_velocities"] = (N, lambda: self._his("dq", 1))
        R["last_actions"] = (N, lambda: self._his("act", 1))
        R["gravity_dir"] = (3, lambda: self._his("grav", 1))
        for n in (4, 10):
            R[f"his_body_joint_positions_{n}frame_step1"] = (N * n, (lambda n=n: self._his("q", n)))
            R[f"his_body_joint_velocities_{n}frame_step1"] = (N * n, (lambda n=n: self._his("dq", n)))
            R[f"his_last_actions_{n}frame_step1"] = (N * n, (lambda n=n: self._his("act", n)))
            R[f"his_base_angular_velocity_{n}frame_step1"] = (3 * n, (lambda n=n: self._his("ang", n)))
            R[f"his_gravity_dir_{n}frame_step1"] = (3 * n, (lambda n=n: self._his("grav", n)))
        return R

    @staticmethod
    def _unsupported(name):
        def f():
            raise NotImplementedError(f"{name}: VR teleop encoder mode is not ported")
        return f

    # ------------------------------------------------------------------ gatherers
    def _frames(self, n, s, always_advance=False):
        T = self.current_motion.timesteps
        out = []
        for k in range(n):
            f = self.current_frame
            if self.play or always_advance:
                f = min(f + k * s, T - 1)
            out.append(f)
        self.saved_frame_for_observation_window = max(self.saved_frame_for_observation_window, (n - 1) * s + 1)
        return out

    def _motion_joint_pos(self, n, s, idx=None):
        jp = self.current_motion.joint_pos
        fr = self._frames(n, s)
        x = jp[fr] if idx is None else jp[fr][:, idx]
        return x.reshape(-1)

    def _motion_joint_vel(self, n, s, idx=None):
        jv = self.current_motion.joint_vel
        fr = self._frames(n, s, always_advance=True)
        x = jv[fr] if idx is None else jv[fr][:, idx]
        if not self.play:
            x = np.zeros_like(x)
        return x.reshape(-1)

    def _root_z(self, n, s):
        bp = self.current_motion.body_pos
        return bp[self._frames(n, s), 0, 2].reshape(-1)

    def _smpl_joints(self, n, s, idx=None):
        sj = self.current_motion.smpl_joints
        if sj is None:
            raise RuntimeError("motion has no SMPL joints")
        fr = self._frames(n, s)
        x = sj[fr] if idx is None else sj[fr][:, idx]
        return x.reshape(-1)

    def _encoder_mode(self, dim):
        v = np.zeros(dim)
        v[0] = float(self.current_motion.encode_mode)
        return v

    def _latest_base_quat(self):
        return self.history[-1].base_quat if self.history else np.zeros(4)

    def apply_delta_heading(self):
        init_heading = rot.calc_heading_quat(self.heading.init_base_quat)
        data_heading_inv = rot.calc_heading_quat_inv(self.init_ref_root_quat)
        q = rot.quat_mul(init_heading, data_heading_inv)
        if self.heading.delta_heading != 0.0:
            q = rot.quat_mul(rot.euler_z_to_quat(self.heading.delta_heading), q)
        return q

    def _anchor_orientation(self, n, s, mode=0):
        base_quat = self._latest_base_quat()
        adh = self.apply_delta_heading()
        bq = self.current_motion.body_quat
        T = self.current_motion.timesteps
        if mode == 2:
            first = min(self.current_frame, T - 1)
            left = rot.calc_heading_quat(rot.quat_mul(adh, bq[first, 0]))
        elif mode == 1:
            left = rot.calc_heading_quat(base_quat)
        else:
            left = base_quat
        out = []
        for k in range(n):
            f = self.current_frame
            if self.play:
                f = min(f + k * s, T - 1)
            new_ref = rot.quat_mul(adh, bq[f, 0])
            out.append(rot.rot6d_rowwise(rot.quat_mul(rot.quat_conj(left), new_ref)))
        return np.concatenate(out)

    def _his(self, key, n):
        h = self.history[-n:]
        h = [self._zero] * (n - len(h)) + h  # oldest -> newest, zeros at the oldest end
        if key == "q":
            return np.concatenate([e.body_q for e in h])
        if key == "dq":
            return np.concatenate([e.body_dq for e in h])
        if key == "act":
            return np.concatenate([e.last_action for e in h])
        if key == "ang":
            return np.concatenate([e.base_ang_vel for e in h])
        if key == "grav":
            return np.concatenate([rot.quat_rotate(rot.quat_conj(e.base_quat), np.array([0.0, 0.0, -1.0])) for e in h])
        raise KeyError(key)

    # ------------------------------------------------------------------ control tick
    def update_heading_state(self):
        m = self.current_motion
        if self.reinitialize_heading:
            self.heading = HeadingState(self._latest_base_quat().copy(), 0.0)
            self.reinitialize_heading = False
            self.init_ref_root_quat = m.body_quat[self.current_frame, 0].copy()
        if self.current_frame == 0:
            self.init_ref_root_quat = m.body_quat[0, 0].copy()

    def gather_encoder_obs(self):
        buf = np.zeros(self.enc_dim)
        mode = self.current_motion.encode_mode
        required = set(self.cfg.encoder_modes[mode][1]) if self.cfg.encoder_modes else None
        for name, off, dim in self.encoder_layout:
            if required is not None and name not in required:
                continue
            buf[off:off + dim] = self.registry[name][1]()
        return buf

    def gather_policy_obs(self):
        buf = np.zeros(self.dec_dim)
        for name, off, dim in self.policy_layout:
            if name == "token_state":
                enc_obs = self.gather_encoder_obs()
                self.last_encoder_obs = enc_obs
                tok = self.encoder.run(None, {self.enc_in: enc_obs[None].astype(np.float32)})[0][0]
                self.token = tok.astype(np.float64)
                buf[off:off + dim] = self.token
            else:
                buf[off:off + dim] = self.registry[name][1]()
        return buf

    def log_state(self, ls):
        q, dq = ls["q"], ls["dq"]
        m2i = self.robot.mujoco_to_isaaclab
        e = LogEntry(
            base_quat=np.asarray(ls["imu_quat"], dtype=np.float64).copy(),
            base_ang_vel=np.asarray(ls["imu_gyro"], dtype=np.float64).copy(),
            body_q=q[m2i] - self.robot.default_angles[m2i],
            body_dq=dq[m2i].copy(),
            last_action=self.last_action.copy(),
        )
        self.history.append(e)
        if len(self.history) > 64:
            del self.history[:-64]

    def step(self, ls):
        """One 50 Hz Control() tick. ls = sim.low_state(). Returns (q_target, kp, kd) in hw order."""
        if self.state == "INIT":
            if self.init_q is None:
                self.init_q = ls["q"].copy()
            self.time_ += CONTROL_DT
            default = self.robot.default_angles
            q_t = default.copy()
            if self.time_ < INIT_DURATION:
                r = np.clip(self.time_ / INIT_DURATION, 0.0, 1.0)
                q_t = ls["q"] * (1.0 - r) + default * r
            else:
                self.state = "WAIT_FOR_CONTROL"
            self.last_cmd = (q_t.astype(np.float32).astype(np.float64), self.robot.kps.copy(), self.robot.kds.copy())
            return self.last_cmd
        if self.state == "WAIT_FOR_CONTROL":
            return self.last_cmd
        # CONTROL
        self.log_state(ls)
        self.saved_frame_for_observation_window = 0
        self.update_heading_state()
        obs = self.gather_policy_obs()
        self.last_policy_obs = obs
        action = self.decoder.run(None, {self.dec_in: obs[None].astype(np.float32)})[0][0].astype(np.float64)
        r = self.robot
        q_t = r.default_angles + action[r.isaaclab_to_mujoco] * r.action_scale
        self.last_action = action.copy()
        self.last_cmd = (q_t.astype(np.float32).astype(np.float64), self.kp.copy(), self.kd.copy())
        self.advance_frame()
        return self.last_cmd

    def advance_frame(self):
        m = self.current_motion
        if m is None or m.timesteps == 0 or not self.play:
            return
        self.current_frame += 1
        if m.name != "streamed":
            if self.current_frame >= m.timesteps:
                self.play = False
                self.current_frame = 0
                self.reinitialize_heading = True
                self.motion_completed = True
        else:
            if self.current_frame >= m.timesteps - self.saved_frame_for_observation_window:
                self.current_frame -= 1


def parse_gain_specs(specs):
    """C++ --motor-kp-scale syntax: "<motor-list>=<factor>", motor-list = "0-11,15" (hardware idx)."""
    out = {}
    for spec in specs or []:
        lst, fac = spec.split("=")
        fac = float(fac)
        if not 0.0 <= fac <= 4.0:
            raise ValueError("scale factor must be in [0, 4]")
        for part in lst.split(","):
            a, _, b = part.partition("-")
            for i in range(int(a), int(b or a) + 1):
                if i in out:
                    raise ValueError(f"motor index scaled more than once: {i}")
                out[i] = fac
    return out
