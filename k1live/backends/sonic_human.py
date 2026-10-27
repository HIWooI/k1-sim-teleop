"""sonic_human backend: the K1 SONIC policy driven directly by the person's whole-body pose (SMPL encoder) - legs and
steps included - instead of a robot reference built from arm IK + walking commands (sonic backend).

Input: protocol.HumanFrame.smpl of the newest fresh person frame (sim_server passes it as `human=` while it follows a
tracked person): root-local SMPL-24 joints (pelvis at 0) + root quaternion, mirrored by the sender, produced by the
same converter the training data went through (k1live/smpl_stream.py, sonic_k1_train/data/human_smpl.py).

Per 20 ms tick:
  person frame -> PresenceGate (sonic/mp2smpl: fade in from / out to a calm standing pose, short dropouts held)
              -> first-order smoothing on the skeleton manifold (tau 30 ms: 30 Hz camera steps -> 50 Hz ramps)
              -> 50 Hz buffer -> the encoder's n future frames ("delay": frame n-1 = newest, i.e. a true future,
                 the robot runs (n-1) x 20 ms behind; "hold": all frames = newest)
              -> MotionSequence(encode_mode 2: SMPL encoder): smpl_joints = local joints, body_quat = root quat,
                 wrist joints = the sender's K1 wrist roll -> K1HumanController (export layout, encoder_index 2).
The deploy's heading alignment (UpdateHeadingState) re-anchors whenever the person stream (re)starts: the person's
heading at that moment = the robot's current heading; turning in place turns the robot.

Without a person (operator stop / estop, nobody in view, attract gestures from the server) the backend runs the robot
encoder exactly like the sonic backend's standing path (default-pose reference + the server's arm targets), switching
over only once the SMPL reference has faded back to the standing pose.

Model: models/k1/k1s_human/deploy (or K1LIVE_HUMAN_VARIANT): model_encoder.onnx / model_decoder.onnx of a K1 SONIC
checkpoint trained with BONES-SEED SMPL data + observation_config.yaml written by prepare_human_variant(). Without it
the backend is not offered (sim_server.available_backends).

    .venv/bin/python -c "from k1live.backends.sonic_human import prepare_human_variant as p; \
        p('models/k1/<run>/exported', 6000, 'models/k1/k1s_human/deploy', n_future=4, step=1, smpl_frames=4)"
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from k1live import smpl_stream
from k1live.backends.sonic import ROOT, SonicBackend
from k1live.protocol import POLICY_JOINTS

DEFAULT_VARIANT = Path(os.environ.get("K1LIVE_HUMAN_VARIANT", ROOT / "models" / "k1" / "k1s_human" / "deploy"))
SMPL_MODE = 2            # encoders_to_iterate = [g1, teleop, smpl]
N_ENCODERS = 3
SMOOTH_TAU = 0.03        # s, smoothing of the 30 Hz person stream into 50 Hz references


# ------------------------------------------------------------------------------------------------ variant
def obs_config_human(n_future: int, step: int, smpl_frames: int) -> dict:
    """Deploy observation_config of a gear_sonic export of sonic_k1_small*(la4) trained WITH SMPL data: the exporter
    input is [encoder_index | tokenizer obs in TokenizerCfg order] (sonic_k1_train/mac/eval_k1_mujoco.obs_config),
    plus the SMPL encoder mode. smpl_frames = smpl_num_future_frames of the training config (step 1 = 20 ms)."""
    sfx = "" if (n_future, step) == (10, 5) else f"_{n_future}f{step}"
    ssf = "" if smpl_frames == 10 else f"_{smpl_frames}f"
    smpl = [f"smpl_joints_multi_future_local_nonflat{ssf}", f"smpl_root_ori_b_multi_future{ssf}",
            f"joint_pos_multi_future_wrist_for_smpl{ssf}"]
    enc = ["encoder_index", "encoder_index_onehot", f"command_multi_future_nonflat{sfx}", "motion_anchor_ori_b",
           f"motion_anchor_ori_b_mf_nonflat{sfx}", f"command_multi_future_lower_body{sfx}",
           "vr_3point_local_target", "vr_3point_local_orn_target"] + smpl
    his = ["his_base_angular_velocity_10frame_step1", "his_body_joint_positions_10frame_step1",
           "his_body_joint_velocities_10frame_step1", "his_last_actions_10frame_step1", "his_gravity_dir_10frame_step1"]
    return {
        "observations": [{"name": "token_state"}] + [{"name": n} for n in his],
        "encoder": {
            "dimension": 64,
            "encoder_observations": [{"name": n} for n in enc],
            "encoder_modes": [{"name": "g1", "mode_id": 0, "required_observations": enc[:3] + enc[4:5]},
                              {"name": "smpl", "mode_id": SMPL_MODE, "required_observations": enc[:2] + smpl}],
        },
        "k1": {"num_future_frames": n_future, "step": step, "smpl_num_future_frames": smpl_frames},
    }


def prepare_human_variant(export_dir, step_id: int, out, n_future: int = 4, step: int = 1, smpl_frames: int = 4):
    import shutil

    export_dir, out = Path(export_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"model_step_{step_id:06d}"
    shutil.copy(export_dir / f"{stem}_encoder.onnx", out / "model_encoder.onnx")
    shutil.copy(export_dir / f"{stem}_decoder.onnx", out / "model_decoder.onnx")
    (out / "observation_config.yaml").write_text(yaml.safe_dump(obs_config_human(n_future, step, smpl_frames),
                                                                sort_keys=False))
    return out


def variant_ok(vdir) -> bool:
    vdir = Path(vdir)
    try:
        cfg = yaml.safe_load((vdir / "observation_config.yaml").read_text())
        modes = {m["mode_id"] for m in cfg["encoder"]["encoder_modes"]}
        return SMPL_MODE in modes and (vdir / "model_encoder.onnx").exists() and (vdir / "model_decoder.onnx").exists()
    except Exception:
        return False


def _controller_cls():
    sys.path.insert(0, str(ROOT / "sonic_k1_train" / "mac"))
    from eval_k1_mujoco import K1Controller

    class K1HumanController(K1Controller):
        """K1Controller + the SMPL encoder mode: encoder_index follows the motion's encode_mode, n-frame SMPL terms."""

        def _build_registry(self):
            R = super()._build_registry()
            R["encoder_index"] = (1, lambda: np.array([float(self.current_motion.encode_mode)]))
            R["encoder_index_onehot"] = (N_ENCODERS, lambda: np.eye(N_ENCODERS)[int(self.current_motion.encode_mode)])
            k1cfg = yaml.safe_load(Path(self._cfg_path).read_text()).get("k1", {}) if getattr(self, "_cfg_path", None) else {}
            n = int(k1cfg.get("smpl_num_future_frames", 10))
            W = list(self.robot.wrist_il)
            if n != 10:
                R[f"smpl_joints_multi_future_local_nonflat_{n}f"] = (72 * n, lambda: self._smpl_joints(n, 1))
                R[f"smpl_root_ori_b_multi_future_{n}f"] = (6 * n, lambda: self._anchor_orientation(n, 1))
                R[f"joint_pos_multi_future_wrist_for_smpl_{n}f"] = (len(W) * n, lambda: self._motion_joint_pos(n, 1, W))
            self.smpl_frames = n
            return R

    return K1HumanController


# ------------------------------------------------------------------------------------------------ backend
class SonicHumanBackend(SonicBackend):
    name = "sonic_human"
    supports = {"arms": True, "waist_yaw": True, "bow": True, "walk": False, "human": True}

    @staticmethod
    def available(variant_dir=None) -> bool:
        return variant_ok(variant_dir or DEFAULT_VARIANT)

    def __init__(self, variant_dir=None, smpl_future: str = "delay", threads: int = 2, band_s: float = 1.0,
                 settle_s: float = 0.5, hold_s: float = 0.5, fade_in_s: float = 0.8, fade_out_s: float = 1.5):
        assert smpl_future in ("delay", "hold")
        from sonic.motion import standing_motion
        from sonic.mp2smpl import PresenceGate
        from sonic.robots import get_robot
        from sonic.sim import G1Sim

        from k1live.backends.base import arm_indices

        variant_dir = Path(variant_dir or DEFAULT_VARIANT)
        if not variant_ok(variant_dir):
            raise FileNotFoundError(f"{variant_dir}: no K1 SONIC checkpoint with an SMPL encoder mode")
        # the SonicBackend attributes its standing path / reset() use (no planner, no walking)
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
        self.variant_dir = variant_dir
        self.ctrl = _controller_cls().load(variant_dir, self.robot)
        self.n_smpl = self.ctrl.smpl_frames
        self._standing = standing_motion(50, robot=self.robot)
        self.kp, self.walk = None, False
        self.arm_future, self.n_future = "extrapolate", int(yaml.safe_load(
            (variant_dir / "observation_config.yaml").read_text()).get("k1", {}).get("num_future_frames", 4))
        self.waist_max, self.bow_max = 0.6, 0.35
        self.band_s, self.settle_s = band_s, settle_s
        self.last_window = None
        self.smpl_future = smpl_future
        idle = smpl_stream.make_converter(live=False).idle_frame()
        self.idle = {"smpl_joints": smpl_stream.zero_pelvis(idle["smpl_joints"]), "body_quat": idle["body_quat"]}
        self._gate_kw = dict(hold_s=hold_s, fade_out_s=fade_out_s, fade_in_s=fade_in_s)
        self._PresenceGate = PresenceGate
        self.wrist_mj = [POLICY_JOINTS.index(f"{s}_wrist_roll_joint") for s in ("left", "right")]
        self.reset()

    # ------------------------------------------------------------------------------------------- lifecycle
    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.mode = "robot"
        self.gate = self._PresenceGate(self.idle, **self._gate_kw)
        self.cur = self.idle
        self.buf = deque(maxlen=self.n_smpl + 2)
        self.wrist = self.robot.default_angles[self.wrist_mj].copy()
        self.n_human_ticks = 0
        super().reset(x, y, yaw)

    # ------------------------------------------------------------------------------------------- reference
    def _gate_tick(self, live: Optional[dict]):
        """Presence gate + smoothing -> the SMPL frame of this tick (dict smpl_joints / body_quat)."""
        from sonic.mp2smpl import blend_frames

        g = self.gate(self.sim_time, live)
        a = float(np.clip(self.dt / SMOOTH_TAU, 0.0, 1.0))
        self.cur = blend_frames(self.cur, g, a)
        return self.cur

    def _gate_settled_idle(self) -> bool:
        g = self.gate
        done = g._t_switch is None or self.sim_time - g._t_switch >= g._dur
        return g.state == g.IDLE and done

    def smpl_motion(self):
        """MotionSequence (frame 0 duplicated, current_frame = 1) of the encoder's n SMPL frames from the buffer."""
        from sonic.motion import MotionSequence

        n = self.n_smpl
        b = list(self.buf)
        if self.smpl_future == "hold" or len(b) < n:
            frames = [b[-1]] * n
        else:
            frames = b[-n:]
        frames = [frames[0]] + frames
        J = np.stack([np.asarray(f["smpl_joints"], np.float64) for f in frames])
        q = np.stack([np.asarray(f["body_quat"], np.float64) for f in frames])
        for k in range(1, len(q)):
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] = -q[k]
        H = len(frames)
        qd = np.tile(self.robot.default_angles, (H, 1))
        qd[:, self.wrist_mj] = self.wrist
        jp = qd[:, self.robot.mujoco_to_isaaclab]
        return MotionSequence("streamed_smpl", jp, np.zeros_like(jp), np.zeros((H, 1, 3)), q[:, None], J, [0],
                              encode_mode=SMPL_MODE)

    def step(self, arm_q: Sequence[float], waist_yaw: Optional[float] = None, bow: Optional[float] = None,
             loco_cmd: Sequence[float] = (0.0, 0.0, 0.0), arm_dq: Optional[Sequence[float]] = None,
             human: Optional[dict] = None) -> dict:
        live = None
        if human is not None and human.get("smpl") is not None:
            try:
                J, q, w = smpl_stream.from_msg(human["smpl"])
                live = {"smpl_joints": J, "body_quat": q}
                if w is not None:
                    self.wrist = np.clip(w, self.q_lo[self.wrist_mj], self.q_hi[self.wrist_mj])
            except (ValueError, KeyError, TypeError):
                live = None
        if live is not None and self.mode == "robot":
            self.mode = "human"
            self.gate.reset()
            self.cur = self.idle
            self.buf.clear()
            self.ctrl.reinitialize_heading = True     # the person's heading now = the robot's heading now
        if self.mode == "human":
            fr = self._gate_tick(live)
            self.buf.append(fr)
            if live is None and self._gate_settled_idle():
                self.mode = "robot"
                self.ctrl.reinitialize_heading = True
        if self.mode == "robot":
            info = super().step(arm_q, waist_yaw=waist_yaw, bow=bow, loco_cmd=(0.0, 0.0, 0.0), arm_dq=arm_dq)
            info["mode"] = "robot"
            return info
        self.n_human_ticks += 1
        ctrl = self.ctrl
        ctrl.current_motion = self.smpl_motion()
        ctrl.current_frame, ctrl.play = 1, True
        q_t, ms = self._physics_tick()
        return {"policy_ms": ms, "planner_ms": 0.0, "target": q_t, "mode": "human", "gate": self.gate.state}
