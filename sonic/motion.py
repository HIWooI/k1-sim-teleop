# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Reference motions (MotionSequence equivalent of motion_data_reader.hpp).

All arrays at 50 Hz:
  joint_pos / joint_vel : (T, 29) IsaacLab order, absolute radians
  body_pos              : (T, B, 3) world
  body_quat             : (T, B, 4) world, wxyz; body 0 == root/pelvis
  smpl_joints           : (T, 24, 3) root-local Z-up SMPL joints (already canonicalised,
                          exactly what the deploy receives over ZMQ protocol v2/v3)
  encode_mode           : 0 = g1 (robot joints), 2 = smpl
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import rot

FPS = 50.0


@dataclass
class MotionSequence:
    name: str
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray
    smpl_joints: np.ndarray | None = None
    body_part_indexes: list = field(default_factory=lambda: [0])
    encode_mode: int = 0

    @property
    def timesteps(self) -> int:
        return int(self.joint_pos.shape[0])


def _read_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)


def load_csv_motion(motion_dir: str | Path) -> MotionSequence:
    """Load one gear_sonic_deploy/reference/<set>/<motion>/ folder."""
    d = Path(motion_dir)
    jp = _read_csv(d / "joint_pos.csv")
    jv = _read_csv(d / "joint_vel.csv")
    bp = _read_csv(d / "body_pos.csv")
    bq = _read_csv(d / "body_quat.csv")
    T = jp.shape[0]
    bp = bp.reshape(T, -1, 3)
    bq = bq.reshape(T, -1, 4)
    idx = [0]
    meta = d / "metadata.txt"
    if meta.exists():
        txt = meta.read_text()
        m = re.search(r"Body part indexes:\s*\[([^\]]*)\]", txt)
        if m:
            idx = [int(x) for x in m.group(1).replace(",", " ").split()]
    smpl = None
    if (d / "smpl_joint.csv").exists():
        smpl = _read_csv(d / "smpl_joint.csv").reshape(T, -1, 3)
    return MotionSequence(d.name, jp, jv, bp, bq, smpl, idx, encode_mode=0)


def list_csv_motions(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "joint_pos.csv").exists())


def smpl_motion_from_arrays(name, root_quat_nobase, smpl_joints_local, wrist_pos=None, num_dof=29,
                            wrist_idx=(23, 24, 25, 26, 27, 28)):
    """Build an encode_mode=2 motion from per-frame SMPL stream fields (protocol v3 view).

    root_quat_nobase : (T,4) wxyz Z-up root orientation, SMPL base rotation removed
    smpl_joints_local: (T,24,3) joints in that root frame
    wrist_pos        : (T,len(wrist_idx)) wrist joints (G1: IsaacLab 23..28); zeros like the GEM-X path
    num_dof/wrist_idx: robot (RobotSpec.num_dof / wrist_il), default G1
    """
    T = smpl_joints_local.shape[0]
    jp = np.zeros((T, num_dof))
    if wrist_pos is not None:
        jp[:, list(wrist_idx)] = wrist_pos
    return MotionSequence(
        name=name,
        joint_pos=jp,
        joint_vel=np.zeros((T, num_dof)),
        body_pos=np.zeros((T, 1, 3)),
        body_quat=np.asarray(root_quat_nobase, dtype=np.float64).reshape(T, 1, 4),
        smpl_joints=np.asarray(smpl_joints_local, dtype=np.float64),
        body_part_indexes=[0],
        encode_mode=2,
    )


def load_smpl_pkl(path: str | Path, smpl_y_up: bool = True, pelvis_mode: str = "j0") -> MotionSequence:
    """Load a SONIC `smpl_filtered` pkl (pose_aa (T,72) Y-up root, smpl_joints (T,24,3) Z-up).

    Mirrors the training observation (commands.py smpl_root_quat_w + observations.py
    smpl_joints_multi_future_local): root = remove_base(ytoz(aa2quat(pose_aa[:, :3]))),
    joints_local = quat_apply(inv(root), smpl_joints). This is also exactly what
    pico_manager_thread_server.process_smpl_joints streams.
    """
    import joblib

    d = joblib.load(path)
    if "pose_aa" not in d:  # dict keyed by motion name
        d = next(iter(d.values()))
    fps = float(d.get("fps", 50.0))
    pose_aa = np.asarray(d["pose_aa"], dtype=np.float64)
    joints = np.asarray(d["smpl_joints"], dtype=np.float64)
    if abs(fps - FPS) > 1e-6:
        T0 = pose_aa.shape[0]
        t_src = np.arange(T0) / fps
        t_dst = np.arange(int(np.floor(t_src[-1] * FPS)) + 1) / FPS
        joints = _interp(t_src, joints, t_dst)
        q = rot.aa_to_quat(pose_aa[:, :3])
        qi = _slerp_series(t_src, q, t_dst)
        pose_aa = np.zeros((len(t_dst), 72))
        pose_aa[:, :3] = rot.quat_to_aa(qi)
    q = rot.aa_to_quat(pose_aa[:, :3])
    if smpl_y_up:
        q = rot.smpl_root_ytoz_up(q)
    q = rot.remove_smpl_base_rot(q)
    if pelvis_mode == "zero":  # GEM-X soma_to_smpl.py convention (root-relative joints)
        joints = joints - joints[:, :1]
    inv = rot.quat_conj(q)[:, None, :].repeat(joints.shape[1], axis=1)
    local = rot.quat_rotate(inv, joints)
    return smpl_motion_from_arrays(Path(path).stem, q, local)


def _interp(t_src, x, t_dst):
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_dst),) + x.shape[1:])


def _slerp_series(t_src, q, t_dst):
    out = np.empty((len(t_dst), 4))
    for i, t in enumerate(t_dst):
        j = int(np.clip(np.searchsorted(t_src, t) - 1, 0, len(t_src) - 2))
        a = (t - t_src[j]) / (t_src[j + 1] - t_src[j])
        out[i] = rot.quat_slerp(q[j], q[j + 1], float(np.clip(a, 0, 1)))
    return out


def standing_motion(n_frames: int = 50, robot=None) -> MotionSequence:
    """Static default-pose reference (pelvis yaw 0). robot: sonic.robots.RobotSpec (default G1)."""
    from .robots import g1

    robot = robot or g1()
    jp = np.tile(robot.default_angles[robot.mujoco_to_isaaclab], (n_frames, 1))
    bq = np.tile([1.0, 0.0, 0.0, 0.0], (n_frames, 1, 1))
    bp = np.tile([0.0, 0.0, robot.standing_height], (n_frames, 1, 1))
    return MotionSequence("standing", jp, np.zeros_like(jp), bp, bq, None, [0], 0)


def load_robot_pkl(path: str | Path, robot=None) -> MotionSequence:
    """Load a SONIC `robot_filtered` pkl (dof MuJoCo order, pose_aa[:,0] = root aa, 30 fps).
    robot: sonic.robots.RobotSpec whose MuJoCo order the pkl uses (default G1)."""
    import joblib

    from .robots import g1

    MUJOCO_TO_ISAACLAB = (robot or g1()).mujoco_to_isaaclab

    d = joblib.load(path)
    name = Path(path).stem
    if "dof" not in d:
        name, d = next(iter(d.items()))
    fps = float(d["fps"])
    dof = np.asarray(d["dof"], dtype=np.float64)
    q = rot.aa_to_quat(np.asarray(d["pose_aa"], dtype=np.float64)[:, 0])
    pos = np.asarray(d["root_trans_offset"], dtype=np.float64)
    T0 = dof.shape[0]
    t_src = np.arange(T0) / fps
    t_dst = np.arange(int(np.floor(t_src[-1] * FPS)) + 1) / FPS
    dof = _interp(t_src, dof, t_dst)
    pos = _interp(t_src, pos, t_dst)
    q = _slerp_series(t_src, q, t_dst)
    jp = dof[:, MUJOCO_TO_ISAACLAB]
    jv = np.gradient(jp, 1.0 / FPS, axis=0)
    return MotionSequence(name, jp, jv, pos[:, None, :], q[:, None, :], None, [0], 0)
