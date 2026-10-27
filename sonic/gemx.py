# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""GEM-X (SOMA) result -> SONIC SMPL stream fields, CPU only, no SOMA layer needed.

Official path (gear_sonic/examples/live_camera_teleop/soma_to_smpl.py) runs the SOMA body
layer on CUDA to get the 77 rig joints. GEM-X also exports the same rig as a BVH
(`<clip>_soma.bvh`), whose Hips joint is exactly global_orient * [0.5,0.5,0.5,0.5] and whose
Hips position == transl (verified on Guap_ver2). So BVH forward kinematics reproduces the
SOMA joints, and the rest of the conversion is copied from SomaToSmpl.convert():

  joints24  = joints77[alias-matched SMPL joints]
  g_quat    = remove_smpl_base_rot(smpl_root_ytoz_up(aa2quat(global_orient)))
  local     = quat_apply(inv(g_quat), (joints24 - joints24[0]) @ YUP_TO_ZUP.T)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import rot, smpl
from .bvh import fk, read_bvh
from .motion import FPS, _interp, _slerp_series, smpl_motion_from_arrays

# copied from soma_to_smpl.py
_ALIASES = {
    "pelvis": ["hips", "pelvis", "root"],
    "left_hip": ["leftleg", "left_hip", "l_hip", "left_upleg"],
    "right_hip": ["rightleg", "right_hip", "r_hip", "right_upleg"],
    "spine1": ["spine1", "spine_1", "spine"],
    "left_knee": ["leftshin", "left_knee", "l_knee"],
    "right_knee": ["rightshin", "right_knee", "r_knee"],
    "spine2": ["spine2", "spine_2"],
    "left_ankle": ["leftfoot", "left_ankle", "l_ankle"],
    "right_ankle": ["rightfoot", "right_ankle", "r_ankle"],
    "spine3": ["chest", "spine3", "spine_3"],
    "left_foot": ["lefttoebase", "left_foot", "l_foot", "left_toe"],
    "right_foot": ["righttoebase", "right_foot", "r_foot", "right_toe"],
    "neck": ["neck1", "neck"],
    "left_collar": ["leftshoulder", "left_collar", "l_collar", "left_clavicle"],
    "right_collar": ["rightshoulder", "right_collar", "r_collar", "right_clavicle"],
    "head": ["head"],
    "left_shoulder": ["leftarm", "left_shoulder", "l_shoulder", "left_upperarm"],
    "right_shoulder": ["rightarm", "right_shoulder", "r_shoulder", "right_upperarm"],
    "left_elbow": ["leftforearm", "left_elbow", "l_elbow", "left_lowerarm"],
    "right_elbow": ["rightforearm", "right_elbow", "r_elbow"],
    "left_wrist": ["lefthand", "left_wrist", "l_wrist"],
    "right_wrist": ["righthand", "right_wrist", "r_wrist"],
    "left_hand": ["lefthandmiddle1", "left_hand", "l_hand", "left_middle1"],
    "right_hand": ["righthandmiddle1", "right_hand", "r_hand", "right_middle1"],
}


def _norm(n):
    return n.lower().replace("-", "_").replace(" ", "_")


def soma_to_smpl_index(names):
    lut = {_norm(n): i for i, n in enumerate(names)}
    out = []
    for s in smpl.SMPL_24_NAMES:
        hit = next((lut[_norm(a)] for a in _ALIASES[s] if _norm(a) in lut), None)
        if hit is None:
            raise KeyError(f"no SOMA joint for {s}")
        out.append(hit)
    return out


def load_gemx_clip(bvh_path, pt_path=None, pelvis_mode="zero", smooth=0.0, resample=True):
    """Build an encode_mode=2 MotionSequence from a GEM-X BVH (+ hpe_results.pt global_orient).

    pelvis_mode: "zero" -> official soma_to_smpl.py (root-relative joints)
                 "j0"   -> training/PICO convention (pelvis at the SMPL rest offset J[0])
    """
    b = read_bvh(bvh_path)
    pos, Rg = fk(b)
    pos = pos / 100.0  # cm -> m
    idx = soma_to_smpl_index(b.names)
    j24 = pos[:, idx]
    if pt_path is not None and Path(pt_path).exists():
        import torch

        p = torch.load(str(pt_path), weights_only=False, map_location="cpu")
        go = p["body_params_global"]["global_orient"].numpy().astype(np.float64)
        q_y = rot.aa_to_quat(go)
    else:  # Hips global rotation = global_orient * B  ->  global_orient = Hips * B^-1
        q_h = rot.rotmat_to_quat(Rg[:, b.names.index("Hips")])
        q_y = rot.quat_mul(q_h, np.broadcast_to(rot.quat_conj(np.array([0.5, 0.5, 0.5, 0.5])), q_h.shape))
    T = min(len(q_y), j24.shape[0])
    q_y, j24 = q_y[:T], j24[:T]
    g = rot.remove_smpl_base_rot(rot.smpl_root_ytoz_up(q_y))
    j0 = j24 - j24[:, :1]
    jz = j0 @ rot.YUP_TO_ZUP.T
    if pelvis_mode == "j0":
        jz = jz + smpl.rest_joints24()[0]
    local = smpl.to_local(jz, g)
    if smooth > 0:
        local, g = ema(local, g, smooth)
    if resample and abs(b.fps - FPS) > 1e-3:
        t_src = np.arange(T) / b.fps
        t_dst = np.arange(int(np.floor(t_src[-1] * FPS)) + 1) / FPS
        local = _interp(t_src, local, t_dst)
        g = _slerp_series(t_src, g, t_dst)
    return smpl_motion_from_arrays(Path(bvh_path).stem, g, local)


def ema(local, q, w):
    """SomaToSmpl temporal smoothing (EMA on joints, sign-aligned normalised lerp on quat)."""
    lo = local.copy()
    qo = q.copy()
    for t in range(1, len(lo)):
        lo[t] = w * lo[t - 1] + (1 - w) * local[t]
        qn = q[t] if np.dot(qo[t - 1], q[t]) >= 0 else -q[t]
        qq = w * qo[t - 1] + (1 - w) * qn
        qo[t] = qq / (np.linalg.norm(qq) + 1e-8)
    return lo, qo
