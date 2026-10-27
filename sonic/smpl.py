# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""SMPL(-X) skeleton helpers: numpy port of gear_sonic.trl.utils.torch_transform.compute_human_joints.

The SONIC smpl encoder is trained on joints produced by compute_human_joints with a *fixed*
neutral rest skeleton (human_joints_info.pkl: J (55,3), parents) and zero hand pose:
  joints[0]   = J[0]  (pelvis stays at the rest offset, NOT at the origin)
  joints[1:]  = FK(global_orient (Z-up), body_pose[:63])
  output      = SMPL-X joints [0..21] + [39 (left_thumb3), 54 (right_thumb3)]  -> 24 joints
The streamed / observed quantity is then quat_apply(inv(root_nobase), joints) with
root_nobase = remove_smpl_base_rot(smpl_root_ytoz_up(aa2quat(global_orient_yup))).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import rot
from .params import HUMAN_JOINTS_INFO, MODELS

SMPL_24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]
OUT_IDX = list(range(22)) + [39, 54]


@lru_cache(maxsize=1)
def human_joints_info():
    cache = MODELS / "human_joints_info.npz"
    if cache.exists():
        z = np.load(cache)
        return z["J"], z["parents"]
    import torch

    hj = torch.load(str(HUMAN_JOINTS_INFO), weights_only=False)
    J = hj["J"].numpy().astype(np.float64)
    parents = np.array(hj["parents_list"], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, J=J, parents=parents)
    return J, parents


def rest_joints24():
    J, _ = human_joints_info()
    return J[OUT_IDX]


def parents24():
    """Parent of each of the 24 output joints (thumb tips hang off the wrists, as in FK)."""
    _, p = human_joints_info()
    par = list(p[:22])
    par += [20, 21]  # left_thumb3 -> left_wrist chain, right_thumb3 -> right_wrist chain
    return par


def aa_to_rotmat(aa):
    aa = np.asarray(aa, dtype=np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s = np.sin(th)[..., None]
    c = np.cos(th)[..., None]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + s * K + (1 - c) * (K @ K)


def compute_human_joints(body_pose63, global_orient):
    """(T,63) body pose aa + (T,3) global orient aa -> (T,24,3), same as the torch version."""
    J, parents = human_joints_info()
    T = body_pose63.shape[0]
    full = np.zeros((T, 55, 3))
    full[:, 0] = global_orient
    full[:, 1:22] = body_pose63.reshape(T, 21, 3)
    R = aa_to_rotmat(full)  # (T,55,3,3)
    rel = J.copy()
    rel[1:] -= J[parents[1:]]
    G_R = np.zeros((T, 55, 3, 3))
    G_t = np.zeros((T, 55, 3))
    G_R[:, 0] = R[:, 0]
    G_t[:, 0] = rel[0]
    for i in range(1, 55):
        p = parents[i]
        G_R[:, i] = G_R[:, p] @ R[:, i]
        G_t[:, i] = G_t[:, p] + np.einsum("tij,j->ti", G_R[:, p], rel[i])
    return G_t[:, OUT_IDX]


def root_nobase_from_yup_aa(global_orient_yup):
    q = rot.aa_to_quat(global_orient_yup)
    return rot.remove_smpl_base_rot(rot.smpl_root_ytoz_up(q))


def to_local(joints_zup, root_nobase):
    inv = rot.quat_conj(root_nobase)[..., None, :].repeat(joints_zup.shape[-2], axis=-2)
    return rot.quat_rotate(inv, joints_zup)
