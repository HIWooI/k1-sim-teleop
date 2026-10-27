"""Whole-body human pose stream for the sonic_human backend: ONE definition shared by training data and the live path.

The K1 SONIC SMPL encoder is trained on BONES-SEED motion turned into exactly what the live sender produces from a
webcam (sonic_k1_train/data/human_smpl.py): MediaPipe-style 33 landmarks -> sonic/mp2smpl.MediaPipeToSmpl (bone
directions re-laid on the canonical SMPL rest bone lengths). Everything both sides must agree on lives here:

* converter settings (make_converter): root frame from the hip line, root pitch share 0.5, tilt cap 40 deg.
  The live-only corrections (arm depth scale, forward-reach cap, leg lock, glitch guard) are off for training data.
* pelvis convention: K1 is trained with the pelvis at the origin ("zero"). mp2smpl's default "j0" (the released
  G1 SONIC convention: the rest pelvis offset J0 added in the world frame) differs from it by one constant world
  vector, so zero_pelvis() turns either into the other's local joints exactly.
* mirror (protocol.py: robot = the person's reflection): reflect y in the root frame, swap left/right joints.
* message format of protocol.HumanFrame.smpl: {"joints": 24x3 root-local, "quat": [w,x,y,z], "wrist": [l, r]}.

Frames: root-local = x forward, y left, z up (robot axes); quat = root orientation in the Z-up world with the SMPL
base rotation removed (identity = upright, facing +x). SMPL-24 joint order = sonic.smpl.SMPL_24_NAMES.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sonic import rot

# left/right joint pairs of SMPL-24 (sonic.smpl.SMPL_24_NAMES)
LR_PAIRS = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
MIRROR_PERM = np.arange(24)
for _a, _b in LR_PAIRS:
    MIRROR_PERM[_a], MIRROR_PERM[_b] = _b, _a
REFLECT_Y = np.array([1.0, -1.0, 1.0])

# MediaPipeToSmpl settings shared by the training data and the live sender (the geometry definition)
SHARED_KW = dict(root_pitch_share=0.5, pelvis_mode="zero", max_root_tilt_deg=40.0)
# live-only extras (MediaPipe artefacts; never applied to mocap): see sonic/mp2smpl.py for each
LIVE_KW = dict(legs="auto", leg_blend_s=0.3, vis_thresh=0.5)


def make_converter(live: bool = False, **over):
    """MediaPipeToSmpl with the shared settings. live=False (training data): legs tracked, no glitch guard / root
    rate limit (clean mocap, any turning speed)."""
    from sonic.mp2smpl import MediaPipeToSmpl

    kw = dict(SHARED_KW, **(LIVE_KW if live else {"legs": "track"}))
    kw.update(over)
    conv = MediaPipeToSmpl(**kw)
    if not live:
        conv.max_jump_rad = np.inf
        conv.max_root_step = np.inf
    return conv


def zero_pelvis(joints_local: np.ndarray) -> np.ndarray:
    """Root-local joints with the pelvis moved to the origin (works for both pelvis conventions)."""
    j = np.asarray(joints_local, np.float64)
    return j - j[..., :1, :]


def mirror(joints_local: np.ndarray, quat: np.ndarray, wrist: Optional[np.ndarray] = None):
    """Reflection of a pose across the person's sagittal plane (x-z plane of the root frame)."""
    j = np.asarray(joints_local, np.float64)[..., MIRROR_PERM, :] * REFLECT_Y
    q = np.asarray(quat, np.float64) * np.array([1.0, -1.0, 1.0, -1.0])  # (w,x,y,z) -> (w,-x,y,-z)
    w = None if wrist is None else -np.asarray(wrist, np.float64)[..., ::-1]
    return j, q, w


def world_joints(joints_local: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Root-local joints rotated into the (heading-carrying) world frame: what motion-lib smpl_joints store."""
    q = np.asarray(quat, np.float64)
    j = np.asarray(joints_local, np.float64)
    return rot.quat_rotate(np.broadcast_to(q[..., None, :], j.shape[:-1] + (4,)), j)


def to_msg(joints_local, quat, wrist=None, nd: int = 4) -> dict:
    d = {"joints": np.round(np.asarray(joints_local, np.float64), nd).tolist(),
         "quat": np.round(np.asarray(quat, np.float64), 6).tolist()}
    if wrist is not None:
        d["wrist"] = np.round(np.asarray(wrist, np.float64), nd).tolist()
    return d


def from_msg(d: dict):
    """-> (joints_local (24,3) pelvis at 0, quat (4,) unit, wrist (2,) or None). Raises ValueError on a bad message."""
    j = np.asarray(d["joints"], np.float64)
    q = np.asarray(d["quat"], np.float64)
    if j.shape != (24, 3) or q.shape != (4,) or not np.all(np.isfinite(j)) or not np.all(np.isfinite(q)):
        raise ValueError("bad smpl message")
    n = np.linalg.norm(q)
    if n < 1e-6:
        raise ValueError("bad smpl quaternion")
    w = d.get("wrist")
    w = None if w is None else np.asarray(w, np.float64).reshape(2)
    return zero_pelvis(j), q / n, w


def root_aa_yup(quat_zup_nobase: np.ndarray) -> np.ndarray:
    """Inverse of gear_sonic's smpl root chain (aa -> quat -> smpl_root_ytoz_up -> remove_smpl_base_rot):
    the Y-up SMPL global_orient axis-angle whose training-time root quaternion is `quat_zup_nobase` (wxyz)."""
    q = np.asarray(quat_zup_nobase, np.float64)
    base = np.array([0.5, 0.5, 0.5, 0.5])
    rx_inv = np.array([np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0, 0.0])  # Rx(-90 deg)
    shape = q.shape
    q = q.reshape(-1, 4)
    out = rot.quat_mul(np.broadcast_to(rx_inv, q.shape), rot.quat_mul(q, np.broadcast_to(base, q.shape)))
    return rot.quat_to_aa(out).reshape(shape[:-1] + (3,))


def root_quat_from_aa_yup(aa: np.ndarray) -> np.ndarray:
    """gear_sonic's chain itself (commands.smpl_root_quat_w): Y-up global_orient aa -> Z-up root quat, base removed."""
    aa = np.asarray(aa, np.float64)
    shape = aa.shape
    q = rot.aa_to_quat(aa.reshape(-1, 3))
    q = rot.smpl_root_ytoz_up(q)
    q = rot.remove_smpl_base_rot(q)
    return q.reshape(shape[:-1] + (4,))
