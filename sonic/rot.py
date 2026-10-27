# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Quaternion helpers (wxyz, scalar first) matching include/math_utils.hpp (the *_d versions).

All functions accept (..., 4) / (..., 3) numpy arrays.
"""
import numpy as np


def quat_mul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q, v):
    """math_utils.hpp quat_rotate_d (no normalisation; zero quat -> zero vector)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qv, v) * qw * 2.0
    c = qv * np.sum(qv * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def quat_from_angle_axis(angle, axis):
    angle = np.asarray(angle, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    half = angle[..., None] * 0.5
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def calc_heading(q):
    rot_dir = quat_rotate(q, np.broadcast_to([1.0, 0.0, 0.0], np.shape(q)[:-1] + (3,)))
    return np.arctan2(rot_dir[..., 1], rot_dir[..., 0])


def calc_heading_quat(q):
    h = calc_heading(q)
    return quat_from_angle_axis(h, np.broadcast_to([0.0, 0.0, 1.0], np.shape(h) + (3,)))


def calc_heading_quat_inv(q):
    h = calc_heading(q)
    return quat_from_angle_axis(-h, np.broadcast_to([0.0, 0.0, 1.0], np.shape(h) + (3,)))


def euler_z_to_quat(angle):
    return quat_from_angle_axis(np.asarray(angle), [0.0, 0.0, 1.0])


def quat_to_rotmat(q):
    """math_utils.hpp quat_to_rotation_matrix_d (normalises first)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot6d_rowwise(q):
    """First two columns of R, flattened row-wise: [R00,R01,R10,R11,R20,R21]."""
    R = quat_to_rotmat(q)
    return R[..., :, :2].reshape(q.shape[:-1] + (6,))


def aa_to_quat(aa):
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1)
    small = angle < 1e-12
    axis = np.where(small[..., None], np.array([1.0, 0.0, 0.0]), aa / np.maximum(angle, 1e-12)[..., None])
    return quat_from_angle_axis(angle, axis)


def quat_to_aa(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., 0:1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    axis = q[..., 1:] / np.maximum(s, 1e-12)[..., None]
    return axis * angle[..., None]


def rotmat_to_quat(R):
    """Robust (Shepperd) rotation-matrix -> wxyz quaternion."""
    R = np.asarray(R, dtype=np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((R.shape[0], 4))
    for i, m in enumerate(R):
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.reshape(shp + (4,))


def quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ---- SMPL helpers (gear_sonic/isaac_utils/rotations.py) ----------------------
_SMPL_BASE = np.array([0.5, 0.5, 0.5, 0.5])
_YTOZ = aa_to_quat(np.array([np.pi / 2, 0.0, 0.0]))


def smpl_root_ytoz_up(q_yup):
    """rotations.smpl_root_ytoz_up: q_z = R_x(+90deg) * q_y."""
    return quat_mul(np.broadcast_to(_YTOZ, np.shape(q_yup)), q_yup)


def remove_smpl_base_rot(q):
    """rotations.remove_smpl_base_rot (w_last=False): q * conj([.5,.5,.5,.5])."""
    return quat_mul(q, np.broadcast_to(quat_conj(_SMPL_BASE), np.shape(q)))


YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
