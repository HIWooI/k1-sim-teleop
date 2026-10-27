"""Error statistics of an SMPL stream against a reference, shared by the GEM-X fit (scripts/gemx_aug_stats.py) and the
augmenter's unit tests (tests/test_smpl_gemx_aug.py): the same estimator measures the live GEM-X stream and the
training-time corruption (overlay .../mdp/smpl_gemx_aug.py), so "the augmentation matches GEM-X" is one comparison.

Decomposition (per frame, live vs reference at the same capture time; root-local joints, pelvis at 0):
  dR       = R_ref^-1 R_live                      root rotation residual as a rotation vector (roll, pitch, yaw)
  aligned  = dR J_live                            the live body in the reference root frame (pelvis wobble removed)
  bone b   = rotation vector taking the reference bone direction to the aligned live one (across the bone)
Per clip: bias = the clip mean, noise = the rest; noise ACF -> slow / fast OU parts (two exponentials).
Dropouts: maximal runs without a delivered pose (between the first and the last delivered frame).
Numpy only.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
BONE_GROUPS = {
    "hip": [1, 2], "thigh": [4, 5], "shin": [7, 8], "foot": [10, 11], "spine": [3, 6, 9], "neck_head": [12, 15],
    "collar": [13, 14], "upper_arm": [16, 17, 18, 19], "forearm": [20, 21], "hand": [22, 23],
}
JOINT_REPORT = {"wrist": [20, 21], "elbow": [18, 19], "ankle": [7, 8], "knee": [4, 5], "head": [15]}


# ------------------------------------------------------------------------------------------------ rotations (wxyz)
def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def qconj(q):
    return np.concatenate((q[..., :1], -q[..., 1:]), -1)


def qapply(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    xyz = np.broadcast_to(xyz, v.shape)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def q_to_rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., :1])
    return q[..., 1:] * np.where(s > 1e-9, ang / np.maximum(s, 1e-9), 2.0)


def tilt_deg(q):
    q = np.asarray(q, float)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1, 1)))


def yaw_of(q):
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ------------------------------------------------------------------------------------------------ residuals
def bone_dirs(J):
    v = J[..., 1:, :] - J[..., PARENTS[1:], :]
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def bone_rotvec(d_ref, d_live):
    """Rotation vectors (across the bone) taking unit directions d_ref to d_live."""
    c = np.cross(d_ref, d_live)
    s = np.linalg.norm(c, axis=-1, keepdims=True)
    ang = np.arctan2(s, np.sum(d_ref * d_live, -1, keepdims=True))
    return c / np.maximum(s, 1e-12) * ang


def residual(J_live, q_live, J_ref, q_ref, remove_yaw=True):
    """(T,24,3)/(T,4) live and reference (root-local, wxyz) -> dict of per-frame residuals.
    remove_yaw: the clip's mean heading offset is not an error (the backend re-anchors the heading per stream)."""
    J_live = np.asarray(J_live, float) - np.asarray(J_live, float)[:, :1]
    J_ref = np.asarray(J_ref, float) - np.asarray(J_ref, float)[:, :1]
    q_live, q_ref = np.asarray(q_live, float), np.asarray(q_ref, float)
    if remove_yaw:
        dy = yaw_of(q_live) - yaw_of(q_ref)
        m = math.atan2(np.mean(np.sin(dy)), np.mean(np.cos(dy)))
        qz = np.array([math.cos(-m / 2), 0.0, 0.0, math.sin(-m / 2)])
        q_live = qmul(np.broadcast_to(qz, q_live.shape), q_live)
    dq = qmul(qconj(q_ref), q_live)
    aligned = qapply(dq[:, None, :], J_live)
    return {"root_rv": q_to_rotvec(dq), "bone_rv": bone_rotvec(bone_dirs(J_ref), bone_dirs(aligned)),
            "joint_cm": 100.0 * np.linalg.norm(J_live - J_ref, axis=-1),
            "tilt_live": tilt_deg(q_live), "tilt_ref": tilt_deg(q_ref)}


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of x (T, ...) (mean removed, pooled over the trailing dims), lags 0..max_lag."""
    x = np.asarray(x, float).reshape(len(x), -1)
    x = x - x.mean(0)
    v = np.sum(x * x)
    out = np.ones(max_lag + 1)
    if v <= 0:
        return out
    for k in range(1, max_lag + 1):
        out[k] = np.sum(x[k:] * x[:-k]) / v if k < len(x) else np.nan
    return out


def fit_two_exp(lag_s: np.ndarray, r: np.ndarray, w: Optional[np.ndarray] = None):
    """ACF ~ (1 - f) exp(-l / tau_s) + f exp(-l / tau_f) (grid search). -> tau_s, tau_f, f (fast variance share)."""
    ok = np.isfinite(r) & (lag_s > 0)
    lag_s, r = lag_s[ok], r[ok]
    w = np.ones_like(r) if w is None else np.asarray(w, float)[ok]
    best = (np.inf, 1.0, 0.05, 0.0)
    for ts in np.geomspace(0.08, 8.0, 60):
        for tf in np.geomspace(0.01, 0.12, 12):
            if tf >= ts:
                continue
            a, b = np.exp(-lag_s / ts), np.exp(-lag_s / tf)
            # r = a + f (b - a): least squares in f, clipped to [0, 1]
            d = b - a
            f = float(np.clip(np.sum(w * d * (r - a)) / max(np.sum(w * d * d), 1e-12), 0.0, 1.0))
            e = float(np.sum(w * (a + f * d - r) ** 2))
            if e < best[0]:
                best = (e, ts, tf, f)
    return best[1], best[2], best[3]


def gaps(delivered: np.ndarray, dt: float) -> List[float]:
    """Lengths (s) of the runs without a delivered pose between the first and the last delivered tick."""
    d = np.asarray(delivered, bool)
    idx = np.flatnonzero(d)
    if len(idx) < 2:
        return []
    d = d[idx[0]:idx[-1] + 1]
    out, run = [], 0
    for v in d:
        if not v:
            run += 1
        elif run:
            out.append(run * dt)
            run = 0
    return out


# ------------------------------------------------------------------------------------------------ summaries
def _rms_angle_deg(rv):
    return float(np.degrees(np.sqrt(np.mean(np.sum(rv * rv, -1))))) if len(rv) else float("nan")


def summarize(clips: Sequence[dict], frame_dt: float, max_lag_s: float = 2.0) -> dict:
    """clips: [{"res": residual(...), "dt": s between frames (optional)}] of one profile -> fitted model numbers.

    bone_bias_deg / bone_noise_deg: RMS angle across the bone of the clip mean / of the rest, per group;
    tau_slow / tau_fast / fast_frac: two-exponential fit of the pooled bone-noise ACF (and root separately);
    root_bias_deg / root_noise_deg: per axis (roll, pitch, yaw) RMS; joint_cm: mean joint error per reported joint."""
    out = {"frames": int(sum(len(c["res"]["root_rv"]) for c in clips)), "clips": len(clips)}
    bias_b, noise_b, bias_r, noise_r = [], [], [], []
    acf_b, acf_r, wts = [], [], []
    for c in clips:
        res = c["res"]
        dt = float(c.get("dt", frame_dt))
        nl = int(round(max_lag_s / dt))
        bm = res["bone_rv"].mean(0)
        rm = res["root_rv"].mean(0)
        bias_b.append(bm)
        bias_r.append(rm)
        nb = res["bone_rv"] - bm
        nr = res["root_rv"] - rm
        noise_b.append(nb)
        noise_r.append(nr)
        lags = np.arange(nl + 1) * dt
        grid = np.arange(0, max_lag_s + 1e-9, frame_dt)
        acf_b.append(np.interp(grid, lags, acf(nb, nl), right=np.nan))
        acf_r.append(np.interp(grid, lags, acf(nr, nl), right=np.nan))
        wts.append(len(nb))
    bias_b, bias_r = np.stack(bias_b), np.stack(bias_r)          # (C, 23, 3), (C, 3)
    nb_all, nr_all = np.concatenate(noise_b), np.concatenate(noise_r)
    out["bone_bias_deg"], out["bone_noise_deg"] = {}, {}
    for g, js in BONE_GROUPS.items():
        b = [j - 1 for j in js]
        out["bone_bias_deg"][g] = round(_rms_angle_deg(bias_b[:, b].reshape(-1, 3)), 2)
        out["bone_noise_deg"][g] = round(_rms_angle_deg(nb_all[:, b].reshape(-1, 3)), 2)
    out["root_bias_deg"] = [round(float(np.degrees(np.sqrt(np.mean(bias_r[:, k] ** 2)))), 2) for k in range(3)]
    out["root_noise_deg"] = [round(float(np.degrees(np.std(nr_all[:, k]))), 2) for k in range(3)]
    grid = np.arange(0, max_lag_s + 1e-9, frame_dt)
    w = np.asarray(wts, float)
    for name, A in (("bone", np.stack(acf_b)), ("root", np.stack(acf_r))):
        m = np.isfinite(A)
        r = np.where(m, A, 0).T @ w / np.maximum(m.T @ w, 1e-9)
        r[~m.any(0)] = np.nan
        ts, tf, f = fit_two_exp(grid, r)
        out[f"{name}_tau_slow"], out[f"{name}_tau_fast"], out[f"{name}_fast_frac"] = round(ts, 3), round(tf, 3), round(f, 3)
        out[f"{name}_acf"] = {f"{l:.2f}": round(float(v), 3) for l, v in zip(grid, r)
                              if np.isfinite(v) and (abs(l / 0.1 - round(l / 0.1)) < 1e-6)}
    jc = np.concatenate([c["res"]["joint_cm"] for c in clips])
    out["joint_cm"] = {k: round(float(jc[:, v].mean()), 2) for k, v in JOINT_REPORT.items()}
    out["joint_cm"]["all"] = round(float(jc.mean()), 2)
    tl = np.concatenate([c["res"]["tilt_live"] for c in clips])
    tr = np.concatenate([c["res"]["tilt_ref"] for c in clips])
    out["tilt_live_p50_p95"] = [round(float(np.percentile(tl, 50)), 1), round(float(np.percentile(tl, 95)), 1)]
    out["tilt_ref_p50_p95"] = [round(float(np.percentile(tr, 50)), 1), round(float(np.percentile(tr, 95)), 1)]
    return out


def split_slow_fast(noise_deg: float, fast_frac: float):
    """RMS noise + fast variance share -> (slow RMS, fast RMS)."""
    return math.sqrt(max(0.0, 1 - fast_frac)) * noise_deg, math.sqrt(max(0.0, fast_frac)) * noise_deg


def gap_summary(gap_lists: Iterable[List[float]], tracked_s: float, hold_s: float = 0.5) -> dict:
    """Pooled dropout lengths of one profile -> rate and a log-normal fit of the lengths up to hold_s."""
    g = np.array([x for gl in gap_lists for x in gl], float)
    out = {"n": int(len(g)), "tracked_s": round(float(tracked_s), 1),
           "per_min": round(60.0 * len(g) / max(tracked_s, 1e-6), 2)}
    if len(g):
        out["len_s_p25_p50_p90"] = [round(float(np.percentile(g, p)), 3) for p in (25, 50, 90)]
        out["over_hold_frac"] = round(float(np.mean(g > hold_s)), 3)
        short = g[g <= hold_s]
        out["per_min_le_hold"] = round(60.0 * len(short) / max(tracked_s, 1e-6), 2)
        if len(short) >= 2:
            lg = np.log(short)
            out["logn_mu"], out["logn_sd"] = round(float(lg.mean()), 3), round(float(max(lg.std(), 0.1)), 3)
    return out
