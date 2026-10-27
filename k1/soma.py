"""GEM-X (SOMA) results -> K1 inputs for k1/run_video.py --source soma_csv | soma_ik.

Two ways to turn one GEM-X run into a K1 reference (both CPU, no SOMA body layer needed on the Mac):

  soma_csv  soma-retargeter K1 CSV (run_soma_retarget.py export, the old K1 path):
            Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z} [deg, scipy Euler 'xyz', Z up], <joint>_dof [deg]
            -> all 23 joints in policy order (rad) + root/torso orientation
  soma_ik   SOMA 77 rig joints (world, m) -> the 14 points the CPU position IK was written for (its TARGET_NAMES are
            SOMA joint names) -> k1/cpu_kinematics.py, exactly the MediaPipe path with a true-3D source

Clip folder contract ($K1_SOMA_CLIPS/<clip>/, default videos/soma_event_clips): hpe_results.pt, soma.bvh,
soma_joints.npz (joints [T,77,3] m, names, fps, frame, kp2d_vitpose, ...), meta.json, DONE (= GEM-X part ready);
k1.csv later, complete when K1CSV_DONE exists. The old Guap result
($K1_GUAP_DIR, default videos/gemx_guap: Guap_ver2_soma.bvh, Guap_ver2_retarget_k1_from_bvh.csv) loads the same way
(SOMA joints from BVH forward kinematics, which equals the SOMA layer output; sonic/gemx.py).
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from k1.cpu_kinematics import TARGET_NAMES

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path(os.environ.get("K1_SOMA_CLIPS", str(ROOT / "videos" / "soma_event_clips")))
GUAP = Path(os.environ.get("K1_GUAP_DIR", str(ROOT / "videos" / "gemx_guap")))


@dataclass
class SomaClip:
    name: str
    joints: np.ndarray       # (T, 77, 3) m, world frame of the source (GEM-X global: Y up)
    names: list
    fps: float               # SOMA / CSV frame rate
    csv_q: np.ndarray        # (Tc, 23) rad, policy order (soma-retargeter); None until k1.csv has arrived
    csv_root_quat: np.ndarray  # (Tc, 4) xyzw, K1 pelvis in the CSV world (Z up)
    csv_root_pos: np.ndarray   # (Tc, 3) m
    csv_fps: float
    video_fps: float         # source video frame rate (meta.json), for the human panel
    video_frames: int
    meta: dict
    dir: Path = None        # folder holding hpe_results.pt (+ preprocess/vitpose.pt)


def _meta_fps(meta, *keys, default=None):
    for k in keys:
        v = meta.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return default


def load_k1_csv(path, policy_names):
    """soma-retargeter K1 CSV -> (q [T,23] rad in policy order, root quat xyzw, root pos m, fps or None).
    A '# fps=<f>' comment line (or an 'fps' column) is honoured if present."""
    path = Path(path)
    lines = path.read_text().splitlines()
    fps = None
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            low = s.lower().replace(" ", "")
            for tag in ("fps=", "fps:"):
                if tag in low:
                    try:
                        fps = float(low.split(tag, 1)[1].split(",")[0])
                    except ValueError:
                        pass
            continue
        rows.append(s)
    header = next(csv.reader([rows[0]]))
    data = np.array([[float(x) for x in r.split(",")] for r in rows[1:]], np.float64)
    col = {n.strip(): i for i, n in enumerate(header)}
    missing = [n for n in policy_names if f"{n}_dof" not in col]
    if missing:
        raise ValueError(f"{path}: missing K1 joint columns {missing}")
    if "fps" in col:
        fps = float(data[0, col["fps"]])
    q = np.radians(np.stack([data[:, col[f"{n}_dof"]] for n in policy_names], 1))
    pos = np.stack([data[:, col[f"root_translate{a}"]] for a in "XYZ"], 1) / 100.0
    eul = np.stack([data[:, col[f"root_rotate{a}"]] for a in "XYZ"], 1)
    quat = Rotation.from_euler("xyz", eul, degrees=True).as_quat()
    return q, quat, pos, fps


def torso_quat(root_quat_xyzw, waist_yaw):
    """K1 torso_link orientation = pelvis * Rz(waist_yaw) (k1.urdf waist_yaw_joint: origin 0, axis z)."""
    return (Rotation.from_quat(root_quat_xyzw) * Rotation.from_rotvec(np.outer(waist_yaw, [0.0, 0.0, 1.0]))).as_quat()


def load_soma_joints(path):
    """(joints [T,77,3] m, names, fps, frame) from soma_joints.npz or a SOMA BVH (forward kinematics, 'Root' dropped)."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        frame = str(z["frame"]) if "frame" in z.files else ""
        return z["joints"].astype(np.float64), names, float(z["fps"]), frame
    from sonic.bvh import fk, read_bvh

    b = read_bvh(path)
    pos, _ = fk(b)
    keep = [i for i, n in enumerate(b.names) if n != "Root"]
    return pos[:, keep] / 100.0, [b.names[i] for i in keep], b.fps, "GEM-X SOMA BVH world (Y up), m"


def load_clip(src, policy_names):
    """A shared clip folder (contract above) or 'guap' (old Guap_ver2 result, validation only)."""
    if str(src) == "guap":
        J, names, fps, _ = load_soma_joints(GUAP / "Guap_ver2_soma.bvh")
        q, rq, rp, cfps = load_k1_csv(GUAP / "Guap_ver2_retarget_k1_from_bvh.csv", policy_names)
        return SomaClip("guap", J, names, fps, q, rq, rp, cfps or fps, 30.0, len(J),
                        {"video": str(GUAP / "Guap_ver2.mp4")}, GUAP)
    d = Path(src)
    if not d.is_dir():
        d = SHARED / str(src)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    npz = d / "soma_joints.npz"
    J, names, fps, _ = load_soma_joints(npz if npz.exists() else d / "soma.bvh")
    # k1.csv comes after the GEM-X files (soma-retargeter is slow); K1CSV_DONE marks it complete
    # (a re-run of GEM-X, e.g. version 2 with DONE_V2, replaces the GEM-X files in place: a k1.csv older than the
    # current soma_joints.npz belongs to the previous run and is ignored until its own K1CSV_DONE is rewritten)
    q = rq = rp = cfps = None
    marker = d / "K1CSV_DONE"
    gem = d / "soma_joints.npz"
    fresh = marker.exists() and (not gem.exists() or marker.stat().st_mtime >= gem.stat().st_mtime)
    if (d / "k1.csv").exists() and (fresh or not (d / "DONE").exists()):
        q, rq, rp, cfps = load_k1_csv(d / "k1.csv", policy_names)
    csv_fps = cfps or _meta_fps(meta.get("k1_csv", {}), "fps", default=None) or _meta_fps(
        meta, "csv_fps", "k1_csv_fps", "fps", "gem_fps", default=fps)
    vfps = _meta_fps(meta, "video_fps", "source_fps", "fps", default=fps)
    vframes = int(meta.get("video_frames") or meta.get("source_frames") or meta.get("frames") or len(J))
    return SomaClip(d.name, J, names, fps, q, rq, rp, csv_fps, vfps, vframes, meta, d)


# ---------------------------------------------------------------------------------------------------------------
def soma_ik_points(J, names):
    """SOMA joints -> the 14 cpu_kinematics targets (TARGET_NAMES), expressed in the person's torso frame and handed
    over in MediaPipe camera axes (x = person's left, y down, z away from the viewer), the convention
    K1CpuRetargeter.scale_human_targets expects. Chest is the shoulder midpoint, as for MediaPipe (the IK's chest
    target is the virtual point between the K1 shoulder roots)."""
    ix = {n: i for i, n in enumerate(names)}
    P = np.zeros((len(J), len(TARGET_NAMES), 3))
    for k, t in enumerate(TARGET_NAMES):
        if t == "Chest":
            P[:, k] = 0.5 * (J[:, ix["LeftArm"]] + J[:, ix["RightArm"]])
        else:
            P[:, k] = J[:, ix[t]]
    hip, chest = P[:, 0], P[:, 1]
    y = P[:, TARGET_NAMES.index("LeftArm")] - P[:, TARGET_NAMES.index("RightArm")]
    y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-6)
    z = chest - hip
    z -= y * np.sum(z * y, -1, keepdims=True)
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-6)
    x = np.cross(y, z)  # left x up = forward (right-handed source frame)
    R = np.stack([x, y, z], axis=-1)  # columns: torso axes in the source frame
    local = np.einsum("tkj,tjc->tkc", P - hip[:, None], R)  # x forward, y left, z up
    return np.stack((local[..., 1], -local[..., 2], -local[..., 0]), axis=-1)
