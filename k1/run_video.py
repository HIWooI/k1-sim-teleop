"""Human video -> MediaPipe or GEM-X (SOMA) -> K1 reference -> K1 physics (split drive) -> two-panel video.

  .venv/bin/python -m k1.run_video --preset                 # the curated event clips -> videos/k1_sbs_<name>.mp4
  .venv/bin/python -m k1.run_video --video in.mp4 --name x  # any clip
  .venv/bin/python -m k1.run_video --preset --source soma_ik   # GEM-X SOMA joints -> the same IK -> k1_sbs_<name>_soma_ik.mp4
  .venv/bin/python -m k1.run_video --preset --source soma_csv  # soma-retargeter K1 CSV -> k1_sbs_<name>_soma_csv.mp4

--source soma_* reads the GEM-X run of the same video from $K1_SOMA_CLIPS/<name>/ (default videos/soma_event_clips;
k1/soma.py; --soma-dir to point elsewhere) instead of MediaPipe:
  soma_ik   SOMA 77-joint rig -> the same position IK as MediaPipe (its targets are SOMA joint names)
  soma_csv  soma-retargeter K1 joint angles (the old offline K1 path, wrist roll included)
  soma_mlp  the distilled retarget MLP (soma-retargeter student) on the SOMA body_pose (the old live path)
--drive mimic (soma_csv / soma_mlp) follows the whole-body reference with the mimic student instead of the standing
loco policy. The MediaPipe landmark cache is still read in every mode, only to place the human crop (same framing as
the MediaPipe reel) and for the 2D arm-agreement metric; the robot never sees it in soma_* modes.

Perception is the same MediaPipe Pose (full) pass as scripts/run_video.py (cached landmarks, One-Euro filter).
Retargeting is the portable CPU position IK (k1/cpu_kinematics.py: MediaPipe points rescaled to K1 bone lengths,
bounded least squares on the URDF). The points are first expressed in the person's own torso frame (shoulder line
+ hips->shoulders; --frame camera keeps the original camera frame, which looked the same on the event clips), so camera
tilt or a slightly leaning presenter does not tilt the robot's arms. --method dir (limb directions x K1 segment
lengths -> the gestures' arm IK) is kept as an alternative; it read worse on the waving clip. Only the 10 arm
joints are used; legs + waist belong to loco_armdriven_r2 in split
mode (the robot stays balanced on its own, it does not step or bend with the person). The arm reference is
resampled to 50 Hz and slew-limited before it reaches the mimic policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from k1.cpu_kinematics import K1CpuRetargeter, TARGET_NAMES
from k1.sim import FPS, URDF, K1Split, Camera

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_video as g1rv  # noqa: E402  (MediaPipe extraction cache, panel fitting)
from sonic.mp2smpl import OneEuro  # noqa: E402

EVENT_VIDEOS = Path(os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))

# curated formal-looking clips (same selection as the G1 teleop showreel, scripts/make_showreel.py TELEOP_REEL)
PRESET = [
    ("pexels_10374292", "raw/pexels_10374292.mp4"),   # office, shirt & tie: waving hello
    ("pexels_10149176", "raw/pexels_10149176.mp4"),   # vest: welcoming with both hands
    ("pexels_10149187", "raw/pexels_10149187.mp4"),   # suit jacket: applause
    ("pexels_10515017", "raw/pexels_10515017.mp4"),   # suit jacket: guiding with open palms
    ("event_raise_arms", "event_raise_arms.mp4"),     # slow raise of both arms (Pexels 8939359, reversed)
    ("event_stand", "event_stand.mp4"),               # standing at ease (Pexels 9558954)
]

# MediaPipe landmarks -> the 14 IK targets (the original perception/demo_cpu.py MP_MAP)
MP_MAP = ((23, 24), (11, 12), (11,), (13,), (15,), (12,), (14,), (16,), (23,), (25,), (27,), (24,), (26,), (28,))
ARM_WRIST_ROLL = ("left_wrist_roll_joint", "right_wrist_roll_joint")


def mp_targets(world, vis):
    pts = np.stack([world[list(ix)].mean(axis=0) for ix in MP_MAP])
    conf = np.array([vis[list(ix)].min() for ix in MP_MAP])
    return pts, conf


def torso_frame(pts):
    """MediaPipe world points -> the same points in the person's torso frame, still in MediaPipe axes
    (so K1CpuRetargeter.scale_human_targets applies its own axis convention unchanged)."""
    k1 = np.stack((-pts[:, 2], pts[:, 0], -pts[:, 1]), axis=-1)  # x forward, y left, z up
    hip, chest = k1[0], k1[1]
    y = k1[TARGET_NAMES.index("LeftArm")] - k1[TARGET_NAMES.index("RightArm")]
    y /= max(np.linalg.norm(y), 1e-6)
    z = chest - hip
    z -= y * (z @ y)
    z /= max(np.linalg.norm(z), 1e-6)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)  # columns: torso axes in camera-K1 coordinates
    local = (k1 - hip) @ R
    return np.stack((local[:, 1], -local[:, 2], -local[:, 0]), axis=-1)


def retarget(W, V, names, frame="camera", max_nfev=20, smoothing=0.5):
    """Per video frame: 23 K1 joints from the IK (only arms are used downstream) + hand target RMS."""
    ret = K1CpuRetargeter(URDF, max_nfev=max_nfev, smoothing=smoothing)
    order = [ret.robot.joint_names.index(n) for n in names]
    Q, rms = [], []
    for t in range(len(W)):
        pts, conf = mp_targets(W[t], V[t])
        targets = ret.scale_human_targets(torso_frame(pts) if frame == "torso" else pts)
        q, pos = ret.solve(targets, conf)
        Q.append(q[order])
        cur = np.stack([pos[n] for n in ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")])
        rms.append(float(np.sqrt(np.mean((cur - targets[[4, 7]]) ** 2))))
    return np.array(Q), np.array(rms)


ARM_LANDMARKS = {"left": (11, 13, 15), "right": (12, 14, 16)}


def retarget_dir(W, V, sim, min_vis=0.5):
    """Direction retargeting: the person's upper-arm and forearm directions (camera frame, K1 axes) times K1's own
    segment lengths -> elbow + palm targets -> 5-DOF arm IK (k1/gestures.ArmIK, the gestures' solver).
    The position IK targets the wrist-link origin, only 7 cm past the K1 elbow, which leaves the forearm direction
    weakly constrained; the palm sits ~27 cm past the elbow."""
    A, ok = [], []
    for t in range(len(W)):
        k1 = np.stack((-W[t][:, 2], W[t][:, 0], -W[t][:, 1]), axis=-1)  # x forward, y left, z up
        A.append([[k1[i] for i in ARM_LANDMARKS[s]] for s in ("left", "right")])
        ok.append([min(V[t][i] for i in ARM_LANDMARKS[s]) >= min_vis for s in ("left", "right")])
    return retarget_dir_arms(np.array(A), np.array(ok), sim)


def retarget_dir_arms(A, ok, sim):
    """A: (T, 2 sides [left, right], 3 [shoulder, elbow, wrist], 3) in K1 axes (x forward, y left, z up);
    ok: (T, 2) usable (else the side holds its last target)."""
    from k1.gestures import ArmIK

    ik = ArmIK(sim)
    lens = {}
    for side in ("left", "right"):
        palm, elbow = ik.points(ik.base, side)
        lens[side] = (np.linalg.norm(elbow), np.linalg.norm(palm - elbow))
    q = ik.base.copy()
    Q, rms = [], []
    last = {}
    for t in range(len(A)):
        err = []
        for si, side in enumerate(("left", "right")):
            if ok[t, si] or side not in last:
                a, b, c = A[t, si]
                u = b - a
                f = c - b
                u /= max(np.linalg.norm(u), 1e-6)
                f /= max(np.linalg.norm(f), 1e-6)
                elbow = u * lens[side][0]
                last[side] = (elbow + f * lens[side][1], elbow)
            palm_t, elbow_t = last[side]
            q, e = ik.solve(side, palm_t, elbow_t, w_elbow=0.8, q_init=q)
            err.append(e)
        Q.append(q.copy())
        rms.append(float(np.mean(err)))
    return np.array(Q), np.array(rms)


def to_reference(Q, vfps, sim, vmax=6.0, wrist=False, whole_body=False):
    """Video-rate arm joints -> 50 Hz reference (policy order): legs/waist = loco standing pose, wrist roll at its
    default (MediaPipe gives no forearm twist; wrist=True keeps Q's wrist roll, e.g. soma-retargeter's),
    slew-limited to vmax rad/s, velocities by finite differences. whole_body=True keeps all 23 joints of Q."""
    T = len(Q)
    t_src = np.arange(T) / vfps
    n = int(np.floor(t_src[-1] * FPS)) + 1
    t50 = np.arange(n) / FPS
    q = np.empty((n, Q.shape[1]))
    for j in range(Q.shape[1]):
        q[:, j] = np.interp(t50, t_src, Q[:, j])
    ref = np.tile(sim.loco.default, (n, 1))
    arms = np.ones_like(sim.arms) if whole_body else sim.arms.copy()
    if not wrist:
        for w in ARM_WRIST_ROLL:
            arms[sim.names.index(w)] = False
    ref[:, arms] = q[:, arms]
    step = vmax / FPS
    for i in range(1, n):
        ref[i, arms] = np.clip(ref[i, arms], ref[i - 1, arms] - step, ref[i - 1, arms] + step)
    ref = np.clip(ref, sim.lo, sim.hi)
    return ref, np.gradient(ref, 1.0 / FPS, axis=0)


def label_overlays(W_, H_):
    import make_showreel as ms

    lab_h = ms.pill([("사람 영상", ms.font("semibold", 26), ms.INK)], pad=(20, 12), radius=14, bar=None,
                    bg=(255, 255, 255, 210))
    lab_r = ms.pill([("AI Sapiens K1 (물리 시뮬레이션)", ms.font("semibold", 26), ms.INK)], pad=(20, 12), radius=14,
                    bar=None, bg=(255, 255, 255, 210))
    from PIL import Image

    hw = W_ // 2
    return [ms.Overlay(lab_h, 40, 36), ms.Overlay(lab_r, hw + 40, 36),
            ms.Overlay(Image.new("RGBA", (4, H_), (255, 255, 255, 255)), hw - 2, 0)]


MP_ARM2D = dict(ls=11, le=13, lw=15, rs=12, re=14, rw=16, lh=23, rh=24)  # MediaPipe landmark ids


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def human_arm_2d(P, C, ix, min_conf=0.5):
    """2D keypoints (T,J,2 px, image y down) + confidences -> (T,4) in-plane angles of [L upper, L fore, R upper,
    R fore] measured from the person's own up axis (perpendicular to the shoulder line; the hips are out of frame
    in the waist-up clips), (T,4) projected segment length in px, (T,4) valid. Image right = the person's left for a
    front-facing camera, so the angles compare 1:1 with the robot's frontal plane (y left, z up) seen from the
    front camera."""
    P = np.asarray(P, np.float64)[..., :2] * np.array([1.0, -1.0])  # x right, y up
    sv = P[:, ix["ls"]] - P[:, ix["rs"]]  # right -> left shoulder (points image-right when facing the camera)
    ta = np.arctan2(-sv[:, 1], sv[:, 0])  # angle of the up axis (-sv_y, sv_x) measured from +y toward +x
    torso_ok = (C[:, [ix["ls"], ix["rs"]]] > min_conf).all(1)
    ang, length, ok = [], [], []
    for s in "lr":
        for a, b in (("s", "e"), ("e", "w")):
            p0, p1 = ix[s + a], ix[s + b]
            v = P[:, p1] - P[:, p0]
            ang.append(_wrap(np.arctan2(v[:, 0], v[:, 1]) - ta))
            length.append(np.linalg.norm(v, axis=-1))
            ok.append((C[:, p0] > min_conf) & (C[:, p1] > min_conf) & torso_ok & ~np.isnan(v[:, 0]))
    return np.stack(ang, 1), np.stack(length, 1), np.stack(ok, 1)


def vitpose_2d(name, soma_dir, n_video=None):
    """GEM-X's own 2D input (ViTPose, 77 SOMA keypoints, px) for the clip if the folder has it, else None:
    (P (T,77,2), conf (T,77), arm index map). With n_video the arrays are re-indexed to the source video frames
    (GEM-X ran on every frame_stride-th frame)."""
    from k1.soma import GUAP, SHARED, load_soma_joints

    d = GUAP if name == "guap" and not soma_dir else (Path(soma_dir) if soma_dir else SHARED) / name
    kp, names, stride = None, None, 1
    if (d / "soma_joints.npz").exists():
        z = np.load(d / "soma_joints.npz", allow_pickle=True)
        names = [str(n) for n in z["names"]]
        if "kp2d_vitpose" in z.files:
            kp = z["kp2d_vitpose"].astype(np.float64)
            if "source_frame_index" in z.files and len(z["source_frame_index"]) > 1:
                stride = int(z["source_frame_index"][1] - z["source_frame_index"][0])
    if kp is None:
        for f in (d / "preprocess" / "vitpose.pt", d / "vitpose.pt"):
            if f.exists():
                import torch

                kp = torch.load(str(f), map_location="cpu", weights_only=False)
                kp = np.asarray(kp.numpy() if hasattr(kp, "numpy") else kp, np.float64)
                break
        if names is None:
            for src in (d / "soma.bvh", d / "Guap_ver2_soma.bvh"):
                if src.exists():
                    names = load_soma_joints(src)[1]
                    break
    if kp is None or names is None or kp.shape[1] != len(names):
        return None
    if n_video is not None:
        kp = kp[np.minimum(np.round(np.arange(n_video) / stride).astype(int), len(kp) - 1)]
    n = {k: names.index(v) for k, v in dict(ls="LeftArm", le="LeftForeArm", lw="LeftHand", rs="RightArm",
                                               re="RightForeArm", rw="RightHand", lh="LeftLeg", rh="RightLeg").items()}
    return kp[..., :2], kp[..., 2], n


class RobotArm2D:
    """Robot limbs (shoulder -> elbow -> palm, physics state) in the torso frame's frontal plane: angle from the
    torso z axis (toward +y = robot's left)."""

    def __init__(self, sim):
        from k1.gestures import HAND_OFFSET

        m = sim.model
        self.sim, self.off = sim, HAND_OFFSET
        self.ids = [(m.body(f"{s}_shoulder_roll_link").id, m.body(f"{s}_elbow_link").id,
                     m.body(f"{s}_wrist_roll_rubber_hand").id) for s in ("left", "right")]

    def __call__(self):
        d = self.sim.data
        Rt = d.xmat[self.sim.torso_id].reshape(3, 3)
        ang = []
        for sh, el, ha in self.ids:
            palm = d.xpos[ha] + d.xmat[ha].reshape(3, 3) @ self.off
            pts = [Rt.T @ (p - d.xpos[self.sim.torso_id]) for p in (d.xpos[sh], d.xpos[el], palm)]
            for p0, p1 in ((pts[0], pts[1]), (pts[1], pts[2])):
                v = p1 - p0
                ang.append(np.arctan2(v[1], v[2]))
        return np.array(ang)


def arm2d_agreement(h_ang, h_rel, h_ok, r_ang, min_rel=0.4):
    """Mean |human 2D limb angle - robot frontal-plane limb angle| (deg) over the frames where the human segment is
    clearly visible in the image plane (keypoints confident, projected length >= min_rel x that segment's clip p95,
    i.e. not pointing at the camera). The frame set depends on the human video only, so every source is scored on the
    same frames."""
    hn = h_rel / np.maximum(np.nanpercentile(np.where(h_ok, h_rel, np.nan), 95, axis=0), 1e-6)
    ok = h_ok & (hn >= min_rel)
    e = np.degrees(np.abs(_wrap(h_ang - r_ang)))
    pick = lambda cols: float(e[:, cols][ok[:, cols]].mean()) if ok[:, cols].any() else float("nan")  # noqa: E731
    return {"mean": round(pick([0, 1, 2, 3]), 2), "upper": round(pick([0, 2]), 2), "fore": round(pick([1, 3]), 2),
            "p90": round(float(np.percentile(e[ok], 90)), 2) if ok.any() else float("nan"),
            "valid_frac": round(float(ok.mean()), 3)}


def perceive(source, mp, name, sim, frame, method, soma_dir, ik_smoothing, mlp_ckpt=""):
    """-> (Q in policy order at its own rate, that rate, wrist roll present, info, SomaClip or None).
    mp = (W, V, vfps, lost frames): the MediaPipe pass, used only by source="mediapipe"."""
    if source == "mediapipe":
        W, V, vfps, lost = mp
        filt = OneEuro(1.0, 0.4)
        Wf = np.array([filt(W[t], t / vfps) for t in range(len(W))])
        if method == "dir":
            Q, ik_rms = retarget_dir(Wf, V, sim)
        else:
            Q, ik_rms = retarget(Wf, V, sim.names, frame=frame)
        return Q, vfps, False, {"lost_frames": lost, "ik_hand_rms_m": round(float(np.mean(ik_rms)), 4),
                                "pipeline": ("MediaPipe Pose full -> arm directions x K1 segment lengths -> 5-DOF arm IK"
                                             if method == "dir" else f"MediaPipe Pose full -> {frame} frame -> K1 CPU IK "
                                             "(position IK)")}, None
    from k1.soma import load_clip, soma_ik_points

    clip = load_clip(Path(soma_dir) / name if soma_dir else name, sim.names)
    info = {"soma_clip": clip.name, "soma_frames": int(len(clip.joints)), "soma_fps": round(clip.fps, 3),
            "csv_frames": None if clip.csv_q is None else int(len(clip.csv_q)), "csv_fps": round(clip.csv_fps, 3)}
    if source == "soma_csv":
        if clip.csv_q is None:
            raise SystemExit(f"{clip.dir}: no complete k1.csv yet (K1CSV_DONE missing)")
        info["pipeline"] = "GEM-X (SOMA) -> soma-retargeter K1 CSV (23 joints, wrist roll included)"
        return clip.csv_q, clip.csv_fps, True, info, clip
    if source == "soma_mlp":
        from k1.soma_mlp import MODELS, SOMA_IDX, load_body_pose, mlp_joint_angles

        bp = load_body_pose(clip.dir / "hpe_results.pt")
        vp = vitpose_2d(name, soma_dir)
        conf = vp[1][:, SOMA_IDX] if vp is not None and len(vp[1]) == len(bp) else None
        Q, mi = mlp_joint_angles(bp, sim.names, clip.fps, ckpt=Path(mlp_ckpt) if mlp_ckpt else MODELS / "k1_retarget.pt",
                                 conf=conf)
        info.update(mi, pipeline="GEM-X (SOMA body_pose) -> distilled retarget MLP (soma-retargeter student, "
                    "wrist roll included)")
        return Q, clip.fps, True, info, clip
    P = soma_ik_points(clip.joints, clip.names)
    if method == "dir":
        K = np.stack((-P[..., 2], P[..., 0], -P[..., 1]), axis=-1)  # torso frame, K1 axes
        A = np.stack([K[:, [TARGET_NAMES.index(f"{s}{j}") for j in ("Arm", "ForeArm", "Hand")]]
                      for s in ("Left", "Right")], axis=1)
        Q, rms = retarget_dir_arms(A, np.ones((len(A), 2), bool), sim)
        info.update(pipeline="GEM-X (SOMA 77 joints) -> torso frame -> arm directions x K1 segment lengths -> "
                    "5-DOF arm IK", ik_palm_err_m=round(float(np.mean(rms)), 4))
        return Q, clip.fps, False, info, clip
    ret = K1CpuRetargeter(URDF, max_nfev=20, smoothing=ik_smoothing)
    order = [ret.robot.joint_names.index(n) for n in sim.names]
    Q, rms = [], []
    for t in range(len(P)):
        targets = ret.scale_human_targets(P[t])
        q, pos = ret.solve(targets, np.ones(len(targets)))
        Q.append(q[order])
        cur = np.stack([pos[n] for n in ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")])
        rms.append(float(np.sqrt(np.mean((cur - targets[[4, 7]]) ** 2))))
    info.update(pipeline="GEM-X (SOMA 77 joints) -> torso frame -> K1 CPU IK (position IK)",
                ik_hand_rms_m=round(float(np.mean(rms)), 4))
    return np.array(Q), clip.fps, False, info, clip


def run(video, name, out_dir, size=(1920, 1080), max_seconds=20.0, labels=True, fps=25, quality=9, settle=1.5,
        tail=1.0, frame="torso", method="shape18", source="mediapipe", soma_dir="", drive="split", wrist="auto",
        ik_smoothing=0.5, suffix="", write_video=True, mlp_ckpt=""):
    t0 = time.perf_counter()
    # MediaPipe pass: the robot input for source="mediapipe"; for every source also the human crop (same framing as
    # the MediaPipe reel) and the 2D arm-agreement metric
    W, V, I, vfps = g1rv.extract(video, "full", None, 1, 1280)
    if max_seconds > 0:
        k = int(max_seconds * vfps)
        W, V, I = W[:k], V[:k], I[:k]
    T = len(W)
    valid = ~np.isnan(W[:, 0, 0])
    first = int(np.argmax(valid))
    for t in range(T):  # hold the last valid detection where MediaPipe lost the person
        if not valid[t]:
            src = t - 1 if t > first else first
            W[t], V[t], I[t] = W[src], V[src], I[src]

    sim = K1Split(mode="mimic" if drive == "mimic" else "split")
    Q, qfps, has_wrist, info, clip = perceive(source, (W, V, vfps, int((~valid).sum())), name, sim, frame, method,
                                              soma_dir, ik_smoothing, mlp_ckpt)
    if max_seconds > 0:
        Q = Q[: int(max_seconds * qfps) + 1]
    use_wrist = has_wrist if wrist == "auto" else wrist == "on"
    if use_wrist and not has_wrist and (clip is None or clip.csv_q is None):
        raise SystemExit("--wrist on needs the soma-retargeter wrist roll: no complete k1.csv for this clip")
    if use_wrist and not has_wrist:  # soma_ik + soma-retargeter's wrist roll (forearm twist the IK cannot see)
        csv_wr = np.stack([np.interp(np.arange(len(Q)) / qfps, np.arange(len(clip.csv_q)) / clip.csv_fps,
                                     clip.csv_q[:, sim.names.index(w)]) for w in ARM_WRIST_ROLL], 1)
        Q = Q.copy()
        Q[:, [sim.names.index(w) for w in ARM_WRIST_ROLL]] = csv_wr
    rq, rdq = to_reference(Q, qfps, sim, wrist=use_wrist, whole_body=drive == "mimic")
    n_ref = len(rq)
    tq = np.tile([0.0, 0.0, 0.0, 1.0], (n_ref, 1))
    if drive == "mimic":
        if clip is None or clip.csv_q is None:
            raise SystemExit("--drive mimic takes the torso orientation from k1.csv, which is not there yet")
        from scipy.spatial.transform import Rotation, Slerp

        from k1.soma import torso_quat

        tcsv = np.arange(len(clip.csv_q)) / clip.csv_fps
        rq_t = torso_quat(clip.csv_root_quat, clip.csv_q[:, sim.names.index("waist_yaw_joint")])
        tq = Slerp(tcsv, Rotation.from_quat(rq_t))(np.clip(np.arange(n_ref) / FPS, 0, tcsv[-1])).as_quat()
        sim.reset(tq[0], q0=rq[0])
    else:
        sim.reset()

    W_, H_ = size
    pw = W_ // 2
    writer = cam = None
    every = max(1, int(round(FPS / fps)))
    out = Path(out_dir) / f"k1_sbs_{name}{suffix}.mp4"
    if write_video:
        import imageio.v2 as imageio

        fit = g1rv.PanelFit(pw, H_, g1rv.person_span(I, V))
        reader = g1rv.FrameReader(video, height=H_, fit=fit)
        cam = Camera(sim, w=W_ - pw, h=H_, azimuth=180.0, distance=2.25, elevation=-5.0, lookat_z=0.74)
        ovs = label_overlays(W_, H_) if labels else []
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(out, fps=fps, quality=quality, macro_block_size=1)
    cap = cv2.VideoCapture(str(video))
    wh = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    refs2d = {"mediapipe": human_arm_2d(I[..., :2] * np.asarray(wh), V, MP_ARM2D)}
    vp = vitpose_2d(name, soma_dir, n_video=T)
    if vp is not None:
        refs2d["vitpose"] = human_arm_2d(*vp)
    r2d = RobotArm2D(sim)
    n_settle, n_tail = int(settle * FPS), int(tail * FPS)
    rec_from = n_settle - int(0.5 * FPS)
    qs, refs, r_ang, kvs = [], [], [], []
    fell, min_z, max_tilt, frames = None, 9.0, 0.0, 0
    for k in range(n_settle + n_ref + n_tail):
        i = min(max(k - n_settle, 0), n_ref - 1)
        moving = 0 <= k - n_settle < n_ref
        sim.step(rq[i], rdq[i] if moving else np.zeros(23), tq[i])
        kv = min(int(i / FPS * vfps), T - 1)  # human frame the robot is executing now
        if moving:
            qs.append(sim.q)
            refs.append(rq[i])
            r_ang.append(r2d())
            kvs.append(kv)
        if k > 25:
            z, tilt = sim.pelvis_height(), sim.tilt_deg()
            min_z, max_tilt = min(min_z, z), max(max_tilt, tilt)
            if fell is None and (z < 0.35 or tilt > 60):
                fell = k / FPS
        if writer is not None and k >= rec_from and (k - rec_from) % every == 0:
            robot = cam.grab()
            hum = cv2.cvtColor(reader.get(kv), cv2.COLOR_BGR2RGB)
            img = np.concatenate([hum, robot], axis=1)
            for ov in ovs:
                ov.blend(img)
            writer.append_data(img)
            frames += 1
    if writer is not None:
        writer.close()
    qs, refs = np.array(qs), np.array(refs)
    arms = sim.arms
    err = np.degrees(np.abs(qs - refs))
    kvs = np.array(kvs)
    acc = lambda x: float(np.sqrt(np.mean(np.diff(x[:, arms], 2, axis=0) ** 2)) * FPS * FPS)  # noqa: E731
    res = {
        "clip": name, "video": str(video), "robot": "ROBOTIS AI Sapiens K1 (23 DOF), MuJoCo", "source": source,
        "pipeline": info.pop("pipeline") + " -> " + ("whole-body reference 50 Hz -> mimic student (all joints)"
                                                     if drive == "mimic" else "arms reference 50 Hz -> split drive "
                                                     "(arms=mimic student, legs+waist=loco_armdriven_r2)"),
        "drive": drive, "wrist_roll": "reference" if use_wrist else "default",
        "human_seconds": round(T / vfps, 2), "fell": fell is not None, "fell_at_s": fell,
        "min_pelvis_z": round(min_z, 3), "max_pelvis_tilt_deg": round(max_tilt, 1),
        **info,
        "tracking_err_deg": {"arms": round(float(err[:, arms].mean()), 2),
                             "arms_p90": round(float(np.percentile(err[:, arms], 90)), 2),
                             "legs_waist_vs_standing": round(float(err[:, ~arms].mean()), 2)},
        "arm2d_vs_human_deg": {k: arm2d_agreement(a[np.minimum(kvs, len(a) - 1)], r[np.minimum(kvs, len(a) - 1)],
                                                  o[np.minimum(kvs, len(a) - 1)], np.array(r_ang))
                               for k, (a, r, o) in refs2d.items()},
        "arm_acc_rms_rad_s2": {"reference": round(acc(refs), 1), "robot": round(acc(qs), 1)},
        "per_joint_err_deg": {n.replace("_joint", ""): round(float(v), 1) for n, v in zip(sim.names, err.mean(0))},
        "video_out": {"path": str(out) if write_video else None, "size": f"{W_}x{H_}", "fps": fps, "frames": frames,
                      "human_t0_in_video_s": round(0.5, 3), "labels": labels},
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (Path(out_dir) / f"k1_sbs_{name}{suffix}.json").write_text(json.dumps(res, indent=2))
    # the 50 Hz reference itself (policy order), e.g. for sonic_k1_train/mac/soma_ref_to_pkl.py
    np.savez(ROOT / "logs" / f"k1_ref_{name}{suffix}.npz", q=rq, torso_quat_xyzw=tq, fps=FPS, names=np.array(sim.names),
             drive=drive)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--preset", action="store_true", help="run the curated event clips")
    ap.add_argument("--only", nargs="*", default=[], help="with --preset: subset of names")
    ap.add_argument("--out-dir", default=str(ROOT / "videos"))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="metrics only")
    ap.add_argument("--frame", default="torso", choices=["torso", "camera"], help="position IK in the person's torso "
                    "frame (default) or in the camera frame (the original)")
    ap.add_argument("--suffix", default=None, help="appended to the output name (default: '' for mediapipe, "
                    "'_<source>' otherwise)")
    ap.add_argument("--method", default="shape18", choices=["shape18", "dir"], help="shape18: cpu_kinematics point IK "
                    "(default); dir: limb directions x K1 lengths -> arm IK")
    ap.add_argument("--source", default="mediapipe", choices=["mediapipe", "soma_ik", "soma_csv", "soma_mlp"])
    ap.add_argument("--mlp-ckpt", default="", help="soma_mlp checkpoint (default: shared shape_models/k1_retarget.pt)")
    ap.add_argument("--soma-dir", default="", help="parent of the <name>/ GEM-X clip folders (default: the shared "
                    "folder); --name guap = the old Guap_ver2 result (validation only)")
    ap.add_argument("--drive", default="split", choices=["split", "mimic"], help="mimic: whole-body reference "
                    "(soma_csv / soma_mlp) with the mimic student instead of the standing loco policy")
    ap.add_argument("--wrist", default="auto", choices=["auto", "on", "off"], help="wrist roll from the reference "
                    "(auto: only soma_csv has it; on with soma_ik = soma-retargeter's wrist roll)")
    ap.add_argument("--ik-smoothing", type=float, default=0.5, help="cpu IK EMA weight of the new solution")
    args = ap.parse_args()
    if args.drive == "mimic" and args.source not in ("soma_csv", "soma_mlp"):
        raise SystemExit("--drive mimic needs a whole-body reference: --source soma_csv or soma_mlp")
    size = tuple(int(v) for v in args.size.split("x"))
    suffix = args.suffix if args.suffix is not None else ("" if args.source == "mediapipe" else f"_{args.source}")
    jobs = [(n, EVENT_VIDEOS / p) for n, p in PRESET if not args.only or n in args.only] if args.preset else \
        [(args.name or Path(args.video).stem, Path(args.video))]
    for name, video in jobs:
        r = run(video, name, args.out_dir, size=size, max_seconds=args.max_seconds, labels=not args.no_labels,
                frame=args.frame, method=args.method, source=args.source, soma_dir=args.soma_dir, drive=args.drive,
                wrist=args.wrist, ik_smoothing=args.ik_smoothing, suffix=suffix, write_video=not args.no_video,
                mlp_ckpt=args.mlp_ckpt)
        t = r["tracking_err_deg"]
        a2 = " ".join(f"2d[{k[:2]}]={v['mean']:5.2f}(up {v['upper']:5.2f} fore {v['fore']:5.2f} n={v['valid_frac']:.2f})"
                      for k, v in r["arm2d_vs_human_deg"].items())
        print(f"{name + suffix:28s} fell={r['fell']} minZ={r['min_pelvis_z']:.3f} tilt={r['max_pelvis_tilt_deg']:5.1f} "
              f"arms={t['arms']:5.2f} p90={t['arms_p90']:5.2f} {a2} acc={r['arm_acc_rms_rad_s2']['robot']:6.1f} "
              f"{r['human_seconds']}s {r['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
