"""Human mocap (SOMA / NOVA-skeleton BVH: BONES-SEED soma_uniform, GEM-X output) -> the live SMPL-24 stream.

The K1 SMPL encoder must see in training exactly what it sees live, and live input is MediaPipe Pose. So the mocap is
NOT fitted to an SMPL body: every frame is turned into the 33 world landmarks a perfect MediaPipe would report and run
through the live converter (k1live/smpl_stream.make_converter -> sonic/mp2smpl.MediaPipeToSmpl: bone directions
re-laid on the canonical SMPL rest bone lengths). Skeleton size, root-frame definition and joint placement are then
identical between training and the webcam - only the pose itself differs.

SOMA joint -> MediaPipe landmark (SOMA rig naming: *Arm = shoulder, *ForeArm = elbow, *Hand = wrist, *Leg = hip,
*Shin = knee, *Foot = ankle, *ToeBase = ball of the foot; vendor soma_to_smpl.py uses the same aliases):
  shoulders 11/12 = Left/RightArm, elbows 13/14 = *ForeArm, wrists 15/16 = *Hand,
  pinky 17/18 = *HandPinky1, index 19/20 = *HandIndex1, thumb 21/22 = *HandThumb2 (palm normal only uses index/pinky)
  hips 23/24 = *Leg, knees 25/26 = *Shin, ankles 27/28 = *Foot, foot index 31/32 = *ToeEnd (MediaPipe = toe tip),
  heels 29/30 = ankle pushed back along the foot, ears 7/8 = 35 % up the Head->HeadEnd bone +- 7 cm along the eye line
  (Left/RightEye), nose 0 = between the eyes + 3 cm forward, eyes 1-6 / mouth 9-10 = filler (unused by the converter).
The processed 26-joint SOMA PKLs (vendor extract_soma_joints_from_bvh.py, GEAR-SONIC sample_data/soma_filtered) lack
the finger / eye / toe-end joints: soma26_landmarks() approximates them from Thumb1 / Middle1, the chest line and
ToeBase (only for validating the pipeline on the public sample - BONES-SEED itself is read from the BVH).

BVH conventions follow vendor gear_sonic/data_process/extract_soma_joints_from_bvh.py: channel order string ->
scipy intrinsic Euler (upper case), joints with position channels use them instead of the OFFSET, centimetres, Y-up.
Output world frame: Z-up metres, (x, y, z)_Yup -> (x, -z, y) (the same rotation as smpl_root_ytoz_up), which puts it in
the frame of the BONES G1 CSV (root headings agree within ~1 deg on the sample clip, docs/training.md).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from k1live import smpl_stream  # noqa: E402
from sonic.mp2smpl import MP_TO_W  # noqa: E402

YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
EAR_UP, EAR_HALF, NOSE_FWD = 0.35, 0.07, 0.03
HELDOUT_FRAC = 0.05


def heldout_of(name: str) -> bool:
    """~5 % of the takes, by name hash; a take and its mirror (<name>_M) always land on the same side."""
    import hashlib

    base = name[:-2] if name.endswith("_M") else name
    return int(hashlib.sha1(base.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < HELDOUT_FRAC


def grid_frames(n_src: int, fps_src: float, fps: float = 50.0) -> int:
    """Frames of the common resampled grid (k1_kin.resample): t = 0 .. (n_src - 1) / fps_src at 1 / fps."""
    return int(np.floor((n_src - 1) / fps_src * fps)) + 1


# ------------------------------------------------------------------------------------------------ BVH
def read_bvh(path):
    """-> dict(names, parents, offsets (J,3) cm, channels [list per joint], motion (T, C), fps)."""
    text = Path(path).read_text()
    head, _, motion = text.partition("MOTION")
    names, parents, offsets, channels, stack = [], [], [], [], []
    for line in head.splitlines():
        s = line.strip()
        m = re.match(r"(ROOT|JOINT)\s+(\S+)", s)
        if m:
            names.append(m.group(2))
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            stack.append(len(names) - 1)
        elif s.startswith("End Site"):
            stack.append(None)  # end sites: own braces, no channels
        elif s.startswith("OFFSET") and stack and stack[-1] is not None:
            offsets[stack[-1]] = np.array([float(x) for x in s.split()[1:4]])
        elif s.startswith("CHANNELS") and stack and stack[-1] is not None:
            parts = s.split()
            channels[stack[-1]] = parts[2:2 + int(parts[1])]
        elif s == "}" and stack:
            stack.pop()
    mlines = motion.strip().splitlines()
    n_frames = int(mlines[0].split(":")[1])
    frame_time = float(mlines[1].split(":")[1])
    data = np.array(" ".join(mlines[2:2 + n_frames]).split(), dtype=np.float64)
    n_ch = sum(len(c) for c in channels)
    data = data.reshape(n_frames, n_ch)
    return {"names": names, "parents": parents, "offsets": np.array(offsets), "channels": channels,
            "motion": data, "fps": 1.0 / frame_time}


def bvh_fk(bvh, stride: int = 1):
    """World positions (T, J, 3) [m, Z-up] and rotations (T, J, 3, 3) [Z-up] of every joint."""
    motion = bvh["motion"][::stride]
    T, J = len(motion), len(bvh["names"])
    pos = np.zeros((T, J, 3))
    R = np.zeros((T, J, 3, 3))
    col = 0
    for j in range(J):
        ch = bvh["channels"][j]
        idx = {c: col + k for k, c in enumerate(ch)}
        col += len(ch)
        if any(c.endswith("position") for c in ch):
            local_p = np.stack([motion[:, idx[a]] if a in idx else np.zeros(T)
                                for a in ("Xposition", "Yposition", "Zposition")], 1)
        else:
            local_p = np.broadcast_to(bvh["offsets"][j], (T, 3))
        rch = [c for c in ch if c.endswith("rotation")]
        if rch:
            order = "".join(c[0] for c in rch).upper()
            local_R = Rot.from_euler(order, motion[:, [idx[c] for c in rch]], degrees=True).as_matrix()
        else:
            local_R = np.broadcast_to(np.eye(3), (T, 3, 3))
        p = bvh["parents"][j]
        if p < 0:
            pos[:, j], R[:, j] = local_p, local_R
        else:
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", R[:, p], local_p)
            R[:, j] = np.einsum("tij,tjk->tik", R[:, p], local_R)
    pos = pos / 100.0 @ YUP_TO_ZUP.T
    R = np.einsum("ij,tkjl->tkil", YUP_TO_ZUP, R)
    return pos, R


# ------------------------------------------------------------------------------------------------ landmarks
def _n(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _fill_face_and_feet(L, head, head_end, eye_l, eye_r, toe_l, toe_r):
    """Ears / nose / eyes / mouth / heels / foot index from head + eye line + toes (all (T,3), Z-up world)."""
    mid_ear = head + EAR_UP * (head_end - head)
    up = _n(head_end - head)
    left = eye_l - eye_r
    left = _n(left - np.sum(left * up, -1, keepdims=True) * up)
    fwd = np.cross(left, up)
    L[:, 7] = mid_ear + EAR_HALF * left
    L[:, 8] = mid_ear - EAR_HALF * left
    eye_mid = 0.5 * (eye_l + eye_r)
    L[:, 0] = eye_mid + NOSE_FWD * fwd
    for k, side in ((1, 1), (2, 1), (3, 1), (4, -1), (5, -1), (6, -1)):
        L[:, k] = eye_mid + side * 0.03 * left + 0.02 * fwd
    L[:, 9] = L[:, 0] - 0.05 * up + 0.015 * left
    L[:, 10] = L[:, 0] - 0.05 * up - 0.015 * left
    L[:, 31], L[:, 32] = toe_l, toe_r
    for heel, ank, toe in ((29, 27, toe_l), (30, 28, toe_r)):
        L[:, heel] = L[:, ank] - 0.25 * (toe - L[:, ank]) - np.array([0.0, 0.0, 0.05])


def soma_landmarks(names, pos):
    """(T, J, 3) SOMA FK positions (Z-up world) -> (T, 33, 3) MediaPipe landmark positions (Z-up world)."""
    ix = {n: i for i, n in enumerate(names)}
    g = lambda n: pos[:, ix[n]]  # noqa: E731
    if any(f"{s}Hand{f}" not in ix for s in ("Left", "Right") for f in ("Index1", "Pinky1", "Thumb2")):
        # a rig without the finger joints: index / pinky / thumb from Thumb1 + Middle1 like the SOMA-26 path
        extra = {}
        for side in ("Left", "Right"):
            wr, th, mi = g(f"{side}Hand"), g(f"{side}HandThumb1"), g(f"{side}HandMiddle1")
            ax = _n(mi - wr)
            u = th - wr
            u = _n(u - np.sum(u * ax, -1, keepdims=True) * ax)
            extra.update({f"{side}HandIndex1": mi + 0.02 * u, f"{side}HandPinky1": mi - 0.03 * u,
                          f"{side}HandThumb2": th})
        extra = {k: v for k, v in extra.items() if k not in ix}
        names = list(names) + list(extra)
        pos = np.concatenate([pos] + [v[:, None] for v in extra.values()], 1)
        ix = {n: i for i, n in enumerate(names)}
    L = soma_landmarks_no_face(names, pos)
    toe ={s: g(f"{s}ToeEnd") if f"{s}ToeEnd" in ix else g(f"{s}ToeBase") for s in ("Left", "Right")}
    if "LeftEye" in ix:
        eye_l, eye_r = g("LeftEye"), g("RightEye")
    else:  # no eyes in the rig: eye line parallel to the shoulder line
        sh = _n(L[:, 11] - L[:, 12])
        eye_l, eye_r = g("Head") + 0.03 * sh, g("Head") - 0.03 * sh
    head_end = g("HeadEnd") if "HeadEnd" in ix else g("Head") + (g("Head") - g("Neck1"))
    _fill_face_and_feet(L, g("Head"), head_end, eye_l, eye_r, toe["Left"], toe["Right"])
    return L


SOMA26 = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
          "LeftHandThumb1", "LeftHandMiddle1", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
          "RightHandThumb1", "RightHandMiddle1", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg",
          "RightShin", "RightFoot", "RightToeBase"]


def soma26_landmarks(j26):
    """Processed SOMA-26 joints (T, 26, 3) [Z-up, any origin] -> (T, 33, 3) landmarks (approximate hands / face)."""
    g = lambda n: j26[:, SOMA26.index(n)]  # noqa: E731
    T = len(j26)
    pos = {}
    for side in ("Left", "Right"):
        wr, th, mi = g(f"{side}Hand"), g(f"{side}HandThumb1"), g(f"{side}HandMiddle1")
        a = _n(mi - wr)
        u = th - wr
        u = _n(u - np.sum(u * a, -1, keepdims=True) * a)  # thumb side, perpendicular to the hand axis
        pos[f"{side}HandIndex1"] = mi + 0.02 * u
        pos[f"{side}HandPinky1"] = mi - 0.03 * u
        pos[f"{side}HandThumb2"] = th
        pos[f"{side}ToeEnd"] = g(f"{side}ToeBase") + 0.05 * _n(g(f"{side}ToeBase") - g(f"{side}Foot")) * \
            np.array([1.0, 1.0, 0.0])
    names = SOMA26 + list(pos)
    allpos = np.concatenate([j26] + [p[:, None] for p in pos.values()], 1)
    L = soma_landmarks_no_face(names, allpos)
    neck, head = g("Neck1"), g("Head")
    sh = _n(L[:, 11] - L[:, 12])
    head_end = head + 1.2 * (head - neck)
    _fill_face_and_feet(L, head, head_end, head + 0.03 * sh, head - 0.03 * sh,
                        pos["LeftToeEnd"], pos["RightToeEnd"])
    return L


def soma_landmarks_no_face(names, pos):
    ix = {n: i for i, n in enumerate(names)}
    L = np.zeros((len(pos), 33, 3))
    for o, side in enumerate(("Left", "Right")):
        for k, n in ((11, "Arm"), (13, "ForeArm"), (15, "Hand"), (17, "HandPinky1"), (19, "HandIndex1"),
                     (21, "HandThumb2"), (23, "Leg"), (25, "Shin"), (27, "Foot")):
            L[:, k + o] = pos[:, ix[side + n]]
    return L


def to_mediapipe(L_w):
    """Z-up world landmarks -> MediaPipe world-landmark axes, hip-centred (inverse of mp2smpl.to_W)."""
    L = np.asarray(L_w, np.float64)
    L = L - 0.5 * (L[:, 23:24] + L[:, 24:25])
    return L @ MP_TO_W


# ------------------------------------------------------------------------------------------------ SMPL track
def smpl_track(L_w, conv=None):
    """(T, 33, 3) Z-up landmarks -> joints_local (T, 24, 3) [pelvis at 0], quat (T, 4) [wxyz, Z-up, base removed]."""
    conv = conv or smpl_stream.make_converter(live=False)
    mp = to_mediapipe(L_w)
    vis = np.ones(33)
    J = np.zeros((len(mp), 24, 3))
    Q = np.zeros((len(mp), 4))
    for t in range(len(mp)):
        f = conv.convert(mp[t], vis)
        J[t], Q[t] = smpl_stream.zero_pelvis(f["smpl_joints"]), f["body_quat"]
    for t in range(1, len(Q)):  # continuous sign for interpolation
        if np.dot(Q[t], Q[t - 1]) < 0:
            Q[t] = -Q[t]
    return J, Q


def resample_track(J, Q, fps_src, t_dst):
    """Joints linear, root slerp, onto the times t_dst (s)."""
    from scipy.spatial.transform import Slerp

    t_src = np.arange(len(J)) / fps_src
    t_dst = np.clip(t_dst, 0.0, t_src[-1])
    i = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(J) - 2)
    w = ((t_dst - t_src[i]) / (t_src[i + 1] - t_src[i]))[:, None, None]
    Jd = (1 - w) * J[i] + w * J[i + 1]
    Qd = Slerp(t_src, Rot.from_quat(Q[:, [1, 2, 3, 0]]))(t_dst).as_quat()[:, [3, 0, 1, 2]]
    for t in range(1, len(Qd)):
        if np.dot(Qd[t], Qd[t - 1]) < 0:
            Qd[t] = -Qd[t]
    return Jd, Qd


def smpl_entry(J_local, Q, fps, transl_zup=None):
    """motion-lib SMPL PKL content (gear_sonic smpl_motion_file; NOT wrapped in {name: ...}).

    smpl_joints = joints rotated into the world (the motion lib re-applies inv(root): commands.smpl_joints_multi_future
    -> observations.smpl_joints_multi_future_local), pose_aa[:, :3] = Y-up global_orient whose training-time root is Q,
    body pose zero (no encoder input uses it), transl = pelvis position Y-up (height only, informational)."""
    T = len(J_local)
    pose_aa = np.zeros((T, 72), np.float32)
    pose_aa[:, :3] = smpl_stream.root_aa_yup(Q)
    tz = np.zeros((T, 3)) if transl_zup is None else np.asarray(transl_zup, np.float64)
    transl_yup = np.stack([tz[:, 0], tz[:, 2], -tz[:, 1]], 1)
    return {"pose_aa": pose_aa, "transl": transl_yup.astype(np.float32),
            "smpl_joints": smpl_stream.world_joints(J_local, Q).astype(np.float32), "fps": float(fps)}
