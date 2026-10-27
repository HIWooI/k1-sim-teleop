"""GEM-X (SOMA-77 skeleton) as the K1 live perception: landmark mapping, walking intent, saved tracks.

GEM-X (NVlabs; the real-time path of a GEM-X perception folder's demo_webcam.py: VitPose SOMA-77 2D -> GEM regression
over a sliding window -> SOMA forward kinematics, CUDA) gives per frame a full 3D skeleton in the camera frame (metres, x right,
y down, z forward) with an absolute depth, plus the VitPose 2D keypoints it was fitted to. This module turns that into
what the MediaPipe path of k1live.human already consumes, so the arm IK, waist, bow, calibration and the sonic_human
whole-body stream run unchanged:

    det = soma_det(joints_cam, kp2d, img_w, img_h, K)     # {"world", "image", "vis"} in MediaPipe-33 layout + "cam"
    HumanMapper(arm_ik="dir", perception="gemx").process(det, t, img_w, img_h)

* world (33, 3): SOMA joints minus the hip centre (MediaPipe world axes are the camera axes, so no rotation). This is
  where the dir IK (k1live/arm_ik.py) takes the depth component of every arm bone from: GEM-X depth instead of
  MediaPipe's world z.
* image (33, 3): VitPose 2D (the direct measurement) for joints it is confident about, else the GEM-X joint projected
  with K and aligned (scale + shift) to the confident VitPose joints; normalised to the image like MediaPipe. The dir
  IK takes the frontal-plane angle of each bone from here.
* vis (33,): VitPose confidence rescaled so that VIS_REF reads as MediaPipe's 0.5 gate (VitPose is ~0.95 on a clear
  joint, ~0.3-0.5 when guessing).
* cam (33, 3): the same joints in the camera frame (absolute) -> PoseObs.lm_cam -> GemxLocoEstimator.

GemxLocoEstimator is k1live.locomotion.LocoEstimator (same interface, profiles, dead zones, hysteresis) with the
per-frame measurement taken from the 3D skeleton: distance = pelvis z, sideways = pelvis x, body turn = yaw of the 3D
shoulder / hip lines. No focal-length guess and no weak-perspective fit; GEM-X's whole-body prior keeps the lines
sensible in side views where MediaPipe's world landmarks fold.

Saved tracks (offline GEM-X results over a whole clip, or a dump of the real-time worker over a video: k1live/gemx_worker.py
--dump) load with load_track() for the offline reel / evaluation scripts. Pure NumPy: no torch, no GEM-X imports.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from k1live.locomotion import LocoEstimator, Measurement

SOMA_NAMES = (
    "Hips Spine1 Spine2 Chest Neck1 Neck2 Head HeadEnd Jaw LeftEye RightEye LeftShoulder LeftArm LeftForeArm LeftHand "
    "LeftHandThumb1 LeftHandThumb2 LeftHandThumb3 LeftHandThumbEnd LeftHandIndex1 LeftHandIndex2 LeftHandIndex3 "
    "LeftHandIndex4 LeftHandIndexEnd LeftHandMiddle1 LeftHandMiddle2 LeftHandMiddle3 LeftHandMiddle4 LeftHandMiddleEnd "
    "LeftHandRing1 LeftHandRing2 LeftHandRing3 LeftHandRing4 LeftHandRingEnd LeftHandPinky1 LeftHandPinky2 "
    "LeftHandPinky3 LeftHandPinky4 LeftHandPinkyEnd RightShoulder RightArm RightForeArm RightHand RightHandThumb1 "
    "RightHandThumb2 RightHandThumb3 RightHandThumbEnd RightHandIndex1 RightHandIndex2 RightHandIndex3 RightHandIndex4 "
    "RightHandIndexEnd RightHandMiddle1 RightHandMiddle2 RightHandMiddle3 RightHandMiddle4 RightHandMiddleEnd "
    "RightHandRing1 RightHandRing2 RightHandRing3 RightHandRing4 RightHandRingEnd RightHandPinky1 RightHandPinky2 "
    "RightHandPinky3 RightHandPinky4 RightHandPinkyEnd LeftLeg LeftShin LeftFoot LeftToeBase LeftToeEnd RightLeg "
    "RightShin RightFoot RightToeBase RightToeEnd").split()
SOMA = {n: i for i, n in enumerate(SOMA_NAMES)}
SOMA_PARENTS = (-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26,
                27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52,
                53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75)
assert len(SOMA_NAMES) == 77 and len(SOMA_PARENTS) == 77

# MediaPipe Pose 33 <- SOMA joints (several = their mean). The ones the K1 chain reads (shoulders, elbows, wrists,
# hips, knees, ankles) are joint centres in both; face / hand / foot tips are close enough for drawing and SMPL.
_S = SOMA
MP_FROM_SOMA = (
    (_S["LeftEye"], _S["RightEye"]),                     # 0 nose
    (_S["LeftEye"],), (_S["LeftEye"],), (_S["LeftEye"],),     # 1-3 left eye inner / eye / outer
    (_S["RightEye"],), (_S["RightEye"],), (_S["RightEye"],),  # 4-6
    (_S["Head"], _S["LeftEye"]), (_S["Head"], _S["RightEye"]),  # 7-8 ears
    (_S["Jaw"],), (_S["Jaw"],),                          # 9-10 mouth
    (_S["LeftArm"],), (_S["RightArm"],),                 # 11-12 shoulders
    (_S["LeftForeArm"],), (_S["RightForeArm"],),         # 13-14 elbows
    (_S["LeftHand"],), (_S["RightHand"],),               # 15-16 wrists
    (_S["LeftHandPinky1"],), (_S["RightHandPinky1"],),   # 17-18 pinky
    (_S["LeftHandIndex1"],), (_S["RightHandIndex1"],),   # 19-20 index
    (_S["LeftHandThumb2"],), (_S["RightHandThumb2"],),   # 21-22 thumb
    (_S["LeftLeg"],), (_S["RightLeg"],),                 # 23-24 hips
    (_S["LeftShin"],), (_S["RightShin"],),               # 25-26 knees
    (_S["LeftFoot"],), (_S["RightFoot"],),               # 27-28 ankles
    (_S["LeftFoot"],), (_S["RightFoot"],),               # 29-30 heels
    (_S["LeftToeBase"],), (_S["RightToeBase"],),         # 31-32 foot index
)
_MP_IDX = np.zeros((33, 2), int)
_MP_W = np.zeros((33, 2))
for _i, _ix in enumerate(MP_FROM_SOMA):
    _MP_IDX[_i] = (_ix[0], _ix[-1])
    _MP_W[_i] = (0.5, 0.5) if len(_ix) == 2 else (1.0, 0.0)

# body bones for drawing (no fingers, no eyes / jaw)
FINGERS = set(range(15, 39)) | set(range(43, 67)) | {7, 8, 9, 10}
DRAW_BONES = tuple((p, c) for c, p in enumerate(SOMA_PARENTS) if p >= 0 and c not in FINGERS)
ARM_BONES = {(_S["LeftArm"], _S["LeftForeArm"]), (_S["LeftForeArm"], _S["LeftHand"]),
             (_S["RightArm"], _S["RightForeArm"]), (_S["RightForeArm"], _S["RightHand"])}

CONF_FIT = 0.5          # VitPose confidence above which a 2D keypoint is trusted (image landmark, 2D alignment)
VIS_REF = 0.35          # VitPose confidence that maps to MediaPipe visibility 0.5 (k1live.human min_vis)
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 23, 24, 27, 28
TORSO = (L_SH, R_SH, L_HIP, R_HIP)


def to_mp33(x77):
    """(..., 77, D) SOMA array -> (..., 33, D) MediaPipe layout (mean of the mapped joints)."""
    x = np.asarray(x77, np.float64)
    return x[..., _MP_IDX[:, 0], :] * _MP_W[:, 0, None] + x[..., _MP_IDX[:, 1], :] * _MP_W[:, 1, None]


def project(J_cam, K):
    """(77, 3) camera-frame points -> (77, 2) pixels."""
    J = np.asarray(J_cam, np.float64)
    uv = J @ np.asarray(K, np.float64).T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-3)


def fit_2d(uv, kp2d, conf_min=CONF_FIT, min_pts=6):
    """Scale + shift that best maps the GEM-X projection uv onto the confident VitPose keypoints (about the
    principal point it does not matter: the fit is a plain similarity without rotation). -> aligned uv (77, 2)."""
    if kp2d is None:
        return uv
    ok = kp2d[:, 2] >= conf_min
    if ok.sum() < min_pts:
        return uv
    a, b = uv[ok], kp2d[ok, :2]
    ma, mb = a.mean(0), b.mean(0)
    da, db = a - ma, b - mb
    s = float((da * db).sum() / max((da * da).sum(), 1e-9))
    if not (0.5 < s < 2.0):
        s = 1.0
    return (uv - ma) * s + mb


def soma_uv(J_cam, kp2d, K):
    """GEM-X joints projected and aligned to VitPose (77, 2) pixels: what the overlay draws."""
    return fit_2d(project(J_cam, K), None if kp2d is None else np.asarray(kp2d, np.float64))


ARM_CHAINS = ((_S["LeftArm"], _S["LeftForeArm"], _S["LeftHand"]), (_S["RightArm"], _S["RightForeArm"], _S["RightHand"]))
DEPTH_MODES = ("gemx", "fused")


def fuse_arm_depth(J, uv, K, min_frac=0.0):
    """Arms re-built from the 2D keypoints with GEM-X's bone lengths: each bone keeps its image direction and image
    length (uv, px), its depth extent follows from the length (|dz| = sqrt(L^2 - in-plane^2) at the parent's depth),
    and only the SIGN of the depth comes from GEM-X. GEM-X's window regression often puts a forearm forward while the
    image shows it full length (or the reverse); the dir IK then trades the frontal angle for that depth. -> J copy."""
    J = np.array(J, np.float64)
    f = 0.5 * (K[0, 0] + K[1, 1])
    for chain in ARM_CHAINS:
        for a, b in zip(chain[:-1], chain[1:]):
            L = float(np.linalg.norm(J[b] - J[a]))       # GEM-X bone length (identity / shape fit: stable)
            if L < 1e-3:
                continue
            sgn = 1.0 if (J[b, 2] - J[a, 2]) >= 0 else -1.0
            za = J[a, 2]
            dz = J[b, 2] - za
            for _ in range(3):   # image vector -> metres at the bone's mid depth (offset-free: a difference of keypoints)
                inpl = (uv[b] - uv[a]) * (za + 0.5 * dz) / f
                m = float(np.linalg.norm(inpl))
                dz = sgn * np.sqrt(max(L * L - m * m, (min_frac * L) ** 2))
            v = np.concatenate([inpl, [dz]])
            n = float(np.linalg.norm(v))
            shift = J[a] + v * (L / max(n, 1e-9)) - J[b]
            # the whole subtree below b moves with it (hand joints follow the wrist)
            J[b] += shift
            if b == chain[-1]:
                for i in range(15, 39) if b == _S["LeftHand"] else range(43, 67):
                    J[i] += shift
    return J


def soma_det(J_cam, kp2d, img_w, img_h, K, conf_fit=CONF_FIT, vis_ref=VIS_REF, image_src="vitpose", depth="gemx"):
    """One GEM-X frame -> a k1live.human det dict (MediaPipe-33 layout).

    J_cam (77, 3) SOMA joints, camera frame (m); kp2d (77, 3) VitPose u, v (px), confidence or None; K (3, 3).
    image_src: "vitpose" = VitPose 2D where confident, aligned projection elsewhere; "proj" = projection only.
    depth: "gemx" = arm depth straight from GEM-X 3D; "fused" = fuse_arm_depth (2D length + GEM-X depth sign)."""
    J = np.asarray(J_cam, np.float64)
    kp = None if kp2d is None else np.asarray(kp2d, np.float64)
    uv = fit_2d(project(J, K), kp)
    if image_src == "vitpose" and kp is not None:
        use = kp[:, 2:3] >= conf_fit
        uv = np.where(use, kp[:, :2], uv)
    if depth == "fused":
        J = fuse_arm_depth(J, uv, np.asarray(K, np.float64))
    conf = np.ones(77) if kp is None else np.clip(kp[:, 2], 0.0, 1.0)
    cam = to_mp33(J)
    hip = 0.5 * (cam[L_HIP] + cam[R_HIP])
    img = to_mp33(uv) / np.array([max(img_w, 1), max(img_h, 1)])
    zrel = (cam[:, 2] - hip[2]) / max(hip[2], 1e-3)          # like MediaPipe's image z: relative depth, small
    c33 = np.minimum(conf[_MP_IDX[:, 0]], conf[_MP_IDX[:, 1]])
    vis = np.clip(0.5 * c33 / max(vis_ref, 1e-3), 0.0, 1.0)
    return {"world": cam - hip, "image": np.concatenate([img, zrel[:, None]], 1), "vis": vis, "cam": cam,
            "soma_uv": uv, "soma_conf": conf}


# ------------------------------------------------------------------------------------------------ walking intent
def measure_cam(obs, min_vis=0.5):
    """PoseObs with lm_cam (33, 4: camera-frame metres + visibility) -> Measurement or None (torso not trusted).
    Also attached: z_pel / x_pel (pelvis), up (unit torso axis), ankle (ankle midpoint, m) and ray (its unit direction)
    when both ankles are visible - the inputs of the ground-plane distance (GemxLocoEstimator dist_src="ground")."""
    cam = obs["lm_cam"] if not isinstance(obs, dict) else obs.get("lm_cam")
    if cam is None:
        return None
    cam = np.asarray(cam, np.float64)
    vis = cam[:, 3] if cam.shape[1] > 3 else np.ones(len(cam))
    if np.any(vis[list(TORSO)] < min_vis) or not np.all(np.isfinite(cam[list(TORSO), :3])):
        return None
    pel = 0.5 * (cam[L_HIP, :3] + cam[R_HIP, :3])
    if pel[2] <= 0.2:
        return None
    d = cam[L_SH, :3] - cam[R_SH, :3]
    dh = cam[L_HIP, :3] - cam[R_HIP, :3]
    t = obs["t"] if not isinstance(obs, dict) else obs.get("t", 0.0)
    m = Measurement(float(t), float(pel[2]), float(pel[0]), math.atan2(d[2], d[0]), math.atan2(dh[2], dh[0]),
                    1.0 / float(pel[2]), True, 0.0)
    m.z_pel, m.x_pel = m.z, m.x
    up = 0.5 * (cam[L_SH, :3] + cam[R_SH, :3]) - pel
    m.up = up / max(np.linalg.norm(up), 1e-9)
    m.ankle = m.ray = None
    if min(vis[L_ANK], vis[R_ANK]) >= min_vis and np.all(np.isfinite(cam[[L_ANK, R_ANK], :3])):
        a = 0.5 * (cam[L_ANK, :3] + cam[R_ANK, :3])
        if a[2] > 0.2:
            m.ankle, m.ray = a, a / np.linalg.norm(a)
    return m


class GemxLocoEstimator(LocoEstimator):
    """LocoEstimator on GEM-X's camera-frame skeleton (see the module docstring). Same profiles and parameters.
    Frames without lm_cam (a MediaPipe frame) fall back to the MediaPipe measurement.

    dist_src: where the distance / sideways position comes from
      "pelvis"  GEM-X's pelvis in the camera frame (its depth = the network's whole-body size / pose fit)
      "ground"  the ray through the ankles (their image position: VitPose-anchored) cut with the floor plane fixed at
                home (normal = the standing torso axis, offset = the ankles at home): a static camera on a flat stage
                needs no size cue at all, so raised arms or a bow cannot move it. Frames without both ankles use the
                pelvis plus the pelvis-to-ground offset measured at home."""

    DIST_SRC = ("pelvis", "ground")

    def __init__(self, mirror=True, profile="visitor", dist_src="pelvis", **params):
        if dist_src not in self.DIST_SRC:
            raise ValueError(f"dist_src {dist_src!r} not in {self.DIST_SRC}")
        self.dist_src = dist_src
        self.plane = None        # (unit normal n, offset d): n . X = d for floor points X (ankle height)
        self.off = (0.0, 0.0)    # ground - pelvis (z, x) at home
        super().__init__(mirror=mirror, profile=profile, **params)

    def forget_home(self):
        super().forget_home()
        self.plane = None

    def _ground(self, m):
        """m.z / m.x from the floor plane (in place); pelvis + home offset when the ankles are not usable."""
        n, d = self.plane
        c = float(n @ m.ray) if getattr(m, "ray", None) is not None else 0.0
        if abs(c) < 0.05 or d / c <= 0.2:
            m.z, m.x = m.z_pel + self.off[0], m.x_pel + self.off[1]
            return m
        X = (d / c) * m.ray
        m.z, m.x = float(X[2]), float(X[0])
        return m

    def _measure(self, obs):
        if obs is None:            # nobody tracked (update(None)): the base class handles the loss
            return None
        cam = obs["lm_cam"] if not isinstance(obs, dict) else obs.get("lm_cam")
        if cam is None:
            return super()._measure(obs)
        m = measure_cam(obs, self.p["min_vis"])
        if m is not None and self.dist_src == "ground" and self.plane is not None:
            self._ground(m)
        return m

    def _set_home(self, ms):
        if self.dist_src == "ground":
            ups = [m.up for m in ms if getattr(m, "up", None) is not None]
            feet = [m for m in ms if getattr(m, "ankle", None) is not None]
            if ups and feet:
                n = np.mean(ups, axis=0)
                n = n / np.linalg.norm(n)
                self.plane = (n, float(np.mean([n @ m.ankle for m in feet])))
                g = [self._ground(m) for m in feet]
                self.off = (float(np.mean([m.z - m.z_pel for m in g])), float(np.mean([m.x - m.x_pel for m in g])))
                ms = [self._ground(m) if getattr(m, "z_pel", None) is not None else m for m in ms]
        super()._set_home(ms)


# ------------------------------------------------------------------------------------------------ saved tracks
@dataclass
class GemxTrack:
    """GEM-X output aligned to the frames of one video (index = video frame). valid[k] False = no person."""
    J: np.ndarray            # (N, 77, 3) camera frame, m
    kp2d: np.ndarray         # (N, 77, 3) VitPose px + confidence (zeros when absent)
    valid: np.ndarray        # (N,) bool
    K: np.ndarray            # (3, 3)
    wh: tuple                # (w, h) of the image GEM-X saw
    fps: float
    kind: str                # "offline" (whole clip at once) | "stream" (real-time worker, causal window)
    meta: dict
    body_pose: Optional[np.ndarray] = None   # (N, 228) SOMA articulation (distilled MLP input) if saved

    def det(self, k, scale_to=None, **kw):
        """soma_det of video frame k (None if no person). scale_to=(w, h): pixel coords of another rendition of the
        same video (e.g. the clip GEM-X saw was downscaled)."""
        if k < 0 or k >= len(self.J) or not self.valid[k]:
            return None
        w, h = self.wh
        K, kp = self.K, self.kp2d[k]
        if scale_to is not None and tuple(scale_to) != (w, h):
            sx, sy = scale_to[0] / w, scale_to[1] / h
            K = np.diag([sx, sy, 1.0]) @ K
            kp = kp * np.array([sx, sy, 1.0])
            w, h = scale_to
        return soma_det(self.J[k], kp, w, h, K, **kw)


def load_track(path, n_video=None) -> GemxTrack:
    """path: a GEM-X clip folder (soma_joints.npz [+ meta.json], an offline run) or an .npz from
    k1live/gemx_worker.py --dump (kind "stream"). Frames are mapped to video frames (source_frame_index; between
    two GEM-X frames the earlier one holds, like a camera feed)."""
    p = Path(path)
    if p.is_dir():
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        p = p / "soma_joints.npz"
    else:
        meta = {}
    z = np.load(p, allow_pickle=True)
    kind = str(z["kind"]) if "kind" in z.files else "offline"
    if "meta" in z.files:
        try:
            meta = {**meta, **json.loads(str(z["meta"]))}
        except ValueError:
            pass
    J = np.asarray(z["joints_incam"], np.float64)
    kp = np.asarray(z["kp2d_vitpose"], np.float64) if "kp2d_vitpose" in z.files else np.zeros(J.shape)
    valid = np.asarray(z["valid"], bool) if "valid" in z.files else np.isfinite(J[:, 0, 0])
    src = np.asarray(z["source_frame_index"]) if "source_frame_index" in z.files else np.arange(len(J))
    wh = tuple(int(v) for v in z["image_wh"])
    fps = float(z["fps"])
    bp = np.asarray(z["body_pose"], np.float64) if "body_pose" in z.files else None
    n = int(n_video if n_video is not None else (src.max() + 1 if len(src) else 0))
    idx = np.searchsorted(src, np.arange(n), side="right") - 1
    ok = idx >= 0
    idx = np.clip(idx, 0, max(len(src) - 1, 0))
    J2 = np.where(ok[:, None, None], J[idx], np.nan)
    v2 = ok & valid[idx]
    return GemxTrack(J2, kp[idx], v2, np.asarray(z["K"], np.float64), wh, fps, kind, meta,
                     None if bp is None else bp[idx])


def draw_soma(frame_bgr, uv, conf=None, color=(40, 150, 255), conf_min=0.0, arms_color=None, thickness=None):
    """GEM-X SOMA body skeleton (77-joint pixel coords uv) on a BGR image (copy). Orange by default (MediaPipe's
    overlay is green), arms optionally in their own colour."""
    import cv2

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    th = thickness or max(2, int(round(min(w, h) / 220)))
    for p, c in DRAW_BONES:
        if conf is not None and min(conf[p], conf[c]) < conf_min:
            continue
        col = arms_color if (arms_color is not None and (p, c) in ARM_BONES) else color
        cv2.line(out, (int(uv[p, 0]), int(uv[p, 1])), (int(uv[c, 0]), int(uv[c, 1])), col, th, cv2.LINE_AA)
    for j in {i for b in DRAW_BONES for i in b}:
        cv2.circle(out, (int(uv[j, 0]), int(uv[j, 1])), th + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return out
