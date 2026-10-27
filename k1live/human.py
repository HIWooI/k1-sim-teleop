"""Person -> K1 targets, one camera frame at a time (used by k1live.sender).

Per frame: MediaPipe landmarks (sonic/pose_mp.PoseTracker) -> One-Euro filter -> the same retargeting as the offline
reel (k1/run_video.py: person's torso frame, CPU position IK on the K1 URDF, k1/cpu_kinematics.py) -> 10 arm joints,
plus torso yaw relative to the pelvis (waist), forward lean (bow) and the pelvis offset from the calibrated home
position (root). Wrist roll stays at the relaxed default (MediaPipe sees no forearm twist), as in the reel.
arm_ik="dir" swaps the position IK for k1live/arm_ik.py (bone directions, frontal-plane angles from the image
landmarks, its own One-Euro filters; docs/architecture.md). Default still "shape18" (the position IK; the name comes
from the project it was ported from and is kept for config compatibility).

Mirror (default on, protocol.py): the robot moves like the person's reflection - the person's right arm drives the
robot's left arm (joint-space left/right swap with roll/yaw negated, exact for the symmetric K1 arms), turning or
stepping to the person's right turns / moves the robot to its left. Everything returned is already in robot terms.

The walking intent is k1live.locomotion.LocoEstimator, imported if present. Its input is PoseObs, the raw
MediaPipe output of one frame:

    est = LocoEstimator(mirror=True)      # est.mirror is reassigned when the operator toggles mirror
    est.update(obs) -> {"vx","vy","wz"}   # every frame with a tracked person (not called otherwise); the estimator
                                          # homes itself once the person stands still 0.8 s
    est.calibrate(obs)                    # only on the operator's CTRL calibrate: landmarks = mean over a still window
    est.root() -> {"x","y","yaw"}         # HumanFrame.root (weak-perspective offset from home; a bow is not a step)
    forget_home                           # source switched: home again on its own

Whole-body stream for the sonic_human backend (HumanFrame.smpl, k1live/smpl_stream.py): the same MediaPipe frame
(One-Euro filtered world landmarks) through sonic/mp2smpl with the settings the K1 SMPL encoder was trained with,
levelled / yaw-zeroed by the same stillness-gated calibration as the arms, a small dead band against MediaPipe
jitter, then mirrored like everything else.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from k1.cpu_kinematics import K1CpuRetargeter
from k1.sim import URDF
from k1live import smpl_stream
from k1live.arm_ik import DirRetargeter
from k1live.protocol import ARM_JOINTS
from sonic.mp2smpl import BoneDeadband, OneEuro

# MediaPipe landmark ids
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
ARM_LM = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}
LEG_LM = (25, 26, 27, 28)
CONF_LM = (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP)

# MediaPipe landmarks -> the 14 IK targets (the original perception/demo_cpu.py MP_MAP, same as k1/run_video.py)
MP_MAP = ((23, 24), (11, 12), (11,), (13,), (15,), (12,), (14,), (16,), (23,), (25,), (27,), (24,), (26,), (28,))

# K1 arm defaults (loco_armdriven_r2 default_position, ARM_JOINTS order): the relaxed pose, wrist roll source
ARM_DEFAULT = np.array([0.2, 0.25, 0.0, 0.95, 0.15, 0.2, -0.25, 0.0, 0.95, -0.15])
WRIST = [ARM_JOINTS.index("left_wrist_roll_joint"), ARM_JOINTS.index("right_wrist_roll_joint")]
TORSO_M = 0.50          # shoulder-centre -> hip-centre of an adult (m), scale for the root offset
FOCAL_PER_H = 0.96      # focal length / image height of a ~55 deg vertical-FOV webcam (root depth guess)
ARM_IK_METHODS = ("shape18", "dir")
# dir IK settings chosen in docs/results.md (used when HumanMapper(arm_ik="dir") gets no explicit ones): its own
# faster One-Euro (less lag; the dir IK has no EMA) and the overhead balance envelope (30 deg forward tilt of raised arms)
DIR_ARM_FILTER = (1.5, 1.0)
DIR_IK_KW = {"overhead_deg": 30.0}
IMG_M = 2.0             # image height in metres of a full-body framing (unit of the dir IK's image-landmark filter)


@dataclass
class PoseObs:
    """Raw MediaPipe result of one camera frame (the LocoEstimator input). Also readable as obs["t"] etc."""
    t: float                 # capture time, time.time()
    img_w: int
    img_h: int
    lm_img: np.ndarray       # (33, 4) pose_landmarks: x, y normalised image coords (raw, unmirrored camera image),
                             #          z relative depth, visibility (= min(visibility, presence))
    lm_world: np.ndarray     # (33, 4) pose_world_landmarks: x, y, z metres (hip-centred, y down, z away from the
                             #          camera), visibility
    lm_cam: Optional[np.ndarray] = None  # (33, 4) camera-frame metres (x right, y down, z forward) + visibility when
                                         # the perception has absolute 3D (GEM-X, k1live.gemx); None for MediaPipe

    def __getitem__(self, k):
        return getattr(self, k)

    def keys(self):
        return ("t", "img_w", "img_h", "lm_img", "lm_world", "lm_cam")

    @staticmethod
    def mean(obs: list) -> "PoseObs":
        cam = None if any(o.lm_cam is None for o in obs) else np.mean([o.lm_cam for o in obs], axis=0)
        return PoseObs(obs[-1].t, obs[-1].img_w, obs[-1].img_h, np.mean([o.lm_img for o in obs], axis=0),
                       np.mean([o.lm_world for o in obs], axis=0), cam)


def obs_from_det(det: dict, t: float, img_w: int, img_h: int) -> PoseObs:
    vis = det["vis"][:, None]
    cam = det.get("cam")
    return PoseObs(t, int(img_w), int(img_h), np.concatenate([det["image"], vis], 1),
                   np.concatenate([det["world"], vis], 1),
                   None if cam is None else np.concatenate([np.asarray(cam)[:, :3], vis], 1))


def mirror_arms(q):
    """Robot arm joints of the person's reflection: left <-> right, roll / yaw / wrist roll negated."""
    q = np.asarray(q, np.float64)
    sgn = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
    return np.concatenate([q[5:] * sgn, q[:5] * sgn])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def k1_axes(w):
    """MediaPipe world (x = person's left for a front camera, y down, z away from the camera) -> K1 axes
    (x = person's forward when facing the camera, y = person's left, z up)."""
    w = np.asarray(w)[..., :3]
    return np.stack((-w[..., 2], w[..., 0], -w[..., 1]), axis=-1)


def line_yaw(v):
    """Yaw of a left-pointing body line (shoulders / hips, K1 axes): 0 when it points along +y."""
    return float(np.arctan2(-v[0], v[1]))


def mp_targets(world, vis):
    pts = np.stack([world[list(ix)].mean(axis=0) for ix in MP_MAP])
    conf = np.array([vis[list(ix)].min() for ix in MP_MAP])
    return pts, conf


def torso_frame(pts):
    """k1/run_video.torso_frame: the points in the person's own torso frame, still in MediaPipe axes."""
    from k1.cpu_kinematics import TARGET_NAMES

    k1 = np.stack((-pts[:, 2], pts[:, 0], -pts[:, 1]), axis=-1)
    hip, chest = k1[0], k1[1]
    y = k1[TARGET_NAMES.index("LeftArm")] - k1[TARGET_NAMES.index("RightArm")]
    y /= max(np.linalg.norm(y), 1e-6)
    z = chest - hip
    z -= y * (z @ y)
    z /= max(np.linalg.norm(z), 1e-6)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    local = (k1 - hip) @ R
    return np.stack((local[:, 1], -local[:, 2], -local[:, 0]), axis=-1)


class JumpGate:
    """Single-person continuity: drop a detection whose hip centre jumped more than max_jump (normalised image
    units) from the tracked one (probably someone else / a glitch), but accept the new position after `patience`
    consecutive rejections, so a bad first lock (e.g. a stale MediaPipe ROI after a source switch) or the person
    re-entering elsewhere cannot lock tracking out. Replaces PoseTracker's own gate (set its max_jump to inf)."""

    def __init__(self, max_jump=0.12, patience=8, forget=15):
        self.max_jump, self.patience, self.forget = max_jump, patience, forget
        self.center, self.rejected, self.missed = None, 0, 0
        self.relocked = False   # this call accepted a jump (probably someone else): the caller re-homes walking

    def reset(self):
        self.center, self.rejected, self.missed = None, 0, 0
        self.relocked = False

    def __call__(self, det):
        self.relocked = False
        if det is None:
            self.missed += 1
            if self.missed >= self.forget:  # nobody for ~0.5 s: whoever comes next is accepted at once
                self.center = None
            return None
        self.missed = 0
        c = 0.5 * (np.asarray(det["image"])[L_HIP, :2] + np.asarray(det["image"])[R_HIP, :2])
        if self.center is not None and np.linalg.norm(c - self.center) > self.max_jump:
            self.rejected += 1
            if self.rejected < self.patience:
                return None
            self.relocked = True
        self.center, self.rejected = c, 0
        return det


class LocoAdapter:
    """Optional LocoEstimator. Missing module or a failing call -> loco None, never an exception.
    perception "gemx": k1live.gemx.GemxLocoEstimator (same interface, GEM-X camera-frame pelvis / body lines)."""

    def __init__(self, mirror=True, profile="visitor", perception="mediapipe", profile_kw=None, **kw):
        self.est, self.error = None, ""
        kw = {**kw, **((profile_kw or {}).get(profile) or {})}   # per-profile overrides (venue tuning)
        try:
            if perception == "gemx":
                from k1live.gemx import GemxLocoEstimator as LocoEstimator
            else:
                from k1live.locomotion import LocoEstimator  # noqa: WPS433 (optional)

            self.est = LocoEstimator(mirror=mirror, profile=profile, **kw)
        except Exception as e:  # ImportError or a constructor problem: run without walking intent
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self):
        return self.est is not None

    def set_mirror(self, mirror):
        if self.est is not None:
            self.est.mirror = bool(mirror)

    def forget_home(self):
        """A new scene (source switch): the estimator homes again on its own once the person stands still."""
        if self.est is None:
            return
        try:
            if hasattr(self.est, "forget_home"):
                self.est.forget_home()
            else:
                self.est.home, self.est.calibrated = None, False
                self.est.reset()
        except Exception as e:
            self.error = f"forget_home: {type(e).__name__}: {e}"

    def root(self) -> Optional[dict]:
        if self.est is None or not hasattr(self.est, "root"):
            return None
        try:
            r = self.est.root()
            return None if r is None else {k: round(float(r[k]), 4) for k in ("x", "y", "yaw")}
        except Exception as e:
            self.error = f"root: {type(e).__name__}: {e}"
            return None

    def calibrate(self, obs):
        if self.est is not None:
            try:
                self.est.calibrate(obs)
            except Exception as e:
                self.error = f"calibrate: {type(e).__name__}: {e}"

    def update(self, obs) -> Optional[dict]:
        if self.est is None:
            return None
        try:
            r = self.est.update(obs)
            return None if r is None else {k: float(r.get(k, 0.0)) for k in ("vx", "vy", "wz")}
        except Exception as e:
            self.error = f"update: {type(e).__name__}: {e}"
            return None


class HumanMapper:
    """Stateful per-frame mapping. process() returns the HumanFrame fields (without seq / t_sent)."""

    def __init__(self, mirror=True, ik_smoothing=0.6, max_nfev=20, min_cutoff=1.0, beta=0.4, calib_frames=15,
                 auto_calibrate=True, min_vis=0.5, loco=True, still_m=0.03, loco_profile="visitor",
                 smpl=True, smpl_deadband=(1.5, 0.8), arm_ik="shape18", arm_ik_kw=None, arm_filter=None,
                 perception="mediapipe", loco_kw=None):
        """arm_ik: "shape18" = K1CpuRetargeter position IK (ik_smoothing / max_nfev apply), "dir" = k1live.arm_ik bone
        directions, frontal-plane angles from the image landmarks (arm_ik_kw: DirRetargeter / DirArmIK options).
        arm_filter: One-Euro (min_cutoff, beta) of the dir IK's own world + image landmark filters (default
        DIR_ARM_FILTER; the main filter keeps feeding waist / bow / root / smpl). arm_ik_kw default DIR_IK_KW.
        perception: "mediapipe" (PoseTracker dets) or "gemx" (k1live.gemx.soma_det dets: the same landmark layout from
        the GEM-X SOMA skeleton, plus camera-frame 3D for the walking intent). loco_kw: extra walking-estimator
        arguments (e.g. GemxLocoEstimator dist_src)."""
        self.mirror = mirror
        self.perception = perception
        self.loco_kw = dict(loco_kw or {})
        if arm_ik not in ARM_IK_METHODS:
            raise ValueError(f"arm_ik {arm_ik!r} not in {ARM_IK_METHODS}")
        self.arm_ik = arm_ik
        self.ret = K1CpuRetargeter(URDF, max_nfev=max_nfev, smoothing=ik_smoothing)
        self._order = [self.ret.robot.joint_names.index(n) for n in ARM_JOINTS]
        self.dir_ret = DirRetargeter(URDF, **(DIR_IK_KW if arm_ik_kw is None else arm_ik_kw)) if arm_ik == "dir" else None
        self.filt_args = (min_cutoff, beta)
        self.filt = OneEuro(*self.filt_args)
        self.arm_filt_args = tuple(DIR_ARM_FILTER if arm_filter is None else arm_filter)
        self._new_arm_filters()
        self.calib_frames = calib_frames
        self.auto_calibrate = auto_calibrate
        self.still_m = still_m
        self._calib_explicit = False
        self.auto_calibrate = auto_calibrate
        self.min_vis = min_vis
        self.loco_profile = loco_profile
        self.loco = LocoAdapter(mirror, loco_profile, perception, **self.loco_kw) if loco else None
        self.home = None              # calibration result
        self._calib = None            # list of (obs, measures) while calibrating
        self._last_arm = None         # person-terms arm joints (before mirror), per-side hold
        self._last_t = None
        self.ik_ms = 0.0
        self.smpl_on, self.smpl_deadband = bool(smpl), smpl_deadband
        self.smpl_ms = 0.0
        self._smpl_conv = self._smpl_db = None
        if self.smpl_on:
            self._reset_smpl(keep_calibration=False)
        if auto_calibrate:
            self.request_calibration(explicit=False)

    # ------------------------------------------------------------------------------------------------ control
    @property
    def calibrated(self):
        return self.home is not None

    def request_calibration(self, explicit=True):
        """explicit = the operator asked (CTRL calibrate): also re-homes the walking estimator. Automatic calibration
        (start-up, new source) leaves the estimator to home itself when the person stands still."""
        self._calib = []
        self._calib_explicit = explicit

    def on_cut(self):
        """The same video looped (fake camera): the person jumps back to the clip's first position. Not a step - the
        walking estimator homes again; arms and calibration are unaffected."""
        self._last_t = None
        if self.loco is not None:
            self.loco.forget_home()
        if self.smpl_on:
            self._reset_smpl(keep_calibration=True)

    def set_loco_profile(self, profile):
        """Operator switched the walking profile (visitor / presenter): a fresh estimator, homed again on its own."""
        self.loco_profile = profile
        if self.loco is not None:
            self.loco = LocoAdapter(self.mirror, profile, self.perception, **self.loco_kw)

    def on_relock(self):
        """JumpGate followed a jump to another position (most likely another person: an audience member, a panel
        guest): the jump is not a step - the walking estimator homes again. Arms just follow the new person."""
        self.on_cut()

    def new_scene(self):
        """Source switched: new person / framing. Forget both homes and calibrate again automatically."""
        self.home = None
        self._last_arm = None
        self._last_t = None
        self._calib = None
        if self.dir_ret is not None:
            self.dir_ret.reset()
        if self.auto_calibrate:
            self.request_calibration(explicit=False)
        if self.loco is not None:
            self.loco.forget_home()
        if self.smpl_on:
            self._reset_smpl(keep_calibration=False)

    def set_mirror(self, mirror):
        self.mirror = bool(mirror)
        if self.loco is not None:
            self.loco.set_mirror(self.mirror)

    # ------------------------------------------------------------------------------------------------ helpers
    def _new_arm_filters(self):
        self.filt_arm_w = OneEuro(*self.arm_filt_args)
        self.filt_arm_i = OneEuro(*self.arm_filt_args)

    def _reset_smpl(self, keep_calibration=True):
        R_cal = self._smpl_conv.R_cal.copy() if (keep_calibration and self._smpl_conv is not None) else None
        self._smpl_conv = smpl_stream.make_converter(live=True)
        if R_cal is not None:
            self._smpl_conv.R_cal = R_cal
        self._smpl_db = BoneDeadband(*self.smpl_deadband) if self.smpl_deadband else None

    def _smpl(self, world_f, vis, t, arm_q_robot):
        """HumanFrame.smpl of one frame: root-local SMPL-24 (pelvis at 0) + root quat, mirrored like arm_q."""
        t0 = time.perf_counter()
        f = self._smpl_conv.convert(world_f, vis, t)
        if self._smpl_db is not None:
            f = self._smpl_db(f)
        J, q = smpl_stream.zero_pelvis(f["smpl_joints"]), np.asarray(f["body_quat"], np.float64)
        if self.mirror:
            J, q, _ = smpl_stream.mirror(J, q)
        wrist = [arm_q_robot[WRIST[0]], arm_q_robot[WRIST[1]]]
        self.smpl_ms = (time.perf_counter() - t0) * 1e3
        return smpl_stream.to_msg(J, q, wrist)

    def _measures(self, obs: PoseObs, world_f):
        """Body measures in person terms (before mirror): image hip centre, torso length px, yaws, lean."""
        im, vis = obs.lm_img, obs.lm_world[:, 3]
        px = im[:, :2] * np.array([obs.img_w, obs.img_h])
        hips_ok = min(vis[L_HIP], vis[R_HIP]) >= self.min_vis
        sh_c = 0.5 * (px[L_SH] + px[R_SH])
        hip_c = 0.5 * (px[L_HIP] + px[R_HIP])
        torso_px = float(np.linalg.norm(sh_c - hip_c))
        k = k1_axes(world_f)
        sh_yaw = line_yaw(k[L_SH] - k[R_SH])
        hip_yaw = line_yaw(k[L_HIP] - k[R_HIP])
        up = 0.5 * (k[L_SH] + k[R_SH]) - 0.5 * (k[L_HIP] + k[R_HIP])
        c, s = np.cos(hip_yaw), np.sin(hip_yaw)
        fwd = c * up[0] + s * up[1]  # along the person's forward axis (hip line rotated -90 deg)
        lean = float(np.arctan2(fwd, up[2]))
        return {"hips_ok": bool(hips_ok), "x_img": float(hip_c[0] if hips_ok else sh_c[0]), "torso_px": torso_px,
                "sh_m": sh_c * TORSO_M / max(torso_px, 1.0),
                "hip_yaw": hip_yaw, "waist": float(_wrap(sh_yaw - hip_yaw)), "lean": lean}

    def _arms(self, world_f, vis, img_f=None):
        t0 = time.perf_counter()
        if self.dir_ret is not None:
            arm = self.dir_ret.solve(world_f, img_f)
        else:
            pts, conf = mp_targets(world_f, vis)
            q, _ = self.ret.solve(self.ret.scale_human_targets(torso_frame(pts)), conf)
            arm = q[self._order]
        self.ik_ms = (time.perf_counter() - t0) * 1e3
        arm[WRIST] = ARM_DEFAULT[WRIST]
        if self._last_arm is None:
            self._last_arm = ARM_DEFAULT.copy()
        for si, side in enumerate(("left", "right")):  # an arm out of view holds its last pose
            if min(vis[i] for i in ARM_LM[side]) >= self.min_vis:
                self._last_arm[si * 5:(si + 1) * 5] = arm[si * 5:(si + 1) * 5]
        return self._last_arm.copy()

    # ------------------------------------------------------------------------------------------------ main
    def process(self, det: Optional[dict], t: float, img_w: int, img_h: int) -> dict:
        """det: PoseTracker.process() result or None. Returns HumanFrame fields + "obs" (PoseObs or None)."""
        out = {"t_capture": t, "present": False, "conf": 0.0, "arm_q": None, "waist_yaw": None, "bow": None,
               "root": None, "legs_visible": False, "loco": None, "calibrated": self.calibrated, "obs": None,
               "smpl": None}
        if det is None:
            self._last_t = None
            return out
        obs = obs_from_det(det, t, img_w, img_h)
        out["obs"] = obs
        vis = det["vis"]
        conf = float(np.mean(vis[list(CONF_LM)]))
        out["conf"] = round(conf, 3)
        if min(vis[L_SH], vis[R_SH]) < self.min_vis:
            self._last_t = None
            return out
        if self._last_t is None or t - self._last_t > 0.5:  # re-acquired: restart the filter, no stale velocity
            self.filt = OneEuro(*self.filt_args)
            self._new_arm_filters()
        self._last_t = t
        world_f = self.filt(det["world"], t)
        if self.dir_ret is not None:
            # own filters (arm_filter); image landmarks aspect-correct, in ~metres (IMG_M per image height) so one
            # beta means the same speed for both
            arm_w = world_f if self.arm_filt_args == self.filt_args else self.filt_arm_w(det["world"], t)
            img = np.asarray(det["image"])[:, :2] * np.array([img_w / max(img_h, 1), 1.0]) * IMG_M
            arm = self._arms(arm_w, vis, self.filt_arm_i(img, t))
        else:
            arm = self._arms(world_f, vis)
        m = self._measures(obs, world_f)
        out["present"] = True
        out["legs_visible"] = bool(min(vis[i] for i in LEG_LM) >= self.min_vis)

        if self._calib is not None and min(vis[L_SH], vis[R_SH]) >= 0.6:
            self._calib.append((obs, m))
            if len(self._calib) >= self.calib_frames:
                if self._still([mm for _, mm in self._calib]):
                    self._finish_calibration()
                else:  # still walking in / moving: slide the window
                    self._calib.pop(0)
        sgn = -1.0 if self.mirror else 1.0
        out["arm_q"] = [float(v) for v in (mirror_arms(arm) if self.mirror else arm)]
        if self.smpl_on:
            out["smpl"] = self._smpl(world_f, vis, t, out["arm_q"])
        if self.home is not None:
            h = self.home
            waist = float(np.clip(m["waist"] - h["waist"], -0.6, 0.6))
            out["waist_yaw"] = round(sgn * waist, 4)
            out["bow"] = round(float(np.clip(m["lean"] - h["lean"], -0.3, 0.8)), 4)
            if m["torso_px"] > 1.0 and h["torso_px"] > 1.0:
                s = m["torso_px"] / h["torso_px"]
                d_home = FOCAL_PER_H * img_h * TORSO_M / h["torso_px"]
                fwd = d_home * (1.0 - 1.0 / s)                     # toward the camera = person's forward
                left = (m["x_img"] - h["x_img"]) * TORSO_M / m["torso_px"]  # image right = person's left
                yaw = float(_wrap(m["hip_yaw"] - h["hip_yaw"]))
                out["root"] = {"x": round(float(fwd), 3), "y": round(float(sgn * left), 3), "yaw": round(sgn * yaw, 4)}
        if self.loco is not None and self.loco.available:
            out["loco"] = self.loco.update(obs)  # zeros until the estimator has a home (it homes itself)
            r = self.loco.root()  # the estimator's weak-perspective offset (a bow does not look like a step)
            if r is not None:
                out["root"] = r
        out["calibrated"] = self.calibrated
        return out

    def _still(self, ms):
        """Shoulder centre within still_m (metres, scaled by the torso length) and torso length within 3 %."""
        sh = np.array([m["sh_m"] for m in ms])
        tp = np.array([m["torso_px"] for m in ms])
        return bool(np.max(np.linalg.norm(sh - sh.mean(0), axis=1)) < self.still_m
                    and np.std(tp) < 0.03 * max(np.mean(tp), 1.0))

    def _finish_calibration(self):
        obs = [o for o, _ in self._calib]
        ms = [m for _, m in self._calib]
        self.home = {k: float(np.median([m[k] for m in ms])) for k in ("x_img", "torso_px", "hip_yaw", "waist", "lean")}
        self.home["hips_ok"] = all(m["hips_ok"] for m in ms)
        self._calib = None
        if self.loco is not None and self._calib_explicit:
            self.loco.calibrate(PoseObs.mean(obs))
        if self.smpl_on:  # level the camera / zero the yaw of the whole-body stream on the same still window
            self._smpl_conv.calibrate([o.lm_world for o in obs])
            if self._smpl_db is not None:
                self._smpl_db.reset()
