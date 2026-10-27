"""MediaPipe Pose (33 world landmarks) -> approximate SONIC SMPL-24 stream fields.

Why not just copy landmark positions: the SONIC smpl encoder was trained on joints from
compute_human_joints() with ONE fixed neutral rest skeleton (human_joints_info.pkl). So we
estimate *directions* from MediaPipe and rebuild the 24 joints with the canonical SMPL bone
lengths — the input then always has an in-distribution skeleton, whatever the operator's
size or the camera distance.

Frames
  MediaPipe world: metres, origin = hip midpoint, x = image right, y = image down,
                   z = away from the camera.
  W (Z-up world):  X = -z_mp (towards the camera), Y = x_mp, Z = -y_mp.
                   An operator facing the camera has heading 0 (faces +X), like the robot.
  robot-local:     x forward, y left, z up  (SMPL rest axes: X left, Y up, Z forward,
                   so v_local = (v_Z, v_X, v_Y), i.e. the [.5,.5,.5,.5] "SMPL base" rotation)

Landmark -> SMPL-24 mapping (indices: MediaPipe BlazePose 33)
  pelvis        : midpoint of hips (23,24)      -> origin; root frame (see below)
  l/r_hip       : rigid pelvis offsets (rest), rotated by the root frame
  spine1/2/3    : rest offsets rotated by slerp(root, chest, 0, 1/3, 2/3) (spine interpolation)
  neck, collars : rest offsets rotated by the chest frame
  head          : neck + rest offset rotated by the head frame (ears 7/8 + nose 0)
  l/r_shoulder  : collar + rest offset rotated by the chest frame (rigid shoulder girdle)
  l/r_elbow     : shoulder + |bone| * dir(13-11 / 14-12)
  l/r_wrist     : elbow    + |bone| * dir(15-13 / 16-14)
  l/r_hand      : SMPL-X thumb tip (joints 39/54): wrist + R_hand * rest offset,
                  R_hand from forearm direction + palm normal (wrist 15/16, pinky 17/18, index 19/20)
  l/r_knee      : hip   + |bone| * dir(25-23 / 26-24)
  l/r_ankle     : knee  + |bone| * dir(27-25 / 28-26)
  l/r_foot      : ankle + |bone| * dir(31-27 / 32-28)  (foot_index)
Root frame: left axis = hip line (23 - 24); up = normalize((1-a) * world_up + a * trunk_up),
  a = root_pitch_share (default 0.5): MediaPipe has no separate pelvis tilt, so the trunk lean
  (hip-mid -> shoulder-mid) is split between the root and the spine chain.
Chest frame: left axis = shoulder line (11 - 12); up = trunk_up.
Output (what soma_to_smpl.py / process_smpl_joints stream):
  body_quat   = root orientation (Z-up, SMPL base removed) wxyz
  smpl_joints = quat_apply(inv(body_quat), joints_W)   with the pelvis at the SMPL rest offset
                J[0] ("j0", default: exact training / PICO convention — compute_human_joints keeps
                the root joint at J[0]) or at 0 ("zero", the GEM-X soma_to_smpl.py convention).
                On the Guap clip j0 tracks better (limb error 8.8 vs 10.3 deg).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import rot, smpl

MP_TO_W = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # rows: X_w,Y_w,Z_w in mp coords
SMPL_TO_LOCAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# MediaPipe indices
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
LEG_LMS = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]


def _n(v):
    return v / max(np.linalg.norm(v), 1e-9)


@lru_cache(maxsize=1)
def parents():
    return [int(x) for x in smpl.parents24()]


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def slerp_dir(a, b, w):
    """Great-circle interpolation between unit vectors a and b."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    th = np.arccos(c)
    if th < 1e-4:
        return _n((1 - w) * a + w * b)
    if th > np.pi - 1e-3:  # opposite: rotate about any perpendicular axis
        perp = _n(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return _n(np.cos(w * th) * a + np.sin(w * th) * perp)
    return (np.sin((1 - w) * th) * a + np.sin(w * th) * b) / np.sin(th)


def _rotate_dir_towards(cur, target, max_ang):
    """Move unit vector `cur` towards `target` by at most max_ang (rad)."""
    th = np.arccos(float(np.clip(np.dot(cur, target), -1.0, 1.0)))
    if th <= max_ang:
        return target
    return slerp_dir(cur, target, max_ang / th)


def _angle_q(a, b):
    return 2.0 * np.arccos(min(abs(float(np.dot(a, b))), 1.0))


def frame_world(frame):
    """Joints of a stream frame in the heading frame: quat_apply(body_quat, smpl_joints)."""
    q = np.asarray(frame["body_quat"], np.float64)
    J = np.asarray(frame["smpl_joints"], np.float64)
    return rot.quat_rotate(np.broadcast_to(q, (24, 4)), J)


def make_frame(q, Jw):
    q = np.asarray(q, np.float64)
    local = rot.quat_rotate(np.broadcast_to(rot.quat_conj(q), (24, 4)), Jw)
    return {"smpl_joints": local.astype(np.float32), "body_quat": q.astype(np.float32), "joints_W": Jw}


def blend_frames(a, b, w):
    """Cross-fade two stream frames without leaving the skeleton manifold: root quaternion slerp,
    every bone direction slerped (bone lengths interpolated), joints re-chained from the pelvis.
    w = 0 -> a, w = 1 -> b."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    qa = np.asarray(a["body_quat"], np.float64)
    qb = np.asarray(b["body_quat"], np.float64)
    if np.dot(qa, qb) < 0:
        qb = -qb
    Wa, Wb = frame_world(a), frame_world(b)
    par = parents()
    out = np.zeros((24, 3))
    out[0] = (1 - w) * Wa[0] + w * Wb[0]
    for i in range(1, 24):
        va, vb = Wa[i] - Wa[par[i]], Wb[i] - Wb[par[i]]
        la, lb = np.linalg.norm(va), np.linalg.norm(vb)
        out[i] = out[par[i]] + ((1 - w) * la + w * lb) * slerp_dir(va / max(la, 1e-9), vb / max(lb, 1e-9), w)
    return make_frame(rot.quat_slerp(qa, qb, w), out)


class BoneDeadband:
    """Soft dead band ('backlash') against MediaPipe jitter while the operator stands still.

    The root orientation and every bone direction (in the output root frame) only move once the
    input has moved more than the dead band away, and then by the excess only — so the output is
    continuous (no stair steps), sits perfectly still for sub-threshold noise, and lags a moving
    limb by at most `bone_deg` (time lag = bone_deg / angular speed, e.g. 2 deg at 60 deg/s = 33 ms).
    """

    def __init__(self, bone_deg=2.0, root_deg=1.0):
        self.bone = np.radians(bone_deg)
        self.root = np.radians(root_deg)
        self.q = None
        self.dirs = None

    def reset(self):
        self.q = None
        self.dirs = None

    def __call__(self, frame):
        if self.bone <= 0 and self.root <= 0:
            return frame
        q_in = np.asarray(frame["body_quat"], np.float64)
        Jw = frame_world(frame)
        par = parents()
        if self.q is None:
            self.q = q_in.copy()
        else:
            if np.dot(self.q, q_in) < 0:
                q_in = -q_in
            ang = _angle_q(self.q, q_in)
            if ang > self.root:
                self.q = rot.quat_slerp(self.q, q_in, (ang - self.root) / ang)
        Jl = rot.quat_rotate(np.broadcast_to(rot.quat_conj(self.q), (24, 4)), Jw)  # in the held root frame
        out = np.zeros((24, 3))
        out[0] = Jl[0]
        new_dirs = np.zeros((24, 3))
        for i in range(1, 24):
            v = Jl[i] - Jl[par[i]]
            ln = np.linalg.norm(v)
            d = v / max(ln, 1e-9)
            if self.dirs is not None:
                th = np.arccos(float(np.clip(np.dot(self.dirs[i], d), -1.0, 1.0)))
                if th <= self.bone:
                    d = self.dirs[i]
                else:
                    d = slerp_dir(self.dirs[i], d, (th - self.bone) / th)
            new_dirs[i] = d
            out[i] = out[par[i]] + ln * d
        self.dirs = new_dirs
        Jw_out = rot.quat_rotate(np.broadcast_to(self.q, (24, 4)), out)
        return make_frame(self.q, Jw_out)


class PresenceGate:
    """What to stream when the operator is (not) there.

    target = live pose while a calibrated person is tracked; the last live pose for `hold_s` after
    they vanish (short MediaPipe dropouts, someone walking in front); then the calm idle pose.
    Every switch of target is a cross-fade that starts from the pose actually being streamed
    (never a jump): to idle over `fade_out_s`, back to live over `fade_in_s` once the person has
    been seen for `confirm_s` (no flicker on a one-frame false detection).
    """

    TRACK, HOLD, IDLE, WAIT = "TRACK", "HOLD", "IDLE", "WAIT"

    def __init__(self, idle, hold_s=0.5, fade_out_s=1.5, fade_in_s=0.8, confirm_s=0.2, quick_s=0.25):
        self.idle = idle
        self.hold_s, self.fade_out_s, self.fade_in_s = hold_s, fade_out_s, fade_in_s
        self.confirm_s, self.quick_s = confirm_s, quick_s
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.out = self.idle
        self.last_live = None
        self.t_lost = None
        self.t_seen = None
        self._from = self.idle
        self._t_switch = None
        self._dur = 1.0

    def _switch(self, t, dur):
        self._from = self.out
        self._t_switch = t
        self._dur = max(dur, 1e-3)

    def crossfade(self, t, dur=0.5):
        """Force a fade from the current output (e.g. after re-calibration changed the mapping)."""
        self._switch(t, dur)

    def __call__(self, t, live):
        """t: capture time (s); live: converted frame or None (nobody / not trackable)."""
        prev = self.state
        if live is not None:
            self.t_lost = None
            self.t_seen = t if self.t_seen is None else self.t_seen
            self.last_live = live
            if prev == self.HOLD:
                self.state = self.TRACK
                self._switch(t, self.quick_s)
            elif prev in (self.IDLE, self.WAIT):
                self.state = self.WAIT
                if t - self.t_seen >= self.confirm_s:
                    self.state = self.TRACK
                    self._switch(t, self.fade_in_s)
        else:
            self.t_seen = None
            if prev == self.TRACK:
                self.state = self.HOLD
                self.t_lost = t
            elif prev == self.HOLD and t - self.t_lost >= self.hold_s:
                self.state = self.IDLE
                self._switch(t, self.fade_out_s)
            elif prev == self.WAIT:
                self.state = self.IDLE
        if self.state == self.TRACK:
            target = live
        elif self.state == self.HOLD:
            target = self.last_live
        else:
            target = self.idle
        w = 1.0 if self._t_switch is None else smoothstep((t - self._t_switch) / self._dur)
        self.out = blend_frames(self._from, target, w)
        return self.out


def frame_from_left_up(left, up):
    """Rotation matrix with columns (forward, left, up), robot convention."""
    y = _n(left)
    z = _n(up - np.dot(up, y) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def _cap_tilt(up, max_tilt):
    """Rotate a unit 'up' vector back towards +Z so that its tilt <= max_tilt (same azimuth)."""
    if np.arccos(np.clip(up[2], -1, 1)) <= max_tilt:
        return up
    h = up[:2] / max(np.linalg.norm(up[:2]), 1e-9)
    return np.array([h[0] * np.sin(max_tilt), h[1] * np.sin(max_tilt), np.cos(max_tilt)])


def _slerp_R(Ra, Rb, t):
    qa = rot.rotmat_to_quat(Ra)
    qb = rot.rotmat_to_quat(Rb)
    return rot.quat_to_rotmat(rot.quat_slerp(qa, qb, t))


class MediaPipeToSmpl:
    def __init__(self, root_pitch_share=0.5, pelvis_mode="j0", legs="auto", vis_thresh=0.5, arm_depth_scale=1.0,
                 max_root_tilt_deg=40.0, leg_blend_s=0.0, leg_unlock_thresh=None, leg_lock_frames=1,
                 leg_unlock_frames=1, max_arm_forward=None):
        J = smpl.rest_joints24()  # SMPL Y-up rest positions
        par = smpl.parents24()
        off = np.zeros_like(J)
        for i in range(1, 24):
            off[i] = J[i] - J[par[i]]
        self.off = off @ SMPL_TO_LOCAL.T  # rest bone offsets in robot-local axes
        self.len = np.linalg.norm(self.off, axis=1)
        self.J0 = J[0].copy()  # added in Z-up world coords, exactly like compute_human_joints
        self.root_pitch_share = root_pitch_share
        # MediaPipe tends to put elbows/hands too far towards the camera; scale the depth (X_w)
        # component of the upper-arm/forearm directions (1.0 = raw)
        self.arm_depth_scale = arm_depth_scale
        # safety: the operator kneeling / lying down is not trackable by a standing robot ->
        # cap the commanded root tilt (a deep bow is ~25-35 deg); None disables
        self.max_root_tilt = None if max_root_tilt_deg is None else np.radians(max_root_tilt_deg)
        # cap on how far in front of the shoulder the elbow/wrist may reach (fraction of the bone length
        # along +X, e.g. 0.5 = 30 deg forward of the frontal plane); None = off. Arms held out in front
        # shift the robot's centre of mass and SONIC answers with a step forward.
        self.max_arm_forward = max_arm_forward
        self.pelvis_mode = pelvis_mode
        self.legs = legs
        self.vis_thresh = vis_thresh
        # auto leg lock with hysteresis + cross-fade (defaults = old behaviour: instant switch per frame)
        self.leg_blend_s = leg_blend_s
        self.leg_unlock_thresh = vis_thresh if leg_unlock_thresh is None else leg_unlock_thresh
        self.leg_lock_frames = leg_lock_frames
        self.leg_unlock_frames = leg_unlock_frames
        self.leg_w = 0.0  # 0 = tracked legs, 1 = locked straight legs
        self._leg_bad = 0
        self._leg_good = 0
        self._t = None
        self.R_cal = np.eye(3)  # tilt/yaw calibration applied to W
        self.legs_locked = False
        # glitch guard: MediaPipe occasionally swaps left/right -> 180 deg root flips
        self.max_jump_rad = 1.2
        self.max_glitch_frames = 5
        self._prev = None
        self._glitch = 0
        self.max_root_step = 0.12  # rad per input frame (~3.6 rad/s at 30 fps)

    # ------------------------------------------------------------------ calibration
    def calibrate(self, world_list):
        """Operator stands upright facing the camera (T/A-pose): level the camera, zero the yaw."""
        ups, lefts = [], []
        for w in world_list:
            P = self.to_W(w, calibrated=False)
            mid_hip = 0.5 * (P[L_HIP] + P[R_HIP])
            mid_sh = 0.5 * (P[L_SH] + P[R_SH])
            mid_ank = 0.5 * (P[L_ANK] + P[R_ANK])
            ups.append(_n(mid_sh - mid_hip) + _n(mid_hip - mid_ank))
            lefts.append(_n(P[L_SH] - P[R_SH]) + _n(P[L_HIP] - P[R_HIP]))
        up = _n(np.mean(ups, axis=0))
        left = np.mean(lefts, axis=0)
        R_meas = frame_from_left_up(left, up)  # measured body frame in uncalibrated W
        self.R_cal = R_meas.T  # maps measured frame -> identity (upright, facing +X)

    def to_W(self, world33, calibrated=True):
        P = np.asarray(world33, dtype=np.float64)[:, :3] @ MP_TO_W.T
        return P @ self.R_cal.T if calibrated else P

    # ------------------------------------------------------------------ conversion
    def convert(self, world33, vis=None, t=None):
        """world33: MediaPipe world landmarks, vis: (33,) visibility, t: capture time (s, for the
        leg cross-fade; None = assume 30 fps)."""
        P = self.to_W(world33)
        vis = np.ones(33) if vis is None else np.asarray(vis)
        Z = np.array([0.0, 0.0, 1.0])
        mid_hip = 0.5 * (P[L_HIP] + P[R_HIP])
        P = P - mid_hip
        mid_sh = 0.5 * (P[L_SH] + P[R_SH])
        trunk_up = _n(mid_sh)
        a = self.root_pitch_share
        if self.max_root_tilt is not None:  # trunk at most 1.5x the root cap (e.g. 60 deg)
            trunk_up = _cap_tilt(trunk_up, 1.5 * self.max_root_tilt)
        up_root = _n((1 - a) * Z + a * trunk_up)
        if self.max_root_tilt is not None:
            up_root = _cap_tilt(up_root, self.max_root_tilt)
        R_root = frame_from_left_up(P[L_HIP] - P[R_HIP], up_root)
        R_chest = frame_from_left_up(P[L_SH] - P[R_SH], trunk_up)
        mid_ear = 0.5 * (P[L_EAR] + P[R_EAR])
        head_up = _n(mid_ear - mid_sh)
        R_head = frame_from_left_up(P[L_EAR] - P[R_EAR], head_up)

        o = self.off
        Jt = np.zeros((24, 3))
        Jt[1] = R_root @ o[1]
        Jt[2] = R_root @ o[2]
        Jt[3] = R_root @ o[3]
        R_s2 = _slerp_R(R_root, R_chest, 1.0 / 3.0)
        R_s3 = _slerp_R(R_root, R_chest, 2.0 / 3.0)
        Jt[6] = Jt[3] + R_s2 @ o[6]
        Jt[9] = Jt[6] + R_s3 @ o[9]
        Jt[12] = Jt[9] + R_chest @ o[12]
        Jt[13] = Jt[9] + R_chest @ o[13]
        Jt[14] = Jt[9] + R_chest @ o[14]
        Jt[15] = Jt[12] + R_head @ o[15]
        Jt[16] = Jt[13] + R_chest @ o[16]
        Jt[17] = Jt[14] + R_chest @ o[17]

        def bone_dir(a_lm, b_lm, depth=1.0, fwd_cap=None):
            v = P[b_lm] - P[a_lm]
            u = _n(np.array([v[0] * depth, v[1], v[2]]))
            if fwd_cap is not None and u[0] > fwd_cap:  # keep the lateral/vertical heading, shorten the reach
                n = np.linalg.norm(u[1:])
                lat = u[1:] / n if n > 1e-6 else np.array([0.0, -1.0])  # straight at the camera -> fold down
                u = np.concatenate([[fwd_cap], lat * np.sqrt(1.0 - fwd_cap ** 2)])
            return u

        ds, cap = self.arm_depth_scale, self.max_arm_forward
        Jt[18] = Jt[16] + self.len[18] * bone_dir(L_SH, L_EL, ds, cap)
        Jt[19] = Jt[17] + self.len[19] * bone_dir(R_SH, R_EL, ds, cap)
        Jt[20] = Jt[18] + self.len[20] * bone_dir(L_EL, L_WR, ds, cap)
        Jt[21] = Jt[19] + self.len[21] * bone_dir(R_EL, R_WR, ds, cap)
        Jt[22] = Jt[20] + self._hand_R(P, True) @ o[22]
        Jt[23] = Jt[21] + self._hand_R(P, False) @ o[23]

        lw = self._update_leg_weight(vis, t)
        R_leg = frame_from_left_up(R_root[:, 1], Z)  # straight rest legs under the (yaw-only) pelvis
        tracked = {4: (L_HIP, L_KNEE), 5: (R_HIP, R_KNEE), 7: (L_KNEE, L_ANK), 8: (R_KNEE, R_ANK),
                   10: (L_ANK, L_TOE), 11: (R_ANK, R_TOE)}
        for c, p in [(4, 1), (5, 2), (7, 4), (8, 5), (10, 7), (11, 8)]:
            locked_dir = _n(R_leg @ o[c])
            if lw >= 1.0:
                d = locked_dir
            elif lw <= 0.0:
                d = bone_dir(*tracked[c])
            else:
                d = slerp_dir(bone_dir(*tracked[c]), locked_dir, lw)
            Jt[c] = Jt[p] + self.len[c] * d

        q_root = rot.rotmat_to_quat(R_root)
        if self._prev is not None:
            q_prev = self._prev["body_quat"].astype(np.float64)
            d = abs(float(np.dot(q_prev, q_root)))
            ang = 2 * np.arccos(min(d, 1.0))
            if ang > self.max_jump_rad and self._glitch < self.max_glitch_frames:
                self._glitch += 1
                return self._prev
            if ang > self.max_root_step:  # persistent big change (e.g. front/back confusion): turn slowly
                q_root = rot.quat_slerp(q_prev, q_root, self.max_root_step / ang)
        self._glitch = 0
        if self.pelvis_mode == "j0":
            Jt = Jt + self.J0
        local = rot.quat_rotate(np.broadcast_to(rot.quat_conj(q_root), (24, 4)), Jt)
        out = {"smpl_joints": local.astype(np.float32), "body_quat": q_root.astype(np.float32), "joints_W": Jt}
        self._prev = out
        return out

    def _update_leg_weight(self, vis, t):
        """Auto leg lock: lock after `leg_lock_frames` frames with a leg landmark below vis_thresh,
        unlock after `leg_unlock_frames` frames with all of them >= leg_unlock_thresh, and
        cross-fade the leg bones over leg_blend_s (0 = switch instantly, the old behaviour)."""
        dt = 1.0 / 30.0 if (t is None or self._t is None) else float(np.clip(t - self._t, 0.0, 0.2))
        self._t = t
        vmin = float(np.min(vis[LEG_LMS])) if vis is not None else 1.0
        self._leg_bad = self._leg_bad + 1 if vmin < self.vis_thresh else 0
        self._leg_good = self._leg_good + 1 if vmin >= self.leg_unlock_thresh else 0
        if self.legs == "lock":
            target = 1.0
        elif self.legs == "track":
            target = 0.0
        else:
            target = 1.0 if self.legs_locked else 0.0
            if not self.legs_locked and self._leg_bad >= self.leg_lock_frames:
                target = 1.0
            elif self.legs_locked and self._leg_good >= self.leg_unlock_frames:
                target = 0.0
        self.legs_locked = target >= 1.0
        if self.leg_blend_s <= 0:
            self.leg_w = target
        else:
            step = dt / self.leg_blend_s
            self.leg_w = float(np.clip(self.leg_w + np.clip(target - self.leg_w, -step, step), 0.0, 1.0))
        return smoothstep(self.leg_w)

    def idle_frame(self, arm_abduct_deg=8.0, elbow_flex_deg=12.0):
        """Calm standing pose (upright, facing the camera, arms relaxed at the sides, palms in).
        Used as the 'nobody there' target of the presence gate and the sim-side watchdog."""
        o = self.off
        Jt = np.zeros((24, 3))
        for i in range(1, 24):  # rest skeleton first (legs, spine, head, collars, shoulders)
            Jt[i] = Jt[parents()[i]] + o[i]
        a, f = np.radians(arm_abduct_deg), np.radians(elbow_flex_deg)
        for side, (sh, el, wr, hd) in ((1.0, (16, 18, 20, 22)), (-1.0, (17, 19, 21, 23))):
            upper = np.array([0.0, side * np.sin(a), -np.cos(a)])
            fore = _n(np.cos(f) * upper + np.sin(f) * np.array([1.0, 0.0, 0.0]))
            Jt[el] = Jt[sh] + self.len[el] * upper
            Jt[wr] = Jt[el] + self.len[wr] * fore
            rest_a = np.array([0.0, side, 0.0])
            rest_n = np.array([0.0, 0.0, -1.0])
            n = np.array([0.0, -side, 0.0])  # palm towards the thigh
            n = _n(n - np.dot(n, fore) * fore)
            M_meas = np.stack([fore, n, np.cross(fore, n)], axis=1)
            M_rest = np.stack([rest_a, rest_n, np.cross(rest_a, rest_n)], axis=1)
            Jt[hd] = Jt[wr] + (M_meas @ M_rest.T) @ o[hd]
        if self.pelvis_mode == "j0":
            Jt = Jt + self.J0
        q = np.array([1.0, 0.0, 0.0, 0.0])
        return {"smpl_joints": Jt.astype(np.float32), "body_quat": q.astype(np.float32), "joints_W": Jt}

    def _hand_R(self, P, left):
        """Global hand rotation relative to the SMPL rest hand (T-pose, palm down, thumb fwd)."""
        if left:
            el, wr, idx, pk, sign, rest_a = L_EL, L_WR, L_INDEX, L_PINKY, -1.0, np.array([0.0, 1.0, 0.0])
        else:
            el, wr, idx, pk, sign, rest_a = R_EL, R_WR, R_INDEX, R_PINKY, 1.0, np.array([0.0, -1.0, 0.0])
        a = _n(P[wr] - P[el])
        n = sign * _n(np.cross(P[idx] - P[wr], P[pk] - P[wr]))
        n = _n(n - np.dot(n, a) * a)
        rest_n = np.array([0.0, 0.0, -1.0])  # palm down in robot-local axes
        M_meas = np.stack([a, n, np.cross(a, n)], axis=1)
        M_rest = np.stack([rest_a, rest_n, np.cross(rest_a, rest_n)], axis=1)
        return M_meas @ M_rest.T


class OneEuro:
    """One-Euro filter for (N,3) landmark arrays."""

    def __init__(self, min_cutoff=1.5, beta=0.3, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        ad = self._alpha(self.dc, dt)
        self.dx = ad * dx + (1 - ad) * self.dx
        cutoff = self.mc + self.beta * np.linalg.norm(self.dx, axis=-1, keepdims=True)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()
