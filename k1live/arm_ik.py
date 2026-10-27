"""Bone-direction arm IK for the K1 (MediaPipe -> 10 arm joints), image plane first.

Why not the position IK (k1/cpu_kinematics.K1CpuRetargeter, arm_ik="shape18", the earlier default): it solves all 23
joints for 14 point targets built from the person's bones rescaled to K1 lengths, with a pull toward q = 0. On the K1
that biases the arms (docs/architecture.md): the "hand" target is the wrist-roll link 7 cm past the elbow, so the forearm
direction is barely constrained and the q = 0 pull (elbow 0 = forearm bent 90 deg) wins; the waist yaw and the
shoulder position (MediaPipe shoulders shrug when the arms go up) absorb part of the arm motion, and only the arm
joints are kept afterwards.

Here each arm is solved on its own for what the audience compares - the directions of the upper arm (shoulder ->
elbow) and forearm (elbow -> palm) in the torso frame:

* target directions: the person's MediaPipe bones in their own torso frame (x forward, y left, z up). The frontal-plane
  angle (what a front camera sees) comes from the 2D image landmarks, measured from the person's image up axis
  (perpendicular to the shoulder line), the depth component from the 3D world landmarks (MediaPipe's z is the noisy
  one). `image_w` blends world (0) and image (1) in-plane angles.
* solver: closed-form start (pitch/roll from the upper arm, yaw/elbow from the forearm, both branches, plus the previous
  solution) -> a few damped Gauss-Newton steps on the exact K1 chain (URDF offsets, joint limits, elbow not past
  straight) with residuals weighted depth < frontal plane, a continuity term to the previous frame (no branch flips on
  fast overhead swings) and a weak pull to the relaxed pose (decides shoulder yaw when the arm is straight).
* ~0.2-0.4 ms per frame (both arms) in NumPy, no new dependencies.

Conventions (k1/gestures.py): shoulder_pitch < 0 = arm forward/up, left shoulder_roll > 0 = out (right < 0, one-sided),
elbow 0 = forearm at 90 deg, pi/2 = straight. Output in ARM_JOINTS order, person terms (the caller mirrors), wrist roll
at the relaxed default (MediaPipe gives no forearm twist).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from k1.cpu_kinematics import UrdfKinematics

SIDES = ("left", "right")
ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
PALM_OFFSET = 0.20          # palm centre along the hand link x axis (k1/gestures.HAND_OFFSET)
ARM_DEFAULT = np.array([0.2, 0.25, 0.0, 0.95, 0.15, 0.2, -0.25, 0.0, 0.95, -0.15])  # loco default, ARM_JOINTS order
ELBOW_STRAIGHT = np.pi / 2  # elbow angle of a straight arm (more = hyperextension, never a person's pose)
# MediaPipe pose landmarks
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
ARM_LM = {"left": (L_SH, L_EL, L_WR), "right": (R_SH, R_EL, R_WR)}


def _rot(axis_i, a):
    c, s = np.cos(a), np.sin(a)
    if axis_i == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis_i == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ArmChain:
    """One K1 arm in the torso frame (URDF origins/axes; all arm joints have rpy 0 and a unit x/y/z axis)."""

    def __init__(self, robot: UrdfKinematics, side: str):
        joints = {j.name: j for j in robot.joints}
        names = [f"{side}_{p}_joint" for p in ARM_PARTS]
        self.side = side
        self.origin, self.axis_i = [], []
        for n in names:
            j = joints[n]
            if not np.allclose(j.origin[:3, :3], np.eye(3)):
                raise ValueError(f"{n}: rotated joint origin not supported")
            ax = np.abs(j.axis)
            if not (np.isclose(ax.max(), 1.0) and np.isclose(ax.sum(), 1.0) and j.axis.max() > 0):
                raise ValueError(f"{n}: axis {j.axis} is not +x/+y/+z")
            self.origin.append(j.origin[:3, 3].copy())
            self.axis_i.append(int(np.argmax(ax)))
        self.lo = np.array([robot.lower[robot.joint_names.index(n)] for n in names])
        self.hi = np.array([robot.upper[robot.joint_names.index(n)] for n in names])
        self.palm = self.origin[4] + np.array([PALM_OFFSET, 0.0, 0.0])  # elbow link -> palm (wrist roll keeps it)
        # closed-form start assumes: pitch about y, roll about x, yaw about z, elbow about y, forearm along +x
        assert self.axis_i[:4] == [1, 0, 2, 1] and abs(self.palm[0]) > 0.1, (side, self.axis_i)

    def fk(self, q):
        """q: 4 (pitch, roll, yaw, elbow) -> shoulder (roll link origin), elbow, palm, joint axes (4, 3)."""
        o, ai = self.origin, self.axis_i
        R1 = _rot(ai[0], q[0])
        sh = o[0] + R1 @ o[1]
        R2 = R1 @ _rot(ai[1], q[1])
        R3 = R2 @ _rot(ai[2], q[2])
        el = sh + R2 @ o[2] + R3 @ o[3]
        R4 = R3 @ _rot(ai[3], q[3])
        palm = el + R4 @ self.palm
        axes = np.stack([np.eye(3)[ai[0]], R1[:, ai[1]], R2[:, ai[2]], R3[:, ai[3]]])
        return sh, el, palm, axes


class DirArmIK:
    """Per-arm direction IK. solve(side, u, f) -> 4 joints (pitch, roll, yaw, elbow)."""

    def __init__(self, urdf=None, depth_w=0.5, w_fore=1.0, cont=0.1, reg=0.03, iters=6, damping=1e-3,
                 elbow_max=ELBOW_STRAIGHT, nominal=ARM_DEFAULT):
        if urdf is None:
            from k1.sim import URDF as urdf
        robot = UrdfKinematics(Path(urdf))
        self.chain = {s: ArmChain(robot, s) for s in SIDES}
        self.W = np.array([depth_w, 1.0, 1.0])     # residual weights in the torso frame: x = depth
        self.w_fore, self.cont, self.reg, self.iters, self.damping = w_fore, cont, reg, iters, damping
        self.nominal = {s: np.asarray(nominal, np.float64)[5 * i:5 * i + 4].copy() for i, s in enumerate(SIDES)}
        self.lo = {s: self.chain[s].lo[:4].copy() for s in SIDES}
        self.hi = {s: self.chain[s].hi[:4].copy() for s in SIDES}
        for s in SIDES:
            self.hi[s][3] = min(self.hi[s][3], elbow_max)
        self.prev = {s: self.nominal[s].copy() for s in SIDES}
        self.last_cost = {s: 0.0 for s in SIDES}

    def reset(self):
        self.prev = {s: self.nominal[s].copy() for s in SIDES}

    # -------------------------------------------------------------------------------------------------- helpers
    def _resid(self, side, q, u_t, f_t, w_u, w_f, jac=False):
        sh, el, palm, axes = self.chain[side].fk(q)
        u = el - sh
        u /= np.linalg.norm(u)
        f = palm - el
        f /= np.linalg.norm(f)
        W = self.W
        r = np.concatenate([w_u * W * (u - u_t), w_f * W * (f - f_t), self.cont * (q - self.prev[side]),
                            self.reg * (q - self.nominal[side])])
        if not jac:
            return r
        J = np.zeros((14, 4))
        for i in range(4):
            if i < 3:
                J[0:3, i] = w_u * W * np.cross(axes[i], u)
            J[3:6, i] = w_f * W * np.cross(axes[i], f)
        J[6:10] = np.eye(4) * self.cont
        J[10:14] = np.eye(4) * self.reg
        return r, J

    def _starts(self, side, u, f):
        """Closed-form candidates (upper arm -> pitch/roll, forearm -> yaw/elbow; 2 x 2 branches) + previous."""
        lo, hi, prev = self.lo[side], self.hi[side], self.prev[side]
        out = [prev.copy()]
        r1 = np.arcsin(np.clip(u[1], -1.0, 1.0))
        for r in (r1, _wrap(np.pi - r1)):
            cr = np.cos(r)
            if abs(cr) < 1e-6:
                p = prev[0]
            else:
                p = np.arctan2(-u[0] / cr, -u[2] / cr)
            # forearm in the roll-link frame: f_l = (Ry(p) Rx(r))^T f = (c_y c_e, s_y c_e, -s_e)
            fl = (_rot(1, p) @ _rot(0, r)).T @ f
            h = np.hypot(fl[0], fl[1])
            for sgn in (1.0, -1.0):
                e = np.arctan2(-fl[2], sgn * h)
                y = np.arctan2(sgn * fl[1], sgn * fl[0])
                if h < 0.35:  # nearly straight: yaw is unobservable, lean on the previous yaw
                    y = prev[2] + (h / 0.35) * _wrap(y - prev[2])
                # put each angle on the 2 pi branch nearest the previous solution (pitch / yaw are +-pi joints)
                q = np.array([p, r, y, e])
                q[[0, 2]] = prev[[0, 2]] + _wrap(q[[0, 2]] - prev[[0, 2]])
                out.append(np.clip(q, lo, hi))
        return out

    # -------------------------------------------------------------------------------------------------- solve
    def solve(self, side, u_t, f_t, w_u=1.0, w_f=None):
        """u_t, f_t: unit target directions (torso frame) of upper arm and forearm. Returns (pitch, roll, yaw, elbow)."""
        w_f = self.w_fore if w_f is None else w_f
        lo, hi = self.lo[side], self.hi[side]
        best, best_c = None, np.inf
        for q0 in self._starts(side, u_t, f_t):
            c = float(np.sum(self._resid(side, q0, u_t, f_t, w_u, w_f) ** 2))
            if c < best_c:
                best, best_c = q0, c
        q = best
        lam = self.damping
        for _ in range(self.iters):
            r, J = self._resid(side, q, u_t, f_t, w_u, w_f, jac=True)
            H = J.T @ J + lam * np.eye(4)
            dq = -np.linalg.solve(H, J.T @ r)
            qn = np.clip(q + dq, lo, hi)
            cn = float(np.sum(self._resid(side, qn, u_t, f_t, w_u, w_f) ** 2))
            if cn <= best_c:
                q, best_c, lam = qn, cn, max(lam * 0.3, 1e-6)
                if np.max(np.abs(dq)) < 1e-4:
                    break
            else:
                lam *= 10.0
        self.prev[side] = q.copy()
        self.last_cost[side] = best_c
        return q

    def directions(self, side, q4):
        """Unit upper-arm / forearm directions of a joint pose (torso frame)."""
        sh, el, palm, _ = self.chain[side].fk(q4)
        u, f = el - sh, palm - el
        return u / np.linalg.norm(u), f / np.linalg.norm(f)


# ------------------------------------------------------------------------------------------------ person side
def k1_axes(w):
    """MediaPipe world (x = person's left for a front camera, y down, z away) -> K1 axes (x fwd, y left, z up)."""
    w = np.asarray(w)[..., :3]
    return np.stack((-w[..., 2], w[..., 0], -w[..., 1]), axis=-1)


def torso_basis(k):
    """Person's torso frame (columns x fwd, y left, z up) from K1-axes landmarks (k1live.human.torso_frame)."""
    y = k[L_SH] - k[R_SH]
    y = y / max(np.linalg.norm(y), 1e-6)
    z = 0.5 * (k[L_SH] + k[R_SH]) - 0.5 * (k[L_HIP] + k[R_HIP])
    z = z - y * (z @ y)
    z = z / max(np.linalg.norm(z), 1e-6)
    return np.stack([np.cross(y, z), y, z], axis=1)


def image_inplane(img_px):
    """Per arm bone: (in-plane angle from the person's image up axis toward the person's left, projected length px)
    for [L upper, L fore, R upper, R fore]; the k1.run_video.human_arm_2d definition. img_px: (33, 2) pixels, y down."""
    P = np.asarray(img_px, np.float64)[:, :2] * np.array([1.0, -1.0])
    sv = P[L_SH] - P[R_SH]
    n = max(np.linalg.norm(sv), 1e-6)
    left = sv / n                      # person's left in the image (image right for a front camera)
    up = np.array([-left[1], left[0]])
    ang, ln = [], []
    for s in SIDES:
        a, b, c = ARM_LM[s]
        for p0, p1 in ((a, b), (b, c)):
            v = P[p1] - P[p0]
            ang.append(np.arctan2(v @ left, v @ up))
            ln.append(np.linalg.norm(v))
    return np.array(ang), np.array(ln)


def person_dirs(world, img_px=None, image_w=1.0):
    """Target unit directions {side: (upper arm, forearm)} in the person's torso frame (x fwd, y left, z up).
    world: (33, 3) MediaPipe world landmarks (filtered); img_px: (33, 2) image landmarks in pixels (filtered) or None.
    image_w: 0 = world landmarks only, 1 = in-plane angle from the image, depth from the world landmarks."""
    k = k1_axes(world)
    R = torso_basis(k)
    out = {}
    if img_px is not None and image_w > 0:
        ang, _ = image_inplane(img_px)
    for si, s in enumerate(SIDES):
        a, b, c = ARM_LM[s]
        dirs = []
        for bi, (p0, p1) in enumerate(((a, b), (b, c))):
            d = R.T @ (k[p1] - k[p0])
            d = d / max(np.linalg.norm(d), 1e-9)
            if img_px is not None and image_w > 0:
                m = np.sqrt(max(1.0 - d[0] ** 2, 0.0))
                th_w = np.arctan2(d[1], d[2])
                th = th_w + image_w * _wrap(ang[2 * si + bi] - th_w)
                d = np.array([d[0], m * np.sin(th), m * np.cos(th)])
            dirs.append(d)
        out[s] = tuple(dirs)
    return out


def overhead_tilt(d, tan_min):
    """A bone pointing above the horizontal gets at least tan_min forward per unit of height (the frontal-plane angle
    y : z is kept, so a front camera sees the same arm). K1-specific balance envelope: with both arms straight up the
    centre of mass moves ~2 cm back and the ROBOTIS stand policy keeps stepping backward (docs/results.md)."""
    if tan_min <= 0 or d[2] <= 0 or d[0] >= tan_min * d[2]:
        return d
    m = np.hypot(d[1], d[2])  # frontal-plane part; (x, y, z) = (tan_min * z', y', z') with y' : z' = y : z
    c = 1.0 / np.sqrt(1.0 + (tan_min * d[2] / m) ** 2)
    out = np.array([tan_min * d[2] * c, d[1] * c, d[2] * c])
    return out / np.linalg.norm(out)


class DirRetargeter:
    """Frame -> 10 arm joints (ARM_JOINTS order, person terms, wrist roll at the relaxed default).
    Same input as the position-IK path (filtered MediaPipe world landmarks + visibility) plus the frame's image landmarks.
    overhead_deg: minimum forward tilt from vertical of a raised bone (overhead_tilt; 0 = off)."""

    def __init__(self, urdf=None, image_w=1.0, overhead_deg=0.0, **ik_kw):
        self.ik = DirArmIK(urdf, **ik_kw)
        self.image_w = image_w
        self.tan_over = float(np.tan(np.radians(overhead_deg)))
        self.ms = 0.0
        self.dirs = None

    def reset(self):
        self.ik.reset()

    def solve(self, world, img_px=None):
        t0 = time.perf_counter()
        self.dirs = person_dirs(world, img_px, self.image_w)
        q = ARM_DEFAULT.copy()
        for i, s in enumerate(SIDES):
            u, f = self.dirs[s]
            if self.tan_over > 0:
                u, f = overhead_tilt(u, self.tan_over), overhead_tilt(f, self.tan_over)
                self.dirs[s] = (u, f)
            q[5 * i:5 * i + 4] = self.ik.solve(s, u, f)
        self.ms = (time.perf_counter() - t0) * 1e3
        return q
