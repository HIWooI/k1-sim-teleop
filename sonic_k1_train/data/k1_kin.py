"""K1 kinematics for building SONIC motion_lib PKLs (Mac CPU: numpy + mujoco + scipy, no Isaac Lab).

The motion library (gear_sonic/utils/motion_lib) reads, per motion, the dict that
gear_sonic/data_process/convert_soma_csv_to_motion_lib.py writes:

    root_trans_offset (T,3) pelvis position (world, z-up)   root_rot (T,4) pelvis quaternion **xyzw**
    dof (T,23) MuJoCo joint order                            pose_aa (T,24,3) [root rotvec, axis_j * dof_j]
    smpl_joints (T,24,3) zeros (no SMPL data for K1)         fps

and does FK itself with the motion-library MJCF (fix_height = no_fix), so the root height written here
is what the policy is asked to track. Every clip is therefore grounded with the K1 foot contact spheres
(ground()), optionally with the stance feet locked in xy (static gestures / bows / weight shifts).

MJCF = the staged k1.xml of the sonic_k1 package (head merged into torso_link, 24 bodies, 23 dofs).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

SIDES = ("left", "right")
# SONIC K1 default standing pose (k1.py K1_DEFAULT_JOINT_POS) and the ROBOTIS loco stance the k1/ gestures use
STANCE_SONIC = {"hip_pitch": -0.3, "knee": 0.63, "ankle_pitch": -0.33}
STANCE_LOCO = {"hip_pitch": -0.18, "knee": 0.36, "ankle_pitch": -0.18}
ARMS_DEFAULT = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
                "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 0.6}


def quat_wxyz_to_xyzw(q):
    return np.asarray(q)[..., [1, 2, 3, 0]]


def quat_xyzw_to_wxyz(q):
    return np.asarray(q)[..., [3, 0, 1, 2]]


class K1Kin:
    def __init__(self, mjcf: str | Path):
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.d = mujoco.MjData(self.m)
        m = self.m
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.names = [m.joint(j).name for j in range(1, m.njnt)]
        self.n = len(self.names)
        self.axes = m.jnt_axis[1:].copy()
        self.lo = m.jnt_range[1:, 0].copy()
        self.hi = m.jnt_range[1:, 1].copy()
        self.J = {n: i for i, n in enumerate(self.names)}
        self.bodies = [m.body(b).name for b in range(1, m.nbody)]  # pelvis first, MuJoCo (DFS) order
        self.foot_geoms = [g for g in range(m.ngeom)
                           if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")]
        self.foot_r = m.geom_size[self.foot_geoms, 0].copy()
        self.foot_body = {s: m.body(f"{s}_ankle_roll_link").id for s in SIDES}
        self.leg_idx = {s: [self.J[f"{s}_{j}_joint"] for j in
                            ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")] for s in SIDES}
        self.mirror_perm, self.mirror_sign = self._mirror_tables()

    # ------------------------------------------------------------------ basics
    def pose(self, stance=STANCE_SONIC, arms=ARMS_DEFAULT):
        q = np.zeros(self.n)
        for s in SIDES:
            for k, v in stance.items():
                q[self.J[f"{s}_{k}_joint"]] = v
        for k, v in arms.items():
            q[self.J[k]] = v
        return q

    def set(self, root_pos, root_quat_wxyz, dof):
        self.d.qpos[:3] = root_pos
        self.d.qpos[3:7] = root_quat_wxyz
        self.d.qpos[7:] = dof
        mujoco.mj_kinematics(self.m, self.d)
        return self.d

    def sole_points(self):
        return self.d.geom_xpos[self.foot_geoms].copy()  # sphere centres; bottom = z - r

    def clip(self, dof):
        return np.clip(dof, self.lo + 1e-3, self.hi - 1e-3)

    # ------------------------------------------------------------------ grounding
    def ground(self, root_pos, root_quat_wxyz, dof, lock_xy=False, clearance=0.0):
        """Shift the root so the lowest foot contact sphere touches z=0 every frame.
        lock_xy: also keep the mid-point of both feet at its frame-0 xy (both feet assumed planted)."""
        root_pos = np.array(root_pos, dtype=np.float64)
        ref_xy = None
        for t in range(len(root_pos)):
            self.set(root_pos[t], root_quat_wxyz[t], dof[t])
            pts = self.sole_points()
            root_pos[t, 2] -= (pts[:, 2] - self.foot_r).min() - clearance
            if lock_xy:
                mid = 0.5 * (self.d.xpos[self.foot_body["left"], :2] + self.d.xpos[self.foot_body["right"], :2])
                if ref_xy is None:
                    ref_xy = mid.copy()
                root_pos[t, :2] -= mid - ref_xy
        return root_pos

    # ------------------------------------------------------------------ mirror (x-z plane)
    def _mirror_tables(self):
        perm, sign = [], []
        for n in self.names:
            other = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(self.J[other])
            ax = self.axes[self.J[n]]
            # reflecting y: rotations about y keep their sign, about x / z flip
            sign.append(1.0 if abs(ax[1]) > 0.5 else -1.0)
        return np.array(perm), np.array(sign)

    def mirror(self, root_pos, root_quat_wxyz, dof):
        p = np.array(root_pos, dtype=np.float64)
        p[:, 1] *= -1
        q = np.array(root_quat_wxyz, dtype=np.float64)
        q[:, 1] *= -1  # (w,x,y,z) -> (w,-x,y,-z)
        q[:, 3] *= -1
        dm = dof[:, self.mirror_perm] * self.mirror_sign[None]
        return p, q, dm

    # ------------------------------------------------------------------ leg IK (feet fixed in the world)
    def leg_ik(self, root_pos, root_quat_wxyz, dof, foot_pose, q_init=None):
        """Solve both legs so each ankle_roll_link has the world pose foot_pose[side] = (pos, R)."""
        dof = np.array(dof, dtype=np.float64)
        for s in SIDES:
            idx = self.leg_idx[s]
            tp, tR = foot_pose[s]
            x0 = dof[idx] if q_init is None else q_init[idx]

            def res(x, idx=idx, tp=tp, tR=tR, s=s):
                dd = dof.copy()
                dd[idx] = x
                self.set(root_pos, root_quat_wxyz, dd)
                b = self.foot_body[s]
                R = self.d.xmat[b].reshape(3, 3)
                e_rot = Rot.from_matrix(tR.T @ R).as_rotvec()
                return np.concatenate([(self.d.xpos[b] - tp) * 10.0, e_rot * 0.5, 1e-3 * (x - x0)])

            sol = least_squares(res, x0, bounds=(self.lo[idx] + 1e-3, self.hi[idx] - 1e-3), xtol=1e-10, ftol=1e-10)
            dof[idx] = sol.x
        return dof

    def foot_poses(self, root_pos, root_quat_wxyz, dof):
        self.set(root_pos, root_quat_wxyz, dof)
        return {s: (self.d.xpos[self.foot_body[s]].copy(), self.d.xmat[self.foot_body[s]].reshape(3, 3).copy())
                for s in SIDES}

    # ------------------------------------------------------------------ motion_lib entry
    def entry(self, root_pos, root_quat_wxyz, dof, fps):
        T = len(dof)
        q = np.array(root_quat_wxyz, dtype=np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        # keep the quaternion sign continuous (motion lib slerps between frames)
        for t in range(1, T):
            if np.dot(q[t], q[t - 1]) < 0:
                q[t] *= -1
        xyzw = quat_wxyz_to_xyzw(q)
        pose_aa = np.zeros((T, self.n + 1, 3), np.float32)
        pose_aa[:, 0] = Rot.from_quat(xyzw).as_rotvec()
        pose_aa[:, 1:] = self.axes[None] * dof[:, :, None]
        return {
            "root_trans_offset": np.asarray(root_pos, np.float32),
            "pose_aa": pose_aa,
            "dof": np.asarray(dof, np.float32),
            "root_rot": xyzw.astype(np.float32),
            "smpl_joints": np.zeros((T, 24, 3), np.float32),
            "fps": int(round(fps)) if abs(fps - round(fps)) < 1e-6 else float(fps),
        }


def save_motion(out_dir: Path, name: str, entry: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({name: entry}, out_dir / f"{name}.pkl")


def resample(fps_src, fps_dst, root_pos, root_quat_wxyz, dof):
    """Linear (joints, position) + slerp (root) resampling."""
    from scipy.spatial.transform import Slerp

    T = len(dof)
    t_src = np.arange(T) / fps_src
    t_dst = np.arange(int(np.floor(t_src[-1] * fps_dst)) + 1) / fps_dst
    lin = lambda x: np.stack([np.interp(t_dst, t_src, x[:, k]) for k in range(x.shape[1])], 1)  # noqa: E731
    sl = Slerp(t_src, Rot.from_quat(quat_wxyz_to_xyzw(root_quat_wxyz)))
    return lin(np.asarray(root_pos)), quat_xyzw_to_wxyz(sl(t_dst).as_quat()), lin(np.asarray(dof))


def smooth_noise(rng, T, fps, n, period=(1.5, 4.0)):
    """(T, n) smooth random curves in [-1, 1] (min-jerk between random keyframes)."""
    out = np.zeros((T, n))
    for k in range(n):
        t, keys = 0.0, [(0.0, 0.0)]
        while t < T / fps:
            t += rng.uniform(*period)
            keys.append((t, rng.uniform(-1, 1)))
        ts = np.arange(T) / fps
        kt = np.array([a for a, _ in keys])
        kv = np.array([b for _, b in keys])
        i = np.clip(np.searchsorted(kt, ts) - 1, 0, len(kt) - 2)
        u = np.clip((ts - kt[i]) / (kt[i + 1] - kt[i]), 0, 1)
        s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
        out[:, k] = kv[i] + s * (kv[i + 1] - kv[i])
    return out
