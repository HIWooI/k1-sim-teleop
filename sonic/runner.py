# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Synchronous sim2sim loop: 50 Hz SonicController <-> 200 Hz MuJoCo PD (4 substeps)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import rot
from .params import CONTROL_DT, SIM_DT

SUBSTEPS = int(round(CONTROL_DT / SIM_DT))  # 4


def place_on_ground(sim, q_mj=None, yaw=0.0):
    """Put the robot in q_mj (default: the robot's default pose) with the feet on the floor
    (alternative to the band drop)."""
    import mujoco

    if q_mj is None:
        q_mj = sim.robot.default_angles
    m, d = sim.model, sim.data
    d.qpos[:] = 0
    d.qpos[3:7] = rot.euler_z_to_quat(yaw)
    d.qpos[sim.qadr] = q_mj
    d.qpos[2] = 1.0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    zmin = 1e9
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        if "ankle_roll" not in m.body(m.geom_bodyid[g]).name:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3)
            zmin = min(zmin, d.geom_xpos[g][2] - float(np.abs(R[2]) @ m.geom_size[g]))
        else:
            zmin = min(zmin, d.geom_xpos[g][2] - m.geom_rbound[g])
    d.qpos[2] = 1.0 - zmin + 0.001
    mujoco.mj_forward(m, d)
    if sim.band is not None:  # hold the spawn pose (height + yaw) during INIT, release later
        sim.band.point[2] = d.qpos[2] + 0.03
        sim.band.target_quat = d.qpos[3:7].copy()


@dataclass
class EpisodeLog:
    t: list = field(default_factory=list)
    pelvis_z: list = field(default_factory=list)
    q_isaac: list = field(default_factory=list)
    ref_isaac: list = field(default_factory=list)
    frame: list = field(default_factory=list)
    playing: list = field(default_factory=list)
    base_quat: list = field(default_factory=list)
    ref_root_quat: list = field(default_factory=list)
    tick_ms: list = field(default_factory=list)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def control_tick(ctrl, sim, log: EpisodeLog | None = None):
    """One 20 ms tick: read state -> policy -> PD command -> 4 physics steps."""
    ls = sim.low_state()
    t0 = time.perf_counter()
    frame_used = ctrl.current_frame
    playing = ctrl.play
    q_t, kp, kd = ctrl.step(ls)
    dt_ms = (time.perf_counter() - t0) * 1e3
    sim.set_command(q_t, kp, kd)
    sim.step(SUBSTEPS)
    if log is not None and ctrl.state == "CONTROL":
        m = ctrl.current_motion
        log.t.append(sim.time)
        log.pelvis_z.append(sim.pelvis_height())
        log.q_isaac.append(ls["q"][sim.robot.mujoco_to_isaaclab])
        log.ref_isaac.append(m.joint_pos[min(frame_used, m.timesteps - 1)])
        log.frame.append(frame_used)
        log.playing.append(playing)
        log.base_quat.append(ls["imu_quat"])
        log.ref_root_quat.append(rot.quat_mul(ctrl.apply_delta_heading(), m.body_quat[min(frame_used, m.timesteps - 1), 0]))
        log.tick_ms.append(dt_ms)
    return ls


def tracking_metrics(log: EpisodeLog, lag_frames: int = 0):
    """Mean |q_robot - q_ref| over the played segment (rad), optionally allowing a fixed lag."""
    a = log.arrays()
    sel = a["playing"].astype(bool)
    if sel.sum() < 5:
        return {}
    q = a["q_isaac"][sel]
    r = a["ref_isaac"][sel]
    if lag_frames > 0:
        q, r = q[lag_frames:], r[:-lag_frames]
    err = np.abs(q - r)
    groups = {
        "legs": [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18],
        "waist": [2, 5, 8],
        "arms": [11, 12, 15, 16, 19, 20, 21, 22],
        "wrists": [23, 24, 25, 26, 27, 28],
    }
    out = {"all_rad": float(err.mean()), "all_deg": float(np.degrees(err.mean()))}
    for k, idx in groups.items():
        out[f"{k}_deg"] = float(np.degrees(err[:, idx].mean()))
    # root orientation error (after the deploy's heading alignment)
    bq = a["base_quat"][sel]
    rq = a["ref_root_quat"][sel]
    if lag_frames > 0:
        bq, rq = bq[lag_frames:], rq[:-lag_frames]
    dq = rot.quat_mul(rot.quat_conj(bq), rq)
    ang = 2 * np.arccos(np.clip(np.abs(dq[:, 0]), 0, 1))
    out["root_ori_deg"] = float(np.degrees(ang.mean()))
    return out


def best_lag(log: EpisodeLog, max_lag=15):
    best = (1e9, 0)
    for lag in range(0, max_lag + 1):
        m = tracking_metrics(log, lag)
        if m and m["all_rad"] < best[0]:
            best = (m["all_rad"], lag)
    return best[1]
