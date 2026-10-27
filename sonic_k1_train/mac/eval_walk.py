"""Closed-loop walking test: kinematic planner (k1live.planner) -> whole-body SONIC policy in MuJoCo, plus the split
baseline (ROBOTIS loco legs on a velocity command + mimic-student arms). CPU heavy: run on a Linux CPU host, not the Mac.

    # G1 check of the planner port (official SONIC release policy, planner reference, no arm overlay)
    python sonic_k1_train/mac/eval_walk.py --backend g1 --scenarios all --json logs/walk_g1.json
    # K1 SONIC checkpoint (80 ms lookahead export) with the K1 planner reference + arm overlays
    python sonic_k1_train/mac/eval_walk.py --backend sonic --export models/k1/<run>/exported --step 2000 --future 4x1 \
        --json logs/walk_k1.json --video videos/k1_sonic_walk_2000.mp4
    # same scenarios on the split drive (track A)
    python sonic_k1_train/mac/eval_walk.py --backend split --json logs/walk_split.json

Each scenario: INIT / band as eval_k1_mujoco.py (--start ground: feet on the floor, band released 2 s after control
starts), 1 s standing reference, then the scenario's 30 s command script (body-frame vx, vy, wz in m/s, rad/s).
sonic / g1: the planner is reset on the robot's pose, gets the command every tick and replans from its own motion
(SONIC deploy behaviour; --anchor robot re-anchors the context root on the robot at every replan); the policy reads
the planner window (current + future frames) as a streamed reference motion. Arms: scenarios with an overlay replace
the reference arm joints by a gesture / a person's arms / smooth random arms (planner_arms.py), legs + waist stay the
planner's. split: the same commands go straight to loco_armdriven_r2's velocity_commands; the student tracks the
overlay arms (or the loco default arm pose).
Metrics per scenario: fall (pelvis z < 0.45 m or tilt > 45 deg), body-frame velocity achieved vs commanded on steady
segments (>= 1.5 s after a command change), yaw rate, foot slip (xy speed of a foot body while it touches the floor),
max tilt, joint errors vs the reference (arms / legs, sonic and split arms), policy time per tick.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sonic_k1_train" / "data"))
sys.path.insert(0, str(REPO / "sonic_k1_train" / "mac"))

FALL_Z, FALL_TILT = 0.45, 45.0
DT = 0.02
SCEN_S = 30.0

# name: (segments [(t0, t1, (vx, vy, wz))] over 0..30 s, arm overlay or None)
SCENARIOS = {
    "fwd_0.3": ([(1, 28, (0.3, 0, 0))], None),
    "fwd_0.6_wave": ([(1, 28, (0.6, 0, 0))], "gesture:wave_hello"),
    "fwd_0.8": ([(1, 28, (0.8, 0, 0))], None),
    "back_0.3_random": ([(1, 28, (-0.3, 0, 0))], "random"),
    "left_0.3_human": ([(1, 28, (0, 0.3, 0))], "human:pexels_10149176_soma_ik"),
    "right_0.3": ([(1, 28, (0, -0.3, 0))], None),
    "turn_0.6": ([(1, 12, (0, 0, 0.6)), (15, 26, (0, 0, -0.6))], "gesture:welcome_open_arms"),
    "walk_turn_0.4": ([(1, 14, (0.4, 0, 0.3)), (14, 27, (0.4, 0, -0.3))], None),
    "start_stop_0.5": ([(1, 5, (0.5, 0, 0)), (8, 12, (0.5, 0, 0)), (15, 19, (-0.3, 0, 0)), (22, 26, (0, 0.3, 0))],
                       "gesture:clap"),
    "mixed_0.4_human": ([(1, 9, (0.35, 0.2, 0)), (9, 15, (0, 0, 0.5)), (15, 24, (0.6, 0, 0)), (24, 27, (0, -0.3, 0))],
                        "human:pexels_10515017"),
}


# standing, arms only (not in "all"; --scenarios standing): the same arm sources with no walking command, to compare
# SONIC with the split student on the arms alone (joint-angle and hand / elbow position errors)
EXTRA_SCENARIOS = {
    "stand_wave": ([], "gesture:wave_hello"),
    "stand_point": ([], "gesture:point_guide"),
    "stand_clap": ([], "gesture:clap"),
    "stand_human_a": ([], "human:pexels_10149176_soma_ik"),
    "stand_human_b": ([], "human:pexels_10515017"),
    "stand_random": ([], "random"),
}
ALL_SCENARIOS = {**SCENARIOS, **EXTRA_SCENARIOS}


def cmd_at(segs, t):
    for a, b, c in segs:
        if a <= t < b:
            return np.array(c, float)
    return np.zeros(3)


def steady_mask(segs, ts, settle=1.5):
    out = np.zeros(len(ts), bool)
    for a, b, c in segs:
        out |= (ts >= a + settle) & (ts < b)
    return out


class Arms:
    """Arm overlay source (k1live.protocol.ARM_JOINTS order) at absolute scenario time."""

    def __init__(self, spec, seed=0):
        import planner_arms as PA

        self.PA, self.spec = PA, spec
        if spec is None:
            self.track = None
        elif spec.startswith("gesture:"):
            self.track = PA.ArmTrack(spec, PA.load_gestures()[spec.split(":", 1)[1]])
        elif spec.startswith("human:"):
            self.track = PA.ArmTrack(spec, PA.load_human()[spec.split(":", 1)[1]])
        else:
            self.track = PA.ArmTrack(spec, PA.random_arms(np.random.default_rng(seed), int(40 * 50), scale=0.7))

    def apply(self, q_win, t0, idx):
        """q_win (H, n) reference joints of the window starting at scenario time t0 -> arms replaced (ramped in)."""
        if self.track is None:
            return q_win
        ts = t0 + np.arange(len(q_win)) * DT
        w = np.clip((ts - 0.5) / 1.0, 0, 1)[:, None]
        w = w * w * (3 - 2 * w)
        q = q_win.copy()
        q[:, idx] = (1 - w) * q_win[:, idx] + w * self.track.at(np.maximum(ts, 0))
        return q


# ------------------------------------------------------------------------------------------------ metrics helpers
def foot_contacts(model, data, foot_bodies, floor_geom):
    on = np.zeros(len(foot_bodies), bool)
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            other = c.geom2 if g == c.geom1 else c.geom1
            if other != floor_geom:
                continue
            b = model.geom_bodyid[g]
            for k, fb in enumerate(foot_bodies):
                if b == fb:
                    on[k] = True
    return on


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def roll_pitch(q):
    w, x, y, z = q
    return (float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))


def log_gait(log, d, feet, on, q, names):
    log.feet.append(np.array([d.xpos[b] for b in feet]))
    log.on.append(np.asarray(on, bool).copy())
    log.knee.append([q[names.index("left_knee_joint")], q[names.index("right_knee_joint")]])
    log.rp.append(roll_pitch(d.qpos[3:7]))


def gait_metrics(A, mask):
    """Walking quality on the ticks in `mask` (steady moving segments): stride (same-foot touchdown to touchdown, m),
    cadence (touchdowns / s), swing clearance (ankle rise above its stance height, cm), knee flexion (deg, mean and
    p10), pelvis roll / pitch sway (std, deg) and vertical bounce (std, cm), double-support share."""
    if len(A["feet"]) != len(mask) or mask.sum() < 50:
        return {}
    F, C, K, RP, z = A["feet"], A["on"], np.degrees(np.abs(A["knee"])), np.degrees(A["rp"]), A["z"]
    strides, clear, n_td = [], [], 0
    for k in range(2):
        c = C[:, k]
        td = np.flatnonzero(c[1:] & ~c[:-1]) + 1
        lo = np.flatnonzero(~c[1:] & c[:-1]) + 1
        td = td[mask[td]]
        n_td += len(td)
        strides += [float(np.linalg.norm(F[b, k, :2] - F[a, k, :2])) for a, b in zip(td[:-1], td[1:]) if b - a < 150]
        for a in lo[mask[lo]]:
            nxt = td[td > a]
            if len(nxt) and nxt[0] - a < 100:
                clear.append(float(F[a:nxt[0], k, 2].max() - F[a, k, 2]))
    T = mask.sum() * DT
    return {"stride_m": round(float(np.median(strides)), 3) if strides else None, "cadence_hz": round(n_td / T, 2),
            "clearance_cm": round(float(np.median(clear)) * 100, 1) if clear else None,
            "knee_deg": round(float(K[mask].mean()), 1), "knee_p10_deg": round(float(np.percentile(K[mask], 10)), 1),
            "roll_std_deg": round(float(RP[mask, 0].std()), 2), "pitch_std_deg": round(float(RP[mask, 1].std()), 2),
            "bounce_cm": round(float(z[mask].std()) * 100, 2),
            "double_support": round(float((C[mask].sum(1) == 2).mean()), 2)}


class Log:
    def __init__(self):
        self.t, self.pos, self.yaw, self.tilt, self.z = [], [], [], [], []
        self.q, self.ref, self.slip, self.ms, self.cmd = [], [], [], [], []
        self.rpos, self.ryaw, self.arm_cm = [], [], []
        self.feet, self.on, self.knee, self.rp = [], [], [], []   # gait: ankle xyz, contact, knee angles, roll/pitch

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


class ArmPos:
    """Elbow + hand position error in the torso frame (cm): the robot's bodies vs the same model posed with the
    reference joints on the robot's root. Joint-angle error over-weights the twist joints (shoulder yaw, wrist roll),
    which move the hand little; SONIC is rewarded on body poses, the split student on joint angles."""

    BODIES = ("left_elbow_link", "left_wrist_roll_rubber_hand", "right_elbow_link", "right_wrist_roll_rubber_hand")

    def __init__(self, model, joint_names):
        import mujoco

        self.mj, self.m = mujoco, model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
        self.torso = model.body("torso_link").id
        self.bodies = [model.body(b).id for b in self.BODIES]

    def _local(self, d):
        R, p = d.xmat[self.torso].reshape(3, 3), d.xpos[self.torso]
        return (d.xpos[self.bodies] - p) @ R

    def err_cm(self, d, q_ref):
        self.d.qpos[:] = d.qpos
        self.d.qpos[self.qadr] = q_ref
        self.mj.mj_kinematics(self.m, self.d)
        return float(np.linalg.norm(self._local(d) - self._local(self.d), axis=1).mean() * 100)


def body_vel(ts, p, yaw, k=12):
    yw = np.unwrap(yaw)
    v = (p[2 * k:, :2] - p[:-2 * k, :2]) / (ts[2 * k:] - ts[:-2 * k])[:, None]
    wz = (yw[2 * k:] - yw[:-2 * k]) / (ts[2 * k:] - ts[:-2 * k])
    y = yw[k:-k]
    vb = np.stack([np.cos(y) * v[:, 0] + np.sin(y) * v[:, 1], -np.sin(y) * v[:, 0] + np.cos(y) * v[:, 1]], 1)
    return vb, wz


def summarize(log: Log, segs, names, fell_at):
    A = log.arrays()
    ts = A["t"]
    out = {"fell": fell_at is not None, "fell_at": fell_at, "max_tilt_deg": round(float(A["tilt"].max()), 1),
           "min_pelvis_z": round(float(A["z"].min()), 3)}
    # velocity in the robot's heading frame, 0.5 s central windows
    k = 12
    vr = wr = None
    if len(A["rpos"]) == len(ts) and len(ts) > 2 * k + 2:   # the reference's own body-frame velocity (planner)
        vr, wr = body_vel(ts, A["rpos"], A["ryaw"], k)
        tc = ts[k:-k]
        st = np.zeros(len(tc), bool)
        for a, b, c in segs:
            st |= (tc >= a + 1.5) & (tc < b)
        cmd = A["cmd"][k:-k][st]
        mv = np.linalg.norm(cmd[:, :2], axis=1) > 0.05
        if mv.any():
            out["ref_speed_ratio"] = round(float(np.linalg.norm(vr[st][mv], axis=1).mean() / np.linalg.norm(cmd[mv, :2], axis=1).mean()), 3)
        out["ref_vel_err_mean"] = [round(float(x), 3) for x in np.abs(np.concatenate([vr[st], wr[st, None]], 1) - cmd).mean(0)]
    if len(ts) > 2 * k + 2:
        p, yw = A["pos"], np.unwrap(A["yaw"])
        v = (p[2 * k:, :2] - p[:-2 * k, :2]) / (ts[2 * k:] - ts[:-2 * k])[:, None]
        wz = (yw[2 * k:] - yw[:-2 * k]) / (ts[2 * k:] - ts[:-2 * k])
        y = yw[k:-k]
        vb = np.stack([np.cos(y) * v[:, 0] + np.sin(y) * v[:, 1], -np.sin(y) * v[:, 0] + np.cos(y) * v[:, 1]], 1)
        tc = ts[k:-k]
        segm = []
        for a, b, c in segs:
            sel = (tc >= a + 1.5) & (tc < b)
            if sel.sum() < 5:
                continue
            sg = {"cmd": list(c), "vx": round(float(vb[sel, 0].mean()), 3), "vy": round(float(vb[sel, 1].mean()), 3),
                  "wz": round(float(wz[sel].mean()), 3)}
            if vr is not None:   # the reference's own body-frame velocity on the same samples
                sg.update(rvx=round(float(vr[sel, 0].mean()), 3), rvy=round(float(vr[sel, 1].mean()), 3),
                          rwz=round(float(wr[sel].mean()), 3))
            segm.append(sg)
        out["segments"] = segm
        st = np.zeros(len(tc), bool)
        for a, b, c in segs:
            st |= (tc >= a + 1.5) & (tc < b)
        if st.any():
            cmd = A["cmd"][k:-k][st]
            ach = np.concatenate([vb[st], wz[st, None]], 1)
            out["vel_err_mean"] = [round(float(x), 3) for x in np.abs(ach - cmd).mean(0)]
            mv = np.linalg.norm(cmd[:, :2], axis=1) > 0.05
            if mv.any():
                out["speed_ratio"] = round(float(np.linalg.norm(ach[mv, :2], axis=1).mean() / np.linalg.norm(cmd[mv, :2], axis=1).mean()), 3)
        idle = np.ones(len(tc), bool)
        for a, b, c in segs:
            idle &= ~((tc >= a) & (tc < b + 1.5))
        idle &= tc > 0.5
        if idle.sum() > 5:
            out["idle_drift_mps"] = round(float(np.linalg.norm(vb[idle], axis=1).mean()), 3)
    moving = np.zeros(len(ts), bool)
    for a, b, c in segs:
        if np.linalg.norm(np.asarray(c, float)[:2]) > 0.05 or abs(c[2]) > 0.05:
            moving |= (ts >= a + 1.5) & (ts < b)
    g = gait_metrics(A, moving)
    if g:
        out["gait"] = g
    sl = A["slip"]
    sl = sl[np.isfinite(sl)]
    if sl.size:
        out["foot_slip_mean"] = round(float(sl.mean()), 3)
        out["foot_slip_p95"] = round(float(np.percentile(sl, 95)), 3)
    if len(A["ref"]):
        err = np.degrees(np.abs(A["q"] - A["ref"]))
        arms = [i for i, n in enumerate(names) if any(t in n for t in ("shoulder", "elbow", "wrist"))]
        legs = [i for i, n in enumerate(names) if any(t in n for t in ("hip", "knee", "ankle"))]
        out["arms_deg"] = round(float(err[:, arms].mean()), 2)
        out["legs_deg"] = round(float(err[:, legs].mean()), 2)
        # robot arms lagging the reference by L ticks: error vs L = 0..10 (delay vs accuracy)
        q, r = A["q"][:, arms], A["ref"][:, arms]
        lag = [float(np.degrees(np.abs(q[L:] - r[:len(r) - L])).mean()) for L in range(0, 11)] if len(q) > 20 else []
        if lag:
            out["arms_best_lag_deg"], out["arms_best_lag_ticks"] = round(min(lag), 2), int(np.argmin(lag))
        out["arms_joint_deg"] = {names[i].replace("_joint", ""): round(float(err[:, i].mean()), 1) for i in arms}
    if len(A["arm_cm"]):
        out["arms_pos_cm"] = round(float(np.mean(A["arm_cm"])), 2)
    out["tick_ms"] = round(float(np.mean(A["ms"])), 2) if len(A["ms"]) else None
    return out


# ------------------------------------------------------------------------------------------------ SONIC backends
class LegsCam:
    """Close-up of the legs: free camera looking at a point under the pelvis (the tracking Recorder looks at the
    pelvis itself)."""

    def __init__(self, model, w=480, h=360, distance=1.25, azimuth=150.0, elevation=-6.0):
        import mujoco

        self.mj = mujoco
        self.renderer = mujoco.Renderer(model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation

    def grab(self, data, xy, z=0.28):
        self.cam.lookat[:] = [xy[0], xy[1], z]
        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render().copy()


def run_sonic(robot_name, vdir, segs, arms_spec, anchor, video, seed, start="ground", label="", latency=2, source=None,
              duration=SCEN_S, vx_gain=1.0, legs=False):
    """legs: with video, every frame also carries leg close-ups of the reference and the robot
    (frames = (t, ref, robot, cmd, ref_legs, robot_legs))."""
    """source: another reference with the KinematicPlanner interface (reset / command / step -> RefWindow / close),
    e.g. eval_walk_clips.ClipSource; default the planner."""
    import mujoco

    from k1live.planner import KinematicPlanner, window_motion
    from k1live.protocol import ARM_JOINTS
    from sonic import rot
    from sonic.deploy import SonicController
    from sonic.motion import standing_motion
    from sonic.robots import get_robot
    from sonic.runner import SUBSTEPS, place_on_ground
    from sonic.sim import G1Sim, Recorder

    robot = get_robot(robot_name)
    sim = G1Sim(robot=robot, elastic_band=True)
    if start == "ground":
        place_on_ground(sim)
    if robot_name == "k1":
        import yaml

        from eval_k1_mujoco import K1Controller

        k1cfg = yaml.safe_load((Path(vdir) / "observation_config.yaml").read_text()).get("k1", {})
        if int(k1cfg.get("smpl_num_future_frames", 10)) != 10:
            # human-encoder checkpoint (k1s_bones: SMPL 4 x 20 ms): the controller that registers the n-frame SMPL
            # terms lives with the human backend (k1live/backends/sonic_human.py); the walk itself runs in the robot (g1) mode
            from k1live.backends.sonic_human import _controller_cls

            ctrl = _controller_cls().load(vdir, robot)
        else:
            ctrl = K1Controller.load(vdir, robot)
    else:
        ctrl = SonicController(variant=vdir.name, models_dir=vdir.parent, robot=robot)
    ctrl.set_motion(standing_motion(50, robot=robot), play=False)
    kp = source or KinematicPlanner(robot=robot_name, anchor=anchor, seed=seed, horizon=60, threads=2, latency_ticks=latency,
                                    vx_gain=vx_gain)
    arms = Arms(arms_spec, seed)
    names = list(robot.mujoco_joint_names)
    arm_idx = [names.index(n) for n in ARM_JOINTS] if robot_name == "k1" else []
    m, d = sim.model, sim.data
    armpos = ArmPos(m, names) if robot_name == "k1" else None
    feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
    floor = m.geom("floor").id
    cam = dict(width=480, height=480, distance=2.6, azimuth=150.0, elevation=-12.0)
    rec = Recorder(sim, **cam) if video else None
    ghost = None
    if video:
        gd = mujoco.MjData(m)
        ghost = Recorder(sim, **cam)
        ghost.sim = type("S", (), {"data": gd})()
    legcam = LegsCam(m) if (video and legs) else None
    frames = []
    log = Log()
    t_ctrl = t_plan = None
    fell_at, tick = None, 0
    vel = np.zeros(6)
    while True:
        if ctrl.state == "WAIT_FOR_CONTROL":
            ctrl.start_control()
            t_ctrl = sim.time
        if t_ctrl is not None and sim.time - t_ctrl >= 2.0:
            sim.set_band(False)
        if t_ctrl is not None and t_plan is None and sim.time - t_ctrl >= 3.0:
            kp.reset(np.concatenate([d.qpos[:7], d.qpos[sim.qadr]]))
            t_plan = sim.time
            ctrl.reinitialize_heading = True
        ts = None
        if t_plan is not None:
            ts = sim.time - t_plan - 1.0          # scenario clock (1 s of planner standing first)
            if ts >= duration:
                break
            c = cmd_at(segs, ts)
            kp.command(*c)
            qpos_now = np.concatenate([d.qpos[:7], d.qpos[sim.qadr]])
            w = kp.step(qpos_now)
            q_ref = w.q if (robot_name == "g1" or arms.track is None) else arms.apply(w.q, ts, arm_idx)
            ctrl.current_motion = window_motion(w, robot, q_ref)   # frame 0 duplicated -> current_frame = 1
            ctrl.current_frame, ctrl.play = 1, True
        t0 = time.perf_counter()
        ls = sim.low_state()
        q_t, kp_, kd_ = ctrl.step(ls)
        ms = (time.perf_counter() - t0) * 1e3
        sim.set_command(q_t, kp_, kd_)
        sim.step(SUBSTEPS)
        if ts is not None and ts >= 0:
            z, tilt = sim.pelvis_height(), sim.tilt_deg()
            on = foot_contacts(m, d, feet, floor)
            sl = []
            for k_, b in enumerate(feet):
                if on[k_]:
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
                    sl.append(float(np.linalg.norm(vel[3:5])))
            log.t.append(ts)
            log.pos.append(d.qpos[:3].copy())
            log.yaw.append(yaw_of(d.qpos[3:7]))
            log.tilt.append(tilt)
            log.z.append(z)
            log.slip.append(np.mean(sl) if sl else np.nan)
            log_gait(log, d, feet, on, d.qpos[sim.qadr], names)
            log.ms.append(ms)
            log.cmd.append(cmd_at(segs, ts))
            log.q.append(ls["q"])
            log.ref.append(q_ref[0][: robot.num_dof])
            if armpos is not None:
                log.arm_cm.append(armpos.err_cm(d, q_ref[0][: robot.num_dof]))
            log.rpos.append(w.pos[0].copy())
            log.ryaw.append(yaw_of(w.quat[0]))
            if fell_at is None and (z < FALL_Z or tilt > FALL_TILT):
                fell_at = round(ts, 2)
                print(f"  FALL at t={ts:.2f} (z={z:.3f}, tilt={tilt:.0f})", flush=True)
                break
        if rec is not None and tick % 2 == 0 and ts is not None:
            img = rec.grab()
            gd = ghost.sim.data
            gd.qpos[:] = 0
            gd.qpos[0:2] = d.qpos[0:2] + [0.0, 0.0]
            gd.qpos[2] = w.pos[0, 2]
            gd.qpos[3:7] = rot.quat_mul(ctrl.apply_delta_heading(), w.quat[0])
            gd.qpos[sim.qadr] = q_ref[0]
            mujoco.mj_forward(m, gd)
            ghost.renderer.update_scene(gd, camera=ghost.cam)
            g = ghost.renderer.render().copy()
            if legcam is not None:
                xy = d.qpos[0:2].copy()
                frames.append((ts, g, img, cmd_at(segs, ts), legcam.grab(gd, xy), legcam.grab(d, xy)))
            else:
                frames.append((ts, g, img, cmd_at(segs, ts)))
        tick += 1
    kp.close()
    res = summarize(log, segs, names, fell_at)
    res["planner_replans"] = kp.stream.n_replans
    res["planner_ms_last"] = round(kp.stream.model.last_ms, 1)
    return res, frames


# ------------------------------------------------------------------------------------------------ split backend
def run_split(segs, arms_spec, video, seed):
    import mujoco

    import k1.sim as ks
    from k1live.protocol import ARM_JOINTS

    sim = ks.K1Split()
    CMD = np.zeros(3)
    orig = sim.loco.act

    def act(terms):
        terms = dict(terms)
        terms[sim._lt["cmd"]] = CMD.copy()
        return orig(terms)

    sim.loco.act = act
    sim.reset()
    arms = Arms(arms_spec, seed)
    names = sim.names
    arm_idx = [names.index(n) for n in ARM_JOINTS]
    m, d = sim.model, sim.data
    armpos = ArmPos(m, names)
    feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
    floor = m.geom("floor").id
    cam = ks.Camera(sim, w=480, h=480, distance=2.6, azimuth=150.0, elevation=-12.0) if video else None
    log, frames, fell_at = Log(), [], None
    base = sim.loco.default.copy()
    vel = np.zeros(6)
    tick = 0
    for i in range(int((SCEN_S + 1.0) / DT)):
        ts = i * DT - 1.0
        CMD[:] = cmd_at(segs, ts)
        rq = arms.apply(base[None], ts, arm_idx)[0]
        nxt = arms.apply(base[None], ts + DT, arm_idx)[0]
        rdq = (nxt - rq) / DT
        t0 = time.perf_counter()
        sim.step(rq, rdq, (0.0, 0.0, 0.0, 1.0))
        ms = (time.perf_counter() - t0) * 1e3
        if ts < 0:
            continue
        z, tilt = sim.pelvis_height(), sim.tilt_deg()
        on = foot_contacts(m, d, feet, floor)
        sl = []
        for k_, b in enumerate(feet):
            if on[k_]:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
                sl.append(float(np.linalg.norm(vel[3:5])))
        log.t.append(ts)
        log.pos.append(d.qpos[:3].copy())
        log.yaw.append(yaw_of(d.qpos[3:7]))
        log.tilt.append(tilt)
        log.z.append(z)
        log.slip.append(np.mean(sl) if sl else np.nan)
        log_gait(log, d, feet, on, d.qpos[sim.qadr], names)
        log.ms.append(ms)
        log.cmd.append(CMD.copy())
        q = sim.q
        log.q.append(q)
        ref = q.copy()
        ref[arm_idx] = rq[arm_idx]          # only the arms have a reference in split mode
        log.ref.append(ref)
        log.arm_cm.append(armpos.err_cm(d, ref))
        if fell_at is None and (z < FALL_Z or tilt > FALL_TILT):
            fell_at = round(ts, 2)
            print(f"  FALL at t={ts:.2f}", flush=True)
            break
        if cam is not None and tick % 2 == 0:
            cam.xy = None                   # k1.sim.Camera smooths the follow point; centre it like the sonic view
            frames.append((ts, None, cam.grab(), CMD.copy()))
        tick += 1
    return summarize(log, segs, names, fell_at), frames


# ------------------------------------------------------------------------------------------------ main
def caption(img, text, y=24):
    import cv2

    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["g1", "sonic", "split"])
    ap.add_argument("--variant-dir", type=Path, help="prepared K1 dir, or the G1 variant dir (default models/release)")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--step", type=int)
    ap.add_argument("--future", default="4x1")
    ap.add_argument("--anchor", default="plan", choices=["plan", "robot"])
    ap.add_argument("--scenarios", nargs="*", default=["all"], help="all (the 10 walking scenarios), standing (arms only), or names")
    ap.add_argument("--no-arms", action="store_true", help="ignore the scenarios' arm overlays")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--latency", type=int, default=2,
                    help="planner result applied this many 20 ms ticks after the request (live: async worker, "
                         "inference 20-80 ms on a laptop CPU); the dataset used 1-4")
    ap.add_argument("--vx-gain", type=float, default=1.0,
                    help="KinematicPlanner vx_gain (planner vx = command vx x gain; slowWalk under-runs fwd/back by ~15 %%)")
    ap.add_argument("--video", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--compare-split", action="store_true",
                    help="with --backend sonic --video: also run the split drive on each scenario and tile "
                         "reference | sonic | split in the video (results of the split runs -> <json>_split)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    names = (list(SCENARIOS) if a.scenarios == ["all"] else list(EXTRA_SCENARIOS) if a.scenarios == ["standing"]
             else a.scenarios)
    vdir = a.variant_dir
    if a.backend == "g1":
        vdir = vdir or (REPO / "models" / "release")
    elif a.backend == "sonic" and a.export is not None:
        from eval_k1_mujoco import prepare_variant

        n, s = (int(x) for x in a.future.split("x"))
        vdir = prepare_variant(a.export, a.step, vdir or (a.export.parent / f"deploy_{a.step:06d}"), n, s)
    results, split_results = {}, {}
    writer = None
    if a.video:   # frames are written scenario by scenario (holding all 10 x 30 s of tiles needs ~15 GB)
        import imageio

        Path(a.video).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(a.video, fps=25, quality=7, macro_block_size=1)
        lab = a.label or a.backend
    for sc in names:
        segs, arms = ALL_SCENARIOS[sc]
        if a.no_arms or a.backend == "g1":
            arms = None
        t0 = time.time()
        if a.backend == "split":
            r, fr = run_split(segs, arms, bool(a.video), a.seed)
        else:
            r, fr = run_sonic("g1" if a.backend == "g1" else "k1", vdir, segs, arms, a.anchor, bool(a.video), a.seed,
                              latency=a.latency, vx_gain=a.vx_gain)
        r["arms_overlay"] = arms
        r["wall_s"] = round(time.time() - t0, 1)
        results[sc] = r
        print(sc, json.dumps({k: v for k, v in r.items() if k != "segments"}), flush=True)
        if a.compare_split and a.backend == "sonic":
            rs, frs = run_split(segs, arms, bool(a.video), a.seed)
            rs["arms_overlay"] = arms
            split_results[sc] = rs
            print(sc, "split", json.dumps({k: v for k, v in rs.items() if k != "segments"}), flush=True)
            if fr:  # align by time: the sonic run may end early (fall) -> hold its last frame
                tsn = np.array([f[0] for f in fr])
                fr = [(t, fr[int(max(0, np.searchsorted(tsn, t, "right") - 1))][1],
                       np.concatenate([fr[int(max(0, np.searchsorted(tsn, t, "right") - 1))][2], caption(im, "split: ROBOTIS loco + student arms")], 1), c)
                      for t, _, im, c in frs]
        if writer is not None:
            for ts, g, img, c in fr:
                txt = f"{sc}  t={ts:4.1f}s  cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}"
                right = caption(img, lab)
                if g is not None:
                    writer.append_data(np.concatenate([caption(caption(g, "reference (planner)"), txt, 470), right], 1))
                else:
                    writer.append_data(caption(right, txt, 470))
        fr = None
    falls = sum(r["fell"] for r in results.values())
    out = {"backend": a.backend, "model": str(vdir) if vdir else "k1/assets split", "anchor": a.anchor, "latency_ticks": a.latency,
           "vx_gain": a.vx_gain,
           "falls": falls, "scenarios": results}
    print(f"falls {falls}/{len(results)}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))
        if split_results:
            sp = {"backend": "split", "model": "k1/assets split", "anchor": "-",
                  "falls": sum(r["fell"] for r in split_results.values()), "scenarios": split_results}
            Path(a.json).with_name(Path(a.json).stem + "_split.json").write_text(json.dumps(sp, indent=1))
    if writer is not None:
        writer.close()
        print("video", a.video)


if __name__ == "__main__":
    main()
