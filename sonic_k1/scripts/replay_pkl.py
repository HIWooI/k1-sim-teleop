"""Kinematic MuJoCo replay of a K1 SONIC PKL — checks the PKL the way the Mac deploy loop reads it.

    .venv/bin/python sonic_k1/scripts/replay_pkl.py sonic_k1/data/robot/Guap_ver2.pkl --video videos/k1_pkl_replay.mp4

Path exercised: PKL -> sonic.motion.load_robot_pkl(robot=k1) (50 Hz resample, MuJoCo -> Isaac order) ->
Isaac -> MuJoCo order (RobotSpec.isaaclab_to_mujoco) -> qpos of the staged K1 MJCF. Prints foot clearance,
joint-limit violations and the body-FK equality against pose_aa (axis * dof per body).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import joblib
    import mujoco

    from sonic import rot
    from sonic.motion import load_robot_pkl
    from sonic.robots import get_robot
    from sonic.sim import G1Sim

    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--fix-height", action="store_true", default=True,
                    help="shift the clip so the lowest foot point touches the floor (motion lib fix_trans_height)")
    args = ap.parse_args()

    k1 = get_robot("k1")
    mot = load_robot_pkl(args.pkl, robot=k1)
    raw = next(iter(joblib.load(args.pkl).values()))
    sim = G1Sim(robot=k1, elastic_band=False)
    m, d = sim.model, sim.data
    feet = [g for g in range(m.ngeom) if "ankle_roll" in m.body(m.geom_bodyid[g]).name and m.geom_contype[g]]

    def pose(f):
        d.qpos[0:3] = mot.body_pos[f, 0]
        d.qpos[3:7] = mot.body_quat[f, 0]
        d.qpos[sim.qadr] = mot.joint_pos[f][k1.isaaclab_to_mujoco]
        mujoco.mj_kinematics(m, d)

    lo = m.jnt_range[[m.joint(n).id for n in k1.mujoco_joint_names], 0]
    hi = m.jnt_range[[m.joint(n).id for n in k1.mujoco_joint_names], 1]
    foot_z, viol = [], 0
    for f in range(mot.timesteps):
        pose(f)
        foot_z.append(min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in feet))
        q = d.qpos[sim.qadr]
        viol += int(((q < lo - 1e-3) | (q > hi + 1e-3)).any())
    foot_z = np.array(foot_z)
    dz = -foot_z.min() if args.fix_height else 0.0
    # pose_aa consistency: each body's local rotation == axis*angle of its joint
    f0 = len(raw["dof"]) // 2
    R_pose = [rot.quat_to_rotmat(rot.aa_to_quat(raw["pose_aa"][f0, i + 1])) for i in range(k1.num_dof)]
    R_joint = [rot.quat_to_rotmat(rot.quat_from_angle_axis(np.array(raw["dof"][f0, i]), m.jnt_axis[m.joint(n).id]))
               for i, n in enumerate(k1.mujoco_joint_names)]
    pose_err = max(float(np.abs(a - b).max()) for a, b in zip(R_pose, R_joint))
    print(f"{Path(args.pkl).name}: {mot.timesteps} frames @50Hz ({len(raw['dof'])} @ {raw['fps']} fps in pkl), "
          f"foot z min {foot_z.min():+.3f} / median {np.median(foot_z):+.3f} m (height shift {dz:+.3f}), "
          f"frames outside joint limits {viol}, pose_aa vs dof max err {pose_err:.1e}")
    if args.video:
        import imageio.v2 as imageio

        r = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = sim.pelvis
        cam.distance, cam.azimuth, cam.elevation = 2.4, 150.0, -12.0
        frames = []
        every = max(1, int(round(50 / args.fps)))
        for f in range(0, mot.timesteps, every):
            pose(f)
            d.qpos[2] += dz
            mujoco.mj_kinematics(m, d)
            r.update_scene(d, camera=cam)
            frames.append(r.render().copy())
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=args.fps, quality=7, macro_block_size=1)
        print("video ->", args.video, len(frames), "frames")


if __name__ == "__main__":
    main()
