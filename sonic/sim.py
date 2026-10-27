# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""MuJoCo G1 sim matching gear_sonic/utils/mujoco_sim/base_sim.py (run_sim_loop.py).

* scene: gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml (29 body DOF + 2x7 Dex3)
* timestep overridden to SIMULATE_DT = 1/200 s
* PD at every physics step: tau = kp (q_des - q) + kd (0 - dq), clipped to the actuator
  effort limits (MOTOR_EFFORT_LIMIT_LIST == actuator ctrlrange)
* IMU: pelvis quaternion = free-joint qpos[3:7] (wxyz), gyro = free-joint qvel[3:6]
  (MuJoCo free-joint angular velocity is body-local, like a real IMU)
* ElasticBand on the pelvis (unitree_mujoco style) — enabled at start, toggled like key '9'
* robot-parameterised (sonic.robots): G1 scene by default; robot=get_robot("k1") loads the staged K1 MJCF
  (+ floor) with the training armatures, K1 joint order and K1 effort limits (actuator ctrlrange)
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .params import HAND_KD, HAND_KP, SCENE_XML, SIM_DT
from .robots import RobotSpec, g1


class ElasticBand:
    def __init__(self):
        self.kp_pos, self.kd_pos = 10000.0, 1000.0
        self.kp_ang, self.kd_ang = 1000.0, 10.0
        self.point = np.array([0.0, 0.0, 1.0])
        self.length = 0.0
        self.enable = True
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # official band always targets identity

    def advance(self, pos, quat_wxyz, lin_vel, ang_vel):
        f = self.kp_pos * (self.point - pos + np.array([0.0, 0.0, self.length])) + self.kd_pos * (0 - lin_vel)
        q = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]) * Rotation.from_quat(self.target_quat[[1, 2, 3, 0]]).inv()
        rv = q.as_rotvec()
        tq = -self.kp_ang * rv - self.kd_ang * ang_vel
        return np.concatenate([f, tq])


class G1Sim:
    def __init__(self, xml_path=None, sim_dt=SIM_DT, elastic_band=True, robot: RobotSpec | None = None):
        self.robot = robot = robot or g1()
        if xml_path is None:
            self.model = robot.load_model()
        else:  # explicit scene file (e.g. the G1 event scene); armature from the spec still applies
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
            if robot.armature is not None:
                for n, a in zip(robot.mujoco_joint_names, robot.armature):
                    self.model.dof_armature[self.model.jnt_dofadr[self.model.joint(n).id]] = a
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        self.sim_dt = sim_dt
        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in robot.mujoco_joint_names]
        assert min(jid) > 0, "body joint missing in scene"
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        # actuator driving each body joint
        act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        self.act = np.array([act_of_joint[j] for j in jid])
        self.hand_j = [j for j in range(m.njnt) if "hand" in m.joint(j).name]
        self.hand_qadr = np.array([m.jnt_qposadr[j] for j in self.hand_j], dtype=int)
        self.hand_vadr = np.array([m.jnt_dofadr[j] for j in self.hand_j], dtype=int)
        self.hand_act = np.array([act_of_joint[j] for j in self.hand_j], dtype=int)
        self.torque_limit = m.actuator_ctrlrange[:, 1].copy()
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso_link").id
        self.band = ElasticBand() if elastic_band else None
        if self.band is not None:
            self.band.point[2] = robot.band_height
        # command (hardware order)
        n = robot.num_dof
        self.q_des = np.zeros(n)
        self.kp = np.zeros(n)
        self.kd = np.zeros(n)
        self.last_tau = np.zeros(n)
        mujoco.mj_forward(m, self.data)

    # ---- state ---------------------------------------------------------------
    def low_state(self):
        d = self.data
        return {
            "q": d.qpos[self.qadr].copy(),
            "dq": d.qvel[self.vadr].copy(),
            "imu_quat": d.qpos[3:7].copy(),  # wxyz
            "imu_gyro": d.qvel[3:6].copy(),  # body frame
            "torso_quat": d.xquat[self.torso].copy(),
            "base_pos": d.qpos[0:3].copy(),
        }

    def set_command(self, q_des, kp, kd):
        self.q_des = np.asarray(q_des, dtype=np.float64)
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)

    def set_band(self, enable: bool):
        if self.band is not None:
            self.band.enable = enable

    # ---- stepping --------------------------------------------------------------
    def step(self, n: int = 1):
        m, d = self.model, self.data
        for _ in range(n):
            q = d.qpos[self.qadr]
            dq = d.qvel[self.vadr]
            tau = self.kp * (self.q_des - q) + self.kd * (0.0 - dq)
            ctrl = np.zeros(m.nu)
            ctrl[self.act] = tau
            if len(self.hand_j):
                ctrl[self.hand_act] = HAND_KP * (0.0 - d.qpos[self.hand_qadr]) + HAND_KD * (0.0 - d.qvel[self.hand_vadr])
            ctrl = np.clip(ctrl, -self.torque_limit, self.torque_limit)
            self.last_tau = ctrl[self.act]
            d.ctrl[:] = ctrl
            if self.band is not None and self.band.enable:
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, vel, 0)
                d.xfrc_applied[self.pelvis] = self.band.advance(
                    d.xpos[self.pelvis].copy(), d.xquat[self.pelvis].copy(), vel[3:6], vel[0:3])
            else:
                d.xfrc_applied[self.pelvis] = 0.0
            t_before = d.time
            mujoco.mj_step(m, d)
            if d.time < t_before or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
                raise RuntimeError(f"MuJoCo became unstable at t={t_before:.3f}s (data was reset)")

    @property
    def time(self):
        return float(self.data.time)

    def pelvis_height(self):
        return float(self.data.qpos[2])

    def tilt_deg(self):
        """Angle between pelvis z-axis and world up."""
        R = self.data.xmat[self.pelvis].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


class Recorder:
    """Offscreen mujoco.Renderer -> list of RGB frames (tracking camera on the pelvis)."""

    def __init__(self, sim: G1Sim, width=640, height=480, distance=3.0, azimuth=135.0, elevation=-15.0):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.pelvis
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.frames = []

    def grab(self):
        self.renderer.update_scene(self.sim.data, camera=self.cam)
        img = self.renderer.render().copy()
        self.frames.append(img)
        return img


EVENT_SCENE = """<mujoco model="g1 event scene">
  <include file="g1_29dof_with_hand.xml"/>
  <statistic center="0 0 0.8" extent="2.0"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.92 0.93 0.95 1"/>
    <quality shadowsize="4096"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.80 0.83 0.88" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.86 0.87 0.89" rgb2="0.82 0.83 0.86"
      markrgb="0.74 0.75 0.78" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="6 6" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.0 3.0" dir="-0.4 0.3 -1" directional="true" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <light pos="-1.0 1.5 3.0" dir="0.3 -0.4 -1" directional="true" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
  <default>
    <geom friction="1.0"/>
  </default>
</mujoco>
"""


def event_scene_path():
    """Same physics as scene_43dof.xml (robot, floor friction 1.0), presentation lighting/colours,
    no com_marker site. Written next to the official scene so the robot include/meshdir resolve."""
    p = SCENE_XML.parent / "scene_43dof_event.xml"
    if not p.exists() or p.read_text() != EVENT_SCENE:
        p.write_text(EVENT_SCENE)
    return p
