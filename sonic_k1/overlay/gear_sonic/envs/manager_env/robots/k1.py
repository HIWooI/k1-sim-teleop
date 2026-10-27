# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
# Derived from ROBOTIS AI Sapiens (https://github.com/ROBOTIS-GIT/ai_sapiens), Apache License 2.0; actuator constants
# from ROBOTIS cyclo_lab (https://github.com/ROBOTIS-GIT/cyclo_lab), Apache License 2.0.
#
# ROBOTIS K1 (AI Sapiens, k1_rev1, 23 DOF) for SONIC — written for sonic_k1,
# following GR00T-WholeBodyControl docs/source/user_guide/new_embodiments.md and robots/h2.py.
#
# Sources
#   kinematics : ai_sapiens_description k1_rev1 (URDF from xacro, MJCF), staged by sonic_k1/scripts/stage_assets.py
#   order      : Isaac Lab articulation order = BFS over the URDF tree (fixed head_joint merged). Identical to the
#                PhysX order ROBOTIS' cyclo_lab K1 training env reports (LEGACY_PHYSX_* of the K1 sim2sim tooling)
#   actuators  : K1_rev1.py of the cyclo_lab K1 env, as reproduced in the K1 sim2sim tooling and
#                k1/assets/student_asset/params/sim2real.yaml (kp 76.5/153/44.6/22.3, kd 4.87/9.73/2.84/1.42):
#                QC080-240-R020-RE 96.864 Nm, 11.519 rad/s; QC060-200-R020-RE 47.277 Nm, 20.944 rad/s;
#                reflected inertia 0.01936542 (QC080) / 0.00564892 (QC060), ankles = 2 motors via linkage (x2)
#   gains      : SONIC/BeyondMimic rule, kp = armature * (2*pi*10 Hz)^2, kd = 2 * 2.0 * armature * (2*pi*10 Hz),
#                action scale = 0.25 * effort / kp (the K1 student asset uses exactly these numbers)
#
# The literal constants up to K1_ISAACLAB_TO_MUJOCO_MAPPING are parsed (ast.literal_eval, no Isaac Lab import)
# by sonic/robots.py on the Mac and by sonic_k1/tests — keep them literal.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

ARMATURE_QC080 = 0.01936542
ARMATURE_QC060 = 0.00564892
EFFORT_QC080 = 96.864
EFFORT_QC060 = 47.277
VELOCITY_QC080 = 11.519
VELOCITY_QC060 = 20.944

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_QC080 = ARMATURE_QC080 * NATURAL_FREQ**2  # 76.45
STIFFNESS_QC060 = ARMATURE_QC060 * NATURAL_FREQ**2  # 22.30
DAMPING_QC080 = 2.0 * DAMPING_RATIO * ARMATURE_QC080 * NATURAL_FREQ  # 4.867
DAMPING_QC060 = 2.0 * DAMPING_RATIO * ARMATURE_QC060 * NATURAL_FREQ  # 1.420

# Bodies in Isaac Lab order (root first). head_link is welded to torso_link (fixed joint) and merged.
K1_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
]

# Joints in MuJoCo / hardware order (ai_sapiens_sim2real/config/k1_config.yaml robot_joint_order)
K1_MUJOCO_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

# Index arrays: position i in the output = position mapping[i] in the input
K1_ISAACLAB_TO_MUJOCO_DOF = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]
K1_MUJOCO_TO_ISAACLAB_DOF = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
K1_ISAACLAB_TO_MUJOCO_BODY = [0, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 3, 6, 10, 14, 18, 22, 7, 11, 15, 19, 23]
K1_MUJOCO_TO_ISAACLAB_BODY = [0, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 6, 12, 18, 23]

# Standing pose (cyclo_lab K1 mimic/student default_position; the robot stands on it in MuJoCo sim2sim)
K1_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.63,
    ".*_ankle_pitch_joint": -0.33,
    ".*_elbow_joint": 0.6,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}
K1_INIT_HEIGHT = 0.77  # pelvis z with K1_DEFAULT_JOINT_POS is 0.765 (feet on the floor)

# per-joint motor tables (regex -> value), used for the actuator groups below and by sonic/robots.py
K1_EFFORT_LIMIT = {
    ".*_hip_.*_joint": 96.864,
    ".*_knee_joint": 96.864,
    ".*_ankle_pitch_joint": 96.864,
    ".*_ankle_roll_joint": 47.277,
    "waist_yaw_joint": 96.864,
    ".*_shoulder_.*_joint": 47.277,
    ".*_elbow_joint": 47.277,
    ".*_wrist_roll_joint": 47.277,
}
K1_VELOCITY_LIMIT = {
    ".*_hip_.*_joint": 11.519,
    ".*_knee_joint": 11.519,
    ".*_ankle_pitch_joint": 11.519,
    ".*_ankle_roll_joint": 20.944,
    "waist_yaw_joint": 11.519,
    ".*_shoulder_.*_joint": 20.944,
    ".*_elbow_joint": 20.944,
    ".*_wrist_roll_joint": 20.944,
}
K1_ARMATURE = {
    ".*_hip_.*_joint": 0.01936542,
    ".*_knee_joint": 0.01936542,
    ".*_ankle_pitch_joint": 0.03873084,
    ".*_ankle_roll_joint": 0.01129784,
    "waist_yaw_joint": 0.01936542,
    ".*_shoulder_.*_joint": 0.00564892,
    ".*_elbow_joint": 0.00564892,
    ".*_wrist_roll_joint": 0.00564892,
}

K1_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": K1_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": K1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": K1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": K1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": K1_MUJOCO_TO_ISAACLAB_BODY,
}


def _kp(armature):
    return armature * NATURAL_FREQ**2


def _kd(armature):
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _group(names):
    """ImplicitActuatorCfg fields for the regexes in `names`, derived from the motor tables above."""
    return dict(
        joint_names_expr=list(names),
        effort_limit_sim={n: K1_EFFORT_LIMIT[n] for n in names},
        velocity_limit_sim={n: K1_VELOCITY_LIMIT[n] for n in names},
        stiffness={n: _kp(K1_ARMATURE[n]) for n in names},
        damping={n: _kd(K1_ARMATURE[n]) for n in names},
        armature={n: K1_ARMATURE[n] for n in names},
    )


K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/k1/k1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, K1_INIT_HEIGHT),
        joint_pos=dict(K1_DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(**_group([".*_hip_.*_joint", ".*_knee_joint"])),
        "feet": ImplicitActuatorCfg(**_group([".*_ankle_pitch_joint", ".*_ankle_roll_joint"])),
        "waist_yaw": ImplicitActuatorCfg(**_group(["waist_yaw_joint"])),
        "arms": ImplicitActuatorCfg(**_group([".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"])),
    },
)

# K1 Action Scale (same rule as G1/H2: 0.25 * effort / kp)
K1_ACTION_SCALE = {}
for a in K1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            K1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
