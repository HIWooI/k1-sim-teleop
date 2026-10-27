# Portions derived from NVIDIA GR00T-WholeBodyControl (GEAR-SONIC), Copyright (c) NVIDIA CORPORATION & AFFILIATES,
# licensed under the Apache License 2.0 (https://github.com/NVlabs/GR00T-WholeBodyControl). Modified for the K1 port.
"""Constants mirrored 1:1 from the official C++ deploy.

Sources (GR00T-WholeBodyControl):
  gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
  gear_sonic/utils/mujoco_sim/configs.py (sim_frequency=200 -> SIMULATE_DT=0.005)
  gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml (ROBOT_SCENE)

Index conventions (same as the C++ names):
  * "mujoco order" == Unitree hardware motor order (legs L/R, waist, L arm, R arm).
  * "isaaclab order" == policy/action/observation order (left/right interleaved).
  * isaaclab_to_mujoco[mj] -> isaaclab index feeding MuJoCo joint mj
  * mujoco_to_isaaclab[il] -> MuJoCo index of IsaacLab joint il
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "GR00T-WholeBodyControl"
MODELS = ROOT / "models"
SCENE_XML = VENDOR / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
REFERENCE_DIR = VENDOR / "gear_sonic_deploy/reference/example"
HUMAN_JOINTS_INFO = VENDOR / "gear_sonic/data/human/human_joints_info.pkl"

G1_NUM_MOTOR = 29
CONTROL_DT = 0.02  # 50 Hz control thread
SIM_DT = 0.005  # run_sim_loop.py: sim_frequency=200
INIT_DURATION = 3.0  # G1Deploy::duration_ (INIT ramp to default pose)

# ---- motor constants (policy_parameters.hpp) --------------------------------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

# motor type per hardware (mujoco) index
_M = ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",  # left leg
      "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",  # right leg
      "7520_14", "5020", "5020",  # waist yaw/roll/pitch
      "5020", "5020", "5020", "5020", "5020", "4010", "4010",  # left arm
      "5020", "5020", "5020", "5020", "5020", "4010", "4010"]  # right arm
_STIFF = {"5020": STIFFNESS_5020, "7520_14": STIFFNESS_7520_14, "7520_22": STIFFNESS_7520_22,
          "4010": STIFFNESS_4010}
_DAMP = {"5020": DAMPING_5020, "7520_14": DAMPING_7520_14, "7520_22": DAMPING_7520_22,
         "4010": DAMPING_4010}
_EFF = {"5020": EFFORT_LIMIT_5020, "7520_14": EFFORT_LIMIT_7520_14,
        "7520_22": EFFORT_LIMIT_7520_22, "4010": EFFORT_LIMIT_4010}
# kps/kds get a 2x factor on ankles (4,5,10,11) and waist roll/pitch (13,14)
_DOUBLE = {4, 5, 10, 11, 13, 14}

G1_ACTION_SCALE = np.array([0.25 * _EFF[m] / _STIFF[m] for m in _M], dtype=np.float64)
# std::array<float,29> in C++ -> keep float32 rounding
KPS = np.array([(2.0 if i in _DOUBLE else 1.0) * _STIFF[m] for i, m in enumerate(_M)],
               dtype=np.float32).astype(np.float64)
KDS = np.array([(2.0 if i in _DOUBLE else 1.0) * _DAMP[m] for i, m in enumerate(_M)],
               dtype=np.float32).astype(np.float64)

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)  # mujoco order

ISAACLAB_TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

UPPER_BODY_JOINT_ISAACLAB_ORDER_IN_ISAACLAB_INDEX = [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23,
                                                     24, 25, 26, 27, 28]
WRIST_JOINT_ISAACLAB_ORDER_IN_ISAACLAB_INDEX = [23, 24, 25, 26, 27, 28]
LOWER_BODY_JOINT_MUJOCO_ORDER_IN_ISAACLAB_INDEX = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
LOWER_BODY_JOINT_ISAACLAB_ORDER_IN_ISAACLAB_INDEX = [0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ISAACLAB_JOINT_NAMES = [MUJOCO_JOINT_NAMES[ISAACLAB_TO_MUJOCO.tolist().index(i)]
                        for i in range(29)]

# Dex3 hands (sim only): dex3_hands_.open() -> q=0, kp=1.5, kd=0.1
HAND_KP = 1.5
HAND_KD = 0.1


def _self_check():
    # mujoco_to_isaaclab[il] is the MuJoCo index of IsaacLab joint il, so the two maps must be inverse
    assert np.all(ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB] == np.arange(29))
    assert np.all(MUJOCO_TO_ISAACLAB[ISAACLAB_TO_MUJOCO] == np.arange(29))
    assert ISAACLAB_JOINT_NAMES[1] == "right_hip_pitch_joint"
    assert ISAACLAB_JOINT_NAMES[23] == "left_wrist_roll_joint"


_self_check()
