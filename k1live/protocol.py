"""Message contract between the K1 live-teleop processes (MuJoCo simulation only - no robot I/O anywhere).

    sender (camera/video -> pose -> K1 targets)  --HUMAN-->  sim_server (controller + MuJoCo + render)
    service (operator panel / audience display)  --CTRL--->  sim_server, sender
    sim_server --SIM_STATE--> service,  sender --SENDER_STATE--> service

Process command lines (the service supervises both; state topics are published >= 1 Hz from start-up):
    python -m k1live.sender --source <camera index | video path> [--loop] [--no-mirror]
    python -m k1live.sim_server [--backend split|sonic]
CTRL is PUB/SUB, so the service re-sends its full settings as one `set` whenever a subscriber (re)connects:
a `set` that repeats the current value must be a no-op (never reopen the same camera/video).

All sockets are on 127.0.0.1. Each ZMQ message is two frames: topic bytes, UTF-8 JSON object.
Publishers bind, subscribers connect (service binds CTRL; sim_server binds SIM_STATE; sender binds HUMAN and
SENDER_STATE). Video is plain HTTP multipart MJPEG: sender -> /cam.mjpg on CAM_HTTP_PORT (camera with skeleton
overlay), sim_server -> /robot.mjpg on ROBOT_HTTP_PORT (robot render). The service page embeds both.

Frames and units: radians, metres, seconds. Robot commands are in the K1 body frame (x forward, y left, z up,
yaw counter-clockwise seen from above). "Mirror" (default on) means the robot moves like the person's reflection:
the person's right arm drives the robot's left arm, a step to the person's right moves the robot to its left.
The sender applies the mirror before publishing, so arm_q and loco are already in robot terms.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

HOST = "127.0.0.1"
# Several stacks can share one machine (e.g. parallel test runs): every port is shifted by K1LIVE_PORT_OFFSET
# (default 0; e.g. 100, 200, ... per stack). Child processes inherit the variable from the service.
# Never hard-code ports elsewhere - use these constants.
PORT_OFFSET = int(os.environ.get("K1LIVE_PORT_OFFSET", "0"))
HUMAN_PORT = 5701 + PORT_OFFSET         # sender -> sim_server, ~30 Hz
CTRL_PORT = 5702 + PORT_OFFSET          # service -> sim_server, sender
SIM_STATE_PORT = 5703 + PORT_OFFSET     # sim_server -> service, 10 Hz
SENDER_STATE_PORT = 5704 + PORT_OFFSET  # sender -> service, 2-5 Hz
UI_HTTP_PORT = 8700 + PORT_OFFSET       # service: operator panel /, audience display /display
CAM_HTTP_PORT = 8701 + PORT_OFFSET      # sender: /cam.mjpg
ROBOT_HTTP_PORT = 8702 + PORT_OFFSET    # sim_server: /robot.mjpg

TOPIC_HUMAN = b"human"
TOPIC_CTRL = b"ctrl"
TOPIC_SIM_STATE = b"sim"
TOPIC_SENDER_STATE = b"snd"

# K1 policy joint order (k1/assets/*/params/sim2real.yaml `policy_joints`)
LEG_JOINTS = [f"{s}_{j}" for s in ("left", "right")
              for j in ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint",
                        "ankle_pitch_joint", "ankle_roll_joint")]
WAIST_JOINTS = ["waist_yaw_joint"]
ARM_JOINTS = [f"{s}_{j}" for s in ("left", "right")
              for j in ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
                        "wrist_roll_joint")]
POLICY_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS  # 23

# operator commands (CtrlMsg.cmd)
CMD_START = "start"          # stream the person into the robot
CMD_STOP = "stop"            # stop streaming; robot returns to standing / idle behaviour
CMD_RESET = "reset"          # respawn the robot standing at the origin
CMD_CALIBRATE = "calibrate"  # sender: capture neutral pose + home position (person stands still, facing camera)
CMD_SET = "set"              # args: any of mode, backend, walk, mirror, max_speed, camera, video,
                             #   loco_profile ("visitor" | "presenter"), crop ([x0, y0, x1, y1] normalized image box or None)
CMD_GESTURE = "gesture"      # args: {"name": <k1.gestures name>} play a built-in gesture (idle/attract mode)
CMD_DRIVE = "drive"          # args: {"vx","vy","wz"} operator walking override; sim drops it 0.3 s after the last one
CMD_ESTOP = "estop"          # stop streaming + walking, robot to standing, ignore person until CMD_START
CMD_QUIT = "quit"

LOCO_PROFILES = ("visitor", "presenter")  # visitor = step-off joystick (default); presenter = no turning, walk along only
MODES = ("live", "idle", "gesture")   # live = follow person; idle = stand / attract loop; gesture = scripted clip
BACKENDS = ("split", "sonic", "sonic_human", "hybrid")  # split = student arms + ROBOTIS loco legs; sonic = K1 SONIC whole-body
                                               # (robot reference: person's arms + planner legs); sonic_human = K1 SONIC
                                               # driven by the person's whole-body pose (SMPL encoder, BONES-SEED training)
                                               # hybrid = split arms while standing, K1 SONIC walk + planner legs while walking


def now() -> float:
    return time.time()


@dataclass
class HumanFrame:
    """One perception result. Everything already mirrored/mapped into robot terms by the sender."""
    seq: int
    t_capture: float                    # time.time() when the camera frame was grabbed
    present: bool                       # a person is tracked with enough confidence
    conf: float = 0.0                   # 0..1
    arm_q: Optional[list] = None        # 10 floats, ARM_JOINTS order, IK result clamped to URDF limits
    waist_yaw: Optional[float] = None   # torso yaw relative to pelvis (robot waist_yaw_joint target)
    bow: Optional[float] = None         # torso forward pitch, 0 = upright, + = leaning forward
    root: Optional[dict] = None         # {"x","y","yaw"}: pelvis offset from the calibrated home, robot terms
    legs_visible: bool = False
    loco: Optional[dict] = None         # {"vx","vy","wz"} walking intent (robot body frame), before sim limits
    calibrated: bool = False
    t_sent: float = field(default_factory=now)
    # whole-body human pose for the sonic_human backend (None if the sender does not compute it):
    # {"joints": 24x3 root-local SMPL joints (Z-up, sonic/mp2smpl output), "quat": [w,x,y,z] root orientation
    #  (Z-up, SMPL base removed, camera-calibrated), "wrist": [l, r] K1 wrist roll}; mirror already applied
    smpl: Optional[dict] = None


@dataclass
class CtrlMsg:
    cmd: str
    args: dict = field(default_factory=dict)
    t: float = field(default_factory=now)


@dataclass
class SimState:
    t: float
    mode: str
    backend: str
    streaming: bool
    walk_enabled: bool
    estop: bool
    fallen: bool
    resets: int                         # automatic + manual resets since start
    pelvis_z: float
    tilt_deg: float
    loco_cmd: list                      # [vx, vy, wz] actually applied this tick
    human_present: bool
    human_age_ms: Optional[float]       # now - t_capture of the last HumanFrame used
    ctrl_hz: float                      # measured control rate (target 50)
    rt_factor: float                    # sim time / wall time over the last second (target 1.0)
    gesture: Optional[str] = None
    arm_err_deg: Optional[float] = None
    note: str = ""                      # "loading" while models load (publish state >= 1 Hz from process start)
    backends: list = field(default_factory=list)  # backends usable right now, e.g. ["split", "sonic"]


@dataclass
class SenderState:
    t: float
    cam_ok: bool
    source: str                         # "camera:0" or a video path
    fps: float
    present: bool
    calibrated: bool
    pose_ms: float                      # pose + IK time per frame
    note: str = ""                      # "loading" while MediaPipe loads


def encode(topic: bytes, msg) -> list:
    return [topic, json.dumps(asdict(msg) if hasattr(msg, "__dataclass_fields__") else msg).encode()]


def decode(frames: list, cls=None):
    d = json.loads(frames[1].decode())
    return cls(**d) if cls is not None else d
