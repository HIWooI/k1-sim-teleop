"""Backend interface of k1live.sim_server: one simulated K1 and the policies that drive it (MuJoCo only).

The sim_server owns everything around the robot: wall-clock pacing (50 Hz), smoothing / rate limits of the arm
targets, walking-speed limits, modes (live / idle / gesture), fall detection + automatic reset, rendering, state
publishing. A backend only turns one control tick of targets into policy actions and physics substeps:

    be = make_backend("split")          # or "sonic", "sonic_human", "hybrid"
    be.reset()
    info = be.step(arm_q, waist_yaw=None, bow=None, loco_cmd=(vx, vy, wz))   # one 20 ms tick
    if be.fallen(): ...

Conventions (k1live/protocol.py): radians, metres, robot body frame (x forward, y left, z up). `arm_q` is 10 floats
in protocol.ARM_JOINTS order, already rate-limited by the server; `loco_cmd` is the walking velocity command after
the server's speed/acceleration limits (zeros = stand). A backend may ignore inputs it cannot follow and says so in
`supports` (the split backend cannot bow: hip pitch belongs to the locomotion policy).

Observation helpers (pelvis_height, tilt_deg, root_pose, arm_q_now, qpos) are read by the server between ticks from
the same thread that calls step(); the renderer only ever sees a qpos copy, never `data` itself.
"""
from __future__ import annotations

from typing import Optional, Sequence

import mujoco
import numpy as np

from k1live.protocol import ARM_JOINTS, POLICY_JOINTS

CONTROL_DT = 0.02      # 50 Hz control tick
FALL_PELVIS_Z = 0.45   # m: below this the robot counts as fallen
FALL_TILT_DEG = 45.0   # pelvis z axis vs world z


class Backend:
    """Base class. Subclasses set the attributes below in __init__ and implement reset() and step()."""

    name = "base"
    dt = CONTROL_DT
    supports = {"arms": True, "waist_yaw": False, "bow": False, "walk": False}

    model: mujoco.MjModel
    data: mujoco.MjData
    names: list = POLICY_JOINTS        # 23 joint names, policy order
    qadr: np.ndarray                   # qpos addresses of `names`
    arm_idx: np.ndarray                # indices of ARM_JOINTS inside `names`
    arm_default: np.ndarray            # (10,) rest arm pose (what the server returns to when nobody is tracked)
    arm_lo: np.ndarray                 # (10,) joint limits
    arm_hi: np.ndarray
    pelvis_body: int                   # body id the follow camera tracks

    # ------------------------------------------------------------------------------------------- to implement
    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Respawn standing at (x, y) facing yaw, policies' histories cleared."""
        raise NotImplementedError

    def step(self, arm_q: Sequence[float], waist_yaw: Optional[float] = None, bow: Optional[float] = None,
             loco_cmd: Sequence[float] = (0.0, 0.0, 0.0), arm_dq: Optional[Sequence[float]] = None) -> dict:
        """One control tick (self.dt of simulated time). arm_dq: reference arm velocity (rad/s); None = the backend
        differentiates successive arm_q itself. A backend with supports["human"] also takes human=<the HumanFrame dict
        being followed> (None while nobody is followed), e.g. sonic_human reads its whole-body `smpl` field. Returns at least {"policy_ms": float (policy inference time of this
        tick), "target": (23,) PD joint targets}."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------- observations
    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis_body][2])

    def tilt_deg(self) -> float:
        R = self.data.xmat[self.pelvis_body].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def fallen(self) -> bool:
        return self.pelvis_height() < FALL_PELVIS_Z or self.tilt_deg() > FALL_TILT_DEG

    def root_pose(self) -> tuple:
        """(x, y, yaw) of the pelvis in the world."""
        R = self.data.xmat[self.pelvis_body].reshape(3, 3)
        p = self.data.xpos[self.pelvis_body]
        return float(p[0]), float(p[1]), float(np.arctan2(R[1, 0], R[0, 0]))

    def q_now(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def arm_q_now(self) -> np.ndarray:
        return self.data.qpos[self.qadr[self.arm_idx]].copy()

    def qpos(self) -> np.ndarray:
        """Copy of the full qpos (what the renderer needs to draw the robot)."""
        return self.data.qpos.copy()


def arm_indices(names) -> np.ndarray:
    return np.array([list(names).index(n) for n in ARM_JOINTS])


def make_backend(name: str, **kw) -> Backend:
    if name == "split":
        from k1live.backends.split import SplitBackend

        return SplitBackend(**kw)
    if name == "sonic":
        from k1live.backends.sonic import SonicBackend

        return SonicBackend(**kw)
    if name == "sonic_human":
        from k1live.backends.sonic_human import SonicHumanBackend  # whole-body SMPL encoder

        return SonicHumanBackend(**kw)
    if name == "hybrid":
        from k1live.backends.hybrid import HybridBackend  # split arms standing, SONIC + planner walking

        return HybridBackend(**kw)
    raise ValueError(f"unknown backend {name!r}")
