"""Robot descriptions for the Mac deploy loop: everything SonicController / G1Sim / StreamBuffer need to know
about a robot, so a SONIC checkpoint trained for another embodiment plugs in with `robot=get_robot("k1")`.

  G1  official SONIC release (Unitree G1 29 DOF + Dex3 hands in the sim), constants from params.py (unchanged)
  K1  ROBOTIS K1 / AI Sapiens k1_rev1 (23 DOF). Single source of truth = the Isaac Lab robot config in
      sonic_k1/overlay/gear_sonic/envs/manager_env/robots/k1.py; its literal tables are read with ast here
      (no Isaac Lab on the Mac). Physics MJCF = the staged motion-library MJCF (sonic_k1/scripts/stage_assets.py)
      plus floor/light, with the training armatures written into the model (the vendor MJCF has armature 0).

Observation sizes follow from num_dof / wrist / lower-body index lists (sonic.deploy registry), e.g. release:
  G1 decoder 994 = 64 + 10*(3+29+29+29+3), encoder 1762;  K1 decoder 814 = 64 + 10*(3+23+23+23+3), encoder 1602
(`python -m sonic.robots` prints the table for every model variant).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import params as P

K1_DIR = P.ROOT / "sonic_k1"
K1_ROBOT_PY = K1_DIR / "overlay/gear_sonic/envs/manager_env/robots/k1.py"
K1_MJCF = K1_DIR / "overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml"


@dataclass(frozen=True, eq=False)
class RobotSpec:
    name: str
    mujoco_joint_names: tuple            # hardware / MuJoCo order
    isaaclab_to_mujoco: np.ndarray       # [mj] -> Isaac Lab index feeding MuJoCo joint mj
    mujoco_to_isaaclab: np.ndarray       # [il] -> MuJoCo index of Isaac Lab joint il
    default_angles: np.ndarray           # MuJoCo order
    kps: np.ndarray                      # MuJoCo order (float32-rounded like the C++ std::array<float>)
    kds: np.ndarray
    action_scale: np.ndarray             # MuJoCo order
    wrist_il: tuple                      # Isaac indices of the wrist joints (SMPL-encoder wrist input)
    lower_body_il: tuple                 # Isaac indices of the leg joints, in MuJoCo leg order (teleop encoder)
    standing_height: float               # pelvis z standing on default_angles (reference motion root z)
    scene: str | Path                    # MJCF path, or a callable-free XML string
    armature: np.ndarray | None = None   # MuJoCo order; written into the MuJoCo model when given
    band_height: float = 1.0             # elastic band anchor z during INIT (official: 1.0 m for G1)

    @property
    def num_dof(self) -> int:
        return len(self.mujoco_joint_names)

    @property
    def isaaclab_joint_names(self) -> list:
        return [self.mujoco_joint_names[self.isaaclab_to_mujoco.tolist().index(i)] for i in range(self.num_dof)]

    def load_model(self):
        import mujoco

        if isinstance(self.scene, Path):
            m = mujoco.MjModel.from_xml_path(str(self.scene))
        else:
            m = mujoco.MjModel.from_xml_string(self.scene)
        if self.armature is not None:
            for n, a in zip(self.mujoco_joint_names, self.armature):
                m.dof_armature[m.jnt_dofadr[m.joint(n).id]] = a
        return m

    def check(self):
        n = self.num_dof
        assert np.array_equal(self.isaaclab_to_mujoco[self.mujoco_to_isaaclab], np.arange(n))
        assert np.array_equal(self.mujoco_to_isaaclab[self.isaaclab_to_mujoco], np.arange(n))
        for a in (self.default_angles, self.kps, self.kds, self.action_scale):
            assert a.shape == (n,)
        return self


def g1() -> RobotSpec:
    return RobotSpec(
        name="g1",
        mujoco_joint_names=tuple(P.MUJOCO_JOINT_NAMES),
        isaaclab_to_mujoco=P.ISAACLAB_TO_MUJOCO,
        mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB,
        default_angles=P.DEFAULT_ANGLES,
        kps=P.KPS,
        kds=P.KDS,
        action_scale=P.G1_ACTION_SCALE,
        wrist_il=tuple(P.WRIST_JOINT_ISAACLAB_ORDER_IN_ISAACLAB_INDEX),
        lower_body_il=tuple(P.LOWER_BODY_JOINT_MUJOCO_ORDER_IN_ISAACLAB_INDEX),
        standing_height=0.78,
        scene=P.SCENE_XML,
    ).check()


# ------------------------------------------------------------------------------------------------ K1
def robot_py_literals(path=K1_ROBOT_PY) -> dict:
    """Module-level literal assignments of an Isaac Lab robot config (no import of isaaclab)."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _per_joint(table: dict, names, default=None):
    out = []
    for n in names:
        hits = [v for k, v in table.items() if re.fullmatch(k, n)]
        if len(hits) > 1:
            raise ValueError(f"{n}: matched by several patterns")
        if not hits and default is None:
            raise KeyError(n)
        out.append(hits[0] if hits else default)
    return np.array(out, dtype=np.float64)


K1_FLOOR = """
  <worldbody>
    <light name="sonic_key" pos="1.5 -1.0 3.0" dir="-0.4 0.3 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.82 0.84 0.87 1" friction="1.0 0.005 0.0001"/>
  </worldbody>
"""


def k1_scene_xml(mjcf=K1_MJCF) -> str:
    """Staged K1 MJCF (head merged, meshes in metres) + floor + light, mesh dir made absolute."""
    mjcf = Path(mjcf)
    if not mjcf.exists():
        raise FileNotFoundError(f"{mjcf} missing — run sonic_k1/scripts/stage_assets.py")
    xml = mjcf.read_text()
    meshdir = re.search(r'meshdir="([^"]+)"', xml).group(1)
    abs_mesh = (mjcf.parent / meshdir).resolve()
    if not any(abs_mesh.glob("*.stl")):
        raise FileNotFoundError(f"no meshes in {abs_mesh} — run sonic_k1/scripts/stage_assets.py")
    xml = xml.replace(f'meshdir="{meshdir}"', f'meshdir="{abs_mesh}"')
    return xml.replace("</mujoco>", K1_FLOOR + "</mujoco>")


@lru_cache(maxsize=1)
def k1() -> RobotSpec:
    L = robot_py_literals()
    names = L["K1_MUJOCO_JOINTS"]
    i2m = np.array(L["K1_ISAACLAB_TO_MUJOCO_DOF"])
    m2i = np.array(L["K1_MUJOCO_TO_ISAACLAB_DOF"])
    w = 10 * 2.0 * 3.1415926535  # NATURAL_FREQ, DAMPING_RATIO = 2 (same rule as G1 / k1.py)
    arm = _per_joint(L["K1_ARMATURE"], names)
    eff = _per_joint(L["K1_EFFORT_LIMIT"], names)
    kp = (arm * w ** 2).astype(np.float32).astype(np.float64)
    kd = (2 * 2.0 * arm * w).astype(np.float32).astype(np.float64)
    isaac = [names[m2i[i]] for i in range(len(names))]
    wrist = tuple(i for i, n in enumerate(isaac) if "wrist" in n)
    legs_mj = [n for n in names if any(t in n for t in ("hip", "knee", "ankle"))]
    lower = tuple(isaac.index(n) for n in legs_mj)
    return RobotSpec(
        name="k1",
        mujoco_joint_names=tuple(names),
        isaaclab_to_mujoco=i2m,
        mujoco_to_isaaclab=m2i,
        default_angles=_per_joint(L["K1_DEFAULT_JOINT_POS"], names, default=0.0),
        kps=kp,
        kds=kd,
        action_scale=0.25 * eff / (arm * w ** 2),
        wrist_il=wrist,
        lower_body_il=lower,
        standing_height=0.765,
        scene=k1_scene_xml(),
        armature=arm,
    ).check()


def get_robot(name: str) -> RobotSpec:
    return {"g1": g1, "k1": k1}[name.lower()]()


def obs_dims(robot: RobotSpec, variant_dir) -> dict:
    """Encoder/decoder input sizes the observation_config.yaml of `variant_dir` implies for `robot`."""
    from .deploy import ObsConfig, SonicController

    cfg = ObsConfig.load(Path(variant_dir) / "observation_config.yaml")
    reg = SonicController.registry_dims(robot, cfg.encoder_dimension or 64)
    dec = sum(reg[n] for n in cfg.observations)
    enc = sum(reg[n] for n in cfg.encoder_observations)
    return {"decoder": dec, "encoder": enc, "action": robot.num_dof, "token": cfg.encoder_dimension or 64}


if __name__ == "__main__":
    for r in ("g1", "k1"):
        spec = get_robot(r)
        for v in ("release", "low_latency", "sonic_v1_1"):
            d = P.MODELS / v
            if (d / "observation_config.yaml").exists():
                print(f"{r:3s} {v:12s}", obs_dims(spec, d))
