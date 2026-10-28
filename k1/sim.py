"""ROBOTIS AI Sapiens K1 (23 DOF) physics loop on a Mac CPU: MuJoCo + onnxruntime.

Port of the author's earlier K1 sim2sim (`sim/sim2sim.py --split`), kept numerically the same:

* model: vendor MJCF `ai_sapiens_description/mujoco/k1/k1.xml` + floor (friction 0.8 0.005 0.0001, the training
  physics material), one torque motor per joint in policy order, per-joint armature and the effort/velocity limits
  of the training env (K1_rev1), joint limits re-injected from `k1.urdf`, 200 Hz physics, 50 Hz control
* PD in code: tau = kp (target - q) - kd qd, torque zeroed when it would accelerate past the velocity limit,
  clipped to the effort limit; kp/kd are those of the policy that commands the joint (sim2real.yaml)
* split drive: arms = mimic student (`student_asset`), legs + waist = `loco_armdriven_r2` (standing, velocity
  command 0). The mimic is told the lower body tracks its reference perfectly (it is OOD otherwise), anchor =
  reference torso orientation in the robot torso frame with the entry yaw alignment
* the policies are driven through the ExpertAdapter they were exported with (observation order, history, action decode)

Only presentation was added: a light event scene and an offscreen camera.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
AI_SAPIENS = Path(os.environ.get("AI_SAPIENS_ROOT", str(HERE.parent / "third_party" / "ai_sapiens")))  # ROBOTIS ai_sapiens
K1_XML = AI_SAPIENS / "ai_sapiens_description/mujoco/k1/k1.xml"
MESHDIR = AI_SAPIENS / "ai_sapiens_description/meshes/k1_rev1"
URDF = ASSETS / "k1.urdf"
SIM_DT = 0.005
FPS = 50.0

PRESENTATION = """
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.38 0.38 0.38" specular="0 0 0"/>
    <rgba haze="0.92 0.93 0.95 1"/>
    <quality shadowsize="4096"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>"""
SCENE = """
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.80 0.83 0.88" width="512" height="3072"/>
    <texture type="2d" name="evfloor" builtin="checker" mark="edge" rgb1="0.86 0.87 0.89" rgb2="0.82 0.83 0.86"
      markrgb="0.74 0.75 0.78" width="300" height="300"/>
    <material name="evfloor" texture="evfloor" texuniform="true" texrepeat="4 4" reflectance="0.08"/>
    <material name="k1_white" rgba="0.86 0.86 0.87 1" specular="0.3" shininess="0.4"/>
  </asset>
  <worldbody>
    <light pos="1.5 -1.0 3.0" dir="-0.4 0.3 -1" directional="true" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <light pos="-1.0 1.5 3.0" dir="0.3 -0.4 -1" directional="true" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="evfloor" friction="0.8 0.005 0.0001"/>
  </worldbody>
  <actuator>
{motors}
  </actuator>
"""


def _effort(n):  # K1_rev1 training env actuator limits (sim2sim._effort)
    return 96.864 if (any(t in n for t in ("hip", "knee", "waist")) or "ankle_pitch" in n) else 47.277


def _vlim(n):  # velocity_limit_sim of the training env (sim2sim._vlim)
    return 20.944 if ("ankle_roll" in n or "shoulder" in n or "elbow" in n or "wrist" in n) else 11.519


def _armature(n):  # reflected inertia (sim2sim._armature)
    if "ankle_pitch" in n:
        return 0.03873084
    if "ankle_roll" in n:
        return 0.01129784
    if "hip" in n or "knee" in n or "waist" in n:
        return 0.01936542
    return 0.00564892


def urdf_position_limits(urdf_path, joint_names):
    """(lower, upper) in joint_names order (as the original bridge/safety_gate.py)."""
    limits = {}
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        node = joint.find("limit")
        if node is None or joint.get("type") in ("fixed", "floating"):
            continue
        lo, hi = node.get("lower"), node.get("upper")
        if lo is not None and hi is not None:
            limits[joint.get("name")] = (float(lo), float(hi))
    lo = np.array([limits.get(n, (-np.inf, np.inf))[0] for n in joint_names])
    hi = np.array([limits.get(n, (-np.inf, np.inf))[1] for n in joint_names])
    return lo, hi


def quat_to_mat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_mat(R):
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_model(joint_names, render_scene=None):
    """Vendor k1.xml with absolute meshdir, its own motors/force ranges replaced by ours, + event scene."""
    xml = K1_XML.read_text()
    xml = xml.replace('meshdir="../../meshes/k1_rev1"', f'meshdir="{MESHDIR}"')
    if 'scale="0.001 0.001 0.001"' not in xml:  # older vendor files had millimetre meshes without a scale
        xml = xml.replace('<mesh name=', '<mesh scale="0.001 0.001 0.001" name=')
    # the code clips torque to the training limits itself (sim2sim removes the vendor force ranges too)
    xml = re.sub(r' actuatorfrcrange="[^"]*"', '', xml)
    xml = re.sub(r' actuatorfrclimited="[^"]*"', '', xml)
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    xml = re.sub(r"<visual>.*?</visual>", PRESENTATION, xml, count=1, flags=re.S)
    motors = "\n".join(f'    <motor name="{n}" joint="{n}" ctrlrange="-500 500"/>' for n in joint_names)
    xml = xml.replace("</mujoco>", SCENE.format(motors=motors) + "</mujoco>")
    if render_scene == "snu":
        # Only the renderer requests this preset. Controller physics keeps the original scene.
        from k1live.scenery import snu_scene_xml
        xml = snu_scene_xml(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    if render_scene == "snu":
        from k1live.scenery import color_snu_sky
        color_snu_sky(model)
    # presentation only: the vendor "white" is a flat grey; a slightly brighter, softly specular white reads
    # better on a projector. Mass/inertia/collision are untouched.
    white = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "white")
    if white >= 0:
        model.mat_rgba[white] = [0.88, 0.88, 0.89, 1.0]
        model.mat_specular[white] = 0.35
        model.mat_shininess[white] = 0.45
    return model


def ort_session(path):
    """onnxruntime session for a small policy MLP: one thread, no spin-waiting. The default pool starts a thread per
    core that spins after every run (122 % CPU and 95 threads for the two K1 policies at 50 Hz on a 32-core PC,
    2 % with this, same outputs, slightly faster)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def _load_adapter(asset):
    """ExpertAdapter loaded by path (as the original sim_policy._load_adapter does), single-threaded session."""
    src = ASSETS / "student_asset" / "expert_adapter.py"
    spec = importlib.util.spec_from_file_location("expert_adapter", str(src))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["expert_adapter"] = mod
    spec.loader.exec_module(mod)
    adapter = mod.ExpertAdapter.from_asset_dir(asset, k1_config_path=ASSETS / "k1_config.yaml", load_session=False)
    adapter._session = ort_session(Path(asset) / "exported/policy.onnx")
    expected = adapter._session.get_inputs()[0].shape[-1]
    if int(expected) != adapter.observation_size:
        raise ValueError(f"{asset}: ONNX expects {expected} observations, sim2real.yaml describes "
                         f"{adapter.observation_size}")
    return adapter


class Student:
    """An exported K1 policy (ONNX + sim2real.yaml), as the original sim_policy.Student."""

    def __init__(self, asset):
        asset = Path(asset)
        cfg = yaml.safe_load((asset / "params/sim2real.yaml").read_text())
        self.cfg = cfg
        self.names = list(cfg["policy_joints"])
        self.dt = float(cfg["step_dt"])
        jp = cfg["joint_properties"]
        self.default = np.array([jp[n]["default_position"] for n in self.names], np.float64)
        self.kp = np.array([jp[n]["stiffness"] for n in self.names], np.float64)
        self.kd = np.array([jp[n]["damping"] for n in self.names], np.float64)
        self.adapter = _load_adapter(asset)
        self.last_action = np.zeros(len(self.names))

    def act(self, terms):
        self.adapter.push_observation(terms)
        raw = np.clip(self.adapter.act(), -10.0, 10.0)  # rsl_rl clips actions to +-10 before env.step
        self.last_action = raw
        return self.adapter.decode_action(raw, canonical=False)


class K1Split:
    """K1 in MuJoCo, arms by the mimic student, legs + waist by loco_armdriven_r2 (sim2sim --split).

    mode="mimic" drives the whole body with the mimic student instead (sim2sim without --split), for references
    that need hip pitch (the bow)."""

    def __init__(self, mimic_asset=ASSETS / "student_asset", loco_asset=ASSETS / "loco_armdriven_r2", mode="split"):
        assert mode in ("split", "mimic")
        self.mode = mode
        self.pol = Student(mimic_asset)
        self.loco = Student(loco_asset)
        assert self.pol.names == self.loco.names
        self.names = self.pol.names
        self.lower = np.array([any(t in n for t in ("hip", "knee", "ankle", "waist")) for n in self.names])
        self.arms = ~self.lower
        terms = list(self.loco.cfg["observations"])

        def pick(*subs):
            for t in terms:
                if any(s in t for s in subs):
                    return t
            raise SystemExit(f"loco obs term not found for {subs}: {terms}")

        self._lt = dict(ang=pick("ang_vel"), grav=pick("gravity"), cmd=pick("velocity_command", "command"),
                        act=pick("action"), jp=pick("joint_pos"), jv=pick("joint_vel"))
        m = self.model = build_model(self.names)
        m.opt.timestep = SIM_DT
        self.data = mujoco.MjData(m)
        self.decim = max(1, int(round(self.pol.dt / SIM_DT)))
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.names])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.names])
        self.torso_id = m.body("torso_link").id
        self.pelvis_id = m.body("pelvis").id
        self.tau_lim = np.array([_effort(n) for n in self.names])
        self.vel_lim = np.array([_vlim(n) for n in self.names])
        m.dof_armature[self.vadr] = [_armature(n) for n in self.names]
        self.lo, self.hi = urdf_position_limits(URDF, self.names)
        for n, lo, hi in zip(self.names, self.lo, self.hi):
            if np.isfinite(lo) and np.isfinite(hi):
                jid = m.joint(n).id
                m.jnt_limited[jid] = 1
                m.jnt_range[jid] = [lo, hi]
        split = self.lower if mode == "split" else np.zeros_like(self.lower)
        self.kp = np.where(split, self.loco.kp, self.pol.kp)
        self.kd = np.where(split, self.loco.kd, self.pol.kd)
        self.init_align = np.eye(3)
        self.t = 0
        self.last_policy_ms = 0.0

    # ---------------------------------------------------------------------------------------------------
    def reset(self, ref_torso_quat_xyzw=(0.0, 0.0, 0.0, 1.0), yaw=0.0, q0=None):
        """Split-mode entry: loco default standing pose, pelvis at 0.79 m (sim2sim.reset_state).
        Mimic mode: the reference's first joint pose, feet 1 cm above the floor (sim2sim teleports to frame 0)."""
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        if self.mode == "split" or q0 is None:
            d.qpos[self.qadr] = self.loco.default
            d.qpos[0:3] = [0.0, 0.0, 0.79]
        else:
            d.qpos[self.qadr] = q0
            d.qpos[2] = 1.0
            mujoco.mj_forward(self.model, d)
            foot = min(d.xpos[self.model.body(f"{s}_ankle_roll_link").id][2] for s in ("left", "right"))
            d.qpos[2] = 1.0 - foot + 0.075  # sole spheres sit 6.5 cm below the ankle_roll origin
        mujoco.mj_forward(self.model, d)
        R_t0 = quat_to_mat(np.asarray(ref_torso_quat_xyzw)[[3, 0, 1, 2]])
        self.init_align = yaw_mat(quat_to_mat(d.xquat[self.torso_id])) @ yaw_mat(R_t0).T
        for p in (self.pol, self.loco):
            p.adapter.reset_history(np.zeros(p.adapter.observation_size))
            p.adapter.set_last_action(np.zeros(len(self.names)), canonical=False)
            p.last_action = np.zeros(len(self.names))
        self.t = 0

    def step(self, rq, rdq, rtq_xyzw, cmd=None, target_override=None):
        """One 50 Hz control tick toward reference joints rq/rdq (policy order) and torso quat (xyzw).
        cmd: loco `velocity_commands` (vx, vy, wz) in the body frame, None = 0 (stand). target_override:
        {joint index: PD target} applied after the policies (k1live split backend: waist yaw)."""
        d = self.data
        t_pol = time.perf_counter()
        R_wb = quat_to_mat(d.xquat[self.torso_id])
        R_wt = quat_to_mat(np.asarray(rtq_xyzw)[[3, 0, 1, 2]])
        anchor = (R_wb.T @ (self.init_align @ R_wt))[:, :2].reshape(-1)
        jp, jv = d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy()
        if self.mode == "split":
            jp_m = np.where(self.lower, rq, jp)
            jv_m = np.where(self.lower, rdq, jv)
        else:
            jp_m, jv_m = jp, jv
        target = self.pol.act({
            "motion_command": np.concatenate([rq, rdq]),
            "motion_anchor_ori_b": anchor,
            "base_ang_vel": d.qvel[3:6].copy(),
            "joint_pos_rel": jp_m - self.pol.default,
            "joint_vel_rel": jv_m,
            "last_action": self.pol.last_action,
        })
        if self.mode == "split":
            R_root = quat_to_mat(d.qpos[3:7].copy())
            lt = self._lt
            loco_target = self.loco.act({
                lt["ang"]: d.qvel[3:6].copy(),
                lt["grav"]: R_root.T @ np.array([0.0, 0.0, -1.0]),
                lt["cmd"]: np.zeros(3) if cmd is None else np.asarray(cmd, np.float64),
                lt["jp"]: jp - self.loco.default,
                lt["jv"]: jv,
                lt["act"]: self.loco.last_action,
            })
            target = np.where(self.lower, loco_target, target)
        if target_override:
            target = target.copy()
            for i, v in target_override.items():
                target[i] = v
        self.last_policy_ms = (time.perf_counter() - t_pol) * 1e3
        for _ in range(self.decim):
            qd = d.qvel[self.vadr]
            tau = self.kp * (target - d.qpos[self.qadr]) - self.kd * qd
            tau = np.where((qd >= self.vel_lim) & (tau > 0), 0.0, tau)
            tau = np.where((qd <= -self.vel_lim) & (tau < 0), 0.0, tau)
            d.ctrl[:] = np.clip(tau, -self.tau_lim, self.tau_lim)
            mujoco.mj_step(self.model, d)
        self.t += 1
        return target

    # ---------------------------------------------------------------------------------------------------
    @property
    def q(self):
        return self.data.qpos[self.qadr].copy()

    def pelvis_height(self):
        return float(self.data.qpos[2])

    def tilt_deg(self):
        R = self.data.xmat[self.pelvis_id].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


class Camera:
    """Offscreen presentation camera with a slow pelvis follow (no shake)."""

    def __init__(self, sim: K1Split, w=1920, h=1080, azimuth=140.0, distance=2.0, elevation=-6.0, lookat_z=0.62):
        self.sim = sim
        self.r = mujoco.Renderer(sim.model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.distance, self.cam.elevation = azimuth, distance, elevation
        self.lookat_z = lookat_z
        self.opt = mujoco.MjvOption()  # geom groups 0-2 visible: visual meshes on, collision (group 3) off
        self.xy = None

    def grab(self):
        p = self.sim.data.xpos[self.sim.pelvis_id]
        xy = p[:2].copy()
        self.xy = xy if self.xy is None else 0.95 * self.xy + 0.05 * xy
        self.cam.lookat[:] = [self.xy[0], self.xy[1], self.lookat_z]
        self.r.update_scene(self.sim.data, camera=self.cam, scene_option=self.opt)
        return self.r.render().copy()
