"""K1 robot config checks that need no Isaac Lab (Mac CPU, stdlib unittest + numpy + mujoco).

    .venv/bin/python -m unittest discover -s sonic_k1/tests -v

1. Isaac Lab order rule (BFS, fixed joints merged) reproduces SONIC's own G1 list and the K1 PhysX order
   recorded by ROBOTIS' cyclo_lab env -> trust it for K1_ISAACLAB_JOINTS.
2. k1.py literal index maps == maps computed from the URDF/MJCF, and they are inverse permutations.
3. MuJoCo FK of the motion-library MJCF == numpy FK of the Isaac URDF for random joint angles (every body).
4. Known-pose round trip: a pose written in Isaac order lands on the same named MuJoCo joints.
5. The motion-library MJCF parses the way gear_sonic Humanoid_Batch parses it (free root, 23 axes, 24 nodes).
6. Gains/action scales equal the K1 student asset's sim2real.yaml (same SONIC/BeyondMimic rule).
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE / "scripts"))
from k1_assets import index_maps, isaaclab_order, mujoco_order, parse_urdf, urdf_fk  # noqa: E402

DESC = HERE / "overlay/gear_sonic/data/assets/robot_description"
K1_PY = HERE / "overlay/gear_sonic/envs/manager_env/robots/k1.py"
VENDOR = REPO / "vendor/GR00T-WholeBodyControl"
STUDENT_SIM2REAL = REPO / "k1/assets/student_asset/params/sim2real.yaml"

# cyclo_lab K1 env PhysX order (LEGACY_PHYSX_* of the K1 sim2sim tooling)
CYCLO_LAB_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint"]
CYCLO_LAB_BODIES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "torso_link", "left_hip_roll_link", "right_hip_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_knee_link", "right_knee_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_elbow_link", "right_elbow_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"]


def literals(path):
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def regex_lookup(table, name):
    hits = [v for k, v in table.items() if re.fullmatch(k, name)]
    assert len(hits) == 1, (name, hits)
    return hits[0]


class TestOrder(unittest.TestCase):
    def test_bfs_rule_reproduces_sonic_g1(self):
        urdf = VENDOR / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
        if not urdf.exists():
            self.skipTest("vendor G1 URDF missing")
        g1 = literals(VENDOR / "gear_sonic/envs/manager_env/robots/g1.py")
        bodies, _ = isaaclab_order(parse_urdf(urdf))
        self.assertEqual(bodies, g1["G1_ISAACLAB_JOINTS"])

    def test_k1_matches_cyclo_lab_physx_order(self):
        bodies, dofs = isaaclab_order(parse_urdf(DESC / "urdf/k1/k1.urdf"))
        self.assertEqual(dofs, CYCLO_LAB_JOINTS)
        self.assertEqual(bodies, CYCLO_LAB_BODIES)
        self.assertEqual(bodies, literals(K1_PY)["K1_ISAACLAB_JOINTS"])

    @pytest.mark.needs_k1_meshes
    def test_index_maps(self):
        import mujoco

        lit = literals(K1_PY)
        _, idofs = isaaclab_order(parse_urdf(DESC / "urdf/k1/k1.urdf"))
        m = mujoco.MjModel.from_xml_path(str(DESC / "mjcf/k1.xml"))
        mb, mdofs = mujoco_order(m)
        self.assertEqual(mdofs, lit["K1_MUJOCO_JOINTS"])
        i2m, m2i = index_maps(idofs, mdofs)
        self.assertEqual(lit["K1_ISAACLAB_TO_MUJOCO_DOF"], i2m)
        self.assertEqual(lit["K1_MUJOCO_TO_ISAACLAB_DOF"], m2i)
        bi2m, bm2i = index_maps(lit["K1_ISAACLAB_JOINTS"], mb)
        self.assertEqual(lit["K1_ISAACLAB_TO_MUJOCO_BODY"], bi2m)
        self.assertEqual(lit["K1_MUJOCO_TO_ISAACLAB_BODY"], bm2i)
        for a, b in [(i2m, m2i), (bi2m, bm2i)]:
            a, b = np.array(a), np.array(b)
            np.testing.assert_array_equal(a[b], np.arange(len(a)))
            np.testing.assert_array_equal(b[a], np.arange(len(a)))


@pytest.mark.needs_k1_meshes
class TestKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco

        cls.mj = mujoco
        cls.m = mujoco.MjModel.from_xml_path(str(DESC / "mjcf/k1.xml"))
        cls.urdf = parse_urdf(DESC / "urdf/k1/k1.urdf")
        cls.lit = literals(K1_PY)

    def _mj_fk(self, q_named, root_pos, root_quat):
        m, mj = self.m, self.mj
        d = mj.MjData(m)
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat
        for n, v in q_named.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mj.mj_kinematics(m, d)
        return d

    def test_mjcf_equals_urdf_fk(self):
        rng = np.random.default_rng(1)
        J = self.urdf["joints"]
        names = self.lit["K1_MUJOCO_JOINTS"]
        for _ in range(20):
            q = {}
            for n in names:
                lo, hi = J[n]["limit"][:2]
                q[n] = rng.uniform(max(lo, -2.0), min(hi, 2.0))
            yaw = rng.uniform(-3, 3)
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            pos = rng.normal(0, 0.5, 3)
            d = self._mj_fk(q, pos, quat)
            Rroot = np.array(d.xmat[1]).reshape(3, 3)
            T = urdf_fk(self.urdf, q, pos, Rroot)
            for b in range(1, self.m.nbody):
                name = self.m.body(b).name
                np.testing.assert_allclose(d.xpos[b], T[name][1], atol=1e-6, err_msg=name)
                np.testing.assert_allclose(np.array(d.xmat[b]).reshape(3, 3), T[name][0], atol=1e-6, err_msg=name)

    def test_axes_and_limits_match_urdf(self):
        m = self.m
        for n in self.lit["K1_MUJOCO_JOINTS"]:
            j = m.joint(n).id
            np.testing.assert_allclose(m.jnt_axis[j], self.urdf["joints"][n]["axis"], err_msg=n)
            np.testing.assert_allclose(m.jnt_range[j], self.urdf["joints"][n]["limit"][:2], atol=1e-4, err_msg=n)

    def test_known_pose_round_trip(self):
        lit = self.lit
        rng = np.random.default_rng(2)
        _, idofs = isaaclab_order(self.urdf)
        q_isaac = rng.uniform(-0.5, 0.5, 23)
        q_mj = q_isaac[lit["K1_ISAACLAB_TO_MUJOCO_DOF"]]  # out[i] = in[mapping[i]]
        d = self._mj_fk(dict(zip(lit["K1_MUJOCO_JOINTS"], q_mj)), [0, 0, 0.77], [1, 0, 0, 0])
        for k, n in enumerate(idofs):  # read back by name from MuJoCo: same value Isaac had at that index
            self.assertAlmostEqual(d.qpos[self.m.jnt_qposadr[self.m.joint(n).id]], q_isaac[k])
        np.testing.assert_allclose(q_mj[lit["K1_MUJOCO_TO_ISAACLAB_DOF"]], q_isaac)
        # body positions: MuJoCo order -> Isaac order lands on the same named bodies
        xpos_mj = np.array(d.xpos[1:])
        xpos_isaac = xpos_mj[lit["K1_MUJOCO_TO_ISAACLAB_BODY"]]
        for k, n in enumerate(lit["K1_ISAACLAB_JOINTS"]):
            np.testing.assert_allclose(xpos_isaac[k], d.xpos[self.m.body(n).id])

    def test_humanoid_batch_parse(self):
        """Mirror of torch_humanoid_batch.Humanoid_Batch.__init__ / from_mjcf decisions (lines ~160-310)."""
        root = ET.parse(DESC / "mjcf/k1.xml").getroot()
        wb = root.find("worldbody")
        joints = wb.findall(".//joint")
        motors = sorted(mt.attrib["name"] for mt in root.find("actuator"))
        self.assertEqual(len(motors), 23)
        self.assertEqual(joints[0].attrib.get("type"), "free")  # else the [6:] branch drops 6 hinge axes
        dof_axis = [[int(i) for i in j.attrib["axis"].split(" ")] for j in joints[1:]]
        np.testing.assert_array_equal(dof_axis, [self.m.jnt_axis[self.m.joint(j.attrib["name"]).id] for j in joints[1:]])
        nodes = [b.attrib["name"] for b in wb.iter("body")]
        self.assertEqual(len(nodes), len(self.lit["K1_ISAACLAB_JOINTS"]))
        ranges = [j for j in joints if j.attrib.get("type") != "free"]
        self.assertTrue(all(j.attrib.get("range") for j in ranges))
        self.assertEqual(len(ranges), len(motors))
        for mt in root.find("actuator"):
            self.assertEqual(mt.attrib["name"], mt.attrib["joint"])
        comp = root.find("compiler")
        mesh_dir = (DESC / "mjcf" / comp.attrib["meshdir"]).resolve()
        if any(mesh_dir.glob("*.stl")):  # staged: metres (load_mesh ignores scale=)
            from k1_assets import read_stl

            tri, _ = read_stl(mesh_dir / "pelvis.stl")
            self.assertLess(np.abs(tri).max(), 1.0)
        self.assertFalse(any("scale" in me.attrib for me in root.find("asset").findall("mesh")))


class TestGains(unittest.TestCase):
    def test_against_student_sim2real(self):
        import yaml

        p = STUDENT_SIM2REAL
        if not p.exists():
            self.skipTest("K1 student asset sim2real.yaml missing")
        cfg = yaml.safe_load(p.read_text())
        lit = literals(K1_PY)
        w = 10 * 2.0 * 3.1415926535
        for k, n in enumerate(cfg["policy_joints"]):
            arm = regex_lookup(lit["K1_ARMATURE"], n)
            eff = regex_lookup(lit["K1_EFFORT_LIMIT"], n)
            kp, kd = arm * w ** 2, 2 * 2.0 * arm * w
            jp = cfg["joint_properties"][n]
            # the yaml stores 3 significant digits (153.0 for 152.9)
            self.assertAlmostEqual(kp, jp["stiffness"], delta=2e-3 * jp["stiffness"], msg=n)
            self.assertAlmostEqual(kd, jp["damping"], delta=max(0.01, 2e-3 * jp["damping"]), msg=n)
            self.assertAlmostEqual(0.25 * eff / kp, cfg["actions"]["joint_pos"]["scale"][k], delta=1e-3, msg=n)
            default = next((v for r, v in lit["K1_DEFAULT_JOINT_POS"].items() if re.fullmatch(r, n)), 0.0)
            self.assertAlmostEqual(default, jp["default_position"], msg=n)

    def test_every_joint_has_one_motor_entry(self):
        lit = literals(K1_PY)
        for n in lit["K1_MUJOCO_JOINTS"]:
            for t in ("K1_ARMATURE", "K1_EFFORT_LIMIT", "K1_VELOCITY_LIMIT"):
                regex_lookup(lit[t], n)


if __name__ == "__main__":
    unittest.main()
