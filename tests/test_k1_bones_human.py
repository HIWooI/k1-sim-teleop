"""K1 BONES-SEED human encoder: data conventions (train == live), pairing on the public BONES sample, sonic_human
backend observation layout / heading alignment / fallback (dummy ONNX), sender SMPL stream. MuJoCo simulation only.

    .venv/bin/python -m unittest tests.test_k1_bones_human -v      # pytest also works

Needs models/sample_data (GEAR-SONIC sample, BONES walk clip) and the staged K1 MJCF (sonic_k1/scripts/stage_assets.py)
for the pairing / backend tests; those are skipped when missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sonic_k1_train" / "data"))

from k1live import smpl_stream  # noqa: E402
from sonic import rot  # noqa: E402

SAMPLE = ROOT / "models" / "sample_data"
MJCF = ROOT / "sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml"
HAVE_SAMPLE = (SAMPLE / "soma_filtered").exists() and (SAMPLE / "robot_filtered").exists()
HAVE_K1 = MJCF.exists() and any((MJCF.parent / "../urdf/k1/meshes").glob("*.stl"))
# sim_server offers sonic_human only next to the sonic checkpoint, which is trained locally (not in the Release)
HAVE_SONIC = (ROOT / "models" / "k1" / "k1s_la4" / "deploy_002000").exists()


def rand_quats(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def sample_landmarks():
    """BONES sample walk (SOMA-26, 30 Hz) -> world landmarks (T, 33, 3)."""
    import joblib

    import human_smpl as H

    key = "walk_forward_amateur_001__A001"
    s = joblib.load(next((SAMPLE / "soma_filtered").rglob(f"{key}.pkl")))[key]
    t = s["soma_transl"]
    jz = s["soma_joints"] + np.stack([t[:, 0], -t[:, 2], t[:, 1]], 1)[:, None]
    return H.soma26_landmarks(jz)


class TestConventions(unittest.TestCase):
    def test_root_chain_roundtrip(self):
        q = rand_quats(200)
        back = smpl_stream.root_quat_from_aa_yup(smpl_stream.root_aa_yup(q))
        self.assertLess(np.abs(np.abs(np.sum(back * q, axis=1)) - 1).max(), 1e-9)

    def test_root_chain_is_gear_sonic(self):
        """root_quat_from_aa_yup == remove_smpl_base_rot(smpl_root_ytoz_up(aa2quat(aa))) written with scipy."""
        from scipy.spatial.transform import Rotation as R

        aa = np.random.default_rng(1).normal(size=(50, 3))
        ref = (R.from_rotvec([np.pi / 2, 0, 0]) * R.from_rotvec(aa) * R.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()
        mine = smpl_stream.root_quat_from_aa_yup(aa)[:, [1, 2, 3, 0]]
        self.assertLess(np.abs(np.abs(np.sum(ref * mine, axis=1)) - 1).max(), 1e-9)

    def test_mirror(self):
        J = np.random.default_rng(2).normal(size=(24, 3))
        q = rand_quats(1, 3)[0]
        w = np.array([0.3, -0.1])
        J2, q2, w2 = smpl_stream.mirror(*smpl_stream.mirror(J, q, w))
        np.testing.assert_allclose(J2, J)
        np.testing.assert_allclose(q2, q)
        np.testing.assert_allclose(w2, w)
        Jm, qm, wm = smpl_stream.mirror(J, q, w)
        np.testing.assert_allclose(Jm[20], J[21] * [1, -1, 1])           # left wrist <- right wrist, y reflected
        np.testing.assert_allclose(wm, [0.1, -0.3])                      # wrist rolls swapped and negated
        yaw = lambda x: rot.calc_heading(x)  # noqa: E731
        self.assertAlmostEqual(float(yaw(qm)), -float(yaw(q)), places=9)
        # world joints of the mirror = reflected world joints of the original (same reflection on both sides)
        np.testing.assert_allclose(smpl_stream.world_joints(Jm, qm),
                                   smpl_stream.world_joints(J, q)[smpl_stream.MIRROR_PERM] * [1, -1, 1], atol=1e-12)

    def test_msg_roundtrip(self):
        J = np.random.default_rng(4).normal(size=(24, 3))
        q = rand_quats(1, 5)[0]
        d = json.loads(json.dumps(smpl_stream.to_msg(J, q, [0.1, 0.2])))
        J2, q2, w2 = smpl_stream.from_msg(d)
        np.testing.assert_allclose(J2, J - J[0], atol=2e-4)
        np.testing.assert_allclose(q2, q, atol=1e-6)
        np.testing.assert_allclose(w2, [0.1, 0.2])
        with self.assertRaises(ValueError):
            smpl_stream.from_msg({"joints": [[0, 0, 0]] * 23, "quat": [1, 0, 0, 0]})


@unittest.skipUnless(HAVE_SAMPLE, "models/sample_data missing")
class TestHumanSmpl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import human_smpl as H

        cls.L = sample_landmarks()[:300]
        cls.J, cls.Q = H.smpl_track(cls.L)

    def test_canonical_skeleton(self):
        """Every bone has the canonical SMPL rest length, whatever the actor: the live converter's skeleton."""
        from sonic import smpl

        par = smpl.parents24()
        rest = smpl.rest_joints24()
        L0 = np.array([np.linalg.norm(rest[i] - rest[par[i]]) for i in range(1, 24)])
        L = np.linalg.norm(self.J[:, 1:] - self.J[:, par[1:]], axis=-1)
        self.assertLess(np.abs(L - L0).max(), 1e-6)
        self.assertLess(np.abs(self.J[:, 0]).max(), 1e-12)               # pelvis at the origin

    def test_tpose_sides(self):
        J = self.J[0]  # the BONES take starts in a T-pose
        self.assertGreater(J[20, 1], 0.4)                                # left wrist on +y (person's left)
        self.assertLess(J[21, 1], -0.4)
        self.assertGreater(J[15, 2], 0.45)                               # head up
        self.assertLess(J[7, 2], -0.7)                                   # ankles down

    def test_train_equals_live_converter(self):
        """Training data and the live sender share one geometry: on clean landmarks (legs visible, no glitches)
        the live converter gives the training converter's output."""
        import human_smpl as H

        conv = smpl_stream.make_converter(live=True)
        mp = H.to_mediapipe(self.L)
        for t in range(len(mp)):
            f = conv.convert(mp[t], np.ones(33), t / 30.0)
            np.testing.assert_allclose(smpl_stream.zero_pelvis(f["smpl_joints"]), self.J[t], atol=1e-9)
            self.assertGreater(abs(float(np.dot(f["body_quat"], self.Q[t]))), 1 - 1e-6)  # float32 output


@unittest.skipUnless(HAVE_SAMPLE and HAVE_K1, "sample data or staged K1 MJCF missing")
class TestPairing(unittest.TestCase):
    def test_sample_pairs(self):
        import joblib

        import bones_to_k1 as B

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            B.main(["sample", "--out", str(out)])
            B.main(["check", "--out", str(out), "--timing", "2"])
            chk = json.loads((out / "check.json").read_text())
            tr = chk["sets"]["."]
            self.assertEqual((tr["robot"], tr["smpl"], tr["n_problems"]), (2, 2, 0))
            self.assertLess(chk["stats"]["heading_diff_deg"]["max"], 2.0)
            self.assertTrue(all(r["best_lag_frames"] == 0 and r["corr_lag0"] > 0.9 for r in chk["timing"]))
            k = "walk_forward_amateur_001__A001"
            e = joblib.load(out / "robot" / "bones_sample" / f"{k}.pkl")[k]
            s = joblib.load(out / "smpl" / f"{k}.pkl")
            self.assertEqual(e["fps"], 50)
            self.assertEqual(s["fps"], 50.0)
            self.assertEqual(len(e["dof"]), len(s["smpl_joints"]))
            self.assertEqual(s["pose_aa"].shape, (len(e["dof"]), 72))


class TestSelect(unittest.TestCase):
    def test_rules_and_budget(self):
        import csv

        import bones_select as S

        cols = ["filename", "move_duration_frames", "package", "category", "content_type_of_movement",
                "content_body_position", "content_props", "content_vertical_move", "content_horizontal_move",
                "content_uniform_style", "content_name", "content_short_description", "move_g1_mujoco_path",
                "move_soma_uniform_path", "actor_uid"]
        base = dict(package="Locomotion", category="Basic Locomotion Neutral", content_type_of_movement="walking",
                    content_body_position="standing", content_props="0", content_vertical_move="0",
                    content_horizontal_move="1", content_uniform_style="neutral", content_name="",
                    content_short_description="", actor_uid="A1")
        rows = []
        def add(name, sec, **kw):
            for m in ("", "_M"):
                r = dict(base, **kw, filename=name + m, move_duration_frames=str(int(sec * 120)))
                r["move_g1_mujoco_path"] = f"g1/csv/210531/{name + m}.csv"
                r["move_soma_uniform_path"] = f"soma_uniform/bvh/210531/{name + m}.bvh"
                rows.append(r)
        add("walk_forward_001__A1", 10)
        add("turn_left_90_001__A1", 5, content_type_of_movement="turning")
        add("wave_hello_R_001__A1", 4, category="Gestures", package="Communication", content_type_of_movement="gesture",
            content_horizontal_move="0")
        add("idle_breathing_001__A1", 6, category="Baseline", content_type_of_movement="idle", content_horizontal_move="0")
        add("jump_forward_001__A1", 3, content_type_of_movement="jumping")
        add("sit_down_chair_001__A1", 5, content_body_position="sitting")
        add("walk_injured_leg_001__A1", 8, content_uniform_style="injured leg")
        add("carry_box_walk_001__A1", 8, content_props="box")
        add("run_fast_001__A1", 8, content_type_of_movement="running")
        add("walk_long_001__A1", 60)
        with tempfile.TemporaryDirectory() as td:
            meta = Path(td) / "seed_metadata_v004.csv"
            with open(meta, "w", newline="") as f:
                w = csv.DictWriter(f, cols)
                w.writeheader()
                w.writerows(rows)
            clips = S.main(meta.with_suffix(".parquet"), Path(td) / "set", max_hours=10.0)
            keys = {c["key"] for c in clips}
            self.assertEqual(keys, {"walk_forward_001__A1", "walk_forward_001__A1_M", "turn_left_90_001__A1",
                                    "turn_left_90_001__A1_M", "wave_hello_R_001__A1", "wave_hello_R_001__A1_M",
                                    "idle_breathing_001__A1", "idle_breathing_001__A1_M"})
            groups = {c["key"]: c["group"] for c in clips}
            self.assertEqual((groups["turn_left_90_001__A1"], groups["wave_hello_R_001__A1_M"],
                              groups["idle_breathing_001__A1"]), ("walk", "gesture", "idle"))
            # budget: walk share = 45 % of 0.012 h = 19.4 s -> the 2 x 10 s walk does not fit, the 2 x 5 s turn does
            clips = S.main(meta, Path(td) / "small", max_hours=0.012)
            self.assertEqual(sorted(c["key"] for c in clips if c["group"] == "walk"),
                             ["turn_left_90_001__A1", "turn_left_90_001__A1_M"])
            self.assertTrue((Path(td) / "small" / "selection.md").exists())


class TestExtract(unittest.TestCase):
    def test_selected_members_only(self):
        import io
        import tarfile

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            arc = td / "g1.tar.gz"
            with tarfile.open(arc, "w:gz") as t:
                for name, data in (("seed/g1/csv/210531/a.csv", b"A"), ("seed/g1/csv/210531/b.csv", b"B"),
                                   ("seed/g1/csv/210601/c.csv", b"C")):
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    t.addfile(ti, io.BytesIO(data))
            sel = td / "selection.json"
            sel.write_text(json.dumps({"clips": [{"g1": "g1/csv/210531/a.csv"}, {"g1": "g1/csv/210601/c.csv"}]}))
            r = subprocess.run([sys.executable, str(ROOT / "sonic_k1_train/box/bones_extract.py"), "--archive", str(arc),
                                "--selection", str(sel), "--field", "g1", "--dest", str(td / "out")],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((td / "out/g1/csv/210531/a.csv").read_bytes(), b"A")
            self.assertEqual((td / "out/g1/csv/210601/c.csv").read_bytes(), b"C")
            self.assertFalse((td / "out/g1/csv/210531/b.csv").exists())
            sel.write_text(json.dumps({"clips": [{"g1": "../evil.csv"}]}))
            r = subprocess.run([sys.executable, str(ROOT / "sonic_k1_train/box/bones_extract.py"), "--archive", str(arc),
                                "--selection", str(sel), "--field", "g1", "--dest", str(td / "out")],
                               capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)


# ------------------------------------------------------------------------------------------------ backend
def make_dummy_variant(vdir: Path, smpl_frames: int, n_future: int = 4, step: int = 1):
    """Zero-weight encoder / decoder ONNX with the input sizes of a human-encoder K1 export (plumbing tests)."""
    sys.path.insert(0, str(ROOT / "sonic_k1" / "scripts"))
    from dummy_onnx import matmul_model
    import yaml

    from k1live.backends.sonic_human import _controller_cls, obs_config_human
    from sonic.robots import get_robot

    vdir.mkdir(parents=True, exist_ok=True)
    cfg = obs_config_human(n_future, step, smpl_frames)
    (vdir / "observation_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cls = _controller_cls()
    probe = cls.__new__(cls)
    probe.robot, probe.token_dim, probe._cfg_path = get_robot("k1"), 64, vdir / "observation_config.yaml"
    reg = probe._build_registry()
    din = sum(reg[o["name"]][0] for o in cfg["encoder"]["encoder_observations"])
    ddec = sum(64 if o["name"] == "token_state" else reg[o["name"]][0] for o in cfg["observations"])
    (vdir / "model_encoder.onnx").write_bytes(matmul_model(din, 64))
    (vdir / "model_decoder.onnx").write_bytes(matmul_model(ddec, 23))
    return din, ddec


def person_msg(yaw=0.0, left_arm_up=False):
    """HumanFrame-like dict of a standing person (idle pose), facing yaw, optionally the left arm straight up."""
    f = smpl_stream.make_converter(live=False).idle_frame()
    J = smpl_stream.zero_pelvis(f["smpl_joints"])
    if left_arm_up:
        J[18] = J[16] + [0, 0, np.linalg.norm(J[18] - J[16])]
        J[20] = J[18] + [0, 0, np.linalg.norm(f["smpl_joints"][20] - f["smpl_joints"][18])]
        J[22] = J[20] + [0, 0, 0.08]
    q = rot.quat_from_angle_axis(np.array(yaw), np.array([0.0, 0.0, 1.0]))
    return {"present": True, "smpl": smpl_stream.to_msg(J, q, [0.1, -0.1])}


@unittest.skipUnless(HAVE_K1, "staged K1 MJCF missing")
class TestSonicHumanBackend(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_available(self):
        from k1live.backends.sonic_human import SonicHumanBackend

        self.assertFalse(SonicHumanBackend.available(self.root / "nothing"))
        make_dummy_variant(self.root / "v4", 4)
        self.assertTrue(SonicHumanBackend.available(self.root / "v4"))

    def _run(self, smpl_frames):
        from k1live.backends.sonic_human import SonicHumanBackend

        din, _ = make_dummy_variant(self.root / f"v{smpl_frames}", smpl_frames)
        be = SonicHumanBackend(self.root / f"v{smpl_frames}")
        be.reset(yaw=0.7)
        be.sim.set_band(True)                                 # zero-action dummy policy: keep it hanging upright
        arm = be.arm_default.copy()
        info = be.step(arm)                                   # nobody: robot encoder, standing
        self.assertEqual(info["mode"], "robot")
        for _ in range(60):                                   # 1.2 s person facing the camera (yaw 0)
            info = be.step(arm, human=person_msg(0.0))
        self.assertEqual(info["mode"], "human")
        self.assertEqual(be.gate.state, "TRACK")
        obs = be.ctrl.last_encoder_obs
        self.assertEqual(len(obs), din)
        lay = {n: (o, d) for n, o, d in be.ctrl.encoder_layout}
        sfx = "" if smpl_frames == 10 else f"_{smpl_frames}f"
        seg = lambda name: obs[lay[name][0]:lay[name][0] + lay[name][1]]  # noqa: E731
        self.assertEqual(seg("encoder_index")[0], 2.0)
        np.testing.assert_allclose(seg("encoder_index_onehot"), [0, 0, 1])
        J = seg(f"smpl_joints_multi_future_local_nonflat{sfx}").reshape(smpl_frames, 24, 3)
        self.assertLess(np.abs(J[-1] - person_msg_joints()).max(), 2e-3)  # newest frame = the person (faded in)
        self.assertLess(np.abs(J[:, 0]).max(), 1e-9)                       # pelvis at the origin, as in training
        np.testing.assert_allclose(seg(f"joint_pos_multi_future_wrist_for_smpl{sfx}").reshape(smpl_frames, 2)[-1],
                                   [0.1, -0.1], atol=1e-9)
        cmd = [n for n in lay if n.startswith("command_multi_future_nonflat")][0]
        self.assertEqual(float(np.abs(seg(cmd)).max()), 0.0)              # g1 inputs off in SMPL mode
        # heading alignment: the person's heading at stream start (0, facing the camera) == the robot's (0.7)
        ctrl = be.ctrl
        q_person = smpl_stream.from_msg(person_msg(0.0)["smpl"])[1]
        adh = ctrl.apply_delta_heading()
        self.assertAlmostEqual(float(rot.calc_heading(rot.quat_mul(adh, q_person))), 0.7, delta=0.02)
        # the encoder's root term is the training one: 6D of inv(robot pelvis) * aligned person root
        r6 = seg(f"smpl_root_ori_b_multi_future{sfx}").reshape(smpl_frames, 6)
        exp = rot.rot6d_rowwise(rot.quat_mul(rot.quat_conj(ctrl._latest_base_quat()), rot.quat_mul(adh, q_person)))
        np.testing.assert_allclose(r6[-1], exp, atol=2e-3)
        # the person turns 0.3 rad to their left -> the aligned reference turns with them
        for _ in range(40):
            be.step(arm, human=person_msg(0.3))
        q_ref = ctrl.current_motion.body_quat[-1, 0]
        self.assertAlmostEqual(float(rot.calc_heading(rot.quat_mul(ctrl.apply_delta_heading(), q_ref))), 1.0,
                               delta=0.03)
        # person gone: hold, fade to the standing pose, then back to the robot encoder
        modes = [be.step(arm)["mode"] for _ in range(150)]
        self.assertEqual(modes[0], "human")
        self.assertEqual(modes[-1], "robot")
        self.assertEqual(be.ctrl.current_motion.encode_mode, 0)

    def test_layout_4_frames(self):
        self._run(4)

    def test_layout_10_frames(self):
        self._run(10)

    @unittest.skipUnless(HAVE_SONIC, "sonic checkpoint models/k1/k1s_la4/deploy_002000 missing")
    def test_available_backends_lists_it(self):
        make_dummy_variant(self.root / "v4", 4)
        env = dict(os.environ, K1LIVE_HUMAN_VARIANT=str(self.root / "v4"))
        out = subprocess.run([sys.executable, "-c", "from k1live.sim_server import available_backends; "
                              "print(','.join(available_backends()))"], cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertIn("sonic_human", out.stdout.strip().split(","), out.stderr[-500:])
        env["K1LIVE_HUMAN_VARIANT"] = str(self.root / "missing")
        out = subprocess.run([sys.executable, "-c", "from k1live.sim_server import available_backends; "
                              "print(','.join(available_backends()))"], cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertNotIn("sonic_human", out.stdout.strip().split(","))


def person_msg_joints():
    return smpl_stream.from_msg(person_msg(0.0)["smpl"])[0]


# ------------------------------------------------------------------------------------------------ sender
@pytest.mark.needs_smpl
class TestSenderSmpl(unittest.TestCase):
    def test_mapper_smpl_and_mirror(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_k1live_core import synthetic_det

        from k1live.human import HumanMapper

        out = {}
        for mirror in (False, True):
            m = HumanMapper(mirror=mirror, loco=False, auto_calibrate=False)
            for i in range(5):
                r = m.process(synthetic_det(left_up=150.0), 100.0 + i / 30, 1280, 720)
            out[mirror] = smpl_stream.from_msg(r["smpl"])
            self.assertEqual(len(r["smpl"]["joints"]), 24)
        J, q, w = out[False]
        self.assertGreater(J[20, 2], 0.3)                    # person's left wrist raised (above the pelvis)
        Jm, qm, wm = out[True]
        self.assertGreater(Jm[21, 2], 0.3)                   # mirror: the robot's right wrist raised
        J2, q2, w2 = smpl_stream.mirror(J, q, w)
        np.testing.assert_allclose(Jm, J2, atol=2e-4)
        np.testing.assert_allclose(wm, w2, atol=1e-4)

    def test_no_person_no_smpl(self):
        from k1live.human import HumanMapper

        m = HumanMapper(loco=False)
        self.assertIsNone(m.process(None, 0.0, 1280, 720)["smpl"])
        self.assertIsNone(HumanMapper(loco=False, smpl=False)._smpl_conv)


if __name__ == "__main__":
    unittest.main()
