"""Mac deploy loop (sonic/) parameterised by robot: G1 unchanged, K1 pluggable.

    .venv/bin/python -m unittest discover -s sonic_k1/tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "scripts"))

from sonic import params as P  # noqa: E402
from sonic.robots import get_robot, obs_dims  # noqa: E402


def _staged():
    return any((HERE / "overlay/gear_sonic/data/assets/robot_description/urdf/k1/meshes").glob("*.stl"))


class TestSpecs(unittest.TestCase):
    def test_g1_spec_is_params(self):
        g1 = get_robot("g1")
        self.assertIs(g1.kps, P.KPS)
        self.assertIs(g1.default_angles, P.DEFAULT_ANGLES)
        self.assertEqual(list(g1.mujoco_joint_names), P.MUJOCO_JOINT_NAMES)
        self.assertEqual(g1.isaaclab_joint_names, P.ISAACLAB_JOINT_NAMES)

    def test_g1_dims_match_released_models(self):
        import onnxruntime as ort

        for v in ("release", "low_latency", "sonic_v1_1"):
            d = P.MODELS / v
            if not (d / "model_encoder.onnx").exists():
                continue
            dims = obs_dims(get_robot("g1"), d)
            enc = ort.InferenceSession(str(d / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
            dec = ort.InferenceSession(str(d / "model_decoder.onnx"), providers=["CPUExecutionProvider"])
            self.assertEqual(dims["encoder"], enc.get_inputs()[0].shape[1], v)
            self.assertEqual(dims["decoder"], dec.get_inputs()[0].shape[1], v)
            self.assertEqual(dims["action"], dec.get_outputs()[0].shape[1], v)

    @unittest.skipUnless(_staged(), "K1 meshes not staged (scripts/stage_assets.py)")
    def test_k1_spec(self):
        k1 = get_robot("k1")
        self.assertEqual(k1.num_dof, 23)
        self.assertEqual([k1.isaaclab_joint_names[i] for i in k1.wrist_il], ["left_wrist_roll_joint", "right_wrist_roll_joint"])
        self.assertEqual(list(k1.wrist_il), [21, 22])  # == sonic_k1.yaml joints_idx for the SMPL encoder
        legs = [k1.isaaclab_joint_names[i] for i in k1.lower_body_il]
        self.assertEqual(legs, list(k1.mujoco_joint_names[:12]))  # commands.py lower_joint_indices_mujoco = range(12)
        np.testing.assert_allclose(k1.action_scale[:4], 0.25 * 96.864 / k1.kps[:4], rtol=1e-6)
        d = obs_dims(k1, P.MODELS / "release")
        self.assertEqual((d["decoder"], d["encoder"], d["action"]), (814, 1602, 23))


@unittest.skipUnless(_staged(), "K1 meshes not staged (scripts/stage_assets.py)")
class TestK1Loop(unittest.TestCase):
    def test_zero_policy_runs_mode0_and_mode2(self):
        """Stand-in zero-action checkpoint with K1 input sizes: INIT ramp, CONTROL, SMPL streaming, no NaN."""
        from dummy_onnx import matmul_model

        from sonic.deploy import SonicController
        from sonic.motion import standing_motion
        from sonic.mp2smpl import MediaPipeToSmpl
        from sonic.runner import control_tick
        from sonic.sim import G1Sim
        from sonic.stream import StreamBuffer, attach, stream_step

        k1 = get_robot("k1")
        d = obs_dims(k1, P.MODELS / "release")
        with tempfile.TemporaryDirectory() as td:
            v = Path(td) / "release"
            v.mkdir()
            (v / "observation_config.yaml").write_text((P.MODELS / "release/observation_config.yaml").read_text())
            (v / "model_encoder.onnx").write_bytes(matmul_model(d["encoder"], d["token"]))
            (v / "model_decoder.onnx").write_bytes(matmul_model(d["decoder"], d["action"]))
            ctrl = SonicController("release", models_dir=td, robot=k1)
        sim = G1Sim(robot=k1, elastic_band=True)
        ctrl.set_motion(standing_motion(robot=k1), play=False)
        idle = MediaPipeToSmpl().idle_frame()
        buf = StreamBuffer(window=10, robot=k1)
        streamed = False
        for k in range(300):  # 6 s, band kept on (a zero policy cannot balance — G1 cannot either)
            if ctrl.state == "WAIT_FOR_CONTROL":
                ctrl.start_control()
            if ctrl.state == "CONTROL" and k > 200 and not streamed:
                for i in range(20):
                    buf.add_sample(sim.time - 0.4 + 0.02 * i, idle["body_quat"], idle["smpl_joints"])
                self.assertTrue(attach(ctrl, buf, sim.time))
                streamed = True
            if streamed:
                stream_step(ctrl, buf, sim.time)
            control_tick(ctrl, sim)
        self.assertEqual(ctrl.state, "CONTROL")
        self.assertEqual(ctrl.current_motion.encode_mode, 2)
        self.assertEqual(ctrl.last_encoder_obs.shape, (1602,))
        self.assertTrue(np.isfinite(sim.data.qpos).all())
        # zero action -> PD target is the K1 default pose
        np.testing.assert_allclose(ctrl.last_cmd[0], k1.default_angles, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
