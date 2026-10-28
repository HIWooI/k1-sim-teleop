"""The display prop must not change the controller model or joint mapping."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from k1live.scenery import snu_scene_xml


def test_gate_is_static_noncolliding_background():
    xml=snu_scene_xml('<mujoco><asset/><visual/><worldbody/></mujoco>')
    root=ET.fromstring(xml)
    gate=root.find(".//geom[@name='snu_gate_visual']")
    assert gate is not None
    assert gate.get('contype') == gate.get('conaffinity') == '0'
    assert root.findall('.//joint') == root.findall('.//freejoint') == []
    assert float(gate.get('pos').split()[0]) <= -4
    asset=root.find(".//mesh[@name='snu_gate_mesh']")
    assert Path(asset.get('file')).is_file()


def test_render_model_preserves_robot_dynamics(monkeypatch):
    import mujoco
    from k1.sim import build_model,K1_XML,MESHDIR
    from k1live.protocol import POLICY_JOINTS
    if not K1_XML.is_file() or not MESHDIR.is_dir():
        pytest.skip('ROBOTIS mesh assets are required')
    monkeypatch.setenv('K1LIVE_RENDER_SCENE','snu')
    physics=build_model(POLICY_JOINTS)
    render=build_model(POLICY_JOINTS,render_scene='snu')
    assert physics.nq == render.nq and physics.nv == render.nv
    assert physics.nbody == render.nbody
    assert mujoco.mj_name2id(physics,mujoco.mjtObj.mjOBJ_GEOM,'snu_gate_visual') == -1
    for field in ('body_mass','body_inertia','dof_damping','dof_armature','jnt_qposadr','jnt_dofadr'):
        np.testing.assert_array_equal(getattr(physics,field),getattr(render,field))
    a,b=mujoco.MjData(physics),mujoco.MjData(render)
    a.qpos[2]=b.qpos[2]=0.8
    for _ in range(10):
        mujoco.mj_step(physics,a);mujoco.mj_step(render,b)
    np.testing.assert_allclose(a.qpos,b.qpos,atol=1e-12)
    np.testing.assert_allclose(a.qvel,b.qvel,atol=1e-12)
