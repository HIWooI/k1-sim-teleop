"""Stage the ROBOTIS K1 description into sonic_k1/overlay/ in the layout SONIC training expects.

    .venv/bin/python sonic_k1/scripts/stage_assets.py \
        --description $AI_SAPIENS_ROOT/ai_sapiens_description \
        --urdf k1/assets/k1.urdf

Writes (paths relative to the GR00T-WholeBodyControl checkout the overlay is copied onto):
  gear_sonic/data/assets/robot_description/urdf/k1/k1.urdf          Isaac Lab physics (mesh paths -> meshes/*.stl)
  gear_sonic/data/assets/robot_description/urdf/k1/meshes/*.stl     STL converted mm -> m   (git-ignored, ROBOTIS asset)
  gear_sonic/data/assets/robot_description/mjcf/k1.xml              motion-library MJCF (FK only), see below

Why a separate motion-library MJCF instead of ai_sapiens_description/mujoco/k1/k1.xml as is:
  * Humanoid_Batch (gear_sonic/utils/motion_lib/torch_humanoid_batch.py) makes every <body> a skeleton
    node. Isaac Lab's URDF importer merges the fixed head_joint, so Isaac has 24 bodies; the head_link
    body is merged into torso_link here too, otherwise pose_aa / body maps would be 25 vs 24.
  * Humanoid_Batch only recognises a floating base written as <joint type="free">; with <freejoint>
    it falls into the "6 root dofs" branch and silently drops the first six hinge axes (left leg).
  * load_mesh() reads STL with open3d and ignores the MJCF scale="0.001" -> the motion library's
    fix_trans_height() would see a 1000x robot. Meshes are converted to metres instead.
  * actuator names = joint names (G1/H2 convention; Humanoid_Batch counts <motor> entries for num_dof).
Sources: ROBOTIS ai_sapiens_description (MJCF + meshes, k1_rev1) and the xacro-generated k1.urdf
(k1/assets/k1.urdf). Joint origins/axes/limits of the two agree (tests/test_k1_robot.py); the
only difference is the fixed head pose (URDF -0.0185,0,0.385 vs MJCF -0.02725,0,0.361) — the URDF value
is used so the merged head matches Isaac physics.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_assets import parse_urdf, read_stl, write_stl  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OVERLAY = HERE / "overlay"
DESC_REL = "gear_sonic/data/assets/robot_description"
# ROBOTIS ai_sapiens checkout (https://github.com/ROBOTIS-GIT/ai_sapiens): $AI_SAPIENS_ROOT, default third_party/ai_sapiens
DEFAULT_DESC = Path(os.environ.get("AI_SAPIENS_ROOT", HERE.parent / "third_party" / "ai_sapiens")) / "ai_sapiens_description"
DEFAULT_URDF = HERE.parent / "k1" / "assets" / "k1.urdf"


def _floats(s):
    return np.array([float(x) for x in s.split()])


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def _full_inertia(I6):
    xx, yy, zz, xy, xz, yz = I6
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def merge_inertial(parent_inertial, child_inertial, child_pos):
    """Combine two <inertial fullinertia=...> (child frame = parent frame translated by child_pos)."""
    m1, c1, I1 = float(parent_inertial.get("mass")), _floats(parent_inertial.get("pos")), _full_inertia(_floats(parent_inertial.get("fullinertia")))
    m2, c2, I2 = float(child_inertial.get("mass")), _floats(child_inertial.get("pos")) + child_pos, _full_inertia(_floats(child_inertial.get("fullinertia")))
    m = m1 + m2
    c = (m1 * c1 + m2 * c2) / m

    def shift(I, mass, d):
        return I + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    I = shift(I1, m1, c1 - c) + shift(I2, m2, c2 - c)
    parent_inertial.set("mass", f"{m:.9g}")
    parent_inertial.set("pos", _fmt(c))
    parent_inertial.set("fullinertia", _fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))


def build_motionlib_mjcf(src_xml: Path, urdf: dict, meshdir: str) -> str:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    root.set("model", "k1")
    comp = root.find("compiler")
    comp.set("meshdir", meshdir)
    for m in root.find("asset").findall("mesh"):
        m.attrib.pop("scale", None)  # meshes are converted to metres
    world = root.find("worldbody")
    pelvis = world.find("body")
    fj = pelvis.find("freejoint")
    if fj is not None:
        idx = list(pelvis).index(fj)
        pelvis.remove(fj)
        j = ET.Element("joint", {"name": fj.get("name", "floating_base_joint"), "type": "free",
                                 "limited": "false", "actuatorfrclimited": "false"})
        pelvis.insert(idx, j)
    # merge fixed head_link into torso_link (Isaac Lab merges the fixed head_joint the same way)
    torso = next(b for b in pelvis.iter("body") if b.get("name") == "torso_link")
    head = next(b for b in torso.findall("body") if b.get("name") == "head_link")
    hj = urdf["joints"]["head_joint"]
    assert hj["type"] == "fixed" and np.allclose(hj["rpy"], 0)
    head_pos = hj["xyz"]
    merge_inertial(torso.find("inertial"), head.find("inertial"), head_pos)
    insert_at = list(torso).index(head)
    for g in head.findall("geom"):
        g = copy.deepcopy(g)
        p = _floats(g.get("pos", "0 0 0")) + head_pos
        g.set("pos", _fmt(p))
        torso.insert(insert_at, g)
        insert_at += 1
    torso.remove(head)
    # actuator names = joint names (G1/H2 convention)
    for mot in root.find("actuator"):
        mot.set("name", mot.get("joint"))
    for tag in ("keyframe",):  # qpos layout is unchanged but keep the file minimal
        for e in root.findall(tag):
            root.remove(e)
    ET.indent(tree, space="  ")
    header = ("<!-- ROBOTIS K1 (AI Sapiens, k1_rev1) for the SONIC motion library (FK only).\n"
              "     Generated by sonic_k1/scripts/stage_assets.py from ai_sapiens_description/mujoco/k1/k1.xml:\n"
              "     fixed head_link merged into torso_link (URDF head_joint pose), <freejoint> -> <joint type=free>,\n"
              "     meshes in metres (no scale attr), actuator name = joint name. Do not edit by hand.\n"
              "     Derived from ROBOTIS AI Sapiens (https://github.com/ROBOTIS-GIT/ai_sapiens), Apache License 2.0. -->\n")
    return header + ET.tostring(root, encoding="unicode") + "\n"


def build_urdf(src_urdf: Path) -> str:
    txt = src_urdf.read_text()
    txt = re.sub(r'filename="package://ai_sapiens_description/meshes/k1_rev1/([^"]+)"', r'filename="meshes/\1"', txt)
    txt = txt.replace(' scale="0.001 0.001 0.001"', "")
    # drop the ROS-only <mujoco> compiler hint (package:// meshdir)
    txt = re.sub(r"\s*<mujoco>.*?</mujoco>", "", txt, flags=re.S)
    note = ("<!-- ROBOTIS K1 (k1_rev1) for SONIC / Isaac Lab. Copied from the xacro-generated k1.urdf by\n"
            "     sonic_k1/scripts/stage_assets.py: mesh paths -> meshes/*.stl (converted to metres, scale removed).\n"
            "     Derived from ROBOTIS AI Sapiens (https://github.com/ROBOTIS-GIT/ai_sapiens), Apache License 2.0. -->\n")
    return txt.replace("<robot ", note + "<robot ", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--description", type=Path, default=DEFAULT_DESC, help="ai_sapiens_description checkout")
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--overlay", type=Path, default=OVERLAY)
    ap.add_argument("--skip-meshes", action="store_true", help="only (re)write the text files")
    args = ap.parse_args()

    src_xml = args.description / "mujoco/k1/k1.xml"
    src_meshes = args.description / "meshes/k1_rev1"
    out_urdf_dir = args.overlay / DESC_REL / "urdf/k1"
    out_mjcf = args.overlay / DESC_REL / "mjcf/k1.xml"
    (out_urdf_dir / "meshes").mkdir(parents=True, exist_ok=True)
    out_mjcf.parent.mkdir(parents=True, exist_ok=True)

    urdf = parse_urdf(args.urdf)
    (out_urdf_dir / "k1.urdf").write_text(build_urdf(args.urdf))
    out_mjcf.write_text(build_motionlib_mjcf(src_xml, urdf, "../urdf/k1/meshes"))
    print("wrote", out_urdf_dir / "k1.urdf")
    print("wrote", out_mjcf)
    if not args.skip_meshes:
        n = 0
        for f in sorted(src_meshes.glob("*.stl")):
            tri, nrm = read_stl(f)
            write_stl(out_urdf_dir / "meshes" / f.name, tri * 1e-3, nrm)
            n += 1
        print(f"converted {n} meshes mm -> m into {out_urdf_dir / 'meshes'}")


if __name__ == "__main__":
    main()
