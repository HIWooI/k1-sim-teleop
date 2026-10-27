"""soma-retargeter K1 CSV -> SONIC motion-library PKL (sonic_k1/data/robot/<name>.pkl).

K1 counterpart of gear_sonic/data_process/convert_soma_csv_to_motion_lib.py, which is G1-only
(29 DOF / 30 bodies / G1 axes hard-coded). Same input conventions as that script and as the soma-retargeter K1 pipeline
(`inputs/padding/<name>_retarget_k1_from_bvh.csv`, `<name>_soma_padded.csv`, 30 fps):
  Frame, root_translate{X,Y,Z} [cm], root_rotate{X,Y,Z} [deg, Euler 'xyz' intrinsic, Z-up], <joint>_dof [deg]
Output entry (new_embodiments.md "Step 5"), keyed by motion name:
  root_trans_offset (T,3) m | pose_aa (T,24,3) MuJoCo body order: [0] root rotvec, [1:] axis*angle |
  dof (T,23) MuJoCo order | root_rot (T,4) xyzw | smpl_joints (T,24,3) zeros placeholder | fps

    .venv/bin/python sonic_k1/scripts/k1_csv_to_pkl.py <motion>_retarget_k1_from_bvh.csv
    .venv/bin/python sonic_k1/scripts/k1_csv_to_pkl.py <dir of csv> --out sonic_k1/data/robot   # bulk (untested at scale)

Not done here (see docs/training.md): mirrored `_M` variants (upstream ships no mirror generator either),
physical-feasibility filtering, real SMPL joints for the SMPL encoder (needs the source BVH / GEM-X output).
"""
from __future__ import annotations

import argparse
import ast
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import transform

HERE = Path(__file__).resolve().parents[1]
K1_PY = HERE / "overlay/gear_sonic/envs/manager_env/robots/k1.py"
K1_MJCF = HERE / "overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml"


def k1_joints_and_axes():
    names = None
    for node in ast.parse(K1_PY.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "K1_MUJOCO_JOINTS":
            names = ast.literal_eval(node.value)
    root = ET.parse(K1_MJCF).getroot()
    joints = [j for j in root.find("worldbody").iter("joint") if j.get("type") != "free"]
    assert [j.get("name") for j in joints] == names, "MJCF joint order != K1_MUJOCO_JOINTS"
    # body i+1 carries joint i (one hinge per body, same depth-first order)
    bodies = [b for b in root.find("worldbody").iter("body")][1:]
    assert all(b.find("joint") is not None and b.find("joint").get("name") == n for b, n in zip(bodies, names))
    axes = np.array([[float(x) for x in j.get("axis").split()] for j in joints], np.float32)
    return names, axes


def load_k1_csv(path, names):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    cols = data.dtype.names
    root_pos = np.stack([data["root_translateX"], data["root_translateY"], data["root_translateZ"]], 1) / 100.0
    euler = np.stack([data["root_rotateX"], data["root_rotateY"], data["root_rotateZ"]], 1)
    missing = [n for n in names if f"{n}_dof" not in cols]
    if missing:
        raise ValueError(f"{path}: missing K1 joint columns {missing}")
    extra = [c for c in cols if c.endswith("_dof") and c[:-4] not in names]
    if extra:
        raise ValueError(f"{path}: unknown joint columns {extra} (not a K1 CSV?)")
    dof = np.deg2rad(np.stack([data[f"{n}_dof"] for n in names], 1))  # reordered by name
    quat_xyzw = transform.Rotation.from_euler("xyz", euler, degrees=True).as_quat()
    return root_pos.astype(np.float32), quat_xyzw.astype(np.float32), dof.astype(np.float32)


def to_entry(root_pos, quat_xyzw, dof, axes, fps):
    T, n = dof.shape
    pose_aa = np.zeros((T, n + 1, 3), np.float32)
    pose_aa[:, 0] = transform.Rotation.from_quat(quat_xyzw).as_rotvec()
    pose_aa[:, 1:] = axes[None] * dof[:, :, None]
    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": quat_xyzw,
        "smpl_joints": np.zeros((T, 24, 3), np.float32),  # placeholder (smpl_motion_file: dummy)
        "fps": int(fps),
    }


def main():
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="K1 CSV files or directories of them")
    ap.add_argument("--out", type=Path, default=HERE / "data/robot")
    ap.add_argument("--fps", type=int, default=30, help="CSV frame rate (k1-motion-pipeline: 30)")
    ap.add_argument("--fps-target", type=int, default=30, help="stride-downsample to this rate (SONIC PKLs: 30)")
    args = ap.parse_args()
    names, axes = k1_joints_and_axes()
    files = []
    for p in map(Path, args.inputs):
        files += sorted(p.rglob("*.csv")) if p.is_dir() else [p]
    args.out.mkdir(parents=True, exist_ok=True)
    jump = max(1, int(args.fps / args.fps_target))
    for f in files:
        name = f.stem.replace("_retarget_k1_from_bvh", "").replace("_soma_padded", "")
        root_pos, q, dof = load_k1_csv(f, names)
        e = to_entry(root_pos[::jump], q[::jump], dof[::jump], axes, args.fps / jump)
        joblib.dump({name: e}, args.out / f"{name}.pkl")
        print(f"{f.name} -> {args.out / (name + '.pkl')}  T={len(e['dof'])} @ {e['fps']} fps, "
              f"root z {e['root_trans_offset'][:, 2].min():.3f}..{e['root_trans_offset'][:, 2].max():.3f} m")


if __name__ == "__main__":
    sys.exit(main())
