"""K1 robot-description helpers shared by the staging script, the verification and sonic/robots.py.

Pure Python + numpy (no Isaac Lab / torch): runs on the Mac and on the GPU training server.

* parse_urdf()            joints/links/origins/axes from a URDF
* isaaclab_order()        Isaac Lab (PhysX) articulation order = breadth-first over the kinematic tree,
                          children in file order, fixed joints merged into their parent (the URDF
                          importer default). Verified against the order cyclo_lab recorded for K1
                          (LEGACY_PHYSX_* of the K1 sim2sim tooling) and against SONIC's own G1 list.
* mujoco_order()          MuJoCo order = depth-first body order of the MJCF (= Unitree/ROBOTIS hardware order)
* urdf_fk()               numpy forward kinematics of the URDF (for the MJCF <-> URDF consistency check)
* read_stl()/write_stl()  binary STL I/O (mesh conversion mm -> m, no trimesh/open3d needed)
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

import numpy as np


def _vec(s, default="0 0 0"):
    return np.array([float(x) for x in (s or default).split()])


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_R(axis, a):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links = [l.attrib["name"] for l in root.findall("link")]
    joints = OrderedDict()
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        lim = j.find("limit")
        joints[j.attrib["name"]] = {
            "type": j.attrib["type"],
            "parent": j.find("parent").attrib["link"],
            "child": j.find("child").attrib["link"],
            "xyz": _vec(o.attrib.get("xyz") if o is not None else None),
            "rpy": _vec(o.attrib.get("rpy") if o is not None else None),
            "axis": _vec(a.attrib.get("xyz") if a is not None else None, "1 0 0"),
            "limit": None if lim is None else (float(lim.attrib.get("lower", 0)), float(lim.attrib.get("upper", 0)),
                                               float(lim.attrib.get("effort", 0)), float(lim.attrib.get("velocity", 0))),
        }
    children = {l: [] for l in links}
    has_parent = set()
    for n, j in joints.items():
        children[j["parent"]].append(n)
        has_parent.add(j["child"])
    roots = [l for l in links if l not in has_parent]
    assert len(roots) == 1, roots
    return {"links": links, "joints": joints, "children": children, "root": roots[0]}


def isaaclab_order(urdf):
    """(bodies, dof_joints) in Isaac Lab articulation order (BFS, fixed joints merged)."""
    J, ch = urdf["joints"], urdf["children"]

    def movable_children(link):  # children through fixed joints are merged into `link`
        out = []
        for jn in ch[link]:
            if J[jn]["type"] == "fixed":
                out += movable_children(J[jn]["child"])
            else:
                out.append(jn)
        return out

    bodies, dofs = [urdf["root"]], []
    frontier = [urdf["root"]]
    while frontier:
        nxt = []
        for link in frontier:
            for jn in movable_children(link):
                dofs.append(jn)
                bodies.append(J[jn]["child"])
                nxt.append(J[jn]["child"])
        frontier = nxt
    return bodies, dofs


def mujoco_order(model):
    """(bodies without the world body, hinge joints) of a compiled mujoco.MjModel."""
    bodies = [model.body(b).name for b in range(1, model.nbody)]
    dofs = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] in (2, 3)]  # slide/hinge
    return bodies, dofs


def index_maps(isaac_names, mujoco_names):
    """SONIC convention: out[i] = in[mapping[i]].
    isaaclab_to_mujoco[i] = Isaac index of MuJoCo element i; mujoco_to_isaaclab[i] = MuJoCo index of Isaac i."""
    i2m = [isaac_names.index(n) for n in mujoco_names]
    m2i = [mujoco_names.index(n) for n in isaac_names]
    return i2m, m2i


def urdf_fk(urdf, q: dict, root_pos=(0, 0, 0), root_R=None):
    """World transform of every link for joint angles q (name -> rad); missing joints = 0."""
    J, ch = urdf["joints"], urdf["children"]
    T = {urdf["root"]: (np.eye(3) if root_R is None else np.asarray(root_R), np.asarray(root_pos, float))}
    stack = [urdf["root"]]
    while stack:
        link = stack.pop()
        R, p = T[link]
        for jn in ch[link]:
            j = J[jn]
            Rj = R @ rpy_to_R(j["rpy"])
            pj = p + R @ j["xyz"]
            if j["type"] in ("revolute", "continuous"):
                Rj = Rj @ axis_angle_R(j["axis"], q.get(jn, 0.0))
            elif j["type"] == "prismatic":
                pj = pj + Rj @ (j["axis"] * q.get(jn, 0.0))
            T[j["child"]] = (Rj, pj)
            stack.append(j["child"])
    return T


# ------------------------------------------------------------------ binary STL
def read_stl(path):
    b = Path(path).read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:400]:  # ASCII STL
        v = [list(map(float, ln.split()[1:4])) for ln in b.decode(errors="ignore").splitlines()
             if ln.strip().startswith("vertex")]
        tri = np.array(v, np.float32).reshape(-1, 3, 3)
        return tri, np.zeros((len(tri), 3), np.float32)
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    return rec["v"].copy(), rec["n"].copy()


def write_stl(path, tri, normals=None, header=b"converted by sonic_k1"):
    tri = np.asarray(tri, np.float32)
    if normals is None:
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        nn = np.cross(e1, e2)
        normals = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = normals, tri
    Path(path).write_bytes(header.ljust(80, b" ")[:80] + struct.pack("<I", len(tri)) + rec.tobytes())
