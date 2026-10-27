"""Minimal BVH reader + forward kinematics (for GEM-X `*_soma.bvh` exports)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BVH:
    names: list
    parents: list
    offsets: np.ndarray  # (J,3)
    channels: list  # per joint list of channel names
    frame_time: float
    motion: np.ndarray  # (T, C)

    @property
    def fps(self):
        return 1.0 / self.frame_time


def read_bvh(path) -> BVH:
    toks = open(path).read().split()
    names, parents, offsets, channels = [], [], [], []
    stack = []
    i = 0
    last = -1
    while toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            names.append(toks[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            last = len(names) - 1
            i += 2
        elif t == "End":  # End Site: skip its block
            depth = 0
            i += 2
            while True:
                if toks[i] == "{":
                    depth += 1
                elif toks[i] == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
        elif t == "{":
            stack.append(last)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            offsets[last] = [float(x) for x in toks[i + 1:i + 4]]
            i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1])
            channels[last] = toks[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    i += 1
    assert toks[i] == "Frames:"
    nf = int(toks[i + 1])
    assert toks[i + 2] == "Frame" and toks[i + 3] == "Time:"
    ft = float(toks[i + 4])
    data = np.array(toks[i + 5:], dtype=np.float64)
    nc = sum(len(c) for c in channels)
    data = data[: nf * nc].reshape(nf, nc)
    return BVH(names, parents, np.array(offsets), channels, ft, data)


def fk(b: BVH):
    """Global joint positions (T,J,3) and rotations (T,J,3,3) in the BVH world frame."""
    T, J = b.motion.shape[0], len(b.names)
    pos = np.zeros((T, J, 3))
    R = np.zeros((T, J, 3, 3))
    col = 0
    for j in range(J):
        ch = b.channels[j]
        local_t = np.broadcast_to(b.offsets[j], (T, 3)).copy()
        rot_axes, rot_cols = "", []
        for k, c in enumerate(ch):
            if c.endswith("position"):
                local_t[:, "XYZ".index(c[0])] = b.motion[:, col + k]
            else:
                rot_axes += c[0]
                rot_cols.append(col + k)
        col += len(ch)
        if rot_axes:
            local_R = Rotation.from_euler(rot_axes, b.motion[:, rot_cols], degrees=True).as_matrix()
        else:
            local_R = np.broadcast_to(np.eye(3), (T, 3, 3))
        p = b.parents[j]
        if p < 0:
            R[:, j] = local_R
            pos[:, j] = local_t
        else:
            R[:, j] = R[:, p] @ local_R
            pos[:, j] = pos[:, p] + np.einsum("tij,tj->ti", R[:, p], local_t)
    return pos, R
