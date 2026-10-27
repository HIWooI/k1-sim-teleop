"""The distilled K1 retarget MLP of the original GEM-X pipeline, offline: GEM-X SOMA body_pose (76 joint rotations,
228) -> 23 K1 joint angles.

The network is a student of soma-retargeter (original perception/make_labels.py + distill.py): it reads the SOMA
articulation only (no root, no points), so it carries forearm twist the 14-point IK cannot see. Live it ran on the
last frame of every GEM-X window (`body_params_incam.body_pose`) with a 2 Hz one-euro low-pass (beta 0) on its
output. Here it reads the per-frame body_pose of the offline GEM-X run (hpe_results.pt), same filter, video time.

Network rebuild (`_ResBlock`, `_PartExperts`, `build_student`, `_remap_block_keys`) and `OneEuro` are copied from
the original perception/demo_webcam.py and perception/motion_command.py unchanged apart from formatting.
Weights: $K1LIVE_SHAPE_MODELS, default models/shape_models/ (k1_retarget.pt etc., not distributed with this repository).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

MODELS = Path(os.environ.get("K1LIVE_SHAPE_MODELS", str(Path(__file__).resolve().parents[1] / "models" / "shape_models")))
# SOMA_IDX: the 14 cpu_kinematics targets in the 77-joint SOMA list (VitPose-77 confidences for pose_conf)
SOMA_IDX = [i - 1 for i in (1, 4, 13, 14, 15, 41, 42, 43, 68, 69, 70, 73, 74, 75)]


class OneEuro:
    """The original motion_command.OneEuro."""

    def __init__(self, n, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = np.zeros(n)
        self.dx_prev = np.zeros(n)
        self.started = False

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        if not self.started:
            self.x_prev, self.started = x.copy(), True
            return x.copy()
        dx = (x - self.x_prev) / max(dt, 1e-6)
        dx = self._alpha(self.d_cutoff, dt) * dx + (1 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx
        return x_hat


class _ResBlock(torch.nn.Module):
    """Pre-norm residual block, matching distill.py --arch res."""

    def __init__(self, w, drop=0.1):
        super().__init__()
        self.n = torch.nn.LayerNorm(w)
        self.f = torch.nn.Sequential(torch.nn.Linear(w, w), torch.nn.GELU(),
                                     torch.nn.Dropout(drop), torch.nn.Linear(w, w))

    def forward(self, x):
        return x + self.f(self.n(x))


def _remap_block_keys(state):
    """distill.py names the block's LayerNorm `norm`, demo_webcam's `_ResBlock` names it `n`."""
    return {k.replace(".norm.", ".n."): v for k, v in state.items()}


class _PartExperts(torch.nn.Module):
    """One expert per body part, each reading only its own chain of the input (normalises internally)."""

    def __init__(self, ck, ndof):
        super().__init__()
        self.ndof = ndof
        self.nets, self.idx = torch.nn.ModuleDict(), {}
        for name, p in ck["parts"].items():
            if name == "_fuse":
                continue
            w, nb = ck["width"], ck.get("blocks", 4)
            net = torch.nn.Sequential(
                torch.nn.Linear(len(p["in"]), w), *[_ResBlock(w) for _ in range(nb)],
                torch.nn.LayerNorm(w), torch.nn.Linear(w, len(p["out"])))
            net.load_state_dict(_remap_block_keys(p["state"]))
            net.eval()
            self.nets[name] = net
            self.idx[name] = (
                torch.tensor(np.asarray(p["in"]), dtype=torch.long),
                torch.tensor(np.asarray(p["out"]), dtype=torch.long),
                torch.tensor(np.asarray(p["mu"]), dtype=torch.float32),
                torch.tensor(np.asarray(p["sd"]), dtype=torch.float32),
            )
        f = ck["parts"].get("_fuse")
        if f is not None:
            self.fuse = torch.nn.Sequential(
                torch.nn.Linear(ndof, 256), torch.nn.GELU(),
                torch.nn.Linear(256, 256), torch.nn.GELU(),
                torch.nn.Linear(256, ndof))
            self.fuse.load_state_dict(f["state"])
            self.fuse.eval()
            self.fmu = torch.tensor(np.asarray(f["mu"]), dtype=torch.float32)
            self.fsd = torch.tensor(np.asarray(f["sd"]), dtype=torch.float32)
        else:
            self.fuse = None

    def forward(self, x):
        out = x.new_zeros(self.ndof)
        for name, net in self.nets.items():
            ci, co, mu, sd = self.idx[name]
            out[co] = net((x[ci] - mu) / sd)
        if self.fuse is not None:
            out = out + self.fuse((out - self.fmu) / self.fsd)
        return out


def build_student(ck, ndof, extra_out=0):
    """Rebuild a distill.py checkpoint's network from the shape it recorded (plain, residual or per-part)."""
    if str(ck.get("arch", "plain")) == "parts":
        return _PartExperts(ck, ndof)
    w, nb = ck["width"], ck.get("blocks", 4)
    n_out = ndof + extra_out
    if str(ck.get("arch", "plain")) == "plain":
        return torch.nn.Sequential(
            torch.nn.Linear(len(ck["mu"]), w), torch.nn.GELU(),
            torch.nn.Linear(w, w), torch.nn.GELU(), torch.nn.Linear(w, n_out))
    return torch.nn.Sequential(
        torch.nn.Linear(len(ck["mu"]), w), *[_ResBlock(w) for _ in range(nb)],
        torch.nn.LayerNorm(w), torch.nn.Linear(w, n_out))


def load_body_pose(hpe_path, which="body_params_incam"):
    p = torch.load(str(hpe_path), map_location="cpu", weights_only=False)
    return p[which]["body_pose"].reshape(-1, 228).float().numpy()


def mlp_joint_angles(body_pose, policy_names, fps, ckpt=MODELS / "k1_retarget.pt", conf=None, smooth_hz=2.0):
    """(T,228) SOMA body_pose -> (T,23) K1 joint angles (rad) in policy order, 1-euro low-passed at smooth_hz
    (0 = off) on the video time base. conf: (T,14) VitPose confidences at SOMA_IDX for pose_conf checkpoints
    (all ones if missing). Returns (Q, checkpoint info)."""
    torch.set_num_threads(2)
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    names = [str(n) for n in ck["names"]]
    net = build_student(ck, len(names))
    if "state" in ck:
        net.load_state_dict(_remap_block_keys(ck["state"]))
    net.eval()
    parts = str(ck.get("arch", "plain")) == "parts"
    mu = None if parts else torch.tensor(np.asarray(ck["mu"]), dtype=torch.float32)
    sd = None if parts else torch.tensor(np.asarray(ck["sd"]), dtype=torch.float32)
    feats = str(ck.get("features", "pose"))
    if feats == "pose_conf" and conf is None:
        conf = np.ones((len(body_pose), 14), np.float32)
    filt = OneEuro(len(names), min_cutoff=smooth_hz, beta=0.0) if smooth_hz else None
    out = []
    with torch.no_grad():
        for t in range(len(body_pose)):
            x = torch.tensor(body_pose[t], dtype=torch.float32)
            if feats == "pose_conf":
                x = torch.cat([x, torch.tensor(conf[t], dtype=torch.float32)])
            q = (net(x) if parts else net((x - mu) / sd)).numpy()[: len(names)].astype(np.float64)
            if filt is not None:
                q = filt(q, 1.0 / fps)
            out.append(q)
    Q = np.array(out)
    order = [names.index(n) for n in policy_names]
    info = {"mlp_ckpt": str(ckpt), "mlp_arch": str(ck.get("arch", "plain")), "mlp_features": feats,
            "mlp_smooth_hz": smooth_hz}
    return Q[:, order], info
