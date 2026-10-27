"""The distilled K1 retarget MLP of the original GEM-X pipeline, per live frame: GEM-X SOMA body_pose (228) -> the 10
K1 arm joints.

Option of k1live/sender_gemx.py (--arm-src mlp) and the offline reel / evaluation. Same network, normalisation and
2 Hz one-euro output low-pass (beta 0, measured dt) as the original perception/demo_webcam.py; the rebuild code is
k1/soma_mlp.py. The network imitates soma-retargeter (no mirror, person's left arm = robot's left arm): the caller
mirrors like any other arm source. Wrist roll: the network's own (it sees the forearm twist), unlike the IK paths.

Weights (not distributed with this repository): $K1LIVE_SHAPE_MODELS/k1_retarget.pt, else models/shape_models/.
CPU, torch.set_num_threads(2) (the default thread pool turns a 0.1 ms forward into 60 ms next to busy jobs).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from k1live.protocol import ARM_JOINTS

ROOT = Path(__file__).resolve().parents[1]
SOMA_IDX = [i - 1 for i in (1, 4, 13, 14, 15, 41, 42, 43, 68, 69, 70, 73, 74, 75)]   # the 14 IK targets


def default_ckpt():
    for d in (os.environ.get("K1LIVE_SHAPE_MODELS"), ROOT / "models" / "shape_models"):
        if d and (Path(d) / "k1_retarget.pt").exists():
            return Path(d) / "k1_retarget.pt"
    return None


class MlpArms:
    """__call__(body_pose (228,), kp2d (77, 3) or None, t) -> 10 arm joints (ARM_JOINTS order, person terms)."""

    def __init__(self, ckpt=None, smooth_hz=2.0):
        import torch

        from k1.soma_mlp import OneEuro, _remap_block_keys, build_student

        ckpt = Path(ckpt) if ckpt else default_ckpt()
        if ckpt is None or not Path(ckpt).exists():
            raise FileNotFoundError("k1_retarget.pt not found (set K1LIVE_SHAPE_MODELS)")
        torch.set_num_threads(2)
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        self.names = [str(n) for n in ck["names"]]
        self.net = build_student(ck, len(self.names))
        if "state" in ck:
            self.net.load_state_dict(_remap_block_keys(ck["state"]))
        self.net.eval()
        self.parts = str(ck.get("arch", "plain")) == "parts"
        self.mu = None if self.parts else torch.tensor(np.asarray(ck["mu"]), dtype=torch.float32)
        self.sd = None if self.parts else torch.tensor(np.asarray(ck["sd"]), dtype=torch.float32)
        self.feats = str(ck.get("features", "pose"))
        self.order = [self.names.index(n) for n in ARM_JOINTS]
        self.smooth_hz = smooth_hz
        self._OneEuro = OneEuro
        self.filt = OneEuro(len(self.names), min_cutoff=smooth_hz, beta=0.0) if smooth_hz else None
        self.t_prev = None
        self.torch = torch
        self.ckpt = str(ckpt)

    def reset(self):
        self.filt = self._OneEuro(len(self.names), min_cutoff=self.smooth_hz, beta=0.0) if self.smooth_hz else None
        self.t_prev = None

    def __call__(self, body_pose, kp2d=None, t=None):
        torch = self.torch
        x = torch.tensor(np.asarray(body_pose, np.float32).reshape(-1))
        if self.feats == "pose_conf":
            conf = np.ones(14, np.float32) if kp2d is None else np.asarray(kp2d, np.float32)[SOMA_IDX, 2]
            x = torch.cat([x, torch.tensor(conf)])
        with torch.no_grad():
            q = (self.net(x) if self.parts else self.net((x - self.mu) / self.sd)).numpy()[: len(self.names)]
        q = q.astype(np.float64)
        if self.filt is not None:
            # measured dt (as the original), capped: a stalled frame must not widen the cutoff just as the pose jumps
            dt = 1 / 30 if (t is None or self.t_prev is None) else min(max(t - self.t_prev, 1e-3), 0.15)
            self.t_prev = t
            q = self.filt(q, dt)
        return q[self.order]
