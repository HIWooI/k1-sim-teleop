"""Arm sources laid over planner walking (K1 arm joints, k1live.protocol.ARM_JOINTS order, 50 Hz).

  gestures  k1/ event gestures (k1/gestures.py palm/elbow IK) except the held-out `heart` and the hip-driven `bow`
  human     people in the event videos, K1 arm IK of MediaPipe / GEM-X SOMA (k1/run_video.py reference logs,
            bundled by `--bundle-human` into data/arms_human.npz: 51 sequences, 12 min, several IK variants per clip)
  random    smooth random arm motion inside a safe box around the default pose (same box as build_dataset synthetic)

ArmTrack(seq, fps).at(t) samples with time wrapping; overlay() blends a source over the planner's own arm swing
with smooth ramps, so a clip can switch between "walking arms" and "waving while walking" without jumps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from k1live.protocol import ARM_JOINTS, POLICY_JOINTS  # noqa: E402

ARM_IDX = [POLICY_JOINTS.index(n) for n in ARM_JOINTS]
GESTURE_NAMES = ("wave_hello", "welcome_open_arms", "clap", "point_guide", "raise_both_arms", "stand_at_ease")
HELDOUT_GESTURES = ("heart",)
DATA = Path(__file__).resolve().parent
# safe box (rad) per joint kind, left side; right roll mirrored. Same as build_dataset.synthetic
BOX = {"shoulder_pitch": (-1.6, 0.5), "shoulder_roll": (0.0, 0.9), "shoulder_yaw": (-0.6, 0.6),
       "elbow": (-0.2, 1.4), "wrist_roll": (-0.8, 0.8)}
ARM_DEFAULT = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
               "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 0.6}


def smooth_noise(rng, T, fps, n, period=(1.2, 3.0)):
    """(T, n) smooth random curves in [-1, 1] (min-jerk between random keyframes, starts at 0)."""
    out = np.zeros((T, n))
    ts = np.arange(T) / fps
    for k in range(n):
        t, keys = 0.0, [(0.0, 0.0)]
        while t < T / fps:
            t += rng.uniform(*period)
            keys.append((t, rng.uniform(-1, 1)))
        kt = np.array([a for a, _ in keys])
        kv = np.array([b for _, b in keys])
        i = np.clip(np.searchsorted(kt, ts) - 1, 0, len(kt) - 2)
        u = np.clip((ts - kt[i]) / (kt[i + 1] - kt[i]), 0, 1)
        out[:, k] = kv[i] + u ** 3 * (10 - 15 * u + 6 * u ** 2) * (kv[i + 1] - kv[i])
    return out


def random_arms(rng, T, fps=50.0, scale=0.7):
    z = smooth_noise(rng, T, fps, len(ARM_JOINTS))
    q = np.zeros((T, len(ARM_JOINTS)))
    for c, n in enumerate(ARM_JOINTS):
        key = next(k for k in BOX if k in n)
        lo, hi = BOX[key]
        if n.startswith("right") and key == "shoulder_roll":
            lo, hi = -hi, -lo
        centre = ARM_DEFAULT.get(n, 0.0)
        q[:, c] = np.clip(centre + scale * 0.5 * (hi - lo) * z[:, c], lo, hi)
    return q


class ArmTrack:
    def __init__(self, name, q, fps=50.0):
        self.name, self.q, self.fps = name, np.asarray(q, float), fps

    @property
    def seconds(self):
        return len(self.q) / self.fps

    def at(self, t):
        """(len(t), 10) arms at times t (s), wrapping around the end."""
        f = (np.asarray(t, float) * self.fps) % len(self.q)
        f0 = np.floor(f).astype(int)
        f1 = (f0 + 1) % len(self.q)
        w = (f - f0)[..., None]
        return (1 - w) * self.q[f0] + w * self.q[f1]


def load_gestures(cache=DATA / "arms_gestures.npz", names=GESTURE_NAMES, rebuild=False):
    """k1/ gestures -> {name: (T, 10)}; built once with k1/gestures.py (needs k1/sim.K1Split for its ArmIK)."""
    if cache.exists() and not rebuild:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from k1 import gestures as G
    from k1 import sim as S

    sim = S.K1Split(mode="split")
    assert sim.names == POLICY_JOINTS
    ik = G.ArmIK(sim)
    out = {}
    for n in list(names) + list(HELDOUT_GESTURES):
        out[n] = G.GESTURES[n](ik).q[:, ARM_IDX].astype(np.float32)
    np.savez_compressed(cache, **out)
    return {k: v for k, v in out.items()}


def load_human(path=DATA / "arms_human.npz"):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def overlay(base, src, w):
    """base, src (T, 10); w (T,) in [0, 1] -> blended arms."""
    w = np.clip(np.asarray(w, float), 0, 1)[:, None]
    return (1 - w) * base + w * src


def ramp_weights(T, fps, windows, ramp=0.5):
    """windows [(t0, t1)] -> (T,) weight 0..1 with min-jerk ramps of `ramp` s at both ends."""
    t = np.arange(T) / fps
    w = np.zeros(T)
    for t0, t1 in windows:
        u_in = np.clip((t - t0) / ramp, 0, 1)
        u_out = np.clip((t1 - t) / ramp, 0, 1)
        s = lambda u: u ** 3 * (10 - 15 * u + 6 * u ** 2)  # noqa: E731
        w = np.maximum(w, s(u_in) * s(u_out))
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-human", type=Path, help="dir with k1/run_video.py reference logs (logs/k1_ref_*.npz)")
    ap.add_argument("--gestures", action="store_true", help="(re)build data/arms_gestures.npz")
    a = ap.parse_args()
    if a.bundle_human:
        out = {}
        for f in sorted(a.bundle_human.glob("k1_ref_*.npz")):
            k = f.stem[len("k1_ref_"):]
            if "guap" in k or "mimic" in k:  # guap: not for the event; mimic: same arms as the split variant
                continue
            z = np.load(f, allow_pickle=True)
            names = [str(n) for n in z["names"]]
            assert names == POLICY_JOINTS, f
            out[k] = z["q"][:, ARM_IDX].astype(np.float32)
        np.savez_compressed(DATA / "arms_human.npz", **out)
        print(len(out), "human arm sequences,", round(sum(len(v) for v in out.values()) / 50 / 60, 1), "min")
    if a.gestures:
        g = load_gestures(rebuild=True)
        print({k: round(len(v) / 50, 1) for k, v in g.items()})


if __name__ == "__main__":
    main()
