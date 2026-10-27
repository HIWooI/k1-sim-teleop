"""People's arms exactly as the live chain produces them: video -> MediaPipe (scripts/run_video.py cache) ->
k1live.human.HumanMapper (mirror, One-Euro, arm IK) -> arm targets, slew-limited to 6 rad/s like the sim server, 50 Hz.
Bundled into data/arms_live.npz ({clip: (T, 10) float32, k1live.protocol.ARM_JOINTS order}) for planner_dataset.py
(--arms-live). Why: the earlier person-arm overlays (arms_human.npz) came from k1/run_video.py's IK, and the K1 SONIC
walking checkpoints track the live HumanMapper's targets 2-8x worse than the split student (hand / elbow position,
scripts/eval_backends.py), while they track the run_video IK arms well.

Needs the k1live tree (k1live/human.py, scripts/run_video.py); run from the repo root on a Linux CPU host:
    .venv/bin/python sonic_k1_train/data/planner_arms_live.py --out data/arms_live.npz <video[@t0-t1]> ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FPS = 50.0
ARM_VMAX = 6.0


def arms_from_video(spec, mirror=True):
    import cv2
    import run_video as g1rv

    from k1live.human import HumanMapper

    path, _, rng = spec.partition("@")
    a, _, b = rng.partition("-")
    t0, t1 = float(a or 0), float(b) if b else None
    W, V, I, vfps = g1rv.extract(path, "full", None, 1, 1280)
    cap = cv2.VideoCapture(path)
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    k0, k1 = int(t0 * vfps), (len(W) if t1 is None else min(len(W), int(t1 * vfps)))
    hm = HumanMapper(mirror=mirror, loco=False)
    ts, qs = [], []
    for k in range(k0, k1):
        det = None if np.isnan(W[k, 0, 0]) else {"world": W[k], "vis": V[k], "image": I[k]}
        f = hm.process(det, k / vfps, *wh)
        if f.get("arm_q") is not None:
            ts.append((k - k0) / vfps)
            qs.append(np.asarray(f["arm_q"], float))
    if len(qs) < 2 * vfps:
        return None
    ts, qs = np.asarray(ts), np.asarray(qs)
    t50 = np.arange(ts[0], ts[-1], 1.0 / FPS)
    q = np.stack([np.interp(t50, ts, qs[:, j]) for j in range(qs.shape[1])], 1)
    out = q.copy()
    for i in range(1, len(q)):   # the server's arm slew limit
        out[i] = out[i - 1] + np.clip(q[i] - out[i - 1], -ARM_VMAX / FPS, ARM_VMAX / FPS)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="path[@t0-t1]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-mirror", action="store_true")
    a = ap.parse_args()
    out = {}
    for spec in a.videos:
        q = arms_from_video(spec, mirror=not a.no_mirror)
        name = Path(spec.partition("@")[0]).stem
        if q is None:
            print("skip (no person)", name, flush=True)
            continue
        out[name] = q
        print(name, f"{len(q) / FPS:.1f} s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(len(out), "live arm sequences,", round(sum(len(v) for v in out.values()) / FPS / 60, 1), "min ->", a.out)


if __name__ == "__main__":
    main()
