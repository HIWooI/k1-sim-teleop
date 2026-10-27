"""Walking dataset clips through the K1 SONIC policy (same closed loop and metrics as eval_walk.py), to separate
"the policy cannot walk this" from "the live reference differs from the training one": each clip's command script is
run twice,
  clip: the built PKL (foot-locked K1 reference the policy was trained on) streamed as the planner would stream it
  live: the same command script through k1live.planner (no foot-lock), latency as the clip was built with
and the achieved body-frame velocity is compared with the reference's own velocity per command segment.
CPU heavy: run on a Linux CPU host.

    python sonic_k1_train/mac/eval_walk_clips.py --export data/ckpt/k1s_walk/exported --step 2000 \
        --dir data/k1_walk_planner --heldout --kinds side backward diag_back --n 8 --json logs/clips_k1s_walk_2000.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sonic_k1_train" / "mac"))

from eval_walk import run_sonic, yaw_of  # noqa: E402


class ClipSource:
    """A built motion_lib PKL replayed with the KinematicPlanner interface: frame 0 held for `hold` ticks (the planner's
    1 s of standing in run_sonic), then one 50 Hz frame per tick; moved rigidly onto the robot's xy / heading at reset."""

    def __init__(self, pkl, horizon=60, hold=50):
        import joblib

        e = next(iter(joblib.load(pkl).values()))
        assert abs(float(e["fps"]) - 50.0) < 1e-6, e["fps"]
        self.pos = np.asarray(e["root_trans_offset"], float).copy()
        self.quat = np.asarray(e["root_rot"], float)[:, [3, 0, 1, 2]].copy()
        self.q = np.asarray(e["dof"], float)
        self.dq = np.zeros_like(self.q)
        self.dq[:-1] = np.diff(self.q, axis=0) * 50.0
        self.dq[-1] = self.dq[-2]
        self.n, self.H, self.i = len(self.pos), horizon + 1, -hold
        self.stream = SimpleNamespace(n_replans=0, model=SimpleNamespace(last_ms=0.0))

    def reset(self, qpos):
        from sonic import rot

        dy = yaw_of(qpos[3:7]) - yaw_of(self.quat[0])
        c, s = np.cos(dy), np.sin(dy)
        xy = self.pos[:, :2] - self.pos[0, :2]
        self.pos[:, :2] = np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], 1) + qpos[:2]
        qz = np.array([np.cos(dy / 2), 0.0, 0.0, np.sin(dy / 2)])
        self.quat = rot.quat_mul(np.broadcast_to(qz, self.quat.shape), self.quat)

    def command(self, *a, **k):
        pass

    def step(self, state):
        from k1live.planner import RefWindow

        idx = np.clip(max(self.i, 0) + np.arange(self.H), 0, self.n - 1)
        self.i += 1
        return RefWindow(self.pos[idx], self.quat[idx], self.q[idx], self.dq[idx] if self.i > 0 else 0 * self.dq[idx])

    def close(self):
        pass


def seg_rows(r, kind_of):
    """(kind, cmd, ref (vx, vy, wz), robot (vx, vy, wz)) per steady moving segment; kind_of: command tuple -> kind."""
    out = []
    for s in r.get("segments", []):
        k = kind_of.get(tuple(s["cmd"]), "stand")
        if k == "stand" or "rvx" not in s:
            continue
        out.append((k, s["cmd"], (s["rvx"], s["rvy"], s["rwz"]), (s["vx"], s["vy"], s["wz"])))
    return out


def ratio(rows, ref=True):
    """achieved / reference (ref=True) or / command, projected on the reference (command) direction; planar for
    translating kinds, yaw rate for turn."""
    num = den = 0.0
    for k, c, rv, av in rows:
        tgt = np.array(rv if ref else c, float)
        a = np.array(av, float)
        if k == "turn":
            num += a[2] * np.sign(tgt[2]); den += abs(tgt[2])
        else:
            n = np.linalg.norm(tgt[:2])
            if n < 1e-3:
                continue
            num += a[:2] @ tgt[:2] / n; den += n
    return round(num / den, 2) if den > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", type=Path)
    ap.add_argument("--export", type=Path)
    ap.add_argument("--step", type=int)
    ap.add_argument("--future", default="4x1")
    ap.add_argument("--dir", type=Path, required=True, help="planner dataset dir (clips.json, robot/planner, heldout/planner)")
    ap.add_argument("--heldout", action="store_true", help="held-out clips (default: training clips)")
    ap.add_argument("--kinds", nargs="*", default=[], help="keep clips with a segment of one of these kinds")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--no-arms", action="store_true", help="skip clips with an arm overlay")
    ap.add_argument("--modes", nargs="*", default=["clip", "live"], choices=["clip", "live"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    vdir = a.variant_dir
    if a.export is not None:
        from eval_k1_mujoco import prepare_variant

        n, s = (int(x) for x in a.future.split("x"))
        vdir = prepare_variant(a.export, a.step, vdir or (a.export.parent / f"deploy_{a.step:06d}"), n, s)
    meta = [c for c in json.loads((a.dir / "clips.json").read_text()) if c["heldout"] == a.heldout]
    if a.kinds:
        meta = [c for c in meta if any(sg[2] in a.kinds for sg in c["segments"])]
    if a.no_arms:
        meta = [c for c in meta if not c.get("arms")]
    rng = np.random.default_rng(a.seed)
    meta = [meta[i] for i in sorted(rng.choice(len(meta), min(a.n, len(meta)), replace=False))]
    sub = "heldout/planner" if a.heldout else "robot/planner"
    results = {}
    rows = {m: defaultdict(list) for m in a.modes}
    for c in meta:
        segs = [(t0, t1, tuple(cmd)) for t0, t1, k, cmd in c["segments"]]
        kind_of = {tuple(cmd): k for t0, t1, k, cmd in c["segments"]}
        dur = c["seconds"] - 0.1
        results[c["name"]] = {}
        for mode in a.modes:
            t0 = time.time()
            src = ClipSource(a.dir / sub / f"{c['name']}.pkl") if mode == "clip" else None
            r, _ = run_sonic("k1", vdir, segs, None, "plan", False, a.seed, latency=c.get("latency_ticks", 2), source=src,
                             duration=dur)
            r["wall_s"] = round(time.time() - t0, 1)
            results[c["name"]][mode] = r
            for k, cmd, rv, av in seg_rows(r, kind_of):
                rows[mode][k].append((k, cmd, rv, av))
            print(c["name"], mode, "fell" if r["fell"] else "ok", json.dumps(r.get("segments")), flush=True)
    summary = {}
    for mode in a.modes:
        summary[mode] = {k: {"n": len(v), "vs_ref": ratio(v), "vs_cmd": ratio(v, ref=False),
                             "ref_vs_cmd": ratio([(k_, c_, c_, r_) for k_, c_, r_, _ in v], ref=False)}
                         for k, v in sorted(rows[mode].items())}
        summary[mode]["falls"] = sum(results[n][mode]["fell"] for n in results)
    print(json.dumps(summary, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"model": str(vdir), "heldout": a.heldout, "summary": summary, "clips": results},
                                           indent=1))


if __name__ == "__main__":
    main()
