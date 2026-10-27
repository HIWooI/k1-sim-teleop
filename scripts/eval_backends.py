"""Split vs sonic backend on the same person stream, through the k1live Backend interface (MuJoCo only).

Person stream = what k1live.sender publishes: the clip's MediaPipe landmarks (scripts/run_video.py cache) ->
k1live.human.HumanMapper (One-Euro, torso-frame position IK, mirror, waist yaw, bow, calibration) at the clip's frame
rate. Control at 50 Hz like k1live.sim_server: the newest HumanFrame (zero-order hold between camera frames), arm
targets slew-limited to 6 rad/s, then backend.step(). Backends (one run each):

  split               student arms + loco legs (the person's waist / bow are ignored: split cannot follow them)
  sonic:<future>      K1 SONIC 80 ms policy, planner idle, arms / waist / bow from the person;
                      future = delay | extrapolate | hold (k1live/backends/sonic.py)
  sonic:<future>:arms same with the person's arms only (waist / bow at the planner's)

Standing scenario (loco_cmd 0) per clip, plus --walk: the walking scenarios (sonic_k1_train/mac/eval_walk.py
SCENARIOS command scripts) with the clip's arms laid over the walk.

Metrics: arm error |q_robot - q_target| (deg, 10 arm joints, target = the slew-limited server target of the same
tick), the same at the best lag 0..20 ticks (and that lag in ms), waist error, falls, max tilt, standing drift.

  .venv/bin/python scripts/eval_backends.py --clips clap=<path> ... --backends split sonic:delay ... --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DT = 0.02
ARM_VMAX = 6.0


def person_stream(video, mirror=True, max_seconds=0.0):
    """-> list of (t_capture, HumanFrame-like dict) at the clip's frame rate (only frames the mapper returned)."""
    import cv2
    import run_video as g1rv

    from k1live.human import HumanMapper

    W, V, I, fps = g1rv.extract(video, "full", None, 1, 1280)
    cap = cv2.VideoCapture(str(video))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    hm = HumanMapper(mirror=mirror, loco=False)
    out = []
    n = len(W) if max_seconds <= 0 else min(len(W), int(max_seconds * fps))
    for k in range(n):
        t = k / fps
        det = None if np.isnan(W[k, 0, 0]) else {"world": W[k], "vis": V[k], "image": I[k]}
        f = hm.process(det, t, *wh)
        f.pop("obs", None)
        out.append((t, f))
    return out, fps


def run(cfg):
    from k1live.backends.base import make_backend

    backend = cfg["backend"]
    parts = backend.split(":")
    use_wb = True
    if parts[0] == "sonic":
        kw = {"arm_future": parts[1] if len(parts) > 1 else "delay"}
        if cfg.get("variant"):
            kw["variant_dir"] = cfg["variant"]
        use_wb = not (len(parts) > 2 and parts[2] == "arms")
        if cfg.get("segs") is not None:
            kw["walk"] = True          # evaluate walking with any checkpoint
        be = make_backend("sonic", **kw)
    else:
        be = make_backend("split")
        use_wb = False
    stream, fps = person_stream(cfg["video"], max_seconds=cfg.get("max_seconds", 0.0))
    segs = cfg.get("segs")
    T = stream[-1][0] + 1.0 if segs is None else max(stream[-1][0] + 1.0, 31.0)
    arm = be.arm_default.copy()
    k_frame = -1
    last = None
    rec = {"t": [], "tgt": [], "q": [], "waist_t": [], "waist_q": [], "tilt": [], "z": [], "xy": [], "ms": []}
    fell = None
    t0w = time.perf_counter()
    n = int(T / DT)
    ts_frames = np.array([s[0] for s in stream])
    period = ts_frames[-1] + 1.0 / fps
    for i in range(n):
        t = i * DT
        tt = t % period if segs is not None else t        # walking runs loop the clip
        k = int(np.searchsorted(ts_frames, tt, side="right")) - 1
        if k >= 0 and (k != k_frame):
            k_frame = k
            last = stream[k][1]
        tgt = be.arm_default if (last is None or last["arm_q"] is None) else np.asarray(last["arm_q"])
        arm = arm + np.clip(tgt - arm, -ARM_VMAX * DT, ARM_VMAX * DT)
        wy = last.get("waist_yaw") if (last is not None and use_wb) else None
        bw = last.get("bow") if (last is not None and use_wb) else None
        cmd = (0.0, 0.0, 0.0)
        if segs is not None:
            from eval_walk import cmd_at

            cmd = tuple(cmd_at(segs, t - 1.0))
        info = be.step(arm, waist_yaw=wy, bow=bw, loco_cmd=cmd)
        rec["t"].append(t)
        rec["tgt"].append(arm.copy())
        rec["q"].append(be.arm_q_now())
        rec["waist_t"].append(np.nan if wy is None else wy)
        rec["waist_q"].append(be.q_now()[be.names.index("waist_yaw_joint")])
        rec["tilt"].append(be.tilt_deg())
        rec["z"].append(be.pelvis_height())
        rec["xy"].append(be.root_pose()[:2])
        rec["ms"].append(info["policy_ms"])
        if fell is None and be.fallen():
            fell = round(t, 2)
            break
    be.close()
    R = {k: np.asarray(v, float) for k, v in rec.items()}
    tgt, q = R["tgt"], R["q"]
    err = np.degrees(np.abs(q - tgt)).mean(1)
    lag_err = []
    for L in range(0, 21):
        lag_err.append(float(np.degrees(np.abs(q[L:] - tgt[:len(tgt) - L])).mean()) if len(q) > L + 5 else np.inf)
    best = int(np.argmin(lag_err))
    moving = np.r_[0.0, np.degrees(np.abs(np.diff(tgt, axis=0)).max(1)) / DT] > 20.0   # target moving > 20 deg/s
    wt = R["waist_t"]
    wok = np.isfinite(wt)
    out = {"clip": cfg["name"], "backend": backend, "scenario": cfg.get("scenario", "stand"),
           "fell": fell is not None, "fell_at": fell, "seconds": round(float(R["t"][-1]), 1),
           "arm_err_deg": round(float(err.mean()), 2), "arm_err_p90_deg": round(float(np.percentile(err, 90)), 2),
           "arm_err_moving_deg": round(float(err[moving].mean()), 2) if moving.any() else None,
           "arm_err_bestlag_deg": round(lag_err[best], 2), "best_lag_ms": best * 20,
           "waist_err_deg": round(float(np.degrees(np.abs(R["waist_q"][wok] - wt[wok])).mean()), 2) if wok.any() else None,
           "max_tilt_deg": round(float(R["tilt"].max()), 1), "min_z": round(float(R["z"].min()), 3),
           "drift_m": round(float(np.linalg.norm(R["xy"][-1] - R["xy"][0])), 3),
           "policy_ms": round(float(R["ms"].mean()), 2), "wall_s": round(time.perf_counter() - t0w, 1)}
    if cfg.get("trace"):
        np.savez(cfg["trace"], **R)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", help="name=path")
    ap.add_argument("--backends", nargs="+", default=["split", "sonic:delay", "sonic:extrapolate", "sonic:hold",
                                                     "sonic:delay:arms"])
    ap.add_argument("--variant", default="", help="sonic deploy dir (default k1s_la4/deploy_002000)")
    ap.add_argument("--walk", nargs="*", default=None, help="eval_walk SCENARIOS names (or 'all') instead of standing")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "sonic_k1_train" / "mac"))
    cfgs = []
    scen = [("stand", None)]
    if a.walk is not None:
        from eval_walk import SCENARIOS

        names = list(SCENARIOS) if a.walk in ([], ["all"]) else a.walk
        scen = [(n, SCENARIOS[n][0]) for n in names]
    for spec in a.clips:
        name, path = spec.split("=", 1)
        for be in a.backends:
            for sname, segs in scen:
                c = {"name": name, "video": path, "backend": be, "variant": a.variant, "scenario": sname,
                     "segs": segs, "max_seconds": a.max_seconds}
                if a.trace_dir:
                    Path(a.trace_dir).mkdir(parents=True, exist_ok=True)
                    c["trace"] = str(Path(a.trace_dir) / f"{name}_{be.replace(':', '-')}_{sname}.npz")
                cfgs.append(c)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, cfgs))
    for r in res:
        print(f"{r['clip']:16s} {r['backend']:20s} {r['scenario']:16s} fell {str(r['fell']):5s} arm {r['arm_err_deg']:5.2f} "
              f"(p90 {r['arm_err_p90_deg']:5.2f}, moving {r['arm_err_moving_deg']}) bestlag {r['arm_err_bestlag_deg']:5.2f} "
              f"@{r['best_lag_ms']:3d}ms waist {r['waist_err_deg']} tilt {r['max_tilt_deg']:5.1f} drift {r['drift_m']:.2f} "
              f"ms {r['policy_ms']}", flush=True)
    summ = {}
    for be in a.backends:
        rs = [r for r in res if r["backend"] == be]
        summ[be] = {"runs": len(rs), "falls": sum(r["fell"] for r in rs),
                    **{k: round(float(np.mean([r[k] for r in rs if r[k] is not None])), 2)
                       for k in ("arm_err_deg", "arm_err_moving_deg", "arm_err_bestlag_deg", "best_lag_ms", "max_tilt_deg",
                                 "drift_m", "policy_ms") if any(r[k] is not None for r in rs)}}
    print(json.dumps(summ, indent=1))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"summary": summ, "runs": res}, indent=1))


if __name__ == "__main__":
    main()
