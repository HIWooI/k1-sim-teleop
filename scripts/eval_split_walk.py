"""Split backend (student arms + ROBOTIS loco legs) walking while the arms follow gestures: falls, shake, speed.

Each trial: 1 s standing, then `--seconds` of a random walking programme (piecewise-constant segments: stop, forward,
back, sideways, turn, forward+turn, diagonal; 1.5-4 s each, speeds in the LocoEstimator output range) passed
through k1live.locomotion.VelocityLimiter at 50 Hz, while the arms play gestures:

  rest      arms at the rest pose
  gestures  K1 event gestures (k1/gestures.py: wave, welcome, clap, guide, heart, hooray, at ease), random order
  video     the arm IK of real gesture videos (k1/run_video.py pipeline: MediaPipe -> One-Euro -> position IK), looped

The arm reference reaches the backend the way sim_server sends it (6 rad/s slew limit).

  .venv/bin/python scripts/eval_split_walk.py --trials 24 --seconds 60 --workers 12 --json logs/split_walk.json

Metrics per trial: fall (pelvis < 0.45 m or tilt > 45 deg), max / RMS pelvis tilt while walking, pelvis roll/pitch
rate RMS (shake), arm tracking error (|q - q_ref|, deg) standing vs walking, achieved/commanded body-frame speed per
axis (second half of each segment), stop distance and time after the command returns to 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FPS = 50
DT = 1.0 / FPS
ARM_VMAX = 6.0  # rad/s, sim_server's arm slew limit

SEG_TYPES = ("stop", "forward", "back", "side", "turn", "forward_turn", "diagonal")
SEG_P = (0.22, 0.22, 0.10, 0.14, 0.14, 0.10, 0.08)


def random_programme(rng, seconds):
    """[(t_start, duration, (vx, vy, wz), type)] covering `seconds`; starts and ends with a stop. Speeds span what
    LocoEstimator emits (forward 0.25-0.6, back 0.25-0.3, side 0.4-0.45, turn 0.6-0.9; the limiter then lifts them
    to the policy start levels)."""
    segs, t = [(0.0, 1.0, (0.0, 0.0, 0.0), "stop")], 1.0
    while t < seconds - 2.0:
        kind = rng.choice(SEG_TYPES, p=SEG_P)
        d = float(rng.uniform(1.5, 4.0))
        s = lambda: float(rng.choice([-1.0, 1.0]))  # noqa: E731
        c = {"stop": (0, 0, 0),
             "forward": (rng.uniform(0.25, 0.6), 0, 0),
             "back": (-rng.uniform(0.25, 0.3), 0, 0),
             "side": (0, s() * rng.uniform(0.4, 0.45), 0),
             "turn": (0, 0, s() * rng.uniform(0.6, 0.9)),
             "forward_turn": (rng.uniform(0.25, 0.5), 0, s() * rng.uniform(0.6, 0.9)),
             "diagonal": (rng.uniform(0.25, 0.45), s() * rng.uniform(0.4, 0.45), 0)}[kind]
        segs.append((t, d, tuple(float(x) for x in c), kind))
        t += d
    segs.append((t, seconds - t + 2.0, (0.0, 0.0, 0.0), "stop"))
    return segs


def gesture_arms(names, rng, seconds):
    """Concatenated K1 event gestures (arm columns, policy order), 50 Hz, random order, 0.5-1.5 s rest between."""
    from k1.gestures import GESTURES, ArmIK
    from k1.sim import K1Split

    from k1live.protocol import ARM_JOINTS

    ik = ArmIK(K1Split())
    arm_idx = [list(names).index(n) for n in ARM_JOINTS]
    cache = {}
    order = [g for g in GESTURES if g != "bow"]
    out = []
    rest = None
    while len(out) < seconds * FPS:
        g = order[int(rng.integers(len(order)))]
        if g not in cache:
            cache[g] = GESTURES[g](ik).q[:, arm_idx]
        q = cache[g]
        rest = q[0] if rest is None else rest
        out.extend([rest] * int(rng.uniform(0.5, 1.5) * FPS))
        out.extend(q)
    return np.array(out[: int(seconds * FPS) + 1]), "+".join(sorted(cache))


def video_arms(names, video, seconds):
    """Arm IK of a real gesture video (k1/run_video pipeline), resampled to 50 Hz, ping-pong looped."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_video as g1rv

    from k1.run_video import retarget
    from k1live.protocol import ARM_JOINTS
    from sonic.mp2smpl import OneEuro

    W, V, I, vfps = g1rv.extract(video, "full", None, 1, 1280)
    valid = ~np.isnan(W[:, 0, 0])
    first = int(np.argmax(valid))
    for t in range(len(W)):
        if not valid[t]:
            W[t], V[t] = W[t - 1 if t > first else first], V[t - 1 if t > first else first]
    filt = OneEuro(1.0, 0.4)
    Wf = np.array([filt(W[t], t / vfps) for t in range(len(W))])
    Q, _ = retarget(Wf, V, names, frame="torso")
    arm_idx = [list(names).index(n) for n in ARM_JOINTS]
    A = Q[:, arm_idx]
    t_src = np.arange(len(A)) / vfps
    t50 = np.arange(int(t_src[-1] * FPS) + 1) / FPS
    A50 = np.stack([np.interp(t50, t_src, A[:, j]) for j in range(A.shape[1])], 1)
    wr = [ARM_JOINTS.index(n) for n in ("left_wrist_roll_joint", "right_wrist_roll_joint")]
    reps = []
    k = 0
    while sum(len(r) for r in reps) < seconds * FPS + 1:
        reps.append(A50 if k % 2 == 0 else A50[::-1])
        k += 1
    out = np.concatenate(reps)[: int(seconds * FPS) + 1]
    return out, wr


def body_vel(be, prev):
    """Body-frame (vx, vy, wz) from two root poses one tick apart."""
    x, y, yaw = be.root_pose()
    if prev is None:
        return (x, y, yaw), np.zeros(3)
    px, py, pyaw = prev
    dx, dy = (x - px) / DT, (y - py) / DT
    c, s = math.cos(yaw), math.sin(yaw)
    dyaw = (yaw - pyaw + math.pi) % (2 * math.pi) - math.pi
    return (x, y, yaw), np.array([c * dx + s * dy, -s * dx + c * dy, dyaw / DT])


def trial(cfg):
    from k1live.backends.split import SplitBackend
    from k1live.locomotion import VelocityLimiter

    seed, seconds, arms_src, lim_params = cfg["seed"], cfg["seconds"], cfg["arms"], cfg.get("limiter", {})
    rng = np.random.default_rng(seed)
    be = SplitBackend()
    names = be.names
    if arms_src == "rest":
        arms, arm_desc = np.tile(be.arm_default, (int((seconds + 1.0) * FPS) + 1, 1)), "rest"
    elif arms_src == "gestures":
        arms, arm_desc = gesture_arms(names, rng, seconds + 1.0)
    else:
        arms, wr = video_arms(names, arms_src, seconds + 1.0)
        arm_desc = Path(arms_src).stem
        arms[:, wr] = be.arm_default[wr]   # MediaPipe gives no forearm twist: wrist roll at rest (k1/run_video)
    prog = cfg.get("programme") or random_programme(rng, seconds)
    lim = VelocityLimiter(**lim_params) if lim_params is not None else None
    be.reset()
    n_settle = FPS
    n = int((seconds + 1.0) * FPS)
    arm_ref = be.arm_default.copy()
    prev = None
    rec = {k: [] for k in ("t", "cmd", "lim", "vel", "tilt", "z", "rate", "arm_err", "ms")}
    fell = None
    t_wall = time.perf_counter()
    for k in range(n):
        t = (k - n_settle) / FPS
        if t < 0:
            cmd = np.zeros(3)
            target = be.arm_default
        else:
            seg = next(s for s in reversed(prog) if s[0] <= t)
            cmd = np.array(seg[2])
            target = arms[min(int(t * FPS), len(arms) - 1)]
        arm_ref = arm_ref + np.clip(target - arm_ref, -ARM_VMAX * DT, ARM_VMAX * DT)
        v = lim({"vx": cmd[0], "vy": cmd[1], "wz": cmd[2]}, DT) if lim is not None else cmd
        info = be.step(arm_ref, loco_cmd=tuple(v))
        prev, vel = body_vel(be, prev)
        d = be.data
        rec["t"].append(t)
        rec["cmd"].append(cmd)
        rec["lim"].append(np.asarray(v, float))
        rec["vel"].append(vel)
        rec["tilt"].append(be.tilt_deg())
        rec["z"].append(be.pelvis_height())
        rec["rate"].append(np.degrees(d.qvel[3:5].copy()))
        rec["arm_err"].append(np.degrees(np.abs(be.arm_q_now() - arm_ref)))
        rec["ms"].append(info["policy_ms"])
        if fell is None and be.fallen():
            fell = round(t, 2)
            break
    R = {k: np.array(v) for k, v in rec.items()}
    t = R["t"]
    run = t >= 0
    walking = run & (np.abs(R["lim"]).max(1) > 0.02)
    standing = run & ~walking
    out = {"seed": seed, "arms": arm_desc, "seconds": seconds, "fell": fell is not None, "fell_at": fell,
           "max_tilt_deg": round(float(R["tilt"][run].max()), 2),
           "tilt_rms_walk_deg": round(float(np.sqrt(np.mean(R["tilt"][walking] ** 2))), 2) if walking.any() else None,
           "rate_rms_walk_dps": round(float(np.sqrt(np.mean(R["rate"][walking] ** 2))), 1) if walking.any() else None,
           "rate_rms_stand_dps": round(float(np.sqrt(np.mean(R["rate"][standing] ** 2))), 1) if standing.any() else None,
           "arm_err_walk_deg": round(float(R["arm_err"][walking].mean()), 2) if walking.any() else None,
           "arm_err_stand_deg": round(float(R["arm_err"][standing].mean()), 2) if standing.any() else None,
           "min_pelvis_z": round(float(R["z"][run].min()), 3),
           "policy_ms_mean": round(float(R["ms"].mean()), 2), "wall_s": round(time.perf_counter() - t_wall, 1),
           "walk_frac": round(float(walking.mean()), 3)}
    # speed ratio per segment (second half), stop distance after each walk -> stop transition
    ratios = {"vx": [], "vx_back": [], "vy": [], "wz": []}
    stops = []
    pos = None
    for i, (ts, dur, c, kind) in enumerate(prog):
        if ts + dur > t[-1]:
            break
        sel = (t >= ts + 0.5 * dur) & (t < ts + dur)
        if kind != "stop" and sel.sum() > 10:
            m = R["vel"][sel].mean(0)
            cl = R["lim"][sel].mean(0)
            for j, key in enumerate(("vx", "vy", "wz")):
                if abs(cl[j]) > 0.1:
                    kk = "vx_back" if key == "vx" and cl[j] < 0 else key
                    ratios[kk].append(m[j] / cl[j])
        if kind == "stop" and i > 0 and prog[i - 1][3] != "stop":
            k0 = int(np.searchsorted(t, ts))
            sp = np.linalg.norm(R["vel"][:, :2], axis=1)
            sp_s = np.convolve(sp, np.ones(10) / 10, mode="same")
            ks = k0 + int(np.argmax(sp_s[k0:k0 + int(dur * FPS)] < 0.05)) if k0 < len(t) else k0
            stops.append({"after": prog[i - 1][3], "time_s": round(float(t[ks] - ts), 2)})
    for k_, v in ratios.items():
        out[f"speed_ratio_{k_}"] = round(float(np.median(v)), 3) if v else None
    out["stop_time_s_max"] = max((s["time_s"] for s in stops), default=None)
    out["stop_time_s_mean"] = round(float(np.mean([s["time_s"] for s in stops])), 2) if stops else None
    if cfg.get("trace"):
        np.savez(cfg["trace"], **R)
    return out


def stop_distance(cfg_base, speeds=(0.3, 0.45, 0.6), lim_params=None):
    """Walk forward at v for 4 s, command 0, measure the distance travelled from the stop command until still."""
    from k1live.backends.split import SplitBackend
    from k1live.locomotion import VelocityLimiter

    res = []
    for v in speeds:
        be = SplitBackend()
        be.reset()
        lim = VelocityLimiter(**(lim_params or {}))
        arm = be.arm_default.copy()
        p_stop = None
        prev = None
        t_still = None
        for k in range(int(9 * FPS)):
            t = k / FPS
            c = v if 1.0 <= t < 5.0 else 0.0
            u = lim({"vx": c}, DT)
            be.step(arm, loco_cmd=tuple(u))
            prev, vel = body_vel(be, prev)
            if t >= 5.0 and p_stop is None:
                p_stop = np.array(be.root_pose()[:2])
            if p_stop is not None and t_still is None and t > 5.3 and abs(vel[0]) < 0.03:
                t_still = t
        x1 = np.array(be.root_pose()[:2])
        res.append({"cmd": v, "stop_distance_m": round(float(np.linalg.norm(x1 - p_stop)), 3),
                    "time_to_still_s": None if t_still is None else round(t_still - 5.0, 2), "fell": be.fallen()})
    return res


SWEEP = ([("vx", v) for v in (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
         + [("vx", -v) for v in (0.2, 0.25, 0.3, 0.4, 0.5)]
         + [("vy", v) for v in (0.3, 0.35, 0.4, 0.5, 0.6, -0.3, -0.4)]
         + [("wz", v) for v in (0.6, 0.7, 0.8, 1.0, 1.2, -0.6, -0.7)])
WIDE = dict(vx_max=1.0, vx_back_max=0.6, vy_max=0.7, wz_max=1.4)   # sweep: no caps below the probed values


def sweep_one(cfg):
    """Steady command for 8 s (after 1 s standing), achieved body-frame speed over the last 4 s."""
    axis, v = cfg["axis_v"]
    c = {"vx": (v, 0, 0), "vy": (0, v, 0), "wz": (0, 0, v), "arc": (0.25, 0, v)}[axis]
    prog = [(0.0, 1.0, (0.0, 0.0, 0.0), "stop"), (1.0, 8.0, c, "seg"), (9.0, 3.0, (0.0, 0.0, 0.0), "stop")]
    r = trial({"seed": cfg["seed"], "seconds": 12.0, "arms": cfg["arms"], "limiter": WIDE, "programme": prog,
               "trace": cfg["trace"]})
    R = np.load(cfg["trace"])
    t = R["t"]
    sel = (t >= 5.0) & (t < 9.0)
    j = {"vx": 0, "vy": 1, "wz": 2, "arc": 2}[axis]
    ach = float(R["vel"][sel, j].mean()) if sel.any() else float("nan")
    return {"axis": axis, "cmd": v, "arms": cfg["arms"] if cfg["arms"] in ("gestures", "rest") else Path(cfg["arms"]).stem,
            "achieved": round(ach, 3), "ratio": round(ach / v, 3), "fell": r["fell"], "max_tilt_deg": r["max_tilt_deg"],
            "cross": [round(float(x), 3) for x in R["vel"][sel].mean(0)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--arms", nargs="*", default=["gestures"], help="'gestures' and/or gesture video paths (cycled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limiter", default="default", help="default | none | JSON dict of VelocityLimiter params")
    ap.add_argument("--stop-test", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="steady-command speed map (each axis value x each --arms)")
    ap.add_argument("--sweep-values", default="", help="axis:v,v,...;axis:... instead of the built-in SWEEP")
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    lim = {} if a.limiter == "default" else (None if a.limiter == "none" else json.loads(a.limiter))
    cfgs = []
    for i in range(a.trials):
        c = {"seed": a.seed + i, "seconds": a.seconds, "arms": a.arms[i % len(a.arms)], "limiter": lim}
        if a.trace_dir:
            Path(a.trace_dir).mkdir(parents=True, exist_ok=True)
            c["trace"] = str(Path(a.trace_dir) / f"trial_{a.seed + i:03d}.npz")
        cfgs.append(c)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.sweep:
        tdir = Path(a.trace_dir or "logs/sweep_traces")
        tdir.mkdir(parents=True, exist_ok=True)
        sweep = SWEEP if not a.sweep_values else [(ax_, float(v)) for part in a.sweep_values.split(";")
                                                   for ax_, vs in [part.split(":")] for v in vs.split(",")]
        cs = [{"axis_v": av, "seed": i, "arms": arm, "trace": str(tdir / f"sw_{i:03d}.npz")}
              for i, (av, arm) in enumerate((av, arm) for arm in a.arms for av in sweep)]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(sweep_one, cs))
        for r in res:
            print(f"{r['arms'][:20]:20s} {r['axis']} cmd {r['cmd']:+.2f} -> {r['achieved']:+.3f} ratio {r['ratio']:.2f} "
                  f"fell {r['fell']} tilt {r['max_tilt_deg']} cross {r['cross']}")
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1))
        return
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(trial, cfgs))
    for r in results:
        print(f"seed {r['seed']:3d} {r['arms'][:28]:28s} fell {r['fell']} at {r['fell_at']} tilt max {r['max_tilt_deg']:5.1f} "
              f"rms {r['tilt_rms_walk_deg']} rate {r['rate_rms_walk_dps']}/{r['rate_rms_stand_dps']} arm {r['arm_err_walk_deg']}/"
              f"{r['arm_err_stand_deg']} ratio vx {r['speed_ratio_vx']} back {r['speed_ratio_vx_back']} vy {r['speed_ratio_vy']} "
              f"wz {r['speed_ratio_wz']} stop {r['stop_time_s_mean']}/{r['stop_time_s_max']} ms {r['policy_ms_mean']}", flush=True)
    def med(key):
        v = [r[key] for r in results if r.get(key) is not None]
        return round(float(np.median(v)), 3) if v else None
    summ = {"trials": len(results), "falls": sum(r["fell"] for r in results), "seconds_each": a.seconds,
            "limiter": a.limiter, "max_tilt_deg": max(r["max_tilt_deg"] for r in results)}
    for k in ("tilt_rms_walk_deg", "rate_rms_walk_dps", "rate_rms_stand_dps", "arm_err_walk_deg", "arm_err_stand_deg",
              "speed_ratio_vx", "speed_ratio_vx_back", "speed_ratio_vy", "speed_ratio_wz", "stop_time_s_mean",
              "policy_ms_mean"):
        summ[k] = med(k)
    summ["stop_time_s_max"] = max((r["stop_time_s_max"] or 0) for r in results)
    out = {"summary": summ, "trials": results}
    if a.stop_test:
        out["stop_distance"] = stop_distance({}, lim_params=lim or {})
        print(json.dumps(out["stop_distance"]))
    print(json.dumps(summ))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
