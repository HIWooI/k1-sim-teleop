"""Walking intent (k1live.locomotion.LocoEstimator) on real videos played as a fake camera.

Each clip runs at its own frame rate as if it were the camera (t = frame / fps, which is what the sender stamps):
MediaPipe Pose full (same cache as scripts/run_video.py) -> home = the estimator's own auto-home (the person stood
still for auto_home_s; what the sender's "stable for 0.5 s" calibration amounts to) or, with name=path@t0-t1:tc,
calibrate() with the landmarks averaged over the 0.5 s before tc -> LocoEstimator.update() on every tracked frame.

  .venv/bin/python scripts/eval_loco_video.py --gesture event_stand=<path> ... --walk w1=<path>[@t0-t1] ... \
      --json logs/loco_eval.json --plots logs/loco_plots

gesture clips: the person stays in place -> any non-zero command is a false walk (reported per axis).
walk clips: start/stop delays vs an offline, zero-phase smoothed version of the same offsets (what a perfect
non-causal observer would say): delay = causal command onset - the smoothed offset crossing --start-thr, stop delay
= causal command zero - the smoothed offset dropping back under --stop-thr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from k1live.locomotion import LocoEstimator, measure  # noqa: E402

AXES = (("vx", "fwd", 1.0), ("vy", "side", 1.0), ("wz", "yaw", 1.0))


def parse_clip(spec):
    name, _, rest = spec.partition("=")
    path, _, rng = rest.partition("@")
    rng, _, tc = rng.partition(":")
    a, _, b = rng.partition("-")
    t0, t1 = float(a or 0.0), float(b) if b else float("inf")
    return name, Path(path), t0, t1, (float(tc) if tc else None)


def load(path, max_side=1280):
    import run_video as g1rv

    W, V, I, fps = g1rv.extract(path, "full", None, 1, max_side)
    cap = cv2.VideoCapture(str(path))
    wh = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return W, V, I, fps, wh


def obs_at(W, V, I, k, fps, wh):
    if np.isnan(W[k, 0, 0]):
        return None
    return {"t": k / fps, "img_w": wh[0], "img_h": wh[1], "lm_img": np.c_[I[k], V[k]], "lm_world": np.c_[W[k], V[k]]}


def run_clip(path, t0=0.0, t1=float("inf"), mirror=True, calib_at=None, params=None):
    W, V, I, fps, wh = load(path)
    est = LocoEstimator(mirror=mirror, **(params or {}))
    est.offset = None
    k0, k1 = int(t0 * fps), min(len(W), int(t1 * fps) if math.isfinite(t1) else len(W))
    window, rows = [], []
    calibrated = calib_at is None   # auto-home: update() from the first frame
    t_cal = None
    for k in range(k0, k1):
        obs = obs_at(W, V, I, k, fps, wh)
        if not calibrated:
            if obs is not None:
                window.append(obs)
            if k / fps >= calib_at and window:
                w = window[-int(0.5 * fps):]
                avg = dict(obs or w[-1], lm_img=np.mean([o["lm_img"] for o in w], 0),
                           lm_world=np.mean([o["lm_world"] for o in w], 0))
                calibrated = est.calibrate(avg)
            continue
        if obs is None:   # the sender does not call update() while nobody is tracked
            rows.append((k / fps, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0))
            continue
        c = est.update(obs)
        if t_cal is None and est.home is not None:
            t_cal = k / fps
        m = est.last
        o = est.offset if est.home is not None and est.offset is not None else {"fwd": np.nan, "side": np.nan, "yaw": np.nan}
        rows.append((k / fps, True, m.z if m else np.nan, m.x if m else np.nan, m.yaw if m else np.nan,
                     o["fwd"], o["side"], o["yaw"], c["vx"], c["vy"], c["wz"]))
    A = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 11))
    return {"t": A[:, 0], "present": A[:, 1] > 0, "z": A[:, 2], "x": A[:, 3], "yaw": A[:, 4],
            "fwd": A[:, 5], "side": A[:, 6], "dyaw": A[:, 7], "cmd": A[:, 8:11], "fps": fps,
            "t_cal": t_cal, "home": est.home, "n_frames": k1 - k0}


def onsets(x):
    nz = np.abs(x) > 0
    return int(np.sum(nz[1:] & ~nz[:-1]) + (1 if len(nz) and nz[0] else 0))


def gesture_metrics(r):
    c = r["cmd"]
    out = {"seconds": round(float(r["t"][-1] - r["t"][0]), 1) if len(r["t"]) else 0.0,
           "tracked_frac": round(float(r["present"].mean()), 3) if len(r["t"]) else 0.0,
           "false_walk_frac": round(float(np.any(c != 0, axis=1).mean()), 4) if len(c) else 0.0,
           "false_onsets": {a: onsets(c[:, i]) for i, (a, _, _) in enumerate(AXES)},
           "max_abs_offset": {"fwd_m": round(float(np.nanmax(np.abs(r["fwd"]))), 3),
                              "side_m": round(float(np.nanmax(np.abs(r["side"]))), 3),
                              "yaw_deg": round(float(np.degrees(np.nanmax(np.abs(r["dyaw"])))), 1)}}
    return out


def _smooth(t, x, sigma=0.15):
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return x
    xi = np.interp(t, t[ok], x[ok])
    dt = np.median(np.diff(t))
    k = np.arange(-int(3 * sigma / dt), int(3 * sigma / dt) + 1) * dt
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    pad = len(k) // 2
    xp = np.pad(xi, pad, mode="edge")
    return np.convolve(xp, g, mode="valid")


def walk_metrics(r, thr):
    """thr: axis -> (start, stop) thresholds in the offset's units (m / rad)."""
    t = r["t"]
    out = {}
    for i, (a, off, _) in enumerate(AXES):
        s = _smooth(t, r["fwd" if off == "fwd" else ("side" if off == "side" else "dyaw")])
        st, sp = thr[a]
        c = r["cmd"][:, i]
        events = []
        k = 0
        while k < len(t):
            if abs(s[k]) >= st:
                kc = k
                ke = kc
                while ke < len(t) and abs(s[ke]) >= sp:
                    ke += 1
                on = np.flatnonzero((np.abs(c) > 0) & (t >= t[kc] - 1.0) & (t <= t[min(ke, len(t) - 1)]))
                t_on = float(t[on[0]]) if len(on) else None
                t_back = float(t[ke]) if ke < len(t) else None
                off_i = np.flatnonzero((c == 0) & (t >= (t_back if t_back is not None else np.inf)))
                t_off = float(t[off_i[0]]) if len(off_i) else None
                events.append({"t_cross": round(float(t[kc]), 2), "start_delay_s": None if t_on is None else round(t_on - float(t[kc]), 3),
                               "t_back": None if t_back is None else round(t_back, 2),
                               "stop_delay_s": None if t_off is None or t_back is None else round(t_off - t_back, 3),
                               "sign": int(np.sign(s[kc]))})
                k = ke + 1
            else:
                k += 1
        moving = np.abs(s) >= thr[a][1]
        spurious = np.flatnonzero((np.abs(c) > 0) & ~_dilate(moving, int(1.2 * r["fps"])))
        out[a] = {"events": events, "spurious_frac": round(float(len(spurious) / max(len(t), 1)), 4)}
    return out


def _dilate(mask, n):
    m = mask.copy()
    for k in range(1, n + 1):
        m[k:] |= mask[:-k]
        m[:-k] |= mask[k:]
    return m


def plot(r, name, out_png, est_p):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = r["t"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    for i, (key, lab, st, sp, unit) in enumerate((("fwd", "forward (m, + = toward camera)", est_p["fwd_start"], est_p["fwd_stop"], 1),
                                                  ("side", "sideways (m, + = image right)", est_p["side_start"], est_p["side_stop"], 1),
                                                  ("dyaw", "body yaw (deg, + = person's left)", est_p["yaw_start"], est_p["yaw_stop"], 0))):
        y = r[key] if unit else np.degrees(r[key])
        ax[i].plot(t, y, color="#4a6fa5", lw=1.2, label="offset vs home")
        for v, ls in ((st, "--"), (sp, ":")):
            ax[i].axhline(v, color="#999", ls=ls, lw=0.8)
            ax[i].axhline(-v, color="#999", ls=ls, lw=0.8)
        ax2 = ax[i].twinx()
        ax2.plot(t, r["cmd"][:, i], color="#c4553a", lw=1.2)
        ax2.set_ylabel(("vx m/s", "vy m/s", "wz rad/s")[i], color="#c4553a")
        ax[i].set_ylabel(lab, fontsize=8)
    ax[-1].set_xlabel("t (s)")
    fig.suptitle(name)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gesture", nargs="*", default=[], help="name=path[@t0-t1]: in-place clips (false walks)")
    ap.add_argument("--walk", nargs="*", default=[], help="name=path[@t0-t1]: clips where the person moves")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--param", action="append", default=[], help="LocoEstimator param override k=v")
    ap.add_argument("--json", default="")
    ap.add_argument("--plots", default="")
    a = ap.parse_args()
    params = {}
    for kv in a.param:
        k, v = kv.split("=")
        params[k] = type(LocoEstimator.DEFAULTS[k])(float(v)) if not isinstance(LocoEstimator.DEFAULTS[k], bool) else v in ("1", "true", "True")
    p = dict(LocoEstimator.DEFAULTS, **params)
    thr = {"vx": (0.40, p["fwd_stop"]), "vy": (0.40, p["side_stop"]), "wz": (math.radians(30), math.radians(p["yaw_stop"]))}
    res = {"params": p, "mirror": not a.no_mirror, "gesture": {}, "walk": {}}
    for kind, specs in (("gesture", a.gesture), ("walk", a.walk)):
        for spec in specs:
            name, path, t0, t1, tc = parse_clip(spec)
            r = run_clip(path, t0, t1, mirror=not a.no_mirror, calib_at=tc, params=params)
            m = gesture_metrics(r) if len(r["t"]) else {"error": "never calibrated"}
            if kind == "walk" and len(r["t"]):
                m["walk"] = walk_metrics(r, thr)
            m.update(video=str(path), t0=t0, t1=None if not math.isfinite(t1) else t1, t_calibrated=r["t_cal"])
            res[kind][name] = m
            if a.plots and len(r["t"]):
                plot(r, f"{kind}: {name}", Path(a.plots) / f"{kind}_{name}.png", p)
            fo = m.get("false_onsets", {})
            print(f"{kind:7s} {name:24s} {m.get('seconds', 0):5.1f}s tracked {m.get('tracked_frac', 0):.2f} "
                  f"nonzero {m.get('false_walk_frac', 0):.3f} onsets {fo} max {m.get('max_abs_offset')}", flush=True)
            if kind == "walk" and "walk" in m:
                for ax_, d in m["walk"].items():
                    for e in d["events"]:
                        print(f"        {ax_} sign {e['sign']:+d} cross {e['t_cross']:5.2f} start {e['start_delay_s']} "
                              f"back {e['t_back']} stop {e['stop_delay_s']}")
                    print(f"        {ax_} spurious {d['spurious_frac']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
