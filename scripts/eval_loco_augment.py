"""Walking intent on real MediaPipe streams with a known walk added (ground truth for start/stop delays).

A clip of a person gesturing in place (real MediaPipe noise, arm motion, bows, weight shifts) is turned into the same
person stepping forward / back / sideways or turning by a known trajectory, applied to the landmarks the way a
pinhole camera would see it:

* distance Z0 -> Z0 - d: image points scale about the principal point by Z0 / Z (world landmarks unchanged)
* lateral X: image points shift by f * X / Z
* turn: world landmarks rotate about the vertical through the hip centre; image points move by s * (R w - w)_xy
  (weak perspective, s = the frame's own fitted px/m)

Trajectory per excursion: 2 s after home -> move `amount` at `speed` -> hold `hold` s -> move back -> 3 s still.
Delays: start = first non-zero command - the moment the true offset crosses `thr` (0.4 m / 35 deg); stop = first zero
command - the moment the person is back home (and - the moment the offset is back under `thr`).

  .venv/bin/python scripts/eval_loco_augment.py --clips event_stand=<path> ... --json logs/loco_augment.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_loco_video import load, obs_at, parse_clip  # noqa: E402
from k1live.locomotion import LocoEstimator, measure  # noqa: E402

# (name, axis, amount, speed, threshold): forward = toward the camera; side + = image right (the person's left);
# turn + = the person's left
EXCURSIONS = [
    ("forward", "fwd", 0.6, 0.8, 0.40), ("back", "fwd", -0.5, 0.6, 0.40),
    ("side_left", "side", 0.5, 0.6, 0.40), ("side_right", "side", -0.5, 0.6, 0.40),
    ("turn_left", "yaw", math.radians(50), math.radians(90), math.radians(35)),
    ("turn_right", "yaw", -math.radians(50), math.radians(90), math.radians(35)),
]
COL = {"fwd": 0, "side": 1, "yaw": 2}


def rot_vertical(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def trajectory(t, t0, amount, speed, hold):
    dur = abs(amount) / speed
    t1, t2, t3 = t0 + dur, t0 + dur + hold, t0 + 2 * dur + hold
    if t < t0 or t >= t3:
        return 0.0
    if t < t1:
        return amount * (t - t0) / dur
    if t < t2:
        return amount
    return amount * (1 - (t - t2) / dur)


def augment(obs, axis, off, f_px, z0):
    o = dict(obs)
    img = obs["lm_img"].copy()
    wld = obs["lm_world"].copy()
    W, H = obs["img_w"], obs["img_h"]
    u = img[:, 0] * W - 0.5 * W
    v = img[:, 1] * H - 0.5 * H
    if axis == "fwd":
        k = z0 / max(z0 - off, 0.3)
        u, v = u * k, v * k
    elif axis == "side":
        u = u + f_px * off / z0
    else:
        m = measure(obs)
        s = m.scale if m is not None else f_px / z0
        w2 = wld[:, :3] @ rot_vertical(off).T
        u = u + s * (w2[:, 0] - wld[:, 0])
        v = v + s * (w2[:, 1] - wld[:, 1])
        wld[:, :3] = w2
    img[:, 0] = (u + 0.5 * W) / W
    img[:, 1] = (v + 0.5 * H) / H
    o["lm_img"], o["lm_world"] = img, wld
    return o


def robot_response(t, cmd, axis, t_cross, t_home_back, amount):
    """The estimator output (held between camera frames) -> VelocityLimiter at 50 Hz -> split backend (arms at
    rest). Robot start = body-frame speed along the axis (0.3 s mean) > 0.1 m/s (0.25 rad/s) in the commanded direction;
    robot stop = speed < 0.05 m/s (0.1 rad/s) for good after the person is back home."""
    from k1live.backends.split import SplitBackend
    from k1live.locomotion import VelocityLimiter

    be = SplitBackend()
    be.reset()
    lim = VelocityLimiter()
    j = COL[axis]
    t0 = t[0] - 1.0
    ts = np.arange(t0, t[-1], 0.02)
    prev, vel = None, []
    for tt in ts:
        k = int(np.searchsorted(t, tt, side="right")) - 1
        c = cmd[k] if k >= 0 else np.zeros(3)
        v = lim({"vx": c[0], "vy": c[1], "wz": c[2]}, 0.02)
        be.step(be.arm_default, loco_cmd=tuple(v))
        x, y, yaw = be.root_pose()
        if prev is not None:
            dx, dy = (x - prev[0]) / 0.02, (y - prev[1]) / 0.02
            cy, sy = math.cos(yaw), math.sin(yaw)
            dyaw = (yaw - prev[2] + math.pi) % (2 * math.pi) - math.pi
            vel.append((cy * dx + sy * dy, -sy * dx + cy * dy, dyaw / 0.02))
        else:
            vel.append((0.0, 0.0, 0.0))
        prev = (x, y, yaw)
    vel = np.array(vel)
    sm = np.convolve(vel[:, j], np.ones(15) / 15, mode="same")
    on_thr, off_thr = (0.25, 0.1) if axis == "yaw" else (0.1, 0.05)
    exp = np.sign(cmd[np.abs(cmd[:, j]) > 0, j].mean()) if np.any(cmd[:, j] != 0) else 0.0
    k_on = np.flatnonzero((ts > t_cross - 1.0) & (sm * exp > on_thr))
    t_on = float(ts[k_on[0]]) if len(k_on) else None
    moving = np.abs(sm) > off_thr
    after = np.flatnonzero(ts > t_home_back)
    t_still = None
    if len(after):
        mv = np.flatnonzero(moving[after])
        t_still = float(ts[after[mv[-1] + 1]]) if len(mv) and mv[-1] + 1 < len(after) else (float(ts[after[0]]) if not len(mv) else None)
    return {"robot_start_s": None if t_on is None else round(t_on - t_cross, 3),
            "robot_stop_after_home_s": None if t_still is None else round(t_still - t_home_back, 3),
            "robot_fell": bool(be.fallen()), "robot_peak": round(float(np.max(np.abs(sm))), 3)}


def run(path, exc, hold=3.0, mirror=True, params=None, t_start=0.0, with_robot=False):
    W, V, I, fps, wh = load(path)
    est = LocoEstimator(mirror=mirror, **(params or {}))
    f_px = 0.5 * wh[0] / math.tan(math.radians(est.p["hfov_deg"]) / 2)
    name, axis, amount, speed, thr = exc
    t_home = None
    z0 = None
    rows = []
    # ping-pong the clip (forward, reversed, ...) so short gesture clips cover the whole excursion
    k_first = int(t_start * fps)
    n = len(W) - k_first
    need = int((t_start + 4.0 + 2 * abs(amount) / speed + hold + 4.0) * fps)
    for j in range(max(n, need)):
        cyc, r = divmod(j, max(n - 1, 1))
        k = k_first + (r if cyc % 2 == 0 else n - 1 - r)
        t = t_start + j / fps
        obs = obs_at(W, V, I, k, fps, wh)
        if obs is not None:
            obs["t"] = t
        if obs is None:
            continue
        if t_home is None:
            est.update(obs)
            if est.home is not None:
                t_home, z0 = t, est.home[0]
            continue
        off = trajectory(t, t_home + 2.0, amount, speed, hold)
        c = est.update(augment(obs, axis, off, f_px, z0))
        rows.append((t, off, c["vx"], c["vy"], c["wz"]))
    if t_home is None:
        return None
    A = np.array(rows)
    t, off, cmd = A[:, 0], A[:, 1], A[:, 2:]
    dur = abs(amount) / speed
    t0 = t_home + 2.0
    t_cross = t0 + thr / speed
    t_back_thr = t0 + dur + hold + (abs(amount) - thr) / speed
    t_home_back = t0 + 2 * dur + hold
    if t[-1] < t_home_back + 2.0:
        return {"error": f"clip too short ({t[-1]:.1f} s) for this excursion"}
    c = cmd[:, COL[axis]]
    nz = np.flatnonzero(np.abs(c) > 0)
    on = [i for i in nz if t[i] >= t0]
    t_on = float(t[on[0]]) if on else None
    zr = np.flatnonzero((c == 0) & (t > (t_on if t_on is not None else t0 + dur)))
    t_off = float(t[zr[0]]) if len(zr) else None
    others = np.delete(cmd, COL[axis], axis=1)
    held = c[(t > t0 + dur + 0.5) & (t < t0 + dur + hold)]
    exp_sign = 1.0 if amount > 0 else -1.0
    if axis in ("side", "yaw") and mirror:
        exp_sign = -exp_sign
    robot = robot_response(t, cmd, axis, t_cross, t_home_back, amount) if with_robot else {}
    return {
        **robot,
        "t_home": round(t_home, 2), "start_delay_s": None if t_on is None else round(t_on - t_cross, 3),
        "stop_after_inside_thr_s": None if t_off is None else round(t_off - t_back_thr, 3),
        "stop_after_home_s": None if t_off is None else round(t_off - t_home_back, 3),
        "sign_ok": bool(len(held) and np.all(np.sign(held) == exp_sign)),
        "held_cmd_mean": round(float(held.mean()), 3) if len(held) else None,
        "false_before": int(np.sum(np.any(cmd[t < t0] != 0, axis=1))),
        "false_after": int(np.sum(np.any(cmd[t > (t_off or 1e9) + 0.05] != 0, axis=1))),
        "cross_axis_frames": int(np.sum(np.any(others != 0, axis=1))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", help="name=path[@t0-...]")
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--robot", action="store_true", help="also drive the split robot (estimator -> limiter -> sim)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    params = {}
    for kv in a.param:
        k, v = kv.split("=")
        params[k] = type(LocoEstimator.DEFAULTS[k])(float(v)) if not isinstance(LocoEstimator.DEFAULTS[k], str) else v
    res = {"params": dict(LocoEstimator.DEFAULTS, **params), "clips": {}}
    agg = {"start": [], "stop_home": [], "stop_thr": [], "fail": 0, "n": 0}
    for spec in a.clips:
        name, path, t0, _, _ = parse_clip(spec)
        res["clips"][name] = {}
        for exc in EXCURSIONS:
            r = run(path, exc, a.hold, not a.no_mirror, params, t_start=t0, with_robot=a.robot)
            res["clips"][name][exc[0]] = r
            if r is None or "error" in r:
                print(f"{name:20s} {exc[0]:11s} {r}")
                continue
            agg["n"] += 1
            ok = (r["start_delay_s"] is not None and r["start_delay_s"] <= 0.5 and r["stop_after_home_s"] is not None
                  and r["stop_after_home_s"] <= 1.0 and r["sign_ok"] and r["false_before"] == 0 and r["false_after"] == 0
                  and r["cross_axis_frames"] == 0)
            agg["fail"] += 0 if ok else 1
            for k, kk in (("start", "start_delay_s"), ("stop_home", "stop_after_home_s"), ("stop_thr", "stop_after_inside_thr_s")):
                if r[kk] is not None:
                    agg[k].append(r[kk])
            print(f"{name:20s} {exc[0]:11s} start {r['start_delay_s']} stop(home) {r['stop_after_home_s']} "
                  f"stop(<thr) {r['stop_after_inside_thr_s']} sign {r['sign_ok']} cmd {r['held_cmd_mean']} "
                  f"false {r['false_before']}/{r['false_after']} cross {r['cross_axis_frames']} {'OK' if ok else 'FAIL'}"
                  + (f" | robot start {r['robot_start_s']} stop(home) {r['robot_stop_after_home_s']} peak "
                     f"{r['robot_peak']} fell {r['robot_fell']}" if a.robot else ""),
                  flush=True)
    summ = {"runs": agg["n"], "failed": agg["fail"]}
    if a.robot:
        rs = [r for c in res["clips"].values() for r in c.values() if r and "robot_start_s" in r]
        for key in ("robot_start_s", "robot_stop_after_home_s"):
            v = [r[key] for r in rs if r[key] is not None]
            summ[key] = {"mean": round(float(np.mean(v)), 3), "max": round(float(np.max(v)), 3), "missing": len(rs) - len(v)} if v else None
        summ["robot_falls"] = sum(r["robot_fell"] for r in rs)
    for k in ("start", "stop_home", "stop_thr"):
        if agg[k]:
            summ[k] = {"mean": round(float(np.mean(agg[k])), 3), "max": round(float(np.max(agg[k])), 3)}
    res["summary"] = summ
    print(json.dumps(summ))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
