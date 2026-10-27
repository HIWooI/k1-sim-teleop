"""Hybrid backend evaluation (MuJoCo only; CPU heavy, e.g. 4 cores / 8 GB).

Backends (one spec each, options after ':' as key=value,key=value):
  split         k1live.backends.split (student arms + ROBOTIS loco legs; the event default until now)
  sonic         k1live.backends.sonic with K1 SONIC walk3 5000 (staged MJCF physics, the SONIC evaluation physics)
  sonic_split   k1live.backends.hybrid lock=walk: SONIC + planner only, on the split physics (vendor k1.xml)
  hybrid        k1live.backends.hybrid (legs=switch: loco while standing, SONIC while walking)
  hybrid_b      k1live.backends.hybrid legs=sonic (alternative B: SONIC legs always, student arms always)
e.g. "hybrid:walk_arms=student,blend_s=0.3".

Subcommands
  walk     the 10 walking scenarios (sonic_k1_train/mac/eval_walk.py SCENARIOS: 30 s command scripts with arm
           overlays). The overlay is "the person's arms": split follows it all the time, hybrid follows it while
           standing (planner arm swing while walking), sonic / sonic_split lay it over the planner (as eval_walk.py).
           Metrics: eval_walk.py's (speed ratios per direction, foot slip, tilt) + gait quality on steady walking segments
           (step length, cadence, swing clearance, knee bend, stance width, torso yaw wobble, crab angle, pelvis
           bounce, tilt), see gait_metrics().
  switch   random walk / stop commands (mixed directions, 0.5-4 s walks, 0.3-4 s pauses incl. stop-and-go) over a
           person arm stream; counts mode transitions, falls, max tilt and arm jumps around transitions.
           --schedule short: 0.3-0.8 s walking intents (GEM-X presenter), --arm-jumps: arm goal glitches around walks.
  stand    person arm streams standing (k1live.human.HumanMapper on the clip's MediaPipe landmarks, as
           scripts/eval_backends.py, or --gemx-root: the GEM-X real-time path): arm joint error and hand / elbow
           position error vs split.

  .venv/bin/python scripts/eval_hybrid.py walk --backends split sonic sonic_split hybrid hybrid_b --json logs/h/walk.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "sonic_k1_train" / "mac"))
sys.path.insert(0, str(ROOT / "sonic_k1_train" / "data"))

DT = 0.02
FALL_Z, FALL_TILT = 0.45, 45.0
WALK3 = ROOT / "models" / "k1" / "k1s_walk3" / "deploy" / "k1s_walk3_005000"


# ------------------------------------------------------------------------------------------------ backends
def safe(spec):
    """Backend spec -> file-name part (options may hold paths, e.g. hybrid:variant_dir=models/k1/<run>/deploy/<step>)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec).strip("_")


def parse_spec(spec):
    name, _, opts = spec.partition(":")
    kw = {}
    for kv in filter(None, opts.split(",")):
        k, v = kv.split("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kw[k] = v
    return name, kw


def make(spec, planner_async=False, latency=2):
    """Backend from a spec. Evaluation default: planner synchronous with the result applied `latency` ticks later
    (deterministic, like eval_walk.py; the live server runs it asynchronously)."""
    name, kw = parse_spec(spec)
    if name == "split":
        from k1live.backends.split import SplitBackend

        return SplitBackend(**kw)
    if name == "sonic":
        from k1live.backends.sonic import SonicBackend

        be = SonicBackend(variant_dir=kw.pop("variant_dir", WALK3), planner_async=planner_async, **kw)
    else:
        from k1live.backends.hybrid import HybridBackend

        base = {"sonic_split": {"lock": "walk", "min_step_v": 0.0, "turn_step_v": 0.0}, "hybrid": {},
                "hybrid_b": {"legs": "sonic"}}[name]
        be = HybridBackend(planner_async=planner_async, **{**base, **kw})
    if not planner_async:
        be.kp.stream.latency_ticks = latency
    return be


# ------------------------------------------------------------------------------------------------ logging
class Probe:
    """Per-tick robot quantities for the gait metrics (works on both MJCFs: same body / joint names)."""

    def __init__(self, be):
        import mujoco

        self.mj, self.be = mujoco, be
        m = be.model
        self.feet = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        self.torso = m.body("torso_link").id
        self.floor = m.geom("floor").id
        names = be.names
        self.knee = [names.index(f"{s}_knee_joint") for s in ("left", "right")]
        self.knee_body = [m.body(f"{s}_knee_link").id for s in ("left", "right")]
        self.v = np.zeros(6)
        self.waist = names.index("waist_yaw_joint")
        self.rec = {k: [] for k in ("t", "cmd", "pos", "yaw", "tyaw", "tilt", "ttilt", "waist", "z", "foot", "con", "fspd", "knee",
                                    "kpos", "arm_q", "arm_t", "phase", "wL", "wA", "ms", "tick_ms")}

    def contacts(self):
        d, m = self.be.data, self.be.model
        on = [False, False]
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 != self.floor and c.geom2 != self.floor:
                continue
            b = m.geom_bodyid[c.geom2 if c.geom1 == self.floor else c.geom1]
            for k, fb in enumerate(self.feet):
                if b == fb:
                    on[k] = True
        return on

    def log(self, t, cmd, arm_t, info):
        be, d, r = self.be, self.be.data, self.rec
        R = d.xmat[be.pelvis_body].reshape(3, 3)
        Rt = d.xmat[self.torso].reshape(3, 3)
        sp = []
        for b in self.feet:
            self.mj.mj_objectVelocity(be.model, d, self.mj.mjtObj.mjOBJ_BODY, b, self.v, 0)
            sp.append(float(np.hypot(self.v[3], self.v[4])))
        q = be.q_now()
        r["t"].append(t)
        r["cmd"].append(np.asarray(cmd, float))
        r["pos"].append(d.xpos[be.pelvis_body].copy())
        r["yaw"].append(float(np.arctan2(R[1, 0], R[0, 0])))
        r["tyaw"].append(float(np.arctan2(Rt[1, 0], Rt[0, 0])))
        r["tilt"].append(be.tilt_deg())
        r["ttilt"].append(float(np.degrees(np.arccos(np.clip(Rt[2, 2], -1.0, 1.0)))))
        r["waist"].append(float(q[self.waist]))
        r["z"].append(be.pelvis_height())
        r["foot"].append(d.xpos[self.feet].copy())
        r["con"].append(self.contacts())
        r["fspd"].append(sp)
        r["knee"].append(q[self.knee])
        r["kpos"].append(d.xpos[self.knee_body].copy())
        r["arm_q"].append(be.arm_q_now())
        r["arm_t"].append(np.asarray(arm_t, float))
        r["phase"].append(info.get("phase", "-"))
        r["wL"].append(info.get("wL", np.nan))
        r["wA"].append(info.get("wA", np.nan))
        r["ms"].append(info.get("policy_ms", 0.0))
        r["tick_ms"].append(info.get("tick_ms", np.nan))

    def arrays(self):
        return {k: (np.asarray(v) if k != "phase" else np.array(v)) for k, v in self.rec.items()}


# ------------------------------------------------------------------------------------------------ metrics
def body_vel(ts, p, yaw, k=12):
    yw = np.unwrap(yaw)
    v = (p[2 * k:, :2] - p[:-2 * k, :2]) / (ts[2 * k:] - ts[:-2 * k])[:, None]
    wz = (yw[2 * k:] - yw[:-2 * k]) / (ts[2 * k:] - ts[:-2 * k])
    y = yw[k:-k]
    vb = np.stack([np.cos(y) * v[:, 0] + np.sin(y) * v[:, 1], -np.sin(y) * v[:, 0] + np.cos(y) * v[:, 1]], 1)
    return vb, wz


def moving_avg(x, n):
    if len(x) < n:
        return np.full_like(x, x.mean())
    c = np.convolve(x, np.ones(n) / n, mode="same")
    h = n // 2
    c[:h], c[-h:] = c[h], c[-h - 1]
    return c


def gait_metrics(A, sel):
    """Gait quality on the ticks `sel` (steady walking). Units: m, m/s, deg, steps/s.

    step_len      distance between the feet along the travel direction at each touchdown (m)
    cadence       touchdowns per second (both feet)
    clearance     peak foot lift per swing (ankle origin z above its value at the touchdown, m)
    knee_width    lateral distance between the knees (m; bow-legged look when > feet width)
    sway_cm       pelvis side-to-side motion minus its 1 s moving average (cm, RMS)
    knee_deg      mean knee flexion (both legs), knee_stance_deg while that foot is on the ground
    width         lateral foot distance in the pelvis-heading frame while both feet are on the ground (m)
    slip          horizontal speed of a foot touching the floor (m/s, mean of contact ticks)
    yaw_wobble    torso yaw minus its 1 s moving average (deg, RMS): the visible twisting per step
    crab_deg      |direction of travel - commanded direction| in the body frame (translation segments)
    bounce_cm     pelvis height minus its 1 s moving average (cm, RMS)
    tilt          pelvis tilt mean / p95 (deg)"""
    idx = np.flatnonzero(sel)
    if len(idx) < 50:
        return {}
    t, foot, con, yaw = A["t"], A["foot"], A["con"].astype(bool), A["yaw"]
    out = {}
    # touchdowns (rising edges, contact off >= 2 ticks before)
    steps, lift = [], []
    for k in (0, 1):
        c = con[:, k]
        for i in idx:
            if i >= 3 and c[i] and not c[i - 1] and not c[i - 2]:
                j = 1 - k
                d = foot[i, k, :2] - foot[i, j, :2]
                v = A["pos"][min(i + 10, len(t) - 1), :2] - A["pos"][max(i - 10, 0), :2]
                nv = np.linalg.norm(v)
                steps.append(abs(d @ (v / nv)) if nv > 1e-3 else np.nan)
                # swing: since the last time this foot left the ground
                i0 = i - 1
                while i0 > 0 and not c[i0]:
                    i0 -= 1
                if i - i0 > 2:
                    lift.append(float(foot[i0:i, k, 2].max() - foot[i, k, 2]))
    dur = len(idx) * DT
    steps = np.asarray(steps, float)
    out["cadence"] = round(len(steps) / dur, 2)
    if np.isfinite(steps).any():
        out["step_len"] = round(float(np.nanmean(steps)), 3)
    if lift:
        out["clearance"] = round(float(np.mean(lift)), 3)
    kn = np.degrees(A["knee"][idx])
    out["knee_deg"] = round(float(kn.mean()), 1)
    st = con[idx]
    if st.any():
        out["knee_stance_deg"] = round(float(kn[st].mean()), 1)
    both = idx[con[idx].all(1)]
    if len(both):
        y = yaw[both]
        dlt = foot[both, 0, :2] - foot[both, 1, :2]
        lat = -np.sin(y) * dlt[:, 0] + np.cos(y) * dlt[:, 1]
        out["width"] = round(float(np.abs(lat).mean()), 3)
    y = yaw[idx]
    dk = A["kpos"][idx, 0, :2] - A["kpos"][idx, 1, :2]
    out["knee_width"] = round(float(np.abs(-np.sin(y) * dk[:, 0] + np.cos(y) * dk[:, 1]).mean()), 3)
    # lateral sway: pelvis position across the travel direction minus its 1 s moving average (cm, RMS)
    p = A["pos"][:, :2]
    pm = np.stack([moving_avg(p[:, 0], 50), moving_avg(p[:, 1], 50)], 1)
    dp = (p - pm)[idx]
    out["sway_cm"] = round(float(np.sqrt(np.mean((-np.sin(y) * dp[:, 0] + np.cos(y) * dp[:, 1]) ** 2)) * 100), 2)
    sp = A["fspd"][idx][st]
    if sp.size:
        out["slip"] = round(float(sp.mean()), 3)
    ty = np.unwrap(A["tyaw"])
    wob = ty - moving_avg(ty, 50)
    out["yaw_wobble_deg"] = round(float(np.degrees(np.sqrt(np.mean(wob[idx] ** 2)))), 2)
    z = A["z"]
    bz = z - moving_avg(z, 50)
    out["bounce_cm"] = round(float(np.sqrt(np.mean(bz[idx] ** 2)) * 100), 2)
    out["z_mean"] = round(float(z[idx].mean()), 3)
    out["tilt_mean"] = round(float(A["tilt"][idx].mean()), 1)
    if "ttilt" in A and len(A["ttilt"]):
        out["torso_tilt_mean"] = round(float(A["ttilt"][idx].mean()), 1)
        out["waist_abs_deg"] = round(float(np.degrees(np.abs(A["waist"][idx])).mean()), 1)
    out["tilt_p95"] = round(float(np.percentile(A["tilt"][idx], 95)), 1)
    return out


def steady(segs, ts, settle=1.5):
    out = np.zeros(len(ts), bool)
    for a, b, c in segs:
        if np.linalg.norm(c) > 0:
            out |= (ts >= a + settle) & (ts < b)
    return out


def walk_summary(A, segs, fell_at):
    from eval_walk_table import direction

    ts = A["t"]
    out = {"fell": fell_at is not None, "fell_at": fell_at, "max_tilt_deg": round(float(A["tilt"].max()), 1),
           "min_pelvis_z": round(float(A["z"].min()), 3)}
    k = 12
    if len(ts) <= 2 * k + 2:
        return out
    vb, wz = body_vel(ts, A["pos"], A["yaw"], k)
    tc = ts[k:-k]
    segm = []
    for a, b, c in segs:
        sel = (tc >= a + 1.5) & (tc < b)
        if sel.sum() < 5:
            continue
        sg = {"cmd": list(c), "dir": direction(c), "vx": round(float(vb[sel, 0].mean()), 3),
              "vy": round(float(vb[sel, 1].mean()), 3), "wz": round(float(wz[sel].mean()), 3)}
        c2 = np.asarray(c[:2], float)
        if np.linalg.norm(c2) > 0.05:   # direction of the mean body-frame velocity vs the command
            ang = np.arctan2(vb[sel, 1].mean(), vb[sel, 0].mean()) - np.arctan2(c2[1], c2[0])
            sg["crab_deg"] = round(float(np.degrees(abs((ang + np.pi) % (2 * np.pi) - np.pi))), 1)
            if abs(c[2]) < 1e-6:        # straight walk: torso heading vs where it should face for this travel
                ii = np.flatnonzero(sel) + k
                dp = A["pos"][ii[-1], :2] - A["pos"][ii[0], :2]
                face = np.arctan2(dp[1], dp[0]) - np.arctan2(c2[1], c2[0])
                e = np.unwrap(A["tyaw"])[ii] - face
                e = (e + np.pi) % (2 * np.pi) - np.pi
                sg["torso_vs_travel_deg"] = round(float(np.degrees(np.abs(e)).mean()), 1)
        full = np.zeros(len(ts), bool)
        full[k:-k] = sel
        sg["gait"] = gait_metrics(A, full)
        # response (causal 0.25 s window): command on after a pause -> robot moving (> 0.1 m/s or > 0.2 rad/s);
        # command off followed by a pause -> robot still (< 0.05 m/s and < 0.1 rad/s for 0.5 s)
        w = int(0.25 / DT)
        dp = np.linalg.norm(A["pos"][w:, :2] - A["pos"][:-w, :2], axis=1) / (w * DT)
        dyw = np.abs(np.diff(np.unwrap(A["yaw"]))[:, None][:, 0])
        yr = np.convolve(np.r_[0.0, dyw], np.ones(w), "full")[:len(ts)][w:] / (w * DT)
        tt = ts[w:]
        moving = (dp > 0.1) | (yr > 0.2)
        still = (dp < 0.05) & (yr < 0.1)
        still_hold = np.convolve(still.astype(float), np.ones(int(0.5 / DT)), "full")[:len(still)] >= int(0.5 / DT)
        prev_end = max([e for s0, e, _ in segs if e <= a + 1e-9], default=-9.0)
        next_start = min([s0 for s0, e, _ in segs if s0 >= b - 1e-9], default=99.0)
        if a - prev_end >= 1.0:
            on = np.flatnonzero((tt >= a) & moving)
            sg["start_s"] = round(float(tt[on[0]] - a), 2) if len(on) else None
        if next_start - b >= 2.0:
            off = np.flatnonzero((tt >= b) & still_hold)
            sg["stop_s"] = round(float(tt[off[0]] - 0.5 - b), 2) if len(off) else None
        segm.append(sg)
    out["segments"] = segm
    walk = steady(segs, ts)
    out["gait"] = gait_metrics(A, walk)
    stand = np.ones(len(ts), bool)
    for a, b, c in segs:
        stand &= ~((ts >= a) & (ts < b + 2.0))
    stand &= ts > 0.5
    if stand.sum() > 50:
        out["stand_tilt_mean"] = round(float(A["tilt"][stand].mean()), 1)
        out["stand_arm_deg"] = round(float(np.degrees(np.abs(A["arm_q"][stand] - A["arm_t"][stand])).mean()), 2)
    out["policy_ms"] = round(float(np.mean(A["ms"])), 3)
    return out


# ------------------------------------------------------------------------------------------------ walk scenarios
def run_walk(cfg):
    from eval_walk import ALL_SCENARIOS, Arms, cmd_at

    spec, scen = cfg["backend"], cfg["scenario"]
    segs, arms_spec = ALL_SCENARIOS[scen]
    name, kw = parse_spec(spec)
    if name == "sonic_split":
        spec = spec + ("," if ":" in spec else ":") + f"walk_arm_human={1.0 if arms_spec else 0.0}"
    be = make(spec)
    arms = Arms(arms_spec, cfg.get("seed", 1234))
    pr = Probe(be)
    fell = None
    base = be.arm_default.copy()
    chain = None
    if cfg.get("server_chain"):   # the live server's walking chain: limiter preset of the backend + heading hold
        from k1live.locomotion import VelocityLimiter

        chain = {"lim": VelocityLimiter.for_backend(getattr(be, "limiter_preset", "split")), "yaw_ref": None}
    t0w = time.perf_counter()
    for i in range(int((30.0 + 1.0) / DT)):
        ts = i * DT - 1.0
        c = cmd_at(segs, ts)
        walk_cmd = None
        if chain is not None:
            raw = c
            c = np.asarray(chain["lim"]({"vx": raw[0], "vy": raw[1], "wz": raw[2]}, DT), float)
            walk_cmd = c.copy()
            yaw = be.root_pose()[2]
            if c[2] != 0.0 or chain["yaw_ref"] is None:
                chain["yaw_ref"] = yaw
            else:
                e = (chain["yaw_ref"] - yaw + np.pi) % (2 * np.pi) - np.pi
                c[2] = float(np.clip(2.0 * e, -0.4, 0.4))
        if arms.track is not None:
            a = arms.apply(np.tile(base, (2, 1)), ts, np.arange(10))
            arm, nxt = a[0], arms.apply(base[None], ts + DT, np.arange(10))[0]
            arm_dq = (nxt - arm) / DT
        elif name in ("sonic",) and be.last_window is not None:
            arm = be.last_window.q[0, be.arm_idx].copy()        # planner arms (one tick late)
            arm_dq = None
        else:
            arm, arm_dq = base, np.zeros(10)
        kw = {"walk_cmd": walk_cmd} if (walk_cmd is not None and be.supports.get("intent")) else {}
        info = be.step(arm, loco_cmd=c, arm_dq=arm_dq, **kw)
        if chain is not None and info.get("phase") not in (None, "stand"):   # as k1live.sim_server
            chain["yaw_ref"] = be.root_pose()[2]
        if ts >= 0:
            pr.log(ts, cmd_at(segs, ts), arm, info)
            if fell is None and be.fallen():
                fell = round(ts, 2)
                break
    be.close()
    stats = dict(be.stats) if hasattr(be, "stats") else None
    pr.be = None
    del be
    gc.collect()
    A = pr.arrays()
    out = {"backend": cfg["backend"], "scenario": scen, "arms": arms_spec, **walk_summary(A, segs, fell),
           "wall_s": round(time.perf_counter() - t0w, 1)}
    if stats is not None:
        out["stats"] = stats
    if cfg.get("trace"):
        np.savez_compressed(cfg["trace"], **{k: v for k, v in A.items()})
    return out


def direction_table(runs):
    """{direction: speed ratio} over the scenarios of one backend (eval_walk_table.direction_ratios)."""
    from eval_walk_table import direction_ratios

    return {k: (round(v, 2) if v is not None else None)
            for k, v in direction_ratios({"scenarios": {r["scenario"]: r for r in runs}}).items()}


def gait_table(runs, dirs=("fwd", "back", "side", "diag", "walk+turn")):
    """Gait metrics averaged over all steady translation segments of one backend (segment-duration weighted)."""
    acc = {}
    for r in runs:
        for sg in r.get("segments", []):
            if sg["dir"] not in dirs or not sg.get("gait"):
                continue
            for k, v in sg["gait"].items():
                acc.setdefault(k, []).append(v)
            for key in ("crab_deg", "torso_vs_travel_deg", "start_s", "stop_s"):
                if sg.get(key) is not None:
                    acc.setdefault(key, []).append(sg[key])
    return {k: round(float(np.mean(v)), 3) for k, v in acc.items()}


# ------------------------------------------------------------------------------------------------ person arms
def person_stream(video, mirror=True, max_seconds=0.0, arm_ik="shape18", gemx_root=None):
    """scripts/eval_backends.person_stream with the sender's arm IK choice (k1live.human arm_ik).
    gemx_root: GEM-X instead of MediaPipe (scripts/gemx_feed.py: the clip's every-frame real-time-worker dump with
    live timing through k1live.sender_gemx's JumpGate + HumanMapper(perception="gemx"); one entry per returned frame)."""
    import cv2
    import run_video as g1rv

    from k1live.human import HumanMapper

    if gemx_root:
        import gemx_feed as GF

        rows = GF.load_rows(GF.find_dump(gemx_root, Path(video).stem)).retime()
        t_end = rows.seconds if max_seconds <= 0 else min(rows.seconds, max_seconds)
        fr = GF.sender_frames(rows, t1=t_end, mirror=mirror, loco=False, smpl=False, arm_ik=arm_ik)
        return [(f["t_capture"], f) for f in fr], rows.fps, t_end
    W, V, I, fps = g1rv.extract(video, "full", None, 1, 1280)
    cap = cv2.VideoCapture(str(video))
    wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    hm = HumanMapper(mirror=mirror, loco=False, arm_ik=arm_ik)
    out = []
    n = len(W) if max_seconds <= 0 else min(len(W), int(max_seconds * fps))
    for k in range(n):
        det = None if np.isnan(W[k, 0, 0]) else {"world": W[k], "vis": V[k], "image": I[k]}
        f = hm.process(det, k / fps, *wh)
        f.pop("obs", None)
        out.append((k / fps, f))
    return out, fps, n / fps


class ArmFeed:
    """A person's arm stream at 50 Hz the way k1live.sim_server feeds a backend: the newest camera frame's IK arms
    (zero-order hold between frames) through the server's ArmLimiter (6 rad/s, 80 rad/s^2); no person -> hold 1 s,
    then back to the rest pose at 1 rad/s. The clip loops."""

    def __init__(self, video, default, max_seconds=0.0, arm_ik="shape18", vmax=6.0, amax=80.0, gemx_root=None):
        from k1live.sim_server import ArmLimiter

        stream, self.fps, self.period = person_stream(video, max_seconds=max_seconds, arm_ik=arm_ik,
                                                      gemx_root=gemx_root)
        self.vmax, self.amax = vmax, amax
        self.ts = np.array([s[0] for s in stream])
        self.arms = [None if s[1].get("arm_q") is None else np.asarray(s[1]["arm_q"], float) for s in stream]
        self.default = np.asarray(default, float)
        self.lim = ArmLimiter(self.default)
        self.goal = self.default.copy()
        self.t_seen = -1e9

    @property
    def seconds(self):
        return self.period

    def __call__(self, t, jump=None):
        """jump: another clip time (s) whose arm pose replaces the goal at this tick (a perception glitch)."""
        k = int(np.searchsorted(self.ts, t % self.period, side="right")) - 1
        if jump is not None:
            k = int(np.searchsorted(self.ts, jump % self.period, side="right")) - 1
        a = self.arms[k] if k >= 0 else None
        vmax, amax = self.vmax, self.amax
        if a is not None:
            self.goal, self.t_seen = a, t
        elif t - self.t_seen > 1.0:
            self.goal, vmax, amax = self.default, 1.0, 80.0
        q, dq = self.lim.step(self.goal, vmax=vmax, amax=amax)
        return q, dq, a is not None


def arm_pos(be):
    from eval_walk import ArmPos

    return ArmPos(be.model, be.names)


# ------------------------------------------------------------------------------------------------ standing
def run_stand(cfg):
    """One person clip standing (no walking command): arm error vs the server's reference (joint deg, hand + elbow
    cm), falls, tilt. hybrid (legs=switch) stands on the split controllers, so it must equal split."""
    be = make(cfg["backend"])
    feed = ArmFeed(cfg["video"], be.arm_default, cfg.get("max_seconds", 0.0), **cfg.get("arm_feed", {}))
    ap_ = arm_pos(be)
    T = feed.seconds + 1.0
    err, cm, tilt, ms, pres = [], [], [], [], []
    fell = None
    for i in range(int(T / DT)):
        t = i * DT
        q, dq, present = feed(t)
        info = be.step(q, loco_cmd=(0.0, 0.0, 0.0), arm_dq=dq)
        if present:
            err.append(np.degrees(np.abs(be.arm_q_now() - q)).mean())
            full = be.q_now()
            full[be.arm_idx] = q
            if i % 2 == 0:
                cm.append(ap_.err_cm(be.data, full))
        tilt.append(be.tilt_deg())
        ms.append(info.get("policy_ms", 0.0))
        if fell is None and be.fallen():
            fell = round(t, 2)
            break
    be.close()
    del be
    gc.collect()
    return {"clip": cfg["name"], "backend": cfg["backend"], "fell": fell is not None, "fell_at": fell,
            "seconds": round(T, 1), "arm_deg": round(float(np.mean(err)), 2) if err else None,
            "arm_p90_deg": round(float(np.percentile(err, 90)), 2) if err else None,
            "hand_elbow_cm": round(float(np.mean(cm)), 2) if cm else None, "max_tilt_deg": round(float(max(tilt)), 1),
            "policy_ms": round(float(np.mean(ms)), 3)}


# ------------------------------------------------------------------------------------------------ transitions
DIRS6 = ("fwd", "back", "left", "right", "turn", "fwd_turn", "diag")


def command_schedule(rng, T, t0=1.0):
    """[(t_start, t_end, (vx, vy, wz))] random walks (0.5-4 s) separated by pauses (0.3-4 s; a third of them
    < 1 s = stop-and-go while the robot is still settling)."""
    segs, t = [], t0
    while t < T - 1.0:
        pause = rng.uniform(0.3, 1.0) if rng.random() < 0.33 else rng.uniform(1.0, 4.0)
        t += pause
        dur = rng.uniform(0.5, 4.0)
        d = DIRS6[rng.integers(len(DIRS6))]
        v = rng.uniform(0.15, 0.6)
        c = {"fwd": (v, 0, 0), "back": (-min(v, 0.3), 0, 0), "left": (0, min(v, 0.3), 0),
             "right": (0, -min(v, 0.3), 0), "turn": (0, 0, rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)),
             "fwd_turn": (v, 0, rng.choice([-1, 1]) * rng.uniform(0.2, 0.4)),
             "diag": (min(v, 0.4), rng.choice([-1, 1]) * 0.2, 0)}[d]
        segs.append((t, min(t + dur, T - 0.5), tuple(float(x) for x in c)))
        t += dur
    return segs


def short_schedule(rng, T, t0=1.0):
    """GEM-X presenter walks (as seen on the event PC): short walking intents of 0.3-0.8 s (forward / backward 0.2-0.3 m/s,
    sometimes sideways, and "edge" = the tiny sideways remainder 0.035-0.08 m/s that the server's 3 m leash leaves of
    an outward step), 1-4 s apart, so most walks end while the first step is still in the air."""
    segs, t = [], t0
    while True:
        t += rng.uniform(1.0, 4.0)
        if t >= T - 1.0:
            break
        dur = rng.uniform(0.3, 0.8)
        v = rng.uniform(0.2, 0.3)
        e = rng.choice([-1, 1]) * rng.uniform(0.035, 0.08)
        d = rng.choice(["fwd", "back", "fwd", "back", "left", "right", "edge"])
        c = {"fwd": (v, 0, 0), "back": (-v, 0, 0), "left": (0, v, 0), "right": (0, -v, 0),
             "edge": (rng.uniform(0.0, 0.02), e, 0)}[d]
        segs.append((t, min(t + dur, T - 0.5), tuple(float(x) for x in c)))
        t += dur
    return segs


def run_switch(cfg):
    """Random walk / stop commands (through the server's limiter + heading hold) over a looping person arm stream.
    A fall is logged, the robot respawned, and the run goes on."""
    from eval_walk import cmd_at

    from k1live.locomotion import VelocityLimiter

    be = make(cfg["backend"], latency=cfg.get("latency", 2))
    rng = np.random.default_rng(cfg["seed"])
    T = cfg["seconds"]
    segs = (short_schedule if cfg.get("schedule") == "short" else command_schedule)(rng, T)
    feed = ArmFeed(cfg["video"], be.arm_default, **cfg.get("arm_feed", {}))
    jumps = []   # (t0, t1, clip time): arm goal glitches during / right after walks (--arm-jumps)
    if cfg.get("arm_jumps"):
        for a, b, _ in segs:
            if b > a and rng.random() < 0.7:
                j0 = rng.uniform(a, b + 0.6)
                jumps.append((j0, j0 + rng.uniform(0.1, 0.4), rng.uniform(0, feed.seconds)))
    lim = VelocityLimiter.for_backend(getattr(be, "limiter_preset", "split"))
    yaw_ref = None
    rec = {k: [] for k in ("t", "phase", "wL", "wA", "tilt", "ttilt", "arm_dq", "tgt", "tick_ms", "cmd")}
    torso = be.model.body("torso_link").id
    falls, fall_t = 0, []
    for i in range(int(T / DT)):
        t = i * DT
        target = cmd_at(segs, t)
        cmd = np.asarray(lim({"vx": target[0], "vy": target[1], "wz": target[2]}, DT), float)
        walk_cmd = cmd.copy()
        yaw = be.root_pose()[2]
        if cmd[2] != 0.0 or yaw_ref is None:
            yaw_ref = yaw
        else:
            err = (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
            cmd[2] = float(np.clip(2.0 * err, -0.4, 0.4))
        q, dq, _ = feed(t, next((c for a, b, c in jumps if a <= t < b), None))
        t0 = time.perf_counter()
        info = be.step(q, loco_cmd=cmd, arm_dq=dq, walk_cmd=walk_cmd)
        tick = (time.perf_counter() - t0) * 1e3
        if info.get("phase") not in (None, "stand"):   # as k1live.sim_server
            yaw_ref = be.root_pose()[2]
        Rt = be.data.xmat[torso].reshape(3, 3)
        rec["t"].append(t)
        rec["phase"].append(info["phase"])
        rec["wL"].append(info["wL"])
        rec["wA"].append(info["wA"])
        rec["tilt"].append(be.tilt_deg())
        rec["ttilt"].append(float(np.degrees(np.arccos(np.clip(Rt[2, 2], -1, 1)))))
        rec["arm_dq"].append(float(np.abs(be.data.qvel[be.sim.vadr[be.arm_idx]]).max()))
        rec["tgt"].append(info["target"].copy())
        rec["tick_ms"].append(tick)
        rec["cmd"].append(walk_cmd)
        if be.fallen():
            falls += 1
            fall_t.append(round(t, 2))
            be.reset(*be.root_pose()[:2])
            lim.reset()
            yaw_ref = None
    stats = dict(be.stats)
    be.close()
    del be
    gc.collect()      # SonicController's observation registry is a reference cycle: free the ONNX sessions now
    A = {k: np.asarray(v) for k, v in rec.items()}
    ph = A["phase"]
    # transitions: legs loco -> SONIC (stand -> walk) and back (settle -> stand), plus settle -> walk re-entries
    ev = [(i, f"{ph[i - 1]}->{ph[i]}") for i in range(1, len(ph)) if ph[i] != ph[i - 1]]
    kinds = {}
    for _, k in ev:
        kinds[k] = kinds.get(k, 0) + 1
    tg = A["tgt"]
    jump = np.r_[0.0, np.abs(np.diff(tg, axis=0)).max(1)]
    win = np.zeros(len(ph), bool)
    for i, _ in ev:
        win[i:i + int(1.5 / DT)] = True
    steady = ~win
    walking = np.array([p == "walk" for p in ph]) & steady
    standing = np.array([p == "stand" for p in ph]) & steady
    f = lambda x, m: round(float(x[m].max()), 2) if m.any() else None  # noqa: E731
    out = {"backend": cfg["backend"], "seed": cfg["seed"], "clip": cfg["name"], "seconds": T, "falls": falls,
           "schedule": cfg.get("schedule", "mixed"), "arm_jumps": len(jumps),
           "fall_t": fall_t, "transitions": len(ev), "kinds": kinds, "stats": stats,
           "walk_s": round(float(np.mean([p == "walk" for p in ph]) * T), 1),
           "tilt_max": {"transition": f(A["tilt"], win), "walking": f(A["tilt"], walking), "standing": f(A["tilt"], standing)},
           "torso_tilt_max": {"transition": f(A["ttilt"], win), "walking": f(A["ttilt"], walking),
                              "standing": f(A["ttilt"], standing)},
           "arm_speed_max": {"transition": f(A["arm_dq"], win), "walking": f(A["arm_dq"], walking),
                             "standing": f(A["arm_dq"], standing)},
           "target_jump_max": {"transition": f(jump, win), "walking": f(jump, walking), "standing": f(jump, standing)},
           "tick_ms": {"mean": round(float(A["tick_ms"].mean()), 2), "p99": round(float(np.percentile(A["tick_ms"], 99)), 2)}}
    if cfg.get("trace"):
        np.savez_compressed(cfg["trace"], **{k: v for k, v in A.items()}, segs=np.array([(a, b, *c) for a, b, c in segs]))
    return out


# ------------------------------------------------------------------------------------------------ video
# operator walking pattern for the comparison video (t0, t1, (vx, vy, wz)); arms = a person clip throughout
DRIVE = [(3, 9, (0.4, 0, 0)), (12, 17, (0, 0, 0.5)), (20, 25, (0, 0.25, 0)), (28, 32, (-0.25, 0, 0)),
         (35, 42, (0.45, 0, -0.3)), (45, 51, (0.5, 0, 0))]
PHASE_KO = {"stand": "서기 · split 팔·다리", "walk": "걷기 · K1 SONIC", "settle": "멈추는 중 · SONIC"}


def cmd_text(v):
    vx, vy, wz = v
    parts = []
    if abs(vx) > 0.02:
        parts.append(f"{'전진' if vx > 0 else '후진'} {abs(vx):.2f} m/s")
    if abs(vy) > 0.02:
        parts.append(f"{'왼쪽' if vy > 0 else '오른쪽'} {abs(vy):.2f} m/s")
    if abs(wz) > 0.02:
        parts.append(f"{'왼쪽' if wz > 0 else '오른쪽'} 회전 {abs(wz):.2f} rad/s")
    return "보행 명령: " + (" · ".join(parts) if parts else "정지")


class Rig:
    """One backend + the server-side walking chain (limiter preset, heading hold) + a follow camera."""

    def __init__(self, spec, w, h):
        from k1.sim import Camera
        from k1live.locomotion import VelocityLimiter

        self.be = make(spec)
        self.lim = VelocityLimiter.for_backend(getattr(self.be, "limiter_preset", "split"))
        self.cam = Camera(self.be.sim, w=w, h=h, azimuth=155.0, distance=2.9, elevation=-10.0, lookat_z=0.62)
        self.cam_yaw = 0.0          # the view turns with the robot (tau 2.5 s): same angle on the gait after turns
        self.yaw_ref = None
        self.info = {}
        self.fell = None
        self.log = {"t": [], "tilt": [], "phase": [], "xy": []}

    def step(self, t, target, q, dq):
        be = self.be
        cmd = np.asarray(self.lim({"vx": target[0], "vy": target[1], "wz": target[2]}, DT), float)
        walk_cmd = cmd.copy()
        yaw = be.root_pose()[2]
        if cmd[2] != 0.0 or self.yaw_ref is None:
            self.yaw_ref = yaw
        else:
            err = (self.yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
            cmd[2] = float(np.clip(2.0 * err, -0.4, 0.4))
        kw = {"walk_cmd": walk_cmd} if be.supports.get("intent") else {}
        self.info = be.step(q, loco_cmd=cmd, arm_dq=dq, **kw)
        self.cmd = walk_cmd
        if self.info.get("phase") not in (None, "stand"):   # as k1live.sim_server
            self.yaw_ref = be.root_pose()[2]
        self.log["t"].append(t)
        self.log["tilt"].append(be.tilt_deg())
        self.log["phase"].append(self.info.get("phase", "-"))
        self.log["xy"].append(be.root_pose()[:2])
        dy = (be.root_pose()[2] - self.cam_yaw + np.pi) % (2 * np.pi) - np.pi
        self.cam_yaw += dy * DT / 2.5
        self.cam.cam.azimuth = 155.0 + np.degrees(self.cam_yaw)
        if self.fell is None and be.fallen():
            self.fell = round(t, 2)


def render_video(a):
    """split | hybrid side by side on the same input: operator walking pattern (DRIVE) + a person's arm stream, or
    a walking person clip (--person: walking intent from k1live.locomotion.LocoEstimator, 3 panels)."""
    import imageio.v2 as imageio
    from PIL import Image

    import make_showreel as ms

    W_, H_ = (int(x) for x in a.size.split("x"))
    sc = W_ / 1920.0
    clips = clip_args(a.clips[:1], a.event_dir)
    name, path = clips[0]
    person = None
    if a.person:
        person = PersonWalk(a.person, a.mirror)
        T = person.seconds
        pw = W_ // 3
    else:
        T = a.seconds
        pw = 0
    rw = (W_ - pw) // 2
    rigs = [Rig(a.backends[0], rw, H_), Rig(a.backends[1], rw, H_)]
    feeds = [ArmFeed(path, r.be.arm_default, arm_ik=a.arm_ik, vmax=a.arm_vmax, amax=a.arm_amax) for r in rigs] \
        if person is None else None
    reader = None
    if person is None:   # picture-in-picture of the arm clip (mirrored like the live view)
        import run_video as g1rv

        reader = g1rv.FrameReader(path, height=int(220 * sc))
    f26, f30 = ms.font("semibold", int(26 * sc)), ms.font("medium", int(30 * sc))
    pad, rad = (int(20 * sc), int(12 * sc)), int(14 * sc)
    labels = [(a.label_left or "split: ROBOTIS 보행 정책 + student 팔 (지금까지의 기본)", 0),
              (a.label_right or "hybrid: 서 있을 때 split · 걸을 때 K1 SONIC 보행", 1)]
    ovs = []
    for txt, k in labels:
        ovs.append(ms.Overlay(ms.pill([(txt, f26, ms.INK)], pad=pad, radius=rad, bar=None, bg=(255, 255, 255, 215)),
                              pw + k * rw + int(28 * sc), int(30 * sc)))
    ovs.append(ms.Overlay(Image.new("RGBA", (4, H_), (255, 255, 255, 255)), pw + rw - 2, 0))
    if pw:
        ovs.append(ms.Overlay(ms.pill([("사람 (카메라 영상)", f26, ms.INK)], pad=pad, radius=rad, bar=None,
                                      bg=(255, 255, 255, 215)), int(28 * sc), int(30 * sc)))
        ovs.append(ms.Overlay(Image.new("RGBA", (4, H_), (255, 255, 255, 255)), pw - 2, 0))
    sprite = {}

    def badge(txt, f=f30):
        if txt not in sprite:
            sprite[txt] = ms.pill([(txt, f, ms.INK)], pad=pad, radius=rad, bar=None, bg=(255, 255, 255, 225))
        return sprite[txt]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1)
    every = max(1, int(round(1.0 / (a.fps * DT))))
    from eval_walk import cmd_at

    t0w = time.perf_counter()
    for i in range(int(T / DT)):
        t = i * DT
        if person is None:
            target = cmd_at(DRIVE, t)
            for r, fd in zip(rigs, feeds):
                q, dq, _ = fd(t)
                r.step(t, target, q, dq)
        else:
            target, q, dq = person(t)
            for r in rigs:
                r.step(t, target, q, dq)
        if i % every:
            continue
        panels = [r.cam.grab() for r in rigs]
        img = np.concatenate(([person.frame(t, pw, H_)] if person is not None else []) + panels, axis=1)
        for ov in ovs:
            ov.blend(img)
        ph = rigs[1].info.get("phase")
        if ph in PHASE_KO:
            b = badge(PHASE_KO[ph])
            ms.Overlay(b, pw + rw + int(28 * sc), int(100 * sc)).blend(img)
        c = badge(cmd_text(rigs[1].cmd))
        ms.Overlay(c, pw + (2 * rw - c.size[0]) // 2, H_ - c.size[1] - int(36 * sc)).blend(img)
        if reader is not None:
            src = reader.get(min(int((t % feeds[0].period) * feeds[0].fps), len(feeds[0].ts) - 1))
            src = np.ascontiguousarray(src[:, ::-1, ::-1])           # BGR -> RGB, mirrored (live view)
            h_, w_ = src.shape[:2]
            x0, y0 = pw + rw - w_ // 2, H_ - h_ - int(130 * sc)
            img[y0:y0 + h_, x0:x0 + w_] = src
        writer.append_data(img)
    writer.close()
    res = {"video": a.out, "backends": a.backends, "seconds": T, "wall_s": round(time.perf_counter() - t0w, 1),
           "arms_clip": None if person else path, "person": a.person}
    for r, spec in zip(rigs, a.backends):
        L = r.log
        res[spec] = {"fell": r.fell, "max_tilt": round(float(np.max(L["tilt"])), 1),
                     "moved_m": round(float(np.linalg.norm(np.asarray(L["xy"][-1]) - L["xy"][0])), 2),
                     **({"stats": dict(r.be.stats)} if hasattr(r.be, "stats") else {})}
        r.be.close()
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


class PersonWalk:
    """A walking person clip as the live chain sees it: MediaPipe landmarks (scripts/run_video.py cache) ->
    HumanMapper (arm IK) + LocoEstimator (walking intent) at the clip's frame rate -> 50 Hz ArmLimiter arms."""

    def __init__(self, spec, mirror):
        import cv2
        import run_video as g1rv

        from k1live.human import HumanMapper
        from k1live.locomotion import LocoEstimator
        from k1live.sim_server import ArmLimiter

        path, _, rng = spec.partition("@")
        t0, _, t1 = rng.partition("-")
        self.path, self.t0 = path, float(t0 or 0)
        self.W, self.V, self.I, self.fps = g1rv.extract(path, "full", None, 1, 1280)
        cap = cv2.VideoCapture(path)
        self.wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        self.k0 = int(self.t0 * self.fps)
        self.k1 = len(self.W) if not t1 else min(len(self.W), int(float(t1) * self.fps))
        self.seconds = (self.k1 - self.k0) / self.fps
        self.hm, self.est = HumanMapper(mirror=mirror, loco=False), LocoEstimator(mirror=mirror)
        self.g1rv, self.reader, self.fit = g1rv, None, None
        self.k_last, self.det, self.cmd, self.goal = -1, None, np.zeros(3), None
        self.lim = None
        self.ArmLimiter = ArmLimiter

    def __call__(self, t):
        k = self.k0 + int(t * self.fps)
        if k != self.k_last and k < self.k1:
            self.k_last = k
            W, V, I = self.W, self.V, self.I
            det = None if np.isnan(W[k, 0, 0]) else {"world": W[k], "vis": V[k], "image": I[k]}
            f = self.hm.process(det, k / self.fps, *self.wh)
            self.det = det
            obs = f.get("obs")
            c = self.est.update(obs) if obs is not None and f["present"] else None
            self.cmd = np.zeros(3) if c is None else np.array([c["vx"], c["vy"], c["wz"]], float)
            if f.get("arm_q") is not None:
                self.goal = np.asarray(f["arm_q"], float)
        if self.lim is None:
            from k1live.backends.split import SplitBackend  # noqa: F401  (rest pose of the K1 policies)
            from k1.sim import ASSETS, Student

            rest = Student(ASSETS / "loco_armdriven_r2").default
            from k1live.protocol import ARM_JOINTS, POLICY_JOINTS

            self.rest = rest[[POLICY_JOINTS.index(n) for n in ARM_JOINTS]]
            self.lim = self.ArmLimiter(self.rest)
        q, dq = self.lim.step(self.goal if self.goal is not None else self.rest)
        return self.cmd, q, dq

    def frame(self, t, w, h):
        import cv2

        from sonic.pose_mp import draw_overlay

        g1rv = self.g1rv
        if self.reader is None:
            self.fit = g1rv.PanelFit(w, h, g1rv.person_span(self.I[self.k0:self.k1], self.V[self.k0:self.k1]))
            self.reader = g1rv.FrameReader(self.path, height=h)
        k = min(self.k0 + int(t * self.fps), self.k1 - 1)
        src = self.reader.get(k)
        if self.det is not None:
            src = draw_overlay(src, self.det)
        return cv2.cvtColor(self.fit(src), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------------------------------------ main
def run_pool(fn, cfgs, workers, json_path):
    """Runs in worker processes (one task per process: each run loads its own planner, ~3 GB), results appended to
    <json>.partial.jsonl as they finish so a killed batch keeps what it did."""
    part = Path(json_path).with_suffix(".partial.jsonl") if json_path else None
    if part is not None:
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text("")
    res = []
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
        for r in ex.map(fn, cfgs):
            res.append(r)
            if part is not None:
                with part.open("a") as f:
                    f.write(json.dumps(r, default=float) + "\n")
    return res


STAND_CLIPS = {   # 9 person clips under the event videos dir (raw/ = Pexels)
    "clap": "raw/pexels_10149187.mp4", "guide": "raw/pexels_10515017.mp4", "wave": "raw/pexels_10374292.mp4",
    "welcome": "raw/pexels_10149176.mp4", "raise_arms": "event_raise_arms.mp4", "open_arms": "event_open_arms.mp4",
    "stand": "event_stand.mp4", "bow": "event_bow.mp4", "exercise": "raw/pexels_8836738.mp4"}


def clip_args(specs, ev_dir):
    out = []
    for s in specs:
        if "=" in s:
            n, p = s.split("=", 1)
        else:
            n, p = s, STAND_CLIPS[s]
        p = Path(p)
        out.append((n, str(p if p.is_absolute() else Path(ev_dir) / p)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["walk", "rescore", "stand", "switch", "video"])
    ap.add_argument("--backends", nargs="+", default=["split", "sonic_split", "hybrid"])
    ap.add_argument("--scenarios", nargs="*", default=["all"])
    ap.add_argument("--server-chain", action="store_true", help="walk: commands through each backend's server "
                    "limiter preset + heading hold (live behaviour); default raw commands as in eval_walk.py")
    ap.add_argument("--clips", nargs="*", default=list(STAND_CLIPS), help="name (the 9 STAND_CLIPS) or name=path")
    ap.add_argument("--event-dir", default=os.environ.get("K1_EVENT_VIDEOS", str(ROOT / "videos" / "event")))
    ap.add_argument("--runs", type=int, default=8, help="switch: runs per backend (seeds 0..)")
    ap.add_argument("--seconds", type=float, default=150.0, help="switch: seconds per run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arm-ik", default="shape18", choices=["shape18", "dir"], help="stand / switch: person arm IK "
                    "(k1live.human arm_ik; dir = the bone-direction IK, the sender's --arm-ik dir)")
    ap.add_argument("--arm-vmax", type=float, default=6.0, help="stand / switch: server arm reference speed limit")
    ap.add_argument("--arm-amax", type=float, default=80.0, help="stand / switch: server arm reference accel limit")
    ap.add_argument("--schedule", default="mixed", choices=["mixed", "short"], help="switch: mixed = random walks "
                    "0.5-4 s (default); short = 0.3-0.8 s walking intents (GEM-X presenter, docs/results.md)")
    ap.add_argument("--arm-jumps", action="store_true", help="switch: the arm goal jumps to another clip pose for "
                    "0.1-0.4 s during / right after 70 %% of the walks (GEM-X arm glitches)")
    ap.add_argument("--latency", type=int, default=2, help="planner result applied this many ticks after the request")
    ap.add_argument("--gemx-root", default="", help="stand: GEM-X instead of MediaPipe - folder of every-frame "
                    "real-time-worker dumps <video stem>.npz, replayed with live timing (scripts/gemx_feed.py)")
    ap.add_argument("--trace-dir", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--out", default=str(ROOT / "videos" / "k1_hybrid_vs_split.mp4"), help="video: output mp4")
    ap.add_argument("--person", default="", help="video: walking person clip path[@t0-t1] instead of DRIVE")
    ap.add_argument("--mirror", action="store_true", help="video --person: mirror (live default); off = side by side")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--label-left", default="", help="video: caption of the left robot")
    ap.add_argument("--label-right", default="", help="video: caption of the right robot")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.cmd == "video":
        if a.backends == ap.get_default("backends"):
            a.backends = ["split", "hybrid"]
        if a.seconds == ap.get_default("seconds"):
            a.seconds = 56.0
        if a.clips == ap.get_default("clips"):
            a.clips = ["welcome"]
        render_video(a)
        return
    if a.cmd in ("stand", "switch"):
        clips = clip_args(a.clips, a.event_dir)
        if a.cmd == "stand":
            feed_kw = {"arm_ik": a.arm_ik, "vmax": a.arm_vmax, "amax": a.arm_amax, "gemx_root": a.gemx_root or None}
            cfgs = [{"backend": b, "name": n, "video": p, "arm_feed": feed_kw} for n, p in clips for b in a.backends]
            fn = run_stand
        else:
            cfgs = []
            for b in a.backends:
                for s in range(a.runs):
                    n, p = clips[s % len(clips)]
                    c = {"backend": b, "seed": s, "name": n, "video": p, "seconds": a.seconds,
                         "arm_feed": {"arm_ik": a.arm_ik, "vmax": a.arm_vmax, "amax": a.arm_amax},
                         "schedule": a.schedule, "arm_jumps": a.arm_jumps, "latency": a.latency}
                    if a.trace_dir:
                        Path(a.trace_dir).mkdir(parents=True, exist_ok=True)
                        c["trace"] = str(Path(a.trace_dir) / f"switch_{safe(b)}_{s}.npz")
                    cfgs.append(c)
            fn = run_switch
        res = run_pool(fn, cfgs, a.workers, a.json)
        summ = {}
        for b in a.backends:
            rs = [r for r in res if r["backend"] == b]
            if a.cmd == "stand":
                summ[b] = {"falls": sum(r["fell"] for r in rs), "max_tilt": max(r["max_tilt_deg"] for r in rs),
                           **{k: round(float(np.mean([r[k] for r in rs if r[k] is not None])), 2)
                              for k in ("arm_deg", "arm_p90_deg", "hand_elbow_cm", "policy_ms")}}
                for r in rs:
                    print(f"{b:24s} {r['clip']:12s} arm {r['arm_deg']} deg  hand/elbow {r['hand_elbow_cm']} cm  "
                          f"tilt {r['max_tilt_deg']}  fell {r['fell']}", flush=True)
            else:
                kinds = {}
                for r in rs:
                    for k, v in r["kinds"].items():
                        kinds[k] = kinds.get(k, 0) + v
                mx = lambda key, part: max((r[key][part] for r in rs if r[key][part] is not None), default=None)  # noqa: E731
                summ[b] = {"runs": len(rs), "falls": sum(r["falls"] for r in rs),
                           "transitions": sum(r["transitions"] for r in rs), "kinds": kinds,
                           "walk_s": round(sum(r["walk_s"] for r in rs), 1),
                           **{key: {part: mx(key, part) for part in ("transition", "walking", "standing")}
                              for key in ("tilt_max", "torso_tilt_max", "arm_speed_max", "target_jump_max")},
                           "tick_ms_p99": max(r["tick_ms"]["p99"] for r in rs)}
            print(b, json.dumps(summ[b]), flush=True)
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.json).write_text(json.dumps({"summary": summ, "runs": res}, indent=1))
        return
    from eval_walk import SCENARIOS

    names = list(SCENARIOS) if a.scenarios in ([], ["all"]) else a.scenarios
    if a.cmd == "rescore":   # recompute the summaries from --trace-dir traces of an earlier walk run
        from eval_walk import ALL_SCENARIOS

        old = json.loads(Path(a.json).read_text())
        res = []
        for r in old["runs"]:
            b = r["backend"]
            tr = Path(a.trace_dir) / f"{safe(b)}_{r['scenario']}.npz"
            A = dict(np.load(tr))
            res.append({**r, **walk_summary(A, ALL_SCENARIOS[r["scenario"]][0], r["fell_at"])})
        a.backends = list(dict.fromkeys(r["backend"] for r in res))
        a.json = a.json.replace(".json", "_rescored.json")
    else:
        res = None
    cfgs = []
    for b in a.backends:
        for s in names:
            c = {"backend": b, "scenario": s, "server_chain": a.server_chain}
            if a.trace_dir:
                Path(a.trace_dir).mkdir(parents=True, exist_ok=True)
                c["trace"] = str(Path(a.trace_dir) / f"{safe(b)}_{s}.npz")
            cfgs.append(c)
    if res is None:
        res = run_pool(run_walk, cfgs, a.workers, a.json)
    summ = {}
    for b in a.backends:
        rs = [r for r in res if r["backend"] == b]
        summ[b] = {"falls": sum(r["fell"] for r in rs), "max_tilt": max(r["max_tilt_deg"] for r in rs),
                   "speed_ratio": direction_table(rs), "gait": gait_table(rs),
                   "turn_gait": gait_table(rs, ("turn",))}
        print(b, json.dumps(summ[b]), flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"summary": summ, "runs": res}, indent=1))


if __name__ == "__main__":
    main()
