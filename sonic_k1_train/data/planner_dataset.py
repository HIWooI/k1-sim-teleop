"""K1 walking motion library from the GEAR-SONIC kinematic planner (CPU; run on a Linux CPU host, not the Mac).

    python sonic_k1_train/data/planner_dataset.py --out <dir> --minutes 75 --workers 8
    python sonic_k1_train/data/planner_dataset.py --out <dir> --stats-only            # re-print stats

Each clip (8-20 s) is a random command script fed to the same command path the live demo uses
(k1live.planner.KinematicPlanner: body-frame vx/vy/wz -> planner mode, world movement / facing direction, change
tolerances, replan logic, 8-frame blends, a simulated 1-4 tick planner latency). The G1 frame that would be played at
each 50 Hz tick is recorded and converted to K1 (k1live.planner.G1ToK1: torso-matched pelvis, foot-pose leg IK,
upper-arm orientation for the shoulders, grounded on the K1 foot spheres), foot-skate cleaned (stance feet pinned, legs re-solved, re-grounded), then mirrored (_M).
Command families: stand, forward, backward, sideways, diagonal (forward / backward), turn in place, walk + turn;
scripts: mix, cruise, start_stop, zigzag. Speeds 0.2-1.0 m/s forward (slowWalk <= 0.8, walk above), <= 0.6 backward,
<= 0.5 sideways, turn 0.3-1.0 rad/s. No run mode: its flight phases would be erased by grounding and K1 is not
trained for running here.
About half the clips get arms laid over the walk (planner_arms.py: k1/ gestures, people's arms from the event videos,
smooth random arms) with 0.5 s ramps; legs, pelvis and waist stay the planner's.
Held out (-> <out>/heldout): every clip of the backward-diagonal family and ~5 % of the rest.
Output: motion_lib PKLs like build_dataset.py (<out>/robot/planner/*.pkl, <out>/heldout/planner/*.pkl),
<out>/clips.json (per clip: script, segments, modes, speeds, IK error, reference foot slip) and stats.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

FPS = 50.0
MJCF = REPO / "sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml"


# ------------------------------------------------------------------------------------------------ command scripts
def seg_cmd(rng, fam):
    """One segment command (vx, vy, wz) of a family, K1 units."""
    sp = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    sgn = lambda: float(rng.choice([-1.0, 1.0]))  # noqa: E731
    if fam == "stand":
        return (0.0, 0.0, 0.0)
    if fam == "forward":
        v = sp(0.2, 1.0) if rng.random() < 0.35 else sp(0.25, 0.7)
        return (v, 0.0, 0.0)
    if fam == "backward":
        return (-sp(0.2, 0.6), 0.0, 0.0)
    if fam == "side":
        return (0.0, sgn() * sp(0.2, 0.5), 0.0)
    if fam in ("diag", "diag_back"):
        a = np.radians(sp(20, 70)) * sgn()
        v = sp(0.2, 0.7)
        vx = v * np.cos(a) * (-1 if fam == "diag_back" else 1)
        return (vx, v * np.sin(a), 0.0)
    if fam == "turn":
        return (0.0, 0.0, sgn() * sp(0.3, 1.0))
    if fam == "walk_turn":
        return (sp(0.2, 0.8), 0.0, sgn() * sp(0.2, 0.6))
    raise KeyError(fam)


FAMS = ["stand", "forward", "backward", "side", "diag", "turn", "walk_turn"]
FAM_P = [0.14, 0.28, 0.12, 0.13, 0.11, 0.11, 0.11]


def make_script(rng, kind, dur):
    """[(t0, t1, family, (vx, vy, wz))] covering [0, dur]."""
    segs, t = [], 0.0
    lead = float(rng.uniform(0.8, 2.0))
    segs.append((0.0, lead, "stand", (0.0, 0.0, 0.0)))
    t = lead
    while t < dur - 0.5:
        if kind == "cruise":
            fam = rng.choice(["forward", "walk_turn", "forward", "diag"])
            L = rng.uniform(3.0, 7.0)
        elif kind == "start_stop":
            fam = "stand" if segs[-1][2] != "stand" else rng.choice(["forward", "backward", "side", "diag", "turn"])
            L = rng.uniform(1.0, 2.5) if fam == "stand" else rng.uniform(2.0, 4.5)
        elif kind == "zigzag":
            fam = rng.choice(["forward", "side", "diag", "backward", "turn"])
            L = rng.uniform(1.2, 2.5)
        elif kind == "diag_back":
            fam = rng.choice(["diag_back", "stand", "diag_back"])
            L = rng.uniform(2.0, 5.0)
        else:  # mix
            fam = rng.choice(FAMS, p=FAM_P)
            L = rng.uniform(1.5, 5.0)
        t1 = min(dur, t + L)
        segs.append((t, t1, str(fam), seg_cmd(rng, str(fam))))
        t = t1
    if rng.random() < 0.6:  # most clips end with a stop
        t0 = max(segs[-1][0] + 1.0, dur - rng.uniform(1.0, 2.0))
        last = segs[-1]
        segs[-1] = (last[0], t0, last[2], last[3])
        segs.append((t0, dur, "stand", (0.0, 0.0, 0.0)))
    return segs


# ------------------------------------------------------------------------------------------------ one clip
CFG = {"prefix": "plan", "live": "", "live_p": 0.0, "arms_always": False, "stand_frac": 0.0}


class Worker:
    def __init__(self, threads=2, cfg=None):
        self.cfg = dict(CFG, **(cfg or {}))
        from k1live.planner import G1ToK1, PlannerModel

        self.model = PlannerModel(threads=threads)
        self.conv = G1ToK1()
        from k1_kin import K1Kin

        self.kin = K1Kin(MJCF)
        import planner_arms as PA

        self.PA = PA
        self.gest = PA.load_gestures()
        self.human = PA.load_human()
        # people's arms as the live chain makes them (planner_arms_live.py -> arms_live.npz)
        self.live = PA.load_human(Path(self.cfg["live"])) if self.cfg["live"] else {}

    def g1_stream(self, segs, dur, latency, seed):
        from k1live.planner import KinematicPlanner

        kp = KinematicPlanner(robot="g1", model=self.model, latency_ticks=latency, seed=seed, horizon=1)
        kp.reset(None)
        T = int(round(dur * FPS))
        out = np.zeros((T, 36))
        modes = np.zeros(T, int)
        cmds = np.zeros((T, 3))
        k = 0
        for f in range(T):
            t = f / FPS
            while k + 1 < len(segs) and t >= segs[k + 1][0]:
                k += 1
            kp.command(*segs[k][3])
            w = kp.step()
            out[f, :3], out[f, 3:7], out[f, 7:] = w.pos[0], w.quat[0], w.q[0]
            modes[f], cmds[f] = w.mode, segs[k][3]
        info = {"replans": kp.stream.n_replans, "planner_errors": kp.stream.errors}
        kp.close()
        return out, modes, cmds, info

    def to_k1(self, g1):
        self.conv.reset()
        k1 = np.zeros((len(g1), 30))
        ik = np.zeros(len(g1))
        for f in range(len(g1)):
            k1[f] = self.conv.convert(g1[f])
            ik[f] = self.conv.ik_err
        return k1, ik

    def foot_lock(self, k1, h_on=0.006, v_on=0.35, ramp=4):
        """Foot-skate cleanup: while a foot is in stance (sole within h_on of the floor, xy speed < v_on, runs >= 5
        frames) pin its xy and yaw to the stance median; ramp in/out over `ramp` frames; legs re-solved by the same
        IK as the retarget, then re-grounded. The planner output itself slides ~5 cm/s in stance (G1 too)."""
        from scipy.spatial.transform import Rotation as Rot

        kin, conv = self.kin, self.conv
        T = len(k1)
        sides = ("left", "right")
        pos = {s: np.zeros((T, 3)) for s in sides}
        R = {s: np.zeros((T, 3, 3)) for s in sides}
        h = {s: np.zeros(T) for s in sides}
        gs = {s: [g for g in kin.foot_geoms if kin.m.geom_bodyid[g] == kin.foot_body[s]] for s in sides}
        for f in range(T):
            kin.set(k1[f, :3], k1[f, 3:7], k1[f, 7:])
            for s in sides:
                b = kin.foot_body[s]
                pos[s][f], R[s][f] = kin.d.xpos[b], kin.d.xmat[b].reshape(3, 3)
                h[s][f] = float((kin.d.geom_xpos[gs[s], 2] - kin.m.geom_size[gs[s], 0]).min())
        tgt = {}
        locked = 0
        for s in sides:
            v = np.zeros(T)
            v[1:-1] = np.linalg.norm(pos[s][2:, :2] - pos[s][:-2, :2], axis=1) * FPS / 2
            c = (h[s] < h_on) & (v < v_on)
            runs, f = [], 0
            while f < T:
                if c[f]:
                    g = f
                    while g < T and c[g]:
                        g += 1
                    if g - f >= 5:
                        runs.append((f, g))
                    f = g
                else:
                    f += 1
            w = np.zeros(T)
            axy = pos[s][:, :2].copy()
            yaw = np.arctan2(R[s][:, 1, 0], R[s][:, 0, 0])
            ayaw = yaw.copy()
            for a, b in runs:
                mxy = np.median(pos[s][a:b, :2], axis=0)
                myaw = float(np.angle(np.mean(np.exp(1j * yaw[a:b]))))
                lo, hi = max(0, a - ramp), min(T, b + ramp)
                for k in range(lo, hi):
                    wk = 1.0 if a <= k < b else (1.0 - (a - k) / (ramp + 1) if k < a else 1.0 - (k - b + 1) / (ramp + 1))
                    if wk > w[k]:
                        w[k], axy[k], ayaw[k] = wk, mxy, myaw
                locked += b - a
            txy = pos[s][:, :2] + w[:, None] * (axy - pos[s][:, :2])
            dyaw = w * ((ayaw - yaw + np.pi) % (2 * np.pi) - np.pi)
            Rt = np.einsum("tij,tjk->tik", Rot.from_euler("z", dyaw[:, None]).as_matrix(), R[s])
            tgt[s] = (np.concatenate([txy, pos[s][:, 2:]], 1), Rt)
        out = k1.copy()
        legs = np.concatenate(list(conv.legs.values()))
        for f in range(T):
            q = k1[f, 7:].copy()
            if f > 0:
                q[legs] = out[f - 1, 7 + legs]   # warm start
            q = conv._leg_ik(k1[f, :7], q, {s: (tgt[s][0][f], tgt[s][1][f]) for s in sides})
            out[f, 7:] = q
            conv.dk.qpos[:7] = out[f, :7]
            conv.dk.qpos[conv.kq] = q
            conv.mj.mj_kinematics(conv.mk, conv.dk)
            out[f, 2] -= float((conv.dk.geom_xpos[conv.foot_geoms, 2] - conv.foot_r).min())
        return out, locked / (2 * T)

    def arms(self, rng, T, base):
        """Optionally lay arms over the walk. Returns (arms (T,10), meta)."""
        PA = self.PA
        r = rng.random()
        if r < 0.5 and not self.cfg["arms_always"]:
            return base, None
        if self.live and rng.random() < self.cfg["live_p"]:
            kind = "live"
        else:
            kind = rng.choice(["gesture", "human", "random"], p=[0.35, 0.40, 0.25])
        if kind == "live":
            name = str(rng.choice(sorted(self.live)))
            src = PA.ArmTrack(name, self.live[name], FPS * rng.uniform(0.85, 1.15))
        elif kind == "gesture":
            name = str(rng.choice(sorted(k for k in self.gest if k not in PA.HELDOUT_GESTURES)))
            src = PA.ArmTrack(name, self.gest[name], FPS * rng.uniform(0.85, 1.2))
        elif kind == "human":
            name = str(rng.choice(sorted(self.human)))
            src = PA.ArmTrack(name, self.human[name], FPS * rng.uniform(0.9, 1.1))
        else:
            name = "random"
            src = PA.ArmTrack(name, PA.random_arms(rng, T + 50, scale=rng.uniform(0.4, 0.9)))
        dur = T / FPS
        if rng.random() < 0.55:
            wins = [(rng.uniform(0.3, 1.5), dur + 1.0)]
        else:
            wins, t = [], rng.uniform(0.5, 2.0)
            while t < dur - 2.0:
                L = rng.uniform(3.0, 8.0)
                wins.append((t, min(dur + 1.0, t + L)))
                t += L + rng.uniform(1.0, 3.0)
        t0 = rng.uniform(0, src.seconds)
        q = src.at(t0 + np.arange(T) / FPS)
        w = PA.ramp_weights(T, FPS, wins)
        return PA.overlay(base, q, w), {"kind": str(kind), "name": name, "windows": [[round(a, 2), round(b, 2)] for a, b in wins]}

    def foot_stats(self, k1):
        """Reference foot slip: xy speed of each ankle_roll_link while its lowest sphere is within 3 mm of z = 0."""
        kin = self.kin
        T = len(k1)
        xy = np.zeros((T, 2, 2))
        low = np.zeros((T, 2))
        bodies = [kin.foot_body["left"], kin.foot_body["right"]]
        side_g = [[g for g in kin.foot_geoms if kin.m.geom_bodyid[g] == b] for b in bodies]
        for f in range(T):
            kin.set(k1[f, :3], k1[f, 3:7], k1[f, 7:])
            for i, b in enumerate(bodies):
                xy[f, i] = kin.d.xpos[b][:2]
                gs = side_g[i]
                low[f, i] = float((kin.d.geom_xpos[gs, 2] - kin.m.geom_size[gs, 0]).min())
        v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * FPS
        c = low[1:] < 0.003
        slip = v[c]
        return {"slip_mean": round(float(slip.mean()), 4) if slip.size else 0.0,
                "slip_p50": round(float(np.median(slip)), 4) if slip.size else 0.0,
                "slip_p95": round(float(np.percentile(slip, 95)), 4) if slip.size else 0.0,
                "contact_frac": round(float(c.mean()), 3), "max_penetration_mm": round(float(-low.min() * 1e3), 2)}

    def clip(self, idx, seed, out: Path):
        rng = np.random.default_rng(seed)
        kind = str(rng.choice(["mix", "cruise", "start_stop", "zigzag", "diag_back"], p=[0.42, 0.18, 0.18, 0.14, 0.08]))
        dur = float(rng.uniform(8.0, 20.0))
        if rng.random() < self.cfg["stand_frac"]:   # planner idle the whole clip (standing, arms only)
            kind = "stand"
            segs = [(0.0, dur, "stand", (0.0, 0.0, 0.0))]
        else:
            segs = make_script(rng, kind, dur)
        latency = int(rng.integers(1, 5))
        t0 = time.time()
        g1, modes, cmds, info = self.g1_stream(segs, dur, latency, int(rng.integers(0, 100000)))
        k1, ik = self.to_k1(g1)
        k1, lock_frac = self.foot_lock(k1)
        arms_idx = self.PA.ARM_IDX
        arms, arm_meta = self.arms(rng, len(k1), k1[:, 7:][:, arms_idx])
        dof = k1[:, 7:].copy()
        dof[:, arms_idx] = arms
        dof = self.kin.clip(dof)
        root, quat = k1[:, :3], k1[:, 3:7]
        held = kind == "diag_back" or rng.random() < 0.05
        name = f"{self.cfg['prefix']}_{idx:04d}_{kind}" + ("_arms" if arm_meta else "")
        from k1_kin import save_motion

        sub = out / ("heldout" if held else "robot") / "planner"
        stats = self.foot_stats(np.concatenate([root, quat, dof], 1))
        variants = [(name, root, quat, dof), (name + "_M",) + self.kin.mirror(root, quat, dof)]
        for n, p, q, d in variants:
            save_motion(sub, n, self.kin.entry(p, q, d, FPS))
        vel = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) * FPS
        yaw = np.unwrap(np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                                   1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)))
        return {"name": name, "kind": kind, "seed": int(seed), "seconds": round(dur, 2), "frames": len(dof),
                "heldout": bool(held), "latency_ticks": latency, "arms": arm_meta, "foot_locked_frac": round(lock_frac, 3),
                "segments": [[round(a, 2), round(b, 2), f, [round(c, 3) for c in cmd]] for a, b, f, cmd in segs],
                "mode_seconds": {str(m): round(float((modes == m).sum()) / FPS, 2) for m in np.unique(modes)},
                "ref_speed_hist": np.histogram(vel, bins=[0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 5])[0].tolist(),
                "ref_speed_max": round(float(np.percentile(vel, 99)), 3),
                "ref_yaw_rate_p99": round(float(np.percentile(np.abs(np.diff(yaw)) * FPS, 99)), 3),
                "pelvis_z": [round(float(root[:, 2].min()), 3), round(float(root[:, 2].max()), 3)],
                "ik_err_mm_max": round(float(ik.max() * 1e3), 2), "ik_err_mm_p99": round(float(np.percentile(ik, 99) * 1e3), 2),
                **stats, **info, "build_s": round(time.time() - t0, 1)}


_W = None


def _init(threads, cfg=None):
    global _W
    _W = Worker(threads, cfg)


def _run(args):
    idx, seed, out = args
    try:
        return _W.clip(idx, seed, Path(out))
    except Exception as e:  # noqa: BLE001
        return {"name": f"plan_{idx:04d}", "error": repr(e)}


def stats(rows, out: Path):
    ok = [r for r in rows if "error" not in r]
    lines = []
    for held in (False, True):
        sel = [r for r in ok if r["heldout"] == held]
        sec = sum(r["seconds"] for r in sel)
        lines.append(f"{'heldout' if held else 'train':7s}: {len(sel)} clips x2 (mirror) = {2 * sec / 60:.1f} min")
    modes = {}
    for r in ok:
        for m, s in r["mode_seconds"].items():
            modes[m] = modes.get(m, 0) + 2 * s
    lines.append("mode minutes (x2 mirror): " + ", ".join(f"{ {'0': 'idle', '1': 'slowWalk', '2': 'walk'}.get(m, m)} {s / 60:.1f}" for m, s in sorted(modes.items())))
    kinds = {}
    for r in ok:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    lines.append("scripts: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    arms = {}
    for r in ok:
        k = r["arms"]["kind"] if r["arms"] else "planner"
        arms[k] = arms.get(k, 0) + 1
    lines.append("arms: " + ", ".join(f"{k} {v}" for k, v in sorted(arms.items())))
    h = np.sum([r["ref_speed_hist"] for r in ok], axis=0)
    edges = ["<0.05", "0.05-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", ">1.2"]
    lines.append("reference pelvis speed (m/s, share of frames): " + ", ".join(f"{e} {100 * c / h.sum():.0f}%" for e, c in zip(edges, h)))
    fams = {}
    for r in ok:
        for a, b, f, _ in r["segments"]:
            fams[f] = fams.get(f, 0) + 2 * (b - a)
    lines.append("command families (min, x2): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in sorted(fams.items())))
    lines.append(f"IK foot error mm: p99 max over clips {max(r['ik_err_mm_p99'] for r in ok):.2f}, max {max(r['ik_err_mm_max'] for r in ok):.1f}")
    lines.append(f"reference foot slip in contact m/s: mean {np.mean([r['slip_mean'] for r in ok]):.3f}, "
                 f"p95 (median over clips) {np.median([r['slip_p95'] for r in ok]):.3f}")
    lines.append(f"pelvis z range {min(r['pelvis_z'][0] for r in ok):.3f}-{max(r['pelvis_z'][1] for r in ok):.3f} m, "
                 f"replans/s {sum(r['replans'] for r in ok) / sum(r['seconds'] for r in ok):.2f}, planner errors {sum(r['planner_errors'] for r in ok)}")
    lines.append(f"failed clips: {len(rows) - len(ok)}")
    txt = "\n".join(lines)
    (out / "stats.txt").write_text(txt + "\n")
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=75.0, help="un-mirrored minutes to generate (mirror doubles)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime threads per worker")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--start", type=int, default=0, help="first clip index (append to an existing set)")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--prefix", default="plan", help="clip name prefix (keep it unique across sets)")
    ap.add_argument("--arms-live", default="", help="arms_live.npz (planner_arms_live.py): people's arms from the live chain")
    ap.add_argument("--live-p", type=float, default=0.0, help="share of arm overlays drawn from --arms-live")
    ap.add_argument("--arms-always", action="store_true", help="every clip gets an arm overlay")
    ap.add_argument("--stand-frac", type=float, default=0.0, help="share of clips that stand the whole time (arms only)")
    a = ap.parse_args()
    cfg = {"prefix": a.prefix, "live": a.arms_live, "live_p": a.live_p, "arms_always": a.arms_always,
           "stand_frac": a.stand_frac}
    a.out.mkdir(parents=True, exist_ok=True)
    js = a.out / "clips.json"
    if a.stats_only:
        print(stats(json.loads(js.read_text()), a.out))
        return
    rows = json.loads(js.read_text()) if (js.exists() and a.start > 0) else []
    n_est = int(np.ceil(a.minutes * 60 / 14.0 * 1.04))
    jobs = [(a.start + i, a.seed + 7919 * (a.start + i), str(a.out)) for i in range(n_est)]
    t0 = time.time()
    if a.workers <= 1:
        _init(a.threads, cfg)
        it = map(_run, jobs)
    else:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.threads, cfg))
        it = pool.imap_unordered(_run, jobs)
    total = 0.0
    for r in it:
        rows.append(r)
        if "error" in r:
            print("ERROR", r["name"], r["error"], flush=True)
            continue
        total += r["seconds"]
        if len(rows) % 20 == 0:
            print(f"{len(rows)} clips, {total / 60:.1f} min, {time.time() - t0:.0f} s", flush=True)
            js.write_text(json.dumps(rows))
    js.write_text(json.dumps(rows, indent=0))
    print(stats(rows, a.out))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
