"""BONES-SEED (+ other paired human / robot sources) -> K1 SONIC motion library with the person's SMPL-24 stream:
same keys, same 50 Hz time axis - the training input of the K1 SMPL ("human") encoder (docs/training.md).

    # 1. pick clips from the metadata (rules + stats: <set>/selection.json, <set>/selection.md)
    python sonic_k1_train/data/bones_to_k1.py select --meta <bones>/metadata/seed_metadata_v004.parquet --out <set>
    # 2. convert the picked clips (CPU; GPU server or a CPU host)
    python sonic_k1_train/data/bones_to_k1.py convert --bones <bones root> --out <set> --workers 12
    # sources available without BONES-SEED (format / pipeline check, smoke training)
    python sonic_k1_train/data/bones_to_k1.py sample --out <set>              # GEAR-SONIC sample_data (BONES walk)
    python sonic_k1_train/data/bones_to_k1.py gemx --clips <soma_event_clips> --out <set>   # GEM-X BVH + K1 CSV
    # 3. checks: key / frame-count match, human vs robot heading and timing, stats -> <set>/check.json
    python sonic_k1_train/data/bones_to_k1.py check --out <set>

Per clip (convert_pair):
  robot  G1 29-dof (BONES g1 CSV, 120 Hz) -> 50 Hz -> k1live.planner.G1ToK1 (K1 pelvis = G1 torso with the waist
         roll/pitch folded in, feet = G1 ankle poses with K1 stance width by leg IK, shoulders by upper-arm orientation,
         elbow / wrist roll / waist yaw copied, K1 limits, grounded on the foot spheres) -> foot-skate cleanup
         (planner_dataset.Worker.foot_lock: stance feet pinned, legs re-solved, re-grounded).
         Or an already-K1 source (soma-retargeter K1 CSV): resample, ground, same cleanup.
  human  SOMA BVH (or the processed SOMA-26 joints) -> MediaPipe-like 33 landmarks -> the live converter
         (human_smpl.py / k1live/smpl_stream.py) -> SMPL-24 local joints + root quat -> the same 50 Hz grid.
  Both streams come from one capture, so frame i of the robot and frame i of the person are the same instant;
  sources whose durations differ by more than 1.5 source frames are rejected.
Mirrors: BONES-SEED ships a mirrored take (<name>_M) for both the G1 CSV and the SOMA BVH -> converted as their own
clips (BONES mirrors across its world y-z plane; the robot motion library does not care which plane). Sources without
mirrors (GEM-X) are mirrored here: K1Kin.mirror + smpl_stream.mirror (x-z plane).

Output (<set>/):
  robot/<group>/<key>.pkl   {key: K1 motion-lib entry, 50 Hz}                 (sonic_k1_train/data/k1_kin.py format)
  smpl/<key>.pkl            SMPL entry, flat folder, not wrapped               (gear_sonic smpl_motion_file format)
  heldout/robot/..., heldout/smpl/...   ~5 % of the takes (a take and its mirror stay together)
  clips.jsonl               one line per clip: source, frames, IK / foot-slip / limit stats, rejection reason
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import human_smpl as H  # noqa: E402
from k1_kin import K1Kin, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, resample, save_motion  # noqa: E402

from k1live import smpl_stream  # noqa: E402

FPS = 50.0
MJCF = REPO / "sonic_k1/overlay/gear_sonic/data/assets/robot_description/mjcf/k1.xml"
SAMPLE = REPO / "models/sample_data"
heldout_of = H.heldout_of


# ------------------------------------------------------------------------------------------------ readers
def read_g1_csv(path):
    """BONES-SEED g1 CSV (vendor convert_soma_csv_to_motion_lib.load_bones_csv conventions): Frame, root_translate
    XYZ [cm], root_rotate XYZ [deg, scipy 'xyz'], 29 *_dof [deg] in G1 MuJoCo order. -> pos (T,3) m, quat wxyz, dof."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    col = lambda *ns: np.stack([a[n] for n in ns], 1)  # noqa: E731
    pos = col("root_translateX", "root_translateY", "root_translateZ") / 100.0
    q = quat_xyzw_to_wxyz(Rot.from_euler("xyz", col("root_rotateX", "root_rotateY", "root_rotateZ"),
                                         degrees=True).as_quat())
    dof = np.deg2rad(np.stack([a[n] for n in a.dtype.names if n.endswith("_dof")], 1))
    assert dof.shape[1] == 29, f"{path}: {dof.shape[1]} joint columns"
    return pos, q, dof


def read_k1_csv(path, names):
    """soma-retargeter K1 CSV (same column conventions, 23 K1 joints) -> pos, quat wxyz, dof (K1 MuJoCo order)."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    col = lambda *ns: np.stack([a[n] for n in ns], 1)  # noqa: E731
    pos = col("root_translateX", "root_translateY", "root_translateZ") / 100.0
    q = quat_xyzw_to_wxyz(Rot.from_euler("xyz", col("root_rotateX", "root_rotateY", "root_rotateZ"),
                                         degrees=True).as_quat())
    dof = np.deg2rad(np.stack([a[f"{n}_dof"] for n in names], 1))
    return pos, q, dof


def human_from_bvh(path, max_fps=60.0):
    """-> landmarks (T, 33, 3) Z-up world, fps, pelvis track (T,3)."""
    b = H.read_bvh(path)
    if abs(b["fps"] - round(b["fps"])) < 0.01:
        b["fps"] = float(round(b["fps"]))
    stride = max(1, int(b["fps"] // max_fps)) if b["fps"] > max_fps + 1e-6 else 1
    pos, _ = H.bvh_fk(b, stride=stride)
    L = H.soma_landmarks(b["names"], pos)
    return L, b["fps"] / stride, pos[:, b["names"].index("Hips")], len(b["motion"]), b["fps"]


# ------------------------------------------------------------------------------------------------ conversion
class Converter:
    """One per worker process (MuJoCo models)."""

    def __init__(self):
        from k1live.planner import G1ToK1
        from planner_dataset import Worker

        self.kin = K1Kin(MJCF)
        w = Worker.__new__(Worker)  # only its foot_lock / foot_stats (they need .kin and .conv)
        w.kin, w.conv = self.kin, G1ToK1()
        self.pw, self.conv = w, w.conv

    def g1_to_k1(self, pos, quat, dof):
        self.conv.reset()
        out = np.zeros((len(pos), 30))
        ik = np.zeros(len(pos))
        for f in range(len(pos)):
            out[f] = self.conv.convert(np.concatenate([pos[f], quat[f], dof[f]]))
            ik[f] = self.conv.ik_err
        return out, ik

    def ground_k1(self, pos, quat, dof):
        dof = self.kin.clip(dof)
        pos = self.kin.ground(pos, quat, dof)
        return np.concatenate([pos, quat, dof], 1)

    def limit_frac(self, dof, tol=0.01):
        """Share of frames with a joint pinned at a K1 limit (a straight knee / elbow at its 0 stop is normal)."""
        k = self.kin
        straight = np.array([("knee" in n or "elbow" in n) for n in k.names])
        at = ((dof <= k.lo + tol) & ~straight) | (dof >= k.hi - tol)
        return float(at.any(axis=1).mean())

    def robot_only(self, pos, quat, dof, fps_src):
        """G1 -> K1 on the 50 Hz grid (the robot half when the person half runs elsewhere: bones_human_part.py)."""
        p50, q50, d50 = resample(fps_src, FPS, pos, quat, dof)
        k1, ik = self.g1_to_k1(p50, q50, d50)
        k1, lock = self.pw.foot_lock(k1)
        dof = self.kin.clip(k1[:, 7:])
        root, quat = k1[:, :3], k1[:, 3:7]
        st = {"seconds": round((len(dof) - 1) / FPS, 2), "frames": len(dof), "src": "g1",
              "ik_err_mm_p99": round(float(np.percentile(ik, 99) * 1e3), 2), "foot_locked_frac": round(lock, 3),
              "limit_frac": round(self.limit_frac(dof), 3),
              "pelvis_z": [round(float(root[:, 2].min()), 3), round(float(root[:, 2].max()), 3)],
              **self.pw.foot_stats(np.concatenate([root, quat, dof], 1))}
        return self.kin.entry(root, quat, dof, FPS), st

    def pair(self, name, robot, human, mirror=False):
        """robot = ("g1" | "k1", pos, quat_wxyz, dof, fps); human = (landmarks (T,33,3), fps, pelvis (T,3)).
        Returns [(key, robot_entry, smpl_entry, stats), ...] (+ the mirror when mirror=True)."""
        kind, rp, rq, rd, rfps = robot
        L, hfps, pelvis = human
        dur_r, dur_h = (len(rp) - 1) / rfps, (len(L) - 1) / hfps
        if abs(dur_r - dur_h) > 1.5 / min(rfps, hfps):
            raise ValueError(f"durations differ: robot {dur_r:.3f} s vs human {dur_h:.3f} s")
        p50, q50, d50 = resample(rfps, FPS, rp, rq, rd)
        t50 = np.arange(len(p50)) / FPS
        stats = {"seconds": round(t50[-1], 2), "frames": len(t50), "src": kind}
        if kind == "g1":
            k1, ik = self.g1_to_k1(p50, q50, d50)
            stats.update(ik_err_mm_p99=round(float(np.percentile(ik, 99) * 1e3), 2))
        else:
            k1 = self.ground_k1(p50, q50, d50)
        k1, lock = self.pw.foot_lock(k1)
        dof = self.kin.clip(k1[:, 7:])
        root, quat = k1[:, :3], k1[:, 3:7]
        stats.update(foot_locked_frac=round(lock, 3), limit_frac=round(self.limit_frac(dof), 3),
                     pelvis_z=[round(float(root[:, 2].min()), 3), round(float(root[:, 2].max()), 3)],
                     **self.pw.foot_stats(np.concatenate([root, quat, dof], 1)))
        J, Q = H.smpl_track(L)
        J50, Q50 = H.resample_track(J, Q, hfps, t50)
        pel50 = np.stack([np.interp(np.clip(t50, 0, dur_h), np.arange(len(pelvis)) / hfps, pelvis[:, k])
                          for k in range(3)], 1)
        # person vs robot root heading (same capture world): mean |yaw difference|
        yaw_h = np.unwrap(_yaw(Q50))
        yaw_r = np.unwrap(_yaw(quat))
        d = (yaw_h - yaw_r + np.pi) % (2 * np.pi) - np.pi
        stats.update(heading_diff_deg=round(float(np.degrees(np.abs(d).mean())), 2),
                     human_tilt_max_deg=round(float(np.degrees(np.arccos(np.clip(1 - 2 * (Q50[:, 1] ** 2 + Q50[:, 2] ** 2),
                                                                                  -1, 1))).max()), 1))
        out = [(name, self.kin.entry(root, quat, dof, FPS), H.smpl_entry(J50, Q50, FPS, pel50), stats)]
        if mirror:
            mp, mq, md = self.kin.mirror(root, quat, dof)
            Jm, Qm, _ = smpl_stream.mirror(J50, Q50)
            pm = pel50 * np.array([1.0, -1.0, 1.0])
            out.append((name + "_M", self.kin.entry(mp, mq, md, FPS), H.smpl_entry(Jm, Qm, FPS, pm),
                        dict(stats, mirrored_here=True)))
        return out


def _yaw(q):
    q = np.asarray(q)
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def write(out: Path, group: str, results, extra=None):
    rows = []
    for key, rob, smp, st in results:
        held = heldout_of(key)
        base = out / "heldout" if held else out
        save_motion(base / "robot" / group, key, rob)
        (base / "smpl").mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(smp, base / "smpl" / f"{key}.pkl")
        rows.append(dict({"key": key, "group": group, "heldout": held}, **st, **(extra or {})))
    return rows


def log_rows(out: Path, rows):
    with open(out / "clips.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------------------------------------ sources
def cmd_sample(a):
    """GEAR-SONIC sample_data: the BONES walk clip (+ its BONES mirror): G1 robot PKL 30 Hz + SOMA-26 PKL 30 Hz."""
    import joblib

    conv = Converter()
    rows = []
    for f in sorted((SAMPLE / "robot_filtered").rglob("*.pkl")):
        key = f.stem
        e = next(iter(joblib.load(f).values()))
        s = next(iter(joblib.load(next((SAMPLE / "soma_filtered").rglob(f"{key}.pkl"))).values()))
        jz = s["soma_joints"] + np.stack([s["soma_transl"][:, 0], -s["soma_transl"][:, 2], s["soma_transl"][:, 1]], 1)[:, None]
        L = H.soma26_landmarks(jz)
        robot = ("g1", e["root_trans_offset"], quat_xyzw_to_wxyz(e["root_rot"]), e["dof"], float(e["fps"]))
        res = conv.pair(key, robot, (L, float(s["fps"]), jz[:, 0]))
        rows += write(a.out, "bones_sample", res, {"source": "gear_sonic_sample", "human": "soma26"})
    log_rows(a.out, rows)
    return rows


def cmd_gemx(a):
    """GEM-X clips: <clip>/soma.bvh (person, video fps) + <clip>/k1.csv (soma-retargeter K1, same frames)."""
    conv = Converter()
    rows = []
    dirs = sorted(d for d in Path(a.clips).iterdir() if (d / "soma.bvh").exists() and (d / "k1.csv").exists())
    pairs = [(f"gemx_{d.name}", d / "soma.bvh", d / "k1.csv") for d in dirs]
    if a.guap:
        pairs.append(("gemx_guap_ver2", Path(a.guap[0]), Path(a.guap[1])))
    for key, bvh, csv in pairs:
        L, hfps, pel, n_src, fps_src = human_from_bvh(bvh)
        rp, rq, rd = read_k1_csv(csv, conv.kin.names)
        if len(rp) != n_src:
            rows.append({"key": key, "rejected": f"frames robot {len(rp)} != bvh {n_src}"})
            continue
        res = conv.pair(key, ("k1", rp, rq, rd, fps_src), (L, hfps, pel), mirror=True)
        rows += write(a.out, "gemx", res, {"source": "gemx", "human": "bvh"})
        print(f"[gemx] {key}: {res[0][3]['seconds']} s", flush=True)
    log_rows(a.out, rows)
    return rows


# ------------------------------------------------------------------------------------------------ BONES-SEED
_CONV = None


def _init():
    global _CONV
    _CONV = Converter()


def _bones_one(task):
    root, key, g1_rel, bvh_rel = task["root"], task["key"], task["g1"], task["bvh"]
    t0 = time.time()
    try:
        L, hfps, pel, n_src, fps_src = human_from_bvh(Path(root) / bvh_rel)
        rp, rq, rd = read_g1_csv(Path(root) / g1_rel)
        if abs(len(rp) - n_src) > 1:
            return [{"key": key, "rejected": f"frames g1 {len(rp)} != bvh {n_src}"}]
        n = min(len(rp), n_src)
        # human_from_bvh may have strided the BVH; the G1 CSV keeps its own rate (same source frames)
        res = _CONV.pair(key, ("g1", rp[:n], rq[:n], rd[:n], fps_src), (L, hfps, pel))
        rows = write(Path(task["out"]), task["group"], res, {"source": "bones_seed", "human": "bvh"})
        rows[0]["build_s"] = round(time.time() - t0, 1)
        return rows
    except Exception as e:  # a bad file must not stop the batch
        return [{"key": key, "rejected": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}]


def _robot_one(task):
    key = task["key"]
    t0 = time.time()
    try:
        held = heldout_of(key)
        dst = (Path(task["out"]) / "heldout" if held else Path(task["out"])) / "robot" / task["group"] / f"{key}.pkl"
        if dst.exists():
            return {"key": key, "skipped": True}
        rp, rq, rd = read_g1_csv(Path(task["root"]) / task["g1"])
        entry, st = _CONV.robot_only(rp, rq, rd, 120.0)
        assert len(entry["dof"]) == H.grid_frames(len(rp), 120.0, FPS), "grid mismatch with the person half"
        dst.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump({key: entry}, dst, compress=3)
        return dict({"key": key, "group": task["group"], "heldout": held, "n_csv": len(rp),
                     "build_s": round(time.time() - t0, 2), "source": "bones_seed"}, **st)
    except Exception as e:
        return {"key": key, "rejected": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}


def cmd_convert_robot(a):
    """Robot half only (G1 CSVs): pairs with bones_human_part.py's SMPL files on the same 50 Hz grid."""
    from multiprocessing import Pool

    sel = json.loads((a.out / "selection.json").read_text())["clips"]
    tasks = [dict(c, root=str(a.bones), out=str(a.out)) for c in sel][:a.limit or None]
    print(f"[robot] {len(tasks)} takes, {a.workers} workers", flush=True)
    t0, n = time.time(), 0
    with Pool(a.workers, initializer=_init) as pool, open(a.out / "robot_clips.jsonl", "a") as log:
        for r in pool.imap_unordered(_robot_one, tasks, chunksize=4):
            log.write(json.dumps(r) + "\n")
            n += 1
            if n % 250 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"[robot] {n}/{len(tasks)} {el / 60:.1f} min, eta {el / n * (len(tasks) - n) / 60:.1f} min",
                      flush=True)


def cmd_convert(a):
    from multiprocessing import Pool

    sel = json.loads((a.out / "selection.json").read_text())["clips"]
    done = set()
    if (a.out / "clips.jsonl").exists():
        done = {json.loads(l)["key"] for l in open(a.out / "clips.jsonl")}
    tasks = [dict(c, root=str(a.bones), out=str(a.out)) for c in sel if c["key"] not in done]
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"[convert] {len(tasks)} clips to do ({len(done)} already logged), {a.workers} workers", flush=True)
    t0, n = time.time(), 0
    with Pool(a.workers, initializer=_init) as pool:
        for rows in pool.imap_unordered(_bones_one, tasks, chunksize=2):
            log_rows(a.out, rows)
            n += 1
            if n % 50 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"[convert] {n}/{len(tasks)}  {el / 60:.1f} min, eta {el / n * (len(tasks) - n) / 60:.1f} min",
                      flush=True)


# ------------------------------------------------------------------------------------------------ checks
def cmd_check(a):
    """Every robot key has an SMPL file with the same frame count; heading / timing agreement on a sample."""
    import joblib

    out = a.out
    res = {"sets": {}}
    for base in (out, out / "heldout"):
        rob = {p.stem: p for p in (base / "robot").rglob("*.pkl")} if (base / "robot").exists() else {}
        smp = {p.stem: p for p in (base / "smpl").glob("*.pkl")} if (base / "smpl").exists() else {}
        bad, secs = [], 0.0
        for k, p in rob.items():
            e = joblib.load(p)
            assert list(e) == [k], f"{p}: key {list(e)} != file stem"
            e = e[k]
            if k not in smp:
                bad.append((k, "no smpl"))
                continue
            s = joblib.load(smp[k])
            if len(s["smpl_joints"]) != len(e["dof"]) or s["fps"] != e["fps"]:
                bad.append((k, f"T {len(s['smpl_joints'])} vs {len(e['dof'])}"))
            r0 = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][:, :3])
            loc = smpl_stream.world_joints(s["smpl_joints"], r0 * np.array([1, -1, -1, -1]))
            if np.abs(loc[:, 0]).max() > 1e-4:
                bad.append((k, "pelvis not at the origin after inv(root)"))
            secs += len(e["dof"]) / e["fps"]
        res["sets"][str(base.relative_to(out)) or "train"] = {
            "robot": len(rob), "smpl": len(smp), "minutes": round(secs / 60, 1), "problems": bad[:20],
            "n_problems": len(bad), "smpl_without_robot": sorted(set(smp) - set(rob))[:20]}
    rows = [json.loads(l) for l in open(out / "clips.jsonl")] if (out / "clips.jsonl").exists() else []
    ok = [r for r in rows if "rejected" not in r]
    if ok:
        def q(k, p):
            v = [r[k] for r in ok if k in r]
            return round(float(np.percentile(v, p)), 3) if v else None
        res["stats"] = {k: {"p50": q(k, 50), "p95": q(k, 95), "max": q(k, 100)} for k in
                        ("heading_diff_deg", "slip_mean", "slip_p95", "limit_frac", "ik_err_mm_p99", "foot_locked_frac",
                         "human_tilt_max_deg", "seconds")}
    res["rejected"] = [{k: r[k] for k in ("key", "rejected")} for r in rows if "rejected" in r][:50]
    res["n_rejected"] = sum("rejected" in r for r in rows)
    if a.timing:
        res["timing"] = timing_check(out, a.timing)
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000])


def timing_check(out: Path, n: int):
    """Same time axis? Best lag (frames, +-15) between the person's and the robot's hand-height signals:
    SMPL wrist z (root-local) vs K1 rubber-hand z (pelvis frame), both differentiated; 0 = aligned."""
    import joblib

    kin = K1Kin(MJCF)
    hands = [kin.m.body(f"{s}_wrist_roll_rubber_hand").id for s in ("left", "right")]
    pel = kin.m.body("pelvis").id
    rows = []
    for p in sorted((out / "robot").rglob("*.pkl"))[:n]:
        k = p.stem
        e = joblib.load(p)[k]
        s = joblib.load(out / "smpl" / f"{k}.pkl")
        T = len(e["dof"])
        rz = np.zeros((T, 2))
        for t in range(T):
            kin.set(e["root_trans_offset"][t], quat_xyzw_to_wxyz(e["root_rot"][t]), e["dof"][t])
            R = kin.d.xmat[pel].reshape(3, 3)
            rz[t] = [(R.T @ (kin.d.xpos[h] - kin.d.xpos[pel]))[2] for h in hands]
        r0 = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][:, :3])
        loc = smpl_stream.world_joints(s["smpl_joints"], r0 * np.array([1, -1, -1, -1]))
        hz = loc[:, [20, 21], 2]
        a_, b_ = np.diff(hz, axis=0).sum(1), np.diff(rz, axis=0).sum(1)
        a_, b_ = a_ - a_.mean(), b_ - b_.mean()
        lags = range(-15, 16)
        cc = [float(np.dot(a_[max(0, l):len(a_) + min(0, l)], b_[max(0, -l):len(b_) - max(0, l)])) for l in lags]
        best = list(lags)[int(np.argmax(cc))]
        corr = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 0 and b_.std() > 0 else None
        rows.append({"key": k, "best_lag_frames": best, "corr_lag0": None if corr is None else round(corr, 3)})
    return rows


# ------------------------------------------------------------------------------------------------ selection
def cmd_select(a):
    import bones_select

    bones_select.main(a.meta, a.out, a.max_hours)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--meta", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--max-hours", type=float, default=10.0)
    s = sub.add_parser("convert")
    s.add_argument("--bones", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("convert-robot")
    s.add_argument("--bones", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("gemx")
    s.add_argument("--clips", type=Path, required=True)
    s.add_argument("--guap", nargs=2, default=None, metavar=("BVH", "K1CSV"))
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("check")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--timing", type=int, default=0, help="also run the timing check on N clips")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    {"select": cmd_select, "convert": cmd_convert, "convert-robot": cmd_convert_robot, "sample": cmd_sample,
     "gemx": cmd_gemx, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
