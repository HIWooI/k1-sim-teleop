"""Person half of the BONES-SEED conversion, run where the BVH files are (the GPU server; numpy / scipy / joblib only, no
MuJoCo): SOMA BVH -> live-equivalent SMPL-24 stream (human_smpl.py) on exactly the 50 Hz grid the robot half
(bones_to_k1.py convert-robot, CPU host) produces from the same take's G1 CSV. Only the CSVs travel; the ~9 GB of BVH
text stays on the server.

    nice -n 15 python bones_human_part.py convert --bones <root> --selection selection.json --out <set> --workers 4
    python bones_human_part.py check --out <set>            # robot/ (from the CPU host) vs smpl/: keys, frames, heading

Grid: both halves use N = the CSV row count and the source rate (120 Hz): T50 = floor((N - 1) / 120 * 50) + 1
(k1_kin.resample). A take whose BVH has more than one frame fewer / more than its CSV is rejected (logged).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import human_smpl as H  # noqa: E402

from k1live import smpl_stream  # noqa: E402

FPS = 50.0


def csv_frames(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip()) - 1


def _one(task):
    key, root, out = task["key"], Path(task["root"]), Path(task["out"])
    t0 = time.time()
    try:
        held = H.heldout_of(key)
        dst = (out / "heldout" if held else out) / "smpl" / f"{key}.pkl"
        if dst.exists():
            return {"key": key, "skipped": True}
        n_csv = csv_frames(root / task["g1"])
        b = H.read_bvh(root / task["bvh"])
        n_bvh, fps_src = len(b["motion"]), b["fps"]
        if abs(fps_src - round(fps_src)) < 0.01:  # "Frame Time: 0.008333" -> 120.005 Hz: use the nominal rate, as
            fps_src = float(round(fps_src))       # the robot half does (else the 50 Hz grid can lose a frame)
        if abs(n_csv - n_bvh) > 1:
            return {"key": key, "rejected": f"frames csv {n_csv} != bvh {n_bvh}"}
        stride = max(1, int(fps_src // 60.0)) if fps_src > 61.0 else 1
        pos, _ = H.bvh_fk(b, stride=stride)
        L = H.soma_landmarks(b["names"], pos)
        J, Q = H.smpl_track(L)
        t50 = np.arange(H.grid_frames(n_csv, fps_src, FPS)) / FPS
        J50, Q50 = H.resample_track(J, Q, fps_src / stride, t50)
        hips = pos[:, b["names"].index("Hips")]
        th = np.arange(len(hips)) * stride / fps_src
        pel = np.stack([np.interp(np.clip(t50, 0, th[-1]), th, hips[:, k]) for k in range(3)], 1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(H.smpl_entry(J50, Q50, FPS, pel), dst, compress=3)
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (Q50[:, 1] ** 2 + Q50[:, 2] ** 2), -1, 1)))
        return {"key": key, "heldout": held, "frames": len(t50), "n_csv": n_csv, "n_bvh": n_bvh,
                "human_tilt_max_deg": round(float(tilt.max()), 1), "build_s": round(time.time() - t0, 2)}
    except Exception as e:  # one bad take must not stop the batch
        return {"key": key, "rejected": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-500:]}


def cmd_convert(a):
    from multiprocessing import Pool

    sel = json.loads(Path(a.selection).read_text())["clips"]
    tasks = [dict(c, root=str(a.bones), out=str(a.out)) for c in sel][:a.limit or None]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"[human] {len(tasks)} takes, {a.workers} workers", flush=True)
    t0, n = time.time(), 0
    with Pool(a.workers) as pool, open(a.out / "human_clips.jsonl", "a") as log:
        for r in pool.imap_unordered(_one, tasks, chunksize=4):
            log.write(json.dumps(r) + "\n")
            n += 1
            if n % 250 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"[human] {n}/{len(tasks)} {el / 60:.1f} min, eta {el / n * (len(tasks) - n) / 60:.1f} min",
                      flush=True)


def _yaw_wxyz(q):
    q = np.asarray(q)
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def cmd_check(a):
    """robot/ vs smpl/ (train and heldout): every robot key has an SMPL file with the same frame count; person vs
    robot root heading agreement; totals. Writes <out>/check_box.json."""
    import joblib

    res = {}
    for base in (a.out, a.out / "heldout"):
        rob = {p.stem: p for p in (base / "robot").rglob("*.pkl")} if (base / "robot").exists() else {}
        smp = {p.stem: p for p in (base / "smpl").glob("*.pkl")} if (base / "smpl").exists() else {}
        bad, secs, hd = [], 0.0, []
        for k, p in sorted(rob.items()):
            e = joblib.load(p)
            if list(e) != [k]:
                bad.append((k, "key != file stem"))
                continue
            e = e[k]
            secs += len(e["dof"]) / float(e["fps"])
            if k not in smp:
                bad.append((k, "no smpl"))
                continue
            s = joblib.load(smp[k])
            if len(s["smpl_joints"]) != len(e["dof"]) or float(s["fps"]) != float(e["fps"]):
                bad.append((k, f"T {len(s['smpl_joints'])} vs {len(e['dof'])}"))
                continue
            if a.heading and len(hd) < a.heading:
                qh = smpl_stream.root_quat_from_aa_yup(s["pose_aa"][:, :3])
                qr = np.asarray(e["root_rot"])[:, [3, 0, 1, 2]]
                d = (np.unwrap(_yaw_wxyz(qh)) - np.unwrap(_yaw_wxyz(qr)) + np.pi) % (2 * np.pi) - np.pi
                hd.append(float(np.degrees(np.abs(d).mean())))
        res[base.name if base != a.out else "train"] = {
            "robot": len(rob), "smpl": len(smp), "hours": round(secs / 3600, 2), "n_problems": len(bad),
            "problems": bad[:30], "smpl_without_robot": sorted(set(smp) - set(rob))[:30],
            "heading_diff_deg": {"n": len(hd), "p50": round(float(np.median(hd)), 2) if hd else None,
                                 "p95": round(float(np.percentile(hd, 95)), 2) if hd else None,
                                 "max": round(float(np.max(hd)), 2) if hd else None}}
    (a.out / "check_box.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("convert")
    s.add_argument("--bones", type=Path, required=True)
    s.add_argument("--selection", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("check")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--heading", type=int, default=400, help="heading check on this many clips")
    a = ap.parse_args(argv)
    {"convert": cmd_convert, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
