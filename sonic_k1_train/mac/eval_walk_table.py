"""eval_walk.py JSON files -> markdown comparison tables (one column block per run); eval_k1_mujoco.py gesture JSON
files (gest_*.json) -> arm / leg error per gesture, one row per run.

    python sonic_k1_train/mac/eval_walk_table.py logs/eval/walk_split.json logs/eval/walk_k1_*.json
    python sonic_k1_train/mac/eval_walk_table.py logs/eval/gest_k1s_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def seg_str(r):
    out = []
    for s in r.get("segments", []):
        c = s["cmd"]
        out.append(f"({c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f})->({s['vx']:+.2f},{s['vy']:+.2f},{s['wz']:+.2f})")
    return " ".join(out)


def ratios(r):
    """Steady-segment means: (|v| achieved / |v| commanded over moving segments, wz achieved / wz commanded over
    turning segments), both signed along the command (projection)."""
    import numpy as np

    lv = [(np.dot([s["vx"], s["vy"]], s["cmd"][:2]) / np.linalg.norm(s["cmd"][:2]), np.linalg.norm(s["cmd"][:2]))
          for s in r.get("segments", []) if np.linalg.norm(s["cmd"][:2]) > 0.05]
    wv = [(s["wz"] * np.sign(s["cmd"][2]), abs(s["cmd"][2])) for s in r.get("segments", []) if abs(s["cmd"][2]) > 0.05]
    f = lambda xs: round(sum(a for a, _ in xs) / sum(b for _, b in xs), 2) if xs else None  # noqa: E731
    return f(lv), f(wv)


def direction(c):
    vx, vy, wz = c
    if abs(vx) < 0.05 and abs(vy) < 0.05:
        return "turn" if abs(wz) > 0.05 else None
    if abs(wz) > 0.05:
        return "walk+turn"
    if abs(vy) < 0.05:
        return "fwd" if vx > 0 else "back"
    return "side" if abs(vx) < 0.05 else "diag"


DIRS = ["fwd", "back", "side", "turn", "walk+turn", "diag"]


def direction_ratios(d):
    """eval_walk result -> {direction: steady-segment speed ratio over all scenarios} (achieved projected on the
    command / |command|; yaw rate for turn)."""
    import numpy as np

    acc = {k: [0.0, 0.0] for k in DIRS}
    for r in d["scenarios"].values():
        for sg in r.get("segments", []):
            k = direction(sg["cmd"])
            if k is None:
                continue
            c = np.array(sg["cmd"], float)
            if k == "turn":
                acc[k][0] += sg["wz"] * np.sign(c[2]); acc[k][1] += abs(c[2])
            else:
                v = np.linalg.norm(c[:2])
                acc[k][0] += (sg["vx"] * c[0] + sg["vy"] * c[1]) / v; acc[k][1] += v
    return {k: (a / b if b else None) for k, (a, b) in acc.items()}


def direction_summary(runs):
    """Per-direction table, and the worst of fwd / back / side / turn."""
    print("\n| run | " + " | ".join(DIRS) + " | min(fwd, back, side, turn) |")
    print("|---|" + "---|" * (len(DIRS) + 1))
    for n, d in runs:
        rat = direction_ratios(d)
        core = [rat[k] for k in ("fwd", "back", "side", "turn") if rat[k] is not None]
        if not core:
            continue
        print(f"| {n} | " + " | ".join(f"{rat[k]:.2f}" if rat[k] is not None else "–" for k in DIRS)
              + f" | **{min(core):.2f}** |")


GAIT = [("stride_m", "stride m"), ("cadence_hz", "steps/s"), ("clearance_cm", "clearance cm"), ("knee_deg", "knee °"),
        ("knee_p10_deg", "knee p10 °"), ("roll_std_deg", "roll sd °"), ("pitch_std_deg", "pitch sd °"),
        ("bounce_cm", "bounce cm"), ("double_support", "double support")]


def gait_summary(runs):
    """Mean over the scenarios of eval_walk's gait block (steady moving ticks), plus the mean foot slip."""
    import numpy as np

    rows = [(n, d) for n, d in runs if any("gait" in r for r in d["scenarios"].values())]
    if not rows:
        return
    print("\n| run (gait, walking ticks) | " + " | ".join(h for _, h in GAIT) + " | foot slip m/s |")
    print("|---|" + "---|" * (len(GAIT) + 1))
    for n, d in rows:
        vals = []
        for k, _ in GAIT:
            v = [r["gait"][k] for r in d["scenarios"].values() if r.get("gait", {}).get(k) is not None]
            vals.append(f"{np.mean(v):.2f}" if v else "–")
        sl = [r["foot_slip_mean"] for r in d["scenarios"].values() if "foot_slip_mean" in r]
        print(f"| {n} | " + " | ".join(vals) + f" | {np.mean(sl):.3f} |")


def gest_stats(d):
    """eval_k1_mujoco result -> (mean, max) of the per-gesture arm errors (stand segments excluded)."""
    per = {}
    for s in d["segments"]:
        c = s["clip"].replace("_sonic_x1.00", "")
        if c != "stand":
            per.setdefault(c, []).append(s["arms_deg"])
    g = [sum(v) / len(v) for v in per.values()]
    return (sum(g) / len(g), max(g)) if g else (None, None)


def gest_table(paths):
    """arms° per gesture (legs° in the last column = mean over the run); stand = mean of the stand segments."""
    runs = [(Path(p).stem, json.loads(Path(p).read_text())) for p in paths]
    clips = []
    for _, d in runs:
        for s in d["segments"]:
            c = s["clip"].replace("_sonic_x1.00", "")
            if c not in clips:
                clips.append(c)
    print("| run | fall | " + " | ".join(f"{c} arms°" for c in clips) + " | gestures mean / max arms° | legs° |")
    print("|---|---|" + "---|" * (len(clips) + 2))
    for n, d in runs:
        cell, g = [], []
        for c in clips:
            v = [s["arms_deg"] for s in d["segments"] if s["clip"].replace("_sonic_x1.00", "") == c]
            cell.append(f"{sum(v) / len(v):.2f}" if v else "–")
            if v and c != "stand":
                g.append(sum(v) / len(v))
        cell.append(f"{sum(g) / len(g):.2f} / {max(g):.2f}" if g else "–")
        legs = sum(s["legs_deg"] * s["seconds"] for s in d["segments"]) / sum(s["seconds"] for s in d["segments"])
        print(f"| {n} | {'**' + str(d['fell_at']) + ' s**' if d['fell'] else '0'} | " + " | ".join(cell) + f" | {legs:.2f} |")


def main(paths, detail=False):
    if all(Path(p).stem.startswith("gest_") for p in paths):
        return gest_table(paths)
    runs = [(Path(p).stem, json.loads(Path(p).read_text())) for p in paths]
    names = list(runs[0][1]["scenarios"])
    print("| scenario | " + " | ".join(f"{n} fall / v ratio / wz ratio / slip m/s / arms° (hand+elbow cm) / legs°"
                                       for n, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for sc in names:
        row = [sc]
        for _, d in runs:
            r = d["scenarios"].get(sc)
            if r is None:
                row.append("–")
                continue
            fall = f"**{r['fell_at']} s**" if r["fell"] else "0"
            lv, wv = ratios(r)
            cm = f" ({r['arms_pos_cm']})" if "arms_pos_cm" in r else ""
            row.append(f"{fall} / {lv if lv is not None else '–'} / {wv if wv is not None else '–'} / {r.get('foot_slip_mean', '–')} / "
                       f"{r.get('arms_deg', '–')}{cm} / {r.get('legs_deg', '–') if d['backend'] != 'split' else '–'}")
        print("| " + " | ".join(row) + " |")
    print("| **falls** | " + " | ".join(f"**{d['falls']}/{len(d['scenarios'])}**" for _, d in runs) + " |")
    direction_summary(runs)
    gait_summary(runs)
    if detail:
        for n, d in runs:
            print(f"\n{n} ({d['backend']}): commanded (vx,vy,wz) -> achieved, steady part of each segment")
            for sc, r in d["scenarios"].items():
                print(f"  {sc:18s} {seg_str(r)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--detail"]
    main(args, "--detail" in sys.argv)
