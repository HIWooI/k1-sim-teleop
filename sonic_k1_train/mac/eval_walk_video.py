"""Side-by-side walking video of several K1 SONIC checkpoints on the same planner scenarios, with a leg close-up row:

    top:    reference (planner, kinematic) | <variant 1> | <variant 2> ...   (full body, camera on the pelvis)
    bottom: the same, close-up of the legs (knee height)

The reference is the first variant's planner window (the planner does not depend on the robot with anchor="plan", so
it is the same for all). Gait numbers of each run (eval_walk gait block: swing clearance, foot slip, knee, sway) are
printed in the caption. CPU heavy: run on a Linux CPU host.

    python sonic_k1_train/mac/eval_walk_video.py --variant walk3=data/deploy/k1s_walk3_005000 \
        --variant walk4=data/deploy/k1s_walk4_008000 --scenarios fwd_0.6_wave right_0.3 back_0.3_random walk_turn_0.4 \
        --seconds 14 --out videos/k1_walk3_vs_walk4_legs.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sonic_k1_train" / "mac"))

from eval_walk import ALL_SCENARIOS, caption, run_sonic  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True, help="label=deploy dir")
    ap.add_argument("--scenarios", nargs="+", default=["fwd_0.6_wave", "right_0.3", "back_0.3_random", "walk_turn_0.4"])
    ap.add_argument("--seconds", type=float, default=14.0, help="per scenario (from the start of the script)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import imageio

    variants = [v.split("=", 1) for v in a.variant]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, quality=7, macro_block_size=1)
    stats = {}
    for sc in a.scenarios:
        segs, arms = ALL_SCENARIOS[sc]
        runs = []
        for lab, vdir in variants:
            r, fr = run_sonic("k1", Path(vdir), segs, arms, "plan", True, 1234, legs=True, duration=a.seconds)
            g = r.get("gait", {})
            stats.setdefault(lab, {})[sc] = {"fell": r["fell"], "foot_slip_mean": r.get("foot_slip_mean"), **g}
            runs.append((lab, r, fr))
            print(sc, lab, json.dumps(stats[lab][sc]), flush=True)
        n = min(len(fr) for _, _, fr in runs)
        for i in range(n):
            ts, gref, _, c, gref_l, _ = runs[0][2][i]
            top = [caption(gref, "reference (planner)")]
            bot = [gref_l]
            for lab, r, fr in runs:
                g = r.get("gait", {})
                txt = f"{lab}  clearance {g.get('clearance_cm', '-')} cm  slip {r.get('foot_slip_mean', '-')} m/s"
                top.append(caption(fr[i][2], txt))
                bot.append(fr[i][5])
            frame = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
            writer.append_data(caption(frame, f"{sc}  t={ts:4.1f}s  cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}",
                                       frame.shape[0] - 10))
        runs = None
    writer.close()
    print("video", a.out)
    if a.json:
        Path(a.json).write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
