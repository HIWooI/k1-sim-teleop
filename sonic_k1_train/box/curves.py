"""Training curves from a gear_sonic train_agent_trl.py stdout log (the rich per-iteration table) -> CSV + summary.

    python curves.py $SONIC_K1_WORK/logs/k1s_main.log [--csv out.csv] [--every 100]

Columns: iteration, wall time, iteration time, mean reward / episode length, the tracking errors
(Env/Metrics/motion/error_*: joint_pos rad, body/anchor pos m), episode termination shares
(time_out = clip finished without a termination; the rest are tracking-failure terminations) and the
adaptive-sampling failure statistics.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KV = re.compile(r"^[│|]?\s*([A-Za-z][\w/ .-]*?):\s+(-?[\d.]+(?:e[-+]?\d+)?)s?\b")
KEEP = {
    "Mean rewards": "reward", "Mean length": "ep_len", "Iteration time": "iter_s", "Total time": "total_s",
    "Env/Metrics/motion/error_joint_pos": "err_joint_pos_rad", "Env/Metrics/motion/error_body_pos": "err_body_pos_m",
    "Env/Metrics/motion/error_anchor_pos": "err_anchor_pos_m", "Env/Metrics/motion/error_anchor_rot": "err_anchor_rot",
    "Env/Metrics/motion/error_body_rot": "err_body_rot",
    "Env/Episode_Termination/time_out": "term_time_out", "Env/Episode_Termination/anchor_pos": "term_anchor_pos",
    "Env/Episode_Termination/anchor_ori_full": "term_anchor_ori", "Env/Episode_Termination/ee_body_pos": "term_ee_pos",
    "Env/Episode_Termination/foot_pos_xyz": "term_foot_pos",
    "Env/Episode_Reward/tracking_vr_5point_local": "rew_vr5",
    "Env/Episode_Reward/tracking_relative_body_pos": "rew_rel_body_pos",
    "Env/Episode_Reward/tracking_anchor_pos": "rew_anchor_pos",
    "Env/adp_samp/failure_rate_max": "adp_fail_max", "Env/adp_samp/failure_rate_mean": "adp_fail_mean",
    "Mean action noise std": "noise_std",
}


def parse(path):
    rows, cur = [], None
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = re.search(r"Learning iteration (\d+)", line)
        if m:
            if cur:
                rows.append(cur)
            cur = {"iteration": int(m.group(1))}
            continue
        if cur is None:
            continue
        m = re.search(r"Computation: (\d+) steps/s", line)
        if m:
            cur["steps_per_s"] = int(m.group(1))
            continue
        m = KV.search(line.strip())
        if m and m.group(1).strip() in KEEP:
            cur[KEEP[m.group(1).strip()]] = float(m.group(2))
    if cur:
        rows.append(cur)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv", default="")
    ap.add_argument("--every", type=int, default=100)
    a = ap.parse_args()
    rows = parse(a.log)
    if not rows:
        sys.exit("no iterations yet")
    cols = ["iteration"] + sorted({k for r in rows for k in r} - {"iteration"})
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)
    show = ["iteration", "total_s", "steps_per_s", "reward", "ep_len", "term_time_out", "err_joint_pos_rad",
            "err_body_pos_m", "err_anchor_pos_m", "adp_fail_max", "adp_fail_mean"]
    print(" ".join(f"{c:>14s}" for c in show))
    for r in rows:
        if r["iteration"] % a.every == 0 or r is rows[-1]:
            print(" ".join(f"{r.get(c, float('nan')):14.4g}" for c in show))


if __name__ == "__main__":
    main()
