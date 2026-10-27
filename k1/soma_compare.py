"""Markdown comparison table of k1/run_video.py runs (MediaPipe vs GEM-X/SOMA variants) from their JSONs.

    .venv/bin/python -m k1.soma_compare --dir videos logs/k1_soma_metrics --variants "" _soma_ik _soma_csv _soma_mlp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from k1.run_video import PRESET

LABEL = {"": "MediaPipe→IK", "_soma_ik": "SOMA→IK", "_soma_ik_wr": "SOMA→IK+CSV 손목", "_soma_ik_s1": "SOMA→IK (EMA 없음)", "_soma_dir": "SOMA→팔 방향 IK", "_soma_dir_wr": "SOMA→팔 방향 IK+CSV 손목", "_soma_csv": "soma-retargeter CSV",
         "_soma_mlp": "증류 MLP", "_soma_csv_mimic": "CSV 전신 mimic", "_soma_mlp_mimic": "MLP 전신 mimic"}


def row(m):
    a2 = m.get("arm2d_vs_human_deg", {})
    f = lambda k: f"{a2[k]['mean']:.1f}" if k in a2 and a2[k]["mean"] == a2[k]["mean"] else "–"  # noqa: E731
    t = m["tracking_err_deg"]
    return [("낙상 " + str(m["fell_at_s"])) if m["fell"] else "0", f"{m['max_pelvis_tilt_deg']:.1f}",
            f"{t['arms']:.2f} / {t['arms_p90']:.2f}", f("mediapipe"), f("vitpose"),
            f"{m['arm_acc_rms_rad_s2']['robot']:.1f}" if "arm_acc_rms_rad_s2" in m else "–"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, nargs="+", default=[Path("logs/k1_soma_metrics")],
                    help="folders searched in order for k1_sbs_<clip><variant>.json")
    ap.add_argument("--variants", nargs="*", default=["", "_soma_ik", "_soma_csv", "_soma_mlp"])
    ap.add_argument("--clips", nargs="*", default=[n for n, _ in PRESET])
    a = ap.parse_args()
    head = ["클립", "경로", "낙상", "골반 최대 기울기°", "팔 추종 오차° 평균/p90", "2D 팔 각도 오차° (MediaPipe 2D)",
            "2D 팔 각도 오차° (VitPose-77)", "팔 가속 RMS rad/s²"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    agg = {v: [] for v in a.variants}
    for c in a.clips:
        for v in a.variants:
            f = next((d / f"k1_sbs_{c}{v}.json" for d in a.dir if (d / f"k1_sbs_{c}{v}.json").exists()), None)
            if f is None:
                continue
            m = json.loads(f.read_text())
            print(f"| {c} | {LABEL.get(v, v)} | " + " | ".join(row(m)) + " |")
            a2 = m.get("arm2d_vs_human_deg", {})
            agg[v].append((m["fell"], m["max_pelvis_tilt_deg"], m["tracking_err_deg"]["arms"],
                           a2.get("mediapipe", {}).get("mean", np.nan), a2.get("vitpose", {}).get("mean", np.nan),
                           m.get("arm_acc_rms_rad_s2", {}).get("robot", np.nan)))
    print()
    print("| 경로 | 클립 수 | 낙상 | 최대 기울기° (최대) | 팔 추종 오차° | 2D 오차° (MediaPipe 2D) | 2D 오차° (VitPose-77) | 팔 가속 RMS |")
    print("|---|---|---|---|---|---|---|---|")
    for v, rs in agg.items():
        if not rs:
            continue
        rs = [tuple(np.nan if x is None else x for x in r) for r in rs]
        r = np.array(rs, dtype=float)
        print(f"| {LABEL.get(v, v)} | {len(r)} | {int(r[:, 0].sum())} | {np.nanmax(r[:, 1]):.1f} | "
              f"{np.nanmean(r[:, 2]):.2f} | {np.nanmean(r[:, 3]):.1f} | {np.nanmean(r[:, 4]):.1f} | {np.nanmean(r[:, 5]):.1f} |")


if __name__ == "__main__":
    main()
