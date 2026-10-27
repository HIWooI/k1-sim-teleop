"""Markdown tables from eval_k1_mujoco.py JSONs (+ baseline_student.json): per-gesture arm / leg error and falls.

    .venv/bin/python sonic_k1_train/mac/summarize.py --sonic name=path.json [name=path.json ...] \
        --baseline logs/eval/baseline_student.json
"""
from __future__ import annotations

import argparse
import json

ORDER = ["stand", "stand_at_ease", "bow", "wave_hello", "welcome_open_arms", "clap", "point_guide", "heart",
         "raise_both_arms"]


def clip_key(label):
    for k in ORDER:
        if label == k or label.startswith(k + "_"):
            return k
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonic", nargs="+", required=True, help="name=eval.json")
    ap.add_argument("--baseline", default="")
    a = ap.parse_args()
    cols, data = [], {}
    for spec in a.sonic:
        name, path = spec.split("=", 1)
        r = json.load(open(path))
        cols.append(name)
        first_stand = True
        for s in r["segments"]:
            k = clip_key(s["clip"])
            if k == "stand":
                if not first_stand:
                    continue
                first_stand = False
            data.setdefault(k, {})[name] = s
        data.setdefault("_fell", {})[name] = r["fell_at"] if r["fell"] else "-"
    base = {}
    if a.baseline:
        for row in json.load(open(a.baseline)):
            base.setdefault(row["clip"], {})[row["drive"]] = row
    head = "| 동작 | " + " | ".join(f"{c} 팔/다리 (°)" for c in cols)
    head += " | student split 팔/다리 (°) | student mimic 팔/다리 (°) |" if base else " |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for k in ORDER:
        if k not in data and k not in base:
            continue
        row = f"| {k} | "
        row += " | ".join(f"{data[k][c]['arms_deg']:.1f} / {data[k][c]['legs_deg']:.1f}" if c in data.get(k, {}) else "–"
                          for c in cols)
        if base:
            for drive in ("split", "mimic"):
                b = base.get("stand_at_ease" if k == "stand" else k, {}).get(drive)
                row += f" | {b['arms_deg']:.1f} / {b['legs_waist_deg']:.1f}" if b else " | –"
        print(row + " |")
    print("| 낙상 | " + " | ".join(str(data["_fell"][c]) for c in cols) + (" | 0 | 0 |" if base else " |"))


if __name__ == "__main__":
    main()
