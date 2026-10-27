"""Baseline for the K1 SONIC comparison: the existing ROBOTIS K1 mimic student (current frame only,
motion_command = [q_ref, dq_ref] of the present tick) on the same k1/ event gestures, in the k1/ MuJoCo loop (k1/sim.py).

    .venv/bin/python sonic_k1_train/mac/baseline_student.py --json logs/k1_baseline_student.json

Two drives per clip: "split" (the k1/ event demo: arms = student, legs + waist = loco_armdriven_r2 standing) and
"mimic" (whole body = student, the apples-to-apples setting for a whole-body SONIC policy). Metrics are the k1/
render_event.run_clip ones (fall, min pelvis z, max tilt, arm / leg joint error in degrees, best-lag arm error).
Needs k1/assets/*/exported/policy.onnx (git-ignored, see k1/README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from k1 import render_event as RE  # noqa: E402
from k1.gestures import GESTURES, ArmIK  # noqa: E402
from k1.sim import K1Split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=["stand_at_ease", "bow", "wave_hello", "welcome_open_arms", "clap",
                                                   "point_guide", "heart", "raise_both_arms"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ik = ArmIK(K1Split())
    orig = dict(GESTURES)
    rows = []
    for drive in ("split", "mimic"):
        for name in a.clips:
            def gen(ik, f=orig[name], drive=drive):
                m = f(ik)
                if drive == "mimic":
                    m.mode = "mimic"
                return m
            RE.GESTURES[name] = gen
            with tempfile.TemporaryDirectory() as td:
                r = RE.run_clip(name, td, video=False, ik=ik)
            t = r["tracking_err_deg"]
            row = {"clip": name, "drive": drive, "fell": r["fell"], "min_pelvis_z": r["min_pelvis_z"],
                   "max_tilt_deg": r["max_pelvis_tilt_deg"], "arms_deg": t["arms"], "arms_p90_deg": t["arms_p90"],
                   "legs_waist_deg": t["legs_waist"], "arms_best_lag_deg": t["arms_best_lag"],
                   "best_lag_frames": t["best_lag_frames"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            RE.GESTURES[name] = orig[name]
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
