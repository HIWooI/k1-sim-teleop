#!/usr/bin/env bash
# Run every example reference motion through the M1 loop (headless) and collect JSON metrics.
set -u
cd "$(dirname "$0")/.."
VARIANT=${1:-release}
OUT=logs/m1_${VARIANT}
mkdir -p "$OUT"
for m in $(.venv/bin/python scripts/run_reference.py --list); do
  .venv/bin/python -u scripts/run_reference.py --variant "$VARIANT" --motion "$m" --min-duration 20 \
     --json "$OUT/$m.json" > "$OUT/$m.log" 2>&1
  echo "$m done"
done
.venv/bin/python - "$OUT" <<'PY'
import json, sys, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    r = json.load(open(f))
    t = r["tracking_lag0"]
    rows.append((r["motion"], r["fell"], r["min_pelvis_z_while_standing"], r["max_pelvis_tilt_deg"], t.get("all_deg", float("nan")), t.get("legs_deg", float("nan")), t.get("arms_deg", float("nan")), t.get("root_ori_deg", float("nan"))))
print(f"{'motion':40s} fell  minZstand  maxTilt  all°   legs°  arms°  root°")
for r in rows:
    print(f"{r[0]:40s} {str(r[1]):5s} {r[2]:8.3f} {r[3]:7.1f} {r[4]:6.2f} {r[5]:6.2f} {r[6]:6.2f} {r[7]:6.2f}")
PY
