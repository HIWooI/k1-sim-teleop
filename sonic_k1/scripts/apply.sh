#!/usr/bin/env bash
# Apply sonic_k1 to a GR00T-WholeBodyControl checkout: copy overlay/** on top, then git-apply patches/*.patch.
#
#   sonic_k1/scripts/apply.sh [SONIC_ROOT]        # default: $SONIC_ROOT or ~/sonic/GR00T-WholeBodyControl
#   sonic_k1/scripts/apply.sh --check [SONIC_ROOT] # dry run: report what would change, touch nothing
#
# Idempotent: overlay files are overwritten in place; a patch that is already applied is skipped.
# Requires the meshes to be staged first (scripts/stage_assets.py), the URDF references urdf/k1/meshes/*.stl.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CHECK=0
if [[ "${1:-}" == "--check" ]]; then CHECK=1; shift; fi
ROOT="${1:-${SONIC_ROOT:-$HOME/sonic/GR00T-WholeBodyControl}}"
[[ -d "$ROOT/gear_sonic" ]] || { echo "not a GR00T-WholeBodyControl checkout: $ROOT" >&2; exit 1; }
MESHES="$HERE/overlay/gear_sonic/data/assets/robot_description/urdf/k1/meshes"
n_mesh=$(find "$MESHES" -name '*.stl' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n_mesh" -lt 25 ]]; then
  echo "only $n_mesh/25 K1 meshes staged in $MESHES — run: python sonic_k1/scripts/stage_assets.py --description <ai_sapiens_description>" >&2
  exit 1
fi

echo "[sonic_k1] overlay -> $ROOT"
(cd "$HERE/overlay" && find . -type f ! -name '.DS_Store' | sort) | while read -r f; do
  dst="$ROOT/${f#./}"
  if [[ $CHECK -eq 1 ]]; then
    if [[ ! -f "$dst" ]]; then echo "  + ${f#./}"; elif ! cmp -s "$HERE/overlay/$f" "$dst"; then echo "  ~ ${f#./}"; fi
  else
    mkdir -p "$(dirname "$dst")"; cp "$HERE/overlay/$f" "$dst"
  fi
done

for p in "$HERE"/patches/*.patch; do
  name=$(basename "$p")
  if git -C "$ROOT" apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "[sonic_k1] $name already applied"
  elif git -C "$ROOT" apply --check "$p"; then
    if [[ $CHECK -eq 1 ]]; then echo "[sonic_k1] $name would apply cleanly"; else git -C "$ROOT" apply "$p"; echo "[sonic_k1] $name applied"; fi
  else
    echo "[sonic_k1] $name does NOT apply to $ROOT (upstream moved?) — resolve by hand" >&2; exit 1
  fi
done
[[ $CHECK -eq 1 ]] && echo "[sonic_k1] dry run only" || echo "[sonic_k1] done. Smoke test on the GPU server: see docs/training.md (num_envs=16)."
