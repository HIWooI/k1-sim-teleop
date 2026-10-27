#!/usr/bin/env bash
# On the GPU server: training set of the k1s_bones run = the walk2 set (planner walking + gestures x3 + the first-run
# clips; no SMPL -> SMPL encoder masked) + the BONES human set (K1 robot clips with SMPL pairs) + the live-arm set
# (planner walking / standing with people's arms from the live HumanMapper chain, planner_arms_live.py, twice), as
# symlinks, so no source folder is touched. motion_lib globs robot/ recursively and keys clips by file basename; smpl/ stays flat
# (the loader only looks for smpl/<basename>.pkl).
#
#   bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/make_bones_train.sh [human set name, default k1_bones_motions]
# OUT=<dir name> (default k1_bones_train); WALK2_COPIES=n (default 1): put the walk2 set in n times (extra copies as _wN-suffixed symlinks), when the human clips
# crowd out the planner walking (backward / sideways regressed in k1s_bones it500 with 884 walk2 vs 4743 BONES clips).
set -euo pipefail
P=${SONIC_K1_WORK:-$HOME/sonic_k1}/pkg
H=$P/${1:-k1_bones_motions}
W=$P/k1_walk2_motions
LA=$P/k1_livearms_motions
D=$P/${OUT:-k1_bones_train}   # OUT=<name>: build elsewhere (never rebuild the set of a running job: clips load lazily)
[ -d "$H/robot" ] && [ -d "$H/smpl" ] || { echo "missing $H/robot or $H/smpl"; exit 1; }
rm -rf "$D"; mkdir -p "$D/robot" "$D/heldout/robot"
ln -sfn "$W/robot" "$D/robot/walk2"
ln -sfn "$H/robot" "$D/robot/human"
ln -sfn "$H/smpl" "$D/smpl"
ln -sfn "$W/heldout" "$D/heldout/robot/walk2"
for k in $(seq 2 "${WALK2_COPIES:-1}"); do
  mkdir -p "$D/robot/walk2_x$k"
  for f in $(find -L "$W/robot" -name '*.pkl'); do ln -sfn "$(readlink -f "$f")" "$D/robot/walk2_x$k/$(basename "$f" .pkl)_w$k.pkl"; done
done
if [ -d "$LA/robot" ]; then   # live-arm clips twice (second copy: _x2 names; motion_lib keys clips by file name)
  ln -sfn "$LA/robot" "$D/robot/livearms"
  mkdir -p "$D/robot/livearms_x2"
  for f in $(find "$LA/robot" -name '*.pkl'); do ln -sfn "$f" "$D/robot/livearms_x2/$(basename "$f" .pkl)_x2.pkl"; done
  ln -sfn "$LA/heldout" "$D/heldout/robot/livearms"
fi
if [ -d "$H/heldout/robot" ]; then ln -sfn "$H/heldout/robot" "$D/heldout/robot/human"; ln -sfn "$H/heldout/smpl" "$D/heldout/smpl"; fi
n_r=$(find -L "$D/robot" -name '*.pkl' | wc -l); n_h=$(find -L "$H/robot" -name '*.pkl' | wc -l); n_s=$(find -L "$D/smpl" -maxdepth 1 -name '*.pkl' | wc -l)
dups=$(find -L "$D/robot" -name '*.pkl' -printf '%f\n' | sort | uniq -d | wc -l)
echo "$D: $n_r robot PKLs ($n_h human), $n_s SMPL, duplicate basenames: $dups"
