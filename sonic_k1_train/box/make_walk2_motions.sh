#!/usr/bin/env bash
# On the GPU server: motion set of the k1s_walk2 fine-tune = pkg/k1_walk_motions with the 84 gesture clips three times
# (symlinks; motion_lib keys clips by file name, so the copies get _x1 / _x2 suffixes). Adaptive sampling draws clips
# by failure count, and gestures rarely fail next to walking, so k1s_walk drew them far less often than k1s_la4 did
# (84 / 716 clips vs 84 / 170) and their arm error stalled at ~4.9 deg mean (k1s_la4 3.5).
#
#   bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/make_walk2_motions.sh      (SONIC_K1_WORK default $HOME/sonic_k1)
set -euo pipefail
P=${SONIC_K1_WORK:-$HOME/sonic_k1}/pkg
S=$P/k1_walk_motions
D=$P/k1_walk2_motions
mkdir -p $D/robot
for d in $S/robot/*/; do
  n=$(basename $d); mkdir -p $D/robot/$n
  for f in $d*.pkl; do ln -sfn $f $D/robot/$n/$(basename $f); done
done
for k in 1 2; do
  mkdir -p $D/robot/gestures_x$k
  for f in $S/robot/gestures/*.pkl; do b=$(basename $f .pkl); ln -sfn $f $D/robot/gestures_x$k/${b}_x$k.pkl; done
done
ln -sfn $S/heldout $D/heldout
cp $S/clips.json $S/manifest.json $D/ 2>/dev/null || true
echo "$(find -L $D/robot -name '*.pkl' | wc -l) PKLs in $D/robot"
