#!/usr/bin/env bash
# Mac -> GPU server: push the K1 package (sonic_k1/, incl. the git-ignored staged meshes), this directory and
# the built motion library to $SONIC_K1_WORK/pkg on the server. Never touches anything outside $SONIC_K1_WORK.
#
#   SONIC_K1_HOST=<ssh host alias> sonic_k1_train/box/sync_to_box.sh <built motion dir: robot/ heldout/ manifest.json> [set name]
# SONIC_K1_WORK: work dir on the server (default sonic_k1 = $HOME/sonic_k1 there).
# set name defaults to k1_motions (the 170-clip set of the first run); the walking set uses k1_walk_motions.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${1:?usage: sync_to_box.sh <k1_motions dir> [set name]}"
NAME="${2:-k1_motions}"
HOST="${SONIC_K1_HOST:?set SONIC_K1_HOST to the ssh host alias of the GPU server}"
W="${SONIC_K1_WORK:-sonic_k1}"
DST=$W/pkg
# a container root without CAP_DAC_OVERRIDE cannot read files with the Mac uid/permissions: never keep them
RS=(rsync -rlz --no-o --no-g --no-perms --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store)
ssh "$HOST" "mkdir -p $DST $W/logs"
"${RS[@]}" --delete "$HERE/sonic_k1/" "$HOST:$DST/sonic_k1/"
"${RS[@]}" --delete --exclude build "$HERE/sonic_k1_train/" "$HOST:$DST/sonic_k1_train/"
"${RS[@]}" --delete "$DATA/" "$HOST:$DST/$NAME/"
ssh "$HOST" "find $DST/$NAME/robot -name '*.pkl' | wc -l; find $DST/sonic_k1/overlay -name '*.stl' | wc -l"
