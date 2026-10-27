#!/usr/bin/env bash
# On the GPU server: apply the K1 embodiment (sonic_k1: overlay + patches) and the sonic_k1_train overlay
# (small-model experiment configs) to the server's own SONIC checkout. Idempotent.
#
#   bash $SONIC_K1_WORK/pkg/sonic_k1_train/box/prepare.sh      (SONIC_K1_WORK default $HOME/sonic_k1)
set -euo pipefail
W=${SONIC_K1_WORK:-$HOME/sonic_k1}
ROOT=$W/src/GR00T-WholeBodyControl   # gear_sonic snapshot of upstream b042411 (git-initialised for the patches)
PKG=$W/pkg
bash "$PKG/sonic_k1/scripts/apply.sh" "$ROOT"
(cd "$PKG/sonic_k1_train/overlay" && find . -type f | sort) | while read -r f; do
  mkdir -p "$(dirname "$ROOT/${f#./}")"; cp "$PKG/sonic_k1_train/overlay/$f" "$ROOT/${f#./}"; echo "  sonic_k1_train overlay: ${f#./}"
done
for p in "$PKG"/sonic_k1_train/patches/*.patch; do   # evaluation fixes (G1-only names)
  if git -C "$ROOT" apply --reverse --check "$p" >/dev/null 2>&1; then echo "  $(basename "$p") already applied"
  else git -C "$ROOT" apply "$p" && echo "  $(basename "$p") applied"; fi
done
git -C "$ROOT" status --short | sed 's/^/    /'
echo "motions: $(find $PKG/k1_motions/robot -name '*.pkl' | wc -l) train PKLs, $(find $PKG/k1_motions/heldout -name '*.pkl' | wc -l) held-out"
