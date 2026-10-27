#!/usr/bin/env bash
# Download the K1 policy weights from the GitHub Release (docs/weights.md) and unpack them where the service looks:
#   k1/assets/{loco_armdriven_r2,student_asset}/exported/policy.onnx               split / hybrid (standing)
#   models/k1/k1s_walk4/deploy/k1s_walk4_007000/{model_encoder,model_decoder}.onnx  hybrid walking (event setting)
# Idempotent: when every file already matches SHA256SUMS nothing is downloaded. Needs curl and tar only (no login).
#   scripts/fetch_weights.sh                 K1_WEIGHTS_TAG=weights-v1, K1_WEIGHTS_URL=<mirror> to override
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${K1_WEIGHTS_TAG:-weights-v1}
URL=${K1_WEIGHTS_URL:-https://github.com/HIWooI/k1-sim-teleop/releases/download/$TAG}
TAR=k1-policies.tar.gz
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sSfL -o "$tmp/SHA256SUMS" "$URL/SHA256SUMS"
grep " $TAR\$" "$tmp/SHA256SUMS" > "$tmp/tar.sha256"
grep -v " $TAR\$" "$tmp/SHA256SUMS" > "$tmp/files.sha256"   # repo-relative paths
if "${SHA[@]}" -c --status "$tmp/files.sha256" 2>/dev/null; then
  echo "K1 weights already in place ($TAG)"
  exit 0
fi
echo "downloading $URL/$TAR"
curl -SfL --progress-bar -o "$tmp/$TAR" "$URL/$TAR"
(cd "$tmp" && "${SHA[@]}" -c tar.sha256)
tar -xzf "$tmp/$TAR"
"${SHA[@]}" -c "$tmp/files.sha256"
echo "K1 weights ready ($TAG)"
