#!/usr/bin/env bash
# Concatenate selected side-by-side clips into one presentation reel (uniform 1920x720,
# light letterbox, first N seconds of each after the stream starts).
#   scripts/make_reel.sh videos/event_sbs_best_reel.mp4 9 pexels_4586958 mixkit_43813_360 ...
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; SECS=$2; shift 2
inputs=(); filters=""; i=0
for c in "$@"; do
  inputs+=(-ss 0.6 -t "$SECS" -i "videos/event_sbs_$c.mp4")
  filters+="[$i:v]scale=1920:720:force_original_aspect_ratio=decrease,pad=1920:720:(ow-iw)/2:(oh-ih)/2:color=0xf5f5f7,setsar=1,fps=25,fade=t=in:st=0:d=0.4,fade=t=out:st=$(echo "$SECS-0.4" | bc):d=0.4[v$i];"
  i=$((i+1))
done
concat=""; for ((k=0; k<i; k++)); do concat+="[v$k]"; done
ffmpeg -loglevel error -y "${inputs[@]}" -filter_complex "${filters}${concat}concat=n=$i:v=1:a=0[out]" \
  -map "[out]" -c:v libx264 -crf 20 -pix_fmt yuv420p "$OUT"
echo "reel -> $OUT ($i clips x ${SECS}s)"
