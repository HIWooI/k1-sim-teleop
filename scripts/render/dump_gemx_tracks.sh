#!/bin/sh
# GEM-X real-time worker over video files (GPU machine): every frame, causal 64-frame window, 1080p, like the live
# sender -> <out>/<stem>[$SUFFIX].npz for k1live.gemx.load_track (render_reel --perception gemx, scripts/gemx_feed.py).
# One process for all clips (models load once); existing outputs are skipped.
#   scripts/render/dump_gemx_tracks.sh <out dir> clip1.mp4 [clip2.mp4 ...]
#   env: K1LIVE_GEMX_PERCEPTION (GEM-X perception folder: demo_webcam.py, GEM-X/ inside; default
#        <repo>/third_party/gemx_perception), GEMX_PY (GEM-X venv python, default $K1LIVE_GEMX_PERCEPTION/GEM-X/.venv/bin/python),
#        REALTIME=1 (camera-rate run with drops, for latency),
#        SUFFIX (appended to the output stem, e.g. _rt)
set -e
OUT=${1:?out dir}; shift
HERE=$(cd "$(dirname "$0")/../.." && pwd)
PY=${GEMX_PY:-${K1LIVE_GEMX_PERCEPTION:-$HERE/third_party/gemx_perception}/GEM-X/.venv/bin/python}
mkdir -p "$OUT"
exec nice -n 10 "$PY" "$HERE/k1live/gemx_worker.py" --dump "$@" --out "$OUT/" --suffix "${SUFFIX:-}" ${REALTIME:+--realtime}
