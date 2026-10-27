#!/bin/bash
# BONES-SEED -> K1 human-encoder training set, after bones_fetch.py has put the archives on the GPU server.
# BONES-SEED is under the BONES-SEED license (https://bones.studio/info/seed-license); nothing of it is committed here.
# Run on the Mac from the repo root (it only orchestrates). This is the flow used for k1_bones_motions:
#
#   SONIC_K1_HOST=<GPU server> SONIC_K1_CPU_HOST=<CPU host> SONIC_K1_CPU_REPO=<repo checkout on the CPU host> \
#     sonic_k1_train/box/bones_pipeline.sh k1_bones_motions [max_hours=10] [workers=8]
#   optional: SONIC_K1_WORK (work dir on the GPU server, default sonic_k1 under its home),
#             SONIC_K1_CPU_AI_SAPIENS (AI_SAPIENS_ROOT on the CPU host, if not the checkout's third_party/ai_sapiens)
#
#   1 metadata CSV GPU server -> CPU host, select on the CPU host (sonic_k1_train/data/bones_select.py)
#   2 GPU server: extract only the selected G1 CSVs and SOMA BVHs (bones_extract.py, ionice idle, nice 15)
#   3 CSVs GPU server -> CPU host (gzip stream), robot half on the CPU host
#     (bones_to_k1.py convert-robot, 8 cores / 16 GB scope), drop unreachable-feet clips, robot PKLs -> GPU server
#   4 person half on the GPU server where the BVHs are (bones_human_part.py with the training env read-only, numpy only)
#   5 drop deep-bend takes, GEM-X extras from the smoke set, check
# A container root without CAP_DAC_OVERRIDE / CAP_CHOWN cannot use files that keep a foreign uid / 700 modes: every tar
# extracted on the GPU server uses --no-same-owner --no-same-permissions. No token or signed URL is involved here.
set -euo pipefail
SET=${1:?set name}
HOURS=${2:-10}
WORKERS=${3:-8}   # CPU host budget: 8 cores, 16 GB (systemd scope below)
GPU=${SONIC_K1_HOST:?set SONIC_K1_HOST to the ssh host alias of the GPU server}
CPU=${SONIC_K1_CPU_HOST:?set SONIC_K1_CPU_HOST to the ssh host alias of the CPU conversion host}
SH=${SONIC_K1_CPU_REPO:?set SONIC_K1_CPU_REPO to the repo checkout on the CPU host (with its .venv)}
W=$(ssh "$GPU" "mkdir -p ${SONIC_K1_WORK:-sonic_k1} && cd ${SONIC_K1_WORK:-sonic_k1} && pwd")   # absolute on the server
BOX=$W/data/bones_seed
PKG=$W/pkg/$SET
CONV=$W/bones_conv
PY=$W/env/bin/python   # used read-only (numpy / scipy / joblib)
ENV="export MUJOCO_GL=egl ${SONIC_K1_CPU_AI_SAPIENS:+AI_SAPIENS_ROOT=$SONIC_K1_CPU_AI_SAPIENS};"
SCOPE="systemd-run --user --scope -q -p CPUQuota=$((WORKERS * 100))% -p MemoryMax=16G nice -n 10"
UNTAR="tar --no-same-owner --no-same-permissions"

echo "[1] metadata -> CPU host, select"
ssh "$CPU" "mkdir -p $SH/data/bones_meta $SH/data/$SET"
ssh "$GPU" "cat $BOX/metadata/seed_metadata_v004.csv" | ssh "$CPU" "cat > $SH/data/bones_meta/seed_metadata_v004.csv"
ssh "$CPU" "cd $SH && $ENV .venv/bin/python sonic_k1_train/data/bones_to_k1.py select \
    --meta data/bones_meta/seed_metadata_v004.csv --out data/$SET --max-hours $HOURS | sed -n 3,20p"
ssh "$CPU" "cat $SH/data/$SET/selection.json" | ssh "$GPU" "mkdir -p $BOX/sel_$SET && cat > $BOX/sel_$SET/selection.json"

echo "[2] extract the selected files on the GPU server"
scp -q sonic_k1_train/box/bones_extract.py "$GPU:$BOX/sel_$SET/"
for pair in "g1.tar.gz g1" "soma_uniform.tar.gz bvh"; do
  set -- $pair
  ssh "$GPU" "cd $BOX && ionice -c3 nice -n 15 python3 sel_$SET/bones_extract.py --archive $1 \
      --selection sel_$SET/selection.json --field $2 --dest subset_$SET | tail -1"
done

echo "[3] robot half on the CPU host"
ssh "$CPU" "mkdir -p $SH/data/bones_src_$SET"
ssh "$GPU" "ionice -c3 nice -n 15 tar -C $BOX/subset_$SET -cf - g1 | gzip -1" | ssh "$CPU" "tar -C $SH/data/bones_src_$SET -xzf -"
ssh "$CPU" "cd $SH && $ENV $SCOPE .venv/bin/python sonic_k1_train/data/bones_to_k1.py convert-robot \
    --bones data/bones_src_$SET --out data/$SET --workers $WORKERS 2>&1 | tail -1"
ssh "$CPU" "cd $SH && python3 - <<'EOF'
import json, os
S = 'data/$SET'
rows = [json.loads(l) for l in open(S + '/robot_clips.jsonl')]
drop = [r for r in rows if 'rejected' not in r and not r.get('skipped') and r['ik_err_mm_p99'] > 50]
for r in drop:
    p = (S + '/heldout' if r['heldout'] else S) + '/robot/%s/%s.pkl' % (r['group'], r['key'])
    if os.path.exists(p):
        os.remove(p)
json.dump(sorted(r['key'] for r in drop), open(S + '/dropped_ik50.json', 'w'))
print('dropped (foot IK > 50 mm):', len(drop))
EOF"
ssh "$GPU" "mkdir -p $PKG"
ssh "$CPU" "cd $SH/data/$SET && tar -cf - robot heldout/robot robot_clips.jsonl selection.json selection.md dropped_ik50.json" | \
    ssh "$GPU" "$UNTAR -C $PKG -xf -"

echo "[4] person half on the GPU server"
ssh "$GPU" "mkdir -p $CONV/k1live $CONV/sonic_k1_train/data $CONV/models"
tar -cf - --exclude __pycache__ sonic k1live/__init__.py k1live/smpl_stream.py sonic_k1_train/data/human_smpl.py \
    sonic_k1_train/data/bones_human_part.py models/human_joints_info.npz | ssh "$GPU" "$UNTAR -C $CONV -xf -"
ssh "$GPU" "cd $PKG && python3 -c \"
import json
drop = set(json.load(open('dropped_ik50.json')))
s = json.load(open('selection.json')); s['clips'] = [c for c in s['clips'] if c['key'] not in drop]
json.dump(s, open('$BOX/sel_$SET/selection_kept.json', 'w'))\""
ssh "$GPU" "cd $CONV && ionice -c3 nice -n 15 $PY -I sonic_k1_train/data/bones_human_part.py convert \
    --bones $BOX/subset_$SET --selection $BOX/sel_$SET/selection_kept.json --out $PKG --workers 4 | tail -1"

echo "[5] deep bends out, GEM-X extras in, check"
ssh "$GPU" "cd $PKG && python3 - <<'EOF'
import glob, json, os, shutil
gone = []
for p in glob.glob('robot/*/*.pkl') + glob.glob('heldout/robot/*/*.pkl'):
    k = os.path.basename(p)[:-4]
    if k.startswith(('body_check', 'body_stretch')):   # trunk near horizontal: yaw undefined, K1 cannot bend so far
        os.remove(p)
        gone.append(k)
        for s in ('smpl/%s.pkl' % k, 'heldout/smpl/%s.pkl' % k):
            if os.path.exists(s):
                os.remove(s)
json.dump(sorted(gone), open('dropped_deep_bend.json', 'w'))
S = '$W/pkg/k1_bones_human_smoke'
have = {os.path.basename(p)[:-4] for p in glob.glob('robot/*/*.pkl') + glob.glob('heldout/robot/*/*.pkl')}
for grp in ('gemx', 'bones_sample'):
    os.makedirs('robot/' + grp, exist_ok=True)
    for p in glob.glob(S + '/robot/%s/*.pkl' % grp):
        k = os.path.basename(p)[:-4]
        if k not in have:
            shutil.copyfile(p, 'robot/%s/%s.pkl' % (grp, k))
            shutil.copyfile(S + '/smpl/%s.pkl' % k, 'smpl/%s.pkl' % k)
print('deep bends dropped:', len(gone))
EOF"
ssh "$GPU" "cd $CONV && ionice -c3 nice -n 15 $PY -I sonic_k1_train/data/bones_human_part.py check --out $PKG --heading 5000 | head -30"
