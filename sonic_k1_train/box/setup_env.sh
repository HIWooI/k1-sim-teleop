#!/usr/bin/env bash
# GPU server: the SONIC training env, everything under $SONIC_K1_WORK (default $HOME/sonic_k1).
# Never writes to the base env (or any other existing env/project); no sudo/apt needed.
#
#   BASE_ENV=<conda env with Isaac Sim 5.1> bash setup_env.sh   # idempotent-ish: skips the clone / checkouts that exist
#
# 1 conda clone (copy, no hard links) of an existing Isaac Sim 5.1 env ($BASE_ENV) -> $SONIC_K1_WORK/env  (~20 GB)
# 2 move the clone's editable links into the base env's projects (isaaclab 2.3.0, cyclo_lab) out of the way
#   (to $SONIC_K1_WORK/attic, nothing deleted) and install our own Isaac Lab v2.3.2 checkout
# 3 gear_sonic (rsync'd snapshot of upstream b042411, git-initialised so sonic_k1 patches can be git-applied)
#   + training deps, with the pins Isaac Sim needs (numpy 1.26.0, click 8.1.7, psutil 5.9.8, websockets 12,
#   protobuf 4.25.9) -> wandb 0.18.7 (0.30 needs protobuf>=5), pyarrow 19 / datasets 3.6 (newer need numpy 2)
# 4 libusb from conda-forge (open3d imports it; system libusb missing) -> train.sh sets LD_PRELOAD
# Code sync from the Mac: sonic_k1_train/box/sync_to_box.sh, then prepare.sh.
set -euo pipefail
W=${SONIC_K1_WORK:-$HOME/sonic_k1}
E=$W/env
SP=$E/lib/python3.11/site-packages
mkdir -p $W/logs $W/src $W/attic
source "$(conda info --base)/etc/profile.d/conda.sh"

[ -x $E/bin/python ] || conda create -y -p $E --clone "${BASE_ENV:?set BASE_ENV to a conda env with Isaac Sim 5.1}" --copy \
  > $W/logs/env_clone.log 2>&1

for f in __editable__.cyclo_lab-2.0.0.pth __editable___cyclo_lab_2_0_0_finder.py \
         __editable__.isaaclab-0.47.2.pth __editable___isaaclab_0_47_2_finder.py __editable__.isaaclab_assets-0.2.3.pth \
         __editable___isaaclab_assets_0_2_3_finder.py __editable__.isaaclab_tasks-0.11.6.pth \
         __editable___isaaclab_tasks_0_11_6_finder.py cyclo_lab-2.0.0.dist-info \
         isaaclab-0.47.2.dist-info isaaclab_assets-0.2.3.dist-info isaaclab_tasks-0.11.6.dist-info; do
  [ -e $SP/$f ] && mv $SP/$f $W/attic/
done
[ -d $W/IsaacLab ] || git clone -q --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git $W/IsaacLab

cat > $W/constraints.txt <<EOF
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
numpy==1.26.0
scipy==1.15.3
warp-lang==1.17.0
protobuf==4.25.9
click==8.1.7
psutil==5.9.8
websockets==12.0
pyarrow==19.0.1
datasets==3.6.0
EOF
export PIP_CONSTRAINT=$W/constraints.txt
PIP="$E/bin/python -m pip install"
(cd $W/IsaacLab/source && $PIP --no-deps -e isaaclab -e isaaclab_assets -e isaaclab_tasks -e isaaclab_rl)
[ -d $W/src/GR00T-WholeBodyControl/gear_sonic ] || { echo "rsync gear_sonic first (sync_to_box.sh / docs/training.md)"; exit 1; }
(cd $W/src/GR00T-WholeBodyControl && $PIP --no-deps -e gear_sonic)
$PIP "hydra-core==1.3.2" "wandb==0.18.7" "trl==0.28.0" "transformers>=4.56.2" "accelerate>=1.3.0" tensorboard open3d \
     vector_quantize_pytorch easydict loguru joblib tqdm rich termcolor "tensordict>=0.8,<0.9" \
     "smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git"
$PIP "packaging==23.2"   # isaaclab_rl wants packaging<24
[ -e $E/lib/libusb-1.0.so.0 ] || conda install -y -p $E -c conda-forge libusb > $W/logs/libusb.log 2>&1
(cd /tmp && LD_PRELOAD=$E/lib/libusb-1.0.so.0 $E/bin/python -c "
import torch, isaaclab, hydra, trl, accelerate, open3d, wandb, tensordict, rich, vector_quantize_pytorch, smpl_sim
print('ok torch', torch.__version__, torch.cuda.is_available(), 'isaaclab', isaaclab.__file__)")
