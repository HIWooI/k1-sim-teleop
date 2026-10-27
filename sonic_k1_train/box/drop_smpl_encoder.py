"""On the GPU server: copy a K1 SONIC checkpoint without its SMPL-encoder weights, for a warm start whose SMPL encoder has
a different input shape (sonic_k1_small_la4_bones: 4 future SMPL frames instead of 10). ppo_trainer loads
policy_state_dict with strict=False, so the dropped encoder starts from its initialisation and everything else
(robot / teleop encoders, FSQ, decoders, critic) carries over. Optimizer / scheduler states are dropped too (a warm
start does not load them).

    env/bin/python pkg/sonic_k1_train/box/drop_smpl_encoder.py <in.pt> <out.pt>
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["policy_state_dict"]
drop = [k for k in sd if ".encoders.smpl." in k]
for k in drop:
    del sd[k]
for k in ("optimizer_state_dict", "lr_scheduler_state_dict"):
    ck.pop(k, None)
torch.save(ck, dst)
print(f"{dst}: dropped {len(drop)} smpl-encoder tensors, kept {len(sd)}")
