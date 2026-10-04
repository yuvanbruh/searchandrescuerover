"""
Convert a trained .pt policy into policy.npz (for the ROS node) and VERIFY that
the NumPy version gives the same answers as the PyTorch one.

Run from sar_training/ (with the sar_venv active):
    python gazebo_deploy/export_policy.py --ckpt runs/run1/final.pt --out policy.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from sar_rl.policy import CandidatePolicy        # noqa: E402
from rl_policy_np import NumpyCandidatePolicy    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="policy.npz")
a = ap.parse_args()

model = CandidatePolicy()
model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
model.eval()
sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()
      if not k.startswith("value_head")}
np.savez(a.out, heads=np.array(4), **sd)
print("saved", a.out)

# ---- verification: torch vs numpy on random inputs --------------------
npol = NumpyCandidatePolicy(a.out)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    cand = rng.uniform(-1, 1, (16, 13)).astype(np.float32)
    glob = rng.uniform(0, 1, (10,)).astype(np.float32)
    mask = rng.random(16) < 0.5
    if not mask.any():
        mask[rng.integers(16)] = True
    with torch.no_grad():
        lt, _ = model(torch.from_numpy(cand)[None], torch.from_numpy(mask)[None],
                      torch.from_numpy(glob)[None])
    ln = npol.logits(cand, mask, glob)
    worst = max(worst, float(np.abs(lt[0].numpy()[mask] - ln[mask]).max()))
    assert int(lt[0].argmax()) == int(np.argmax(ln)), "argmax mismatch"
print(f"VERIFIED: max logit difference torch vs numpy = {worst:.2e}")
if worst > 1e-3:
    sys.exit("difference too large - do NOT deploy this file")
