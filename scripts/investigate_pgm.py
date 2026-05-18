#!/usr/bin/env python3
"""Probe the FTS-Diffusion PGM: generate a segment for every SISC pattern and
check whether the diffusion latent / decoder output reproduce the conditioning
pattern (shape = net change, magnitude = range).

Run:  python scripts/investigate_pgm.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "fts-diffusion-ref")
sys.path.insert(0, REF)
os.chdir(REF)

import torch  # noqa: E402
from models.model_params import prm_params  # noqa: E402

prm_params["dataname"] = "sp500"
prm_params["k"] = 14

from models.load_models import load_ftsdiffusion  # noqa: E402
from models.utils_sampling import sampling_inputs  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

model = load_ftsdiffusion()
gen = model["generation"]
_, _, patterns = sampling_inputs()
patterns = np.asarray(patterns, dtype=float)
print(f"patterns shape = {patterns.shape}   gen.condition = {getattr(gen, 'condition', '?')}")
print()

NREP = 8
L = 10  # decoder unroll length (= the collapsed alpha)


def net_rng(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    return a[-1] - a[0], a.max() - a.min()


print(f"{'pat':>3} | {'centroid':>16} | {'diffusion z_':>22} | {'decoder x_ (->segment)':>30}")
print(f"{'':>3} | {'net':>7} {'range':>7} | {'net':>9} {'range':>10} | "
      f"{'net':>13} {'range':>13}")
print("-" * 84)

for p in range(prm_params["k"]):
    cnet, crng = net_rng(patterns[p])
    xnets, xrngs, znets, zrngs = [], [], [], []
    for _ in range(NREP):
        pat = torch.tensor(patterns[p]).unsqueeze(0).float()
        x_, z_ = gen.generate(pat, np.array([L]))
        x = x_.detach().cpu().numpy().reshape(-1)[:L]
        z = z_.detach().cpu().numpy().reshape(-1)[:L]
        xn, xr = net_rng(x)
        zn, zr = net_rng(z)
        xnets.append(xn); xrngs.append(xr); znets.append(zn); zrngs.append(zr)
    print(f"p{p:2d} | {cnet:+7.3f} {crng:7.3f} | "
          f"{np.mean(znets):+9.4f} {np.mean(zrngs):10.4f} | "
          f"{np.mean(xnets):+9.5f}±{np.std(xnets):.5f} {np.mean(xrngs):9.5f}")

print()
print("Reading: centroid = the SISC pattern the PGM is conditioned on.")
print("z_ = diffusion latent (after denoising, +pattern if condition).")
print("x_ = decoder output = the actual generated segment (before beta scaling).")
print("If x_ net/range are ~0 and identical across all patterns -> PGM is degenerate")
print("and ignores the conditioning pattern.")
