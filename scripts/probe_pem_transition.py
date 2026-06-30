"""Probe the trained PEM next-pattern distribution.

Refutes the "near-absorbing self-transition" hypothesis for the PEM collapse: the
pattern head is near-uniform, peaked by a thin margin on the data mode; the collapse
comes from deterministic argmax, not a learned absorbing transition.

Run:  python scripts/probe_pem_transition.py
"""
import sys, glob, numpy as np, torch

REPO = "/home/deli/ETH_projects/ml-in-fcs"
sys.path.insert(0, f"{REPO}/fts-diffusion-ref")
from models.pattern_evolution_module import PatternEvolutionModule  # noqa: E402

dev = torch.device("cpu")
ckpt = glob.glob(f"{REPO}/fts-diffusion-ref/trained_models/pem_sp500_k14_*.pth.pt")[0]
model = torch.load(ckpt, map_location=dev, weights_only=False)  # our own checkpoint
model.eval(); model.device = dev
K = model.n_patterns

states = np.load(f"{REPO}/synthetic/sp500/k14/split/run_00_states.npy")
l_typ, m_typ = float(np.median(states[:, 1])), float(np.median(states[:, 2]))
print(f"checkpoint: {ckpt.split('/')[-1]}")
print(f"probe at median length={l_typ:.0f}, magnitude={m_typ:.2f}  (uniform softmax = {1/K:.3f})\n")
print(f"{'in p':>4} | {'next p':>6} | {'maxprob':>7} | top-3")
ams = []
with torch.no_grad():
    for p in range(K):
        prob = torch.softmax(model(torch.tensor([[float(p), l_typ, m_typ]]))[:, :K], 1).squeeze(0).numpy()
        ams.append(int(prob.argmax()))
        t3 = " ".join(f"{int(i)}:{prob[i]:.2f}" for i in np.argsort(-prob)[:3])
        print(f"{p:>4} | {ams[-1]:>6} | {prob.max():>7.3f} | {t3}")

outs = set()
with torch.no_grad():
    for p in range(K):
        for l in range(10, 22):
            for m in (m_typ * 0.2, m_typ, m_typ * 5):
                outs.add(int(torch.argmax(model(torch.tensor([[float(p), float(l), float(m)]]))[:, :K]).item()))
print(f"\nargmax over the 14 input patterns : {sorted(set(ams))}")
print(f"argmax over 504 varied (p,l,m)    : {sorted(outs)}  (all high-frequency SISC patterns)")
