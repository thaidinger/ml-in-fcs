#!/usr/bin/env python3
"""Step-by-step replay of FTS-Diffusion synthetic generation, from saved data.

Uses the trajectories the team already saved (no model re-run):
  synthetic/<asset>/k<K>/<protocol>/run_<r>_states.npy      -> (M+1, 3) = (p, alpha, beta)
  synthetic/<asset>/k<K>/<protocol>/run_<r>_{continuous,syn}.npy  -> the price series

For every segment it prints a trace of which component runs and the state it
transitions through (PEM -> diffusion -> decoder -> beta-scale+anchor -> append),
and renders a GIF of the price series being assembled segment by segment.

Examples
  python scripts/animate_generation.py                       # goog k11 split, run 0
  python scripts/animate_generation.py --asset sp500 --k 14
  python scripts/animate_generation.py --detail 20 --anim-segments 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="goog")
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--protocol", default="split", choices=["split", "single"])
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--detail", type=int, default=14,
                    help="segments traced in full detail in the console")
    ap.add_argument("--anim-segments", type=int, default=32,
                    help="segments shown in the animation")
    ap.add_argument("--out", default=None, help="output GIF path")
    return ap.parse_args()


def load(asset, k, protocol, run):
    d = os.path.join(ROOT, "synthetic", asset, f"k{k}", protocol)
    states = np.load(os.path.join(d, f"run_{run:02d}_states.npy")).astype(float)
    traj_name = "continuous" if protocol == "split" else "syn"
    series = np.load(os.path.join(d, f"run_{run:02d}_{traj_name}.npy")).astype(float)
    return states, series


def segment_bounds(states, T):
    """(start, end) of every segment in the trajectory; segment 0 = init segment."""
    lengths = states[:, 1].astype(int)
    cum = np.concatenate([[0], np.cumsum(lengths)])
    bounds = []
    for m in range(len(lengths)):
        a, b = cum[m], min(cum[m + 1], T)
        bounds.append((a, b))
        if b >= T:
            break
    return bounds


# ---------------------------------------------------------------- text trace
def trace(states, series, bounds, detail):
    p0, a0, b0 = states[0]
    print("=" * 70)
    print(f"INIT  seed the series with the initial real segment")
    print(f"      state S0 = (pattern p={int(p0)}, alpha={int(a0)}, beta={b0:.3f})")
    print(f"      series starts at y={series[0]:.3f}, length {bounds[0][1]}")
    print("=" * 70)

    n = min(detail, len(bounds) - 1)
    for m in range(1, n + 1):
        pp, pa, pb = states[m - 1]
        cp, ca, cb = states[m]
        a, b = bounds[m]
        seg = series[a:b]
        if len(seg) == 0:
            break
        print(f"\n----- segment {m} "
              f"(days {a}-{b}) " + "-" * (46 - len(str(m))))
        print(f" [PEM  | evolution]  S{m-1}=(p={int(pp):2d}, a={int(pa)}, b={pb:.3f})"
              f"  --->  S{m}=(p={int(cp):2d}, a={int(ca)}, b={cb:.3f})")
        print(f" [PGM  | diffusion]  denoise a latent shape, conditioned on "
              f"pattern p={int(cp)}")
        print(f" [PGM  | decoder]    unroll the LSTM alpha={int(ca)} steps  "
              f"--->  unit-magnitude segment (len {len(seg)})")
        print(f" [x beta | anchor]   scale x beta={cb:.3f}, then shift to start "
              f"at series end y={seg[0]:.3f}")
        print(f" [append]            segment spans [{seg[0]:.3f} -> {seg[-1]:.3f}]"
              f"   net {seg[-1]-seg[0]:+.3f}   series length {a} ---> {b}")

    # --- summary over the whole run ---
    patt = states[1:, 1 * 0].astype(int)  # column 0 = pattern
    uniq, cnt = np.unique(patt, return_counts=True)
    order = np.argsort(-cnt)
    print("\n" + "=" * 70)
    print(f"SUMMARY  {len(patt)} generated segments")
    print("  pattern usage: " + ", ".join(
        f"p={int(uniq[i])}:{cnt[i]} ({100*cnt[i]/len(patt):.1f}%)" for i in order))
    print(f"  first 24 patterns: {list(map(int, patt[:24]))}")
    if len(uniq) <= 2:
        print("  --> the PEM has COLLAPSED: only "
              f"{len(uniq)} distinct pattern(s) ever visited.")
    print("=" * 70)


# ---------------------------------------------------------------- animation
def animate(states, series, bounds, n_seg, out):
    n_seg = min(n_seg, len(bounds) - 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.6),
                                   gridspec_kw={"height_ratios": [3, 1]})
    patt_all = states[:, 0].astype(int)
    end_max = bounds[n_seg][1]
    y = series[:end_max]
    ymin, ymax = y.min(), y.max()
    pad = 0.08 * (ymax - ymin + 1e-9)

    def update(m):
        ax1.clear(); ax2.clear()
        a, b = bounds[m]
        # series so far
        ax1.plot(np.arange(b), series[:b], color="steelblue", lw=1.3)
        if m == 0:
            ax1.plot(np.arange(a, b), series[a:b], color="gray", lw=2.4,
                     label="init segment")
        else:
            ax1.plot(np.arange(a, b), series[a:b], color="#d62728", lw=2.6,
                     label=f"segment {m} just appended")
        ax1.set_xlim(-5, end_max + 5)
        ax1.set_ylim(ymin - pad, ymax + pad)
        ax1.set_ylabel("price")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(alpha=0.3)

        cp, ca, cb = states[m]
        if m == 0:
            head = f"INIT  —  seed state S0 = (p={int(cp)}, alpha={int(ca)}, beta={cb:.3f})"
            log = "the series is seeded with the initial real segment"
        else:
            pp, pa, pb = states[m - 1]
            head = (f"segment {m}  —  PEM:  (p={int(pp)}, a={int(pa)}, b={pb:.2f})"
                    f"  -->  (p={int(cp)}, a={int(ca)}, b={cb:.2f})")
            log = (f"diffusion: denoise | pattern p={int(cp)}      "
                   f"decoder: unroll alpha={int(ca)} steps      "
                   f"x beta={cb:.2f} + anchor to y={series[a]:.2f}")
        ax1.set_title(head, fontsize=11, fontweight="bold")
        ax1.text(0.015, 0.04, log, transform=ax1.transAxes, fontsize=8.5,
                 family="monospace", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e6", ec="gray"))

        # pattern chain
        xs = np.arange(m + 1)
        ax2.step(xs, patt_all[:m + 1], where="mid", color="darkblue", lw=1.4)
        ax2.scatter(xs, patt_all[:m + 1], s=22, color="#d62728", zorder=3)
        ax2.set_xlim(-0.5, n_seg + 0.5)
        ax2.set_ylim(patt_all[:n_seg + 1].min() - 1, patt_all[:n_seg + 1].max() + 1)
        ax2.set_xlabel("segment index")
        ax2.set_ylabel("pattern  p")
        ax2.set_title("state chain — pattern visited at each segment "
                      "(flat / 2-cycle = PEM collapse)", fontsize=9)
        ax2.grid(alpha=0.3)
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=n_seg + 1, interval=700)
    anim.save(out, writer=PillowWriter(fps=1.6))
    plt.close(fig)
    print(f"\nanimation written: {out}  ({n_seg} segments)")


def main():
    args = parse_args()
    states, series = load(args.asset, args.k, args.protocol, args.run)
    bounds = segment_bounds(states, len(series))
    tag = f"{args.asset}_k{args.k}_{args.protocol}_run{args.run:02d}"
    out = args.out or os.path.join(ROOT, "figures", "synthetic_trajectories",
                                   f"generation_{tag}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"\nReplaying generation: {args.asset} k{args.k} "
          f"{args.protocol} run {args.run}  "
          f"({len(bounds)-1} segments, series length {len(series)})")
    trace(states, series, bounds, args.detail)
    animate(states, series, bounds, args.anim_segments, out)


if __name__ == "__main__":
    main()
