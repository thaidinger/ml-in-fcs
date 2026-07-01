#!/usr/bin/env python
"""Probe the trained S&P 500 PEM next-pattern distribution.

This diagnostic supports the discussion in the paper around deterministic
argmax rollout. It loads only retained repository artifacts:

- the S&P 500 PEM checkpoint in `fts-diffusion-ref/trained_models/`
- the S&P 500 SISC labels, lengths, and subsequences in `fts-diffusion-ref/res/`

It does not require saved synthetic state trajectories.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "fts-diffusion-ref"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="sp500", choices=["sp500"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def reference_paths(checkpoint: Path | None) -> tuple[Path, Path, Path, Path]:
    labels = REF / "res/sisc_sp500_k14_l10-21_dba_kmpp_labels.csv"
    segmentation = REF / "res/sisc_sp500_k14_l10-21_dba_kmpp_segmentation.csv"
    subsequences = REF / "res/sisc_sp500_k14_l10-21_dba_kmpp_subsequences.csv"
    ckpt = checkpoint or (
        REF
        / "trained_models/pem_sp500_k14_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.pth.pth"
    )
    for path in (labels, segmentation, subsequences, ckpt):
        if not path.exists():
            raise FileNotFoundError(path)
    return labels, segmentation, subsequences, ckpt


def load_reference_states() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    labels_path, segmentation_path, subsequences_path, _ = reference_paths(None)
    labels = pd.read_csv(labels_path, index_col=0).iloc[:, 0].to_numpy(dtype=int)
    boundaries = pd.read_csv(segmentation_path, index_col=0).iloc[:, 0].to_numpy(dtype=int)
    lengths = np.diff(boundaries).astype(int)
    subsequences = pd.read_csv(subsequences_path, index_col=0).iloc[:, 0].astype(str)
    magnitudes = []
    for raw in subsequences:
        segment = np.fromstring(raw.strip().strip("[]"), sep=" ")
        if segment.size:
            magnitudes.append(float(segment.max() - segment.min()))
    magnitudes = np.asarray(magnitudes, dtype=float)
    return labels.astype(int), lengths.astype(int), magnitudes.astype(float)


def load_pem(checkpoint: Path):
    import torch

    sys.path.insert(0, str(REF))
    from models.pattern_evolution_module import PatternEvolutionModule  # noqa: E402

    device = torch.device("cpu")
    n_patterns = 14
    range_length = 12
    model = PatternEvolutionModule(
        n_patterns=n_patterns,
        range_length=range_length,
        embedding_dim=196,
        hidden_dim=32,
        device=device,
    ).to(device)

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def next_pattern_probabilities(model, device, pattern: int, length: int, magnitude: float) -> np.ndarray:
    import torch

    with torch.no_grad():
        x = torch.tensor([[float(pattern), float(length), float(magnitude)]], device=device)
        logits = model(x)[:, : model.n_patterns]
        return torch.softmax(logits, dim=1).cpu().numpy().squeeze(0)


def main() -> int:
    args = parse_args()
    _, _, _, checkpoint = reference_paths(args.checkpoint)
    labels, lengths, magnitudes = load_reference_states()
    model, device = load_pem(checkpoint)

    typical_length = int(np.clip(round(float(np.median(lengths))), 10, 21))
    typical_magnitude = float(np.median(magnitudes))
    magnitude_grid = np.quantile(magnitudes, [0.1, 0.5, 0.9]).astype(float)

    print(f"checkpoint: {checkpoint.relative_to(ROOT)}")
    print(f"patterns: {model.n_patterns}")
    print(f"typical state: length={typical_length}, magnitude={typical_magnitude:.4f}")
    print(f"real SISC pattern counts: {dict(sorted(Counter(labels).items()))}")
    print()
    print(f"{'in p':>4} | {'next p':>6} | {'maxprob':>7} | top-3")

    typical_argmax: list[int] = []
    for pattern in range(model.n_patterns):
        probs = next_pattern_probabilities(model, device, pattern, typical_length, typical_magnitude)
        next_pattern = int(probs.argmax())
        typical_argmax.append(next_pattern)
        top3 = " ".join(f"{int(idx)}:{probs[idx]:.3f}" for idx in np.argsort(-probs)[:3])
        print(f"{pattern:>4} | {next_pattern:>6} | {probs.max():>7.3f} | {top3}")

    grid_argmax: list[int] = []
    for pattern in range(model.n_patterns):
        for length in range(10, 22):
            for magnitude in magnitude_grid:
                probs = next_pattern_probabilities(model, device, pattern, length, float(magnitude))
                grid_argmax.append(int(probs.argmax()))

    print()
    print(f"argmax at typical state: {dict(sorted(Counter(typical_argmax).items()))}")
    print(f"argmax over {len(grid_argmax)} varied (p,l,m) states: {dict(sorted(Counter(grid_argmax).items()))}")
    print(f"unique varied-state argmax patterns: {sorted(set(grid_argmax))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
