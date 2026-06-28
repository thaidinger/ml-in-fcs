from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "fts-diffusion-ref" / "res"
OUT = ROOT / "reports" / "generated_outputs" / "04_presentation_figures"

COLORS = {
    "ink": "#1f2933",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "green": "#2f855a",
    "blue": "#2563eb",
    "orange": "#d97706",
    "red": "#c2410c",
    "gray": "#6b7280",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "legend.frameon": False,
        }
    )


def parse_array(text: str) -> np.ndarray:
    cleaned = str(text).replace("[", " ").replace("]", " ").replace("\n", " ")
    values = np.fromstring(cleaned, sep=" ")
    if values.size == 0:
        raise ValueError(f"Could not parse subsequence array from: {text!r}")
    return values.astype(float)


def centered_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    scale = np.nanmax(np.abs(x))
    return x / scale if scale > 0 else x


def diffusion_schedule(n_steps: int = 30, min_beta: float = 1e-4, max_beta: float = 0.02) -> np.ndarray:
    betas = np.linspace(min_beta, max_beta, n_steps)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def load_segment(asset: str, segment_index: int) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    prefix = RES / f"sisc_{asset}_k11_l10-21_dba_kmpp"
    subsequences = pd.read_csv(f"{prefix}_subsequences.csv", index_col=0)
    labels = pd.read_csv(f"{prefix}_labels.csv", index_col=0)
    segmentation = pd.read_csv(f"{prefix}_segmentation.csv", index_col=0)
    centroids = pd.read_csv(f"{prefix}_centroids.csv", index_col=0)

    if segment_index < 0 or segment_index >= len(subsequences):
        raise IndexError(f"{asset} has {len(subsequences)} segments; requested {segment_index}.")

    segment = parse_array(subsequences.iloc[segment_index, 0])
    pattern_id = int(labels.iloc[segment_index, 0])
    centroid = centroids.iloc[pattern_id].to_numpy(dtype=float)[: len(segment)]
    start = int(segmentation.iloc[segment_index, 0])
    end = int(segmentation.iloc[segment_index + 1, 0])
    return segment, centroid, pattern_id, start, end


def make_gif(asset: str, segment_index: int, frames: int = 84, fps: int = 12) -> Path:
    setup()
    segment_raw, centroid_raw, pattern_id, start, end = load_segment(asset, segment_index)
    segment = centered_unit(segment_raw)
    condition = centered_unit(centroid_raw)
    x_axis = np.arange(len(segment))

    rng = np.random.default_rng(1000 + segment_index)
    forward_noise = rng.normal(size=segment.shape)
    reverse_noise = rng.normal(size=segment.shape)
    alpha_bars = diffusion_schedule()

    out_name = f"13_specific_segment_diffusion_{asset}_idx{segment_index:03d}.gif"
    out_path = OUT / out_name

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.22)
    fig.suptitle(
        f"{asset.upper()} Segment {segment_index}: Pattern-Conditioned Diffusion",
        fontsize=17,
        fontweight="bold",
        x=0.09,
        y=0.96,
        ha="left",
    )
    fig.text(
        0.09,
        0.875,
        f"SISC label p={pattern_id}, original index span {start}-{end}, segment length {len(segment)}.",
        fontsize=10.5,
        color=COLORS["muted"],
        ha="left",
    )

    ax.set_xlim(0, len(segment) - 1)
    ax.set_ylim(-2.25, 2.25)
    ax.set_xlabel("Within-segment time index")
    ax.set_ylabel("Centered and scaled value")
    ax.grid(axis="y")
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.40)
    ax.plot(x_axis, condition, color=COLORS["gray"], linewidth=2.4, alpha=0.45, label="conditioning motif")
    ax.plot(x_axis, segment, color="#111827", linewidth=2.0, alpha=0.24, label="actual segment")
    current_line, = ax.plot(x_axis, segment, color=COLORS["blue"], linewidth=3.2, label="animated state")
    phase_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=12, fontweight="bold", color=COLORS["ink"])
    equation_text = ax.text(0.02, 0.84, "", transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"])
    ax.legend(loc="lower left")

    progress_bg = mpl.lines.Line2D([0.09, 0.98], [0.105, 0.105], transform=fig.transFigure, color=COLORS["grid"], linewidth=7)
    progress_fg = mpl.lines.Line2D([0.09, 0.09], [0.105, 0.105], transform=fig.transFigure, color=COLORS["blue"], linewidth=7)
    fig.add_artist(progress_bg)
    fig.add_artist(progress_fg)

    def smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def update(frame: int):
        f = frame / (frames - 1)
        forward_phase = smoothstep(min(f / 0.52, 1.0))
        reverse_phase = smoothstep(max((f - 0.44) / 0.56, 0.0))
        step = min(len(alpha_bars) - 1, int(round(forward_phase * (len(alpha_bars) - 1))))
        a_bar = alpha_bars[step]

        if f < 0.56:
            current = np.sqrt(a_bar) * segment + np.sqrt(1.0 - a_bar) * forward_noise
            color = COLORS["blue"]
            phase_text.set_text(f"Forward diffusion: add scheduled noise, t={step}")
            equation_text.set_text(r"$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$")
        else:
            noisy_start = np.sqrt(alpha_bars[-1]) * segment + np.sqrt(1.0 - alpha_bars[-1]) * reverse_noise
            current = (1.0 - reverse_phase) * noisy_start + reverse_phase * segment
            color = COLORS["green"]
            phase_text.set_text(f"Reverse denoising: recover the segment, {int(reverse_phase * 100)}%")
            equation_text.set_text("Schematic reverse path conditioned on the SISC motif")

        current_line.set_data(x_axis, current)
        current_line.set_color(color)
        progress_fg.set_data([0.09, 0.09 + 0.89 * f], [0.105, 0.105])
        progress_fg.set_color(color)
        return current_line, phase_text, equation_text, progress_fg

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=90, blit=False)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    manifest_path = OUT / "specific_segment_diffusion_manifest.json"
    manifest = {
        "asset": asset,
        "segment_index": segment_index,
        "pattern_id": pattern_id,
        "segment_start": start,
        "segment_end": end,
        "segment_length": len(segment),
        "output": str(out_path),
        "frames": frames,
        "fps": fps,
        "note": "Forward diffusion uses the paper-code beta schedule. Reverse denoising is schematic for presentation.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GIF for one specific SISC segment.")
    parser.add_argument("--asset", choices=["goog", "zcf"], default="goog")
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=84)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    out_path = make_gif(args.asset, args.segment_index, frames=args.frames, fps=args.fps)
    print(out_path)


if __name__ == "__main__":
    main()
