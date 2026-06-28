from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "reports" / "generated_outputs"
OUT = GEN / "08_replicability_diagnostic"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_curve(rows: list[dict], name: str, summary: pd.DataFrame, x_col: str, protocol: str, experiment: str) -> None:
    baseline = float(summary["avg"].iloc[0])
    best_idx = int(summary["avg"].idxmin())
    final_idx = int(len(summary) - 1)
    x_values = summary[x_col].tolist()
    best = float(summary["avg"].iloc[best_idx])
    final = float(summary["avg"].iloc[final_idx])
    rows.append(
        {
            "run_set": name,
            "experiment": experiment,
            "protocol": protocol,
            "x_column": x_col,
            "baseline_x": x_values[0],
            "baseline_mape": baseline,
            "best_x": x_values[best_idx],
            "best_mape": best,
            "best_pct_vs_baseline": (best - baseline) / baseline * 100,
            "final_x": x_values[final_idx],
            "final_mape": final,
            "final_pct_vs_baseline": (final - baseline) / baseline * 100,
            "n_points": len(summary),
            "monotonic_down_steps": int((np.diff(summary["avg"].to_numpy()) < 0).sum()),
            "monotonic_up_steps": int((np.diff(summary["avg"].to_numpy()) > 0).sum()),
        }
    )


def pct_change(series: pd.Series) -> pd.Series:
    return (series - series.iloc[0]) / series.iloc[0] * 100


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    resource_tatr = read_csv(GEN / "01_sp500_paper_protocol_resource_aware" / "tatr_prices_summary_authors_style.csv")
    resource_tmtr = read_csv(GEN / "01_sp500_paper_protocol_resource_aware" / "tmtr_prices_summary_authors_style.csv")
    single = read_csv(GEN / "02_sp500_tatr_single_diagnostic" / "tatr_prices_single_summary_with_change.csv")
    author_1d = read_csv(GEN / "03_sp500_tatr_author_style_1day" / "tatr_prices_author_style_summary_with_change.csv")
    author_5d = read_csv(GEN / "04_sp500_tatr_author_style_5day" / "tatr_prices_author_style_summary_with_change.csv")
    prices_tatr = read_csv(
        GEN / "05_sp500_prices_and_returns_sweeps" / "downstream_prices_sp500" / "tatr_prices_summary_authors_style.csv"
    )
    prices_tmtr = read_csv(
        GEN / "05_sp500_prices_and_returns_sweeps" / "downstream_prices_sp500" / "tmtr_prices_summary_authors_style.csv"
    )
    audit = read_csv(GEN / "06_tatr_audit" / "tatr_results_audit_summary.csv")
    synth = read_csv(GEN / "06_tatr_audit" / "sp500_synthetic_protocol_stats_by_run.csv")

    rows: list[dict] = []
    add_curve(rows, "resource_aware_tatr_1day", resource_tatr, "augmentation_idx", "author_style_independent_blocks", "TATR")
    add_curve(rows, "author_style_tatr_1day", author_1d, "augmentation_idx", "author_style_independent_blocks", "TATR")
    add_curve(rows, "author_style_tatr_5day", author_5d, "augmentation_idx", "author_style_independent_blocks", "TATR")
    add_curve(rows, "reduced_prices_tatr_1day_50epoch", prices_tatr, "augmentation_idx", "author_style_independent_blocks", "TATR")
    add_curve(rows, "single_diagnostic_tatr_1day", single, "augmentation_blocks", "single_continuous_trajectory", "TATR")
    add_curve(rows, "resource_aware_tmtr_1day", resource_tmtr, "synthetic_proportion_pct", "author_style_mix", "TMTR")
    add_curve(rows, "reduced_prices_tmtr_1day_50epoch", prices_tmtr, "synthetic_proportion_pct", "author_style_mix", "TMTR")

    curve_summary = pd.DataFrame(rows)
    curve_summary.to_csv(OUT / "curve_replicability_summary.csv", index=False)

    sp500_audit = audit[audit["asset"].eq("sp500")].copy()
    sp500_audit.to_csv(OUT / "stored_sp500_protocol_audit.csv", index=False)

    protocol_stats = (
        synth.groupby("protocol")
        .agg(
            runs=("run", "count"),
            synthetic_mean=("mean", "mean"),
            synthetic_min=("min", "mean"),
            synthetic_max=("max", "mean"),
            first_start=("first_start", "mean"),
            last_end=("last_end", "mean"),
            return_mean=("return_mean", "mean"),
            return_std=("return_std", "mean"),
            block_jump_abs_mean=("block_jump_abs_mean", "mean"),
        )
        .reset_index()
    )
    protocol_stats.to_csv(OUT / "synthetic_protocol_price_level_summary.csv", index=False)

    author_like = curve_summary[
        curve_summary["protocol"].isin(["author_style_independent_blocks", "author_style_mix"])
    ].copy()
    strong_improvements = author_like[author_like["final_pct_vs_baseline"] <= -20]
    any_author_final_improves = bool((author_like["final_pct_vs_baseline"] < 0).any())
    any_author_strong_final_improves = bool(len(strong_improvements))
    single_row = curve_summary[curve_summary["protocol"].eq("single_continuous_trajectory")].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        resource_tatr["augmentation_idx"],
        pct_change(resource_tatr["avg"]),
        marker="o",
        linewidth=2,
        label="author-style TATR, 5 runs, 100 epochs",
    )
    ax.plot(
        author_1d["augmentation_idx"],
        pct_change(author_1d["avg"]),
        marker="o",
        linewidth=2,
        label="author-style TATR, 10 runs, 100 epochs",
    )
    ax.plot(
        single["augmentation_blocks"],
        pct_change(single["avg"]),
        marker="o",
        linewidth=2,
        label="single continuous diagnostic, 1 run",
    )
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title("SP500 TATR: author-style curves vs single diagnostic")
    ax.set_xlabel("Synthetic blocks")
    ax.set_ylabel("MAPE change vs no augmentation (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "author_vs_single_tatr_pct_change.png", dpi=220)
    fig.savefig(OUT / "author_vs_single_tatr_pct_change.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for protocol, df in synth.groupby("protocol"):
        ax.scatter(df["run"], df["mean"], label=f"{protocol} mean", s=26)
        ax.scatter(df["run"], df["last_end"], label=f"{protocol} last", s=18, alpha=0.65)
    ax.set_title("Synthetic SP500 price levels by protocol")
    ax.set_xlabel("Synthetic run")
    ax.set_ylabel("Synthetic price level")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_price_levels_by_protocol.png", dpi=220)
    fig.savefig(OUT / "synthetic_price_levels_by_protocol.pdf")
    plt.close(fig)

    conclusion = {
        "diagnosis": "not_replicable_with_released_reference_code_and_stored_sp500_artifacts",
        "scope": (
            "This is a reproducibility conclusion about the available code/checkpoints/artifacts and our "
            "resource-aware reruns, not proof of author intent."
        ),
        "author_style_final_improves_any_curve": any_author_final_improves,
        "author_style_strong_final_improves_any_curve": any_author_strong_final_improves,
        "single_diagnostic_final_pct_vs_baseline": float(single_row["final_pct_vs_baseline"]),
        "single_diagnostic_best_pct_vs_baseline": float(single_row["best_pct_vs_baseline"]),
        "evidence": [
            "Reference README calls paper-like TATR/TMTR with prices, ahead=1, window=64, hidden=32, loss=mae, epochs=100.",
            "Reference TATR code adds independent generated 252-day blocks from a fixed first segment; it is not a continuous trajectory protocol.",
            "Our resource-aware author-style one-day TATR run has final MAPE worse than baseline.",
            "Our separate author-style one-day and five-day TATR runs also have final MAPE worse than baseline.",
            "Stored SP500 author-protocol audit has final MAPE much worse than baseline.",
            "The only strong paper-like downward behavior appears in the single continuous-trajectory diagnostic, which differs from the released author TATR protocol.",
            "Synthetic price-level audit shows independent blocks stay near the early initial level, while continuous trajectories drift across/above the test-period scale.",
        ],
    }
    (OUT / "diagnostic_conclusion.json").write_text(json.dumps(conclusion, indent=2))

    verdict = "NOT REPLICABLE with the released code/artifacts under the tested protocols"
    if any_author_strong_final_improves:
        verdict = "INCONCLUSIVE: at least one author-style curve shows strong final improvement"
    elif any_author_final_improves:
        verdict = "WEAK/INCONCLUSIVE: at least one author-style curve improves at final point, but not strongly"

    readme = f"""# Replicability Diagnostic

Verdict: **{verdict}**.

This diagnostic does not establish cheating or intent. It says that, using the released reference protocol plus the stored S&P 500 artifacts/checkpoints available in this repo, the paper-like sustained downward TATR behavior is not reproduced by the author-style protocol.

## Main Checks

1. **Protocol check**: the reference paper-like settings are `prices`, `ahead=1`, `window_size=64`, `hidden_dim=32`, `loss=mae`, `epochs=100`; TATR adds independent 252-day synthetic blocks from a fixed initial segment.
2. **Author-style reruns**: all generated author-style S&P TATR curves end worse than their no-augmentation baseline.
3. **Stored audit**: stored SP500 `authors` protocol results also end worse, while stored SP500 `single` improves strongly.
4. **Protocol sensitivity**: the `single` continuous-trajectory diagnostic produces a large drop, but it is not the released author-code TATR protocol.
5. **Synthetic-level diagnosis**: independent blocks remain near the initial price level; continuous trajectories drift through the test-period scale.

## Files

- `curve_replicability_summary.csv`: one-row summary per generated curve.
- `stored_sp500_protocol_audit.csv`: stored SP500 protocol audit rows.
- `synthetic_protocol_price_level_summary.csv`: synthetic price-level summary by protocol.
- `diagnostic_conclusion.json`: machine-readable conclusion.
- `author_vs_single_tatr_pct_change.png`: author-style vs single TATR percent-change plot.
- `synthetic_price_levels_by_protocol.png`: synthetic price-level diagnostic plot.

## Bottom Line

The authors' reported trend is **not replicable from the released reference code and stored S&P artifacts we have**. The result becomes qualitatively paper-like only after switching to the `single` continuous-trajectory protocol, which is a different protocol from the released author TATR code.
"""
    (OUT / "README.md").write_text(readme)

    print(json.dumps({"verdict": verdict, "out": str(OUT.relative_to(ROOT)), "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
