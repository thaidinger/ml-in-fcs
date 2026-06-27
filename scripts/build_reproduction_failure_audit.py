from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "reports" / "generated_outputs"
OUT = GEN / "05_original_claim_reproduction_audit"


COLORS = {
    "ink": "#1f2933",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "green": "#2f855a",
    "blue": "#2563eb",
    "orange": "#d97706",
    "red": "#c2410c",
    "purple": "#7c3aed",
    "teal": "#0f766e",
    "gray": "#6b7280",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
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


def pct(x: float) -> str:
    return f"{x:+.1f}%"


def load_metrics() -> dict[str, float]:
    tatr_stored = pd.read_csv(
        GEN
        / "01_sp500_downstream_replication"
        / "07_replication_report"
        / "tables"
        / "tatr_summary.csv"
    )
    long_tatr = pd.read_csv(
        GEN
        / "01_sp500_downstream_replication"
        / "11_long_replication_batch"
        / "long_tatr_protocol_summary.csv"
    )
    long_tmtr = pd.read_csv(
        GEN
        / "01_sp500_downstream_replication"
        / "11_long_replication_batch"
        / "long_tmtr_protocol_summary.csv"
    )
    b3_sweep = json.loads(
        (
            GEN
            / "03_appendix_b3_simulated_sisc"
            / "16_b3_sisc_sweep"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    one_pattern_files = [
        GEN
        / "03_appendix_b3_simulated_sisc"
        / "14_b3_sisc_simulated_replication"
        / "fig8_one_pattern_metrics.csv",
        GEN
        / "03_appendix_b3_simulated_sisc"
        / "15_b3_sisc_simulated_replication_max20"
        / "fig8_one_pattern_metrics.csv",
    ]
    one_pattern = pd.concat([pd.read_csv(path) for path in one_pattern_files], ignore_index=True)

    def stored(dataset: str, variant: str) -> float:
        row = tatr_stored[(tatr_stored["dataset"].eq(dataset)) & (tatr_stored["variant"].eq(variant))].iloc[0]
        return float(row["change_100_vs_0pct"])

    def long(stage: str, column: str = "final_pct_mean") -> float:
        row = long_tatr[long_tatr["stage"].eq(stage)].iloc[0]
        return float(row[column])

    def tmtr(protocol: str, column: str = "final_pct_mean") -> float:
        row = long_tmtr[long_tmtr["protocol"].eq(protocol)].iloc[0]
        return float(row[column])

    return {
        "sp500_paper": -17.9,
        "goog_paper": -15.3,
        "zcf_paper": -17.4,
        "sp500_stored_authors": stored("SP500", "authors"),
        "goog_stored_authors": stored("GOOG", "authors"),
        "zcf_stored_authors": stored("ZCF", "authors"),
        "sp500_continuous_best": long("tatr_continuous_chunked", "best_pct_mean"),
        "sp500_continuous_final": long("tatr_continuous_chunked", "final_pct_mean"),
        "sp500_refit_final": long("tatr_continuous_refit", "final_pct_mean"),
        "sp500_independent_best": long("tatr_independent_fixed", "best_pct_mean"),
        "sp500_independent_final": long("tatr_independent_fixed", "final_pct_mean"),
        "tmtr_reference_best": tmtr("tmtr_reference", "best_pct_mean"),
        "tmtr_reference_final": tmtr("tmtr_reference", "final_pct_mean"),
        "tmtr_offset30_final": tmtr("tmtr_continuous_offset30", "final_pct_mean"),
        "tmtr_offset50_final": tmtr("tmtr_continuous_offset50", "final_pct_mean"),
        "b3_one_paper_dtw": 0.009,
        "b3_one_paper_jaccard": 0.938,
        "b3_one_best_dtw": float(one_pattern["per_unit_dtw"].min()),
        "b3_one_best_boundary_jaccard": float(one_pattern["boundary_jaccard_tol2"].max()),
        "b3_multi_paper_dtw": 0.01,
        "b3_multi_paper_iou": 0.784,
        "b3_multi_best_dtw": float(b3_sweep["best_by_dtw"]["avg_per_unit_dtw"]),
        "b3_multi_best_interval_iou": float(b3_sweep["best_by_author_interval_iou"]["author_interval_iou_pred"]),
        "b3_multi_best_boundary_jaccard": float(b3_sweep["best_by_boundary_jaccard_tol2"]["boundary_jaccard_tol2"]),
    }


def build_rows(m: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "claim_area": "TATR S&P 500",
            "paper_claim": "Appending 100 synthetic years reduces one-day MAPE by 17.9%.",
            "closest_local_evidence": (
                f"Stored author matrix: {pct(m['sp500_stored_authors'])}; "
                f"six-seed independent fixed: {pct(m['sp500_independent_final'])}; "
                f"continuous rollout final: {pct(m['sp500_continuous_final'])} "
                f"and best: {pct(m['sp500_continuous_best'])}."
            ),
            "status": "Not reproduced as an author-specified result.",
            "what_failed": (
                "The released-code-style independent 252-day reset path worsens MAPE. "
                "Paper-like gains appear only when synthetic data are generated as one continuous pattern-evolution rollout, "
                "so the Markov rollout, initialization, block slicing, scaler fitting, and warm-up choices do not uniquely recover the paper curve."
            ),
        },
        {
            "claim_area": "TATR GOOG",
            "paper_claim": "Appending 100 synthetic years reduces one-day MAPE by 15.3%.",
            "closest_local_evidence": f"Stored GOOG author matrix finishes {pct(m['goog_stored_authors'])} versus baseline.",
            "status": "Not reproduced; strict rerun blocked by split logic.",
            "what_failed": (
                "The released S&P-style downstream split consumes 1260 held-out initialization points, "
                "but the GOOG 80/20 held-out window has 744 points. The stored author matrix also trends in the wrong direction."
            ),
        },
        {
            "claim_area": "TATR ZC=F",
            "paper_claim": "Appending 100 synthetic years reduces one-day MAPE by 17.4%.",
            "closest_local_evidence": f"Stored ZC=F author matrix finishes {pct(m['zcf_stored_authors'])} versus baseline.",
            "status": "Not reproduced; strict rerun blocked by split logic.",
            "what_failed": (
                "The released S&P-style downstream split consumes 1260 held-out initialization points, "
                "but the ZC=F 80/20 held-out window has 963 points. The stored author matrix also worsens rather than improves."
            ),
        },
        {
            "claim_area": "TMTR robustness",
            "paper_claim": "FTS-Diffusion maintains comparable prediction accuracy across synthetic mixing proportions.",
            "closest_local_evidence": (
                f"Six-seed S&P reference TMTR best {pct(m['tmtr_reference_best'])} but final {pct(m['tmtr_reference_final'])}; "
                f"continuous offset-30 final {pct(m['tmtr_offset30_final'])}; offset-50 final {pct(m['tmtr_offset50_final'])}."
            ),
            "status": "Only partially supported under our controls.",
            "what_failed": (
                "TMTR is not stable across plausible generation offsets; some settings help, while others collapse at high synthetic proportions."
            ),
        },
        {
            "claim_area": "Appendix B.3 one-pattern SISC",
            "paper_claim": "One-pattern toy reaches per-unit DTW 0.009 and Jaccard 0.938.",
            "closest_local_evidence": (
                f"Best available one-pattern DTW {m['b3_one_best_dtw']:.4f}; "
                f"best boundary-Jaccard@2 {m['b3_one_best_boundary_jaccard']:.4f}."
            ),
            "status": "Not reproduced from released materials.",
            "what_failed": (
                "The released tree does not provide the exact one-pattern generator, standard pattern arrays, seed state, or standalone metric implementation. "
                "Derived one-pattern runs on the shipped toy data remain below the reported target."
            ),
        },
        {
            "claim_area": "Appendix B.3 multi-pattern SISC",
            "paper_claim": "Four-pattern toy reaches average per-unit DTW 0.01 and Jaccard/IoU 0.784.",
            "closest_local_evidence": (
                f"Best sweep DTW {m['b3_multi_best_dtw']:.4f}; "
                f"best author-style interval IoU {m['b3_multi_best_interval_iou']:.4f}; "
                f"best strict boundary-Jaccard@2 {m['b3_multi_best_boundary_jaccard']:.4f}."
            ),
            "status": "Not reproduced from released materials.",
            "what_failed": (
                "The gap persisted across seed, l_max, and iteration sweeps. Exact reproduction appears to need the unpublished synthetic generator details and metric conventions."
            ),
        },
    ]
    return pd.DataFrame(rows)


def save_audit_chart(m: dict[str, float]) -> None:
    fig = plt.figure(figsize=(14.2, 9.1))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.86, bottom=0.11, hspace=0.42, wspace=0.26)
    fig.text(0.07, 0.94, "Original Claims vs Local Reproduction Evidence", fontsize=20, fontweight="bold")
    fig.text(
        0.07,
        0.902,
        "Negative MAPE change means lower error. Local author-style endpoints are the closest available released-code/stored-matrix comparisons.",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    ax = fig.add_subplot(gs[0, 0])
    assets = ["S&P 500", "GOOG", "ZC=F"]
    paper = [m["sp500_paper"], m["goog_paper"], m["zcf_paper"]]
    local = [m["sp500_stored_authors"], m["goog_stored_authors"], m["zcf_stored_authors"]]
    x = np.arange(len(assets))
    width = 0.34
    ax.bar(x - width / 2, paper, width, color=COLORS["green"], label="Original paper")
    ax.bar(x + width / 2, local, width, color=COLORS["red"], label="Local author-style endpoint")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_yscale("symlog", linthresh=25)
    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel("MAPE change at 100 synthetic years (%)")
    ax.set_title("TATR Endpoints Do Not Match the Claimed Reductions", loc="left", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    ax.set_ylim(-70, 1700)
    for xpos, val in zip(x - width / 2, paper):
        ax.text(xpos, val - 5, f"{val:.1f}%", ha="center", va="top", fontsize=9, color=COLORS["green"])
    for xpos, val in zip(x + width / 2, local):
        ypos = val * 0.58 if val > 150 else val * 0.72
        ax.text(xpos, ypos, f"+{val:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["Paper\nS&P target", "Continuous\nbest", "Continuous\nfinal", "Independent\nfinal"]
    values = [
        m["sp500_paper"],
        m["sp500_continuous_best"],
        m["sp500_continuous_final"],
        m["sp500_independent_final"],
    ]
    colors = [COLORS["purple"], COLORS["green"], COLORS["blue"], COLORS["red"]]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_ylabel("MAPE change (%)")
    ax.set_title("S&P Success Requires a Different Rollout", loc="left", fontweight="bold")
    ax.grid(axis="y")
    ax.set_ylim(-105, 230)
    for bar, val in zip(bars, values):
        va = "bottom" if val >= 0 else "top"
        offset = 5 if val >= 0 else -5
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.1f}%", ha="center", va=va, fontsize=9.5)

    ax = fig.add_subplot(gs[1, 0])
    metric_labels = ["One DTW", "One Jaccard*", "Multi DTW", "Multi IoU"]
    targets = [m["b3_one_paper_dtw"], m["b3_one_paper_jaccard"], m["b3_multi_paper_dtw"], m["b3_multi_paper_iou"]]
    ours = [
        m["b3_one_best_dtw"],
        m["b3_one_best_boundary_jaccard"],
        m["b3_multi_best_dtw"],
        m["b3_multi_best_interval_iou"],
    ]
    x = np.arange(len(metric_labels))
    ax.bar(x - width / 2, targets, width, color=COLORS["green"], label="Original paper")
    ax.bar(x + width / 2, ours, width, color=COLORS["orange"], label="Best local")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_title("Appendix B.3 Targets Remain Out of Reach", loc="left", fontweight="bold")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    for xpos, val in zip(x - width / 2, targets):
        ax.text(xpos, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=8.8)
    for xpos, val in zip(x + width / 2, ours):
        ax.text(xpos, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=8.8)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    failure_text = (
        "Primary failure modes\n\n"
        "1. TATR is under-specified: continuous pattern-evolution rollouts and independent reset blocks answer different experimental questions.\n\n"
        "2. GOOG/ZC=F author-faithful TATR is not defined under the released S&P-style split because the held-out windows are too short.\n\n"
        "3. Appendix B.3 lacks the exact toy generator, seed state, standard patterns, and metric implementation needed to recover the reported values."
    )
    ax.text(
        0.02,
        0.98,
        failure_text,
        va="top",
        ha="left",
        fontsize=11.2,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.55,rounding_size=0.04", "facecolor": "#f8fafc", "edgecolor": COLORS["grid"]},
    )

    fig.text(
        0.50,
        0.035,
        "Generated from retained result tables; *local one-pattern Jaccard is boundary-Jaccard@2 because the exact paper metric implementation is unavailable.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"],
    )
    fig.savefig(OUT / "original_claim_reproduction_audit.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "original_claim_reproduction_audit.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_readme(df: pd.DataFrame, m: dict[str, float]) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"### {row['claim_area']}\n\n"
            f"- Paper claim: {row['paper_claim']}\n"
            f"- Closest local evidence: {row['closest_local_evidence']}\n"
            f"- Status: {row['status']}\n"
            f"- What failed: {row['what_failed']}\n"
        )
    body = "\n".join(rows)
    text = f"""# Original Claim Reproduction Audit

This folder is a compact audit of the original FTS-Diffusion claims that we could not reproduce from the released materials.

The original paper passages checked were the abstract/Section 5.3/Figure 6 for TMTR/TATR and Appendix B.3 for SISC toy-data metrics. The key TATR claim is that appending 100 synthetic years reduces one-day-ahead MAPE by 17.9%, 15.3%, and 17.4% on S&P 500, GOOG, and ZC=F. The key B.3 claims are one-pattern DTW/Jaccard 0.009/0.938 and multi-pattern DTW/Jaccard 0.01/0.784.

## Files

- `claim_reproduction_audit.csv`: row-level claim, closest local evidence, status, and failure reason.
- `audit_metrics.json`: machine-readable numeric summary used by the plot.
- `original_claim_reproduction_audit.png` / `.pdf`: visual summary for discussion or appendix material.

## Bottom Line

The failed reproduction is not one vague failure. It breaks at three concrete points:

- The S&P 500 TATR result is not uniquely recovered from released details: continuous rollouts improve, while released-code-style independent blocks worsen.
- GOOG and ZC=F author-faithful TATR is blocked by the released split logic and the stored author matrices worsen.
- Appendix B.3 needs unreleased toy-generator and metric details to recover the reported DTW/Jaccard values.

## Claim Rows

{body}
"""
    (OUT / "README.md").write_text(textwrap.dedent(text), encoding="utf-8")
    (OUT / "audit_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


def main() -> None:
    setup()
    metrics = load_metrics()
    audit = build_rows(metrics)
    audit.to_csv(OUT / "claim_reproduction_audit.csv", index=False)
    save_audit_chart(metrics)
    save_readme(audit, metrics)
    manifest = {
        "output": str(OUT),
        "source_tables": [
            "01_sp500_downstream_replication/07_replication_report/tables/tatr_summary.csv",
            "01_sp500_downstream_replication/11_long_replication_batch/long_tatr_protocol_summary.csv",
            "01_sp500_downstream_replication/11_long_replication_batch/long_tmtr_protocol_summary.csv",
            "03_appendix_b3_simulated_sisc/14_b3_sisc_simulated_replication/fig8_one_pattern_metrics.csv",
            "03_appendix_b3_simulated_sisc/15_b3_sisc_simulated_replication_max20/fig8_one_pattern_metrics.csv",
            "03_appendix_b3_simulated_sisc/16_b3_sisc_sweep/manifest.json",
        ],
        "files": [
            "README.md",
            "claim_reproduction_audit.csv",
            "audit_metrics.json",
            "original_claim_reproduction_audit.png",
            "original_claim_reproduction_audit.pdf",
            "manifest.json",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
