from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "tables"
PAPER_K_CONFIGS = {
    "GOOG": "K11",
    "ZCF": "K11",
    "SP500": "K14",
}


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(x: float | int | str) -> str:
    if isinstance(x, str):
        return tex_escape(x)
    if pd.isna(x):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.4f}"


def latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    columns: list[str] | None = None,
    font_size: str = r"\small",
    resize_to_width: bool = True,
) -> str:
    if columns is not None:
        df = df[columns]
    aligns = "l" + "r" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        font_size,
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
    ]
    if resize_to_width:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [
        rf"\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        " & ".join(tex_escape(col) for col in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(row[col]) for col in df.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%"]
    if resize_to_width:
        lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def parse_tatr_path(path: Path) -> dict[str, str]:
    parts = path.parts
    idx = parts.index("tatr")
    dataset = parts[idx + 1]
    k = parts[idx + 2]
    variant = parts[idx + 3]
    return {"dataset": dataset.upper(), "k": k.upper(), "variant": variant}


def is_paper_k_config(dataset: str, k: str) -> bool:
    """Return True for the K choices used in the paper/result protocol."""
    return PAPER_K_CONFIGS.get(dataset.upper()) == k.upper()


def load_tatr_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    for path in sorted((ROOT / "results" / "tatr").glob("**/results_matrix.csv")):
        meta = parse_tatr_path(path)
        if not is_paper_k_config(meta["dataset"], meta["k"]):
            continue
        df = pd.read_csv(path)
        pct_cols = [col for col in df.columns if col != "run_idx"]
        long = df.melt(id_vars=["run_idx"], value_vars=pct_cols, var_name="synthetic_pct", value_name="mae")
        long["synthetic_pct"] = long["synthetic_pct"].astype(int)
        for key, value in meta.items():
            long[key] = value
        long["source_file"] = str(path.relative_to(ROOT))
        rows.append(long)

        means = long.groupby("synthetic_pct")["mae"].mean()
        stds = long.groupby("synthetic_pct")["mae"].std(ddof=0)
        baseline = float(means.loc[0])
        best_pct = int(means.idxmin())
        best_mean = float(means.loc[best_pct])
        final_mean = float(means.loc[100]) if 100 in means.index else float("nan")
        summary.append(
            {
                **meta,
                "runs": int(df.shape[0]),
                "baseline_0pct": baseline,
                "best_pct": best_pct,
                "best_mean": best_mean,
                "best_delta_vs_0pct": best_mean - baseline,
                "best_change_pct": 100.0 * (best_mean - baseline) / baseline,
                "mean_100pct": final_mean,
                "change_100_vs_0pct": 100.0 * (final_mean - baseline) / baseline,
                "std_at_best": float(stds.loc[best_pct]),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    if rows:
        return pd.concat(rows, ignore_index=True), pd.DataFrame(summary)
    return pd.DataFrame(), pd.DataFrame()


def plot_tatr_by_dataset(tatr_long: pd.DataFrame) -> None:
    if tatr_long.empty:
        return
    for (dataset, k), group in tatr_long.groupby(["dataset", "k"], sort=True):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for variant, sub in group.groupby("variant", sort=True):
            agg = sub.groupby("synthetic_pct")["mae"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["synthetic_pct"], agg["mean"], marker="o", linewidth=1.8, label=variant)
            ax.fill_between(
                agg["synthetic_pct"].to_numpy(),
                (agg["mean"] - agg["std"]).to_numpy(),
                (agg["mean"] + agg["std"]).to_numpy(),
                alpha=0.12,
            )
        ax.set_title(f"TATR replication matrix: {dataset} {k}")
        ax.set_xlabel("Synthetic augmentation percentage")
        ax.set_ylabel("One-step MAE")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / f"tatr_{dataset.lower()}_{k.lower()}.pdf")
        plt.close(fig)


def plot_tatr_best_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values(["dataset", "k", "variant"]).copy()
    labels = [f"{r.dataset} {r.k} {r.variant}" for r in ordered.itertuples()]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#2f6f9f" if val <= 0 else "#a34f4f" for val in ordered["best_change_pct"]]
    ax.bar(x, ordered["best_change_pct"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Best mean MAE change vs 0% synthetic (%)")
    ax.set_title("Best observed TATR change across pulled matrices")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "tatr_best_change_summary.pdf")
    plt.close(fig)


def load_reference_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_rows = []
    for name in ["res_tatr_summary.csv", "res_tmtr_summary.csv"]:
        path = ROOT / "fts-diffusion-ref" / "res" / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        first = df.columns[0]
        df = df.rename(columns={first: "index"})
        df["experiment"] = name.replace("res_", "").replace("_summary.csv", "").upper()
        df["source_file"] = str(path.relative_to(ROOT))
        ref_rows.append(df)
    ref = pd.concat(ref_rows, ignore_index=True) if ref_rows else pd.DataFrame()

    detail_rows = []
    for name in ["res_tatr_sp500-prices_1ahead_h32_mae.csv", "res_tmtr_sp500-prices_1ahead_h32_mae.csv"]:
        path = ROOT / "fts-diffusion-ref" / "res" / name
        if not path.exists():
            continue
        df = pd.read_csv(path, header=None)
        long = df.reset_index(names="run_idx").melt(id_vars=["run_idx"], var_name="column", value_name="mae")
        long["experiment"] = "TATR" if "tatr" in name else "TMTR"
        detail_rows.append(long)
    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return ref, detail


def plot_reference_summaries(ref_detail: pd.DataFrame) -> None:
    if ref_detail.empty:
        return
    for exp, group in ref_detail.groupby("experiment"):
        agg = group.groupby("column")["mae"].agg(["mean", "min", "max"]).reset_index()
        x = agg["column"].astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.plot(x, agg["mean"], marker="o", label="mean")
        ax.fill_between(x, agg["min"], agg["max"], alpha=0.18, label="min-max")
        ax.set_title(f"Reference {exp} SP500 MAE summary")
        ax.set_xlabel("Result column index")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / f"reference_{exp.lower()}_summary.pdf")
        plt.close(fig)


def parse_array(text: str) -> np.ndarray:
    values = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", str(text))
    return np.asarray([float(v) for v in values], dtype=float)


def plot_sisc_outputs() -> tuple[int, int]:
    centroid_path = ROOT / "fts-diffusion-ref" / "res" / "sisc_sp500_k14_l10-21_dba_kmpp_centroids.csv"
    segmentation_path = ROOT / "fts-diffusion-ref" / "res" / "sisc_sp500_k14_l10-21_dba_kmpp_segmentation.csv"
    n_centroids = 0
    n_segments = 0
    if centroid_path.exists():
        centroids = pd.read_csv(centroid_path, index_col=0)
        n_centroids = int(centroids.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(8.0, 6.4), sharex=True)
        axes = axes.ravel()
        for i, (_, row) in enumerate(centroids.iterrows()):
            ax = axes[i]
            ax.plot(row.to_numpy(dtype=float), color="#2f6f9f", linewidth=1.3)
            ax.set_title(f"Pattern {i}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15)
        for ax in axes[n_centroids:]:
            ax.axis("off")
        fig.suptitle("SISC centroids from reference SP500 replication artifacts", y=0.995)
        fig.tight_layout()
        fig.savefig(FIG / "sisc_centroids.pdf")
        plt.close(fig)
    if segmentation_path.exists():
        seg = pd.read_csv(segmentation_path)
        n_segments = int(seg.shape[0])
        y = pd.to_numeric(seg.iloc[:, 1], errors="coerce").to_numpy()
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        ax.plot(np.arange(len(y)), y, linewidth=1.0)
        ax.set_title("SISC segmentation boundary sequence")
        ax.set_xlabel("Segment index")
        ax.set_ylabel("Boundary / position")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "sisc_segmentation_boundaries.pdf")
        plt.close(fig)
    return n_centroids, n_segments


def copy_existing_figures_section() -> str:
    fig_dir = ROOT / "fts-diffusion-ref" / "figs"
    lines = []
    for name, caption, label in [
        ("stylized_fact.pdf", "Reference stylized-fact figure bundled with the FTS-Diffusion code.", "fig:ref-stylized"),
    ]:
        path = fig_dir / name
        if path.exists():
            rel = os.path.relpath(path, OUT)
            lines += [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.92\linewidth]{{{str(rel)}}}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{figure}",
            ]
    return "\n".join(lines)


def local_final_figures_section() -> tuple[str, list[str]]:
    """Include all final result PDFs under the repository-level figures/ tree."""
    figure_root = ROOT / "figures"
    paths = []
    for path in sorted(figure_root.glob("**/final.pdf")):
        parts = path.relative_to(figure_root).parts
        if len(parts) >= 4 and is_paper_k_config(parts[0].upper(), parts[1].upper()):
            paths.append(path)
    lines: list[str] = []
    included: list[str] = []
    for path in paths:
        rel_from_root = path.relative_to(ROOT)
        parts = path.relative_to(figure_root).parts
        if len(parts) >= 4:
            dataset, k, variant = parts[0], parts[1], parts[2]
            caption = f"Pulled final result figure for {dataset.upper()} {k.upper()} {variant}."
        else:
            caption = f"Pulled final result figure: {rel_from_root}."
        lines += [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.90\linewidth]{{{os.path.relpath(path, OUT)}}}",
            rf"\caption{{{tex_escape(caption)}}}",
            r"\end{figure}",
        ]
        included.append(str(rel_from_root))
    return "\n".join(lines), included


def relative(path: Path) -> str:
    return str(path.relative_to(OUT))


def write_report(tatr_summary: pd.DataFrame, ref_summary: pd.DataFrame, n_centroids: int, n_segments: int) -> None:
    tatr_summary_for_table = tatr_summary.copy()
    if not tatr_summary_for_table.empty:
        tatr_summary_for_table = tatr_summary_for_table[
            [
                "dataset",
                "k",
                "variant",
                "runs",
                "baseline_0pct",
                "best_pct",
                "best_mean",
                "best_change_pct",
                "mean_100pct",
                "change_100_vs_0pct",
            ]
        ].sort_values(["dataset", "k", "variant"])
        tatr_summary_for_table.columns = [
            "Data",
            "K",
            "Variant",
            "Runs",
            "0% mean",
            "Best %",
            "Best mean",
            "Best change %",
            "100% mean",
            "100% change %",
        ]

    ref_table = ref_summary.copy()
    if not ref_table.empty:
        ref_table = ref_table[["experiment", "index", "avg"]]
        ref_table.columns = ["Exp.", "Idx", "Avg"]
        ref_table = ref_table.head(16)

    result_files = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "results" / "tatr").glob("**/results_matrix.csv")
        if is_paper_k_config(parse_tatr_path(path)["dataset"], parse_tatr_path(path)["k"])
    )
    has_drift_outputs = (ROOT / "drift_component" / "outputs").exists()
    drift_sentence = (
        "Drift diagnostic outputs were found and can be appended to this report."
        if has_drift_outputs
        else "No real generated-sample drift diagnostic outputs are present in this checkout; this report therefore does not claim no-drift results for FTS-Diffusion samples."
    )

    def figure_section(paths: list[Path], caption_prefix: str) -> str:
        section_lines: list[str] = []
        for path in paths:
            section_lines += [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.90\linewidth]{{{relative(path)}}}",
                rf"\caption{{{caption_prefix}: {tex_escape(path.stem.replace('_', ' '))}.}}",
                r"\end{figure}",
            ]
        return "\n".join(section_lines)

    sisc_plots = sorted(FIG.glob("sisc_*.pdf"))

    sisc_plot_section = figure_section(sisc_plots, "SISC artifact plot")

    generated_plot_section = []
    for path in sorted(FIG.glob("*.pdf")):
        generated_plot_section += [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{{relative(path)}}}",
            rf"\caption{{Generated plot: {tex_escape(path.stem.replace('_', ' '))}.}}",
            r"\end{figure}",
        ]

    tatr_table_tex = (
        latex_table(tatr_summary_for_table, "Summary of pulled TATR result matrices. Negative change means lower MAE than the 0% synthetic baseline.", "tab:tatr-summary")
        if not tatr_summary_for_table.empty
        else "No TATR result matrices were found."
    )
    ref_table_tex = (
        latex_table(
            ref_table,
            "Reference summary CSV rows bundled with the original FTS-Diffusion repository.",
            "tab:reference-summary",
            font_size=r"\scriptsize",
            resize_to_width=False,
        )
        if not ref_table.empty
        else "No reference summary CSVs were found."
    )
    local_figures_tex, local_figure_files = local_final_figures_section()

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{microtype}}

\title{{FTS-Diffusion Replication Attempt: Results and Diagnostic Report}}
\author{{}}
\date{{May 11, 2026}}

\begin{{document}}
\maketitle

\section{{What We Did}}

\begin{{itemize}}
\item Collected the paper-consistent artifacts: {len(result_files)} TATR matrices, {len(local_figure_files)} final figures under \texttt{{figures/}}, reference TMTR/TATR summaries, SISC outputs, and bundled reference figures.
\item Aggregated each TATR matrix over 30 runs and plotted mean MAE with one-standard-deviation bands.
\item Added the predictable-drift component as a separate diagnostic, but did not report FTS-Diffusion drift numbers because no ordered generated sample paths are present in this checkout.
\end{{itemize}}

\section{{Available Artifacts}}

The report uses the paper-consistent K choices: GOOG K=11, ZCF K=11, and SP500 K=14. Other local artifacts, such as ZCF K=14, are excluded from the tables and figures for consistency. The TATR matrices have columns 0, 10, \ldots, 100, interpreted as synthetic augmentation percentages. The full file list and plot list are stored in \texttt{{manifest.json}} and \texttt{{tables/tatr\_summary.csv}}. {tex_escape(drift_sentence)}

\section{{TATR Matrix Results}}

Table~\ref{{tab:tatr-summary}} summarizes the pulled TATR matrices. Negative change means lower MAE than the 0\% synthetic baseline.

{tatr_table_tex}

\section{{Updated TATR/TMTR Result Figures}}

These are the newest final result PDFs under the repository \texttt{{figures}} directory, filtered to GOOG K=11, ZCF K=11, and SP500 K=14.

{local_figures_tex}

\section{{Reference Summary CSVs}}

Table~\ref{{tab:reference-summary}} reproduces the bundled reference TMTR/TATR summary rows.

{ref_table_tex}

\section{{SISC Pattern-Recognition Artifacts and Plots}}

The local reference artifacts contain {n_centroids} SISC centroids and {n_segments} segmentation boundary rows for SP500 K=14.

{sisc_plot_section}

\section{{Bundled Stylized-Fact Figure}}

{copy_existing_figures_section()}

\section{{Predictable-Drift Extension Status}}

The drift component targets the gap that matching stylized facts does not imply
\[
  \mathbb{{E}}[r_{{t+1}}\mid \mathcal{{F}}_t] = 0.
\]
It fixes
\[
  \omega_t = (1, r_t, r_{{t-1}}, |r_t|, |r_{{t-1}}|)
\]
and computes
\[
  \widehat{{\delta}} =
  \left\|
  \frac{{1}}{{T}}\sum_t r_{{t+1}}\omega_t
  \right\|_2.
\]
This is a finite-dimensional check, not a full no-arbitrage proof. Because no ordered generated return paths are present here, this PDF does not report drift statistics for actual FTS-Diffusion samples.

\section{{Replication Caveats}}

\begin{{itemize}}
\item The pulled \texttt{{results/tatr/**/results\_matrix.csv}} files are result matrices, not generated financial return paths.
\item The report does not include RCGAN, TimeGAN, or CSDI reruns.
\item The SISC artifacts and reference figures are bundled outputs from the reference directory; they support comparison, but are not proof that the full pipeline was freshly retrained in this checkout.
\item A full replication report should add training logs, generated sample CSVs, exact seeds, and final drift diagnostics once those outputs are available.
\end{{itemize}}

\end{{document}}
"""
    (OUT / "replication_report.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    for old_plot in FIG.glob("*.pdf"):
        old_plot.unlink()
    tatr_long, tatr_summary = load_tatr_matrices()
    if not tatr_long.empty:
        tatr_long.to_csv(TAB / "tatr_long.csv", index=False)
        tatr_summary.to_csv(TAB / "tatr_summary.csv", index=False)
    ref_summary, ref_detail = load_reference_summaries()
    if not ref_summary.empty:
        ref_summary.to_csv(TAB / "reference_summary.csv", index=False)
    if not ref_detail.empty:
        ref_detail.to_csv(TAB / "reference_detail.csv", index=False)
    n_centroids, n_segments = plot_sisc_outputs()
    manifest = {
        "tatr_matrices": int(tatr_summary.shape[0]) if not tatr_summary.empty else 0,
        "tatr_matrix_files": sorted(
            str(path.relative_to(ROOT))
            for path in (ROOT / "results" / "tatr").glob("**/results_matrix.csv")
            if is_paper_k_config(parse_tatr_path(path)["dataset"], parse_tatr_path(path)["k"])
        ),
        "local_final_figures": sorted(
            str(path.relative_to(ROOT))
            for path in (ROOT / "figures").glob("**/final.pdf")
            if len(path.relative_to(ROOT / "figures").parts) >= 4
            and is_paper_k_config(
                path.relative_to(ROOT / "figures").parts[0].upper(),
                path.relative_to(ROOT / "figures").parts[1].upper(),
            )
        ),
        "reference_summary_rows": int(ref_summary.shape[0]) if not ref_summary.empty else 0,
        "sisc_centroids": n_centroids,
        "sisc_segmentation_rows": n_segments,
        "generated_figures": sorted(path.name for path in FIG.glob("*.pdf")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(tatr_summary, ref_summary, n_centroids, n_segments)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
