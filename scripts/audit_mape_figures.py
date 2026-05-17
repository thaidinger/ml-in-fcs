#!/usr/bin/env python3
"""Audit + fill MAPE downstream figures (TATR / TMTR).

Scans every results folder on disk, checks for the two expected figure
variants, and generates ONLY the missing ones from the per-run CSVs.

Variant 1 ("authors' style")  -> final.{pdf,png}
    Replicates utils_downstream.py::plot_dowmstream_{tatr,tmtr}.
    Band = trimmed (2.5%) min/max. Plain line, no legend/title.

Variant 2 ("annotated")       -> final_enhanced.{pdf,png}
    Same authors' band (2.5%-trimmed min/max) plus per-point MAPE value
    annotations, markers, baseline, legend and title.

Both variants use the SAME per-run data, the SAME trimmed mean curve and
the SAME 2.5%-trimmed min/max band; variant 2 only adds presentation
chrome (markers, value annotations, legend, title).

Usage:
    python scripts/audit_mape_figures.py            # scan + fill + report
    python scripts/audit_mape_figures.py --dry-run  # report only, no writes
    python scripts/audit_mape_figures.py --force    # regenerate every figure
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo root (this file lives in <root>/scripts/) ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- constants lifted verbatim from the notebooks -------------------------
AUG_LENGTH = 252  # trading days per eval_year (TATR x-axis scale)

ASSETS_CONFIG = {
    "sp500":    {"pretty": "S&P 500"},
    "sp500_us": {"pretty": "S&P 500 (retrained from scratch)"},
    "goog":     {"pretty": "GOOG"},
    "zcf":      {"pretty": "ZC=F (Corn futures)"},
}

# Per-experiment plotting metadata.
EXP_META = {
    "tatr": {
        "x_col": "eval_year",
        "x_scale": AUG_LENGTH,
        "authors_xlabel": "Aug. Size",
        "enh_xlabel": "Augmented Size (days)",
        "baseline_label": "Initial MAPE (year 0) = {:.4f}",
        "title": "TATR",
        "x_unit": "eval points",
        "paper_ref": "\n(paper: -17.9%)",
    },
    "tmtr": {
        "x_col": "proportion_pct",
        "x_scale": 1,
        "authors_xlabel": "Syn. Prop. (%)",
        "enh_xlabel": "Synthetic Proportion (%)",
        "baseline_label": "Real-only baseline (0%) = {:.4f}",
        "title": "TMTR",
        "x_unit": "proportions",
        "paper_ref": "",
    },
}


# --- aggregation formulas (verbatim from the notebooks) -------------------
def summarize_authors_style(errors_matrix: np.ndarray) -> pd.DataFrame:
    """Replicates summarize_results from utils_downstream.py.

    errors_matrix : (n_runs, n_iters). Returns DataFrame[avg, min, max].
    """
    n_runs, n_iters = errors_matrix.shape
    summary = np.zeros((3, n_iters))
    for i in range(n_iters):
        res = errors_matrix[:, i].copy()
        res.sort()
        pencentile = max(int(np.ceil(len(res) * 0.025)), 1)
        summary[0, i] = (np.mean(res[pencentile:-pencentile])
                         if len(res) > 2 * pencentile else np.mean(res))
        summary[1, i] = res[pencentile]
        summary[2, i] = res[-pencentile]
    return pd.DataFrame({"avg": summary[0, :], "min": summary[1, :],
                         "max": summary[2, :]})


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 0):
    """Bootstrap CI of the mean (verbatim from the notebooks)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = np.array([rng.choice(x, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return lo, hi


# --- data loading ---------------------------------------------------------
def load_run_matrix(res_dir: str, x_col: str):
    """Load all run_*.csv into a (run x x_value) pivot table.

    Returns (df_pivot, x_values_sorted) or (None, None) if no usable runs.
    """
    run_files = sorted(glob.glob(os.path.join(res_dir, "run_*.csv")))
    rows = {}
    for rf in run_files:
        df = pd.read_csv(rf)
        if x_col not in df.columns or "mape" not in df.columns:
            continue
        idx = os.path.basename(rf)
        rows[idx] = dict(zip(df[x_col].astype(int), df["mape"].astype(float)))
    if not rows:
        return None, None
    df_pivot = pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)
    x_values = list(df_pivot.columns)
    return df_pivot, x_values


def compute_summaries(df_pivot: pd.DataFrame, x_values: list):
    """Build both summary tables from the per-run pivot."""
    errors_matrix = df_pivot[x_values].to_numpy()

    auth = summarize_authors_style(errors_matrix)
    auth["x"] = x_values
    auth["n_runs"] = (~np.isnan(errors_matrix)).sum(axis=0)

    boot_rows = []
    for j, xv in enumerate(x_values):
        vals = errors_matrix[:, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_ci(vals)
        boot_rows.append({"x": xv, "mean_mape": vals.mean(),
                          "std_mape": vals.std(), "ci95_low": lo,
                          "ci95_high": hi, "n_runs": len(vals)})
    boot = pd.DataFrame(boot_rows)
    return auth, boot


def load_existing_summary(res_dir: str, fname: str, x_col_csv: str):
    """Reuse a saved summary CSV if present; returns DataFrame or None.

    Renames the experiment-specific x column to a generic 'x'.
    """
    path = os.path.join(res_dir, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if x_col_csv in df.columns:
        df = df.rename(columns={x_col_csv: "x"})
    return df


# --- figure generation ----------------------------------------------------
def plot_authors_style(auth: pd.DataFrame, meta: dict, fig_dir: str):
    """Variant 1 -> final.{pdf,png}. Replicates plot_dowmstream_* verbatim."""
    fig, ax = plt.subplots()
    x_range = np.array(auth["x"].values) * meta["x_scale"]
    error_avg = auth["avg"].values
    ax.plot(x_range, error_avg)
    ax.fill_between(x_range, auth["min"].values, auth["max"].values, alpha=0.2)
    ax.axhline(y=error_avg[0], color="gray", linestyle="--")
    ax.set_xlabel(meta["authors_xlabel"])
    ax.set_ylabel("MAPE")
    plt.tight_layout()
    pdf = os.path.join(fig_dir, "final.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_enhanced_style(summ: pd.DataFrame, meta: dict, pretty: str,
                        k: int, protocol: str, fig_dir: str):
    """Variant 2 -> final_enhanced.{pdf,png}.

    Authors' confidence band (2.5%-trimmed min/max, fill_between alpha=0.2,
    exactly as plot_dowmstream_tatr) plus per-point MAPE annotations,
    markers, baseline, legend and title.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    mean = summ["avg"].values
    lo = summ["min"].values
    hi = summ["max"].values
    xv = np.array(summ["x"].values) * meta["x_scale"]
    n_runs = int(np.max(summ["n_runs"].values))

    ax.plot(xv, mean, "o-", linewidth=2.5, color="steelblue", markersize=7,
            label=f"FTS-Diffusion (n={n_runs} runs)")
    ax.fill_between(xv, lo, hi, alpha=0.2, color="steelblue",
                    label="min/max band (2.5% trimmed)")
    ax.axhline(mean[0], color="gray", linestyle="--", alpha=0.7,
               label=meta["baseline_label"].format(mean[0]))

    for x, y in zip(xv, mean):
        if np.isnan(y):
            continue
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color="darkblue",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="darkblue", alpha=0.85, linewidth=0.8))

    if int(summ["x"].iloc[-1]) == 100 and not np.isnan(mean[-1]) \
            and not np.isnan(mean[0]):
        pct = 100 * (mean[-1] - mean[0]) / mean[0]
        tag = ("ΔMAPE @ year 100" if meta["title"] == "TATR"
               else "ΔMAPE @ 100%")
        ax.annotate(f"{tag}: {pct:+.1f}%{meta['paper_ref']}",
                    xy=(xv[-1], mean[-1]),
                    xytext=(xv[-1] * 0.55, mean[-1] + 0.02),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=11, color="red")

    ax.set_xlabel(meta["enh_xlabel"], fontsize=12)
    ax.set_ylabel("MAPE (real test set)", fontsize=12)
    ax.set_title(f"{meta['title']} — {pretty} (K={k}, {protocol!r}) — "
                 f"{n_runs} runs × {len(summ)} {meta['x_unit']}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.015)
    plt.tight_layout()
    pdf = os.path.join(fig_dir, "final_enhanced.pdf")
    fig.savefig(pdf, bbox_inches="tight", dpi=150)
    fig.savefig(pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# --- audit driver ---------------------------------------------------------
def variant_present(fig_dir: str, stem: str) -> bool:
    """A variant counts as present only if BOTH .pdf and .png exist."""
    return (os.path.exists(os.path.join(fig_dir, f"{stem}.pdf"))
            and os.path.exists(os.path.join(fig_dir, f"{stem}.png")))


def discover_result_folders(experiment: str):
    """Yield (asset, k, protocol, res_dir) for every leaf results folder."""
    base = os.path.join(ROOT, "results", experiment)
    if not os.path.isdir(base):
        return
    for asset in sorted(os.listdir(base)):
        a_dir = os.path.join(base, asset)
        if not os.path.isdir(a_dir):
            continue
        for kdir in sorted(os.listdir(a_dir)):
            if not (kdir.startswith("k") and kdir[1:].isdigit()):
                continue
            k = int(kdir[1:])
            kp = os.path.join(a_dir, kdir)
            for protocol in sorted(os.listdir(kp)):
                res_dir = os.path.join(kp, protocol)
                if os.path.isdir(res_dir):
                    yield asset, k, protocol, res_dir


def audit(dry_run: bool = False, force: bool = False):
    results = []  # list of row dicts
    for experiment in ("tatr", "tmtr"):
        meta = EXP_META[experiment]
        for asset, k, protocol, res_dir in discover_result_folders(experiment):
            n_runs = len(glob.glob(os.path.join(res_dir, "run_*.csv")))
            fig_dir = os.path.join(ROOT, "figures", experiment, asset,
                                   f"k{k}", protocol)
            auth_was = variant_present(fig_dir, "final")
            enh_was = variant_present(fig_dir, "final_enhanced")
            need_auth = force or not auth_was
            need_enh = force or not enh_was

            row = {"exp": experiment, "asset": asset, "k": k,
                   "protocol": protocol, "n_runs": n_runs,
                   "auth_was": auth_was, "auth_now": auth_was,
                   "enh_was": enh_was, "enh_now": enh_was,
                   "status": "OK", "note": ""}

            if n_runs == 0:
                row["status"] = "NO RESULTS"
                results.append(row)
                continue

            if not need_auth and not need_enh:
                results.append(row)  # OK, nothing to do
                continue

            if dry_run:
                row["status"] = "FILLED"  # would fill
                row["note"] = "(dry-run: not written)"
                results.append(row)
                continue

            try:
                os.makedirs(fig_dir, exist_ok=True)
                df_pivot, x_values = load_run_matrix(res_dir, meta["x_col"])
                if df_pivot is None:
                    row["status"] = "FAILED"
                    row["note"] = "run CSVs unreadable"
                    results.append(row)
                    continue

                # Authors' trimmed (2.5%) summary -> avg / min / max, always
                # recomputed from the per-run CSVs. Both figure variants share
                # this table; they only differ in presentation chrome.
                auth, _ = compute_summaries(df_pivot, x_values)

                if need_auth:
                    plot_authors_style(auth, meta, fig_dir)
                    row["auth_now"] = True

                if need_enh:
                    pretty = ASSETS_CONFIG.get(asset, {}).get("pretty", asset)
                    plot_enhanced_style(auth, meta, pretty, k, protocol,
                                        fig_dir)
                    row["enh_now"] = True

                row["status"] = "FILLED"
            except Exception:
                row["status"] = "FAILED"
                row["note"] = traceback.format_exc().splitlines()[-1]
            results.append(row)
    return results


def yn(was: bool, now: bool) -> str:
    return f"{'Y' if was else 'N'}->{'Y' if now else 'N'}"


def print_report(results: list):
    hdr = (f"{'EXPERIMENT':<32} {'#runs':>6}  {'authors':>9}  "
           f"{'enhanced':>9}  STATUS")
    print("\n" + "=" * 78)
    print("MAPE FIGURE AUDIT")
    print("=" * 78)
    print(hdr)
    print("-" * 78)
    action = []
    for r in sorted(results, key=lambda r: (r["exp"], r["asset"], r["k"],
                                            r["protocol"])):
        name = f"{r['exp']}/{r['asset']}/k{r['k']}/{r['protocol']}"
        line = (f"{name:<32} {r['n_runs']:>6}  "
                f"{yn(r['auth_was'], r['auth_now']):>9}  "
                f"{yn(r['enh_was'], r['enh_now']):>9}  {r['status']}")
        if r["note"]:
            line += f"  {r['note']}"
        print(line)
        if r["status"] != "OK":
            action.append((name, r["status"], r["note"]))

    print("-" * 78)
    n_ok = sum(r["status"] == "OK" for r in results)
    n_fill = sum(r["status"] == "FILLED" for r in results)
    n_none = sum(r["status"] == "NO RESULTS" for r in results)
    n_fail = sum(r["status"] == "FAILED" for r in results)
    print(f"Totals: {len(results)} folders | OK={n_ok}  FILLED={n_fill}  "
          f"NO RESULTS={n_none}  FAILED={n_fail}")

    if action:
        print("\nFolders needing attention:")
        for name, status, note in action:
            print(f"  [{status:<10}] {name}" + (f"  {note}" if note else ""))
    else:
        print("\nAll folders OK — nothing to do.")
    print("=" * 78)
    return n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; do not write any figure files")
    ap.add_argument("--force", action="store_true",
                    help="regenerate every figure even if it already exists")
    args = ap.parse_args()
    results = audit(dry_run=args.dry_run, force=args.force)
    n_fail = print_report(results)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
