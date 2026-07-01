#!/usr/bin/env python
"""Evaluate predictable-drift diagnostics for real and synthetic returns."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from fts_diffusion.evaluation.drift import (  # noqa: E402
    DRIFT_COMPONENTS,
    as_1d_float_array,
    block_shuffle_returns,
    drift_report,
    null_drift_reports,
    rolling_drift_report,
)
from fts_diffusion.evaluation.stylized import stylized_report  # noqa: E402

COMMON_COLUMNS = ("return", "returns", "r", "close", "value", "generated", "synthetic")


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def infer_value_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Column {requested!r} not found. Available columns: {list(df.columns)}")
        return requested

    lower_to_original = {str(column).lower(): column for column in df.columns}
    for candidate in COMMON_COLUMNS:
        if candidate in lower_to_original:
            return str(lower_to_original[candidate])

    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] == 1:
        return str(numeric.columns[0])
    if numeric.shape[1] > 1:
        return str(numeric.columns[0])
    raise ValueError("Could not infer a numeric value column.")


def load_return_series(
    path: Path,
    value_column: str | None = None,
    date_column: str | None = None,
    input_type: str = "return",
) -> np.ndarray:
    df = pd.read_csv(path)
    if date_column and date_column in df.columns:
        df = df.sort_values(date_column)

    column = infer_value_column(df, value_column)
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    kind = input_type or "return"
    if kind in {"return", "value"}:
        returns = values
    elif kind == "close":
        returns = values.pct_change().dropna()
    elif kind == "log_return":
        returns = np.log(values / values.shift(1)).dropna()
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    return as_1d_float_array(returns)


def chronological_split(
    values: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float | None,
) -> dict[str, np.ndarray]:
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1].")
    if val_fraction < 0:
        raise ValueError("val_fraction must be non-negative.")
    if test_fraction is not None and test_fraction < 0:
        raise ValueError("test_fraction must be non-negative.")
    if test_fraction is not None and train_fraction + val_fraction + test_fraction > 1.0000001:
        raise ValueError("train_fraction + val_fraction + test_fraction cannot exceed 1.")
    if test_fraction is None and train_fraction + val_fraction > 1.0000001:
        raise ValueError("train_fraction + val_fraction cannot exceed 1.")

    n = values.size
    train_end = int(math.floor(n * train_fraction))
    val_end = train_end + int(math.floor(n * val_fraction))
    if test_fraction is not None:
        test_end = val_end + int(math.floor(n * test_fraction))
    else:
        test_end = n
    return {
        "train": values[:train_end],
        "val": values[train_end:val_end],
        "test": values[val_end:test_end],
        "full": values,
    }


def _source_id(path: Path) -> str:
    return path.stem


def _flatten_row(
    source_type: str,
    sample_id: str,
    returns: np.ndarray,
    train_mean: float,
    train_std: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pooled = drift_report(returns, train_mean=train_mean, train_std=train_std)
    rolling = rolling_drift_report(
        returns,
        window_size=args.window_size,
        step_size=args.step_size,
        min_window=args.min_window,
        train_mean=train_mean,
        train_std=train_std,
    )
    stylized = stylized_report(returns, max_lag=args.max_lag)

    drift_row = {
        "source_type": source_type,
        "sample_id": sample_id,
        "n_obs": pooled["n_obs"],
        "pooled_delta": pooled["delta"],
        "rolling_delta_mean": rolling["rolling_delta_mean"],
        "rolling_delta_std": rolling["rolling_delta_std"],
        "rolling_delta_min": rolling["rolling_delta_min"],
        "rolling_delta_max": rolling["rolling_delta_max"],
        "n_windows": rolling["n_windows"],
        "train_mean": pooled["train_mean"],
        "train_std": pooled["train_std"],
    }
    component_row = {
        "source_type": source_type,
        "sample_id": sample_id,
        **{name: pooled[name] for name in DRIFT_COMPONENTS},
    }
    stylized_row = {
        "source_type": source_type,
        "sample_id": sample_id,
        **stylized,
    }
    return drift_row, component_row, stylized_row


def _null_summary(null_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not null_rows:
        return {"n_reps": 0}
    df = pd.DataFrame(null_rows)
    return {
        "n_reps": int(len(df)),
        "delta_mean": float(df["delta"].mean()),
        "delta_std": float(df["delta"].std(ddof=0)),
        "delta_min": float(df["delta"].min()),
        "delta_max": float(df["delta"].max()),
        "delta_q05": float(df["delta"].quantile(0.05)),
        "delta_q50": float(df["delta"].quantile(0.50)),
        "delta_q95": float(df["delta"].quantile(0.95)),
    }


def _write_report(
    output_dir: Path,
    drift_df: pd.DataFrame,
    stylized_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    merged = drift_df.merge(
        stylized_df[["source_type", "sample_id", "acf_abs_r_lag1", "excess_kurtosis"]],
        on=["source_type", "sample_id"],
        how="left",
    )
    table_columns = [
        "source_type",
        "sample_id",
        "n_obs",
        "pooled_delta",
        "rolling_delta_mean",
        "rolling_delta_std",
        "acf_abs_r_lag1",
        "excess_kurtosis",
    ]
    table = merged[table_columns].copy()
    markdown_table = _markdown_table(table)

    lines = [
        "# Predictable-Drift Diagnostic Report",
        "",
        markdown_table,
        "",
        "Lower delta means less predictable drift under the fixed feature map; this is not a full martingale/no-arbitrage proof.",
        "",
        "The diagnostic uses standardized returns with training-split mean and standard deviation.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not np.isfinite(value) else f"{value:.6g}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [
        [_format_markdown_value(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-csv", type=Path, required=True)
    parser.add_argument("--synthetic-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--value-column")
    parser.add_argument("--date-column")
    parser.add_argument(
        "--input-type",
        choices=["return", "log_return", "close", "value"],
        default=None,
    )
    parser.add_argument(
        "--synthetic-input-type",
        choices=["return", "log_return", "close", "value"],
        default=None,
        help="Input type for synthetic CSVs; defaults to return because generated files are usually returns.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--eval-split", choices=["train", "val", "test", "full"], default="test")
    parser.add_argument("--window-size", type=int, default=252)
    parser.add_argument("--step-size", type=int, default=21)
    parser.add_argument("--min-window", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=21)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lag", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drift_diagnostics"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_yaml(args.config)
    if not config and args.run_dir:
        config = _read_yaml(args.run_dir / "resolved_config.yaml")

    input_type = args.input_type or _nested_get(config, ["data", "input_type"], "return")
    synthetic_input_type = args.synthetic_input_type or "return"
    value_column = args.value_column or _nested_get(config, ["data", "value_column"])
    date_column = args.date_column or _nested_get(config, ["data", "date_column"])

    real_returns = load_return_series(
        args.real_csv,
        value_column=value_column,
        date_column=date_column,
        input_type=input_type,
    )
    splits = chronological_split(
        real_returns,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    train = splits["train"]
    real_eval = splits[args.eval_split]
    if train.size == 0:
        raise ValueError("Training split is empty; cannot compute standardization statistics.")
    if real_eval.size < 3:
        raise ValueError(f"Selected eval split {args.eval_split!r} has fewer than 3 observations.")

    train_mean = float(np.mean(train))
    train_std = float(np.std(train, ddof=0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    drift_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    stylized_rows: list[dict[str, Any]] = []

    row_triplet = _flatten_row("real", args.eval_split, real_eval, train_mean, train_std, args)
    for target, row in zip((drift_rows, component_rows, stylized_rows), row_triplet):
        target.append(row)

    for synthetic_path in args.synthetic_csv:
        synthetic_returns = load_return_series(
            synthetic_path,
            value_column=args.value_column,
            date_column=args.date_column,
            input_type=synthetic_input_type,
        )
        row_triplet = _flatten_row(
            "synthetic",
            _source_id(synthetic_path),
            synthetic_returns,
            train_mean,
            train_std,
            args,
        )
        for target, row in zip((drift_rows, component_rows, stylized_rows), row_triplet):
            target.append(row)

    null_reports = null_drift_reports(
        real_eval,
        n_reps=args.null_reps,
        block_size=args.block_size,
        seed=args.seed,
        train_mean=train_mean,
        train_std=train_std,
    )
    null_stylized_rows: list[dict[str, Any]] = []
    null_drift_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for rep, null_report in enumerate(null_reports):
        shuffled = block_shuffle_returns(
            real_eval,
            block_size=args.block_size,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        rolling = rolling_drift_report(
            shuffled,
            window_size=args.window_size,
            step_size=args.step_size,
            min_window=args.min_window,
            train_mean=train_mean,
            train_std=train_std,
        )
        sample_id = f"block_shuffle_{rep:03d}"
        null_drift_rows.append(
            {
                "source_type": "null",
                "sample_id": sample_id,
                "rep": rep,
                "n_obs": null_report["n_obs"],
                "delta": null_report["delta"],
                "pooled_delta": null_report["delta"],
                "rolling_delta_mean": rolling["rolling_delta_mean"],
                "rolling_delta_std": rolling["rolling_delta_std"],
            }
        )
        null_stylized_rows.append(
            {
                "source_type": "null",
                "sample_id": sample_id,
                **stylized_report(shuffled, max_lag=args.max_lag),
            }
        )

    if null_drift_rows:
        null_delta_values = pd.DataFrame(null_drift_rows)["delta"]
        drift_rows.append(
            {
                "source_type": "null",
                "sample_id": "block_shuffle_summary",
                "n_obs": int(real_eval.size),
                "pooled_delta": float(null_delta_values.mean()),
                "rolling_delta_mean": float(pd.DataFrame(null_drift_rows)["rolling_delta_mean"].mean()),
                "rolling_delta_std": float(pd.DataFrame(null_drift_rows)["rolling_delta_mean"].std(ddof=0)),
                "rolling_delta_min": float(null_delta_values.min()),
                "rolling_delta_max": float(null_delta_values.max()),
                "n_windows": int(args.null_reps),
                "train_mean": train_mean,
                "train_std": train_std,
            }
        )
        stylized_null_df = pd.DataFrame(null_stylized_rows)
        stylized_rows.append(
            {
                "source_type": "null",
                "sample_id": "block_shuffle_summary",
                **{
                    column: float(stylized_null_df[column].mean())
                    for column in stylized_null_df.select_dtypes(include=["number"]).columns
                    if column != "n_obs"
                },
                "n_obs": int(real_eval.size),
            }
        )

    drift_df = pd.DataFrame(drift_rows)
    component_df = pd.DataFrame(component_rows)
    stylized_df = pd.DataFrame(stylized_rows)
    null_df = pd.DataFrame(null_drift_rows)
    null_summary = _null_summary(null_reports)

    drift_df.to_csv(args.output_dir / "drift_summary.csv", index=False)
    component_df.to_csv(args.output_dir / "drift_components.csv", index=False)
    stylized_df.to_csv(args.output_dir / "stylized_summary.csv", index=False)
    null_df.to_csv(args.output_dir / "null_summary.csv", index=False)

    metadata = {
        "real_csv": str(args.real_csv),
        "synthetic_csv": [str(path) for path in args.synthetic_csv],
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "config": str(args.config) if args.config else None,
        "input_type": input_type,
        "synthetic_input_type": synthetic_input_type,
        "value_column": value_column,
        "date_column": date_column,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "eval_split": args.eval_split,
        "train_mean": train_mean,
        "train_std": train_std,
        "n_real_total": int(real_returns.size),
        "n_train": int(train.size),
        "n_eval": int(real_eval.size),
        "null_summary": null_summary,
        "interpretation": "Lower delta means less predictable drift under the fixed feature map; this is not a full martingale/no-arbitrage proof.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(args.output_dir, drift_df, stylized_df, metadata)
    print(f"Wrote predictable-drift diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
