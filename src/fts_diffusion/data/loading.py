from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fts_diffusion.config import DataConfig


@dataclass
class LoadedSeries:
    train_values: np.ndarray
    test_values: np.ndarray
    all_values: np.ndarray
    train_mean: float
    train_std: float


def _read_frame(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def _resolve_value_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_columns) == 1:
        return numeric_columns[0]
    raise ValueError(
        f"Could not resolve value column '{requested}'. Available numeric columns: {numeric_columns}"
    )


def load_financial_series(config: DataConfig) -> LoadedSeries:
    frame = _read_frame(config.path)
    if config.date_column and config.date_column in frame.columns:
        frame = frame.sort_values(config.date_column).reset_index(drop=True)

    value_column = _resolve_value_column(frame, config.value_column)
    values = frame[value_column].astype("float32").to_numpy()

    if config.input_type == "close":
        values = np.diff(values) / np.clip(values[:-1], 1e-8, None)
    elif config.input_type == "log_return":
        values = np.diff(np.log(np.clip(values, 1e-8, None)))
    elif config.input_type in {"return", "value"}:
        values = values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported input_type={config.input_type}")

    split_index = int(len(values) * config.train_ratio)
    if split_index <= 0 or split_index >= len(values):
        raise ValueError("train_ratio produces an empty train or test split.")

    train_values = values[:split_index].astype(np.float32)
    test_values = values[split_index:].astype(np.float32)
    train_mean = float(train_values.mean()) if config.standardize else 0.0
    train_std = float(train_values.std()) if config.standardize else 1.0
    train_std = max(train_std, 1e-6)

    if config.standardize:
        train_values = ((train_values - train_mean) / train_std).astype(np.float32)
        test_values = ((test_values - train_mean) / train_std).astype(np.float32)

    all_values = np.concatenate([train_values, test_values]).astype(np.float32)
    return LoadedSeries(
        train_values=train_values,
        test_values=test_values,
        all_values=all_values,
        train_mean=train_mean,
        train_std=train_std,
    )
