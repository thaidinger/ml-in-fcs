from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    path: str = "data/asset.csv"
    value_column: str = "close"
    date_column: Optional[str] = "date"
    input_type: str = "close"
    train_ratio: float = 0.8
    standardize: bool = True
    output_series_name: str = "returns"


@dataclass
class PatternConfig:
    num_patterns: int = 14
    min_length: int = 10
    max_length: int = 21
    max_iters: int = 10
    init_length: Optional[int] = None
    centroid_length: Optional[int] = None
    dtw_window: Optional[int] = None
    random_restarts: int = 1


@dataclass
class GenerationConfig:
    fixed_length: Optional[int] = None
    hidden_dim: int = 64
    rnn_type: str = "gru"
    rnn_layers: int = 2
    diffusion_steps: int = 100
    residual_blocks: int = 6
    tcn_channels: int = 64
    kernel_size: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 200
    grad_clip_norm: float = 1.0


@dataclass
class EvolutionConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 4e-4
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 1000
    grad_clip_norm: float = 1.0


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    alpha_noise: float = 0.05
    beta_noise: float = 0.05
    min_beta: float = 1e-3
    default_terminal_length: int = 2048


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    output_dir: str = "outputs/default"
    checkpoint_every: int = 25


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_name, field_info in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name in values:
            raw_value = values[field_name]
        elif field_info.default is not MISSING:
            raw_value = field_info.default
        elif field_info.default_factory is not MISSING:
            raw_value = field_info.default_factory()
        else:
            raise KeyError(f"Missing required config field: {field_name}")
        if is_dataclass(field_info.type) and isinstance(raw_value, dict):
            kwargs[field_name] = _dataclass_from_dict(field_info.type, raw_value)
            continue
        nested_default = getattr(cls(), field_name)
        if is_dataclass(nested_default) and isinstance(raw_value, dict):
            kwargs[field_name] = _dataclass_from_dict(type(nested_default), raw_value)
            continue
        kwargs[field_name] = raw_value
    return cls(**kwargs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = _dataclass_from_dict(ExperimentConfig, payload)
    if config.pattern.init_length is None:
        config.pattern.init_length = config.pattern.max_length
    if config.pattern.centroid_length is None:
        config.pattern.centroid_length = config.pattern.max_length
    if config.generation.fixed_length is None:
        config.generation.fixed_length = config.pattern.centroid_length
    return config


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
