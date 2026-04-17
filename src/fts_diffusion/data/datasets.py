from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from fts_diffusion.models.sisc import SISCResult


@dataclass
class SegmentExample:
    raw: np.ndarray
    normalized: np.ndarray
    pattern: np.ndarray
    length: int
    alpha: float
    beta: float
    cluster_id: int


class SegmentDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, series: np.ndarray, sisc_result: SISCResult) -> None:
        self.examples: list[SegmentExample] = []
        for segment in sisc_result.segments:
            raw = np.asarray(series[segment.start : segment.end], dtype=np.float32)
            normalized = raw / max(segment.beta, 1e-6)
            self.examples.append(
                SegmentExample(
                    raw=raw,
                    normalized=normalized.astype(np.float32),
                    pattern=np.asarray(sisc_result.patterns[segment.cluster_id], dtype=np.float32),
                    length=segment.length,
                    alpha=segment.alpha,
                    beta=segment.beta,
                    cluster_id=segment.cluster_id,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "raw": torch.from_numpy(example.raw),
            "normalized": torch.from_numpy(example.normalized),
            "pattern": torch.from_numpy(example.pattern),
            "length": torch.tensor(example.length, dtype=torch.long),
            "alpha": torch.tensor(example.alpha, dtype=torch.float32),
            "beta": torch.tensor(example.beta, dtype=torch.float32),
            "cluster_id": torch.tensor(example.cluster_id, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = int(max(item["length"].item() for item in batch))
        raw = torch.zeros(len(batch), max_length, dtype=torch.float32)
        normalized = torch.zeros(len(batch), max_length, dtype=torch.float32)
        pattern = torch.stack([item["pattern"] for item in batch], dim=0)
        lengths = torch.stack([item["length"] for item in batch], dim=0)
        alphas = torch.stack([item["alpha"] for item in batch], dim=0)
        betas = torch.stack([item["beta"] for item in batch], dim=0)
        cluster_ids = torch.stack([item["cluster_id"] for item in batch], dim=0)

        for index, item in enumerate(batch):
            length = int(item["length"].item())
            raw[index, :length] = item["raw"]
            normalized[index, :length] = item["normalized"]

        return {
            "raw": raw,
            "normalized": normalized,
            "pattern": pattern,
            "length": lengths,
            "alpha": alphas,
            "beta": betas,
            "cluster_id": cluster_ids,
        }


class TransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, sisc_result: SISCResult) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        segments = sisc_result.segments
        for current, nxt in zip(segments[:-1], segments[1:]):
            self.items.append(
                {
                    "current_pattern": torch.tensor(current.cluster_id, dtype=torch.long),
                    "current_alpha": torch.tensor(current.alpha, dtype=torch.float32),
                    "current_beta": torch.tensor(current.beta, dtype=torch.float32),
                    "next_pattern": torch.tensor(nxt.cluster_id, dtype=torch.long),
                    "next_alpha": torch.tensor(nxt.alpha, dtype=torch.float32),
                    "next_beta": torch.tensor(nxt.beta, dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]

