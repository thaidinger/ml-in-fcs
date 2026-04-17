from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from fts_diffusion.config import PatternConfig
from fts_diffusion.utils.interpolation import resample_1d, rms_scale, z_normalize


@dataclass
class SegmentRecord:
    start: int
    end: int
    cluster_id: int
    length: int
    alpha: float
    beta: float
    distance: float


@dataclass
class SISCResult:
    patterns: np.ndarray
    segments: list[SegmentRecord]
    config: dict

    def to_dict(self) -> dict:
        return {
            "patterns": self.patterns.tolist(),
            "segments": [asdict(segment) for segment in self.segments],
            "config": self.config,
        }


def dtw_distance(x: np.ndarray, y: np.ndarray, window: Optional[int] = None) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n, m = len(x), len(y)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    table = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    table[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = abs(x[i - 1] - y[j - 1])
            table[i, j] = cost + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(table[n, m] / max(n + m, 1))


class SISC:
    """Scale-Invariant Subsequence Clustering from the paper."""

    def __init__(self, config: PatternConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.patterns_: Optional[np.ndarray] = None
        self.segments_: Optional[list[SegmentRecord]] = None

    @property
    def centroid_length(self) -> int:
        assert self.config.centroid_length is not None
        return self.config.centroid_length

    @property
    def init_length(self) -> int:
        assert self.config.init_length is not None
        return self.config.init_length

    def fit(self, series: np.ndarray) -> SISCResult:
        series = np.asarray(series, dtype=np.float32)
        best_score = float("inf")
        best_patterns: Optional[np.ndarray] = None
        best_segments: Optional[list[SegmentRecord]] = None

        for _ in range(self.config.random_restarts):
            patterns = self._initialize_patterns(series)
            segments = []
            for _iteration in range(self.config.max_iters):
                segments = self._greedy_segmentation(series, patterns)
                updated_patterns = self._update_patterns(series, patterns, segments)
                patterns = updated_patterns
            score = float(np.mean([segment.distance for segment in segments]))
            if score < best_score:
                best_score = score
                best_patterns = patterns.copy()
                best_segments = segments

        if best_patterns is None or best_segments is None:
            raise RuntimeError("SISC failed to produce a valid clustering result.")

        self.patterns_ = best_patterns
        self.segments_ = best_segments
        return SISCResult(
            patterns=best_patterns,
            segments=best_segments,
            config={
                "num_patterns": self.config.num_patterns,
                "min_length": self.config.min_length,
                "max_length": self.config.max_length,
                "init_length": self.init_length,
                "centroid_length": self.centroid_length,
                "max_iters": self.config.max_iters,
            },
        )

    def _initialize_patterns(self, series: np.ndarray) -> np.ndarray:
        candidates: list[np.ndarray] = []
        for start in range(0, len(series) - self.init_length + 1):
            segment = series[start : start + self.init_length]
            candidates.append(z_normalize(resample_1d(segment, self.centroid_length)))
        if len(candidates) < self.config.num_patterns:
            raise ValueError("Not enough candidates to initialize the requested number of clusters.")

        chosen_indices: list[int] = [int(self.rng.integers(0, len(candidates)))]
        while len(chosen_indices) < self.config.num_patterns:
            distances = []
            for index, candidate in enumerate(candidates):
                if index in chosen_indices:
                    distances.append(0.0)
                    continue
                nearest = min(
                    dtw_distance(candidate, candidates[chosen], self.config.dtw_window)
                    for chosen in chosen_indices
                )
                distances.append(max(nearest, 1e-6))
            probabilities = np.asarray(distances, dtype=np.float64)
            probabilities = probabilities / probabilities.sum()
            next_index = int(self.rng.choice(len(candidates), p=probabilities))
            while next_index in chosen_indices:
                next_index = int(self.rng.choice(len(candidates), p=probabilities))
            chosen_indices.append(next_index)
        return np.stack([candidates[index] for index in chosen_indices], axis=0)

    def _greedy_segmentation(self, series: np.ndarray, patterns: np.ndarray) -> list[SegmentRecord]:
        segments: list[SegmentRecord] = []
        position = 0
        total_length = len(series)

        while position < total_length:
            remaining = total_length - position
            if remaining < self.config.min_length:
                if not segments:
                    raise ValueError("Series is shorter than the configured minimum segment length.")
                previous = segments[-1]
                previous.end = total_length
                previous.length = previous.end - previous.start
                previous.alpha = previous.length / self.centroid_length
                previous.beta = rms_scale(series[previous.start : previous.end])
                previous.distance = dtw_distance(
                    z_normalize(series[previous.start : previous.end]),
                    patterns[previous.cluster_id],
                    self.config.dtw_window,
                )
                break

            best_length: Optional[int] = None
            best_cluster = 0
            best_distance = float("inf")
            upper = min(self.config.max_length, remaining)
            for length in range(self.config.min_length, upper + 1):
                candidate = z_normalize(series[position : position + length])
                for cluster_id, pattern in enumerate(patterns):
                    distance = dtw_distance(candidate, pattern, self.config.dtw_window)
                    if distance < best_distance:
                        best_distance = distance
                        best_length = length
                        best_cluster = cluster_id

            assert best_length is not None
            raw_segment = series[position : position + best_length]
            magnitude = rms_scale(raw_segment)
            segments.append(
                SegmentRecord(
                    start=position,
                    end=position + best_length,
                    cluster_id=best_cluster,
                    length=best_length,
                    alpha=best_length / self.centroid_length,
                    beta=magnitude,
                    distance=best_distance,
                )
            )
            position += best_length

        return segments

    def _update_patterns(
        self, series: np.ndarray, current_patterns: np.ndarray, segments: list[SegmentRecord]
    ) -> np.ndarray:
        updated = []
        for cluster_id in range(self.config.num_patterns):
            cluster_segments = [
                z_normalize(series[segment.start : segment.end])
                for segment in segments
                if segment.cluster_id == cluster_id
            ]
            if not cluster_segments:
                updated.append(current_patterns[cluster_id])
                continue
            stacked = np.stack(
                [resample_1d(segment, self.centroid_length) for segment in cluster_segments],
                axis=0,
            )
            centroid = z_normalize(stacked.mean(axis=0))
            updated.append(centroid)
        return np.stack(updated, axis=0)
