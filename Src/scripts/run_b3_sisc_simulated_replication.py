from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report, confusion_matrix
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw as tslearn_dtw


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "fts-diffusion-ref"
DATA = REF / "data"


def patch_reference_sisc() -> None:
    from models.pattern_recognition_module import SISC

    def _weight_next_centroid(self, series, remain_idxs, centroids):
        weights = np.zeros(len(remain_idxs), dtype=float)
        for pos, idx in enumerate(remain_idxs):
            seq = series[idx : idx + self.l_max]
            weights[pos] = min(self._compute_scaled_dtw(seq, c) for c in centroids)
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(len(remain_idxs), dtype=float) / len(remain_idxs)
        return weights / total

    def stop_criteria(self, new_centroids, new_loss, epsilon=1e-6):
        if not hasattr(self, "coverage"):
            self.coverage = 0
        for old, new in zip(self.centroids, new_centroids):
            if self._compute_scaled_dtw(old, new) > epsilon:
                self.coverage = 0
                return False
        if abs(self.total_loss - new_loss) > epsilon:
            self.coverage = 0
            return False
        if self.coverage <= 3:
            self.coverage += 1
            return False
        return True

    SISC._weight_next_centroid = _weight_next_centroid
    SISC.stop_criteria = stop_criteria


def normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def per_unit_dtw_error(c_real: np.ndarray, c_sisc: np.ndarray) -> float:
    c_real_n = normalize_to_unit(c_real)
    c_sisc_n = normalize_to_unit(c_sisc)
    return float(tslearn_dtw(c_real_n, c_sisc_n) / len(c_real_n))


def jaccard_segmentation(boundaries_real: np.ndarray, boundaries_sisc: np.ndarray, tol: int = 2) -> float:
    matched_real = 0
    for boundary in boundaries_real:
        if any(abs(boundary - pred_boundary) <= tol for pred_boundary in boundaries_sisc):
            matched_real += 1
    union = len(boundaries_real) + len(boundaries_sisc) - matched_real
    return matched_real / union if union else 0.0


def dense_labels(labels: np.ndarray, boundaries: np.ndarray, total_length: int) -> np.ndarray:
    dense = np.full(total_length, -1, dtype=int)
    for i, label in enumerate(labels):
        if i + 1 < len(boundaries):
            dense[boundaries[i] : boundaries[i + 1]] = label
    return dense


def pointwise_label_accuracy(
    real_labels: np.ndarray,
    real_boundaries: np.ndarray,
    sisc_labels: np.ndarray,
    sisc_boundaries: np.ndarray,
    total_length: int,
    alignment: dict[int, int],
) -> float:
    real_dense = dense_labels(real_labels, real_boundaries, total_length)
    sisc_dense = dense_labels(sisc_labels, sisc_boundaries, total_length)
    inv_alignment = {sisc_k: gt_k for gt_k, sisc_k in alignment.items()}
    sisc_aligned = np.array([inv_alignment.get(s, -1) if s >= 0 else -1 for s in sisc_dense])
    valid = (real_dense >= 0) & (sisc_aligned >= 0)
    return float((real_dense[valid] == sisc_aligned[valid]).mean()) if valid.sum() else 0.0


def load_toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = pd.read_csv(DATA / "data_toy_l10-20_timeseries.csv", index_col=0).values.flatten().astype(np.float32)
    labels = pd.read_csv(DATA / "data_toy_l10-20_labels.csv", index_col=0).values.flatten().astype(int)
    boundaries = pd.read_csv(DATA / "data_toy_l10-20_segmentation.csv", index_col=0).values.flatten().astype(int)
    return ts, labels, boundaries


def compute_ground_truth_centroids(
    ts: np.ndarray,
    labels: np.ndarray,
    boundaries: np.ndarray,
    k_true: int,
    l_ref: int,
) -> list[np.ndarray]:
    centroids = []
    for k in range(k_true):
        segments = [ts[boundaries[i] : boundaries[i + 1]].reshape(-1, 1) for i in np.where(labels == k)[0]]
        barycenter = dtw_barycenter_averaging(segments, max_iter=20).flatten()
        if len(barycenter) != l_ref:
            barycenter = np.interp(np.linspace(0, 1, l_ref), np.linspace(0, 1, len(barycenter)), barycenter)
        centroids.append(barycenter)
    return centroids


def run_sisc(series: np.ndarray, k: int, l_min: int, l_max: int, max_iters: int):
    from models.pattern_recognition_module import SISC

    sisc = SISC(n_clusters=k, l_min=l_min, l_max=l_max)
    sisc.fit(
        series,
        max_iters=max_iters,
        init_strategy="kmeans++",
        barycenter="dba",
        plot_progress=False,
        store_res=False,
    )
    return sisc


def align_centroids(gt_centroids: list[np.ndarray], sisc_centroids: list[np.ndarray]) -> tuple[dict[int, int], np.ndarray]:
    k_true = len(gt_centroids)
    dist_matrix = np.zeros((k_true, k_true), dtype=float)
    for i in range(k_true):
        for j in range(k_true):
            dist_matrix[i, j] = per_unit_dtw_error(gt_centroids[i], sisc_centroids[j])
    rows, cols = linear_sum_assignment(dist_matrix)
    return dict(zip(rows, cols)), dist_matrix


def per_segment_majority_accuracy(
    real_labels: np.ndarray,
    real_boundaries: np.ndarray,
    sisc_labels: np.ndarray,
    sisc_boundaries: np.ndarray,
    total_length: int,
    alignment: dict[int, int],
) -> tuple[float, np.ndarray, str]:
    sisc_dense = dense_labels(sisc_labels, sisc_boundaries, total_length)
    inv_alignment = {sisc_k: gt_k for gt_k, sisc_k in alignment.items()}
    sisc_dense_aligned = np.array([inv_alignment.get(s, -1) for s in sisc_dense])
    predicted = []
    for i in range(len(real_labels)):
        segment_labels = sisc_dense_aligned[real_boundaries[i] : real_boundaries[i + 1]]
        segment_labels = segment_labels[segment_labels >= 0]
        predicted.append(Counter(segment_labels).most_common(1)[0][0] if len(segment_labels) else -1)
    predicted = np.asarray(predicted)
    valid = predicted >= 0
    cm = confusion_matrix(real_labels[valid], predicted[valid])
    report = classification_report(real_labels[valid], predicted[valid], output_dict=False, zero_division=0)
    return float((real_labels[valid] == predicted[valid]).mean()), cm, report


def build_one_pattern_series(
    ts: np.ndarray,
    labels: np.ndarray,
    boundaries: np.ndarray,
    target_pattern: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    indices = np.where(labels == target_pattern)[0]
    raw_segments = [ts[boundaries[i] : boundaries[i + 1]] for i in indices]
    parts = [raw_segments[0]]
    for segment in raw_segments[1:]:
        parts.append(segment + parts[-1][-1] - segment[0])
    new_ts = np.concatenate(parts).astype(np.float32)
    new_boundaries = np.concatenate([[0], np.cumsum([len(s) for s in raw_segments])])
    return new_ts, parts, new_boundaries


def plot_multi_pattern(
    out: Path,
    ts: np.ndarray,
    real_labels: np.ndarray,
    real_boundaries: np.ndarray,
    gt_centroids: list[np.ndarray],
    sisc_centroids: list[np.ndarray],
    alignment: dict[int, int],
    errors: list[float],
    avg_dtw: float,
    boundary_jaccard: float,
    pointwise_acc: float,
) -> None:
    colors = ["#386cb0", "#f46d43", "#4daf4a", "#984ea3"]
    fig, axes = plt.subplots(len(gt_centroids), 3, figsize=(13, 3 * len(gt_centroids)))
    for i in range(len(gt_centroids)):
        sisc_idx = alignment[i]
        seg_indices = np.where(real_labels == i)[0]
        seg_start = real_boundaries[seg_indices[0]]
        seg_end = real_boundaries[seg_indices[0] + 1]
        ctx_start = max(0, seg_start - 20)
        ctx_end = min(len(ts), seg_end + 20)
        axes[i, 0].plot(range(ctx_start, ctx_end), ts[ctx_start:ctx_end], color="0.55", linewidth=0.8)
        axes[i, 0].plot(range(seg_start, seg_end), ts[seg_start:seg_end], color=colors[i], linewidth=2)
        axes[i, 0].axvspan(seg_start, seg_end, alpha=0.16, color=colors[i])
        axes[i, 0].set_title(f"Series excerpt: pattern {i}")
        axes[i, 1].plot(normalize_to_unit(gt_centroids[i]), color="#2a7f3f", linewidth=2)
        axes[i, 1].set_title(f"Ground truth p{i + 1}")
        axes[i, 2].plot(normalize_to_unit(sisc_centroids[sisc_idx]), color="#d95f02", linewidth=2)
        axes[i, 2].set_title(f"SISC p{i + 1}, DTW={errors[i]:.4f}")
        for ax in axes[i]:
            ax.grid(alpha=0.25, linewidth=0.7)
    fig.suptitle(
        "Appendix B.3 / Fig. 9 Replication: Multi-pattern SISC\n"
        f"avg DTW={avg_dtw:.4f} (paper ~0.01), boundary Jaccard={boundary_jaccard:.4f}, "
        f"pointwise accuracy={pointwise_acc:.4f} (paper Jaccard 0.784)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig9_multi_pattern_replication.png", dpi=220)
    fig.savefig(out / "fig9_multi_pattern_replication.pdf")
    plt.close(fig)


def plot_confusion(out: Path, cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted SISC label, aligned")
    ax.set_ylabel("Ground truth label")
    ax.set_title("B.3 Multi-pattern Segment Majority Confusion")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(out / "fig9_confusion_matrix.png", dpi=220)
    fig.savefig(out / "fig9_confusion_matrix.pdf")
    plt.close(fig)


def plot_one_pattern(out: Path, gt_centroids: list[np.ndarray], one_results: list[dict[str, object]]) -> None:
    colors = ["#386cb0", "#f46d43", "#4daf4a", "#984ea3"]
    fig, axes = plt.subplots(len(one_results), 4, figsize=(16, 3 * len(one_results)))
    for i, res in enumerate(one_results):
        ts_i = res["series"]
        boundaries = res["real_boundaries"]
        zoom = min(200, len(ts_i))
        axes[i, 0].plot(ts_i[:zoom], color="black", linewidth=0.8, alpha=0.65)
        for j in range(min(12, len(boundaries) - 1)):
            if boundaries[j + 1] > zoom:
                break
            axes[i, 0].axvspan(boundaries[j], boundaries[j + 1], color=colors[i], alpha=0.15)
        axes[i, 0].set_title(f"One-pattern series p{i + 1}")
        axes[i, 1].plot(normalize_to_unit(gt_centroids[i]), color="#2a7f3f", linewidth=2)
        axes[i, 1].set_title(f"Ground truth p{i + 1}")
        axes[i, 2].plot(normalize_to_unit(res["sisc_centroid"]), color="#d95f02", linewidth=2)
        axes[i, 2].set_title(f"SISC p{i + 1}, DTW={res['dtw_error']:.4f}")
        axes[i, 3].plot(normalize_to_unit(gt_centroids[i]), color="#2a7f3f", linewidth=2, label="GT")
        axes[i, 3].plot(normalize_to_unit(res["sisc_centroid"]), color="#d95f02", linewidth=2, linestyle="--", label="SISC")
        axes[i, 3].set_title("Overlay")
        axes[i, 3].legend(fontsize=8)
        for ax in axes[i]:
            ax.grid(alpha=0.25, linewidth=0.7)
    fig.suptitle("Appendix B.3 / Fig. 8 Replication: One-pattern SISC", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig8_one_pattern_replication.png", dpi=220)
    fig.savefig(out / "fig8_one_pattern_replication.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate Appendix B.3 SISC simulated-data investigation.")
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/generated_outputs/14_b3_sisc_simulated_replication")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    os.chdir(REF)
    sys.path.insert(0, str(REF))
    patch_reference_sisc()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ts, real_labels, real_boundaries = load_toy_data()
    k_true = len(np.unique(real_labels))
    l_min, l_max = 10, 20
    gt_centroids = compute_ground_truth_centroids(ts, real_labels, real_boundaries, k_true, l_max)

    sisc = run_sisc(ts, k_true, l_min, l_max, args.max_iters)
    sisc_centroids = [np.asarray(c) for c in sisc.centroids]
    sisc_boundaries = np.asarray(sisc.segmentation)
    sisc_labels = np.asarray(sisc.labels)
    alignment, dist_matrix = align_centroids(gt_centroids, sisc_centroids)
    errors = [per_unit_dtw_error(gt_centroids[i], sisc_centroids[alignment[i]]) for i in range(k_true)]
    avg_dtw = float(np.mean(errors))
    boundary_jaccard = jaccard_segmentation(real_boundaries, sisc_boundaries, tol=2)
    pointwise_acc = pointwise_label_accuracy(real_labels, real_boundaries, sisc_labels, sisc_boundaries, len(ts), alignment)
    segment_acc, cm, cls_report = per_segment_majority_accuracy(
        real_labels, real_boundaries, sisc_labels, sisc_boundaries, len(ts), alignment
    )

    one_results = []
    for k in range(k_true):
        ts_one, segments_one, boundaries_one = build_one_pattern_series(ts, real_labels, real_boundaries, k)
        sisc_one = run_sisc(ts_one, 1, l_min, l_max, args.max_iters)
        centroid = np.asarray(sisc_one.centroids[0])
        one_results.append(
            {
                "pattern": k,
                "series": ts_one,
                "real_boundaries": boundaries_one,
                "series_length": len(ts_one),
                "n_segments_real": len(segments_one),
                "n_segments_sisc": len(sisc_one.subsequences),
                "sisc_centroid": centroid,
                "dtw_error": per_unit_dtw_error(gt_centroids[k], centroid),
                "boundary_jaccard": jaccard_segmentation(boundaries_one, np.asarray(sisc_one.segmentation), tol=2),
            }
        )

    plot_multi_pattern(out, ts, real_labels, real_boundaries, gt_centroids, sisc_centroids, alignment, errors, avg_dtw, boundary_jaccard, pointwise_acc)
    plot_confusion(out, cm)
    plot_one_pattern(out, gt_centroids, one_results)

    multi_rows = [
        {"metric": f"per_unit_dtw_pattern_{i}", "paper": "approx 0.01", "ours": errors[i]} for i in range(k_true)
    ]
    multi_rows.extend(
        [
            {"metric": "avg_per_unit_dtw", "paper": "0.01", "ours": avg_dtw},
            {"metric": "boundary_jaccard_tol2", "paper": "0.784", "ours": boundary_jaccard},
            {"metric": "pointwise_label_accuracy", "paper": "", "ours": pointwise_acc},
            {"metric": "per_segment_majority_accuracy", "paper": "", "ours": segment_acc},
        ]
    )
    pd.DataFrame(multi_rows).to_csv(out / "fig9_multi_pattern_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "pattern": r["pattern"],
                "series_length": r["series_length"],
                "n_segments_real": r["n_segments_real"],
                "n_segments_sisc": r["n_segments_sisc"],
                "per_unit_dtw": r["dtw_error"],
                "boundary_jaccard_tol2": r["boundary_jaccard"],
                "paper_per_unit_dtw": 0.009,
                "paper_boundary_jaccard": 0.938,
            }
            for r in one_results
        ]
    ).to_csv(out / "fig8_one_pattern_metrics.csv", index=False)
    pd.DataFrame(dist_matrix).to_csv(out / "fig9_centroid_distance_matrix.csv", index=False)
    pd.DataFrame(cm).to_csv(out / "fig9_confusion_matrix.csv", index=False)
    (out / "classification_report.txt").write_text(cls_report, encoding="utf-8")

    summary = {
        "output": str(out),
        "seed": args.seed,
        "max_iters": args.max_iters,
        "k_true": k_true,
        "series_length": int(len(ts)),
        "n_ground_truth_segments": int(len(real_labels)),
        "multi_pattern": {
            "paper_avg_per_unit_dtw": 0.01,
            "ours_avg_per_unit_dtw": avg_dtw,
            "paper_jaccard": 0.784,
            "ours_boundary_jaccard_tol2": boundary_jaccard,
            "ours_pointwise_label_accuracy": pointwise_acc,
            "ours_per_segment_majority_accuracy": segment_acc,
            "alignment_gt_to_sisc": {str(k): int(v) for k, v in alignment.items()},
        },
        "one_pattern": {
            "paper_per_unit_dtw": 0.009,
            "ours_avg_per_unit_dtw": float(np.mean([r["dtw_error"] for r in one_results])),
            "paper_boundary_jaccard": 0.938,
            "ours_avg_boundary_jaccard_tol2": float(np.mean([r["boundary_jaccard"] for r in one_results])),
        },
        "files": [
            "fig8_one_pattern_metrics.csv",
            "fig8_one_pattern_replication.pdf",
            "fig8_one_pattern_replication.png",
            "fig9_centroid_distance_matrix.csv",
            "fig9_confusion_matrix.csv",
            "fig9_confusion_matrix.pdf",
            "fig9_confusion_matrix.png",
            "fig9_multi_pattern_metrics.csv",
            "fig9_multi_pattern_replication.pdf",
            "fig9_multi_pattern_replication.png",
            "classification_report.txt",
            "manifest.json",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
