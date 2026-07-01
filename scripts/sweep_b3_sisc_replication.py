from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from run_b3_sisc_simulated_replication import (
    compute_ground_truth_centroids,
    jaccard_segmentation,
    load_toy_data,
    patch_reference_sisc,
    per_unit_dtw_error,
    per_segment_majority_accuracy,
    pointwise_label_accuracy,
)


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "fts-diffusion-ref"


def author_interval_iou(real_boundaries: np.ndarray, pred_boundaries: np.ndarray, direction: str) -> float:
    """Mirror the reference metrics_clustering.compute_iou behavior for interval segmentations."""
    real = np.asarray(real_boundaries, dtype=int)
    pred = np.asarray(pred_boundaries, dtype=int)
    real_intervals = np.array([(real[i], real[i + 1]) for i in range(len(real) - 1)])
    pred_intervals = np.array([(pred[i], pred[i + 1]) for i in range(len(pred) - 1)])
    source, target = (real_intervals, pred_intervals) if direction == "pred" else (pred_intervals, real_intervals)
    values = []
    for interval in target:
        max_iou = 0.0
        for candidate in source:
            intersection = max(0, min(candidate[1], interval[1]) - max(candidate[0], interval[0]))
            union = max(candidate[1], interval[1]) - min(candidate[0], interval[0])
            max_iou = max(max_iou, intersection / union if union else 0.0)
        values.append(max_iou)
    return float(np.mean(values)) if values else 0.0


def align_centroids(gt_centroids: list[np.ndarray], sisc_centroids: list[np.ndarray]) -> tuple[dict[int, int], np.ndarray]:
    k = len(gt_centroids)
    distances = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            distances[i, j] = per_unit_dtw_error(gt_centroids[i], sisc_centroids[j])
    rows, cols = linear_sum_assignment(distances)
    return dict(zip(rows, cols)), distances


def run_sisc(series: np.ndarray, k: int, l_min: int, l_max: int, max_iters: int, init: str, barycenter: str):
    from models.pattern_recognition_module import SISC

    sisc = SISC(n_clusters=k, l_min=l_min, l_max=l_max)
    sisc.fit(
        series,
        max_iters=max_iters,
        init_strategy=init,
        barycenter=barycenter,
        plot_progress=False,
        store_res=False,
    )
    return sisc


def evaluate_run(
    ts: np.ndarray,
    real_labels: np.ndarray,
    real_boundaries: np.ndarray,
    seed: int,
    k: int,
    l_min: int,
    l_max: int,
    max_iters: int,
    init: str,
    barycenter: str,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    started = time.time()
    gt_centroids = compute_ground_truth_centroids(ts, real_labels, real_boundaries, k, l_max)
    sisc = run_sisc(ts, k, l_min, l_max, max_iters, init, barycenter)

    sisc_centroids = [np.asarray(c) for c in sisc.centroids]
    sisc_boundaries = np.asarray(sisc.segmentation)
    sisc_labels = np.asarray(sisc.labels)
    alignment, distance_matrix = align_centroids(gt_centroids, sisc_centroids)
    errors = [per_unit_dtw_error(gt_centroids[i], sisc_centroids[alignment[i]]) for i in range(k)]
    segment_acc, _, _ = per_segment_majority_accuracy(
        real_labels,
        real_boundaries,
        sisc_labels,
        sisc_boundaries,
        len(ts),
        alignment,
    )

    return {
        "seed": seed,
        "k": k,
        "l_min": l_min,
        "l_max": l_max,
        "max_iters": max_iters,
        "init": init,
        "barycenter": barycenter,
        "runtime_sec": round(time.time() - started, 3),
        "n_segments_real": int(len(real_labels)),
        "n_segments_sisc": int(len(sisc_labels)),
        "final_loss": float(sisc.hist_loss[-1]) if sisc.hist_loss else None,
        "min_loss": float(np.min(sisc.hist_loss)) if sisc.hist_loss else None,
        "avg_per_unit_dtw": float(np.mean(errors)),
        "per_unit_dtw_p0": float(errors[0]),
        "per_unit_dtw_p1": float(errors[1]),
        "per_unit_dtw_p2": float(errors[2]),
        "per_unit_dtw_p3": float(errors[3]),
        "boundary_jaccard_tol0": jaccard_segmentation(real_boundaries, sisc_boundaries, tol=0),
        "boundary_jaccard_tol1": jaccard_segmentation(real_boundaries, sisc_boundaries, tol=1),
        "boundary_jaccard_tol2": jaccard_segmentation(real_boundaries, sisc_boundaries, tol=2),
        "boundary_jaccard_tol5": jaccard_segmentation(real_boundaries, sisc_boundaries, tol=5),
        "author_interval_iou_pred": author_interval_iou(real_boundaries, sisc_boundaries, direction="pred"),
        "author_interval_iou_real": author_interval_iou(real_boundaries, sisc_boundaries, direction="real"),
        "pointwise_label_accuracy": pointwise_label_accuracy(
            real_labels,
            real_boundaries,
            sisc_labels,
            sisc_boundaries,
            len(ts),
            alignment,
        ),
        "per_segment_majority_accuracy": segment_acc,
        "alignment_gt_to_sisc": json.dumps({str(key): int(value) for key, value in alignment.items()}),
        "distance_matrix": json.dumps(distance_matrix.tolist()),
    }


def build_experiments() -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []

    for seed in range(10):
        experiments.append({"seed": seed, "l_max": 20, "max_iters": 10, "init": "kmeans++", "barycenter": "dba"})

    for seed in range(5):
        experiments.append({"seed": seed, "l_max": 20, "max_iters": 20, "init": "kmeans++", "barycenter": "dba"})

    for seed in range(5):
        experiments.append({"seed": seed, "l_max": 21, "max_iters": 20, "init": "kmeans++", "barycenter": "dba"})

    for seed in range(3):
        experiments.append({"seed": seed, "l_max": 20, "max_iters": 20, "init": "random_sample", "barycenter": "dba"})

    for seed in range(3):
        experiments.append({"seed": seed, "l_max": 20, "max_iters": 20, "init": "random_noise", "barycenter": "dba"})

    experiments.extend(
        [
            {"seed": 42, "l_max": 20, "max_iters": 50, "init": "kmeans++", "barycenter": "dba"},
            {"seed": 42, "l_max": 21, "max_iters": 50, "init": "kmeans++", "barycenter": "dba"},
        ]
    )
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Appendix B.3 SISC replication variants.")
    parser.add_argument("--output", default="reports/generated_outputs/03_appendix_b3_simulated_sisc/16_b3_sisc_sweep")
    parser.add_argument("--max-runtime-min", type=float, default=55)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "sweep_results.csv"
    manifest_path = out / "manifest.json"

    os.chdir(REF)
    sys.path.insert(0, str(REF))
    patch_reference_sisc()

    ts, real_labels, real_boundaries = load_toy_data()
    k = int(len(np.unique(real_labels)))
    experiments = build_experiments()
    if args.limit is not None:
        experiments = experiments[: args.limit]

    completed_keys = set()
    rows: list[dict[str, object]] = []
    if results_path.exists():
        existing = pd.read_csv(results_path)
        rows = existing.to_dict("records")
        for row in rows:
            completed_keys.add((row["seed"], row["l_max"], row["max_iters"], row["init"], row["barycenter"]))

    started_all = time.time()
    for index, spec in enumerate(experiments, start=1):
        key = (spec["seed"], spec["l_max"], spec["max_iters"], spec["init"], spec["barycenter"])
        if key in completed_keys:
            continue
        elapsed_min = (time.time() - started_all) / 60
        if elapsed_min >= args.max_runtime_min:
            break

        print(f"\n=== Experiment {index}/{len(experiments)}: {spec} ===", flush=True)
        try:
            row = evaluate_run(
                ts=ts,
                real_labels=real_labels,
                real_boundaries=real_boundaries,
                seed=int(spec["seed"]),
                k=k,
                l_min=10,
                l_max=int(spec["l_max"]),
                max_iters=int(spec["max_iters"]),
                init=str(spec["init"]),
                barycenter=str(spec["barycenter"]),
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            best_dtw = min(r["avg_per_unit_dtw"] for r in rows)
            best_iou = max(r["author_interval_iou_pred"] for r in rows)
            print(
                "result: "
                f"dtw={row['avg_per_unit_dtw']:.4f}, "
                f"boundaryJ2={row['boundary_jaccard_tol2']:.4f}, "
                f"authorIoU={row['author_interval_iou_pred']:.4f}, "
                f"segAcc={row['per_segment_majority_accuracy']:.4f}; "
                f"best_dtw={best_dtw:.4f}, best_authorIoU={best_iou:.4f}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "seed": spec["seed"],
                "k": k,
                "l_min": 10,
                "l_max": spec["l_max"],
                "max_iters": spec["max_iters"],
                "init": spec["init"],
                "barycenter": spec["barycenter"],
                "error": repr(exc),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            print(f"error: {exc!r}", flush=True)

    df = pd.DataFrame(rows)
    completed = int(df["avg_per_unit_dtw"].notna().sum()) if "avg_per_unit_dtw" in df else 0
    summary = {
        "output": str(out.relative_to(ROOT)),
        "results": str(results_path.relative_to(ROOT)),
        "completed_successful_runs": completed,
        "attempted_rows": int(len(df)),
        "paper_multi_pattern": {"avg_per_unit_dtw": 0.01, "jaccard": 0.784},
    }
    if completed:
        successful = df.dropna(subset=["avg_per_unit_dtw"])
        summary["best_by_dtw"] = successful.sort_values("avg_per_unit_dtw").iloc[0].to_dict()
        summary["best_by_author_interval_iou"] = successful.sort_values("author_interval_iou_pred", ascending=False).iloc[0].to_dict()
        summary["best_by_boundary_jaccard_tol2"] = successful.sort_values("boundary_jaccard_tol2", ascending=False).iloc[0].to_dict()
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
