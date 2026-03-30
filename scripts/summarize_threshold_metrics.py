#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


THRESHOLDS = {
    "rmsd": 0.75,
    "ring-rmsd": 0.10,
    "ring-tfd": 0.05,
}


def harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    return float(2.0 * a * b / (a + b + eps))


def choose_threshold(cov_df: pd.DataFrame, target: float) -> float:
    thresholds = cov_df.index.get_level_values("threshold").to_numpy(dtype=float)
    unique_thresholds = np.unique(thresholds)
    idx = int(np.argmin(np.abs(unique_thresholds - target)))
    return float(unique_thresholds[idx])


def get_all_molecules(samples_path: Optional[Path]) -> Optional[pd.Index]:
    if samples_path is None:
        return None
    if not samples_path.exists():
        raise FileNotFoundError(f"samples file not found: {samples_path}")

    with open(samples_path, "rb") as source:
        samples = pickle.load(source)
    # Match metrics index naming (basename only)
    return pd.Index([Path(k).name for k in samples.keys()], name="molecule")


def summarize_metric(
    metric_name: str,
    metric_data: Dict[str, pd.DataFrame],
    threshold: float,
    all_molecules: Optional[pd.Index],
    include_failures: bool,
) -> Dict[str, float]:
    cov_df = metric_data["cov"].copy()
    mat_df = metric_data["mat"].copy()

    chosen_threshold = choose_threshold(cov_df, threshold)
    cov_at_t = cov_df.xs(chosen_threshold, level="threshold").copy()
    cov_at_t.index = cov_at_t.index.rename("molecule")

    # Count successes before any reindexing/filling
    n_cov_success = int(cov_at_t.index.nunique())

    if all_molecules is not None:
        cov_at_t = cov_at_t.reindex(all_molecules)
        if include_failures:
            cov_at_t = cov_at_t.fillna(0.0)

    cov_r = float(cov_at_t["cov-r"].mean())
    cov_p = float(cov_at_t["cov-p"].mean())
    cov_f1 = harmonic_mean(cov_p, cov_r)

    # MAT is a distance metric (lower is better).
    # We report raw MAT means and raw harmonic mean on successful reconstructions only.
    mat_df = mat_df.copy()
    mat_df.index = mat_df.index.rename("molecule")
    mat_success = mat_df[["mat-r", "mat-p"]]
    n_mat_success = int(mat_success.index.nunique())

    mat_r_success = float(mat_success["mat-r"].mean())
    mat_p_success = float(mat_success["mat-p"].mean())
    mat_f1_raw_success = harmonic_mean(mat_p_success, mat_r_success)

    # Optional full-test-set adjustment for a bounded MAT-like score.
    # Convert distance to similarity in [0,1] with s = 1/(1+d), then compute F1.
    if all_molecules is not None:
        mat_full = mat_success.reindex(all_molecules)
        mat_sim = 1.0 / (1.0 + mat_full)
        if include_failures:
            mat_sim = mat_sim.fillna(0.0)
    else:
        mat_sim = 1.0 / (1.0 + mat_success)

    mat_sim_r = float(mat_sim["mat-r"].mean())
    mat_sim_p = float(mat_sim["mat-p"].mean())
    mat_sim_f1 = harmonic_mean(mat_sim_p, mat_sim_r)

    if all_molecules is not None:
        n_total = int(len(all_molecules))
    else:
        n_total = n_cov_success

    return {
        "metric": metric_name,
        "threshold_requested": float(threshold),
        "threshold_used": float(chosen_threshold),
        "n_total": n_total,
        "n_cov_success": n_cov_success,
        "n_mat_success": n_mat_success,
        "success_rate_cov": float(n_cov_success / n_total),
        "success_rate_mat": float(n_mat_success / n_total),
        "cov_recall": cov_r,
        "cov_precision": cov_p,
        "cov_f1": cov_f1,
        "mat_r_mean_success_only": mat_r_success,
        "mat_p_mean_success_only": mat_p_success,
        "mat_f1_raw_success_only": mat_f1_raw_success,
        "mat_sim_recall": mat_sim_r,
        "mat_sim_precision": mat_sim_p,
        "mat_sim_f1": mat_sim_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize COV and MAT metrics at fixed thresholds: "
            "0.75 (all-atom RMSD), 0.10 (ring RMSD), 0.05 (ring TFD)."
        )
    )
    parser.add_argument("--metrics-path", default="sample/metrics.pickle")
    parser.add_argument(
        "--samples-path",
        default="sample/samples.pickle",
        help=(
            "Used to count full test-set molecules and optionally penalize failed reconstructions. "
            "Set to empty string to disable."
        ),
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help=(
            "When samples-path is provided, include failed reconstructions as zeros for COV and "
            "for MAT similarity scores (mat_sim_*)."
        ),
    )
    parser.add_argument("--out-csv", default="sample/threshold_metrics_summary.csv")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    with open(metrics_path, "rb") as source:
        metrics = pickle.load(source)

    samples_path = None if args.samples_path == "" else Path(args.samples_path)
    all_molecules = get_all_molecules(samples_path)

    rows = []
    for metric_name, threshold in THRESHOLDS.items():
        if metric_name not in metrics:
            continue
        row = summarize_metric(
            metric_name=metric_name,
            metric_data=metrics[metric_name],
            threshold=threshold,
            all_molecules=all_molecules,
            include_failures=args.include_failures,
        )
        rows.append(row)

    if not rows:
        raise ValueError(
            "No requested metrics found in metrics file. Expected keys among: "
            f"{list(THRESHOLDS.keys())}"
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by="metric").reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 100)
    print("\nThreshold metric summary")
    print(summary.to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
