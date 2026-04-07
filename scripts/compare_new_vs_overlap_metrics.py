#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import stats


TARGET_THRESHOLDS = {
    "rmsd": 0.75,
    "ring-rmsd": 0.10,
    "ring-tfd": 0.05,
}

VALUE_COLUMNS = ["cov_r", "cov_p", "cov_f1", "mat_r", "mat_p", "mat_f1_raw"]


def f1(a: float, b: float, eps: float = 1e-12) -> float:
    return float(2.0 * a * b / (a + b + eps))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return np.nan

    greater = 0
    less = 0
    for xi in x:
        greater += int(np.sum(xi > y))
        less += int(np.sum(xi < y))
    return (greater - less) / float(n_x * n_y)


def load_metrics_dir(metrics_dir: Path) -> Dict[str, dict]:
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    metrics = {}
    for path in sorted(metrics_dir.glob("*.pickle")):
        with open(path, "rb") as source:
            obj = pickle.load(source)
        if isinstance(obj, dict):
            metrics[path.name] = obj
    return metrics


def load_filename_list(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "filename" in df.columns:
        series = df["filename"]
    else:
        series = df.iloc[:, 0]

    names = set()
    for value in series.astype(str):
        name = value.strip()
        if not name:
            continue
        if not name.endswith(".pickle"):
            name = f"{name}.pickle"
        names.add(name)
    return names


def extract_threshold_metrics(mol_metrics: dict, mol_name: str) -> List[dict]:
    rows: List[dict] = []
    for metric_name, threshold in TARGET_THRESHOLDS.items():
        if metric_name not in mol_metrics:
            continue

        cov = mol_metrics[metric_name]["cov"]
        mat = mol_metrics[metric_name]["mat"]

        thresholds = np.asarray(cov["threshold"], dtype=float)
        cov_r_vals = np.asarray(cov["cov-r"], dtype=float)
        cov_p_vals = np.asarray(cov["cov-p"], dtype=float)

        idx = int(np.argmin(np.abs(thresholds - threshold)))
        threshold_used = float(thresholds[idx])
        cov_r = float(cov_r_vals[idx])
        cov_p = float(cov_p_vals[idx])

        mat_r = float(mat["mat-r"])
        mat_p = float(mat["mat-p"])

        rows.append(
            {
                "molecule": mol_name,
                "metric": metric_name,
                "threshold_requested": float(threshold),
                "threshold_used": threshold_used,
                "cov_r": cov_r,
                "cov_p": cov_p,
                "cov_f1": f1(cov_r, cov_p),
                "mat_r": mat_r,
                "mat_p": mat_p,
                "mat_f1_raw": f1(mat_r, mat_p),
            }
        )

    return rows


def build_threshold_df(metrics_by_mol: Dict[str, dict], group_label: str) -> pd.DataFrame:
    rows = []
    for mol_name, mol_metrics in metrics_by_mol.items():
        mol_rows = extract_threshold_metrics(mol_metrics, mol_name)
        for row in mol_rows:
            row["group"] = group_label
            rows.append(row)
    return pd.DataFrame(rows)


def compare_groups(overlap_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = sorted(set(overlap_df["metric"]).intersection(set(new_df["metric"])))

    for metric_name in metrics:
        overlap_metric = overlap_df[overlap_df["metric"] == metric_name]
        new_metric = new_df[new_df["metric"] == metric_name]

        for col in VALUE_COLUMNS:
            x = overlap_metric[col].to_numpy(dtype=float)
            y = new_metric[col].to_numpy(dtype=float)

            if len(x) == 0 or len(y) == 0:
                continue

            t_res = stats.ttest_ind(y, x, equal_var=False)
            mw_res = stats.mannwhitneyu(y, x, alternative="two-sided")
            delta = cliffs_delta(y, x)

            rows.append(
                {
                    "metric": metric_name,
                    "value": col,
                    "n_overlap": len(x),
                    "n_new": len(y),
                    "mean_overlap": float(np.mean(x)),
                    "mean_new": float(np.mean(y)),
                    "delta_new_minus_overlap": float(np.mean(y) - np.mean(x)),
                    "welch_t_stat": float(t_res.statistic),
                    "welch_t_pvalue": float(t_res.pvalue),
                    "mannwhitney_u": float(mw_res.statistic),
                    "mannwhitney_pvalue": float(mw_res.pvalue),
                    "cliffs_delta_new_vs_overlap": float(delta),
                }
            )

    return pd.DataFrame(rows)


def run_level_means(overlap_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = sorted(set(overlap_df["metric"]).intersection(set(new_df["metric"])))
    for metric_name in metrics:
        overlap_metric = overlap_df[overlap_df["metric"] == metric_name]
        new_metric = new_df[new_df["metric"] == metric_name]
        for col in VALUE_COLUMNS:
            mean_overlap = float(overlap_metric[col].mean())
            mean_new = float(new_metric[col].mean())
            n_overlap = int(len(overlap_metric))
            n_new = int(len(new_metric))
            mean_combined = float((mean_overlap * n_overlap + mean_new * n_new) / (n_overlap + n_new))
            rows.append(
                {
                    "metric": metric_name,
                    "value": col,
                    "overlap_mean": mean_overlap,
                    "new_mean": mean_new,
                    "new_minus_overlap": float(mean_new - mean_overlap),
                    "combined_mean_weighted": mean_combined,
                    "overlap_n": n_overlap,
                    "new_n": n_new,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare overlap molecules vs run2-only new molecules at fixed thresholds: "
            "rmsd=0.75, ring-rmsd=0.10, ring-tfd=0.05."
        )
    )
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--overlap-csv", required=True)
    parser.add_argument("--new-csv", required=True)
    parser.add_argument("--out-dir", default="sample/analysis/new_vs_overlap")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    overlap_csv = Path(args.overlap_csv)
    new_csv = Path(args.new_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_mol = load_metrics_dir(metrics_dir)
    overlap_list = load_filename_list(overlap_csv)
    new_list = load_filename_list(new_csv)

    overlap_mols = sorted(overlap_list.intersection(metrics_by_mol.keys()))
    new_mols = sorted(new_list.intersection(metrics_by_mol.keys()))
    overlap_missing = sorted(overlap_list - metrics_by_mol.keys())
    new_missing = sorted(new_list - metrics_by_mol.keys())

    print(f"Metrics files loaded: {len(metrics_by_mol)}")
    print(f"Overlap list size (CSV): {len(overlap_list)}")
    print(f"New list size (CSV): {len(new_list)}")
    print(f"Overlap molecules with metrics: {len(overlap_mols)}")
    print(f"New molecules with metrics: {len(new_mols)}")
    print(f"Missing overlap metrics: {len(overlap_missing)}")
    print(f"Missing new metrics: {len(new_missing)}")

    if len(new_mols) == 0:
        raise ValueError("No new molecules with metrics found from --new-csv")
    if len(overlap_mols) == 0:
        raise ValueError("No overlap molecules with metrics found from --overlap-csv")

    overlap_metrics = {mol: metrics_by_mol[mol] for mol in overlap_mols}
    new_metrics = {mol: metrics_by_mol[mol] for mol in new_mols}

    overlap_df = build_threshold_df(overlap_metrics, group_label="overlap")
    new_df = build_threshold_df(new_metrics, group_label="new")

    comparison_df = compare_groups(overlap_df, new_df)
    means_df = run_level_means(overlap_df, new_df)

    overlap_df.to_csv(out_dir / "overlap_molecules_threshold_metrics.csv", index=False)
    new_df.to_csv(out_dir / "new_molecules_threshold_metrics.csv", index=False)
    comparison_df.to_csv(out_dir / "new_vs_overlap_significance.csv", index=False)
    means_df.to_csv(out_dir / "run1_vs_run2_means.csv", index=False)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 100)

    print("\n=== Group mean summary (overlap vs new) ===")
    print(means_df.to_string(index=False))

    print("\n=== New vs overlap significance ===")
    print(comparison_df.to_string(index=False))

    print(f"\nWrote results to: {out_dir}")


if __name__ == "__main__":
    main()
