#!/usr/bin/env python
"""Check ring bond lengths against training means for all conformers in pickled molecules."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ringer.utils import chem, internal_coords


def _compare_ring_bonds_single_conf(conf_id, bond_idxs, positions, expected_per_pos, atom_label_per_pos):
    rows = []
    for ring_pos, (idx1, idx2) in enumerate(bond_idxs):
        expected = expected_per_pos[ring_pos]
        actual = float(np.linalg.norm(positions[idx1] - positions[idx2]))
        delta = actual - expected
        rows.append(
            {
                "conf_id": int(conf_id),
                "ring_bond_pos": ring_pos,
                "atom_i": int(idx1),
                "atom_j": int(idx2),
                "first_atom_label": atom_label_per_pos[ring_pos],
                "expected_length": expected,
                "actual_length": actual,
                "delta": delta,
            }
        )
    return rows


def compare_ring_bond_lengths_all_conformers(mol, mean_distances, n_jobs=-1):
    macrocycle_idxs = chem.get_macrocycle_idxs(mol, n_to_c=True)
    if macrocycle_idxs is None:
        return None, None

    bond_idxs = internal_coords.get_macrocycle_bond_idxs(macrocycle_idxs)
    if mol.GetNumConformers() == 0:
        return None, None

    atom_labels = ["N", "Calpha", "CO"]
    atom_label_per_pos = [atom_labels[i % 3] for i in range(len(bond_idxs))]
    expected_per_pos = [float(mean_distances[lbl]) for lbl in atom_label_per_pos]

    conf_positions = [(conf.GetId(), np.asarray(conf.GetPositions())) for conf in mol.GetConformers()]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compare_ring_bonds_single_conf)(
            conf_id, bond_idxs, positions, expected_per_pos, atom_label_per_pos
        )
        for conf_id, positions in conf_positions
    )

    df = pd.DataFrame([row for conf_rows in results for row in conf_rows])
    df["abs_delta"] = df["delta"].abs()
    df["rel_abs_delta_pct"] = 100.0 * df["abs_delta"] / df["expected_length"]

    worst_per_conf = (
        df.loc[df.groupby("conf_id")["abs_delta"].idxmax()]
        .sort_values("abs_delta", ascending=False)
        .reset_index(drop=True)
    )

    return df, worst_per_conf


def process_directory(input_dir, means_path, output_csv, n_jobs, threshold_pct):
    with open(means_path, "r") as source:
        mean_distances = json.load(source)

    pickle_paths = sorted(input_dir.glob("*.pickle"))
    if not pickle_paths:
        print(f"No pickle files found in {input_dir}")
        return

    print(f"Found {len(pickle_paths)} molecules in {input_dir}")

    all_worst_rows = []
    for i, pickle_path in enumerate(pickle_paths):
        mol_name = pickle_path.stem
        with open(pickle_path, "rb") as source:
            mol = pickle.load(source)

        n_conf = mol.GetNumConformers()
        df, worst_per_conf = compare_ring_bond_lengths_all_conformers(mol, mean_distances, n_jobs=n_jobs)

        if df is None:
            print(f"[{i + 1}/{len(pickle_paths)}] {mol_name}: no macrocycle found, skipping")
            continue

        n_flagged = int((worst_per_conf["rel_abs_delta_pct"] > threshold_pct).sum())
        worst_delta = worst_per_conf["abs_delta"].iloc[0]
        worst_pct = worst_per_conf["rel_abs_delta_pct"].iloc[0]

        print(
            f"[{i + 1}/{len(pickle_paths)}] {mol_name}: "
            f"{n_conf} conformers, {n_flagged} flagged (>{threshold_pct}%), "
            f"worst abs_delta={worst_delta:.4f} A ({worst_pct:.2f}%)"
        )

        worst_per_conf.insert(0, "molecule", mol_name)
        all_worst_rows.append(worst_per_conf)

    if all_worst_rows:
        combined = pd.concat(all_worst_rows, ignore_index=True)
        combined.to_csv(output_csv, index=False)
        print(f"\nWrote worst-per-conformer results to {output_csv} ({len(combined)} rows)")
    else:
        print("No molecules processed.")


def main():
    parser = argparse.ArgumentParser(description="Check ring bond lengths against training means.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("sample/reconstructed_mols"),
    )
    parser.add_argument(
        "--means-path",
        type=Path,
        default=Path("assets/models/conditional/training_mean_distances.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ring_bond_length_check.csv"),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=5.0,
        help="Relative absolute delta percentage threshold for flagging conformers.",
    )
    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        means_path=args.means_path,
        output_csv=args.output_csv,
        n_jobs=args.n_jobs,
        threshold_pct=args.threshold_pct,
    )


if __name__ == "__main__":
    main()
