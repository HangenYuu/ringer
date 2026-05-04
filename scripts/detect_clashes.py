#!/usr/bin/env python

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem


def detect_steric_clashes(
    mol,
    conf_id=0,
    scale=0.75,
    tol=0.0,
    heavy_only=True,
    min_topological_distance=3,
):
    """Detect steric clashes in one conformer using vdW overlap criterion.

    A pair (i,j) is flagged as a clash when:
        distance(i,j) < scale * (Rvdw_i + Rvdw_j) - tol

    Args:
        mol: RDKit molecule with conformers.
        conf_id: Conformer ID to analyze.
        scale: Overlap scale on vdW sum; stricter when larger.
        tol: Absolute tolerance in Angstrom subtracted from threshold.
        heavy_only: If True, ignore hydrogens.
        min_topological_distance: Only check atom pairs with shortest path >= this value
            (default=3 excludes 1-2 and 1-3 pairs).

    Returns:
        clashes_df: DataFrame of clash pairs sorted by overlap descending.
        summary: Dict with counts and settings.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers")
    if conf_id < 0 or conf_id >= mol.GetNumConformers():
        raise ValueError(
            f"conf_id={conf_id} out of range for {mol.GetNumConformers()} conformers"
        )

    conf = mol.GetConformer(int(conf_id))
    positions = np.asarray(conf.GetPositions())
    topological_dist = Chem.GetDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    atom_indices = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if heavy_only and atom.GetAtomicNum() == 1:
            continue
        atom_indices.append(idx)

    rows = []
    num_pairs_checked = 0
    for ii, i in enumerate(atom_indices):
        atom_i = mol.GetAtomWithIdx(int(i))
        zi = atom_i.GetAtomicNum()
        ri = float(pt.GetRvdw(int(zi)))
        type_i = f"{atom_i.GetSymbol()}(Z={zi},hyb={str(atom_i.GetHybridization())})"

        for j in atom_indices[ii + 1 :]:
            topod = float(topological_dist[i, j])
            if topod < min_topological_distance:
                continue

            atom_j = mol.GetAtomWithIdx(int(j))
            zj = atom_j.GetAtomicNum()
            rj = float(pt.GetRvdw(int(zj)))
            type_j = (
                f"{atom_j.GetSymbol()}(Z={zj},hyb={str(atom_j.GetHybridization())})"
            )

            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            bond_type = str(bond.GetBondType()) if bond is not None else "nonbonded"
            relation = f"1-{int(topod) + 1}"

            d = float(np.linalg.norm(positions[i] - positions[j]))
            threshold = float(scale * (ri + rj) - tol)
            num_pairs_checked += 1

            if d < threshold:
                rows.append(
                    {
                        "conf_id": int(conf_id),
                        "atom_i": int(i),
                        "atom_j": int(j),
                        "symbol_i": atom_i.GetSymbol(),
                        "symbol_j": atom_j.GetSymbol(),
                        "atom_type_i": type_i,
                        "atom_type_j": type_j,
                        "bond_type": bond_type,
                        "relation": relation,
                        "topological_distance": topod,
                        "distance": d,
                        "vdw_i": ri,
                        "vdw_j": rj,
                        "threshold": threshold,
                        "overlap": float(threshold - d),
                    }
                )

    columns = [
        "conf_id",
        "atom_i",
        "atom_j",
        "symbol_i",
        "symbol_j",
        "atom_type_i",
        "atom_type_j",
        "bond_type",
        "relation",
        "topological_distance",
        "distance",
        "vdw_i",
        "vdw_j",
        "threshold",
        "overlap",
    ]
    clashes_df = pd.DataFrame(rows, columns=columns)
    if len(clashes_df) > 0:
        clashes_df = clashes_df.sort_values("overlap", ascending=False).reset_index(
            drop=True
        )

    summary = {
        "conf_id": int(conf_id),
        "num_atoms_considered": int(len(atom_indices)),
        "num_pairs_checked": int(num_pairs_checked),
        "num_clashes": int(len(clashes_df)),
        "scale": float(scale),
        "tol": float(tol),
        "heavy_only": bool(heavy_only),
        "min_topological_distance": int(min_topological_distance),
    }
    return clashes_df, summary


def detect_clashes_all_conformers(
    mol,
    scale=0.75,
    tol=0.0,
    heavy_only=True,
    min_topological_distance=3,
    n_jobs=-1,
):
    """Run steric clash detection on all conformers of a molecule in parallel."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers")

    num_confs = mol.GetNumConformers()
    print(
        f"Detecting clashes for {num_confs} conformers using {n_jobs if n_jobs > 0 else 'all'} cores..."
    )

    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(detect_steric_clashes)(
            mol,
            conf_id=conf.GetId(),
            scale=scale,
            tol=tol,
            heavy_only=heavy_only,
            min_topological_distance=min_topological_distance,
        )
        for conf in mol.GetConformers()
    )

    clash_dfs = [res[0] for res in results if len(res[0]) > 0]
    summaries = [res[1] for res in results]

    if clash_dfs:
        all_clashes_df = pd.concat(clash_dfs, ignore_index=True)
    else:
        all_clashes_df = pd.DataFrame()

    summary_df = pd.DataFrame(summaries)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds.")
    print(f"Total clashes found across all conformers: {len(all_clashes_df)}")

    return all_clashes_df, summary_df


def process_directory(input_dir, output_csv, summary_csv, scale, tol, heavy_only, min_topo_dist, n_jobs):
    pickle_paths = sorted(input_dir.glob("*.pickle"))
    if not pickle_paths:
        print(f"No pickle files found in {input_dir}")
        return

    print(f"Found {len(pickle_paths)} molecules in {input_dir}")

    all_clash_dfs = []
    all_summary_dfs = []
    for i, pickle_path in enumerate(pickle_paths):
        mol_name = pickle_path.stem
        with open(pickle_path, "rb") as f:
            mol = pickle.load(f)

        clashes_df, summary_df = detect_clashes_all_conformers(
            mol,
            scale=scale,
            tol=tol,
            heavy_only=heavy_only,
            min_topological_distance=min_topo_dist,
            n_jobs=n_jobs,
        )

        n_conf = mol.GetNumConformers()
        n_clashing = int((summary_df["num_clashes"] > 0).sum())
        print(
            f"[{i + 1}/{len(pickle_paths)}] {mol_name}: "
            f"{n_conf} conformers, {n_clashing} with clashes, "
            f"{len(clashes_df)} total clashes"
        )

        if len(clashes_df) > 0:
            clashes_df.insert(0, "molecule", mol_name)
            all_clash_dfs.append(clashes_df)

        summary_df.insert(0, "molecule", mol_name)
        all_summary_dfs.append(summary_df)

    if all_clash_dfs:
        combined_clashes = pd.concat(all_clash_dfs, ignore_index=True)
        if output_csv:
            combined_clashes.to_csv(output_csv, index=False)
            print(f"\nWrote all clashes to {output_csv} ({len(combined_clashes)} rows)")
    else:
        print("\nNo clashes detected in any molecule.")

    if all_summary_dfs:
        combined_summary = pd.concat(all_summary_dfs, ignore_index=True)
        if summary_csv:
            combined_summary.to_csv(summary_csv, index=False)
            print(f"Wrote summary to {summary_csv} ({len(combined_summary)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Detect steric clashes across all conformers of RDKit molecules"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing .pickle files to process",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Single .pickle file to process",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to save all clashes as CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to save summary stats as CSV",
    )
    parser.add_argument(
        "--scale", type=float, default=0.75, help="vdW overlap scale (default: 0.75)"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="Absolute distance tolerance in A (default: 0.0)",
    )
    parser.add_argument(
        "--include-hydrogens",
        action="store_true",
        help="Include hydrogens in clash check (default: heavy atoms only)",
    )
    parser.add_argument(
        "--min-topo-dist",
        type=int,
        default=3,
        help="Min topological distance to check (default: 3)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 for all cores)",
    )

    args = parser.parse_args()
    heavy_only = not args.include_hydrogens

    if args.input_dir:
        process_directory(
            input_dir=args.input_dir,
            output_csv=args.output_csv,
            summary_csv=args.summary_csv,
            scale=args.scale,
            tol=args.tol,
            heavy_only=heavy_only,
            min_topo_dist=args.min_topo_dist,
            n_jobs=args.n_jobs,
        )
    else:
        with open(args.input_file, "rb") as f:
            mol = pickle.load(f)

        all_clashes_df, summary_df = detect_clashes_all_conformers(
            mol,
            scale=args.scale,
            tol=args.tol,
            heavy_only=heavy_only,
            min_topological_distance=args.min_topo_dist,
            n_jobs=args.n_jobs,
        )

        if args.output_csv and len(all_clashes_df) > 0:
            all_clashes_df.to_csv(args.output_csv, index=False)
            print(f"Saved detailed clashes to {args.output_csv}")

        if args.summary_csv:
            summary_df.to_csv(args.summary_csv, index=False)
            print(f"Saved clash summary per conformer to {args.summary_csv}")

        if len(all_clashes_df) > 0:
            worst_conf = summary_df.loc[summary_df["num_clashes"].idxmax()]
            print(
                f"\nConformer with most clashes: {worst_conf['conf_id']} "
                f"({worst_conf['num_clashes']} clashes)"
            )
            worst_clash = all_clashes_df.loc[all_clashes_df["overlap"].idxmax()]
            print(
                f"Worst overall overlap: {worst_clash['overlap']:.3f}A in conf "
                f"{worst_clash['conf_id']} (atoms {worst_clash['atom_i']}-{worst_clash['atom_j']})"
            )
        else:
            print("\nNo clashes detected in any conformer with current settings.")


if __name__ == "__main__":
    main()
