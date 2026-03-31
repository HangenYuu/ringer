#!/usr/bin/env python
"""Find conformers with collinear-neighbor geometry issues in pickled molecules."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

Issue = Tuple[int, int, int, float, str]


def find_collinear_atoms(
    mol,
    conf_id: int = -1,
    cross_threshold: float = 1e-6,
    overlap_threshold: float = 1e-10,
) -> List[Issue]:
    """Return atom triplets with near-collinear vectors at a center atom."""
    conf = mol.GetConformer(conf_id)
    problems: List[Issue] = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbor_idxs = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbor_idxs) < 2:
            continue

        pos = np.array(conf.GetAtomPosition(idx))
        for i in range(len(neighbor_idxs)):
            for j in range(i + 1, len(neighbor_idxs)):
                p1 = np.array(conf.GetAtomPosition(neighbor_idxs[i]))
                p2 = np.array(conf.GetAtomPosition(neighbor_idxs[j]))
                v1 = p1 - pos
                v2 = p2 - pos
                v1n = np.linalg.norm(v1)
                v2n = np.linalg.norm(v2)

                if v1n < overlap_threshold or v2n < overlap_threshold:
                    problems.append(
                        (
                            idx,
                            neighbor_idxs[i],
                            neighbor_idxs[j],
                            0.0,
                            "overlapping atoms",
                        )
                    )
                    continue

                unit_v1 = v1 / v1n
                unit_v2 = v2 / v2n
                cross = float(np.linalg.norm(np.cross(unit_v1, unit_v2)))
                angle_deg = float(
                    np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))
                )

                if cross < cross_threshold:
                    problems.append(
                        (
                            idx,
                            neighbor_idxs[i],
                            neighbor_idxs[j],
                            angle_deg,
                            f"cross product magnitude = {cross:.2e}",
                        )
                    )

    return problems


def scan_pickle_for_problematic_conformers(
    pickle_path: Path,
    include_issues: bool = False,
) -> Dict[str, Any]:
    """Scan all conformers in one pickle file and return issue summary."""
    with open(pickle_path, "rb") as source:
        mol = pickle.load(source)

    problematic_conf_ids: List[int] = []
    issue_counts_by_conf_id: Dict[int, int] = {}
    issues_by_conf_id: Dict[int, List[Issue]] = {}

    for conf_id in range(mol.GetNumConformers()):
        issues = find_collinear_atoms(mol, conf_id)
        if issues:
            problematic_conf_ids.append(conf_id)
            issue_counts_by_conf_id[conf_id] = len(issues)
            if include_issues:
                issues_by_conf_id[conf_id] = issues

    result: Dict[str, Any] = {
        "molecule": pickle_path.name,
        "path": str(pickle_path),
        "n_conformers": mol.GetNumConformers(),
        "n_problematic_conformers": len(problematic_conf_ids),
        "problematic_conf_ids": problematic_conf_ids,
        "issue_counts_by_conf_id": issue_counts_by_conf_id,
    }

    if include_issues:
        result["issues_by_conf_id"] = issues_by_conf_id

    return result


def scan_directory_parallel(
    input_dir: Path,
    pattern: str,
    n_jobs: int,
    backend: str,
    verbose: int,
    include_issues: bool,
) -> List[Dict[str, Any]]:
    """Scan all pickle files in a directory using joblib parallel workers."""
    pickle_paths = sorted(input_dir.glob(pattern))

    if not pickle_paths:
        return []

    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(scan_pickle_for_problematic_conformers)(
            path, include_issues=include_issues
        )
        for path in pickle_paths
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find problematic conformers in pickled molecules with joblib."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("sample/reconstructed_mols/retry-noH"),
        help="Directory containing pickled molecules.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pickle",
        help="Glob pattern for input pickle files.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (joblib).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="loky",
        choices=["loky", "threading", "multiprocessing"],
        help="Joblib backend.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=10,
        help="Joblib verbosity level.",
    )
    parser.add_argument(
        "--include-issues",
        action="store_true",
        help="Include full per-conformer issue tuples in output JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full results JSON.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    results = scan_directory_parallel(
        input_dir=args.input_dir,
        pattern=args.pattern,
        n_jobs=args.n_jobs,
        backend=args.backend,
        verbose=args.verbose,
        include_issues=args.include_issues,
    )

    if not results:
        print(f"No files found in {args.input_dir} matching pattern: {args.pattern}")
        return

    problematic_only = [r for r in results if r["n_problematic_conformers"] > 0]

    print(f"Scanned {len(results)} molecules")
    print(f"Molecules with problematic conformers: {len(problematic_only)}")

    for result in problematic_only:
        print(
            f"{result['molecule']}: {result['n_problematic_conformers']} problematic conformers "
            f"-> {result['problematic_conf_ids']}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as sink:
            json.dump(results, sink, indent=2)
        print(f"Wrote JSON results to: {args.output_json}")


if __name__ == "__main__":
    main()
