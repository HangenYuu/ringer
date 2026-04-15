#!/usr/bin/env python

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import typer
from joblib import Parallel, delayed
from rdkit import Chem

from ringer.utils import chem, evaluation, internal_coords


def load_mol(path: Union[str, Path]) -> Chem.Mol:
    with open(path, "rb") as f:
        mol = pickle.load(f)
    if isinstance(mol, dict):
        mol = mol["rd_mol"]
    return mol


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def compute_diag_rmsd_from_map(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    atom_map: List[Sequence[Tuple[int, int]]],
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    """Diagonal version of evaluation.compute_rmsd_matrix_from_map: only pair i of
    probe with pair i of ref."""
    if conf_ids_ref is None:
        conf_ids_ref = [conf.GetId() for conf in ref_mol.GetConformers()]
    if conf_ids_probe is None:
        conf_ids_probe = [conf.GetId() for conf in probe_mol.GetConformers()]

    num_pairs = min(len(conf_ids_ref), len(conf_ids_probe))

    args = [(i, i, conf_ids_probe[i], conf_ids_ref[i]) for i in range(num_pairs)]
    rmsds_with_idxs = Parallel(n_jobs=ncpu)(
        delayed(evaluation._get_best_rms)(a, probe_mol=probe_mol, ref_mol=ref_mol, atom_map=atom_map)
        for a in args
    )

    diag = np.empty(num_pairs)
    for i, _, rmsd in rmsds_with_idxs:
        diag[i] = rmsd
    return diag


def compute_diag_rmsd(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    """Mirror of evaluation.compute_rmsd_matrix but only for corresponding
    conformer pairs (diagonal)."""
    probe_mol = Chem.RemoveHs(probe_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    atom_map = evaluation.get_atom_map(probe_mol, ref_mol)
    atom_map = [list(map_.items()) for map_ in atom_map]

    return compute_diag_rmsd_from_map(
        probe_mol,
        ref_mol,
        atom_map,
        conf_ids_probe=conf_ids_probe,
        conf_ids_ref=conf_ids_ref,
        ncpu=ncpu,
    )


def compute_diag_ring_rmsd(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    """Mirror of evaluation.compute_ring_rmsd_matrix but only for corresponding
    conformer pairs (diagonal)."""
    probe_mol = Chem.RemoveHs(probe_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    atom_map = evaluation.get_atom_map(probe_mol, ref_mol)

    probe_macrocycle_idxs = chem.get_macrocycle_idxs(probe_mol, n_to_c=False)
    if probe_macrocycle_idxs is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(probe_mol)}'"
        )
    atom_map = list(set(tuple((k, map_[k]) for k in probe_macrocycle_idxs) for map_ in atom_map))

    ref_macrocycle_idxs = chem.get_macrocycle_idxs(ref_mol, n_to_c=False)
    if ref_macrocycle_idxs is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(ref_mol)}'"
        )
    ref_macrocycle_idxs = set(ref_macrocycle_idxs)
    for map_ in atom_map:
        if set(ref_idx for _, ref_idx in map_) != ref_macrocycle_idxs:
            raise ValueError("Inconsistent macrocycle indices")

    return compute_diag_rmsd_from_map(
        probe_mol,
        ref_mol,
        atom_map,
        conf_ids_probe=conf_ids_probe,
        conf_ids_ref=conf_ids_ref,
        ncpu=ncpu,
    )


def compute_diag_ring_tfd(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    """Mirror of evaluation.compute_ring_tfd_matrix but only for corresponding
    conformer pairs (diagonal). Uses the same wrap/normalize logic."""
    probe_mol = Chem.RemoveHs(probe_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    atom_map = evaluation.get_atom_map(probe_mol, ref_mol)

    probe_macrocycle_idxs = chem.get_macrocycle_idxs(probe_mol, n_to_c=False)
    if probe_macrocycle_idxs is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(probe_mol)}'"
        )
    atom_map = list(set(tuple((k, map_[k]) for k in probe_macrocycle_idxs) for map_ in atom_map))

    num_torsions = len(probe_macrocycle_idxs)

    all_probe_conf_ids = [conf.GetId() for conf in probe_mol.GetConformers()]
    all_ref_conf_ids = [conf.GetId() for conf in ref_mol.GetConformers()]
    if conf_ids_probe is None:
        conf_ids_probe = all_probe_conf_ids
    if conf_ids_ref is None:
        conf_ids_ref = all_ref_conf_ids

    num_pairs = min(len(conf_ids_probe), len(conf_ids_ref))
    probe_idxs = np.array(
        [all_probe_conf_ids.index(cid) for cid in conf_ids_probe[:num_pairs]]
    )
    ref_idxs = np.array(
        [all_ref_conf_ids.index(cid) for cid in conf_ids_ref[:num_pairs]]
    )

    probe_torsions = internal_coords.get_macrocycle_dihedrals(
        probe_mol, probe_macrocycle_idxs
    ).to_numpy()[probe_idxs]

    ref_macrocycle_idxs_check = chem.get_macrocycle_idxs(ref_mol, n_to_c=False)
    if ref_macrocycle_idxs_check is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(ref_mol)}'"
        )
    ref_macrocycle_idxs_check = set(ref_macrocycle_idxs_check)

    def _tfd_for_map(map_: Sequence[Tuple[int, int]]) -> np.ndarray:
        ref_macrocycle_idxs = [ref_idx for _, ref_idx in map_]
        if set(ref_macrocycle_idxs) != ref_macrocycle_idxs_check:
            raise ValueError("Inconsistent macrocycle indices")
        ref_torsions = internal_coords.get_macrocycle_dihedrals(
            ref_mol, ref_macrocycle_idxs
        ).to_numpy()[ref_idxs]

        torsion_deviation = probe_torsions - ref_torsions
        torsion_deviation = (torsion_deviation + np.pi) % (2 * np.pi) - np.pi
        return np.sum(np.abs(torsion_deviation) / np.pi, axis=-1) / num_torsions

    tfds_per_map = Parallel(n_jobs=ncpu)(delayed(_tfd_for_map)(m) for m in atom_map)

    return np.minimum.reduce(tfds_per_map)


def compute_metrics(
    mol_probe_path: str,
    mol_ref_path: str,
    out_dir: str = "metrics_corresponding",
    rrmsd_threshold: float = 0.10,
    rtfd_threshold: float = 0.05,
    rmsd_threshold: float = 0.75,
    ncpu: int = 1,
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_path = Path(mol_probe_path)
    ref_path = Path(mol_ref_path)

    probe_mol = load_mol(probe_path)
    ref_mol = load_mol(ref_path)

    probe_conf_ids = [conf.GetId() for conf in probe_mol.GetConformers()]
    ref_conf_ids = [conf.GetId() for conf in ref_mol.GetConformers()]
    num_pairs = min(len(probe_conf_ids), len(ref_conf_ids))

    if num_pairs == 0:
        raise ValueError("No conformers found to compare")

    if len(probe_conf_ids) != len(ref_conf_ids):
        logging.warning(
            "Conformer count mismatch: probe=%d, ref=%d. Using first %d corresponding pairs",
            len(probe_conf_ids),
            len(ref_conf_ids),
            num_pairs,
        )

    probe_conf_ids = probe_conf_ids[:num_pairs]
    ref_conf_ids = ref_conf_ids[:num_pairs]

    diag_ring_rmsd = compute_diag_ring_rmsd(
        probe_mol,
        ref_mol,
        conf_ids_probe=probe_conf_ids,
        conf_ids_ref=ref_conf_ids,
        ncpu=ncpu,
    )
    diag_ring_tfd = compute_diag_ring_tfd(
        probe_mol,
        ref_mol,
        conf_ids_probe=probe_conf_ids,
        conf_ids_ref=ref_conf_ids,
    )
    diag_rmsd = compute_diag_rmsd(
        probe_mol,
        ref_mol,
        conf_ids_probe=probe_conf_ids,
        conf_ids_ref=ref_conf_ids,
        ncpu=ncpu,
    )

    rrmsd_pass_count = int(np.count_nonzero(diag_ring_rmsd <= rrmsd_threshold))
    rtfd_pass_count = int(np.count_nonzero(diag_ring_tfd <= rtfd_threshold))
    rmsd_pass_count = int(np.count_nonzero(diag_rmsd <= rmsd_threshold))
    rrmsd_pass_fraction = rrmsd_pass_count / num_pairs
    rtfd_pass_fraction = rtfd_pass_count / num_pairs
    rmsd_pass_fraction = rmsd_pass_count / num_pairs

    results: Dict[str, Any] = {
        "probe_mol": str(probe_path),
        "ref_mol": str(ref_path),
        "num_pairs_compared": num_pairs,
        "probe_conf_ids": probe_conf_ids,
        "ref_conf_ids": ref_conf_ids,
        "diag_rmsd": diag_rmsd,
        "diag_ring_rmsd": diag_ring_rmsd,
        "diag_ring_tfd": diag_ring_tfd,
        "thresholds": {
            "rmsd": float(rmsd_threshold),
            "ring-rmsd": float(rrmsd_threshold),
            "ring-tfd": float(rtfd_threshold),
        },
        "summary": {
            "rmsd_mean": float(diag_rmsd.mean()),
            "rmsd_median": float(np.median(diag_rmsd)),
            "rmsd_below_threshold_count": rmsd_pass_count,
            "rmsd_below_threshold_fraction": rmsd_pass_fraction,
            "rmsd_below_threshold_percent": 100.0 * rmsd_pass_fraction,
            "ring_rmsd_mean": float(diag_ring_rmsd.mean()),
            "ring_rmsd_median": float(np.median(diag_ring_rmsd)),
            "ring_tfd_mean": float(diag_ring_tfd.mean()),
            "ring_tfd_median": float(np.median(diag_ring_tfd)),
            "ring_rmsd_below_threshold_count": rrmsd_pass_count,
            "ring_rmsd_below_threshold_fraction": rrmsd_pass_fraction,
            "ring_rmsd_below_threshold_percent": 100.0 * rrmsd_pass_fraction,
            "ring_tfd_below_threshold_count": rtfd_pass_count,
            "ring_tfd_below_threshold_fraction": rtfd_pass_fraction,
            "ring_tfd_below_threshold_percent": 100.0 * rtfd_pass_fraction,
        },
    }

    out_path = output_dir / f"{probe_path.stem}.pickle"
    save_pickle(out_path, results)

    logging.info("Compared %d corresponding conformer pairs", num_pairs)
    logging.info(
        "Ring RMSD <= %.3f: %d/%d (%.2f%%)",
        rrmsd_threshold,
        rrmsd_pass_count,
        num_pairs,
        100.0 * rrmsd_pass_fraction,
    )
    logging.info(
        "Ring TFD <= %.3f: %d/%d (%.2f%%)",
        rtfd_threshold,
        rtfd_pass_count,
        num_pairs,
        100.0 * rtfd_pass_fraction,
    )
    logging.info(
        "RMSD <= %.3f: %d/%d (%.2f%%)",
        rmsd_threshold,
        rmsd_pass_count,
        num_pairs,
        100.0 * rmsd_pass_fraction,
    )
    logging.info("Saved corresponding metrics to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(compute_metrics)


if __name__ == "__main__":
    main()
