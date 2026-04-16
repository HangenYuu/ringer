#!/usr/bin/env python

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Set, Union

import typer
from rdkit import Chem


def load_mol(path: Union[str, Path]) -> Chem.Mol:
    with open(path, "rb") as f:
        mol = pickle.load(f)
    if isinstance(mol, dict):
        mol = mol["rd_mol"]
    return mol


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def parse_failed_indices(log_path: Path) -> Set[int]:
    """Extract failed conformer indices from a reconstruction log file."""
    pattern = re.compile(r"Failed conformer indices:\s+([\d\s]+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return set(int(x) for x in m.group(1).split())
    return set()


def filter_conformers(
    log_path: str,
    ref_mol_path: str,
    out_dir: str = "filtered_mols",
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(log_path)
    ref_path = Path(ref_mol_path)

    failed_indices = parse_failed_indices(log_file)
    if not failed_indices:
        logging.info("No failed conformers in %s; copying ref as-is", log_file)

    ref_mol = load_mol(ref_path)
    num_before = ref_mol.GetNumConformers()

    existing_ids = {conf.GetId() for conf in ref_mol.GetConformers()}
    missing = failed_indices - existing_ids
    if missing:
        logging.warning(
            "Conformer IDs %s not found in ref mol (has %d conformers); skipping them",
            sorted(missing),
            num_before,
        )

    for idx in sorted(failed_indices & existing_ids, reverse=True):
        ref_mol.RemoveConformer(idx)

    num_after = ref_mol.GetNumConformers()
    logging.info(
        "Filtered %s: %d -> %d conformers (removed %d)",
        ref_path.name,
        num_before,
        num_after,
        num_before - num_after,
    )

    out_path = output_dir / ref_path.name
    save_pickle(out_path, ref_mol)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(filter_conformers)


if __name__ == "__main__":
    main()
