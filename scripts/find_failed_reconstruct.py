#!/usr/bin/env python3

import argparse
import pickle
import re
from pathlib import Path
from typing import Iterable, List


def chunked(seq: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find failed reconstruction indices from logs and missing outputs."
    )
    parser.add_argument("--logs-dir", default="sample/logs/reconstruct")
    parser.add_argument("--samples", default="sample/samples.pickle")
    parser.add_argument("--reconstructed-dir", default="sample/reconstructed_mols")
    parser.add_argument("--jobscript", default="jobscripts/reconstruct-pretrain.sh")
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    samples_path = Path(args.samples)
    reconstructed_dir = Path(args.reconstructed_dir)

    with open(samples_path, "rb") as source:
        samples = pickle.load(source)

    keys = list(samples.keys())
    idx_to_name = {i: Path(k).name for i, k in enumerate(keys)}

    failed_runtime = set()
    failed_traceback = set()

    for log_path in sorted(logs_dir.glob("reconstruct_idx*.log")):
        match = re.search(r"reconstruct_idx(\d+)\.log$", log_path.name)
        if not match:
            continue

        idx = int(match.group(1))
        text = log_path.read_text(errors="ignore")
        if "RuntimeError" in text:
            failed_runtime.add(idx)
        if "Traceback (most recent call last):" in text:
            failed_traceback.add(idx)

    missing_outputs = set()
    for idx, mol_name in idx_to_name.items():
        out_path = reconstructed_dir / mol_name
        if not out_path.exists():
            missing_outputs.add(idx)

    failed = sorted(failed_runtime | failed_traceback | missing_outputs)

    print(f"Total molecules in samples: {len(keys)}")
    print(f"RuntimeError in logs: {len(failed_runtime)}")
    print(f"Traceback in logs: {len(failed_traceback)}")
    print(f"Missing reconstructed outputs: {len(missing_outputs)}")
    print(f"Unique failed indices: {len(failed)}")

    if not failed:
        print("\nNo failed indices detected.")
        return

    print("\nFailed indices:")
    print(" ".join(map(str, failed)))

    print("\nFailed index -> molecule:")
    for idx in failed:
        print(f"{idx}\t{idx_to_name.get(idx, '<out-of-range>')}")

    print("\nRerun commands (single-index submits):")
    for idx in failed:
        print(f"qsub -v IDX={idx} {args.jobscript}")

    print(f"\nRerun commands (chunked arrays, chunk_size={args.chunk_size}):")
    for group in chunked(failed, args.chunk_size):
        # Convenience output; if indices are sparse, edit ranges manually.
        print(f"qsub -J {group[0]}-{group[-1]} {args.jobscript}")


if __name__ == "__main__":
    main()
