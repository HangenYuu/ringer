"""Count conformers in pickle files and save results to CSV."""

import csv
import os
import pickle
import sys


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "sample/reconstructed_mols"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "conformer_counts.csv"

    results = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pickle"):
            fpath = os.path.join(data_dir, fname)
            fsize = os.path.getsize(fpath)
            with open(fpath, "rb") as f:
                mol = pickle.load(f)
            n_conformers = mol.GetNumConformers()
            results.append((fname, fsize, n_conformers))

    results.sort(key=lambda x: (x[1], x[2]), reverse=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "file_size_bytes", "num_conformers"])
        writer.writerows(results)

    print(f"Wrote {len(results)} entries to {output_csv}")


if __name__ == "__main__":
    main()
