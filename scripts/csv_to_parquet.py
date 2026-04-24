"""Convert ring_bond_length_check CSV to Parquet in chunks."""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DTYPES = {
    "molecule": "str",
    "conf_id": "uint32",
    "ring_bond_pos": "uint16",
    "atom_i": "uint32",
    "atom_j": "uint32",
    "first_atom_label": "category",
    "expected_length": "float32",
    "actual_length": "float32",
    "delta": "float32",
    "abs_delta": "float32",
    "rel_abs_delta_pct": "float32",
}

ARROW_SCHEMA = pa.schema(
    [
        ("molecule", pa.string()),
        ("conf_id", pa.uint32()),
        ("ring_bond_pos", pa.uint16()),
        ("atom_i", pa.uint32()),
        ("atom_j", pa.uint32()),
        ("first_atom_label", pa.dictionary(pa.int8(), pa.string())),
        ("expected_length", pa.float32()),
        ("actual_length", pa.float32()),
        ("delta", pa.float32()),
        ("abs_delta", pa.float32()),
        ("rel_abs_delta_pct", pa.float32()),
    ]
)

CHUNK_SIZE = 500_000


def convert(csv_path: Path, out_path: Path) -> None:
    reader = pd.read_csv(csv_path, dtype=DTYPES, chunksize=CHUNK_SIZE)
    writer = None
    rows = 0
    try:
        for chunk in reader:
            table = pa.Table.from_pandas(chunk, schema=ARROW_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, schema=ARROW_SCHEMA, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            print(f"\r{rows:,} rows written", end="", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"\nDone — {rows:,} rows → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or args.csv.with_suffix(".parquet")
    convert(args.csv, out)
