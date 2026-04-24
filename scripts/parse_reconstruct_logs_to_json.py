#!/usr/bin/env python3

"""Parse reconstruction logs into a JSON summary."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from joblib import Parallel, delayed


LOG_NAME_PATTERN = re.compile(r"reconstruct_idx(\d+)\.log$")
MOLECULE_PATTERN = re.compile(r"INFO:root:Reconstructing\s+(.+?\.pickle)")
SUMMARY_PATTERN = re.compile(
    r"Conformer reconstruction summary:\s*attempted=(\d+)\s+successful=(\d+)\s+failed=(\d+)"
)
FAILED_INDICES_PATTERN = re.compile(r"Failed conformer indices:\s*(.*)")
PROGRESS_TOTAL_PATTERN = re.compile(r"\b\d+/(?P<total>\d+)\b")


def extract_idx(log_path: Path) -> int:
    match = LOG_NAME_PATTERN.search(log_path.name)
    if match is None:
        raise ValueError(f"Unexpected log filename format: {log_path.name}")
    return int(match.group(1))


def to_display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def parse_log(log_path: Path, base_dir: Path) -> Tuple[int, Dict[str, Optional[Any]]]:
    text = log_path.read_text(errors="ignore")

    molecule_match = MOLECULE_PATTERN.search(text)
    summary_match = SUMMARY_PATTERN.search(text)

    n_conformers: Optional[int] = None
    n_conformers_failed: Optional[int] = None

    if summary_match is not None:
        n_conformers = int(summary_match.group(1))
        n_conformers_failed = int(summary_match.group(3))
    else:
        totals = [int(match.group("total")) for match in PROGRESS_TOTAL_PATTERN.finditer(text)]
        if totals:
            n_conformers = max(totals)

        failed_indices_match = FAILED_INDICES_PATTERN.search(text)
        if failed_indices_match is not None:
            indices_text = failed_indices_match.group(1).strip()
            n_conformers_failed = 0 if not indices_text else len(indices_text.split())

    record = {
        "molecule": molecule_match.group(1) if molecule_match is not None else None,
        "path": to_display_path(log_path, base_dir),
        "n_conformers": n_conformers,
        "n_conformers_failed": n_conformers_failed,
    }
    return extract_idx(log_path), record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse reconstruction logs into JSON records")
    parser.add_argument(
        "--logs-dir",
        default="sample/logs/reconstruct/retry-drop",
        help="Directory containing reconstruct_idx*.log files",
    )
    parser.add_argument(
        "--output",
        default="sample/logs/reconstruct/retry-drop.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="joblib parallel workers (-1 uses all CPU cores)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)
    base_dir = Path.cwd()

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory does not exist: {logs_dir}")

    log_paths = sorted(logs_dir.glob("reconstruct_idx*.log"), key=extract_idx)
    if not log_paths:
        raise FileNotFoundError(f"No reconstruct_idx*.log files found in: {logs_dir}")

    parsed = Parallel(n_jobs=args.n_jobs)(
        delayed(parse_log)(log_path, base_dir) for log_path in log_paths
    )
    parsed.sort(key=lambda item: item[0])
    records = [record for _, record in parsed]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        json.dump(records, target, indent=args.indent)
        target.write("\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
