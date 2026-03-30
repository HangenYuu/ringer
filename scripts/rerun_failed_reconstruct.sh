#!/usr/bin/env bash

set -uo pipefail

IDX_LIST=(
  4 24 27 42 50 63 65 77 78 84 92 96 101 107 117 118 120 142 146 147 156 166 172 174 175
  176 193 195 197 204 206 238 247 251 252 262 273 282 299 306 328 368 376 385 392 408 409
  420 428 437 438 444 455 468 483 486 488 494 506 526 533 551 568 573 579 588 592 596 599
  607 614 630 634 643 653 661 668 693 694 695 706 732 737 743 750 774 776 788 807 816 831
  835 844 848 854 856 868 883 890 897 900 904 907 929 932 938 955 963 973 974 988 994
)

MOL_DIR="data/cremp/test"
SAMPLES_PATH="sample/samples.pickle"
OUT_DIR="sample/reconstructed_mols/retry-noH"
MEAN_DIST_PATH="assets/models/conditional/training_mean_distances.json"
LOG_DIR="sample/logs/reconstruct/retry-noH"
NCPU="${NCPU:-32}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

failed_idxs=()

for idx in "${IDX_LIST[@]}"; do
  log_path="$LOG_DIR/reconstruct_idx${idx}.log"
  echo "[$(date '+%F %T')] Running idx=${idx}"

  if python scripts/reconstruct_single.py \
    "$idx" \
    "$MOL_DIR" \
    "$SAMPLES_PATH" \
    "$OUT_DIR" \
    "$MEAN_DIST_PATH" \
    --ncpu "$NCPU" \
    --no-add-hydrogens \
    > "$log_path" 2>&1; then
    echo "  success idx=${idx}"
  else
    echo "  failed idx=${idx} (see $log_path)"
    failed_idxs+=("$idx")
  fi
done

if ((${#failed_idxs[@]} > 0)); then
  echo
  echo "Completed with failures (${#failed_idxs[@]}): ${failed_idxs[*]}"
  exit 1
fi

echo
echo "Completed all ${#IDX_LIST[@]} indices successfully."
