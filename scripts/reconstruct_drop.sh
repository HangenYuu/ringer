#!/bin/bash
set -euo pipefail

MOL_DIR="data/cremp/test"
SAMPLES_PATH="sample/samples.pickle"
OUT_DIR="sample/reconstructed_mols/drop"
MEAN_DIST_PATH="assets/models/conditional/training_mean_distances.json"
LOG_DIR="sample/logs/reconstruct/drop"
NCPU=$(nproc)

mkdir -p "$OUT_DIR" "$LOG_DIR"

for IDX in $(seq 0 999); do
  echo "[$(date +%H:%M:%S)] Processing molecule $IDX/999..." | tee -a "$LOG_DIR/reconstruct_all.log"
  python scripts/reconstruct_single.py \
    "$IDX" \
    "$MOL_DIR" \
    "$SAMPLES_PATH" \
    "$OUT_DIR" \
    "$MEAN_DIST_PATH" \
    --ncpu "$NCPU" \
    > "$LOG_DIR/reconstruct_idx${IDX}.log" 2>&1
done

echo "Done. All 1000 molecules processed." | tee -a "$LOG_DIR/reconstruct_all.log"
