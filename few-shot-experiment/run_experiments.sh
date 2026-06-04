#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run few-shot genre classification experiments.
#
# Trains each model on every data-size split found in DATA_DIR.
# Results are written to output/<train_file>__<checkpoint_name>/
#
# Usage:
#   bash run_experiments.sh                       # all models, all sizes
#   bash run_experiments.sh configs/train_hmbert.yaml   # one model, all sizes
#
# GPU selection:
#   CUDA_VISIBLE_DEVICES=1 bash run_experiments.sh
# ---------------------------------------------------------------------------

set -euo pipefail

DATA_DIR="data_few-shot-data"

# If a config is passed as argument, use only that; otherwise run all three.
if [ $# -ge 1 ]; then
    CONFIGS=("$1")
else
    CONFIGS=(
        "configs/train_xlmroberta.yaml"
        "configs/train_hmbert.yaml"
        "configs/train_mbert.yaml"
    )
fi

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Config: $config"
    echo "========================================"

    for train_file in "$DATA_DIR"/*.train.csv; do
        fname=$(basename "$train_file")
        echo "  --> Training on: $fname"
        python train.py "$config" --train_file "$fname"
        echo "  Done: $fname"
        echo
    done
done

echo "All experiments complete."
