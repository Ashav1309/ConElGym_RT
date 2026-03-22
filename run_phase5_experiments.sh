#!/usr/bin/env bash
# Phase 5: Run all remaining framediff/TSM experiments in sequence
# Run after mv3_framediff_bilstm_attn HPO completes
set -e

source venv/Scripts/activate

MODELS=(
    "efficientnet_b0_framediff_bilstm"
    "efficientnet_b0_framediff_tcn"
    "mobilenet_v3_small_framediff_bilstm"
    "mobilenet_v3_small_framediff_tcn"
    "efficientnet_b0_tsm_bilstm"
    "efficientnet_b0_tsm_tcn"
    "mobilenet_v3_small_tsm_bilstm_attn"
    "mobilenet_v3_small_tsm_bilstm"
    "mobilenet_v3_small_tsm_tcn"
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "HPO: $MODEL"
    echo "========================================"
    python src/scripts/hpo.py --model "$MODEL" --n-trials 20 \
        > "logs/hpo_${MODEL}.log" 2>&1

    echo "--- Train seed=42: $MODEL ---"
    python src/scripts/train.py \
        --config "configs/${MODEL}_opt.yaml" \
        --seed 42 \
        > "logs/train_${MODEL}.log" 2>&1

    echo "--- LOAO: $MODEL ---"
    python src/scripts/loao_cv.py \
        --config "configs/${MODEL}_opt.yaml" \
        --seed 42 \
        > "logs/loao_${MODEL}.log" 2>&1

    echo "DONE: $MODEL"
done

echo "========================================"
echo "All Phase 5 experiments complete!"
echo "========================================"
