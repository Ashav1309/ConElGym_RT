#!/bin/bash
# Sequential HPO runner for all remaining models
# mobilenet_v3_small is already running (PID check skipped)

set -e
source venv/Scripts/activate

MODELS=(
    "mobilenet_v3_bilstm"
    "mobilenet_v3_bilstm_attn"
    "mobilenet_v3_tcn"
    "efficientnet_b0_bilstm_attn"
    "efficientnet_b0_bilstm"
    "efficientnet_b0_mlp"
)

for MODEL in "${MODELS[@]}"; do
    echo "======================================"
    echo "Starting HPO for: $MODEL"
    echo "======================================"
    python src/scripts/hpo.py --model "$MODEL" --n-trials 20 \
        > "logs/hpo_${MODEL}.log" 2>&1
    echo "Done: $MODEL"
done

echo "All HPO jobs complete!"
