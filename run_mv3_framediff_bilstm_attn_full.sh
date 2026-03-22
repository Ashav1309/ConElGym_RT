#!/usr/bin/env bash
# Wait for HPO to finish then run train seed=42 + LOAO for mv3_framediff_bilstm_attn
source venv/Scripts/activate

MODEL="mobilenet_v3_small_framediff_bilstm_attn"

# Train seed=42
echo "=== Train seed=42 ==="
python src/scripts/train.py \
    --config "configs/${MODEL}_opt.yaml" \
    --seed 42 \
    2>&1 | tee "logs/train_${MODEL}.log"

# LOAO
echo "=== LOAO ==="
python src/scripts/loao_cv.py \
    --config "configs/${MODEL}_opt.yaml" \
    --seed 42 \
    2>&1 | tee "logs/loao_${MODEL}.log"
