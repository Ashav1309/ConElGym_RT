#!/usr/bin/env bash
# Full experiment pipeline: HPO → train seed=42 → LOAO
# Usage: ./run_full_experiment.sh <model_name>
# e.g.:  ./run_full_experiment.sh mobilenet_v3_small_framediff_bilstm_attn

set -e
source venv/Scripts/activate

MODEL="$1"
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "[$(date '+%H:%M:%S')] === $MODEL ==="

# Step 1: HPO (skip if _opt.yaml already exists)
OPT_CFG="configs/${MODEL}_opt.yaml"
if [ -f "$OPT_CFG" ]; then
    echo "[$(date '+%H:%M:%S')] HPO: skipped (opt config exists)"
else
    echo "[$(date '+%H:%M:%S')] HPO: starting 20 trials"
    python src/scripts/hpo.py --model "$MODEL" --n-trials 20 \
        2>&1 | tee "${LOG_DIR}/hpo_${MODEL}.log"
    echo "[$(date '+%H:%M:%S')] HPO: done — $(grep 'mAP' ${LOG_DIR}/hpo_${MODEL}.log | tail -1)"
fi

# Step 2: Train seed=42
CHECKPOINT="models/${MODEL}_opt_seed42_best.pt"
if [ -f "$CHECKPOINT" ]; then
    echo "[$(date '+%H:%M:%S')] Train: skipped (checkpoint exists)"
else
    echo "[$(date '+%H:%M:%S')] Train seed=42: starting"
    python src/scripts/train.py --config "$OPT_CFG" --seed 42 \
        2>&1 | tee "${LOG_DIR}/train_${MODEL}.log"
    echo "[$(date '+%H:%M:%S')] Train: done"
fi

# Step 3: LOAO
echo "[$(date '+%H:%M:%S')] LOAO: starting"
python src/scripts/loao_cv.py --config "$OPT_CFG" --seed 42 \
    2>&1 | tee "${LOG_DIR}/loao_${MODEL}.log"
echo "[$(date '+%H:%M:%S')] LOAO: done"

# Summary
echo "[$(date '+%H:%M:%S')] === COMPLETE: $MODEL ==="
grep -E "mAP@0.5|Mean:|Criterion" "${LOG_DIR}/loao_${MODEL}.log" | tail -10
