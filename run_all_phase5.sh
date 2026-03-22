#!/usr/bin/env bash
# Phase 5: Complete experiment pipeline — all models sequentially on single GPU
# Order: mv3_framediff_bilstm_attn (after current HPO) → framediff×5 → TSM×6 → Pose

set -e
source venv/Scripts/activate
mkdir -p logs

# ── 1. Wait for mv3_framediff_bilstm_attn HPO then run train+LOAO ──────────
echo "[$(date '+%H:%M:%S')] Waiting for mv3_framediff_bilstm_attn HPO..."
while ! [ -f "configs/mobilenet_v3_small_framediff_bilstm_attn_opt.yaml" ]; do
    sleep 30
done
echo "[$(date '+%H:%M:%S')] HPO done. Running train+LOAO..."
./run_full_experiment.sh mobilenet_v3_small_framediff_bilstm_attn

# ── 2. Remaining framediff + TSM models ────────────────────────────────────
for MODEL in \
    efficientnet_b0_framediff_bilstm \
    efficientnet_b0_framediff_tcn \
    mobilenet_v3_small_framediff_bilstm \
    mobilenet_v3_small_framediff_tcn \
    efficientnet_b0_tsm_bilstm \
    efficientnet_b0_tsm_tcn \
    mobilenet_v3_small_tsm_bilstm_attn \
    mobilenet_v3_small_tsm_bilstm \
    mobilenet_v3_small_tsm_tcn
do
    ./run_full_experiment.sh "$MODEL"
done

# ── 3. Pose: wait for feature extraction, then HPO + train + LOAO ──────────
echo "[$(date '+%H:%M:%S')] Waiting for pose feature extraction (200 train files)..."
while [ "$(ls data/pose_features/train/ 2>/dev/null | wc -l)" -lt 200 ]; do
    echo "  pose train=$(ls data/pose_features/train/ 2>/dev/null | wc -l)/200"
    sleep 120
done
echo "[$(date '+%H:%M:%S')] Pose features ready. Running pose experiment..."
./run_full_experiment.sh pose_bilstm_attn

echo "[$(date '+%H:%M:%S')] ========================================="
echo "All Phase 5 experiments complete!"
echo "========================================="
