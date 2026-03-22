#!/usr/bin/env bash
# Pose experiment: wait for feature extraction → HPO → train → LOAO

source venv/Scripts/activate

POSE_TRAIN_COUNT=$(ls data/pose_features/train/ 2>/dev/null | wc -l)
echo "Pose features: train=$POSE_TRAIN_COUNT (need 200)"

# Wait for all 200 train features (valid/test extracted in same run)
while [ "$(ls data/pose_features/train/ 2>/dev/null | wc -l)" -lt "200" ]; do
    echo "  Waiting... train=$(ls data/pose_features/train/ 2>/dev/null | wc -l)/200"
    sleep 120
done
echo "Pose features ready. Starting HPO..."

./run_full_experiment.sh pose_bilstm_attn
