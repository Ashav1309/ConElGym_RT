#!/usr/bin/env bash
# Reruns for framediff TCN experiments (config bug fixed)
set -e
cd "$(dirname "$0")"
source venv/Scripts/activate

run_experiment() {
    local MODEL=$1
    echo "=== HPO: $MODEL ==="
    python src/scripts/hpo.py --model "$MODEL" --n-trials 20
    echo "=== Train: $MODEL ==="
    python src/train.py --config "configs/${MODEL}_opt.yaml" --seed 42
    echo "=== LOAO: $MODEL ==="
    python src/scripts/loao_cv.py --config "configs/${MODEL}_opt.yaml" --seed 42
    echo "=== DONE: $MODEL ==="
}

run_experiment efficientnet_b0_framediff_tcn
run_experiment mobilenet_v3_small_framediff_tcn
