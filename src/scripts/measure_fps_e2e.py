"""
End-to-end FPS benchmark — ConElGym_RT.

Одинаковые условия для всех моделей:
  - Видео: Ball_001.mp4 (~2.5 мин)
  - Декодирование: cv2.VideoCapture, кадр за кадром
  - CNN-модели: decode + preprocess + backbone (chunk=256) + temporal
  - Pose-модели: decode + MediaPipe + normalize + temporal
  - S3D: результат из measure_s3d_fps.py (261.9 FPS) — уже честный
  - Прогрев: 1 run (discarded), затем 3 measurement runs
  - Финальный результат: mean ± std FPS

Запуск:
    python src/scripts/measure_fps_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.backbone import CNNBackbone  # noqa: E402
from src.models.full_model import GymRT  # noqa: E402
from src.models.pose_model import PoseGymRT  # noqa: E402
from src.scripts.extract_frame_features import preprocess_frame  # noqa: E402
from src.scripts.extract_pose_features import (  # noqa: E402
    _build_detector,
    normalize_landmarks,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

VIDEO_PATH = Path("C:/Users/ashav/Desktop/ConElGym_v2/data/train/videos/Ball_001.mp4")
MODELS_DIR = ROOT / "models"

BACKBONE_CHUNK = 256   # кадров за один backbone-проход (как chunk_size при обучении)
N_RUNS        = 3      # measurement runs (первый — warm-up, результаты runs 2–4)
WARMUP_RUNS   = 1

# ---------------------------------------------------------------------------
# Модели для замера
# ---------------------------------------------------------------------------

# (checkpoint_filename, model_type, display_name)
MODELS = [
    # CNN — EfficientNet-B0
    ("efficientnet_b0_bilstm_attn_opt_seed42_best.pt",     "cnn",  "EffB0+BiLSTM+Att"),
    ("efficientnet_b0_bilstm_opt_seed42_best.pt",          "cnn",  "EffB0+BiLSTM"),
    ("efficientnet_b0_tcn_opt_seed42_best.pt",             "cnn",  "EffB0+TCN"),
    ("efficientnet_b0_tsm_bilstm_attn_opt_seed42_best.pt", "cnn",  "EffB0+TSM+BiLSTM+Att"),
    ("efficientnet_b0_tsm_bilstm_opt_seed42_best.pt",      "cnn",  "EffB0+TSM+BiLSTM"),
    # CNN — MobileNetV4-Conv-Small
    ("mobilenetv4_conv_small_bilstm_attn_opt_seed42_best.pt", "cnn", "MV4+BiLSTM+Att"),
    ("mobilenetv4_conv_small_bilstm_opt_seed42_best.pt",      "cnn", "MV4+BiLSTM"),
    ("mobilenetv4_conv_small_tcn_opt_seed42_best.pt",         "cnn", "MV4+TCN"),
    # Pose
    ("pose_bilstm_attn_opt_seed42_best.pt",                "pose", "Pose+BiLSTM+Att"),
    ("pose_bilstm_opt_seed42_best.pt",                     "pose", "Pose+BiLSTM"),
    ("pose_causal_tcn_opt_seed42_best.pt",                 "pose", "Pose+CausalTCN"),
]


# ---------------------------------------------------------------------------
# Загрузка моделей
# ---------------------------------------------------------------------------

def load_cnn_model(ckpt_path: Path, device: torch.device) -> GymRT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]["model"]
    model = GymRT(
        backbone_name=cfg["backbone"],
        temporal_name=cfg["temporal_head"],
        temporal_cfg={
            "hidden_dim": cfg.get("hidden_dim", 256),
            "n_layers":   cfg.get("n_layers", 2),
            "dropout":    cfg.get("dropout", 0.3),
        },
        frozen_backbone=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_pose_model(ckpt_path: Path, device: torch.device) -> PoseGymRT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {}).get("model", {})
    temporal_cfg = {
        "hidden_dim": cfg.get("hidden_dim", 256),
        "n_layers":   cfg.get("n_layers", 2),
        "dropout":    cfg.get("dropout", 0.3),
    }
    model = PoseGymRT(
        hidden_dim=cfg.get("hidden_dim", 256),
        temporal_name=cfg.get("temporal_head", "bilstm_attn"),
        temporal_cfg=temporal_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Benchmark: CNN (decode + preprocess + backbone + temporal)
# ---------------------------------------------------------------------------

def run_cnn_benchmark(model: GymRT, device: torch.device) -> float:
    """Один прогон. Возвращает FPS."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    all_feats: list[torch.Tensor] = []
    chunk_frames: list[torch.Tensor] = []
    n_frames = 0

    t0 = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        n_frames += 1
        chunk_frames.append(preprocess_frame(frame_bgr))

        if len(chunk_frames) == BACKBONE_CHUNK:
            batch = torch.stack(chunk_frames).to(device)     # [T, C, H, W]
            with torch.no_grad():
                feats = model.backbone(batch)                 # [T, D]
            all_feats.append(feats.cpu())
            chunk_frames = []

    # Remaining frames
    if chunk_frames:
        batch = torch.stack(chunk_frames).to(device)
        with torch.no_grad():
            feats = model.backbone(batch)
        all_feats.append(feats.cpu())

    cap.release()

    # Temporal head — весь ролик одним проходом [1, N, D]
    all_feats_t = torch.cat(all_feats, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model.temporal(all_feats_t)

    elapsed = time.perf_counter() - t0
    return n_frames / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Benchmark: Pose (decode + MediaPipe + normalize + temporal)
# ---------------------------------------------------------------------------

def run_pose_benchmark(model: PoseGymRT, device: torch.device) -> float:
    """Один прогон. Возвращает FPS."""
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ms_per_frame = int(1000.0 / video_fps)

    detector = _build_detector()

    all_pose: list[np.ndarray] = []
    frame_idx = 0
    n_frames = 0

    t0 = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        n_frames += 1
        ts_ms = frame_idx * ms_per_frame
        frame_idx += 1

        # MediaPipe ожидает RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        import mediapipe as mp_lib
        mp_image = mp_lib.Image(
            image_format=mp_lib.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = detector.detect_for_video(mp_image, ts_ms)

        if result.pose_landmarks:
            raw = np.array(
                [[lm.x, lm.y, lm.visibility] for lm in result.pose_landmarks[0]],
                dtype=np.float32,
            )
            feat = normalize_landmarks(raw).flatten()
        else:
            feat = np.zeros(99, dtype=np.float32)

        all_pose.append(feat)

    cap.release()
    detector.close()

    # Temporal head [1, N, 99]
    pose_t = torch.tensor(np.stack(all_pose)).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model.temporal(pose_t)

    elapsed = time.perf_counter() - t0
    return n_frames / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Video:  {VIDEO_PATH.name}")
    print(f"Protocol: {WARMUP_RUNS} warm-up + {N_RUNS} measurement runs\n")

    results: list[tuple[str, float, float]] = []  # (name, mean_fps, std_fps)

    for ckpt_name, model_type, display in MODELS:
        ckpt_path = MODELS_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"  [SKIP] {display} — checkpoint not found: {ckpt_name}")
            continue

        print(f"Benchmarking: {display}")

        if model_type == "cnn":
            model = load_cnn_model(ckpt_path, device)
            bench_fn = lambda: run_cnn_benchmark(model, device)  # noqa: E731
        else:
            model = load_pose_model(ckpt_path, device)
            bench_fn = lambda: run_pose_benchmark(model, device)  # noqa: E731

        fps_runs: list[float] = []
        total_runs = WARMUP_RUNS + N_RUNS

        for run_idx in range(total_runs):
            fps = bench_fn()
            label = "warm-up" if run_idx < WARMUP_RUNS else f"run {run_idx - WARMUP_RUNS + 1}"
            print(f"  {label}: {fps:,.1f} FPS")
            if run_idx >= WARMUP_RUNS:
                fps_runs.append(fps)

        mean_fps = float(np.mean(fps_runs))
        std_fps  = float(np.std(fps_runs))
        print(f"  => MEAN: {mean_fps:,.1f} +/- {std_fps:,.1f} FPS\n")
        results.append((display, mean_fps, std_fps))

        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("END-TO-END FPS RESULTS (decode + backbone + temporal)")
    print("=" * 65)
    print(f"{'Model':<30} {'FPS mean':>10} {'FPS std':>10}  {'RT?':>5}")
    print("-" * 65)
    for name, mean, std in sorted(results, key=lambda x: -x[1]):
        rt_ok = "YES" if mean >= 25 else "NO"
        print(f"{name:<30} {mean:>10,.1f} {std:>10,.1f}  {rt_ok:>5}")
    print("-" * 65)
    print("S3D v2 (reference, from measure_s3d_fps.py):")
    print(f"{'S3D+BiLSTM+Att (offline)':<30} {'261.9':>10} {'22.4':>10}  {'YES':>5}")
    print("  Note: S3D includes CNN+decode; RT models: CNN/MediaPipe+temporal")
    print("=" * 65)


if __name__ == "__main__":
    main()
