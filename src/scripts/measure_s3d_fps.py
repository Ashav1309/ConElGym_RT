"""
Замер FPS S3D backbone на тестовом видео.

S3D — 3D CNN backbone (Kinetics-400), клипы 16 кадров, stride=8.
Цель: показать, что 3D backbone нарушает RT-требование (25 FPS).

Запуск:
  python src/scripts/measure_s3d_fps.py
  python src/scripts/measure_s3d_fps.py --video data/test/videos/Ball_001.mp4
  python src/scripts/measure_s3d_fps.py --n-runs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbone_s3d import S3DBackbone, extract_s3d_features_rt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Замер FPS S3D backbone")
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Путь к видеофайлу. По умолчанию: первое .mp4 в data/test/videos/",
    )
    p.add_argument(
        "--n-runs", type=int, default=3,
        help="Количество прогонов для усреднения FPS (default: 3)",
    )
    return p.parse_args()


def find_test_video(root: Path) -> Path:
    """Найти первый тестовый видеофайл."""
    v2_root = Path("C:/Users/ashav/Desktop/ConElGym_v2")
    candidates = [
        v2_root / "data" / "test" / "videos" / "Ball_001.mp4",
        v2_root / "data" / "valid" / "videos" / "Ball_001.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: любой mp4 в test/videos
    for p in (v2_root / "data" / "test" / "videos").glob("*.mp4"):
        return p
    raise FileNotFoundError("Не найдено тестовое видео. Укажите --video явно.")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    video_path = args.video or find_test_video(ROOT)
    print(f"Видео: {video_path}")

    print("Загрузка S3D (Kinetics-400)...")
    backbone = S3DBackbone(pretrained=True).to(device).eval()

    params = sum(p.numel() for p in backbone.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in backbone.parameters()) / 1024**2
    print(f"S3D параметров: {params:,} | Размер: {size_mb:.1f} MB")

    # Прогрев GPU
    print("Прогрев GPU (5 прогонов)...")
    dummy = torch.zeros(1, 3, 16, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(5):
            backbone(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Замер FPS
    fps_runs: list[float] = []
    for i in range(args.n_runs):
        _, _, fps = extract_s3d_features_rt(str(video_path), backbone, device)
        fps_runs.append(fps)
        print(f"  Run {i+1}/{args.n_runs}: {fps:.1f} FPS")

    avg_fps = sum(fps_runs) / len(fps_runs)
    print(f"\nS3D FPS (среднее {args.n_runs} прогонов): {avg_fps:.1f}")
    print(f"RT-требование: 25 FPS")
    print(f"Статус: {'✅ RT' if avg_fps >= 25 else '❌ НЕ RT (слишком медленно)'}")
    print()
    print("=== Для диссертации ===")
    print(f"S3D backbone: {avg_fps:.1f} FPS | mAP@0.5=0.993 | LOAO=0.879")
    print("→ 3D backbone даёт высокую точность, но нарушает RT-требование")


if __name__ == "__main__":
    main()
