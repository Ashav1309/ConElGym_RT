"""
Извлечение frame-level CNN-признаков из видео и сохранение в кэш.

Кэш сохраняется в: data/frame_features/<split>/<stem>_<backbone>.pt
Формат:
  {
    "features":     Tensor[F, D],
    "fps":          float,
    "total_frames": int,
    "backbone":     str,
  }

Запуск:
  python src/scripts/extract_frame_features.py --split train --backbone mobilenet_v3_small
  python src/scripts/extract_frame_features.py --split all
  python src/scripts/extract_frame_features.py --split valid --force  # перезаписать
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbone import BACKBONE_CONFIGS, CNNBackbone

# Нормализация ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

V2_ROOT = Path("C:/Users/ashav/Desktop/ConElGym_v2")
SPLITS = {
    "train": V2_ROOT / "data" / "train" / "videos",
    "valid": V2_ROOT / "data" / "valid" / "videos",
    "test":  V2_ROOT / "data" / "test"  / "videos",
}


def preprocess_frame(frame_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """BGR кадр → нормализованный тензор [3, size, size]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


@torch.no_grad()
def extract_video(
    video_path: Path,
    backbone: nn.Module,
    device: torch.device,
    frame_size: int = 224,
    batch_size: int = 32,
    frame_diff: bool = False,
    tsm_mode: bool = False,
) -> tuple[torch.Tensor, float, int]:
    """Извлекает признаки из всех кадров видео.

    Args:
        frame_diff: если True — вход backbone = [I_t | I_t - I_{t-1}] (6 каналов).
                    Для первого кадра разность = 0.
        tsm_mode:   если True — всё видео подаётся как [T, C, H, W] одним проходом
                    (TSM требует полного контекста для корректного временного сдвига).

    Returns:
        features: Tensor[F, D]
        fps:      float
        total_frames: int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    all_features: list[torch.Tensor] = []

    if tsm_mode:
        # TSM: загружаем все кадры, подаём как [T, C, H, W] одним проходом
        all_frames: list[torch.Tensor] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(preprocess_frame(frame, frame_size))

        cap.release()
        if not all_frames:
            raise RuntimeError(f"Нет кадров в видео: {video_path}")

        x = torch.stack(all_frames).to(device)   # [T, 3, H, W]
        feats = backbone(x)                        # [T, D]
        return feats.cpu(), fps, feats.shape[0]

    elif frame_diff:
        # Последовательная обработка: нужен предыдущий кадр для разности
        prev: torch.Tensor | None = None
        batch_diff: list[torch.Tensor] = []

        def flush_diff() -> None:
            if not batch_diff:
                return
            x = torch.stack(batch_diff).to(device)  # [B, 6, H, W]
            feats = backbone(x)                       # [B, D]
            all_features.append(feats.cpu())
            batch_diff.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            curr = preprocess_frame(frame, frame_size)          # [3, H, W]
            diff = curr - prev if prev is not None else torch.zeros_like(curr)
            batch_diff.append(torch.cat([curr, diff], dim=0))  # [6, H, W]
            prev = curr
            if len(batch_diff) == batch_size:
                flush_diff()

        flush_diff()
    else:
        batch: list[torch.Tensor] = []

        def flush_batch() -> None:
            if not batch:
                return
            x = torch.stack(batch).to(device)   # [B, 3, H, W]
            feats = backbone(x)                  # [B, D]
            all_features.append(feats.cpu())
            batch.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(preprocess_frame(frame, frame_size))
            if len(batch) == batch_size:
                flush_batch()

        flush_batch()

    cap.release()

    if not all_features:
        raise RuntimeError(f"Нет кадров в видео: {video_path}")

    features = torch.cat(all_features, dim=0)  # [F, D]
    return features, fps, features.shape[0]


def process_split(
    split: str,
    backbone_name: str,
    out_root: Path,
    device: torch.device,
    force: bool,
    frame_size: int,
    batch_size: int,
) -> None:
    videos_dir = SPLITS[split]
    if not videos_dir.exists():
        print(f"[WARN] Папка не найдена: {videos_dir}")
        return

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"[WARN] Нет .mp4 в {videos_dir}")
        return

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = CNNBackbone(backbone_name, frozen=True).to(device).eval()
    frame_diff = backbone_name == "efficientnet_b0_framediff"
    tsm_mode = backbone_name == "efficientnet_b0_tsm"

    print(f"\n=== {split.upper()} — {len(video_files)} видео | {backbone_name} ===")

    skipped = 0
    errors = 0

    for video_path in tqdm(video_files, desc=split, unit="video"):
        out_path = out_dir / f"{video_path.stem}_{backbone_name}.pt"
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            features, fps, n_frames = extract_video(
                video_path, backbone, device, frame_size, batch_size,
                frame_diff=frame_diff,
                tsm_mode=tsm_mode,
            )
            torch.save({
                "features":     features,
                "fps":          fps,
                "total_frames": n_frames,
                "backbone":     backbone_name,
            }, out_path)
        except Exception as exc:
            print(f"\n[ERROR] {video_path.name}: {exc}")
            errors += 1

    print(f"  Сохранено: {len(video_files) - skipped - errors} | "
          f"Пропущено: {skipped} | Ошибок: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение frame-level признаков")
    parser.add_argument("--split", default="all",
                        choices=["train", "valid", "test", "all"],
                        help="Какой сплит обрабатывать")
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=list(BACKBONE_CONFIGS.keys()),
                        help="Имя backbone")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "data" / "frame_features",
                        help="Директория для кэша признаков")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписать существующие файлы")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Кадров за раз через backbone")
    parser.add_argument("--cpu", action="store_true", help="Принудительно CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(
            split=split,
            backbone_name=args.backbone,
            out_root=args.out_dir,
            device=device,
            force=args.force,
            frame_size=args.frame_size,
            batch_size=args.batch_size,
        )

    print("\nГотово.")


if __name__ == "__main__":
    main()
