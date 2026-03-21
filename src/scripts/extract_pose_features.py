"""
Извлечение pose-признаков из видео с помощью MediaPipe Pose.

Каждый кадр → 33 ключевые точки × 3 (x, y, visibility) = 99 чисел.
Координаты нормализованы: центр = midpoint бёдер (landmarks 23, 24);
масштаб = расстояние нос (0) — щиколотка (27).

Кэш сохраняется в: data/pose_features/<split>/<stem>.pt
Формат:
  {
    "features":     Tensor[F, 99],
    "fps":          float,
    "total_frames": int,
  }

Запуск:
  python src/scripts/extract_pose_features.py --split train
  python src/scripts/extract_pose_features.py --split all
  python src/scripts/extract_pose_features.py --split valid --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

V2_ROOT = Path("C:/Users/ashav/Desktop/ConElGym_v2")
SPLITS = {
    "train": V2_ROOT / "data" / "train" / "videos",
    "valid": V2_ROOT / "data" / "valid" / "videos",
    "test":  V2_ROOT / "data" / "test"  / "videos",
}

N_LANDMARKS = 33
POSE_DIM = N_LANDMARKS * 3   # x, y, visibility

# Индексы для нормализации (MediaPipe Pose)
IDX_NOSE       = 0
IDX_HIP_LEFT   = 23
IDX_HIP_RIGHT  = 24
IDX_ANKLE_LEFT = 27


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Нормализует позу относительно центра бёдер и роста.

    Args:
        raw: [33, 3] — (x, y, visibility) в нормированных координатах [0, 1]

    Returns:
        [33, 3] — нормализованные координаты
    """
    # Центр бёдер
    hip_center = (raw[IDX_HIP_LEFT, :2] + raw[IDX_HIP_RIGHT, :2]) / 2.0

    # Масштаб: расстояние нос — щиколотка (по оси y)
    scale = abs(raw[IDX_NOSE, 1] - raw[IDX_ANKLE_LEFT, 1])
    if scale < 1e-6:
        scale = 1.0

    out = raw.copy()
    out[:, 0] = (raw[:, 0] - hip_center[0]) / scale
    out[:, 1] = (raw[:, 1] - hip_center[1]) / scale
    # visibility без изменений
    return out


def extract_pose_video(
    video_path: Path,
    pose_detector: mp.solutions.pose.Pose,  # type: ignore[name-defined]
) -> tuple[torch.Tensor, float, int]:
    """Извлекает pose-признаки из всех кадров видео.

    Returns:
        features: Tensor[F, 99]
        fps: float
        total_frames: int
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    all_features: list[np.ndarray] = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(frame_rgb)

        if results.pose_landmarks:
            raw = np.array([
                [lm.x, lm.y, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ], dtype=np.float32)  # [33, 3]
            norm = normalize_landmarks(raw)
        else:
            # Нет человека на кадре → нули
            norm = np.zeros((N_LANDMARKS, 3), dtype=np.float32)

        all_features.append(norm.flatten())  # [99]

    cap.release()

    features = torch.from_numpy(np.stack(all_features, axis=0))  # [F, 99]
    return features, fps, features.shape[0]


def process_split(
    split: str,
    out_root: Path,
    force: bool,
) -> None:
    video_dir = SPLITS[split]
    video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"[WARN] Нет .mp4 файлов в {video_dir}")
        return

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print(f"\n=== {split.upper()} — {len(video_files)} видео | MediaPipe Pose ===")

    skipped = 0
    errors = 0

    for video_path in tqdm(video_files, desc=split, unit="video"):
        out_path = out_dir / f"{video_path.stem}.pt"
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            features, fps, n_frames = extract_pose_video(video_path, pose)
            torch.save({
                "features":     features,
                "fps":          fps,
                "total_frames": n_frames,
            }, out_path)
        except Exception as exc:
            print(f"\n[ERROR] {video_path.name}: {exc}")
            errors += 1

    pose.close()
    print(f"  Сохранено: {len(video_files) - skipped - errors} | "
          f"Пропущено: {skipped} | Ошибок: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение pose-признаков (MediaPipe)")
    parser.add_argument("--split", default="all",
                        choices=["train", "valid", "test", "all"])
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "data" / "pose_features",
                        help="Директория для кэша pose-признаков")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписать существующие файлы")
    args = parser.parse_args()

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, args.out_dir, args.force)


if __name__ == "__main__":
    main()
