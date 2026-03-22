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
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

V2_ROOT = Path("C:/Users/ashav/Desktop/ConElGym_v2")
SPLITS = {
    "train": V2_ROOT / "data" / "train" / "videos",
    "valid": V2_ROOT / "data" / "valid" / "videos",
    "test":  V2_ROOT / "data" / "test"  / "videos",
}

MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"

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


def _build_detector() -> mp_vision.PoseLandmarker:
    """Создаёт PoseLandmarker из Tasks API (mediapipe 0.10+), режим VIDEO.

    VIDEO mode использует temporal tracking → быстрее IMAGE (~2× на длинных видео).
    Один детектор на весь сплит с глобальными монотонными timestamp.
    """
    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_pose_video(
    video_path: Path,
    detector: mp_vision.PoseLandmarker,
    start_ts_ms: int = 0,
) -> tuple[torch.Tensor, float, int, int]:
    """Извлекает pose-признаки из всех кадров видео.

    Args:
        start_ts_ms: начальный глобальный timestamp (мс) — монотонный счётчик
                     поверх всех видео в сплите.

    Returns:
        features: Tensor[F, 99]
        fps: float
        total_frames: int
        next_ts_ms: следующий глобальный timestamp для продолжения
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        all_features: list[np.ndarray] = []
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts_ms = start_ts_ms + int(frame_idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                raw = np.array(
                    [[lm.x, lm.y, lm.visibility] for lm in lms],
                    dtype=np.float32,
                )  # [33, 3]
                norm = normalize_landmarks(raw)
            else:
                norm = np.zeros((N_LANDMARKS, 3), dtype=np.float32)

            all_features.append(norm.flatten())  # [99]
            frame_idx += 1
    finally:
        cap.release()

    if not all_features:
        raise ValueError(f"No frames decoded from {video_path}")

    n_frames = len(all_features)
    # +1 s gap between videos to reset tracking context
    next_ts_ms = start_ts_ms + int(n_frames * 1000.0 / fps) + 1000
    features = torch.from_numpy(np.stack(all_features, axis=0))  # [F, 99]
    return features, fps, n_frames, next_ts_ms


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

    print(f"\n=== {split.upper()} — {len(video_files)} видео | MediaPipe Pose ===")

    skipped = 0
    errors = 0

    # Один детектор на весь сплит (VIDEO mode + глобальный timestamp)
    detector = _build_detector()
    global_ts_ms = 0

    try:
        for video_path in tqdm(video_files, desc=split, unit="video"):
            out_path = out_dir / f"{video_path.stem}.pt"
            if out_path.exists() and not force:
                # Читаем реальные total_frames/fps из кэша для точного advance timestamp
                cached = torch.load(out_path, map_location="cpu", weights_only=True)
                n = cached["total_frames"]
                fps_tmp = cached["fps"]
                global_ts_ms += int(n * 1000.0 / fps_tmp) + 1000
                skipped += 1
                continue
            try:
                features, fps, n_frames, global_ts_ms = extract_pose_video(
                    video_path, detector, global_ts_ms
                )
                torch.save({
                    "features":     features,
                    "fps":          fps,
                    "total_frames": n_frames,
                }, out_path)
            except Exception as exc:
                print(f"\n[ERROR] {video_path.name}: {exc}")
                errors += 1
                # Сбрасываем детектор — иначе следующее видео получит неверный timestamp
                try:
                    detector.close()
                except Exception:
                    pass
                detector = _build_detector()
                global_ts_ms = 0
    finally:
        detector.close()

    print(f"  Saved: {len(video_files) - skipped - errors} | "
          f"Skipped: {skipped} | Errors: {errors}")


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
