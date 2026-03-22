"""
Pose-level dataset для ConElGym_RT.

Загружает кэш MediaPipe Pose-признаков (pre-extracted) и строит
покадровые метки из аннотаций ConElGym_v2.

Формат кэша (.pt):
  {
    "features":     Tensor[F, 99],  # 33 ключевые точки × (x, y, vis)
    "fps":          float,
    "total_frames": int,
  }

Интерфейс идентичен FrameDataset: возвращает VideoSample с features=[F, 99].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.annotation_parser import load_annotations

POSE_DIM = 99  # 33 landmarks × 3 (x, y, visibility)


@dataclass
class VideoSample:
    """Одно видео: признаки + метки + метаданные."""
    video_name: str
    features: Tensor    # [F, 99]
    labels: Tensor      # [F]  float32, 0.0 или 1.0
    fps: float
    has_element: bool


@dataclass
class TBPTTChunk:
    """Один чанк для TBPTT."""
    features: Tensor    # [chunk_size, 99]
    labels: Tensor      # [chunk_size]
    is_first: bool
    valid_len: int


class PoseDataset(Dataset):
    """Dataset с pose-признаками. Интерфейс совместим с FrameDataset.

    Args:
        features_dir: путь к data/pose_features/<split>/
        ann_dir:      путь к data/<split>/annotations/
    """

    def __init__(self, features_dir: Path, ann_dir: Path) -> None:
        self.features_dir = Path(features_dir)
        annotations = load_annotations(Path(ann_dir))

        self.samples: list[VideoSample] = []
        for video_name, ann in sorted(annotations.items()):
            feat_path = self.features_dir / f"{Path(video_name).stem}.pt"
            if not feat_path.exists():
                continue
            data = torch.load(feat_path, weights_only=True)
            features: Tensor = data["features"].float()  # [F, 99]
            fps: float = float(data.get("fps", 25.0))
            total_frames: int = features.shape[0]

            labels = torch.tensor(
                [ann.frame_label(t) for t in range(total_frames)],
                dtype=torch.float32,
            )
            self.samples.append(VideoSample(
                video_name=video_name,
                features=features,
                labels=labels,
                fps=fps,
                has_element=ann.has_element,
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]

    def pos_weight(self) -> float:
        """Вычисляет pos_weight для BCEWithLogitsLoss."""
        total = sum(s.labels.numel() for s in self.samples)
        pos = sum(s.labels.sum().item() for s in self.samples)
        neg = total - pos
        if pos == 0:
            return 1.0
        return neg / pos

    def iter_tbptt_chunks(
        self, sample: VideoSample, chunk_size: int
    ) -> Iterator[TBPTTChunk]:
        """Разбивает одно видео на чанки для TBPTT."""
        F = sample.features.shape[0]
        n_chunks = max(1, math.ceil(F / chunk_size))

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, F)
            feat_chunk = sample.features[start:end]
            lab_chunk = sample.labels[start:end]

            pad = chunk_size - feat_chunk.shape[0]
            if pad > 0:
                feat_chunk = torch.cat([
                    feat_chunk,
                    torch.zeros(pad, feat_chunk.shape[1], device=feat_chunk.device),
                ], dim=0)
                lab_chunk = torch.cat([lab_chunk, torch.zeros(pad, device=lab_chunk.device)], dim=0)

            yield TBPTTChunk(
                features=feat_chunk,
                labels=lab_chunk,
                is_first=(i == 0),
                valid_len=(end - start),
            )
