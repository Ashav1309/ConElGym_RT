"""
Frame-level dataset для ConElGym_RT.

Загружает кэш CNN-признаков (pre-extracted) и строит покадровые метки
из аннотаций ConElGym_v2.

Формат кэша (.pt):
  {
    "features":      Tensor[F, D],  # F кадров, D признаков
    "fps":           float,
    "total_frames":  int,
    "backbone":      str,
  }

Два режима:
  - sequence mode (по умолчанию): возвращает всё видео как один пример
    → используется в TBPTT training loop
  - flat mode: возвращает отдельные кадры (не учитывает порядок)
    → используется для статистики / диагностики
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


@dataclass
class VideoSample:
    """Одно видео: признаки + метки + метаданные."""
    video_name: str
    features: Tensor    # [F, D]
    labels: Tensor      # [F]  float32, 0.0 или 1.0
    fps: float
    has_element: bool


@dataclass
class TBPTTChunk:
    """Один чанк для TBPTT."""
    features: Tensor    # [chunk_size, D]
    labels: Tensor      # [chunk_size]  float32
    is_first: bool      # сбрасывать hidden state перед этим чанком


class FrameDataset(Dataset):
    """Dataset для sequence-mode обучения (одно видео = один пример).

    Args:
        features_dir: путь к data/frame_features/<split>/
        ann_dir:      путь к data/<split>/annotations/
        backbone:     имя backbone (суффикс в имени .pt файла)
    """

    def __init__(self, features_dir: Path, ann_dir: Path, backbone: str) -> None:
        self.features_dir = Path(features_dir)
        self.backbone = backbone
        annotations = load_annotations(Path(ann_dir))

        self.samples: list[VideoSample] = []
        for video_name, ann in sorted(annotations.items()):
            feat_path = self.features_dir / f"{Path(video_name).stem}_{backbone}.pt"
            if not feat_path.exists():
                continue
            data = torch.load(feat_path, weights_only=True)
            features: Tensor = data["features"].float()  # [F, D]
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
        """Разбивает одно видео на чанки для TBPTT.

        Yields TBPTTChunk; первый чанк имеет is_first=True.
        Последний чанк дополняется нулями если короче chunk_size.
        """
        F = sample.features.shape[0]
        n_chunks = max(1, math.ceil(F / chunk_size))

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, F)
            feat_chunk = sample.features[start:end]   # [L, D]
            lab_chunk = sample.labels[start:end]       # [L]

            # Паддинг последнего чанка если нужно
            pad = chunk_size - feat_chunk.shape[0]
            if pad > 0:
                feat_chunk = torch.cat([
                    feat_chunk,
                    torch.zeros(pad, feat_chunk.shape[1])
                ], dim=0)
                lab_chunk = torch.cat([lab_chunk, torch.zeros(pad)], dim=0)

            yield TBPTTChunk(
                features=feat_chunk,
                labels=lab_chunk,
                is_first=(i == 0),
            )


class FlatFrameDataset(Dataset):
    """Dataset для flat mode (покадровые сэмплы без порядка).

    Используется для диагностики и статистики, не для обучения.
    """

    def __init__(self, features_dir: Path, ann_dir: Path, backbone: str) -> None:
        seq_dataset = FrameDataset(features_dir, ann_dir, backbone)
        self._features: list[Tensor] = []
        self._labels: list[float] = []
        for sample in seq_dataset.samples:
            for t in range(sample.features.shape[0]):
                self._features.append(sample.features[t])
                self._labels.append(float(sample.labels[t]))

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._features[idx], torch.tensor(self._labels[idx])
