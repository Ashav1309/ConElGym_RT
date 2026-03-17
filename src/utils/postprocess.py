"""
Постобработка score-последовательности → список детекций.

Pipeline:
  scores[T] → threshold → merge_intervals → duration_filter → detections
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Detection:
    """Одна детектированная сцена."""
    start_frame: int
    end_frame: int
    score: float         # максимальный score внутри сегмента

    def start_sec(self, fps: float) -> float:
        return self.start_frame / fps

    def end_sec(self, fps: float) -> float:
        return self.end_frame / fps

    def duration_sec(self, fps: float) -> float:
        return (self.end_frame - self.start_frame) / fps


def scores_to_detections(
    scores: Tensor | list[float],
    fps: float,
    threshold: float = 0.5,
    min_duration_sec: float = 1.5,
    max_duration_sec: float = 12.0,
) -> list[Detection]:
    """Преобразует frame-level scores в список Detection.

    Args:
        scores:           [T] — sigmoid-вероятности (или логиты если < 0)
        fps:              кадров в секунду
        threshold:        порог бинаризации
        min_duration_sec: минимальная длительность детекции
        max_duration_sec: максимальная длительность детекции

    Returns:
        Список Detection (обычно 0 или 1 для нашей задачи).
    """
    if isinstance(scores, Tensor):
        probs = torch.sigmoid(scores) if scores.min() < 0 else scores
        probs_list = probs.tolist()
    else:
        probs_list = list(scores)

    min_frames = int(min_duration_sec * fps)
    max_frames = int(max_duration_sec * fps)

    # Бинаризация
    binary = [p >= threshold for p in probs_list]

    # Слияние последовательных активных кадров в сегменты
    segments: list[tuple[int, int, float]] = []  # (start, end, max_score)
    in_seg = False
    seg_start = 0
    seg_max = 0.0

    for t, (active, prob) in enumerate(zip(binary, probs_list)):
        if active and not in_seg:
            in_seg = True
            seg_start = t
            seg_max = prob
        elif active and in_seg:
            seg_max = max(seg_max, prob)
        elif not active and in_seg:
            segments.append((seg_start, t, seg_max))
            in_seg = False
            seg_max = 0.0

    if in_seg:
        segments.append((seg_start, len(probs_list), seg_max))

    # Duration filter
    detections = []
    for start, end, score in segments:
        duration = end - start
        if min_frames <= duration <= max_frames:
            detections.append(Detection(start_frame=start, end_frame=end, score=score))

    return detections
