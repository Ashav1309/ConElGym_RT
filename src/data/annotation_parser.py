"""
Парсинг JSON-аннотаций ConElGym.

Формат аннотации:
  {
    "video_name": "Ball_001.mp4",
    "annotations": [
      {"start_frame": 50, "end_frame": 173, "element_type": "element", "confidence": 1.0}
    ]
  }

Пустой список annotations → негативный пример (нет элемента).
Индексы кадров 0-based.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    video_name: str
    has_element: bool
    start_frame: int  # включительно; 0 если has_element=False
    end_frame: int    # не включительно; 0 если has_element=False

    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame if self.has_element else 0

    def frame_label(self, frame_idx: int) -> int:
        """Возвращает 1 если кадр внутри элемента, иначе 0."""
        if not self.has_element:
            return 0
        return int(self.start_frame <= frame_idx < self.end_frame)


def load_annotation(json_path: Path) -> Annotation:
    """Загружает одну аннотацию из JSON-файла."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    video_name = data["video_name"]
    annotations = data.get("annotations", [])

    if not annotations:
        return Annotation(
            video_name=video_name,
            has_element=False,
            start_frame=0,
            end_frame=0,
        )

    ann = annotations[0]
    start_frame = int(ann["start_frame"])
    end_frame = int(ann["end_frame"])

    if start_frame >= end_frame:
        raise ValueError(
            f"{json_path.name}: start_frame={start_frame} >= end_frame={end_frame}"
        )

    return Annotation(
        video_name=video_name,
        has_element=True,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def load_annotations(ann_dir: Path) -> dict[str, Annotation]:
    """Загружает все аннотации из директории.

    Returns:
        dict: video_name → Annotation
    """
    result: dict[str, Annotation] = {}
    for json_path in sorted(ann_dir.glob("*.json")):
        ann = load_annotation(json_path)
        result[ann.video_name] = ann
    return result
