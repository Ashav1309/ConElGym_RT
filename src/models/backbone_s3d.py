"""
S3D Backbone — RT-reference для ConElGym_RT.

Адаптирован из ConElGym_v2/src/models/feature_extractor.py.
Используется ТОЛЬКО для замера FPS и сравнения в диссертации.
Не участвует в обучении — веса S3D заморожены.

Вход клипа: [C, T, H, W] или [B, C, T, H, W]
Выход:      [1024] или [B, 1024]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models.video as vm
from torchvision.models.video import S3D_Weights

FEATURE_DIM = 1024
WINDOW_SIZE = 16
STRIDE = 8
FRAME_SIZE = 224

# ImageNet нормализация (как в v2)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


class S3DBackbone(nn.Module):
    """S3D feature extractor (замороженный, Kinetics-400).

    Используется как RT-reference: измеряем FPS клип-за-клипом,
    чтобы показать что 3D backbone нарушает RT-требование (>25 FPS).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = S3D_Weights.DEFAULT if pretrained else None
        model = vm.s3d(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.output_dim = FEATURE_DIM

        # S3D всегда заморожен — используется только как feature extractor
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: [C, T, H, W] или [B, C, T, H, W], float32, нормализованный
        Returns:
            [1024] или [B, 1024]
        """
        squeeze = clip.ndim == 4
        if squeeze:
            clip = clip.unsqueeze(0)
        feat = self.features(clip)   # [B, 1024, t, h, w]
        feat = self.avgpool(feat)    # [B, 1024, 1, 1, 1]
        feat = feat.flatten(1)       # [B, 1024]
        return feat.squeeze(0) if squeeze else feat


def preprocess_frame(frame_bgr, device: torch.device) -> torch.Tensor:
    """BGR кадр (numpy HxWx3 uint8) → нормализованный тензор [3, H, W]."""
    import cv2
    import numpy as np

    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = _MEAN.squeeze(-1).to(device)
    std  = _STD.squeeze(-1).to(device)
    return (t.to(device) - mean) / std


def extract_s3d_features_rt(
    video_path: str,
    backbone: S3DBackbone,
    device: torch.device,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tuple[torch.Tensor, list[int], float]:
    """
    Извлечь S3D признаки в режиме скользящего окна (RT-симуляция).

    Returns:
        features:     [N, 1024]
        start_frames: [N]
        fps:          кадров/сек (включая декодирование + S3D inference)
    """
    import time

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(str(video_path))
    buffer: list[torch.Tensor] = []
    all_features: list[torch.Tensor] = []
    start_frames: list[int] = []
    frame_idx = 0

    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(preprocess_frame(frame, device))
        frame_idx += 1

        if len(buffer) == window_size:
            clip = torch.stack(buffer, dim=1).unsqueeze(0)  # [1, C, T, H, W]
            feat = backbone(clip)                            # [1024]
            all_features.append(feat.cpu())
            start_frames.append(frame_idx - window_size)
            # Скользим на stride кадров
            buffer = buffer[stride:]

    cap.release()
    elapsed = time.perf_counter() - t_start
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    if not all_features:
        return torch.zeros(0, FEATURE_DIM), [], fps

    features = torch.stack(all_features)  # [N, 1024]
    return features, start_frames, fps
