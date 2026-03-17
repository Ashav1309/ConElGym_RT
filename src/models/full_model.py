"""
GymRT — полная модель ConElGym_RT.

Объединяет CNN Backbone + Temporal Head в единый модуль.

Режимы:
  forward(x)              — обучение/оффлайн оценка: [B, T, C, H, W] → logits [B, T]
  forward_frame(x, state) — стриминг: [B, C, H, W] → (logit [B], new_state)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbone import CNNBackbone
from src.models.temporal import (
    BiLSTMAttentionHead,
    BiLSTMHead,
    CausalTCNHead,
    PerFrameMLP,
)

TEMPORAL_HEADS = {
    "per_frame_mlp": PerFrameMLP,
    "bilstm": BiLSTMHead,
    "bilstm_attn": BiLSTMAttentionHead,
    "causal_tcn": CausalTCNHead,
}


def build_temporal_head(
    name: str, input_dim: int, cfg: dict
) -> nn.Module:
    if name not in TEMPORAL_HEADS:
        raise ValueError(
            f"Неизвестный temporal head: '{name}'. "
            f"Доступные: {list(TEMPORAL_HEADS.keys())}"
        )
    cfg = dict(cfg)
    # PerFrameMLP и CausalTCN не принимают n_layers
    if name in ("per_frame_mlp", "causal_tcn"):
        cfg.pop("n_layers", None)
    cls = TEMPORAL_HEADS[name]
    return cls(input_dim=input_dim, **cfg)


class GymRT(nn.Module):
    """Backbone + Temporal Head.

    Args:
        backbone_name:   имя CNN backbone (см. backbone.py)
        temporal_name:   имя temporal head (см. temporal.py)
        temporal_cfg:    kwargs для temporal head (hidden_dim, n_layers, ...)
        frozen_backbone: заморозить backbone при инициализации
    """

    def __init__(
        self,
        backbone_name: str,
        temporal_name: str,
        temporal_cfg: dict | None = None,
        frozen_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone(backbone_name, frozen=frozen_backbone)
        self.temporal = build_temporal_head(
            temporal_name,
            input_dim=self.backbone.output_dim,
            cfg=temporal_cfg or {},
        )
        self._backbone_name = backbone_name
        self._temporal_name = temporal_name

    @property
    def backbone_name(self) -> str:
        return self._backbone_name

    @property
    def temporal_name(self) -> str:
        return self._temporal_name

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    def forward(self, x: Tensor) -> Tensor:
        """Батчевый forward для обучения.

        Args:
            x: [B, T, C, H, W]  — батч последовательностей кадров
               ИЛИ [T, C, H, W] → автоматически добавляется batch dim

        Returns:
            logits: [B, T]
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)                    # [1, T, C, H, W]

        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)           # [B*T, C, H, W]
        feats = self.backbone(frames)              # [B*T, D]
        feats = feats.view(B, T, -1)              # [B, T, D]
        return self.temporal(feats)                # [B, T]

    def forward_frame(
        self, frame: Tensor, state: Any
    ) -> tuple[Tensor, Any]:
        """Пошаговый стриминг-инференс.

        Args:
            frame: [B, C, H, W]  — один кадр
            state: hidden state temporal head (None для первого кадра)

        Returns:
            (logit [B], new_state)
        """
        feat = self.backbone(frame)                # [B, D]
        return self.temporal.forward_step(feat, state)

    def init_state(self, batch_size: int = 1, device: torch.device | None = None) -> Any:
        """Инициализирует hidden state для стриминга."""
        if device is None:
            device = next(self.parameters()).device
        return self.temporal.init_state(batch_size, device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def size_mb(self) -> float:
        """Размер модели в MB (только параметры float32)."""
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        return total / (1024 ** 2)
