"""
Pose-based модель для ConElGym_RT.

Вместо CNN backbone использует MediaPipe Pose-признаки (99-dim).
Архитектура: Linear(99→hidden_dim) → BiLSTM+Attention → Linear→logit

PoseHead — temporal head с встроенной проекцией входа.
  Интерфейс совместим с BiLSTMAttentionHead:
  - forward([B, T, 99])                → logits [B, T]
  - forward_train([B, T, 99], state)   → (logits [B, T], new_state)
  - forward_step([B, 99], state)       → (logit [B], new_state)
  - init_state(batch_size, device)     → state

PoseGymRT — обёртка над PoseHead, аналог GymRT для pose моделей.
  Экспортирует .temporal, .size_mb(), .count_parameters()
  для совместимости с train.py / hpo.py / loao_cv.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.temporal import BiLSTMAttentionHead, BiLSTMHead, CausalTCNHead

POSE_DIM = 99   # 33 landmarks × 3

POSE_TEMPORAL_HEADS = {
    "bilstm_attn": BiLSTMAttentionHead,
    "bilstm":      BiLSTMHead,
    "causal_tcn":  CausalTCNHead,
}


class PoseHead(nn.Module):
    """Linear projection + temporal head для pose-признаков.

    Args:
        pose_dim:      размерность входа (99 по умолчанию)
        hidden_dim:    размер скрытого слоя проекции и temporal head
        temporal_name: имя temporal head ('bilstm_attn', 'bilstm', 'causal_tcn')
        temporal_cfg:  kwargs для temporal head (n_layers, dropout, ...)
    """

    def __init__(
        self,
        pose_dim: int = POSE_DIM,
        hidden_dim: int = 256,
        temporal_name: str = "bilstm_attn",
        temporal_cfg: dict | None = None,
    ) -> None:
        super().__init__()

        if temporal_name not in POSE_TEMPORAL_HEADS:
            raise ValueError(
                f"Неизвестный temporal head: '{temporal_name}'. "
                f"Доступные: {list(POSE_TEMPORAL_HEADS.keys())}"
            )

        self.projection = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
        )

        cfg = dict(temporal_cfg or {})
        if temporal_name == "causal_tcn":
            cfg.pop("n_layers", None)
            cfg.pop("hidden_dim", None)

        cls = POSE_TEMPORAL_HEADS[temporal_name]
        self._temporal = cls(input_dim=hidden_dim, **cfg)
        self._temporal_name = temporal_name

    def _project(self, x: Tensor) -> Tensor:
        """[B, T, 99] → [B, T, hidden_dim]"""
        return self.projection(x)

    def forward(self, x: Tensor) -> Tensor:
        """[B, T, 99] → logits [B, T]"""
        return self._temporal(self._project(x))

    def forward_train(
        self, x: Tensor, state: Any
    ) -> tuple[Tensor, Any]:
        """TBPTT-совместимый forward. [B, T, 99], state → (logits [B, T], new_state)"""
        return self._temporal.forward_train(self._project(x), state)

    def forward_step(
        self, x: Tensor, state: Any
    ) -> tuple[Tensor, Any]:
        """Стриминг: [B, 99], state → (logit [B], new_state)"""
        proj = self.projection(x)       # [B, hidden_dim]
        return self._temporal.forward_step(proj, state)

    def init_state(self, batch_size: int, device: torch.device) -> Any:
        return self._temporal.init_state(batch_size, device)

    def _copy_fwd_weights(self) -> None:
        """Синхронизирует BiLSTM fwd-веса перед стримингом (если применимо)."""
        if hasattr(self._temporal, "_copy_fwd_weights"):
            self._temporal._copy_fwd_weights()


class PoseGymRT(nn.Module):
    """Полная pose-модель: PoseHead без CNN backbone.

    Интерфейс совместим с GymRT для использования в train.py / hpo.py / loao_cv.py:
      - .temporal  — expose PoseHead как temporal module
      - .size_mb()
      - .count_parameters()
    """

    def __init__(
        self,
        pose_dim: int = POSE_DIM,
        hidden_dim: int = 256,
        temporal_name: str = "bilstm_attn",
        temporal_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.temporal = PoseHead(
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            temporal_name=temporal_name,
            temporal_cfg=temporal_cfg,
        )
        self._temporal_name = temporal_name

    def forward(self, x: Tensor) -> Tensor:
        """[B, T, 99] → logits [B, T]"""
        return self.temporal(x)

    def forward_frame(
        self, frame_pose: Tensor, state: Any
    ) -> tuple[Tensor, Any]:
        """Стриминг: [B, 99] → (logit [B], new_state)"""
        return self.temporal.forward_step(frame_pose, state)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> Any:
        if device is None:
            device = next(self.parameters()).device
        self.temporal._copy_fwd_weights()
        return self.temporal.init_state(batch_size, device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def size_mb(self) -> float:
        total = sum(p.numel() * p.element_size() for p in self.parameters())
        return total / (1024 ** 2)
