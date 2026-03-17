"""
CNN Backbone для извлечения признаков из кадров.

Поддерживаемые backbone:
  - mobilenet_v3_small   → output_dim=576
  - mobilenet_v3_large   → output_dim=960
  - efficientnet_b0      → output_dim=1280

Все веса — ImageNet pretrained.
При frozen=True веса backbone не обновляются (используется при первых N эпохах).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)

BACKBONE_CONFIGS: dict[str, dict] = {
    "mobilenet_v3_small": {
        "factory": lambda: models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        ),
        "output_dim": 576,
    },
    "mobilenet_v3_large": {
        "factory": lambda: models.mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
        ),
        "output_dim": 960,
    },
    "efficientnet_b0": {
        "factory": lambda: models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        ),
        "output_dim": 1280,
    },
}


class CNNBackbone(nn.Module):
    """Обёртка над torchvision backbone.

    Убирает classification head, оставляет только feature extractor.
    forward(x) → [B, output_dim] (после AdaptiveAvgPool + flatten).

    Args:
        name:     имя backbone из BACKBONE_CONFIGS
        frozen:   если True — замораживает все параметры backbone
    """

    def __init__(self, name: str, frozen: bool = False) -> None:
        super().__init__()

        if name not in BACKBONE_CONFIGS:
            raise ValueError(
                f"Неизвестный backbone: '{name}'. "
                f"Доступные: {list(BACKBONE_CONFIGS.keys())}"
            )

        cfg = BACKBONE_CONFIGS[name]
        self._name = name
        self._output_dim: int = cfg["output_dim"]

        full_model = cfg["factory"]()

        # Убираем classifier head
        if name.startswith("mobilenet"):
            # MobileNetV3: features → avgpool → flatten → classifier
            # Оставляем features + avgpool
            self.features = full_model.features
            self.avgpool = full_model.avgpool
        elif name.startswith("efficientnet"):
            # EfficientNet: features → avgpool → classifier
            self.features = full_model.features
            self.avgpool = full_model.avgpool
        else:
            raise ValueError(f"Нет обработчика для backbone: {name}")

        if frozen:
            self.freeze()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def name(self) -> str:
        return self._name

    def freeze(self) -> None:
        """Замораживает все параметры backbone."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Размораживает все параметры backbone."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  — нормализованные кадры
        Returns:
            [B, output_dim]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
