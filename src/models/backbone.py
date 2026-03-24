"""
CNN Backbone для извлечения признаков из кадров.

Поддерживаемые backbone:
  - mobilenet_v3_small              → output_dim=576
  - mobilenet_v3_large              → output_dim=960
  - efficientnet_b0                 → output_dim=1280
  - efficientnet_b0_framediff       → output_dim=1280, вход 6 каналов [I_t | I_t - I_{t-1}]
  - mobilenet_v3_small_framediff    → output_dim=576,  вход 6 каналов [I_t | I_t - I_{t-1}]
  - efficientnet_b0_tsm             → output_dim=1280, вход [T,C,H,W], TSM на MBConv блоках
  - mobilenet_v3_small_tsm          → output_dim=576,  вход [T,C,H,W], TSM на InvertedResidual блоках
  - mobilenetv4_conv_small          → output_dim=1280, timm (Google 2024, 2.5M params)

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
    "efficientnet_b0_framediff": {
        "factory": None,   # создаётся в _build_framediff_backbone()
        "output_dim": 1280,
    },
    "mobilenet_v3_small_framediff": {
        "factory": None,   # создаётся в _build_mv3_framediff_backbone()
        "output_dim": 576,
    },
    "efficientnet_b0_tsm": {
        "factory": None,   # создаётся в tsm.build_tsm_efficientnet_b0()
        "output_dim": 1280,
    },
    "mobilenet_v3_small_tsm": {
        "factory": None,   # создаётся в tsm.build_tsm_mobilenet_v3_small()
        "output_dim": 576,
    },
    "mobilenetv4_conv_small": {
        "factory": None,   # создаётся в _build_mv4_small_backbone()
        "output_dim": 1280,  # после head conv + global avg pool
    },
}


def _build_mv4_small_backbone() -> nn.Module:
    """MobileNetV4-Conv-Small (timm, pretrained on ImageNet-1k).

    2.5M параметров — легче EfficientNet-B0 (5.3M) при сопоставимом output_dim=1280.
    timm с num_classes=0 и global_pool='avg' возвращает [B, 1280] из forward().
    """
    import timm
    return timm.create_model(
        "mobilenetv4_conv_small",
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )


def _build_mv3_framediff_backbone() -> nn.Module:
    """MobileNetV3-Small с 6-канальным входом для frame difference.

    Первые 3 канала — текущий кадр (веса из pretrained).
    Следующие 3 — разность I_t - I_{t-1} (инициализированы нулями).
    """
    full = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # features[0] = Conv2dNormActivation → [0] = Conv2d(3, 16, ...)
    old_conv = full.features[0][0]
    new_conv = nn.Conv2d(
        6, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = 0.0
    full.features[0][0] = new_conv
    return full


def _build_framediff_backbone() -> nn.Module:
    """EfficientNet-B0 с 6-канальным входом для frame difference.

    Первые 3 канала — текущий кадр (веса из pretrained).
    Следующие 3 — разность I_t - I_{t-1} (инициализированы нулями).
    Остальные слои без изменений.
    """
    full = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # features[0] = Conv2dNormActivation → [0] = Conv2d(3, 32, ...)
    old_conv = full.features[0][0]
    new_conv = nn.Conv2d(
        6, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    # Копируем pretrained веса для первых 3 каналов
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = 0.0   # frame diff каналы = 0
    full.features[0][0] = new_conv
    return full


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

        self._name = name
        self._output_dim: int = BACKBONE_CONFIGS[name]["output_dim"]

        if name == "efficientnet_b0_framediff":
            full_model = _build_framediff_backbone()
        elif name == "mobilenet_v3_small_framediff":
            full_model = _build_mv3_framediff_backbone()
        elif name == "efficientnet_b0_tsm":
            from src.models.tsm import build_tsm_efficientnet_b0
            full_model = build_tsm_efficientnet_b0()
        elif name == "mobilenet_v3_small_tsm":
            from src.models.tsm import build_tsm_mobilenet_v3_small
            full_model = build_tsm_mobilenet_v3_small()
        elif name == "mobilenetv4_conv_small":
            full_model = _build_mv4_small_backbone()
        else:
            full_model = BACKBONE_CONFIGS[name]["factory"]()

        # Убираем classifier head (timm-модели с num_classes=0 уже являются feature extractor)
        if name == "mobilenetv4_conv_small":
            self._timm_model = full_model   # forward() возвращает [B, output_dim] напрямую
        elif name.startswith("mobilenet"):
            self.features = full_model.features
            self.avgpool = full_model.avgpool
        elif name.startswith("efficientnet"):
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
            x: [B, 3, H, W] или [B, 6, H, W] для frame diff backbone
        Returns:
            [B, output_dim]
        """
        if hasattr(self, "_timm_model"):
            return self._timm_model(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
