"""
Temporal Shift Module (TSM) для ConElGym_RT.

Ссылка: Lin et al., "TSM: Temporal Shift Module for Efficient Video Understanding"
        https://arxiv.org/abs/1811.08383

В режиме обучения/оффлайн оценки:
  - Входной тензор: [T, C, H, W] — последовательность кадров одного видео
  - 1/fold_div каналов сдвигается из t-1 (левый сдвиг)
  - 1/fold_div каналов сдвигается из t+1 (правый сдвиг) — только в bidirectional
  - Остальные каналы не меняются

В режиме стриминга (causal=True):
  - Только левый сдвиг (причинный): каналы из t-1
  - Буфер `shift_buffer` хранит shifted каналы предыдущего кадра

Использование:
  backbone = CNNBackbone('efficientnet_b0_tsm')
  # При extract_frame_features: обрабатываем всё видео [T, C, H, W] одним проходом
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    """Обёртка-сдвиг: применяет temporal shift, затем вызывает wrapped модуль.

    Args:
        module:     оборачиваемый модуль (MBConv, InvertedResidual, ...)
        fold_div:   1/fold_div каналов сдвигается в каждую сторону (default 8 → 1/8)
        bidirectional: если True — сдвиг влево И вправо (train/offline eval)
                       если False — только влево (causal, streaming)
    """

    def __init__(
        self,
        module: nn.Module,
        fold_div: int = 8,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.fold_div = fold_div
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, C, H, W] — последовательность кадров одного видео
        Returns:
            [T, C, H, W] после temporal shift + module forward
        """
        x = temporal_shift(x, self.fold_div, self.bidirectional)
        return self.module(x)

    def forward_streaming(
        self, x: torch.Tensor, buffer: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Пошаговый каузальный инференс для одного кадра.

        Args:
            x:      [1, C, H, W] — один кадр
            buffer: [1, fold, H, W] буфер из предыдущего шага (None = нули)

        Returns:
            (out [1, C, H, W], new_buffer [1, fold, H, W])
        """
        T, C, H, W = x.shape
        fold = C // self.fold_div

        if buffer is None:
            buffer = torch.zeros(T, fold, H, W, device=x.device, dtype=x.dtype)

        x_shifted = x.clone()
        x_shifted[:, :fold] = buffer           # каналы из t-1
        new_buffer = x[:, :fold].clone()       # сохраняем для следующего шага

        return self.module(x_shifted), new_buffer


def temporal_shift(
    x: torch.Tensor,
    fold_div: int = 8,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Применяет temporal shift к тензору [T, C, H, W].

    Args:
        x:            [T, C, H, W]
        fold_div:     знаменатель для доли каналов (1/fold_div)
        bidirectional: True — сдвиг влево + вправо; False — только влево (каузальный)

    Returns:
        [T, C, H, W] с shifted каналами
    """
    T, C, H, W = x.shape
    fold = C // fold_div

    # Левый сдвиг: канал [t] ← канал [t-1]
    left = torch.cat([x.new_zeros(1, fold, H, W), x[:-1, :fold]], dim=0)

    if bidirectional:
        # Правый сдвиг: канал [t] ← канал [t+1]
        right = torch.cat([x[1:, fold:2 * fold], x.new_zeros(1, fold, H, W)], dim=0)
        return torch.cat([left, right, x[:, 2 * fold:]], dim=1)

    return torch.cat([left, x[:, fold:]], dim=1)


def wrap_mbconv_with_tsm(
    module: nn.Module,
    fold_div: int = 8,
    bidirectional: bool = True,
) -> nn.Module:
    """Рекурсивно оборачивает все MBConv блоки в Sequential с TemporalShift.

    Применяется к `features[1..7]` EfficientNet-B0.
    """
    for name, child in module.named_children():
        # Оборачиваем только MBConv (не первый и последний Conv2dNormActivation)
        child_type = type(child).__name__
        if child_type == "MBConv":
            setattr(module, name, TemporalShift(child, fold_div, bidirectional))
        else:
            wrap_mbconv_with_tsm(child, fold_div, bidirectional)

    return module


def wrap_inverted_residual_with_tsm(
    module: nn.Module,
    fold_div: int = 8,
    bidirectional: bool = True,
) -> nn.Module:
    """Рекурсивно оборачивает все InvertedResidual блоки в TemporalShift.

    Применяется к `features[1..11]` MobileNetV3-Small.
    """
    for name, child in module.named_children():
        child_type = type(child).__name__
        if child_type == "InvertedResidual":
            setattr(module, name, TemporalShift(child, fold_div, bidirectional))
        else:
            wrap_inverted_residual_with_tsm(child, fold_div, bidirectional)
    return module


def build_tsm_mobilenet_v3_small(
    fold_div: int = 8,
    bidirectional: bool = True,
) -> nn.Module:
    """MobileNetV3-Small с TSM, обёрнутым вокруг всех InvertedResidual блоков.

    Веса — ImageNet pretrained. TSM не добавляет параметров.
    Вход: [T, C, H, W] — ОБЯЗАТЕЛЬНО полная последовательность для корректного сдвига.

    Примечание: в MobileNetV3 features[1..11] — это САМИ InvertedResidual (в отличие от
    EfficientNet, где features[i] — Sequential, содержащий MBConv). Поэтому оборачиваем
    напрямую, без рекурсии через wrap_inverted_residual_with_tsm.
    """
    from torchvision import models
    from torchvision.models import MobileNet_V3_Small_Weights

    full = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # features[1..11] включительно — прямые InvertedResidual, заменяем напрямую
    for i in range(1, 12):
        full.features[i] = TemporalShift(full.features[i], fold_div, bidirectional)

    return full


def build_tsm_efficientnet_b0(
    fold_div: int = 8,
    bidirectional: bool = True,
) -> nn.Module:
    """EfficientNet-B0 с TSM, обёрнутым вокруг всех MBConv блоков.

    Веса — ImageNet pretrained. TSM не добавляет параметров.
    Вход: [T, C, H, W] — ОБЯЗАТЕЛЬНО полная последовательность для корректного сдвига.
    """
    from torchvision import models
    from torchvision.models import EfficientNet_B0_Weights

    full = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Оборачиваем блоки features[1..7] (Sequential с MBConv)
    for i in range(1, 8):
        wrap_mbconv_with_tsm(full.features[i], fold_div, bidirectional)

    return full


class TSMState:
    """Состояние TSM для стриминг-инференса.

    Хранит буферы сдвига для каждого TemporalShift слоя.
    """

    def __init__(self) -> None:
        self.buffers: dict[int, torch.Tensor | None] = {}

    def get(self, layer_id: int) -> torch.Tensor | None:
        return self.buffers.get(layer_id)

    def set(self, layer_id: int, buf: torch.Tensor) -> None:
        self.buffers[layer_id] = buf

    def to(self, device: torch.device) -> "TSMState":
        self.buffers = {
            k: v.to(device) if v is not None else None
            for k, v in self.buffers.items()
        }
        return self

    def __repr__(self) -> str:
        return f"TSMState(layers={list(self.buffers.keys())})"
