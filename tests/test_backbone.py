"""Тесты для backbone.py."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.backbone import BACKBONE_CONFIGS, CNNBackbone

# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected_dim", [
    ("mobilenet_v3_small", 576),
    ("mobilenet_v3_large", 960),
    ("efficientnet_b0", 1280),
])
def test_output_shape(name, expected_dim):
    backbone = CNNBackbone(name, frozen=True)
    backbone.eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (2, expected_dim), f"{name}: got {out.shape}"


def test_output_dim_property():
    for name, cfg in BACKBONE_CONFIGS.items():
        backbone = CNNBackbone(name, frozen=True)
        assert backbone.output_dim == cfg["output_dim"]


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

def test_frozen_weights_not_updated():
    backbone = CNNBackbone("mobilenet_v3_small", frozen=True)
    # Все параметры должны быть frozen
    for p in backbone.parameters():
        assert not p.requires_grad, "Параметр должен быть заморожен"


def test_unfreeze():
    backbone = CNNBackbone("mobilenet_v3_small", frozen=True)
    backbone.unfreeze()
    for p in backbone.parameters():
        assert p.requires_grad, "Параметр должен быть разморожен"


def test_default_not_frozen():
    backbone = CNNBackbone("mobilenet_v3_small", frozen=False)
    # По умолчанию — не заморожен
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    assert len(trainable) > 0


# ---------------------------------------------------------------------------
# Unknown backbone
# ---------------------------------------------------------------------------

def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Неизвестный backbone"):
        CNNBackbone("resnet50")


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------

def test_name_property():
    backbone = CNNBackbone("mobilenet_v3_small", frozen=True)
    assert backbone.name == "mobilenet_v3_small"
