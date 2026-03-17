"""Тесты для full_model.py, temporal.py, postprocess.py."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.full_model import GymRT
from src.models.temporal import (
    BiLSTMAttentionHead,
    BiLSTMHead,
    CausalTCNHead,
    PerFrameMLP,
)
from src.utils.postprocess import Detection, scores_to_detections

DEVICE = torch.device("cpu")
B, T, D = 2, 20, 576  # batch, timesteps, mobilenet_v3_small dim


# ---------------------------------------------------------------------------
# PerFrameMLP
# ---------------------------------------------------------------------------

def test_perframe_mlp_forward():
    head = PerFrameMLP(input_dim=D, hidden_dim=64)
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T)


def test_perframe_mlp_step():
    head = PerFrameMLP(input_dim=D, hidden_dim=64)
    x = torch.randn(B, D)
    logit, state = head.forward_step(x, None)
    assert logit.shape == (B,)
    assert state is None


# ---------------------------------------------------------------------------
# BiLSTMHead
# ---------------------------------------------------------------------------

def test_bilstm_forward():
    head = BiLSTMHead(input_dim=D, hidden_dim=32, n_layers=2)
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T)


def test_bilstm_step():
    head = BiLSTMHead(input_dim=D, hidden_dim=32, n_layers=2)
    state = head.init_state(B, DEVICE)
    x = torch.randn(B, D)
    logit, new_state = head.forward_step(x, state)
    assert logit.shape == (B,)
    assert new_state[0].shape == (2, B, 32)  # (h, c)


def test_bilstm_step_none_state():
    head = BiLSTMHead(input_dim=D, hidden_dim=32, n_layers=1)
    x = torch.randn(1, D)
    logit, state = head.forward_step(x, None)
    assert logit.shape == (1,)


# ---------------------------------------------------------------------------
# BiLSTMAttentionHead
# ---------------------------------------------------------------------------

def test_bilstm_attn_forward():
    head = BiLSTMAttentionHead(input_dim=D, hidden_dim=32, n_layers=1, attn_window=10)
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T)


def test_bilstm_attn_step():
    head = BiLSTMAttentionHead(input_dim=D, hidden_dim=32, n_layers=1, attn_window=10)
    x = torch.randn(B, D)
    logit, state = head.forward_step(x, None)
    assert logit.shape == (B,)
    h, c, buf = state
    assert buf.shape == (B, 10, 32)


# ---------------------------------------------------------------------------
# CausalTCNHead
# ---------------------------------------------------------------------------

def test_causal_tcn_forward():
    head = CausalTCNHead(input_dim=D, num_channels=[32, 32], kernel_size=3)
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T)


def test_causal_tcn_causality():
    """Вывод в момент t не должен зависеть от кадров t+1, t+2, ..."""
    head = CausalTCNHead(input_dim=16, num_channels=[8], kernel_size=3)
    head.eval()
    x = torch.randn(1, 30, 16)
    with torch.no_grad():
        out_full = head(x)

    # Обрезаем последние 5 кадров и проверяем что первые 25 не изменились
    x_short = x[:, :25, :]
    with torch.no_grad():
        out_short = head(x_short)

    assert torch.allclose(out_full[:, :25], out_short, atol=1e-5), \
        "CausalTCN нарушает причинность: вывод зависит от будущих кадров"


def test_causal_tcn_step_raises():
    head = CausalTCNHead(input_dim=D)
    with pytest.raises(NotImplementedError):
        head.forward_step(torch.randn(1, D), None)


# ---------------------------------------------------------------------------
# GymRT
# ---------------------------------------------------------------------------

def test_gymrt_forward_4d():
    """[T, C, H, W] → [1, T] (автоматически добавляется batch dim)."""
    model = GymRT("mobilenet_v3_small", "per_frame_mlp",
                  {"hidden_dim": 32}, frozen_backbone=True)
    model.eval()
    x = torch.zeros(5, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


def test_gymrt_forward_5d():
    """[B, T, C, H, W] → [B, T]."""
    model = GymRT("mobilenet_v3_small", "per_frame_mlp",
                  {"hidden_dim": 32}, frozen_backbone=True)
    model.eval()
    x = torch.zeros(2, 3, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3)


def test_gymrt_forward_frame():
    model = GymRT("mobilenet_v3_small", "bilstm",
                  {"hidden_dim": 32, "n_layers": 1}, frozen_backbone=True)
    model.eval()
    frame = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        logit, state = model.forward_frame(frame, None)
    assert logit.shape == (1,)


def test_gymrt_size_mb():
    model = GymRT("mobilenet_v3_small", "per_frame_mlp", {"hidden_dim": 32})
    assert model.size_mb() > 0


def test_gymrt_unknown_head():
    with pytest.raises(ValueError, match="Неизвестный temporal head"):
        GymRT("mobilenet_v3_small", "transformer_xl")


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------

def test_scores_to_detections_basic():
    fps = 25.0
    scores = [0.0] * 10 + [0.9] * 50 + [0.0] * 10  # элемент 10..59 = 2.0s
    dets = scores_to_detections(scores, fps, threshold=0.5,
                                min_duration_sec=1.5, max_duration_sec=12.0)
    assert len(dets) == 1
    assert dets[0].start_frame == 10
    assert dets[0].end_frame == 60
    assert abs(dets[0].duration_sec(fps) - 2.0) < 0.01


def test_scores_too_short_filtered():
    fps = 25.0
    # 10 кадров = 0.4s < min 1.5s → отфильтровывается
    scores = [0.0] * 10 + [0.9] * 10 + [0.0] * 10
    dets = scores_to_detections(scores, fps, threshold=0.5)
    assert len(dets) == 0


def test_scores_no_detection():
    scores = [0.1] * 100
    dets = scores_to_detections(scores, fps=25.0, threshold=0.5)
    assert len(dets) == 0


def test_scores_tensor_input():
    fps = 25.0
    scores = torch.cat([
        torch.zeros(10),
        torch.ones(50) * 0.9,
        torch.zeros(10),
    ])
    dets = scores_to_detections(scores, fps, threshold=0.5)
    assert len(dets) == 1


def test_detection_sec():
    d = Detection(start_frame=25, end_frame=75, score=0.9)
    assert d.start_sec(25.0) == pytest.approx(1.0)
    assert d.end_sec(25.0) == pytest.approx(3.0)
    assert d.duration_sec(25.0) == pytest.approx(2.0)
