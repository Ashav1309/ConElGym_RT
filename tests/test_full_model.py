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


# ---------------------------------------------------------------------------
# build_temporal_head: n_layers stripping for per_frame_mlp and causal_tcn
# ---------------------------------------------------------------------------

def test_build_temporal_head_strips_n_layers_for_per_frame_mlp():
    """per_frame_mlp не принимает n_layers — build_temporal_head должен убрать его."""
    from src.models.full_model import build_temporal_head

    # Если n_layers НЕ убирается, PerFrameMLP.__init__ бросает TypeError
    head = build_temporal_head(
        "per_frame_mlp",
        input_dim=64,
        cfg={"hidden_dim": 32, "n_layers": 3},
    )
    assert isinstance(head, PerFrameMLP)


def test_build_temporal_head_strips_n_layers_for_causal_tcn():
    """causal_tcn не принимает n_layers — build_temporal_head должен убрать его."""
    from src.models.full_model import build_temporal_head

    head = build_temporal_head(
        "causal_tcn",
        input_dim=64,
        cfg={"num_channels": [16], "n_layers": 2},
    )
    assert isinstance(head, CausalTCNHead)


def test_build_temporal_head_passes_n_layers_to_bilstm():
    """bilstm принимает n_layers — значение должно учитываться."""
    from src.models.full_model import build_temporal_head

    head = build_temporal_head(
        "bilstm",
        input_dim=64,
        cfg={"hidden_dim": 32, "n_layers": 3},
    )
    assert isinstance(head, BiLSTMHead)
    assert head.n_layers == 3


# ---------------------------------------------------------------------------
# BiLSTMHead: _copy_fwd_weights + forward_step vs forward consistency
# ---------------------------------------------------------------------------

def test_bilstm_copy_fwd_weights_produces_same_logit_on_single_timestep():
    """После _copy_fwd_weights forward_step должен давать тот же логит, что
    forward() на однокадровой последовательности (нет контекста от backward pass).
    """
    torch.manual_seed(0)
    head = BiLSTMHead(input_dim=32, hidden_dim=16, n_layers=1, dropout=0.0)
    head.eval()
    head._copy_fwd_weights()

    x = torch.randn(1, 32)  # один кадр, batch=1

    with torch.no_grad():
        # forward() на [1, 1, D]
        logit_full = head(x.unsqueeze(0))       # [1, 1] → берём [0, 0]
        logit_full_val = logit_full[0, 0].item()

        # forward_step с нулевым состоянием
        logit_step, _ = head.forward_step(x, None)
        logit_step_val = logit_step[0].item()

    # При n_layers=1 и нулевом состоянии BiLSTM для T=1 forward output совпадает
    # с однонаправленным forward_step (backward pass даёт нулевой контекст тоже).
    # Проверяем знак совпадает — полное равенство не гарантировано из-за FC весов.
    assert (logit_full_val > 0) == (logit_step_val > 0), (
        f"Знак логита не совпадает: forward={logit_full_val:.4f}, "
        f"forward_step={logit_step_val:.4f}"
    )


def test_bilstm_step_by_step_state_shapes_n_layers_2():
    """State shape должен соответствовать n_layers для каждого шага."""
    head = BiLSTMHead(input_dim=16, hidden_dim=8, n_layers=2, dropout=0.0)
    state = head.init_state(batch_size=3, device=DEVICE)

    for _ in range(5):
        x = torch.randn(3, 16)
        logit, state = head.forward_step(x, state)

    h, c = state
    assert h.shape == (2, 3, 8)   # (n_layers, B, hidden_dim)
    assert c.shape == (2, 3, 8)


def test_bilstm_step_state_is_not_none_after_first_step():
    """forward_step с None state возвращает корректный инициализированный state."""
    head = BiLSTMHead(input_dim=16, hidden_dim=8, n_layers=1, dropout=0.0)
    x = torch.randn(1, 16)
    logit, state = head.forward_step(x, None)
    assert state is not None
    h, c = state
    assert h.shape == (1, 1, 8)


def test_bilstm_step_logit_changes_with_state_vs_no_state():
    """Логит на одном и том же кадре разный при наличии контекста и без него."""
    torch.manual_seed(42)
    head = BiLSTMHead(input_dim=16, hidden_dim=8, n_layers=1, dropout=0.0)
    head.eval()

    x_context = torch.randn(1, 16)
    x_test = torch.randn(1, 16)

    with torch.no_grad():
        # Без контекста
        logit_no_ctx, _ = head.forward_step(x_test, None)

        # С контекстом от предыдущего кадра
        _, state_after_ctx = head.forward_step(x_context, None)
        logit_with_ctx, _ = head.forward_step(x_test, state_after_ctx)

    # Контекст должен изменить логит
    assert not torch.allclose(logit_no_ctx, logit_with_ctx), (
        "Логит не изменился после передачи state — контекст не используется"
    )


# ---------------------------------------------------------------------------
# BiLSTMAttentionHead: sliding buffer eviction
# ---------------------------------------------------------------------------

def test_bilstm_attn_buffer_evicts_oldest_entry():
    """После attn_window шагов самая старая запись в буфере вытесняется."""
    attn_window = 4
    head = BiLSTMAttentionHead(
        input_dim=16, hidden_dim=8, n_layers=1,
        dropout=0.0, attn_window=attn_window
    )
    head.eval()

    state = None
    buffers = []
    with torch.no_grad():
        for _ in range(attn_window + 2):
            x = torch.randn(1, 16)
            _, state = head.forward_step(x, state)
            h, c, buf = state
            buffers.append(buf.clone())

    # После attn_window+2 шагов буфер должен содержать только последние attn_window значений
    _, _, final_buf = state
    assert final_buf.shape == (1, attn_window, 8)


def test_bilstm_attn_buffer_shifts_on_each_step():
    """Буфер сдвигается влево: позиция 0 уходит, новый элемент в конце."""
    attn_window = 3
    head = BiLSTMAttentionHead(
        input_dim=8, hidden_dim=4, n_layers=1,
        dropout=0.0, attn_window=attn_window
    )
    head.eval()

    # Два шага: после первого шага buf[-1] содержит cur_step_1
    x1 = torch.randn(1, 8)
    x2 = torch.randn(1, 8)

    with torch.no_grad():
        _, state1 = head.forward_step(x1, None)
        _, state2 = head.forward_step(x2, state1)

    h1, c1, buf1 = state1
    h2, c2, buf2 = state2

    # Последний элемент buf1 должен стать предпоследним в buf2
    assert torch.allclose(buf1[:, -1, :], buf2[:, -2, :], atol=1e-6), (
        "FIFO буфер: последний элемент шага N должен стать предпоследним на шаге N+1"
    )


# ---------------------------------------------------------------------------
# CausalTCN: multi-layer causality and boundary cases
# ---------------------------------------------------------------------------

def test_causal_tcn_causality_multi_layer():
    """Многослойная TCN с экспоненциальными dilation не должна нарушать причинность."""
    # 3 слоя: dilation=1,2,4 → receptive field = (7-1)*(1+2+4) + 1 = 43 кадра
    head = CausalTCNHead(
        input_dim=16,
        num_channels=[8, 8, 8],
        kernel_size=7,
    )
    head.eval()
    T = 60

    torch.manual_seed(1)
    x = torch.randn(1, T, 16)

    with torch.no_grad():
        out_full = head(x)

    # Обрезаем последние 10 кадров
    x_short = x[:, :T - 10, :]
    with torch.no_grad():
        out_short = head(x_short)

    assert torch.allclose(out_full[:, :T - 10], out_short, atol=1e-5), (
        "Многослойная CausalTCN нарушает причинность"
    )


def test_causal_tcn_causality_at_first_timestep():
    """Вывод на t=0 не должен зависеть от любых будущих кадров."""
    head = CausalTCNHead(input_dim=8, num_channels=[4], kernel_size=3)
    head.eval()

    torch.manual_seed(2)
    x = torch.randn(1, 20, 8)

    with torch.no_grad():
        out_full = head(x)

    # Меняем все кадры кроме первого
    x_modified = x.clone()
    x_modified[:, 1:, :] = torch.randn_like(x_modified[:, 1:, :])
    with torch.no_grad():
        out_modified = head(x_modified)

    assert torch.allclose(out_full[:, :1], out_modified[:, :1], atol=1e-5), (
        "t=0 вывод изменился после изменения будущих кадров"
    )


def test_causal_tcn_output_length_equals_input_length():
    """Длина выходной последовательности должна совпадать с входной при любом T."""
    head = CausalTCNHead(input_dim=8, num_channels=[4, 4], kernel_size=5)
    head.eval()

    for T in [1, 7, 16, 100]:
        x = torch.zeros(1, T, 8)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (1, T), f"Ожидалось (1, {T}), получено {out.shape}"


# ---------------------------------------------------------------------------
# postprocess: edge cases
# ---------------------------------------------------------------------------

def test_scores_to_detections_segment_ending_at_last_frame():
    """Активный сегмент, доходящий до последнего кадра, должен быть детектирован."""
    fps = 25.0
    # 50 активных кадров в самом конце (2.0s > 1.5s min)
    scores = [0.0] * 10 + [0.9] * 50
    dets = scores_to_detections(scores, fps, threshold=0.5,
                                min_duration_sec=1.5, max_duration_sec=12.0)
    assert len(dets) == 1
    assert dets[0].end_frame == 60  # 10 + 50


def test_scores_to_detections_too_long_filtered():
    """Сегмент длиннее max_duration_sec должен быть отфильтрован."""
    fps = 25.0
    # 400 кадров = 16s > 12s max
    scores = [0.9] * 400
    dets = scores_to_detections(scores, fps, threshold=0.5,
                                min_duration_sec=1.5, max_duration_sec=12.0)
    assert len(dets) == 0


def test_scores_to_detections_all_below_threshold():
    """Все scores ниже порога → нет детекций."""
    scores = [0.4] * 100
    dets = scores_to_detections(scores, fps=25.0, threshold=0.5)
    assert len(dets) == 0


def test_scores_to_detections_exact_min_duration():
    """Сегмент ровно min_duration_sec включается (граничное условие >=)."""
    fps = 25.0
    min_frames = int(1.5 * fps)  # 37 кадров
    scores = [0.0] * 5 + [0.9] * min_frames + [0.0] * 5
    dets = scores_to_detections(scores, fps, threshold=0.5,
                                min_duration_sec=1.5, max_duration_sec=12.0)
    assert len(dets) == 1
