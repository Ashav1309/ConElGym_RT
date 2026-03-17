"""
Temporal heads для ConElGym_RT.

Все heads принимают последовательность признаков [B, T, D] и возвращают
логиты [B, T] (до sigmoid).

Для стриминг-инференса каждый head реализует forward_step(x, state) → (logit, state),
где state — hidden state между кадрами.

Реализованные heads:
  PerFrameMLP        — per-frame, без контекста (baseline)
  BiLSTMHead         — двунаправленный LSTM (обучение), однонаправленный (стриминг)
  BiLSTMAttentionHead — BiLSTM + dot-product self-attention
  CausalTCNHead      — причинная TCN (только левостороннее паддинг)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# PerFrameMLP
# ---------------------------------------------------------------------------

class PerFrameMLP(nn.Module):
    """Per-frame baseline без temporal context.

    forward:      [B, T, D] → [B, T]
    forward_step: [B, D], None → (logit [B], None)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, T, D] → logits [B, T]"""
        return self.net(x).squeeze(-1)

    def forward_step(self, x: Tensor, state: Any) -> tuple[Tensor, None]:
        """Args: x [B, D] → (logit [B], None)"""
        return self.net(x).squeeze(-1), None

    def init_state(self, batch_size: int, device: torch.device) -> None:
        return None


# ---------------------------------------------------------------------------
# BiLSTMHead
# ---------------------------------------------------------------------------

class BiLSTMHead(nn.Module):
    """Двунаправленный LSTM.

    При обучении: forward() использует полный BiLSTM (двунаправленный).
    При стриминге: forward_step() использует только forward-направление (причинный).

    State для стриминга: (h, c) для forward-LSTM, shape [n_layers, B, hidden_dim].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.norm = nn.LayerNorm(input_dim)
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Для стриминга — однонаправленный LSTM с теми же весами (forward-часть)
        self.fwd_lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # BiLSTM → hidden_dim*2
        self.fc_fwd = nn.Linear(hidden_dim, 1)  # forward-only (стриминг)

    def _copy_fwd_weights(self) -> None:
        """Копирует forward-веса из bilstm в fwd_lstm (вызывать перед стримингом)."""
        with torch.no_grad():
            for layer in range(self.n_layers):
                for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                    src = getattr(self.bilstm, f"{name}_l{layer}")
                    dst = getattr(self.fwd_lstm, f"{name}_l{layer}")
                    dst.copy_(src)
            # fc_fwd ← первые hidden_dim строк fc (соответствуют forward)
            self.fc_fwd.weight.copy_(self.fc.weight[:, :self.hidden_dim])
            self.fc_fwd.bias.copy_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, T, D] → logits [B, T]"""
        x = self.norm(x)
        out, _ = self.bilstm(x)       # [B, T, hidden*2]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)

    def forward_step(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Single-step inference (причинный).

        Args:
            x:     [B, D]
            state: (h, c) или None
        Returns:
            logit [B], new_state
        """
        x = self.norm(x).unsqueeze(1)   # [B, 1, D]
        out, new_state = self.fwd_lstm(x, state)  # [B, 1, hidden]
        logit = self.fc_fwd(out.squeeze(1)).squeeze(-1)  # [B]
        return logit, new_state

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return h, c


# ---------------------------------------------------------------------------
# AttentionLayer
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    """Dot-product self-attention (single head)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, T, D] → [B, T, D]"""
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, T, T]
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V)


# ---------------------------------------------------------------------------
# BiLSTMAttentionHead
# ---------------------------------------------------------------------------

class BiLSTMAttentionHead(nn.Module):
    """BiLSTM + dot-product self-attention.

    При стриминге: attention по буферу последних attn_window кадров.
    State: (h, c, buffer) где buffer [B, attn_window, hidden*2].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        attn_window: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attn_window = attn_window

        self.norm = nn.LayerNorm(input_dim)
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fwd_lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_dim * 2)
        self.attn_fwd = AttentionLayer(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.fc_fwd = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, T, D] → logits [B, T]"""
        x = self.norm(x)
        lstm_out, _ = self.bilstm(x)          # [B, T, hidden*2]
        attn_out = self.attention(lstm_out)    # [B, T, hidden*2]
        out = self.dropout(lstm_out + attn_out)
        return self.fc(out).squeeze(-1)

    def forward_step(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Single-step с attention по скользящему буферу.

        State: (h, c, buffer [B, attn_window, hidden])
        """
        B = x.shape[0]
        device = x.device

        if state is None:
            h = torch.zeros(self.n_layers, B, self.hidden_dim, device=device)
            c = torch.zeros(self.n_layers, B, self.hidden_dim, device=device)
            buf = torch.zeros(B, self.attn_window, self.hidden_dim, device=device)
        else:
            h, c, buf = state

        x_n = self.norm(x).unsqueeze(1)              # [B, 1, D]
        lstm_out, (h, c) = self.fwd_lstm(x_n, (h, c))  # [B, 1, hidden]
        cur = lstm_out.squeeze(1)                    # [B, hidden]

        # Обновляем скользящий буфер (FIFO)
        buf = torch.cat([buf[:, 1:, :], cur.unsqueeze(1)], dim=1)  # [B, W, hidden]

        # Attention по буферу
        attn_out = self.attn_fwd(buf)                # [B, W, hidden]
        out = (cur + attn_out[:, -1, :])             # residual с последним шагом
        logit = self.fc_fwd(out).squeeze(-1)

        return logit, (h, c, buf)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        buf = torch.zeros(batch_size, self.attn_window, self.hidden_dim, device=device)
        return h, c, buf


# ---------------------------------------------------------------------------
# CausalTCNHead
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Свёртка с причинным (левосторонним) паддингом."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.pad, dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, C, T] → [B, out_ch, T]"""
        return self.conv(x)[:, :, :x.shape[2]]


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, C, T]"""
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x).transpose(1, 2)   # → [B, T, out_ch]
        out = F.relu(self.norm1(out)).transpose(1, 2)
        out = self.drop(out)
        out = self.conv2(out).transpose(1, 2)
        out = F.relu(self.norm2(out)).transpose(1, 2)
        return F.relu(out + res)


class CausalTCNHead(nn.Module):
    """Причинная TCN — только левостороннее паддинг.

    forward_step — не реализован (используется скользящий буфер в demo_live).
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [128, 128, 64]

        self.norm = nn.LayerNorm(input_dim)
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x [B, T, D] → logits [B, T]"""
        x = self.norm(x)
        x = x.transpose(1, 2)          # [B, D, T]
        x = self.tcn(x)                # [B, out_ch, T]
        x = x.transpose(1, 2)          # [B, T, out_ch]
        return self.fc(x).squeeze(-1)

    def forward_step(self, x: Tensor, state: Any) -> tuple[Tensor, Any]:
        raise NotImplementedError(
            "CausalTCN не поддерживает пошаговый стриминг. "
            "Используй скользящий буфер в demo_live.py."
        )

    def init_state(self, batch_size: int, device: torch.device) -> None:
        return None
