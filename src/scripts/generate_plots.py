"""
Генерация графиков для магистерской диссертации — ConElGym_RT.

Создаёт publication-quality фигуры в data/plots/results/:

  Fig 1  — mAP@0.5: Valid vs Test для всех 8 моделей (grouped bar)
  Fig 2  — Boundary Error (sorted horizontal bar, target line)
  Fig 3  — Ablation heatmap: backbone × temporal head (mAP@0.5 и BE)
  Fig 4  — FPS vs mAP scatter (accuracy-efficiency trade-off, RT zone)
  Fig 5  — Multi-seed errorbar: mean±std mAP и BE (все 8 моделей, 3 seed)
  Fig 6  — Radar chart: топ-4 модели по 5 осям
  Fig 7  — Model size vs mAP bubble chart
  Fig 8  — Summary table (appendix)
  Fig 9  — HPO before/after comparison
  Fig 10 — LOAO heatmap: 8 моделей × 4 снаряда
  Fig 11 — LOAO grouped bar: топ-3 модели × 4 снаряда
  Fig 12 — Phase 5: LOAO сравнение baseline vs framediff vs TSM (eff_b0)
  Fig 13 — Phase 5: FPS vs mAP scatter (Phase 3 + Phase 5 модели)

Запуск:
    python src/scripts/generate_plots.py

Выходные файлы: data/plots/results/fig{1..13}_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "data" / "plots" / "results"

# ---------------------------------------------------------------------------
# Данные экспериментов (seed=42, threshold=0.5, test split)
# Источники: compare_models.py output, logs/compare_seed42.log
# ---------------------------------------------------------------------------

# fmt: off
# name, map3, map5, map7, precision, recall, be, fps_k, size_mb, valid_map5
# Источник: compare_models.py, seed=42, opt-конфиг, test split, RTX 5060 Ti
# Дата последнего измерения: 2026-03-20
MODELS_SEED42 = [
    # name                      map3   map5   map7   prec   rec    be     fps_k    size_mb  valid_map5
    ("mv3_small",               0.000, 0.000, 0.000, 0.000, 0.000, None,  1908.3,  4.1,    0.091),  # base конфиг, FPS не переизмерен
    ("mv3_bilstm",              0.771, 0.771, 0.750, 0.757, 0.841, 0.307, 220.8,   6.4,    0.795),
    ("mv3_bilstm_attn",         0.762, 0.687, 0.565, 0.758, 0.794, 0.378, 260.3,   6.2,    0.701),
    ("mv3_tcn",                 0.735, 0.560, 0.295, 0.595, 0.698, 0.598, 499.4,   7.5,    0.560),
    ("eff_b0_mlp",              0.182, 0.182, 0.091, 0.667, 0.159, 0.894, 883.0,   16.5,   0.182),
    ("eff_b0_bilstm",           0.791, 0.791, 0.774, 0.771, 0.857, 0.325, 191.8,   20.3,   0.894),
    ("eff_b0_bilstm_attn",      0.853, 0.788, 0.774, 0.700, 0.889, 0.316, 7.8,     45.1,   0.895),
    ("eff_b0_tcn",              0.900, 0.900, 0.708, 0.568, 1.000, 0.507, 345.1,   22.0,   0.810),
]
# fmt: on

# Multi-seed test mAP и BE (seed=42/123/2024, opt-конфиг, все seed на opt)
MULTI_SEED = {
    "eff_b0_bilstm_attn": {"map": [0.788, 0.862, 0.839], "be": [0.320, 0.250, 0.210]},
    "eff_b0_bilstm":      {"map": [0.791, 0.867, 0.823], "be": [0.330, 0.240, 0.390]},
    "eff_b0_tcn":         {"map": [0.900, 0.714, 0.676], "be": [0.510, 0.530, 0.540]},
    "mv3_bilstm":         {"map": [0.771, 0.763, 0.769], "be": [0.310, 0.420, 0.310]},
    "mv3_bilstm_attn":    {"map": [0.687, 0.749, 0.737], "be": [0.380, 0.320, 0.350]},
    "mv3_tcn":            {"map": [0.560, 0.470, 0.439], "be": [0.600, 0.520, 0.600]},
    "eff_b0_mlp":         {"map": [0.182, 0.149, 0.146], "be": [0.890, 0.910, 0.890]},
    "mv3_small":          {"map": [0.000, 0.091, 0.045], "be": [None,  0.780, 0.870]},
}

# LOAO results (seed=42, mAP@0.5 per apparatus)
LOAO = {
    "eff_b0_bilstm_attn": {"Ball": 0.720, "Clubs": 0.797, "Hoop": 0.636, "Ribbon": 0.800},
    "eff_b0_bilstm":      {"Ball": 0.707, "Clubs": 0.818, "Hoop": 0.628, "Ribbon": 0.536},
    "mv3_bilstm_attn":    {"Ball": 0.776, "Clubs": 0.470, "Hoop": 0.697, "Ribbon": 0.447},
    "eff_b0_tcn":         {"Ball": 0.303, "Clubs": 0.818, "Hoop": 0.527, "Ribbon": 0.713},
    "mv3_bilstm":         {"Ball": 0.588, "Clubs": 0.628, "Hoop": 0.588, "Ribbon": 0.361},
    "mv3_tcn":            {"Ball": 0.358, "Clubs": 0.535, "Hoop": 0.345, "Ribbon": 0.247},
    "eff_b0_mlp":         {"Ball": 0.000, "Clubs": 0.182, "Hoop": 0.000, "Ribbon": 0.182},
    "mv3_small":          {"Ball": 0.000, "Clubs": 0.000, "Hoop": 0.045, "Ribbon": 0.011},
}

TARGET_MAP = 0.75
TARGET_BE  = 2.0   # PRD требование для RT (в секундах)
TARGET_FPS = 25

# ---------------------------------------------------------------------------
# Phase 5 данные — motion-aware backbones (eff_b0 + bilstm_attn, seed=42)
# Источник: compare_models.py, test split, RTX 5060 Ti
# Дата измерения: 2026-03-21
# ---------------------------------------------------------------------------

# name, map5_test, be_test, recall_test, fps, size_mb, valid_map5
PHASE5_MODELS = [
    # name                             map5   be     recall fps_k   size_mb  valid_map5
    ("eff_b0_bilstm_attn",             0.788, 0.316, 0.889, 7.8,    45.1,   0.895),  # Phase 3 baseline
    ("eff_b0_framediff_bilstm_attn",   0.777, 0.260, 0.889, 9.0,    45.2,   0.894),  # Phase 5 framediff
    ("eff_b0_tsm_bilstm_attn",         0.886, 0.250, 0.921, 13.1,   45.1,   0.896),  # Phase 5 TSM
]

PHASE5_LOAO = {
    "eff_b0_bilstm_attn":           {"Ball": 0.785, "Clubs": 0.775, "Hoop": 0.636, "Ribbon": 0.800},  # Phase 3 baseline (LOAO seed42)
    "eff_b0_framediff_bilstm_attn": {"Ball": 0.995, "Clubs": 0.966, "Hoop": 0.740, "Ribbon": 0.429},  # Phase 5 framediff
    "eff_b0_tsm_bilstm_attn":       {"Ball": 1.000, "Clubs": 0.790, "Hoop": 0.903, "Ribbon": 0.621},  # Phase 5 TSM
}

PHASE5_DISPLAY = {
    "eff_b0_bilstm_attn":           "EffB0+BiLSTM+Att\n(Phase 3 baseline)",
    "eff_b0_framediff_bilstm_attn": "EffB0+FrameDiff\n+BiLSTM+Att",
    "eff_b0_tsm_bilstm_attn":       "EffB0+TSM\n+BiLSTM+Att",
}

PHASE5_PALETTE = {
    "eff_b0_bilstm_attn":           "#E53935",   # красный — лучшая Phase 3
    "eff_b0_framediff_bilstm_attn": "#1565C0",   # синий
    "eff_b0_tsm_bilstm_attn":       "#2E7D32",   # зелёный
}

PALETTE = {
    "mv3_small":        "#BDBDBD",
    "mv3_bilstm":       "#80CBC4",
    "mv3_bilstm_attn":  "#26A69A",
    "mv3_tcn":          "#00796B",
    "eff_b0_mlp":       "#FFCC80",
    "eff_b0_bilstm":    "#FFA726",
    "eff_b0_bilstm_attn": "#E53935",  # лучшая модель — красный
    "eff_b0_tcn":       "#FF7043",
}

DISPLAY = {
    "mv3_small":          "MV3+MLP",
    "mv3_bilstm":         "MV3+BiLSTM",
    "mv3_bilstm_attn":    "MV3+BiLSTM+Att",
    "mv3_tcn":            "MV3+TCN",
    "eff_b0_mlp":         "EffB0+MLP",
    "eff_b0_bilstm":      "EffB0+BiLSTM",
    "eff_b0_bilstm_attn": "EffB0+BiLSTM+Att",
    "eff_b0_tcn":         "EffB0+TCN",
}


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path)
    print(f"  Saved: {path.relative_to(ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 — mAP@0.5 grouped bar: Valid vs Test, все 8 моделей
# ---------------------------------------------------------------------------

def fig1_map_comparison() -> None:
    names      = [DISPLAY[m[0]] for m in MODELS_SEED42]
    colors     = [PALETTE[m[0]] for m in MODELS_SEED42]
    valid_map  = [m[9] for m in MODELS_SEED42]
    test_map   = [m[2] for m in MODELS_SEED42]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - width / 2, valid_map, width, label="Valid",
           color=colors, alpha=0.5, edgecolor="white")
    ax.bar(x + width / 2, test_map, width, label="Test",
           color=colors, alpha=0.90, edgecolor="white")

    ax.axhline(TARGET_MAP, color="red", linestyle="--", linewidth=1.2,
               label=f"PRD target (≥{TARGET_MAP})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("mAP@IoU=0.5")
    ax.set_ylim(0, 1.1)
    ax.set_title("Figure 1. mAP@0.5: Valid vs Test — All 8 Models\n"
                 "(seed=42, opt config, threshold=0.5)")

    model_handles = [mpatches.Patch(color=PALETTE[m[0]], label=DISPLAY[m[0]])
                     for m in MODELS_SEED42]
    metric_handles = [
        mpatches.Patch(color="gray", alpha=0.5, label="Valid"),
        mpatches.Patch(color="gray", alpha=0.9, label="Test"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
               label=f"Target ≥ {TARGET_MAP}"),
    ]
    leg1 = ax.legend(handles=metric_handles, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="upper right", ncol=2, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig1_map_comparison.png")


# ---------------------------------------------------------------------------
# Fig 2 — Boundary Error horizontal bar
# ---------------------------------------------------------------------------

def fig2_boundary_error() -> None:
    data = [(DISPLAY[m[0]], m[6], PALETTE[m[0]])
            for m in MODELS_SEED42 if m[6] is not None]
    data.sort(key=lambda x: x[1])

    names, be_vals, colors = zip(*data)
    bar_colors = ["#4CAF50" if v <= TARGET_BE else "#F44336" for v in be_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, be_vals, color=bar_colors, edgecolor="none", height=0.6)
    ax.axvline(TARGET_BE, color="red", linestyle="--", linewidth=1.5,
               label=f"PRD target: BE ≤ {TARGET_BE}s")
    ax.axvline(1.0, color="orange", linestyle=":", linewidth=1.2,
               label="Strict target: BE ≤ 1.0s")

    for bar, val in zip(bars, be_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", fontsize=9.5)

    ax.set_xlabel("Boundary Error (seconds)")
    ax.set_title("Figure 2. Boundary Error — All Models\n"
                 "(seed=42, test split; green = passes PRD target ≤2.0s)")
    good = mpatches.Patch(color="#4CAF50", label=f"Passes PRD (BE ≤ {TARGET_BE}s)")
    bad  = mpatches.Patch(color="#F44336", label=f"Fails PRD (BE > {TARGET_BE}s)")
    ax.legend(handles=[good, bad,
                        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5,
                               label=f"PRD = {TARGET_BE}s"),
                        Line2D([0], [0], color="orange", linestyle=":", linewidth=1.2,
                               label="Strict = 1.0s")],
              loc="lower right")
    ax.set_xlim(0, max(be_vals) * 1.15)
    fig.tight_layout()
    _save(fig, "fig2_boundary_error.png")


# ---------------------------------------------------------------------------
# Fig 3 — Ablation heatmap: backbone × temporal head
# ---------------------------------------------------------------------------

def fig3_ablation_heatmap() -> None:
    # Rows: MV3-Small, EffB0 | Cols: MLP, BiLSTM, BiLSTM+Att, TCN
    map_grid = np.array([
        [0.000, 0.672, 0.644, 0.552],   # MV3
        [0.091, 0.791, 0.788, 0.900],   # EffB0
    ])
    be_grid = np.array([
        [np.nan, 0.394, 0.359, 0.609],
        [0.943,  0.330, 0.320, 0.510],
    ])
    row_labels = ["MobileNetV3-Small", "EfficientNet-B0"]
    col_labels = ["MLP\n(baseline)", "BiLSTM", "BiLSTM\n+Attention", "CausalTCN"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # mAP@0.5
    im1 = ax1.imshow(map_grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im1, ax=ax1, label="mAP@IoU=0.5")
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(2))
    ax1.set_xticklabels(col_labels, fontsize=10)
    ax1.set_yticklabels(row_labels, fontsize=11)
    ax1.set_title("mAP@IoU=0.5 (test, seed=42)", fontweight="bold")
    for i in range(2):
        for j in range(4):
            val = map_grid[i, j]
            color = "white" if val > 0.75 else "black"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)
    ax1.grid(False)

    # Boundary Error
    im2 = ax2.imshow(be_grid, cmap="RdYlGn_r", vmin=0.2, vmax=1.1, aspect="auto")
    plt.colorbar(im2, ax=ax2, label="Boundary Error (s)")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(2))
    ax2.set_xticklabels(col_labels, fontsize=10)
    ax2.set_yticklabels(row_labels, fontsize=11)
    ax2.set_title("Boundary Error (s) — lower is better", fontweight="bold")
    for i in range(2):
        for j in range(4):
            val = be_grid[i, j]
            if np.isnan(val):
                ax2.text(j, i, "n/a", ha="center", va="center", fontsize=11)
            else:
                color = "white" if val < 0.40 else "black"
                ax2.text(j, i, f"{val:.3f}s", ha="center", va="center",
                         fontsize=12, fontweight="bold", color=color)
    ax2.grid(False)

    fig.suptitle(
        "Figure 3. Ablation Study: Backbone × Temporal Head\n"
        "(seed=42, test split; green = better)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_ablation_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 4 — FPS vs mAP scatter (accuracy-efficiency trade-off)
# ---------------------------------------------------------------------------

def fig4_fps_vs_map() -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    # RT boundary zone
    ax.axvline(TARGET_FPS, color="green", linestyle="--", linewidth=1.5,
               label=f"RT boundary (FPS={TARGET_FPS})", zorder=2)
    ax.axhline(TARGET_MAP, color="red", linestyle="--", linewidth=1.2,
               label=f"PRD mAP target ({TARGET_MAP})", zorder=2)
    ax.fill_betweenx([TARGET_MAP, 1.05], TARGET_FPS, 2e6,
                     alpha=0.06, color="green", label="RT + PRD zone")

    for m in MODELS_SEED42:
        name, _, map5, _, _, _, be, fps_k, size_mb, _ = m
        fps = fps_k * 1000
        color = PALETTE[name]
        size = max(size_mb * 8, 40)
        ax.scatter(fps, map5, s=size, color=color, alpha=0.85,
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(DISPLAY[name], (fps, map5),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color="black")

    ax.set_xscale("log")
    ax.set_xlabel("FPS (inference, RTX 5060 Ti, log scale)")
    ax.set_ylabel("mAP@IoU=0.5 (test, seed=42)")
    ax.set_title("Figure 4. Accuracy–Efficiency Trade-off\n"
                 "(bubble size ∝ model size in MB; green zone = passes RT + PRD requirements)")
    ax.set_ylim(-0.02, 1.1)

    size_legend = [
        ax.scatter([], [], s=4.1 * 8, color="gray", alpha=0.6,
                   edgecolors="black", linewidths=0.5, label="4 MB"),
        ax.scatter([], [], s=22 * 8, color="gray", alpha=0.6,
                   edgecolors="black", linewidths=0.5, label="22 MB"),
        ax.scatter([], [], s=88 * 8, color="gray", alpha=0.6,
                   edgecolors="black", linewidths=0.5, label="88 MB"),
    ]
    leg1 = ax.legend(handles=size_legend, title="Model size", loc="lower right",
                     framealpha=0.9, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    _save(fig, "fig4_fps_vs_map.png")


# ---------------------------------------------------------------------------
# Fig 5 — Multi-seed errorbar
# ---------------------------------------------------------------------------

def fig5_multiseed() -> None:
    model_keys = [
        "eff_b0_bilstm_attn", "eff_b0_bilstm", "eff_b0_tcn",
        "mv3_bilstm", "mv3_bilstm_attn", "mv3_tcn",
    ]
    display_names = [DISPLAY[k] for k in model_keys]
    colors = [PALETTE[k] for k in model_keys]

    map_means, map_stds, be_means, be_stds = [], [], [], []
    for k in model_keys:
        maps = MULTI_SEED[k]["map"]
        bes  = [v for v in MULTI_SEED[k]["be"] if v is not None]
        map_means.append(np.mean(maps))
        map_stds.append(np.std(maps))
        be_means.append(np.mean(bes))
        be_stds.append(np.std(bes))

    x = np.arange(len(model_keys))
    fig, (ax_map, ax_be) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, means, stds, ylabel, title, ylim, target, target_label in [
        (ax_map, map_means, map_stds, "mAP@IoU=0.5",
         "mAP@0.5: Mean ± Std (3 seeds)", (0.35, 1.08),
         TARGET_MAP, f"Target ≥ {TARGET_MAP}"),
        (ax_be, be_means, be_stds, "Boundary Error (s)",
         "Boundary Error: Mean ± Std (3 seeds)", (0.0, 1.2),
         TARGET_BE, f"PRD target ≤ {TARGET_BE}s"),
    ]:
        ax.bar(x, means, 0.55, color=colors, alpha=0.75, edgecolor="white")
        ax.errorbar(x, means, yerr=stds,
                    fmt="none", color="black", capsize=8, linewidth=2)

        # Individual seed points
        for i, key in enumerate(model_keys):
            seed_vals = [v for v in (MULTI_SEED[key]["map"] if ax is ax_map
                                     else MULTI_SEED[key]["be"]) if v is not None]
            jitter = np.linspace(-0.13, 0.13, len(seed_vals))
            for sv, jit in zip(seed_vals, jitter):
                ax.scatter(i + jit, sv, zorder=5, color=colors[i],
                           edgecolors="black", s=55, linewidths=0.8)

        ax.axhline(target, color="red", linestyle="--", linewidth=1.2,
                   label=target_label)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, ylim[0] + 0.02, f"{mean:.3f}\n±{std:.3f}",
                    ha="center", fontsize=8.5, color=colors[i], fontweight="bold")

    fig.suptitle(
        "Figure 5. Multi-Seed Stability: Top-6 Models (seeds 42, 123, 2024)\n"
        "Bars = mean, whiskers = population std, dots = individual runs",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "fig5_multiseed_errorbar.png")


# ---------------------------------------------------------------------------
# Fig 6 — Radar chart: top-4 models
# ---------------------------------------------------------------------------

def fig6_radar() -> None:
    def be_score(be: float | None) -> float:
        if be is None:
            return 0.0
        return max(0.0, 1.0 - be / 2.0)

    def fps_score(fps_k: float) -> float:
        return min(1.0, fps_k / 1000.0)

    def loao_score(name: str) -> float:
        vals = list(LOAO[name].values())
        return np.mean(vals)

    axes_labels = ["mAP@0.5", "Recall", "Boundary\nEfficiency", "FPS\nEfficiency", "LOAO\nmean"]

    top4 = ["eff_b0_bilstm_attn", "eff_b0_bilstm", "eff_b0_tcn", "mv3_bilstm"]
    radar_colors = [PALETTE[k] for k in top4]

    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for key, color in zip(top4, radar_colors):
        m = next(m for m in MODELS_SEED42 if m[0] == key)
        values = [
            m[2],               # mAP@0.5
            m[5],               # Recall
            be_score(m[6]),     # Boundary Efficiency
            fps_score(m[7]),    # FPS Efficiency
            loao_score(key),    # LOAO mean
        ]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=DISPLAY[key])
        ax.fill(angles, values, color=color, alpha=0.10)

    # Target ring
    target_vals = [TARGET_MAP, 0.80, be_score(1.0), fps_score(25), 0.70]
    target_vals += target_vals[:1]
    ax.plot(angles, target_vals, "r--", linewidth=1.2, label="PRD minimum")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8)
    ax.set_title("Figure 6. Radar Chart: Top-4 Models\n"
                 "(seed=42; Boundary Eff = 1−BE/2s; FPS Eff = min(FPS/1000,1))",
                 pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.15))

    fig.tight_layout()
    _save(fig, "fig6_radar_chart.png")


# ---------------------------------------------------------------------------
# Fig 7 — Model size vs mAP scatter (bubble = 1/BE)
# ---------------------------------------------------------------------------

def fig7_size_vs_map() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for m in MODELS_SEED42:
        name, _, map5, _, _, _, be, fps_k, size_mb, _ = m
        bubble = (1.0 / be * 400) if be and be > 0 else 30
        ax.scatter(size_mb, map5, s=bubble, color=PALETTE[name],
                   alpha=0.80, edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(DISPLAY[name], (size_mb, map5),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.axhline(TARGET_MAP, color="red", linestyle="--", linewidth=1.2,
               label=f"PRD target mAP ≥ {TARGET_MAP}")
    ax.axvline(50, color="orange", linestyle=":", linewidth=1.2,
               label="PRD model size ≤ 50 MB")
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("mAP@IoU=0.5 (test, seed=42)")
    ax.set_title("Figure 7. Model Size vs Performance\n"
                 "(bubble size ∝ 1/Boundary_Error; larger = more accurate boundaries)")

    for be_val, label in [(0.3, "BE=0.3s"), (0.5, "BE=0.5s"), (1.0, "BE=1.0s")]:
        ax.scatter([], [], s=1.0 / be_val * 400, color="gray",
                   alpha=0.5, label=label, edgecolors="black", linewidths=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig7_size_vs_map.png")


# ---------------------------------------------------------------------------
# Fig 8 — Summary table
# ---------------------------------------------------------------------------

def fig8_summary_table() -> None:
    col_labels = ["Model", "mAP@0.5", "mAP@0.7", "Precision",
                  "Recall", "BE (s)", "FPS (K)", "Size MB", "PRD"]
    rows = []
    for m in MODELS_SEED42:
        name, map3, map5, map7, prec, rec, be, fps_k, size_mb, _ = m
        be_str = f"{be:.3f}" if be is not None else "n/a"
        prd_ok = (map5 >= TARGET_MAP and rec >= 0.80
                  and (be is None or be <= TARGET_BE) and fps_k * 1000 >= TARGET_FPS)
        rows.append([
            DISPLAY[name],
            f"{map5:.3f}",
            f"{map7:.3f}",
            f"{prec:.3f}",
            f"{rec:.3f}",
            be_str,
            f"{fps_k:.0f}K",
            f"{size_mb:.1f}",
            "OK" if prd_ok else "FAIL",
        ])

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#37474F")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    best_name = "eff_b0_bilstm_attn"
    for i, m in enumerate(MODELS_SEED42):
        row_idx = i + 1
        if m[0] == best_name:
            bg = "#FFEBEE"
        elif m[2] >= TARGET_MAP:
            bg = "#F1F8E9"
        else:
            bg = "#FFFFFF"
        for j in range(len(col_labels)):
            tbl[(row_idx, j)].set_facecolor(bg)
        prd_cell = tbl[(row_idx, len(col_labels) - 1)]
        prd_ok = rows[i][-1] == "OK"
        prd_cell.set_text_props(color="#2E7D32" if prd_ok else "#C62828",
                                fontweight="bold")

    ax.set_title(
        "Figure 8 / Table 1. Full Results Summary (seed=42, threshold=0.5, test split)\n"
        "Red row = best model (EffB0+BiLSTM+Att); Green rows = mAP ≥ 0.75",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    _save(fig, "fig8_summary_table.png")


# ---------------------------------------------------------------------------
# Fig 9 — HPO before vs after
# ---------------------------------------------------------------------------

def fig9_hpo_comparison() -> None:
    # valid mAP@0.5: base (Phase 2) vs opt (Phase 3 best HPO trial)
    models_hpo = [
        ("eff_b0_bilstm_attn", 0.800, 0.895),
        ("eff_b0_bilstm",      0.813, 0.894),
        ("eff_b0_tcn",         0.552, 0.810),
        ("mv3_bilstm",         0.692, 0.795),
        ("mv3_bilstm_attn",    0.541, 0.701),
        ("mv3_tcn",            0.554, 0.560),
    ]
    names = [DISPLAY[m[0]] for m in models_hpo]
    pre   = [m[1] for m in models_hpo]
    post  = [m[2] for m in models_hpo]
    colors = [PALETTE[m[0]] for m in models_hpo]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, pre,  width, label="Before HPO (Phase 2)", color=colors,
           alpha=0.45, edgecolor="white")
    ax.bar(x + width / 2, post, width, label="After HPO (Optuna best trial)", color=colors,
           alpha=0.90, edgecolor="white")

    # Delta arrows
    for i, (p, o) in enumerate(zip(pre, post)):
        delta = o - p
        if abs(delta) > 0.01:
            ax.annotate(f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                        xy=(i + width / 2, o), xytext=(i + width / 2, o + 0.025),
                        ha="center", fontsize=8.5,
                        color="#2E7D32" if delta >= 0 else "#C62828",
                        fontweight="bold")

    ax.axhline(TARGET_MAP, color="red", linestyle="--", linewidth=1.2,
               label=f"PRD target ≥ {TARGET_MAP}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Valid mAP@IoU=0.5")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Figure 9. HPO Effect: Valid mAP Before vs After Hyperparameter Optimization\n"
                 "(Optuna TPE, 20–42 trials per model; annotation = Δ mAP)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig9_hpo_comparison.png")


# ---------------------------------------------------------------------------
# Fig 10 — LOAO heatmap: 8 models × 4 apparatus
# ---------------------------------------------------------------------------

def fig10_loao_heatmap() -> None:
    apparatus = ["Ball", "Clubs", "Hoop", "Ribbon"]
    model_keys = [
        "eff_b0_bilstm_attn", "eff_b0_bilstm", "mv3_bilstm_attn",
        "eff_b0_tcn", "mv3_bilstm", "mv3_tcn", "eff_b0_mlp", "mv3_small",
    ]
    display_names = [DISPLAY[k] for k in model_keys]

    grid = np.array([[LOAO[k][a] for a in apparatus] for k in model_keys])

    # Добавляем столбец mean
    means = grid.mean(axis=1, keepdims=True)
    full_grid = np.hstack([grid, means])
    col_labels = apparatus + ["Mean"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(full_grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="mAP@IoU=0.5")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(model_keys)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("Apparatus (left-out fold)")

    for i in range(len(model_keys)):
        for j in range(len(col_labels)):
            val = full_grid[i, j]
            color = "white" if val > 0.75 else ("black" if val > 0.30 else "white")
            marker = ""
            if j < 4 and val >= 0.70:
                marker = " ✓"
            elif j < 4 and val < 0.70:
                marker = " ✗"
            ax.text(j, i, f"{val:.2f}{marker}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    # Target threshold line
    ax.axhline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(3.5, color="white", linestyle="-", linewidth=1.5, alpha=0.7)

    ax.set_title("Figure 10. LOAO Cross-Validation Heatmap\n"
                 "(✓ = passes ≥0.70; red = fails; last column = mean; "
                 "S3D v2 reference: 0.879 mean, all ✓)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig10_loao_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 11 — LOAO grouped bar: top-3 models
# ---------------------------------------------------------------------------

def fig11_loao_bar() -> None:
    apparatus = ["Ball", "Clubs", "Hoop", "Ribbon"]
    top3 = ["eff_b0_bilstm_attn", "eff_b0_bilstm", "eff_b0_tcn"]
    colors_top3 = [PALETTE[k] for k in top3]
    labels_top3 = [DISPLAY[k] for k in top3]

    n_app = len(apparatus)
    n_models = len(top3)
    width = 0.25
    x = np.arange(n_app)
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (key, color, label) in enumerate(zip(top3, colors_top3, labels_top3)):
        vals = [LOAO[key][a] for a in apparatus]
        bars = ax.bar(x + offsets[i], vals, width, color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.8, label=label)
        for bar, val in zip(bars, vals):
            check = " ✓" if val >= 0.70 else " ✗"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.015,
                    f"{val:.3f}{check}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.axhline(TARGET_MAP, color="red", linestyle="--", linewidth=1.4,
               label=f"Target ≥ {TARGET_MAP}", zorder=3)
    ax.axhline(0.879, color="#5C6BC0", linestyle=":", linewidth=1.2,
               label="S3D v2 LOAO mean (0.879)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(apparatus, fontsize=12)
    ax.set_xlabel("Apparatus type (left-out fold)")
    ax.set_ylabel("mAP@IoU=0.5")
    ax.set_ylim(0.0, 1.12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Figure 11. LOAO Cross-Validation: Top-3 2D Models vs S3D Reference\n"
                 "(✓ = passes ≥0.70 criterion; dotted line = S3D v2 LOAO mean)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig11_loao_grouped_bar.png")


# ---------------------------------------------------------------------------
# Fig 12 — Phase 5: LOAO сравнение baseline vs framediff vs TSM
# ---------------------------------------------------------------------------

def fig12_phase5_loao() -> None:
    apparatus = ["Ball", "Clubs", "Hoop", "Ribbon"]
    model_keys = list(PHASE5_LOAO.keys())
    colors = [PHASE5_PALETTE[k] for k in model_keys]
    labels = [PHASE5_DISPLAY[k] for k in model_keys]

    n_app = len(apparatus)
    n_models = len(model_keys)
    width = 0.25
    x = np.arange(n_app)
    offsets = np.linspace(-(n_models - 1) * width / 2,
                          (n_models - 1) * width / 2, n_models)

    fig, (ax_bar, ax_mean) = plt.subplots(1, 2, figsize=(15, 6),
                                           gridspec_kw={"width_ratios": [3, 1]})

    # --- Left: grouped bar per apparatus ---
    for i, (key, color, label) in enumerate(zip(model_keys, colors, labels)):
        vals = [PHASE5_LOAO[key][a] for a in apparatus]
        bars = ax_bar.bar(x + offsets[i], vals, width, color=color,
                          alpha=0.85, edgecolor="white", linewidth=0.8, label=label)
        for bar, val in zip(bars, vals):
            check = "✓" if val >= 0.70 else "✗"
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.015, f"{val:.3f}\n{check}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_bar.axhline(0.70, color="red", linestyle="--", linewidth=1.4,
                   label="Criterion ≥ 0.70", zorder=3)
    ax_bar.axhline(0.879, color="#5C6BC0", linestyle=":", linewidth=1.2,
                   label="S3D v2 LOAO mean (0.879)", zorder=3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(apparatus, fontsize=12)
    ax_bar.set_xlabel("Apparatus (left-out fold)")
    ax_bar.set_ylabel("mAP@IoU=0.5")
    ax_bar.set_ylim(0.0, 1.15)
    ax_bar.legend(fontsize=9, loc="lower right")
    ax_bar.set_title("LOAO per Apparatus", fontweight="bold")

    # --- Right: mean bar ---
    means = [np.mean(list(PHASE5_LOAO[k].values())) for k in model_keys]
    short_labels = ["Baseline\n(Ph3)", "FrameDiff", "TSM"]
    bar_h = ax_mean.bar(short_labels, means, color=colors, alpha=0.85,
                        edgecolor="white", width=0.5)
    for bar, val in zip(bar_h, means):
        ax_mean.text(bar.get_x() + bar.get_width() / 2,
                     val + 0.015, f"{val:.3f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_mean.axhline(0.70, color="red", linestyle="--", linewidth=1.4)
    ax_mean.axhline(0.879, color="#5C6BC0", linestyle=":", linewidth=1.2)
    ax_mean.set_ylim(0.0, 1.15)
    ax_mean.set_ylabel("LOAO Mean mAP@0.5")
    ax_mean.set_title("LOAO Mean", fontweight="bold")

    fig.suptitle(
        "Figure 12. Phase 5: LOAO Cross-Validation — EfficientNet-B0 + BiLSTM+Attention\n"
        "Baseline (Phase 3) vs Frame Difference vs TSM  |  seed=42  |  ✓ = ≥0.70",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig12_phase5_loao.png")


# ---------------------------------------------------------------------------
# Fig 13 — Phase 5: FPS vs mAP scatter (Phase 3 + Phase 5)
# ---------------------------------------------------------------------------

def fig13_phase5_fps_map() -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axvline(TARGET_FPS, color="green", linestyle="--", linewidth=1.5,
               label=f"RT boundary (FPS={TARGET_FPS})", zorder=2)
    ax.axhline(TARGET_MAP, color="red", linestyle="--", linewidth=1.2,
               label=f"PRD mAP target ({TARGET_MAP})", zorder=2)
    ax.fill_betweenx([TARGET_MAP, 1.05], TARGET_FPS, 2e6,
                     alpha=0.06, color="green", label="RT + PRD zone")

    # Phase 3 models (полупрозрачные, серые)
    for m in MODELS_SEED42:
        name, _, map5, _, _, _, be, fps_k, size_mb, _ = m
        fps = fps_k * 1000
        size = max(size_mb * 6, 30)
        ax.scatter(fps, map5, s=size, color=PALETTE[name], alpha=0.35,
                   edgecolors="gray", linewidths=0.6, zorder=2)
        ax.annotate(DISPLAY[name], (fps, map5),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=8, color="gray", alpha=0.7)

    # Phase 5 models (яркие, с outline)
    for name, map5, be, recall, fps_k, size_mb, _ in PHASE5_MODELS:
        fps = fps_k * 1000
        size = max(size_mb * 6, 60)
        color = PHASE5_PALETTE[name]
        ax.scatter(fps, map5, s=size, color=color, alpha=0.95,
                   edgecolors="black", linewidths=1.5, zorder=4, marker="D")
        ax.annotate(PHASE5_DISPLAY[name].replace("\n", " "), (fps, map5),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("FPS (inference, RTX 5060 Ti, log scale)")
    ax.set_ylabel("mAP@IoU=0.5 (test, seed=42)")
    ax.set_title(
        "Figure 13. Phase 5: Accuracy–Efficiency Trade-off\n"
        "Circles = Phase 3 models  |  Diamonds = Phase 5 (framediff, TSM)  |  bubble ∝ model size",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.1)

    p3_patch = mpatches.Patch(color="gray", alpha=0.5, label="Phase 3 models")
    p5_scatter = ax.scatter([], [], s=80, color="black", marker="D",
                             edgecolors="black", label="Phase 5 models")
    p5_handles = [
        mpatches.Patch(color=PHASE5_PALETTE[k], label=PHASE5_DISPLAY[k].replace("\n", " "))
        for k in PHASE5_LOAO.keys()
    ]
    leg1 = ax.legend(handles=[p3_patch, p5_scatter], loc="lower right", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=p5_handles, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig13_phase5_fps_map.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating publication figures for ConElGym_RT thesis...")
    print(f"Output: {PLOTS_DIR.relative_to(ROOT)}\n")

    fig1_map_comparison()
    fig2_boundary_error()
    fig3_ablation_heatmap()
    fig4_fps_vs_map()
    fig5_multiseed()
    fig6_radar()
    fig7_size_vs_map()
    fig8_summary_table()
    fig9_hpo_comparison()
    fig10_loao_heatmap()
    fig11_loao_bar()
    fig12_phase5_loao()
    fig13_phase5_fps_map()

    print(f"\nDone. All figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
