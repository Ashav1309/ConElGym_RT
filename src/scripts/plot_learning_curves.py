"""
Кривые обучения для всех 8 моделей ConElGym_RT.

Генерирует 3 фигуры в data/plots/results/:
  Fig 1 — 4×2 grid: train_loss + val_mAP@0.5 по эпохам (opt_seed42, все 8 моделей)
  Fig 2 — Convergence: когда достигнут best checkpoint, сколько эпох обучалось
  Fig 3 — Multi-seed: train_loss + val_mAP для топ-3 моделей (seeds 42/123/2024)

Использование:
    python src/scripts/plot_learning_curves.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,   # managed manually per axis to avoid twinx double-grid
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import mlflow

ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = ROOT / "data" / "plots" / "results"
TRACKING_URI = (ROOT / "mlruns").as_uri()

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Resolve experiment ID once at module load (robust across MLflow versions)
_exp = client.get_experiment_by_name("ConElGym_RT")
_EXPERIMENT_ID: str | None = _exp.experiment_id if _exp else None

_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9_\-\.]+$")

# ---------------------------------------------------------------------------
# Модели и цвета
# ---------------------------------------------------------------------------

MODELS = [
    ("mv3_small",          "mobilenet_v3_small",          "#9E9E9E"),
    ("mv3_bilstm",         "mobilenet_v3_bilstm",          "#66BB6A"),
    ("mv3_bilstm_attn",    "mobilenet_v3_bilstm_attn",     "#26A69A"),
    ("mv3_tcn",            "mobilenet_v3_tcn",              "#FFA726"),
    ("eff_b0_mlp",         "efficientnet_b0_mlp",           "#BDBDBD"),
    ("eff_b0_bilstm",      "efficientnet_b0_bilstm",        "#42A5F5"),
    ("eff_b0_bilstm_attn", "efficientnet_b0_bilstm_attn",   "#EF5350"),
    ("eff_b0_tcn",         "efficientnet_b0_tcn",           "#AB47BC"),
]

TOP3 = ["efficientnet_b0_bilstm", "efficientnet_b0_bilstm_attn", "efficientnet_b0_tcn"]
SEEDS = [42, 123, 2024]
SEED_COLORS = ["#1565C0", "#42A5F5", "#90CAF9"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_run(run_name: str) -> mlflow.entities.Run | None:
    """Find a FINISHED run by exact run_name in ConElGym_RT experiment."""
    if _EXPERIMENT_ID is None:
        return None
    if not _SAFE_RUN_NAME.match(run_name):
        raise ValueError(f"Unsafe run_name for filter: {run_name!r}")
    results = client.search_runs(
        experiment_ids=[_EXPERIMENT_ID],
        filter_string=f"attributes.run_name = '{run_name}' AND attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return results[0] if results else None


def _get_history(run_id: str, metric: str) -> tuple[list[int], list[float]]:
    """Return (steps, values) sorted by step."""
    history = client.get_metric_history(run_id, metric)
    if not history:
        return [], []
    pairs = sorted((m.step, m.value) for m in history)
    steps, values = zip(*pairs)
    return list(steps), list(values)


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path)
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    print(f"  Saved: {display}")
    plt.close(fig)


def _opt_run_name(model_id: str, seed: int) -> str:
    return f"{model_id}_opt_seed{seed}"


def _base_run_name(model_id: str, seed: int) -> str:
    return f"{model_id}_seed{seed}"


# ---------------------------------------------------------------------------
# Figure 1 — 4×2 grid: all 8 models, opt_seed42
# ---------------------------------------------------------------------------

def fig1_learning_curves_grid() -> list[dict]:
    """Train loss + val mAP per epoch for all 8 models (opt seed42)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    convergence_data = []

    for idx, (label, model_id, color) in enumerate(MODELS):
        ax = axes[idx]
        ax2 = ax.twinx()
        ax2.set_visible(False)  # hidden by default; enabled only if mAP data exists
        ax.grid(True, alpha=0.3)
        ax2.grid(False)

        run = _find_run(_opt_run_name(model_id, 42)) or _find_run(_base_run_name(model_id, 42))

        if run is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        steps_loss, loss = _get_history(run.info.run_id, "train_loss")
        steps_map, val_map = _get_history(run.info.run_id, "val_mAP_at_0.5")

        if not steps_loss and not steps_map:
            ax.text(0.5, 0.5, "No metrics", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        is_opt = "opt" in run.info.run_name
        suffix = " (opt)" if is_opt else " (base)"

        if steps_loss:
            ax.plot(steps_loss, loss, color=color, linewidth=1.8, label="Train loss", alpha=0.9)
            ax.set_ylabel("Train Loss", color=color)
            ax.tick_params(axis="y", labelcolor=color)

        if steps_map:
            ax2.set_visible(True)
            ax2.spines["right"].set_visible(True)
            ax2.plot(steps_map, val_map, color=color, linewidth=1.8,
                     linestyle="--", label="Val mAP@0.5", alpha=0.7)
            ax2.set_ylabel("Val mAP@0.5", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")
            ax2.set_ylim(0, 1.05)

            best_idx = int(np.argmax(val_map))
            best_epoch = steps_map[best_idx]
            best_val = val_map[best_idx]
            final_map = val_map[-1]
            ax2.axvline(best_epoch, color="green", linestyle=":", linewidth=1.2, alpha=0.8)
            ax2.scatter([best_epoch], [best_val], color="green", s=50, zorder=5)

            # total_epochs = count of logged epochs (correct regardless of step base)
            total_epochs = len(steps_map)
            convergence_data.append({
                "label": label,
                "best_epoch": best_epoch,
                "total_epochs": total_epochs,
                "best_mAP": best_val,
                "final_mAP": final_map,
                "final_loss": loss[-1] if steps_loss else None,
                "is_opt": is_opt,
            })
            ax.set_title(
                f"{label}{suffix}\nbest ep={best_epoch}, mAP={best_val:.3f} | final={final_map:.3f}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{label}{suffix}", fontsize=10)

        ax.set_xlabel("Epoch")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        ax.set_xlim(left=0)

    fig.suptitle(
        "Figure 1. Learning Curves: Train Loss & Val mAP@0.5 per Epoch (all models, opt seed=42)\n"
        "Solid = train loss (left axis) | Dashed = val mAP@0.5 (right axis) | Green dot = best checkpoint",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig1_learning_curves_grid.png")
    return convergence_data


# ---------------------------------------------------------------------------
# Figure 2 — Convergence: best checkpoint epoch + early stopping efficiency
# ---------------------------------------------------------------------------

def fig2_convergence(convergence_data: list[dict]) -> None:
    """When did each model converge? best_epoch/total_epochs bar."""
    if not convergence_data:
        print("  [skip] No convergence data")
        return

    labels = [d["label"] for d in convergence_data]
    best_epochs = [d["best_epoch"] for d in convergence_data]
    total_epochs = [d["total_epochs"] for d in convergence_data]
    best_maps = [d["best_mAP"] for d in convergence_data]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    patience_epochs = [t - b for b, t in zip(best_epochs, total_epochs)]
    ax1.bar(x, best_epochs, color="#42A5F5", label="Epochs to best ckpt")
    ax1.bar(x, patience_epochs, bottom=best_epochs, color="#EF9A9A",
            alpha=0.6, label="Early stopping patience")
    for i, (be, te) in enumerate(zip(best_epochs, total_epochs)):
        ax1.text(i, te + 0.5, f"{be}/{te}", ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("Epochs")
    ax1.set_title("Convergence: Epochs to Best Checkpoint", fontweight="bold")
    ax1.legend()

    colors = ["#4CAF50" if m >= 0.75 else "#FFA726" if m >= 0.5 else "#EF5350"
              for m in best_maps]
    ax2.bar(x, best_maps, color=colors, alpha=0.85)
    ax2.axhline(0.75, color="green", linestyle="--", linewidth=1.2, label="PRD target (0.75)")
    for i, m in enumerate(best_maps):
        ax2.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_ylabel("Best Val mAP@0.5")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Best Val mAP@0.5 per Model\n(green ≥ 0.75 PRD | orange ≥ 0.5 | red < 0.5)",
                  fontweight="bold")
    ax2.legend()

    fig.suptitle("Figure 2. Convergence Analysis (opt seed=42)", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig2_convergence.png")


# ---------------------------------------------------------------------------
# Figure 3 — Multi-seed curves for top-3 models
# ---------------------------------------------------------------------------

def fig3_multiseed_curves() -> None:
    """Train loss + val mAP across 3 seeds for top-3 models."""
    top3_labels = {
        "efficientnet_b0_bilstm":      "EfficientNet-B0 + BiLSTM",
        "efficientnet_b0_bilstm_attn": "EfficientNet-B0 + BiLSTM+Attn",
        "efficientnet_b0_tcn":         "EfficientNet-B0 + CausalTCN",
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for row, model_id in enumerate(TOP3):
        for col, (seed, seed_color) in enumerate(zip(SEEDS, SEED_COLORS)):
            ax = axes[row, col]
            ax2 = ax.twinx()
            ax2.set_visible(False)
            ax.grid(True, alpha=0.3)
            ax2.grid(False)

            run = _find_run(_opt_run_name(model_id, seed))
            if run is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{top3_labels[model_id]}\nseed={seed}")
                continue

            steps_loss, loss = _get_history(run.info.run_id, "train_loss")
            steps_map, val_map = _get_history(run.info.run_id, "val_mAP_at_0.5")

            if steps_loss:
                ax.plot(steps_loss, loss, color=seed_color, linewidth=2,
                        label="Train loss", alpha=0.9)
                ax.set_ylabel("Train Loss", color=seed_color)
                ax.tick_params(axis="y", labelcolor=seed_color)

            if steps_map:
                ax2.set_visible(True)
                ax2.spines["right"].set_visible(True)
                ax2.plot(steps_map, val_map, color=seed_color, linewidth=2,
                         linestyle="--", label="Val mAP@0.5", alpha=0.8)
                ax2.set_ylabel("Val mAP@0.5", color="gray")
                ax2.tick_params(axis="y", labelcolor="gray")
                ax2.set_ylim(0, 1.05)

                best_idx = int(np.argmax(val_map))
                best_epoch = steps_map[best_idx]
                best_val = val_map[best_idx]
                final_map = val_map[-1]
                ax2.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5, alpha=0.9)
                ax2.scatter([best_epoch], [best_val], color="green", s=60, zorder=5)

                ax.set_title(
                    f"{top3_labels[model_id]}\n"
                    f"seed={seed} | best={best_val:.3f} @ ep{best_epoch} | final={final_map:.3f}",
                    fontsize=9,
                )
            else:
                ax.set_title(f"{top3_labels[model_id]}\nseed={seed}")

            ax.set_xlabel("Epoch")
            ax.set_xlim(left=0)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")

    fig.suptitle(
        "Figure 3. Multi-Seed Learning Curves: Top-3 Models (3 seeds × 3 models)\n"
        "Solid = train loss | Dashed = val mAP@0.5 | Green dot = best checkpoint",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig3_multiseed_curves.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating learning curve figures...")
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Output: {PLOTS_DIR.relative_to(ROOT)}\n")

    convergence_data = fig1_learning_curves_grid()
    fig2_convergence(convergence_data)
    fig3_multiseed_curves()

    print("\nAll figures saved.")

    if convergence_data:
        print("\n=== Convergence Summary (opt seed=42) ===")
        print(f"{'Model':<22} {'Best ep':>8} {'Total':>7} {'Best mAP':>10} {'Final mAP':>10}")
        print("-" * 62)
        for d in convergence_data:
            flag = " ✅" if d["best_mAP"] >= 0.75 else " ❌"
            print(
                f"{d['label']:<22} {d['best_epoch']:>8} {d['total_epochs']:>7} "
                f"{d['best_mAP']:>10.3f} {d['final_mAP']:>10.3f}{flag}"
            )


if __name__ == "__main__":
    main()
