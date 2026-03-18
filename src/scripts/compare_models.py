"""
Сравнение всех обученных моделей GymRT.

Собирает все чекпоинты из models/, оценивает на valid+test,
строит графики и сохраняет таблицу в docs/comparison_table.md.

Запуск:
  python src/scripts/compare_models.py
  python src/scripts/compare_models.py --split test
  python src/scripts/compare_models.py --checkpoints models/bilstm*.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.frame_dataset import FrameDataset  # noqa: E402
from src.models.full_model import GymRT  # noqa: E402
from src.scripts.train import evaluate_split, load_config  # noqa: E402

# Соответствие имён чекпоинтов → конфигов
CHECKPOINT_TO_CONFIG: dict[str, str] = {
    "mobilenet_v3_small":           "mobilenet_v3_small",
    "mobilenet_v3_bilstm":          "mobilenet_v3_bilstm",
    "mobilenet_v3_bilstm_attn":     "mobilenet_v3_bilstm_attn",
    "mobilenet_v3_tcn":             "mobilenet_v3_tcn",
    "efficientnet_b0_bilstm_attn":  "efficientnet_b0_bilstm_attn",
    "efficientnet_b0_bilstm":       "efficientnet_b0_bilstm",
    "efficientnet_b0_mlp":          "efficientnet_b0_mlp",
    "efficientnet_b0_tcn":          "efficientnet_b0_tcn",
    # HPO-оптимальные версии
    "mobilenet_v3_small_opt":           "mobilenet_v3_small_opt",
    "mobilenet_v3_bilstm_opt":          "mobilenet_v3_bilstm_opt",
    "mobilenet_v3_bilstm_attn_opt":     "mobilenet_v3_bilstm_attn_opt",
    "mobilenet_v3_tcn_opt":             "mobilenet_v3_tcn_opt",
    "efficientnet_b0_bilstm_attn_opt":  "efficientnet_b0_bilstm_attn_opt",
    "efficientnet_b0_bilstm_opt":       "efficientnet_b0_bilstm_opt",
    "efficientnet_b0_mlp_opt":          "efficientnet_b0_mlp_opt",
    "efficientnet_b0_tcn_opt":          "efficientnet_b0_tcn_opt",
}


def infer_config_name(checkpoint_path: Path) -> str | None:
    """Выводит имя конфига из имени чекпоинта."""
    stem = checkpoint_path.stem  # e.g. mobilenet_v3_bilstm_seed42_best
    for key in sorted(CHECKPOINT_TO_CONFIG.keys(), key=len, reverse=True):
        if stem.startswith(key):
            return CHECKPOINT_TO_CONFIG[key]
    return None


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: dict,
    device: torch.device,
) -> GymRT:
    model_cfg = cfg["model"]
    temporal_cfg: dict = {
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "n_layers":   model_cfg.get("n_layers", 2),
        "dropout":    model_cfg.get("dropout", 0.3),
    }
    if model_cfg["temporal_head"] == "causal_tcn":
        temporal_cfg.pop("n_layers", None)
        temporal_cfg.pop("hidden_dim", None)

    model = GymRT(
        backbone_name=model_cfg["backbone"],
        temporal_name=model_cfg["temporal_head"],
        temporal_cfg=temporal_cfg,
        frozen_backbone=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate_model(
    checkpoint_path: Path,
    cfg: dict,
    device: torch.device,
    split: str,
) -> dict:
    """Оценивает один чекпоинт на заданном сплите."""
    data_cfg = cfg["data"]
    post_cfg = cfg["postprocess"]
    feat_root = ROOT / data_cfg["features_dir"]
    v2_root   = Path(data_cfg["v2_root"])
    backbone  = cfg["model"]["backbone"]

    ann_key = f"{split}_annotations"
    dataset = FrameDataset(
        feat_root / split,
        v2_root / data_cfg[ann_key],
        backbone,
    )

    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    import time
    t0 = time.perf_counter()
    metrics = evaluate_split(
        model, dataset, device,
        post_cfg["threshold"],
        post_cfg["min_duration_sec"],
        post_cfg["max_duration_sec"],
    )
    elapsed = time.perf_counter() - t0
    total_frames = sum(s.features.shape[0] for s in dataset.samples)
    metrics["fps"] = total_frames / elapsed if elapsed > 0 else 0.0
    metrics["size_mb"] = model.size_mb()
    metrics["params"] = model.count_parameters()
    return metrics


def collect_checkpoints(models_dir: Path, pattern: str = "*_best.pt") -> list[Path]:
    """Собирает все чекпоинты из директории."""
    ckpts = sorted(models_dir.glob(pattern))
    # Группируем по базовому имени (берём один seed=42 для каждой модели)
    seen: dict[str, Path] = {}
    for ckpt in ckpts:
        stem = ckpt.stem  # mobilenet_v3_bilstm_seed42_best
        # Извлекаем базовое имя (до _seed)
        if "_seed" in stem:
            base = stem[:stem.index("_seed")]
        else:
            base = stem.replace("_best", "")
        # Приоритет: seed42 > остальные
        if base not in seen or "_seed42_" in ckpt.name:
            seen[base] = ckpt
    return sorted(seen.values())


def format_table(rows: list[dict], split: str) -> str:
    """Форматирует Markdown таблицу результатов."""
    lines = [
        f"## Результаты на {split} (threshold=0.50)\n",
        "| Модель | mAP@0.3 | mAP@0.5 | mAP@0.7 | Precision | Recall | BE (s) | FPS | Size MB |",
        "|--------|:-------:|:-------:|:-------:|:---------:|:------:|:------:|----:|--------:|",
    ]
    for r in rows:
        be = f"{r['boundary_error']:.3f}" if not np.isnan(r.get("boundary_error", float("nan"))) else "n/a"
        lines.append(
            f"| {r['name']} "
            f"| {r.get('mAP@0.3', 0):.3f} "
            f"| {r.get('mAP@0.5', 0):.3f} "
            f"| {r.get('mAP@0.7', 0):.3f} "
            f"| {r.get('precision', 0):.3f} "
            f"| {r.get('recall', 0):.3f} "
            f"| {be} "
            f"| {r.get('fps', 0):,.0f} "
            f"| {r.get('size_mb', 0):.1f} |"
        )
    return "\n".join(lines)


def save_comparison_table(
    valid_rows: list[dict],
    test_rows: list[dict],
    out_path: Path,
) -> None:
    lines = [
        "# Сравнение всех моделей GymRT\n",
        "*Дата генерации: автоматически из compare_models.py*\n",
        "",
        format_table(valid_rows, "valid"),
        "",
        format_table(test_rows, "test"),
        "",
        "## Примечания",
        "- Все модели: MobileNetV3-Small backbone (если не указано иное)",
        "- Threshold=0.50 для всех",
        "- FPS = кадров/сек (inference, RTX 5060 Ti, pre-extracted features)",
        "- _opt = HPO-оптимальные гиперпараметры",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[+] Таблица сохранена: {out_path}")


def make_plots(valid_rows: list[dict], test_rows: list[dict], plots_dir: Path) -> None:
    """Генерирует графики сравнения."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib не установлен, графики не генерируются")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in test_rows]
    x = range(len(names))

    # 1. mAP@0.5 comparison (valid vs test)
    fig, ax = plt.subplots(figsize=(10, 5))
    valid_map = [r.get("mAP@0.5", 0) for r in valid_rows]
    test_map  = [r.get("mAP@0.5", 0) for r in test_rows]
    width = 0.35
    if valid_map and len(valid_map) == len(names):
        ax.bar([i - width/2 for i in x], valid_map, width, label="Valid", color="#4C72B0")
    if test_map and len(test_map) == len(names):
        ax.bar([i + width/2 for i in x], test_map, width, label="Test", color="#DD8452")
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1, label="PRD target (0.75)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("mAP@0.5: Valid vs Test")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_map.png", dpi=150)
    plt.close(fig)

    # 2. FPS vs mAP@0.5 scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in test_rows:
        ax.scatter(r.get("fps", 0), r.get("mAP@0.5", 0), s=80, zorder=3)
        ax.annotate(r["name"], (r.get("fps", 0), r.get("mAP@0.5", 0)),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.axhline(0.75, color="red", linestyle="--", linewidth=1, label="PRD mAP target")
    ax.axvline(25, color="green", linestyle="--", linewidth=1, label="PRD FPS target (25)")
    ax.set_xlabel("FPS (inference)")
    ax.set_ylabel("mAP@0.5 (test)")
    ax.set_title("Speed vs Accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_fps.png", dpi=150)
    plt.close(fig)

    # 3. Boundary Error
    be_values = [
        r.get("boundary_error", float("nan")) for r in test_rows
    ]
    valid_be = [(n, v) for n, v in zip(names, be_values) if not np.isnan(v)]
    if valid_be:
        be_names, be_vals = zip(*valid_be)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2ca02c" if v <= 1.5 else "#d62728" for v in be_vals]
        ax.barh(be_names, be_vals, color=colors)
        ax.axvline(1.5, color="red", linestyle="--", linewidth=1, label="PRD target (1.5s)")
        ax.set_xlabel("Boundary Error (s)")
        ax.set_title("Boundary Error (test, lower is better)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "comparison_be.png", dpi=150)
        plt.close(fig)

    print(f"[+] Графики сохранены в {plots_dir}")


def run_comparison(
    splits: list[str],
    checkpoint_paths: list[Path] | None,
    models_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if checkpoint_paths is None:
        checkpoint_paths = collect_checkpoints(models_dir)

    if not checkpoint_paths:
        print("[ERROR] Нет чекпоинтов для сравнения")
        return

    print(f"\nЧекпоинтов: {len(checkpoint_paths)}")

    results: dict[str, dict] = {}  # split → list of row dicts
    for split in splits:
        results[split] = []

    for ckpt_path in checkpoint_paths:
        config_name = infer_config_name(ckpt_path)
        if config_name is None:
            print(f"[WARN] Не удалось определить конфиг для {ckpt_path.name}, пропускаю")
            continue

        config_path = ROOT / "configs" / f"{config_name}.yaml"
        if not config_path.exists():
            print(f"[WARN] Конфиг не найден: {config_path}, пропускаю")
            continue

        cfg = load_config(config_path)
        display_name = config_name.replace("mobilenet_v3_", "").replace("efficientnet_b0_", "eff_b0_")
        # Добавляем seed если не seed42
        stem = ckpt_path.stem
        if "_seed" in stem:
            seed_part = stem[stem.index("_seed")+1:stem.index("_best")]
            if seed_part != "seed42":
                display_name += f" ({seed_part})"

        print(f"\n--- {ckpt_path.name} ---")

        for split in splits:
            ann_key = f"{split}_annotations"
            if ann_key not in cfg.get("data", {}):
                print(f"  [WARN] Нет аннотаций для split={split}")
                continue
            try:
                metrics = evaluate_model(ckpt_path, cfg, device, split)
                metrics["name"] = display_name
                results[split].append(metrics)
                map50 = metrics.get("mAP@0.5", 0)
                be = metrics.get("boundary_error", float("nan"))
                rec = metrics.get("recall", 0)
                print(f"  {split}: mAP@0.5={map50:.3f} | BE={be:.2f}s | Recall={rec:.3f} | FPS={metrics['fps']:,.0f}")
            except Exception as e:
                print(f"  [ERROR] {split}: {e}")

    # Сохраняем таблицу
    docs_dir = ROOT / "docs"
    valid_rows = results.get("valid", [])
    test_rows  = results.get("test", [])

    if valid_rows or test_rows:
        save_comparison_table(valid_rows, test_rows, docs_dir / "comparison_table.md")
        make_plots(valid_rows, test_rows, ROOT / "data" / "plots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение всех моделей GymRT")
    parser.add_argument("--split", default="both",
                        choices=["valid", "test", "both"],
                        help="Какой сплит оценивать")
    parser.add_argument("--checkpoints", nargs="*", type=Path,
                        help="Конкретные чекпоинты (по умолчанию — все из models/)")
    parser.add_argument("--models-dir", type=Path,
                        default=ROOT / "models")
    args = parser.parse_args()

    splits = ["valid", "test"] if args.split == "both" else [args.split]
    run_comparison(splits, args.checkpoints, args.models_dir)


if __name__ == "__main__":
    main()
