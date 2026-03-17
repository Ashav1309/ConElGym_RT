"""
Leave-One-Apparatus-Out (LOAO) Cross-Validation.

4 фолда: Ball, Clubs, Hoop, Ribbon.
На каждом фолде: исключаем один снаряд из train+valid, оцениваем на исключённом.

Запуск:
  python src/scripts/loao_cv.py --config configs/mobilenet_v3_bilstm_attn_opt.yaml --seed 42
  python src/scripts/loao_cv.py --config configs/mobilenet_v3_bilstm_attn_opt.yaml --seed 42 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.frame_dataset import FrameDataset  # noqa: E402
from src.models.full_model import GymRT  # noqa: E402
from src.scripts.train import (  # noqa: E402
    _detach_state,
    evaluate_split,
    load_config,
    set_seed,
)

APPARATUS = ["Ball", "Clubs", "Hoop", "Ribbon"]


def filter_by_apparatus(
    dataset: FrameDataset, apparatus: str, keep: bool
) -> FrameDataset:
    """Возвращает новый FrameDataset с отфильтрованными samples.

    Args:
        dataset:    исходный датасет
        apparatus:  имя снаряда (Ball/Clubs/Hoop/Ribbon)
        keep:       True → оставить только этот снаряд, False → исключить
    """
    filtered = object.__new__(FrameDataset)
    filtered.features_dir = dataset.features_dir
    filtered.backbone = dataset.backbone
    filtered.samples = [
        s for s in dataset.samples
        if (apparatus.lower() in s.video_name.lower()) == keep
    ]
    return filtered


def train_one_fold(
    cfg: dict,
    train_ds: FrameDataset,
    valid_ds: FrameDataset,
    device: torch.device,
    seed: int,
) -> GymRT:
    """Обучает одну модель на одном фолде. Возвращает обученную модель."""
    set_seed(seed)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    pos_weight = torch.tensor(train_ds.pos_weight(), device=device)

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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    patience     = train_cfg.get("early_stopping_patience", 10)
    chunk_size   = int(train_cfg.get("chunk_size", 256))
    best_map     = 0.0
    patience_ctr = 0
    best_state   = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        n_chunks = 0

        for sample in train_ds.samples:
            hidden: Any = None
            for chunk in train_ds.iter_tbptt_chunks(sample, chunk_size):
                chunk_feat = chunk.features.unsqueeze(0).to(device)
                chunk_lab  = chunk.labels.unsqueeze(0).to(device)

                logits, hidden = model.temporal.forward_train(chunk_feat, hidden)
                logits = logits.squeeze(0)
                vl = chunk.valid_len
                loss = criterion(logits[:vl], chunk_lab.squeeze(0)[:vl])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if hidden is not None:
                    hidden = _detach_state(hidden)
                n_chunks += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == train_cfg["epochs"]:
            post_cfg = cfg["postprocess"]
            metrics = evaluate_split(
                model, valid_ds, device,
                post_cfg["threshold"],
                post_cfg["min_duration_sec"],
                post_cfg["max_duration_sec"],
            )
            map50 = metrics.get("mAP@0.5", 0.0)

            if map50 > best_map:
                best_map   = map50
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def run_loao(config_path: Path, seed: int, eval_split: str) -> None:
    cfg = load_config(config_path, seed)

    data_cfg = cfg["data"]
    post_cfg = cfg["postprocess"]
    feat_root = ROOT / data_cfg["features_dir"]
    v2_root   = Path(data_cfg["v2_root"])
    backbone  = cfg["model"]["backbone"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LOAO CV | config={config_path.name} | seed={seed} | eval_split={eval_split}")
    print(f"Device: {device}")

    # Загружаем полные датасеты
    full_train = FrameDataset(
        feat_root / "train",
        v2_root / data_cfg["train_annotations"],
        backbone,
    )
    full_valid = FrameDataset(
        feat_root / "valid",
        v2_root / data_cfg["valid_annotations"],
        backbone,
    )
    if eval_split == "test":
        full_eval = FrameDataset(
            feat_root / "test",
            v2_root / data_cfg["test_annotations"],
            backbone,
        )
    else:
        full_eval = full_valid

    results: dict[str, float] = {}

    for apparatus in APPARATUS:
        print(f"\n{'='*50}")
        print(f"Fold: exclude {apparatus}")

        train_fold = filter_by_apparatus(full_train, apparatus, keep=False)
        valid_fold = filter_by_apparatus(full_valid, apparatus, keep=False)
        test_fold  = filter_by_apparatus(full_eval, apparatus, keep=True)

        print(f"  Train: {len(train_fold)} videos | "
              f"Valid: {len(valid_fold)} videos | "
              f"Test({apparatus}): {len(test_fold)} videos")

        if len(test_fold) == 0:
            print(f"  [WARN] No test samples for {apparatus}, skipping")
            continue

        model = train_one_fold(cfg, train_fold, valid_fold, device, seed)

        with torch.no_grad():
            fold_metrics = evaluate_split(
                model, test_fold, device,
                post_cfg["threshold"],
                post_cfg["min_duration_sec"],
                post_cfg["max_duration_sec"],
            )

        map50 = fold_metrics.get("mAP@0.5", 0.0)
        be    = fold_metrics.get("boundary_error", float("nan"))
        rec   = fold_metrics.get("recall", 0.0)
        results[apparatus] = map50

        print(f"  {apparatus}: mAP@0.5={map50:.3f} | BE={be:.2f}s | Recall={rec:.3f}")

    # Итоговая статистика
    values = list(results.values())
    mean_map = float(np.mean(values)) if values else 0.0
    std_map  = float(np.std(values))  if values else 0.0

    print(f"\n{'='*50}")
    print("LOAO Results:")
    for app, v in results.items():
        status = "OK" if v >= 0.70 else "FAIL"
        print(f"  {app:8s}: mAP@0.5={v:.3f}  [{status}]")
    print(f"  Mean: {mean_map:.3f} +/- {std_map:.3f}")
    all_pass = all(v >= 0.70 for v in values)
    print(f"  Criterion (all >= 0.70): {'PASS' if all_pass else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LOAO cross-validation для GymRT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--split",  default="valid", choices=["valid", "test"])
    args = parser.parse_args()
    run_loao(args.config, args.seed, args.split)


if __name__ == "__main__":
    main()
