"""
Обучение GymRT моделей.

Запуск:
  python src/scripts/train.py --config configs/mobilenet_v3_small.yaml
  python src/scripts/train.py --config configs/mobilenet_v3_bilstm_attn.yaml --seed 123
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.frame_dataset import FrameDataset  # noqa: E402
from src.data.pose_dataset import PoseDataset  # noqa: E402
from src.models.full_model import GymRT  # noqa: E402
from src.models.pose_model import PoseGymRT  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    compute_boundary_error,
    compute_map,
    compute_precision_recall,
)
from src.utils.postprocess import scores_to_detections  # noqa: E402
from src.utils.tracking import MLflowTracker  # noqa: E402


def _detach_state(state: Any) -> Any:
    """Detach hidden state tensors to truncate TBPTT computation graph."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, (tuple, list)):
        detached = tuple(_detach_state(s) for s in state)
        return detached if isinstance(state, tuple) else list(detached)
    return state


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: Path, seed_override: int | None = None) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Наследование base.yaml
    base_path = config_path.parent / "base.yaml"
    if base_path.exists() and config_path != base_path:
        with open(base_path, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        base = _deep_merge(base, cfg)
        cfg = base

    if seed_override is not None:
        cfg["experiment"]["seed"] = seed_override

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@torch.no_grad()
def evaluate_split(
    model: GymRT,
    dataset: FrameDataset,
    device: torch.device,
    threshold: float,
    min_dur: float,
    max_dur: float,
) -> dict[str, float]:
    """Оценка модели на сплите. Возвращает метрики."""
    model.eval()
    predictions = []
    ground_truths = []

    for sample in dataset.samples:
        feats = sample.features.to(device)  # [T, D]
        logits = model.temporal(feats.unsqueeze(0)).squeeze(0)  # [T]
        scores = torch.sigmoid(logits).cpu()
        del feats, logits

        dets = scores_to_detections(scores, sample.fps, threshold, min_dur, max_dur)
        for d in dets:
            predictions.append({
                "video": sample.video_name,
                "start": d.start_sec(sample.fps),
                "end":   d.end_sec(sample.fps),
                "score": d.score,
            })

        if sample.has_element:
            pos_frames = sample.labels.nonzero(as_tuple=True)[0]
            if len(pos_frames) > 0:
                gt_start = pos_frames[0].item() / sample.fps
                gt_end   = pos_frames[-1].item() / sample.fps
                ground_truths.append({
                    "video": sample.video_name,
                    "start": gt_start,
                    "end":   gt_end,
                })

    metrics = compute_map(predictions, ground_truths)
    p, r = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    metrics["precision"] = p
    metrics["recall"]    = r

    # Boundary Error (только TP предсказания)
    be_list = []
    matched_gt = set()
    for pred in predictions:
        for gt_i, gt in enumerate(ground_truths):
            if gt["video"] != pred["video"] or gt_i in matched_gt:
                continue
            from src.utils.metrics import compute_iou
            if compute_iou((pred["start"], pred["end"]), (gt["start"], gt["end"])) >= 0.5:
                be_list.append(compute_boundary_error(
                    (pred["start"], pred["end"]), (gt["start"], gt["end"])
                ))
                matched_gt.add(gt_i)
                break

    metrics["boundary_error"] = float(np.mean(be_list)) if be_list else float("nan")
    return metrics


def train(config_path: Path, seed: int | None = None) -> None:
    cfg = load_config(config_path, seed)

    exp_cfg   = cfg["experiment"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    post_cfg  = cfg["postprocess"]
    mlf_cfg   = cfg["mlflow"]

    seed_val = exp_cfg.get("seed", 42)
    set_seed(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {seed_val}")

    # --- Данные ---
    v2_root   = Path(data_cfg["v2_root"])
    feat_root = ROOT / data_cfg["features_dir"]
    backbone  = model_cfg.get("backbone", "")
    is_pose   = model_cfg.get("model_type") == "pose"

    if is_pose:
        pose_root = ROOT / data_cfg.get("pose_features_dir", "data/pose_features")
        train_ds = PoseDataset(pose_root / "train", v2_root / data_cfg["train_annotations"])
        valid_ds = PoseDataset(pose_root / "valid", v2_root / data_cfg["valid_annotations"])
    else:
        train_ds = FrameDataset(feat_root / "train", v2_root / data_cfg["train_annotations"], backbone)
        valid_ds = FrameDataset(feat_root / "valid", v2_root / data_cfg["valid_annotations"], backbone)

    pos_weight = torch.tensor(train_ds.pos_weight(), device=device)
    print(f"Train: {len(train_ds)} videos | pos_weight={pos_weight.item():.1f}")
    print(f"Valid: {len(valid_ds)} videos")

    # --- Модель ---
    temporal_cfg = {
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "n_layers":   model_cfg.get("n_layers", 2),
        "dropout":    model_cfg.get("dropout", 0.3),
    }
    if model_cfg.get("temporal_head") == "causal_tcn":
        temporal_cfg.pop("n_layers", None)
        temporal_cfg.pop("hidden_dim", None)

    if is_pose:
        model = PoseGymRT(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            temporal_name=model_cfg["temporal_head"],
            temporal_cfg=temporal_cfg,
        ).to(device)
    else:
        model = GymRT(
            backbone_name=backbone,
            temporal_name=model_cfg["temporal_head"],
            temporal_cfg=temporal_cfg,
            frozen_backbone=True,
        ).to(device)

    print(f"Model params: {model.count_parameters():,}  |  Size: {model.size_mb():.1f} MB")

    # --- Оптимизатор ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    # --- MLflow ---
    tracker = MLflowTracker(
        experiment_name=mlf_cfg["experiment_name"],
        tracking_uri=(ROOT / mlf_cfg["tracking_uri"]).as_uri(),
    )
    exp_name = exp_cfg["name"]
    run_name = f"{exp_name}_seed{seed_val}"

    best_map = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 10)
    chunk_size = train_cfg.get("chunk_size", 256)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    best_ckpt = models_dir / f"{run_name}_best.pt"

    with tracker.start_run(run_name=run_name):
        tracker.log_params({
            "backbone":      backbone,
            "temporal_head": model_cfg["temporal_head"],
            "hidden_dim":    model_cfg.get("hidden_dim", 128),
            "n_layers":      model_cfg.get("n_layers", 2),
            "dropout":       model_cfg.get("dropout", 0.3),
            "lr":            train_cfg["lr"],
            "weight_decay":  train_cfg["weight_decay"],
            "chunk_size":    chunk_size,
            "seed":          seed_val,
            "pos_weight":    pos_weight.item(),
        })
        tracker.set_tag("model_size_mb", f"{model.size_mb():.1f}")

        for epoch in range(1, train_cfg["epochs"] + 1):
            # NOTE: backbone всегда заморожен при pre-extracted features.
            # Разморозка backbone не используется — фичи кэшированы, backbone
            # не участвует в forward/backward pass во время обучения.

            model.train()
            total_loss = 0.0
            n_chunks = 0

            for sample in train_ds.samples:
                # TBPTT: передаём hidden state между чанками одного видео
                hidden: Any = None
                for chunk in train_ds.iter_tbptt_chunks(sample, chunk_size):
                    chunk_feat = chunk.features.unsqueeze(0).to(device)   # [1, L, D]
                    chunk_lab  = chunk.labels.unsqueeze(0).to(device)     # [1, L]

                    # forward_train возвращает (logits, new_state)
                    logits, hidden = model.temporal.forward_train(chunk_feat, hidden)
                    logits = logits.squeeze(0)  # [L]

                    # Маскируем паддинг последнего чанка
                    vl = chunk.valid_len
                    loss = criterion(logits[:vl], chunk_lab.squeeze(0)[:vl])

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
                    optimizer.step()

                    # Detach hidden state для TBPTT (обрываем граф, но передаём значения)
                    if hidden is not None:
                        hidden = _detach_state(hidden)

                    total_loss += loss.item()
                    n_chunks += 1

            scheduler.step()
            avg_loss = total_loss / max(n_chunks, 1)

            # Валидация каждые 5 эпох и в последнюю
            if epoch % 5 == 0 or epoch == train_cfg["epochs"]:
                metrics = evaluate_split(
                    model, valid_ds, device,
                    post_cfg["threshold"],
                    post_cfg["min_duration_sec"],
                    post_cfg["max_duration_sec"],
                )
                map50 = metrics.get("mAP@0.5", 0.0)
                be    = metrics.get("boundary_error", float("nan"))
                rec   = metrics.get("recall", 0.0)

                print(
                    f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                    f"mAP@0.5={map50:.3f} | BE={be:.2f}s | Recall={rec:.3f}"
                )

                tracker.log_metrics({
                    "train_loss":       avg_loss,
                    "val_mAP_at_0.5":  metrics.get("mAP@0.5", 0.0),
                    "val_mAP_at_0.3":  metrics.get("mAP@0.3", 0.0),
                    "val_mAP_at_0.7":  metrics.get("mAP@0.7", 0.0),
                    "val_precision":    metrics.get("precision", 0.0),
                    "val_recall":       rec,
                    "val_BE":           be if not np.isnan(be) else 0.0,
                }, step=epoch)

                if map50 > best_map:
                    best_map = map50
                    patience_counter = 0
                    torch.save({
                        "epoch":        epoch,
                        "model_state":  model.state_dict(),
                        "optim_state":  optimizer.state_dict(),
                        "config":       cfg,
                        "val_metrics":  metrics,
                    }, best_ckpt)
                    print(f"  [+] Saved best checkpoint (mAP@0.5={best_map:.3f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping на эпохе {epoch}")
                        break
            else:
                tracker.log_metric("train_loss", avg_loss, step=epoch)

        tracker.log_params({"best_val_mAP_at_0.5": best_map})
        tracker.log_artifact(str(best_ckpt))
        print(f"\nTraining done. Best mAP@0.5={best_map:.3f} -> {best_ckpt.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение GymRT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.seed)


if __name__ == "__main__":
    main()
