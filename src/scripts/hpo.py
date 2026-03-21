"""
HPO (Hyperparameter Optimization) с Optuna для GymRT моделей.

Запуск:
  python src/scripts/hpo.py --model mobilenet_v3_bilstm_attn --n-trials 20
  python src/scripts/hpo.py --model mobilenet_v3_bilstm --n-trials 20
  python src/scripts/hpo.py --model mobilenet_v3_tcn --n-trials 20

Лучшие HP сохраняются в configs/<model>_opt.yaml.
История trials — в hpo_studies.db (Optuna SQLite).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.frame_dataset import FrameDataset  # noqa: E402
from src.data.pose_dataset import PoseDataset  # noqa: E402
from src.models.full_model import GymRT  # noqa: E402
from src.models.pose_model import PoseGymRT  # noqa: E402
from src.scripts.train import (  # noqa: E402
    _detach_state,
    evaluate_split,
    load_config,
    set_seed,
)

# Базовый конфиг для старта
BASE_CONFIG = ROOT / "configs" / "base.yaml"

# Пространство поиска (общее для всех моделей)
SEARCH_SPACE = {
    "lr":           ("log_float", 1e-4, 5e-3),
    "weight_decay": ("log_float", 1e-5, 1e-2),
    "dropout":      ("float",     0.1, 0.5),
    "chunk_size":   ("int",       128, 512),
}

# Дополнительно для BiLSTM-based моделей
BILSTM_SPACE = {
    "hidden_dim": ("int_pow2", 64, 256),   # 64, 128, 256
    "n_layers":   ("int",      1, 3),
}

# Дополнительно для TCN
TCN_SPACE: dict = {}   # num_channels задаётся через hidden_dim → [H, H, H//2]


def suggest_hp(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "log_float":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "int_pow2":
        choices = [v for v in [32, 64, 128, 256, 512] if spec[1] <= v <= spec[2]]
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unknown spec kind: {kind}")


def build_cfg_from_trial(
    base_cfg: dict,
    trial: optuna.Trial,
    temporal_head: str,
) -> dict:
    """Создаёт конфиг с HP из trial поверх base_cfg."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    # Общие HP
    for name, spec in SEARCH_SPACE.items():
        val = suggest_hp(trial, name, spec)
        if name in ("lr", "weight_decay", "chunk_size"):
            cfg["training"][name] = val
        elif name == "dropout":
            cfg["model"]["dropout"] = val

    # BiLSTM-specific
    if temporal_head in ("bilstm", "bilstm_attn"):
        for name, spec in BILSTM_SPACE.items():
            val = suggest_hp(trial, name, spec)
            cfg["model"][name] = val

    # Фиксируем patience для честного сравнения
    cfg["training"]["early_stopping_patience"] = 10

    return cfg


def run_trial(
    cfg: dict,
    device: torch.device,
    trial: optuna.Trial,
) -> float:
    """Один trial: обучение + валидация. Возвращает valid mAP@0.5."""
    seed_val = cfg["experiment"].get("seed", 42)
    set_seed(seed_val)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    post_cfg  = cfg["postprocess"]

    feat_root = ROOT / data_cfg["features_dir"]
    v2_root   = Path(data_cfg["v2_root"])
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

    temporal_cfg: dict = {
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

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_chunks   = 0

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

                total_loss += loss.item()
                n_chunks   += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == train_cfg["epochs"]:
            metrics = evaluate_split(
                model, valid_ds, device,
                post_cfg["threshold"],
                post_cfg["min_duration_sec"],
                post_cfg["max_duration_sec"],
            )
            map50 = metrics.get("mAP@0.5", 0.0)

            # Optuna pruning
            trial.report(map50, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if map50 > best_map:
                best_map     = map50
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    return best_map


def save_opt_config(base_cfg: dict, best_params: dict, model_name: str) -> Path:
    """Сохраняет оптимальный конфиг в configs/<model>_opt.yaml."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    # Применяем лучшие HP
    for k, v in best_params.items():
        if k in ("lr", "weight_decay", "chunk_size"):
            cfg["training"][k] = v
        elif k == "dropout":
            cfg["model"]["dropout"] = v
        elif k in ("hidden_dim", "n_layers"):
            cfg["model"][k] = v

    cfg["training"]["early_stopping_patience"] = 10
    cfg["experiment"]["name"] = f"{model_name}_opt"

    out_path = ROOT / "configs" / f"{model_name}_opt.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n[+] Optimal config saved: {out_path}")
    return out_path


def run_hpo(model_name: str, n_trials: int, seed: int) -> None:
    base_cfg = load_config(ROOT / "configs" / f"{model_name}.yaml")
    temporal_head = base_cfg["model"]["temporal_head"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HPO: {model_name} | {n_trials} trials | device={device}")

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{ROOT / f'hpo_{model_name}.db'}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=model_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = build_cfg_from_trial(base_cfg, trial, temporal_head)
        try:
            return run_trial(cfg, device, trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[WARN] Trial {trial.number} failed: {e}")
            return 0.0

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  mAP@0.5 = {study.best_value:.4f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    save_opt_config(base_cfg, study.best_params, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO для GymRT моделей (Optuna)")
    parser.add_argument("--model", required=True,
                        help="Имя модели (без .yaml), напр. mobilenet_v3_bilstm_attn")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_hpo(args.model, args.n_trials, args.seed)


if __name__ == "__main__":
    main()
