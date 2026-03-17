"""
Оценка обученной GymRT модели на тестовом или valid сплите.

Запуск:
  python src/scripts/evaluate.py --config configs/mobilenet_v3_small.yaml \
      --checkpoint models/mobilenet_v3_small_seed42_best.pt

  # Sweep порогов 0.1–0.9:
  python src/scripts/evaluate.py --config configs/mobilenet_v3_bilstm_attn.yaml \
      --checkpoint models/mobilenet_v3_bilstm_attn_seed42_best.pt \
      --sweep-threshold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.frame_dataset import FrameDataset  # noqa: E402
from src.models.full_model import GymRT  # noqa: E402
from src.scripts.train import load_config  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    compute_boundary_error,
    compute_iou,
    compute_map,
    compute_precision_recall,
)
from src.utils.postprocess import scores_to_detections  # noqa: E402


def load_model(checkpoint_path: Path, cfg: dict, device: torch.device) -> GymRT:
    model_cfg = cfg["model"]
    temporal_cfg = {
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "n_layers":   model_cfg.get("n_layers", 2),
        "dropout":    model_cfg.get("dropout", 0.3),
    }
    if model_cfg["temporal_head"] == "causal_tcn":
        temporal_cfg.pop("n_layers", None)

    model = GymRT(
        backbone_name=model_cfg["backbone"],
        temporal_name=model_cfg["temporal_head"],
        temporal_cfg=temporal_cfg,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: GymRT,
    dataset: FrameDataset,
    device: torch.device,
    threshold: float,
    min_dur: float,
    max_dur: float,
) -> tuple[dict[str, float], float]:
    """Returns (metrics_dict, fps)."""
    predictions = []
    ground_truths = []
    total_frames = 0
    t0 = time.perf_counter()

    for sample in dataset.samples:
        feats = sample.features.to(device)
        logits = model.temporal(feats.unsqueeze(0)).squeeze(0)
        scores = torch.sigmoid(logits).cpu()
        total_frames += feats.shape[0]

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
                ground_truths.append({
                    "video": sample.video_name,
                    "start": pos_frames[0].item() / sample.fps,
                    "end":   pos_frames[-1].item() / sample.fps,
                })

    elapsed = time.perf_counter() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    metrics = compute_map(predictions, ground_truths)
    p, r = compute_precision_recall(predictions, ground_truths, iou_threshold=0.5)
    metrics["precision"] = p
    metrics["recall"] = r

    be_list = []
    matched_gt = set()
    for pred in predictions:
        for gt_i, gt in enumerate(ground_truths):
            if gt["video"] != pred["video"] or gt_i in matched_gt:
                continue
            if compute_iou((pred["start"], pred["end"]), (gt["start"], gt["end"])) >= 0.5:
                be_list.append(compute_boundary_error(
                    (pred["start"], pred["end"]), (gt["start"], gt["end"])
                ))
                matched_gt.add(gt_i)
                break

    metrics["boundary_error"] = float(np.mean(be_list)) if be_list else float("nan")
    metrics["n_preds"] = len(predictions)
    metrics["n_gt"] = len(ground_truths)

    return metrics, fps


def print_metrics(metrics: dict, fps: float, threshold: float) -> None:
    print(f"\n  threshold={threshold:.2f}")
    print(f"  mAP@0.3={metrics.get('mAP@0.3', 0):.3f}  "
          f"mAP@0.5={metrics.get('mAP@0.5', 0):.3f}  "
          f"mAP@0.7={metrics.get('mAP@0.7', 0):.3f}")
    print(f"  Precision={metrics.get('precision', 0):.3f}  "
          f"Recall={metrics.get('recall', 0):.3f}")
    be = metrics.get('boundary_error', float('nan'))
    be_str = f"{be:.3f}s" if not np.isnan(be) else "n/a"
    print(f"  Boundary Error={be_str}")
    print(f"  Preds={metrics.get('n_preds', 0)} / GT={metrics.get('n_gt', 0)}")
    print(f"  FPS={fps:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка GymRT модели")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--sweep-threshold", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    post_cfg = cfg["postprocess"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v2_root  = Path(data_cfg["v2_root"])
    feat_root = ROOT / data_cfg["features_dir"]
    backbone  = cfg["model"]["backbone"]

    ann_key = f"{args.split}_annotations"
    dataset = FrameDataset(
        feat_root / args.split,
        v2_root / data_cfg[ann_key],
        backbone,
    )
    print(f"Split: {args.split}  |  Videos: {len(dataset)}")

    model = load_model(args.checkpoint, cfg, device)
    print(f"Model: {model.backbone_name} + {model.temporal_name}  |  "
          f"{model.count_parameters():,} params  |  {model.size_mb():.1f} MB")

    if args.sweep_threshold:
        print("\n=== Threshold sweep ===")
        for t in [round(x * 0.1, 1) for x in range(1, 10)]:
            metrics, fps = evaluate(model, dataset, device, t,
                                    post_cfg["min_duration_sec"],
                                    post_cfg["max_duration_sec"])
            print_metrics(metrics, fps, t)
    else:
        metrics, fps = evaluate(model, dataset, device,
                                post_cfg["threshold"],
                                post_cfg["min_duration_sec"],
                                post_cfg["max_duration_sec"])
        print("\n=== Результаты ===")
        print_metrics(metrics, fps, post_cfg["threshold"])


if __name__ == "__main__":
    main()
