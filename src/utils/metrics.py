"""Метрики для TAL: mAP, IoU, Boundary Error."""

from __future__ import annotations

import numpy as np


def compute_iou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """Intersection over Union двух интервалов (в секундах или кадрах)."""
    p_start, p_end = pred
    g_start, g_end = gt
    inter = max(0.0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)
    return inter / union if union > 0 else 0.0


def compute_boundary_error(
    pred: tuple[float, float], gt: tuple[float, float]
) -> float:
    """Средняя абсолютная ошибка границ (start + end) / 2."""
    start_err = abs(pred[0] - gt[0])
    end_err = abs(pred[1] - gt[1])
    return (start_err + end_err) / 2.0


def compute_ap(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float,
) -> float:
    """Вычисляет Average Precision при заданном IoU threshold.

    Args:
        predictions: список dict с ключами 'video', 'start', 'end', 'score'
        ground_truths: список dict с ключами 'video', 'start', 'end'
        iou_threshold: минимальный IoU для совпадения

    Returns:
        AP ∈ [0, 1]
    """
    if not ground_truths:
        return 0.0

    # Сортируем предсказания по убыванию score
    preds = sorted(predictions, key=lambda x: x["score"], reverse=True)

    n_gt = len(ground_truths)
    matched = set()  # индексы gt, уже сопоставленные

    tp_list = []
    fp_list = []

    for pred in preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt["video"] != pred["video"]:
                continue
            if gt_idx in matched:
                continue
            iou = compute_iou((pred["start"], pred["end"]), (gt["start"], gt["end"]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            matched.add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall = tp_cum / n_gt

    # Interpolated AP (11-point)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 11.0

    return float(ap)


def compute_map(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Вычисляет mAP при нескольких IoU thresholds.

    Returns:
        dict: {'mAP@0.3': ..., 'mAP@0.5': ..., 'mAP@0.7': ...}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.3, 0.5, 0.7]

    return {
        f"mAP@{t}": compute_ap(predictions, ground_truths, t)
        for t in iou_thresholds
    }


def compute_precision_recall(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Precision и Recall при заданном IoU threshold.

    Каждое предсказание — TP если IoU ≥ threshold с незанятым GT, иначе FP.
    """
    if not ground_truths:
        return 0.0, 0.0
    if not predictions:
        return 0.0, 0.0

    matched_gt = set()
    tp = 0

    for pred in predictions:
        for gt_idx, gt in enumerate(ground_truths):
            if gt["video"] != pred["video"]:
                continue
            if gt_idx in matched_gt:
                continue
            if compute_iou((pred["start"], pred["end"]), (gt["start"], gt["end"])) >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                break

    precision = tp / len(predictions)
    recall = tp / len(ground_truths)
    return precision, recall
