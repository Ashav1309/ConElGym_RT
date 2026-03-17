"""Тесты для metrics.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.metrics import (
    compute_boundary_error,
    compute_iou,
    compute_map,
    compute_precision_recall,
)

# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def test_iou_perfect_overlap():
    assert compute_iou((0.0, 10.0), (0.0, 10.0)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert compute_iou((0.0, 5.0), (6.0, 10.0)) == pytest.approx(0.0)


def test_iou_partial():
    # pred=[0,10], gt=[5,15] → inter=5, union=15
    iou = compute_iou((0.0, 10.0), (5.0, 15.0))
    assert iou == pytest.approx(5.0 / 15.0)


def test_iou_contained():
    # pred внутри gt → inter=pred_len, union=gt_len
    iou = compute_iou((2.0, 8.0), (0.0, 10.0))
    assert iou == pytest.approx(6.0 / 10.0)


def test_iou_zero_length():
    assert compute_iou((5.0, 5.0), (5.0, 5.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Boundary Error
# ---------------------------------------------------------------------------

def test_boundary_error_perfect():
    assert compute_boundary_error((1.0, 5.0), (1.0, 5.0)) == pytest.approx(0.0)


def test_boundary_error_symmetric():
    # start_err=1, end_err=1 → mean=1.0
    assert compute_boundary_error((0.0, 4.0), (1.0, 5.0)) == pytest.approx(1.0)


def test_boundary_error_asymmetric():
    # start_err=0, end_err=2 → mean=1.0
    assert compute_boundary_error((1.0, 3.0), (1.0, 5.0)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mAP / Precision / Recall
# ---------------------------------------------------------------------------

def _make_pred(video, start, end, score):
    return {"video": video, "start": start, "end": end, "score": score}


def _make_gt(video, start, end):
    return {"video": video, "start": start, "end": end}


def test_map_perfect_prediction():
    preds = [_make_pred("v1.mp4", 0.0, 10.0, 1.0)]
    gts = [_make_gt("v1.mp4", 0.0, 10.0)]
    result = compute_map(preds, gts, iou_thresholds=[0.5])
    assert result["mAP@0.5"] == pytest.approx(1.0)


def test_map_no_predictions():
    gts = [_make_gt("v1.mp4", 0.0, 10.0)]
    result = compute_map([], gts, iou_thresholds=[0.5])
    assert result["mAP@0.5"] == pytest.approx(0.0)


def test_map_no_ground_truths():
    preds = [_make_pred("v1.mp4", 0.0, 10.0, 1.0)]
    result = compute_map(preds, [], iou_thresholds=[0.5])
    assert result["mAP@0.5"] == pytest.approx(0.0)


def test_map_low_iou_miss():
    # pred и gt почти не пересекаются → miss
    preds = [_make_pred("v1.mp4", 0.0, 2.0, 1.0)]
    gts = [_make_gt("v1.mp4", 8.0, 10.0)]
    result = compute_map(preds, gts, iou_thresholds=[0.5])
    assert result["mAP@0.5"] == pytest.approx(0.0)


def test_precision_recall_perfect():
    preds = [_make_pred("v1.mp4", 0.0, 10.0, 1.0)]
    gts = [_make_gt("v1.mp4", 0.0, 10.0)]
    p, r = compute_precision_recall(preds, gts, iou_threshold=0.5)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_precision_recall_false_positive():
    preds = [
        _make_pred("v1.mp4", 0.0, 10.0, 0.9),   # TP
        _make_pred("v1.mp4", 20.0, 30.0, 0.8),  # FP
    ]
    gts = [_make_gt("v1.mp4", 0.0, 10.0)]
    p, r = compute_precision_recall(preds, gts, iou_threshold=0.5)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(1.0)


def test_precision_recall_no_predictions():
    gts = [_make_gt("v1.mp4", 0.0, 10.0)]
    p, r = compute_precision_recall([], gts)
    assert p == pytest.approx(0.0)
    assert r == pytest.approx(0.0)
