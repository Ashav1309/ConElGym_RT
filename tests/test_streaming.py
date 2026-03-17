"""Tests for StreamingDetector."""
from __future__ import annotations

import pytest

from src.models.streaming_state import DetectionState, StreamingDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def feed(detector: StreamingDetector, scores: list[float]) -> list[DetectionState]:
    return [detector.update(s, i) for i, s in enumerate(scores)]


# ---------------------------------------------------------------------------
# Basic activation / deactivation
# ---------------------------------------------------------------------------

class TestActivation:
    def test_no_activation_below_high(self):
        det = StreamingDetector(threshold_high=0.6, threshold_low=0.3, patience_frames=5)
        states = feed(det, [0.5] * 20)
        assert all(not s.is_active for s in states)

    def test_activates_above_high(self):
        det = StreamingDetector(threshold_high=0.6, threshold_low=0.3, patience_frames=5,
                                ema_alpha=1.0)  # alpha=1 → ema == score instantly
        states = feed(det, [0.0] * 5 + [0.9])
        assert states[-1].is_active
        assert states[-1].start_frame == 5

    def test_start_frame_set_on_activation(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=3,
                                ema_alpha=1.0)
        states = feed(det, [0.0, 0.0, 0.8])
        assert states[2].start_frame == 2

    def test_no_deactivation_before_patience(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.3, patience_frames=5,
                                ema_alpha=1.0)
        # activate then go low for patience-1 frames
        scores = [0.8] + [0.1] * 4
        states = feed(det, scores)
        # still active — haven't hit patience yet
        assert states[-1].is_active

    def test_deactivates_after_patience(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.3, patience_frames=3,
                                ema_alpha=1.0)
        scores = [0.8] + [0.1] * 3
        states = feed(det, scores)
        assert not states[-1].is_active

    def test_low_counter_resets_on_high_score(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.3, patience_frames=3,
                                ema_alpha=1.0)
        # activate, go low 2 frames, recover, go low again — should not deactivate
        scores = [0.8, 0.1, 0.1, 0.8, 0.1, 0.1]
        states = feed(det, scores)
        assert states[-1].is_active  # only 2 low in a row after recovery


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_smoothing_reduces_spike(self):
        det = StreamingDetector(threshold_high=0.6, threshold_low=0.3, patience_frames=5,
                                ema_alpha=0.3)
        # single spike should be smoothed
        state = feed(det, [0.0] * 5 + [1.0])[-1]
        # ema after one step from 0: 0.3*1.0 + 0.7*0.0 = 0.3 < 0.6
        assert not state.is_active

    def test_ema_accumulates_over_time(self):
        det = StreamingDetector(threshold_high=0.6, threshold_low=0.3, patience_frames=5,
                                ema_alpha=0.5)
        states = feed(det, [1.0] * 10)
        # ema should converge to 1.0 → activation
        assert any(s.is_active for s in states)

    def test_ema_score_in_state(self):
        det = StreamingDetector(ema_alpha=1.0)
        state = det.update(0.7, 0)
        assert abs(state.ema_score - 0.7) < 1e-6

    def test_current_score_is_raw(self):
        det = StreamingDetector(ema_alpha=0.3)
        state = det.update(0.9, 0)
        assert abs(state.current_score - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Duration filter
# ---------------------------------------------------------------------------

class TestDurationFilter:
    def test_short_segment_not_recorded(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=3,
                                ema_alpha=1.0, min_frames=5)
        # activate for 1 frame then deactivate
        scores = [0.8] + [0.1] * 3
        feed(det, scores)
        assert len(det.completed) == 0

    def test_long_enough_segment_recorded(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=3,
                                ema_alpha=1.0, min_frames=3)
        scores = [0.8] * 5 + [0.1] * 3
        feed(det, scores)
        assert len(det.completed) == 1

    def test_max_frames_forces_deactivation(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=100,
                                ema_alpha=1.0, max_frames=5)
        scores = [0.9] * 10
        states = feed(det, scores)
        # should deactivate at frame 4 (0-based, len=5)
        assert not states[-1].is_active


# ---------------------------------------------------------------------------
# Completed segments
# ---------------------------------------------------------------------------

class TestCompletedSegments:
    def test_completed_segment_boundaries(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=2,
                                ema_alpha=1.0)
        # activate at 0, deactivate after 2 low frames (frames 3,4)
        scores = [0.8, 0.8, 0.8, 0.1, 0.1]
        feed(det, scores)
        assert len(det.completed) == 1
        seg = det.completed[0]
        assert seg.start_frame == 0
        assert seg.end_frame == 4

    def test_multiple_segments(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=2,
                                ema_alpha=1.0, min_frames=1)
        scores = [0.8, 0.8, 0.1, 0.1, 0.0, 0.8, 0.8, 0.1, 0.1]
        feed(det, scores)
        assert len(det.completed) == 2


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

class TestFinalize:
    def test_finalize_closes_active_segment(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=100,
                                ema_alpha=1.0, min_frames=1)
        feed(det, [0.9, 0.9, 0.9])
        det.finalize(last_frame_idx=2)
        assert len(det.completed) == 1
        assert det.completed[0].end_frame == 2

    def test_finalize_no_active_segment(self):
        det = StreamingDetector()
        state = det.finalize(last_frame_idx=10)
        assert not state.is_active
        assert len(det.completed) == 0

    def test_finalize_respects_min_frames(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=100,
                                ema_alpha=1.0, min_frames=10)
        feed(det, [0.9] * 3)
        det.finalize(last_frame_idx=2)
        assert len(det.completed) == 0  # too short


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        det = StreamingDetector(threshold_high=0.5, threshold_low=0.2, patience_frames=100,
                                ema_alpha=1.0)
        feed(det, [0.9] * 5)
        assert det._is_active
        det.reset()
        assert not det._is_active
        assert det._start_frame is None
        assert det._ema == 0.0
        assert det.completed == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_ema_alpha_zero(self):
        with pytest.raises(ValueError):
            StreamingDetector(ema_alpha=0.0)

    def test_invalid_ema_alpha_negative(self):
        with pytest.raises(ValueError):
            StreamingDetector(ema_alpha=-0.1)

    def test_threshold_low_gt_high_raises(self):
        with pytest.raises(ValueError):
            StreamingDetector(threshold_high=0.3, threshold_low=0.7)


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_config_basic(self):
        cfg = {
            "threshold_high": 0.7,
            "threshold_low": 0.3,
            "patience_frames": 20,
            "ema_alpha": 0.4,
            "fps": 25.0,
            "min_duration_sec": 2.0,
            "max_duration_sec": 10.0,
        }
        det = StreamingDetector.from_config(cfg)
        assert det.threshold_high == 0.7
        assert det.min_frames == 50   # 2.0 * 25
        assert det.max_frames == 250  # 10.0 * 25

    def test_from_config_defaults(self):
        det = StreamingDetector.from_config({})
        assert det.threshold_high == 0.6
        assert det.threshold_low == 0.3
        assert det.patience_frames == 15
