"""
tests/test_pose_model.py

Comprehensive pytest tests for:
  - src/data/pose_dataset.py  (PoseDataset, VideoSample, TBPTTChunk)
  - src/models/pose_model.py  (PoseHead, PoseGymRT)

External I/O (torch.load, load_annotations) is mocked via tmp_path fixtures
and monkeypatching so no real video files or annotation directories are required.

Run with:
    pytest tests/test_pose_model.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from src.data.pose_dataset import (
    POSE_DIM,
    PoseDataset,
    TBPTTChunk,
    VideoSample,
)
from src.models.pose_model import PoseGymRT, PoseHead

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


def make_features(n_frames: int, pose_dim: int = POSE_DIM) -> Tensor:
    """Return a deterministic float32 tensor of shape [n_frames, pose_dim]."""
    torch.manual_seed(0)
    return torch.randn(n_frames, pose_dim)


def make_video_sample(
    n_frames: int = 50,
    start_frame: int = 10,
    end_frame: int = 30,
    video_name: str = "Ball_001.mp4",
) -> VideoSample:
    """Build a VideoSample in memory without any file I/O."""
    features = make_features(n_frames)
    labels = torch.zeros(n_frames)
    labels[start_frame:end_frame] = 1.0
    return VideoSample(
        video_name=video_name,
        features=features,
        labels=labels,
        fps=25.0,
        has_element=True,
    )


def make_annotation(
    video_name: str,
    start_frame: int = 10,
    end_frame: int = 30,
    has_element: bool = True,
) -> dict:
    """Build a ConElGym annotation dict."""
    anns = []
    if has_element:
        anns = [
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "element_type": "element",
                "confidence": 1.0,
            }
        ]
    return {"video_name": video_name, "annotations": anns}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pose_dir(tmp_path: Path) -> Path:
    """Temporary directory that mimics data/pose_features/<split>/."""
    d = tmp_path / "pose_features"
    d.mkdir()
    return d


@pytest.fixture()
def ann_dir(tmp_path: Path) -> Path:
    """Temporary directory that mimics data/<split>/annotations/."""
    d = tmp_path / "annotations"
    d.mkdir()
    return d


def _write_annotation(ann_dir: Path, ann_dict: dict) -> None:
    stem = Path(ann_dict["video_name"]).stem
    json_path = ann_dir / f"{stem}.json"
    json_path.write_text(json.dumps(ann_dict), encoding="utf-8")


def _write_pt(pose_dir: Path, video_name: str, features: Tensor, fps: float = 25.0) -> None:
    stem = Path(video_name).stem
    torch.save(
        {"features": features, "fps": fps, "total_frames": features.shape[0]},
        pose_dir / f"{stem}.pt",
    )


# ---------------------------------------------------------------------------
# VideoSample dataclass
# ---------------------------------------------------------------------------


class TestVideoSample:
    def test_fields_accessible(self):
        sample = make_video_sample(n_frames=20, start_frame=5, end_frame=10)
        assert sample.video_name == "Ball_001.mp4"
        assert sample.features.shape == (20, POSE_DIM)
        assert sample.labels.shape == (20,)
        assert sample.fps == 25.0
        assert sample.has_element is True

    def test_labels_dtype_is_float32(self):
        sample = make_video_sample()
        assert sample.labels.dtype == torch.float32

    def test_features_dim_is_99(self):
        sample = make_video_sample(n_frames=10)
        assert sample.features.shape[1] == POSE_DIM


# ---------------------------------------------------------------------------
# PoseDataset construction
# ---------------------------------------------------------------------------


class TestPoseDatasetLoading:
    def test_len_matches_number_of_pt_files_with_annotations(
        self, pose_dir, ann_dir
    ):
        for i in range(3):
            name = f"Ball_{i:03d}.mp4"
            feats = make_features(40)
            _write_pt(pose_dir, name, feats)
            _write_annotation(ann_dir, make_annotation(name))

        ds = PoseDataset(pose_dir, ann_dir)
        assert len(ds) == 3

    def test_missing_pt_file_is_skipped(self, pose_dir, ann_dir):
        # Write annotation for two videos but only one .pt file
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4"))
        _write_annotation(ann_dir, make_annotation("Ball_002.mp4"))
        feats = make_features(30)
        _write_pt(pose_dir, "Ball_001.mp4", feats)

        ds = PoseDataset(pose_dir, ann_dir)
        assert len(ds) == 1

    def test_empty_features_dir_gives_empty_dataset(self, pose_dir, ann_dir):
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4"))
        ds = PoseDataset(pose_dir, ann_dir)
        assert len(ds) == 0

    def test_getitem_returns_video_sample(self, pose_dir, ann_dir):
        feats = make_features(50)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 10, 30))

        ds = PoseDataset(pose_dir, ann_dir)
        sample = ds[0]
        assert isinstance(sample, VideoSample)

    def test_features_shape_matches_pt_file(self, pose_dir, ann_dir):
        n_frames = 77
        feats = make_features(n_frames)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 10, 40))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].features.shape == (n_frames, POSE_DIM)

    def test_features_dtype_is_float32(self, pose_dir, ann_dir):
        # Store as float16 to verify the dataset casts to float32
        feats = make_features(30).half()
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 5, 20))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].features.dtype == torch.float32

    def test_fps_stored_in_sample(self, pose_dir, ann_dir):
        feats = make_features(30)
        _write_pt(pose_dir, "Ball_001.mp4", feats, fps=30.0)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 5, 20))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].fps == pytest.approx(30.0)

    def test_fps_defaults_to_25_when_not_in_file(self, pose_dir, ann_dir):
        # Manually save a .pt without 'fps' key
        feats = make_features(30)
        stem = "Ball_001"
        torch.save({"features": feats, "total_frames": 30}, pose_dir / f"{stem}.pt")
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 5, 20))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].fps == pytest.approx(25.0)

    def test_has_element_true_for_positive_sample(self, pose_dir, ann_dir):
        feats = make_features(50)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 10, 30, has_element=True))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].has_element is True

    def test_has_element_false_for_negative_sample(self, pose_dir, ann_dir):
        feats = make_features(50)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", has_element=False))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].has_element is False

    def test_labels_are_correct_for_annotated_segment(self, pose_dir, ann_dir):
        n_frames, s, e = 60, 15, 35
        feats = make_features(n_frames)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", s, e))

        ds = PoseDataset(pose_dir, ann_dir)
        labels = ds[0].labels

        # Frames inside segment must be 1, outside must be 0
        assert labels[:s].sum().item() == pytest.approx(0.0)
        assert labels[s:e].sum().item() == pytest.approx(float(e - s))
        assert labels[e:].sum().item() == pytest.approx(0.0)

    def test_labels_all_zero_for_negative_sample(self, pose_dir, ann_dir):
        feats = make_features(40)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", has_element=False))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds[0].labels.sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PoseDataset.pos_weight()
# ---------------------------------------------------------------------------


class TestPosWeight:
    def test_pos_weight_returns_float(self, pose_dir, ann_dir):
        feats = make_features(100)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 10, 20))

        ds = PoseDataset(pose_dir, ann_dir)
        pw = ds.pos_weight()
        assert isinstance(pw, float)

    def test_pos_weight_value_neg_over_pos(self, pose_dir, ann_dir):
        # 100 frames, 10 positive -> pos_weight = 90/10 = 9.0
        feats = make_features(100)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", 0, 10))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds.pos_weight() == pytest.approx(9.0)

    def test_pos_weight_returns_1_when_no_positives(self, pose_dir, ann_dir):
        feats = make_features(50)
        _write_pt(pose_dir, "Ball_001.mp4", feats)
        _write_annotation(ann_dir, make_annotation("Ball_001.mp4", has_element=False))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds.pos_weight() == pytest.approx(1.0)

    def test_pos_weight_aggregates_across_multiple_videos(self, pose_dir, ann_dir):
        # Video 1: 100 frames, 20 positive (frames 0-20)
        # Video 2: 100 frames, 0 positive
        # Total: 200 frames, 20 pos, 180 neg -> pos_weight = 180/20 = 9.0
        for i, (s, e, has_el) in enumerate([(0, 20, True), (0, 0, False)]):
            name = f"Ball_{i:03d}.mp4"
            _write_pt(pose_dir, name, make_features(100))
            _write_annotation(ann_dir, make_annotation(name, s, e, has_el))

        ds = PoseDataset(pose_dir, ann_dir)
        assert ds.pos_weight() == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# PoseDataset.iter_tbptt_chunks()
# ---------------------------------------------------------------------------


class TestIterTbpttChunks:
    def _collect_chunks(
        self, sample: VideoSample, chunk_size: int
    ) -> list[TBPTTChunk]:
        ds = PoseDataset.__new__(PoseDataset)
        ds.samples = [sample]
        return list(ds.iter_tbptt_chunks(sample, chunk_size))

    def test_single_chunk_when_frames_fit_exactly(self):
        sample = make_video_sample(n_frames=32, start_frame=0, end_frame=10)
        chunks = self._collect_chunks(sample, chunk_size=32)
        assert len(chunks) == 1

    def test_number_of_chunks_is_ceil_division(self):
        n_frames, chunk_size = 70, 32
        expected = math.ceil(n_frames / chunk_size)
        sample = make_video_sample(n_frames=n_frames)
        chunks = self._collect_chunks(sample, chunk_size=chunk_size)
        assert len(chunks) == expected

    def test_first_chunk_has_is_first_true(self):
        sample = make_video_sample(n_frames=64)
        chunks = self._collect_chunks(sample, chunk_size=32)
        assert chunks[0].is_first is True

    def test_subsequent_chunks_have_is_first_false(self):
        sample = make_video_sample(n_frames=64)
        chunks = self._collect_chunks(sample, chunk_size=32)
        for chunk in chunks[1:]:
            assert chunk.is_first is False

    def test_all_chunks_have_correct_feature_shape(self):
        sample = make_video_sample(n_frames=70)
        chunk_size = 32
        chunks = self._collect_chunks(sample, chunk_size=chunk_size)
        for chunk in chunks:
            assert chunk.features.shape == (chunk_size, POSE_DIM)

    def test_last_chunk_is_padded_to_chunk_size(self):
        # 70 frames, chunk_size=32 -> last chunk has 70-64=6 real, 26 padded
        sample = make_video_sample(n_frames=70)
        chunks = self._collect_chunks(sample, chunk_size=32)
        last = chunks[-1]
        assert last.features.shape[0] == 32
        # Padded region must be zeros
        real_len = last.valid_len
        assert last.features[real_len:].abs().sum().item() == pytest.approx(0.0)

    def test_last_chunk_label_padding_is_zero(self):
        sample = make_video_sample(n_frames=70)
        chunks = self._collect_chunks(sample, chunk_size=32)
        last = chunks[-1]
        real_len = last.valid_len
        assert last.labels[real_len:].sum().item() == pytest.approx(0.0)

    def test_valid_len_of_last_chunk_is_remainder(self):
        n_frames, chunk_size = 70, 32
        remainder = n_frames % chunk_size  # 6
        sample = make_video_sample(n_frames=n_frames)
        chunks = self._collect_chunks(sample, chunk_size=chunk_size)
        assert chunks[-1].valid_len == remainder

    def test_valid_len_of_non_last_chunk_equals_chunk_size(self):
        sample = make_video_sample(n_frames=64)
        chunks = self._collect_chunks(sample, chunk_size=32)
        assert chunks[0].valid_len == 32

    def test_features_content_matches_original_sample(self):
        sample = make_video_sample(n_frames=32)
        chunks = self._collect_chunks(sample, chunk_size=32)
        assert torch.allclose(chunks[0].features, sample.features)

    def test_single_frame_video_yields_one_chunk(self):
        sample = make_video_sample(n_frames=1, start_frame=0, end_frame=0)
        chunks = self._collect_chunks(sample, chunk_size=32)
        assert len(chunks) == 1
        assert chunks[0].valid_len == 1

    def test_chunk_size_larger_than_video_pads_full_chunk(self):
        n_frames = 5
        chunk_size = 20
        sample = make_video_sample(n_frames=n_frames)
        chunks = self._collect_chunks(sample, chunk_size=chunk_size)
        assert len(chunks) == 1
        assert chunks[0].features.shape == (chunk_size, POSE_DIM)
        assert chunks[0].valid_len == n_frames


# ---------------------------------------------------------------------------
# PoseHead: forward
# ---------------------------------------------------------------------------


class TestPoseHeadForward:
    @pytest.mark.parametrize("temporal_name", ["bilstm_attn", "bilstm"])
    def test_output_shape_batch_time(self, temporal_name):
        B, T = 2, 30
        head = PoseHead(hidden_dim=64, temporal_name=temporal_name, temporal_cfg={"n_layers": 1})
        x = torch.randn(B, T, POSE_DIM)
        out = head(x)
        assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"

    def test_output_shape_with_causal_tcn(self):
        B, T = 2, 40
        head = PoseHead(hidden_dim=64, temporal_name="causal_tcn")
        x = torch.randn(B, T, POSE_DIM)
        out = head(x)
        assert out.shape == (B, T)

    def test_output_is_not_nan(self):
        B, T = 2, 20
        head = PoseHead(hidden_dim=64, temporal_name="bilstm_attn", temporal_cfg={"n_layers": 1})
        x = torch.randn(B, T, POSE_DIM)
        out = head(x)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_unknown_temporal_name_raises_value_error(self):
        with pytest.raises(ValueError, match="bilstm_attn"):
            PoseHead(temporal_name="nonexistent_head")

    def test_forward_with_batch_size_one(self):
        head = PoseHead(hidden_dim=32, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        x = torch.randn(1, 15, POSE_DIM)
        out = head(x)
        assert out.shape == (1, 15)

    def test_forward_with_single_timestep(self):
        head = PoseHead(hidden_dim=32, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        x = torch.randn(2, 1, POSE_DIM)
        out = head(x)
        assert out.shape == (2, 1)

    def test_projection_maps_99_to_hidden_dim(self):
        hidden_dim = 128
        head = PoseHead(hidden_dim=hidden_dim, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        x = torch.randn(1, 5, POSE_DIM)
        projected = head._project(x)
        assert projected.shape == (1, 5, hidden_dim)


# ---------------------------------------------------------------------------
# PoseHead: forward_train (TBPTT)
# ---------------------------------------------------------------------------


class TestPoseHeadForwardTrain:
    def _make_head(self) -> PoseHead:
        return PoseHead(
            hidden_dim=64,
            temporal_name="bilstm_attn",
            temporal_cfg={"n_layers": 1},
        )

    def test_returns_logits_and_state(self):
        head = self._make_head()
        x = torch.randn(2, 20, POSE_DIM)
        # BiLSTM TBPTT: first chunk always uses state=None (bidirectional init)
        logits, new_state = head.forward_train(x, None)
        assert logits.shape == (2, 20)
        assert new_state is not None

    def test_state_is_none_on_first_chunk(self):
        head = self._make_head()
        x = torch.randn(2, 20, POSE_DIM)
        logits, new_state = head.forward_train(x, None)
        assert logits.shape == (2, 20)
        assert new_state is not None

    def test_state_threads_between_chunks(self):
        """Hidden state from chunk N is accepted by chunk N+1 without error."""
        head = self._make_head()
        B, T = 1, 16
        x1 = torch.randn(B, T, POSE_DIM)
        x2 = torch.randn(B, T, POSE_DIM)
        logits1, state1 = head.forward_train(x1, None)
        logits2, state2 = head.forward_train(x2, state1)
        assert logits2.shape == (B, T)

    def test_detach_state_before_next_chunk_does_not_break_forward(self):
        """Verify that detaching the TBPTT state (standard training practice) works."""
        head = self._make_head()
        B, T = 1, 16
        x = torch.randn(B, T, POSE_DIM)
        _, state = head.forward_train(x, None)
        # Detach all tensors in the state tuple before passing to next chunk
        detached = tuple(s.detach() for s in state)
        logits, _ = head.forward_train(torch.randn(B, T, POSE_DIM), detached)
        assert logits.shape == (B, T)

    def test_causal_tcn_forward_train_returns_none_state(self):
        head = PoseHead(hidden_dim=64, temporal_name="causal_tcn")
        x = torch.randn(2, 20, POSE_DIM)
        logits, state = head.forward_train(x, None)
        assert logits.shape == (2, 20)
        assert state is None

    def test_bilstm_forward_train_returns_tuple_state(self):
        head = PoseHead(
            hidden_dim=64,
            temporal_name="bilstm",
            temporal_cfg={"n_layers": 1},
        )
        x = torch.randn(2, 20, POSE_DIM)
        _, state = head.forward_train(x, None)
        # BiLSTM state: (h, c), each [n_layers, B, hidden_dim]
        assert isinstance(state, tuple)
        assert len(state) == 2


# ---------------------------------------------------------------------------
# PoseHead: forward_step (streaming)
# ---------------------------------------------------------------------------


class TestPoseHeadForwardStep:
    def _make_head(self) -> PoseHead:
        return PoseHead(
            hidden_dim=64,
            temporal_name="bilstm_attn",
            temporal_cfg={"n_layers": 1},
        )

    def test_output_shape_is_batch(self):
        head = self._make_head()
        B = 3
        x = torch.randn(B, POSE_DIM)
        state = head.init_state(B, DEVICE)
        logit, _ = head.forward_step(x, state)
        assert logit.shape == (B,)

    def test_output_is_not_nan(self):
        head = self._make_head()
        x = torch.randn(2, POSE_DIM)
        state = head.init_state(2, DEVICE)
        logit, _ = head.forward_step(x, state)
        assert not torch.isnan(logit).any()

    def test_state_updates_after_each_step(self):
        """Calling forward_step twice gives different states."""
        head = self._make_head()
        B = 1
        state = head.init_state(B, DEVICE)
        x = torch.randn(B, POSE_DIM)
        _, state1 = head.forward_step(x, state)
        _, state2 = head.forward_step(x, state1)
        # The h tensor in state should have changed
        h_init = state[0]
        h1 = state1[0]
        h2 = state2[0]
        assert not torch.allclose(h_init, h1), "State did not update after step 1"
        assert not torch.allclose(h1, h2), "State did not update after step 2"

    def test_streaming_ten_steps_produces_ten_logits(self):
        head = self._make_head()
        B = 1
        state = head.init_state(B, DEVICE)
        logits = []
        for _ in range(10):
            x = torch.randn(B, POSE_DIM)
            logit, state = head.forward_step(x, state)
            logits.append(logit)
        assert len(logits) == 10
        assert all(l.shape == (B,) for l in logits)

    def test_bilstm_forward_step_output_shape(self):
        head = PoseHead(
            hidden_dim=64,
            temporal_name="bilstm",
            temporal_cfg={"n_layers": 1},
        )
        B = 2
        x = torch.randn(B, POSE_DIM)
        state = head.init_state(B, DEVICE)
        logit, _ = head.forward_step(x, state)
        assert logit.shape == (B,)

    def test_causal_tcn_forward_step_raises_not_implemented(self):
        head = PoseHead(hidden_dim=64, temporal_name="causal_tcn")
        x = torch.randn(1, POSE_DIM)
        state = head.init_state(1, DEVICE)
        with pytest.raises(NotImplementedError):
            head.forward_step(x, state)


# ---------------------------------------------------------------------------
# PoseHead: init_state
# ---------------------------------------------------------------------------


class TestPoseHeadInitState:
    def test_bilstm_attn_state_is_tuple_of_three_tensors(self):
        head = PoseHead(
            hidden_dim=64,
            temporal_name="bilstm_attn",
            temporal_cfg={"n_layers": 1},
        )
        state = head.init_state(2, DEVICE)
        assert isinstance(state, tuple)
        assert len(state) == 3  # h, c, attention_buffer

    def test_bilstm_state_is_tuple_of_two_tensors(self):
        head = PoseHead(
            hidden_dim=64,
            temporal_name="bilstm",
            temporal_cfg={"n_layers": 1},
        )
        state = head.init_state(2, DEVICE)
        assert isinstance(state, tuple)
        assert len(state) == 2  # h, c

    def test_init_state_tensors_are_zero(self):
        head = PoseHead(
            hidden_dim=64,
            temporal_name="bilstm",
            temporal_cfg={"n_layers": 1},
        )
        h, c = head.init_state(3, DEVICE)
        assert h.sum().item() == pytest.approx(0.0)
        assert c.sum().item() == pytest.approx(0.0)

    def test_init_state_h_shape_matches_n_layers_and_batch(self):
        n_layers, B, hidden_dim = 2, 4, 64
        head = PoseHead(
            hidden_dim=hidden_dim,
            temporal_name="bilstm",
            # Must pass hidden_dim explicitly; default is 128
            temporal_cfg={"n_layers": n_layers, "hidden_dim": hidden_dim},
        )
        h, c = head.init_state(B, DEVICE)
        assert h.shape == (n_layers, B, hidden_dim)
        assert c.shape == (n_layers, B, hidden_dim)

    def test_causal_tcn_init_state_returns_none(self):
        head = PoseHead(hidden_dim=64, temporal_name="causal_tcn")
        state = head.init_state(1, DEVICE)
        assert state is None


# ---------------------------------------------------------------------------
# PoseGymRT: construction and introspection
# ---------------------------------------------------------------------------


class TestPoseGymRTConstruction:
    def test_temporal_attribute_is_pose_head(self):
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        assert isinstance(model.temporal, PoseHead)

    def test_default_temporal_name_is_bilstm_attn(self):
        model = PoseGymRT(hidden_dim=64, temporal_cfg={"n_layers": 1})
        assert model._temporal_name == "bilstm_attn"

    def test_invalid_temporal_name_raises_value_error(self):
        with pytest.raises(ValueError):
            PoseGymRT(temporal_name="bad_head")

    def test_count_parameters_returns_positive_int(self):
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        n_params = model.count_parameters()
        assert isinstance(n_params, int)
        assert n_params > 0

    def test_count_parameters_excludes_non_trainable_params(self):
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        # Freeze all params; count should drop to 0
        for p in model.parameters():
            p.requires_grad_(False)
        assert model.count_parameters() == 0

    def test_size_mb_returns_positive_float(self):
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        mb = model.size_mb()
        assert isinstance(mb, float)
        assert mb > 0.0

    def test_size_mb_consistent_with_count_parameters(self):
        # float32 params: n_params * 4 bytes / 1024^2
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        expected = model.count_parameters() * 4 / (1024 ** 2)
        assert model.size_mb() == pytest.approx(expected, rel=1e-4)

    def test_larger_hidden_dim_has_more_parameters(self):
        model_small = PoseGymRT(hidden_dim=32, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        model_large = PoseGymRT(hidden_dim=128, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        assert model_large.count_parameters() > model_small.count_parameters()


# ---------------------------------------------------------------------------
# PoseGymRT: forward
# ---------------------------------------------------------------------------


class TestPoseGymRTForward:
    def test_forward_output_shape(self):
        B, T = 2, 25
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        x = torch.randn(B, T, POSE_DIM)
        out = model(x)
        assert out.shape == (B, T)

    def test_forward_output_not_nan(self):
        B, T = 2, 25
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        x = torch.randn(B, T, POSE_DIM)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_forward_with_bilstm_head(self):
        B, T = 3, 10
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        x = torch.randn(B, T, POSE_DIM)
        out = model(x)
        assert out.shape == (B, T)

    def test_forward_with_causal_tcn_head(self):
        B, T = 2, 50
        model = PoseGymRT(hidden_dim=64, temporal_name="causal_tcn")
        x = torch.randn(B, T, POSE_DIM)
        out = model(x)
        assert out.shape == (B, T)

    def test_forward_delegates_to_temporal(self):
        """PoseGymRT.forward must produce identical results to PoseHead.forward."""
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        model.eval()
        x = torch.randn(2, 15, POSE_DIM)
        with torch.no_grad():
            out_gymrt = model(x)
            out_head = model.temporal(x)
        assert torch.allclose(out_gymrt, out_head)


# ---------------------------------------------------------------------------
# PoseGymRT: init_state and forward_frame (streaming)
# ---------------------------------------------------------------------------


class TestPoseGymRTStreaming:
    def test_init_state_returns_non_none_for_bilstm(self):
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        state = model.init_state(batch_size=1, device=DEVICE)
        assert state is not None

    def test_init_state_calls_copy_fwd_weights(self):
        """init_state must synchronise fwd weights before streaming."""
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        with patch.object(
            model.temporal, "_copy_fwd_weights", wraps=model.temporal._copy_fwd_weights
        ) as mock_copy:
            model.init_state(batch_size=1, device=DEVICE)
            mock_copy.assert_called_once()

    def test_forward_frame_output_shape(self):
        B = 2
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        state = model.init_state(batch_size=B, device=DEVICE)
        x = torch.randn(B, POSE_DIM)
        logit, _ = model.forward_frame(x, state)
        assert logit.shape == (B,)

    def test_forward_frame_state_updates(self):
        B = 1
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        state0 = model.init_state(batch_size=B, device=DEVICE)
        x = torch.randn(B, POSE_DIM)
        _, state1 = model.forward_frame(x, state0)
        # h component should differ after one step
        assert not torch.allclose(state0[0], state1[0])

    def test_streaming_produces_one_logit_per_frame(self):
        B, n_steps = 1, 15
        model = PoseGymRT(hidden_dim=64, temporal_name="bilstm_attn",
                          temporal_cfg={"n_layers": 1})
        state = model.init_state(batch_size=B, device=DEVICE)
        all_logits = []
        for _ in range(n_steps):
            x = torch.randn(B, POSE_DIM)
            logit, state = model.forward_frame(x, state)
            all_logits.append(logit)
        assert len(all_logits) == n_steps
        assert all(l.shape == (B,) for l in all_logits)

    def test_init_state_device_inferred_from_parameters(self):
        """When device=None, init_state infers device from model parameters."""
        model = PoseGymRT(hidden_dim=32, temporal_name="bilstm", temporal_cfg={"n_layers": 1})
        state = model.init_state(batch_size=1)  # device=None
        h, c = state
        assert h.device == next(model.parameters()).device
