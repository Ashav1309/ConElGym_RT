"""Тесты для frame_dataset."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.frame_dataset import FrameDataset


def make_annotation(directory: Path, stem: str, start: int, end: int) -> None:
    has = start < end
    ann = {
        "video_name": f"{stem}.mp4",
        "annotations": [
            {"start_frame": start, "end_frame": end,
             "element_type": "element", "confidence": 1.0}
        ] if has else [],
    }
    (directory / f"{stem}.json").write_text(json.dumps(ann), encoding="utf-8")


def make_features(directory: Path, stem: str, n_frames: int, dim: int = 8,
                  fps: float = 25.0, backbone: str = "mob") -> None:
    data = {
        "features": torch.randn(n_frames, dim),
        "fps": fps,
        "total_frames": n_frames,
        "backbone": backbone,
    }
    torch.save(data, directory / f"{stem}_{backbone}.pt")


@pytest.fixture
def dataset_dirs(tmp_path):
    feat_dir = tmp_path / "features"
    ann_dir = tmp_path / "annotations"
    feat_dir.mkdir()
    ann_dir.mkdir()
    return feat_dir, ann_dir


# ---------------------------------------------------------------------------
# FrameDataset
# ---------------------------------------------------------------------------

def test_loads_positive_video(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=10, end=50)
    make_features(feat_dir, "Ball_001", n_frames=100)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    assert len(ds) == 1
    sample = ds[0]
    assert sample.has_element is True
    assert sample.features.shape == (100, 8)
    assert sample.labels.shape == (100,)
    assert sample.labels.sum().item() == 40  # кадры 10..49


def test_loads_negative_video(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_006", start=0, end=0)
    make_features(feat_dir, "Ball_006", n_frames=80)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    assert len(ds) == 1
    sample = ds[0]
    assert sample.has_element is False
    assert sample.labels.sum().item() == 0.0


def test_label_boundaries(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=5, end=10)
    make_features(feat_dir, "Ball_001", n_frames=20)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    labels = ds[0].labels.tolist()
    assert labels[4] == 0.0   # до start
    assert labels[5] == 1.0   # start включительно
    assert labels[9] == 1.0   # end-1
    assert labels[10] == 0.0  # end не включительно


def test_skips_missing_features(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=10, end=50)
    # features НЕ созданы → видео пропускается

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    assert len(ds) == 0


def test_pos_weight_positive(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    # 100 кадров, 10 позитивных → pos_weight = 90/10 = 9.0
    make_annotation(ann_dir, "Ball_001", start=0, end=10)
    make_features(feat_dir, "Ball_001", n_frames=100)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    assert abs(ds.pos_weight() - 9.0) < 0.01


def test_pos_weight_all_negative(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_006", start=0, end=0)
    make_features(feat_dir, "Ball_006", n_frames=50)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    assert ds.pos_weight() == 1.0  # нет позитивных → fallback


# ---------------------------------------------------------------------------
# TBPTT chunking
# ---------------------------------------------------------------------------

def test_tbptt_chunk_count(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=10, end=50)
    make_features(feat_dir, "Ball_001", n_frames=100)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    chunks = list(ds.iter_tbptt_chunks(ds[0], chunk_size=30))
    # 100 кадров / 30 = ceil(3.33) = 4 чанка
    assert len(chunks) == 4


def test_tbptt_first_chunk_flag(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=10, end=50)
    make_features(feat_dir, "Ball_001", n_frames=100)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    chunks = list(ds.iter_tbptt_chunks(ds[0], chunk_size=30))
    assert chunks[0].is_first is True
    assert all(not c.is_first for c in chunks[1:])


def test_tbptt_chunk_size_padded(dataset_dirs):
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=10, end=50)
    make_features(feat_dir, "Ball_001", n_frames=100)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    chunks = list(ds.iter_tbptt_chunks(ds[0], chunk_size=30))
    # все чанки должны быть chunk_size=30 (последний дополняется паддингом)
    for chunk in chunks:
        assert chunk.features.shape[0] == 30
        assert chunk.labels.shape[0] == 30


def test_tbptt_short_video(dataset_dirs):
    """Видео короче chunk_size → 1 чанк с паддингом."""
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=2, end=8)
    make_features(feat_dir, "Ball_001", n_frames=15)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    chunks = list(ds.iter_tbptt_chunks(ds[0], chunk_size=256))
    assert len(chunks) == 1
    assert chunks[0].features.shape[0] == 256
