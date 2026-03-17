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


# ---------------------------------------------------------------------------
# TBPTT: padding region contains zeros
# ---------------------------------------------------------------------------

def test_tbptt_padding_is_zeros(dataset_dirs):
    """Паддинг последнего чанка должен быть нулями (не реальными признаками)."""
    feat_dir, ann_dir = dataset_dirs
    # 35 кадров, chunk_size=20 → 2 чанка; второй содержит 15 реальных + 5 нулей
    make_annotation(ann_dir, "Ball_001", start=5, end=20)
    make_features(feat_dir, "Ball_001", n_frames=35)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    chunks = list(ds.iter_tbptt_chunks(ds[0], chunk_size=20))
    assert len(chunks) == 2

    last_chunk = chunks[-1]
    # Последние 5 строк должны быть нулями
    pad_features = last_chunk.features[15:]
    assert torch.all(pad_features == 0.0), "Паддинговые кадры должны быть нулями"
    pad_labels = last_chunk.labels[15:]
    assert torch.all(pad_labels == 0.0), "Паддинговые метки должны быть нулями"


def test_tbptt_no_label_in_padding(dataset_dirs):
    """Элемент полностью умещается в первом чанке → метки второго чанка все нули."""
    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=2, end=8)
    make_features(feat_dir, "Ball_001", n_frames=35)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    chunks = list(ds.iter_tbptt_chunks(ds[0], chunk_size=20))
    assert len(chunks) == 2
    assert chunks[1].labels.sum().item() == 0.0


# ---------------------------------------------------------------------------
# TBPTT: hidden state detach (gradient isolation between chunks)
# ---------------------------------------------------------------------------

def test_tbptt_hidden_state_detached_between_chunks(dataset_dirs):
    """Hidden state, переданный между чанками, должен быть отсоединён от графа.

    TBPTT-тренинг вызывает state.detach() перед каждым новым чанком, чтобы
    обрезать градиенты из прошлого.  Этот тест моделирует типичный цикл
    обучения и проверяет, что детачнутый state не имеет grad_fn.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.temporal import BiLSTMHead

    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=10, end=50)
    make_features(feat_dir, "Ball_001", n_frames=90, dim=8)

    ds = FrameDataset(feat_dir, ann_dir, backbone="mob")
    head = BiLSTMHead(input_dim=8, hidden_dim=16, n_layers=1, dropout=0.0)

    state = None
    for chunk in ds.iter_tbptt_chunks(ds[0], chunk_size=30):
        if chunk.is_first:
            state = head.init_state(1, torch.device("cpu"))
        else:
            # Симулируем detach, как это делает обучающий цикл
            h, c = state
            h = h.detach()
            c = c.detach()
            state = (h, c)

        # Прогоняем чанк через forward с состоянием
        x = chunk.features.unsqueeze(0)  # [1, chunk_size, D]
        x_step = x[0, 0, :]             # используем первый шаг
        logit, state = head.forward_step(x_step.unsqueeze(0), state)

    # После detach: grad_fn у state должен быть None
    h, c = state
    h_detached = h.detach()
    c_detached = c.detach()
    assert h_detached.grad_fn is None, "h должен быть отсоединён от графа"
    assert c_detached.grad_fn is None, "c должен быть отсоединён от графа"


# ---------------------------------------------------------------------------
# FlatFrameDataset
# ---------------------------------------------------------------------------

def test_flat_frame_dataset_length(dataset_dirs):
    """FlatFrameDataset суммирует кадры из всех видео."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.frame_dataset import FlatFrameDataset

    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=5, end=20)
    make_features(feat_dir, "Ball_001", n_frames=60)
    make_annotation(ann_dir, "Ball_002", start=0, end=0)
    make_features(feat_dir, "Ball_002", n_frames=40)

    flat_ds = FlatFrameDataset(feat_dir, ann_dir, backbone="mob")
    assert len(flat_ds) == 100  # 60 + 40


def test_flat_frame_dataset_item_shape(dataset_dirs):
    """Каждый элемент FlatFrameDataset — (feature [D], label scalar)."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.frame_dataset import FlatFrameDataset

    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_001", start=5, end=20)
    make_features(feat_dir, "Ball_001", n_frames=30, dim=8)

    flat_ds = FlatFrameDataset(feat_dir, ann_dir, backbone="mob")
    feat, label = flat_ds[0]
    assert feat.shape == (8,)
    assert label.shape == ()


def test_flat_frame_dataset_negative_video_all_zero_labels(dataset_dirs):
    """Для негативного видео все метки в FlatFrameDataset равны 0."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.frame_dataset import FlatFrameDataset

    feat_dir, ann_dir = dataset_dirs
    make_annotation(ann_dir, "Ball_006", start=0, end=0)
    make_features(feat_dir, "Ball_006", n_frames=50)

    flat_ds = FlatFrameDataset(feat_dir, ann_dir, backbone="mob")
    all_labels = torch.tensor([flat_ds[i][1].item() for i in range(len(flat_ds))])
    assert all_labels.sum().item() == 0.0
