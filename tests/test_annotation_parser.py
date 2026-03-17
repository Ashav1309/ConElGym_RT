"""Тесты для annotation_parser."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.annotation_parser import Annotation, load_annotation, load_annotations


@pytest.fixture
def tmp_ann_dir(tmp_path):
    return tmp_path


def write_json(directory: Path, filename: str, data: dict) -> Path:
    p = directory / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_annotation
# ---------------------------------------------------------------------------

def test_positive_annotation(tmp_ann_dir):
    write_json(tmp_ann_dir, "Ball_001.json", {
        "video_name": "Ball_001.mp4",
        "annotations": [{"start_frame": 50, "end_frame": 173,
                         "element_type": "element", "confidence": 1.0}],
    })
    ann = load_annotation(tmp_ann_dir / "Ball_001.json")
    assert ann.video_name == "Ball_001.mp4"
    assert ann.has_element is True
    assert ann.start_frame == 50
    assert ann.end_frame == 173
    assert ann.duration_frames() == 123


def test_negative_annotation_empty_list(tmp_ann_dir):
    write_json(tmp_ann_dir, "Ball_006.json", {
        "video_name": "Ball_006.mp4",
        "annotations": [],
    })
    ann = load_annotation(tmp_ann_dir / "Ball_006.json")
    assert ann.has_element is False
    assert ann.start_frame == 0
    assert ann.end_frame == 0
    assert ann.duration_frames() == 0


def test_start_less_than_end(tmp_ann_dir):
    write_json(tmp_ann_dir, "bad.json", {
        "video_name": "bad.mp4",
        "annotations": [{"start_frame": 100, "end_frame": 50,
                         "element_type": "element", "confidence": 1.0}],
    })
    with pytest.raises(ValueError, match="start_frame"):
        load_annotation(tmp_ann_dir / "bad.json")


def test_start_equals_end_raises(tmp_ann_dir):
    write_json(tmp_ann_dir, "bad2.json", {
        "video_name": "bad2.mp4",
        "annotations": [{"start_frame": 100, "end_frame": 100,
                         "element_type": "element", "confidence": 1.0}],
    })
    with pytest.raises(ValueError):
        load_annotation(tmp_ann_dir / "bad2.json")


# ---------------------------------------------------------------------------
# frame_label
# ---------------------------------------------------------------------------

def test_frame_label_inside(tmp_ann_dir):
    ann = Annotation(video_name="x.mp4", has_element=True, start_frame=50, end_frame=100)
    assert ann.frame_label(50) == 1   # включительно start
    assert ann.frame_label(75) == 1
    assert ann.frame_label(99) == 1   # последний внутри


def test_frame_label_outside(tmp_ann_dir):
    ann = Annotation(video_name="x.mp4", has_element=True, start_frame=50, end_frame=100)
    assert ann.frame_label(49) == 0   # до начала
    assert ann.frame_label(100) == 0  # end не включительно


def test_frame_label_negative_video():
    ann = Annotation(video_name="x.mp4", has_element=False, start_frame=0, end_frame=0)
    assert ann.frame_label(0) == 0
    assert ann.frame_label(100) == 0


# ---------------------------------------------------------------------------
# load_annotations
# ---------------------------------------------------------------------------

def test_load_annotations_multiple(tmp_ann_dir):
    write_json(tmp_ann_dir, "Ball_001.json", {
        "video_name": "Ball_001.mp4",
        "annotations": [{"start_frame": 10, "end_frame": 50,
                         "element_type": "element", "confidence": 1.0}],
    })
    write_json(tmp_ann_dir, "Ball_006.json", {
        "video_name": "Ball_006.mp4",
        "annotations": [],
    })
    anns = load_annotations(tmp_ann_dir)
    assert len(anns) == 2
    assert anns["Ball_001.mp4"].has_element is True
    assert anns["Ball_006.mp4"].has_element is False


def test_load_annotations_empty_dir(tmp_ann_dir):
    anns = load_annotations(tmp_ann_dir)
    assert anns == {}
