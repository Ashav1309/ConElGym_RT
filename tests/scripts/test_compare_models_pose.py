"""
Tests for the three pose-related functions added to compare_models.py:
  - is_pose_config(cfg)
  - load_pose_model_from_checkpoint(path, cfg, device)
  - evaluate_pose_model(path, cfg, device, split)

No real checkpoint files or video data are required.
All file I/O, model loading, and evaluation are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_cfg(
    *,
    temporal_head: str = "bilstm_attn",
    hidden_dim: int = 256,
    n_layers: int = 2,
    dropout: float = 0.3,
    v2_root: str = "/fake/v2",
    valid_annotations: str = "valid/annotations",
    test_annotations: str = "test/annotations",
    threshold: float = 0.5,
    min_duration_sec: float = 1.5,
    max_duration_sec: float = 12.0,
) -> dict:
    """Build a minimal pose-model config dict (mirrors real YAML structure)."""
    return {
        "model": {
            "model_type": "pose",
            "temporal_head": temporal_head,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
        },
        "data": {
            "v2_root": v2_root,
            "valid_annotations": valid_annotations,
            "test_annotations": test_annotations,
        },
        "postprocess": {
            "threshold": threshold,
            "min_duration_sec": min_duration_sec,
            "max_duration_sec": max_duration_sec,
        },
    }


def _non_pose_cfg() -> dict:
    """Config dict that does NOT declare model_type=pose."""
    return {
        "model": {
            "backbone": "efficientnet_b0",
            "temporal_head": "bilstm_attn",
        }
    }


# ---------------------------------------------------------------------------
# is_pose_config
# ---------------------------------------------------------------------------

class TestIsPoseConfig:

    def test_returns_true_when_model_type_is_pose(self):
        from src.scripts.compare_models import is_pose_config
        cfg = _pose_cfg()
        assert is_pose_config(cfg) is True

    def test_returns_false_when_model_type_is_missing(self):
        from src.scripts.compare_models import is_pose_config
        cfg = _non_pose_cfg()
        assert is_pose_config(cfg) is False

    def test_returns_false_when_model_type_is_different_string(self):
        from src.scripts.compare_models import is_pose_config
        cfg = {"model": {"model_type": "frame"}}
        assert is_pose_config(cfg) is False

    def test_returns_false_when_model_key_missing_entirely(self):
        from src.scripts.compare_models import is_pose_config
        assert is_pose_config({}) is False

    def test_returns_false_when_model_section_empty(self):
        from src.scripts.compare_models import is_pose_config
        assert is_pose_config({"model": {}}) is False

    def test_case_sensitive_value_check(self):
        # "Pose" (capital P) must not match
        from src.scripts.compare_models import is_pose_config
        cfg = {"model": {"model_type": "Pose"}}
        assert is_pose_config(cfg) is False


# ---------------------------------------------------------------------------
# load_pose_model_from_checkpoint
# ---------------------------------------------------------------------------

class TestLoadPoseModelFromCheckpoint:

    def _make_real_checkpoint(self, tmp_path: Path, cfg: dict) -> Path:
        """Build a real PoseGymRT, save its state_dict, return checkpoint path."""
        from src.models.pose_model import PoseGymRT
        model_cfg = cfg["model"]
        temporal_cfg = {
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "n_layers":   model_cfg.get("n_layers", 2),
            "dropout":    model_cfg.get("dropout", 0.3),
        }
        model = PoseGymRT(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            temporal_name=model_cfg.get("temporal_head", "bilstm_attn"),
            temporal_cfg=temporal_cfg,
        )
        ckpt_path = tmp_path / "pose_bilstm_attn_seed42_best.pt"
        torch.save({"model_state": model.state_dict()}, str(ckpt_path))
        return ckpt_path

    def test_returns_pose_gymrt_instance(self, tmp_path):
        from src.models.pose_model import PoseGymRT
        from src.scripts.compare_models import load_pose_model_from_checkpoint

        # Use small dims to keep the test fast
        cfg = _pose_cfg(hidden_dim=32, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg)
        device = torch.device("cpu")

        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        assert isinstance(model, PoseGymRT)

    def test_model_is_in_eval_mode_after_load(self, tmp_path):
        from src.scripts.compare_models import load_pose_model_from_checkpoint

        cfg = _pose_cfg(hidden_dim=32, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg)
        device = torch.device("cpu")

        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        assert not model.training

    def test_model_is_on_correct_device(self, tmp_path):
        from src.scripts.compare_models import load_pose_model_from_checkpoint

        cfg = _pose_cfg(hidden_dim=32, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg)
        device = torch.device("cpu")

        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        param_device = next(model.parameters()).device
        assert param_device == device

    def test_model_uses_hidden_dim_from_config(self, tmp_path):
        from src.scripts.compare_models import load_pose_model_from_checkpoint
        from src.models.pose_model import PoseGymRT

        cfg = _pose_cfg(hidden_dim=64, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg)
        device = torch.device("cpu")

        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        # PoseHead projection Linear(99 → hidden_dim) — check output dim
        assert model.temporal.projection[1].out_features == 64

    def test_default_hidden_dim_256_when_not_in_config(self, tmp_path):
        """When hidden_dim is absent from config, function defaults to 256."""
        from src.scripts.compare_models import load_pose_model_from_checkpoint

        cfg = _pose_cfg(hidden_dim=256, n_layers=1)
        # Strip hidden_dim from model section
        del cfg["model"]["hidden_dim"]
        ckpt_path = self._make_real_checkpoint(
            tmp_path,
            # Use 256 to build the checkpoint that matches default
            _pose_cfg(hidden_dim=256, n_layers=1),
        )
        device = torch.device("cpu")

        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        assert model.temporal.projection[1].out_features == 256

    def test_causal_tcn_head_does_not_pass_n_layers(self, tmp_path):
        """causal_tcn temporal head must not receive hidden_dim or n_layers kwargs."""
        from src.scripts.compare_models import load_pose_model_from_checkpoint
        from src.models.pose_model import PoseGymRT

        cfg = _pose_cfg(temporal_head="causal_tcn", hidden_dim=32, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg)
        device = torch.device("cpu")

        # Should not raise even though n_layers is in config
        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        assert isinstance(model, PoseGymRT)

    def test_loaded_weights_match_saved_weights(self, tmp_path):
        """State dict loaded from disk is bit-identical to original."""
        from src.scripts.compare_models import load_pose_model_from_checkpoint
        from src.models.pose_model import PoseGymRT

        cfg = _pose_cfg(hidden_dim=32, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg)
        device = torch.device("cpu")

        # Load the original state dict directly
        saved_state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)["model_state"]

        model = load_pose_model_from_checkpoint(ckpt_path, cfg, device)

        for key, tensor in saved_state.items():
            assert torch.equal(model.state_dict()[key], tensor), f"Mismatch in parameter {key}"

    def test_raises_if_checkpoint_has_wrong_architecture(self, tmp_path):
        """Loading a checkpoint whose dims don't match the config must raise RuntimeError."""
        from src.scripts.compare_models import load_pose_model_from_checkpoint

        # Save a checkpoint with hidden_dim=32
        cfg_save = _pose_cfg(hidden_dim=32, n_layers=1)
        ckpt_path = self._make_real_checkpoint(tmp_path, cfg_save)

        # Try to load it with hidden_dim=128 — shapes will be incompatible
        cfg_wrong = _pose_cfg(hidden_dim=128, n_layers=1)
        device = torch.device("cpu")

        with pytest.raises(RuntimeError):
            load_pose_model_from_checkpoint(ckpt_path, cfg_wrong, device)


# ---------------------------------------------------------------------------
# evaluate_pose_model
# ---------------------------------------------------------------------------

def _make_pose_checkpoint(tmp_path: Path, hidden_dim: int = 32) -> Path:
    """Persist a minimal PoseGymRT checkpoint and return its path."""
    from src.models.pose_model import PoseGymRT
    model = PoseGymRT(
        hidden_dim=hidden_dim,
        temporal_name="bilstm_attn",
        temporal_cfg={"hidden_dim": hidden_dim, "n_layers": 1, "dropout": 0.0},
    )
    ckpt_path = tmp_path / "pose_bilstm_attn_seed42_best.pt"
    torch.save({"model_state": model.state_dict()}, str(ckpt_path))
    return ckpt_path


def _make_pose_feature_file(tmp_path: Path, video_stem: str, n_frames: int = 50) -> Path:
    """Write a fake pose feature .pt file and return its path."""
    features = torch.zeros(n_frames, 99)
    feat_path = tmp_path / f"{video_stem}.pt"
    torch.save({"features": features, "fps": 25.0, "total_frames": n_frames}, str(feat_path))
    return feat_path


def _make_annotation_json(ann_dir: Path, video_name: str, start_frame: int = 10, end_frame: int = 40) -> None:
    """Write a minimal annotation JSON into ann_dir."""
    ann_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_name": video_name,
        "annotations": [
            {"start_frame": start_frame, "end_frame": end_frame, "element_type": "element", "confidence": 1.0}
        ],
    }
    (ann_dir / f"{Path(video_name).stem}.json").write_text(json.dumps(payload), encoding="utf-8")


class TestEvaluatePoseModel:

    def _build_env(self, tmp_path: Path, split: str = "valid") -> tuple[Path, dict]:
        """
        Set up a minimal filesystem environment:
          tmp_path/pose_features/<split>/Ball_001.pt
          tmp_path/v2/<split>/annotations/Ball_001.json

        Returns (ckpt_path, cfg).
        """
        video_name = "Ball_001.mp4"
        video_stem = "Ball_001"

        # Pose feature cache — evaluate_pose_model builds:
        #   ROOT / "data" / "pose_features" / split
        # We patch ROOT = tmp_path, so the directory must be:
        #   tmp_path / "data" / "pose_features" / split
        pose_split_dir = tmp_path / "data" / "pose_features" / split
        pose_split_dir.mkdir(parents=True, exist_ok=True)
        _make_pose_feature_file(pose_split_dir, video_stem, n_frames=50)

        # Annotation
        ann_dir = tmp_path / "v2" / split / "annotations"
        _make_annotation_json(ann_dir, video_name, start_frame=10, end_frame=40)

        ckpt_path = _make_pose_checkpoint(tmp_path, hidden_dim=32)

        cfg = {
            "model": {
                "model_type": "pose",
                "temporal_head": "bilstm_attn",
                "hidden_dim": 32,
                "n_layers": 1,
                "dropout": 0.0,
            },
            "data": {
                "v2_root": str(tmp_path / "v2"),
                f"{split}_annotations": f"{split}/annotations",
            },
            "postprocess": {
                "threshold": 0.5,
                "min_duration_sec": 0.0,   # disabled so any detection passes
                "max_duration_sec": 120.0,
            },
        }
        return ckpt_path, cfg

    def test_returns_dict_with_required_metric_keys(self, tmp_path):
        from src.scripts.compare_models import evaluate_pose_model

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        # Patch ROOT so pose_root resolves to tmp_path/pose_features
        with patch("src.scripts.compare_models.ROOT", tmp_path):
            metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        for key in ("mAP@0.5", "precision", "recall", "fps", "size_mb", "params"):
            assert key in metrics, f"Missing key: {key}"

    def test_fps_key_is_positive_float(self, tmp_path):
        from src.scripts.compare_models import evaluate_pose_model

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        assert isinstance(metrics["fps"], float)
        assert metrics["fps"] >= 0.0

    def test_size_mb_matches_model_size_mb(self, tmp_path):
        from src.scripts.compare_models import evaluate_pose_model
        from src.models.pose_model import PoseGymRT

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        # Compute expected size independently
        model_cfg = cfg["model"]
        ref_model = PoseGymRT(
            hidden_dim=model_cfg["hidden_dim"],
            temporal_name=model_cfg["temporal_head"],
            temporal_cfg={"hidden_dim": model_cfg["hidden_dim"], "n_layers": 1, "dropout": 0.0},
        )
        expected_size_mb = ref_model.size_mb()

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        assert abs(metrics["size_mb"] - expected_size_mb) < 1e-6

    def test_params_key_is_positive_integer(self, tmp_path):
        from src.scripts.compare_models import evaluate_pose_model

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        assert isinstance(metrics["params"], int)
        assert metrics["params"] > 0

    def test_uses_split_specific_annotation_key(self, tmp_path):
        """evaluate_pose_model must look up f'{split}_annotations' in data config."""
        from src.scripts.compare_models import evaluate_pose_model

        # Build environment for 'test' split instead of 'valid'
        split = "test"
        ckpt_path, cfg = self._build_env(tmp_path, split=split)
        device = torch.device("cpu")

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            metrics = evaluate_pose_model(ckpt_path, cfg, device, split=split)

        assert "mAP@0.5" in metrics

    def test_metric_values_are_finite_floats(self, tmp_path):
        """Metric values must be finite (not inf/NaN), except boundary_error which can be NaN."""
        from src.scripts.compare_models import evaluate_pose_model
        import math

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        for key in ("mAP@0.5", "precision", "recall", "fps", "size_mb"):
            val = metrics[key]
            assert math.isfinite(val), f"{key}={val} is not finite"

    def test_empty_dataset_produces_zero_recall(self, tmp_path):
        """
        When the dataset has no samples (no matching .pt files), evaluate_pose_model
        must not crash and recall must be 0.
        """
        from src.scripts.compare_models import evaluate_pose_model

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        # Return an empty dataset; bypass real filesystem lookup with mock
        with patch("src.scripts.compare_models.ROOT", tmp_path):
            with patch("src.scripts.compare_models.PoseDataset") as mock_ds_cls:
                mock_ds = MagicMock()
                mock_ds.samples = []
                mock_ds_cls.return_value = mock_ds

                metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        assert metrics.get("recall", 0.0) == 0.0

    def test_evaluate_pose_model_calls_evaluate_split(self, tmp_path):
        """evaluate_pose_model must delegate to evaluate_split from train.py."""
        from src.scripts.compare_models import evaluate_pose_model

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        fake_metrics = {
            "mAP@0.3": 0.9, "mAP@0.5": 0.85, "mAP@0.7": 0.7,
            "precision": 0.8, "recall": 0.9, "boundary_error": 0.4,
        }

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            with patch("src.scripts.compare_models.evaluate_split", return_value=fake_metrics) as mock_eval:
                # Need a non-empty dataset so fps denominator > 0
                with patch("src.scripts.compare_models.PoseDataset") as mock_ds_cls:
                    sample = MagicMock()
                    sample.features = torch.zeros(50, 99)
                    mock_ds = MagicMock()
                    mock_ds.samples = [sample]
                    mock_ds_cls.return_value = mock_ds

                    metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        mock_eval.assert_called_once()
        # Verify the threshold and duration bounds are forwarded from postprocess config
        call_args = mock_eval.call_args
        assert call_args.args[3] == cfg["postprocess"]["threshold"]
        assert call_args.args[4] == cfg["postprocess"]["min_duration_sec"]
        assert call_args.args[5] == cfg["postprocess"]["max_duration_sec"]

    def test_evaluate_pose_model_passes_model_to_evaluate_split(self, tmp_path):
        """The loaded PoseGymRT model must be passed (not a bare PoseHead) to evaluate_split."""
        from src.scripts.compare_models import evaluate_pose_model
        from src.models.pose_model import PoseGymRT

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        fake_metrics = {"mAP@0.5": 0.9, "precision": 1.0, "recall": 1.0, "boundary_error": 0.3}
        captured = {}

        def _fake_evaluate_split(model, dataset, dev, thr, min_d, max_d):
            captured["model"] = model
            return fake_metrics

        with patch("src.scripts.compare_models.ROOT", tmp_path):
            with patch("src.scripts.compare_models.evaluate_split", side_effect=_fake_evaluate_split):
                with patch("src.scripts.compare_models.PoseDataset") as mock_ds_cls:
                    sample = MagicMock()
                    sample.features = torch.zeros(50, 99)
                    mock_ds = MagicMock()
                    mock_ds.samples = [sample]
                    mock_ds_cls.return_value = mock_ds

                    evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        assert isinstance(captured["model"], PoseGymRT)

    def test_fps_is_zero_when_elapsed_time_is_zero(self, tmp_path):
        """If elapsed == 0, fps should be 0.0 (no division by zero)."""
        from src.scripts.compare_models import evaluate_pose_model

        ckpt_path, cfg = self._build_env(tmp_path, split="valid")
        device = torch.device("cpu")

        fake_metrics = {"mAP@0.5": 0.0, "precision": 0.0, "recall": 0.0, "boundary_error": float("nan")}

        # Force perf_counter to return the same value twice → elapsed == 0
        with patch("src.scripts.compare_models.ROOT", tmp_path):
            with patch("src.scripts.compare_models.evaluate_split", return_value=fake_metrics):
                with patch("src.scripts.compare_models.PoseDataset") as mock_ds_cls:
                    sample = MagicMock()
                    sample.features = torch.zeros(50, 99)
                    mock_ds = MagicMock()
                    mock_ds.samples = [sample]
                    mock_ds_cls.return_value = mock_ds

                    with patch("time.perf_counter", side_effect=[0.0, 0.0]):
                        metrics = evaluate_pose_model(ckpt_path, cfg, device, split="valid")

        assert metrics["fps"] == 0.0


# ---------------------------------------------------------------------------
# Integration: is_pose_config routes evaluate_pose_model inside run_comparison
# ---------------------------------------------------------------------------

class TestRunComparisonPoseRouting:
    """Verify that run_comparison dispatches to evaluate_pose_model for pose configs."""

    def test_run_comparison_calls_evaluate_pose_model_for_pose_checkpoint(self, tmp_path):
        from src.scripts.compare_models import run_comparison

        ckpt_path = tmp_path / "pose_bilstm_attn_opt_seed42_best.pt"
        # Write a dummy (non-loadable) checkpoint — we will mock the evaluate call
        ckpt_path.write_bytes(b"")

        pose_cfg = _pose_cfg(hidden_dim=32, n_layers=1)
        # Add annotation key required by run_comparison
        pose_cfg["data"]["valid_annotations"] = "valid/annotations"

        fake_metrics = {
            "mAP@0.3": 0.9, "mAP@0.5": 0.85, "mAP@0.7": 0.7,
            "precision": 0.8, "recall": 0.9, "boundary_error": 0.4,
            "fps": 1000.0, "size_mb": 5.0, "params": 100000,
        }

        with patch("src.scripts.compare_models.infer_config_name", return_value="pose_bilstm_attn_opt"):
            with patch("src.scripts.compare_models.load_config", return_value=pose_cfg):
                with patch("src.scripts.compare_models.evaluate_pose_model", return_value=fake_metrics) as mock_ep:
                    with patch("src.scripts.compare_models.evaluate_model") as mock_em:
                        with patch("src.scripts.compare_models.save_comparison_table"):
                            with patch("src.scripts.compare_models.make_plots"):
                                run_comparison(
                                    splits=["valid"],
                                    checkpoint_paths=[ckpt_path],
                                    models_dir=tmp_path / "models",
                                )

        mock_ep.assert_called_once()
        mock_em.assert_not_called()

    def test_run_comparison_does_not_call_evaluate_pose_model_for_frame_checkpoint(self, tmp_path):
        from src.scripts.compare_models import run_comparison

        ckpt_path = tmp_path / "efficientnet_b0_bilstm_attn_opt_seed42_best.pt"
        ckpt_path.write_bytes(b"")

        frame_cfg = {
            "model": {"backbone": "efficientnet_b0", "temporal_head": "bilstm_attn"},
            "data": {
                "v2_root": str(tmp_path),
                "features_dir": "data/features",
                "valid_annotations": "valid/annotations",
            },
            "postprocess": {"threshold": 0.5, "min_duration_sec": 1.5, "max_duration_sec": 12.0},
        }

        fake_metrics = {
            "mAP@0.3": 0.9, "mAP@0.5": 0.85, "mAP@0.7": 0.7,
            "precision": 0.8, "recall": 0.9, "boundary_error": 0.4,
            "fps": 1000.0, "size_mb": 5.0, "params": 100000,
        }

        with patch("src.scripts.compare_models.infer_config_name", return_value="efficientnet_b0_bilstm_attn_opt"):
            with patch("src.scripts.compare_models.load_config", return_value=frame_cfg):
                with patch("src.scripts.compare_models.evaluate_model", return_value=fake_metrics) as mock_em:
                    with patch("src.scripts.compare_models.evaluate_pose_model") as mock_ep:
                        with patch("src.scripts.compare_models.save_comparison_table"):
                            with patch("src.scripts.compare_models.make_plots"):
                                run_comparison(
                                    splits=["valid"],
                                    checkpoint_paths=[ckpt_path],
                                    models_dir=tmp_path / "models",
                                )

        mock_ep.assert_not_called()
        mock_em.assert_called_once()
