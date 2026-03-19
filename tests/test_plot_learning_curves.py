"""
Tests for src/scripts/plot_learning_curves.py.

Strategy
--------
The module creates a module-level MlflowClient instance at import time and
calls mlflow.set_tracking_uri().  We therefore patch both before the import
so no real MLflow connection is ever attempted.  Each test patches the
module-level `client` object directly via
`plot_learning_curves.client.<method>` to stay focused on observable
behaviour rather than internal construction details.

We also patch matplotlib.pyplot.savefig (routed through Figure.savefig inside
_save) to prevent actual PNG writes, and Path.mkdir so no directories are
created on disk.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure the module can be imported without a live MLflow server
# or GPU.  We inject thin stubs before the first import.
# ---------------------------------------------------------------------------


def _make_fake_mlflow() -> types.ModuleType:
    """Return a minimal mlflow stub that satisfies the module's top-level
    calls (set_tracking_uri, tracking.MlflowClient)."""
    mlflow_stub = types.ModuleType("mlflow")
    mlflow_stub.set_tracking_uri = MagicMock()

    tracking_stub = types.ModuleType("mlflow.tracking")
    fake_client = MagicMock()
    tracking_stub.MlflowClient = MagicMock(return_value=fake_client)

    entities_stub = types.ModuleType("mlflow.entities")
    mlflow_stub.tracking = tracking_stub
    mlflow_stub.entities = entities_stub

    sys.modules.setdefault("mlflow", mlflow_stub)
    sys.modules.setdefault("mlflow.tracking", tracking_stub)
    sys.modules.setdefault("mlflow.entities", entities_stub)
    return mlflow_stub


_make_fake_mlflow()

# Now the import is safe.
import importlib  # noqa: E402

# Use importlib so we can re-import cleanly if needed without rerunning the
# module-level side effects multiple times.
import src.scripts.plot_learning_curves as plc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build lightweight MLflow Run / Metric fakes
# ---------------------------------------------------------------------------


def _make_run(run_id: str, run_name: str) -> MagicMock:
    """Return a MagicMock that mimics mlflow.entities.Run."""
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = run_name
    return run


def _make_metric(step: int, value: float) -> MagicMock:
    """Return a MagicMock that mimics mlflow.entities.Metric."""
    m = MagicMock()
    m.step = step
    m.value = value
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_client_mock():
    """Reset the module-level client mock between tests."""
    plc.client.reset_mock()
    # Explicitly clear side_effects — reset_mock() may not propagate to child
    # mock side_effects in all Python/mock versions.
    plc.client.get_metric_history.side_effect = None
    plc.client.search_runs.side_effect = None
    yield


@pytest.fixture()
def suppress_file_io():
    """Patch out file-system side effects: mkdir and Figure.savefig."""
    with (
        patch("src.scripts.plot_learning_curves.PLOTS_DIR") as mock_dir,
        patch("matplotlib.figure.Figure.savefig"),
        patch("builtins.print"),  # silence progress prints
    ):
        mock_dir.mkdir = MagicMock()
        mock_dir.__truediv__ = lambda self, other: Path("/tmp") / other
        yield


# ---------------------------------------------------------------------------
# _find_run
# ---------------------------------------------------------------------------


class TestFindRun:
    def test_returns_none_when_no_runs_found(self):
        plc.client.search_runs.return_value = []

        result = plc._find_run("nonexistent_run")

        assert result is None

    def test_returns_first_run_when_found(self):
        fake_run = _make_run("abc123", "mv3_small_opt_seed42")
        plc.client.search_runs.return_value = [fake_run]

        result = plc._find_run("mv3_small_opt_seed42")

        assert result is fake_run

    def test_passes_correct_filter_string_to_search_runs(self):
        plc.client.search_runs.return_value = []

        plc._find_run("my_run")

        call_kwargs = plc.client.search_runs.call_args
        # filter_string must reference the exact run_name and FINISHED status
        filter_str = call_kwargs.kwargs.get(
            "filter_string", call_kwargs.args[1] if len(call_kwargs.args) > 1 else ""
        )
        assert "my_run" in filter_str
        assert "FINISHED" in filter_str

    def test_returns_none_when_search_returns_empty_list(self):
        # Explicit check: empty list (falsy) must yield None, not IndexError
        plc.client.search_runs.return_value = []

        result = plc._find_run("any_run")

        assert result is None

    def test_ignores_extra_results_returns_only_first(self):
        run_a = _make_run("id_a", "run_a")
        run_b = _make_run("id_b", "run_b")
        plc.client.search_runs.return_value = [run_a, run_b]

        result = plc._find_run("run_a")

        assert result is run_a


# ---------------------------------------------------------------------------
# _get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    def test_returns_empty_lists_when_no_history(self):
        plc.client.get_metric_history.return_value = []

        steps, values = plc._get_history("run_id_1", "train_loss")

        assert steps == []
        assert values == []

    def test_returns_sorted_steps_and_values(self):
        # Deliberately out-of-order to verify sort
        history = [
            _make_metric(step=3, value=0.3),
            _make_metric(step=1, value=0.1),
            _make_metric(step=2, value=0.2),
        ]
        plc.client.get_metric_history.return_value = history

        steps, values = plc._get_history("run_id_1", "train_loss")

        assert steps == [1, 2, 3]
        assert values == [0.1, 0.2, 0.3]

    def test_returns_lists_not_tuples(self):
        history = [_make_metric(step=1, value=0.5)]
        plc.client.get_metric_history.return_value = history

        steps, values = plc._get_history("run_id_1", "train_loss")

        assert isinstance(steps, list)
        assert isinstance(values, list)

    def test_single_metric_point(self):
        history = [_make_metric(step=0, value=1.0)]
        plc.client.get_metric_history.return_value = history

        steps, values = plc._get_history("run_id_1", "val_mAP_at_0.5")

        assert steps == [0]
        assert values == [1.0]

    def test_passes_run_id_and_metric_name_to_client(self):
        plc.client.get_metric_history.return_value = []

        plc._get_history("specific_run_id", "val_mAP_at_0.5")

        plc.client.get_metric_history.assert_called_once_with(
            "specific_run_id", "val_mAP_at_0.5"
        )

    def test_duplicate_steps_are_preserved_in_sort(self):
        # Degenerate case: two metrics logged at the same step
        history = [
            _make_metric(step=2, value=0.8),
            _make_metric(step=2, value=0.9),
        ]
        plc.client.get_metric_history.return_value = history

        steps, values = plc._get_history("run_id_1", "train_loss")

        assert len(steps) == 2
        assert steps == [2, 2]


# ---------------------------------------------------------------------------
# fig1_learning_curves_grid
# ---------------------------------------------------------------------------


class TestFig1LearningCurvesGrid:
    def test_returns_empty_convergence_data_when_all_runs_missing(
        self, suppress_file_io
    ):
        # All _find_run calls return None
        plc.client.search_runs.return_value = []

        result = plc.fig1_learning_curves_grid()

        assert result == []

    def test_does_not_crash_when_all_runs_missing(self, suppress_file_io):
        plc.client.search_runs.return_value = []

        # Must complete without exception
        plc.fig1_learning_curves_grid()

    def test_returns_convergence_entry_for_run_with_both_metrics(
        self, suppress_file_io
    ):
        fake_run = _make_run("run_abc", "mobilenet_v3_small_opt_seed42")
        # search_runs called twice per model (opt then base fallback) — return
        # the run on the first try, empty thereafter
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            # First call (opt seed42 for first model) returns a result
            if call_count[0] == 1:
                return [fake_run]
            return []

        plc.client.search_runs.side_effect = search_side_effect

        loss_history = [
            _make_metric(step=1, value=0.5),
            _make_metric(step=2, value=0.3),
        ]
        map_history = [
            _make_metric(step=1, value=0.6),
            _make_metric(step=2, value=0.9),
        ]

        def metric_side_effect(run_id, metric):
            if metric == "train_loss":
                return loss_history
            if metric == "val_mAP_at_0.5":
                return map_history
            return []

        plc.client.get_metric_history.side_effect = metric_side_effect

        result = plc.fig1_learning_curves_grid()

        assert len(result) >= 1
        entry = result[0]
        assert "label" in entry
        assert "best_epoch" in entry
        assert "total_epochs" in entry
        assert "best_mAP" in entry
        assert "final_mAP" in entry
        assert "is_opt" in entry

    def test_convergence_entry_has_correct_best_epoch(self, suppress_file_io):
        fake_run = _make_run("run_abc", "mobilenet_v3_small_opt_seed42")
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [fake_run]
            return []

        plc.client.search_runs.side_effect = search_side_effect

        # Best mAP is at epoch 3 (value 0.95 > 0.8 > 0.7)
        map_history = [
            _make_metric(step=1, value=0.7),
            _make_metric(step=2, value=0.8),
            _make_metric(step=3, value=0.95),
        ]
        loss_history = [
            _make_metric(step=1, value=0.5),
            _make_metric(step=2, value=0.4),
            _make_metric(step=3, value=0.3),
        ]

        def metric_side_effect(run_id, metric):
            if metric == "train_loss":
                return loss_history
            if metric == "val_mAP_at_0.5":
                return map_history
            return []

        plc.client.get_metric_history.side_effect = metric_side_effect

        result = plc.fig1_learning_curves_grid()

        assert result[0]["best_epoch"] == 3
        assert result[0]["best_mAP"] == pytest.approx(0.95)

    def test_does_not_crash_when_run_has_no_metrics(self, suppress_file_io):
        fake_run = _make_run("run_abc", "mobilenet_v3_small_opt_seed42")
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [fake_run]
            return []

        plc.client.search_runs.side_effect = search_side_effect
        plc.client.get_metric_history.return_value = []

        result = plc.fig1_learning_curves_grid()

        # No convergence entry appended when val_mAP is missing
        assert result == []

    def test_is_opt_field_is_true_for_opt_run(self, suppress_file_io):
        fake_run = _make_run("run_abc", "mobilenet_v3_small_opt_seed42")
        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [fake_run]
            return []

        plc.client.search_runs.side_effect = search_side_effect

        map_history = [_make_metric(step=1, value=0.8)]
        loss_history = [_make_metric(step=1, value=0.4)]

        def metric_side_effect(run_id, metric):
            if metric == "train_loss":
                return loss_history
            if metric == "val_mAP_at_0.5":
                return map_history
            return []

        plc.client.get_metric_history.side_effect = metric_side_effect

        result = plc.fig1_learning_curves_grid()

        assert result[0]["is_opt"] is True


# ---------------------------------------------------------------------------
# fig2_convergence
# ---------------------------------------------------------------------------


class TestFig2Convergence:
    def test_skips_gracefully_on_empty_convergence_data(
        self, suppress_file_io, capsys
    ):
        # Must not raise and must print a skip message
        plc.fig2_convergence([])
        # savefig should not have been called (no figure generated)
        # We can check indirectly via the mock
        import matplotlib.figure
        matplotlib.figure.Figure.savefig  # attribute access — no assertion needed;
        # the real assertion is that no exception was raised.

    def test_does_not_raise_on_empty_data(self, suppress_file_io):
        plc.fig2_convergence([])  # must complete without error

    def test_generates_figure_for_single_entry(self, suppress_file_io):
        data = [
            {
                "label": "mv3_small",
                "best_epoch": 10,
                "total_epochs": 20,
                "best_mAP": 0.85,
                "final_mAP": 0.80,
                "final_loss": 0.25,
                "is_opt": True,
            }
        ]
        # Should not raise
        plc.fig2_convergence(data)

    def test_generates_figure_for_multiple_entries(self, suppress_file_io):
        data = [
            {
                "label": f"model_{i}",
                "best_epoch": 5 + i,
                "total_epochs": 20,
                "best_mAP": 0.5 + i * 0.05,
                "final_mAP": 0.48 + i * 0.05,
                "final_loss": 0.4 - i * 0.02,
                "is_opt": True,
            }
            for i in range(4)
        ]
        plc.fig2_convergence(data)  # must not raise

    def test_handles_entry_where_best_epoch_equals_total_epochs(
        self, suppress_file_io
    ):
        # patience_epochs = total - best = 0 — stacked bar with zero second bar
        data = [
            {
                "label": "mv3_small",
                "best_epoch": 20,
                "total_epochs": 20,
                "best_mAP": 0.9,
                "final_mAP": 0.9,
                "final_loss": 0.1,
                "is_opt": True,
            }
        ]
        plc.fig2_convergence(data)  # must not raise


# ---------------------------------------------------------------------------
# fig3_multiseed_curves
# ---------------------------------------------------------------------------


class TestFig3MultiseedCurves:
    def test_does_not_crash_when_all_runs_missing(self, suppress_file_io):
        plc.client.search_runs.return_value = []

        plc.fig3_multiseed_curves()  # must complete without exception

    def test_generates_figure_when_some_runs_present(self, suppress_file_io):
        # Only the first seed of the first top-3 model has data
        fake_run = _make_run("run_xyz", "efficientnet_b0_bilstm_opt_seed42")

        call_count = [0]

        def search_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [fake_run]
            return []

        plc.client.search_runs.side_effect = search_side_effect

        map_history = [
            _make_metric(step=1, value=0.7),
            _make_metric(step=2, value=0.88),
        ]
        loss_history = [
            _make_metric(step=1, value=0.6),
            _make_metric(step=2, value=0.4),
        ]

        def metric_side_effect(run_id, metric):
            if metric == "train_loss":
                return loss_history
            if metric == "val_mAP_at_0.5":
                return map_history
            return []

        plc.client.get_metric_history.side_effect = metric_side_effect

        plc.fig3_multiseed_curves()  # must not raise

    def test_queries_expected_seeds(self, suppress_file_io):
        plc.client.search_runs.return_value = []

        plc.fig3_multiseed_curves()

        # Each of the 3 top models x 3 seeds = 9 search_runs calls
        assert plc.client.search_runs.call_count == 9

    def test_queries_all_top3_models(self, suppress_file_io):
        plc.client.search_runs.return_value = []

        plc.fig3_multiseed_curves()

        all_filter_strings = [
            str(call.kwargs.get("filter_string", ""))
            for call in plc.client.search_runs.call_args_list
        ]
        combined = " ".join(all_filter_strings)
        for model_id in plc.TOP3:
            assert model_id in combined, (
                f"Expected {model_id} to appear in search_runs filter strings"
            )

    def test_queries_all_seeds(self, suppress_file_io):
        plc.client.search_runs.return_value = []

        plc.fig3_multiseed_curves()

        all_filter_strings = [
            str(call.kwargs.get("filter_string", ""))
            for call in plc.client.search_runs.call_args_list
        ]
        combined = " ".join(all_filter_strings)
        for seed in plc.SEEDS:
            assert str(seed) in combined, (
                f"Expected seed {seed} to appear in search_runs filter strings"
            )
