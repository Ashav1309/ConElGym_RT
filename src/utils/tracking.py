"""MLflow wrapper для ConElGym_RT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class MLflowTracker:
    """Тонкая обёртка над MLflow для логирования экспериментов.

    Использование:
        tracker = MLflowTracker("ConElGym_RT", tracking_uri="./mlruns")
        with tracker.start_run(run_name="mobilenet_v3_bilstm_seed42"):
            tracker.log_params({"lr": 1e-3, "hidden_dim": 128})
            tracker.log_metrics({"mAP@0.5": 0.85}, step=10)
            tracker.log_artifact("models/best.pt")
    """

    def __init__(
        self,
        experiment_name: str = "ConElGym_RT",
        tracking_uri: str = "./mlruns",
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None) -> mlflow.ActiveRun:
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def set_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)
