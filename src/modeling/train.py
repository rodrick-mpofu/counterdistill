"""Model training module with MLflow tracking and Hydra configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_loader import AdultIncomeLoader

logger = logging.getLogger(__name__)


def train_model(
    cfg: DictConfig,
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_test: pl.DataFrame,
    y_test: pl.Series,
) -> dict[str, Any]:
    """Train and evaluate the configured sklearn-compatible model."""
    import importlib

    model_class_path = cfg.model._class_
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    model_params = {k: v for k, v in cfg.model.items() if not k.startswith("_")}
    model = model_class(**model_params)

    x_train_np = x_train.to_numpy()
    y_train_np = y_train.to_numpy()
    x_test_np = x_test.to_numpy()
    y_test_np = y_test.to_numpy()

    logger.info("Training %s...", class_name)
    model.fit(x_train_np, y_train_np)

    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(x_test_np)
    accuracy = accuracy_score(y_test_np, y_pred)
    report = classification_report(y_test_np, y_pred, output_dict=True)

    logger.info("Test accuracy: %.4f", accuracy)

    return {
        "model": model,
        "accuracy": accuracy,
        "classification_report": report,
        "x_train": x_train_np,
        "y_train": y_train_np,
        "x_test": x_test_np,
        "y_test": y_test_np,
        "feature_names": x_train.columns,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train using preprocessing state learned only from the training split."""
    import random

    import numpy as np

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting model training...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    loader = AdultIncomeLoader(data_dir=cfg.data_dir)
    x_train, x_test, y_train, y_test = loader.get_train_test_split(
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    logger.info("Training data: %d samples", x_train.height)
    logger.info("Test data: %d samples", x_test.height)

    engine = FeatureEngineer(
        x_train,
        config=cfg.feature_engineering if hasattr(cfg, "feature_engineering") else None,
    )
    x_train_fe = engine.fit_transform()
    x_test_fe = engine.transform(x_test)

    logger.info("Features shape: %d", x_train_fe.width)

    tracking_uri = cfg.mlflow.tracking_uri
    if not tracking_uri.startswith(("http://", "https://", "sqlite://")):
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
    elif tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model_class_name = str(cfg.model._class_.split(".")[-1])

    model_params = {
        f"model.{key}": value
        for key, value in cfg.model.items()
        if not key.startswith("_")
    }

    data_params = {f"data.{key}": value for key, value in cfg.data.items()}

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow.log_params(
            {
                **model_params,
                **data_params,
                "seed": int(cfg.seed),
                "train_rows": x_train_fe.height,
                "test_rows": x_test_fe.height,
                "feature_count": x_train_fe.width,
            }
        )

        mlflow.set_tags(
            {
                "project": "counterdistill",
                "stage": "training",
                "model_class": model_class_name,
                "dataset": str(cfg.data.name),
            }
        )

        resolved_config_raw = OmegaConf.to_container(
            cfg,
            resolve=True,
        )

        if not isinstance(
            resolved_config_raw,
            dict,
        ):
            raise TypeError("Expected Hydra configuration to resolve to a dictionary.")

        resolved_config = cast(
            dict[str, Any],
            resolved_config_raw,
        )

        mlflow.log_dict(
            resolved_config,
            "config/resolved_config.yaml",
        )

        result = train_model(cfg, x_train_fe, y_train, x_test_fe, y_test)
        mlflow.log_metric("accuracy", result["accuracy"])

        report = result["classification_report"]
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, int | float):
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)

        trusted_types_map = {
            "XGBClassifier": [
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
            "LGBMClassifier": [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        }

        mlflow.sklearn.log_model(  # type: ignore
            result["model"],
            name="model",
            registered_model_name=model_class_name,
            skops_trusted_types=trusted_types_map.get(model_class_name),
        )

        mlflow.log_dict(
            {
                "feature_names": engine.feature_names_,
                "scale": engine.scale_,
                "scale_stats": engine.scale_stats_,
            },
            "preprocessing/feature_schema.json",
        )

        if hasattr(result["model"], "feature_importances_"):
            importance_dict = {
                name: float(importance)
                for name, importance in zip(
                    result["feature_names"],
                    result["model"].feature_importances_,
                    strict=True,
                )
            }
            importance_dict = dict(
                sorted(
                    importance_dict.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            mlflow.log_dict(
                importance_dict,
                "model/feature_importance.json",
            )

        logger.info("MLflow run ID: %s", run.info.run_id)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
