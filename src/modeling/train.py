"""Model training module with MLflow tracking and Hydra configuration."""

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import mlflow
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_loader import AdultIncomeLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def train_model(
    cfg: DictConfig,
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_test: pl.DataFrame,
    y_test: pl.Series,
) -> dict[str, Any]:
    """Train a model based on configuration."""
    # Import the model class
    model_class_path = cfg.model._class_
    module_path, class_name = model_class_path.rsplit(".", 1)

    # Dynamically import the class
    import importlib

    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Extract model parameters (excluding _class_)
    model_params = {k: v for k, v in cfg.model.items() if not k.startswith("_")}

    # Initialize model
    model = model_class(**model_params)

    # Convert Polars to numpy for sklearn compatibility
    x_train_np = x_train.to_numpy()
    y_train_np = y_train.to_numpy()
    x_test_np = x_test.to_numpy()
    y_test_np = y_test.to_numpy()

    # Train model
    logger.info(f"Training {class_name}...")
    model.fit(x_train_np, y_train_np)

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report

    y_pred = model.predict(x_test_np)
    accuracy = accuracy_score(y_test_np, y_pred)

    # Classification report
    report = classification_report(y_test_np, y_pred, output_dict=True)

    logger.info(f"Test accuracy: {accuracy:.4f}")

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
    """Main training function with Hydra configuration."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting model training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    import random

    import numpy as np

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 1. Load data
    loader = AdultIncomeLoader(data_dir=cfg.data_dir)

    # Get train-test split
    x_train, x_test, y_train, y_test = loader.get_train_test_split(
        test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    logger.info(f"Training data: {x_train.height} samples")
    logger.info(f"Test data: {x_test.height} samples")

    # 2. Feature engineering - combine train and test first
    # Concatenate train and test to ensure consistent encoding
    combined_df = pl.concat([x_train, x_test])

    # Apply feature engineering to combined data
    engine = FeatureEngineer(
        combined_df,
        config=cfg.feature_engineering if hasattr(cfg, "feature_engineering") else None,
    )
    combined_fe = engine.build_pipeline()

    # Split back into train and test
    n_train = x_train.height
    x_train_fe = combined_fe[:n_train]
    x_test_fe = combined_fe[n_train:]

    logger.info(f"Features shape: {x_train_fe.width}")

    # 3. Set up MLflow
    # Only create directory if using file-based tracking (not SQLite or HTTP)
    tracking_uri = cfg.mlflow.tracking_uri
    if not tracking_uri.startswith(("http://", "https://", "sqlite://")):
        output_dir = Path(tracking_uri)
        output_dir.mkdir(parents=True, exist_ok=True)
    # For SQLite, ensure the database directory exists
    elif tracking_uri.startswith("sqlite:///"):
        # Extract the database path from sqlite:///path/to/db
        db_path = tracking_uri.replace("sqlite:///", "")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # 4. Train model with MLflow tracking
    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        # Log parameters
        mlflow.log_params({**cfg.model, **cfg.data})

        # Train the model
        result = train_model(cfg, x_train_fe, y_train, x_test_fe, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", result["accuracy"])

        # Log classification report metrics
        report = result["classification_report"]
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, int | float):
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)

        # Log model with trusted types based on model type
        model_class_name = cfg.model._class_.split(".")[-1]

        # Define trusted types for different frameworks
        trusted_types_map = {
            "XGBClassifier": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
            "LGBMClassifier": [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        }

        trusted_types = trusted_types_map.get(model_class_name, None)

        mlflow.sklearn.log_model(  # type: ignore
            result["model"],
            name="model",
            registered_model_name=model_class_name,
            skops_trusted_types=trusted_types,
        )

        # Log feature importance if available
        if hasattr(result["model"], "feature_importances_"):
            import json

            importance = result["model"].feature_importances_
            feature_names = result["feature_names"]
            importance_dict = {
                name: float(imp)
                for name, imp in zip(
                    feature_names,
                    importance,
                    strict=True,
                )
            }
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            importance_path = Path("feature_importance.json")
            with open(importance_path, "w") as f:
                json.dump(importance_dict, f, indent=2)
            mlflow.log_artifact(str(importance_path))
            # Clean up
            importance_path.unlink()

        # Log the configuration
        mlflow.log_params(dict(cfg))  # type: ignore

        logger.info(f"MLflow run ID: {run.info.run_id}")
        experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)

        if experiment is not None:
            logger.info(f"Experiment URI: {experiment.artifact_location}")
        else:
            logger.warning(f"Experiment '{cfg.mlflow.experiment_name}' was not found")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
