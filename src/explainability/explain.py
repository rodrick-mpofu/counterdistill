"""Main explainability orchestration script."""

import logging
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.explainability.dice import DiceExplainer
from src.explainability.shap import ShapExplainer
from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_loader import AdultIncomeLoader
from src.storage.duckdb import DuckDBStorage

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main explainability pipeline."""
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting explainability pipeline...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    import random

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

    # 2. Feature engineering
    combined_df = pl.concat([x_train, x_test])
    engine = FeatureEngineer(
        combined_df,
        config=cfg.feature_engineering if hasattr(cfg, "feature_engineering") else None,
    )
    combined_fe = engine.build_pipeline()

    # Split back
    n_train = x_train.height
    x_train_fe = combined_fe[:n_train]
    x_test_fe = combined_fe[n_train:]

    logger.info(f"Features shape: {x_train_fe.width}")

    # 3. Load model from MLflow
    model_name = cfg.model._class_.split(".")[-1]
    run_id = cfg.mlflow.get("run_id", None)

    if run_id:
        # Load from MLflow
        logger.info(f"Loading model from MLflow run {run_id}")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")  # type: ignore
    else:
        # Load from file or train new
        logger.warning("No run_id provided, using previously trained model")
        # Import train function
        import importlib

        # Reload train module to get train_model function
        import src.modeling.train as train_module

        importlib.reload(train_module)

        result = train_module.train_model(cfg, x_train_fe, y_train, x_test_fe, y_test)
        model = result["model"]

    # 4. Initialize storage
    storage = DuckDBStorage(db_path=cfg.database_dir + "/counterdistill.db")

    # 5. Generate DiCE counterfactuals
    logger.info("Generating DiCE counterfactuals...")
    dice = DiceExplainer(
        model=model,
        data_df=x_train_fe,
        target="target",
        categorical_features=cfg.feature_engineering.categorical_columns,
        continuous_features=cfg.feature_engineering.numeric_columns,
    )

    # Generate counterfactuals for test samples
    x_test_np = x_test_fe.to_numpy()
    cf_df = dice.generate_batch(
        x_test_np,
        num_samples=cfg.get("num_counterfactuals", 100),
        total_cfs=5,
        desired_class=1,
    )

    # Store in DuckDB
    storage.store_counterfactuals(
        cf_df, model_name=model_name, run_id=run_id or "local"
    )

    logger.info(f"Generated {cf_df.height} counterfactuals")

    # 6. Compute SHAP values
    logger.info("Computing SHAP values...")
    feature_names = x_train_fe.columns

    shap_explainer = ShapExplainer(
        model=model,
        x_train=x_train_fe.to_numpy(),
        feature_names=feature_names,
        model_type="tree",
    )

    # Compute SHAP values for test samples
    shap_df = shap_explainer.compute_batch(
        x_test_np[: cfg.get("num_counterfactuals", 100)],
        feature_names=feature_names,
    )

    # Store in DuckDB
    storage.store_shap_values(shap_df, model_name=model_name, run_id=run_id or "local")

    logger.info(f"Computed {shap_df.height} SHAP values")

    # 7. Save summary statistics
    logger.info("Saving summary statistics...")

    # Get model metrics from MLflow
    if run_id:
        client = mlflow.tracking.MlflowClient()  # type: ignore
        run = client.get_run(run_id)
        metrics = run.data.metrics

        # Store metrics in DuckDB
        storage.store_metrics(  # type: ignore
            model_name=model_name, metrics=metrics, run_id=run_id
        )
        logger.info(f"Stored {len(metrics)} metrics for {model_name}")
    else:
        logger.warning("No run_id provided, skipping metric storage")

    logger.info("Explainability pipeline complete!")


if __name__ == "__main__":
    main()
