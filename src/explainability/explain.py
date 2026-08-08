"""Main explainability orchestration script."""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.explainability.dice import DiceExplainer
from src.explainability.shap import ShapExplainer
from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_loader import AdultIncomeLoader
from src.storage.duckdb import DuckDBStorage

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def _validate_model_width(model: object, feature_count: int) -> None:
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and int(expected) != feature_count:
        raise ValueError(
            "Loaded model expects "
            f"{expected} features, but the fitted FeatureEngineer produced "
            f"{feature_count}. Retrain the model with the current preprocessing "
            "pipeline or load a compatible MLflow run."
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run counterfactual and SHAP explanation pipelines."""
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting explainability pipeline...")
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
    logger.info("Encoded feature count: %d", x_train_fe.width)

    model_name = cfg.model._class_.split(".")[-1]

    provided_run_id = cfg.mlflow.get("run_id")

    run_id = (
        provided_run_id if provided_run_id else f"local-{datetime.now():%Y%m%d-%H%M%S}"
    )

    if provided_run_id:
        logger.info("Loading model from MLflow run %s", run_id)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")  # type: ignore
    else:
        logger.warning("No run_id provided; training a local model for explanations.")
        from src.modeling.train import train_model

        result = train_model(
            cfg,
            x_train_fe,
            y_train,
            x_test_fe,
            y_test,
        )
        model = result["model"]

    _validate_model_width(model, x_train_fe.width)

    storage = DuckDBStorage(db_path=str(Path(cfg.database_dir) / "counterdistill.db"))
    storage_run_id = str(run_id or "local")

    logger.info("Generating DiCE counterfactuals...")
    dice = DiceExplainer(
        model=model,
        data_df=x_train,
        target_values=y_train,
        feature_engineer=engine,
        target=cfg.data.target_column,
        categorical_features=list(cfg.feature_engineering.categorical_columns),
        continuous_features=list(cfg.feature_engineering.numeric_columns),
        method="random",
    )

    num_samples = int(cfg.get("num_counterfactuals", 100))

    sample_count = min(
        num_samples,
        x_test.height,
    )

    rng = np.random.default_rng(int(cfg.seed))

    if x_test.height > sample_count:
        selected_indices = np.sort(
            rng.choice(
                x_test.height,
                sample_count,
                replace=False,
            )
        )
    else:
        selected_indices = np.arange(
            x_test.height,
            dtype=int,
        )

    logger.info(
        "Selected %d shared instances for DiCE and SHAP",
        len(selected_indices),
    )

    cf_df = dice.generate_batch(
        x_test,
        num_samples=num_samples,
        total_cfs=5,
        desired_class="opposite",
        random_seed=int(cfg.seed),
        selected_indices=selected_indices,
    )

    if cf_df.height:
        storage.store_counterfactuals(
            cf_df,
            model_name=model_name,
            run_id=storage_run_id,
        )
    else:
        logger.warning("No counterfactuals were generated.")

    logger.info("Generated %d counterfactual rows", cf_df.height)

    logger.info("Computing SHAP values...")
    feature_names = x_train_fe.columns
    x_train_np = x_train_fe.to_numpy()
    x_test_np = x_test_fe.to_numpy()

    shap_explainer = ShapExplainer(
        model=model,
        x_train=x_train_np,
        feature_names=feature_names,
        model_type="tree",
        class_index=1,
    )

    x_shap = x_test_np[selected_indices]

    shap_df = shap_explainer.compute_batch(
        x_shap,
        feature_names=feature_names,
        instance_ids=selected_indices,
    )

    if shap_df.height:
        storage.store_shap_values(
            shap_df,
            model_name=model_name,
            run_id=storage_run_id,
        )
    else:
        logger.warning("No SHAP values were generated.")

    logger.info("Computed %d SHAP rows", shap_df.height)

    if provided_run_id:
        client = mlflow.tracking.MlflowClient()  # type: ignore
        run = client.get_run(provided_run_id)
        metrics = run.data.metrics

        storage.store_metrics(
            model_name=model_name,
            metrics=metrics,
            run_id=provided_run_id,
        )

        logger.info(
            "Stored %d metrics for %s",
            len(metrics),
            model_name,
        )
    else:
        logger.info(
            "Local run %s; skipping MLflow metric storage.",
            run_id,
        )

    logger.info("Explainability pipeline complete!")


if __name__ == "__main__":
    main()
