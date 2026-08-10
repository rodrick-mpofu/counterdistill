"""Hyperparameter optimization with Optuna and MLflow."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, cast

import hydra
import mlflow
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_loader import AdultIncomeLoader

logger = logging.getLogger(__name__)


def suggest_parameters(
    trial: optuna.Trial,
    model_class_name: str,
) -> dict[str, Any]:
    """Sample model-specific hyperparameters."""
    if model_class_name == "RandomForestClassifier":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
                step=50,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                4,
                24,
                step=2,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10,
            ),
            "max_features": trial.suggest_float(
                "max_features",
                0.3,
                1.0,
            ),
        }

    if model_class_name == "XGBClassifier":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
                step=50,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
        }

    if model_class_name == "LGBMClassifier":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
                step=50,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                127,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
        }

    raise ValueError(f"Optuna search space is not configured for {model_class_name!r}.")


def build_model(
    cfg: DictConfig,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Instantiate the configured model with optional overrides."""
    model_class_path = str(cfg.model._class_)

    module_path, class_name = model_class_path.rsplit(
        ".",
        1,
    )

    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    params = {
        str(key): value
        for key, value in cfg.model.items()
        if not str(key).startswith("_")
    }

    if overrides:
        params.update(overrides)

    return model_class(**params)


def evaluate_classifier(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate a fitted binary classifier."""
    predictions = model.predict(x)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
            )
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[:, 1]

        metrics["roc_auc"] = float(
            roc_auc_score(
                y,
                probabilities,
            )
        )

    return metrics


def prepare_sqlite_storage(
    uri: str,
) -> None:
    """Create the parent directory for a SQLite URI."""
    if not uri.startswith("sqlite:///"):
        return

    database_path = uri.removeprefix("sqlite:///")

    Path(database_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Tune the configured model and log the study to MLflow."""
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Optuna hyperparameter optimization...")

    np.random.seed(int(cfg.seed))

    # -------------------------------------------------
    # Data
    # -------------------------------------------------

    loader = AdultIncomeLoader(data_dir=cfg.data_dir)

    x_train, x_test, y_train, y_test = loader.get_train_test_split(
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    row_indices = np.arange(x_train.height)

    tune_indices, validation_indices = train_test_split(
        row_indices,
        test_size=float(cfg.optuna.validation_size),
        random_state=int(cfg.seed),
        stratify=y_train_np,
    )

    x_tune = x_train.gather(tune_indices)

    x_validation = x_train.gather(validation_indices)

    y_tune = y_train_np[tune_indices]
    y_validation = y_train_np[validation_indices]

    # -------------------------------------------------
    # Fit preprocessing ONLY on tuning data
    # -------------------------------------------------

    tuning_engine = FeatureEngineer(
        x_tune,
        config=cfg.feature_engineering,
    )

    x_tune_fe = tuning_engine.fit_transform().to_numpy()

    x_validation_fe = tuning_engine.transform(x_validation).to_numpy()

    model_class_name = str(cfg.model._class_.split(".")[-1])

    objective_metric = str(cfg.optuna.metric)

    # -------------------------------------------------
    # MLflow
    # -------------------------------------------------

    tracking_uri = str(cfg.mlflow.tracking_uri)

    prepare_sqlite_storage(tracking_uri)

    optuna_storage = str(cfg.optuna.storage)

    prepare_sqlite_storage(optuna_storage)

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

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

    # -------------------------------------------------
    # Parent study run
    # -------------------------------------------------

    parent_run_name = f"optuna-{model_class_name}"

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tags(
            {
                "project": "counterdistill",
                "stage": "hyperparameter_tuning",
                "model_class": model_class_name,
                "dataset": str(cfg.data.name),
            }
        )

        mlflow.log_params(
            {
                "optuna.n_trials": int(cfg.optuna.n_trials),
                "optuna.metric": (objective_metric),
                "optuna.direction": str(cfg.optuna.direction),
                "optuna.validation_size": float(cfg.optuna.validation_size),
                "seed": int(cfg.seed),
            }
        )

        mlflow.log_dict(
            resolved_config,
            "config/resolved_config.yaml",
        )

        study_name = (
            f"{cfg.optuna.study_name}-{model_class_name}-{parent_run.info.run_id[:8]}"
        )

        sampler = optuna.samplers.TPESampler(seed=int(cfg.optuna.sampler.seed))

        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_storage,
            direction=str(cfg.optuna.direction),
            sampler=sampler,
            load_if_exists=False,
        )

        # ---------------------------------------------
        # Objective
        # ---------------------------------------------

        def objective(
            trial: optuna.Trial,
        ) -> float:
            sampled_params = suggest_parameters(
                trial,
                model_class_name,
            )

            with mlflow.start_run(
                run_name=(f"trial-{trial.number:03d}"),
                nested=True,
            ):
                mlflow.set_tags(
                    {
                        "stage": "optuna_trial",
                        "trial_number": str(trial.number),
                    }
                )

                mlflow.log_params(sampled_params)

                model = build_model(
                    cfg,
                    overrides=sampled_params,
                )

                model.fit(
                    x_tune_fe,
                    y_tune,
                )

                metrics = evaluate_classifier(
                    model,
                    x_validation_fe,
                    y_validation,
                )

                mlflow.log_metrics(
                    {f"validation_{name}": value for name, value in metrics.items()}
                )

                if objective_metric not in metrics:
                    raise ValueError(
                        f"Unsupported Optuna objective metric: {objective_metric!r}"
                    )

                return metrics[objective_metric]

        timeout = int(cfg.optuna.timeout)

        study.optimize(
            objective,
            n_trials=int(cfg.optuna.n_trials),
            timeout=(timeout if timeout > 0 else None),
        )

        # -------------------------------------------------
        # Study summary
        # -------------------------------------------------

        best_params = dict(study.best_params)

        mlflow.log_metric(
            f"best_validation_{objective_metric}",
            float(study.best_value),
        )

        mlflow.log_metric(
            "completed_trials",
            float(len(study.trials)),
        )

        mlflow.log_params({f"best.{key}": value for key, value in best_params.items()})

        mlflow.log_dict(
            {
                "study_name": (study.study_name),
                "best_trial": (study.best_trial.number),
                "best_value": float(study.best_value),
                "objective_metric": (objective_metric),
                "best_params": (best_params),
            },
            "optuna/best_trial.json",
        )

        trials = []

        for trial in study.trials:
            trials.append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "state": (trial.state.name),
                    "params": trial.params,
                }
            )

        mlflow.log_dict(
            {"trials": trials},
            "optuna/trials.json",
        )

        # -------------------------------------------------
        # Retrain best model on FULL training split
        # -------------------------------------------------

        logger.info(
            "Best trial: %d",
            study.best_trial.number,
        )

        logger.info(
            "Best %s: %.4f",
            objective_metric,
            study.best_value,
        )

        logger.info(
            "Best parameters: %s",
            best_params,
        )

        final_engine = FeatureEngineer(
            x_train,
            config=cfg.feature_engineering,
        )

        x_train_final = final_engine.fit_transform().to_numpy()

        x_test_final = final_engine.transform(x_test).to_numpy()

        final_model = build_model(
            cfg,
            overrides=best_params,
        )

        final_model.fit(
            x_train_final,
            y_train_np,
        )

        test_metrics = evaluate_classifier(
            final_model,
            x_test_final,
            y_test_np,
        )

        mlflow.log_metrics(
            {f"test_{name}": value for name, value in test_metrics.items()}
        )

        mlflow.log_metric(
            "feature_count",
            float(len(final_engine.feature_names_)),
        )

        mlflow.log_dict(
            {
                "feature_names": (final_engine.feature_names_),
                "scale": (final_engine.scale_),
                "scale_stats": (final_engine.scale_stats_),
            },
            ("preprocessing/feature_schema.json"),
        )

        trusted_types_map = {
            "XGBClassifier": [
                "xgboost.core.Booster",
                ("xgboost.sklearn.XGBClassifier"),
            ],
            "LGBMClassifier": [
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                ("lightgbm.sklearn.LGBMClassifier"),
            ],
        }

        mlflow.sklearn.log_model(  # type: ignore
            final_model,
            name="model",
            registered_model_name=(model_class_name),
            skops_trusted_types=(trusted_types_map.get(model_class_name)),
        )

        if hasattr(
            final_model,
            "feature_importances_",
        ):
            importance = {
                name: float(value)
                for name, value in zip(
                    final_engine.feature_names_,
                    final_model.feature_importances_,
                    strict=True,
                )
            }

            importance = dict(
                sorted(
                    importance.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            mlflow.log_dict(
                importance,
                ("model/feature_importance.json"),
            )

        logger.info(
            "Final test metrics: %s",
            test_metrics,
        )

        logger.info(
            "MLflow parent run ID: %s",
            parent_run.info.run_id,
        )


if __name__ == "__main__":
    main()
