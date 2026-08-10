from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pytest
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier

from src.modeling.tune import (
    build_model,
    evaluate_classifier,
    prepare_sqlite_storage,
    suggest_parameters,
)


def test_random_forest_search_space() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 250,
            "max_depth": 12,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "max_features": 0.7,
        }
    )

    params = suggest_parameters(
        trial,  # type: ignore[arg-type]
        "RandomForestClassifier",
    )

    assert params == {
        "n_estimators": 250,
        "max_depth": 12,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "max_features": 0.7,
    }


def test_xgboost_search_space() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    )

    params = suggest_parameters(
        trial,  # type: ignore[arg-type]
        "XGBClassifier",
    )

    assert params["n_estimators"] == 300
    assert params["max_depth"] == 6
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.9)
    assert params["reg_alpha"] == pytest.approx(0.1)
    assert params["reg_lambda"] == pytest.approx(1.0)


def test_lightgbm_search_space() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 0.5,
        }
    )

    params = suggest_parameters(
        trial,  # type: ignore[arg-type]
        "LGBMClassifier",
    )

    assert params["n_estimators"] == 200
    assert params["num_leaves"] == 31
    assert params["learning_rate"] == pytest.approx(0.1)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.8)
    assert params["reg_alpha"] == pytest.approx(0.01)
    assert params["reg_lambda"] == pytest.approx(0.5)


def test_unknown_model_search_space_raises() -> None:
    trial = optuna.trial.FixedTrial({})

    with pytest.raises(
        ValueError,
        match="search space is not configured",
    ):
        suggest_parameters(
            trial,  # type: ignore[arg-type]
            "UnsupportedClassifier",
        )


def test_build_model_applies_overrides() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "_class_": ("sklearn.ensemble.RandomForestClassifier"),
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            }
        }
    )

    model = build_model(
        cfg,
        overrides={
            "n_estimators": 250,
            "max_depth": 16,
        },
    )

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.n_estimators == 250
    assert model.max_depth == 16

    # Values not tuned should still come
    # from the Hydra model configuration.
    assert model.min_samples_split == 5
    assert model.min_samples_leaf == 2
    assert model.random_state == 42


class FakeProbabilisticClassifier:
    """Minimal classifier for metric testing."""

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [0, 1, 1, 0],
            dtype=np.int64,
        )

    def predict_proba(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )


def test_evaluate_classifier_returns_binary_metrics() -> None:
    model = FakeProbabilisticClassifier()

    x = np.zeros(
        (4, 2),
        dtype=np.float64,
    )

    y = np.array(
        [0, 1, 1, 0],
        dtype=np.int64,
    )

    metrics = evaluate_classifier(
        model,
        x,
        y,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


class FakeClassifierWithoutProbabilities:
    """Classifier without predict_proba."""

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            [0, 1],
            dtype=np.int64,
        )


def test_evaluate_classifier_handles_no_probabilities() -> None:
    model = FakeClassifierWithoutProbabilities()

    x = np.zeros(
        (2, 1),
        dtype=np.float64,
    )

    y = np.array(
        [0, 1],
        dtype=np.int64,
    )

    metrics = evaluate_classifier(
        model,
        x,
        y,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)

    assert "roc_auc" not in metrics


def test_prepare_sqlite_storage_creates_parent_directory(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "database" / "study.db"

    uri = f"sqlite:///{database_path}"

    assert not database_path.parent.exists()

    prepare_sqlite_storage(uri)

    assert database_path.parent.exists()


def test_prepare_sqlite_storage_ignores_non_sqlite_uri(
    tmp_path: Path,
) -> None:
    target = tmp_path / "should-not-exist"

    prepare_sqlite_storage("http://127.0.0.1:5000")

    assert not target.exists()
