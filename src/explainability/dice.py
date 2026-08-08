"""DiCE counterfactual explanation generator over the raw feature space."""

from __future__ import annotations

import json
import logging
from typing import Any

import dice_ml
import numpy as np
import pandas as pd
import polars as pl

from src.features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class FeatureEngineeringModelAdapter:
    """Expose a raw-feature sklearn-like interface to an encoded-feature model."""

    def __init__(self, model: Any, feature_engineer: FeatureEngineer) -> None:
        self.model = model
        self.feature_engineer = feature_engineer
        self.classes_ = getattr(model, "classes_", np.array([0, 1]))

    def _to_polars(self, x: Any) -> pl.DataFrame:
        if isinstance(x, pl.DataFrame):
            return x

        if isinstance(x, dict):
            x = pd.DataFrame([x])

        if not isinstance(x, pd.DataFrame):
            raise TypeError(f"Unsupported input type for prediction: {type(x)!r}")

        # DiCE may create candidate DataFrames containing mixed
        # pandas object/float dtypes. Normalize them before converting
        # back to Polars.
        df = x.copy()

        for col in self.feature_engineer.numeric_columns:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float64)
                )

        for col in self.feature_engineer.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        return pl.from_pandas(df)

    def _transform(self, x: Any) -> np.ndarray:
        raw = self._to_polars(x)
        encoded = self.feature_engineer.transform(raw)
        return encoded.to_numpy()

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform(x)))

    def predict_proba(self, x: Any) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "DiCE classification requires a model with predict_proba()."
            )
        return np.asarray(self.model.predict_proba(self._transform(x)))


class DiceExplainer:
    """Generate human-readable counterfactuals in the original Adult feature space."""

    DEFAULT_FEATURES_TO_VARY = [
        "workclass",
        "education",
        "occupation",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]

    IMMUTABLE_FEATURES = [
        "race",
        "sex",
        "native_country",
    ]

    NON_ACTIONABLE_FEATURES = [
        "fnlwgt",
        "education_num",
        "age",
        "marital_status",
        "relationship",
    ]

    def __init__(
        self,
        model: Any,
        data_df: pl.DataFrame,
        target_values: pl.Series | np.ndarray | list[int],
        feature_engineer: FeatureEngineer,
        target: str = "target",
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
        method: str = "random",
    ) -> None:
        if not feature_engineer.is_fitted_:
            raise ValueError("feature_engineer must be fitted before DiceExplainer.")

        self.model = model
        self.feature_engineer = feature_engineer
        self.target = target
        self.method = method

        self.raw_data = feature_engineer.clean_data(data_df)
        self.raw_columns = self.raw_data.columns

        if categorical_features is None:
            categorical_features = [
                col
                for col in self.raw_columns
                if self.raw_data.schema[col]
                in (pl.String, pl.Utf8, pl.Categorical, pl.Enum)
            ]

        categorical_features = [
            col for col in categorical_features if col in self.raw_columns
        ]

        if continuous_features is None:
            continuous_features = [
                col for col in self.raw_columns if col not in categorical_features
            ]

        continuous_features = [
            col for col in continuous_features if col in self.raw_columns
        ]

        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        target_array = np.asarray(
            target_values.to_numpy()
            if isinstance(target_values, pl.Series)
            else target_values
        ).reshape(-1)

        if len(target_array) != self.raw_data.height:
            raise ValueError("target_values length must match data_df height.")

        dice_frame = self.raw_data.to_pandas()
        dice_frame = self._sanitize_pandas(dice_frame)

        dice_frame[self.target] = target_array.astype(np.int32)

        self.adapter = FeatureEngineeringModelAdapter(model, feature_engineer)

        logger.info(
            "DiCE dataframe dtypes:\n%s",
            dice_frame.dtypes,
        )

        self.dice_data = dice_ml.Data(
            dataframe=dice_frame,
            continuous_features=self.continuous_features,
            categorical_features=self.categorical_features,
            outcome_name=self.target,
        )
        self.dice_model = dice_ml.Model(
            model=self.adapter,
            backend="sklearn",
            model_type="classifier",
        )
        self.explainer = dice_ml.Dice(
            data_interface=self.dice_data,
            model_interface=self.dice_model,
            method=self.method,
        )

        self._ranges = self._continuous_ranges(self.raw_data)

        self._bounds = self._continuous_bounds(self.raw_data)

        logger.info(
            "DiCE explainer initialized in raw feature space (%s method)",
            self.method,
        )

    def _continuous_ranges(self, df: pl.DataFrame) -> dict[str, float]:
        ranges: dict[str, float] = {}
        for col in self.continuous_features:
            series = df[col].cast(pl.Float64, strict=False)
            minimum = series.min()
            maximum = series.max()
            if minimum is None or maximum is None:
                ranges[col] = 1.0
            else:
                ranges[col] = max(float(maximum) - float(minimum), 1.0)
        return ranges

    def _continuous_bounds(
        self,
        df: pl.DataFrame,
    ) -> dict[str, list[float]]:
        """Return observed training bounds for continuous features."""
        bounds: dict[str, list[float]] = {}

        for col in self.continuous_features:
            if col not in df.columns:
                continue

            series = df[col].cast(
                pl.Float64,
                strict=False,
            )

            minimum = series.min()
            maximum = series.max()

            if minimum is None or maximum is None:
                continue

            bounds[col] = [
                float(minimum),
                float(maximum),
            ]

        return bounds

    def _default_permitted_range(
        self,
    ) -> dict[str, list[float]]:
        """Return practical ranges for actionable continuous features."""
        ranges: dict[str, list[float]] = {}

        if "hours_per_week" in self._bounds:
            ranges["hours_per_week"] = [
                1.0,
                60.0,
            ]

        if "capital_gain" in self._bounds:
            ranges["capital_gain"] = self._bounds["capital_gain"]

        if "capital_loss" in self._bounds:
            ranges["capital_loss"] = self._bounds["capital_loss"]

        return ranges

    @staticmethod
    def _python_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if pd.isna(value):
            return None
        return value

    def _row_to_dict(self, row: pd.Series) -> dict[str, Any]:
        return {
            col: self._python_value(row[col])
            for col in self.raw_columns
            if col in row.index
        }

    def _mixed_distance(self, original: pd.Series, counterfactual: pd.Series) -> float:
        """Gower-style distance across raw continuous and categorical features."""
        distances: list[float] = []

        for col in self.continuous_features:
            if col not in original.index or col not in counterfactual.index:
                continue
            try:
                left = float(original[col])
                right = float(counterfactual[col])
            except (TypeError, ValueError):
                continue
            distances.append(abs(left - right) / self._ranges[col])

        for col in self.categorical_features:
            if col not in original.index or col not in counterfactual.index:
                continue
            distances.append(float(original[col] != counterfactual[col]))

        return float(np.mean(distances)) if distances else 0.0

    def generate_counterfactuals(
        self,
        query_instance: pl.DataFrame | pd.DataFrame | dict[str, Any],
        total_cfs: int = 5,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] | None = None,
        permitted_range: dict[str, list[float]] | None = None,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate counterfactuals for one raw-feature query instance."""
        if isinstance(query_instance, pl.DataFrame):
            query_df = query_instance.to_pandas()
        elif isinstance(query_instance, dict):
            query_df = pd.DataFrame([query_instance])
        else:
            query_df = query_instance.copy()

        query_df = query_df[self.raw_columns]

        # DiCE rejects query instances containing missing values
        query_df = self._sanitize_pandas(query_df)

        if len(query_df) != 1:
            raise ValueError("generate_counterfactuals expects exactly one row.")

        if features_to_vary is None:
            features_to_vary = [
                feature
                for feature in self.DEFAULT_FEATURES_TO_VARY
                if feature in self.raw_columns
            ]

        if permitted_range is None:
            permitted_range = self._default_permitted_range()

        kwargs: dict[str, Any] = {
            "total_CFs": total_cfs,
            "desired_class": desired_class,
            "features_to_vary": features_to_vary,
            "permitted_range": permitted_range,
        }

        if random_seed is not None and self.method == "random":
            kwargs["random_seed"] = random_seed

        try:
            explanation = self.explainer.generate_counterfactuals(query_df, **kwargs)
            cf_df = explanation.cf_examples_list[0].final_cfs_df

            if cf_df is None or cf_df.empty:
                return {"error": "DiCE returned no counterfactuals."}

            if self.target in cf_df.columns:
                cf_df = cf_df.drop(columns=[self.target])

            cf_df = cf_df[self.raw_columns].reset_index(drop=True)
            original = query_df.iloc[0]

            valid_rows = [
                self._validate_counterfactual(
                    original,
                    cf_df.iloc[i],
                )
                for i in range(len(cf_df))
            ]

            cf_df = cf_df.loc[valid_rows].reset_index(drop=True)

            if cf_df.empty:
                return {"error": "DiCE returned no feasible counterfactuals."}

            distances = [
                self._mixed_distance(original, cf_df.iloc[i]) for i in range(len(cf_df))
            ]

            return {
                "counterfactuals": cf_df,
                "distances": distances,
                "original_instance": self._row_to_dict(original),
                "desired_class": desired_class,
            }
        except Exception as exc:
            logger.exception("Error generating counterfactuals")
            return {"error": str(exc)}

    def generate_batch(
        self,
        x_test: pl.DataFrame,
        num_samples: int = 100,
        total_cfs: int = 5,
        desired_class: str | int = "opposite",
        features_to_vary: str | list[str] | None = None,
        permitted_range: dict[str, list[float]] | None = None,
        random_seed: int = 42,
        selected_indices: list[int] | np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Generate semantic counterfactual records for multiple raw test rows."""
        if not isinstance(x_test, pl.DataFrame):
            raise TypeError("x_test must be a raw Polars DataFrame.")

        if selected_indices is not None:
            selected = np.asarray(
                selected_indices,
                dtype=int,
            )

            if selected.ndim != 1:
                raise ValueError("selected_indices must be one-dimensional.")

            if np.any(selected < 0) or np.any(selected >= x_test.height):
                raise ValueError("selected_indices contains an invalid row index.")

        else:
            rng = np.random.default_rng(random_seed)
            count = min(num_samples, x_test.height)

            if x_test.height > count:
                selected = np.sort(
                    rng.choice(
                        x_test.height,
                        count,
                        replace=False,
                    )
                )
            else:
                selected = np.arange(x_test.height)

        results: list[dict[str, Any]] = []
        logger.info("Generating counterfactuals for %d instances...", len(selected))

        for position, row_index in enumerate(selected, start=1):
            logger.info(
                "Processing counterfactual instance %d/%d",
                position,
                len(selected),
            )
            query = x_test.slice(int(row_index), 1)
            result = self.generate_counterfactuals(
                query,
                total_cfs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
                random_seed=random_seed + int(row_index),
            )

            if "error" in result:
                logger.warning(
                    "Skipping instance %d: %s",
                    row_index,
                    result["error"],
                )
                continue

            query_pd = query.to_pandas()[self.raw_columns]
            original_class = int(self.adapter.predict(query_pd)[0])

            cf_df: pd.DataFrame = result["counterfactuals"]
            cf_classes = self.adapter.predict(cf_df)

            for j in range(len(cf_df)):
                cf_dict = self._row_to_dict(cf_df.iloc[j])
                results.append(
                    {
                        "instance_id": int(row_index),
                        "original_features": json.dumps(
                            result["original_instance"],
                            sort_keys=True,
                        ),
                        "counterfactual_features": json.dumps(
                            cf_dict,
                            sort_keys=True,
                        ),
                        "target_class": int(cf_classes[j]),
                        "original_class": original_class,
                        "distance": float(result["distances"][j]),
                    }
                )

        if not results:
            return pl.DataFrame(
                schema={
                    "instance_id": pl.Int64,
                    "original_features": pl.String,
                    "counterfactual_features": pl.String,
                    "target_class": pl.Int64,
                    "original_class": pl.Int64,
                    "distance": pl.Float64,
                }
            )

        df = pl.DataFrame(results)
        logger.info("Generated %d counterfactuals", df.height)
        return df

    def _validate_counterfactual(
        self,
        original: pd.Series,
        counterfactual: pd.Series,
    ) -> bool:
        """Validate that a generated counterfactual respects feasibility rules."""

        frozen_features = self.IMMUTABLE_FEATURES + self.NON_ACTIONABLE_FEATURES

        for feature in frozen_features:
            if (
                feature in original.index
                and feature in counterfactual.index
                and original[feature] != counterfactual[feature]
            ):
                return False

        for feature, bounds in self._default_permitted_range().items():
            if feature not in counterfactual.index:
                continue

            value = float(counterfactual[feature])
            lower, upper = bounds

            if value < lower or value > upper:
                return False

        return True

    def _sanitize_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw feature dtypes and remove missing values for DiCE."""
        df = df.copy()

        for col in self.continuous_features:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float64)
                )

        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        return df
