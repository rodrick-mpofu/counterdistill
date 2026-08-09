"""Encode semantic counterfactual changes for clustering."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class CounterfactualEncoder:
    """Encode original -> counterfactual transitions into numeric vectors.

    The encoded representation focuses on what changed rather than the
    absolute state of the original observation.

    Numeric features are represented by signed normalized deltas.
    Categorical features are represented by one-hot transition indicators.
    """

    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.numeric_features = numeric_features or [
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ]

        self.categorical_features = categorical_features or [
            "workclass",
            "education",
            "occupation",
        ]

        self.numeric_ranges_: dict[str, float] = {}
        self.transition_columns_: list[str] = []
        self.feature_names_: list[str] = []
        self.is_fitted_: bool = False

    @staticmethod
    def _parse_json(
        value: str | dict[str, Any],
    ) -> dict[str, Any]:
        """Parse a DuckDB JSON value into a Python dictionary."""
        if isinstance(value, dict):
            return value

        parsed = json.loads(value)

        if not isinstance(parsed, dict):
            raise ValueError("Expected counterfactual JSON to contain an object.")

        return parsed

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        """Convert a raw value to float, defaulting missing values to zero."""
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _transition_column(
        feature: str,
        original: Any,
        counterfactual: Any,
    ) -> str:
        """Build a stable categorical transition column name."""
        source = str(original).strip().replace(" ", "_")
        destination = str(counterfactual).strip().replace(" ", "_")

        return f"{feature}__transition__" f"{source}__to__{destination}"

    def _extract_records(
        self,
        df: pl.DataFrame,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Parse original/counterfactual JSON pairs."""
        required = {
            "original_features",
            "counterfactual_features",
        }

        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                "Counterfactual DataFrame is missing required columns: "
                f"{sorted(missing)}"
            )

        records: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for row in df.iter_rows(named=True):
            original = self._parse_json(row["original_features"])
            counterfactual = self._parse_json(row["counterfactual_features"])

            records.append((original, counterfactual))

        return records

    def _fit_numeric_ranges(
        self,
        records: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Learn normalization ranges from the explanation corpus."""
        self.numeric_ranges_.clear()

        for feature in self.numeric_features:
            values: list[float] = []

            for original, counterfactual in records:
                if feature in original:
                    values.append(self._safe_float(original[feature]))

                if feature in counterfactual:
                    values.append(self._safe_float(counterfactual[feature]))

            if not values:
                self.numeric_ranges_[feature] = 1.0
                continue

            feature_range = max(values) - min(values)

            self.numeric_ranges_[feature] = max(
                feature_range,
                1.0,
            )

    def _fit_transition_schema(
        self,
        records: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Learn categorical transition columns present in the corpus."""
        transitions: set[str] = set()

        for original, counterfactual in records:
            for feature in self.categorical_features:
                original_value = original.get(feature)
                counterfactual_value = counterfactual.get(feature)

                if original_value == counterfactual_value:
                    continue

                transitions.add(
                    self._transition_column(
                        feature,
                        original_value,
                        counterfactual_value,
                    )
                )

        self.transition_columns_ = sorted(transitions)

    def fit(
        self,
        df: pl.DataFrame,
    ) -> CounterfactualEncoder:
        """Learn the encoding schema from counterfactual records."""
        records = self._extract_records(df)

        if not records:
            raise ValueError("Cannot fit CounterfactualEncoder on an empty DataFrame.")

        self._fit_numeric_ranges(records)
        self._fit_transition_schema(records)

        numeric_columns = [f"{feature}__delta" for feature in self.numeric_features]

        numeric_change_columns = [
            f"{feature}__changed" for feature in self.numeric_features
        ]

        categorical_change_columns = [
            f"{feature}__changed" for feature in self.categorical_features
        ]

        self.feature_names_ = (
            numeric_columns
            + numeric_change_columns
            + categorical_change_columns
            + self.transition_columns_
        )

        self.is_fitted_ = True

        logger.info(
            "CounterfactualEncoder fitted with %d clustering features",
            len(self.feature_names_),
        )

        return self

    def transform(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Encode counterfactual transitions using the fitted schema."""
        if not self.is_fitted_:
            raise RuntimeError(
                "CounterfactualEncoder must be fitted before transform()."
            )

        records = self._extract_records(df)

        encoded_rows: list[dict[str, Any]] = []

        source_rows = df.iter_rows(named=True)

        for source_row, (
            original,
            counterfactual,
        ) in zip(
            source_rows,
            records,
            strict=True,
        ):
            encoded: dict[str, Any] = {}

            # Keep provenance but do not use it as clustering input.
            for metadata_column in (
                "id",
                "run_id",
                "instance_id",
                "model_name",
                "distance",
            ):
                if metadata_column in source_row:
                    encoded[metadata_column] = source_row[metadata_column]

            for feature in self.numeric_features:
                original_value = self._safe_float(original.get(feature))
                counterfactual_value = self._safe_float(counterfactual.get(feature))

                raw_delta = counterfactual_value - original_value

                feature_range = self.numeric_ranges_.get(
                    feature,
                    1.0,
                )

                encoded[f"{feature}__delta"] = raw_delta / feature_range

                encoded[f"{feature}__changed"] = int(
                    not np.isclose(
                        original_value,
                        counterfactual_value,
                    )
                )

            for feature in self.categorical_features:
                original_category = original.get(feature)
                counterfactual_category = counterfactual.get(feature)

                changed = original_category != counterfactual_category

                encoded[f"{feature}__changed"] = int(changed)

                if changed:
                    transition = self._transition_column(
                        feature,
                        original_category,
                        counterfactual_category,
                    )

                    if transition in self.transition_columns_:
                        encoded[transition] = 1

            # Fill every absent transition with zero.
            for column in self.transition_columns_:
                encoded.setdefault(column, 0)

            encoded_rows.append(encoded)

        if not encoded_rows:
            return pl.DataFrame()

        return pl.DataFrame(encoded_rows)

    def fit_transform(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Fit the encoder and transform the supplied records."""
        return self.fit(df).transform(df)

    def clustering_matrix(
        self,
        df: pl.DataFrame,
    ) -> np.ndarray:
        """Return only numeric clustering features as a NumPy matrix."""
        if not self.is_fitted_:
            raise RuntimeError(
                "CounterfactualEncoder must be fitted before " "clustering_matrix()."
            )

        missing = [column for column in self.feature_names_ if column not in df.columns]

        if missing:
            raise ValueError(
                "Encoded DataFrame is missing clustering features: " f"{missing}"
            )

        return df.select(self.feature_names_).cast(pl.Float64).to_numpy()
