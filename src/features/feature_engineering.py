"""Feature engineering module using Polars with fitted train-only state."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Train-aware Polars feature engineering pipeline.

    The engineer learns all state from the training split only:
    - output dummy-column schema
    - optional scaling statistics

    Calling ``transform`` on validation/test/counterfactual data then aligns the
    output to exactly the feature order learned during ``fit``.
    """

    def __init__(
        self,
        df: pl.DataFrame | None = None,
        config: DictConfig | dict[str, Any] | None = None,
    ) -> None:
        self.df = df
        self.config = config if config is not None else self._get_default_config()

        self.feature_names_: list[str] = []
        self.scale_stats_: dict[str, tuple[float, float]] = {}
        self.scale_: bool = False
        self.is_fitted_: bool = False

    @staticmethod
    def _get_default_config() -> dict[str, Any]:
        return {
            "numeric_columns": [
                "age",
                "fnlwgt",
                "education_num",
                "capital_gain",
                "capital_loss",
                "hours_per_week",
            ],
            "categorical_columns": [
                "workclass",
                "education",
                "marital_status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native_country",
            ],
            "age_bins": [18, 25, 35, 45, 55, 65],
            "age_labels": [
                "child",
                "teen",
                "young",
                "young_adult",
                "middle_age",
                "senior",
                "elderly",
            ],
            "hours_bins": [0, 20, 30, 40, 50],
            "hours_labels": [
                "none",
                "part_time",
                "reduced",
                "full_time",
                "overtime",
                "extreme",
            ],
            "education_bins": [0, 6, 10, 12, 14],
            "education_labels": [
                "no_edu",
                "basic",
                "high_school",
                "some_college",
                "bachelor",
                "advanced",
            ],
            "scale_features": False,
        }

    def _get(self, key: str, default: Any) -> Any:
        if isinstance(self.config, DictConfig):
            return self.config.get(key, default)
        return self.config.get(key, default)

    def _resolve_df(self, df: pl.DataFrame | None) -> pl.DataFrame:
        resolved = df if df is not None else self.df
        if resolved is None:
            raise ValueError("No DataFrame provided to FeatureEngineer.")
        return resolved

    @property
    def numeric_columns(self) -> list[str]:
        return list(self._get("numeric_columns", []))

    @property
    def categorical_columns(self) -> list[str]:
        return list(self._get("categorical_columns", []))

    @property
    def raw_feature_names(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns

    def clean_data(self, df: pl.DataFrame | None = None) -> pl.DataFrame:
        """Clean numeric and categorical columns using vectorized expressions."""
        frame = self._resolve_df(df)
        exprs: list[pl.Expr] = []

        for col in self.numeric_columns:
            if col in frame.columns:
                exprs.append(
                    pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col)
                )

        for col in self.categorical_columns:
            if col in frame.columns:
                exprs.append(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .str.strip_chars()
                    .fill_null("Unknown")
                    .alias(col)
                )

        return frame.with_columns(exprs) if exprs else frame

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create deterministic derived features."""
        feature_exprs: list[pl.Expr] = []

        age_bins = list(self._get("age_bins", [18, 25, 35, 45, 55, 65]))
        age_labels = list(
            self._get(
                "age_labels",
                [
                    "child",
                    "teen",
                    "young",
                    "young_adult",
                    "middle_age",
                    "senior",
                    "elderly",
                ],
            )
        )
        if "age" in df.columns:
            age_expr = pl.when(pl.col("age") <= age_bins[0]).then(pl.lit(age_labels[0]))
            for i in range(len(age_bins) - 1):
                age_expr = age_expr.when(
                    (pl.col("age") > age_bins[i]) & (pl.col("age") <= age_bins[i + 1])
                ).then(pl.lit(age_labels[i + 1]))
            feature_exprs.append(
                age_expr.otherwise(pl.lit(age_labels[-1])).alias("age_group")
            )

        hours_bins = list(self._get("hours_bins", [0, 20, 30, 40, 50]))
        hours_labels = list(
            self._get(
                "hours_labels",
                ["none", "part_time", "reduced", "full_time", "overtime", "extreme"],
            )
        )
        if "hours_per_week" in df.columns:
            hours_expr = pl.when(pl.col("hours_per_week") <= hours_bins[0]).then(
                pl.lit(hours_labels[0])
            )
            for i in range(len(hours_bins) - 1):
                hours_expr = hours_expr.when(
                    (pl.col("hours_per_week") > hours_bins[i])
                    & (pl.col("hours_per_week") <= hours_bins[i + 1])
                ).then(pl.lit(hours_labels[i + 1]))
            feature_exprs.append(
                hours_expr.otherwise(pl.lit(hours_labels[-1])).alias("hours_category")
            )

        edu_bins = list(self._get("education_bins", [0, 6, 10, 12, 14]))
        edu_labels = list(
            self._get(
                "education_labels",
                [
                    "no_edu",
                    "basic",
                    "high_school",
                    "some_college",
                    "bachelor",
                    "advanced",
                ],
            )
        )
        if "education_num" in df.columns:
            edu_expr = pl.when(pl.col("education_num") <= edu_bins[0]).then(
                pl.lit(edu_labels[0])
            )
            for i in range(len(edu_bins) - 1):
                edu_expr = edu_expr.when(
                    (pl.col("education_num") > edu_bins[i])
                    & (pl.col("education_num") <= edu_bins[i + 1])
                ).then(pl.lit(edu_labels[i + 1]))
            feature_exprs.append(
                edu_expr.otherwise(pl.lit(edu_labels[-1])).alias("education_level")
            )

        if all(c in df.columns for c in ("capital_gain", "capital_loss", "fnlwgt")):
            feature_exprs.append(
                (
                    (pl.col("capital_gain") - pl.col("capital_loss"))
                    / (pl.col("fnlwgt") + 1.0)
                )
                .fill_nan(0.0)
                .fill_null(0.0)
                .cast(pl.Float64)
                .alias("capital_ratio")
            )

        return df.with_columns(feature_exprs) if feature_exprs else df

    @staticmethod
    def _categorical_features_present(df: pl.DataFrame) -> list[str]:
        candidates = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
            "age_group",
            "hours_category",
            "education_level",
        ]
        return [col for col in candidates if col in df.columns]

    def encode_categorical(self, df: pl.DataFrame) -> pl.DataFrame:
        """One-hot encode all categorical and derived categorical features."""
        existing = self._categorical_features_present(df)
        if not existing:
            return df

        cleaned = df.with_columns(
            [
                pl.col(col)
                .cast(pl.Utf8)
                .str.strip_chars()
                .fill_null("Unknown")
                .alias(col)
                for col in existing
            ]
        )
        return cleaned.to_dummies(
            columns=existing,
            separator="_",
            drop_first=False,
        )

    def _fit_scale_stats(self, df: pl.DataFrame) -> None:
        self.scale_stats_.clear()
        columns = self.numeric_columns + ["capital_ratio"]

        for col in columns:
            if col not in df.columns:
                continue

            mean = df[col].mean()
            std = df[col].std()
            if mean is None or std is None or std <= 0:
                continue

            self.scale_stats_[col] = (float(mean), float(std))

    def scale_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply scaling statistics learned by ``fit``."""
        if not self.scale_stats_:
            return df

        exprs = [
            ((pl.col(col) - mean) / std)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"{col}_scaled")
            for col, (mean, std) in self.scale_stats_.items()
            if col in df.columns
        ]
        drop_cols = [col for col in self.scale_stats_ if col in df.columns]

        if exprs:
            df = df.with_columns(exprs)
        if drop_cols:
            df = df.drop(drop_cols)
        return df

    def _prepare_before_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        prepared = self.clean_data(df)
        prepared = self.create_features(prepared)
        if self.scale_:
            prepared = self.scale_features(prepared)
        return prepared

    def fit(
        self,
        df: pl.DataFrame | None = None,
        scale: bool | None = None,
    ) -> FeatureEngineer:
        """Learn preprocessing state from training data only."""
        frame = self._resolve_df(df)
        self.scale_ = bool(
            self._get("scale_features", False) if scale is None else scale
        )

        prepared = self.clean_data(frame)
        prepared = self.create_features(prepared)

        if self.scale_:
            self._fit_scale_stats(prepared)
            prepared = self.scale_features(prepared)
        else:
            self.scale_stats_.clear()

        encoded = self.encode_categorical(prepared)
        self.feature_names_ = encoded.columns
        self.is_fitted_ = True

        logger.info(
            "FeatureEngineer fitted on %d rows with %d output features",
            frame.height,
            len(self.feature_names_),
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data and align it to the fitted training feature schema."""
        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before transform().")

        encoded = self.encode_categorical(self._prepare_before_encoding(df))

        missing = [col for col in self.feature_names_ if col not in encoded.columns]
        if missing:
            encoded = encoded.with_columns(
                [pl.lit(0, dtype=pl.UInt8).alias(col) for col in missing]
            )

        extra = [col for col in encoded.columns if col not in self.feature_names_]
        if extra:
            logger.debug("Dropping unseen encoded columns: %s", extra)
            encoded = encoded.drop(extra)

        return encoded.select(self.feature_names_)

    def fit_transform(
        self,
        df: pl.DataFrame | None = None,
        scale: bool | None = None,
    ) -> pl.DataFrame:
        """Fit on the supplied training data and transform it."""
        frame = self._resolve_df(df)
        self.fit(frame, scale=scale)
        return self.transform(frame)

    def build_pipeline(self, scale: bool | None = None) -> pl.DataFrame:
        """Backward-compatible entry point for the full pipeline."""
        frame = self._resolve_df(None)
        if self.is_fitted_:
            return self.transform(frame)
        return self.fit_transform(frame, scale=scale)
