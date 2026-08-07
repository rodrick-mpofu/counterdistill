"""Feature engineering module using Polars with configuration-driven design."""

import logging
from typing import Any

import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline using Polars with vectorized operations.

    This class follows modern Polars patterns:
    - Single expression graphs for each transformation
    - Minimal DataFrame copying
    - Fail-fast error handling
    - Configuration-driven design
    """

    def __init__(
        self,
        df: pl.DataFrame,
        config: DictConfig | None = None,
    ):
        """
        Initialize the feature engineer.

        Args:
            df: Input Polars DataFrame
            config: Hydra configuration for feature engineering
        """
        self.df = df
        self.config = config

        # Default configuration if not provided
        if self.config is None:
            self.config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for feature engineering."""
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

    def clean_data(self) -> pl.DataFrame:
        """
        Clean numeric and categorical columns in one vectorized operation.

        Returns:
            Cleaned Polars DataFrame
        """
        numeric_cols = self.config.get("numeric_columns", [])  # type: ignore
        categorical_cols = self.config.get("categorical_columns", [])  # type: ignore

        # Build cleaning expressions
        clean_exprs = []

        # Clean numeric columns
        for col in numeric_cols:
            if col in self.df.columns:
                clean_exprs.append(
                    pl.col(col).cast(pl.Float64, strict=False).fill_null(0).alias(col)
                )

        # Clean categorical columns
        for col in categorical_cols:
            if col in self.df.columns:
                clean_exprs.append(
                    pl.col(col).cast(pl.Utf8).str.strip_chars().alias(col)
                )

        # Execute all cleaning in one with_columns call
        return self.df.with_columns(clean_exprs)

    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create new features using vectorized expressions.

        Args:
            df: Input DataFrame to add features to

        Returns:
            DataFrame with new features added
        """
        # Build feature expressions
        feature_exprs = []

        # Age group expression
        age_bins = self.config.get("age_bins", [18, 25, 35, 45, 55, 65])  # type: ignore
        age_labels = self.config.get(  # type: ignore
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

        if "age" in df.columns:
            age_expr = pl.when(pl.col("age") <= age_bins[0]).then(pl.lit(age_labels[0]))
            for i in range(len(age_bins) - 1):
                age_expr = age_expr.when(
                    (pl.col("age") > age_bins[i]) & (pl.col("age") <= age_bins[i + 1])
                ).then(pl.lit(age_labels[i + 1]))
            age_expr = age_expr.otherwise(pl.lit(age_labels[-1])).alias("age_group")
            feature_exprs.append(age_expr)

        # Hours category expression
        hours_bins = self.config.get("hours_bins", [0, 20, 30, 40, 50])  # type: ignore
        hours_labels = self.config.get(  # type: ignore
            "hours_labels",
            ["none", "part_time", "reduced", "full_time", "overtime", "extreme"],
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
            hours_expr = hours_expr.otherwise(pl.lit(hours_labels[-1])).alias(
                "hours_category"
            )
            feature_exprs.append(hours_expr)

        # Education level expression
        edu_bins = self.config.get("education_bins", [0, 6, 10, 12, 14])  # type: ignore
        edu_labels = self.config.get(  # type: ignore
            "education_labels",
            ["no_edu", "basic", "high_school", "some_college", "bachelor", "advanced"],
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
            edu_expr = edu_expr.otherwise(pl.lit(edu_labels[-1])).alias(
                "education_level"
            )
            feature_exprs.append(edu_expr)

        # Capital ratio expression
        if all(col in df.columns for col in ["capital_gain", "capital_loss", "fnlwgt"]):
            capital_expr = (
                (
                    (pl.col("capital_gain") - pl.col("capital_loss"))
                    / (pl.col("fnlwgt") + 1)
                )
                .fill_null(0)
                .cast(pl.Float64)
                .alias("capital_ratio")
            )
            feature_exprs.append(capital_expr)

        # Execute all feature creation in one with_columns call
        return df.with_columns(feature_exprs)

    def encode_categorical(self, df: pl.DataFrame) -> pl.DataFrame:
        """One-hot encode categorical features using Polars."""
        cat_cols = [
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

        existing_cols = [c for c in cat_cols if c in df.columns]

        if not existing_cols:
            return df

        # Clean string columns
        df = df.with_columns(
            [pl.col(c).cast(pl.Utf8).str.strip_chars().alias(c) for c in existing_cols]
        )

        # One-hot encode all categorical columns at once
        df = df.to_dummies(
            columns=existing_cols,
            separator="_",
            drop_first=False,
        )

        return df

    def scale_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Scale numerical features using standardization.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with scaled features
        """
        numeric_cols = self.config.get("numeric_columns", [])  # type: ignore
        numeric_cols.append("capital_ratio")

        # Collect all scaling expressions
        scale_exprs = []
        drop_cols = []

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # Compute mean and std
            mean = df[col].mean()
            std = df[col].std()

            if std is not None and std > 0:
                scale_exprs.append(
                    ((pl.col(col) - mean) / std).fill_null(0).alias(f"{col}_scaled")
                )
                drop_cols.append(col)

        # Apply scaling
        if scale_exprs:
            df = df.with_columns(scale_exprs)
            if drop_cols:
                df = df.drop(drop_cols)

        return df

    def build_pipeline(self, scale: bool | None = None) -> pl.DataFrame:
        """
        Build complete feature engineering pipeline.

        This method orchestrates all transformations in the optimal order:
        1. Clean data
        2. Create new features
        3. Encode categorical variables
        4. Scale numerical features (optional)

        Args:
            scale: Whether to scale numerical features (overrides config)

        Returns:
            Fully processed Polars DataFrame
        """
        logger.info("Building feature engineering pipeline...")

        # Determine if scaling should be applied
        if scale is None:
            scale = self.config.get("scale_features", False)  # type: ignore

        # Chain transformations with minimal DataFrame copies
        df = self.clean_data()
        df = self.create_features(df)
        df = self.encode_categorical(df)

        if scale:
            df = self.scale_features(df)

        logger.info(f"Feature engineering complete: {df.width} features")
        return df
