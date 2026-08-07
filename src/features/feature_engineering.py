"""Feature engineering module using Polars."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline using Polars."""

    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _clean_numeric_column(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Clean a column and convert to numeric."""
        if col in df.columns:
            # First convert to string, clean, then cast to float
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.replace_all(",", "")
                .str.replace_all(" ", "")
                .cast(pl.Float64, strict=False)
                .fill_null(0)
                .alias(col)
            )
        return df

    def create_features(self) -> pl.DataFrame:
        """Create additional features."""
        df = self.df.clone()

        # Clean numeric columns first
        numeric_cols = [
            "age",
            "fnlwgt",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = self._clean_numeric_column(df, col)

        # Age groups - manual approach
        if "age" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("age") <= 18)
                .then("child")
                .when(pl.col("age") <= 25)
                .then("teen")
                .when(pl.col("age") <= 35)
                .then("young")
                .when(pl.col("age") <= 45)
                .then("young_adult")
                .when(pl.col("age") <= 55)
                .then("middle_age")
                .when(pl.col("age") <= 65)
                .then("senior")
                .otherwise("elderly")
                .alias("age_group")
            )

        # Capital ratio
        if (
            "capital_gain" in df.columns
            and "capital_loss" in df.columns
            and "fnlwgt" in df.columns
        ):
            df = df.with_columns(
                (
                    (pl.col("capital_gain") - pl.col("capital_loss"))
                    / (pl.col("fnlwgt") + 1)
                )
                .fill_null(0)
                .cast(pl.Float64)
                .alias("capital_ratio")
            )

        # Hours category - manual approach
        if "hours_per_week" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("hours_per_week") <= 0)
                .then("none")
                .when(pl.col("hours_per_week") <= 20)
                .then("part_time")
                .when(pl.col("hours_per_week") <= 30)
                .then("reduced")
                .when(pl.col("hours_per_week") <= 40)
                .then("full_time")
                .when(pl.col("hours_per_week") <= 50)
                .then("overtime")
                .otherwise("extreme")
                .alias("hours_category")
            )

        # Education level - manual approach
        if "education_num" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("education_num") <= 0)
                .then("no_edu")
                .when(pl.col("education_num") <= 6)
                .then("basic")
                .when(pl.col("education_num") <= 10)
                .then("high_school")
                .when(pl.col("education_num") <= 12)
                .then("some_college")
                .when(pl.col("education_num") <= 14)
                .then("bachelor")
                .otherwise("advanced")
                .alias("education_level")
            )

        return df

    def encode_categorical(self, df: pl.DataFrame) -> pl.DataFrame:
        """Encode categorical features using Polars."""
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

        existing_cat_cols = [col for col in cat_cols if col in df.columns]
        df_encoded = df.clone()

        for col in existing_cat_cols:
            try:
                # Clean categorical columns
                df_encoded = df_encoded.with_columns(
                    pl.col(col).cast(pl.Utf8).str.strip_chars().alias(col)
                )
                dummies = df_encoded.select(
                    pl.col(col).cast(pl.Categorical).to_dummies(separator="_")
                )
                df_encoded = df_encoded.drop(col)
                df_encoded = pl.concat([df_encoded, dummies], how="horizontal")
            except Exception as e:
                logger.warning(f"Could not encode column {col}: {e}")
                continue

        return df_encoded

    def build_pipeline(self, scale: bool = False) -> pl.DataFrame:
        """Build complete feature engineering pipeline."""
        logger.info("Building feature engineering pipeline...")

        try:
            df_features = self.create_features()
            df_encoded = self.encode_categorical(df_features)

            if scale:
                # Scale numerical features
                numeric_cols = [
                    "age",
                    "fnlwgt",
                    "education_num",
                    "capital_gain",
                    "capital_loss",
                    "hours_per_week",
                    "capital_ratio",
                ]
                df_final = df_encoded.clone()
                for col in numeric_cols:
                    if col in df_final.columns:
                        try:
                            mean = df_final[col].mean()
                            std = df_final[col].std()
                            if std is not None and std > 0:
                                df_final = df_final.with_columns(
                                    ((pl.col(col) - mean) / std)
                                    .fill_null(0)
                                    .alias(f"{col}_scaled")
                                )
                                df_final = df_final.drop(col)
                        except Exception as e:
                            logger.warning(f"Could not scale column {col}: {e}")
                            continue
            else:
                df_final = df_encoded

            logger.info(f"Feature engineering complete: {df_final.width} features")
            return df_final

        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            raise
