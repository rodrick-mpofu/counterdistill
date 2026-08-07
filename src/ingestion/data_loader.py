"""Data loading module for the Adult Income dataset using Polars."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class AdultIncomeLoader:
    """Loader for the Adult Income dataset."""

    def __init__(
        self,
        data_dir: str = "data",
        raw_filename: str = "adult.data",
        processed_filename: str = "adult_processed.parquet",
    ):
        self.data_dir = Path(data_dir)
        self.raw_path = self.data_dir / "raw" / raw_filename
        self.processed_path = self.data_dir / "processed" / processed_filename

        # Column names based on UCI Adult dataset
        self.column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ]

        # Categorical and numerical column definitions
        self.categorical_features = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]

        self.numerical_features = [
            "age",
            "fnlwgt",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ]

    def load_raw(self) -> pl.DataFrame:
        """Load raw data from CSV file."""
        logger.info(f"Loading raw data from {self.raw_path}")

        if not self.raw_path.exists():
            # Download the data if it doesn't exist
            self.download_data()

        # Load with polars
        df = pl.read_csv(
            self.raw_path,
            has_header=False,
            new_columns=self.column_names,
            null_values=[" ?", " ?", "?"],
            ignore_errors=True,  # Skip malformed rows
        )

        logger.info(f"Loaded {df.height} rows and {df.width} columns")
        return df

    def download_data(self) -> None:
        """Download the Adult Income dataset if not available."""
        import urllib.request

        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading data from {url}")
        urllib.request.urlretrieve(url, self.raw_path)
        logger.info("Download complete!")

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess the data using Polars."""
        logger.info("Preprocessing data...")

        # Clean column names (strip whitespace)
        df = df.with_columns(
            [
                pl.col(col).str.strip_chars()
                for col in self.categorical_features
                if col in df.columns
            ]
        )

        # Clean income target - handle potential issues
        df = df.with_columns(
            pl.col("income")
            .str.strip_chars()
            .str.replace(",", "")
            .str.replace(" ", "")
            .alias("income")
        )

        # Filter out rows with empty or invalid income
        df = df.filter((pl.col("income") == "<=50K") | (pl.col("income") == ">50K"))

        # Convert target to binary (1 if >50K, 0 otherwise)
        df = df.with_columns(
            (pl.col("income") == ">50K").cast(pl.Int32).alias("target")
        )

        # Drop the original income column
        df = df.drop("income")

        # Drop any rows with null values in target
        df = df.drop_nulls(subset=["target"])

        logger.info(f"Preprocessing complete! Rows: {df.height}, Columns: {df.width}")
        return df

    def process(self) -> pl.DataFrame:
        """Complete data processing pipeline."""
        # Load raw data
        df_raw = self.load_raw()

        # Preprocess
        df_processed = self.preprocess(df_raw)

        # Save processed data
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.write_parquet(self.processed_path)

        logger.info(f"Saved processed data to {self.processed_path}")
        return df_processed

    def load_processed(self) -> pl.DataFrame:
        """Load processed data from parquet file."""
        if self.processed_path.exists():
            return pl.read_parquet(self.processed_path)
        else:
            return self.process()

    def get_features_and_target(
        self, df: pl.DataFrame | None = None
    ) -> tuple[pl.DataFrame, pl.Series]:
        """Separate features and target."""
        if df is None:
            df = self.load_processed()

        # Separate features and target
        x = df.drop("target")
        y = df["target"]

        return x, y

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """Split data into train and test sets while preserving Polars dtypes."""
        from sklearn.model_selection import train_test_split

        x, y = self.get_features_and_target()

        # Create row indices
        indices = list(range(x.height))

        # Split indices using the target for stratification
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y.to_numpy(),
        )

        # Slice the Polars DataFrames directly
        x_train = x[train_idx]
        x_test = x[test_idx]

        # Slice the target Series directly
        y_train = y[train_idx]
        y_test = y[test_idx]

        logger.info(
            "Train/test split complete: "
            f"{x_train.height} train, {x_test.height} test samples"
        )

        return x_train, x_test, y_train, y_test
