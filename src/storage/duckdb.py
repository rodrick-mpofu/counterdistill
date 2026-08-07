"""DuckDB storage module for counterfactuals and explanations."""

import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class DuckDBStorage:
    """DuckDB storage for counterfactual explanations."""

    def __init__(self, db_path: str = "database/counterdistill.db"):
        """Initialize DuckDB storage."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with duckdb.connect(str(self.db_path)) as conn:
            # Create counterfactuals table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS counterfactuals (
                    id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    instance_id INTEGER,
                    original_features JSON,
                    counterfactual_features JSON,
                    target_class INTEGER,
                    original_class INTEGER,
                    distance FLOAT,
                    created_at TIMESTAMP
                )
            """
            )

            # Create SHAP values table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shap_values (
                    id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    instance_id INTEGER,
                    feature_name VARCHAR,
                    shap_value FLOAT,
                    feature_value FLOAT,
                    created_at TIMESTAMP
                )
            """
            )

            # Create explanations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS explanations (
                    id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    instance_id INTEGER,
                    explanation_type VARCHAR,
                    explanation JSON,
                    created_at TIMESTAMP
                )
            """
            )

            # Create metrics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    model_name VARCHAR,
                    metric_name VARCHAR,
                    metric_value FLOAT,
                    run_id VARCHAR,
                    created_at TIMESTAMP
                )
            """
            )

            logger.info(f"Database initialized at {self.db_path}")

    def store_counterfactuals(
        self, df: pl.DataFrame, model_name: str, run_id: str
    ) -> None:
        """Store counterfactual explanations in DuckDB."""
        with duckdb.connect(str(self.db_path)) as conn:
            # Convert to DuckDB
            conn.register("temp_df", df.to_pandas())

            # Insert counterfactuals
            conn.execute(
                """
                INSERT INTO counterfactuals (
                    model_name,
                    instance_id,
                    original_features,
                    counterfactual_features,
                    target_class,
                    original_class,
                    distance,
                    created_at
                )
                SELECT
                    ?,
                    instance_id,
                    original_features,
                    counterfactual_features,
                    target_class,
                    original_class,
                    distance,
                    CURRENT_TIMESTAMP
                FROM temp_df
            """,
                [model_name],
            )

            logger.info(f"Stored {df.height} counterfactuals for {model_name}")

    def store_shap_values(self, df: pl.DataFrame, model_name: str, run_id: str) -> None:
        """Store SHAP values in DuckDB."""
        with duckdb.connect(str(self.db_path)) as conn:
            conn.register("temp_df", df.to_pandas())

            conn.execute(
                """
                INSERT INTO shap_values (
                    model_name,
                    instance_id,
                    feature_name,
                    shap_value,
                    feature_value,
                    created_at
                )
                SELECT
                    ?,
                    instance_id,
                    feature_name,
                    shap_value,
                    feature_value,
                    CURRENT_TIMESTAMP
                FROM temp_df
            """,
                [model_name],
            )

            logger.info(f"Stored {df.height} SHAP values for {model_name}")

    def query_counterfactuals(
        self, model_name: str | None = None, limit: int = 100
    ) -> pl.DataFrame:
        """Query counterfactuals from DuckDB."""
        with duckdb.connect(str(self.db_path)) as conn:
            query = "SELECT * FROM counterfactuals"
            if model_name:
                query += f" WHERE model_name = '{model_name}'"
            query += f" ORDER BY created_at DESC LIMIT {limit}"

            result = conn.execute(query).fetchdf()
            return pl.from_pandas(result)

    def get_metrics(self, model_name: str | None = None) -> dict[str, float]:
        """Get metrics for a model."""
        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute(
                """
                SELECT metric_name, metric_value
                FROM metrics
                WHERE model_name = ?
                ORDER BY created_at DESC
            """,
                [model_name],
            ).fetchdf()

            if len(result) > 0:
                return {row[0]: row[1] for row in result.itertuples(index=False)}
            return {}
