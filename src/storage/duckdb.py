"""DuckDB storage for counterfactuals, SHAP values, and metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


class DuckDBStorage:
    """Persist explanation artifacts with model/run provenance."""

    def __init__(self, db_path: str = "database/counterdistill.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with duckdb.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS counterfactuals (
                    id BIGINT PRIMARY KEY,
                    model_name VARCHAR,
                    run_id VARCHAR,
                    instance_id BIGINT,
                    original_features JSON,
                    counterfactual_features JSON,
                    target_class INTEGER,
                    original_class INTEGER,
                    distance DOUBLE,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shap_values (
                    id BIGINT PRIMARY KEY,
                    model_name VARCHAR,
                    run_id VARCHAR,
                    instance_id BIGINT,
                    feature_name VARCHAR,
                    shap_value DOUBLE,
                    feature_value DOUBLE,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS explanations (
                    id BIGINT PRIMARY KEY,
                    model_name VARCHAR,
                    run_id VARCHAR,
                    instance_id BIGINT,
                    explanation_type VARCHAR,
                    explanation JSON,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id BIGINT PRIMARY KEY,
                    model_name VARCHAR,
                    metric_name VARCHAR,
                    metric_value DOUBLE,
                    run_id VARCHAR,
                    created_at TIMESTAMP
                )
                """
            )

            # Migrate databases created by the previous schema.
            conn.execute(
                "ALTER TABLE counterfactuals ADD COLUMN IF NOT EXISTS run_id VARCHAR"
            )
            conn.execute(
                "ALTER TABLE shap_values ADD COLUMN IF NOT EXISTS run_id VARCHAR"
            )
            conn.execute(
                "ALTER TABLE explanations ADD COLUMN IF NOT EXISTS run_id VARCHAR"
            )

        logger.info("Database initialized at %s", self.db_path)

    @staticmethod
    def _next_id_expression(table: str) -> str:
        # This project writes explanation batches serially. The expression keeps
        # compatibility with existing INTEGER PRIMARY KEY tables that have no
        # identity/default sequence.
        return f"(SELECT COALESCE(MAX(id), 0) FROM {table}) " "+ ROW_NUMBER() OVER ()"

    def store_counterfactuals(
        self,
        df: pl.DataFrame,
        model_name: str,
        run_id: str,
    ) -> None:
        if df.is_empty():
            logger.warning("No counterfactuals to store.")
            return

        with duckdb.connect(str(self.db_path)) as conn:
            conn.register("temp_df", df.to_pandas())
            conn.execute(
                f"""
                INSERT INTO counterfactuals (
                    id,
                    model_name,
                    run_id,
                    instance_id,
                    original_features,
                    counterfactual_features,
                    target_class,
                    original_class,
                    distance,
                    created_at
                )
                SELECT
                    {self._next_id_expression("counterfactuals")},
                    ?,
                    ?,
                    instance_id,
                    CAST(original_features AS JSON),
                    CAST(counterfactual_features AS JSON),
                    target_class,
                    original_class,
                    distance,
                    CURRENT_TIMESTAMP
                FROM temp_df
                """,
                [model_name, run_id],
            )

        logger.info(
            "Stored %d counterfactuals for %s (run: %s)",
            df.height,
            model_name,
            run_id,
        )

    def store_shap_values(
        self,
        df: pl.DataFrame,
        model_name: str,
        run_id: str,
    ) -> None:
        if df.is_empty():
            logger.warning("No SHAP values to store.")
            return

        with duckdb.connect(str(self.db_path)) as conn:
            conn.register("temp_df", df.to_pandas())
            conn.execute(
                f"""
                INSERT INTO shap_values (
                    id,
                    model_name,
                    run_id,
                    instance_id,
                    feature_name,
                    shap_value,
                    feature_value,
                    created_at
                )
                SELECT
                    {self._next_id_expression("shap_values")},
                    ?,
                    ?,
                    instance_id,
                    feature_name,
                    shap_value,
                    feature_value,
                    CURRENT_TIMESTAMP
                FROM temp_df
                """,
                [model_name, run_id],
            )

        logger.info(
            "Stored %d SHAP values for %s (run: %s)",
            df.height,
            model_name,
            run_id,
        )

    def store_metrics(
        self,
        model_name: str,
        metrics: dict[str, float],
        run_id: str,
    ) -> None:
        records = [
            {
                "metric_name": name,
                "metric_value": float(value),
            }
            for name, value in metrics.items()
            if isinstance(value, int | float)
        ]

        if not records:
            logger.warning("No metrics to store.")
            return

        df = pl.DataFrame(records)

        with duckdb.connect(str(self.db_path)) as conn:
            conn.register("temp_df", df.to_pandas())
            conn.execute(
                f"""
                INSERT INTO metrics (
                    id,
                    model_name,
                    metric_name,
                    metric_value,
                    run_id,
                    created_at
                )
                SELECT
                    {self._next_id_expression("metrics")},
                    ?,
                    metric_name,
                    metric_value,
                    ?,
                    CURRENT_TIMESTAMP
                FROM temp_df
                """,
                [model_name, run_id],
            )

        logger.info(
            "Stored %d metrics for %s (run: %s)",
            len(records),
            model_name,
            run_id,
        )

    def query_counterfactuals(
        self,
        model_name: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        clauses: list[str] = []
        params: list[object] = []

        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(int(limit), 0)

        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute(
                f"""
                SELECT *
                FROM counterfactuals
                {where}
                ORDER BY created_at DESC
                LIMIT {safe_limit}
                """,
                params,
            ).fetchdf()

        return pl.from_pandas(result)

    def get_metrics(
        self,
        model_name: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, float]:
        clauses: list[str] = []
        params: list[object] = []

        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute(
                f"""
                SELECT metric_name, metric_value
                FROM metrics
                {where}
                ORDER BY created_at DESC
                """,
                params,
            ).fetchdf()

        metrics: dict[str, float] = {}
        for row in result.itertuples(index=False):
            metrics.setdefault(str(row[0]), float(row[1]))
        return metrics

    def get_model_metrics(
        self,
        model_name: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        clauses: list[str] = []
        params: list[object] = []

        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(int(limit), 0)

        with duckdb.connect(str(self.db_path)) as conn:
            result = conn.execute(
                f"""
                SELECT
                    model_name,
                    metric_name,
                    metric_value,
                    run_id,
                    created_at
                FROM metrics
                {where}
                ORDER BY created_at DESC
                LIMIT {safe_limit}
                """,
                params,
            ).fetchdf()

        return pl.from_pandas(result)
