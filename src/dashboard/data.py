"""Data access helpers for the CounterDistill dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


class DashboardData:
    """Read CounterDistill artifacts from DuckDB."""

    def __init__(
        self,
        db_path: str = "database/counterdistill.db",
    ) -> None:
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"CounterDistill database not found: {self.db_path}"
            )

    def _query(
        self,
        query: str,
        params: list[object] | None = None,
    ) -> pl.DataFrame:
        """Execute a read-only dashboard query."""
        with duckdb.connect(
            str(self.db_path),
            read_only=True,
        ) as conn:
            result = conn.execute(
                query,
                params or [],
            ).fetch_arrow_table()

        return pl.from_arrow(result)

    def available_runs(self) -> pl.DataFrame:
        """Return explanation runs ordered from newest to oldest."""
        return self._query(
            """
            SELECT
                run_id,
                model_name,
                COUNT(*) AS counterfactual_count,
                MAX(created_at) AS created_at
            FROM counterfactuals
            WHERE run_id IS NOT NULL
            GROUP BY run_id, model_name
            ORDER BY created_at DESC
            """
        )

    def counterfactuals(
        self,
        run_id: str,
    ) -> pl.DataFrame:
        """Return counterfactual records with cluster assignments."""
        return self._query(
            """
            SELECT
                cf.id,
                cf.instance_id,
                cf.model_name,
                cf.run_id,
                cf.original_features,
                cf.counterfactual_features,
                cf.target_class,
                cf.original_class,
                cf.distance,
                clusters.cluster_id,
                cf.created_at
            FROM counterfactuals AS cf
            LEFT JOIN counterfactual_clusters AS clusters
                ON cf.id = clusters.counterfactual_id
               AND cf.run_id = clusters.run_id
            WHERE cf.run_id = ?
            ORDER BY cf.instance_id, cf.distance
            """,
            [run_id],
        )

    def global_rules(
        self,
        run_id: str,
    ) -> pl.DataFrame:
        """Return distilled global rules for a run."""
        return self._query(
            """
            SELECT
                cluster_id,
                conditions,
                support,
                support_share,
                avg_distance,
                quality_score,
                created_at
            FROM global_rules
            WHERE run_id = ?
            ORDER BY quality_score DESC
            """,
            [run_id],
        )

    def shap_importance(
        self,
        run_id: str,
        limit: int = 20,
    ) -> pl.DataFrame:
        """Return mean absolute SHAP importance for a run."""
        safe_limit = max(
            int(limit),
            1,
        )

        return self._query(
            f"""
            SELECT
                feature_name,
                AVG(ABS(shap_value)) AS mean_abs_shap
            FROM shap_values
            WHERE run_id = ?
            GROUP BY feature_name
            ORDER BY mean_abs_shap DESC
            LIMIT {safe_limit}
            """,
            [run_id],
        )

    def shap_for_instance(
        self,
        run_id: str,
        instance_id: int,
    ) -> pl.DataFrame:
        """Return SHAP values for one explained instance."""
        return self._query(
            """
            SELECT
                feature_name,
                feature_value,
                shap_value
            FROM shap_values
            WHERE run_id = ?
              AND instance_id = ?
            ORDER BY ABS(shap_value) DESC
            """,
            [
                run_id,
                instance_id,
            ],
        )

    @staticmethod
    def parse_json_object(
        value: str | dict[str, Any],
    ) -> dict[str, Any]:
        """Parse a stored DuckDB JSON object."""
        if isinstance(value, dict):
            return value

        parsed = json.loads(value)

        if not isinstance(parsed, dict):
            raise ValueError("Expected stored JSON object.")

        return parsed

    def run_summary(
        self,
        run_id: str,
    ) -> dict[str, int]:
        """Return high-level artifact counts for a run."""
        result = self._query(
            """
            SELECT
                (
                    SELECT COUNT(*)
                    FROM counterfactuals
                    WHERE run_id = ?
                ) AS counterfactual_count,

                (
                    SELECT COUNT(*)
                    FROM shap_values
                    WHERE run_id = ?
                ) AS shap_value_count,

                (
                    SELECT COUNT(DISTINCT instance_id)
                    FROM shap_values
                    WHERE run_id = ?
                ) AS explained_instance_count,

                (
                    SELECT COUNT(DISTINCT cluster_id)
                    FROM counterfactual_clusters
                    WHERE run_id = ?
                ) AS cluster_count,

                (
                    SELECT COUNT(*)
                    FROM global_rules
                    WHERE run_id = ?
                ) AS rule_count
            """,
            [
                run_id,
                run_id,
                run_id,
                run_id,
                run_id,
            ],
        )

        row = result.row(
            0,
            named=True,
        )

        return {
            "counterfactual_count": int(row["counterfactual_count"]),
            "shap_value_count": int(row["shap_value_count"]),
            "explained_instance_count": int(row["explained_instance_count"]),
            "cluster_count": int(row["cluster_count"]),
            "rule_count": int(row["rule_count"]),
        }

    def counterfactual_changes(
        self,
        counterfactual_id: int,
        run_id: str,
    ) -> pl.DataFrame:
        """Return changed features for one counterfactual."""
        result = self._query(
            """
            SELECT
                original_features,
                counterfactual_features
            FROM counterfactuals
            WHERE id = ?
            AND run_id = ?
            LIMIT 1
            """,
            [
                counterfactual_id,
                run_id,
            ],
        )

        if result.is_empty():
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "original": pl.String,
                    "counterfactual": pl.String,
                }
            )

        row = result.row(
            0,
            named=True,
        )

        original = self.parse_json_object(row["original_features"])

        counterfactual = self.parse_json_object(row["counterfactual_features"])

        changes: list[dict[str, str]] = []

        all_features = sorted(set(original) | set(counterfactual))

        for feature in all_features:
            original_value = original.get(feature)
            counterfactual_value = counterfactual.get(feature)

            if original_value == counterfactual_value:
                continue

            changes.append(
                {
                    "feature": feature,
                    "original": str(original_value),
                    "counterfactual": str(counterfactual_value),
                }
            )

        if not changes:
            return pl.DataFrame(
                schema={
                    "feature": pl.String,
                    "original": pl.String,
                    "counterfactual": pl.String,
                }
            )

        return pl.DataFrame(changes)

    def parse_json_list(
        self,
        value: str | list[str],
    ) -> list[str]:
        """Parse a stored DuckDB JSON list."""
        if isinstance(value, list):
            return [str(item) for item in value]

        parsed = json.loads(value)

        if not isinstance(parsed, list):
            raise ValueError("Expected stored JSON list.")

        return [str(item) for item in parsed]

    def shap_instances(
        self,
        run_id: str,
    ) -> list[int]:
        """Return instance IDs with stored SHAP explanations."""
        result = self._query(
            """
            SELECT DISTINCT instance_id
            FROM shap_values
            WHERE run_id = ?
            ORDER BY instance_id
            """,
            [run_id],
        )

        if result.is_empty():
            return []

        return [int(value) for value in result["instance_id"].to_list()]
