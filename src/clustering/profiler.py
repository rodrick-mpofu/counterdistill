"""Profile counterfactual clusters for interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl


@dataclass
class ClusterProfile:
    """Summary statistics for one counterfactual cluster."""

    cluster_id: int
    size: int
    share: float
    avg_distance: float
    change_rates: dict[str, float]
    avg_numeric_deltas: dict[str, float]
    top_transitions: list[tuple[str, float]]


class CounterfactualClusterProfiler:
    """Summarize the dominant intervention patterns in each cluster."""

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

    def _change_columns(
        self,
        df: pl.DataFrame,
    ) -> list[str]:
        """Return available changed-indicator columns."""
        features = self.numeric_features + self.categorical_features

        return [
            f"{feature}__changed"
            for feature in features
            if f"{feature}__changed" in df.columns
        ]

    def _delta_columns(
        self,
        df: pl.DataFrame,
    ) -> list[str]:
        """Return available normalized numeric delta columns."""
        return [
            f"{feature}__delta"
            for feature in self.numeric_features
            if f"{feature}__delta" in df.columns
        ]

    @staticmethod
    def _transition_columns(
        df: pl.DataFrame,
    ) -> list[str]:
        """Return categorical transition indicator columns."""
        return [column for column in df.columns if "__transition__" in column]

    def profile_cluster(
        self,
        clustered_df: pl.DataFrame,
        cluster_id: int,
        top_n_transitions: int = 10,
    ) -> ClusterProfile:
        """Create an interpretable summary for one cluster."""
        if "cluster_id" not in clustered_df.columns:
            raise ValueError("clustered_df must contain cluster_id.")

        cluster_df = clustered_df.filter(pl.col("cluster_id") == cluster_id)

        if cluster_df.is_empty():
            raise ValueError(f"Cluster {cluster_id} contains no rows.")

        total_rows = clustered_df.height
        size = cluster_df.height

        if "distance" in cluster_df.columns:
            avg_distance_value = cast(
                float | None,
                cluster_df["distance"].mean(),
            )
            avg_distance = (
                avg_distance_value if avg_distance_value is not None else float("nan")
            )
        else:
            avg_distance = float("nan")

        change_rates: dict[str, float] = {}

        for column in self._change_columns(cluster_df):
            rate_value = cast(
                float | None,
                cluster_df[column].mean(),
            )

            rate = rate_value if rate_value is not None else 0.0

            feature = column.removesuffix("__changed")

            change_rates[feature] = rate

        avg_numeric_deltas: dict[str, float] = {}

        for column in self._delta_columns(cluster_df):
            value_result = cast(
                float | None,
                cluster_df[column].mean(),
            )

            value = value_result if value_result is not None else 0.0

            feature = column.removesuffix("__delta")

            avg_numeric_deltas[feature] = value

        transitions: list[tuple[str, float]] = []

        for column in self._transition_columns(cluster_df):
            rate_value = cast(
                float | None,
                cluster_df[column].mean(),
            )

            rate = rate_value if rate_value is not None else 0.0

            if rate > 0:
                transitions.append((column, rate))

        transitions.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return ClusterProfile(
            cluster_id=cluster_id,
            size=size,
            share=size / total_rows,
            avg_distance=avg_distance,
            change_rates=change_rates,
            avg_numeric_deltas=avg_numeric_deltas,
            top_transitions=transitions[:top_n_transitions],
        )

    def profile_all(
        self,
        clustered_df: pl.DataFrame,
        top_n_transitions: int = 10,
    ) -> list[ClusterProfile]:
        """Profile every cluster in the clustered DataFrame."""
        cluster_ids = (
            clustered_df.select("cluster_id")
            .unique()
            .sort("cluster_id")["cluster_id"]
            .to_list()
        )

        return [
            self.profile_cluster(
                clustered_df,
                int(cluster_id),
                top_n_transitions=top_n_transitions,
            )
            for cluster_id in cluster_ids
        ]

    @staticmethod
    def format_transition(
        transition_column: str,
    ) -> str:
        """Convert an encoded transition column into readable text."""
        try:
            feature, values = transition_column.split(
                "__transition__",
                maxsplit=1,
            )

            source, destination = values.split(
                "__to__",
                maxsplit=1,
            )

            return (
                f"{feature}: "
                f"{source.replace('_', ' ')}"
                f" -> "
                f"{destination.replace('_', ' ')}"
            )

        except ValueError:
            return transition_column
