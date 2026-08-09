"""K-Means clustering for encoded counterfactual explanations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Container for counterfactual clustering outputs."""

    labels: np.ndarray
    centers: np.ndarray
    silhouette: float
    inertia: float


class CounterfactualKMeans:
    """Cluster encoded counterfactual intervention vectors."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )

    def fit(
        self,
        matrix: np.ndarray,
    ) -> ClusteringResult:
        """Fit K-Means and compute unsupervised evaluation metrics."""
        matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )

        if matrix.ndim != 2:
            raise ValueError("Clustering matrix must be two-dimensional.")

        if len(matrix) < self.n_clusters:
            raise ValueError(
                "Number of samples must be greater than or equal to n_clusters."
            )

        labels = self.model.fit_predict(matrix)

        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            silhouette = float("nan")
        else:
            silhouette = float(
                silhouette_score(
                    matrix,
                    labels,
                    metric="euclidean",
                )
            )

        logger.info(
            "K-Means fitted: clusters=%d silhouette=%.4f inertia=%.4f",
            len(unique_labels),
            silhouette,
            self.model.inertia_,
        )

        return ClusteringResult(
            labels=labels,
            centers=self.model.cluster_centers_,
            silhouette=silhouette,
            inertia=float(self.model.inertia_),
        )

    @staticmethod
    def attach_labels(
        encoded_df: pl.DataFrame,
        labels: np.ndarray,
    ) -> pl.DataFrame:
        """Attach cluster assignments to encoded counterfactual rows."""
        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        if encoded_df.height != len(labels):
            raise ValueError("Number of cluster labels must match encoded rows.")

        return encoded_df.with_columns(
            pl.Series(
                name="cluster_id",
                values=labels,
            )
        )

    @staticmethod
    def cluster_sizes(
        clustered_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Return counterfactual count and share for each cluster."""
        if "cluster_id" not in clustered_df.columns:
            raise ValueError("clustered_df must contain cluster_id.")

        total = clustered_df.height

        return (
            clustered_df.group_by("cluster_id")
            .agg(pl.len().alias("count"))
            .with_columns((pl.col("count") / total).alias("share"))
            .sort("cluster_id")
        )
