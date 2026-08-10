"""Run the counterfactual aggregation and rule distillation pipeline."""

from __future__ import annotations

import argparse
import logging

from src.clustering.kmeans import CounterfactualKMeans
from src.clustering.profiler import CounterfactualClusterProfiler
from src.explainability.encoder import CounterfactualEncoder
from src.rules.evaluator import CounterfactualRuleEvaluator
from src.rules.extractor import CounterfactualRuleExtractor
from src.storage.duckdb import (
    DuckDBStorage,
    GlobalRuleRecord,
)

logger = logging.getLogger(__name__)


def aggregate_counterfactuals(
    run_id: str,
    model_name: str,
    n_clusters: int = 6,
) -> None:
    """Aggregate stored counterfactuals into global distilled rules."""
    storage = DuckDBStorage()

    counterfactuals = storage.query_counterfactuals(
        run_id=run_id,
        limit=100_000,
    )

    if counterfactuals.is_empty():
        raise ValueError(f"No counterfactuals found for run_id={run_id!r}.")

    logger.info(
        "Loaded %d counterfactuals for run %s",
        counterfactuals.height,
        run_id,
    )

    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(counterfactuals)

    matrix = encoder.clustering_matrix(encoded)

    logger.info(
        "Encoded counterfactual matrix shape: %s",
        matrix.shape,
    )

    clusterer = CounterfactualKMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    clustering_result = clusterer.fit(matrix)

    clustered = clusterer.attach_labels(
        encoded,
        clustering_result.labels,
    )

    logger.info(
        "Clustering complete: silhouette=%.4f inertia=%.4f",
        clustering_result.silhouette,
        clustering_result.inertia,
    )

    profiler = CounterfactualClusterProfiler()

    profiles = profiler.profile_all(
        clustered,
        top_n_transitions=10,
    )

    extractor = CounterfactualRuleExtractor(
        min_change_rate=0.80,
        min_transition_rate=0.10,
        min_abs_delta=0.05,
    )

    rules = extractor.extract_all(profiles)

    evaluator = CounterfactualRuleEvaluator()

    qualities = evaluator.evaluate_all(rules)

    quality_by_cluster = {quality.cluster_id: quality for quality in qualities}

    rule_records: list[GlobalRuleRecord] = []

    for rule in rules:
        quality = quality_by_cluster[rule.cluster_id]

        rule_records.append(
            {
                "cluster_id": rule.cluster_id,
                "conditions": rule.conditions,
                "support": rule.support,
                "support_share": rule.support_share,
                "avg_distance": rule.avg_distance,
                "quality_score": quality.quality_score,
            }
        )

    storage.clear_aggregation_results(
        model_name=model_name,
        run_id=run_id,
    )

    storage.store_counterfactual_clusters(
        clustered,
        model_name=model_name,
        run_id=run_id,
    )

    storage.store_global_rules(
        rule_records,
        model_name=model_name,
        run_id=run_id,
    )

    logger.info(
        "Aggregation complete: %d clusters, %d global rules",
        len(profiles),
        len(rules),
    )


def parse_args() -> argparse.Namespace:
    """Parse aggregation command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Aggregate CounterDistill counterfactuals into global rules.")
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
    )

    return parser.parse_args()


def main() -> None:
    """Run counterfactual aggregation."""
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    aggregate_counterfactuals(
        run_id=args.run_id,
        model_name=args.model_name,
        n_clusters=args.n_clusters,
    )


if __name__ == "__main__":
    main()
