from src.clustering.kmeans import CounterfactualKMeans
from src.clustering.profiler import (
    CounterfactualClusterProfiler,
)
from src.explainability.encoder import CounterfactualEncoder
from src.rules.evaluator import CounterfactualRuleEvaluator
from src.rules.extractor import CounterfactualRuleExtractor
from src.storage.duckdb import DuckDBStorage

RUN_ID = "local-20260808-220006"


def main() -> None:
    storage = DuckDBStorage()

    counterfactuals = storage.query_counterfactuals(
        run_id=RUN_ID,
        limit=1000,
    )

    encoder = CounterfactualEncoder()
    encoded = encoder.fit_transform(counterfactuals)

    matrix = encoder.clustering_matrix(encoded)

    clusterer = CounterfactualKMeans(
        n_clusters=6,
        random_state=42,
        n_init=10,
    )

    result = clusterer.fit(matrix)

    clustered = clusterer.attach_labels(
        encoded,
        result.labels,
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

    print(
        f"{'cluster':<10}"
        f"{'support':<10}"
        f"{'coverage':<12}"
        f"{'conditions':<12}"
        f"{'compactness':<14}"
        f"{'distance':<12}"
        f"{'quality':<10}"
    )

    print("-" * 80)

    for quality in sorted(
        qualities,
        key=lambda item: item.quality_score,
        reverse=True,
    ):
        print(
            f"{quality.cluster_id:<10}"
            f"{quality.support:<10}"
            f"{quality.coverage:<12.3f}"
            f"{quality.condition_count:<12}"
            f"{quality.compactness:<14.3f}"
            f"{quality.avg_distance:<12.4f}"
            f"{quality.quality_score:<10.4f}"
        )

    print("\nPairwise rule similarity:")

    similarities = evaluator.pairwise_similarities(rules)

    for similarity in sorted(
        similarities,
        key=lambda item: item.similarity,
        reverse=True,
    ):
        print(
            f"Cluster "
            f"{similarity.cluster_a} vs "
            f"{similarity.cluster_b}: "
            f"{similarity.similarity:.3f}"
        )


if __name__ == "__main__":
    main()
