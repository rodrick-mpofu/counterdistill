from src.clustering.kmeans import CounterfactualKMeans
from src.clustering.profiler import (
    CounterfactualClusterProfiler,
)
from src.explainability.encoder import CounterfactualEncoder
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

    for rule in rules:
        print("\n" + "=" * 60)
        print(f"Cluster {rule.cluster_id}")
        print("=" * 60)

        print(f"Support: {rule.support} " f"({rule.support_share:.2%})")

        print(f"Average distance: " f"{rule.avg_distance:.4f}")

        print("\nRule:")

        for index, condition in enumerate(rule.conditions):
            prefix = "IF" if index == 0 else "AND"
            print(f"  {prefix} {condition}")

        print("  THEN model prediction reaches " "the counterfactual target class")


if __name__ == "__main__":
    main()
