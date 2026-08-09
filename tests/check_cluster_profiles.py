from src.clustering.kmeans import CounterfactualKMeans
from src.clustering.profiler import (
    CounterfactualClusterProfiler,
)
from src.explainability.encoder import CounterfactualEncoder
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
        top_n_transitions=5,
    )

    for profile in profiles:
        print("\n" + "=" * 60)

        print(f"Cluster {profile.cluster_id}")

        print("=" * 60)

        print(f"Size: {profile.size}")

        print(f"Share: {profile.share:.2%}")

        print(f"Average distance: " f"{profile.avg_distance:.4f}")

        print("\nChange rates:")

        for feature, rate in sorted(
            profile.change_rates.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {feature:<20} " f"{rate:.2%}")

        print("\nAverage normalized deltas:")

        for feature, delta in sorted(
            profile.avg_numeric_deltas.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        ):
            print(f"  {feature:<20} " f"{delta:+.4f}")

        print("\nTop transitions:")

        for transition, rate in profile.top_transitions:
            readable = profiler.format_transition(transition)

            print(f"  {readable:<50} " f"{rate:.2%}")


if __name__ == "__main__":
    main()
