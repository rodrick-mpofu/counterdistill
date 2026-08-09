from src.clustering.kmeans import CounterfactualKMeans
from src.explainability.encoder import CounterfactualEncoder
from src.storage.duckdb import DuckDBStorage

RUN_ID = "local-20260808-184609"


def main() -> None:
    storage = DuckDBStorage()

    counterfactuals = storage.query_counterfactuals(
        run_id=RUN_ID,
        limit=1000,
    )

    encoder = CounterfactualEncoder()
    encoded = encoder.fit_transform(counterfactuals)
    matrix = encoder.clustering_matrix(encoded)

    print(
        f"{'k':<5}"
        f"{'silhouette':<15}"
        f"{'inertia':<15}"
        f"{'min_cluster':<15}"
        f"{'max_cluster':<15}"
    )

    print("-" * 65)

    for k in range(2, 11):
        clusterer = CounterfactualKMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        result = clusterer.fit(matrix)

        clustered = clusterer.attach_labels(
            encoded,
            result.labels,
        )

        sizes = clusterer.cluster_sizes(clustered)

        counts = sizes["count"].to_list()

        print(
            f"{k:<5}"
            f"{result.silhouette:<15.4f}"
            f"{result.inertia:<15.4f}"
            f"{min(counts):<15}"
            f"{max(counts):<15}"
        )


if __name__ == "__main__":
    main()
