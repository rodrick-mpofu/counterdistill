from src.clustering.kmeans import CounterfactualKMeans
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
        n_clusters=5,
        random_state=42,
        n_init=10,
    )

    result = clusterer.fit(matrix)

    clustered = clusterer.attach_labels(
        encoded,
        result.labels,
    )

    sizes = clusterer.cluster_sizes(clustered)

    print(f"Matrix shape: {matrix.shape}")

    print(f"Silhouette score: " f"{result.silhouette:.4f}")

    print(f"Inertia: " f"{result.inertia:.4f}")

    print("\nCluster sizes:")

    print(sizes)

    print("\nSample assignments:")

    print(
        clustered.select(
            [
                "instance_id",
                "distance",
                "cluster_id",
            ]
        ).head(20)
    )


if __name__ == "__main__":
    main()
