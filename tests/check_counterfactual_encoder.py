from src.explainability.encoder import CounterfactualEncoder
from src.storage.duckdb import DuckDBStorage

RUN_ID = "local-20260808-220006"


def main() -> None:
    storage = DuckDBStorage()

    counterfactuals = storage.query_counterfactuals(
        run_id=RUN_ID,
        limit=1000,
    )

    print(f"Loaded {counterfactuals.height} counterfactuals")

    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(counterfactuals)

    print(f"Encoded shape: {encoded.shape}")

    print(f"Clustering features: {len(encoder.feature_names_)}")

    print("\nNumeric ranges:")

    for feature, feature_range in encoder.numeric_ranges_.items():
        print(f"  {feature}: {feature_range:.4f}")

    print("\nSample:")

    print(
        encoded.select(
            [
                "instance_id",
                "distance",
                "capital_gain__delta",
                "capital_loss__delta",
                "hours_per_week__delta",
                "workclass__changed",
                "education__changed",
                "occupation__changed",
            ]
        ).head(10)
    )

    matrix = encoder.clustering_matrix(encoded)

    print(f"\nClustering matrix shape: {matrix.shape}")

    print(f"Contains NaN: {bool((matrix != matrix).any())}")


if __name__ == "__main__":
    main()
