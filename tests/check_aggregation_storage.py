from src.clustering.kmeans import CounterfactualKMeans
from src.clustering.profiler import CounterfactualClusterProfiler
from src.explainability.encoder import CounterfactualEncoder
from src.rules.evaluator import CounterfactualRuleEvaluator
from src.rules.extractor import CounterfactualRuleExtractor
from src.storage.duckdb import (
    DuckDBStorage,
    GlobalRuleRecord,
)

RUN_ID = "local-20260808-220006"
MODEL_NAME = "RandomForestClassifier"


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

    storage.store_counterfactual_clusters(
        clustered,
        model_name=MODEL_NAME,
        run_id=RUN_ID,
    )

    storage.store_global_rules(
        rule_records,
        model_name=MODEL_NAME,
        run_id=RUN_ID,
    )

    print(f"Stored {clustered.height} cluster assignments")

    print(f"Stored {len(rule_records)} global rules")


if __name__ == "__main__":
    main()
