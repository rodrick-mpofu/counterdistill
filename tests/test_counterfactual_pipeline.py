from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from src.clustering.kmeans import CounterfactualKMeans
from src.clustering.profiler import CounterfactualClusterProfiler
from src.explainability.encoder import CounterfactualEncoder
from src.rules.evaluator import CounterfactualRuleEvaluator
from src.rules.extractor import CounterfactualRuleExtractor


@pytest.fixture
def counterfactual_df() -> pl.DataFrame:
    """Create two clearly separated intervention patterns."""
    records = [
        # Pattern A: capital gain increases
        {
            "id": 1,
            "run_id": "test-run",
            "instance_id": 101,
            "model_name": "RandomForestClassifier",
            "distance": 0.05,
            "original_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
            "counterfactual_features": json.dumps(
                {
                    "capital_gain": 5000,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
        },
        {
            "id": 2,
            "run_id": "test-run",
            "instance_id": 102,
            "model_name": "RandomForestClassifier",
            "distance": 0.06,
            "original_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
            "counterfactual_features": json.dumps(
                {
                    "capital_gain": 6000,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
        },
        {
            "id": 3,
            "run_id": "test-run",
            "instance_id": 103,
            "model_name": "RandomForestClassifier",
            "distance": 0.07,
            "original_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
            "counterfactual_features": json.dumps(
                {
                    "capital_gain": 7000,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
        },
        # Pattern B: more hours + higher education
        {
            "id": 4,
            "run_id": "test-run",
            "instance_id": 104,
            "model_name": "RandomForestClassifier",
            "distance": 0.08,
            "original_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
            "counterfactual_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 50,
                    "education": "Bachelors",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
        },
        {
            "id": 5,
            "run_id": "test-run",
            "instance_id": 105,
            "model_name": "RandomForestClassifier",
            "distance": 0.09,
            "original_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
            "counterfactual_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 55,
                    "education": "Bachelors",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
        },
        {
            "id": 6,
            "run_id": "test-run",
            "instance_id": 106,
            "model_name": "RandomForestClassifier",
            "distance": 0.10,
            "original_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "education": "HS-grad",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
            "counterfactual_features": json.dumps(
                {
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 60,
                    "education": "Bachelors",
                    "workclass": "Private",
                    "occupation": "Sales",
                }
            ),
        },
    ]

    return pl.DataFrame(records)


def test_encoder_represents_intervention_changes(
    counterfactual_df: pl.DataFrame,
) -> None:
    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(counterfactual_df)

    assert encoded.height == 6
    assert encoder.is_fitted_

    assert "capital_gain__delta" in encoded.columns
    assert "capital_gain__changed" in encoded.columns
    assert "hours_per_week__changed" in encoded.columns
    assert "education__changed" in encoded.columns

    transition = "education__transition__HS-grad__to__Bachelors"

    assert transition in encoded.columns

    capital_gain_row = encoded.row(
        0,
        named=True,
    )

    assert capital_gain_row["capital_gain__changed"] == 1
    assert capital_gain_row["hours_per_week__changed"] == 0
    assert capital_gain_row["education__changed"] == 0

    education_row = encoded.row(
        3,
        named=True,
    )

    assert education_row["capital_gain__changed"] == 0
    assert education_row["hours_per_week__changed"] == 1
    assert education_row["education__changed"] == 1
    assert education_row[transition] == 1


def test_clustering_matrix_is_numeric_and_complete(
    counterfactual_df: pl.DataFrame,
) -> None:
    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(counterfactual_df)

    matrix = encoder.clustering_matrix(encoded)

    assert matrix.shape == (
        6,
        len(encoder.feature_names_),
    )

    assert np.issubdtype(
        matrix.dtype,
        np.number,
    )

    assert not np.isnan(matrix).any()


def test_kmeans_separates_intervention_patterns(
    counterfactual_df: pl.DataFrame,
) -> None:
    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(counterfactual_df)

    matrix = encoder.clustering_matrix(encoded)

    clustering = CounterfactualKMeans(
        n_clusters=2,
        random_state=42,
    )

    result = clustering.fit(matrix)

    assert len(result.labels) == 6
    assert len(np.unique(result.labels)) == 2
    assert result.centers.shape[0] == 2
    assert result.inertia >= 0.0
    assert result.silhouette > 0.0

    clustered = clustering.attach_labels(
        encoded,
        result.labels,
    )

    sizes = clustering.cluster_sizes(clustered)

    assert sorted(sizes["count"].to_list()) == [3, 3]

    assert sizes["share"].sum() == pytest.approx(1.0)


def test_pipeline_extracts_expected_global_patterns(
    counterfactual_df: pl.DataFrame,
) -> None:
    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(counterfactual_df)

    matrix = encoder.clustering_matrix(encoded)

    clustering = CounterfactualKMeans(
        n_clusters=2,
        random_state=42,
    )

    result = clustering.fit(matrix)

    clustered = clustering.attach_labels(
        encoded,
        result.labels,
    )

    profiler = CounterfactualClusterProfiler()

    profiles = profiler.profile_all(clustered)

    assert len(profiles) == 2

    assert sum(profile.share for profile in profiles) == pytest.approx(1.0)

    extractor = CounterfactualRuleExtractor()

    rules = extractor.extract_all(profiles)

    assert len(rules) == 2

    all_conditions = [condition for rule in rules for condition in rule.conditions]

    assert any(
        "capital_gain tends to increase" in condition for condition in all_conditions
    )

    assert any(
        "hours_per_week tends to increase" in condition for condition in all_conditions
    )

    assert any("education changes" in condition for condition in all_conditions)

    assert any(
        "education: HS-grad -> Bachelors" in condition for condition in all_conditions
    )


def test_rule_evaluation_produces_valid_quality_scores(
    counterfactual_df: pl.DataFrame,
) -> None:
    encoder = CounterfactualEncoder()
    encoded = encoder.fit_transform(counterfactual_df)

    matrix = encoder.clustering_matrix(encoded)

    clustering = CounterfactualKMeans(
        n_clusters=2,
        random_state=42,
    )

    result = clustering.fit(matrix)

    clustered = clustering.attach_labels(
        encoded,
        result.labels,
    )

    profiler = CounterfactualClusterProfiler()
    profiles = profiler.profile_all(clustered)

    extractor = CounterfactualRuleExtractor()
    rules = extractor.extract_all(profiles)

    evaluator = CounterfactualRuleEvaluator()
    qualities = evaluator.evaluate_all(rules)

    assert len(qualities) == 2

    for quality in qualities:
        assert 0.0 <= quality.quality_score <= 1.0
        assert 0.0 <= quality.coverage <= 1.0
        assert 0.0 <= quality.compactness <= 1.0
        assert quality.condition_count >= 1

    similarities = evaluator.pairwise_similarities(rules)

    assert len(similarities) == 1
    assert 0.0 <= similarities[0].similarity <= 1.0


def test_encoder_requires_fit_before_transform(
    counterfactual_df: pl.DataFrame,
) -> None:
    encoder = CounterfactualEncoder()

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        encoder.transform(counterfactual_df)


def test_kmeans_rejects_too_many_clusters() -> None:
    matrix = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    clustering = CounterfactualKMeans(n_clusters=3)

    with pytest.raises(
        ValueError,
        match="greater than or equal",
    ):
        clustering.fit(matrix)
