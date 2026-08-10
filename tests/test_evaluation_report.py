from __future__ import annotations

import pytest

from src.evaluation.report import (
    counterfactual_coverage_metrics,
    intervention_signature,
    rule_jaccard,
)


def test_intervention_signature_returns_changed_features() -> None:
    original = {
        "education": "HS-grad",
        "occupation": "Sales",
        "hours_per_week": 40,
    }

    counterfactual = {
        "education": "Bachelors",
        "occupation": "Sales",
        "hours_per_week": 45,
    }

    signature = intervention_signature(
        original,
        counterfactual,
    )

    assert signature == (
        "education",
        "hours_per_week",
    )


def test_counterfactual_coverage_metrics() -> None:
    counterfactuals = {
        "counterfactual_count": 399,
        "explained_instance_count": 96,
    }

    shap = {
        "explained_instance_count": 100,
    }

    metrics = counterfactual_coverage_metrics(
        counterfactuals,
        shap,
    )

    assert metrics["attempted_instances"] == 100
    assert metrics["covered_instances"] == 96

    assert metrics["instance_coverage"] == pytest.approx(0.96)

    assert metrics["mean_counterfactuals_per_covered_instance"] == pytest.approx(
        399 / 96
    )


def test_counterfactual_coverage_handles_zero_attempts() -> None:
    counterfactuals = {
        "counterfactual_count": 0,
        "explained_instance_count": 0,
    }

    shap = {
        "explained_instance_count": 0,
    }

    metrics = counterfactual_coverage_metrics(
        counterfactuals,
        shap,
    )

    assert metrics["instance_coverage"] is None

    assert metrics["mean_counterfactuals_per_covered_instance"] is None


def test_rule_jaccard_identical_rules() -> None:
    left = [
        "capital_gain tends to increase (avg normalized delta +0.50)",
        "occupation changes (100% of cluster)",
    ]

    right = [
        "capital_gain tends to increase (avg normalized delta +0.40)",
        "occupation changes (90% of cluster)",
    ]

    assert rule_jaccard(
        left,
        right,
    ) == pytest.approx(1.0)


def test_rule_jaccard_partially_overlapping_rules() -> None:
    left = [
        "capital_gain tends to increase (avg normalized delta +0.50)",
        "occupation changes (100% of cluster)",
    ]

    right = [
        "capital_gain tends to increase (avg normalized delta +0.40)",
        "education changes (100% of cluster)",
    ]

    assert rule_jaccard(
        left,
        right,
    ) == pytest.approx(1 / 3)


def test_rule_jaccard_disjoint_rules() -> None:
    left = [
        "occupation changes (100% of cluster)",
    ]

    right = [
        "education changes (100% of cluster)",
    ]

    assert rule_jaccard(
        left,
        right,
    ) == pytest.approx(0.0)
