"""Evaluate quality of distilled counterfactual rules."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from src.rules.extractor import GlobalRule


@dataclass
class RuleQuality:
    """Quality metrics for one distilled counterfactual rule."""

    cluster_id: int
    support: int
    coverage: float
    compactness: float
    condition_count: int
    avg_distance: float
    quality_score: float


@dataclass
class RuleSimilarity:
    """Pairwise similarity between two distilled rules."""

    cluster_a: int
    cluster_b: int
    similarity: float


class CounterfactualRuleEvaluator:
    """Evaluate interpretability and redundancy of global rules."""

    def __init__(
        self,
        distance_weight: float = 0.30,
        coverage_weight: float = 0.40,
        compactness_weight: float = 0.30,
    ) -> None:
        total_weight = distance_weight + coverage_weight + compactness_weight

        if not 0.99 <= total_weight <= 1.01:
            raise ValueError("Rule quality weights must sum to 1.0.")

        self.distance_weight = distance_weight
        self.coverage_weight = coverage_weight
        self.compactness_weight = compactness_weight

    @staticmethod
    def _compactness(
        condition_count: int,
    ) -> float:
        """Score shorter rules more highly."""
        if condition_count <= 0:
            return 0.0

        return 1.0 / condition_count

    @staticmethod
    def _distance_score(
        avg_distance: float,
    ) -> float:
        """Convert counterfactual distance into a 0-1 quality score."""
        return max(
            0.0,
            min(
                1.0,
                1.0 - avg_distance,
            ),
        )

    def evaluate_rule(
        self,
        rule: GlobalRule,
    ) -> RuleQuality:
        """Evaluate one distilled rule."""
        condition_count = len(rule.conditions)

        compactness = self._compactness(condition_count)

        distance_score = self._distance_score(rule.avg_distance)

        quality_score = (
            self.coverage_weight * rule.support_share
            + self.compactness_weight * compactness
            + self.distance_weight * distance_score
        )

        return RuleQuality(
            cluster_id=rule.cluster_id,
            support=rule.support,
            coverage=rule.support_share,
            compactness=compactness,
            condition_count=condition_count,
            avg_distance=rule.avg_distance,
            quality_score=quality_score,
        )

    def evaluate_all(
        self,
        rules: list[GlobalRule],
    ) -> list[RuleQuality]:
        """Evaluate all rules."""
        return [self.evaluate_rule(rule) for rule in rules]

    @staticmethod
    def _condition_tokens(
        rule: GlobalRule,
    ) -> set[str]:
        """Normalize rule conditions for similarity comparison."""
        return {
            condition.split("(", maxsplit=1)[0].strip().lower()
            for condition in rule.conditions
        }

    def rule_similarity(
        self,
        left: GlobalRule,
        right: GlobalRule,
    ) -> float:
        """Compute Jaccard similarity between rule conditions."""
        left_conditions = self._condition_tokens(left)
        right_conditions = self._condition_tokens(right)

        union = left_conditions | right_conditions

        if not union:
            return 0.0

        intersection = left_conditions & right_conditions

        return len(intersection) / len(union)

    def pairwise_similarities(
        self,
        rules: list[GlobalRule],
    ) -> list[RuleSimilarity]:
        """Compare every pair of distilled rules."""
        similarities: list[RuleSimilarity] = []

        for left, right in combinations(
            rules,
            2,
        ):
            similarity = self.rule_similarity(
                left,
                right,
            )

            similarities.append(
                RuleSimilarity(
                    cluster_a=left.cluster_id,
                    cluster_b=right.cluster_id,
                    similarity=similarity,
                )
            )

        return similarities
