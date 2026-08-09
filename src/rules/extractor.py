"""Extract global counterfactual rules from cluster profiles."""

from __future__ import annotations

from dataclasses import dataclass

from src.clustering.profiler import ClusterProfile


@dataclass
class GlobalRule:
    """Interpretable rule distilled from one counterfactual cluster."""

    cluster_id: int
    conditions: list[str]
    support: int
    support_share: float
    avg_distance: float


class CounterfactualRuleExtractor:
    """Convert cluster profiles into concise global rules."""

    def __init__(
        self,
        min_change_rate: float = 0.80,
        min_transition_rate: float = 0.10,
        min_abs_delta: float = 0.05,
    ) -> None:
        self.min_change_rate = min_change_rate
        self.min_transition_rate = min_transition_rate
        self.min_abs_delta = min_abs_delta

    @staticmethod
    def _format_numeric_condition(
        feature: str,
        delta: float,
    ) -> str:
        direction = "increase" if delta > 0 else "decrease"

        return f"{feature} tends to {direction} " f"(avg normalized delta {delta:+.3f})"

    @staticmethod
    def _format_transition(
        transition: str,
    ) -> str:
        try:
            feature, values = transition.split(
                "__transition__",
                maxsplit=1,
            )

            source, destination = values.split(
                "__to__",
                maxsplit=1,
            )

            return (
                f"{feature}: "
                f"{source.replace('_', ' ')}"
                f" -> "
                f"{destination.replace('_', ' ')}"
            )

        except ValueError:
            return transition

    def extract_rule(
        self,
        profile: ClusterProfile,
    ) -> GlobalRule:
        """Extract one rule from a cluster profile."""
        conditions: list[str] = []

        for feature, change_rate in sorted(
            profile.change_rates.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            if change_rate < self.min_change_rate:
                continue

            delta = profile.avg_numeric_deltas.get(feature)

            if delta is not None and abs(delta) >= self.min_abs_delta:
                conditions.append(
                    self._format_numeric_condition(
                        feature,
                        delta,
                    )
                )
            else:
                conditions.append(
                    f"{feature} changes " f"({change_rate:.0%} of cluster)"
                )

        for transition, rate in profile.top_transitions:
            if rate < self.min_transition_rate:
                continue

            conditions.append(f"{self._format_transition(transition)} " f"({rate:.0%})")

        if not conditions:
            conditions.append("No dominant intervention pattern")

        return GlobalRule(
            cluster_id=profile.cluster_id,
            conditions=conditions,
            support=profile.size,
            support_share=profile.share,
            avg_distance=profile.avg_distance,
        )

    def extract_all(
        self,
        profiles: list[ClusterProfile],
    ) -> list[GlobalRule]:
        """Extract rules for every counterfactual cluster."""
        return [self.extract_rule(profile) for profile in profiles]
