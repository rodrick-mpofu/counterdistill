import json
from typing import Any

import duckdb

DB_PATH = "database/counterdistill.db"


def parse_json(value: str | dict[str, Any]) -> dict[str, Any]:
    """Return a dictionary whether DuckDB gives JSON as text or dict-like data."""
    if isinstance(value, dict):
        return value

    parsed = json.loads(value)

    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")

    return parsed


def get_changed_features(
    original: dict,
    counterfactual: dict,
) -> dict[str, tuple[object, object]]:
    """Return only features whose values changed."""
    changes = {}

    for feature, original_value in original.items():
        counterfactual_value = counterfactual.get(feature)

        if original_value != counterfactual_value:
            changes[feature] = (
                original_value,
                counterfactual_value,
            )

    return changes


def print_counterfactuals(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 10,
) -> None:
    """Print counterfactuals in a readable format."""
    rows = conn.execute(
        """
        SELECT
            instance_id,
            original_class,
            target_class,
            distance,
            original_features,
            counterfactual_features
        FROM counterfactuals
        ORDER BY instance_id, distance
        LIMIT ?
        """,
        [limit],
    ).fetchall()

    print("\n" + "=" * 70)
    print("COUNTERFACTUAL EXPLANATIONS")
    print("=" * 70)

    if not rows:
        print("No counterfactuals found.")
        return

    for (
        instance_id,
        original_class,
        target_class,
        distance,
        original_features,
        counterfactual_features,
    ) in rows:
        original = parse_json(original_features)
        counterfactual = parse_json(counterfactual_features)

        changes = get_changed_features(
            original,
            counterfactual,
        )

        print(f"\nInstance: {instance_id}")
        print(f"Original class: {original_class}")
        print(f"Counterfactual class: {target_class}")
        print(f"Distance: {distance:.4f}")

        print("\nChanges:")

        if not changes:
            print("  No changed features found.")
        else:
            for feature, (
                original_value,
                counterfactual_value,
            ) in changes.items():
                print(f"  {feature}: " f"{original_value} -> {counterfactual_value}")

        print("-" * 70)


def print_global_shap(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 20,
) -> None:
    """Print the globally most important SHAP features."""
    rows = conn.execute(
        """
        SELECT
            feature_name,
            AVG(ABS(shap_value)) AS mean_abs_shap
        FROM shap_values
        GROUP BY feature_name
        ORDER BY mean_abs_shap DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()

    print("\n" + "=" * 70)
    print("GLOBAL SHAP IMPORTANCE")
    print("=" * 70)

    if not rows:
        print("No SHAP values found.")
        return

    for rank, (feature_name, mean_abs_shap) in enumerate(
        rows,
        start=1,
    ):
        print(f"{rank:>2}. " f"{feature_name:<45} " f"{mean_abs_shap:.6f}")


def print_instance_shap(
    conn: duckdb.DuckDBPyConnection,
    instance_id: int,
    limit: int = 15,
) -> None:
    """Print the most influential SHAP features for one instance."""
    rows = conn.execute(
        """
        SELECT
            feature_name,
            feature_value,
            shap_value
        FROM shap_values
        WHERE instance_id = ?
        ORDER BY ABS(shap_value) DESC
        LIMIT ?
        """,
        [instance_id, limit],
    ).fetchall()

    print("\n" + "=" * 70)
    print(f"SHAP VALUES FOR INSTANCE {instance_id}")
    print("=" * 70)

    if not rows:
        print(f"No SHAP values found for instance {instance_id}.")
        return

    for rank, (
        feature_name,
        feature_value,
        shap_value,
    ) in enumerate(rows, start=1):
        print(
            f"{rank:>2}. "
            f"{feature_name:<45} "
            f"value={feature_value!s:<12} "
            f"SHAP={shap_value:+.6f}"
        )


def get_example_instance_id(
    conn: duckdb.DuckDBPyConnection,
) -> int | None:
    """Get one explained instance ID to use for the detailed SHAP view."""
    result = conn.execute(
        """
        SELECT instance_id
        FROM shap_values
        ORDER BY instance_id
        LIMIT 1
        """
    ).fetchone()

    if result is None:
        return None

    return int(result[0])


def main() -> None:
    conn = duckdb.connect(DB_PATH)

    try:
        print_counterfactuals(
            conn,
            limit=10,
        )

        print_global_shap(
            conn,
            limit=20,
        )

        example_instance_id = get_example_instance_id(conn)

        if example_instance_id is not None:
            print_instance_shap(
                conn,
                instance_id=example_instance_id,
                limit=15,
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
