"""Generate the final CounterDistill evaluation report."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import duckdb
import mlflow
import polars as pl
from mlflow import MlflowClient
from sklearn.metrics import silhouette_score

from src.explainability.encoder import CounterfactualEncoder

logger = logging.getLogger(__name__)


def parse_json_object(
    value: str | dict[str, Any],
) -> dict[str, Any]:
    """Parse a stored JSON object."""
    if isinstance(value, dict):
        return value

    parsed = json.loads(value)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")

    return parsed


def parse_json_list(
    value: str | list[str],
) -> list[str]:
    """Parse a stored JSON list."""
    if isinstance(value, list):
        return [str(item) for item in value]

    parsed = json.loads(value)

    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")

    return [str(item) for item in parsed]


def resolve_explanation_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: str | None,
    model_name: str | None,
) -> tuple[str, str]:
    """Resolve the explanation run to evaluate."""
    if run_id is not None:
        if model_name is None:
            row = conn.execute(
                """
                SELECT model_name
                FROM counterfactuals
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [run_id],
            ).fetchone()

            if row is None:
                raise ValueError(f"No counterfactuals found for run_id={run_id!r}.")

            return run_id, str(row[0])

        return run_id, model_name

    params: list[object] = []
    model_filter = ""

    if model_name is not None:
        model_filter = "AND cf.model_name = ?"
        params.append(model_name)

    row = conn.execute(
        f"""
        SELECT
            cf.run_id,
            cf.model_name,
            MAX(cf.created_at) AS latest_created_at
        FROM counterfactuals AS cf
        WHERE cf.run_id IS NOT NULL
          {model_filter}
          AND EXISTS (
              SELECT 1
              FROM global_rules AS rules
              WHERE rules.run_id = cf.run_id
                AND rules.model_name = cf.model_name
          )
        GROUP BY
            cf.run_id,
            cf.model_name
        ORDER BY latest_created_at DESC
        LIMIT 1
        """,
        params,
    ).fetchone()

    if row is None:
        raise ValueError("No completed explanation run with global rules was found.")

    return str(row[0]), str(row[1])


def load_counterfactuals(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    model_name: str,
) -> pl.DataFrame:
    """Load counterfactuals and their cluster assignments."""
    result = conn.execute(
        """
        SELECT
            cf.id,
            cf.model_name,
            cf.run_id,
            cf.instance_id,
            cf.original_features,
            cf.counterfactual_features,
            cf.distance,
            clusters.cluster_id
        FROM counterfactuals AS cf
        LEFT JOIN counterfactual_clusters AS clusters
            ON cf.id = clusters.counterfactual_id
           AND cf.run_id = clusters.run_id
           AND cf.model_name = clusters.model_name
        WHERE cf.run_id = ?
          AND cf.model_name = ?
        ORDER BY cf.id
        """,
        [
            run_id,
            model_name,
        ],
    ).fetchdf()

    return pl.from_pandas(result)


def intervention_signature(
    original: dict[str, Any],
    counterfactual: dict[str, Any],
) -> tuple[str, ...]:
    """Return the names of features changed by one counterfactual."""
    features = sorted(set(original) | set(counterfactual))

    return tuple(
        feature
        for feature in features
        if original.get(feature) != counterfactual.get(feature)
    )


def counterfactual_metrics(
    counterfactuals: pl.DataFrame,
) -> dict[str, Any]:
    """Compute corpus-level counterfactual metrics."""
    if counterfactuals.is_empty():
        raise ValueError("Counterfactual corpus is empty.")

    signatures: set[tuple[str, ...]] = set()
    changed_feature_counts: list[int] = []

    for row in counterfactuals.iter_rows(named=True):
        original = parse_json_object(row["original_features"])

        counterfactual = parse_json_object(row["counterfactual_features"])

        signature = intervention_signature(
            original,
            counterfactual,
        )

        signatures.add(signature)
        changed_feature_counts.append(len(signature))

    distances = counterfactuals["distance"]

    mean_distance = cast(
        float,
        distances.mean(),
    )

    min_distance = cast(
        float,
        distances.min(),
    )

    max_distance = cast(
        float,
        distances.max(),
    )

    return {
        "counterfactual_count": counterfactuals.height,
        "explained_instance_count": (counterfactuals["instance_id"].n_unique()),
        "mean_distance": mean_distance,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "mean_changed_features": (
            sum(changed_feature_counts) / len(changed_feature_counts)
        ),
        "unique_intervention_signatures": len(signatures),
        "intervention_signature_diversity": (len(signatures) / counterfactuals.height),
    }


def clustering_metrics(
    counterfactuals: pl.DataFrame,
) -> dict[str, Any]:
    """Recompute clustering quality for stored assignments."""
    clustered = counterfactuals.filter(pl.col("cluster_id").is_not_null())

    if clustered.is_empty():
        return {
            "cluster_count": 0,
            "silhouette_score": None,
            "largest_cluster_share": None,
            "cluster_sizes": {},
        }

    labels = clustered["cluster_id"].cast(pl.Int64).to_numpy()

    encoder = CounterfactualEncoder()

    encoded = encoder.fit_transform(clustered)

    matrix = encoder.clustering_matrix(encoded)

    unique_labels = sorted({int(label) for label in labels})

    silhouette: float | None = None

    if 1 < len(unique_labels) < len(labels):
        silhouette = float(
            silhouette_score(
                matrix,
                labels,
            )
        )

    cluster_counts = clustered.group_by("cluster_id").len().sort("cluster_id")

    cluster_sizes = {
        str(int(row["cluster_id"])): int(row["len"])
        for row in cluster_counts.iter_rows(named=True)
    }

    largest_cluster = max(cluster_sizes.values())

    return {
        "cluster_count": len(cluster_sizes),
        "silhouette_score": silhouette,
        "largest_cluster_share": (largest_cluster / clustered.height),
        "cluster_sizes": cluster_sizes,
    }


def shap_metrics(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    model_name: str,
) -> dict[str, Any]:
    """Compute SHAP summary metrics."""
    summary = conn.execute(
        """
        SELECT
            COUNT(*) AS shap_value_count,
            COUNT(DISTINCT instance_id)
                AS explained_instance_count,
            COUNT(DISTINCT feature_name)
                AS feature_count
        FROM shap_values
        WHERE run_id = ?
          AND model_name = ?
        """,
        [
            run_id,
            model_name,
        ],
    ).fetchone()

    if summary is None:
        raise ValueError("Could not query SHAP summary.")

    top_features = conn.execute(
        """
        SELECT
            feature_name,
            AVG(ABS(shap_value)) AS mean_abs_shap
        FROM shap_values
        WHERE run_id = ?
          AND model_name = ?
        GROUP BY feature_name
        ORDER BY mean_abs_shap DESC
        LIMIT 10
        """,
        [
            run_id,
            model_name,
        ],
    ).fetchall()

    return {
        "shap_value_count": int(summary[0]),
        "explained_instance_count": int(summary[1]),
        "feature_count": int(summary[2]),
        "top_features": [
            {
                "feature": str(feature),
                "mean_abs_shap": float(value),
            }
            for feature, value in top_features
        ],
    }


def condition_tokens(
    conditions: list[str],
) -> set[str]:
    """Normalize conditions for rule similarity."""
    return {
        condition.split(
            "(",
            maxsplit=1,
        )[0]
        .strip()
        .lower()
        for condition in conditions
    }


def rule_jaccard(
    left: list[str],
    right: list[str],
) -> float:
    """Compute Jaccard similarity between two rule condition sets."""
    left_tokens = condition_tokens(left)
    right_tokens = condition_tokens(right)

    union = left_tokens | right_tokens

    if not union:
        return 0.0

    return len(left_tokens & right_tokens) / len(union)


def rule_metrics(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    model_name: str,
) -> dict[str, Any]:
    """Compute global-rule quality statistics."""
    rows = conn.execute(
        """
        SELECT
            cluster_id,
            conditions,
            support,
            support_share,
            avg_distance,
            quality_score
        FROM global_rules
        WHERE run_id = ?
          AND model_name = ?
        ORDER BY quality_score DESC
        """,
        [
            run_id,
            model_name,
        ],
    ).fetchall()

    if not rows:
        return {
            "rule_count": 0,
            "mean_quality_score": None,
            "best_quality_score": None,
            "mean_condition_count": None,
            "max_pairwise_jaccard": None,
            "mean_pairwise_jaccard": None,
            "rules": [],
        }

    rules: list[dict[str, Any]] = []
    condition_lists: list[list[str]] = []

    for (
        cluster_id,
        conditions,
        support,
        support_share,
        avg_distance,
        quality_score,
    ) in rows:
        parsed_conditions = parse_json_list(conditions)

        condition_lists.append(parsed_conditions)

        rules.append(
            {
                "cluster_id": int(cluster_id),
                "conditions": parsed_conditions,
                "support": int(support),
                "support_share": float(support_share),
                "avg_distance": float(avg_distance),
                "quality_score": float(quality_score),
            }
        )

    similarities = [
        rule_jaccard(
            left,
            right,
        )
        for left, right in combinations(
            condition_lists,
            2,
        )
    ]

    quality_scores = [rule["quality_score"] for rule in rules]

    condition_counts = [len(rule["conditions"]) for rule in rules]

    return {
        "rule_count": len(rules),
        "mean_quality_score": (sum(quality_scores) / len(quality_scores)),
        "best_quality_score": max(quality_scores),
        "mean_condition_count": (sum(condition_counts) / len(condition_counts)),
        "max_pairwise_jaccard": (max(similarities) if similarities else 0.0),
        "mean_pairwise_jaccard": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        "rules": rules,
    }


def latest_model_metrics(
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
) -> dict[str, Any]:
    """Return metrics from the most relevant recent MLflow model run."""
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"MLflow experiment {experiment_name!r} was not found.")

    filters = [
        (
            "tags.project = 'counterdistill' "
            f"and tags.model_class = '{model_name}' "
            "and tags.stage = 'hyperparameter_tuning'"
        ),
        (
            "tags.project = 'counterdistill' "
            f"and tags.model_class = '{model_name}' "
            "and tags.stage = 'training'"
        ),
    ]

    selected_run = None

    for filter_string in filters:
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=filter_string,
            max_results=20,
        )

        for run in runs:
            metrics = run.data.metrics

            if "test_accuracy" in metrics or "accuracy" in metrics:
                selected_run = run
                break

        if selected_run is not None:
            break

    if selected_run is None:
        raise ValueError("No completed model run with evaluation metrics was found.")

    metrics = selected_run.data.metrics
    params = selected_run.data.params
    tags = selected_run.data.tags

    accuracy = metrics.get(
        "test_accuracy",
        metrics.get("accuracy"),
    )

    f1 = metrics.get(
        "test_f1",
        metrics.get("weighted avg_f1-score"),
    )

    roc_auc = metrics.get("test_roc_auc")

    feature_count = metrics.get("feature_count")

    if feature_count is None:
        raw_feature_count = params.get("feature_count")

        if raw_feature_count is not None:
            feature_count = float(raw_feature_count)

    return {
        "mlflow_run_id": (selected_run.info.run_id),
        "stage": tags.get("stage"),
        "model_name": model_name,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "feature_count": (int(feature_count) if feature_count is not None else None),
        "train_rows": params.get("train_rows"),
        "test_rows": params.get("test_rows"),
    }


def format_number(
    value: Any,
    digits: int = 4,
) -> str:
    """Format optional report values."""
    if value is None:
        return "N/A"

    if isinstance(
        value,
        float,
    ):
        return f"{value:.{digits}f}"

    return str(value)


def markdown_report(
    report: dict[str, Any],
) -> str:
    """Render report data as Markdown."""
    model = report["model"]
    counterfactuals = report["counterfactuals"]
    coverage = report["counterfactual_coverage"]
    shap = report["shap"]
    clustering = report["clustering"]
    rules = report["rules"]

    lines = [
        "# CounterDistill Final Evaluation",
        "",
        "## Run provenance",
        "",
        (f"- Explanation run: `{report['explanation_run_id']}`"),
        (f"- Model: `{report['model_name']}`"),
        (f"- MLflow run: `{model['mlflow_run_id']}`"),
        "",
        "## Model performance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (f"| Accuracy | {format_number(model['accuracy'])} |"),
        (f"| F1 | {format_number(model['f1'])} |"),
        (f"| ROC AUC | {format_number(model['roc_auc'])} |"),
        (f"| Model features | {format_number(model['feature_count'], 0)} |"),
        "",
        "## Explainability",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (f"| Counterfactuals | {counterfactuals['counterfactual_count']} |"),
        (
            "| Counterfactual instances | "
            f"{counterfactuals['explained_instance_count']} |"
        ),
        (
            "| Mean counterfactual distance | "
            f"{format_number(counterfactuals['mean_distance'])} |"
        ),
        (
            "| Counterfactual instance coverage | "
            f"{format_number(coverage['instance_coverage'])} |"
        ),
        (
            "| Mean CFs per covered instance | "
            + format_number(
                coverage["mean_counterfactuals_per_covered_instance"],
                2,
            )
            + " |"
        ),
        (
            "| Mean changed features | "
            f"{format_number(counterfactuals['mean_changed_features'], 2)} |"
        ),
        (
            "| Unique intervention signatures | "
            f"{counterfactuals['unique_intervention_signatures']} |"
        ),
        (
            "| Intervention signature diversity | "
            f"{format_number(counterfactuals['intervention_signature_diversity'])} |"
        ),
        (f"| SHAP values | {shap['shap_value_count']} |"),
        (f"| SHAP instances | {shap['explained_instance_count']} |"),
        (f"| SHAP features | {shap['feature_count']} |"),
        "",
        "## Distillation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (f"| Clusters | {clustering['cluster_count']} |"),
        (f"| Silhouette score | {format_number(clustering['silhouette_score'])} |"),
        (
            "| Largest cluster share | "
            f"{format_number(clustering['largest_cluster_share'])} |"
        ),
        (f"| Global rules | {rules['rule_count']} |"),
        (f"| Mean rule quality | {format_number(rules['mean_quality_score'])} |"),
        (f"| Best rule quality | {format_number(rules['best_quality_score'])} |"),
        (
            "| Mean conditions per rule | "
            f"{format_number(rules['mean_condition_count'], 2)} |"
        ),
        (
            "| Mean rule Jaccard similarity | "
            f"{format_number(rules['mean_pairwise_jaccard'])} |"
        ),
        (
            "| Max rule Jaccard similarity | "
            f"{format_number(rules['max_pairwise_jaccard'])} |"
        ),
        "",
        "## Top SHAP features",
        "",
        "| Rank | Feature | Mean absolute SHAP |",
        "| ---: | --- | ---: |",
    ]

    for index, feature in enumerate(
        shap["top_features"],
        start=1,
    ):
        lines.append(
            f"| {index} | {feature['feature']} | {feature['mean_abs_shap']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Distilled rules",
            "",
        ]
    )

    for rule in rules["rules"]:
        lines.append(f"### Cluster {rule['cluster_id']}")
        lines.append("")
        lines.append(f"- Support: {rule['support']} ({rule['support_share']:.2%})")
        lines.append(f"- Quality: {rule['quality_score']:.4f}")
        lines.append(f"- Average distance: {rule['avg_distance']:.4f}")
        lines.append("- Conditions:")

        for condition in rule["conditions"]:
            lines.append(f"  - {condition}")

        lines.append("")

    lines.extend(
        [
            "## Interpretation note",
            "",
            (
                "Counterfactuals and distilled rules describe "
                "model behavior under hypothetical feature changes. "
                "They are not causal or prescriptive claims."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def generate_report(
    db_path: str,
    tracking_uri: str,
    experiment_name: str,
    output_dir: str,
    run_id: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Generate JSON and Markdown evaluation reports."""
    database = Path(db_path)

    if not database.exists():
        raise FileNotFoundError(f"CounterDistill database not found: {database}")

    with duckdb.connect(
        str(database),
        read_only=True,
    ) as conn:
        resolved_run_id, resolved_model_name = resolve_explanation_run(
            conn,
            run_id=run_id,
            model_name=model_name,
        )

        logger.info(
            "Evaluating explanation run %s (%s)",
            resolved_run_id,
            resolved_model_name,
        )

        counterfactuals = load_counterfactuals(
            conn,
            run_id=resolved_run_id,
            model_name=resolved_model_name,
        )

        # ---------------------------------------------
        # Explanation metrics
        # ---------------------------------------------

        cf_summary = counterfactual_metrics(counterfactuals)

        shap_summary = shap_metrics(
            conn,
            run_id=resolved_run_id,
            model_name=resolved_model_name,
        )

        coverage_summary = counterfactual_coverage_metrics(
            cf_summary,
            shap_summary,
        )

        clustering_summary = clustering_metrics(counterfactuals)

        rule_summary = rule_metrics(
            conn,
            run_id=resolved_run_id,
            model_name=resolved_model_name,
        )

        # ---------------------------------------------
        # Model metrics
        # ---------------------------------------------

        model_summary = latest_model_metrics(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            model_name=resolved_model_name,
        )

        # ---------------------------------------------
        # Final report
        # ---------------------------------------------

        report = {
            "explanation_run_id": resolved_run_id,
            "model_name": resolved_model_name,
            "model": model_summary,
            "counterfactuals": cf_summary,
            "counterfactual_coverage": coverage_summary,
            "shap": shap_summary,
            "clustering": clustering_summary,
            "rules": rule_summary,
        }

    # ---------------------------------------------
    # Write report artifacts
    # ---------------------------------------------

    destination = Path(output_dir)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = destination / "final_report.json"

    markdown_path = destination / "final_report.md"

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_path.write_text(
        markdown_report(report),
        encoding="utf-8",
    )

    logger.info(
        "Wrote %s",
        json_path,
    )

    logger.info(
        "Wrote %s",
        markdown_path,
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Generate CounterDistill's final evaluation report.")
    )

    parser.add_argument(
        "--db-path",
        default="database/counterdistill.db",
    )

    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///database/mlflow.db",
    )

    parser.add_argument(
        "--experiment-name",
        default="counterdistill_exp",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--model-name",
        default=None,
    )

    return parser.parse_args()


def counterfactual_coverage_metrics(
    counterfactuals: dict[str, Any],
    shap: dict[str, Any],
) -> dict[str, float | int | None]:
    """Measure counterfactual coverage over the shared explanation sample."""
    attempted = int(shap["explained_instance_count"])

    covered = int(counterfactuals["explained_instance_count"])

    counterfactual_count = int(counterfactuals["counterfactual_count"])

    coverage = None if attempted == 0 else covered / attempted

    mean_per_covered = None if covered == 0 else counterfactual_count / covered

    return {
        "attempted_instances": attempted,
        "covered_instances": covered,
        "instance_coverage": coverage,
        "mean_counterfactuals_per_covered_instance": (mean_per_covered),
    }


def main() -> None:
    """Run final evaluation reporting."""
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    report = generate_report(
        db_path=args.db_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
    )

    print(markdown_report(report))


if __name__ == "__main__":
    main()
