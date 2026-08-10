from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from src.dashboard.data import DashboardData

RUN_ID = "test-run-001"
MODEL_NAME = "RandomForestClassifier"


@pytest.fixture
def dashboard_data(
    tmp_path: Path,
) -> DashboardData:
    """Create a temporary populated CounterDistill database."""
    db_path = tmp_path / "counterdistill.db"

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE counterfactuals (
                id BIGINT PRIMARY KEY,
                model_name VARCHAR,
                run_id VARCHAR,
                instance_id BIGINT,
                original_features JSON,
                counterfactual_features JSON,
                original_class INTEGER,
                target_class INTEGER,
                distance DOUBLE,
                created_at TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE counterfactual_clusters (
                id BIGINT PRIMARY KEY,
                model_name VARCHAR,
                run_id VARCHAR,
                instance_id BIGINT,
                counterfactual_id BIGINT,
                cluster_id INTEGER,
                created_at TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE global_rules (
                id BIGINT PRIMARY KEY,
                model_name VARCHAR,
                run_id VARCHAR,
                cluster_id INTEGER,
                conditions JSON,
                support INTEGER,
                support_share DOUBLE,
                avg_distance DOUBLE,
                quality_score DOUBLE,
                created_at TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE shap_values (
                id BIGINT PRIMARY KEY,
                model_name VARCHAR,
                run_id VARCHAR,
                instance_id BIGINT,
                feature_name VARCHAR,
                feature_value VARCHAR,
                shap_value DOUBLE,
                created_at TIMESTAMP
            )
            """
        )

        original_1 = json.dumps(
            {
                "education": "HS-grad",
                "education_num": 9,
                "capital_gain": 0,
                "hours_per_week": 40,
            }
        )

        counterfactual_1 = json.dumps(
            {
                "education": "Bachelors",
                "education_num": 13,
                "capital_gain": 5000,
                "hours_per_week": 40,
            }
        )

        original_2 = json.dumps(
            {
                "education": "Bachelors",
                "education_num": 13,
                "capital_gain": 0,
                "hours_per_week": 40,
            }
        )

        counterfactual_2 = json.dumps(
            {
                "education": "Bachelors",
                "education_num": 13,
                "capital_gain": 7000,
                "hours_per_week": 45,
            }
        )

        conn.executemany(
            """
            INSERT INTO counterfactuals
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                (
                    1,
                    MODEL_NAME,
                    RUN_ID,
                    101,
                    original_1,
                    counterfactual_1,
                    0,
                    1,
                    0.05,
                ),
                (
                    2,
                    MODEL_NAME,
                    RUN_ID,
                    102,
                    original_2,
                    counterfactual_2,
                    0,
                    1,
                    0.08,
                ),
            ],
        )

        conn.executemany(
            """
            INSERT INTO counterfactual_clusters
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                (
                    1,
                    MODEL_NAME,
                    RUN_ID,
                    101,
                    1,
                    0,
                ),
                (
                    2,
                    MODEL_NAME,
                    RUN_ID,
                    102,
                    2,
                    1,
                ),
            ],
        )

        conn.executemany(
            """
            INSERT INTO global_rules
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                (
                    1,
                    MODEL_NAME,
                    RUN_ID,
                    0,
                    json.dumps(
                        [
                            "capital_gain tends to increase",
                        ]
                    ),
                    1,
                    0.5,
                    0.05,
                    0.80,
                ),
                (
                    2,
                    MODEL_NAME,
                    RUN_ID,
                    1,
                    json.dumps(
                        [
                            "hours_per_week tends to increase",
                            "capital_gain tends to increase",
                        ]
                    ),
                    1,
                    0.5,
                    0.08,
                    0.65,
                ),
            ],
        )

        conn.executemany(
            """
            INSERT INTO shap_values
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                (
                    1,
                    MODEL_NAME,
                    RUN_ID,
                    101,
                    "capital_gain",
                    "0",
                    0.30,
                ),
                (
                    2,
                    MODEL_NAME,
                    RUN_ID,
                    101,
                    "hours_per_week",
                    "40",
                    -0.10,
                ),
                (
                    3,
                    MODEL_NAME,
                    RUN_ID,
                    102,
                    "capital_gain",
                    "0",
                    -0.50,
                ),
                (
                    4,
                    MODEL_NAME,
                    RUN_ID,
                    102,
                    "hours_per_week",
                    "40",
                    0.20,
                ),
            ],
        )

    return DashboardData(db_path=str(db_path))


def test_available_runs(
    dashboard_data: DashboardData,
) -> None:
    runs = dashboard_data.available_runs()

    assert runs.height == 1
    assert runs["run_id"][0] == RUN_ID
    assert runs["model_name"][0] == MODEL_NAME
    assert runs["counterfactual_count"][0] == 2


def test_run_summary(
    dashboard_data: DashboardData,
) -> None:
    summary = dashboard_data.run_summary(RUN_ID)

    assert summary == {
        "counterfactual_count": 2,
        "shap_value_count": 4,
        "explained_instance_count": 2,
        "cluster_count": 2,
        "rule_count": 2,
    }


def test_counterfactuals_include_clusters(
    dashboard_data: DashboardData,
) -> None:
    counterfactuals = dashboard_data.counterfactuals(RUN_ID)

    assert counterfactuals.height == 2
    assert set(counterfactuals["cluster_id"].to_list()) == {0, 1}


def test_counterfactual_changes(
    dashboard_data: DashboardData,
) -> None:
    changes = dashboard_data.counterfactual_changes(
        counterfactual_id=1,
        run_id=RUN_ID,
    )

    changed_features = set(changes["feature"].to_list())

    assert changed_features == {
        "education",
        "education_num",
        "capital_gain",
    }

    # hours_per_week did not change.
    assert "hours_per_week" not in (changed_features)


def test_global_rules_are_ranked_by_quality(
    dashboard_data: DashboardData,
) -> None:
    rules = dashboard_data.global_rules(RUN_ID)

    assert rules.height == 2
    assert rules["cluster_id"][0] == 0
    assert rules["quality_score"][0] == pytest.approx(0.80)


def test_parse_rule_conditions(
    dashboard_data: DashboardData,
) -> None:
    rules = dashboard_data.global_rules(RUN_ID)

    conditions = dashboard_data.parse_json_list(rules["conditions"][0])

    assert conditions == ["capital_gain tends to increase"]


def test_shap_global_importance(
    dashboard_data: DashboardData,
) -> None:
    importance = dashboard_data.shap_importance(
        RUN_ID,
        limit=10,
    )

    assert importance.height == 2

    assert importance["feature_name"][0] == "capital_gain"

    # mean(|0.30|, |-0.50|) = 0.40
    assert importance["mean_abs_shap"][0] == pytest.approx(0.40)


def test_shap_instances(
    dashboard_data: DashboardData,
) -> None:
    instances = dashboard_data.shap_instances(RUN_ID)

    assert instances == [101, 102]


def test_shap_for_instance(
    dashboard_data: DashboardData,
) -> None:
    explanation = dashboard_data.shap_for_instance(
        run_id=RUN_ID,
        instance_id=101,
    )

    assert explanation.height == 2

    # Ordered by absolute SHAP magnitude.
    assert explanation["feature_name"][0] == "capital_gain"

    assert explanation["shap_value"][0] == pytest.approx(0.30)
