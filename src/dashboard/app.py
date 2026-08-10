"""Streamlit dashboard for CounterDistill."""

from __future__ import annotations

import polars as pl
import streamlit as st

from src.dashboard.data import DashboardData

st.set_page_config(
    page_title="CounterDistill",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def get_data() -> DashboardData:
    """Create the dashboard data-access layer."""
    return DashboardData()


def render_overview(
    data: DashboardData,
    run_id: str,
    model_name: str,
) -> None:
    """Render the dashboard overview."""
    st.title("CounterDistill")

    st.caption(
        "Distilling local counterfactual explanations "
        "into interpretable global rules."
    )

    st.subheader("Run Overview")

    st.write(f"**Model:** `{model_name}`  \n" f"**Run:** `{run_id}`")

    summary = data.run_summary(run_id)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Counterfactuals",
        summary["counterfactual_count"],
    )

    col2.metric(
        "Explained Instances",
        summary["explained_instance_count"],
    )

    col3.metric(
        "SHAP Values",
        summary["shap_value_count"],
    )

    col4.metric(
        "Clusters",
        summary["cluster_count"],
    )

    col5.metric(
        "Global Rules",
        summary["rule_count"],
    )

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Global SHAP Importance")

        shap = data.shap_importance(
            run_id,
            limit=10,
        )

        if shap.is_empty():
            st.info("No SHAP values are available " "for this run.")
        else:
            shap_chart = shap.sort("mean_abs_shap").to_pandas()

            st.bar_chart(
                shap_chart,
                x="feature_name",
                y="mean_abs_shap",
                horizontal=True,
            )

    with right:
        st.subheader("Top Distilled Rules")

        rules = data.global_rules(run_id)

        if rules.is_empty():
            st.info("No distilled rules are available " "for this run.")

        else:
            for row in rules.head(3).iter_rows(named=True):
                with st.container(border=True):
                    st.markdown(f"**Cluster " f"{row['cluster_id']}**")

                    st.metric(
                        "Quality",
                        f"{row['quality_score']:.3f}",
                    )

                    st.write(f"Coverage: " f"{row['support_share']:.1%}")

                    st.write(f"Support: " f"{row['support']}")

    st.divider()

    st.info(
        "Counterfactual rules describe patterns in "
        "model behavior and should not be interpreted "
        "as causal real-world recommendations."
    )


def render_counterfactuals(
    data: DashboardData,
    run_id: str,
) -> None:
    """Render the counterfactual explorer."""
    st.title("Counterfactual Explorer")

    st.caption("Explore feasible feature changes that alter " "the model's prediction.")

    counterfactuals = data.counterfactuals(run_id)

    if counterfactuals.is_empty():
        st.info("No counterfactuals are available " "for this run.")
        return

    # -------------------------------------------------
    # Filters
    # -------------------------------------------------

    st.subheader("Filters")

    filter_col1, filter_col2 = st.columns(2)

    cluster_values = sorted(
        int(value)
        for value in (counterfactuals["cluster_id"].drop_nulls().unique().to_list())
    )

    with filter_col1:
        selected_clusters = st.multiselect(
            "Clusters",
            options=cluster_values,
            default=cluster_values,
        )

    distance_series = counterfactuals["distance"].drop_nulls()

    max_distance = 1.0 if distance_series.is_empty() else float(distance_series.max())

    slider_max = max(
        max_distance,
        0.001,
    )

    with filter_col2:
        distance_limit = st.slider(
            "Maximum counterfactual distance",
            min_value=0.0,
            max_value=slider_max,
            value=slider_max,
            step=0.001,
            format="%.3f",
        )

    filtered = counterfactuals.filter(pl.col("distance") <= distance_limit)

    if selected_clusters:
        filtered = filtered.filter(pl.col("cluster_id").is_in(selected_clusters))

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Counterfactuals",
        filtered.height,
    )

    unique_instances = (
        filtered["instance_id"].n_unique() if not filtered.is_empty() else 0
    )

    metric2.metric(
        "Instances",
        unique_instances,
    )

    if filtered.is_empty():
        avg_distance = 0.0
    else:
        mean_distance = filtered["distance"].mean()

        avg_distance = float(mean_distance) if mean_distance is not None else 0.0

    metric3.metric(
        "Average Distance",
        f"{avg_distance:.3f}",
    )

    st.divider()

    if filtered.is_empty():
        st.warning("No counterfactuals match " "the selected filters.")
        return

    # -------------------------------------------------
    # Counterfactual table
    # -------------------------------------------------

    st.subheader("Counterfactuals")

    display_columns = [
        "id",
        "instance_id",
        "cluster_id",
        "original_class",
        "target_class",
        "distance",
    ]

    existing_columns = [
        column for column in display_columns if column in filtered.columns
    ]

    st.dataframe(
        filtered.select(existing_columns).sort(
            [
                "instance_id",
                "distance",
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.divider()

    # -------------------------------------------------
    # Individual explanation
    # -------------------------------------------------

    st.subheader("Inspect Counterfactual")

    instance_ids = sorted(
        int(value) for value in (filtered["instance_id"].unique().to_list())
    )

    selected_instance = st.selectbox(
        "Instance",
        options=instance_ids,
    )

    instance_counterfactuals = filtered.filter(
        pl.col("instance_id") == selected_instance
    ).sort("distance")

    counterfactual_ids = [
        int(value) for value in (instance_counterfactuals["id"].to_list())
    ]

    selected_counterfactual = st.selectbox(
        "Counterfactual",
        options=counterfactual_ids,
    )

    selected_row = instance_counterfactuals.filter(
        pl.col("id") == selected_counterfactual
    ).row(
        0,
        named=True,
    )

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Instance",
        selected_instance,
    )

    info2.metric(
        "Cluster",
        (
            selected_row["cluster_id"]
            if selected_row["cluster_id"] is not None
            else "N/A"
        ),
    )

    info3.metric(
        "Distance",
        f"{float(selected_row['distance']):.3f}",
    )

    info4.metric(
        "Target Class",
        selected_row["target_class"],
    )

    st.markdown("#### Feature Changes")

    changes = data.counterfactual_changes(
        counterfactual_id=(selected_counterfactual),
        run_id=run_id,
    )

    if changes.is_empty():
        st.info("No feature changes were found.")
    else:
        st.dataframe(
            changes,
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "These counterfactuals describe model "
        "behavior under feasible feature changes. "
        "They are not causal recommendations."
    )


def render_global_rules(
    data: DashboardData,
    run_id: str,
) -> None:
    """Render distilled global counterfactual rules."""
    st.title("Global Rules")

    st.caption(
        "Explore interpretable rules distilled from "
        "clusters of local counterfactual explanations."
    )

    rules = data.global_rules(run_id)

    if rules.is_empty():
        st.info("No distilled global rules are available " "for this run.")
        return

    # -------------------------------------------------
    # Filters
    # -------------------------------------------------

    st.subheader("Filters")

    filter_col1, filter_col2 = st.columns(2)

    cluster_ids = sorted(
        int(value) for value in (rules["cluster_id"].drop_nulls().unique().to_list())
    )

    with filter_col1:
        selected_clusters = st.multiselect(
            "Clusters",
            options=cluster_ids,
            default=cluster_ids,
        )

    quality_values = rules["quality_score"].drop_nulls()

    if quality_values.is_empty():
        max_quality = 1.0
    else:
        maximum = quality_values.max()

        max_quality = float(maximum) if maximum is not None else 1.0

    with filter_col2:
        minimum_quality = st.slider(
            "Minimum quality score",
            min_value=0.0,
            max_value=max(
                max_quality,
                0.01,
            ),
            value=0.0,
            step=0.01,
            format="%.2f",
        )

    filtered = rules.filter(pl.col("quality_score") >= minimum_quality)

    if selected_clusters:
        filtered = filtered.filter(pl.col("cluster_id").is_in(selected_clusters))

    # -------------------------------------------------
    # Summary metrics
    # -------------------------------------------------

    if filtered.is_empty():
        st.warning("No global rules match " "the selected filters.")
        return

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Rules",
        filtered.height,
    )

    total_support_value = filtered["support"].sum()

    total_support = int(total_support_value) if total_support_value is not None else 0

    metric2.metric(
        "Total Support",
        total_support,
    )

    avg_quality_value = filtered["quality_score"].mean()

    avg_quality = float(avg_quality_value) if avg_quality_value is not None else 0.0

    metric3.metric(
        "Average Quality",
        f"{avg_quality:.3f}",
    )

    avg_distance_value = filtered["avg_distance"].mean()

    avg_distance = float(avg_distance_value) if avg_distance_value is not None else 0.0

    metric4.metric(
        "Average Distance",
        f"{avg_distance:.3f}",
    )

    st.divider()

    # -------------------------------------------------
    # Rule comparison table
    # -------------------------------------------------

    st.subheader("Rule Comparison")

    comparison = filtered.select(
        [
            "cluster_id",
            "support",
            "support_share",
            "avg_distance",
            "quality_score",
        ]
    ).sort(
        "quality_score",
        descending=True,
    )

    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "cluster_id": st.column_config.NumberColumn(
                "Cluster",
                format="%d",
            ),
            "support": st.column_config.NumberColumn(
                "Support",
                format="%d",
            ),
            "support_share": (
                st.column_config.ProgressColumn(
                    "Coverage",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.1%%",
                )
            ),
            "avg_distance": (
                st.column_config.NumberColumn(
                    "Avg Distance",
                    format="%.3f",
                )
            ),
            "quality_score": (
                st.column_config.ProgressColumn(
                    "Quality",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                )
            ),
        },
    )

    st.divider()

    # -------------------------------------------------
    # Rule cards
    # -------------------------------------------------

    st.subheader("Distilled Rules")

    ranked_rules = filtered.sort(
        "quality_score",
        descending=True,
    )

    for rank, row in enumerate(
        ranked_rules.iter_rows(named=True),
        start=1,
    ):
        cluster_id = int(row["cluster_id"])

        conditions = data.parse_json_list(row["conditions"])

        with st.container(border=True):
            title_col, quality_col = st.columns([4, 1])

            with title_col:
                st.markdown(f"### #{rank} — Cluster " f"{cluster_id}")

            with quality_col:
                st.metric(
                    "Quality",
                    f"{float(row['quality_score']):.3f}",
                )

            st.markdown("#### Rule")

            if not conditions:
                st.write("No conditions extracted.")

            else:
                rule_lines: list[str] = []

                for index, condition in enumerate(conditions):
                    prefix = "IF" if index == 0 else "AND"

                    rule_lines.append(f"**{prefix}** {condition}")

                st.markdown("  \n".join(rule_lines))

            stat1, stat2, stat3 = st.columns(3)

            stat1.metric(
                "Support",
                int(row["support"]),
            )

            stat2.metric(
                "Coverage",
                (f"{float(row['support_share']):.1%}"),
            )

            stat3.metric(
                "Avg Distance",
                (f"{float(row['avg_distance']):.3f}"),
            )

    st.divider()

    st.info(
        "These rules summarize recurring model "
        "counterfactual patterns. They describe "
        "model behavior and should not be interpreted "
        "as causal or prescriptive recommendations."
    )


def render_shap(
    data: DashboardData,
    run_id: str,
) -> None:
    """Render global and local SHAP explanations."""
    st.title("SHAP Explorer")

    st.caption(
        "Explore global feature importance and " "instance-level model contributions."
    )

    # -------------------------------------------------
    # Global SHAP importance
    # -------------------------------------------------

    st.subheader("Global Feature Importance")

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
    )

    importance = data.shap_importance(
        run_id=run_id,
        limit=top_n,
    )

    if importance.is_empty():
        st.info("No SHAP values are available " "for this run.")
        return

    global_left, global_right = st.columns([2, 1])

    with global_left:
        chart_data = importance.sort("mean_abs_shap")

        st.bar_chart(
            chart_data,
            x="feature_name",
            y="mean_abs_shap",
            horizontal=True,
            sort=False,
            x_label="Mean |SHAP value|",
            y_label="Feature",
        )

    with global_right:
        st.markdown("#### Top Features")

        ranked_importance = importance.with_row_index(
            name="rank",
            offset=1,
        )

        st.dataframe(
            ranked_importance,
            hide_index=True,
            width="stretch",
            column_config={
                "rank": (
                    st.column_config.NumberColumn(
                        "Rank",
                        format="%d",
                    )
                ),
                "feature_name": (st.column_config.TextColumn("Feature")),
                "mean_abs_shap": (
                    st.column_config.NumberColumn(
                        "Mean |SHAP|",
                        format="%.4f",
                    )
                ),
            },
        )

    st.caption(
        "Mean absolute SHAP value measures how "
        "strongly a feature influences predictions "
        "on average, regardless of direction."
    )

    st.divider()

    # -------------------------------------------------
    # Local explanations
    # -------------------------------------------------

    st.subheader("Instance-Level Explanation")

    instance_ids = data.shap_instances(run_id)

    if not instance_ids:
        st.info("No instance-level SHAP explanations " "are available for this run.")
        return

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        selected_instance = st.selectbox(
            "Instance",
            options=instance_ids,
        )

    with control_col2:
        local_top_n = st.slider(
            "Local features",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
        )

    local_shap = data.shap_for_instance(
        run_id=run_id,
        instance_id=selected_instance,
    )

    if local_shap.is_empty():
        st.warning("No SHAP explanation was found " "for this instance.")
        return

    # The query is already ordered by absolute SHAP value.
    local_top = local_shap.head(local_top_n)

    # -------------------------------------------------
    # Local summary metrics
    # -------------------------------------------------

    positive = local_shap.filter(pl.col("shap_value") > 0)

    negative = local_shap.filter(pl.col("shap_value") < 0)

    strongest_positive_feature = "N/A"
    strongest_positive_value = 0.0

    if not positive.is_empty():
        positive_row = positive.sort(
            "shap_value",
            descending=True,
        ).row(
            0,
            named=True,
        )

        strongest_positive_feature = str(positive_row["feature_name"])

        strongest_positive_value = float(positive_row["shap_value"])

    strongest_negative_feature = "N/A"
    strongest_negative_value = 0.0

    if not negative.is_empty():
        negative_row = negative.sort("shap_value").row(
            0,
            named=True,
        )

        strongest_negative_feature = str(negative_row["feature_name"])

        strongest_negative_value = float(negative_row["shap_value"])

    total_abs_value = local_shap.select(
        pl.col("shap_value").abs().sum().alias("total_abs_shap")
    ).item()

    total_abs_shap = float(total_abs_value if total_abs_value is not None else 0.0)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Strongest Positive",
        strongest_positive_feature,
        delta=(f"{strongest_positive_value:+.4f}"),
    )

    metric2.metric(
        "Strongest Negative",
        strongest_negative_feature,
        delta=(f"{strongest_negative_value:+.4f}"),
        delta_color="inverse",
    )

    metric3.metric(
        "Total |SHAP|",
        f"{total_abs_shap:.4f}",
    )

    st.divider()

    # -------------------------------------------------
    # Local SHAP chart
    # -------------------------------------------------

    st.markdown(f"#### Contributions for Instance " f"{selected_instance}")

    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        local_chart = local_top.sort("shap_value")

        st.bar_chart(
            local_chart,
            x="feature_name",
            y="shap_value",
            horizontal=True,
            sort=False,
            x_label="SHAP Value",
            y_label="Feature",
        )

        st.caption(
            "Positive values push the model output "
            "higher; negative values push it lower "
            "relative to the model baseline."
        )

    with table_col:
        st.markdown("#### Feature Details")

        st.dataframe(
            local_top.select(
                [
                    "feature_name",
                    "feature_value",
                    "shap_value",
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "feature_name": (st.column_config.TextColumn("Feature")),
                "feature_value": (st.column_config.TextColumn("Value")),
                "shap_value": (
                    st.column_config.NumberColumn(
                        "SHAP",
                        format="%.4f",
                    )
                ),
            },
        )

    st.divider()

    st.info(
        "SHAP values explain the model's prediction "
        "behavior. They identify influential model "
        "features but do not establish causal effects."
    )


def main() -> None:
    """Run the CounterDistill dashboard."""
    data = get_data()

    runs = data.available_runs()

    if runs.is_empty():
        st.error("No explainability runs were found.")
        st.stop()

    run_ids = runs["run_id"].to_list()

    st.sidebar.title("CounterDistill")

    selected_run = st.sidebar.selectbox(
        "Explanation run",
        options=run_ids,
        index=0,
    )

    selected = runs.filter(runs["run_id"] == selected_run)

    model_name = str(selected["model_name"][0])

    page = st.sidebar.radio(
        "View",
        [
            "Overview",
            "Counterfactuals",
            "Global Rules",
            "SHAP",
        ],
    )

    if page == "Overview":
        render_overview(
            data=data,
            run_id=selected_run,
            model_name=model_name,
        )

    elif page == "Counterfactuals":
        render_counterfactuals(
            data=data,
            run_id=selected_run,
        )

    elif page == "Global Rules":
        render_global_rules(
            data=data,
            run_id=selected_run,
        )

    elif page == "SHAP":
        render_shap(
            data=data,
            run_id=selected_run,
        )


if __name__ == "__main__":
    main()
