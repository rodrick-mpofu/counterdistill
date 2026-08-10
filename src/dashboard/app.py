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
        st.title("Global Rules")

        st.info("Global rule explorer coming next.")

    elif page == "SHAP":
        st.title("SHAP Explanations")

        st.info("SHAP explorer coming next.")


if __name__ == "__main__":
    main()
