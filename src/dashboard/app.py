"""Streamlit dashboard for CounterDistill."""

from __future__ import annotations

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
        st.title("Counterfactual Explorer")

        st.info("Counterfactual explorer coming next.")

    elif page == "Global Rules":
        st.title("Global Rules")

        st.info("Global rule explorer coming next.")

    elif page == "SHAP":
        st.title("SHAP Explanations")

        st.info("SHAP explorer coming next.")


if __name__ == "__main__":
    main()
