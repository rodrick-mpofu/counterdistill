from src.dashboard.data import DashboardData

RUN_ID = "local-20260808-220006"


def main() -> None:
    data = DashboardData()

    runs = data.available_runs()

    print("\nAvailable runs:")
    print(runs.head())

    counterfactuals = data.counterfactuals(RUN_ID)

    print(f"\nCounterfactuals: " f"{counterfactuals.height}")

    rules = data.global_rules(RUN_ID)

    print(f"Global rules: " f"{rules.height}")

    shap = data.shap_importance(
        RUN_ID,
        limit=10,
    )

    print("\nTop SHAP features:")
    print(shap)


if __name__ == "__main__":
    main()
