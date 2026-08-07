"""SHAP value computation for model interpretability."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import shap

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP value explainer for model interpretability."""

    def __init__(
        self,
        model,
        x_train: np.ndarray,
        feature_names: list[str],
        model_type: str = "tree",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model
            x_train: Training data
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'deep')
        """
        self.model = model
        self.X_train = x_train
        self.feature_names = feature_names

        # Create appropriate explainer
        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, x_train)
        elif model_type == "deep":
            self.explainer = shap.DeepExplainer(model, x_train)
        else:
            # Default to KernelExplainer (slower but general)
            self.explainer = shap.KernelExplainer(model.predict, x_train[:100])

        logger.info(f"SHAP explainer initialized ({model_type})")

    def compute_shap_values(
        self,
        x: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compute SHAP values for instances.

        Args:
            x: Instances to explain
            feature_names: Optional feature names override

        Returns:
            Dictionary with SHAP values and metadata
        """
        if feature_names is None:
            feature_names = self.feature_names

        logger.info(f"Computing SHAP values for {len(x)} instances...")

        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(x)

            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output
                shap_values = shap_values[1]  # Take positive class

            # Convert to DataFrame for easier handling
            shap_df = pd.DataFrame(shap_values, columns=feature_names)

            return {
                "shap_values": shap_values,
                "shap_df": shap_df,
                "feature_names": feature_names,
                "base_value": self.explainer.expected_value,
            }

        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return {"error": str(e)}

    def compute_batch(
        self,
        x: np.ndarray,
        batch_size: int = 100,
        feature_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Compute SHAP values in batches.

        Args:
            x: Instances to explain
            batch_size: Number of instances per batch
            feature_names: Optional feature names override

        Returns:
            Polars DataFrame with SHAP values
        """
        if feature_names is None:
            feature_names = self.feature_names

        all_results = []

        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            current_batch = i // batch_size + 1
            total_batches = (len(x) - 1) // batch_size + 1
            logger.info(f"Processing batch {current_batch}/{total_batches}")

            result = self.compute_shap_values(batch, feature_names)

            if "error" in result:
                logger.warning(f"Error in batch: {result['error']}")
                continue

            shap_values = result["shap_values"]

            # Convert to records
            for j in range(len(batch)):
                for k, feature in enumerate(feature_names):
                    all_results.append(
                        {
                            "instance_id": i + j,
                            "feature_name": feature,
                            "shap_value": shap_values[j][k],
                            "feature_value": batch[j][k],
                        }
                    )

        # Convert to Polars DataFrame
        df = pl.DataFrame(all_results)
        logger.info(
            f"Computed SHAP values for {len(x)} instances, "
            f"{len(feature_names)} features"
        )

        return df
