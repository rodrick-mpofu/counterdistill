"""DiCE counterfactual explanation generator."""

import logging
from typing import Any

import dice_ml
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class DiceExplainer:
    """DiCE counterfactual explanation generator."""

    def __init__(
        self,
        model,
        data_df: pl.DataFrame,
        target: str = "target",
        categorical_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
    ):
        """
        Initialize DiCE explainer.

        Args:
            model: Trained sklearn model
            data_df: Training data (Polars DataFrame)
            target: Target column name
            categorical_features: List of categorical feature names
            continuous_features: List of continuous feature names
        """
        self.model = model

        # Convert to pandas for DiCE compatibility
        self.data_df = data_df.to_pandas()

        # Determine feature types
        if categorical_features is None:
            categorical_features = [
                col
                for col in data_df.columns
                if data_df[col].dtype in [pl.Utf8, pl.String, pl.Categorical]
            ]

        if continuous_features is None:
            continuous_features = [
                col
                for col in data_df.columns
                if col not in categorical_features and col != target
            ]

        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        # Create DiCE data object
        self.dice_data = dice_ml.Data(
            dataframe=self.data_df,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            target=target,
        )

        # Create DiCE model object
        self.dice_model = dice_ml.Model(
            model=self.model,
            backend="sklearn",
        )

        # Create explainer
        self.explainer = dice_ml.Dice(
            data_interface=self.dice_data,
            model_interface=self.dice_model,
            method="random",
        )

        logger.info("DiCE explainer initialized")

    def generate_counterfactuals(
        self,
        query_instance: np.ndarray,
        total_cfs: int = 5,
        desired_class: str | int = 1,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
    ) -> dict[str, Any]:
        """
        Generate counterfactual explanations for a query instance.

        Args:
            query_instance: Input instance to explain
            total_cfs: Number of counterfactuals to generate
            desired_class: Target class label
            proximity_weight: Weight for proximity
            diversity_weight: Weight for diversity

        Returns:
            Dictionary with counterfactuals and metadata
        """
        try:
            # Convert to DataFrame
            query_df = pd.DataFrame([query_instance], columns=self.data_df.columns[:-1])

            # Generate counterfactuals
            cf = self.explainer.generate_counterfactuals(
                query_df,
                total_CFs=total_cfs,
                desired_class=str(desired_class),
                proximity_weight=proximity_weight,
                diversity_weight=diversity_weight,
            )

            # Extract counterfactuals
            cf_df = cf.cf_examples_list[0].final_cfs_df
            cf_df = cf_df.drop(columns=["target"])  # Remove target column

            # Calculate distances
            distances = []
            for i in range(len(cf_df)):
                # Euclidean distance between original and counterfactual
                diff = cf_df.iloc[i].values - query_instance
                distance = np.sqrt(np.sum(diff**2))
                distances.append(distance)

            return {
                "counterfactuals": cf_df,
                "distances": distances,
                "original_instance": query_instance,
                "desired_class": desired_class,
            }

        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return {"error": str(e)}

    def generate_batch(
        self,
        x_test: np.ndarray,
        num_samples: int = 100,
        total_cfs: int = 5,
        desired_class: int = 1,
    ) -> pl.DataFrame:
        """
        Generate counterfactuals for multiple instances.

        Args:
            x_test: Test instances to explain
            num_samples: Number of instances to explain
            total_cfs: Number of counterfactuals per instance
            desired_class: Target class

        Returns:
            Polars DataFrame with counterfactuals
        """
        results = []

        # Sample if too many
        if len(x_test) > num_samples:
            indices = np.random.choice(len(x_test), num_samples, replace=False)
            x_test = x_test[indices]

        logger.info(f"Generating counterfactuals for {len(x_test)} instances...")

        for i, instance in enumerate(x_test):
            logger.info(f"Processing instance {i+1}/{len(x_test)}")

            # Generate counterfactuals
            result = self.generate_counterfactuals(
                instance,
                total_cfs=total_cfs,
                desired_class=desired_class,
            )

            if "error" in result:
                logger.warning(f"Error for instance {i}: {result['error']}")
                continue

            # Convert counterfactuals to list of dicts
            cf_df = result["counterfactuals"]
            for j in range(len(cf_df)):
                record = {
                    "instance_id": i,
                    "original_features": instance.tolist(),
                    "counterfactual_features": cf_df.iloc[j].values.tolist(),
                    "target_class": desired_class,
                    "original_class": self.model.predict([instance])[0],
                    "distance": result["distances"][j],
                }
                results.append(record)

        # Convert to Polars DataFrame
        df = pl.DataFrame(results)
        logger.info(f"Generated {len(results)} counterfactuals")

        return df
