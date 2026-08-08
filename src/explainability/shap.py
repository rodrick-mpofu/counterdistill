"""SHAP value computation for encoded model features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
import shap

logger = logging.getLogger(__name__)


class ShapExplainer:
    """Compute SHAP values in the exact feature space consumed by the model."""

    def __init__(
        self,
        model: Any,
        x_train: np.ndarray,
        feature_names: list[str],
        model_type: str = "tree",
        class_index: int = 1,
    ) -> None:
        self.model = model
        self.x_train = np.asarray(x_train)
        self.feature_names = feature_names
        self.class_index = class_index

        if self.x_train.ndim != 2:
            raise ValueError("x_train must be a 2-D array.")
        if self.x_train.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match x_train columns.")

        if model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            self.explainer = shap.LinearExplainer(model, self.x_train)
        elif model_type == "deep":
            self.explainer = shap.DeepExplainer(model, self.x_train)
        else:
            background = self.x_train[: min(100, len(self.x_train))]
            if hasattr(model, "predict_proba"):

                def prediction_fn(x):
                    return model.predict_proba(x)[:, self.class_index]

            else:
                prediction_fn = model.predict
            self.explainer = shap.KernelExplainer(prediction_fn, background)

        logger.info("SHAP explainer initialized (%s)", model_type)

    def _select_output(self, values: Any) -> np.ndarray:
        """Normalize legacy/new SHAP classifier output to (rows, features)."""
        if isinstance(values, list):
            if not values:
                raise ValueError("SHAP returned an empty list.")
            index = min(self.class_index, len(values) - 1)
            selected = np.asarray(values[index])
        else:
            selected = np.asarray(values)

        if selected.ndim == 3:
            if selected.shape[2] <= self.class_index:
                raise ValueError(
                    f"SHAP output has {selected.shape[2]} outputs; "
                    f"class_index={self.class_index} is invalid."
                )
            selected = selected[:, :, self.class_index]

        if selected.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP output shape {selected.shape}; "
                "expected (samples, features) after class selection."
            )

        if selected.shape[1] != len(self.feature_names):
            raise ValueError(
                "SHAP feature dimension does not match feature_names: "
                f"{selected.shape[1]} != {len(self.feature_names)}"
            )

        return selected

    def _base_value(self) -> float | list[float]:
        expected = np.asarray(self.explainer.expected_value)

        if expected.ndim == 0:
            return float(expected)

        flat = expected.reshape(-1)
        if len(flat) > self.class_index:
            return float(flat[self.class_index])
        return [float(value) for value in flat]

    def compute_shap_values(
        self,
        x: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP values and normalize binary/multiclass output shapes."""
        names = feature_names or self.feature_names
        x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError("x must be a 2-D array.")
        if x.shape[1] != len(names):
            raise ValueError("feature_names length must match x columns.")

        logger.info("Computing SHAP values for %d instances...", len(x))

        try:
            raw_values = self.explainer.shap_values(x)
            values = self._select_output(raw_values)

            return {
                "shap_values": values,
                "feature_names": names,
                "base_value": self._base_value(),
            }
        except Exception as exc:
            logger.exception("Error computing SHAP values")
            return {"error": str(exc)}

    def compute_batch(
        self,
        x: np.ndarray,
        batch_size: int = 100,
        feature_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """Compute long-form SHAP records suitable for DuckDB storage."""
        names = feature_names or self.feature_names
        x = np.asarray(x)

        all_results: list[dict[str, Any]] = []

        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            current_batch = start // batch_size + 1
            total_batches = (len(x) - 1) // batch_size + 1
            logger.info(
                "Processing SHAP batch %d/%d",
                current_batch,
                total_batches,
            )

            result = self.compute_shap_values(batch, names)
            if "error" in result:
                logger.warning("Skipping SHAP batch: %s", result["error"])
                continue

            values = result["shap_values"]

            for row_offset, row in enumerate(batch):
                instance_id = start + row_offset
                for feature_idx, feature in enumerate(names):
                    all_results.append(
                        {
                            "instance_id": instance_id,
                            "feature_name": feature,
                            "shap_value": float(values[row_offset, feature_idx]),
                            "feature_value": float(row[feature_idx]),
                        }
                    )

        if not all_results:
            return pl.DataFrame(
                schema={
                    "instance_id": pl.Int64,
                    "feature_name": pl.String,
                    "shap_value": pl.Float64,
                    "feature_value": pl.Float64,
                }
            )

        df = pl.DataFrame(all_results)
        logger.info(
            "Computed SHAP values for %d instances and %d features",
            len(x),
            len(names),
        )
        return df
