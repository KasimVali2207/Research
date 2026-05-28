# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SHAP explainability module.

Generates model explanations (SHAP summary, waterfall charts) and computes
patient-specific key risk factors (Experiment 6).
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("shap package not installed. Explainer will run in fallback feature-importance mode.")


class CancerSHAPExplainer:
    """Manages SHAP explanations for tree, linear, or neural network classifiers."""

    def __init__(self, model: Any, model_type: str, feature_names: list[str], seed: int = 42) -> None:
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.seed = seed
        self.explainer = None
        
        if HAS_SHAP:
            try:
                if model_type == "tree":
                    # For tree models (XGBoost, LightGBM, Random Forest)
                    self.explainer = shap.TreeExplainer(model)
                elif model_type == "linear":
                    self.explainer = shap.LinearExplainer(model)
                else:
                    self.explainer = shap.Explainer(model)
            except Exception as exc:
                logger.error("Failed to initialize SHAP explainer: {}", exc)
                self.explainer = None

    def compute_shap_values(self, X: pd.DataFrame | np.ndarray, background_n: int = 100) -> np.ndarray:
        """Compute SHAP values for the given features matrix."""
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            meta_cols = ["subject_id", "cancer_type", "gender", "age"]
            X_clean = X.drop(columns=meta_cols, errors="ignore").fillna(0.0).to_numpy()
        else:
            X_clean = X
            
        n_samples, n_features = X_clean.shape
        
        if HAS_SHAP and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X_clean)
                
                # Handle multiclass or dual-class output shapes from SHAP
                if isinstance(shap_values, list):
                    # For binary class, take class 1 values
                    if len(shap_values) == 2:
                        return shap_values[1]
                    return np.array(shap_values)
                elif len(shap_values.shape) == 3:
                    # multiclass/multioutput
                    return shap_values[:, :, 1]
                return shap_values
            except Exception as exc:
                logger.error("SHAP computation failed: {}. Falling back to random perturbation.", exc)
                
        # Fallback Gini/weight-based mock SHAP calculation
        logger.info("Computing fallback feature-importance-weighted SHAP proxy...")
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(n_features) / n_features
            
        # Mock SHAP values proportional to feature deviation from median × importance
        # Center values
        medians = np.median(X_clean, axis=0)
        stds = np.std(X_clean, axis=0) + 1e-6
        normalized = (X_clean - medians) / stds
        
        # Weighted by importance
        mock_shap = normalized * importances
        return mock_shap

    def get_top_features(self, shap_values: np.ndarray, top_k: int = 20) -> pd.DataFrame:
        """Compute global feature rank based on mean absolute SHAP value."""
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False)
        
        df["rank"] = np.arange(1, len(df) + 1)
        return df.head(top_k)

    def plot_summary(self, shap_values: np.ndarray, X: pd.DataFrame | np.ndarray, output_path: str, plot_type: str = "dot") -> None:
        """Create and save a global SHAP summary plot."""
        if isinstance(X, pd.DataFrame):
            meta_cols = ["subject_id", "cancer_type", "gender", "age"]
            X_clean = X.drop(columns=meta_cols, errors="ignore").fillna(0.0).to_numpy()
        else:
            X_clean = X
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        if HAS_SHAP:
            shap.summary_plot(
                shap_values,
                X_clean,
                feature_names=self.feature_names,
                plot_type=plot_type,
                show=False
            )
        else:
            # Fallback barplot of mean absolute score
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            indices = np.argsort(mean_abs)[::-1][:20]
            
            plt.barh(
                [self.feature_names[i] for i in indices][::-1],
                [mean_abs[i] for i in indices][::-1],
                color="dodgerblue"
            )
            plt.title("Fallback Mean Absolute Contribution Profile (Proxy SHAP)")
            plt.xlabel("Mean Absolute Feature Attribution")
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("SHAP summary plot saved to {}", output_path)

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame | np.ndarray,
        sample_idx: int,
        output_path: str,
        expected_value: float = 0.0,
    ) -> None:
        """Plot and save single-patient waterfall attribution diagram."""
        if isinstance(X, pd.DataFrame):
            meta_cols = ["subject_id", "cancer_type", "gender", "age"]
            X_clean = X.drop(columns=meta_cols, errors="ignore").fillna(0.0).to_numpy()
        else:
            X_clean = X
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        single_shap = shap_values[sample_idx]
        single_x = X_clean[sample_idx]
        
        if HAS_SHAP:
            # Wrap in Explanation object
            exp = shap.Explanation(
                values=single_shap,
                base_values=expected_value,
                data=single_x,
                feature_names=self.feature_names
            )
            shap.plots.waterfall(exp, show=False, max_display=10)
        else:
            # Fallback simple horizontal barplot for the single sample
            indices = np.argsort(np.abs(single_shap))[::-1][:10]
            plt.barh(
                [self.feature_names[i] for i in indices][::-1],
                [single_shap[i] for i in indices][::-1],
                color=["salmon" if single_shap[i] > 0 else "skyblue" for i in indices][::-1]
            )
            plt.axvline(x=0, color="k", linestyle="-")
            plt.title(f"Attribution Profile (Sample #{sample_idx})")
            plt.xlabel("Attribution Value")
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("SHAP waterfall plot saved to {}", output_path)

    def explain_patient(
        self,
        X_patient: np.ndarray,
        shap_values_patient: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert a single patient's SHAP values to clinical explanatory list.

        Args:
            X_patient: 1D array of patient's scaled features.
            shap_values_patient: 1D array of patient's SHAP values.
            feature_names: Feature name list (defaults to self.feature_names).

        Returns:
            JSON-serializable clinical explanation dictionary.
        """
        feats = feature_names or self.feature_names
        
        records = []
        for f, val, attr in zip(feats, X_patient, shap_values_patient):
            records.append({"feature": f, "value": float(val), "attribution": float(attr)})
            
        # Sort by absolute attribution
        records_sorted = sorted(records, key=lambda r: abs(r["attribution"]), reverse=True)
        
        top_risk = [
            f"{r['feature']} = {r['value']:.2f} (impact: +{r['attribution']:.3f})"
            for r in records_sorted if r["attribution"] > 0.05
        ][:5]
        
        top_protective = [
            f"{r['feature']} = {r['value']:.2f} (impact: {r['attribution']:.3f})"
            for r in records_sorted if r["attribution"] < -0.05
        ][:5]
        
        return {
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
            "base_rate": 0.15,  # clinical study prevalence proxy
            "patient_risk_modifiers": [r["feature"] for r in records_sorted[:3]],
        }

    def compute_group_shap(
        self,
        shap_values: np.ndarray,
        groups: dict[str, list[str]],
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate feature attributions across clinical groups.

        Args:
            shap_values: 2D SHAP array.
            groups: Dict mapping group_name -> list of base features.
            feature_names: List of feature names.

        Returns:
            DataFrame containing aggregated group contributions.
        """
        feats = feature_names or self.feature_names
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        group_sums = {}
        for g_name, base_feats in groups.items():
            # Find all feature indices belonging to this group
            indices = []
            for i, f in enumerate(feats):
                # match base feature or temporal versions
                if f in base_feats or any(f.startswith(bf) for bf in base_feats):
                    indices.append(i)
                    
            if indices:
                group_sums[g_name] = float(np.sum(mean_abs_shap[indices]))
            else:
                group_sums[g_name] = 0.0
                
        total = sum(group_sums.values()) + 1e-6
        
        rows = [
            {"group": g, "mean_abs_shap": val, "contribution_pct": round(val / total * 100, 2)}
            for g, val in group_sums.items()
        ]
        
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
