# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Temporal modeling and comparative evaluation module.

Implements Experiment 2: comparing static snapshots (mean, last-value)
against full temporal trajectories using gradient boosted trees.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, brier_score_loss
from xgboost import XGBClassifier
from typing import Any

from src.features.temporal_features import TemporalFeatureExtractor, create_static_snapshot


# Simple DeLong test approximation or permutation test for AUROC comparison
def compare_auroc_p_value(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray, n_boot: int = 100) -> float:
    """Compute permutation/bootstrap p-value comparing two model predictions."""
    y_true = np.array(y_true)
    y_prob_a = np.array(y_prob_a)
    y_prob_b = np.array(y_prob_b)
    
    auc_diff = abs(roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b))
    
    if auc_diff == 0:
        return 1.0
        
    # Bootstrap method
    count = 0
    np.random.seed(42)
    for _ in range(n_boot):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        boot_y = y_true[indices]
        if len(np.unique(boot_y)) < 2:
            continue
        auc_a = roc_auc_score(boot_y, y_prob_a[indices])
        auc_b = roc_auc_score(boot_y, y_prob_b[indices])
        if abs(auc_a - auc_b) >= auc_diff:
            count += 1
            
    return float(count / n_boot)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate the Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)


class TemporalModelEvaluator:
    """Compares static and temporal models and analyzes trajectory feature importance."""

    def __init__(self, base_model_cfg: dict, seed: int = 42) -> None:
        self.base_model_cfg = base_model_cfg
        self.seed = seed

    def compare_static_vs_temporal(
        self,
        X_static_train: pd.DataFrame,
        X_static_test: pd.DataFrame,
        X_temporal_train: pd.DataFrame,
        X_temporal_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train XGBoost classifiers and compare different static/temporal feature subsets."""
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        
        # Datasets configurations to evaluate
        configs = {
            "static_last": [c for c in X_static_train.columns if c not in meta_cols],
            "temporal_full": [c for c in X_temporal_train.columns if c not in meta_cols],
            "temporal_slope_only": [c for c in X_temporal_train.columns if c.endswith("_trend_slope")],
            "temporal_trend_only": [c for c in X_temporal_train.columns if c.endswith(("_trend_slope", "_velocity", "_delta"))],
        }

        results = []
        prob_dict = {}
        
        for name, cols in configs.items():
            if len(cols) == 0:
                continue
                
            # Select correct training set
            X_tr = X_temporal_train if "temporal" in name else X_static_train
            X_te = X_temporal_test if "temporal" in name else X_static_test
            
            # Clean data
            X_tr_clean = X_tr[cols].fillna(0.0)
            X_te_clean = X_te[cols].fillna(0.0)
            
            # Fit best baseline (XGBoost)
            model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=self.seed, n_jobs=-1)
            model.fit(X_tr_clean, y_train)
            
            # Predict
            probs = model.predict_proba(X_te_clean)[:, 1]
            prob_dict[name] = probs
            preds = (probs >= 0.5).astype(int)
            
            # Calculate metrics
            auroc = roc_auc_score(y_test, probs)
            
            precision, recall, _ = precision_recall_curve(y_test, probs)
            auprc = auc(recall, precision)
            
            f1 = f1_score(y_test, preds)
            ece = compute_ece(y_test.to_numpy(), probs)
            brier = brier_score_loss(y_test, probs)
            
            results.append({
                "setting": name,
                "auroc": float(auroc),
                "auprc": float(auprc),
                "f1": float(f1),
                "ece": float(ece),
                "brier": float(brier),
                "p_value_vs_static": 1.0  # calculated below
            })
            
        results_df = pd.DataFrame(results)
        
        # Calculate statistical significance vs static_last
        if "static_last" in prob_dict:
            static_probs = prob_dict["static_last"]
            for i, row in results_df.iterrows():
                setting = row["setting"]
                if setting != "static_last":
                    p_val = compare_auroc_p_value(y_test.to_numpy(), static_probs, prob_dict[setting])
                    results_df.at[i, "p_value_vs_static"] = p_val
                    
        return results_df

    def analyze_temporal_importance(self, model: Any, feature_names: list[str]) -> pd.DataFrame:
        """Aggregate feature importances by their temporal statistic types."""
        # Get importances from XGBoost / LightGBM model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "feature_importances"):
            importances = model.feature_importances
        else:
            raise ValueError("Model does not expose feature importances.")
            
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        
        # Map feature suffixes to temporal stat groups
        # stats = ['mean', 'median', 'std', 'trend_slope', 'delta', 'velocity', 'moving_avg_last', 'exp_smooth_last', 'n_measurements', 'n_abnormal', 'first_value', 'last_value']
        def get_stat_type(feat):
            for suffix in ["_trend_slope", "_velocity", "_delta", "_exp_smooth_last", "_moving_avg_last", "_mean", "_median", "_std", "_min", "_max", "_range", "_cv", "_first_value", "_last_value", "_n_measurements", "_n_abnormal"]:
                if feat.endswith(suffix):
                    return suffix.lstrip("_")
            return "static"

        df["stat_type"] = df["feature"].apply(get_stat_type)
        
        # Group and sum importances
        grouped = df.groupby("stat_type").agg(
            importance_sum=("importance", "sum"),
            feature_count=("importance", "count")
        ).reset_index()
        
        grouped["importance_pct"] = grouped["importance_sum"] / grouped["importance_sum"].sum() * 100
        grouped = grouped.sort_values(by="importance_sum", ascending=False)
        
        return grouped

    def plot_temporal_comparison(self, results_df: pd.DataFrame, output_path: str) -> None:
        """Create a comparative bar chart of model performances."""
        plt.figure(figsize=(8, 5))
        
        # Plot AUROC and AUPRC
        df_melted = pd.melt(
            results_df,
            id_vars=["setting"],
            value_vars=["auroc", "auprc"],
            var_name="Metric",
            value_name="Value"
        )
        
        sns.barplot(data=df_melted, x="setting", y="Value", hue="Metric", palette="Set1")
        plt.title("Performance Comparison: Static vs. Temporal Trajectories")
        plt.ylim(0.4, 1.0)
        plt.ylabel("Score")
        plt.xlabel("Feature Set Setting")
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("Temporal comparison plot saved to {}", output_path)


def build_longitudinal_dataset(
    labs_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    horizon_months: int,
    feature_names: list[str],
) -> pd.DataFrame:
    """Helper to slice labs and compute temporal features for a cohort in one call."""
    # Ensure dates are correct type
    cohort_df = cohort_df.copy()
    cohort_df["first_diag_date"] = pd.to_datetime(cohort_df["first_diag_date"])
    
    labs_df = labs_df.copy()
    labs_df["charttime"] = pd.to_datetime(labs_df["charttime"])
    
    # Merge cohorts and labs
    merged = labs_df.merge(
        cohort_df[["subject_id", "first_diag_date", "label", "cancer_type", "age", "gender"]],
        on="subject_id",
        how="inner"
    )
    
    # Observe window: [diag_date - 12m, diag_date - Hm]
    window_start = merged["first_diag_date"] - pd.Timedelta(days=365)
    window_end = merged["first_diag_date"] - pd.Timedelta(days=horizon_months * 30.4)
    
    filtered = merged[(merged["charttime"] >= window_start) & (merged["charttime"] <= window_end)].copy()
    
    # Format to long structure
    long_df = filtered[["subject_id", "charttime", "feature_name", "value"]]
    
    # Extract features
    extractor = TemporalFeatureExtractor(feature_names=feature_names)
    wide_features = extractor.fit_transform(long_df)
    
    # Join labels back
    final_df = cohort_df[["subject_id", "label", "cancer_type", "age", "gender"]].merge(
        wide_features, on="subject_id", how="inner"
    )
    
    return final_df
