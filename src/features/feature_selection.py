# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Feature selection, correlation filtering, and bootstrap stability analysis.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


def compute_feature_correlations(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Identify highly correlated features to drop, keeping one from each pair.

    Args:
        X: Feature DataFrame.
        threshold: Correlation coefficient threshold.

    Returns:
        List of feature names to drop.
    """
    meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
    numeric_cols = [c for c in X.columns if c not in meta_cols]
    
    corr_matrix = X[numeric_cols].corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    logger.info("Correlation analysis (threshold={}): dropping {}/{} collinear features", threshold, len(to_drop), len(numeric_cols))
    return to_drop


def select_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "xgboost",
    top_k: int = 50,
    seed: int = 42,
) -> list[str]:
    """Select the top K features using model importance or statistical metrics.

    Args:
        X_train: Training features.
        y_train: Training labels.
        method: 'xgboost', 'permutation', or 'mutual_info'.
        top_k: Number of features to return.
        seed: Random seed.

    Returns:
        List of selected feature names.
    """
    meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
    numeric_cols = [c for c in X_train.columns if c not in meta_cols]
    
    # Impute missing values for tree fitting
    X_numeric = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    
    if method == "xgboost":
        # Fit XGBoost classifier
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_numeric, y_train)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"feature": numeric_cols, "score": importances})
        
    elif method == "permutation":
        model = XGBClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_numeric, y_train)
        r = permutation_importance(model, X_numeric, y_train, n_repeats=5, random_state=seed)
        importance_df = pd.DataFrame({"feature": numeric_cols, "score": r.importances_mean})
        
    elif method == "mutual_info":
        scores = mutual_info_classif(X_numeric, y_train, random_state=seed)
        importance_df = pd.DataFrame({"feature": numeric_cols, "score": scores})
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
        
    # Sort and take top K
    importance_df = importance_df.sort_values(by="score", ascending=False)
    selected = importance_df["feature"].head(top_k).tolist()
    
    logger.info("Feature selection via {}: selected top {} features", method, len(selected))
    return selected


def analyze_feature_importance_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_bootstrap: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Assess the stability of feature importances across bootstrap resamples.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        DataFrame summarizing stability metrics for all features.
    """
    np.random.seed(seed)
    meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
    numeric_cols = [c for c in X_train.columns if c not in meta_cols]
    
    importances_list = []
    
    for i in range(n_bootstrap):
        # Sample indices with replacement
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[indices]
        y_boot = y_train.iloc[indices]
        
        # Impute
        X_boot_numeric = X_boot[numeric_cols].fillna(X_boot[numeric_cols].median())
        
        # Fit model
        model = XGBClassifier(n_estimators=50, max_depth=4, random_state=seed + i, n_jobs=-1)
        model.fit(X_boot_numeric, y_boot)
        
        importances_list.append(model.feature_importances_)
        
    importances_matrix = np.array(importances_list)
    mean_imp = np.mean(importances_matrix, axis=0)
    std_imp = np.std(importances_matrix, axis=0)
    
    # Ranks
    ranks_list = []
    for row in importances_matrix:
        # Rank features descending (high importance -> rank 1)
        ranks_list.append(len(numeric_cols) - np.argsort(np.argsort(row)))
        
    ranks_matrix = np.array(ranks_list)
    mean_rank = np.mean(ranks_matrix, axis=0)
    std_rank = np.std(ranks_matrix, axis=0)
    
    stability_df = pd.DataFrame({
        "feature": numeric_cols,
        "mean_importance": mean_imp,
        "std_importance": std_imp,
        "rank_mean": mean_rank,
        "rank_std": std_rank
    }).sort_values(by="mean_importance", ascending=False)
    
    return stability_df


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str, top_n: int = 30) -> None:
    """Plot feature importances color-coded by clinical category.

    Categories: CBC, Metabolic, Inflammatory, Derived, Temporal.

    Args:
        importance_df: DataFrame containing 'feature' and 'mean_importance' or 'score'.
        output_path: Path to save the plot image.
        top_n: Number of top features to plot.
    """
    df = importance_df.copy()
    if "mean_importance" in df.columns:
        df.rename(columns={"mean_importance": "score"}, inplace=True)
        
    df = df.sort_values(by="score", ascending=False).head(top_n)
    
    # Assign categories based on name patterns
    def get_category(feat):
        feat = feat.lower()
        # Derived check
        if feat.startswith(("nlr", "plr", "sii", "agr", "de_ritis", "fib4")):
            return "Derived Biomarkers"
        # Check standard lists
        from src.agents.base_agent import _LAB_CATEGORIES
        
        # Strip temporal suffixes to identify base feature
        base_feat = feat
        for suffix in ["_mean", "_median", "_std", "_trend_slope", "_delta", "_velocity", "_min", "_max", "_range", "_cv", "_last_value", "_first_value", "_moving_avg_last", "_exp_smooth_last", "_n_measurements", "_n_abnormal"]:
            if feat.endswith(suffix):
                base_feat = feat.replace(suffix, "")
                break
                
        for cat, list_feats in _LAB_CATEGORIES.items():
            if base_feat in list_feats:
                # If it has a temporal suffix, mark as Temporal category or keep clinical category
                if base_feat != feat:
                    return f"Temporal ({cat})"
                return cat
                
        if "_trend_slope" in feat or "_velocity" in feat or "_delta" in feat or "_exp_smooth" in feat:
            return "Temporal Trajectories"
            
        return "Other"

    df["category"] = df["feature"].apply(get_category)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df,
        y="feature",
        x="score",
        hue="category",
        palette="Set2",
        dodge=False
    )
    
    plt.title(f"Top {top_n} Features by ML Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.legend(title="Clinical Category", loc="lower right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Feature importance plot saved to {}", output_path)
