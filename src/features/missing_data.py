# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Missing data analysis, simulation, and imputation module.

Used to test model robustness against clinical data sparsity and missingness
mechanisms (MCAR, MAR, MNAR) according to Experiment 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score


def compute_missingness_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missingness correlations and return correlation matrix.

    A correlation coefficient near 1.0 indicates that if lab A is missing,
    lab B is almost certainly missing as well (e.g., CBC panel components).

    Args:
        df: Feature DataFrame.

    Returns:
        DataFrame containing correlation matrix of missingness indicators.
    """
    indicator_df = df.isna().astype(float)
    # Filter out columns with zero missingness variance
    variances = indicator_df.var()
    cols_to_keep = variances[variances > 0.0].index
    
    if len(cols_to_keep) < 2:
        logger.warning("Fewer than 2 features have missingness variance. Correlation matrix is empty.")
        return pd.DataFrame()
        
    corr_matrix = indicator_df[cols_to_keep].corr()
    return corr_matrix


def multiple_imputation(df: pd.DataFrame, n_imputations: int = 5, seed: int = 42) -> list[pd.DataFrame]:
    """Perform MICE multiple imputation using Scikit-Learn's IterativeImputer.

    Args:
        df: Wide features DataFrame.
        n_imputations: Number of imputed datasets to return.
        seed: Random seed.

    Returns:
        List of n_imputations imputed DataFrames.
    """
    # Exclude metadata columns
    meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
    present_meta = {c: df[c] for c in meta_cols if c in df.columns}
    numeric_df = df.drop(columns=list(present_meta.keys()), errors="ignore")
    
    imputed_datasets = []
    
    for i in range(n_imputations):
        logger.info("Running MICE imputation {}/{}...", i + 1, n_imputations)
        imputer = IterativeImputer(
            max_iter=10,
            random_state=seed + i,
            n_nearest_features=10,
            initial_strategy="median"
        )
        
        imputed_array = imputer.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed_array, columns=numeric_df.columns, index=df.index)
        
        # Add metadata columns back
        for col, s in present_meta.items():
            imputed_df[col] = s.values
            
        imputed_datasets.append(imputed_df)
        
    return imputed_datasets


def simulate_missing_data(
    df: pd.DataFrame,
    missing_rate: float,
    mechanism: str = "MCAR",
    seed: int = 42,
) -> pd.DataFrame:
    """Artificially introduce missingness into a complete-case or baseline dataset.

    Mechanisms:
      - MCAR (Missing Completely at Random): Drop values uniformly at random.
      - MAR (Missing at Random): Drop value based on an observed covariate (e.g. elderly
        patients have fewer tests ordered because they are in hospice).
      - MNAR (Missing Not at Random): Drop values based on the value itself (e.g. normal
        values are omitted / not recorded, while critical highs are always recorded).

    Args:
        df: Input DataFrame.
        missing_rate: Proportion of values to mask (0.0 to 1.0).
        mechanism: 'MCAR', 'MAR', or 'MNAR'.
        seed: Random seed.

    Returns:
        DataFrame with simulated missing values (NaNs).
    """
    np.random.seed(seed)
    out = df.copy()
    
    meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
    numeric_cols = [c for c in out.columns if c not in meta_cols]
    
    if mechanism == "MCAR":
        # Random uniform masking
        mask = np.random.rand(*out[numeric_cols].shape) < missing_rate
        out[numeric_cols] = out[numeric_cols].mask(mask)
        
    elif mechanism == "MAR":
        # Missingness probability correlates with age
        if "age" in out.columns:
            age_norm = (out["age"] - out["age"].min()) / (out["age"].max() - out["age"].min() + 1e-6)
            for col in numeric_cols:
                # Probability of missingness is proportional to age
                prob = age_norm * missing_rate * 2.0  # Scale to average out to missing_rate
                prob = np.clip(prob, 0.0, 0.95)
                mask = np.random.rand(len(out)) < prob
                out[col] = out[col].mask(mask)
        else:
            # Fallback to MCAR
            logger.warning("Age covariate missing for MAR, falling back to MCAR.")
            return simulate_missing_data(df, missing_rate, "MCAR", seed)
            
    elif mechanism == "MNAR":
        # Values closer to normal are more likely to be missing (undocumented)
        # while extreme highs/lows are recorded
        for col in numeric_cols:
            vals = out[col].dropna()
            if len(vals) == 0:
                continue
            median = vals.median()
            std = vals.std() + 1e-6
            # Distance from center
            dist = np.abs(out[col] - median) / std
            # Higher distance -> lower probability of missingness
            # prob = exp(-dist) scaled
            prob = np.exp(-dist) * missing_rate * 2.0
            prob = np.clip(prob, 0.0, 0.95)
            mask = np.random.rand(len(out)) < prob
            out[col] = out[col].mask(mask)
            
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")
        
    return out


def missingness_robustness_experiment(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    rates: list[float] | None = None,
    n_trials: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Run simulated missingness robustness experiment for a trained model.

    Evaluates the model's performance decay as data becomes sparser.

    Args:
        model: Trained classifier (with predict_proba method).
        X_test: Test features.
        y_test: Test labels.
        rates: Missing rates to simulate.
        n_trials: Number of trials per rate to calculate variance.
        seed: Random seed.

    Returns:
        DataFrame containing results table.
    """
    if rates is None:
        rates = [0.1, 0.2, 0.4]
        
    results = []
    
    # Exclude metadata columns for model prediction
    meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
    numeric_cols = [c for c in X_test.columns if c not in meta_cols]
    
    # Compute baseline score (0% missingness)
    # Fit simple median imputer for test set if model doesn't support NaNs
    baseline_X = X_test[numeric_cols].fillna(X_test[numeric_cols].median())
    y_prob = model.predict_proba(baseline_X)[:, 1]
    baseline_auroc = roc_auc_score(y_test, y_prob)
    results.append({
        "missing_rate": 0.0,
        "mechanism": "None",
        "auroc_mean": baseline_auroc,
        "auroc_std": 0.0
    })
    
    for r in rates:
        for mech in ["MCAR", "MAR", "MNAR"]:
            trial_scores = []
            for t in range(n_trials):
                X_sim = simulate_missing_data(X_test, r, mech, seed=seed + t)
                # Median impute
                X_imputed = X_sim[numeric_cols].fillna(X_sim[numeric_cols].median())
                try:
                    y_prob = model.predict_proba(X_imputed)[:, 1]
                    if len(np.unique(y_test)) < 2:
                        auroc = 0.5
                    else:
                        auroc = roc_auc_score(y_test, y_prob)
                except Exception:
                    auroc = 0.5
                trial_scores.append(auroc)
                
            results.append({
                "missing_rate": r,
                "mechanism": mech,
                "auroc_mean": float(np.mean(trial_scores)),
                "auroc_std": float(np.std(trial_scores))
            })
            
            logger.info(
                "Missingness robustness (rate={}, mech={}): AUROC = {:.4f} ± {:.4f}",
                r,
                mech,
                np.mean(trial_scores),
                np.std(trial_scores),
            )
            
    return pd.DataFrame(results)
