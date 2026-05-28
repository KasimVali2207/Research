# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Cohort case-control matching module.

Performs 1:N stratified case-control matching using demographic strata.
Measures match balance using Standardized Mean Difference (SMD) and
visualizes balance using a Love plot.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def create_matching_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Create stratum identifiers based on age, sex, and admission year.

    Age is grouped into 10-year bins.
    Admission year is grouped into 5-year bins.

    Args:
        df: DataFrame containing at least ['age', 'gender', 'admission_year'].

    Returns:
        df with 'stratum' column and intermediate bin columns.
    """
    df = df.copy()
    
    # 10-year age decade bins (e.g. 50-59, 60-69)
    df["age_decade"] = (df["age"] // 10) * 10
    
    # 5-year admission year bins
    df["year_bin"] = (df["admission_year"] // 5) * 5
    
    # Format stratum string
    df["stratum"] = (
        df["age_decade"].astype(str)
        + "_"
        + df["gender"].astype(str)
        + "_"
        + df["year_bin"].astype(str)
    )
    
    # Also define relaxed stratum (only age_decade + gender)
    df["stratum_relaxed"] = df["age_decade"].astype(str) + "_" + df["gender"].astype(str)
    
    return df


def match_controls(
    cancer_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    ratio: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each cancer case to `ratio` controls based on strata.

    Tries exact stratum matching first, then falls back to relaxed stratum
    (age_decade + gender) if the stratum is exhausted.

    Args:
        cancer_df: Case DataFrame.
        control_pool_df: Pool of possible control patients.
        ratio: Number of controls per case.
        seed: Random seed.

    Returns:
        Tuple of (matched_cancer_df, matched_control_df).
    """
    np.random.seed(seed)
    
    # Add strata columns if not present
    if "stratum" not in cancer_df.columns:
        cancer_df = create_matching_strata(cancer_df)
    if "stratum" not in control_pool_df.columns:
        control_pool_df = create_matching_strata(control_pool_df)

    matched_controls_list = []
    matched_cases_list = []
    
    used_control_ids = set()
    
    # Pre-group controls for fast lookup
    control_by_stratum = {
        name: group.copy() for name, group in control_pool_df.groupby("stratum")
    }
    control_by_relaxed = {
        name: group.copy() for name, group in control_pool_df.groupby("stratum_relaxed")
    }

    exact_matches = 0
    relaxed_matches = 0
    failed_matches = 0

    for idx, case in cancer_df.iterrows():
        case_id = case["subject_id"]
        stratum = case["stratum"]
        stratum_relaxed = case["stratum_relaxed"]
        
        # 1. Try exact matching
        candidates = control_by_stratum.get(stratum, pd.DataFrame())
        # Filter out already used controls
        candidates = candidates[~candidates["subject_id"].isin(used_control_ids)]
        
        selected_controls = pd.DataFrame()
        match_type = "exact"
        
        if len(candidates) >= ratio:
            selected_controls = candidates.sample(n=ratio, random_state=seed + idx)
            exact_matches += 1
        else:
            # 2. Try relaxed matching
            candidates_relaxed = control_by_relaxed.get(stratum_relaxed, pd.DataFrame())
            candidates_relaxed = candidates_relaxed[~candidates_relaxed["subject_id"].isin(used_control_ids)]
            
            if len(candidates_relaxed) >= ratio:
                selected_controls = candidates_relaxed.sample(n=ratio, random_state=seed + idx)
                relaxed_matches += 1
                match_type = "relaxed"
            else:
                # 3. Exhausted relaxed candidates as well, grab closest available or skip
                # (Grab whatever is left in the relaxed stratum, and pad if necessary)
                combined = pd.concat([candidates, candidates_relaxed]).drop_duplicates(subset=["subject_id"])
                combined = combined[~combined["subject_id"].isin(used_control_ids)]
                if len(combined) > 0:
                    take_n = min(len(combined), ratio)
                    selected_controls = combined.sample(n=take_n, random_state=seed + idx)
                    relaxed_matches += 1
                    match_type = "partial"
                else:
                    failed_matches += 1
                    match_type = "failed"
                    
        if len(selected_controls) > 0:
            # Assign match ID linking case and control
            match_id = f"match_{case_id}"
            
            case_entry = case.copy()
            case_entry["match_id"] = match_id
            case_entry["match_type"] = match_type
            matched_cases_list.append(case_entry.to_frame().T)
            
            selected_controls = selected_controls.copy()
            selected_controls["match_id"] = match_id
            selected_controls["match_type"] = match_type
            selected_controls["matched_case_id"] = case_id
            matched_controls_list.append(selected_controls)
            
            used_control_ids.update(selected_controls["subject_id"].tolist())

    if matched_cases_list:
        matched_cases_df = pd.concat(matched_cases_list, ignore_index=True)
    else:
        matched_cases_df = pd.DataFrame(columns=cancer_df.columns.tolist() + ["match_id", "match_type"])

    if matched_controls_list:
        matched_controls_df = pd.concat(matched_controls_list, ignore_index=True)
    else:
        matched_controls_df = pd.DataFrame(columns=control_pool_df.columns.tolist() + ["match_id", "match_type", "matched_case_id"])

    total_cases = len(cancer_df)
    logger.info(
        "Strata matching complete: Exact matched={} ({:.1f}%), Relaxed/Partial matched={} ({:.1f}%), Failed={} ({:.1f}%)",
        exact_matches,
        exact_matches / total_cases * 100,
        relaxed_matches,
        relaxed_matches / total_cases * 100,
        failed_matches,
        failed_matches / total_cases * 100,
    )
    
    return matched_cases_df, matched_controls_df


def compute_smd(
    cancer_df: pd.DataFrame,
    control_df: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """Calculate the Standardized Mean Difference (SMD) for matched cohorts.

    SMD = (Mean_cases - Mean_controls) / sqrt((Var_cases + Var_controls) / 2)

    Args:
        cancer_df: Matched cases.
        control_df: Matched controls.
        covariates: List of covariates (numeric or boolean columns).

    Returns:
        DataFrame containing SMD values per covariate.
    """
    smd_records = []
    
    for cov in covariates:
        if cov not in cancer_df.columns or cov not in control_df.columns:
            logger.warning("Covariate '{}' not found in cohorts, skipping SMD calculation.", cov)
            continue
            
        case_vals = cancer_df[cov].dropna().astype(float)
        ctrl_vals = control_df[cov].dropna().astype(float)
        
        mean_case = case_vals.mean()
        mean_ctrl = ctrl_vals.mean()
        
        var_case = case_vals.var()
        var_ctrl = ctrl_vals.var()
        
        # Prevent division by zero
        denom = np.sqrt((var_case + var_ctrl) / 2.0)
        if denom == 0:
            smd = 0.0
        else:
            smd = (mean_case - mean_ctrl) / denom
            
        smd_records.append(
            {
                "covariate": cov,
                "mean_cases": mean_case,
                "mean_controls": mean_ctrl,
                "std_cases": np.sqrt(var_case),
                "std_controls": np.sqrt(var_ctrl),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )
        
    return pd.DataFrame(smd_records)


def assess_matching_quality(
    cancer_df: pd.DataFrame,
    control_df: pd.DataFrame,
    covariates: list[str] | None = None,
) -> dict:
    """Assess cohort match quality and return summary stats.

    Args:
        cancer_df: Case DataFrame.
        control_df: Control DataFrame.
        covariates: Optional list of columns to assess.

    Returns:
        Summary dictionary.
    """
    if covariates is None:
        covariates = ["age", "gender_encoded", "admission_year"]
        
    # Helper to encode gender if needed
    for df in (cancer_df, control_df):
        if "gender_encoded" not in df.columns and "gender" in df.columns:
            df["gender_encoded"] = df["gender"].map({"M": 1, "F": 0, "m": 1, "f": 0}).fillna(0)
            
    smd_df = compute_smd(cancer_df, control_df, covariates)
    
    max_smd = smd_df["abs_smd"].max()
    unbalanced_covs = smd_df[smd_df["abs_smd"] >= 0.1]["covariate"].tolist()
    
    balanced = len(unbalanced_covs) == 0
    
    summary = {
        "smd_table": smd_df.to_dict(orient="records"),
        "max_abs_smd": float(max_smd) if not pd.isna(max_smd) else 0.0,
        "unbalanced_covariates": unbalanced_covs,
        "is_well_balanced": bool(balanced),
        "match_rate": len(cancer_df) / max(len(cancer_df) + len(control_df), 1),
    }
    
    return summary


def plot_matching_balance(smd_df: pd.DataFrame, output_path: str) -> None:
    """Create and save a Love Plot showing covariate balance.

    Args:
        smd_df: DataFrame output from compute_smd.
        output_path: Path to save the plot image.
    """
    plt.figure(figsize=(8, 6))
    
    # Sort by absolute SMD for readability
    smd_sorted = smd_df.sort_values(by="abs_smd", ascending=True)
    
    y_pos = np.arange(len(smd_sorted))
    plt.hlines(y_pos, 0, smd_sorted["abs_smd"], colors="skyblue", linewidth=2)
    plt.plot(smd_sorted["abs_smd"], y_pos, "o", color="blue", markersize=8)
    
    plt.yticks(y_pos, smd_sorted["covariate"])
    plt.xlabel("Absolute Standardized Mean Difference (SMD)")
    plt.title("Covariate Balance (Love Plot)")
    
    # Add vertical reference lines at 0.05 and 0.10
    plt.axvline(x=0.1, color="red", linestyle="--", alpha=0.7, label="Negligible Threshold (0.10)")
    plt.axvline(x=0.05, color="orange", linestyle=":", alpha=0.7, label="Strict Threshold (0.05)")
    plt.axvline(x=0.0, color="gray", linestyle="-", alpha=0.5)
    
    plt.xlim(-0.02, max(0.2, smd_sorted["abs_smd"].max() * 1.2))
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Love plot saved to {}", output_path)
