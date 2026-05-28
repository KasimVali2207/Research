# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the early cancer triage preprocessing, features, and models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lab_features import compute_derived_biomarkers, flag_clinical_alerts
from src.preprocessing.cohort_matching import create_matching_strata, match_controls, compute_smd
from src.models.calibration import compute_calibration_metrics


def test_derived_biomarkers():
    """Verify immune-inflammation indices calculations (NLR, PLR, SII, FIB-4)."""
    df = pd.DataFrame({
        "neutrophils": [4.0, 8.0],
        "lymphocytes": [2.0, 1.0],
        "platelets": [200.0, 400.0],
        "albumin": [4.0, 3.0],
        "total_protein": [7.0, 7.0],
        "ast": [20.0, 40.0],
        "alt": [20.0, 80.0],
        "age": [50, 60]
    })
    
    out = compute_derived_biomarkers(df)
    
    # Check NLR = neutrophils / lymphocytes
    assert out.loc[0, "nlr"] == 2.0
    assert out.loc[1, "nlr"] == 8.0
    
    # Check PLR = platelets / lymphocytes
    assert out.loc[0, "plr"] == 100.0
    assert out.loc[1, "plr"] == 400.0
    
    # Check SII = (platelets * neutrophils) / lymphocytes
    assert out.loc[0, "sii"] == 400.0
    assert out.loc[1, "sii"] == 3200.0
    
    # Check De Ritis = AST / ALT
    assert out.loc[0, "de_ritis_ratio"] == 1.0
    assert out.loc[1, "de_ritis_ratio"] == 0.5


def test_clinical_alert_flags():
    """Verify clinical alert flagging thresholds and ordinal values."""
    df = pd.DataFrame({
        "hemoglobin": [14.0, 6.5, 10.0],  # normal, critically low, low
        "wbc": [7.0, 35.0, 1.0],         # normal, critically high, critically low
    })
    
    normal_ranges = {
        "hemoglobin": {"low": 12.0, "high": 17.5},
        "wbc": {"low": 4.0, "high": 11.0}
    }
    
    out = flag_clinical_alerts(df, normal_ranges)
    
    # Hemoglobin flags
    assert out.loc[0, "hemoglobin_flag"] == 0   # Normal
    assert out.loc[1, "hemoglobin_flag"] == -2  # Critical low (<7.0)
    assert out.loc[2, "hemoglobin_flag"] == -1  # Low (<12.0)
    
    # WBC flags
    assert out.loc[0, "wbc_flag"] == 0   # Normal
    assert out.loc[1, "wbc_flag"] == 2   # Critical high (>30.0)
    assert out.loc[2, "wbc_flag"] == -2  # Critical low (<1.5)


def test_cohort_matching():
    """Verify propensity-like matching and Standardized Mean Difference balance checks."""
    np.random.seed(42)
    
    # Generate cases
    cases = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "age": [52, 63, 71],
        "gender": ["M", "F", "M"],
        "admission_year": [2150, 2151, 2150]
    })
    
    # Generate large control pool
    controls = pd.DataFrame({
        "subject_id": list(range(10, 210)),
        "age": np.random.randint(40, 80, size=200),
        "gender": np.random.choice(["M", "F"], size=200),
        "admission_year": np.random.choice([2150, 2151, 2152], size=200)
    })
    
    matched_cases, matched_controls = match_controls(cases, controls, ratio=2, seed=42)
    
    # Check 1:2 ratio
    assert len(matched_cases) == 3
    assert len(matched_controls) == 6
    
    # Check SMD calculation
    covariates = ["age"]
    smd_df = compute_smd(matched_cases, matched_controls, covariates)
    assert "smd" in smd_df.columns
    assert len(smd_df) == 1


def test_calibration_metrics():
    """Verify ECE and Brier score calculations."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.05])
    
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    
    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier" in metrics
    
    # Perfect calibration check
    assert metrics["ece"] < 0.15
    assert metrics["brier"] < 0.1
