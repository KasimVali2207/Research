# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lab feature extraction, normalization, and derived biomarker computation.

Derived biomarkers (NLR, PLR, SII, FIB-4, De Ritis ratio) are among the
strongest routine-blood-only signals for cancer triage — hence the central role
of this module in the feature pipeline.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.lab_itemids import NORMAL_RANGES


# ---------------------------------------------------------------------------
# LabFeatureExtractor
# ---------------------------------------------------------------------------

class LabFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that:
      1. Median-imputes missing labs (fit on train, apply to test — no leakage).
      2. Clips extreme outliers at 1st / 99th percentile (learned from fit).
      3. Adds {feature}_missing boolean indicator columns.
      4. Adds {feature}_abnormal_low / _abnormal_high boolean columns.
    """

    def __init__(
        self,
        feature_config: dict,
        normal_ranges: Optional[dict] = None,
    ) -> None:
        self.feature_config = feature_config
        self.normal_ranges = normal_ranges or NORMAL_RANGES

        # Populated during fit()
        self._medians: dict[str, float] = {}
        self._clip_low: dict[str, float] = {}
        self._clip_high: dict[str, float] = {}
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None) -> "LabFeatureExtractor":
        all_features = (
            self.feature_config.get("cbc", [])
            + self.feature_config.get("metabolic", [])
            + self.feature_config.get("inflammatory", [])
        )
        self._feature_cols = [f for f in all_features if f in df.columns]

        for feat in self._feature_cols:
            col = df[feat].replace([np.inf, -np.inf], np.nan)
            self._medians[feat] = col.median()
            self._clip_low[feat] = col.quantile(0.01)
            self._clip_high[feat] = col.quantile(0.99)

        logger.info(
            f"LabFeatureExtractor fitted on {len(self._feature_cols)} features, "
            f"{df.shape[0]} samples."
        )
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # If no raw lab columns were found during fit, pass through as-is
        if not self._medians:
            return df.copy()

        out = df.copy()

        for feat in self._feature_cols:
            if feat not in out.columns:
                out[feat] = np.nan

            # Missing indicator BEFORE imputation (ground truth of missingness)
            out[f"{feat}_missing"] = out[feat].isna().astype(np.int8)

            # Clip and impute
            out[feat] = out[feat].replace([np.inf, -np.inf], np.nan)
            out[feat] = out[feat].clip(
                lower=self._clip_low[feat], upper=self._clip_high[feat]
            )
            out[feat] = out[feat].fillna(self._medians[feat])

            # Abnormal flags (use normal ranges from clinical reference)
            if feat in self.normal_ranges:
                lo = self.normal_ranges[feat]["low"]
                hi = self.normal_ranges[feat]["high"]
                out[f"{feat}_abnormal_low"] = (out[feat] < lo).astype(np.int8)
                out[f"{feat}_abnormal_high"] = (out[feat] > hi).astype(np.int8)

        return out

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    # ------------------------------------------------------------------
    def get_missingness_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns per-feature missingness statistics."""
        rows = []
        for feat in self._feature_cols:
            pct = df[feat].isna().mean() * 100 if feat in df.columns else 100.0
            rows.append(
                {
                    "feature": feat,
                    "pct_missing": round(pct, 2),
                    "imputation_value": self._medians.get(feat, np.nan),
                }
            )
        return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


# ---------------------------------------------------------------------------
# Derived Biomarkers
# ---------------------------------------------------------------------------

def compute_derived_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes clinically established derived biomarkers from raw labs.

    NLR, PLR, SII are independently validated predictors of cancer prognosis
    and early detection (Templeton et al., Lancet Oncol 2014; Chen et al., 2015).
    FIB-4 is a non-invasive liver fibrosis index with strong hepatocellular
    carcinoma prediction utility.
    """
    out = df.copy()

    # Guard against zero denominators — real labs are never exactly zero,
    # but imputed data or artifacts may produce them.
    eps = 1e-6

    # --- Immune-Inflammation Indices ---
    if {"neutrophils", "lymphocytes"}.issubset(out.columns):
        denom = out["lymphocytes"].clip(lower=eps)
        out["nlr"] = (out["neutrophils"] / denom).clip(upper=50.0)

    if {"platelets", "lymphocytes"}.issubset(out.columns):
        denom = out["lymphocytes"].clip(lower=eps)
        out["plr"] = (out["platelets"] / denom).clip(upper=2000.0)

    if {"platelets", "neutrophils", "lymphocytes"}.issubset(out.columns):
        denom = out["lymphocytes"].clip(lower=eps)
        out["sii"] = (out["platelets"] * out["neutrophils"] / denom).clip(
            upper=1e5
        )

    # --- Protein Indices ---
    if {"albumin", "total_protein"}.issubset(out.columns):
        globulin = (out["total_protein"] - out["albumin"]).clip(lower=eps)
        out["agr"] = (out["albumin"] / globulin).clip(upper=10.0)

    # --- Liver Indices ---
    if {"ast", "alt"}.issubset(out.columns):
        denom = out["alt"].clip(lower=eps)
        out["de_ritis_ratio"] = (out["ast"] / denom).clip(upper=20.0)

    # FIB-4: validated for HCC risk stratification (Sterling et al., Hepatology 2006)
    # FIB-4 = (age × AST) / (platelets × √ALT)
    required_fib4 = {"age", "ast", "platelets", "alt"}
    if required_fib4.issubset(out.columns):
        sqrt_alt = np.sqrt(out["alt"].clip(lower=eps))
        denom = (out["platelets"].clip(lower=eps) * sqrt_alt).clip(lower=eps)
        out["fib4"] = ((out["age"] * out["ast"]) / denom).clip(upper=20.0)

    new_cols = [c for c in out.columns if c not in df.columns]
    logger.debug(f"Derived biomarkers computed: {new_cols}")
    return out


# ---------------------------------------------------------------------------
# Clinical Alert Flags
# ---------------------------------------------------------------------------

# Critical thresholds beyond normal range — not just abnormal but alarm-level.
# These are commonly used in clinical decision support systems.
_CRITICAL_LOW: dict[str, float] = {
    "hemoglobin": 7.0,       # g/dL — transfusion trigger
    "platelets": 50.0,       # K/μL — bleeding risk
    "wbc": 1.5,              # K/μL — severe leukopenia
    "sodium": 120.0,         # mEq/L — severe hyponatremia
    "potassium": 2.5,        # mEq/L — severe hypokalemia
    "albumin": 2.0,          # g/dL — severe hypoalbuminemia
    "glucose": 50.0,         # mg/dL — hypoglycemia
}
_CRITICAL_HIGH: dict[str, float] = {
    "wbc": 30.0,             # K/μL — leukocytosis / leukemia concern
    "platelets": 1000.0,     # K/μL — thrombocytosis
    "potassium": 6.0,        # mEq/L — severe hyperkalemia
    "sodium": 155.0,         # mEq/L — severe hypernatremia
    "bilirubin_total": 10.0, # mg/dL — severe jaundice
    "creatinine": 8.0,       # mg/dL — severe renal failure
    "glucose": 500.0,        # mg/dL — hyperglycemic crisis
    "alt": 500.0,            # U/L — severe hepatocellular injury
    "ast": 500.0,
}


def flag_clinical_alerts(
    df: pd.DataFrame, normal_ranges: Optional[dict] = None
) -> pd.DataFrame:
    """
    Encodes lab values as ordinal severity flags.

    Flag encoding:
      -2: critically low   (below critical low threshold)
      -1: low              (below normal low but above critical low)
       0: normal
       1: high             (above normal high but below critical high)
       2: critically high  (above critical high threshold)

    These flags are clinical features in their own right — a patient with
    hemoglobin flag = -2 is qualitatively different from flag = -1.
    """
    nr = normal_ranges or NORMAL_RANGES
    out = df.copy()

    for feat in nr:
        if feat not in out.columns:
            continue

        lo_norm = nr[feat]["low"]
        hi_norm = nr[feat]["high"]
        lo_crit = _CRITICAL_LOW.get(feat, lo_norm - abs(lo_norm))  # fallback: very far below
        hi_crit = _CRITICAL_HIGH.get(feat, hi_norm + abs(hi_norm))

        flags = pd.Series(0, index=out.index, dtype=np.int8)
        flags = flags.where(out[feat] >= lo_norm, -1)   # low
        flags = flags.where(out[feat] > lo_crit, -2)    # critically low (overrides -1)
        flags = flags.where(out[feat] <= hi_norm, 1)    # high (only if still 0)
        # Re-check: critically high
        flags[out[feat] > hi_crit] = 2

        # Recombine: normal if within range
        in_range = (out[feat] >= lo_norm) & (out[feat] <= hi_norm)
        flags[in_range] = 0

        # Missing values get 0 (unknown, not normal — imputation already done)
        out[f"{feat}_flag"] = flags.astype(np.int8)

    return out
