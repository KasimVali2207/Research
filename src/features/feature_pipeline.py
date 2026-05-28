# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Feature engineering pipeline module.

Chains the different extraction steps (temporal stats, imputation,
abnormal flagging, derived biomarker computations) into an sklearn-compatible pipeline.
"""

from __future__ import annotations

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from loguru import logger

from src.features.lab_features import LabFeatureExtractor, compute_derived_biomarkers, flag_clinical_alerts
from src.preprocessing.lab_itemids import NORMAL_RANGES


class DerivedBiomarkerComputer(BaseEstimator, TransformerMixin):
    """Transformer wrapper around compute_derived_biomarkers and flag_clinical_alerts."""

    def __init__(self, normal_ranges: dict | None = None) -> None:
        self.normal_ranges = normal_ranges or NORMAL_RANGES

    def fit(self, X: pd.DataFrame, y=None) -> DerivedBiomarkerComputer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = compute_derived_biomarkers(X)
        df = flag_clinical_alerts(df, self.normal_ranges)
        return df


class CancerTriageFeaturePipeline:
    """Standardized pipeline for processing early-detection feature matrices."""

    def __init__(self, cfg: dict, use_temporal: bool = True, seed: int = 42) -> None:
        self.cfg = cfg
        self.use_temporal = use_temporal
        self.seed = seed
        
        # Initialize sub-transformers
        # Imputer & outlier clipper
        self.lab_extractor = LabFeatureExtractor(
            feature_config=cfg.get("features", {}),
            normal_ranges=NORMAL_RANGES
        )
        self.derived_computer = DerivedBiomarkerComputer(normal_ranges=NORMAL_RANGES)
        self.scaler = StandardScaler()
        self.selector = VarianceThreshold(threshold=0.0)
        
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> CancerTriageFeaturePipeline:
        """Fit all steps of the feature pipeline."""
        df = X.copy()
        
        # 1. Fit lab extractor (learns medians & quantiles)
        df = self.lab_extractor.fit_transform(df, y)
        
        # 2. Add derived features and abnormal flags
        df = self.derived_computer.fit_transform(df, y)
        
        # Exclude metadata columns before scaling
        meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
        numeric_cols = [c for c in df.columns if c not in meta_cols]
        
        # 3. Fit scaler
        self.scaler.fit(df[numeric_cols])
        scaled_data = self.scaler.transform(df[numeric_cols])
        
        # 4. Fit variance threshold selector
        self.selector.fit(scaled_data)
        
        # Save feature name lists
        selected_indices = self.selector.get_support()
        self.feature_names_ = [numeric_cols[i] for i, val in enumerate(selected_indices) if val]
        
        logger.info("CancerTriageFeaturePipeline fitted. Selected {} features.", len(self.feature_names_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame through the pipeline."""
        df = X.copy()
        
        # Keep metadata columns to carry along
        meta_cols = ["subject_id", "label", "cancer_type", "gender", "age"]
        present_meta = {c: df[c] for c in meta_cols if c in df.columns}
        
        # Transform steps
        df = self.lab_extractor.transform(df)
        df = self.derived_computer.transform(df)
        
        # Isolate numeric columns
        numeric_cols = [c for c in df.columns if c not in meta_cols]
        
        # Scale and select
        scaled = self.scaler.transform(df[numeric_cols])
        selected = self.selector.transform(scaled)
        
        # Re-build DataFrame
        out_df = pd.DataFrame(selected, columns=self.feature_names_, index=df.index)
        for c, s in present_meta.items():
            out_df[c] = s.values
            
        return out_df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

    def save(self, path: str) -> None:
        """Serialize pipeline to a pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved feature pipeline to {}", path)

    @classmethod
    def load(cls, path: str) -> CancerTriageFeaturePipeline:
        """Load serialized pipeline from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded feature pipeline from {}", path)
        return obj


def load_horizon_data(
    processed_dir: str,
    horizon_months: int,
    cancer_type: str | None = None,
    dataset: str = "mimic",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load horizon features parquet and optionally filter to a specific cancer type.

    Args:
        processed_dir: Path to directory containing processed parquets.
        horizon_months: The study horizon (3, 6, or 12).
        cancer_type: Type of cancer (colorectal, lung, liver) or None for all.
        dataset: The dataset name ('mimic' or 'eicu').

    Returns:
        Tuple of (X_df, y_series)
    """
    file_name = f"features_horizon_{horizon_months}m_{dataset}.parquet"
    file_path = os.path.join(processed_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found at {file_path}")
        
    df = pd.read_parquet(file_path)
    
    if cancer_type is not None:
        # Keep cases matching cancer_type, and controls matched to those cases
        # Check matching columns first
        if "cancer_type" in df.columns:
            # For mimic controls, we need to map matched_case_id's cancer type
            # Find subject_ids of cases of this cancer type
            case_subset = df[(df["label"] == 1) & (df["cancer_type"] == cancer_type)]
            case_subject_ids = set(case_subset["subject_id"])
            
            # Read cohorts to map controls
            control_cohort_path = os.path.join(processed_dir, f"{dataset}_control_cohort.parquet")
            if os.path.exists(control_cohort_path):
                controls_meta = pd.read_parquet(control_cohort_path)
                # Find control subject_ids matched to these cases
                matched_ctrls = controls_meta[controls_meta["matched_case_id"].isin(case_subject_ids)]["subject_id"]
                control_subject_ids = set(matched_ctrls)
            else:
                control_subject_ids = set(df[df["label"] == 0]["subject_id"])
                
            df = df[df["subject_id"].isin(case_subject_ids.union(control_subject_ids))]
            
    y = df["label"]
    # Drop standard label columns from X
    X = df.drop(columns=["label"])
    
    return X, y


def create_train_val_test_split(
    df: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split the cohort cleanly by subject_id to prevent any patient leakage."""
    # Ensure subject_id is a column in df
    if "subject_id" not in df.columns:
        df = df.reset_index()
        
    unique_subjects = df[["subject_id", "cancer_type"]].drop_duplicates().copy()
    unique_subjects["label_strata"] = df.loc[unique_subjects.index, "subject_id"].map(y)
    
    # Fill missing cancer type for controls
    unique_subjects["cancer_type"] = unique_subjects["cancer_type"].fillna("control")
    unique_subjects["split_strata"] = unique_subjects["label_strata"].astype(str) + "_" + unique_subjects["cancer_type"]

    # First split off test set — fall back to non-stratified if any class is too small
    try:
        train_val_subjs, test_subjs = train_test_split(
            unique_subjects["subject_id"],
            test_size=test_size,
            random_state=seed,
            stratify=unique_subjects["split_strata"]
        )
    except ValueError:
        logger.warning("Stratified split failed (too few members), falling back to random split.")
        train_val_subjs, test_subjs = train_test_split(
            unique_subjects["subject_id"],
            test_size=test_size,
            random_state=seed
        )
    
    # Second split train and val
    train_val_meta = unique_subjects[unique_subjects["subject_id"].isin(train_val_subjs)]
    
    # Calculate adjusted validation size relative to remaining data
    adj_val_size = val_size / (1.0 - test_size)
    
    try:
        train_subjs, val_subjs = train_test_split(
            train_val_meta["subject_id"],
            test_size=adj_val_size,
            random_state=seed,
            stratify=train_val_meta["split_strata"]
        )
    except ValueError:
        logger.warning("Stratified val split failed, falling back to random split.")
        train_subjs, val_subjs = train_test_split(
            train_val_meta["subject_id"],
            test_size=adj_val_size,
            random_state=seed
        )
    
    train_set_ids = set(train_subjs)
    val_set_ids = set(val_subjs)
    test_set_ids = set(test_subjs)
    
    # Filter datasets
    X_train = df[df["subject_id"].isin(train_set_ids)]
    y_train = y[df["subject_id"].isin(train_set_ids)]
    
    X_val = df[df["subject_id"].isin(val_set_ids)]
    y_val = y[df["subject_id"].isin(val_set_ids)]
    
    X_test = df[df["subject_id"].isin(test_set_ids)]
    y_test = y[df["subject_id"].isin(test_set_ids)]
    
    logger.info(
        "Train-Val-Test Split: Train={} subjects, Val={} subjects, Test={} subjects",
        len(train_set_ids),
        len(val_set_ids),
        len(test_set_ids),
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
