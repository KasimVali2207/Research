# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MIMIC-IV and eICU features pivot pipeline.

Applies observation window slicing for 3, 6, and 12 months horizons.
Builds the wide feature tables for models.
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
from loguru import logger

from src.features.temporal_features import TemporalFeatureExtractor
from src.preprocessing.lab_itemids import MIMIC_LAB_ITEMIDS


def parse_args():
    parser = argparse.ArgumentParser(description="Transform cohort labs to wide features.")
    parser.add_argument("--data-processed-dir", type=str, default="data/processed", help="Path to processed directory")
    parser.add_argument("--dataset", type=str, default="mimic", choices=["mimic", "eicu"], help="Dataset to process")
    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    
    cases_path = os.path.join(args.data_processed_dir, f"{args.dataset}_cancer_cohort.parquet")
    controls_path = os.path.join(args.data_processed_dir, f"{args.dataset}_control_cohort.parquet")
    labs_path = os.path.join(args.data_processed_dir, f"{args.dataset}_cohort_labs.parquet")
    
    if not (os.path.exists(cases_path) and os.path.exists(controls_path) and os.path.exists(labs_path)):
        logger.error("Missing cohort files. Run extract_cohort.py or extract_eicu.py first.")
        return
        
    logger.info("Loading cohort and labs for dataset: {}", args.dataset)
    cases = pd.read_parquet(cases_path)
    controls = pd.read_parquet(controls_path)
    labs = pd.read_parquet(labs_path)
    
    # Label cases and controls
    cases["label"] = 1
    controls["label"] = 0
    
    # Align timelines: map case diagnosis date to matched controls
    if args.dataset == "mimic":
        controls_aligned = controls.merge(
            cases[["subject_id", "first_diag_date"]].rename(columns={"subject_id": "matched_case_id", "first_diag_date": "case_diag_date"}),
            on="matched_case_id",
            how="inner"
        )
        controls_aligned["first_diag_date"] = controls_aligned["case_diag_date"]
        controls_aligned.drop(columns=["case_diag_date"], inplace=True)
    else:
        # eICU already has standard aligned offsets relative to stay, mock dates are used
        controls["first_diag_date"] = pd.Timestamp("2150-01-01") + pd.to_timedelta(500, unit="m")
        controls_aligned = controls
        
    cohort = pd.concat([cases, controls_aligned], ignore_index=True)
    cohort["first_diag_date"] = pd.to_datetime(cohort["first_diag_date"])
    
    # Map itemid/labname to standard feature names if raw mapping not yet applied
    if "feature_name" not in labs.columns:
        # MIMIC itemid mapping
        rev_item_map = {}
        for feat, itemids in MIMIC_LAB_ITEMIDS.items():
            for iid in itemids:
                rev_item_map[iid] = feat
        labs["feature_name"] = labs["itemid"].map(rev_item_map)
        labs = labs.dropna(subset=["feature_name"])
        
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    
    # Feature list
    feature_list = sorted(list(MIMIC_LAB_ITEMIDS.keys()))
    
    horizons = [3, 6, 12]
    
    for h in horizons:
        logger.info("Processing features for horizon {}m...", h)
        
        # Merge cohort first_diag_date into labs to filter per-patient window
        merged_labs = labs.merge(cohort[["subject_id", "first_diag_date", "label", "cancer_type"]], on="subject_id", how="inner")
        
        # Observation window: [first_diag_date - 12m, first_diag_date - H months]
        # Using 30.4 days per month as an approximation
        window_start = merged_labs["first_diag_date"] - pd.Timedelta(days=365)
        window_end = merged_labs["first_diag_date"] - pd.Timedelta(days=h * 30.4)
        
        filtered_labs = merged_labs[
            (merged_labs["charttime"] >= window_start) & 
            (merged_labs["charttime"] <= window_end)
        ].copy()
        
        # Ensure we only have standard columns for feature extraction
        longitudinal_df = filtered_labs[["subject_id", "charttime", "feature_name", "valuenum"]].rename(columns={"valuenum": "value"})
        
        # Apply TemporalFeatureExtractor
        extractor = TemporalFeatureExtractor(feature_names=feature_list)
        wide_features = extractor.fit_transform(longitudinal_df)
        
        # Join labels and metadata back
        final_dataset = cohort[["subject_id", "label", "cancer_type", "age", "gender"]].merge(
            wide_features, on="subject_id", how="inner"
        )
        
        # Save Parquet
        out_path = os.path.join(args.data_processed_dir, f"features_horizon_{h}m_{args.dataset}.parquet")
        final_dataset.to_parquet(out_path)
        logger.info("Saved {} features for horizon {}m to {}", len(final_dataset), h, out_path)


if __name__ == "__main__":
    main()
