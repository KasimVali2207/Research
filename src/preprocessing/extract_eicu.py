# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
eICU external validation cohort extraction script.

Extracts external validation cohorts (cases and matched controls) from eICU
tables, applying matching and leakage prevention. Includes synthetic eICU generator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

from src.preprocessing.cohort_matching import match_controls, assess_matching_quality
from src.preprocessing.lab_itemids import EICU_LAB_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Extract eICU cancer cohort.")
    parser.add_argument("--data-raw-dir", type=str, default="data/raw/eicu", help="Path to raw eICU directory")
    parser.add_argument("--data-processed-dir", type=str, default="data/processed", help="Path to processed directory")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic raw data if not present")
    parser.add_argument("--control-ratio", type=int, default=3, help="Control matching ratio")
    return parser.parse_known_args()[0]


def generate_synthetic_eicu(output_dir: str):
    """Generate synthetic raw eICU tables for testing and reproducibility."""
    logger.warning("Raw eICU files not found. Generating synthetic raw data at {}...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    n_stays = 500

    # 1. Patient table
    patientunitstayids = np.arange(200001, 200001 + n_stays)
    genders = np.random.choice(["Male", "Female"], size=n_stays)
    ages = np.random.randint(18, 90, size=n_stays)
    ethnicities = np.random.choice(["Caucasian", "African American", "Asian"], size=n_stays)
    
    patient_df = pd.DataFrame({
        "patientunitstayid": patientunitstayids,
        "patienthealthsystemstayid": patientunitstayids + 100000,
        "uniquepid": [f"002-{i:05d}" for i in range(n_stays)],
        "age": ages.astype(str),  # eICU age is sometimes string (e.g. "> 89")
        "gender": genders,
        "ethnicity": ethnicities,
        "hospitalid": np.random.randint(100, 200, size=n_stays),
        "unittype": np.random.choice(["MICU", "SICU", "CCU"]),
        "admissionheight": np.random.uniform(150, 190, size=n_stays),
        "admissionweight": np.random.uniform(50, 110, size=n_stays),
        "unitdischargestatus": np.random.choice(["Alive", "Expired"], p=[0.92, 0.08], size=n_stays),
    })
    
    # 2. Diagnosis table
    diagnoses = []
    cancer_codes = {
        "colorectal": "153.9",
        "lung": "162.9",
        "liver": "155.0",
    }
    
    for stay_id in patientunitstayids:
        prob = np.random.rand()
        # 12% cancer rate in eICU synthetic subset
        if prob < 0.04:
            diagnoses.append({
                "patientunitstayid": stay_id,
                "diagnosisoffset": int(np.random.randint(10, 500)),
                "icd9code": cancer_codes["colorectal"],
                "diagnosisstring": "oncology|colorectal cancer",
            })
        elif prob < 0.08:
            diagnoses.append({
                "patientunitstayid": stay_id,
                "diagnosisoffset": int(np.random.randint(10, 500)),
                "icd9code": cancer_codes["lung"],
                "diagnosisstring": "oncology|lung cancer",
            })
        elif prob < 0.12:
            diagnoses.append({
                "patientunitstayid": stay_id,
                "diagnosisoffset": int(np.random.randint(10, 500)),
                "icd9code": cancer_codes["liver"],
                "diagnosisstring": "oncology|liver cancer",
            })
        else:
            diagnoses.append({
                "patientunitstayid": stay_id,
                "diagnosisoffset": int(np.random.randint(10, 500)),
                "icd9code": "401.9",
                "diagnosisstring": "cardiovascular|hypertension",
            })
            
    diagnosis_df = pd.DataFrame(diagnoses)
    
    # 3. Lab table
    labs = []
    # Names match EICU_LAB_NAMES mapping
    lab_names = {
        "wbc": "WBC x 1000",
        "hemoglobin": "Hgb",
        "platelets": "platelets x 1000",
        "albumin": "albumin",
        "alt": "ALT (SGPT)",
        "ast": "AST (SGOT)",
    }
    
    for stay_id in patientunitstayids:
        # eICU offsets are in minutes. Simulate 3-8 labs over time
        n_labs = np.random.randint(3, 9)
        has_cancer = stay_id in diagnosis_df[diagnosis_df["diagnosisstring"].str.contains("oncology")]["patientunitstayid"].values
        
        for l_idx in range(n_labs):
            offset = -1 * l_idx * 1440  # Days prior in minutes
            for feat_name, eicu_name in lab_names.items():
                val = np.random.uniform(5.0, 150.0) if feat_name in ["alt", "ast"] else np.random.uniform(5.0, 15.0)
                
                # Introduce cancer drop in hemoglobin
                if feat_name == "hemoglobin" and has_cancer:
                    val = max(5.0, val - (5 - l_idx) * 0.5)
                    
                labs.append({
                    "patientunitstayid": stay_id,
                    "labresultoffset": offset,
                    "labname": eicu_name,
                    "labresult": float(round(val, 2)),
                    "labresulttext": str(round(val, 2)),
                })
                
    lab_df = pd.DataFrame(labs)
    
    # Save
    patient_df.to_parquet(os.path.join(output_dir, "patient.parquet"))
    diagnosis_df.to_parquet(os.path.join(output_dir, "diagnosis.parquet"))
    lab_df.to_parquet(os.path.join(output_dir, "lab.parquet"))
    logger.info("Synthetic raw eICU data generated successfully.")


def load_raw_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw eICU tables."""
    tables = ["patient", "diagnosis", "lab"]
    dfs = {}
    
    for tbl in tables:
        pq_path = os.path.join(data_dir, f"{tbl}.parquet")
        csv_path = os.path.join(data_dir, f"{tbl}.csv")
        
        if os.path.exists(pq_path):
            logger.info("Loading table {} from {}", tbl, pq_path)
            dfs[tbl] = pd.read_parquet(pq_path)
        elif os.path.exists(csv_path):
            logger.info("Loading table {} from {}", tbl, csv_path)
            dfs[tbl] = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Could not find table {tbl} in {data_dir} (.parquet or .csv)")
            
    return dfs["patient"], dfs["diagnosis"], dfs["lab"]


def main():
    args = parse_args()
    
    os.makedirs(args.data_processed_dir, exist_ok=True)
    
    raw_files_exist = all(
        os.path.exists(os.path.join(args.data_raw_dir, f"{t}.parquet")) or 
        os.path.exists(os.path.join(args.data_raw_dir, f"{t}.csv"))
        for t in ["patient", "diagnosis", "lab"]
    )
    
    if not raw_files_exist or args.generate_synthetic:
        generate_synthetic_eicu(args.data_raw_dir)
        
    # Load raw data
    try:
        patient, diagnosis, lab = load_raw_data(args.data_raw_dir)
    except Exception as exc:
        logger.error("Failed to load raw eICU data: {}", exc)
        sys.exit(1)
        
    # Parse ages, mapping >89 to 90
    def clean_age(x):
        try:
            return float(x)
        except ValueError:
            if isinstance(x, str) and ">" in x:
                return 90.0
            return np.nan

    patient["age_clean"] = patient["age"].apply(clean_age)
    patient = patient.dropna(subset=["age_clean"])
    
    # Identify cases using ICD-9 codes
    cancer_codes = {
        "colorectal": ["153", "154"],
        "lung": ["162"],
        "liver": ["155"],
    }
    
    def get_cancer_type(icd):
        icd = str(icd).strip()
        for ct, codes in cancer_codes.items():
            if any(icd.startswith(c) for c in codes):
                return ct
        return None

    diagnosis["cancer_type"] = diagnosis["icd9code"].apply(get_cancer_type)
    cancer_dxs = diagnosis[diagnosis["cancer_type"].notna()].copy()
    
    # Find first diagnosis offset (in minutes from stay start)
    first_dx = cancer_dxs.groupby("patientunitstayid").agg(
        first_diag_offset=("diagnosisoffset", "min"),
        cancer_type=("cancer_type", "first")
    ).reset_index()
    
    logger.info("Found {} unique cancer cases in eICU.", len(first_dx))
    
    # Censor labs post diagnosis
    # join labs with diagnosis offset
    labs_merged = lab.merge(first_dx, on="patientunitstayid", how="left")
    # For cases, exclude labs with labresultoffset > first_diag_offset
    labs_merged["is_post_dx"] = (labs_merged["labresultoffset"] > labs_merged["first_diag_offset"]) & labs_merged["first_diag_offset"].notna()
    pre_dx_labs = labs_merged[~labs_merged["is_post_dx"]].copy()
    
    # Map eICU lab names to standard features
    # Build reverse mapping from EICU_LAB_NAMES
    eicu_rev_map = {}
    for feat_name, names in EICU_LAB_NAMES.items():
        for name in names:
            eicu_rev_map[name.lower()] = feat_name
            
    pre_dx_labs["feature_name"] = pre_dx_labs["labname"].str.lower().map(eicu_rev_map)
    clean_labs = pre_dx_labs.dropna(subset=["feature_name"]).copy()
    
    # Exclude patients with insufficient observation window
    # eICU offsets are in minutes (1 day = 1440 minutes)
    stats = clean_labs.groupby("patientunitstayid").agg(
        min_offset=("labresultoffset", "min"),
        max_offset=("labresultoffset", "max"),
        count=("labresult", "count")
    )
    stats["span_days"] = (stats["max_offset"] - stats["min_offset"]) / 1440.0
    
    # Relax observation window for eICU validation (e.g. min 5 days, 3 labs)
    valid_stays = stats[(stats["span_days"] >= 5) & (stats["count"] >= 3)].index
    clean_labs = clean_labs[clean_labs["patientunitstayid"].isin(valid_stays)]
    
    final_cases = first_dx[first_dx["patientunitstayid"].isin(valid_stays)].copy()
    final_cases = final_cases.merge(patient[["patientunitstayid", "gender", "age_clean"]], on="patientunitstayid", how="inner")
    final_cases.rename(columns={"age_clean": "age", "patientunitstayid": "subject_id"}, inplace=True)
    
    # Match controls
    cancer_stay_ids = set(diagnosis[diagnosis["cancer_type"].notna()]["patientunitstayid"])
    control_pool_stays = set(patient["patientunitstayid"]) - cancer_stay_ids
    
    control_pool = patient[patient["patientunitstayid"].isin(control_pool_stays)].copy()
    control_pool.rename(columns={"age_clean": "age", "patientunitstayid": "subject_id"}, inplace=True)
    # Mock admission year since eICU doesn't have it
    control_pool["admission_year"] = 2150
    final_cases["admission_year"] = 2150
    
    matched_cases_df, matched_controls_df = match_controls(
        final_cases, control_pool, ratio=args.control_ratio
    )
    
    # Output to processed files
    matched_cases_df.to_parquet(os.path.join(args.data_processed_dir, "eicu_cancer_cohort.parquet"))
    matched_controls_df.to_parquet(os.path.join(args.data_processed_dir, "eicu_control_cohort.parquet"))
    
    cohort_subjects = set(matched_cases_df["subject_id"]).union(set(matched_controls_df["subject_id"]))
    cohort_labs = clean_labs[clean_labs["patientunitstayid"].isin(cohort_subjects)].copy()
    cohort_labs.rename(columns={"patientunitstayid": "subject_id", "labresult": "valuenum"}, inplace=True)
    
    # Map offsets back to time stamps if needed, otherwise keep labresultoffset
    # eICU doesn't have explicit dates, so we use relative offsets as charttime
    cohort_labs["charttime"] = pd.Timestamp("2150-01-01") + pd.to_timedelta(cohort_labs["labresultoffset"], unit="m")
    cohort_labs.to_parquet(os.path.join(args.data_processed_dir, "eicu_cohort_labs.parquet"))
    
    logger.info("eICU cohort extraction completed successfully.")


if __name__ == "__main__":
    main()
