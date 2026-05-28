# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MIMIC-IV cohort extraction script.

Extracts colorectal, lung, and liver cancer cohorts along with 1:3 matched
controls from raw MIMIC-IV tables, applying leakage prevention.
Supports generating synthetic raw files if they are not present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

from src.preprocessing.cohort_matching import (
    match_controls,
    assess_matching_quality,
    compute_smd,
    plot_matching_balance,
)
from src.preprocessing.leakage_prevention import apply_all_leakage_filters


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV cancer cohort.")
    parser.add_argument("--data-raw-dir", type=str, default="data/raw/mimic", help="Path to raw MIMIC-IV directory")
    parser.add_argument("--data-processed-dir", type=str, default="data/processed", help="Path to processed directory")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic raw data if not present")
    parser.add_argument("--control-ratio", type=int, default=3, help="Control matching ratio")
    return parser.parse_known_args()[0]


def generate_synthetic_mimic(output_dir: str):
    """Generate synthetic raw MIMIC-IV files for testing and reproducibility."""
    logger.warning("Raw MIMIC-IV files not found. Generating synthetic raw data at {}...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    n_patients = 1000

    # 1. Patients table
    # gender, anchor_age, anchor_year, dod
    subject_ids = np.arange(100001, 100001 + n_patients)
    genders = np.random.choice(["M", "F"], size=n_patients)
    ages = np.random.randint(18, 90, size=n_patients)
    anchor_years = np.random.randint(2100, 2200, size=n_patients)
    
    patients_df = pd.DataFrame({
        "subject_id": subject_ids,
        "gender": genders,
        "anchor_age": ages,
        "anchor_year": anchor_years,
        "anchor_year_group": "2010 - 2012",
        "dod": [np.nan] * n_patients,
    })
    
    # 2. Admissions table
    # hadm_id, admittime, dischtime, admission_type
    admissions = []
    for sid in subject_ids:
        n_admissions = np.random.randint(1, 4)
        for i in range(n_admissions):
            hadm_id = sid * 10 + i
            admit_year = int(patients_df[patients_df["subject_id"] == sid]["anchor_year"].values[0])
            admit_time = pd.Timestamp(f"{admit_year}-01-01") + pd.Timedelta(days=int(np.random.randint(0, 365)))
            discharge_time = admit_time + pd.Timedelta(days=int(np.random.randint(1, 15)))
            admissions.append({
                "subject_id": sid,
                "hadm_id": hadm_id,
                "admittime": admit_time,
                "dischtime": discharge_time,
                "admission_type": np.random.choice(["EW EMER", "URGENT", "ELECTIVE"]),
                "insurance": np.random.choice(["Medicare", "Other", "Medicaid"]),
            })
    admissions_df = pd.DataFrame(admissions)
    
    # 3. Diagnoses ICD table
    # icd_code, icd_version
    # assign colorectal (C18, 153), lung (C34, 162), liver (C22, 155) or control
    diagnoses = []
    cancer_types = {
        "colorectal": [("C18", 10), ("153", 9)],
        "lung": [("C34", 10), ("162", 9)],
        "liver": [("C22", 10), ("155", 9)],
    }
    
    for _, adm in admissions_df.iterrows():
        sid = adm["subject_id"]
        hadm_id = adm["hadm_id"]
        # 15% cancer rate overall in synthetic data
        prob = np.random.rand()
        if prob < 0.05:
            code, ver = cancer_types["colorectal"][np.random.choice([0, 1])]
            diagnoses.append({"subject_id": sid, "hadm_id": hadm_id, "seq_num": 1, "icd_code": code, "icd_version": ver})
        elif prob < 0.10:
            code, ver = cancer_types["lung"][np.random.choice([0, 1])]
            diagnoses.append({"subject_id": sid, "hadm_id": hadm_id, "seq_num": 1, "icd_code": code, "icd_version": ver})
        elif prob < 0.15:
            code, ver = cancer_types["liver"][np.random.choice([0, 1])]
            diagnoses.append({"subject_id": sid, "hadm_id": hadm_id, "seq_num": 1, "icd_code": code, "icd_version": ver})
        else:
            # benign diagnoses
            diagnoses.append({"subject_id": sid, "hadm_id": hadm_id, "seq_num": 1, "icd_code": "I10", "icd_version": 10})
            
    diagnoses_df = pd.DataFrame(diagnoses)
    
    # 4. Labevents table
    # itemid, charttime, valuenum
    # Need to simulate CBC, metabolic, inflammatory measurements
    labs = []
    # itemids: WBC (51301), Hemoglobin (51222), Platelets (51265), Albumin (50862), ALT (50861), AST (50878), CRP (50889)
    item_map = {
        51301: ("wbc", 4.0, 11.0),
        51222: ("hemoglobin", 12.0, 17.5),
        51265: ("platelets", 150.0, 400.0),
        50862: ("albumin", 3.5, 5.0),
        50861: ("alt", 7.0, 56.0),
        50878: ("ast", 10.0, 40.0),
        50889: ("crp", 0.0, 10.0),
    }
    
    for _, adm in admissions_df.iterrows():
        sid = adm["subject_id"]
        hadm_id = adm["hadm_id"]
        # Generate longitudinal lab observations over prior year
        # Simulate 4-10 timepoints
        n_times = np.random.randint(4, 11)
        base_time = adm["admittime"]
        
        # Determine if patient is cancer case (to inject trajectory trends)
        has_colorectal = not diagnoses_df[(diagnoses_df["subject_id"] == sid) & diagnoses_df["icd_code"].isin(["C18", "153"])].empty
        has_liver = not diagnoses_df[(diagnoses_df["subject_id"] == sid) & diagnoses_df["icd_code"].isin(["C22", "155"])].empty
        
        for t_idx in range(n_times):
            chart_time = base_time - pd.Timedelta(days=t_idx * 30)
            # Add labs
            for itemid, (name, lo, hi) in item_map.items():
                val = np.random.uniform(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
                
                # Inject signal for colorectal cancer: declining hemoglobin
                if name == "hemoglobin" and has_colorectal:
                    val -= (10 - t_idx) * 0.4  # Hemoglobin drops closer to admission
                # Inject signal for liver cancer: elevated AST/ALT
                if name in ["alt", "ast"] and has_liver:
                    val += (10 - t_idx) * 12.0  # AST/ALT rise closer to admission
                    
                labs.append({
                    "subject_id": sid,
                    "hadm_id": hadm_id,
                    "itemid": itemid,
                    "charttime": chart_time,
                    "valuenum": float(round(val, 2)),
                    "valueuom": "unit",
                    "flag": "abnormal" if val < lo or val > hi else None,
                })
                
    labevents_df = pd.DataFrame(labs)
    
    # 5. D_labitems table
    d_labitems_df = pd.DataFrame([
        {"itemid": itemid, "label": name, "fluid": "Blood", "category": "Chemistry" if itemid < 51000 else "Hematology"}
        for itemid, (name, _, _) in item_map.items()
    ])
    
    # 6. Services table
    services_df = pd.DataFrame({
        "subject_id": admissions_df["subject_id"],
        "hadm_id": admissions_df["hadm_id"],
        "curr_service": np.random.choice(["MED", "SURG", "OMED"], size=len(admissions_df)),
    })
    
    # 7. Procedures ICD table
    procedures_df = pd.DataFrame({
        "subject_id": admissions_df["subject_id"],
        "hadm_id": admissions_df["hadm_id"],
        "icd_code": ["9904"] * len(admissions_df),
        "icd_version": [9] * len(admissions_df),
    })

    # Save to parquet
    patients_df.to_parquet(os.path.join(output_dir, "patients.parquet"))
    admissions_df.to_parquet(os.path.join(output_dir, "admissions.parquet"))
    diagnoses_df.to_parquet(os.path.join(output_dir, "diagnoses_icd.parquet"))
    labevents_df.to_parquet(os.path.join(output_dir, "labevents.parquet"))
    d_labitems_df.to_parquet(os.path.join(output_dir, "d_labitems.parquet"))
    services_df.to_parquet(os.path.join(output_dir, "services.parquet"))
    procedures_df.to_parquet(os.path.join(output_dir, "procedures_icd.parquet"))
    logger.info("Synthetic raw MIMIC-IV data generated successfully.")


def load_raw_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw MIMIC-IV tables from parquet or csv."""
    tables = ["patients", "admissions", "diagnoses_icd", "labevents", "d_labitems", "services", "procedures_icd"]
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
            
    # Ensure datetimes are loaded correctly
    dfs["admissions"]["admittime"] = pd.to_datetime(dfs["admissions"]["admittime"])
    dfs["labevents"]["charttime"] = pd.to_datetime(dfs["labevents"]["charttime"])
    
    return (
        dfs["patients"],
        dfs["admissions"],
        dfs["diagnoses_icd"],
        dfs["labevents"],
        dfs["d_labitems"],
        dfs["services"],
        dfs["procedures_icd"],
    )


def main():
    args = parse_args()
    
    os.makedirs(args.data_processed_dir, exist_ok=True)
    
    # Handle synthetic generation check
    raw_files_exist = all(
        os.path.exists(os.path.join(args.data_raw_dir, f"{t}.parquet")) or 
        os.path.exists(os.path.join(args.data_raw_dir, f"{t}.csv"))
        for t in ["patients", "admissions", "diagnoses_icd", "labevents"]
    )
    
    if not raw_files_exist or args.generate_synthetic:
        generate_synthetic_mimic(args.data_raw_dir)
        
    # Load raw data
    try:
        patients, admissions, diagnoses, labevents, d_labitems, services, procedures = load_raw_data(args.data_raw_dir)
    except Exception as exc:
        logger.error("Failed to load raw data: {}. Make sure data path is correct.", exc)
        sys.exit(1)
        
    # ICD Codes mappings
    cancer_codes_icd10 = {
        "colorectal": ["C18", "C19", "C20"],
        "lung": ["C33", "C34"],
        "liver": ["C22"],
    }
    cancer_codes_icd9 = {
        "colorectal": ["153", "154"],
        "lung": ["162"],
        "liver": ["155"],
    }
    
    # Identify case diagnoses
    def get_cancer_type(icd, ver):
        icd = str(icd).strip()
        ver = int(ver)
        if ver == 10:
            for ct, codes in cancer_codes_icd10.items():
                if any(icd.startswith(c) for c in codes):
                    return ct
        elif ver == 9:
            for ct, codes in cancer_codes_icd9.items():
                if any(icd.startswith(c) for c in codes):
                    return ct
        return None

    diagnoses["cancer_type"] = diagnoses.apply(lambda r: get_cancer_type(r["icd_code"], r["icd_version"]), axis=1)
    cancer_dxs = diagnoses[diagnoses["cancer_type"].notna()].copy()
    
    # Find first cancer diagnosis time
    # Join with admissions to get admittime
    cancer_dxs = cancer_dxs.merge(admissions[["hadm_id", "admittime"]], on="hadm_id", how="inner")
    first_dx = cancer_dxs.groupby("subject_id").agg(
        first_diag_date=("admittime", "min"),
        cancer_type=("cancer_type", "first"),
        hadm_id=("hadm_id", "first"),
    ).reset_index()
    
    logger.info("Found {} unique cancer cases.", len(first_dx))
    
    # Apply leakage filters to cancer cases
    clean_cancer_cohort, clean_labs, exclusions = apply_all_leakage_filters(
        first_dx, labevents, procedures, admissions, services, min_days=30, min_labs=3
    )
    
    # Extract controls (must not have any cancer diagnosis code)
    cancer_subject_ids = set(diagnoses[diagnoses["cancer_type"].notna()]["subject_id"])
    all_neoplasm_subjects = set(diagnoses[
        diagnoses["icd_code"].astype(str).str.startswith(("C", "D0", "D1", "D2", "D3", "D4")) |
        diagnoses["icd_code"].astype(str).str.startswith(tuple(str(x) for x in range(140, 240)))
    ]["subject_id"])
    
    control_pool_subjects = set(patients["subject_id"]) - cancer_subject_ids - all_neoplasm_subjects
    
    # Build control pool data frame
    control_pool = patients[patients["subject_id"].isin(control_pool_subjects)].copy()
    # Merge with admissions to get admission year and hadm_id
    control_pool = control_pool.merge(admissions[["subject_id", "hadm_id", "admittime"]], on="subject_id", how="inner")
    control_pool["admission_year"] = control_pool["admittime"].dt.year
    control_pool = control_pool.groupby("subject_id").first().reset_index()
    control_pool.rename(columns={"anchor_age": "age", "gender": "gender"}, inplace=True)
    
    # Prepare cancer cohort for matching
    clean_cancer_cohort = clean_cancer_cohort.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="inner")
    clean_cancer_cohort.rename(columns={"anchor_age": "age"}, inplace=True)
    clean_cancer_cohort["admission_year"] = clean_cancer_cohort["first_diag_date"].dt.year
    
    # Match controls
    matched_cases_df, matched_controls_df = match_controls(
        clean_cancer_cohort, control_pool, ratio=args.control_ratio
    )
    
    # Save processed datasets
    matched_cases_df.to_parquet(os.path.join(args.data_processed_dir, "mimic_cancer_cohort.parquet"))
    matched_controls_df.to_parquet(os.path.join(args.data_processed_dir, "mimic_control_cohort.parquet"))
    
    # Save labs for the cohort
    cohort_subjects = set(matched_cases_df["subject_id"]).union(set(matched_controls_df["subject_id"]))
    cohort_labs = clean_labs[clean_labs["subject_id"].isin(cohort_subjects)].copy()
    cohort_labs.to_parquet(os.path.join(args.data_processed_dir, "mimic_cohort_labs.parquet"))
    
    # Assess matching and write stats
    covariates = [c for c in ["age", "admission_year"] if c in matched_cases_df.columns and c in matched_controls_df.columns]
    smd_df = compute_smd(matched_cases_df, matched_controls_df, covariates)
    match_quality = assess_matching_quality(matched_cases_df, matched_controls_df)

    # Save Love plot
    try:
        plot_matching_balance(smd_df, os.path.join(args.data_processed_dir, "love_plot.png"))
    except Exception as e:
        logger.warning("Could not save Love plot: {}", e)
    
    # Attrition report
    consort_report = {
        "original_cancer_cases": len(first_dx),
        "exclusions": exclusions,
        "clean_cancer_cases": len(clean_cancer_cohort),
        "matched_cancer_cases": len(matched_cases_df),
        "matched_controls": len(matched_controls_df),
        "matching_quality": {
            "max_abs_smd": match_quality["max_abs_smd"],
            "is_well_balanced": match_quality["is_well_balanced"],
        }
    }
    
    with open(os.path.join(args.data_processed_dir, "cohort_stats.json"), "w") as f:
        json.dump(consort_report, f, indent=2)
        
    logger.info("Cohort extraction completed successfully. Stats saved to cohort_stats.json.")


if __name__ == "__main__":
    main()
