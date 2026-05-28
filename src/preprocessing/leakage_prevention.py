# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Leakage prevention module for multi-cancer early detection cohort.

Prevents target and temporal leakage by:
- Identifying and removing lab events that occur after the cancer diagnosis.
- Excluding oncology admissions and oncology service lines.
- Excluding oncology treatments (chemo/radiotherapy) prior to baseline.
- Validating the observation window for each patient.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def identify_post_diagnosis_labs(labs_df: pd.DataFrame, diagnosis_dates_df: pd.DataFrame) -> pd.DataFrame:
    """Flag labs occurring after the first cancer diagnosis date.

    Args:
        labs_df: DataFrame with at least ['subject_id', 'charttime'].
        diagnosis_dates_df: DataFrame with ['subject_id', 'first_diag_date'].

    Returns:
        labs_df with 'is_post_diagnosis' boolean column added.
    """
    df = labs_df.merge(diagnosis_dates_df, on="subject_id", how="left")
    # For controls, first_diag_date is NaT, so they will not be marked as post-diagnosis.
    df["is_post_diagnosis"] = (df["charttime"] > df["first_diag_date"]) & df["first_diag_date"].notna()
    return df.drop(columns=["first_diag_date"])


def exclude_oncology_admissions(admissions_df: pd.DataFrame, services_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out admissions associated with oncology or hematology service lines.

    MIMIC services include: 'HEM' (Hematology), 'ONC' (Oncology), 'GYNO' (Gynecologic Oncology), etc.

    Args:
        admissions_df: DataFrame of admissions.
        services_df: DataFrame of service transfers with ['hadm_id', 'curr_service'].

    Returns:
        Filtered admissions_df.
    """
    oncology_services = ["HEM", "ONC", "GYNO", "MEDONC", "SURGONC"]
    
    # Identify admissions that touched oncology service lines
    onc_hadms = services_df[services_df["curr_service"].isin(oncology_services)]["hadm_id"].unique()
    
    initial_count = len(admissions_df)
    filtered_df = admissions_df[~admissions_df["hadm_id"].isin(onc_hadms)]
    removed_count = initial_count - len(filtered_df)
    
    logger.info(
        "Oncology admissions exclusion: removed {}/{} admissions ({}%)",
        removed_count,
        initial_count,
        round(removed_count / max(initial_count, 1) * 100, 2),
    )
    return filtered_df


def exclude_treatment_procedures(
    df: pd.DataFrame,
    procedures_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Exclude patients who received oncology-related treatment (chemo/radiotherapy) in baseline window.

    ICD-9 procedures:
      - Chemotherapy: 99.25, 00.10, etc.
      - Radiotherapy: 92.2x, 92.3x
    ICD-10 procedures:
      - Chemotherapy: D0002ZZ, etc. (often Z51.1 in ICD-10-CM diagnoses, but this is procedures)
      - Radiotherapy: DW0xxxx, etc.

    Args:
        df: Cohort DataFrame with ['subject_id'].
        procedures_df: DataFrame of procedures with ['subject_id', 'icd_code', 'icd_version'].

    Returns:
        Tuple of (filtered_df, exclusions_dict).
    """
    chemo_icd9_prefixes = ("9925", "0010")
    radio_icd9_prefixes = ("922", "923")
    chemo_icd10_prefixes = ("3E03005", "3E03305", "3E033HZ", "3E0D329", "3E0E329", "3E0F329")
    radio_icd10_prefixes = ("D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "DB", "DC", "DD", "DF", "DG", "DH", "DJ", "DK", "DL", "DM", "DN", "DP", "DQ", "DR", "DS", "DT", "DV", "DW", "DX", "DY")

    def is_oncology_treatment(row):
        code = str(row["icd_code"]).replace(".", "").strip()
        version = int(row["icd_version"])
        if version == 9:
            return code.startswith(chemo_icd9_prefixes) or code.startswith(radio_icd9_prefixes)
        elif version == 10:
            return code.startswith(chemo_icd10_prefixes) or code.startswith(radio_icd10_prefixes)
        return False

    onc_proc_df = procedures_df[procedures_df.apply(is_oncology_treatment, axis=1)]
    onc_subjects = onc_proc_df["subject_id"].unique()

    initial_count = len(df)
    filtered_df = df[~df["subject_id"].isin(onc_subjects)]
    removed_count = initial_count - len(filtered_df)

    exclusions_dict = {
        "treatment_procedure_exclusion": int(removed_count)
    }

    logger.info(
        "Treatment procedure exclusion: removed {}/{} patients ({}%)",
        removed_count,
        initial_count,
        round(removed_count / max(initial_count, 1) * 100, 2),
    )
    return filtered_df, exclusions_dict


def validate_observation_window(
    labs_df: pd.DataFrame,
    min_days: int = 30,
    min_labs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure patients have sufficient observation window and lab frequency.

    Args:
        labs_df: DataFrame of lab events.
        min_days: Minimum number of days between first and last lab.
        min_labs: Minimum number of total labs.

    Returns:
        Tuple of (valid_labs_df, excluded_labs_df).
    """
    # Calculate stats per subject_id
    stats = labs_df.groupby("subject_id").agg(
        first_time=("charttime", "min"),
        last_time=("charttime", "max"),
        count=("charttime", "count"),
    )
    stats["span_days"] = (stats["last_time"] - stats["first_time"]).dt.total_seconds() / (24 * 3600)
    
    valid_subjects = stats[(stats["span_days"] >= min_days) & (stats["count"] >= min_labs)].index
    
    valid_labs = labs_df[labs_df["subject_id"].isin(valid_subjects)]
    excluded_labs = labs_df[~labs_df["subject_id"].isin(valid_subjects)]
    
    logger.info(
        "Observation window validation: {} valid subjects, {} excluded (minimum days={}, minimum labs={})",
        len(valid_subjects),
        len(stats) - len(valid_subjects),
        min_days,
        min_labs,
    )
    return valid_labs, excluded_labs


def create_leakage_report(original_n: int, exclusions_dict: dict[str, int]) -> str:
    """Generate CONSORT-style attrition text report.

    Args:
        original_n: Initial cohort count.
        exclusions_dict: Dictionary mapping exclusion name to count.

    Returns:
        Markdown-formatted string report.
    """
    lines = [
        "## CONSORT Attrition Report",
        "",
        f"Initial Cohort Pool: N = {original_n}",
    ]
    current_n = original_n
    for key, count in exclusions_dict.items():
        percent = round(count / max(original_n, 1) * 100, 2)
        lines.append(f"  - Excluded due to {key.replace('_', ' ')}: N = {count} ({percent}%)")
        current_n -= count
    lines.append(f"Final Study Cohort: N = {current_n}")
    return "\n".join(lines)


def apply_all_leakage_filters(
    cancer_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    services_df: pd.DataFrame,
    min_days: int = 30,
    min_labs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Orchestrate the full leakage prevention workflow in sequential steps.

    Args:
        cancer_df: Base cancer patient cohort DataFrame.
        labs_df: Longitudinal lab measurements DataFrame.
        procedures_df: Procedures recorded for these patients.
        admissions_df: Admissions DataFrame.
        services_df: Service transfers DataFrame.
        min_days: Minimum duration of history required.
        min_labs: Minimum number of lab measurements.

    Returns:
        Tuple of (clean_cancer_df, clean_labs_df, exclusion_report_dict).
    """
    exclusions = {}
    original_patients = len(cancer_df)

    # Step 1: Filter admissions by oncology services
    clean_admissions = exclude_oncology_admissions(admissions_df, services_df)
    valid_hadms = set(clean_admissions["hadm_id"])
    
    # Filter cancer cohort by valid admissions
    filtered_cancer_df = cancer_df[cancer_df["hadm_id"].isin(valid_hadms)]
    exclusions["oncology_admissions"] = original_patients - len(filtered_cancer_df)
    
    # Step 2: Exclude oncology treatments
    filtered_cancer_df, treatment_excl = exclude_treatment_procedures(filtered_cancer_df, procedures_df)
    exclusions.update(treatment_excl)

    # Step 3: Flag and censor post-diagnosis labs
    diagnosis_dates = filtered_cancer_df[["subject_id", "first_diag_date"]].drop_duplicates()
    labs_with_flags = identify_post_diagnosis_labs(labs_df, diagnosis_dates)
    
    # Filter out post-diagnosis labs
    pre_dx_labs = labs_with_flags[~labs_with_flags["is_post_diagnosis"]].copy()
    
    # Step 4: Enforce minimum observation window
    clean_labs, excluded_labs = validate_observation_window(
        pre_dx_labs, min_days=min_days, min_labs=min_labs
    )
    
    valid_subjects = set(clean_labs["subject_id"])
    final_cancer_df = filtered_cancer_df[filtered_cancer_df["subject_id"].isin(valid_subjects)]
    
    exclusions["insufficient_observation"] = len(filtered_cancer_df) - len(final_cancer_df)
    
    return final_cancer_df, clean_labs, exclusions
