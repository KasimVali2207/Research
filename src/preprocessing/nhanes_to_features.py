"""
NHANES → Feature Matrix Pipeline
Merges demographics, CBC, biochemistry, CRP, ferritin, cancer labels
into a clean ML-ready dataset with real cancer cases.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/raw/nhanes")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CYCLES = {
    "2013": ("H", "2013-2014"),
    "2015": ("I", "2015-2016"),
    "2017": ("J", "2017-2018"),
}

# Cancer type codes in MCQ230x that map to our 3 cancer types
# 10=colorectal, 14=lung, 22=liver (NHANES codes)
CANCER_TYPE_MAP = {
    10: "colorectal", 11: "colorectal", 12: "colorectal",
    14: "lung",  15: "lung",
    22: "liver", 23: "liver",
}

def read_xpt(path):
    """Read SAS XPT file safely."""
    try:
        return pd.read_sas(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_sas(path, encoding="latin-1")
        except Exception as e:
            print(f"  Could not read {path.name}: {e}")
            return None

def load_cycle(year_start, letter, year_label):
    """Load and merge all tables for one NHANES cycle."""
    def p(prefix):
        return DATA_DIR / f"{prefix}_{letter}_{year_label.replace('-','_')}.XPT"

    # --- Demographics ---
    demo = read_xpt(p("DEMO"))
    if demo is None:
        return None
    demo = demo[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]].copy()
    demo.columns = ["seqn", "gender", "age", "ethnicity"]
    demo["gender"]    = demo["gender"].map({1: "M", 2: "F"})
    demo["ethnicity"] = demo["ethnicity"].map({
        1:"Mexican American", 2:"Other Hispanic", 3:"Non-Hispanic White",
        4:"Non-Hispanic Black", 6:"Non-Hispanic Asian", 7:"Other"
    })

    # --- CBC ---
    cbc = read_xpt(p("CBC"))
    if cbc is not None:
        cbc_cols = {
            "SEQN":     "seqn",
            "LBXWBCSI": "wbc",
            "LBXRBCSI": "rbc",
            "LBXHGB":   "hemoglobin",
            "LBXHCT":   "hematocrit",
            "LBXMCVSI": "mcv",
            "LBXMCHSI": "mch",
            "LBXRDW":   "rdw",
            "LBXPLTSI": "platelets",
            "LBDNENO":  "neutrophils",
            "LBDLYMNO": "lymphocytes",
            "LBDMONO":  "monocytes",
            "LBXEOPCT": "eosinophils_pct",
            "LBXBAPCT": "basophils_pct",
        }
        cbc = cbc[[c for c in cbc_cols if c in cbc.columns]].rename(columns=cbc_cols)
    else:
        cbc = pd.DataFrame({"seqn": demo["seqn"]})

    # --- Biochemistry (BIOPRO) ---
    bio = read_xpt(p("BIOPRO"))
    if bio is not None:
        bio_cols = {
            "SEQN":    "seqn",
            "LBXSAL":  "albumin",
            "LBXSATSI":"alt",
            "LBXSASSI":"ast",
            "LBXSAPSI":"alp",
            "LBXSTB":  "bilirubin_total",
            "LBXSCR":  "creatinine",
            "LBXSBU":  "bun",
            "LBXSNASI":"sodium",
            "LBXSKSI": "potassium",
            "LBXSCA":  "calcium",
            "LBXSTP":  "total_protein",
            "LBXSGB":  "globulin",
            "LBXSGL":  "glucose",
        }
        bio = bio[[c for c in bio_cols if c in bio.columns]].rename(columns=bio_cols)
    else:
        bio = pd.DataFrame({"seqn": demo["seqn"]})

    # --- CRP ---
    crp = read_xpt(p("HSCRP"))
    if crp is not None and "LBXHSCRP" in crp.columns:
        crp = crp[["SEQN", "LBXHSCRP"]].rename(columns={"SEQN":"seqn","LBXHSCRP":"crp"})
    else:
        crp = pd.DataFrame({"seqn": demo["seqn"], "crp": np.nan})

    # --- Ferritin ---
    ferr = read_xpt(p("FERTIN"))
    if ferr is not None:
        ferr_col = "LBXFER" if "LBXFER" in ferr.columns else (
                   "LBDFERSI" if "LBDFERSI" in ferr.columns else None)
        if ferr_col:
            ferr = ferr[["SEQN", ferr_col]].rename(columns={"SEQN":"seqn", ferr_col:"ferritin"})
        else:
            ferr = pd.DataFrame({"seqn": demo["seqn"], "ferritin": np.nan})
    else:
        ferr = pd.DataFrame({"seqn": demo["seqn"], "ferritin": np.nan})

    # --- Cancer Labels (MCQ) ---
    mcq = read_xpt(p("MCQ"))
    if mcq is None:
        return None

    mcq_sub = mcq[["SEQN", "MCQ220"]].copy()
    mcq_sub.columns = ["seqn", "ever_cancer"]
    mcq_sub["cancer"] = (mcq_sub["ever_cancer"] == 1).astype(int)

    # Extract cancer type from MCQ230A-D
    type_cols = [c for c in mcq.columns if c.startswith("MCQ230")]
    mcq_sub["cancer_type"] = "none"
    if type_cols:
        for col in type_cols:
            mapped = mcq[col].map(CANCER_TYPE_MAP)
            mask   = mapped.notna() & (mcq_sub["cancer"] == 1)
            mcq_sub.loc[mask.values, "cancer_type"] = mapped[mask].values

    mcq_sub = mcq_sub[["seqn", "cancer", "cancer_type"]]

    # --- Merge everything ---
    df = demo.merge(cbc,      on="seqn", how="left")
    df = df.merge(bio,        on="seqn", how="left")
    df = df.merge(crp,        on="seqn", how="left")
    df = df.merge(ferr,       on="seqn", how="left")
    df = df.merge(mcq_sub,    on="seqn", how="left")
    df["cycle"] = year_label

    # --- Derived features ---
    if "neutrophils" in df.columns and "lymphocytes" in df.columns:
        df["nlr"] = df["neutrophils"] / df["lymphocytes"].replace(0, np.nan)
    if "platelets" in df.columns and "lymphocytes" in df.columns:
        df["plr"] = df["platelets"] / df["lymphocytes"].replace(0, np.nan)
    if "wbc" in df.columns and "platelets" in df.columns and "neutrophils" in df.columns:
        df["sii"] = df["neutrophils"] * df["platelets"] / df["lymphocytes"].replace(0, np.nan)

    # Filter to adults (18+) with cancer label available
    df = df[df["age"] >= 18].copy()
    df = df[df["cancer"].notna()].copy()

    return df


# ---- Main ----
print("=" * 65)
print("NHANES Feature Pipeline")
print("=" * 65)

all_cycles = []
for year_start, (letter, year_label) in CYCLES.items():
    print(f"\nProcessing {year_label}...")
    df = load_cycle(year_start, letter, year_label)
    if df is not None:
        n_cancer = int(df["cancer"].sum())
        n_total  = len(df)
        print(f"  Subjects  : {n_total:,}")
        print(f"  Cancer    : {n_cancer:,} ({100*n_cancer/n_total:.1f}%)")
        print(f"  Features  : {df.shape[1]}")
        types = df[df["cancer"]==1]["cancer_type"].value_counts().to_dict()
        print(f"  Types     : {types}")
        all_cycles.append(df)
    else:
        print(f"  SKIP (missing files)")

if not all_cycles:
    print("No data loaded!")
    exit(1)

combined = pd.concat(all_cycles, ignore_index=True)

# Keep only colorectal / lung / liver as cancer+ cases; rest as controls
target_cancers = {"colorectal", "lung", "liver"}
combined["label"] = 0
combined.loc[(combined["cancer"]==1) & (combined["cancer_type"].isin(target_cancers)), "label"] = 1
combined.loc[(combined["cancer"]==1) & (~combined["cancer_type"].isin(target_cancers)), "label"] = -1  # exclude other cancers

# Exclude "other cancer" patients (not cases, not controls)
combined = combined[combined["label"] != -1].copy()

# Final feature set
feature_cols = [c for c in combined.columns if c not in
    ("seqn","cancer","cancer_type","ever_cancer","label","cycle","gender","ethnicity")]

print(f"\n{'='*65}")
print(f"FINAL COMBINED DATASET")
print(f"  Total subjects       : {len(combined):,}")
print(f"  Cancer cases (target): {int((combined['label']==1).sum()):,}")
print(f"  Controls             : {int((combined['label']==0).sum()):,}")
print(f"  Feature columns      : {len(feature_cols)}")
print(f"  Cancer breakdown     :")
print(combined[combined["label"]==1]["cancer_type"].value_counts().to_string())

# Save outputs
combined.to_parquet(OUTPUT_DIR / "nhanes_features.parquet", index=False)

# Also save cancer + matched controls separately
cases    = combined[combined["label"] == 1].copy()
controls = combined[combined["label"] == 0].copy()
cases.to_parquet(OUTPUT_DIR / "nhanes_cancer_cases.parquet",    index=False)
controls.to_parquet(OUTPUT_DIR / "nhanes_controls.parquet",     index=False)

# Stats JSON
stats = {
    "total_subjects"  : int(len(combined)),
    "cancer_cases"    : int((combined["label"]==1).sum()),
    "controls"        : int((combined["label"]==0).sum()),
    "feature_columns" : len(feature_cols),
    "cancer_types"    : combined[combined["label"]==1]["cancer_type"].value_counts().to_dict(),
    "cycles"          : ["2013-2014", "2015-2016", "2017-2018"],
    "missing_pct"     : {c: round(100*combined[c].isna().mean(),1) for c in feature_cols},
}
with open(OUTPUT_DIR / "nhanes_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved:")
print(f"  data/processed/nhanes_features.parquet")
print(f"  data/processed/nhanes_cancer_cases.parquet")
print(f"  data/processed/nhanes_controls.parquet")
print(f"  data/processed/nhanes_stats.json")
print(f"\nNext: python -m src.models.train_nhanes")
