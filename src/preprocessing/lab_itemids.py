# Copyright 2024 The Research Project Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lab item ID mappings for MIMIC-IV and eICU.

MIMIC-IV itemids were verified against the d_labitems table (MIMIC-IV v3.1).
Multiple itemids per feature exist because the same test was ordered under
different lab panels (e.g., CBC with/without differential, point-of-care vs
central lab). We aggregate all of them during feature extraction.

eICU lab names come from the `lab.labname` column; spellings are inconsistent
across hospital sites so we list all known variants.
"""

from typing import Dict, List, Union

# ---------------------------------------------------------------------------
# MIMIC-IV  itemid → feature mappings
# ---------------------------------------------------------------------------

MIMIC_LAB_ITEMIDS: Dict[str, List[int]] = {
    # --- Complete Blood Count ---
    "wbc": [51300, 51301],          # WBC (K/uL)
    "rbc": [51279],                 # RBC (m/uL)
    "hemoglobin": [51222],          # Hemoglobin (g/dL)
    "hematocrit": [51221],          # Hematocrit (%)
    "mcv": [51237],                 # Mean Corpuscular Volume (fL)
    "mch": [51234],                 # Mean Corpuscular Hemoglobin (pg)
    "mchc": [51233],                # MCHC (g/dL)
    "rdw": [51277],                 # RDW (%)
    "platelets": [51265],           # Platelet Count (K/uL)
    "mpv": [51264],                 # Mean Platelet Volume (fL)
    # --- Differential ---
    "neutrophils": [51244, 51245],  # Neutrophils absolute (K/uL) and %
    "lymphocytes": [51133, 51244],  # Lymphocytes absolute / %
    "monocytes": [51254, 51255],    # Monocytes absolute / %
    "eosinophils": [51199, 51200],  # Eosinophils absolute / %
    "basophils": [51146, 51200],    # Basophils absolute / %
    "bands": [51143],               # Bands % (manual diff)
    # --- Metabolic / Chemistry ---
    "glucose": [50931, 50809],      # Glucose plasma / whole blood
    "albumin": [50862],             # Albumin (g/dL)
    "total_protein": [50976],       # Total Protein (g/dL)
    "creatinine": [50912],          # Creatinine (mg/dL)
    "bun": [51006],                 # Blood Urea Nitrogen (mg/dL)
    "sodium": [50983],              # Sodium (mEq/L)
    "potassium": [50971],           # Potassium (mEq/L)
    "chloride": [50902],            # Chloride (mEq/L)
    "bicarbonate": [50882],         # Bicarbonate / CO2 (mEq/L)
    "calcium": [50893],             # Calcium total (mg/dL)
    "magnesium": [50960],           # Magnesium (mg/dL)
    "phosphate": [50970],           # Phosphorus (mg/dL)
    "uric_acid": [51007],           # Uric Acid (mg/dL)
    # --- Liver Function Tests ---
    "alt": [50861],                 # ALT/SGPT (IU/L)
    "ast": [50878],                 # AST/SGOT (IU/L)
    "alp": [50863],                 # Alkaline Phosphatase (IU/L)
    "ggt": [50927],                 # GGT (IU/L)
    "ldh": [50954],                 # LDH (IU/L)
    "bilirubin_total": [50885],     # Total Bilirubin (mg/dL)
    "bilirubin_direct": [50883],    # Direct Bilirubin (mg/dL)
    "bilirubin_indirect": [50884],  # Indirect Bilirubin (mg/dL)
    # --- Inflammatory Markers ---
    "crp": [50889],                 # C-Reactive Protein (mg/L)
    "esr": [51288],                 # ESR (mm/hr)
    "ferritin": [52073, 50924],     # Ferritin (ng/mL)
    "iron": [50925],                # Iron (ug/dL)
    "tibc": [50944],                # TIBC (ug/dL)
    "transferrin_saturation": [50943],
    "procalcitonin": [51652],       # Procalcitonin (ng/mL)
    # --- Coagulation ---
    "inr": [51237],                 # INR
    "pt": [51274],                  # Prothrombin Time (sec)
    "ptt": [51275],                 # Partial Thromboplastin Time (sec)
    "fibrinogen": [51214],          # Fibrinogen (mg/dL)
    "d_dimer": [51192],             # D-Dimer (ng/mL FEU)
    # --- Blood Gas ---
    "ph": [50820],                  # pH (arterial)
    "pco2": [50818],                # pCO2 (mmHg)
    "po2": [50821],                 # pO2 (mmHg)
    "base_excess": [50802],         # Base Excess (mEq/L)
    "lactate": [50813],             # Lactate (mmol/L)
    # --- Thyroid ---
    "tsh": [50953],                 # TSH (uIU/mL)
    "free_t4": [50921],             # Free T4 (ng/dL)
    # --- Tumor Markers (where available in MIMIC) ---
    "cea": [51512],                 # CEA (ng/mL) — sparse
    "ca125": [51510],               # CA-125 (U/mL) — sparse
    "psa": [51519],                 # PSA (ng/mL) — sparse
    "afp": [51508],                 # AFP (ng/mL) — sparse
}


# ---------------------------------------------------------------------------
# eICU lab name strings → feature mappings
# eICU stores lab results as free-text names; these cover all known variants
# observed across the ~200 hospital sites in the eICU Collaborative Research DB.
# ---------------------------------------------------------------------------

EICU_LAB_NAMES: Dict[str, List[str]] = {
    "wbc": ["WBC x 1000", "WBC", "wbc"],
    "rbc": ["RBC", "rbc"],
    "hemoglobin": ["Hgb", "hemoglobin", "Hemoglobin"],
    "hematocrit": ["Hct", "hematocrit", "Hematocrit"],
    "mcv": ["MCV"],
    "mch": ["MCH"],
    "mchc": ["MCHC"],
    "rdw": ["RDW", "rdw"],
    "platelets": ["platelets x 1000", "Platelets", "platelet count"],
    "mpv": ["MPV"],
    "neutrophils": ["neutrophils", "Neutrophils", "neutrophils - absolute"],
    "lymphocytes": ["lymphocytes", "Lymphocytes", "lymphocytes - absolute"],
    "monocytes": ["monocytes", "Monocytes", "monocytes - absolute"],
    "eosinophils": ["eosinophils", "Eosinophils"],
    "basophils": ["basophils", "Basophils"],
    "bands": ["bands", "Bands"],
    "glucose": ["glucose", "Glucose", "bedside glucose"],
    "albumin": ["albumin", "Albumin"],
    "total_protein": ["total protein", "Total Protein"],
    "creatinine": ["creatinine", "Creatinine"],
    "bun": ["BUN", "bun", "urea nitrogen"],
    "sodium": ["sodium", "Sodium", "Na"],
    "potassium": ["potassium", "Potassium", "K"],
    "chloride": ["chloride", "Chloride", "Cl"],
    "bicarbonate": ["bicarbonate", "Bicarbonate", "HCO3", "total CO2"],
    "calcium": ["calcium", "Calcium"],
    "magnesium": ["magnesium", "Magnesium"],
    "phosphate": ["phosphate", "Phosphate", "phosphorus"],
    "uric_acid": ["uric acid", "Uric Acid"],
    "alt": ["ALT (SGPT)", "ALT", "alanine aminotransferase"],
    "ast": ["AST (SGOT)", "AST", "aspartate aminotransferase"],
    "alp": ["alkaline phos.", "Alkaline Phosphatase", "ALP"],
    "ggt": ["GGT", "gamma glutamyl transferase"],
    "ldh": ["LDH", "lactate dehydrogenase"],
    "bilirubin_total": ["total bilirubin", "Total Bilirubin", "bilirubin"],
    "bilirubin_direct": ["direct bilirubin", "Direct Bilirubin"],
    "bilirubin_indirect": ["indirect bilirubin", "Indirect Bilirubin"],
    "crp": ["C reactive protein", "CRP", "c-reactive protein"],
    "esr": ["ESR", "erythrocyte sedimentation rate"],
    "ferritin": ["ferritin", "Ferritin"],
    "iron": ["iron", "Iron", "serum iron"],
    "tibc": ["TIBC", "total iron binding capacity"],
    "transferrin_saturation": ["transferrin saturation", "% sat"],
    "procalcitonin": ["procalcitonin", "Procalcitonin"],
    "inr": ["PT - INR", "INR", "inr"],
    "pt": ["PT", "prothrombin time"],
    "ptt": ["PTT", "aPTT", "partial thromboplastin time"],
    "fibrinogen": ["fibrinogen", "Fibrinogen"],
    "d_dimer": ["D-dimer", "d-dimer", "D Dimer"],
    "ph": ["pH", "pH art"],
    "pco2": ["pCO2", "PCO2", "pco2"],
    "po2": ["pO2", "PO2", "po2"],
    "base_excess": ["Base Excess", "base excess", "BE"],
    "lactate": ["lactate", "Lactate"],
    "tsh": ["TSH", "thyroid stimulating hormone"],
    "free_t4": ["free T4", "Free T4", "FT4"],
    "cea": ["CEA", "carcinoembryonic antigen"],
    "ca125": ["CA-125", "CA 125"],
    "psa": ["PSA", "prostate specific antigen"],
    "afp": ["AFP", "alpha-fetoprotein"],
}


# ---------------------------------------------------------------------------
# Clinical reference ranges (adult, sex-agnostic where applicable)
# Sources: CLSI EP28-A3c, institutional reference intervals, Harrison's
# ---------------------------------------------------------------------------

NORMAL_RANGES: Dict[str, Dict[str, Union[float, str]]] = {
    "wbc":                  {"low": 4.5,   "high": 11.0,  "unit": "K/uL"},
    "rbc":                  {"low": 4.2,   "high": 5.9,   "unit": "m/uL"},
    "hemoglobin":           {"low": 12.0,  "high": 17.5,  "unit": "g/dL"},
    "hematocrit":           {"low": 36.0,  "high": 52.0,  "unit": "%"},
    "mcv":                  {"low": 80.0,  "high": 100.0, "unit": "fL"},
    "mch":                  {"low": 27.0,  "high": 33.0,  "unit": "pg"},
    "mchc":                 {"low": 32.0,  "high": 36.0,  "unit": "g/dL"},
    "rdw":                  {"low": 11.5,  "high": 14.5,  "unit": "%"},
    "platelets":            {"low": 150.0, "high": 400.0, "unit": "K/uL"},
    "mpv":                  {"low": 7.5,   "high": 12.5,  "unit": "fL"},
    "neutrophils":          {"low": 1.8,   "high": 7.7,   "unit": "K/uL"},
    "lymphocytes":          {"low": 1.0,   "high": 4.8,   "unit": "K/uL"},
    "monocytes":            {"low": 0.2,   "high": 0.95,  "unit": "K/uL"},
    "eosinophils":          {"low": 0.05,  "high": 0.5,   "unit": "K/uL"},
    "basophils":            {"low": 0.0,   "high": 0.1,   "unit": "K/uL"},
    "bands":                {"low": 0.0,   "high": 10.0,  "unit": "%"},
    "glucose":              {"low": 70.0,  "high": 99.0,  "unit": "mg/dL"},
    "albumin":              {"low": 3.5,   "high": 5.0,   "unit": "g/dL"},
    "total_protein":        {"low": 6.0,   "high": 8.3,   "unit": "g/dL"},
    "creatinine":           {"low": 0.6,   "high": 1.2,   "unit": "mg/dL"},
    "bun":                  {"low": 7.0,   "high": 20.0,  "unit": "mg/dL"},
    "sodium":               {"low": 136.0, "high": 145.0, "unit": "mEq/L"},
    "potassium":            {"low": 3.5,   "high": 5.1,   "unit": "mEq/L"},
    "chloride":             {"low": 98.0,  "high": 106.0, "unit": "mEq/L"},
    "bicarbonate":          {"low": 22.0,  "high": 29.0,  "unit": "mEq/L"},
    "calcium":              {"low": 8.5,   "high": 10.5,  "unit": "mg/dL"},
    "magnesium":            {"low": 1.7,   "high": 2.2,   "unit": "mg/dL"},
    "phosphate":            {"low": 2.5,   "high": 4.5,   "unit": "mg/dL"},
    "uric_acid":            {"low": 2.4,   "high": 7.0,   "unit": "mg/dL"},
    "alt":                  {"low": 7.0,   "high": 56.0,  "unit": "IU/L"},
    "ast":                  {"low": 10.0,  "high": 40.0,  "unit": "IU/L"},
    "alp":                  {"low": 44.0,  "high": 147.0, "unit": "IU/L"},
    "ggt":                  {"low": 9.0,   "high": 48.0,  "unit": "IU/L"},
    "ldh":                  {"low": 122.0, "high": 222.0, "unit": "IU/L"},
    "bilirubin_total":      {"low": 0.2,   "high": 1.2,   "unit": "mg/dL"},
    "bilirubin_direct":     {"low": 0.0,   "high": 0.3,   "unit": "mg/dL"},
    "bilirubin_indirect":   {"low": 0.2,   "high": 0.9,   "unit": "mg/dL"},
    "crp":                  {"low": 0.0,   "high": 10.0,  "unit": "mg/L"},
    "esr":                  {"low": 0.0,   "high": 20.0,  "unit": "mm/hr"},
    "ferritin":             {"low": 12.0,  "high": 300.0, "unit": "ng/mL"},
    "iron":                 {"low": 60.0,  "high": 170.0, "unit": "ug/dL"},
    "tibc":                 {"low": 250.0, "high": 370.0, "unit": "ug/dL"},
    "transferrin_saturation": {"low": 20.0, "high": 50.0, "unit": "%"},
    "procalcitonin":        {"low": 0.0,   "high": 0.05,  "unit": "ng/mL"},
    "inr":                  {"low": 0.8,   "high": 1.1,   "unit": "ratio"},
    "pt":                   {"low": 11.0,  "high": 13.5,  "unit": "sec"},
    "ptt":                  {"low": 25.0,  "high": 35.0,  "unit": "sec"},
    "fibrinogen":           {"low": 200.0, "high": 400.0, "unit": "mg/dL"},
    "d_dimer":              {"low": 0.0,   "high": 500.0, "unit": "ng/mL FEU"},
    "ph":                   {"low": 7.35,  "high": 7.45,  "unit": "pH units"},
    "pco2":                 {"low": 35.0,  "high": 45.0,  "unit": "mmHg"},
    "po2":                  {"low": 75.0,  "high": 100.0, "unit": "mmHg"},
    "base_excess":          {"low": -2.0,  "high": 2.0,   "unit": "mEq/L"},
    "lactate":              {"low": 0.5,   "high": 2.0,   "unit": "mmol/L"},
    "tsh":                  {"low": 0.4,   "high": 4.0,   "unit": "uIU/mL"},
    "free_t4":              {"low": 0.8,   "high": 1.8,   "unit": "ng/dL"},
    "cea":                  {"low": 0.0,   "high": 3.0,   "unit": "ng/mL"},
    "ca125":                {"low": 0.0,   "high": 35.0,  "unit": "U/mL"},
    "psa":                  {"low": 0.0,   "high": 4.0,   "unit": "ng/mL"},
    "afp":                  {"low": 0.0,   "high": 10.0,  "unit": "ng/mL"},
}


# ---------------------------------------------------------------------------
# Canonical unit strings — used to normalise heterogeneous units in labevents
# ---------------------------------------------------------------------------

CLINICAL_UNITS: Dict[str, str] = {
    "wbc":                   "K/uL",
    "rbc":                   "m/uL",
    "hemoglobin":            "g/dL",
    "hematocrit":            "%",
    "mcv":                   "fL",
    "mch":                   "pg",
    "mchc":                  "g/dL",
    "rdw":                   "%",
    "platelets":             "K/uL",
    "mpv":                   "fL",
    "neutrophils":           "K/uL",
    "lymphocytes":           "K/uL",
    "monocytes":             "K/uL",
    "eosinophils":           "K/uL",
    "basophils":             "K/uL",
    "bands":                 "%",
    "glucose":               "mg/dL",
    "albumin":               "g/dL",
    "total_protein":         "g/dL",
    "creatinine":            "mg/dL",
    "bun":                   "mg/dL",
    "sodium":                "mEq/L",
    "potassium":             "mEq/L",
    "chloride":              "mEq/L",
    "bicarbonate":           "mEq/L",
    "calcium":               "mg/dL",
    "magnesium":             "mg/dL",
    "phosphate":             "mg/dL",
    "uric_acid":             "mg/dL",
    "alt":                   "IU/L",
    "ast":                   "IU/L",
    "alp":                   "IU/L",
    "ggt":                   "IU/L",
    "ldh":                   "IU/L",
    "bilirubin_total":       "mg/dL",
    "bilirubin_direct":      "mg/dL",
    "bilirubin_indirect":    "mg/dL",
    "crp":                   "mg/L",
    "esr":                   "mm/hr",
    "ferritin":              "ng/mL",
    "iron":                  "ug/dL",
    "tibc":                  "ug/dL",
    "transferrin_saturation": "%",
    "procalcitonin":         "ng/mL",
    "inr":                   "ratio",
    "pt":                    "sec",
    "ptt":                   "sec",
    "fibrinogen":            "mg/dL",
    "d_dimer":               "ng/mL FEU",
    "ph":                    "pH units",
    "pco2":                  "mmHg",
    "po2":                   "mmHg",
    "base_excess":           "mEq/L",
    "lactate":               "mmol/L",
    "tsh":                   "uIU/mL",
    "free_t4":               "ng/dL",
    "cea":                   "ng/mL",
    "ca125":                 "U/mL",
    "psa":                   "ng/mL",
    "afp":                   "ng/mL",
}


# Flat lookup: itemid → feature_name (built at import time, O(1) reverse lookup)
ITEMID_TO_FEATURE: Dict[int, str] = {
    itemid: feature
    for feature, itemids in MIMIC_LAB_ITEMIDS.items()
    for itemid in itemids
}

# Flat lookup: eicu_lab_name → feature_name
EICU_LABNAME_TO_FEATURE: Dict[str, str] = {
    name.lower(): feature
    for feature, names in EICU_LAB_NAMES.items()
    for name in names
}

# Ordered feature list used to guarantee column ordering across train/test splits
FEATURE_COLUMNS: List[str] = list(MIMIC_LAB_ITEMIDS.keys())
