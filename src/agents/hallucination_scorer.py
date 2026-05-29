"""
Formal Hallucination Scorer
============================
Defines and measures LLM clinical hallucination as:
  "A numeric claim in LLM text that cannot be matched to any known
   patient biomarker value within ±TOLERANCE% relative error."

Algorithm:
1. Extract all numeric values from LLM free-text using regex
2. For each extracted number, check if it is within ±TOL% of any
   ground-truth patient biomarker value
3. Hallucination = extracted number that does NOT match any biomarker
4. Rate = n_hallucinated_numbers / n_total_extracted_numbers

This is reproducible, deterministic, and requires no human labeling.
"""
import re
import sys
import json
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Configuration ────────────────────────────────────────────────────────────
TOLERANCE   = 0.15   # ±15% relative tolerance = acceptable paraphrase
MIN_VALUE   = 0.001  # ignore near-zero extracted numbers (common stopwords)
EXCLUDE_PAT = re.compile(
    r'\b(19|20)\d{2}\b'        # years like 2023
    r'|\b\d{1,3}(?:\.\d+)?%'  # percentages  (handled separately)
    r'|\b[1-9]\d?\b(?!\.\d)'  # single 1-99 ints that are likely not biomarkers
)

NUM_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from LLM text."""
    if not text or text.startswith("["):
        return []
    raw = NUM_PATTERN.findall(text)
    nums = []
    for r in raw:
        v = float(r)
        if v >= MIN_VALUE and v < 100000:    # exclude implausibly large values
            if not (v == int(v) and 1900 <= int(v) <= 2100):  # exclude years
                nums.append(v)
    return nums


def is_grounded(value: float, ground_truth_values: list[float],
                tolerance: float = TOLERANCE) -> bool:
    """Return True if 'value' matches any ground-truth value within tolerance."""
    for gt in ground_truth_values:
        if gt == 0:
            continue
        rel_err = abs(value - gt) / abs(gt)
        if rel_err <= tolerance:
            return True
    return False


def score_response(llm_text: str, patient_biomarkers: dict) -> dict:
    """
    Score one LLM response for hallucination.

    Parameters
    ----------
    llm_text : str
        Raw LLM output (any agent)
    patient_biomarkers : dict
        Dict of feature_name -> numeric_value for the actual patient

    Returns
    -------
    dict with keys:
        hallucination_rate    : float  [0, 1]
        n_extracted           : int    total numbers found in text
        n_hallucinated        : int    numbers not matching any biomarker
        n_grounded            : int    numbers matching a biomarker
        extracted_numbers     : list   all numbers extracted
        hallucinated_numbers  : list   numbers that are hallucinations
    """
    if not llm_text or llm_text.startswith("["):
        return {
            "hallucination_rate": 0.0,
            "n_extracted": 0, "n_hallucinated": 0, "n_grounded": 0,
            "extracted_numbers": [], "hallucinated_numbers": [],
            "note": "Empty or rate-limited response — excluded from scoring"
        }

    gt_values = [float(v) for v in patient_biomarkers.values()
                 if v is not None and not np.isnan(float(v) if isinstance(v, float) else 0)]

    extracted = extract_numbers(llm_text)
    if not extracted:
        return {
            "hallucination_rate": 0.0,
            "n_extracted": 0, "n_hallucinated": 0, "n_grounded": 0,
            "extracted_numbers": [], "hallucinated_numbers": [],
            "note": "No numeric values found in LLM output"
        }

    hallucinated = [v for v in extracted if not is_grounded(v, gt_values)]
    grounded     = [v for v in extracted if     is_grounded(v, gt_values)]
    rate         = len(hallucinated) / len(extracted)

    return {
        "hallucination_rate":  round(rate, 4),
        "n_extracted":         len(extracted),
        "n_hallucinated":      len(hallucinated),
        "n_grounded":          len(grounded),
        "extracted_numbers":   extracted,
        "hallucinated_numbers":hallucinated,
    }


def score_all_agents(record: dict, patient_biomarkers: dict) -> dict:
    """
    Score all 5 agent responses for one patient.

    Parameters
    ----------
    record : dict
        One entry from agent_results.json
    patient_biomarkers : dict
        Numeric biomarker values for this patient

    Returns
    -------
    dict: per-agent scores + aggregate
    """
    agent_keys = ["a1_biomarker", "a2_risk", "a3_differential",
                  "a4_evidence", "a5_triage"]
    per_agent = {}
    all_rates = []

    for key in agent_keys:
        text  = str(record.get(key, ""))
        score = score_response(text, patient_biomarkers)
        per_agent[key] = score
        if score["n_extracted"] > 0:
            all_rates.append(score["hallucination_rate"])

    return {
        "per_agent":            per_agent,
        "aggregate_rate":       round(np.mean(all_rates), 4) if all_rates else 0.0,
        "total_extracted":      sum(p["n_extracted"] for p in per_agent.values()),
        "total_hallucinated":   sum(p["n_hallucinated"] for p in per_agent.values()),
        "tolerance_pct":        int(TOLERANCE * 100),
        "algorithm":            "regex_numeric_extraction_with_15pct_relative_tolerance",
    }


# ─── CLI: score all records in agent_results.json ────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("data/processed/nhanes_features.parquet")
    FEAT = [c for c in df.columns if c not in
            ("seqn","cancer","cancer_type","ever_cancer","label","cycle","gender","ethnicity","age")]

    with open("results/agent_results.json") as f:
        records = json.load(f)

    print("FORMAL HALLUCINATION SCORING")
    print(f"  Algorithm: regex extraction + {int(TOLERANCE*100)}% relative tolerance")
    print(f"  n={len(records)} patients\n")
    print(f"{'Patient':>8} {'Cancer':>12} {'Extracted':>10} {'Halluc.':>9} {'Rate':>7}")
    print("-"*50)

    all_rates = []
    for r in records:
        row  = df.iloc[r["patient_idx"]].to_dict()
        bios = {k: row[k] for k in FEAT if k in row and isinstance(row.get(k), (int,float)) and not (isinstance(row.get(k),float) and np.isnan(row[k]))}
        res  = score_all_agents(r, bios)
        rate = res["aggregate_rate"]
        all_rates.append(rate)
        print(f"{r['patient_idx']:>8} {r['cancer_type']:>12} "
              f"{res['total_extracted']:>10} {res['total_hallucinated']:>9} {rate:>7.3f}")
        r["hallucination_formal"] = res

    print("-"*50)
    print(f"{'MEAN':>21} {sum(r['hallucination_formal']['total_extracted'] for r in records):>10} "
          f"{sum(r['hallucination_formal']['total_hallucinated'] for r in records):>9} "
          f"{np.mean(all_rates):>7.3f}")

    print(f"\nMethod: {records[0]['hallucination_formal']['algorithm']}")
    print(f"Tolerance: ±{TOLERANCE*100:.0f}% relative error = acceptable paraphrase")
    print(f"\nInterpretation:")
    print(f"  Rate=0.000 = all LLM numbers match actual patient values (no hallucination)")
    print(f"  Rate=1.000 = no LLM number matches any actual patient value (complete hallucination)")
    print(f"  Rate={np.mean(all_rates):.3f} = {np.mean(all_rates)*100:.1f}% of cited numbers are not grounded in patient data")

    with open("results/agent_results.json","w") as f:
        json.dump(records, f, indent=2)
    print("\nUpdated agent_results.json with formal hallucination scores.")
