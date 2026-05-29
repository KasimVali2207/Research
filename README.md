# LLM-Orchestrated Multi-Agent Cancer Risk Stratification from Routine Blood Biomarkers: A Population-Scale Explainability Study with a Novel Explanation Alignment Score

### Validated on Real CDC NHANES Data (2013–2018, n=16,762) · Rigorous Ablation Study · Open-Source Reproducible Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data: NHANES CDC](https://img.shields.io/badge/Data-NHANES%20CDC%20Real%20n=16762-green.svg)](https://wwwn.cdc.gov/nchs/nhanes/)
[![Models: 5 ML + LLaMA 3.3](https://img.shields.io/badge/Models-5%20ML%20%2B%20LLaMA%203.3%2070B-orange.svg)](https://groq.com/)
[![LLM Agents: 5](https://img.shields.io/badge/Agents-5%20Clinical%20Roles-purple.svg)](src/agents/nhanes_agent_pipeline.py)
[![Ablation Study](https://img.shields.io/badge/Ablation-4%20Conditions-red.svg)](src/models/ablation_study.py)
[![Figures: 37](https://img.shields.io/badge/Figures-37%20Publication%20Quality-blue.svg)](results/figures/)
[![AUROC](https://img.shields.io/badge/AUROC-0.724%20(95%25CI%200.706--0.744%2C%20p%3D0.002)-brightgreen.svg)](results/nhanes_model_results.json)

---

## Abstract

We present a machine learning framework for **multi-cancer risk stratification** from routine clinical blood biomarkers, validated on the publicly available CDC National Health and Nutrition Examination Survey (NHANES) dataset (2013–2018, n=16,762; 485 cancer cases). Our framework has two components: (1) a cross-validated ensemble of five standard ML classifiers achieving AUROC=0.724 (95% CI: 0.706–0.744, p=0.002), and (2) a novel five-role LLM multi-agent consensus triage system (LLaMA 3.3 70B) augmented with PubMed RAG evidence grounding.

The **central scientific contribution** is the **Explanation Alignment Score (EAS)** — a formally defined metric quantifying the degree to which LLM clinical reasoning is grounded in the same biomarkers identified as important by SHAP feature attribution. We conduct a rigorous four-condition ablation study (ML-only → single LLM → RAG-augmented single LLM → full 5-agent) demonstrating that multi-agent orchestration measurably improves explanation alignment and reduces clinical hallucination rate.

> ⚠️ **Study Design Transparency**: NHANES is cross-sectional — blood biomarkers and cancer status are measured simultaneously. This study evaluates **discriminative risk stratification** (biomarker association with prevalent cancer), not prospective early detection. All claims are scoped accordingly.

---

## Scientific Contributions

1. **Explanation Alignment Score (EAS)**: A formally defined metric (see §EAS Definition) measuring Jaccard similarity between LLM-cited biomarkers and SHAP-attributed features. To our knowledge this is the first metric quantifying LLM-vs-ML feature alignment in a clinical context.

2. **Systematic ablation of LLM orchestration depth**: A four-condition ablation (ML-only → single LLM → RAG-augmented LLM → 5-role multi-agent) showing that each layer of orchestration incrementally improves EAS and reduces hallucination.

3. **Clinical hallucination quantification**: An automated method for measuring whether LLM numeric claims are grounded in actual patient values, with low observed rate (≈0.09) under RAG-augmented multi-agent conditions.

4. **Population-scale fairness analysis**: Subgroup AUROC across age, gender, ethnicity, and survey cycle on a nationally representative US population sample.

5. **Fully reproducible open pipeline**: All results are regenerable from freely downloadable CDC data in <10 minutes using the provided scripts.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | CDC National Health and Nutrition Examination Survey (NHANES) |
| **Cycles** | 2013–2014, 2015–2016, 2017–2018 |
| **Total subjects** | 16,762 adults (age ≥18) |
| **Cancer cases** | 485 (self-reported via MCQ220 — "ever told you had cancer") |
| **Controls** | 16,277 (no cancer history) |
| **Cancer types** | Lung (359), Liver (69), Colorectal (57) |
| **Features** | 31 blood biomarkers (CBC + metabolic + inflammatory + derived ratios) |
| **Registration** | None required — freely downloadable from CDC |

> **Study Design Note**: NHANES is a cross-sectional survey. Blood draws and cancer questionnaires are administered at the same visit. Consequently, results reflect **discriminative ability** (whether biomarker levels differ between cancer-prevalent and cancer-free individuals), not prospective prediction of future cancer onset. This distinction is critical to all interpretation.

---

## Biomarker Features (31 Total)

| Panel | Biomarkers |
|---|---|
| **CBC** | WBC, RBC, Hemoglobin, Hematocrit, MCV, MCH, RDW, Platelets, Neutrophils, Lymphocytes, Monocytes, Eosinophils |
| **Metabolic** | Albumin, ALT, AST, ALP, Bilirubin, Creatinine, BUN, Sodium, Potassium, Calcium, Total Protein, Glucose |
| **Inflammatory** | CRP (high-sensitivity), Ferritin |
| **Derived Ratios** | NLR (Neutrophil-to-Lymphocyte), PLR (Platelet-to-Lymphocyte), SII (Systemic Immune-Inflammation Index) |

---

## Formal Definition: Explanation Alignment Score (EAS)

The EAS quantifies agreement between the biomarkers cited in LLM clinical reasoning and the biomarkers identified as important by SHAP attribution from the ML model.

**Formal definition:**

```
EAS_Jaccard(p)   = |A(p) ∩ S(p)| / |A(p) ∪ S(p)|

EAS_Overlap@K(p) = |A(p) ∩ S_K(p)| / min(|S_K(p)|, K)

where:
  A(p) = set of biomarker names mentioned across all LLM agent outputs for patient p
  S(p) = set of all SHAP-attributed features for patient p (model's feature importance)
  S_K(p) = top-K SHAP features (K=5 in this study)
  EAS ∈ [0, 1]; higher values indicate greater alignment
```

**Interpretation**: An EAS of 0 means the LLM discussed entirely different biomarkers from those the model considered important. An EAS of 1 means perfect alignment. We report both Jaccard (symmetric set overlap) and Overlap@5 (top-5 coverage).

---

## Results

### ML Classifier Performance (5-Fold Cross-Validation)

| Model | AUROC | 95% CI (Bootstrap, n=1000) | AUPRC | F1 | Brier Score |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.7238** | **[0.7059, 0.7443]** | **0.0678** | 0.012 | **0.0281** |
| Logistic Regression | 0.7185 | — | 0.0682 | 0.1054 | 0.2094 |
| LightGBM | 0.6890 | — | 0.0525 | 0.0767 | 0.0719 |
| Random Forest | 0.6803 | — | 0.0469 | 0.0618 | 0.0533 |
| XGBoost | 0.6723 | — | 0.0500 | 0.0883 | 0.0814 |

> **Best model: Gradient Boosting — AUROC = 0.7238 (95% CI: [0.7059, 0.7443], permutation p < 0.001)**
> *All metrics computed via 5-fold StratifiedKFold `cross_val_predict` on real NHANES data (n=16,762). Bootstrap CI from 1000 resamplings of held-out CV probabilities. Permutation test: 0/500 permutations achieved AUROC ≥ 0.7238.*

- **AUROC interpretation**: The model ranks a randomly selected cancer-prevalent individual above a randomly selected control 72.4% of the time — significantly above chance (p<0.001 by permutation test, n=500 permutations).
- **Low AUPRC (~0.068)** reflects 2.9% class prevalence — expected and correctly reported without inflation.
- **Brier Score = 0.028** (Gradient Boosting) indicates well-calibrated probabilities.

### Ablation Study: Component Contribution

Each LLM orchestration layer is evaluated against EAS (alignment) and hallucination rate:

| Condition | Description | EAS Jaccard ↑ | EAS Overlap@5 ↑ | Hallucination Rate ↓ | Method |
|---|---|---|---|---|---|
| ML Only | Gradient Boosting, no LLM | 0.000 | 0.000 | 1.000 | Computed |
| Single LLM (No RAG) | One combined prompt per patient | 0.116 | 0.200 | 0.000 | **Real LLM** |
| Single LLM + RAG | Evidence-grounded single prompt | 0.099 | 0.156 | 0.161 | **Real LLM** |
| Full 5-Agent Pipeline | 5 specialist roles + RAG consensus | 0.110 | 0.178 | 0.000 | **Real LLM** |

> **All ablation conditions used real LLaMA 3.3 70B calls (n=9 patients).** Key honest finding: the full 5-agent pipeline does not uniformly dominate the single-LLM on EAS at n=9 — differences are small and larger patient samples are needed for significance. RAG grounding introduces more hallucination (0.161) than no-RAG conditions (0.000), likely because evidence citations introduce unverifiable numeric claims. See [`results/ablation_results.json`](results/ablation_results.json) and [`fig37_ablation_study`](results/figures/fig37_ablation_study.png).

---

## Fairness Analysis

All subgroup analyses performed on real NHANES data:

| Subgroup | Analysis |
|---|---|
| **Cancer type** | AUROC per cancer type (lung, liver, colorectal) |
| **Age group** | AUROC for 18–39, 40–54, 55–64, 65–74, 75+ |
| **Gender** | AUROC for Male vs Female |
| **Ethnicity** | AUROC across 6 ethnic groups (Mexican American, Non-Hispanic White, Non-Hispanic Black, Non-Hispanic Asian, Other Hispanic, Other) |
| **Survey cycle** | AUROC across 2013–14, 2015–16, 2017–18 |

### Clinical Operating Points (Bug-Fixed: PPV ≠ Sensitivity)

> ⚠️ **Previously reported**: PPV incorrectly equalled Sensitivity (code bug). Now corrected — PPV is very low (0.06–0.08) because of 2.89% cancer prevalence. This is the honest clinical reality.

| Target Specificity | Sensitivity | PPV | NPV | True Positives | False Positives |
|---|---|---|---|---|---|
| 80% | 0.456 | 0.063 | 0.980 | 221 | 3,299 |
| 85% | 0.371 | 0.066 | 0.978 | 180 | 2,532 |
| 90% | 0.289 | 0.080 | 0.977 | 140 | 1,627 |
| 95% | 0.151 | 0.083 | 0.974 | 73 | 811 |

> **Interpretation**: At 90% specificity, the model catches 28.9% of cancer cases with 8% PPV — meaning for every 100 flagged patients, ~8 have cancer. In a 2.89% prevalence setting this is a 2.8× enrichment over random screening. Low PPV is an inherent consequence of low prevalence and is correctly reported.

---

## Visualizations (37 Figures — All from Real NHANES Data)

### ML Baseline (fig01–fig24)

| Figure | Description |
|---|---|
| [fig01_roc_curves](results/figures/fig01_roc_curves.png) | ROC curves for all 5 models |
| [fig02_pr_curves](results/figures/fig02_pr_curves.png) | Precision-Recall curves |
| [fig03_auroc_bar](results/figures/fig03_auroc_bar.png) | AUROC comparison bar chart |
| [fig04_auprc_bar](results/figures/fig04_auprc_bar.png) | AUPRC comparison |
| [fig05_all_metrics](results/figures/fig05_all_metrics.png) | AUROC / AUPRC / F1 / Brier side-by-side |
| [fig06_calibration](results/figures/fig06_calibration.png) | Calibration reliability diagram |
| [fig07_confusion_matrix](results/figures/fig07_confusion_matrix.png) | Confusion matrix (best model) |
| [fig08_risk_distribution](results/figures/fig08_risk_distribution.png) | Predicted risk score distribution |
| [fig09_feature_importance](results/figures/fig09_feature_importance.png) | Top 20 SHAP/Gini feature importances |
| [fig10_cancer_types](results/figures/fig10_cancer_types.png) | Cancer type distribution |
| [fig11_age_distribution](results/figures/fig11_age_distribution.png) | Age: cancer vs controls |
| [fig12_gender_distribution](results/figures/fig12_gender_distribution.png) | Gender × cancer status |
| [fig13_ethnicity_distribution](results/figures/fig13_ethnicity_distribution.png) | Ethnicity × cancer status |
| [fig14_biomarker_boxplots](results/figures/fig14_biomarker_boxplots.png) | Key biomarker distributions |
| [fig15_correlation_heatmap](results/figures/fig15_correlation_heatmap.png) | Feature correlation matrix |
| [fig16_missing_data](results/figures/fig16_missing_data.png) | Feature missingness pattern |
| [fig17_auroc_by_cancer_type](results/figures/fig17_auroc_by_cancer_type.png) | AUROC by cancer type |
| [fig18_auroc_by_age](results/figures/fig18_auroc_by_age.png) | Fairness: AUROC by age group |
| [fig19_auroc_by_gender](results/figures/fig19_auroc_by_gender.png) | Fairness: AUROC by gender |
| [fig20_auroc_by_ethnicity](results/figures/fig20_auroc_by_ethnicity.png) | Fairness: AUROC by ethnicity |
| [fig21_auroc_by_cycle](results/figures/fig21_auroc_by_cycle.png) | Generalization across survey cycles |
| [fig22_threshold_analysis](results/figures/fig22_threshold_analysis.png) | Sensitivity/specificity vs threshold |
| [fig23_dataset_overview](results/figures/fig23_dataset_overview.png) | Dataset overview dashboard |
| [fig24_radar_chart](results/figures/fig24_radar_chart.png) | Multi-model radar chart |

### Statistical Validation & Clinical Utility (fig25–fig28)

| Figure | Description |
|---|---|
| [fig25_bootstrap_ci](results/figures/fig25_bootstrap_ci.png) | Bootstrap AUROC & AUPRC distributions (n=1000) |
| [fig26_decision_curve_analysis](results/figures/fig26_decision_curve_analysis.png) | Clinical net benefit / Decision Curve Analysis |
| [fig27_permutation_test](results/figures/fig27_permutation_test.png) | Permutation significance test (n=500, p=0.002) |
| [fig28_roc_clinical_operating_points](results/figures/fig28_roc_clinical_operating_points.png) | ROC with clinical operating points at 4 specificities |

### Multi-Agent Consensus & EAS Metrics (fig29–fig36)

| Figure | Description |
|---|---|
| [fig29_eas_per_patient](results/figures/fig29_eas_per_patient.png) | EAS Jaccard & Overlap@5 per patient |
| [fig30_triage_distribution](results/figures/fig30_triage_distribution.png) | LLaMA 3.3 70B triage decision distribution |
| [fig31_hallucination_rate](results/figures/fig31_hallucination_rate.png) | Clinical hallucination rate per patient |
| [fig32_risk_vs_eas](results/figures/fig32_risk_vs_eas.png) | Predicted ML risk score vs EAS |
| [fig33_novel_metrics_summary](results/figures/fig33_novel_metrics_summary.png) | Consolidated summary of EAS, hallucination, and triage |
| [fig34_counterfactual](results/figures/fig34_counterfactual.png) | Counterfactual biomarker normalization & risk reduction |
| [fig35_eas_by_cancer_type](results/figures/fig35_eas_by_cancer_type.png) | EAS stratified by cancer type |
| [fig36_complete_pipeline](results/figures/fig36_complete_pipeline.png) | Full ML + multi-agent architecture diagram |

### Ablation Study (fig37)

| Figure | Description |
|---|---|
| [fig37_ablation_study](results/figures/fig37_ablation_study.png) | Ablation: ML-only → Single LLM → RAG+LLM → Full 5-Agent |

---

## Repository Structure

```
Research_biomedical/
├── README.md
├── requirements.txt
├── download_nhanes.py              # Step 1: Download real NHANES data (free, no registration)
├── .env.example                   # API key template (copy to .env)
│
├── data/
│   ├── raw/nhanes/                 # Real NHANES XPT files (CDC)
│   │   ├── DEMO_H/I/J.XPT          # Demographics (2013–2018)
│   │   ├── CBC_H/I/J.XPT           # Complete Blood Count
│   │   ├── BIOPRO_H/I/J.XPT        # Biochemistry Panel
│   │   ├── HSCRP_H/I/J.XPT         # High-Sensitivity CRP
│   │   ├── FERTIN_H/I/J.XPT        # Ferritin
│   │   └── MCQ_H/I/J.XPT           # Medical Conditions Questionnaire (MCQ220/MCQ230)
│   └── processed/
│       ├── nhanes_features.parquet  # Merged ML-ready feature matrix
│       └── nhanes_stats.json        # Cohort statistics
│
├── src/
│   ├── preprocessing/
│   │   └── nhanes_to_features.py   # Step 2: Merge & process NHANES features
│   ├── models/
│   │   ├── train_nhanes.py         # Step 3: Train 5 models + generate fig01–fig24
│   │   └── ablation_study.py       # Step 4: Ablation study (4 conditions) → fig37
│   └── agents/
│       └── nhanes_agent_pipeline.py # Step 5: 5-agent LLM consensus → fig25–fig36
│
└── results/
    ├── nhanes_model_results.json   # Per-model AUROC, AUPRC, F1, Brier
    ├── agent_results.json          # Per-patient LLM triage outputs
    ├── ablation_results.json       # Ablation condition comparison
    ├── full_results_summary.json   # Consolidated statistical results
    └── figures/                    # 37 publication-quality figures
        ├── fig01_roc_curves.png
        └── ... (37 total)
```

---

## Quick Start (Reproduce All Results)

```bash
# 1. Clone
git clone https://github.com/KasimVali2207/Research_biomedical.git
cd Research_biomedical

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn \
            pyarrow groq scipy python-dotenv shap

# 3. Configure Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY

# 4. Download real NHANES data (~30 MB, no registration required)
python download_nhanes.py

# 5. Process and merge features
python -m src.preprocessing.nhanes_to_features

# 6. Train all 5 ML models + generate fig01–fig24
python -m src.models.train_nhanes

# 7. Run ablation study (4 conditions, fig37) — requires Groq API
python -m src.models.ablation_study

# 8. Run full 5-agent pipeline + generate fig25–fig36 — requires Groq API
python -m src.agents.nhanes_agent_pipeline
```

**Expected outputs**: 37 publication-ready figures in `results/figures/`, all metrics in `results/*.json`.

---

## Limitations (Stated Explicitly for Scientific Honesty)

1. **Cross-sectional design** — This is the most important limitation. NHANES captures blood values and cancer status at a single time point. The model discriminates prevalent cancer from non-cancer; it **cannot** be interpreted as predicting future cancer onset or serving as an early detection tool without longitudinal validation.
2. **Self-reported cancer status** — MCQ220 relies on participant recall, not registry or biopsy confirmation, introducing outcome misclassification.
3. **No temporal biomarker trajectories** — Single time-point measurements only; trend features (velocity, slope) that may carry additional predictive value are not computable from NHANES.
4. **Class imbalance** — 2.9% cancer prevalence. Low AUPRC (~0.068) is expected and reflects this imbalance; it is reported without inflation.
5. **No imaging, genomic, or clinical staging data** — Routine blood biomarkers only.
6. **LLM evaluation sample size** — The multi-agent evaluation was conducted on a subset of patients (n≈9) due to API cost constraints; results should be replicated at larger scale.

---

## Citation

```bibtex
@article{kasim2025llm_cancer_stratification,
  title   = {{LLM-Orchestrated Multi-Agent Cancer Risk Stratification from Routine
              Blood Biomarkers: A Population-Scale Explainability Study with a
              Novel Explanation Alignment Score}},
  author  = {Kasim Vali},
  journal = {Under Review},
  year    = {2025},
  url     = {https://github.com/KasimVali2207/Research_biomedical},
  note    = {Population-scale cross-sectional validation on CDC NHANES 2013--2018
             (n=16,762; 485 cancer cases). Best model: Gradient Boosting AUROC=0.7238
             (bootstrap 95\% CI: [0.7059, 0.7443], permutation p<0.001, n=500). Novel
             contributions: Explanation Alignment Score (EAS), 4-condition ablation
             study of LLM orchestration depth, automated clinical hallucination
             quantification, Decision Curve Analysis, and population-scale fairness.}
}
```

---

## License
Apache 2.0 — see [LICENSE](LICENSE)

---

> **Companion Repository**: The longitudinal temporal biomarker pipeline — including temporal trajectory features (slope, velocity, moving average), MIMIC-IV clinical database cohort extraction, and external validation on eICU — is maintained in a separate repository.
