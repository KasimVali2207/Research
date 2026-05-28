# Multi-Cancer Detection from Routine Blood Biomarkers
### Machine Learning Study on Real NHANES Data (CDC, n=16,762)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data: NHANES CDC](https://img.shields.io/badge/Data-NHANES%20CDC%20Real-green.svg)](https://wwwn.cdc.gov/nchs/nhanes/)
[![Models: 5 ML](https://img.shields.io/badge/Models-5%20ML%20Classifiers-orange.svg)]()
[![Figures: 24](https://img.shields.io/badge/Figures-24%20Publication%20Quality-blue.svg)]()

---

## About This Repository

This repository contains a **fully realized ML and LLM-augmented study** for detecting cancer (colorectal, lung, liver) from routine blood panel biomarkers, validated on **real, publicly available CDC NHANES data** (2013–2018, n=16,762).

### Key Highlights of This Study
- **Machine Learning Core**: Trains and evaluates **5 standard ML classifiers** (Gradient Boosting, Random Forest, XGBoost, LightGBM, Logistic Regression) using robust 5-fold cross-validation.
- **LLM Multi-Agent Orchestration**: Implements a novel **5-Agent LLM Consensus Triage System** powered by LLaMA 3.3 70B (Biomarker, Risk Explanation, Differential Diagnosis, PubMed RAG Evidence Grounding, and Clinical Triage agents) specifically adapted to NHANES cross-sectional data.
- **Top-Journal Novelty Suite**: Includes advanced validations such as **Bootstrap 95% Confidence Intervals**, **Decision Curve Analysis (DCA)**, **Permutation significance tests**, **Hallucination Rate scoring**, and a novel **Explanation Alignment Score (EAS)** matching LLM clinical reasoning with SHAP feature importances.
- **Publication-Ready Figures**: Generates **36 publication-quality figures** (fig01–fig36) representing standard machine learning metrics, clinical utility curves, multi-agent agreements, and fairness across demographics (age, gender, ethnicity, cycle).
- **100% Reproducible**: Freely downloadable CDC NHANES raw data can be processed and analyzed in under 10 minutes.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | CDC National Health and Nutrition Examination Survey (NHANES) |
| **Cycles** | 2013–2014, 2015–2016, 2017–2018 |
| **Total subjects** | 16,762 adults (age ≥18) |
| **Cancer cases** | 485 (confirmed via MCQ220 — "ever told you had cancer") |
| **Controls** | 16,277 (no cancer history) |
| **Cancer types** | Lung (359), Liver (69), Colorectal (57) |
| **Features** | 31 blood biomarkers (CBC + metabolic + inflammatory) |
| **Registration** | None required — freely downloadable from CDC |
| **Download link** | [wwwn.cdc.gov/nchs/nhanes/](https://wwwn.cdc.gov/nchs/nhanes/) |

### Important Note on Study Design
NHANES is a **cross-sectional survey** — blood tests and cancer status are measured
at the same time point. This means results reflect **discriminative ability**
(can blood tests distinguish cancer vs no-cancer at a given moment), not
prospective early detection. This is clearly stated for scientific accuracy.

---

## Biomarker Features Used

| Panel | Biomarkers |
|---|---|
| **CBC** | WBC, RBC, Hemoglobin, Hematocrit, MCV, MCH, RDW, Platelets, Neutrophils, Lymphocytes, Monocytes, Eosinophils |
| **Metabolic** | Albumin, ALT, AST, ALP, Bilirubin, Creatinine, BUN, Sodium, Potassium, Calcium, Total Protein, Glucose |
| **Inflammatory** | CRP (high-sensitivity), Ferritin |
| **Derived ratios** | NLR (Neutrophil-to-Lymphocyte Ratio), PLR (Platelet-to-Lymphocyte Ratio), SII (Systemic Immune-Inflammation Index) |

---

## Results (5-Fold Cross-Validation, Real NHANES Data)

### Model Performance

| Model | AUROC | AUPRC | F1 | Brier Score |
|---|---|---|---|---|
| **Gradient Boosting** | **0.724** | **0.068** | 0.012 | **0.028** |
| Logistic Regression | 0.718 | 0.068 | 0.105 | 0.209 |
| Random Forest | 0.697 | 0.054 | 0.102 | 0.114 |
| LightGBM | 0.680 | 0.049 | 0.069 | 0.062 |
| XGBoost | 0.674 | 0.052 | 0.097 | 0.087 |

> **Best model: Gradient Boosting — AUROC = 0.724**
> Achieved using only routine blood panel tests available from any standard lab.

### Interpretation
- AUROC = 0.724 means the model correctly ranks a randomly selected cancer patient
  above a randomly selected control 72.4% of the time — significantly above chance (0.5)
- Low AUPRC (~0.068) reflects the strong class imbalance (485 cancer / 16,277 controls)
  and is expected and correctly reported
- Brier Score = 0.028 (Gradient Boosting) indicates excellent probability calibration

---

## Fairness Analysis

All subgroup analyses performed on real NHANES data:

| Subgroup | Analysis |
|---|---|
| **Cancer type** | AUROC per cancer (lung, liver, colorectal) |
| **Age group** | AUROC for 18–39, 40–54, 55–64, 65–74, 75+ |
| **Gender** | AUROC for Male vs Female |
| **Ethnicity** | AUROC for 6 ethnic groups (Mexican American, Non-Hispanic White, Non-Hispanic Black, Non-Hispanic Asian, Other Hispanic, Other) |
| **Survey cycle** | AUROC across 2013–14, 2015–16, 2017–18 (temporal generalization) |

---

## Visualizations (24 Figures — All from Real NHANES Data)

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
| [fig09_feature_importance](results/figures/fig09_feature_importance.png) | Top 20 feature importances |
| [fig10_cancer_types](results/figures/fig10_cancer_types.png) | Cancer type breakdown |
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
| [fig23_dataset_overview](results/figures/fig23_dataset_overview.png) | Full dataset overview dashboard |
| [fig24_radar_chart](results/figures/fig24_radar_chart.png) | Multi-model radar chart |
| [fig25_bootstrap_ci](results/figures/fig25_bootstrap_ci.png) | Bootstrap distribution of AUROC and AUPRC (n=1000) |
| [fig26_decision_curve_analysis](results/figures/fig26_decision_curve_analysis.png) | Clinical utility and Net Benefit (DCA) |
| [fig27_permutation_test](results/figures/fig27_permutation_test.png) | Permutation significance testing (n=500) |
| [fig28_roc_clinical_operating_points](results/figures/fig28_roc_clinical_operating_points.png) | ROC curve with clinical operating points |
| [fig29_eas_per_patient](results/figures/fig29_eas_per_patient.png) | EAS Jaccard & Overlap@5 per patient |
| [fig30_triage_distribution](results/figures/fig30_triage_distribution.png) | LLaMA 3.3 70B triage decisions |
| [fig31_hallucination_rate](results/figures/fig31_hallucination_rate.png) | Automated clinical hallucination rate per patient |
| [fig32_risk_vs_eas](results/figures/fig32_risk_vs_eas.png) | Predicted risk vs Explanation Alignment |
| [fig33_novel_metrics_summary](results/figures/fig33_novel_metrics_summary.png) | Summary dashboard of agentic and validation metrics |
| [fig34_counterfactual](results/figures/fig34_counterfactual.png) | Counterfactual risk reduction analysis |
| [fig35_eas_by_cancer_type](results/figures/fig35_eas_by_cancer_type.png) | EAS stratified by target cancer types |
| [fig36_complete_pipeline](results/figures/fig36_complete_pipeline.png) | Full ML + Multi-Agent Consensus pipeline architecture |

---

## Repository Structure

```
Research_biomedical/
├── README.md
├── requirements.txt
├── download_nhanes.py              # Step 1: Download real NHANES data (free)
│
├── data/
│   └── raw/nhanes/                 # Real NHANES XPT files (CDC)
│       ├── DEMO_H/I/J.XPT          # Demographics (2013-2018)
│       ├── CBC_H/I/J.XPT           # Complete Blood Count
│       ├── BIOPRO_H/I/J.XPT        # Biochemistry Panel
│       ├── HSCRP_I/J.XPT           # High-Sensitivity CRP
│       ├── FERTIN_I/J.XPT          # Ferritin
│       └── MCQ_H/I/J.XPT           # Cancer Questionnaire (MCQ220)
│
├── src/
│   ├── preprocessing/
│   │   └── nhanes_to_features.py   # Step 2: Merge & process features
│   ├── models/
│   │   └── train_nhanes.py         # Step 3: Train models + generate 24 figures
│   └── agents/
│       └── nhanes_agent_pipeline.py # Step 4: LLaMA 3.3 70B Multi-Agent Triage Pipeline (figures 25-36)
│
└── results/
    ├── nhanes_model_results.json   # Model results (AUROC, AUPRC, F1, Brier)
    ├── agent_results.json          # Individual patient LLM triage outputs
    ├── full_results_summary.json   # Consolidated statistical results (Bootstrap CI, DCA, EAS)
    └── figures/                    # 36 publication-quality figures
        ├── fig01_roc_curves.png
        ├── fig02_pr_curves.png
        └── ... (36 total)
```

---

## Quick Start (Reproduce Everything in ~10 Minutes)

```bash
# 1. Clone
git clone https://github.com/KasimVali2207/Research_biomedical.git
cd Research_biomedical

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn pyarrow groq scipy python-dotenv shap

# 3. Configure Groq API Key (for LLM Agents)
# Create a .env file in the root folder with your console.groq.com API Key:
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 4. Download real NHANES data (free, no registration, ~30 MB)
python download_nhanes.py

# 5. Process features
python -m src.preprocessing.nhanes_to_features

# 6. Train all 5 models + generate 24 figures (~5 minutes)
python -m src.models.train_nhanes

# 7. Run 5-agent LLaMA 3.3 consensus pipeline + generate figures 25-36 (~2 minutes)
python -m src.agents.nhanes_agent_pipeline
```

**That's it.** All 36 publication-ready figures will be in `results/figures/` and metrics in `results/`.

---

## Limitations (Stated for Scientific Honesty)

1. **Cross-sectional design** — NHANES measures blood tests and cancer status simultaneously; this cannot confirm that abnormal labs preceded diagnosis
2. **Self-reported cancer** — Cancer status from MCQ220 is patient-reported, not registry-confirmed
3. **No temporal features** — Only single-timepoint measurements; no slope or trend features possible
4. **Class imbalance** — 2.9% cancer prevalence causes low AUPRC despite good AUROC
5. **No imaging or genomic data** — Blood biomarkers only

---

## Citation

```bibtex
@misc{kasim2025nhanes_cancer,
  title  = {Multi-Cancer Detection from Routine Blood Biomarkers:
             A Machine Learning Study on Real NHANES Data},
  author = {Kasim Vali},
  year   = {2025},
  url    = {https://github.com/KasimVali2207/Research_biomedical},
  note   = {Data: CDC NHANES 2013-2018 (n=16,762, 485 cancer cases).
             Best model: Gradient Boosting AUROC=0.724.}
}
```

---

## License
Apache 2.0 — see [LICENSE](LICENSE)

---

*For the full LLM-orchestrated agentic pipeline with temporal features on MIMIC-IV,
see the companion repository.*
