# LLM-Orchestrated Agentic Triage for Multi-Cancer Early Detection
### A Retrospective Validation Study using Routine Blood Biomarkers

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Overview

This repository contains the full research pipeline for the study:

> **LLM-Orchestrated Agentic Triage for Multi-Cancer Early Detection from Routine Blood Biomarkers: A Retrospective Validation Study**

We develop and validate a **5-agent LLM orchestration system** that identifies elevated risk for:
- 🟠 **Colorectal cancer** (ICD: C18–C20)
- 🔵 **Lung cancer** (ICD: C33–C34)
- 🟢 **Liver cancer** (ICD: C22)

Using **only routine blood tests** — CBC, metabolic panels, inflammatory markers, and their temporal trajectories — **without imaging or genomics**.

---

## Architecture

```
MIMIC-IV / eICU
      │
      ▼
Cohort Construction (leakage-safe, 1:3 matched)
      │
      ▼
Feature Engineering
  ├── Static: CBC, Metabolic, Inflammatory, NLR/PLR/SII
  └── Temporal: slope, velocity, delta, exp_smooth, moving_avg
      │
      ▼
Baseline ML Models
  LR · RF · XGBoost · LightGBM · CatBoost · TabNet
      │
      ▼
5-Agent LLM Pipeline
  [1] TemporalBiomarkerAgent  → abnormal patterns
  [2] RiskPredictionAgent     → calibrated risk scores
  [3] DifferentialDiagnosisAgent → top differentials
  [4] EvidenceGroundingAgent  → RAG (PubMed) grounding
  [5] ClinicalTriageAgent     → urgency + recommendations
      │
      ▼
Evaluation Suite
  AUROC · AUPRC · ECE · Brier · DeLong · McNemar
  Fairness · Calibration · Hallucination · EAS (novel)
```

---

## Repository Structure

```
├── configs/                    # Hydra experiment configs
│   ├── base.yaml               # Shared hyperparameters
│   ├── experiment_baseline.yaml
│   ├── experiment_temporal.yaml
│   ├── experiment_agentic.yaml
│   ├── experiment_ablation.yaml
│   └── experiment_external_val.yaml
│
├── docker/                     # Reproducible Docker setup
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/                  # One notebook per experiment
│   ├── 01_baseline_performance.ipynb
│   ├── 02_temporal_vs_static.ipynb
│   ├── 03_ablation_study.ipynb
│   ├── 04_calibration_analysis.ipynb
│   ├── 05_missing_data_robustness.ipynb
│   ├── 06_fairness_analysis.ipynb
│   ├── 07_external_validation.ipynb
│   ├── 08_hallucination_faithfulness.ipynb
│   └── 09_explanation_alignment.ipynb
│
├── src/
│   ├── agents/                 # 5 LLM agents + orchestrator
│   ├── evaluation/             # Metrics, fairness, hallucination
│   ├── explainability/         # SHAP + Explanation Alignment Score
│   ├── features/               # Feature engineering pipeline
│   ├── models/                 # Baselines, TabNet, calibration
│   ├── preprocessing/          # Cohort extraction, leakage prevention
│   ├── retrieval/              # PubMed RAG pipeline
│   ├── utils/                  # Logging, seeding, I/O
│   └── run_pipeline.py         # Main entry point
│
├── tests/
│   └── test_pipeline.py        # Unit tests
│
└── requirements.txt
```

---

## Experiments

| # | Notebook | Question |
|---|----------|---------|
| 1 | `01_baseline_performance` | Which ML model performs best? |
| 2 | `02_temporal_vs_static` | Do trajectory features improve AUROC? |
| 3 | `03_ablation_study` | What does each agent contribute? |
| 4 | `04_calibration_analysis` | Are probabilities well-calibrated? |
| 5 | `05_missing_data_robustness` | How robust under 10/20/40% missingness? |
| 6 | `06_fairness_analysis` | Are predictions equitable across subgroups? |
| 7 | `07_external_validation` | Does it generalize MIMIC → eICU? |
| 8 | `08_hallucination_faithfulness` | Do agents hallucinate biomarker values? |
| 9 | `09_explanation_alignment` | Do agent citations align with SHAP? (EAS) |

---

## Target Performance

| Cancer | AUROC Goal | Key Biomarkers |
|--------|------------|----------------|
| Colorectal | 0.82–0.88 | Hemoglobin↓, NLR↑, Platelets↑ |
| Liver | 0.84–0.90 | ALT↑, AST↑, Albumin↓, Bilirubin↑ |
| Lung | 0.76–0.84 | NLR↑, Albumin↓, LDH↑, PLR↑ |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Place MIMIC-IV tables in `data/raw/mimic/` and eICU tables in `data/raw/eicu/`.

For a quick demo with synthetic data:
```bash
python -m src.preprocessing.extract_cohort --generate-synthetic
python -m src.preprocessing.extract_eicu --generate-synthetic
python -m src.preprocessing.mimic_to_features --dataset mimic
python -m src.preprocessing.mimic_to_features --dataset eicu
```

### 3. Run the full pipeline
```bash
python src/run_pipeline.py
```

### 4. Run individual experiments
```bash
jupyter notebook notebooks/
```

### 5. Docker (fully reproducible)
```bash
docker-compose -f docker/docker-compose.yml up cancer-triage
```

---

## Data Requirements

| Dataset | Access | Tables Used |
|---------|--------|-------------|
| MIMIC-IV v3.1 | [PhysioNet](https://physionet.org/content/mimiciv/) | diagnoses_icd, labevents, admissions, patients |
| eICU-CRD | [PhysioNet](https://physionet.org/content/eicu-crd/) | patient, diagnosis, lab |

**Note**: Raw patient data is never committed to this repository (see `.gitignore`).

---

## Citation

If you use this work, please cite:
```bibtex
@misc{kasim2024cancertriage,
  title={LLM-Orchestrated Agentic Triage for Multi-Cancer Early Detection},
  author={Kasim Vali},
  year={2024},
  url={https://github.com/KasimVali2207/Research}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
