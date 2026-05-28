# LLM-Orchestrated Agentic Triage for Multi-Cancer Early Detection from Routine Blood Biomarkers

> **Retrospective Validation Study** — Colorectal · Lung · Liver

---

## Overview

This project develops an agentic clinical reasoning system that identifies elevated risk for colorectal, lung, and liver cancer using only routine CBC, metabolic panels, inflammatory biomarkers, demographics, and temporal lab trajectories — without imaging, genomics, or specialist diagnostics.

### Research Hypothesis

> Structured agentic clinical reasoning over longitudinal blood biomarkers improves multi-cancer triage performance, interpretability, and clinical usability compared with classical ML and monolithic LLM systems.

---

## Datasets

| Dataset | Role |
|---------|------|
| MIMIC-IV v3.1 | Primary training + validation |
| eICU Collaborative Research Database | External validation |
| SEER Program | Epidemiology justification only |

**Access**: Both MIMIC-IV and eICU require credentialed PhysioNet access. Place data under `data/raw/mimic/` and `data/raw/eicu/`.

---

## Prediction Windows

| Horizon | Meaning |
|---------|---------|
| 3 months | Near-term detection |
| 6 months | Clinically meaningful |
| 12 months | Strongest novelty |

---

## Cancer ICD Codes

| Cancer | ICD-10 |
|--------|--------|
| Colorectal | C18–C20 |
| Lung | C33–C34 |
| Liver | C22 |

---

## Repository Structure

```
project/
├── configs/              # Hydra YAML experiment configs
├── data/
│   ├── raw/              # MIMIC-IV, eICU (not committed)
│   ├── processed/        # Extracted cohorts
│   └── external/         # SEER stats (reference only)
├── notebooks/            # EDA + figures
├── src/
│   ├── preprocessing/    # Cohort construction, leakage prevention
│   ├── features/         # Feature engineering, temporal modeling
│   ├── models/           # Baseline ML + TabNet
│   ├── agents/           # 5-agent LLM orchestration system
│   ├── retrieval/        # RAG + FAISS + biomedical embeddings
│   ├── evaluation/       # Metrics, calibration, fairness
│   ├── explainability/   # SHAP + explanation alignment
│   └── utils/            # Logging, seeding, I/O
├── experiments/          # MLflow experiment configs
├── results/              # Outputs, figures, tables
├── tests/                # Unit + integration tests
└── docker/               # Reproducibility containers
```

---

## Coding Phases

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Dataset extraction | ✅ |
| 2 | Preprocessing pipeline | ✅ |
| 3 | Feature engineering | ✅ |
| 4 | Baseline ML | ✅ |
| 5 | Temporal modeling | ✅ |
| 6 | Explainability | ✅ |
| 7 | Agent system | ✅ |
| 8 | RAG grounding | ✅ |
| 9 | Evaluation suite | ✅ |
| 10 | External validation | ✅ |

---

## Target Performance

| Cancer | AUROC Goal |
|--------|------------|
| Colorectal | 0.82–0.88 |
| Liver | 0.84–0.90 |
| Lung | 0.76–0.84 |

---

## Setup

```bash
# Create environment
conda create -n cancer-triage python=3.11
conda activate cancer-triage
pip install -r requirements.txt

# Configure paths
cp configs/paths.yaml.example configs/paths.yaml
# Edit configs/paths.yaml with your data locations

# Run extraction (requires MIMIC-IV access)
python src/preprocessing/extract_cohort.py

# Run full pipeline
python src/run_pipeline.py experiment=baseline

# Launch MLflow UI
mlflow ui --port 5000
```

---

## Reproducibility

- All random seeds fixed via `configs/base.yaml`
- Docker image in `docker/`
- MLflow tracking for all experiments
- Hydra config management

---

## Target Venues

- Journal of Biomedical Informatics
- JAMIA
- npj Digital Medicine
- The Lancet Digital Health (stretch)

---

## Citation

*Manuscript in preparation.*
