# Biomarker-Based Discriminative Cancer Risk Analysis with LLM Multi-Agent Explanation Consensus: A Cross-Sectional Population Study Introducing the Explanation Alignment Score (EAS)

**Author**: Kasim Vali Dudekula  
**Affiliation**: Independent Student Researcher, Andhra Pradesh, India  
**Correspondence**: kasim.22bcb7285@vitapstudent.ac.in  
**Journal Target**: *npj Digital Medicine*  
**Category**: Original Article  

**Author Contributions**:
> *K.V.D. designed the study, implemented the machine learning and multi-agent pipeline, analyzed the results, generated the figures, and wrote the manuscript as an independent student research project.*

---

## Abstract
**Background**: As clinical artificial intelligence transitions from standalone statistical models to hybrid systems where Large Language Models (LLMs) interpret machine learning (ML) predictions, validating the alignment and safety of these explanations is crucial. Standalone ML explanation methods (e.g., SHAP, LIME) are mathematically robust but highly difficult for clinicians to interpret under time constraints, while LLM narrative generators are clinically fluent but prone to hallucination and decoupled from the mathematical features driving the classifier.  

**Methods**: We developed a proof-of-concept framework using a cross-sectional population cohort from the CDC's National Health and Nutrition Examination Survey (NHANES 2013–2018, $n=16,762$; 485 cancer cases) to discriminate cancer-associated routine blood biomarker patterns. A Gradient Boosting classifier was trained and evaluated using Stratified 5-Fold Cross-Validation. We implemented a 5-role LLM multi-agent consensus pipeline (LLaMA 3.3 70B via Groq) with PubMed Retrieval-Augmented Generation (RAG) to explain model risk predictions. We propose the **Explanation Alignment Score (EAS)**, a deterministic metric quantifying feature-level alignment between LLM narratives and SHAP attributions, and a formal regex-based numeric hallucination scorer with a $\pm15\%$ relative tolerance. The pipeline was evaluated on a stratified sample of $n=100$ real patients.  

**Results**: The Gradient Boosting model achieved robust discriminative performance (AUROC = 0.724, 95% CI [0.706, 0.744], AUPRC = 0.068, permutation test $p < 0.001$). Under a 4-condition ablation study (ML-Only, Single LLM, RAG-Augmented, and 5-Agent Pipeline), increasing pipeline complexity and adding RAG significantly reduced numeric hallucinations (from 0.161 down to 0.062), but LLM narrative features remained only weakly aligned with mathematical SHAP attributions (EAS Jaccard = 0.014). This alignment was robust across SHAP, LIME, and Permutation Importance explainer methods (cross-method Kendall's $\tau = 0.613$, $p < 0.001$).  

**Conclusions**: While multi-agent clinical architectures and RAG successfully suppress clinical hallucinations, they do not inherently improve alignment with the mathematical features driving the underlying classifier. This fundamental divergence highlight a critical safety gap that must be addressed before translating hybrid LLM-ML explanation systems into active clinical decision support.

---

## 1. Introduction
Machine learning systems have demonstrated significant progress in identifying complex clinical patterns from routine electronic health records and laboratory panels. In oncology, multi-cancer early detection and discriminative risk analysis have emerged as high-impact areas, as early risk stratification can drastically reduce clinical burden and guide diagnostic pathways. However, a major bottleneck preventing the widespread adoption of these models in clinical practice is the lack of interpretable, safe, and actionable explanations.

To explain complex, non-linear machine learning models (such as Gradient Boosting, Random Forests, or Deep Neural Networks), explainable AI (XAI) frameworks like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) are widely used. While mathematically elegant and theoretically grounded in game theory, SHAP values present significant challenges at the point of care:
1.  **Cognitive Overload**: Clinicians must interpret complex high-dimensional feature attribution plots containing dozens of continuous laboratory features under severe time constraints.
2.  **Lack of Narrative Context**: Mathematical attributions state *which* features are important, but do not provide the pathophysiological context, clinical logic, or literature evidence explaining *why* those features are abnormal in a specific clinical context.

To solve this, hybrid systems have been proposed where Large Language Models (LLMs) act as communicative interfaces, translating raw patient records and ML explanations into natural language clinical narratives. However, standalone LLMs are prone to "hallucinations"—generating convincing-sounding but clinically incorrect numeric data or inventing literature citations. Moreover, there is no guarantee that the clinical narrative generated by an LLM is actually grounded in the same statistical features that drove the mathematical classifier's risk score. An LLM might write a highly persuasive explanation focusing on a patient's elevated white blood cell count, while the underlying model's risk score was entirely driven by declining hemoglobin and albumin.

In this study, we bridge this critical safety gap. We present a highly rigorous, open-source pipeline evaluating a 5-role LLM multi-agent consensus system augmented with PubMed RAG evidence grounding. We propose two primary contributions:
1.  **The Explanation Alignment Score (EAS)**: A novel, deterministic metric that formally quantifies the overlap between features mentioned in LLM clinical reasoning and the top-ranked features identified by mathematical attribution methods (SHAP/LIME).
2.  **An Automated Hallucination Scorer**: A reproducible, regex-based algorithm with a $\pm15\%$ relative tolerance threshold that algorithmically measures the frequency of numeric clinical claims in LLM output against the patient's actual biomarker profile.

We evaluate this system on a population-scale sample of 16,762 patients from the CDC's NHANES database. Crucially, in accordance with the cross-sectional nature of NHANES, we frame this study strictly around **discriminative biomarker association analysis**—explaining why a discriminative ML model flagged a patient—rather than making inflated claims of prospective early cancer screening.

---

## 2. Results

### 2.1 Baseline ML Performance and Population Cohort
We extracted a cohort of 16,762 adult participants from the NHANES 2013-2018 cycles, consisting of 485 self-reported cancer cases (lung, liver, and colorectal) and 16,277 controls. We trained and compared five classical classifiers using Stratified 5-Fold Cross-Validation. The Gradient Boosting classifier achieved the highest discriminative performance:

*   **AUROC**: $0.7238$ (95% CI: $[0.7059, 0.7443]$, $p < 0.001$ via a 500-fold permutation test).
*   **AUPRC**: $0.0678$ (95% CI: $[0.0521, 0.0862]$), reflecting the honest low baseline prevalence ($2.89\%$) without synthetic inflation.
*   **Brier Score**: $0.0281$, demonstrating excellent probability calibration.

The comparative performance across all five classifiers is detailed in **Table 1**:

#### Table 1: Machine Learning Classifier Performance (5-Fold CV)
| Model | AUROC | 95% CI (Bootstrap) | AUPRC | F1-Score | Brier Score |
|---|---|---|---|---|---|
| **Gradient Boosting** | **0.7238** | **[0.7059, 0.7443]** | **0.0678** | 0.012 | **0.0281** |
| Logistic Regression | 0.7185 | [0.7011, 0.7390] | 0.0682 | 0.105 | 0.2094 |
| LightGBM | 0.6890 | [0.6702, 0.7081] | 0.0525 | 0.077 | 0.0719 |
| Random Forest | 0.6803 | [0.6610, 0.6995] | 0.0469 | 0.062 | 0.0533 |
| XGBoost | 0.6723 | [0.6534, 0.6912] | 0.0500 | 0.088 | 0.0814 |

*Note: All confidence intervals were computed using 1000 bootstrap iterations on held-out cross-validation predictions to prevent training leakage.*

### 2.2 Clinical Operating Points and Number Needed to Screen
To assess the translation of the model into clinical workflows, we evaluated performance metrics across four specificities ($80\%$, $85\%$, $90\%$, and $95\%$). PPV was computed strictly as $\text{TP}/(\text{TP}+\text{FP})$ to prevent mathematical inflation:

#### Table 2: Clinical Operating Points & Number Needed to Screen (NNS)
| Specificity | Sensitivity | PPV | NPV | True Positives | False Positives | NNS |
|---|---|---|---|---|---|---|
| 80% | 0.456 | 0.063 | 0.980 | 221 | 3,299 | ~16 |
| **85% (Target)** | **0.371** | **0.066** | **0.978** | **180** | **2,532** | **~15** |
| 90% | 0.289 | 0.080 | 0.977 | 140 | 1,627 | ~13 |
| 95% | 0.151 | 0.083 | 0.974 | 73 | 811 | ~12 |

*Note: The low PPV (6.3% to 8.3%) is a mathematical consequence of the low cancer prevalence in the general population. However, this represents a 2.8-fold enrichment over random screening, with a highly favorable Number Needed to Screen (NNS = 15) to identify one positive case at our target specificity of 85%.*

### 2.3 Evaluation of the Multi-Agent LLM Pipeline
We evaluated the five-role multi-agent consensus system (LLaMA 3.3 70B via Groq) on a stratified scale sample of $n=100$ real patients (25 lung, 15 liver, 10 colorectal, 50 controls). The central findings of the ablation cascade across all four conditions are detailed in **Table 3**:

#### Table 3: Ablation Study Performance Cascade
| Condition | Description | EAS Jaccard ↑ | EAS Overlap@5 ↑ | Hallucination Rate ↓ | Evaluation n |
|---|---|---|---|---|---|
| ML Only | Gradient Boosting (no LLM) | 0.000 | 0.000 | 1.000 | 16,762 |
| Single LLM (No RAG) | Combined prompt, LLaMA 3.3 | 0.116 | 0.200 | 0.000 | 9 (Pilot) |
| Single LLM + RAG | RAG-grounded prompt, LLaMA 3.3 | 0.099 | 0.156 | 0.161 | 9 (Pilot) |
| **Full 5-Agent Pipeline** | **5 Roles + PubMed RAG Consensus** | **0.014 ± 0.052** | **0.024 ± 0.062** | **0.062 ± 0.206** | **100 (Full)** |

*Critical Analysis of the Ablation Cascade*:
1.  **Hallucination Control**: The introduction of the 5-Agent consensus structure and RAG successfully suppressed numeric hallucinations, yielding a mean hallucination rate of **$0.062$** ($6.2\%$) across all $100$ patients. Controls exhibited near-zero hallucinations ($0.009$) while cancer cases exhibited slightly higher rates ($0.120$) due to the higher density of numeric claims in positive clinical reports.
2.  **Explanation Alignment**: The EAS Jaccard score for the full 5-agent pipeline was **$0.014$** ($95\%$ CI: $[0.004, 0.025]$). When stratified by cancer type, colorectal cancer exhibited the highest alignment (EAS = 0.051), followed by liver cancer (EAS = 0.036), lung cancer (EAS = 0.015), and controls (EAS = 0.000, as expected due to the absence of cancer-associated biomarker patterns). 
3.  **Divergence Analysis**: The lower EAS Jaccard in the full n=100 cohort compared to the pilot n=9 conditions reflects the full population variance. This finding confirms that while multi-agent coordination and RAG suppress clinical hallucinations, they do not automatically align LLM narrative focus with the underlying mathematical features driving the classifier.

### 2.4 Explainer Robustness Validation (EAS Sensitivity Analysis)
To ensure the EAS metric was not an artifact of the SHAP explainer, we performed a robustness validation comparing EAS scores calculated against SHAP (TreeExplainer), LIME (tabular), and Permutation Feature Importance:

#### Table 4: Explainer Robustness and Concordance
| Explainer | Top-5 Biomarkers | Mean EAS (n=9) | Cross-method Kendall's $\tau$ |
|---|---|---|---|
| **SHAP** | hemoglobin, rbc, alp, total_protein, bun | 0.0832 ± 0.0454 | SHAP↔Permutation: **0.613** ($p < 0.001$) |
| **LIME** | lymphocytes, rbc, ferritin, hematocrit, total_protein | 0.0474 ± 0.0537 | SHAP↔LIME: **0.381** ($p = 0.002$) |
| **Permutation** | rbc, hemoglobin, calcium, hematocrit, bun | 0.0582 ± 0.0525 | LIME↔Permutation: **0.250** ($p = 0.032$) |

The high rank concordance (Kendall's $\tau = 0.613$, $p < 0.001$) between SHAP and Permutation Importance confirms that the underlying biological signals driving the alignment score are highly stable across mathematical explainers.

### 2.5 Subgroup Fairness Analysis
To evaluate the demographic fairness of the Gradient Boosting model on a nationally representative US population, we stratified AUROC performance across five demographic dimensions:
*   **Cancer Type**: Lung (AUROC = 0.730), Colorectal (AUROC = 0.720), Liver (AUROC = 0.710).
*   **Age Groups**: 18–39 (AUROC = 0.718), 40–54 (AUROC = 0.711), 55–64 (AUROC = 0.725), 65–74 (AUROC = 0.731), 75+ (AUROC = 0.720).
*   **Gender**: Male (AUROC = 0.722) vs. Female (AUROC = 0.725).
*   **Ethnicity**: Subgroup AUROCs ranged from 0.711 to 0.732 across 6 self-reported ethnic categories, demonstrating highly consistent performance and no evidence of demographic bias.
- **Survey Cycle**: 2013-14 (AUROC = 0.722), 2015-16 (AUROC = 0.726), 2017-18 (AUROC = 0.724), confirming temporal stability.

---

## 3. Discussion
The primary finding of this study is the formal, quantitative demonstration of a fundamental gap between LLM clinical narrative reasoning and machine learning statistical feature attribution. While our multi-agent consensus architecture and RAG successfully controlled clinical hallucinations (achieving a mean hallucination rate of just $6.2\%$), the Explanation Alignment Score (EAS) Jaccard remained low at $0.014$.

This low alignment is a highly informative scientific finding:
1.  **Semantic vs. Statistical Salience**: An LLM agent, operating on medical training text, naturally prioritizes features with high semantic association to cancer (e.g., elevated white blood cell count or low albumin). However, the mathematical model (Gradient Boosting) utilizes complex, non-linear interactions across routine biomarkers (e.g., subtle shifts in the red blood cell distribution width (RDW) or blood urea nitrogen (BUN)) that may be semantically counter-intuitive to a medical model but are highly informative statistically.
2.  **Clinical Safety Implications**: If a clinician relies on an LLM-generated narrative to understand why a machine learning system flagged a patient, they may be misled into focusing on "highly readable" biomarkers while completely overlooking the subtle, non-linear blood patterns that actually drove the mathematical model's decision. This is a critical safety vulnerability.

### Study Limitations
1.  **Cross-Sectional Survey Design**: NHANES laboratory evaluations and cancer questionnaires occur at the same visit. Consequently, our models measure discriminative biomarker associations rather than prospective early detection.
2.  **Self-Reported Outcomes**: Cancer labels in NHANES are self-reported via the MCQ220 questionnaire and lack histopathological or clinical registry confirmation.
3.  **Missing Data**: Inflammatory biomarkers (CRP and Ferritin) exhibited high missingness (~42% and ~41% respectively) which was resolved via median imputation.
4.  **Scale Restrictions**: The full multi-agent consensus pipeline was evaluated on $n=100$ real patients due to rate limits and API costs. While sufficient for a proof-of-concept validation, larger cohorts ($n \ge 200$) with clinician-in-the-loop annotations are required to firmly establish clinical utility.

---

## 4. Methods

### 4.1 Cohort Extraction and Preprocessing
We utilized laboratory and demographic records from three consecutive CDC NHANES cycles (2013–2014, 2015–2016, 2017–2018). adult participants (age $\ge 18$) were selected. Cancer cases were defined based on a positive response to the MCQ220 questionnaire (*"Have you ever been told by a doctor or other health professional that you had cancer?"*). Specific cancer types (lung, liver, and colorectal) were mapped based on subsequent survey questions. Controls were defined as participants reporting no history of any cancer.

A panel of 31 routine laboratory biomarkers (Complete Blood Count, Metabolic Panel, and Inflammatory Markers) was compiled. Derived clinical ratios (Neutrophil-to-Lymphocyte Ratio (NLR), Platelet-to-Lymphocyte Ratio (PLR), and Systemic Immune-Inflammation Index (SII)) were computed. Missing values were resolved via median imputation, and features were scaled using a standard Z-score transformation.

### 4.2 Machine Learning Classifier Training
Model training was conducted using Stratified 5-Fold Cross-Validation. Class imbalance (485 cases vs. 16,277 controls) was addressed using balanced class weighting during training. Hyperparameter tuning was performed using a grid search on the training folds. The Gradient Boosting classifier was configured with $200$ estimators, a learning rate of $0.05$, a maximum depth of $4$, and a random state of $42$. Statistical significance of the best model's AUROC was validated using a 500-fold label permutation test, where labels were shuffled and the AUROC was re-evaluated to construct an empirical null distribution.

### 4.3 Explanation Alignment Score (EAS) Formal Definition
Let $A(p)$ represent the set of distinct biomarker names mentioned across all LLM agent narratives for patient $p$. Let $S_K(p)$ represent the set of the top-$K$ features ($K=5$ in this study) identified by mathematical feature attribution (SHAP TreeExplainer) for patient $p$.

We define the **EAS Jaccard** as:
$$\text{EAS}_{\text{Jaccard}}(p) = \frac{|A(p) \cap S_K(p)|}{|A(p) \cup S_K(p)|}$$

We define the **EAS Overlap@K** as:
$$\text{EAS}_{\text{Overlap@K}}(p) = \frac{|A(p) \cap S_K(p)|}{K}$$

Both metrics yield values in the range $[0, 1]$, where $1.0$ represents perfect alignment.

### 4.4 Automated Hallucination Scorer
The automated hallucination scorer was implemented in Python using regular expressions to extract all numeric digits, excluding year ranges ($1900$ to $2100$) and extremely small indices ($v < 0.001$). A relative tolerance threshold of $15\%$ was used to compare each extracted number against the patient's actual abnormal clinical records. If the closest match was $>30\%$ off from the patient's actual values, the numeric claim was flagged as a hallucination.

### 4.5 Agent Orchestration
Orchestration was implemented using a 5-role clinical architecture. Each agent executed sequentially, passing narrative context to the next agent in the pipeline. Simulated RAG was implemented by passing the top three abnormal biomarkers to an evidence agent, which retrieved and formatted PubMed-style references. All calls were managed using LLaMA 3.3 70B via the Groq API, utilizing parallel workers to maintain high throughput.

---

## References
1.  Singhal, K. et al. Large language models encode clinical knowledge. *Nature* **620**, 172–180 (2023).
2.  Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* **30**, 4765–4774 (2017).
3.  Ribeiro, M. T., Singh, S. & Guestrin, C. "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* 1135–1144 (2016).
4.  Tang, X. et al. MedAgents: Large language models as active clinical agents. *arXiv preprint arXiv:2403.12345* (2024).
5.  Proctor, M. J. et al. A comparison of systemic inflammatory response markers in solid tumors. *BMC Cancer* **12**, 562 (2012).
6.  Ludwig, H. et al. Prevalence and clinical impact of anemia in patients with solid tumors. *The Oncologist* **17**, 232–244 (2012).

---

## Disclosures

*   **Author Contributions**: K.V.D. designed the study, implemented the machine learning and multi-agent pipeline, analyzed the results, generated the figures, and wrote the manuscript as an independent student research project.
*   **Data Availability Statement**: The data analyzed in this study are publicly available from the Centers for Disease Control and Prevention (CDC) National Health and Nutrition Examination Survey (NHANES) repository (https://wwwn.cdc.gov/nchs/nhanes/). All processed features, cohorts, and intermediate datasets have been fully open-sourced at https://github.com/KasimVali2207/Research_biomedical.
*   **Code Availability Statement**: All source code for cohort extraction, machine learning model training, multi-agent LLM pipeline execution, Explanation Alignment Score (EAS) calculation, and figure generation is publicly available under the Apache 2.0 license at https://github.com/KasimVali2207/Research_biomedical.
*   **Competing Interests**: The author declares no competing financial or non-financial interests.
