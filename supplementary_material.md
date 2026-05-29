# Supplementary Information

## Multi-Cancer Risk Stratification + LLM Agent Pipeline (NHANES)
**Study Title**: Biomarker-Based Discriminative Cancer Risk Analysis with LLM Multi-Agent Explanation Consensus: A Cross-Sectional Population Study Introducing the Explanation Alignment Score (EAS)  
**Author**: Kasim Vali Dudekula (Independent Student Researcher, Andhra Pradesh, India)  
**Code Repository**: [https://github.com/KasimVali2207/Research_biomedical](https://github.com/KasimVali2207/Research_biomedical)

---

## Section S1: Clinical Agent Prompt Templates

All agents utilize **LLaMA 3.3 70B Versatile** (via the Groq inference engine) with temperature set to $0.0$ to ensure deterministic clinical reasoning and minimize random variations in consensus formatting.

### 1. Biomarker Analyst Agent
*   **Role**: Clinical Hematologist / Biochemist
*   **Objective**: Interpret patient abnormal biomarker patterns.
*   **Input Schema**: Age, sex, and list of biomarker values flagged as LOW or HIGH based on standard NHANES laboratory reference thresholds.
*   **System & User Prompt**:
    ```text
    You are a clinical hematologist.
    Patient: age={age}, sex={gender}
    Abnormal blood values:
    {abnormal_biomarkers}
    
    In 3 clinical sentences: What do these patterns suggest? Focus on cancer-relevant patterns only.
    Reference specific values you see above.
    ```

### 2. Risk Stratifier Agent
*   **Role**: Oncology Risk AI
*   **Objective**: Contextualize the mathematical risk score output by the Gradient Boosting classifier.
*   **Input Schema**: Patient predicted risk score (percentage), abnormal biomarker summary.
*   **System & User Prompt**:
    ```text
    You are an oncology risk AI.
    Cancer risk score: {risk}%. Abnormal biomarkers: {abnormal_biomarkers}
    
    2 sentences: Explain this risk score clinically. Which specific values drive this risk most?
    Cite actual numeric values from the data above.
    ```

### 3. Differential Diagnoser Agent
*   **Role**: Diagnostic Oncologist
*   **Objective**: Rank the three primary solid cancer types (colorectal, lung, and liver) targeted by the classification model based on biomarker abnormalities.
*   **Input Schema**: Patient age, sex, risk, and top abnormal biomarkers.
*   **System & User Prompt**:
    ```text
    You are a diagnostic oncologist.
    Patient: age={age}, sex={gender}, risk={risk}%
    Abnormal: {abnormal_biomarkers}
    
    Rank by likelihood: 1)Colorectal 2)Lung 3)Liver cancer.
    One specific reason per cancer type citing the exact biomarker values.
    ```

### 4. PubMed RAG Evidence Grounding Agent
*   **Role**: Medical Research Librarian
*   **Objective**: Link abnormal biomarker findings to published medical literature via simulated retrieval-augmented generation (RAG).
*   **Input Schema**: Top abnormal biomarkers.
*   **System & User Prompt**:
    ```text
    Cite 2 peer-reviewed studies linking {top_features} abnormalities to cancer detection.
    Format: [First Author Year Journal]: key finding with effect size.
    Be specific. Do not invent citations.
    ```

### 5. Triage Coordinator Agent
*   **Role**: Clinical Coordinator / Triage Director
*   **Objective**: Synthesize the preceding agent narrative explanations and establish a formal clinical triage category.
*   **Input Schema**: Patient risk score, Biomarker Analyst narrative summary, Differential Diagnoser summary.
*   **System & User Prompt**:
    ```text
    Clinical triage system. Cancer risk={risk}%.
    Biomarker summary: {analyst_output}
    Differential: {diagnoser_output}
    
    Output EXACTLY:
    TRIAGE: [URGENT/ROUTINE/MONITOR/LOW_RISK]
    ACTION: [specific next step]
    TIMEFRAME: [when]
    RATIONALE: [one sentence]
    ```

---

## Section S2: Automated Hallucination Scorer Matching Logic

To ensure completely objective, reproducible evaluation of LLM clinical narratives, we designed and implemented a deterministic regex-based hallucination scoring algorithm. 

### Algorithmic Execution Steps:
1.  **Regex Extraction**: The algorithm scans the concatenated narrative outputs of Agents 1, 2, 3, and 5 and extracts all numeric floating-point values using the pattern:
    $$\b\d+(?:\.\d+)?\b$$
2.  **Filter Mask**: To prevent false positives, we automatically filter out:
    *   Calendar years (numbers falling in the range $1900 \le v \le 2100$).
    *   Extremely small numbers used for formatting or indices ($v < 0.001$).
3.  **Relative Distance Scoring**: For each extracted numeric value $v$, we calculate the relative distance against all actual abnormal biomarker values $g$ in the patient’s clinical record:
    $$\text{Dist}(v, g) = \frac{|v - g|}{g}$$
4.  **Tolerance Check**: A numeric claim $v$ is classified as **grounded** if there exists at least one clinical biomarker $g$ such that:
    $$\text{Dist}(v, g) \le 0.15$$
    Otherwise, the claim $v$ is classified as a **hallucination**. The $15\%$ relative tolerance accommodates standard clinical paraphrasing (e.g., a white blood cell count of $8.2 \times 10^3/\mu\text{L}$ described in prose as *"around 8"*).
5.  **Rate Calculation**: The final patient-level hallucination rate is defined as:
    $$\text{Hallucination Rate} = \frac{N_{\text{hallucinated}}}{N_{\text{total extracted}}}$$

---

## Section S3: Supplementary Figure Registry

The following **34 supplementary figures** are fully generated by the pipeline and contained in the `results/figures/` folder. They provide granular statistical details supporting the main findings:

| Figure ID | Filename | Description |
|---|---|---|
| **Fig. S1** | `fig03_auroc_bar.png` | Bar chart comparison of raw AUROC metrics across all 5 classical models. |
| **Fig. S2** | `fig04_auprc_bar.png` | Bar chart comparison of raw AUPRC metrics under low prevalence. |
| **Fig. S3** | `fig05_all_metrics.png` | Grouped comparison of secondary metrics (F1-score, Brier score, specificity, sensitivity). |
| **Fig. S4** | `fig07_confusion_matrix.png` | Confusion matrix of the best model (Gradient Boosting) at $85\%$ target specificity. |
| **Fig. S5** | `fig10_cancer_types.png` | Cohort distribution plot showing self-reported lung (359), liver (69), and colorectal (57) cancer cases. |
| **Fig. S6** | `fig11_age_distribution.png` | Age pyramid and density distribution of the study population (n=16,762). |
| **Fig. S7** | `fig12_gender_distribution.png` | Gender distribution breakdown showing balanced demographics. |
| **Fig. S8** | `fig13_ethnicity_distribution.png` | Distribution of the 6 self-reported NHANES racial/ethnic categories in the study. |
| **Fig. S9** | `fig14_biomarker_boxplots.png` | Boxplots showing clinical distributions of all 31 biomarker features for cases vs. controls. |
| **Fig. S10** | `fig15_correlation_heatmap.png` | Pearson correlation matrix heatmap for all routine blood biomarkers. |
| **Fig. S11** | `fig16_missing_data.png` | Missingness pattern heat map (highlighting high missingness in CRP and Ferritin). |
| **Fig. S12** | `fig17_auroc_by_cancer_type.png` | Stratified AUROC showing high discriminative performance for specific cancer sites. |
| **Fig. S13** | `fig18_auroc_by_age.png` | Fairness analysis: AUROC stratified across 5 age groups (18–39, 40–54, 55–64, 65–74, 75+). |
| **Fig. S14** | `fig19_auroc_by_gender.png` | Fairness analysis: AUROC comparison for Male vs. Female subgroups. |
| **Fig. S15** | `fig20_auroc_by_ethnicity.png` | Fairness analysis: Model AUROC compared across all 6 ethnic subgroups. |
| **Fig. S16** | `fig21_auroc_by_cycle.png` | Fairness analysis: Model AUROC compared across survey cycles (2013-14, 2015-16, 2017-18). |
| **Fig. S17** | `fig22_threshold_analysis.png` | Plot mapping sensitivity, specificity, and F1-score as continuous functions of decision thresholds. |
| **Fig. S18** | `fig23_dataset_overview.png` | Cohort dashboard summarizing key statistics and missingness rates. |
| **Fig. S19** | `fig24_radar_chart.png` | Radar chart of all 5 classifiers across 5 key performance metrics. |
| **Fig. S20** | `fig26_triage_distribution.png` | Breakdown of multi-agent triage recommendation levels across pilot samples. |
| **Fig. S21** | `fig29_eas_per_patient.png` | Explanation Alignment Score (EAS) across individual pilot patients. |
| **Fig. S22** | `fig30_triage_distribution.png` | Triage recommendation level distribution across the n=100 evaluation. |
| **Fig. S23** | `fig31_hallucination_rate.png` | Distribution of LLM numeric hallucination rates across the n=100 scale evaluation. |
| **Fig. S24** | `fig32_risk_vs_eas.png` | Scatter plot correlating patient ML predicted risk score against Explanation Alignment Score. |
| **Fig. S25** | `fig33_novel_metrics_summary.png` | Boxplot summary of novel metric distributions (EAS and Hallucination Rates) across the cohort. |
| **Fig. S26** | `fig34_counterfactual.png` | Patient explanation walk-through showing risk factors, protective factors, and agent consensus. |
| **Fig. S27** | `fig35_eas_by_cancer_type.png` | Analysis of pilot alignment scores (EAS) by specific cancer subgroups. |
| **Fig. S28** | `results/figures/fig38_shap_vs_lime.png` | Cross-method explainer validation comparing EAS scores calculated with SHAP, LIME, and Permutation. |

---

## Section S4: Patient Consensus Trace (Example Case)

Below is a complete, unedited narrative output trace generated by the multi-agent consensus pipeline for a Representative Patient Case:

*   **Clinical Summary**: 68-year-old Male, ML Predicted Risk: $45.6\%$ (URGENT Triage category).
*   **Abnormal Biomarkers**: Hemoglobin = 10.2 g/dL (LOW), RDW = 16.5% (HIGH), Ferritin = 8.5 ng/mL (LOW), Albumin = 3.1 g/dL (LOW).

### Agent Narrative Outputs:
*   **Agent 1 (Biomarker Analyst)**: *"The patient's blood profile is highly suggestive of a severe microcytic anemia, as evidenced by a low Hemoglobin of 10.2 g/dL and an elevated RDW of 16.5%. The critically low Ferritin of 8.5 ng/mL points to severe iron deficiency, which in an older male is highly suspicious for occult gastrointestinal blood loss secondary to malignancy. The concomitant hypoalbuminemia (Albumin 3.1 g/dL) suggests nutritional depletion or a chronic inflammatory state."*
*   **Agent 2 (Risk Stratifier)**: *"The patient's mathematical risk score of 45.6% represents a highly significant deviation from the population baseline risk. This risk is primarily driven by the severe microcytic anemia pattern, notably the combination of Hemoglobin (10.2 g/dL) and Ferritin (8.5 ng/mL) which represent values in the bottom 5th percentile of the NHANES population."*
*   **Agent 3 (Differential Diagnoser)**: *"1) Colorectal cancer: The combination of iron deficiency anemia in a 68-year-old male is a classic presentation for colorectal cancer, driven by chronic tumor bleeding. 2) Liver cancer: Hypoalbuminemia of 3.1 g/dL suggests impaired hepatic protein synthesis or chronic inflammation, which can occur in hepatocellular carcinoma, though less specific than GI bleeding signs. 3) Lung cancer: Possible, but the biomarker pattern is dominated by GI/hepatic findings rather than respiratory inflammatory patterns."*
*   **Agent 4 (PubMed RAG Grounder)**: *"1. [Proctor 2012 BMC Cancer]: Outlined that systemic inflammation (elevated RDW, low albumin) serves as a key discriminative marker in solid tumors with a hazard ratio of 1.8. 2. [Ludwig 2012 Oncologist]: Confirmed that microcytic iron deficiency anemia (Hemoglobin < 11 g/dL) is present in up to 40% of newly diagnosed colorectal cancer cases due to chronic mucosal blood loss."*
*   **Agent 5 (Triage Coordinator)**:
    ```text
    TRIAGE: URGENT
    ACTION: Referral for diagnostic colonoscopy and abdominal ultrasound
    TIMEFRAME: Within 14 days
    RATIONALE: Older male patient presenting with high-risk mathematical score of 45.6% accompanied by clinical evidence of microcytic iron deficiency anemia and hypoalbuminemia, highly indicative of occult gastrointestinal malignancy.
    ```
