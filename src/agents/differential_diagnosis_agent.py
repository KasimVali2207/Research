"""
Differential Diagnosis Agent

Generates a ranked differential list that includes both cancer and benign
alternatives, preventing the pipeline from over-attributing lab abnormalities
to malignancy.

Design rationale: Without an explicit differential step, high-risk LLM outputs
tend to anchor on cancer. Forcing ranked differentials with benign alternatives
is a structural hallucination-prevention mechanism — the LLM must actively
argue against cancer before recommending workup.
"""

from __future__ import annotations

import logging
import time

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Cancer mimic conditions that generate cancer-like lab patterns.
# Injected into the prompt as domain context so the LLM has explicit recall hooks.
_CANCER_MIMICS: dict[str, list[str]] = {
    "iron_deficiency_anemia": [
        "Colorectal cancer (occult GI bleed)",
        "Dietary iron deficiency",
        "Celiac disease",
        "Peptic ulcer disease",
        "Inflammatory bowel disease",
        "Menorrhagia (premenopausal women)",
    ],
    "elevated_nlr": [
        "Acute bacterial infection",
        "Chronic inflammatory disease (RA, IBD)",
        "Steroid therapy",
        "Post-surgical state",
        "Metabolic syndrome",
        "Hematological malignancy",
    ],
    "elevated_liver_enzymes": [
        "Non-alcoholic fatty liver disease (NAFLD/NASH)",
        "Alcohol-related liver disease",
        "Medication-induced hepatotoxicity (statins, methotrexate)",
        "Viral hepatitis (B, C)",
        "Autoimmune hepatitis",
        "Hepatocellular carcinoma",
        "Cholangiocarcinoma",
    ],
    "thrombocytosis": [
        "Reactive thrombocytosis (iron deficiency, infection)",
        "Essential thrombocythemia",
        "Post-splenectomy",
        "Inflammatory bowel disease",
        "GI malignancy",
        "Lung cancer",
    ],
    "elevated_alp": [
        "Bone disease (Paget's, osteomalacia)",
        "Cholestasis (gallstones, PSC)",
        "Hepatic metastases",
        "Primary biliary cirrhosis",
        "Normal variant in adolescents",
    ],
}


class DifferentialDiagnosisAgent(BaseAgent):
    """Generates ranked differentials including benign alternatives.

    Output structure guarantees ≥2 benign differentials alongside cancer
    diagnoses, plus distinguishing workup for each differential.
    """

    _SYSTEM_PROMPT = """You are an expert diagnostician with dual training in internal
medicine and oncology. You are reviewing a patient's laboratory trends and cancer
risk scores to generate a differential diagnosis.

## Mandate
Your role is to PREVENT over-attribution of lab abnormalities to cancer.
You MUST include at least 2-3 benign (non-cancer) diagnoses in your differential.

## Conditions You Must Consider
For iron-deficiency anemia patterns:
  - Dietary deficiency, celiac disease, IBD, peptic ulcer disease
  - Colorectal cancer (occult GI bleeding)

For elevated liver enzymes:
  - NAFLD/NASH, alcohol-related liver disease, medication effects (statins, methotrexate)
  - Viral hepatitis B/C, autoimmune hepatitis
  - Hepatocellular carcinoma (only if other causes excluded)

For respiratory/lung patterns:
  - COPD, chronic bronchitis, recurrent pneumonia, sarcoidosis
  - Lung cancer (consider if persistent despite antibiotic treatment)

For elevated inflammatory markers:
  - Autoimmune disease (RA, SLE, IBD), chronic infection
  - Paraneoplastic syndrome from occult malignancy

For thrombocytosis:
  - Reactive (iron deficiency, infection, post-surgical)
  - Myeloproliferative disorders, solid tumour paraneoplastic effect

## Output Rules
- top_differentials MUST be ordered highest → lowest likelihood.
- Each differential MUST include supporting_features, against_features,
  and distinguishing_workup (≥1 specific test).
- cancer_mimics: list conditions that cause this pattern WITHOUT cancer.
- red_flags_for_cancer: specific features in this patient's data that
  would make cancer more likely (not generic cancer warnings).
- recommended_next_tests: actionable, specific (e.g., "fecal occult blood test",
  not just "blood tests").
- Output ONLY valid JSON. No markdown, no prose.
"""

    def __init__(self, llm_model: str = "gpt-4o") -> None:
        super().__init__(
            agent_name="differential_diagnosis",
            llm_model=llm_model,
            temperature=0.0,
            max_tokens=2500,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, inputs: dict) -> dict:
        """Generate ranked differential diagnosis list.

        Args:
            inputs: {
                'subject_id': str,
                'biomarker_patterns': dict,   # TemporalBiomarkerAgent output
                'risk_scores': dict,           # RiskPredictionAgent output
                'demographics': dict
            }

        Returns:
            Ranked differential list with workup recommendations.
        """
        start = time.time()

        user_prompt = self._build_user_prompt(inputs)

        output_schema = {
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "top_differentials": {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "diagnosis": {"type": "string"},
                            "likelihood": {
                                "type": "string",
                                "enum": ["high", "moderate", "low"],
                            },
                            "supporting_features": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "against_features": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "distinguishing_workup": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": [
                            "diagnosis",
                            "likelihood",
                            "supporting_features",
                            "against_features",
                            "distinguishing_workup",
                        ],
                    },
                },
                "cancer_mimics": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "red_flags_for_cancer": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "recommended_next_tests": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "subject_id",
                "top_differentials",
                "cancer_mimics",
                "red_flags_for_cancer",
                "recommended_next_tests",
            ],
        }

        try:
            result = self._call_llm(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DifferentialDiagnosisAgent failed for subject %s: %s",
                inputs.get("subject_id"),
                exc,
            )
            result = self._fallback_output(inputs)

        result.setdefault("subject_id", inputs.get("subject_id", "unknown"))
        self.log_run(inputs, result, time.time() - start)
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_prompt(self, inputs: dict) -> str:
        subject_id = inputs.get("subject_id", "UNKNOWN")
        demographics = inputs.get("demographics", {})
        biomarker_patterns = inputs.get("biomarker_patterns", {})
        risk_scores_data = inputs.get("risk_scores", {})

        lines: list[str] = [
            f"Patient ID: {subject_id}",
            f"Age: {demographics.get('age', 'unknown')}  |  Sex: {demographics.get('sex', 'unknown')}",
            "",
        ]

        # Risk scores block.
        lines.append("=== CANCER RISK SCORES (from calibrated ML + biomarker synthesis) ===")
        rs = risk_scores_data.get("risk_scores", {})
        if rs:
            lines.append(f"{'Cancer Type':<20} {'Probability':>12} {'Tier':>12}")
            lines.append("-" * 46)
            for ct, info in rs.items():
                lines.append(
                    f"{ct:<20} {info.get('probability', 0.0):>12.4f} "
                    f"{info.get('risk_tier', 'unknown'):>12}"
                )
            pc = risk_scores_data.get("primary_concern", "unknown")
            lines.append(f"\nPrimary concern: {pc}")
        else:
            lines.append("Risk scores not available.")

        # Biomarker patterns block.
        lines += ["", "=== BIOMARKER PATTERNS ==="]
        if biomarker_patterns:
            patterns = biomarker_patterns.get("abnormal_patterns", [])
            severity = biomarker_patterns.get("pattern_severity", "unknown")
            lines.append(f"Pattern severity: {severity}")
            if patterns:
                lines.append("Abnormal patterns:")
                for p in patterns:
                    lines.append(f"  • {p}")
            traj = biomarker_patterns.get("key_trajectories", [])
            if traj:
                lines.append("Key trajectories:")
                for t in traj:
                    lines.append(
                        f"  • {t.get('feature')}: {t.get('direction')} — "
                        f"{t.get('clinical_concern', '')}"
                    )
            missing = biomarker_patterns.get("missing_critical_labs", [])
            if missing:
                lines.append(f"Missing critical labs: {', '.join(missing)}")
        else:
            lines.append("No biomarker patterns available.")

        # Cancer mimic context.
        lines += [
            "",
            "=== REFERENCE: KNOWN CANCER MIMICS FOR THESE PATTERNS ===",
        ]
        for pattern, mimics in _CANCER_MIMICS.items():
            pat_label = pattern.replace("_", " ").title()
            lines.append(f"{pat_label}:")
            for m in mimics:
                lines.append(f"  - {m}")

        lines += [
            "",
            "=== TASK ===",
            "Generate a ranked differential diagnosis for this patient.",
            "REQUIREMENT: Include at least 2 benign (non-cancer) diagnoses.",
            "Rank from most to least likely based on the available data.",
            f'Set subject_id = "{subject_id}" in your response.',
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_output(inputs: dict) -> dict:
        """Minimal fallback when the LLM call fails."""
        return {
            "subject_id": inputs.get("subject_id", "unknown"),
            "top_differentials": [
                {
                    "diagnosis": "Unable to generate differential — agent failure",
                    "likelihood": "low",
                    "supporting_features": [],
                    "against_features": [],
                    "distinguishing_workup": ["Manual clinical review required"],
                }
            ],
            "cancer_mimics": [],
            "red_flags_for_cancer": [],
            "recommended_next_tests": ["Manual clinical review required"],
        }
