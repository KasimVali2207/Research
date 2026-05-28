"""
Temporal Biomarker Agent

Analyzes serial lab measurements over time and maps statistically significant
trends to clinically meaningful cancer-risk signals.

Design rationale: Separating biomarker pattern recognition from risk scoring
allows independent evaluation of the NLP layer (hallucination testing) from
the probabilistic layer, and lets clinicians audit each stage independently.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agents.base_agent import BaseAgent, _DEFAULT_NORMAL_RANGES

logger = logging.getLogger(__name__)

# Features where declining trend is clinically concerning.
_DECLINING_CONCERN = {
    "hemoglobin": "progressive anemia — GI blood loss or bone marrow suppression",
    "hematocrit": "progressive anemia",
    "albumin": "malnutrition or hepatic synthetic failure",
    "lymphocytes": "lymphopenia — immune suppression or hematologic malignancy",
    "mcv": "microcytic anemia — chronic iron deficiency (e.g., occult colorectal bleeding)",
}

# Features where rising trend is clinically concerning.
_RISING_CONCERN = {
    "nlr": "systemic inflammation — independent cancer-risk biomarker",
    "plr": "platelet-to-lymphocyte ratio elevation — paraneoplastic signal",
    "sii": "systemic immune-inflammation index — validated pan-cancer prognostic",
    "alt": "hepatocellular stress — NAFLD progression or hepatocellular carcinoma",
    "ast": "hepatocellular or muscle damage",
    "alp": "cholestasis or bone/liver malignancy",
    "bilirubin_total": "hepatic obstruction or haemolysis",
    "platelets": "reactive thrombocytosis — chronic bleeding or inflammatory state",
    "wbc": "leukocytosis — infection, inflammation, or haematological malignancy",
    "crp": "chronic low-grade inflammation — pan-cancer risk modifier",
    "ferritin": "acute-phase reactant elevation or iron overload",
}


class TemporalBiomarkerAgent(BaseAgent):
    """Identifies clinically significant lab trends and maps them to cancer risk.

    The agent intentionally does *not* assign a cancer-type probability —
    that is the job of RiskPredictionAgent. Here we focus purely on
    pattern recognition grounded in the actual feature values.
    """

    _SYSTEM_PROMPT = """You are a board-certified clinical hematologist and oncologist
with expertise in early cancer detection via routine laboratory biomarkers.

Your task is to analyze serial (longitudinal) laboratory data for a single patient
and identify clinically significant trends that may indicate early cancer risk.

## Focus Areas
1. **CBC trends**: unexplained anemia (↓ Hb, ↓ MCV), thrombocytosis, neutrophilia,
   lymphopenia, elevated NLR/PLR/SII.
2. **Liver function trends**: rising ALT/AST/ALP/bilirubin, falling albumin.
3. **Inflammatory markers**: rising CRP, ferritin, ESR.
4. **Cancer-risk pattern mapping**:
   - Unexplained iron-deficiency anemia → colorectal cancer (occult GI bleeding)
   - Elevated NLR (>3) + rising ALP → hepatocellular or biliary malignancy
   - Thrombocytosis + rising CRP + falling albumin → lung or GI malignancy
   - Lymphopenia + high ferritin → hematologic malignancy
   - Isolated ALP elevation + normal transaminases → bone metastasis

## Rules
- ONLY report patterns that are actually present in the provided data.
- Do NOT invent features not listed in the input.
- Confidence MUST reflect data completeness (fewer measurements → lower confidence).
- Output ONLY valid JSON — no prose, no markdown fences.
"""

    def __init__(
        self,
        llm_model: str = "gpt-4o",
        normal_ranges: dict | None = None,
    ) -> None:
        super().__init__(
            agent_name="temporal_biomarker",
            llm_model=llm_model,
            temperature=0.0,
            max_tokens=2000,
        )
        self.normal_ranges = normal_ranges or _DEFAULT_NORMAL_RANGES

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, inputs: dict) -> dict:
        """Analyze temporal features and return biomarker pattern summary.

        Args:
            inputs: {
                'subject_id': str,
                'temporal_features': {
                    feature_name: {
                        'mean': float, 'slope': float, 'delta': float,
                        'last_value': float, 'n_measurements': int
                    }
                },
                'demographics': {'age': int, 'sex': str},
                'horizon_months': int
            }

        Returns:
            Structured biomarker pattern dict (JSON-serializable).
        """
        start = time.time()

        # Confidence degrades with sparse data — penalise heavily because
        # unreliable slope estimates are worse than no estimate.
        confidence = self._estimate_confidence(inputs.get("temporal_features", {}))

        user_prompt = self._build_user_prompt(inputs)

        output_schema = {
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "abnormal_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Snake_case pattern names e.g. declining_hemoglobin",
                },
                "trend_summary": {
                    "type": "string",
                    "description": "Free-text narrative, ≤3 sentences",
                },
                "pattern_severity": {
                    "type": "string",
                    "enum": ["mild", "moderate", "severe"],
                },
                "key_trajectories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string"},
                            "direction": {"type": "string"},
                            "magnitude": {"type": "string"},
                            "clinical_concern": {"type": "string"},
                        },
                        "required": ["feature", "direction", "magnitude", "clinical_concern"],
                    },
                },
                "missing_critical_labs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Data completeness confidence, 0-1",
                },
            },
            "required": [
                "subject_id",
                "abnormal_patterns",
                "trend_summary",
                "pattern_severity",
                "key_trajectories",
                "missing_critical_labs",
                "confidence",
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
                "TemporalBiomarkerAgent failed for subject %s: %s",
                inputs.get("subject_id"),
                exc,
            )
            result = self._fallback_output(inputs, str(exc))

        # Override LLM confidence with data-driven estimate — LLM tends to be
        # overconfident when given sparse timeseries.
        result["confidence"] = min(result.get("confidence", confidence), confidence)
        result.setdefault("subject_id", inputs.get("subject_id", "unknown"))

        self.log_run(inputs, result, time.time() - start)
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_prompt(self, inputs: dict) -> str:
        """Render temporal features as a tabular string the LLM can parse.

        Slope is converted to a qualitative trend descriptor so the LLM
        doesn't need to interpret raw regression coefficients per se, but
        the raw value is kept for transparency.
        """
        subject_id = inputs.get("subject_id", "UNKNOWN")
        demographics = inputs.get("demographics", {})
        horizon = inputs.get("horizon_months", 12)
        temporal_features: dict[str, dict] = inputs.get("temporal_features", {})

        lines: list[str] = [
            f"Patient ID: {subject_id}",
            f"Age: {demographics.get('age', 'unknown')}  |  "
            f"Sex: {demographics.get('sex', 'unknown')}",
            f"Observation horizon: {horizon} months before index date",
            "",
            "=== TEMPORAL LAB SUMMARY ===",
            f"{'Feature':<28} {'Last Value':>12} {'Mean':>10} {'Slope/mo':>12} "
            f"{'Delta':>10} {'N meas':>8} {'Trend':>12} {'Ref':>20}",
            "-" * 115,
        ]

        for feat, stats in sorted(temporal_features.items()):
            last_val = stats.get("last_value")
            mean_val = stats.get("mean")
            slope = stats.get("slope", 0.0)
            delta = stats.get("delta")
            n_meas = stats.get("n_measurements", 0)

            trend_str = _slope_to_trend(slope)
            ref_range = self.normal_ranges.get(feat)
            ref_str = f"[{ref_range[0]:.1f}-{ref_range[1]:.1f}]" if ref_range else "N/A"

            # Flag whether last value is outside normal range.
            abnormal_marker = ""
            if last_val is not None and ref_range is not None:
                lo, hi = ref_range
                if last_val > hi:
                    abnormal_marker = "↑"
                elif last_val < lo:
                    abnormal_marker = "↓"

            lines.append(
                f"{feat:<28} "
                f"{_fmt(last_val):>12}{abnormal_marker:<1} "
                f"{_fmt(mean_val):>10} "
                f"{slope:>+12.4f} "
                f"{_fmt(delta):>10} "
                f"{n_meas:>8} "
                f"{trend_str:>12} "
                f"{ref_str:>20}"
            )

        lines += [
            "",
            "=== INTERPRETATION CONTEXT ===",
            f"- Observation window spans {horizon} months prior to potential diagnosis date.",
            f"- Patient is a {demographics.get('age', '?')}-year-old "
            f"{demographics.get('sex', 'patient')}.",
            "- Slope is per-month from linear regression; delta = last_value - first_value.",
            "- ↑/↓ markers indicate last value outside reference range.",
            "",
            "Task: Identify all clinically significant patterns from the data above.",
            "Map them to cancer risk signals where supported by epidemiological evidence.",
            f'Set subject_id = "{subject_id}" in your response.',
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_confidence(self, temporal_features: dict) -> float:
        """Estimate data completeness confidence.

        Logic: average fraction of critical labs present × average measurement
        density, capped at 1.0.  Fewer than 3 measurements per feature
        makes slope estimates unreliable.
        """
        critical = [
            "hemoglobin", "wbc", "platelets", "neutrophils", "lymphocytes",
            "alt", "ast", "albumin", "nlr",
        ]
        present = sum(1 for c in critical if c in temporal_features)
        coverage = present / len(critical)

        avg_n = (
            sum(v.get("n_measurements", 0) for v in temporal_features.values())
            / max(len(temporal_features), 1)
        )
        # ≥5 measurements → full measurement confidence; linear below.
        measurement_conf = min(avg_n / 5.0, 1.0)

        return round(coverage * measurement_conf, 3)

    @staticmethod
    def _fallback_output(inputs: dict, error_msg: str) -> dict:
        """Graceful fallback when LLM call fails."""
        return {
            "subject_id": inputs.get("subject_id", "unknown"),
            "abnormal_patterns": [],
            "trend_summary": f"Analysis unavailable: {error_msg}",
            "pattern_severity": "mild",
            "key_trajectories": [],
            "missing_critical_labs": [],
            "confidence": 0.0,
        }


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------

def _fmt(val: Any) -> str:
    """Format a numeric value or return 'N/A'."""
    if val is None:
        return "N/A"
    return f"{val:.3f}"


def _slope_to_trend(slope: float) -> str:
    """Convert regression slope to qualitative label.

    Thresholds are deliberately loose — a slope of 0.005 g/dL/month for
    haemoglobin is clinically meaningless noise, while 0.5 g/dL/month
    represents rapid decline.
    """
    abs_slope = abs(slope)
    if abs_slope < 0.005:
        return "stable"
    elif abs_slope < 0.05:
        direction = "slowly↑" if slope > 0 else "slowly↓"
    elif abs_slope < 0.2:
        direction = "rising↑" if slope > 0 else "falling↓"
    else:
        direction = "rapid↑↑" if slope > 0 else "rapid↓↓"
    return direction
