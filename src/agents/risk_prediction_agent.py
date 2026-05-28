"""
Risk Prediction Agent

Synthesizes ML model probabilities with biomarker pattern signals to produce
calibrated, tiered cancer risk estimates with uncertainty bounds.

Design rationale: ML models give point probabilities; the LLM layer adds
clinical reasoning that can up/down-weight based on pattern severity,
demographics, and data quality — functioning as a learned prior adjustment.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Risk tier boundaries (probability thresholds).
_TIERS = [
    (0.30, "high"),
    (0.15, "elevated"),
    (0.05, "borderline"),
    (0.00, "low"),
]

# Age/sex risk modifiers — labels injected into prompts so the LLM can
# reference them explicitly rather than re-deriving from demographic numbers.
_AGE_MODIFIERS: list[tuple[int, str]] = [
    (75, "elderly (≥75): substantially elevated baseline cancer risk"),
    (60, "older adult (60-74): above-average baseline risk"),
    (45, "middle-aged (45-59): moderate baseline risk"),
    (0,  "younger adult (<45): lower baseline risk, unusual presentation is more concerning"),
]


def _risk_tier(prob: float) -> str:
    for threshold, label in _TIERS:
        if prob >= threshold:
            return label
    return "low"


def _approx_ci(prob: float, n_effective: int = 500) -> tuple[float, float]:
    """Wilson score interval approximation for a proportion.

    n_effective is a pessimistic proxy when no bootstrap samples are
    available — chosen as 500 because our training cohorts are O(1000).
    """
    from math import sqrt

    z = 1.96  # 95% CI
    n = max(n_effective, 10)
    centre = (prob + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = z * sqrt(prob * (1 - prob) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    lo = max(0.0, round(centre - margin, 4))
    hi = min(1.0, round(centre + margin, 4))
    return lo, hi


class RiskPredictionAgent(BaseAgent):
    """Synthesizes ML probabilities and biomarker patterns into tiered risk.

    The LLM is used for *reasoning* about consistency between the ML signal
    and biomarker patterns, not for recomputing probabilities from scratch.
    The final probability is a weighted blend anchored to the ML model output.
    """

    _SYSTEM_PROMPT = """You are a senior clinical oncologist reviewing AI-generated
cancer risk scores for early detection research.

Your task: Given ML model probabilities for several cancer types AND a summary
of the patient's laboratory biomarker patterns, synthesize a final risk assessment.

## Synthesis Rules
1. The ML probabilities are your PRIMARY source of truth — they come from calibrated
   gradient-boosted models trained on thousands of similar patients.
2. Biomarker patterns can MODESTLY adjust your reasoning (+/- one tier at most),
   but NEVER override a low ML probability to high without extremely compelling evidence.
3. Risk tiers:
   - low:        probability < 5%
   - borderline: 5-15%
   - elevated:   15-30%
   - high:       > 30%
4. primary_concern = cancer type with highest adjusted probability.
5. risk_modifiers: list demographic or clinical flags that influenced reasoning
   (e.g., "age >65: elevated baseline colorectal risk").
6. reasoning: 2-3 sentences maximum. Clinical, not conversational.
7. Output ONLY valid JSON — no markdown, no prose outside JSON.
"""

    def __init__(
        self,
        cancer_types: list[str] | None = None,
        llm_model: str = "gpt-4o",
    ) -> None:
        super().__init__(
            agent_name="risk_prediction",
            llm_model=llm_model,
            temperature=0.0,
            max_tokens=1500,
        )
        self.cancer_types = cancer_types or ["colorectal", "lung", "liver"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, inputs: dict) -> dict:
        """Produce tiered risk scores with CIs for each cancer type.

        Args:
            inputs: {
                'subject_id': str,
                'ml_probabilities': {'colorectal': float, ...},
                'biomarker_patterns': dict,   # TemporalBiomarkerAgent output
                'demographics': dict,
                'horizon_months': int,
                'model_calibrated': bool
            }

        Returns:
            Structured risk scores dict.
        """
        start = time.time()

        ml_probs: dict[str, float] = inputs.get("ml_probabilities", {})
        demographics: dict = inputs.get("demographics", {})

        # Pre-compute CIs from ML probabilities; LLM reasoning may shift
        # the central estimate but we keep CI anchored to ML uncertainty.
        pre_computed_risk: dict[str, dict] = {}
        for ct in self.cancer_types:
            prob = float(ml_probs.get(ct, 0.0))
            lo, hi = _approx_ci(prob)
            pre_computed_risk[ct] = {
                "ml_probability": prob,
                "ci_lower": lo,
                "ci_upper": hi,
                "tier_from_ml": _risk_tier(prob),
            }

        user_prompt = self._build_user_prompt(inputs, pre_computed_risk)

        output_schema = {
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "risk_scores": {
                    "type": "object",
                    "description": "One key per cancer type",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "probability": {"type": "number"},
                            "risk_tier": {"type": "string", "enum": ["low", "borderline", "elevated", "high"]},
                            "ci_lower": {"type": "number"},
                            "ci_upper": {"type": "number"},
                        },
                        "required": ["probability", "risk_tier", "ci_lower", "ci_upper"],
                    },
                },
                "primary_concern": {"type": "string"},
                "risk_modifiers": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["subject_id", "risk_scores", "primary_concern", "risk_modifiers", "reasoning"],
        }

        try:
            result = self._call_llm(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "RiskPredictionAgent failed for subject %s: %s",
                inputs.get("subject_id"),
                exc,
            )
            result = self._fallback_output(inputs, pre_computed_risk)

        # Enforce: LLM probability cannot deviate >15% from ML probability
        # to prevent the LLM from completely overriding the calibrated model.
        result = self._clamp_probabilities(result, ml_probs)
        result.setdefault("subject_id", inputs.get("subject_id", "unknown"))

        self.log_run(inputs, result, time.time() - start)
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        inputs: dict,
        pre_computed_risk: dict[str, dict],
    ) -> str:
        subject_id = inputs.get("subject_id", "UNKNOWN")
        demographics = inputs.get("demographics", {})
        horizon = inputs.get("horizon_months", 12)
        biomarker_patterns = inputs.get("biomarker_patterns", {})
        model_calibrated = inputs.get("model_calibrated", True)

        age = demographics.get("age", "unknown")
        sex = demographics.get("sex", "unknown")

        # Select age modifier text.
        age_modifier_text = "unknown age group"
        if isinstance(age, (int, float)):
            for threshold, text in _AGE_MODIFIERS:
                if age >= threshold:
                    age_modifier_text = text
                    break

        lines: list[str] = [
            f"Patient ID: {subject_id}",
            f"Age: {age} ({age_modifier_text})",
            f"Sex: {sex}",
            f"Prediction horizon: {horizon} months",
            f"Model calibrated: {model_calibrated}",
            "",
            "=== ML MODEL PROBABILITIES (calibrated) ===",
            f"{'Cancer Type':<20} {'ML Prob':>10} {'95% CI':>20} {'ML Tier':>12}",
            "-" * 65,
        ]

        for ct, risk_info in pre_computed_risk.items():
            prob = risk_info["ml_probability"]
            ci_str = f"[{risk_info['ci_lower']:.3f}, {risk_info['ci_upper']:.3f}]"
            lines.append(
                f"{ct:<20} {prob:>10.4f} {ci_str:>20} {risk_info['tier_from_ml']:>12}"
            )

        lines += ["", "=== BIOMARKER PATTERN SUMMARY ==="]
        if biomarker_patterns:
            lines.append(f"Pattern severity: {biomarker_patterns.get('pattern_severity', 'unknown')}")
            lines.append(f"Data confidence:  {biomarker_patterns.get('confidence', 0.0):.2f}")
            lines.append("")
            patterns = biomarker_patterns.get("abnormal_patterns", [])
            if patterns:
                lines.append("Abnormal patterns detected:")
                for p in patterns:
                    lines.append(f"  • {p}")
            traj = biomarker_patterns.get("key_trajectories", [])
            if traj:
                lines.append("")
                lines.append("Key trajectories:")
                for t in traj:
                    lines.append(
                        f"  • {t.get('feature', '?')}: {t.get('direction', '?')} "
                        f"({t.get('magnitude', '?')}) — {t.get('clinical_concern', '')}"
                    )
            summary = biomarker_patterns.get("trend_summary", "")
            if summary:
                lines += ["", f"Narrative: {summary}"]
        else:
            lines.append("No biomarker pattern data available.")

        lines += [
            "",
            "=== TASK ===",
            f"Synthesize the ML probabilities and biomarker patterns above into a final",
            f"risk assessment for patient {subject_id}.",
            "Keep final probabilities within ±0.15 of ML probability (the model is calibrated).",
            "List risk_modifiers as short phrases (≤10 words each).",
            f'Set subject_id = "{subject_id}" in your response.',
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_probabilities(
        self,
        result: dict,
        ml_probs: dict[str, float],
    ) -> dict:
        """Prevent LLM from drifting more than 15pp from ML probability.

        This is a hard safety rail — without it the LLM occasionally gives
        cancer probabilities of 0.95 for patients with ML score 0.3.
        """
        max_drift = 0.15
        risk_scores = result.get("risk_scores", {})
        for ct, ml_prob in ml_probs.items():
            if ct not in risk_scores:
                lo, hi = _approx_ci(ml_prob)
                risk_scores[ct] = {
                    "probability": ml_prob,
                    "risk_tier": _risk_tier(ml_prob),
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
                continue
            entry = risk_scores[ct]
            raw_prob = float(entry.get("probability", ml_prob))
            clamped = float(np.clip(raw_prob, ml_prob - max_drift, ml_prob + max_drift))
            clamped = round(max(0.0, min(1.0, clamped)), 4)
            entry["probability"] = clamped
            entry["risk_tier"] = _risk_tier(clamped)
            # Recompute CI around clamped value.
            lo, hi = _approx_ci(clamped)
            entry["ci_lower"] = lo
            entry["ci_upper"] = hi

        # Ensure primary_concern matches the highest probability cancer type.
        if risk_scores:
            result["primary_concern"] = max(
                risk_scores, key=lambda ct: risk_scores[ct].get("probability", 0.0)
            )

        result["risk_scores"] = risk_scores
        return result

    @staticmethod
    def _fallback_output(inputs: dict, pre_computed_risk: dict) -> dict:
        """Return ML-only risk scores when LLM call fails."""
        risk_scores = {}
        for ct, info in pre_computed_risk.items():
            prob = info["ml_probability"]
            risk_scores[ct] = {
                "probability": prob,
                "risk_tier": _risk_tier(prob),
                "ci_lower": info["ci_lower"],
                "ci_upper": info["ci_upper"],
            }
        primary = max(risk_scores, key=lambda ct: risk_scores[ct]["probability"]) if risk_scores else "unknown"
        return {
            "subject_id": inputs.get("subject_id", "unknown"),
            "risk_scores": risk_scores,
            "primary_concern": primary,
            "risk_modifiers": ["LLM synthesis unavailable — ML probabilities used directly"],
            "reasoning": "Agent failure — ML model output used as fallback.",
        }
