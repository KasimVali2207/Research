# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Clinical Triage Agent.

Performs the final synthesis of biomarkers, risk scores, differentials, and
grounding evidence to recommend actionable triage pathways (Experiment 7).
"""

from __future__ import annotations

import logging
import time

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ClinicalTriageAgent(BaseAgent):
    """Synthesizes risk scores, differentials, and evidence to output clinical recommendations."""

    _SYSTEM_PROMPT = """You are a senior physician and medical director overseeing early detection programs.
Your task is to synthesize the patient's laboratory trends, ML risk predictions, differentials,
and grounding evidence, and produce a final clinical triage recommendation.

## Urgency Classifications (Strict Rules)
1. **urgent**:
   - Any cancer type adjusted probability > 40%, OR
   - Extreme critical lab alerts matching clinical deterioration.
2. **expedited**:
   - Any cancer type probability is 20-40%, OR
   - Multiple cancer types > 10%.
3. **routine**:
   - All cancer probabilities < 20%.

## Triage Pathways
- **watchful_waiting**: Routine follow-up, low risk.
- **primary_care_followup**: Re-test labs, evaluate for benign mimics in primary care.
- **specialist_referral**: Refer to gastroenterology, hepatology, or pulmonology.
- **urgent_workup**: Immediate diagnostic workup (colonoscopy, abdominal CT, chest CT).

## Output Rules
- recommended_actions: list specific steps (e.g., "Refer to GI for colonoscopy scheduling").
- patient_communication_summary: Clear, empathetic, non-alarmist plain language explaining the findings.
- Output ONLY valid JSON — no prose, no markdown fences.
"""

    def __init__(self, llm_model: str = "gpt-4o") -> None:
        super().__init__(
            agent_name="clinical_triage",
            llm_model=llm_model,
            temperature=0.0,
            max_tokens=2000,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, inputs: dict) -> dict:
        """Produce final actionable clinical triage plan.

        Args:
            inputs: {
                'subject_id': str,
                'risk_scores': dict,
                'differentials': dict,
                'evidence': dict,
                'biomarker_patterns': dict,
                'demographics': dict,
                'horizon_months': int
            }

        Returns:
            Actionable triage checklist and urgency mapping.
        """
        start = time.time()
        
        subject_id = inputs.get("subject_id", "unknown")
        user_prompt = self._build_user_prompt(inputs)
        
        output_schema = {
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "urgency": {"type": "string", "enum": ["routine", "expedited", "urgent"]},
                "triage_category": {
                    "type": "string",
                    "enum": ["watchful_waiting", "primary_care_followup", "specialist_referral", "urgent_workup"],
                },
                "primary_concern": {"type": "string"},
                "reason": {"type": "string", "description": "Clinical reasoning (2-3 sentences)"},
                "recommended_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "timeframe": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        },
                        "required": ["action", "timeframe", "priority"],
                    },
                },
                "monitoring_plan": {"type": "string"},
                "patient_communication_summary": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "caveats": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "subject_id",
                "urgency",
                "triage_category",
                "primary_concern",
                "reason",
                "recommended_actions",
                "monitoring_plan",
                "patient_communication_summary",
                "confidence",
                "caveats",
            ],
        }
        
        try:
            result = self._call_llm(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
        except Exception as exc:
            logger.error("ClinicalTriageAgent failed for subject {}: {}", subject_id, exc)
            result = self._fallback_output(inputs, str(exc))
            
        # Overwrite/reinforce urgency rules programmatically as safety rail
        result = self._apply_rule_based_override(result, inputs)
        result.setdefault("subject_id", subject_id)
        
        self.log_run(inputs, result, time.time() - start)
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_prompt(self, inputs: dict) -> str:
        subject_id = inputs.get("subject_id", "UNKNOWN")
        demographics = inputs.get("demographics", {})
        horizon = inputs.get("horizon_months", 12)
        risk_scores = inputs.get("risk_scores", {})
        differentials = inputs.get("differentials", {})
        evidence = inputs.get("evidence", {})
        biomarker_patterns = inputs.get("biomarker_patterns", {})
        
        lines: list[str] = [
            f"Patient ID: {subject_id}",
            f"Demographics: {demographics.get('age', '?')}-year-old {demographics.get('gender', 'patient')}",
            f"Study Horizon: {horizon} months",
            "",
            "=== SYNTHESIZED RISK SCORES ===",
        ]
        
        rs = risk_scores.get("risk_scores", {})
        for ct, info in rs.items():
            lines.append(f"  • {ct}: probability = {info.get('probability', 0.0):.4f} (tier: {info.get('risk_tier')})")
        lines.append(f"Primary Cancer Concern: {risk_scores.get('primary_concern', 'unknown')}")
        
        lines += ["", "=== CLINICAL DIFFERENTIALS ==="]
        for diff in differentials.get("top_differentials", [])[:3]:
            lines.append(f"  • {diff.get('diagnosis')} (Likelihood: {diff.get('likelihood')})")
            lines.append(f"    Workup: {', '.join(diff.get('distinguishing_workup', []))}")
            
        lines += ["", "=== EVIDENCE GROUNDING SUMMARY ==="]
        lines.append(f"Grounding Score: {evidence.get('grounding_score', 0.0):.2f}")
        lines.append(f"Evidence Summary: {evidence.get('evidence_summary', 'No literature summary available.')}")
        for ev in evidence.get("supporting_evidence", [])[:2]:
            lines.append(f"  • Claim: {ev.get('claim')} (Strength: {ev.get('strength')})")
            
        lines += ["", "=== BIOMARKER TRENDS ==="]
        for pat in biomarker_patterns.get("abnormal_patterns", []):
            lines.append(f"  • {pat}")
            
        lines += [
            "",
            "=== TASK ===",
            "Review the aggregated diagnostic profile and recommend the appropriate triage urgency",
            "and next steps. Formulate recommended actions and monitoring plan.",
            f'Set subject_id = "{subject_id}" in your response.',
        ]
        
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Urgency override
    # ------------------------------------------------------------------

    def _apply_rule_based_override(self, result: dict, inputs: dict) -> dict:
        """Reinforce clinical safety rules programmatically."""
        risk_scores = inputs.get("risk_scores", {})
        rs = risk_scores.get("risk_scores", {})
        
        probs = [float(info.get("probability", 0.0)) for info in rs.values()]
        
        # Hard clinical triage rails
        if probs:
            max_prob = max(probs)
            if max_prob > 0.40:
                result["urgency"] = "urgent"
                result["triage_category"] = "urgent_workup"
            elif max_prob > 0.20 or sum(1 for p in probs if p > 0.10) >= 2:
                # If LLM under-triage, push to expedited
                if result.get("urgency") == "routine":
                    result["urgency"] = "expedited"
                    result["triage_category"] = "specialist_referral"
                    
        return result

    @staticmethod
    def _fallback_output(inputs: dict, error_msg: str) -> dict:
        """Minimal fallback when the LLM call fails."""
        return {
            "subject_id": inputs.get("subject_id", "unknown"),
            "urgency": "expedited",
            "triage_category": "primary_care_followup",
            "primary_concern": "unknown",
            "reason": f"Triage unavailable: {error_msg}. Prefit fallback clinical review recommended.",
            "recommended_actions": [{"action": "Schedule manual clinical review", "timeframe": "1 week", "priority": "high"}],
            "monitoring_plan": "Monitor laboratory values at regular intervals.",
            "patient_communication_summary": "We encountered a clinical processing issue. A manual review of your laboratory data is underway.",
            "confidence": 0.0,
            "caveats": ["Technical processing error"],
        }
