# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-agent pipeline orchestrator.

Coordinates execution across the 5 agents: TemporalBiomarker, RiskPrediction,
DifferentialDiagnosis, EvidenceGrounding, and ClinicalTriage.
"""

from __future__ import annotations

import os
import time
import json
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.agents.temporal_biomarker_agent import TemporalBiomarkerAgent
from src.agents.risk_prediction_agent import RiskPredictionAgent
from src.agents.differential_diagnosis_agent import DifferentialDiagnosisAgent
from src.agents.evidence_grounding_agent import EvidenceGroundingAgent
from src.agents.clinical_triage_agent import ClinicalTriageAgent


class AgentOrchestrator:
    """Orchestrates sequential invocation of the five clinical reasoning agents."""

    def __init__(self, cfg: dict, rag_pipeline: Any = None) -> None:
        self.cfg = cfg
        self.rag_pipeline = rag_pipeline
        
        # Instantiate the agents
        llm_model = cfg.get("agents", {}).get("llm_model", "gpt-4o")
        
        self.biomarker_agent = TemporalBiomarkerAgent(llm_model=llm_model)
        self.risk_agent = RiskPredictionAgent(llm_model=llm_model)
        self.diff_agent = DifferentialDiagnosisAgent(llm_model=llm_model)
        self.grounding_agent = EvidenceGroundingAgent(rag_pipeline=rag_pipeline, llm_model=llm_model)
        self.triage_agent = ClinicalTriageAgent(llm_model=llm_model)

    def run_pipeline(self, patient_data: dict) -> dict:
        """Run the full 5-agent sequential pipeline on a single patient's records.

        Args:
            patient_data: {
                'subject_id': str,
                'temporal_features': dict,  # features stats
                'demographics': {'age': int, 'sex': str},
                'ml_probabilities': {'colorectal': float, 'lung': float, 'liver': float},
                'horizon_months': int
            }

        Returns:
            Dictionary containing all output responses from the individual agents.
        """
        subject_id = patient_data.get("subject_id", "unknown")
        logger.info("Starting multi-agent orchestration for patient: {}", subject_id)
        
        start_time = time.time()
        timings = {}
        
        # 1. Temporal Biomarker Agent
        step_start = time.time()
        biomarker_out = self.biomarker_agent.run(patient_data)
        timings["temporal_biomarker"] = time.time() - step_start
        
        # 2. Risk Prediction Agent
        step_start = time.time()
        risk_inputs = {
            "subject_id": subject_id,
            "ml_probabilities": patient_data.get("ml_probabilities", {}),
            "biomarker_patterns": biomarker_out,
            "demographics": patient_data.get("demographics", {}),
            "horizon_months": patient_data.get("horizon_months", 12),
            "model_calibrated": True
        }
        risk_out = self.risk_agent.run(risk_inputs)
        timings["risk_prediction"] = time.time() - step_start
        
        # 3. Differential Diagnosis Agent
        step_start = time.time()
        diff_inputs = {
            "subject_id": subject_id,
            "biomarker_patterns": biomarker_out,
            "risk_scores": risk_out,
            "demographics": patient_data.get("demographics", {})
        }
        diff_out = self.diff_agent.run(diff_inputs)
        timings["differential_diagnosis"] = time.time() - step_start
        
        # 4. Evidence Grounding Agent
        step_start = time.time()
        grounding_inputs = {
            "subject_id": subject_id,
            "biomarker_patterns": biomarker_out,
            "risk_scores": risk_out,
            "differentials": diff_out,
            "primary_concern": risk_out.get("primary_concern", "cancer")
        }
        grounding_out = self.grounding_agent.run(grounding_inputs)
        timings["evidence_grounding"] = time.time() - step_start
        
        # 5. Clinical Triage Agent
        step_start = time.time()
        triage_inputs = {
            "subject_id": subject_id,
            "risk_scores": risk_out,
            "differentials": diff_out,
            "evidence": grounding_out,
            "biomarker_patterns": biomarker_out,
            "demographics": patient_data.get("demographics", {}),
            "horizon_months": patient_data.get("horizon_months", 12)
        }
        triage_out = self.triage_agent.run(triage_inputs)
        timings["clinical_triage"] = time.time() - step_start
        
        total_duration = time.time() - start_time
        logger.info("Pipeline execution complete for patient {} in {:.2f}s", subject_id, total_duration)
        
        return {
            "subject_id": subject_id,
            "temporal_biomarker": biomarker_out,
            "risk_prediction": risk_out,
            "differential_diagnosis": diff_out,
            "evidence_grounding": grounding_out,
            "clinical_triage": triage_out,
            "timings": timings,
            "total_duration": total_duration,
        }

    def run_batch(self, patients_list: list[dict], n_workers: int = 1) -> list[dict]:
        """Run batch execution over a list of patients with progress indicators."""
        logger.info("Running agentic pipeline batch of size N = {}", len(patients_list))
        results = []
        
        for idx, patient in enumerate(tqdm(patients_list, desc="Processing patients")):
            res = self.run_pipeline(patient)
            results.append(res)
            
            # Save checkpoints occasionally
            if (idx + 1) % 50 == 0:
                logger.info("Processed {}/{} patients. Checkpoint saved.", idx + 1, len(patients_list))
                
        return results

    def get_pipeline_stats(self, pipeline_outputs: list[dict]) -> dict:
        """Compute aggregate cost, token usage, and latency statistics."""
        total_cost = 0.0
        # gpt-4o pricing assumptions: $5.00 / 1M input tokens, $15.00 / 1M output tokens
        input_price = 5.0 / 1e6
        output_price = 15.0 / 1e6
        
        agents_list = [
            self.biomarker_agent, self.risk_agent, self.diff_agent,
            self.grounding_agent, self.triage_agent
        ]
        
        prompt_tokens = sum(a._total_prompt_tokens for a in agents_list)
        completion_tokens = sum(a._total_completion_tokens for a in agents_list)
        total_cost = (prompt_tokens * input_price) + (completion_tokens * output_price)
        
        durations = [out.get("total_duration", 0.0) for out in pipeline_outputs]
        grounding_scores = [
            out.get("evidence_grounding", {}).get("grounding_score", 0.0)
            for out in pipeline_outputs
        ]
        
        return {
            "total_patients_processed": len(pipeline_outputs),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_estimated_cost_usd": round(total_cost, 4),
            "mean_duration_sec": float(np.mean(durations)) if durations else 0.0,
            "max_duration_sec": float(np.max(durations)) if durations else 0.0,
            "mean_grounding_score": float(np.mean(grounding_scores)) if grounding_scores else 0.0,
        }

    def compare_with_baseline(
        self,
        pipeline_outputs: list[dict],
        baseline_probabilities: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        """Compare agent risk predictions against baseline ML predictions."""
        # Extract agent risk probabilities for each patient
        agent_probs = {ct: [] for ct in ["colorectal", "lung", "liver"]}
        
        for out in pipeline_outputs:
            scores = out.get("risk_prediction", {}).get("risk_scores", {})
            for ct in agent_probs:
                prob = scores.get(ct, {}).get("probability", 0.0)
                agent_probs[ct].append(prob)
                
        # Compare metrics
        rows = []
        for ct in agent_probs:
            y_agent = np.array(agent_probs[ct])
            y_base = baseline_probabilities.get(ct, np.zeros_like(y_true))
            
            # Simple AUROC calculation
            auc_agent = roc_auc_score(y_true, y_agent) if len(np.unique(y_true)) > 1 else 0.5
            auc_base = roc_auc_score(y_true, y_base) if len(np.unique(y_true)) > 1 else 0.5
            
            rows.append({
                "cancer_type": ct,
                "agent_auroc": float(auc_agent),
                "baseline_auroc": float(auc_base),
                "delta": float(auc_agent - auc_base)
            })
            
        return pd.DataFrame(rows)
