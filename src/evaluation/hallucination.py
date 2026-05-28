# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Hallucination detection and clinical faithfulness evaluation module.

Quantifies the fidelity of clinical agent reasoning by checking whether mentioned
biomarker patterns align with the actual numeric features (Experiment 8).
"""

from __future__ import annotations

import logging
from typing import Any

from src.preprocessing.lab_itemids import NORMAL_RANGES

logger = logging.getLogger(__name__)


def detect_ungrounded_claims(agent_output: dict, X_patient_raw: dict) -> list[str]:
    """Check if the agent's claim list contains features not supported by numeric data.

    Checks claims like 'declining_hemoglobin' against actual slopes and values.

    Args:
        agent_output: Response dict from clinical agents.
        X_patient_raw: Dictionary containing raw patient lab values and slopes.
                       Format: {feature_name: {stat_name: value}} or flat {feature_name: value}.

    Returns:
        List of strings detailing identified ungrounded claims.
    """
    hallucinations = []
    
    # 1. Check abnormal patterns list
    patterns = agent_output.get("abnormal_patterns", [])
    
    # Standard thresholds
    for pat in patterns:
        pat_lower = str(pat).lower()
        
        # Anemia check: declining hemoglobin
        if "hemoglobin" in pat_lower or "anemia" in pat_lower:
            hgb_data = X_patient_raw.get("hemoglobin", {})
            # If hgb_data is dict of stats
            if isinstance(hgb_data, dict):
                slope = hgb_data.get("slope", hgb_data.get("trend_slope", 0.0))
                last_val = hgb_data.get("last_value", hgb_data.get("mean"))
                
                # Anemia is low hemoglobin (normally <12) or declining trend (negative slope)
                if last_val is not None and last_val >= 12.0 and slope is not None and slope >= -0.005:
                    hallucinations.append(
                        f"Claimed 'anemia/declining hemoglobin' but hemoglobin was normal ({last_val}) and stable (slope={slope:.4f})"
                    )
            elif hgb_data is not None:
                # Flat value
                val = float(hgb_data)
                if val >= 12.0:
                    hallucinations.append(f"Claimed 'anemia/low hemoglobin' but hemoglobin was normal ({val})")
                    
        # Inflammatory check: elevated NLR
        elif "nlr" in pat_lower:
            nlr_data = X_patient_raw.get("nlr", {})
            if isinstance(nlr_data, dict):
                last_val = nlr_data.get("last_value", nlr_data.get("mean"))
                if last_val is not None and last_val < 3.0:
                    hallucinations.append(f"Claimed 'elevated NLR' but NLR was normal ({last_val:.2f})")
            elif nlr_data is not None:
                val = float(nlr_data)
                if val < 3.0:
                    hallucinations.append(f"Claimed 'elevated NLR' but NLR was normal ({val:.2f})")
                    
        # Liver check: elevated ALT/AST
        elif "alt" in pat_lower or "ast" in pat_lower or "hepatocellular" in pat_lower:
            alt_data = X_patient_raw.get("alt", {})
            ast_data = X_patient_raw.get("ast", {})
            
            # Extract last value
            alt_val = alt_data.get("last_value", alt_data.get("mean")) if isinstance(alt_data, dict) else alt_data
            ast_val = ast_data.get("last_value", ast_data.get("mean")) if isinstance(ast_data, dict) else ast_data
            
            alt_limit = NORMAL_RANGES.get("alt", {}).get("high", 56.0)
            ast_limit = NORMAL_RANGES.get("ast", {}).get("high", 40.0)
            
            alt_ok = alt_val is None or alt_val <= alt_limit
            ast_ok = ast_val is None or ast_val <= ast_limit
            
            if alt_ok and ast_ok:
                hallucinations.append(
                    f"Claimed 'elevated transaminases' but ALT ({alt_val}) and AST ({ast_val}) were within normal limits."
                )
                
        # Platelets: thrombocytosis
        elif "thrombocytosis" in pat_lower or "platelets" in pat_lower:
            plt_data = X_patient_raw.get("platelets", {})
            plt_val = plt_data.get("last_value", plt_data.get("mean")) if isinstance(plt_data, dict) else plt_data
            
            if plt_val is not None and plt_val < 400.0:
                hallucinations.append(f"Claimed 'thrombocytosis/high platelets' but platelets were normal ({plt_val})")
                
        # Albumin: hypoalbuminemia
        elif "albumin" in pat_lower or "hypoalbuminemia" in pat_lower:
            alb_data = X_patient_raw.get("albumin", {})
            alb_val = alb_data.get("last_value", alb_data.get("mean")) if isinstance(alb_data, dict) else alb_data
            
            if alb_val is not None and alb_val >= 3.5:
                hallucinations.append(f"Claimed 'hypoalbuminemia/low albumin' but albumin was normal ({alb_val})")

    # 2. Check key trajectories list
    traj_list = agent_output.get("key_trajectories", [])
    for t in traj_list:
        feat = t.get("feature")
        direction = t.get("direction", "").lower()
        
        if feat and feat in X_patient_raw:
            feat_data = X_patient_raw[feat]
            if isinstance(feat_data, dict):
                slope = feat_data.get("slope", feat_data.get("trend_slope", 0.0))
                
                # Check direction claims against actual slopes
                if "declining" in direction or "down" in direction or "falling" in direction or "↓" in direction:
                    if slope is not None and slope > 0.005:
                        hallucinations.append(
                            f"Claimed trajectory '{feat}' was declining, but regression slope was positive ({slope:.4f})"
                        )
                elif "rising" in direction or "up" in direction or "elevating" in direction or "↑" in direction:
                    if slope is not None and slope < -0.005:
                        hallucinations.append(
                            f"Claimed trajectory '{feat}' was rising, but regression slope was negative ({slope:.4f})"
                        )
                        
    return hallucinations


def evaluate_agent_faithfulness(agent_output: dict, X_patient_raw: dict) -> float:
    """Calculate the faithfulness score (1.0 - error_rate) for the agent output.

    Score of 1.0 indicates perfect grounding.

    Args:
        agent_output: Response dict from clinical agents.
        X_patient_raw: Raw patient lab trends dict.

    Returns:
        Float score between 0.0 and 1.0.
    """
    hallucinations = detect_ungrounded_claims(agent_output, X_patient_raw)
    
    # Calculate denominator: total assertions
    # we count abnormal patterns and key trajectories
    total_claims = len(agent_output.get("abnormal_patterns", [])) + len(agent_output.get("key_trajectories", []))
    
    if total_claims == 0:
        # Default to 1.0 if agent made no assertions
        return 1.0
        
    error_rate = len(hallucinations) / total_claims
    faithfulness = max(0.0, 1.0 - error_rate)
    
    logger.info(
        "Faithfulness Evaluation: total claims = {}, hallucinations found = {}, score = {:.3f}",
        total_claims,
        len(hallucinations),
        faithfulness
    )
    return float(faithfulness)
