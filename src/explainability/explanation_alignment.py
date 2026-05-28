# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Explanation Alignment Score (EAS) evaluation module.

Evaluates how well the agentic narrative matches the mathematical SHAP feature
importance rankings (Experiment 9).
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr


class ExplanationAlignmentScorer:
    """Computes the Explanation Alignment Score (EAS) between agent notes and SHAP values."""

    def __init__(self, feature_categories: dict[str, str] | None = None) -> None:
        # Default mapping of agent clinical phrases/concepts to base lab feature names
        self.concept_to_feature = feature_categories or {
            "declining_hemoglobin": "hemoglobin",
            "progressive_anemia": "hemoglobin",
            "elevated_nlr": "nlr",
            "rising_nlr": "nlr",
            "elevated_plr": "plr",
            "elevated_sii": "sii",
            "thrombocytosis": "platelets",
            "falling_albumin": "albumin",
            "hypoalbuminemia": "albumin",
            "elevated_alt": "alt",
            "elevated_ast": "ast",
            "elevated_alp": "alp",
            "rising_alp": "alp",
            "jaundice": "bilirubin_total",
            "elevated_crp": "crp",
            "rising_crp": "crp",
            "leukocytosis": "wbc",
            "lymphopenia": "lymphocytes",
        }

    def parse_agent_explanation(self, agent_output: dict) -> list[str]:
        """Extract standardized feature names mentioned in the agent's reasoning.

        Args:
            agent_output: Output dictionary from ClinicalTriageAgent or others.

        Returns:
            List of unique, mapped feature names mentioned.
        """
        mentioned_concepts = set()
        
        # Check abnormal patterns list
        for pat in agent_output.get("abnormal_patterns", []):
            mentioned_concepts.add(str(pat).lower())
            
        # Check key trajectories
        for traj in agent_output.get("key_trajectories", []):
            feat = traj.get("feature")
            if feat:
                mentioned_concepts.add(str(feat).lower())
                
        # Parse free-text trend summaries and reasoning for concept strings
        text_sources = [
            agent_output.get("trend_summary", ""),
            agent_output.get("reasoning", ""),
            agent_output.get("reason", "")
        ]
        
        for text in text_sources:
            if not text:
                continue
            # Simple keyword matching for registered concepts
            text_lower = text.lower()
            for concept in self.concept_to_feature:
                concept_clean = concept.replace("_", " ")
                if concept_clean in text_lower or concept in text_lower:
                    mentioned_concepts.add(concept)
                    
        # Map concepts to standard base feature names
        mapped_features = set()
        for concept in mentioned_concepts:
            # Direct match
            if concept in self.concept_to_feature:
                mapped_features.add(self.concept_to_feature[concept])
            elif concept in self.concept_to_feature.values():
                mapped_features.add(concept)
                
        return sorted(list(mapped_features))

    def compute_alignment_score(
        self,
        agent_mentioned_features: list[str],
        shap_top_k_features: list[str],
        k: int = 10,
    ) -> dict[str, float]:
        """Calculate overlap@k, Jaccard similarity, and rank correlations.

        Args:
            agent_mentioned_features: List of features parsed from agent outputs.
            shap_top_k_features: Top K features based on SHAP absolute attribution.
            k: Depth of check for metrics.

        Returns:
            Dictionary containing alignment metrics.
        """
        agent_set = set(agent_mentioned_features)
        shap_set = set(shap_top_k_features[:k])
        
        if not agent_set or not shap_set:
            return {"overlap_at_k": 0.0, "jaccard": 0.0, "rank_correlation": 0.0}
            
        intersection = agent_set.intersection(shap_set)
        
        # 1. Overlap @ k (percentage of SHAP features identified by agent)
        overlap = len(intersection) / len(shap_set)
        
        # 2. Jaccard Index
        union = agent_set.union(shap_set)
        jaccard = len(intersection) / len(union)
        
        # 3. Spearman Rank Correlation (based on SHAP ranking vs mentions order)
        # We assign rank = index in agent list (as proxy for mention prominence)
        # features not mentioned are given a low default rank
        rank_correlation = 0.0
        if len(intersection) >= 2:
            agent_ranks = {feat: idx for idx, feat in enumerate(agent_mentioned_features)}
            shap_ranks = {feat: idx for idx, feat in enumerate(shap_top_k_features[:k])}
            
            common_feats = list(intersection)
            a_ranks = [agent_ranks[f] for f in common_feats]
            s_ranks = [shap_ranks[f] for f in common_feats]
            
            corr, _ = spearmanr(a_ranks, s_ranks)
            if not np.isnan(corr):
                rank_correlation = float(corr)
                
        return {
            "overlap_at_k": float(overlap),
            "jaccard": float(jaccard),
            "rank_correlation": float(rank_correlation)
        }

    def batch_evaluate(
        self,
        agent_outputs: list[dict],
        shap_values: np.ndarray,
        feature_names: list[str],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Run explanation alignment scoring across a batch of test samples.

        Args:
            agent_outputs: List of agent outputs dictionaries.
            shap_values: 2D array of SHAP values.
            feature_names: Feature names matching SHAP columns.
            top_k: Evaluated depth.

        Returns:
            DataFrame containing alignment scores per sample.
        """
        results = []
        
        for idx, agent_out in enumerate(agent_outputs):
            subj_id = agent_out.get("subject_id", f"sample_{idx}")
            
            # Extract features mentioned by agent
            agent_feats = self.parse_agent_explanation(agent_out)
            
            # Find SHAP rankings for this patient
            patient_shap = np.abs(shap_values[idx])
            # Strip temporal suffixes to get base feature name for matching
            base_feats_in_shap = []
            for f in feature_names:
                base_f = f
                # Strip typical temporal suffixes
                for suffix in ["_mean", "_median", "_std", "_trend_slope", "_delta", "_velocity", "_min", "_max", "_range", "_cv", "_last_value", "_first_value", "_moving_avg_last", "_exp_smooth_last", "_n_measurements", "_n_abnormal"]:
                    if f.endswith(suffix):
                        base_f = f.replace(suffix, "")
                        break
                base_feats_in_shap.append(base_f)
                
            # Aggregate SHAP by base feature name to prevent matching issues
            shap_by_base = pd.DataFrame({"base_feature": base_feats_in_shap, "shap": patient_shap})
            shap_agg = shap_by_base.groupby("base_feature").max().sort_values("shap", ascending=False).reset_index()
            
            top_shap_feats = shap_agg["base_feature"].tolist()
            
            scores = self.compute_alignment_score(agent_feats, top_shap_feats, k=top_k)
            scores["subject_id"] = subj_id
            scores["num_agent_mentions"] = len(agent_feats)
            scores["num_shap_targets"] = min(top_k, len(top_shap_feats))
            
            results.append(scores)
            
        return pd.DataFrame(results)

    def summarize_alignment(self, alignment_df: pd.DataFrame) -> dict[str, float]:
        """Aggregate alignment scores across a study population."""
        if alignment_df.empty:
            return {"mean_jaccard": 0.0, "mean_overlap": 0.0, "mean_rank_corr": 0.0}
            
        summary = {
            "mean_jaccard": float(alignment_df["jaccard"].mean()),
            "std_jaccard": float(alignment_df["jaccard"].std()),
            "mean_overlap": float(alignment_df["overlap_at_k"].mean()),
            "std_overlap": float(alignment_df["overlap_at_k"].std()),
            "mean_rank_corr": float(alignment_df["rank_correlation"].mean()),
        }
        
        logger.info(
            "Explanation Alignment Score Summary: Jaccard = {:.3f} ± {:.3f}, Overlap@K = {:.3f}",
            summary["mean_jaccard"],
            summary["std_jaccard"],
            summary["mean_overlap"]
        )
        
        return summary


def compare_agent_vs_shap_ranking(
    agent_feature_mentions: list[str],
    shap_importance_df: pd.DataFrame,
    top_k: int = 10,
) -> dict:
    """Compare specific features identified by agent vs SHAP importance table.

    Identifies consensus, missed clinical signs, or agent hallucinations.
    """
    agent_set = set(agent_feature_mentions)
    shap_top_set = set(shap_importance_df["feature"].head(top_k))
    
    consensus = list(agent_set.intersection(shap_top_set))
    missed = list(shap_top_set - agent_set)
    hallucinated = list(agent_set - set(shap_importance_df["feature"]))
    
    return {
        "agreement_rate": len(consensus) / max(len(shap_top_set), 1),
        "consensus_features": consensus,
        "missed_important_features": missed,
        "hallucinated_features": hallucinated
    }
