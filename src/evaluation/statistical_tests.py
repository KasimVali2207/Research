# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Statistical significance tests for diagnostic models.

Implements McNemar's test for paired classification choices, Wilcoxon signed-rank
tests for continuous distributions, and Benjamini-Hochberg FDR correction (Experiment 1 & 2).
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
from loguru import logger

from src.evaluation.metrics import delong_roc_test


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """Run McNemar's test on paired binary predictions.

    Tests whether the error rates of Model A and Model B are significantly different.
    Contingency table:
                     Model B Correct  Model B Incorrect
      Model A Correct       n_00             n_01
      Model A Incorrect     n_10             n_11

    McNemar Chi2 = (|n_01 - n_10| - 1)^2 / (n_01 + n_10) with Edwards continuity correction.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from Model A.
        y_pred_b: Predictions from Model B.

    Returns:
        p-value.
    """
    y_true = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    
    # Discordant cells
    n_01 = np.sum(correct_a & ~correct_b)  # A correct, B incorrect
    n_10 = np.sum(~correct_a & correct_b)  # A incorrect, B correct
    
    discordant_sum = n_01 + n_10
    
    if discordant_sum == 0:
        return 1.0
        
    if discordant_sum < 25:
        # Exact binomial test for small sample sizes
        # p-value is cumulative binomial probability of getting min(n_01, n_10) or fewer successes out of discordant_sum trials
        k = min(n_01, n_10)
        p_val = 2.0 * stats.binom.cdf(k, discordant_sum, 0.5)
        # Cap p-value at 1.0
        p_val = min(1.0, p_val)
    else:
        # Chi-square approximation with continuity correction
        chi2 = (abs(n_01 - n_10) - 1.0) ** 2 / discordant_sum
        p_val = stats.chi2.sf(chi2, df=1)
        
    logger.info(
        "McNemar Test: A correct/B incorrect = {}, A incorrect/B correct = {}. p-value = {:.4e}",
        n_01,
        n_10,
        p_val,
    )
    return float(p_val)


def delong_test_wrapper(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> float:
    """Wrapper around DeLong ROC significance test."""
    return delong_roc_test(y_true, y_prob_a, y_prob_b)


def wilcoxon_signed_rank_test(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Run Wilcoxon signed-rank test comparing paired score distributions.

    Used to compare patient-level risk predictions or bootstrap scores.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    if np.array_equal(scores_a, scores_b):
        return 1.0
        
    _, p_val = stats.wilcoxon(scores_a, scores_b)
    logger.info("Wilcoxon Signed-Rank Test: p-value = {:.4e}", p_val)
    return float(p_val)


def fdr_correction(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) p-value correction.

    Args:
        p_values: List of raw p-values.
        alpha: Target FDR rate (e.g. 0.05).

    Returns:
        Tuple of (rejected_boolean_mask, corrected_p_values).
    """
    p_vals = np.array(p_values)
    n = len(p_vals)
    
    if n == 0:
        return np.array([]), np.array([])
        
    # Sort indices
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    
    # Calculate Benjamini-Hochberg step cutoffs
    q_vals = np.zeros(n)
    
    # Corrected p-values: p_adj_i = p_i * n / rank_i
    # taking cumulative minimum from tail to ensure monotonicity
    prev_q = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        q = sorted_p[idx] * (n / rank)
        q = min(q, prev_q)
        q_vals[idx] = q
        prev_q = q
        
    # Restore original order
    adj_p_values = np.zeros(n)
    adj_p_values[sorted_indices] = q_vals
    
    rejected = adj_p_values <= alpha
    
    logger.info(
        "FDR BH Correction completed: rejected {}/{} hypotheses at alpha={}",
        np.sum(rejected),
        n,
        alpha,
    )
    return rejected, adj_p_values
