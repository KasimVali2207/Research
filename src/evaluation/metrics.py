# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Clinical performance evaluation metrics.

Computes AUROC, AUPRC, Sensitivity, Specificity, F1, ECE, Brier score,
and implements bootstrap confidence intervals and the DeLong test for ROC comparison.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
from loguru import logger
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, brier_score_loss


def calculate_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Calculate standard diagnostic performance metrics.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probability estimates.
        threshold: Classification cutoff.

    Returns:
        Dictionary of performance scores.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Check if we have at least one of each class for ROC/PR calculations
    if len(np.unique(y_true)) < 2:
        auroc = 0.5
        auprc = 0.0
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = float(auc(recall, precision))
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0  # positive predictive value / precision
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0  # negative predictive value
    
    f1 = float(2 * (ppv * sensitivity) / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0
    
    # Calibration metrics
    brier = float(brier_score_loss(y_true, y_prob))
    ece = calculate_ece(y_true, y_prob)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1_score": f1,
        "brier_score": brier,
        "ece": ece,
    }


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error calculation."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "auroc",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate metric with bootstrap confidence intervals.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probability estimates.
        metric_name: Key in calculate_clinical_metrics dict.
        n_bootstrap: Resampling iterations count.
        ci_level: Confidence level (e.g., 0.95).
        seed: Random seed.

    Returns:
        Tuple of (mean_estimate, lower_ci, upper_ci).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    np.random.seed(seed)
    scores = []
    
    # Calculate baseline
    baseline_metrics = calculate_clinical_metrics(y_true, y_prob)
    baseline_score = baseline_metrics.get(metric_name, 0.0)
    
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        boot_y_true = y_true[indices]
        boot_y_prob = y_prob[indices]
        
        # Ensure we have both classes represented in bootstrap sample
        if len(np.unique(boot_y_true)) < 2:
            continue
            
        metrics = calculate_clinical_metrics(boot_y_true, boot_y_prob)
        scores.append(metrics.get(metric_name, 0.0))
        
    if not scores:
        return baseline_score, baseline_score, baseline_score
        
    alpha = (1.0 - ci_level) / 2.0
    lower_idx = int(alpha * len(scores))
    upper_idx = int((1.0 - alpha) * len(scores))
    
    scores_sorted = sorted(scores)
    
    lower = scores_sorted[max(0, lower_idx)]
    upper = scores_sorted[min(len(scores) - 1, upper_idx)]
    
    return float(baseline_score), float(lower), float(upper)


# DeLong Test implementation based on structural components
def delong_roc_variance(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Compute AUC and its variance using structural components of DeLong's method."""
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    
    # Get case-control splits
    pos = y_true == 1
    neg = y_true == 0
    
    m = np.sum(neg)  # controls
    n = np.sum(pos)  # cases
    
    k_pos = y_prob[pos]
    k_neg = y_prob[neg]
    
    # Structural components
    v_10 = np.zeros(n)
    v_01 = np.zeros(m)
    
    # Mid-rank calculations for ties
    for i in range(n):
        v_10[i] = np.sum(k_neg < k_pos[i]) / m
        # handle ties
        v_10[i] += np.sum(k_neg == k_pos[i]) / (2 * m)
        
    for j in range(m):
        v_01[j] = np.sum(k_pos > k_neg[j]) / n
        v_01[j] += np.sum(k_pos == k_neg[j]) / (2 * n)
        
    auc = np.mean(v_10)
    
    # Variances
    s_10 = np.var(v_10, ddof=1) if n > 1 else 0.0
    s_01 = np.var(v_01, ddof=1) if m > 1 else 0.0
    
    variance = (s_10 / n) + (s_01 / m)
    return float(auc), float(variance)


def delong_roc_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> float:
    """Compare two ROC curves using the DeLong covariance method.

    Returns:
        One-tailed or two-tailed p-value.
    """
    y_true = np.array(y_true)
    y_prob_a = np.array(y_prob_a)
    y_prob_b = np.array(y_prob_b)
    
    auc_a, var_a = delong_roc_variance(y_true, y_prob_a)
    auc_b, var_b = delong_roc_variance(y_true, y_prob_b)
    
    # Compute covariance term via structural components
    pos = y_true == 1
    neg = y_true == 0
    m = np.sum(neg)
    n = np.sum(pos)
    
    k_pos_a = y_prob_a[pos]
    k_neg_a = y_prob_a[neg]
    k_pos_b = y_prob_b[pos]
    k_neg_b = y_prob_b[neg]
    
    v_10_a = np.zeros(n)
    v_01_a = np.zeros(m)
    v_10_b = np.zeros(n)
    v_01_b = np.zeros(m)
    
    for i in range(n):
        v_10_a[i] = np.sum(k_neg_a < k_pos_a[i]) / m + np.sum(k_neg_a == k_pos_a[i]) / (2 * m)
        v_10_b[i] = np.sum(k_neg_b < k_pos_b[i]) / m + np.sum(k_neg_b == k_pos_b[i]) / (2 * m)
        
    for j in range(m):
        v_01_a[j] = np.sum(k_pos_a > k_neg_a[j]) / n + np.sum(k_pos_a == k_neg_a[j]) / (2 * n)
        v_01_b[j] = np.sum(k_pos_b > k_neg_b[j]) / n + np.sum(k_pos_b == k_neg_b[j]) / (2 * n)
        
    cov_10 = np.cov(v_10_a, v_10_b)[0, 1] if n > 1 else 0.0
    cov_01 = np.cov(v_01_a, v_01_b)[0, 1] if m > 1 else 0.0
    
    covariance = (cov_10 / n) + (cov_01 / m)
    
    # Standard error of difference
    sd_diff = np.sqrt(max(1e-15, var_a + var_b - 2 * covariance))
    
    z = (auc_a - auc_b) / sd_diff
    p_value = 2 * (1.0 - stats.norm.cdf(abs(z)))
    
    logger.info("DeLong Test: AUC_A = {:.4f}, AUC_B = {:.4f}, Z = {:.4f}, p-value = {:.4e}", auc_a, auc_b, z, p_value)
    return float(p_value)
