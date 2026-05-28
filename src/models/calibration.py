# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Calibration diagnostics, visualization, and adjustment module.

Implements calibration metrics (ECE, MCE), isotonic calibration, and temperature
scaling for early cancer risk predictions, producing reliability diagrams (Experiment 4).
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Calculate ECE, MCE, Brier score, and bin-wise reliability metrics.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probability estimates.
        n_bins: Number of confidence bins.

    Returns:
        Dictionary of metrics and reliability curves data.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    
    mean_predicted_value = []
    fraction_of_positives = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Filter samples in current bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_count = np.sum(in_bin)
        bin_counts.append(int(bin_count))
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            
            mean_predicted_value.append(float(avg_confidence_in_bin))
            fraction_of_positives.append(float(accuracy_in_bin))
            
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
        else:
            mean_predicted_value.append(float((bin_lower + bin_upper) / 2.0))
            fraction_of_positives.append(0.0)
            
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "reliability_data": {
            "mean_predicted_value": mean_predicted_value,
            "fraction_of_positives": fraction_of_positives,
            "bin_counts": bin_counts,
            "bin_boundaries": bin_boundaries.tolist()
        }
    }


def plot_calibration_curve(
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Calibration Curves (Reliability Diagram)",
) -> None:
    """Plot reliability diagram comparing multiple models on the same plot.

    Args:
        models_dict: Dict mapping model_name -> (y_true, y_prob).
        output_path: Path to save the plot.
        title: Title of the chart.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})
    
    # Perfect calibration diagonal
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for model_name, (y_true, y_prob) in models_dict.items():
        metrics = compute_calibration_metrics(y_true, y_prob)
        rel_data = metrics["reliability_data"]
        
        # Plot reliability curve
        ax1.plot(
            rel_data["mean_predicted_value"],
            rel_data["fraction_of_positives"],
            "s-",
            label=f"{model_name} (ECE={metrics['ece']:.3f})"
        )
        
        # Plot predicted probabilities histogram
        ax2.hist(
            y_prob,
            bins=10,
            label=model_name,
            histtype="step",
            lw=2,
            alpha=0.8
        )
        
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(title)
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Calibration reliability curve saved to {}", output_path)


def isotonic_calibration(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit IsotonicRegression on validation probabilities and return calibrated test probabilities.

    Args:
        model:Prefit model classifier.
        X_val: Validation features.
        y_val: Validation true labels.
        X_test: Test features to predict.

    Returns:
        Calibrated probabilities on test set.
    """
    meta_cols = ["subject_id", "cancer_type", "gender", "age"]
    X_va = X_val.drop(columns=meta_cols, errors="ignore").fillna(0.0)
    X_te = X_test.drop(columns=meta_cols, errors="ignore").fillna(0.0)
    
    # Prefit probabilities
    val_probs = model.predict_proba(X_va)[:, 1]
    test_probs = model.predict_proba(X_te)[:, 1]
    
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)
    
    calibrated_test_probs = iso.predict(test_probs)
    return calibrated_test_probs


def temperature_scaling(logits: np.ndarray, y_val: np.ndarray) -> tuple[float, np.ndarray]:
    """Learn optimal temperature parameter T minimizing negative log likelihood on validation set.

    Calibrated prob = sigmoid(logits / T).

    Args:
        logits: Uncalibrated classifier logits (pre-activation outputs).
        y_val: Validation labels.

    Returns:
        Tuple of (optimal_temperature, calibrated_probabilities).
    """
    y_val = np.array(y_val)
    
    # Loss function (negative log likelihood / cross entropy)
    def nll_loss(t):
        temp = t[0]
        # Avoid division by zero
        if temp <= 0:
            return 1e9
        scaled_logits = logits / temp
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        loss = -np.mean(y_val * np.log(probs) + (1.0 - y_val) * np.log(1.0 - probs))
        return loss

    # Optimize T
    init_temp = 1.0
    res = minimize(nll_loss, [init_temp], bounds=[(0.01, 10.0)], method="L-BFGS-B")
    optimal_T = float(res.x[0])
    
    # Calculate calibrated probabilities
    scaled_logits = logits / optimal_T
    calibrated_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    
    logger.info("Temperature scaling search complete: optimal T = {:.4f}", optimal_T)
    return optimal_T, calibrated_probs


def calibration_comparison_table(results_dict: dict) -> pd.DataFrame:
    """Create a summary comparison table showing calibration improvement.

    Input structure:
      {
        model_name: {
          'y_true': np.ndarray,
          'y_prob_uncal': np.ndarray,
          'y_prob_cal': np.ndarray
        }
      }

    Returns:
        DataFrame table.
    """
    rows = []
    for model_name, data in results_dict.items():
        y_true = data["y_true"]
        uncal = compute_calibration_metrics(y_true, data["y_prob_uncal"])
        cal = compute_calibration_metrics(y_true, data["y_prob_cal"])
        
        rows.append({
            "model": model_name,
            "ECE_before": round(uncal["ece"], 4),
            "ECE_after": round(cal["ece"], 4),
            "Brier_before": round(uncal["brier"], 4),
            "Brier_after": round(cal["brier"], 4)
        })
        
    return pd.DataFrame(rows)
