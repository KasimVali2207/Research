# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Temporal feature engineering from serial lab measurements.

The core novelty: instead of treating labs as a single snapshot, we extract
trajectory statistics that mimic how a clinician reads a trending lab panel.
A declining hemoglobin over 6 months tells a different story than a single
low reading — this module captures that story quantitatively.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Core Temporal Feature Extractor
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms a long-format longitudinal lab DataFrame into a wide-format
    feature matrix with one row per subject.

    Expected input schema:
        subject_id | charttime | feature_name | value

    Output: one row per subject_id, columns = {feature}_{stat}
    """

    DEFAULT_STATS = [
        "mean", "median", "std", "min", "max",
        "trend_slope", "delta", "velocity",
        "moving_avg_last", "exp_smooth_last",
        "n_measurements", "n_abnormal",
        "first_value", "last_value",
        "range", "cv",  # coefficient of variation — captures volatility
    ]

    def __init__(
        self,
        feature_names: list[str],
        stats: Optional[list[str]] = None,
        normal_ranges: Optional[dict] = None,
        exp_smooth_alpha: float = 0.3,
        moving_avg_window: int = 3,
    ) -> None:
        self.feature_names = feature_names
        self.stats = stats or self.DEFAULT_STATS
        self.normal_ranges = normal_ranges or {}
        self.exp_smooth_alpha = exp_smooth_alpha
        self.moving_avg_window = moving_avg_window

    # ------------------------------------------------------------------
    def fit(self, longitudinal_df: pd.DataFrame, y=None) -> "TemporalFeatureExtractor":
        required = {"subject_id", "charttime", "feature_name", "value"}
        missing = required - set(longitudinal_df.columns)
        if missing:
            raise ValueError(f"longitudinal_df missing columns: {missing}")
        return self  # stateless — nothing to learn

    # ------------------------------------------------------------------
    def transform(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main transformation: per-subject, per-feature trajectory stats.

        Patients with a single measurement get slope=NaN, delta=0, velocity=0.
        Patients with zero measurements get all-NaN (handled by imputer downstream).
        """
        # Ensure charttime is datetime
        df = longitudinal_df.copy()
        df["charttime"] = pd.to_datetime(df["charttime"])
        df = df.sort_values(["subject_id", "feature_name", "charttime"])

        all_subjects = df["subject_id"].unique()
        records = []

        for subject_id, subj_df in df.groupby("subject_id"):
            row: dict = {"subject_id": subject_id}

            for feat in self.feature_names:
                feat_df = subj_df[subj_df["feature_name"] == feat].copy()
                feat_df = feat_df.dropna(subset=["value"]).sort_values("charttime")

                prefix = feat
                feat_stats = self._compute_feature_stats(feat_df, feat)

                for stat_name, val in feat_stats.items():
                    row[f"{prefix}_{stat_name}"] = val

            records.append(row)

        # Subjects with zero lab data at all still need a row
        subject_ids_in_data = {r["subject_id"] for r in records}
        missing_subjects = set(all_subjects) - subject_ids_in_data
        for sid in missing_subjects:
            records.append({"subject_id": sid})

        result = pd.DataFrame(records)
        logger.info(
            f"Temporal features: {result.shape[1]-1} features for "
            f"{result.shape[0]} subjects."
        )
        return result

    # ------------------------------------------------------------------
    def fit_transform(self, longitudinal_df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(longitudinal_df).transform(longitudinal_df)

    # ------------------------------------------------------------------
    def _compute_feature_stats(
        self, feat_df: pd.DataFrame, feat_name: str
    ) -> dict[str, float]:
        """Computes all trajectory statistics for one patient × one feature."""
        stats_out: dict[str, float | int] = {}

        n = len(feat_df)
        stats_out["n_measurements"] = n

        if n == 0:
            # All stats NaN — imputer will handle
            for s in self.stats:
                if s != "n_measurements":
                    stats_out[s] = np.nan
            return stats_out

        values = feat_df["value"].to_numpy(dtype=float)
        times = feat_df["charttime"].to_numpy()
        times_days = _to_days_since_first(times)

        # Basic descriptives
        stats_out["mean"] = float(np.nanmean(values))
        stats_out["median"] = float(np.nanmedian(values))
        stats_out["std"] = float(np.nanstd(values)) if n > 1 else 0.0
        stats_out["min"] = float(np.nanmin(values))
        stats_out["max"] = float(np.nanmax(values))
        stats_out["range"] = stats_out["max"] - stats_out["min"]
        stats_out["cv"] = (
            stats_out["std"] / abs(stats_out["mean"])
            if stats_out["mean"] != 0 else 0.0
        )
        stats_out["first_value"] = float(values[0])
        stats_out["last_value"] = float(values[-1])

        # Trend: slope of linear regression (value vs. days) — the key novelty
        stats_out["trend_slope"] = compute_trajectory_trend(values, times_days)

        # Delta and velocity
        if n >= 2:
            elapsed = times_days[-1] - times_days[0]
            stats_out["delta"] = float(values[-1] - values[0])
            stats_out["velocity"] = (
                stats_out["delta"] / elapsed if elapsed > 0 else 0.0
            )
        else:
            stats_out["delta"] = 0.0
            stats_out["velocity"] = 0.0

        # Smoothed estimates
        stats_out["moving_avg_last"] = float(
            _moving_average(values, self.moving_avg_window)[-1]
        )
        stats_out["exp_smooth_last"] = compute_exp_smooth(
            values, self.exp_smooth_alpha
        )

        # Count abnormal measurements (for clinical context)
        if feat_name in self.normal_ranges:
            lo = self.normal_ranges[feat_name]["low"]
            hi = self.normal_ranges[feat_name]["high"]
            n_abnormal = int(np.sum((values < lo) | (values > hi)))
        else:
            n_abnormal = 0
        stats_out["n_abnormal"] = n_abnormal

        return stats_out


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def compute_trajectory_trend(
    values: np.ndarray, times_days: np.ndarray
) -> float:
    """
    Linear regression slope of lab value over time (units: value/day).

    Returns NaN for < 2 points — slope is meaningless with a single measurement.
    """
    if len(values) < 2:
        return np.nan

    # Remove NaN pairs
    mask = ~(np.isnan(values) | np.isnan(times_days))
    v, t = values[mask], times_days[mask]
    if len(v) < 2:
        return np.nan

    # scipy linregress fails if all x values are identical (e.g. same-day measurements)
    if len(np.unique(t)) < 2:
        return 0.0

    slope, _, _, _, _ = stats.linregress(t, v)
    return float(slope)


def compute_exp_smooth(values: np.ndarray, alpha: float = 0.3) -> float:
    """
    Simple exponential smoothing. Returns the last smoothed value.

    alpha=0.3 gives more weight to recent measurements — appropriate since
    labs closer to diagnosis are more clinically relevant.
    """
    if len(values) == 0:
        return np.nan

    s = float(values[0])
    for v in values[1:]:
        if not np.isnan(v):
            s = alpha * v + (1 - alpha) * s
    return s


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average with edge padding."""
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _to_days_since_first(times: np.ndarray) -> np.ndarray:
    """Converts datetime64 array to float days since the earliest timestamp."""
    t0 = times[0]
    return (times - t0).astype("timedelta64[h]").astype(float) / 24.0


# ---------------------------------------------------------------------------
# Static snapshot (Experiment 2 baseline)
# ---------------------------------------------------------------------------

def create_static_snapshot(
    longitudinal_df: pd.DataFrame,
    method: str = "last",
    horizon_date_col: str = "horizon_date",
) -> pd.DataFrame:
    """
    Creates a static single-timepoint feature set from longitudinal data.

    Used in Experiment 2 to establish the temporal vs. static comparison.
    Without this, we can't attribute performance gains to temporal modeling.

    Args:
        method: 'last' (most recent), 'mean' (all-time mean), or
                'closest_to_horizon' (measurement closest to prediction horizon).
    """
    df = longitudinal_df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])

    records = []

    for subject_id, subj_df in df.groupby("subject_id"):
        row: dict = {"subject_id": subject_id}

        for feat_name, feat_df in subj_df.groupby("feature_name"):
            feat_df = feat_df.dropna(subset=["value"]).sort_values("charttime")

            if len(feat_df) == 0:
                row[feat_name] = np.nan
                continue

            if method == "last":
                row[feat_name] = float(feat_df["value"].iloc[-1])
            elif method == "mean":
                row[feat_name] = float(feat_df["value"].mean())
            elif method == "closest_to_horizon":
                if horizon_date_col in feat_df.columns:
                    horizon = feat_df[horizon_date_col].iloc[0]
                    idx = (feat_df["charttime"] - horizon).abs().idxmin()
                    row[feat_name] = float(feat_df.loc[idx, "value"])
                else:
                    row[feat_name] = float(feat_df["value"].iloc[-1])
            else:
                raise ValueError(f"Unknown method: {method}")

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Second-derivative acceleration (bonus feature for detecting rapid deterioration)
# ---------------------------------------------------------------------------

def compute_change_acceleration(
    longitudinal_df: pd.DataFrame, feature_name: str
) -> pd.Series:
    """
    Computes the second derivative of a lab trajectory per patient.

    Detecting acceleration in decline (e.g., hemoglobin dropping faster each
    month) is a stronger signal than detecting decline alone. This is the
    clinical equivalent of 'velocity of deterioration.'

    Returns a Series indexed by subject_id.
    """
    df = longitudinal_df[longitudinal_df["feature_name"] == feature_name].copy()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df = df.sort_values(["subject_id", "charttime"])

    results = {}

    for subject_id, subj_df in df.groupby("subject_id"):
        subj_df = subj_df.dropna(subset=["value"])
        if len(subj_df) < 3:
            results[subject_id] = np.nan
            continue

        values = subj_df["value"].to_numpy(dtype=float)
        times_days = _to_days_since_first(subj_df["charttime"].to_numpy())

        # Split into two halves and compare slopes
        mid = len(values) // 2
        slope_early = compute_trajectory_trend(values[:mid+1], times_days[:mid+1])
        slope_late = compute_trajectory_trend(values[mid:], times_days[mid:])

        if np.isnan(slope_early) or np.isnan(slope_late):
            results[subject_id] = np.nan
        else:
            results[subject_id] = slope_late - slope_early  # acceleration

    return pd.Series(results, name=f"{feature_name}_acceleration")
