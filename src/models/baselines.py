# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Baseline ML model training, HPO, calibration, and serialization.
"""

from __future__ import annotations

import os
import time
import pickle
from typing import Any
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

# Disable Optuna logs to keep terminal output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaselineModelTrainer:
    """Manages baseline ML model initialization, training, Optuna HPO, and calibration."""

    def __init__(self, cfg: dict, cancer_type: str | None = None, horizon_months: int = 12, seed: int = 42) -> None:
        self.cfg = cfg
        self.cancer_type = cancer_type
        self.horizon_months = horizon_months
        self.seed = seed

    def get_models(self, scale_pos_weight: float = 1.0) -> dict[str, Any]:
        """Initialize and return baseline models."""
        return {
            "logistic_regression": LogisticRegression(
                C=self.cfg.get("models", {}).get("logistic_regression", {}).get("C", 1.0),
                max_iter=self.cfg.get("models", {}).get("logistic_regression", {}).get("max_iter", 1000),
                class_weight=self.cfg.get("models", {}).get("logistic_regression", {}).get("class_weight", "balanced"),
                random_state=self.seed,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=self.cfg.get("models", {}).get("random_forest", {}).get("n_estimators", 500),
                max_depth=self.cfg.get("models", {}).get("random_forest", {}).get("max_depth", 10),
                class_weight=self.cfg.get("models", {}).get("random_forest", {}).get("class_weight", "balanced"),
                n_jobs=-1,
                random_state=self.seed,
            ),
            "xgboost": XGBClassifier(
                n_estimators=self.cfg.get("models", {}).get("xgboost", {}).get("n_estimators", 500),
                learning_rate=self.cfg.get("models", {}).get("xgboost", {}).get("learning_rate", 0.05),
                max_depth=self.cfg.get("models", {}).get("xgboost", {}).get("max_depth", 6),
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.seed,
                n_jobs=-1,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=self.cfg.get("models", {}).get("lightgbm", {}).get("n_estimators", 500),
                learning_rate=self.cfg.get("models", {}).get("lightgbm", {}).get("learning_rate", 0.05),
                num_leaves=self.cfg.get("models", {}).get("lightgbm", {}).get("num_leaves", 63),
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            ),
            "catboost": CatBoostClassifier(
                iterations=self.cfg.get("models", {}).get("catboost", {}).get("iterations", 500),
                learning_rate=self.cfg.get("models", {}).get("catboost", {}).get("learning_rate", 0.05),
                depth=self.cfg.get("models", {}).get("catboost", {}).get("depth", 6),
                auto_class_weights="Balanced",
                random_state=self.seed,
                verbose=0,
            ),
        }

    def train_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Train a single model with logging and timing."""
        # Drop metadata columns for training
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        X_tr = X_train.drop(columns=meta_cols, errors="ignore").fillna(0.0)
        X_va = X_val.drop(columns=meta_cols, errors="ignore").fillna(0.0)
        
        start_time = time.time()
        
        # Fit models, handling early stopping for gradient boosters
        if model_name == "xgboost":
            model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
                verbose=False
            )
        elif model_name == "lightgbm":
            # For LightGBM compatibility
            model.fit(
                X_tr, y_train,
                eval_set=[(X_va, y_val)],
                callbacks=[]
            )
        elif model_name == "catboost":
            model.fit(
                X_tr, y_train,
                eval_set=(X_va, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model.fit(X_tr, y_train)
            
        train_time = time.time() - start_time
        
        # Predict probability and calculate validation AUROC
        y_prob = model.predict_proba(X_va)[:, 1]
        val_auroc = roc_auc_score(y_val, y_prob)
        
        logger.info(
            "Model '{}' training completed in {:.2f}s. Val AUROC = {:.4f}",
            model_name,
            train_time,
            val_auroc,
        )
        
        return {
            "model": model,
            "val_auroc": float(val_auroc),
            "train_time": float(train_time),
        }

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, dict[str, Any]]:
        """Train all baseline models and return results."""
        # Calculate case-control ratio for XGBoost scale_pos_weight
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        ratio = float(num_neg / max(num_pos, 1))
        
        models = self.get_models(scale_pos_weight=ratio)
        results = {}
        
        for name, model in models.items():
            try:
                res = self.train_model(name, model, X_train, y_train, X_val, y_val)
                results[name] = res
            except Exception as exc:
                logger.error("Failed to train model '{}': {}", name, exc)
                
        return results

    def calibrate_model(
        self,
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        method: str = "isotonic",
    ) -> CalibratedClassifierCV:
        """Wrap model in calibration layer using validation set."""
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        X_va = X_val.drop(columns=meta_cols, errors="ignore").fillna(0.0)
        
        logger.info("Calibrating model using method '{}'...", method)
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit"
        )
        calibrated.fit(X_va, y_val)
        return calibrated

    def hyperparameter_search(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 20,
    ) -> dict[str, Any]:
        """Optimize hyperparameters for XGBoost or LightGBM using Optuna."""
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        X_tr = X_train.drop(columns=meta_cols, errors="ignore").fillna(0.0)
        X_va = X_val.drop(columns=meta_cols, errors="ignore").fillna(0.0)
        
        logger.info("Starting Optuna HPO search for {} ({} trials)...", model_name, n_trials)
        
        def objective(trial):
            if model_name == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "eval_metric": "logloss",
                    "random_state": self.seed,
                    "n_jobs": -1
                }
                clf = XGBClassifier(**params)
                clf.fit(X_tr, y_train, verbose=False)
            elif model_name == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "class_weight": "balanced",
                    "random_state": self.seed,
                    "n_jobs": -1,
                    "verbose": -1
                }
                clf = LGBMClassifier(**params)
                clf.fit(X_tr, y_train)
            else:
                raise ValueError("HPO only supported for xgboost or lightgbm baseline.")
                
            y_prob = clf.predict_proba(X_va)[:, 1]
            return roc_auc_score(y_val, y_prob)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info("Best HPO parameters found: {}", study.best_params)
        logger.info("Best HPO AUROC = {:.4f}", study.best_value)
        
        return study.best_params

    def save_model(self, model: Any, output_path: str) -> None:
        """Serialize a model to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to {}", output_path)

    @classmethod
    def load_model(cls, model_path: str) -> Any:
        """Load a serialized model from disk."""
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded from {}", model_path)
        return model
