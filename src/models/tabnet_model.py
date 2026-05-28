# Copyright 2024 The Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
TabNet neural network model trainer and built-in self-attention visualizer.
"""

from __future__ import annotations

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import roc_auc_score

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False
    logger.warning("pytorch-tabnet or torch is not installed. TabNet model will run in fallback mode.")


class TabNetTrainer:
    """Trainer wrapper for TabNet neural net classifier with built-in interpretability."""

    def __init__(self, cfg: dict, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed
        self.model = None
        self.feature_names: list[str] = []
        
        # Hyperparameters
        self.n_d = cfg.get("models", {}).get("tabnet", {}).get("n_d", 64)
        self.n_a = cfg.get("models", {}).get("tabnet", {}).get("n_a", 64)
        self.n_steps = cfg.get("models", {}).get("tabnet", {}).get("n_steps", 5)
        self.max_epochs = cfg.get("models", {}).get("tabnet", {}).get("max_epochs", 200)
        self.patience = cfg.get("models", {}).get("tabnet", {}).get("patience", 20)
        
        if HAS_TABNET:
            # Deterministic PyTorch operations
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, Any]:
        """Train TabNet model, handling class weights and training loops."""
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        X_tr = X_train.drop(columns=meta_cols, errors="ignore")
        X_va = X_val.drop(columns=meta_cols, errors="ignore")
        
        self.feature_names = X_tr.columns.tolist()
        
        # Impute missing values for PyTorch feeding
        X_tr_np = X_tr.fillna(X_tr.median()).to_numpy(dtype=np.float32)
        X_va_np = X_va.fillna(X_tr.median()).to_numpy(dtype=np.float32)
        y_tr_np = y_train.to_numpy(dtype=np.int64)
        y_va_np = y_val.to_numpy(dtype=np.int64)
        
        if HAS_TABNET:
            # Compute class weights for cross-entropy
            class_counts = np.bincount(y_tr_np)
            # Inverse frequency weighting
            weights = {0: 1.0, 1: float(class_counts[0] / max(class_counts[1], 1))}
            
            logger.info("Initializing TabNet model (n_d={}, n_a={}, n_steps={})...", self.n_d, self.n_a, self.n_steps)
            self.model = TabNetClassifier(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=1.3,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params=dict(step_size=10, gamma=0.9),
                mask_type="entmax",
                seed=self.seed,
            )
            
            # Train
            self.model.fit(
                X_train=X_tr_np,
                y_train=y_tr_np,
                eval_set=[(X_va_np, y_va_np)],
                eval_name=["val"],
                eval_metric=["auc"],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=128,
                virtual_batch_size=16,
                num_workers=0,
                drop_last=False,
                weights=1,  # Uses class-weighted loss internally if weights=1
            )
            
            y_prob = self.model.predict_proba(X_va_np)[:, 1]
            val_auroc = roc_auc_score(y_val, y_prob)
            importances = self.model.feature_importances_
            
        else:
            # Fallback to Random Forest if TabNet is missing
            logger.warning("TabNet unavailable, falling back to Random Forest.")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=self.seed)
            self.model.fit(X_tr_np, y_tr_np)
            
            y_prob = self.model.predict_proba(X_va_np)[:, 1]
            val_auroc = roc_auc_score(y_val, y_prob)
            importances = self.model.feature_importances_

        logger.info("TabNet training completed. Val AUROC = {:.4f}", val_auroc)
        
        return {
            "model": self,
            "val_auroc": float(val_auroc),
            "feature_importances": importances,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target probabilities."""
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        X_clean = X.drop(columns=meta_cols, errors="ignore").fillna(0.0).to_numpy(dtype=np.float32)
        
        if HAS_TABNET and isinstance(self.model, TabNetClassifier):
            return self.model.predict_proba(X_clean)
        else:
            return self.model.predict_proba(X_clean)

    def get_attention_masks(self, X: pd.DataFrame) -> np.ndarray:
        """Get attention masks indicating which features were attended to per step.

        Returns:
            Numpy array of shape (n_steps, n_samples, n_features)
        """
        meta_cols = ["subject_id", "cancer_type", "gender", "age"]
        X_clean = X.drop(columns=meta_cols, errors="ignore").fillna(0.0).to_numpy(dtype=np.float32)
        
        if HAS_TABNET and isinstance(self.model, TabNetClassifier):
            explain_matrix, masks = self.model.explain(X_clean)
            # masks is a dict of step_idx -> array (n_samples, n_features)
            # Reconstruct to 3D array
            n_samples = X_clean.shape[0]
            n_features = X_clean.shape[1]
            arr = np.zeros((self.n_steps, n_samples, n_features))
            for step in range(self.n_steps):
                if step in masks:
                    arr[step] = masks[step]
            return arr
        else:
            # Mock masks
            n_samples = X_clean.shape[0]
            n_features = X_clean.shape[1]
            return np.ones((self.n_steps, n_samples, n_features)) / n_features

    def plot_attention(self, attention_masks: np.ndarray, output_path: str, top_k: int = 25) -> None:
        """Plot step-wise self-attention heatmap for top features."""
        # Calculate average attention across all samples per step
        # Shape: (n_steps, n_features)
        avg_att = np.mean(attention_masks, axis=1)
        
        # Sort features by total attention across all steps
        total_att = np.sum(avg_att, axis=0)
        top_indices = np.argsort(total_att)[::-1][:top_k]
        
        top_features = [self.feature_names[i] for i in top_indices]
        top_att_matrix = avg_att[:, top_indices].T  # Shape: (top_k, n_steps)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            top_att_matrix,
            annot=True,
            cmap="YlGnBu",
            yticklabels=top_features,
            xticklabels=[f"Step {i+1}" for i in range(self.n_steps)],
            fmt=".3f"
        )
        
        plt.title("TabNet Step-wise Feature Attention Heatmap")
        plt.xlabel("Decision Step")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("TabNet attention plot saved to {}", output_path)

    def save(self, path: str) -> None:
        """Serialize TabNetTrainer wrapper to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved TabNet model to {}", path)

    @classmethod
    def load(cls, path: str) -> TabNetTrainer:
        """Load serialized TabNetTrainer wrapper from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded TabNet model from {}", path)
        return obj
