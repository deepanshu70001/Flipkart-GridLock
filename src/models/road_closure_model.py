"""
Road Closure Prediction Model
==============================
CatBoost classifier for predicting whether an event requires road closure.
Handles severe class imbalance (only 8.3% positive) using class weights
and threshold optimization.
"""

import numpy as np
import pandas as pd
import yaml
import joblib
import os
import warnings
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class RoadClosureModel:
    """
    Classifier for predicting road closure requirement.
    
    Uses CatBoost (primary) with LightGBM fallback.
    Handles class imbalance via:
    - Automatic class weighting
    - Threshold optimization on validation set
    - SMOTE-aware evaluation
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config["models"]["closure"]
        self.random_state = self.config["models"]["random_state"]
        self.model = None
        self.optimal_threshold = 0.5
        self.feature_importances = None
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the road closure classifier.
        
        Automatically handles class imbalance via class weights.
        If validation data is provided, optimizes the decision threshold
        for maximum F1 score.
        """
        print("=" * 60)
        print("TRAINING ROAD CLOSURE MODEL")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Positive rate: {y_train.mean():.3f} ({y_train.sum()} closures)")

        # Compute class weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        print(f"Scale pos weight: {scale_pos_weight:.2f}")

        if HAS_CATBOOST:
            cb_params = self.model_config["catboost_params"].copy()
            cb_params["random_seed"] = self.random_state
            cb_params["verbose"] = 0
            cb_params["loss_function"] = "Logloss"
            cb_params["auto_class_weights"] = "Balanced"
            self.model = CatBoostClassifier(**cb_params)
            self.model.fit(X_train, y_train)
            print("  CatBoost model trained")
        else:
            # Fallback to LightGBM with class weights
            lgbm_params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 7,
                "num_leaves": 63,
                "min_child_samples": 20,
                "scale_pos_weight": scale_pos_weight,
                "random_state": self.random_state,
                "verbose": -1,
                "objective": "binary",
            }
            self.model = lgb.LGBMClassifier(**lgbm_params)
            self.model.fit(X_train, y_train)
            print("  LightGBM model trained (CatBoost fallback)")

        # Feature importances
        self.feature_importances = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)

        self._fitted = True

        # ── Optimize threshold ──
        if X_val is not None and y_val is not None:
            self._optimize_threshold(X_val, y_val)
            self.evaluate(X_val, y_val)

        return self

    def _optimize_threshold(self, X_val, y_val):
        """Find the optimal threshold that maximizes F1 score."""
        y_proba = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

        # Compute F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
        print(f"  Optimal threshold: {self.optimal_threshold:.3f} (F1: {f1_scores[best_idx]:.4f})")

    def predict_proba(self, X):
        """Predict probability of road closure."""
        assert self._fitted, "Model not fitted!"
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=None):
        """Predict binary class using optimized threshold."""
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X_val, y_val):
        """Comprehensive evaluation on validation set."""
        y_proba = self.predict_proba(X_val)
        y_pred = self.predict(X_val)

        print("\n" + "─" * 40)
        print("ROAD CLOSURE MODEL — VALIDATION RESULTS")
        print("─" * 40)

        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)

        print(f"  AUC:        {auc:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  Threshold:  {self.optimal_threshold:.3f}")
        print(f"\n  Classification Report:\n{classification_report(y_val, y_pred, target_names=['No Closure', 'Closure'])}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

        self.metrics = {
            "roc_auc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": self.optimal_threshold,
        }
        return self.metrics

    def get_top_features(self, n=20):
        """Return top N features."""
        if self.feature_importances is not None:
            return self.feature_importances.head(n)
        return None

    def save(self, path="outputs/models/closure_model.joblib"):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "optimal_threshold": self.optimal_threshold,
                "feature_importances": self.feature_importances,
                "metrics": getattr(self, "metrics", {}),
            },
            path,
        )
        print(f"[RoadClosureModel] Saved to {path}")

    def load(self, path="outputs/models/closure_model.joblib"):
        """Load model."""
        data = joblib.load(path)
        self.model = data["model"]
        self.optimal_threshold = data["optimal_threshold"]
        self.feature_importances = data["feature_importances"]
        self.metrics = data.get("metrics", {})
        self._fitted = True
        print(f"[RoadClosureModel] Loaded from {path}")
