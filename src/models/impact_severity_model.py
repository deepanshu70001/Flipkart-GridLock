"""
Impact Severity Model
=====================
Stacked ensemble for predicting event priority (High/Low).
Uses LightGBM, XGBoost, CatBoost, and Random Forest as base learners,
with Logistic Regression as the meta-learner.

Evaluation: ROC-AUC, F1, Precision, Recall with temporal cross-validation.
"""

import numpy as np
import pandas as pd
import yaml
import joblib
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# CatBoost is optional (heavier dependency)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[SeverityModel] CatBoost not available, using 3-model ensemble")


class SeverityModel:
    """
    Stacked ensemble classifier for event severity (priority) prediction.
    
    Architecture:
        Layer 1: LightGBM + XGBoost + CatBoost + RandomForest
        Layer 2: Logistic Regression meta-learner
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config["models"]["severity"]
        self.random_state = self.config["models"]["random_state"]
        self.base_models = {}
        self.meta_model = None
        self.feature_importances = None
        self._fitted = False

    def _build_base_models(self):
        """Initialize base learner models."""
        models = {}

        # LightGBM
        lgbm_params = self.model_config["lgbm_params"].copy()
        lgbm_params["random_state"] = self.random_state
        lgbm_params["verbose"] = -1
        lgbm_params["objective"] = "binary"
        models["lgbm"] = lgb.LGBMClassifier(**lgbm_params)

        # XGBoost
        xgb_params = self.model_config["xgb_params"].copy()
        xgb_params["random_state"] = self.random_state
        xgb_params["eval_metric"] = "auc"
        xgb_params["use_label_encoder"] = False
        models["xgb"] = xgb.XGBClassifier(**xgb_params)

        # CatBoost
        if HAS_CATBOOST:
            cb_params = self.model_config["catboost_params"].copy()
            cb_params["random_seed"] = self.random_state
            cb_params["verbose"] = 0
            cb_params["loss_function"] = "Logloss"
            models["catboost"] = CatBoostClassifier(**cb_params)

        # Random Forest
        rf_params = self.model_config["rf_params"].copy()
        rf_params["random_state"] = self.random_state
        rf_params["n_jobs"] = -1
        models["rf"] = RandomForestClassifier(**rf_params)

        return models

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the stacked ensemble.
        
        Step 1: Generate cross-validated predictions from base models.
        Step 2: Train meta-learner on stacked predictions.
        Step 3: Refit base models on full training data.
        """
        print("=" * 60)
        print("TRAINING SEVERITY MODEL (Stacked Ensemble)")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Class balance: {y_train.mean():.3f} (High priority rate)")

        self.base_models = self._build_base_models()
        n_models = len(self.base_models)

        # ── Step 1: Cross-validated base model predictions ──
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        meta_train = np.zeros((len(X_train), n_models))

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            for model_idx, (name, model) in enumerate(self.base_models.items()):
                # Clone model for this fold
                model_clone = self._clone_model(name)
                model_clone.fit(X_fold_train, y_fold_train)
                meta_train[val_idx, model_idx] = model_clone.predict_proba(X_fold_val)[:, 1]

            print(f"  Fold {fold_idx + 1}/5 complete")

        # ── Step 2: Train meta-learner ──
        self.meta_model = LogisticRegression(
            C=1.0, random_state=self.random_state, max_iter=1000
        )
        self.meta_model.fit(meta_train, y_train)
        print(f"  Meta-learner trained on stacked predictions")

        # ── Step 3: Refit base models on full training data ──
        for name, model in self.base_models.items():
            model.fit(X_train, y_train)
            print(f"  {name} refitted on full training data")

        # Feature importances from LightGBM (most interpretable)
        self.feature_importances = pd.Series(
            self.base_models["lgbm"].feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)

        self._fitted = True

        # ── Evaluate ──
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)

        return self

    def predict_proba(self, X):
        """Predict probability of High priority."""
        assert self._fitted, "Model not fitted!"
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.base_models.values()
        ])
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def predict(self, X, threshold=0.5):
        """Predict binary class (0=Low, 1=High)."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X_val, y_val):
        """Comprehensive evaluation on validation set."""
        y_proba = self.predict_proba(X_val)
        y_pred = self.predict(X_val)

        print("\n" + "─" * 40)
        print("SEVERITY MODEL — VALIDATION RESULTS")
        print("─" * 40)

        # Individual model AUCs
        for name, model in self.base_models.items():
            auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            print(f"  {name:10s} AUC: {auc:.4f}")

        # Ensemble AUC
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        print(f"\n  ENSEMBLE AUC:       {auc:.4f}")
        print(f"  ENSEMBLE F1:        {f1:.4f}")
        print(f"  ENSEMBLE Precision: {precision:.4f}")
        print(f"  ENSEMBLE Recall:    {recall:.4f}")
        print(f"\n  Classification Report:\n{classification_report(y_val, y_pred, target_names=['Low', 'High'])}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

        self.metrics = {
            "roc_auc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        return self.metrics

    def get_top_features(self, n=20):
        """Return top N most important features."""
        if self.feature_importances is not None:
            return self.feature_importances.head(n)
        return None

    def save(self, path="outputs/models/severity_model.joblib"):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "base_models": self.base_models,
                "meta_model": self.meta_model,
                "feature_importances": self.feature_importances,
                "metrics": getattr(self, "metrics", {}),
            },
            path,
        )
        print(f"[SeverityModel] Saved to {path}")

    def load(self, path="outputs/models/severity_model.joblib"):
        """Load a trained model."""
        data = joblib.load(path)
        self.base_models = data["base_models"]
        self.meta_model = data["meta_model"]
        self.feature_importances = data["feature_importances"]
        self.metrics = data.get("metrics", {})
        self._fitted = True
        print(f"[SeverityModel] Loaded from {path}")

    def _clone_model(self, name):
        """Create a fresh copy of a base model by name."""
        if name == "lgbm":
            params = self.model_config["lgbm_params"].copy()
            params["random_state"] = self.random_state
            params["verbose"] = -1
            params["objective"] = "binary"
            return lgb.LGBMClassifier(**params)
        elif name == "xgb":
            params = self.model_config["xgb_params"].copy()
            params["random_state"] = self.random_state
            params["eval_metric"] = "auc"
            params["use_label_encoder"] = False
            return xgb.XGBClassifier(**params)
        elif name == "catboost" and HAS_CATBOOST:
            params = self.model_config["catboost_params"].copy()
            params["random_seed"] = self.random_state
            params["verbose"] = 0
            params["loss_function"] = "Logloss"
            return CatBoostClassifier(**params)
        elif name == "rf":
            params = self.model_config["rf_params"].copy()
            params["random_state"] = self.random_state
            params["n_jobs"] = -1
            return RandomForestClassifier(**params)
