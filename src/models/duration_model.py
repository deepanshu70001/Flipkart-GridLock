"""
Duration Prediction Model
==========================
LightGBM-based quantile regression for predicting event resolution time.
Provides uncertainty estimates via P10, P25, P50, P75, P90 quantile predictions.

This is critical for resource planning — knowing not just the expected duration
but the range of possible outcomes.
"""

import numpy as np
import pandas as pd
import yaml
import joblib
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")


class DurationModel:
    """
    Quantile regression ensemble for event resolution duration.
    
    Predicts multiple quantiles (P10, P25, P50, P75, P90) to provide
    uncertainty-aware duration estimates.
    
    Uses log-transformed target to handle the heavy right-tail distribution.
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config["models"]["duration"]
        self.random_state = self.config["models"]["random_state"]
        self.quantiles = self.model_config["quantiles"]
        self.models = {}  # One model per quantile
        self.mean_model = None  # Standard regression for mean
        self._fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit quantile regression models and a mean regression model.
        
        Args:
            X_train: Feature DataFrame
            y_train: Resolution hours (raw, will be log-transformed internally)
        """
        print("=" * 60)
        print("TRAINING DURATION MODEL (Quantile Regression)")
        print("=" * 60)

        # Filter out NaN targets
        valid_mask = y_train.notna() & (y_train > 0)
        X_train_valid = X_train[valid_mask]
        y_train_valid = y_train[valid_mask]
        print(f"Training samples (valid duration): {len(X_train_valid)} / {len(X_train)}")

        # Log-transform target
        y_log = np.log1p(y_train_valid)
        print(f"Log-transformed target — mean: {y_log.mean():.2f}, std: {y_log.std():.2f}")

        # ── Train quantile models ──
        base_params = self.model_config["lgbm_params"].copy()
        base_params["random_state"] = self.random_state
        base_params["verbose"] = -1

        for q in self.quantiles:
            params = base_params.copy()
            params["objective"] = "quantile"
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_valid, y_log)
            self.models[q] = model
            print(f"  Quantile {q:.2f} model trained")

        # ── Train mean regression model ──
        mean_params = base_params.copy()
        mean_params["objective"] = "regression"
        self.mean_model = lgb.LGBMRegressor(**mean_params)
        self.mean_model.fit(X_train_valid, y_log)
        print(f"  Mean regression model trained")

        # Feature importances
        self.feature_importances = pd.Series(
            self.mean_model.feature_importances_,
            index=X_train_valid.columns,
        ).sort_values(ascending=False)

        self._fitted = True

        # ── Evaluate ──
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)

        return self

    def predict(self, X):
        """
        Predict resolution duration with quantile estimates.
        
        Returns:
            dict with keys: 'mean', 'p10', 'p25', 'p50', 'p75', 'p90'
            All values are in original scale (hours).
        """
        assert self._fitted, "Model not fitted!"

        result = {}
        for q in self.quantiles:
            y_log_pred = self.models[q].predict(X)
            result[f"p{int(q*100)}"] = np.expm1(y_log_pred)

        y_log_mean = self.mean_model.predict(X)
        result["mean"] = np.expm1(y_log_mean)

        return result

    def predict_single(self, X):
        """Predict mean duration (scalar output for pipeline integration)."""
        assert self._fitted, "Model not fitted!"
        y_log = self.mean_model.predict(X)
        return np.expm1(y_log)

    def evaluate(self, X_val, y_val):
        """Evaluate on validation set."""
        valid_mask = y_val.notna() & (y_val > 0)
        X_val_valid = X_val[valid_mask]
        y_val_valid = y_val[valid_mask]

        if len(y_val_valid) == 0:
            print("  [DurationModel] No valid validation samples!")
            return {}

        predictions = self.predict(X_val_valid)
        y_pred_mean = predictions["mean"]
        y_pred_median = predictions["p50"]

        print("\n" + "─" * 40)
        print("DURATION MODEL — VALIDATION RESULTS")
        print("─" * 40)

        # Mean prediction metrics
        mae = mean_absolute_error(y_val_valid, y_pred_mean)
        rmse = np.sqrt(mean_squared_error(y_val_valid, y_pred_mean))
        r2 = r2_score(y_val_valid, y_pred_mean)

        # Log-scale metrics (more meaningful for heavy-tailed distributions)
        y_log_true = np.log1p(y_val_valid)
        y_log_pred = np.log1p(y_pred_mean)
        mae_log = mean_absolute_error(y_log_true, y_log_pred)
        rmse_log = np.sqrt(mean_squared_error(y_log_true, y_log_pred))

        print(f"  MAE (hours):         {mae:.2f}")
        print(f"  RMSE (hours):        {rmse:.2f}")
        print(f"  R² Score:            {r2:.4f}")
        print(f"  MAE (log-scale):     {mae_log:.4f}")
        print(f"  RMSE (log-scale):    {rmse_log:.4f}")

        # Median prediction metrics
        mae_median = mean_absolute_error(y_val_valid, y_pred_median)
        print(f"  MAE (median pred):   {mae_median:.2f}")

        # Calibration: % of true values within predicted intervals
        p10 = predictions["p10"]
        p90 = predictions["p90"]
        coverage_80 = ((y_val_valid.values >= p10) & (y_val_valid.values <= p90)).mean()
        print(f"  80% PI Coverage:     {coverage_80:.1%} (target: 80%)")

        p25 = predictions["p25"]
        p75 = predictions["p75"]
        coverage_50 = ((y_val_valid.values >= p25) & (y_val_valid.values <= p75)).mean()
        print(f"  50% PI Coverage:     {coverage_50:.1%} (target: 50%)")

        self.metrics = {
            "mae_hours": mae,
            "rmse_hours": rmse,
            "r2": r2,
            "mae_log": mae_log,
            "coverage_80pct": coverage_80,
            "coverage_50pct": coverage_50,
        }
        return self.metrics

    def get_top_features(self, n=20):
        """Return top N most important features."""
        if self.feature_importances is not None:
            return self.feature_importances.head(n)
        return None

    def save(self, path="outputs/models/duration_model.joblib"):
        """Save all quantile models + mean model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "mean_model": self.mean_model,
                "quantiles": self.quantiles,
                "feature_importances": self.feature_importances,
                "metrics": getattr(self, "metrics", {}),
            },
            path,
        )
        print(f"[DurationModel] Saved to {path}")

    def load(self, path="outputs/models/duration_model.joblib"):
        """Load trained models."""
        data = joblib.load(path)
        self.models = data["models"]
        self.mean_model = data["mean_model"]
        self.quantiles = data["quantiles"]
        self.feature_importances = data["feature_importances"]
        self.metrics = data.get("metrics", {})
        self._fitted = True
        print(f"[DurationModel] Loaded from {path}")
