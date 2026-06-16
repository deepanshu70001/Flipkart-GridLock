"""
Post-Event Learning System
===========================
Closes the feedback loop by comparing predicted vs actual outcomes.
Identifies systematic biases and auto-updates resource recommendation
multipliers.

This directly addresses the problem statement gap:
"No post-event learning system."
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
from datetime import datetime


class PostEventLearningSystem:
    """
    Compares predicted vs actual event outcomes to:
    1. Track prediction accuracy over time
    2. Identify systematic biases per event cause / corridor
    3. Generate correction factors for resource recommendations
    4. Produce performance reports
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.prediction_log = []
        self.correction_factors = {}

    def log_prediction(self, event_id, event_profile, predictions, actuals=None):
        """
        Log a prediction for future comparison.
        
        Args:
            event_id: Unique event identifier
            event_profile: dict with event details
            predictions: dict with model predictions
            actuals: dict with actual outcomes (filled in post-event)
        """
        entry = {
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "event_cause": event_profile.get("event_cause"),
            "corridor": event_profile.get("corridor"),
            "predicted_severity": predictions.get("severity"),
            "predicted_severity_proba": predictions.get("severity_proba"),
            "predicted_duration_hours": predictions.get("duration_hours", {}).get("p50"),
            "predicted_closure": predictions.get("closure_needed"),
            "predicted_closure_proba": predictions.get("closure_proba"),
            "predicted_manpower": predictions.get("resource_plan", {}).get("manpower", {}).get("total_personnel"),
        }

        if actuals:
            entry.update({
                "actual_severity": actuals.get("severity"),
                "actual_duration_hours": actuals.get("duration_hours"),
                "actual_closure": actuals.get("closure"),
                "actual_manpower_deployed": actuals.get("manpower_deployed"),
            })

        self.prediction_log.append(entry)

    def update_actuals(self, event_id, actuals):
        """Update a logged prediction with actual outcomes."""
        for entry in self.prediction_log:
            if entry["event_id"] == event_id:
                entry["actual_severity"] = actuals.get("severity")
                entry["actual_duration_hours"] = actuals.get("duration_hours")
                entry["actual_closure"] = actuals.get("closure")
                entry["actual_manpower_deployed"] = actuals.get("manpower_deployed")
                entry["updated_at"] = datetime.now().isoformat()
                return True
        return False

    def analyze_from_dataset(self, df, severity_model, duration_model, closure_model, feature_engine):
        """
        Run post-event analysis on historical dataset.
        Compare model predictions against actual outcomes.
        
        Args:
            df: Full preprocessed DataFrame with actuals
            severity_model: Trained severity model
            duration_model: Trained duration model
            closure_model: Trained road closure model
            feature_engine: Fitted feature engine
        
        Returns:
            dict: Analysis results with correction factors
        """
        print("=" * 60)
        print("POST-EVENT LEARNING ANALYSIS")
        print("=" * 60)

        # Use model-specific feature subsets (must match training)
        sev_features = [f for f in feature_engine.get_severity_features() if f in df.columns]
        dur_features = [f for f in feature_engine.get_duration_features() if f in df.columns]
        clos_features = [f for f in feature_engine.get_closure_features() if f in df.columns]

        X_sev = df[sev_features]
        X_dur_all = df[dur_features]
        X_clos = df[clos_features]

        # -- Severity Analysis --
        y_sev_actual = df["priority_binary"]
        y_sev_pred = severity_model.predict(X_sev)
        y_sev_proba = severity_model.predict_proba(X_sev)

        sev_accuracy = (y_sev_actual == y_sev_pred).mean()
        print(f"\nSeverity Prediction Accuracy: {sev_accuracy:.1%}")

        # Per-cause severity bias
        sev_bias = {}
        for cause in df["event_cause"].unique():
            mask = df["event_cause"] == cause
            if mask.sum() < 5:
                continue
            actual_rate = y_sev_actual[mask].mean()
            pred_rate = y_sev_proba[mask].mean()
            bias = pred_rate - actual_rate
            sev_bias[cause] = {
                "actual_high_rate": round(actual_rate, 3),
                "predicted_high_rate": round(pred_rate, 3),
                "bias": round(bias, 3),
                "count": int(mask.sum()),
            }
        print(f"\nSeverity bias by cause:")
        for cause, info in sorted(sev_bias.items(), key=lambda x: abs(x[1]["bias"]), reverse=True)[:5]:
            print(f"  {cause:20s}: actual={info['actual_high_rate']:.1%}, pred={info['predicted_high_rate']:.1%}, bias={info['bias']:+.3f}")

        # -- Duration Analysis --
        valid_dur = df["resolution_hours"].notna() & (df["resolution_hours"] > 0)
        if valid_dur.sum() > 0:
            y_dur_actual = df.loc[valid_dur, "resolution_hours"]
            X_dur = df.loc[valid_dur, dur_features]
            y_dur_pred = duration_model.predict_single(X_dur)

            dur_errors = y_dur_pred - y_dur_actual.values
            print(f"\nDuration Prediction Error (hours):")
            print(f"  Mean Error (bias): {dur_errors.mean():+.2f}h")
            print(f"  Median Error:      {np.median(dur_errors):+.2f}h")
            print(f"  MAE:               {np.abs(dur_errors).mean():.2f}h")

            # Per-cause duration bias
            dur_bias = {}
            for cause in df.loc[valid_dur, "event_cause"].unique():
                mask = df.loc[valid_dur, "event_cause"] == cause
                if mask.sum() < 5:
                    continue
                actual_mean = y_dur_actual[mask].mean()
                pred_mean = y_dur_pred[mask.values].mean()
                ratio = pred_mean / max(actual_mean, 0.1)
                dur_bias[cause] = {
                    "actual_mean_hours": round(actual_mean, 2),
                    "predicted_mean_hours": round(pred_mean, 2),
                    "ratio": round(ratio, 3),
                    "count": int(mask.sum()),
                }
            print(f"\nDuration bias by cause (ratio > 1 = over-prediction):")
            for cause, info in sorted(dur_bias.items(), key=lambda x: abs(x[1]["ratio"] - 1), reverse=True)[:5]:
                direction = "OVER" if info["ratio"] > 1 else "UNDER"
                print(f"  {cause:20s}: actual={info['actual_mean_hours']:.1f}h, pred={info['predicted_mean_hours']:.1f}h, {direction} by {abs(info['ratio']-1):.0%}")

        # -- Closure Analysis --
        y_clos_actual = df["road_closure_binary"]
        y_clos_pred = closure_model.predict(X_clos)

        clos_accuracy = (y_clos_actual == y_clos_pred).mean()
        print(f"\nClosure Prediction Accuracy: {clos_accuracy:.1%}")

        # ── Compute Correction Factors ──
        correction_factors = self._compute_correction_factors(df, sev_bias, dur_bias if valid_dur.sum() > 0 else {})

        # ── Save Report ──
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_events": len(df),
            "severity_accuracy": round(sev_accuracy, 4),
            "severity_bias_by_cause": sev_bias,
            "closure_accuracy": round(clos_accuracy, 4),
            "correction_factors": correction_factors,
        }
        if valid_dur.sum() > 0:
            report["duration_mae_hours"] = round(np.abs(dur_errors).mean(), 2)
            report["duration_bias_hours"] = round(dur_errors.mean(), 2)
            report["duration_bias_by_cause"] = dur_bias

        report_path = os.path.join(
            self.config["outputs"]["reports_dir"],
            f"post_event_report.json",
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[PostEventLearning] Report saved to {report_path}")

        self.correction_factors = correction_factors
        return report

    def _compute_correction_factors(self, df, sev_bias, dur_bias):
        """
        Compute resource recommendation correction factors.
        
        If we systematically under-predict severity for a cause,
        increase the resource multiplier.
        If we systematically over-predict duration for a cause,
        decrease the duration-based multiplier.
        """
        factors = {}

        for cause, info in sev_bias.items():
            bias = info["bias"]
            if bias < -0.1:
                # Under-predicting severity → need more resources
                factors[cause] = round(1 + abs(bias) * 0.5, 2)
            elif bias > 0.1:
                # Over-predicting severity → can reduce
                factors[cause] = round(max(0.8, 1 - bias * 0.3), 2)
            else:
                factors[cause] = 1.0

        # Adjust for duration bias
        for cause, info in dur_bias.items():
            ratio = info["ratio"]
            if cause in factors:
                # If we under-predict duration, increase factor
                if ratio < 0.7:
                    factors[cause] *= 1.2
                elif ratio > 1.3:
                    factors[cause] *= 0.9

        print(f"\nCorrection factors computed for {len(factors)} event causes")
        for cause, factor in sorted(factors.items(), key=lambda x: abs(x[1] - 1), reverse=True)[:5]:
            direction = "↑ increase" if factor > 1 else "↓ decrease" if factor < 1 else "= no change"
            print(f"  {cause:20s}: {factor:.2f} ({direction})")

        return factors

    def get_correction_factors(self):
        """Return current correction factors for resource recommender."""
        return self.correction_factors

    def get_worst_predictions(self, n=10):
        """Return the N worst predictions for review."""
        log_df = pd.DataFrame(self.prediction_log)
        if log_df.empty or "actual_duration_hours" not in log_df.columns:
            return pd.DataFrame()

        log_df["duration_error"] = abs(
            log_df["predicted_duration_hours"] - log_df["actual_duration_hours"]
        )
        return log_df.nlargest(n, "duration_error")
