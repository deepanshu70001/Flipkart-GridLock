"""
Pipeline Orchestrator
=====================
End-to-end ML pipeline that:
1. Preprocesses data
2. Engineers features
3. Trains all models (severity, duration, closure, clustering)
4. Evaluates with temporal train/test split
5. Runs post-event learning analysis
6. Generates resource recommendations for sample events
7. Saves all artifacts

Usage:
    python -m src.pipeline              # Full training pipeline
    python -m src.pipeline --predict    # Predict on sample events
"""

import sys
import os
import json
import yaml
import pandas as pd
import warnings
import argparse

# Set stdout to utf-8 to handle special characters printed by models
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngine
from src.models.impact_severity_model import SeverityModel
from src.models.duration_model import DurationModel
from src.models.road_closure_model import RoadClosureModel
from src.models.spatial_temporal_clustering import SpatialTemporalClustering
from src.resource_recommender import ResourceRecommender
from src.post_event_learning import PostEventLearningSystem


class CongestionPipeline:
    """
    End-to-end pipeline for event-driven congestion prediction
    and resource recommendation.
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engine = FeatureEngine(config_path)
        self.severity_model = SeverityModel(config_path)
        self.duration_model = DurationModel(config_path)
        self.closure_model = RoadClosureModel(config_path)
        self.clustering = SpatialTemporalClustering(config_path)
        self.recommender = ResourceRecommender(config_path)
        self.learning_system = PostEventLearningSystem(config_path)
        
        # Create output dirs
        for dir_key in ["models_dir", "plots_dir", "reports_dir"]:
            os.makedirs(self.config["outputs"][dir_key], exist_ok=True)

    def run_training(self):
        """
        Execute the full training pipeline.
        
        Returns:
            dict: All model metrics and summaries
        """
        print("+" + "=" * 58 + "+")
        print("|  EVENT-DRIVEN CONGESTION ML PIPELINE -- TRAINING          |")
        print("+" + "=" * 58 + "+")

        # ── Step 1: Data Preprocessing ──
        print("\n" + "#" * 60)
        print("#  STEP 1: DATA PREPROCESSING")
        print("#" * 60)
        df = self.preprocessor.process()

        # ── Step 2: Feature Engineering ──
        print("\n" + "#" * 60)
        print("#  STEP 2: FEATURE ENGINEERING")
        print("#" * 60)
        df = self.feature_engine.fit_transform(df)

        # ── Step 3: Temporal Train/Test Split ──
        print("\n" + "#" * 60)
        print("#  STEP 3: TEMPORAL TRAIN/TEST SPLIT")
        print("#" * 60)
        split_date = pd.Timestamp(
            self.config["models"]["temporal_split_date"],
            tz="Asia/Kolkata"
        )
        train_mask = df["start_datetime"] < split_date
        test_mask = df["start_datetime"] >= split_date

        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        print(f"Training set: {len(df_train)} events (before {split_date.date()})")
        print(f"Test set:     {len(df_test)} events (from {split_date.date()})")

        # Get feature lists
        severity_features = self.feature_engine.get_severity_features()
        duration_features = self.feature_engine.get_duration_features()
        closure_features = self.feature_engine.get_closure_features()

        # Filter to available features
        severity_features = [f for f in severity_features if f in df.columns]
        duration_features = [f for f in duration_features if f in df.columns]
        closure_features = [f for f in closure_features if f in df.columns]

        X_train_sev = df_train[severity_features]
        X_test_sev = df_test[severity_features]
        y_train_sev = df_train["priority_binary"]
        y_test_sev = df_test["priority_binary"]

        X_train_dur = df_train[duration_features]
        X_test_dur = df_test[duration_features]
        y_train_dur = df_train["resolution_hours"]
        y_test_dur = df_test["resolution_hours"]

        X_train_clos = df_train[closure_features]
        X_test_clos = df_test[closure_features]
        y_train_clos = df_train["road_closure_binary"]
        y_test_clos = df_test["road_closure_binary"]

        # ── Step 4: Train Severity Model ──
        print("\n" + "#" * 60)
        print("#  STEP 4: SEVERITY MODEL")
        print("#" * 60)
        self.severity_model.fit(X_train_sev, y_train_sev, X_test_sev, y_test_sev)
        self.severity_model.save()

        print("\nTop 15 Features (Severity):")
        top_feats = self.severity_model.get_top_features(15)
        if top_feats is not None:
            for feat, imp in top_feats.items():
                print(f"  {feat:45s}: {imp}")

        # ── Step 5: Train Duration Model ──
        print("\n" + "#" * 60)
        print("#  STEP 5: DURATION MODEL")
        print("#" * 60)
        self.duration_model.fit(X_train_dur, y_train_dur, X_test_dur, y_test_dur)
        self.duration_model.save()

        print("\nTop 15 Features (Duration):")
        top_feats = self.duration_model.get_top_features(15)
        if top_feats is not None:
            for feat, imp in top_feats.items():
                print(f"  {feat:45s}: {imp}")

        # ── Step 6: Train Closure Model ──
        print("\n" + "#" * 60)
        print("#  STEP 6: ROAD CLOSURE MODEL")
        print("#" * 60)
        self.closure_model.fit(X_train_clos, y_train_clos, X_test_clos, y_test_clos)
        self.closure_model.save()

        print("\nTop 15 Features (Closure):")
        top_feats = self.closure_model.get_top_features(15)
        if top_feats is not None:
            for feat, imp in top_feats.items():
                print(f"  {feat:45s}: {imp}")

        # ── Step 7: Spatial-Temporal Clustering ──
        print("\n" + "#" * 60)
        print("#  STEP 7: SPATIAL-TEMPORAL CLUSTERING")
        print("#" * 60)
        df_clustered, cluster_labels = self.clustering.fit(df)
        self.clustering.save()

        print("\nTop 10 Hotspots:")
        hotspots = self.clustering.get_hotspots(10)
        if hotspots is not None:
            for _, row in hotspots.iterrows():
                print(f"  Cluster {row['cluster_id']:3d}: {row['size']:4d} events | "
                      f"({row['center_lat']:.4f}, {row['center_lng']:.4f}) | "
                      f"Cause: {row['dominant_cause']:20s} | "
                      f"High-priority: {row['high_priority_rate']:.0%}")

        print("\nHigh-Risk Clusters:")
        high_risk = self.clustering.get_high_risk_clusters(0.7)
        if high_risk is not None and len(high_risk) > 0:
            for _, row in high_risk.iterrows():
                print(f"  Cluster {row['cluster_id']:3d}: {row['high_priority_rate']:.0%} high-priority | "
                      f"{row['closure_rate']:.0%} closure rate | {row['size']} events")

        # ── Step 8: Post-Event Learning ──
        print("\n" + "#" * 60)
        print("#  STEP 8: POST-EVENT LEARNING")
        print("#" * 60)
        self.learning_system.analyze_from_dataset(
            df_test, self.severity_model, self.duration_model, self.closure_model,
            self.feature_engine
        )

        # Update resource recommender with correction factors
        correction_factors = self.learning_system.get_correction_factors()
        self.recommender.update_correction_factors(correction_factors)

        # ── Step 9: Sample Predictions ──
        print("\n" + "#" * 60)
        print("#  STEP 9: SAMPLE PREDICTIONS & RECOMMENDATIONS")
        print("#" * 60)
        self._demo_predictions(df_test, severity_features, duration_features, closure_features)

        # ── Final Summary ──
        print("\n" + "+" + "=" * 58 + "+")
        print("|  PIPELINE COMPLETE -- FINAL METRICS SUMMARY               |")
        print("+" + "=" * 58 + "+")

        metrics = {
            "severity": getattr(self.severity_model, "metrics", {}),
            "duration": getattr(self.duration_model, "metrics", {}),
            "closure": getattr(self.closure_model, "metrics", {}),
            "training_samples": len(df_train),
            "test_samples": len(df_test),
            "total_features": len(severity_features),
        }

        print("\n  Severity Model:")
        for k, v in metrics.get("severity", {}).items():
            print(f"    {k:20s}: {v:.4f}")

        print("\n  Duration Model:")
        for k, v in metrics.get("duration", {}).items():
            print(f"    {k:20s}: {v:.4f}" if isinstance(v, float) else f"    {k:20s}: {v}")

        print("\n  Closure Model:")
        for k, v in metrics.get("closure", {}).items():
            print(f"    {k:20s}: {v:.4f}" if isinstance(v, float) else f"    {k:20s}: {v}")

        # Save final metrics
        metrics_path = os.path.join(self.config["outputs"]["reports_dir"], "final_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n  Metrics saved to {metrics_path}")

        return metrics

    def _demo_predictions(self, df_test, severity_features, duration_features, closure_features):
        """Run sample predictions to demonstrate the system."""
        # Select diverse sample events
        sample_events = []
        for cause in ["public_event", "procession", "construction", "accident", "vehicle_breakdown"]:
            subset = df_test[df_test["event_cause"] == cause]
            if len(subset) > 0:
                sample_events.append(subset.iloc[0])

        if not sample_events:
            sample_events = [df_test.iloc[i] for i in range(min(3, len(df_test)))]

        for i, event in enumerate(sample_events):
            print(f"\n{'-' * 50}")
            print(f"SAMPLE EVENT {i+1}: {event.get('event_cause', 'unknown')} @ {event.get('corridor', 'unknown')}")
            print(f"{'-' * 50}")

            # Get predictions
            X_sev = pd.DataFrame([event[severity_features]])
            X_dur = pd.DataFrame([event[duration_features]])
            X_clos = pd.DataFrame([event[closure_features]])

            severity_proba = float(self.severity_model.predict_proba(X_sev)[0])
            severity = "High" if severity_proba > 0.5 else "Low"
            duration_pred = self.duration_model.predict(X_dur)
            duration_dict = {k: float(v[0]) for k, v in duration_pred.items()}
            closure_proba = float(self.closure_model.predict_proba(X_clos)[0])
            closure_needed = closure_proba > self.closure_model.optimal_threshold

            predictions = {
                "severity": severity,
                "severity_proba": severity_proba,
                "duration_hours": duration_dict,
                "closure_needed": closure_needed,
                "closure_proba": closure_proba,
            }

            event_profile = {
                "event_cause": event.get("event_cause", "others"),
                "corridor": event.get("corridor", "unknown"),
                "is_rush_hour": bool(event.get("is_rush_hour", False)),
                "is_weekend": bool(event.get("is_weekend", False)),
                "is_planned": bool(event.get("is_planned", False)),
                "latitude": float(event.get("latitude", 12.97)),
                "longitude": float(event.get("longitude", 77.59)),
                "junction": event.get("junction", "unknown"),
            }

            # Get resource recommendation
            plan = self.recommender.recommend(event_profile, predictions)

            # Print results
            print(f"  Predicted Severity:    {severity} (prob: {severity_proba:.2%})")
            print(f"  Predicted Duration:    {duration_dict.get('mean', 0):.1f}h (P10={duration_dict.get('p10', 0):.1f}, P90={duration_dict.get('p90', 0):.1f})")
            print(f"  Road Closure:          {'YES' if closure_needed else 'NO'} (prob: {closure_proba:.2%})")
            print(f"  Alert Level:           {plan['alert_level']['level']} ({plan['alert_level']['color']})")
            print(f"  Manpower:              {plan['manpower']['total_personnel']} personnel ({plan['manpower']['shifts_needed']} shifts)")
            print(f"  Barricading:           {plan['barricading']['type']} ({plan['barricading']['barricade_count']} barricades)")
            print(f"  Estimated Cost:        {plan['estimated_cost']['total_formatted']}")
            print(f"  Diversion:             {plan['diversion']['primary']}")

            # Compare with actual
            actual_priority = event.get("priority", "unknown")
            actual_closure = bool(event.get("requires_road_closure", False))
            actual_duration = event.get("resolution_hours")
            print(f"\n  ACTUAL Severity:       {actual_priority}")
            print(f"  ACTUAL Closure:        {'YES' if actual_closure else 'NO'}")
            if pd.notna(actual_duration):
                print(f"  ACTUAL Duration:       {actual_duration:.1f}h")

    def predict_event(self, event_info):
        """
        Predict for a new event (inference mode).
        
        Args:
            event_info: dict with event details
        
        Returns:
            dict: Complete predictions + resource plan
        """
        # Convert to DataFrame
        event_df = pd.DataFrame([event_info])
        
        # Preprocess
        event_df = self.feature_engine.transform(event_df)
        
        # Get features
        severity_features = [f for f in self.feature_engine.get_severity_features() if f in event_df.columns]
        duration_features = [f for f in self.feature_engine.get_duration_features() if f in event_df.columns]
        closure_features = [f for f in self.feature_engine.get_closure_features() if f in event_df.columns]
        
        # Predictions
        severity_proba = float(self.severity_model.predict_proba(event_df[severity_features])[0])
        duration_pred = self.duration_model.predict(event_df[duration_features])
        closure_proba = float(self.closure_model.predict_proba(event_df[closure_features])[0])
        
        predictions = {
            "severity": "High" if severity_proba > 0.5 else "Low",
            "severity_proba": severity_proba,
            "duration_hours": {k: float(v[0]) for k, v in duration_pred.items()},
            "closure_needed": closure_proba > self.closure_model.optimal_threshold,
            "closure_proba": closure_proba,
        }
        
        # Resource recommendation
        event_profile = {
            "event_cause": event_info.get("event_cause", "others"),
            "corridor": event_info.get("corridor", "unknown"),
            "is_rush_hour": bool(event_info.get("is_rush_hour", False)),
            "is_weekend": bool(event_info.get("is_weekend", False)),
            "is_planned": bool(event_info.get("is_planned", False)),
            "latitude": float(event_info.get("latitude", 12.97)),
            "longitude": float(event_info.get("longitude", 77.59)),
        }
        
        predictions["resource_plan"] = self.recommender.recommend(event_profile, predictions)
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description="Event-Driven Congestion ML Pipeline")
    parser.add_argument("--predict", action="store_true", help="Run prediction demo instead of training")
    args = parser.parse_args()

    pipeline = CongestionPipeline()

    if args.predict:
        print("Running prediction demo...")
        # TODO: Load saved models and run inference
    else:
        pipeline.run_training()
        print("\n[OK] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
