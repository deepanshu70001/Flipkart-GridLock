import sys
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set stdout to utf-8 to handle special characters printed by duration_model.py
sys.stdout.reconfigure(encoding='utf-8')

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngine
from src.models.duration_model import DurationModel

def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(config_path)
    df = preprocessor.process()

    feature_engine = FeatureEngine(config_path)
    df = feature_engine.fit_transform(df)

    split_date = pd.Timestamp(
        config["models"]["temporal_split_date"],
        tz="Asia/Kolkata"
    )
    train_mask = df["start_datetime"] < split_date
    test_mask = df["start_datetime"] >= split_date

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    duration_features = feature_engine.get_duration_features()
    duration_features = [f for f in duration_features if f in df.columns]

    X_train = df_train[duration_features]
    X_test = df_test[duration_features]
    y_train = df_train["resolution_hours"]
    y_test = df_test["resolution_hours"]

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [500, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [4, 5, 6],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [20, 30, 50],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.0, 0.1, 1.0],
        'reg_lambda': [0.0, 0.1, 1.0]
    }

    # Generate 10 random combinations
    rng = np.random.RandomState(42)
    param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=rng))

    best_score = float('inf')  # We want to minimize MAE + Penalty for bad PI coverage
    best_params = None
    best_metrics = None

    print(f"Starting Random Search for {len(param_list)} iterations...")

    for i, params in enumerate(param_list):
        print(f"\n--- Iteration {i+1}/{len(param_list)} ---")
        print(f"Params: {params}")

        # Temporarily update config
        config["models"]["duration"]["lgbm_params"] = params
        
        # Save temp config
        with open("config_temp.yaml", "w") as f:
            yaml.dump(config, f)

        # Initialize and train model
        try:
            model = DurationModel("config_temp.yaml")
            model.fit(X_train, y_train, X_test, y_test)
            metrics = model.metrics

            mae = metrics["mae_hours"]
            coverage_80 = metrics["coverage_80pct"]
            coverage_50 = metrics["coverage_50pct"]

            # Score function: Minimize MAE, but heavily penalize if 80% coverage is < 0.75
            coverage_penalty = 0
            if coverage_80 < 0.78:
                coverage_penalty += (0.78 - coverage_80) * 10
            if coverage_50 < 0.48:
                coverage_penalty += (0.48 - coverage_50) * 10

            score = mae + coverage_penalty

            print(f"Score: {score:.4f} (MAE: {mae:.4f}, 80% PI: {coverage_80:.2%}, 50% PI: {coverage_50:.2%})")

            if score < best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
        except Exception as e:
            print(f"Error during training: {e}")

    print("\n" + "="*50)
    print("BEST PARAMETERS FOUND:")
    print("="*50)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    print("\nBest Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")

    if os.path.exists("config_temp.yaml"):
        os.remove("config_temp.yaml")

if __name__ == "__main__":
    main()
