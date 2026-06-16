"""
Spatial-Temporal Clustering Model
=================================
HDBSCAN-based clustering to identify congestion hotspots and
temporal event patterns. Cluster membership is fed back as 
features to supervised models.
"""

import numpy as np
import pandas as pd
import yaml
import joblib
import os
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    from sklearn.cluster import DBSCAN


class SpatialTemporalClustering:
    """
    Identifies spatial-temporal event clusters using HDBSCAN.
    
    Clustering dimensions:
    - Latitude, Longitude (spatial)
    - Hour of day (temporal, cyclical)
    - Event cause severity (event profile)
    
    Outputs:
    - Cluster labels for each event
    - Cluster statistics (hotspot analysis)
    - Noise point identification
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.cluster_config = self.config["models"]["clustering"]
        self.scaler = StandardScaler()
        self.model = None
        self.cluster_stats = None
        self._fitted = False

    def fit(self, df):
        """
        Fit clustering model on event data.
        
        Args:
            df: DataFrame with columns: latitude, longitude, hour, cause_severity_rank
        """
        print("=" * 60)
        print("TRAINING SPATIAL-TEMPORAL CLUSTERING")
        print("=" * 60)

        # Prepare clustering features
        cluster_features = self._prepare_features(df)
        print(f"Clustering on {len(cluster_features)} events with {cluster_features.shape[1]} dimensions")

        # Scale features
        X_scaled = self.scaler.fit_transform(cluster_features)

        # Fit HDBSCAN or DBSCAN
        if HAS_HDBSCAN:
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.cluster_config["min_cluster_size"],
                min_samples=self.cluster_config["min_samples"],
                metric=self.cluster_config["metric"],
                cluster_selection_method="eom",
            )
        else:
            self.model = DBSCAN(
                eps=0.5,
                min_samples=self.cluster_config["min_samples"],
                metric=self.cluster_config["metric"],
            )

        labels = self.model.fit_predict(X_scaled)
        df = df.copy()
        df["cluster_label"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"  Found {n_clusters} clusters")
        print(f"  Noise points: {n_noise} ({n_noise / len(df):.1%})")

        # Compute cluster statistics
        self.cluster_stats = self._compute_cluster_stats(df)
        self._fitted = True

        return df, labels

    def predict(self, df):
        """
        Assign cluster labels to new events using approximate nearest cluster.
        Since HDBSCAN doesn't natively predict, we use centroid distance.
        """
        assert self._fitted, "Must fit first!"

        cluster_features = self._prepare_features(df)
        X_scaled = self.scaler.transform(cluster_features)

        if HAS_HDBSCAN and hasattr(self.model, "approximate_predict"):
            labels, _ = hdbscan.approximate_predict(self.model, X_scaled)
        else:
            # Fallback: assign to nearest cluster centroid
            labels = self._assign_nearest_centroid(X_scaled)

        return labels

    def _prepare_features(self, df):
        """Prepare features for clustering."""
        features = pd.DataFrame()
        features["latitude"] = df["latitude"]
        features["longitude"] = df["longitude"]

        # Cyclical hour encoding
        if "hour" in df.columns:
            features["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            features["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        elif "hour_sin" in df.columns:
            features["hour_sin"] = df["hour_sin"]
            features["hour_cos"] = df["hour_cos"]

        if "cause_severity_rank" in df.columns:
            features["cause_severity_rank"] = df["cause_severity_rank"]

        # Weight spatial dimensions more heavily (3x)
        features["latitude"] = features["latitude"] * 3
        features["longitude"] = features["longitude"] * 3

        return features.fillna(0)

    def _compute_cluster_stats(self, df):
        """Compute statistics for each cluster."""
        stats = []
        for cluster_id in sorted(df["cluster_label"].unique()):
            if cluster_id == -1:
                continue

            cluster_df = df[df["cluster_label"] == cluster_id]
            stat = {
                "cluster_id": cluster_id,
                "size": len(cluster_df),
                "center_lat": cluster_df["latitude"].mean(),
                "center_lng": cluster_df["longitude"].mean(),
                "lat_spread": cluster_df["latitude"].std(),
                "lng_spread": cluster_df["longitude"].std(),
                "dominant_cause": cluster_df["event_cause"].mode().iloc[0] if "event_cause" in cluster_df.columns else "unknown",
                "avg_hour": cluster_df["hour"].mean() if "hour" in cluster_df.columns else 0,
                "high_priority_rate": cluster_df["priority_binary"].mean() if "priority_binary" in cluster_df.columns else 0,
                "closure_rate": cluster_df["road_closure_binary"].mean() if "road_closure_binary" in cluster_df.columns else 0,
            }

            # Top corridors in cluster
            if "corridor" in cluster_df.columns:
                stat["top_corridor"] = cluster_df["corridor"].mode().iloc[0]

            stats.append(stat)

        return pd.DataFrame(stats)

    def _assign_nearest_centroid(self, X_scaled):
        """Assign to nearest cluster centroid (fallback for prediction)."""
        if self.cluster_stats is None or len(self.cluster_stats) == 0:
            return np.full(len(X_scaled), -1)

        # Build centroids from cluster stats
        centroids = self.scaler.transform(
            self._prepare_features(
                pd.DataFrame({
                    "latitude": self.cluster_stats["center_lat"],
                    "longitude": self.cluster_stats["center_lng"],
                    "hour_sin": np.sin(2 * np.pi * self.cluster_stats["avg_hour"] / 24),
                    "hour_cos": np.cos(2 * np.pi * self.cluster_stats["avg_hour"] / 24),
                    "cause_severity_rank": 5,  # default mid severity
                })
            )
        )

        # Assign to nearest centroid
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X_scaled, centroids)
        labels = np.argmin(distances, axis=1)
        # Map back to cluster IDs
        cluster_ids = self.cluster_stats["cluster_id"].values
        labels = cluster_ids[labels]

        return labels

    def get_hotspots(self, top_n=10):
        """Return top N hotspot clusters by event count."""
        if self.cluster_stats is None:
            return None
        return self.cluster_stats.nlargest(top_n, "size")

    def get_high_risk_clusters(self, priority_threshold=0.7):
        """Return clusters with high-priority event rates above threshold."""
        if self.cluster_stats is None:
            return None
        return self.cluster_stats[
            self.cluster_stats["high_priority_rate"] >= priority_threshold
        ].sort_values("high_priority_rate", ascending=False)

    def save(self, path="outputs/models/clustering_model.joblib"):
        """Save clustering model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "cluster_stats": self.cluster_stats,
            },
            path,
        )
        print(f"[Clustering] Saved to {path}")

    def load(self, path="outputs/models/clustering_model.joblib"):
        """Load clustering model."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.cluster_stats = data["cluster_stats"]
        self._fitted = True
        print(f"[Clustering] Loaded from {path}")
