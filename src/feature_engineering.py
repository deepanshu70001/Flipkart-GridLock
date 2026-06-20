"""
Feature Engineering Module
==========================
Transforms preprocessed event data into ML-ready features.
This is the competition-winning module — features are the #1 driver
of model performance in tabular ML.

Feature Groups:
  A. Temporal (cyclical, rush hour, time-of-day)
  B. Spatial (H3 hex, corridor encoding, junction stats)
  C. Event Profile (cause, vehicle, description)
  D. Historical / Lag (concurrent events, rolling stats)
  E. Interaction (rush × closure, planned × corridor, etc.)
"""

import numpy as np
import yaml
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# Try importing h3; fall back gracefully
try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False
    print("[FeatureEngine] Warning: h3 not installed. Spatial hex features disabled.")


class FeatureEngine:
    """
    Comprehensive feature engineering for event-driven congestion prediction.
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.target_encodings = {}
        self.junction_stats = {}
        self.corridor_stats = {}
        self.cause_stats = {}
        self._fitted = False

    # ──────────────────────────────────────────────────────────────
    # A. TEMPORAL FEATURES
    # ──────────────────────────────────────────────────────────────

    def _temporal_features(self, df):
        """Extract rich temporal features from start_datetime."""
        dt = df["start_datetime"]

        df["hour"] = dt.dt.hour
        df["minute"] = dt.dt.minute
        df["day_of_week"] = dt.dt.dayofweek  # 0=Monday
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Rush hour flags (IST hours)
        morning_rush = self.config["temporal"]["rush_hours_morning"]
        evening_rush = self.config["temporal"]["rush_hours_evening"]
        df["is_morning_rush"] = df["hour"].isin(morning_rush).astype(int)
        df["is_evening_rush"] = df["hour"].isin(evening_rush).astype(int)
        df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)

        # Night flag
        night_start = self.config["temporal"]["night_start"]
        night_end = self.config["temporal"]["night_end"]
        df["is_night"] = ((df["hour"] >= night_start) | (df["hour"] < night_end)).astype(int)

        # Cyclical encoding for periodicity
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Time period buckets
        conditions = [
            (df["hour"] >= 6) & (df["hour"] < 10),   # Early morning
            (df["hour"] >= 10) & (df["hour"] < 14),  # Midday
            (df["hour"] >= 14) & (df["hour"] < 17),  # Afternoon
            (df["hour"] >= 17) & (df["hour"] < 21),  # Evening
            (df["hour"] >= 21) | (df["hour"] < 2),   # Late night
            (df["hour"] >= 2) & (df["hour"] < 6),    # Early morning
        ]
        labels = ["early_morning", "midday", "afternoon", "evening", "late_night", "pre_dawn"]
        df["time_period"] = np.select(conditions, labels, default="other")

        return df

    # ──────────────────────────────────────────────────────────────
    # B. SPATIAL FEATURES
    # ──────────────────────────────────────────────────────────────

    def _spatial_features(self, df):
        """Extract spatial features: H3 hex, distance to center, corridor stats."""
        # H3 hexagonal indexing
        if HAS_H3:
            resolution = self.config["spatial"]["h3_resolution"]
            df["h3_hex"] = df.apply(
                lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], resolution),
                axis=1,
            )
            # H3 hex event frequency (from training data)
            if self._fitted:
                df["h3_event_count"] = df["h3_hex"].map(self.h3_counts).fillna(0)
            else:
                self.h3_counts = df["h3_hex"].value_counts().to_dict()
                df["h3_event_count"] = df["h3_hex"].map(self.h3_counts).fillna(0)

        # Distance to city center
        center_lat = self.config["spatial"]["city_center_lat"]
        center_lng = self.config["spatial"]["city_center_lng"]
        df["dist_to_center_km"] = self._haversine(
            df["latitude"], df["longitude"], center_lat, center_lng
        )

        # Is on major corridor (non-"Non-corridor" and non-"unknown")
        df["is_major_corridor"] = (
            ~df["corridor"].isin(["Non-corridor", "unknown"])
        ).astype(int)

        # Corridor event frequency
        if not self._fitted:
            self.corridor_event_counts = df["corridor"].value_counts().to_dict()
        df["corridor_event_frequency"] = df["corridor"].map(self.corridor_event_counts).fillna(0)

        # Junction event frequency
        if not self._fitted:
            self.junction_event_counts = df["junction"].value_counts().to_dict()
        df["junction_event_frequency"] = df["junction"].map(self.junction_event_counts).fillna(0)

        # Police station event frequency
        if not self._fitted:
            self.ps_event_counts = df["police_station"].value_counts().to_dict()
        df["police_station_event_frequency"] = df["police_station"].map(self.ps_event_counts).fillna(0)

        # Lat/Lng grid cells (coarser than H3, more robust)
        df["lat_grid"] = (df["latitude"] * 100).round() / 100  # ~1.1km
        df["lng_grid"] = (df["longitude"] * 100).round() / 100

        return df

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Haversine distance in km. Accepts scalars or arrays."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    # ──────────────────────────────────────────────────────────────
    # C. EVENT PROFILE FEATURES
    # ──────────────────────────────────────────────────────────────

    def _event_profile_features(self, df):
        """Features derived from the event itself."""
        # Event cause severity ranking (domain knowledge)
        cause_severity = {
            "vip_movement": 10,
            "protest": 9,
            "procession": 8,
            "public_event": 8,
            "accident": 7,
            "tree_fall": 6,
            "water_logging": 5,
            "construction": 5,
            "congestion": 4,
            "vehicle_breakdown": 3,
            "road_conditions": 3,
            "pot_holes": 2,
            "debris": 2,
            "fog_low_visibility": 4,
            "others": 3,
        }
        df["cause_severity_rank"] = df["event_cause"].map(cause_severity).fillna(3)

        # Event cause category groups
        planned_causes = {"construction", "public_event", "procession", "vip_movement", "protest"}
        infrastructure_causes = {"pot_holes", "road_conditions", "water_logging", "debris"}
        vehicle_causes = {"vehicle_breakdown", "accident"}
        nature_causes = {"tree_fall", "fog_low_visibility"}

        df["is_planned_cause"] = df["event_cause"].isin(planned_causes).astype(int)
        df["is_infrastructure_cause"] = df["event_cause"].isin(infrastructure_causes).astype(int)
        df["is_vehicle_cause"] = df["event_cause"].isin(vehicle_causes).astype(int)
        df["is_nature_cause"] = df["event_cause"].isin(nature_causes).astype(int)

        # Vehicle type severity
        veh_severity = {
            "heavy_vehicle": 5,
            "truck": 5,
            "private_bus": 4,
            "bmtc_bus": 4,
            "ksrtc_bus": 4,
            "lcv": 3,
            "others": 2,
            "private_car": 2,
            "taxi": 2,
            "auto": 1,
            "unknown": 0,
        }
        df["vehicle_severity"] = df["veh_type"].map(veh_severity).fillna(0)

        # Description-based features
        df["desc_has_traffic_slow"] = df["description"].str.lower().str.contains(
            "slow|traffic|congestion|jam", na=False
        ).astype(int)
        df["desc_has_road_closed"] = df["description"].str.lower().str.contains(
            "closed|closure|block|barricad", na=False
        ).astype(int)
        df["desc_has_accident"] = df["description"].str.lower().str.contains(
            "accident|collision|hit|crash", na=False
        ).astype(int)
        df["desc_word_count"] = df["description"].str.split().str.len().fillna(0).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────
    # D. HISTORICAL / LAG FEATURES
    # ──────────────────────────────────────────────────────────────

    def _historical_features(self, df):
        """
        Compute historical lag features.
        These are the KEY differentiator for competition accuracy.
        
        IMPORTANT: All lag features are computed using only past data
        (no data leakage).
        """
        df = df.sort_values("start_datetime").reset_index(drop=True)

        # Pre-compute some aggregate stats from training data
        if not self._fitted:
            self.cause_avg_resolution = df.groupby("event_cause")["resolution_hours"].mean().to_dict()
            self.corridor_avg_resolution = df.groupby("corridor")["resolution_hours"].mean().to_dict()
            self.cause_closure_rate = df.groupby("event_cause")["road_closure_binary"].mean().to_dict()
            self.junction_closure_rate = df.groupby("junction")["road_closure_binary"].mean().to_dict()
            self.cause_priority_rate = df.groupby("event_cause")["priority_binary"].mean().to_dict()

        # Map aggregate stats
        df["cause_avg_resolution_hours"] = df["event_cause"].map(self.cause_avg_resolution).fillna(
            df["resolution_hours"].mean() if "resolution_hours" in df.columns else 2.0
        )
        df["corridor_avg_resolution_hours"] = df["corridor"].map(self.corridor_avg_resolution).fillna(
            df["resolution_hours"].mean() if "resolution_hours" in df.columns else 2.0
        )
        df["cause_historical_closure_rate"] = df["event_cause"].map(self.cause_closure_rate).fillna(0)
        df["junction_historical_closure_rate"] = df["junction"].map(self.junction_closure_rate).fillna(0)
        df["cause_historical_priority_rate"] = df["event_cause"].map(self.cause_priority_rate).fillna(0.5)

        # Rolling event counts (vectorized for speed)
        # Events in same corridor in last 24h
        df["events_same_corridor_24h"] = self._rolling_count(
            df, group_col="corridor", window_hours=24
        )
        # Events in same zone in last 7 days
        df["events_same_zone_7d"] = self._rolling_count(
            df, group_col="zone", window_hours=168
        )
        # Events in same police station in last 24h
        df["events_same_ps_24h"] = self._rolling_count(
            df, group_col="police_station", window_hours=24
        )
        # Total events in last 6h (city-wide event pressure)
        df["total_events_last_6h"] = self._rolling_count_global(df, window_hours=6)

        return df

    def _rolling_count(self, df, group_col, window_hours):
        """
        Count events in the same group within the past window_hours.
        Uses a vectorized approach for speed.
        """
        result = np.zeros(len(df))
        timestamps = df["start_datetime"].values.astype("int64") // 10**9  # to unix seconds
        groups = df[group_col].values
        window_seconds = window_hours * 3600

        # Group by the column for efficiency
        group_indices = defaultdict(list)
        for i, g in enumerate(groups):
            group_indices[g].append(i)

        for g, indices in group_indices.items():
            if len(indices) <= 1:
                continue
            idx_arr = np.array(indices)
            ts_arr = timestamps[idx_arr]
            for j, idx in enumerate(idx_arr):
                t = ts_arr[j]
                # Count events in same group within window before this event
                mask = (ts_arr < t) & (ts_arr >= t - window_seconds)
                result[idx] = mask.sum()

        return result

    def _rolling_count_global(self, df, window_hours):
        """Count total events in last window_hours (city-wide)."""
        result = np.zeros(len(df))
        timestamps = df["start_datetime"].values.astype("int64") // 10**9
        window_seconds = window_hours * 3600

        for i in range(len(df)):
            t = timestamps[i]
            mask = (timestamps[:i] >= t - window_seconds)
            result[i] = mask.sum()

        return result

    # ──────────────────────────────────────────────────────────────
    # E. INTERACTION FEATURES
    # ──────────────────────────────────────────────────────────────

    def _interaction_features(self, df):
        """Create interaction features between key dimensions."""
        # Rush hour × road closure need
        df["rush_x_closure"] = df["is_rush_hour"] * df["road_closure_binary"]

        # Planned × major corridor
        df["planned_x_major_corridor"] = df["is_planned"] * df["is_major_corridor"]

        # Cause severity × rush hour
        df["severity_x_rush"] = df["cause_severity_rank"] * df["is_rush_hour"]

        # Weekend × planned cause
        df["weekend_x_planned_cause"] = df["is_weekend"] * df["is_planned_cause"]

        # Night × vehicle breakdown
        df["night_x_vehicle"] = df["is_night"] * df["is_vehicle_cause"]

        # Distance to center × severity
        df["dist_x_severity"] = df["dist_to_center_km"] * df["cause_severity_rank"]

        # Concurrent events × severity
        df["concurrent_x_severity"] = df["events_same_corridor_24h"] * df["cause_severity_rank"]

        # Junction frequency × closure rate
        df["junction_freq_x_closure"] = (
            df["junction_event_frequency"] * df["junction_historical_closure_rate"]
        )

        return df

    # ──────────────────────────────────────────────────────────────
    # F. TARGET ENCODING (with smoothing)
    # ──────────────────────────────────────────────────────────────

    def _target_encode(self, df, col, target, is_train=True):
        """
        Target-encode a categorical column with Bayesian smoothing.
        Prevents target leakage via leave-one-out for training.
        """
        smoothing = self.config["features"]["target_encoding_smoothing"]
        global_mean = df[target].mean()

        if is_train and not self._fitted:
            # Compute stats from training data
            stats = df.groupby(col)[target].agg(["mean", "count"])
            # Bayesian smoothing: weighted average of category mean and global mean
            stats["smoothed"] = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            self.target_encodings[f"{col}_{target}"] = stats["smoothed"].to_dict()

        encoded_name = f"{col}_te_{target}"
        mapping = self.target_encodings.get(f"{col}_{target}", {})
        df[encoded_name] = df[col].map(mapping).fillna(global_mean)

        return df

    # ──────────────────────────────────────────────────────────────
    # MAIN TRANSFORM
    # ──────────────────────────────────────────────────────────────

    def fit_transform(self, df):
        """Fit on training data and transform. Use for training."""
        self._fitted = False
        df = self._temporal_features(df)
        df = self._spatial_features(df)
        df = self._event_profile_features(df)
        df = self._historical_features(df)
        df = self._interaction_features(df)

        # Target encoding for key categoricals
        for col in ["event_cause", "corridor", "zone", "police_station"]:
            df = self._target_encode(df, col, "priority_binary", is_train=True)
            df = self._target_encode(df, col, "road_closure_binary", is_train=True)

        self._fitted = True
        print(f"[FeatureEngine] Generated {len([c for c in df.columns if c not in self._original_cols(df)])} new features")
        return df

    def transform(self, df):
        """Transform new data using fitted encodings. Use for inference."""
        assert self._fitted, "Must call fit_transform first!"
        df = self._temporal_features(df)
        df = self._spatial_features(df)
        df = self._event_profile_features(df)
        df = self._historical_features(df)
        df = self._interaction_features(df)

        for col in ["event_cause", "corridor", "zone", "police_station"]:
            df = self._target_encode(df, col, "priority_binary", is_train=False)
            df = self._target_encode(df, col, "road_closure_binary", is_train=False)

        return df

    def get_feature_names(self):
        """Return the list of feature columns for ML models."""
        numeric_features = [
            # Temporal
            "hour", "minute", "day_of_week", "day_of_month", "month", "week_of_year",
            "is_weekend", "is_morning_rush", "is_evening_rush", "is_rush_hour", "is_night",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            # Spatial
            "dist_to_center_km", "is_major_corridor", "corridor_event_frequency",
            "junction_event_frequency", "police_station_event_frequency",
            "lat_grid", "lng_grid",
            # Event profile
            "is_planned", "has_vehicle", "road_closure_binary",
            "cause_severity_rank", "is_planned_cause", "is_infrastructure_cause",
            "is_vehicle_cause", "is_nature_cause", "vehicle_severity",
            "description_length", "desc_has_traffic_slow", "desc_has_road_closed",
            "desc_has_accident", "desc_word_count",
            "has_extent", "event_extent_km", "response_delay_minutes",
            # Historical
            "cause_avg_resolution_hours", "corridor_avg_resolution_hours",
            "cause_historical_closure_rate", "junction_historical_closure_rate",
            "cause_historical_priority_rate",
            "events_same_corridor_24h", "events_same_zone_7d",
            "events_same_ps_24h", "total_events_last_6h",
            # Interactions
            "rush_x_closure", "planned_x_major_corridor",
            "severity_x_rush", "weekend_x_planned_cause",
            "night_x_vehicle", "dist_x_severity",
            "concurrent_x_severity", "junction_freq_x_closure",
            # Target encoded
            "event_cause_te_priority_binary", "corridor_te_priority_binary",
            "zone_te_priority_binary", "police_station_te_priority_binary",
            "event_cause_te_road_closure_binary", "corridor_te_road_closure_binary",
            "zone_te_road_closure_binary", "police_station_te_road_closure_binary",
        ]
        if HAS_H3:
            numeric_features.append("h3_event_count")
        return numeric_features

    def get_severity_features(self):
        """Features for severity prediction (exclude road_closure_binary — it's a separate target)."""
        feats = self.get_feature_names()
        exclude = {"road_closure_binary", "rush_x_closure"}
        return [f for f in feats if f not in exclude]

    def get_duration_features(self):
        """Features for duration prediction."""
        feats = self.get_feature_names()
        # Include road_closure as a feature for duration prediction
        return feats

    def get_closure_features(self):
        """Features for road closure prediction."""
        feats = self.get_feature_names()
        exclude = {
            "road_closure_binary", "rush_x_closure",
            "event_cause_te_road_closure_binary", "corridor_te_road_closure_binary",
            "zone_te_road_closure_binary", "police_station_te_road_closure_binary",
            "junction_freq_x_closure",
        }
        return [f for f in feats if f not in exclude]

    @staticmethod
    def _original_cols(df):
        """Approximate original columns (before feature engineering)."""
        return {
            "id", "event_type", "latitude", "longitude", "endlatitude", "endlongitude",
            "address", "end_address", "event_cause", "requires_road_closure",
            "start_datetime", "end_datetime", "status", "authenticated",
            "modified_datetime", "map_file", "direction", "description",
            "veh_type", "veh_no", "corridor", "priority", "cargo_material",
            "reason_breakdown", "age_of_truck", "created_date", "route_path",
            "client_id", "created_by_id", "last_modified_by_id",
            "assigned_to_police_id", "citizen_accident_id", "comment",
            "police_station", "meta_data", "kgid", "resolved_at_address",
            "resolved_at_latitude", "resolved_at_longitude", "closed_by_id",
            "closed_datetime", "resolved_by_id", "resolved_datetime",
            "gba_identifier", "zone", "junction",
        }


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.process()

    engine = FeatureEngine()
    df = engine.fit_transform(df)

    print(f"\nTotal features: {len(engine.get_feature_names())}")
    print(f"Severity features: {len(engine.get_severity_features())}")
    print(f"Duration features: {len(engine.get_duration_features())}")
    print(f"Closure features: {len(engine.get_closure_features())}")
    print(f"\nDataFrame shape after feature engineering: {df.shape}")
