"""
Data Preprocessing Module
=========================
Handles raw CSV ingestion, cleaning, parsing, and transformation
for the Astram event data. Produces a clean DataFrame ready for
feature engineering.
"""

import pandas as pd
import numpy as np
import yaml
import os
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    End-to-end data preprocessor for Astram traffic event data.
    
    Handles:
    - Mixed datetime parsing
    - NULL handling and imputation
    - Event cause normalization
    - Duration computation (resolution time)
    - Coordinate validation
    - Timezone conversion (UTC → IST)
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.timezone = self.config["data"]["timezone"]

    def load_raw(self):
        """Load raw CSV data."""
        csv_path = self.config["data"]["raw_csv"]
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[DataPreprocessor] Loaded {len(df)} records from {csv_path}")
        return df

    def _parse_datetimes(self, df):
        """Parse all datetime columns with mixed formats."""
        datetime_cols = [
            "start_datetime",
            "end_datetime",
            "created_date",
            "modified_datetime",
            "closed_datetime",
            "resolved_datetime",
        ]
        for col in datetime_cols:
            if col in df.columns:
                # Replace literal 'NULL' strings
                df[col] = df[col].replace("NULL", pd.NaT)
                df[col] = pd.to_datetime(df[col], format="mixed", utc=True, errors="coerce")
                # Convert to IST
                df[col] = df[col].dt.tz_convert(self.timezone)
        return df

    def _normalize_event_causes(self, df):
        """Normalize inconsistent event_cause values."""
        cause_mapping = {
            "Debris": "debris",
            "Fog / Low Visibility": "fog_low_visibility",
            "test_demo": "others",
        }
        df["event_cause"] = df["event_cause"].str.strip().replace(cause_mapping)
        return df

    def _validate_coordinates(self, df):
        """Validate and clean geographic coordinates."""
        bbox = self.config["spatial"]["bbox"]

        # Flag invalid start coordinates
        valid_start = (
            (df["latitude"] >= bbox["min_lat"])
            & (df["latitude"] <= bbox["max_lat"])
            & (df["longitude"] >= bbox["min_lng"])
            & (df["longitude"] <= bbox["max_lng"])
        )
        n_invalid = (~valid_start).sum()
        if n_invalid > 0:
            print(f"[DataPreprocessor] Warning: {n_invalid} records with start coords outside Bengaluru bbox")

        # Clean end coordinates: replace 0 with NaN
        df.loc[df["endlatitude"] == 0, "endlatitude"] = np.nan
        df.loc[df["endlongitude"] == 0, "endlongitude"] = np.nan

        # Replace NULL strings
        for col in ["endlatitude", "endlongitude", "resolved_at_latitude", "resolved_at_longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _compute_durations(self, df):
        """Compute resolution/closure durations in hours."""
        # Resolution time: from start to closed/resolved
        df["closed_datetime_combined"] = df["closed_datetime"].fillna(df["resolved_datetime"])
        df["resolution_hours"] = (
            df["closed_datetime_combined"] - df["start_datetime"]
        ).dt.total_seconds() / 3600.0

        # Clip extreme values: negative durations and >720h (30 days)
        df.loc[df["resolution_hours"] < 0, "resolution_hours"] = np.nan
        df.loc[df["resolution_hours"] > 720, "resolution_hours"] = np.nan

        # Planned event duration (if end_datetime exists)
        df["planned_duration_hours"] = (
            df["end_datetime"] - df["start_datetime"]
        ).dt.total_seconds() / 3600.0
        df.loc[df["planned_duration_hours"] < 0, "planned_duration_hours"] = np.nan

        # Response time: from start to created (how fast was it reported)
        df["response_delay_minutes"] = (
            df["created_date"] - df["start_datetime"]
        ).dt.total_seconds() / 60.0
        df.loc[df["response_delay_minutes"] < 0, "response_delay_minutes"] = np.nan
        df.loc[df["response_delay_minutes"] > 1440, "response_delay_minutes"] = np.nan  # cap at 24h

        return df

    def _handle_missing_values(self, df):
        """Handle missing values with domain-appropriate strategies."""
        # Boolean: requires_road_closure — already clean (TRUE/FALSE)
        # Categorical: fill with 'unknown'
        cat_fill_cols = [
            "corridor",
            "zone",
            "police_station",
            "junction",
            "veh_type",
            "direction",
            "gba_identifier",
        ]
        for col in cat_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        # Description: fill with empty string
        df["description"] = df["description"].fillna("")

        # Priority: should not be missing, but if so
        if df["priority"].isnull().any():
            df["priority"] = df["priority"].fillna("Low")

        return df

    def _create_binary_targets(self, df):
        """Create binary/numeric targets for ML models."""
        # Priority: High=1, Low=0
        df["priority_binary"] = (df["priority"] == "High").astype(int)

        # Road closure: already boolean
        df["road_closure_binary"] = df["requires_road_closure"].astype(int)

        # Is planned
        df["is_planned"] = (df["event_type"] == "planned").astype(int)

        # Has vehicle
        df["has_vehicle"] = (df["veh_type"] != "unknown").astype(int)

        # Status encoding
        status_map = {"active": 0, "resolved": 1, "closed": 2}
        df["status_encoded"] = df["status"].map(status_map).fillna(0).astype(int)

        return df

    def _add_computed_columns(self, df):
        """Add derived columns useful for analysis."""
        # Description length as proxy for severity
        df["description_length"] = df["description"].str.len()

        # Has end address (indicates extent of event)
        df["has_extent"] = (
            df["endlatitude"].notna() & df["endlongitude"].notna()
        ).astype(int)

        # Spatial extent (haversine distance between start and end)
        mask = df["has_extent"] == 1
        if mask.any():
            df.loc[mask, "event_extent_km"] = self._haversine(
                df.loc[mask, "latitude"],
                df.loc[mask, "longitude"],
                df.loc[mask, "endlatitude"],
                df.loc[mask, "endlongitude"],
            )
        df["event_extent_km"] = df.get("event_extent_km", pd.Series(dtype=float)).fillna(0)

        return df

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Compute haversine distance in km between two coordinate arrays."""
        R = 6371.0  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def process(self):
        """Run the full preprocessing pipeline."""
        print("=" * 60)
        print("RUNNING DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        df = self.load_raw()
        df = self._parse_datetimes(df)
        df = self._normalize_event_causes(df)
        df = self._validate_coordinates(df)
        df = self._compute_durations(df)
        df = self._handle_missing_values(df)
        df = self._create_binary_targets(df)
        df = self._add_computed_columns(df)

        # Sort by start_datetime for temporal consistency
        df = df.sort_values("start_datetime").reset_index(drop=True)

        # Save processed data
        os.makedirs("outputs", exist_ok=True)
        output_path = self.config["data"]["processed_parquet"]
        df.to_parquet(output_path, index=False)
        print(f"\n[DataPreprocessor] Saved {len(df)} processed records to {output_path}")

        # Print summary stats
        self._print_summary(df)

        return df

    def _print_summary(self, df):
        """Print preprocessing summary."""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['start_datetime'].min()} -> {df['start_datetime'].max()}")
        print(f"Planned events: {df['is_planned'].sum()}")
        print(f"Unplanned events: {(~df['is_planned'].astype(bool)).sum()}")
        print(f"High priority: {df['priority_binary'].sum()} ({df['priority_binary'].mean():.1%})")
        print(f"Road closures: {df['road_closure_binary'].sum()} ({df['road_closure_binary'].mean():.1%})")
        print(f"Resolution time (median): {df['resolution_hours'].median():.1f}h")
        print(f"Resolution time (mean): {df['resolution_hours'].mean():.1f}h")
        print(f"Events with vehicles: {df['has_vehicle'].sum()}")
        print(f"Unique corridors: {df['corridor'].nunique()}")
        print(f"Unique junctions: {df['junction'].nunique()}")
        print(f"Unique police stations: {df['police_station'].nunique()}")
        print("=" * 60)


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.process()
