"""
EDA Visualizations Module
==========================
Generates publication-quality visualizations for the competition presentation.
All plots are saved to outputs/plots/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
import os
import yaml
import warnings

warnings.filterwarnings("ignore")

# Style configuration
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


class EDAVisualizer:
    """Generate comprehensive EDA visualizations."""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.output_dir = self.config["outputs"]["plots_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all(self, df):
        """Generate all visualizations."""
        print("=" * 60)
        print("GENERATING EDA VISUALIZATIONS")
        print("=" * 60)

        self.plot_event_cause_distribution(df)
        self.plot_temporal_heatmap(df)
        self.plot_hourly_distribution(df)
        self.plot_corridor_analysis(df)
        self.plot_severity_by_cause(df)
        self.plot_duration_distribution(df)
        self.plot_spatial_scatter(df)
        self.plot_closure_analysis(df)
        self.plot_zone_analysis(df)
        self.plot_planned_vs_unplanned(df)

        print(f"\n[EDA] All visualizations saved to {self.output_dir}/")

    def plot_event_cause_distribution(self, df):
        """Event cause distribution with planned/unplanned breakdown."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Overall distribution
        cause_counts = df["event_cause"].value_counts()
        colors = sns.color_palette("husl", len(cause_counts))
        ax = axes[0]
        bars = ax.barh(cause_counts.index[::-1], cause_counts.values[::-1], color=colors[::-1])
        ax.set_title("Event Cause Distribution", fontweight="bold")
        ax.set_xlabel("Count")
        for bar, val in zip(bars, cause_counts.values[::-1]):
            ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=10)

        # Planned vs Unplanned breakdown
        ax = axes[1]
        ct = pd.crosstab(df["event_cause"], df["event_type"])
        ct = ct.reindex(cause_counts.index)
        ct.plot(kind="barh", stacked=True, ax=ax, color=["#e74c3c", "#2ecc71"])
        ax.set_title("Planned vs Unplanned by Cause", fontweight="bold")
        ax.legend(title="Event Type")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_event_cause_distribution.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Event cause distribution")

    def plot_temporal_heatmap(self, df):
        """Hour × Day-of-week heatmap."""
        if "hour" not in df.columns:
            df["hour"] = df["start_datetime"].dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["start_datetime"].dt.dayofweek

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # All events
        pivot = df.pivot_table(index="hour", columns="day_of_week", aggfunc="size", fill_value=0)
        pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", ax=axes[0], linewidths=0.5)
        axes[0].set_title("All Events: Hour × Day-of-Week", fontweight="bold")
        axes[0].set_ylabel("Hour (IST)")

        # High priority only
        high = df[df["priority"] == "High"]
        pivot_high = high.pivot_table(index="hour", columns="day_of_week", aggfunc="size", fill_value=0)
        pivot_high.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.heatmap(pivot_high, cmap="YlOrRd", annot=True, fmt="d", ax=axes[1], linewidths=0.5)
        axes[1].set_title("High Priority Events: Hour × Day-of-Week", fontweight="bold")
        axes[1].set_ylabel("Hour (IST)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "02_temporal_heatmap.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Temporal heatmap")

    def plot_hourly_distribution(self, df):
        """Hourly event distribution by event type."""
        if "hour" not in df.columns:
            df["hour"] = df["start_datetime"].dt.hour

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Overall hourly
        ax = axes[0, 0]
        hourly = df.groupby("hour").size()
        ax.fill_between(hourly.index, hourly.values, alpha=0.3, color="#3498db")
        ax.plot(hourly.index, hourly.values, "-o", color="#3498db", linewidth=2)
        ax.set_title("Events per Hour (IST)", fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, 24))

        # By event cause (top 5)
        ax = axes[0, 1]
        top_causes = df["event_cause"].value_counts().head(5).index
        for cause in top_causes:
            subset = df[df["event_cause"] == cause]
            hourly_cause = subset.groupby("hour").size().reindex(range(24), fill_value=0)
            ax.plot(hourly_cause.index, hourly_cause.values, "-o", label=cause, linewidth=2)
        ax.set_title("Hourly by Top Event Causes", fontweight="bold")
        ax.set_xlabel("Hour")
        ax.legend()
        ax.set_xticks(range(0, 24))

        # By priority
        ax = axes[1, 0]
        for priority in ["High", "Low"]:
            subset = df[df["priority"] == priority]
            hourly_p = subset.groupby("hour").size().reindex(range(24), fill_value=0)
            ax.plot(hourly_p.index, hourly_p.values, "-o", label=priority, linewidth=2)
        ax.set_title("Hourly by Priority", fontweight="bold")
        ax.set_xlabel("Hour")
        ax.legend()
        ax.set_xticks(range(0, 24))

        # Weekly distribution
        ax = axes[1, 1]
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["start_datetime"].dt.dayofweek
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_counts = df["day_of_week"].value_counts().sort_index()
        ax.bar(dow_names, dow_counts.values, color=sns.color_palette("viridis", 7))
        ax.set_title("Events per Day of Week", fontweight="bold")
        ax.set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "03_hourly_distribution.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Hourly distribution")

    def plot_corridor_analysis(self, df):
        """Corridor-level analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Top corridors
        ax = axes[0]
        top_corridors = df["corridor"].value_counts().head(15)
        colors = ["#e74c3c" if c != "Non-corridor" else "#95a5a6" for c in top_corridors.index]
        bars = ax.barh(top_corridors.index[::-1], top_corridors.values[::-1], color=colors[::-1])
        ax.set_title("Top 15 Corridors by Event Count", fontweight="bold")
        ax.set_xlabel("Events")

        # Priority rate by corridor
        ax = axes[1]
        corridor_priority = df.groupby("corridor")["priority_binary"].agg(["mean", "count"])
        corridor_priority = corridor_priority[corridor_priority["count"] >= 10].sort_values("mean", ascending=True)
        corridor_priority = corridor_priority.tail(15)
        ax.barh(corridor_priority.index, corridor_priority["mean"], color="#e67e22")
        ax.set_title("High-Priority Rate by Corridor (min 10 events)", fontweight="bold")
        ax.set_xlabel("High Priority Rate")
        ax.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "04_corridor_analysis.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Corridor analysis")

    def plot_severity_by_cause(self, df):
        """Severity (priority) breakdown by event cause."""
        fig, ax = plt.subplots(figsize=(14, 8))

        ct = pd.crosstab(df["event_cause"], df["priority"], normalize="index")
        ct = ct.reindex(df["event_cause"].value_counts().index)
        ct.plot(kind="barh", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"])
        ax.set_title("Priority Distribution by Event Cause", fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.legend(title="Priority", loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "05_severity_by_cause.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Severity by cause")

    def plot_duration_distribution(self, df):
        """Resolution time distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        valid = df[df["resolution_hours"].notna() & (df["resolution_hours"] > 0) & (df["resolution_hours"] < 200)]

        # Overall distribution (log scale)
        ax = axes[0, 0]
        ax.hist(np.log1p(valid["resolution_hours"]), bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.set_title("Resolution Time (Log Scale)", fontweight="bold")
        ax.set_xlabel("log(hours + 1)")
        ax.set_ylabel("Count")

        # By event cause (box plot, top causes)
        ax = axes[0, 1]
        top_causes = valid["event_cause"].value_counts().head(8).index
        data_to_plot = [valid[valid["event_cause"] == c]["resolution_hours"].values for c in top_causes]
        bp = ax.boxplot(data_to_plot, labels=top_causes, vert=True, patch_artist=True)
        for patch, color in zip(bp["boxes"], sns.color_palette("husl", 8)):
            patch.set_facecolor(color)
        ax.set_title("Resolution Time by Cause", fontweight="bold")
        ax.set_ylabel("Hours")
        ax.set_xticklabels(top_causes, rotation=45, ha="right")
        ax.set_ylim(0, 50)

        # By priority
        ax = axes[1, 0]
        for priority in ["High", "Low"]:
            subset = valid[valid["priority"] == priority]
            ax.hist(np.log1p(subset["resolution_hours"]), bins=40, alpha=0.6, label=priority)
        ax.set_title("Resolution Time by Priority (Log Scale)", fontweight="bold")
        ax.set_xlabel("log(hours + 1)")
        ax.legend()

        # By planned/unplanned
        ax = axes[1, 1]
        for etype in ["planned", "unplanned"]:
            subset = valid[valid["event_type"] == etype]
            if len(subset) > 0:
                ax.hist(np.log1p(subset["resolution_hours"]), bins=40, alpha=0.6, label=etype)
        ax.set_title("Resolution Time: Planned vs Unplanned (Log Scale)", fontweight="bold")
        ax.set_xlabel("log(hours + 1)")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "06_duration_distribution.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Duration distribution")

    def plot_spatial_scatter(self, df):
        """Spatial scatter plot of events."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Color by priority
        ax = axes[0]
        low = df[df["priority"] == "Low"]
        high = df[df["priority"] == "High"]
        ax.scatter(low["longitude"], low["latitude"], c="#2ecc71", alpha=0.3, s=5, label="Low")
        ax.scatter(high["longitude"], high["latitude"], c="#e74c3c", alpha=0.3, s=5, label="High")
        ax.set_title("Event Locations by Priority", fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        # Color by event type
        ax = axes[1]
        unplanned = df[df["event_type"] == "unplanned"]
        planned = df[df["event_type"] == "planned"]
        ax.scatter(unplanned["longitude"], unplanned["latitude"], c="#3498db", alpha=0.3, s=5, label="Unplanned")
        ax.scatter(planned["longitude"], planned["latitude"], c="#e74c3c", alpha=0.5, s=20, label="Planned", marker="*")
        ax.set_title("Event Locations: Planned vs Unplanned", fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "07_spatial_scatter.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Spatial scatter")

    def plot_closure_analysis(self, df):
        """Road closure analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Closure rate by cause
        ax = axes[0]
        closure_rate = df.groupby("event_cause")["requires_road_closure"].mean().sort_values(ascending=True)
        closure_rate = closure_rate[closure_rate > 0]
        ax.barh(closure_rate.index, closure_rate.values, color="#9b59b6")
        ax.set_title("Road Closure Rate by Event Cause", fontweight="bold")
        ax.set_xlabel("Closure Rate")
        for i, (idx, val) in enumerate(closure_rate.items()):
            ax.text(val + 0.01, i, f"{val:.1%}", va="center")

        # Closure events over time
        ax = axes[1]
        df_closures = df[df["requires_road_closure"]].copy()
        if len(df_closures) > 0:
            df_closures["date"] = df_closures["start_datetime"].dt.date
            daily = df_closures.groupby("date").size()
            ax.fill_between(daily.index, daily.values, alpha=0.4, color="#9b59b6")
            ax.plot(daily.index, daily.values, color="#9b59b6", linewidth=1)
            ax.set_title("Road Closures Over Time", fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Closures per Day")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "08_closure_analysis.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Closure analysis")

    def plot_zone_analysis(self, df):
        """Zone-level analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Events by zone
        ax = axes[0]
        zone_counts = df["zone"].value_counts()
        zone_counts = zone_counts[zone_counts.index != "unknown"]
        ax.barh(zone_counts.index[::-1], zone_counts.values[::-1], color=sns.color_palette("coolwarm", len(zone_counts)))
        ax.set_title("Events by Zone", fontweight="bold")

        # High priority rate by zone
        ax = axes[1]
        zone_priority = df[df["zone"] != "unknown"].groupby("zone")["priority_binary"].mean().sort_values()
        ax.barh(zone_priority.index, zone_priority.values, color="#e67e22")
        ax.set_title("High-Priority Rate by Zone", fontweight="bold")
        ax.set_xlabel("Rate")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "09_zone_analysis.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Zone analysis")

    def plot_planned_vs_unplanned(self, df):
        """Compare planned vs unplanned events."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        planned = df[df["event_type"] == "planned"]
        unplanned = df[df["event_type"] == "unplanned"]

        # Cause distribution for planned
        ax = axes[0, 0]
        planned["event_cause"].value_counts().plot(kind="pie", ax=ax, autopct="%1.0f%%",
                                                    colors=sns.color_palette("Set2"))
        ax.set_title(f"Planned Events (n={len(planned)})", fontweight="bold")
        ax.set_ylabel("")

        # Cause distribution for unplanned (top 8)
        ax = axes[0, 1]
        unplanned["event_cause"].value_counts().head(8).plot(kind="pie", ax=ax, autopct="%1.0f%%",
                                                              colors=sns.color_palette("Set2"))
        ax.set_title(f"Unplanned Events (n={len(unplanned)})", fontweight="bold")
        ax.set_ylabel("")

        # Closure rate comparison
        ax = axes[1, 0]
        data = pd.DataFrame({
            "Planned": [planned["requires_road_closure"].mean()],
            "Unplanned": [unplanned["requires_road_closure"].mean()],
        })
        data.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
        ax.set_title("Road Closure Rate", fontweight="bold")
        ax.set_xticklabels([""], rotation=0)
        ax.set_ylabel("Rate")

        # Priority comparison
        ax = axes[1, 1]
        ct = pd.crosstab(df["event_type"], df["priority"], normalize="index")
        ct.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
        ax.set_title("Priority by Event Type", fontweight="bold")
        ax.set_xticklabels(ct.index, rotation=0)
        ax.legend(title="Priority")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "10_planned_vs_unplanned.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Planned vs unplanned")

    def plot_model_results(self, severity_model, duration_model, closure_model):
        """Plot model-specific results (feature importances, etc.)."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 10))

        # Severity feature importance
        ax = axes[0]
        top_feats = severity_model.get_top_features(15)
        if top_feats is not None:
            ax.barh(top_feats.index[::-1], top_feats.values[::-1], color="#3498db")
            ax.set_title("Severity Model — Top Features", fontweight="bold")

        # Duration feature importance
        ax = axes[1]
        top_feats = duration_model.get_top_features(15)
        if top_feats is not None:
            ax.barh(top_feats.index[::-1], top_feats.values[::-1], color="#2ecc71")
            ax.set_title("Duration Model — Top Features", fontweight="bold")

        # Closure feature importance
        ax = axes[2]
        top_feats = closure_model.get_top_features(15)
        if top_feats is not None:
            ax.barh(top_feats.index[::-1], top_feats.values[::-1], color="#e74c3c")
            ax.set_title("Closure Model — Top Features", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "11_model_feature_importance.png"), bbox_inches="tight")
        plt.close()
        print("  ✓ Model feature importances")


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.process()

    viz = EDAVisualizer()
    viz.generate_all(df)
