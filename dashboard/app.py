"""
Interactive Dashboard for Event-Driven Congestion System
=========================================================
Streamlit-based dashboard providing:
1. Live Event Map with clustered markers
2. Prediction Interface for new events
3. Historical Analysis with filters
4. Post-Event Learning Reports
5. Model Performance Metrics

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Traffic Event Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main .block-container {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        color: #f0f0f0 !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .css-1d391kg {  /* sidebar */
        background: rgba(15, 12, 41, 0.95);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load preprocessed data."""
    try:
        df = pd.read_parquet("outputs/processed_events.parquet")
        return df
    except FileNotFoundError:
        # Fall back to raw CSV
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = yaml.safe_load(f)
            csv_path = config["data"]["raw_csv"]
        else:
            csv_path = "Astram event data_anonymized - Astram event data_anonymizedb40ac87 (1).csv"
        df = pd.read_csv(csv_path, low_memory=False)
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], format="mixed", utc=True)
        return df


@st.cache_data
def load_metrics():
    """Load model metrics."""
    try:
        with open("outputs/reports/final_metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_post_event_report():
    """Load post-event learning report."""
    try:
        with open("outputs/reports/post_event_report.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    # ── Sidebar ──
    st.sidebar.title("🚦 Traffic Command Center")
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🗺️ Event Map", "🔮 Predict Event", "📈 Model Performance", "📋 Post-Event Learning"],
    )

    df = load_data()

    if page == "📊 Dashboard":
        render_dashboard(df)
    elif page == "🗺️ Event Map":
        render_event_map(df)
    elif page == "🔮 Predict Event":
        render_prediction_interface(df)
    elif page == "📈 Model Performance":
        render_model_performance()
    elif page == "📋 Post-Event Learning":
        render_post_event_learning()


def render_dashboard(df):
    """Main dashboard with KPI overview."""
    st.title("🚦 Bengaluru Traffic Event Intelligence")
    st.markdown("*Real-time insights into event-driven congestion*")

    # ── KPI Row ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Events", f"{len(df):,}")
    with col2:
        planned = df[df["event_type"] == "planned"] if "event_type" in df.columns else df.head(0)
        st.metric("Planned Events", f"{len(planned):,}")
    with col3:
        if "priority" in df.columns:
            high_pct = (df["priority"] == "High").mean()
            st.metric("High Priority %", f"{high_pct:.1%}")
    with col4:
        if "requires_road_closure" in df.columns:
            closures = df["requires_road_closure"].sum()
            st.metric("Road Closures", f"{closures:,}")
    with col5:
        if "event_cause" in df.columns:
            st.metric("Event Types", f"{df['event_cause'].nunique()}")

    st.divider()

    # ── Charts Row 1 ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Events by Cause")
        if "event_cause" in df.columns:
            cause_counts = df["event_cause"].value_counts().reset_index()
            cause_counts.columns = ["cause", "count"]
            fig = px.bar(
                cause_counts,
                x="count",
                y="cause",
                orientation="h",
                color="count",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⏰ Events by Hour (IST)")
        if "start_datetime" in df.columns:
            hours = df["start_datetime"].dt.hour
            hourly = hours.value_counts().sort_index().reset_index()
            hourly.columns = ["hour", "count"]
            fig = px.area(
                hourly,
                x="hour",
                y="count",
                color_discrete_sequence=["#00d2ff"],
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛣️ Top Corridors")
        if "corridor" in df.columns:
            corridor_data = df["corridor"].value_counts().head(12).reset_index()
            corridor_data.columns = ["corridor", "count"]
            fig = px.bar(
                corridor_data,
                x="count",
                y="corridor",
                orientation="h",
                color="count",
                color_continuous_scale="Inferno",
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔴 Priority Distribution")
        if "priority" in df.columns:
            fig = px.pie(
                df,
                names="priority",
                color="priority",
                color_discrete_map={"High": "#e74c3c", "Low": "#2ecc71"},
                hole=0.4,
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_event_map(df):
    """Interactive event map."""
    st.title("🗺️ Event Location Map")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        if "event_cause" in df.columns:
            causes = ["All"] + sorted(df["event_cause"].unique().tolist())
            selected_cause = st.selectbox("Event Cause", causes)
    with col2:
        if "priority" in df.columns:
            priorities = ["All", "High", "Low"]
            selected_priority = st.selectbox("Priority", priorities)
    with col3:
        if "event_type" in df.columns:
            types = ["All", "planned", "unplanned"]
            selected_type = st.selectbox("Event Type", types)

    # Filter data
    filtered = df.copy()
    if "event_cause" in df.columns and selected_cause != "All":
        filtered = filtered[filtered["event_cause"] == selected_cause]
    if "priority" in df.columns and selected_priority != "All":
        filtered = filtered[filtered["priority"] == selected_priority]
    if "event_type" in df.columns and selected_type != "All":
        filtered = filtered[filtered["event_type"] == selected_type]

    st.info(f"Showing {len(filtered):,} events")

    # Plotly map
    if "latitude" in filtered.columns and "longitude" in filtered.columns:
        color_col = "priority" if "priority" in filtered.columns else None
        color_map = {"High": "#e74c3c", "Low": "#2ecc71"} if color_col else None

        fig = px.scatter_map(
            filtered.sample(min(2000, len(filtered))),
            lat="latitude",
            lon="longitude",
            color=color_col,
            color_discrete_map=color_map,
            hover_data=["event_cause", "corridor", "address"] if "address" in filtered.columns else ["event_cause"],
            zoom=10,
            height=700,
            opacity=0.6,
        )
        fig.update_layout(
            map_style="carto-darkmatter",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_prediction_interface(df):
    """Interface for predicting new events."""
    st.title("🔮 Event Impact Predictor")
    st.markdown("*Enter event details to get predictions and resource recommendations*")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Event Details")
        event_cause = st.selectbox(
            "Event Cause",
            ["vehicle_breakdown", "accident", "construction", "public_event",
             "procession", "vip_movement", "protest", "tree_fall",
             "water_logging", "congestion", "pot_holes", "road_conditions", "others"],
        )
        event_type = st.selectbox("Event Type", ["planned", "unplanned"])
        corridor = st.selectbox(
            "Corridor",
            ["Non-corridor"] + sorted([c for c in df["corridor"].unique() if c != "Non-corridor" and c != "unknown"]) if "corridor" in df.columns else ["Non-corridor"],
        )
        priority_guess = st.selectbox("Initial Priority Estimate", ["High", "Low"])

    with col2:
        st.subheader("Timing & Location")
        hour = st.slider("Hour (IST)", 0, 23, 9)
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        is_rush = hour in [8, 9, 10, 17, 18, 19, 20]
        st.info(f"{'🔴 RUSH HOUR' if is_rush else '🟢 Off-peak'}")

        requires_closure = st.checkbox("Requires Road Closure?")

    if st.button("🔮 Predict Impact & Generate Plan", type="primary"):
        st.divider()

        # Simulated predictions (would use actual model in production)
        severity_scores = {
            "vip_movement": 0.95, "protest": 0.90, "procession": 0.85,
            "public_event": 0.80, "accident": 0.75, "tree_fall": 0.60,
            "water_logging": 0.55, "construction": 0.50, "congestion": 0.65,
            "vehicle_breakdown": 0.40, "pot_holes": 0.25, "road_conditions": 0.30,
            "others": 0.35,
        }

        base_severity = severity_scores.get(event_cause, 0.5)
        rush_boost = 0.15 if is_rush else 0
        corridor_boost = 0.1 if corridor != "Non-corridor" else 0
        severity_proba = min(0.99, base_severity + rush_boost + corridor_boost)

        duration_base = {
            "construction": 48, "public_event": 8, "procession": 4,
            "vip_movement": 3, "protest": 6, "accident": 2,
            "vehicle_breakdown": 1.5, "tree_fall": 4, "water_logging": 6,
            "congestion": 2, "pot_holes": 24, "road_conditions": 12, "others": 2,
        }
        predicted_duration = duration_base.get(event_cause, 2) * (1.3 if is_rush else 1.0)

        # Display predictions
        col1, col2, col3 = st.columns(3)
        with col1:
            color = "🔴" if severity_proba > 0.6 else "🟡" if severity_proba > 0.4 else "🟢"
            st.metric(f"{color} Severity", f"{'High' if severity_proba > 0.5 else 'Low'}", f"{severity_proba:.0%} confidence")
        with col2:
            st.metric("⏱️ Est. Duration", f"{predicted_duration:.1f}h", f"P90: {predicted_duration * 2:.1f}h")
        with col3:
            st.metric("🚧 Road Closure", "YES" if requires_closure or severity_proba > 0.8 else "NO")

        st.divider()

        # Resource Recommendation
        st.subheader("📋 Resource Deployment Plan")
        manpower_base = {
            "vehicle_breakdown": 2, "accident": 4, "construction": 3,
            "public_event": 8, "procession": 10, "vip_movement": 12,
            "protest": 10, "tree_fall": 3, "water_logging": 3,
            "congestion": 4, "pot_holes": 2, "road_conditions": 2, "others": 2,
        }
        base = manpower_base.get(event_cause, 2)
        total_personnel = int(base * (1.5 if severity_proba > 0.5 else 1.0) * (1.3 if is_rush else 1.0))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👮 Total Personnel", total_personnel)
        with col2:
            st.metric("🚧 Barricades", 12 if requires_closure else 6 if severity_proba > 0.5 else 0)
        with col3:
            st.metric("📡 Shifts Needed", max(1, int(predicted_duration / 8)))
        with col4:
            cost = total_personnel * 400 * predicted_duration
            st.metric("💰 Est. Cost", f"₹{cost:,.0f}")

        # Diversion routes
        st.subheader("🔄 Suggested Diversions")
        diversion_map = {
            "Bellary Road 1": "Palace Road → Sankey Road → Sadashivanagar",
            "Mysore Road": "Chord Road → Vijayanagar",
            "Tumkur Road": "ORR → Yeshwanthpur via Rajajinagar",
            "Hosur Road": "Bannerghatta Road → JP Nagar",
            "ORR East 1": "Old Airport Road → HAL",
        }
        diversion = diversion_map.get(corridor, "Use adjacent parallel roads — check Google Maps")
        st.info(f"🔄 **Primary Diversion**: {diversion}")

        # Equipment
        st.subheader("🧰 Recommended Equipment")
        base_equip = ["Walkie-talkies", "Reflective jackets", "Whistles"]
        cause_equip = {
            "accident": ["First aid kit", "Fire extinguisher", "Tow truck", "Ambulance"],
            "tree_fall": ["Chainsaw", "JCB/crane", "Warning lights"],
            "public_event": ["PA system", "Crowd barriers", "CCTV van"],
            "vip_movement": ["Escort vehicles", "Radio comm", "Route clearance"],
            "protest": ["Crowd barriers", "Water cannon (standby)", "Riot gear"],
        }
        equipment = base_equip + cause_equip.get(event_cause, [])
        st.write(", ".join(f"✅ {e}" for e in equipment))


def render_model_performance():
    """Model performance metrics."""
    st.title("📈 Model Performance")

    metrics = load_metrics()
    if metrics is None:
        st.warning("⚠️ No model metrics found. Run the training pipeline first: `python -m src.pipeline`")
        return

    # Severity model
    st.subheader("🎯 Impact Severity Model")
    sev = metrics.get("severity", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{sev.get('roc_auc', 0):.4f}")
    with col2:
        st.metric("F1 Score", f"{sev.get('f1', 0):.4f}")
    with col3:
        st.metric("Precision", f"{sev.get('precision', 0):.4f}")
    with col4:
        st.metric("Recall", f"{sev.get('recall', 0):.4f}")

    st.divider()

    # Duration model
    st.subheader("⏱️ Duration Prediction Model")
    dur = metrics.get("duration", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE (hours)", f"{dur.get('mae_hours', 0):.2f}")
    with col2:
        st.metric("R² Score", f"{dur.get('r2', 0):.4f}")
    with col3:
        st.metric("80% PI Coverage", f"{dur.get('coverage_80pct', 0):.1%}")
    with col4:
        st.metric("50% PI Coverage", f"{dur.get('coverage_50pct', 0):.1%}")

    st.divider()

    # Closure model
    st.subheader("🚧 Road Closure Model")
    clos = metrics.get("closure", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{clos.get('roc_auc', 0):.4f}")
    with col2:
        st.metric("F1 Score", f"{clos.get('f1', 0):.4f}")
    with col3:
        st.metric("Precision", f"{clos.get('precision', 0):.4f}")
    with col4:
        st.metric("Recall", f"{clos.get('recall', 0):.4f}")

    st.divider()
    st.subheader("📊 Training Info")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", f"{metrics.get('training_samples', 0):,}")
    with col2:
        st.metric("Test Samples", f"{metrics.get('test_samples', 0):,}")
    with col3:
        st.metric("Total Features", f"{metrics.get('total_features', 0)}")

    # Feature importance plots
    st.subheader("🏆 Feature Importances")
    plot_path = "outputs/plots/11_model_feature_importance.png"
    if os.path.exists(plot_path):
        st.image(plot_path)
    else:
        st.info("Run EDA visualizations to generate feature importance plots")


def render_post_event_learning():
    """Post-event learning analysis."""
    st.title("📋 Post-Event Learning System")

    report = load_post_event_report()
    if report is None:
        st.warning("⚠️ No post-event report found. Run the training pipeline first.")
        return

    # Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Events Analyzed", f"{report.get('total_events', 0):,}")
    with col2:
        st.metric("Severity Accuracy", f"{report.get('severity_accuracy', 0):.1%}")
    with col3:
        st.metric("Closure Accuracy", f"{report.get('closure_accuracy', 0):.1%}")

    st.divider()

    # Severity bias analysis
    st.subheader("🎯 Severity Prediction Bias by Event Cause")
    sev_bias = report.get("severity_bias_by_cause", {})
    if sev_bias:
        bias_df = pd.DataFrame(sev_bias).T
        bias_df.index.name = "Event Cause"
        st.dataframe(bias_df.style.background_gradient(subset=["bias"], cmap="RdYlGn_r"))

    st.divider()

    # Duration bias
    st.subheader("⏱️ Duration Prediction Bias")
    if "duration_bias_hours" in report:
        st.metric("Mean Duration Bias", f"{report['duration_bias_hours']:+.2f}h")
        st.metric("Duration MAE", f"{report.get('duration_mae_hours', 0):.2f}h")

    dur_bias = report.get("duration_bias_by_cause", {})
    if dur_bias:
        dur_df = pd.DataFrame(dur_bias).T
        dur_df.index.name = "Event Cause"
        st.dataframe(dur_df.style.background_gradient(subset=["ratio"], cmap="RdYlGn_r"))

    st.divider()

    # Correction factors
    st.subheader("🔧 Auto-Generated Correction Factors")
    corrections = report.get("correction_factors", {})
    if corrections:
        corr_df = pd.DataFrame(list(corrections.items()), columns=["Event Cause", "Correction Factor"])
        corr_df["Direction"] = corr_df["Correction Factor"].apply(
            lambda x: "↑ Increase resources" if x > 1 else "↓ Decrease resources" if x < 1 else "= No change"
        )
        st.dataframe(corr_df)


if __name__ == "__main__":
    main()
