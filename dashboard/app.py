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
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import yaml
import math
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.resource_recommender import ResourceRecommender
from src.pdf_generator import generate_action_plan_pdf

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Traffic Event Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# PREMIUM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg-primary: #06080F;
        --bg-secondary: #0D1117;
        --bg-card: rgba(13, 17, 23, 0.7);
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.06);
        --glass-hover: rgba(255, 255, 255, 0.08);
        --accent-indigo: #6366F1;
        --accent-violet: #8B5CF6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --accent-amber: #F59E0B;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --gradient-1: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%);
        --gradient-2: linear-gradient(135deg, #F43F5E 0%, #F59E0B 100%);
        --gradient-card: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.05) 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
    }

    .stApp {
        background: var(--bg-primary);
        background-image:
            radial-gradient(ellipse at 10% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(6,182,212,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(139,92,246,0.04) 0%, transparent 60%);
        color: var(--text-primary);
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 40%, rgba(6,182,212,0.1) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.1) 0%, transparent 40%);
        animation: hero-glow 8s ease-in-out infinite alternate;
    }
    @keyframes hero-glow {
        0% { transform: translate(0, 0); }
        100% { transform: translate(5%, 5%); }
    }
    .hero-header h1 {
        font-size: 2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #fff 0%, #c7d2fe 50%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        position: relative;
        z-index: 1;
    }
    .hero-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        background: var(--glass-hover);
        border-color: rgba(99,102,241,0.2);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.1);
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-1);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(99,102,241,0.25);
        box-shadow: 0 12px 40px rgba(99,102,241,0.12);
    }
    [data-testid="stMetric"]:hover::before {
        opacity: 1;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6,8,15,0.98) 0%, rgba(13,17,23,0.98) 100%);
        border-right: 1px solid var(--glass-border);
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: var(--text-secondary) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background: transparent !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
        margin: 0.2rem 0 !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: var(--glass-hover) !important;
        border-color: var(--glass-border) !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: rgba(99,102,241,0.1) !important;
        border-color: rgba(99,102,241,0.3) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.4) !important;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div {
        background: rgba(99,102,241,0.3) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--glass-border);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.02);
        border-radius: 14px;
        padding: 0.3rem;
        border: 1px solid var(--glass-border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.15) !important;
        color: var(--text-primary) !important;
    }

    /* ── Dividers ── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--glass-border) 50%, transparent 100%);
        margin: 1.5rem 0;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .section-header .icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .section-header .icon-indigo { background: rgba(99,102,241,0.15); }
    .section-header .icon-emerald { background: rgba(16,185,129,0.15); }
    .section-header .icon-rose { background: rgba(244,63,94,0.15); }
    .section-header .icon-amber { background: rgba(245,158,11,0.15); }
    .section-header .icon-cyan { background: rgba(6,182,212,0.15); }

    /* ── Result Cards ── */
    .result-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(12px);
    }
    .result-card.severity-high {
        border-left: 4px solid #F43F5E;
        background: rgba(244,63,94,0.04);
    }
    .result-card.severity-low {
        border-left: 4px solid #10B981;
        background: rgba(16,185,129,0.04);
    }

    /* ── Status Badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-critical { background: rgba(244,63,94,0.15); color: #F43F5E; border: 1px solid rgba(244,63,94,0.3); }
    .badge-high { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .badge-medium { background: rgba(6,182,212,0.15); color: #06B6D4; border: 1px solid rgba(6,182,212,0.3); }
    .badge-low { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }

    /* ── Diversion Route Card ── */
    .diversion-card {
        background: linear-gradient(135deg, rgba(6,182,212,0.06) 0%, rgba(99,102,241,0.04) 100%);
        border: 1px solid rgba(6,182,212,0.15);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .diversion-card .route-label {
        color: #06B6D4;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .diversion-card .route-path {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
    }

    /* ── Equipment Tag ── */
    .equip-tag {
        display: inline-block;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--glass-border);
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }
    .equip-tag:hover {
        background: rgba(99,102,241,0.1);
        border-color: rgba(99,102,241,0.2);
        color: var(--text-primary);
    }

    /* ── Communication Action Item ── */
    .comm-action {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0.8rem;
        margin: 0.25rem 0;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border-left: 3px solid var(--accent-indigo);
    }
    .comm-action .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .dot-green { background: #10B981; }
    .dot-yellow { background: #F59E0B; }
    .dot-red { background: #F43F5E; }

    /* ── Animated pulse ── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .pulse { animation: pulse 2s ease-in-out infinite; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load preprocessed data."""
    try:
        df = pd.read_parquet("outputs/processed_events.parquet")
        return df
    except FileNotFoundError:
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


@st.cache_data
def compute_data_driven_stats(df):
    """
    Pre-compute per-cause statistics from the actual data.
    These are more accurate than hardcoded heuristics, especially for duration.
    """
    stats = {}

    # Per-cause severity (priority) rates
    if "priority_binary" in df.columns and "event_cause" in df.columns:
        stats["severity_rate"] = df.groupby("event_cause")["priority_binary"].mean().to_dict()

    # Per-cause average resolution hours (filter valid durations)
    if "resolution_hours" in df.columns and "event_cause" in df.columns:
        valid = df[df["resolution_hours"].notna() & (df["resolution_hours"] > 0)]
        stats["avg_duration"] = valid.groupby("event_cause")["resolution_hours"].mean().to_dict()
        stats["median_duration"] = valid.groupby("event_cause")["resolution_hours"].median().to_dict()
        stats["p10_duration"] = valid.groupby("event_cause")["resolution_hours"].quantile(0.1).to_dict()
        stats["p90_duration"] = valid.groupby("event_cause")["resolution_hours"].quantile(0.9).to_dict()

    # Per-cause road closure rates
    if "road_closure_binary" in df.columns and "event_cause" in df.columns:
        stats["closure_rate"] = df.groupby("event_cause")["road_closure_binary"].mean().to_dict()

    # Per-corridor severity rates
    if "priority_binary" in df.columns and "corridor" in df.columns:
        stats["corridor_severity_rate"] = df.groupby("corridor")["priority_binary"].mean().to_dict()

    # Rush hour impact
    if "is_rush_hour" in df.columns or "hour" in df.columns:
        if "hour" not in df.columns and "start_datetime" in df.columns:
            df["hour"] = df["start_datetime"].dt.hour
        if "hour" in df.columns:
            rush_hours = {8, 9, 10, 17, 18, 19, 20}
            rush_mask = df["hour"].isin(rush_hours)
            if "priority_binary" in df.columns:
                stats["rush_severity_rate"] = df[rush_mask]["priority_binary"].mean()
                stats["nonrush_severity_rate"] = df[~rush_mask]["priority_binary"].mean()

    return stats


@st.cache_resource
def load_recommender():
    """Load the ResourceRecommender."""
    try:
        recommender = ResourceRecommender("config.yaml")
        # Try to load correction factors from post-event report
        try:
            with open("outputs/reports/post_event_report.json") as f:
                report = json.load(f)
            factors = report.get("correction_factors", {})
            if factors:
                recommender.update_correction_factors(factors)
        except FileNotFoundError:
            pass
        return recommender
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_osrm_route(start_lon, start_lat, end_lon, end_lat):
    """Fetch real road geometries from OSRM API."""
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=3.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("routes"):
                coords = data["routes"][0]["geometry"]["coordinates"]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                return lats, lons
    except Exception:
        pass
    # Fallback to straight line if OSRM fails
    return [start_lat, end_lat], [start_lon, end_lon]


# ──────────────────────────────────────────────────────────────
# CHART THEME
# ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
    colorway=["#6366F1", "#06B6D4", "#8B5CF6", "#10B981", "#F59E0B", "#F43F5E",
              "#EC4899", "#14B8A6", "#A78BFA", "#34D399"],
)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🚦</div>
            <div style="font-size: 1.1rem; font-weight: 800; 
                        background: linear-gradient(135deg, #fff 0%, #c7d2fe 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Traffic Command Center
            </div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.2rem;">
                Bengaluru • Real-time Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "NAVIGATION",
            ["📊 Dashboard", "🗺️ Event Map", "🔮 Predict Event",
             "📈 Model Performance", "📋 Post-Event Learning"],
        )

        st.markdown("---")

        # Sidebar footer
        st.markdown("""
        <div style="position: fixed; bottom: 1rem; padding: 0 1rem;">
            <div style="color: #475569; font-size: 0.7rem;">
                Built for Flipkart GridLock Challenge
            </div>
        </div>
        """, unsafe_allow_html=True)

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


# ──────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ──────────────────────────────────────────────────────────────

def render_dashboard(df):
    """Main dashboard with KPI overview."""
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🚦 Bengaluru Traffic Event Intelligence</h1>
        <p>Real-time insights into event-driven congestion across the city</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Events", f"{len(df):,}")
    with col2:
        planned = df[df["event_type"] == "planned"] if "event_type" in df.columns else df.head(0)
        unplanned_count = len(df) - len(planned)
        st.metric("Planned Events", f"{len(planned):,}", f"{unplanned_count:,} unplanned")
    with col3:
        if "priority" in df.columns:
            high_pct = (df["priority"] == "High").mean()
            st.metric("High Priority", f"{high_pct:.1%}", f"{(df['priority'] == 'High').sum():,} events")
    with col4:
        if "requires_road_closure" in df.columns:
            closures = df["requires_road_closure"].sum()
            st.metric("Road Closures", f"{closures:,}", f"{closures/len(df):.1%} of events")
    with col5:
        if "event_cause" in df.columns:
            st.metric("Event Types", f"{df['event_cause'].nunique()}", f"{df['corridor'].nunique()} corridors")

    st.markdown("")

    # ── Charts Row 1 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📊 Events by Cause")
        if "event_cause" in df.columns:
            cause_counts = df["event_cause"].value_counts().reset_index()
            cause_counts.columns = ["cause", "count"]
            fig = px.bar(
                cause_counts,
                x="count",
                y="cause",
                orientation="h",
                color="count",
                color_continuous_scale=[[0, "#1e1b4b"], [0.5, "#6366F1"], [1, "#06B6D4"]],
            )
            fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False, **PLOTLY_LAYOUT)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### ⏰ Events by Hour (IST)")
        if "start_datetime" in df.columns:
            hours = df["start_datetime"].dt.hour
            hourly = hours.value_counts().sort_index().reset_index()
            hourly.columns = ["hour", "count"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["count"],
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.1)",
                line=dict(color="#6366F1", width=2.5),
                mode="lines",
            ))
            # Mark rush hours
            rush_hours = hourly[hourly["hour"].isin([8, 9, 10, 17, 18, 19, 20])]
            fig.add_trace(go.Scatter(
                x=rush_hours["hour"], y=rush_hours["count"],
                mode="markers",
                marker=dict(color="#F43F5E", size=8, symbol="circle"),
                name="Rush Hour",
            ))
            fig.update_layout(height=420, showlegend=True, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🛣️ Top Corridors")
        if "corridor" in df.columns:
            corridor_data = df["corridor"].value_counts().head(15).reset_index()
            corridor_data.columns = ["corridor", "count"]
            fig = px.bar(
                corridor_data,
                x="count",
                y="corridor",
                orientation="h",
                color="count",
                color_continuous_scale=[[0, "#1a1a2e"], [0.5, "#8B5CF6"], [1, "#F59E0B"]],
            )
            fig.update_layout(height=450, showlegend=False, coloraxis_showscale=False, **PLOTLY_LAYOUT)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 🔴 Priority Distribution by Cause")
        if "priority" in df.columns and "event_cause" in df.columns:
            priority_by_cause = df.groupby(["event_cause", "priority"]).size().reset_index(name="count")
            fig = px.bar(
                priority_by_cause,
                x="count",
                y="event_cause",
                color="priority",
                orientation="h",
                color_discrete_map={"High": "#F43F5E", "Low": "#10B981"},
                barmode="stack",
            )
            fig.update_layout(height=450, **PLOTLY_LAYOUT)
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: EVENT MAP
# ──────────────────────────────────────────────────────────────

def render_event_map(df):
    """Interactive event map."""
    st.markdown("""
    <div class="hero-header">
        <h1>🗺️ Event Location Map</h1>
        <p>Visualize traffic events across Bengaluru corridors</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters in columns
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

    # Stats bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Showing Events", f"{len(filtered):,}")
    with col2:
        if "priority" in filtered.columns:
            st.metric("High Priority", f"{(filtered['priority'] == 'High').sum():,}")
    with col3:
        if "corridor" in filtered.columns:
            st.metric("Corridors Hit", f"{filtered['corridor'].nunique()}")
    with col4:
        if "requires_road_closure" in filtered.columns:
            st.metric("Closures", f"{filtered['requires_road_closure'].sum():,}")

    # Plotly map
    if "latitude" in filtered.columns and "longitude" in filtered.columns:
        color_col = "priority" if "priority" in filtered.columns else None
        color_map = {"High": "#F43F5E", "Low": "#10B981"} if color_col else None
        hover_cols = []
        for c in ["event_cause", "corridor", "address"]:
            if c in filtered.columns:
                hover_cols.append(c)

        sample_size = min(3000, len(filtered))
        fig = px.scatter_map(
            filtered.sample(sample_size) if len(filtered) > sample_size else filtered,
            lat="latitude",
            lon="longitude",
            color=color_col,
            color_discrete_map=color_map,
            hover_data=hover_cols if hover_cols else None,
            zoom=10,
            height=700,
            opacity=0.65,
            size_max=10,
        )
        fig.update_layout(
            map_style="carto-darkmatter",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: PREDICT EVENT
# ──────────────────────────────────────────────────────────────

def render_prediction_interface(df):
    """Interface for predicting new events with data-driven accuracy."""
    st.markdown("""
    <div class="hero-header">
        <h1>🔮 Event Impact Predictor</h1>
        <p>Enter event details to get data-driven predictions, diversion routes, and resource deployment plans</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data-driven stats and recommender
    stats = compute_data_driven_stats(df)
    recommender = load_recommender()

    # ── INPUT FORM ──
    st.markdown("##### 📝 Step 1: What Happened & Where?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p style="color:#94A3B8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Event Profile</p>', unsafe_allow_html=True)
        event_causes = sorted(df["event_cause"].unique().tolist()) if "event_cause" in df.columns else [
             "vehicle_breakdown", "accident", "construction", "public_event",
             "procession", "vip_movement", "protest", "tree_fall",
             "water_logging", "congestion", "pot_holes", "road_conditions", "others"]
        event_cause = st.selectbox("Event Cause", event_causes)
        
        # Smart default for event type
        planned_causes = {"construction", "public_event", "procession", "vip_movement"}
        default_type_idx = 0 if event_cause in planned_causes else 1
        event_type = st.selectbox("Event Type", ["planned", "unplanned"], index=default_type_idx, help="Auto-selected based on cause, but you can override.")

    with col2:
        st.markdown('<p style="color:#94A3B8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Primary Location</p>', unsafe_allow_html=True)
        corridors_list = sorted([c for c in df["corridor"].unique()
                                 if c != "Non-corridor" and c != "unknown"]) if "corridor" in df.columns else []
        corridor = st.selectbox("Corridor", ["Non-corridor"] + corridors_list)
        
        veh_types = sorted([v for v in df["veh_type"].unique()
                            if v != "unknown" and pd.notna(v)]) if "veh_type" in df.columns else []
        st.selectbox("Vehicle Type", ["unknown"] + veh_types)

    with col3:
        st.markdown('<p style="color:#94A3B8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Specific Location</p>', unsafe_allow_html=True)
        
        # Auto-fill Zone based on corridor
        corridor_df = df[df["corridor"] == corridor]
        top_zones = corridor_df["zone"].value_counts() if not corridor_df.empty and "zone" in df.columns else pd.Series(dtype=int)
        default_zone = top_zones.index[0] if not top_zones.empty else "Unknown"
        
        zones = ["Unknown"] + sorted([z for z in df["zone"].unique() if pd.notna(z)]) if "zone" in df.columns else ["Unknown"]
        zone_idx = zones.index(default_zone) if default_zone in zones else 0
        st.selectbox("Zone", zones, index=zone_idx, help="Auto-filled based on Corridor")

        # Auto-fill Junction based on corridor
        top_junctions = corridor_df["junction"].value_counts() if not corridor_df.empty and "junction" in df.columns else pd.Series(dtype=int)
        valid_junctions = [j for j in top_junctions.index if pd.notna(j) and j != "unknown"]
        default_junction = valid_junctions[0] if valid_junctions else "unknown"
        
        if valid_junctions:
            junctions = ["unknown"] + valid_junctions
        else:
            all_juncs = sorted([j for j in df["junction"].unique() if pd.notna(j) and j != "unknown"]) if "junction" in df.columns else []
            junctions = ["unknown"] + all_juncs[:50]
            
        junc_idx = junctions.index(default_junction) if default_junction in junctions else 0
        junction = st.selectbox("Nearest Junction", junctions, index=junc_idx, help="Filtered & auto-filled based on Corridor")

    st.markdown("---")
    st.markdown("##### 🕒 Step 2: When & Additional Context")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p style="color:#94A3B8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Timing</p>', unsafe_allow_html=True)
        hour = st.slider("Hour (IST)", 0, 23, 9)
        day_of_week = st.selectbox("Day of Week",
                                   ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
        is_weekend = day_idx >= 5
        is_rush = hour in [8, 9, 10, 17, 18, 19, 20]

    with col2:
        st.markdown('<p style="color:#94A3B8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Traffic Conditions</p>', unsafe_allow_html=True)
        # Dynamic rush hour indicator
        if is_rush:
            st.markdown("""
            <div style="background: rgba(244,63,94,0.1); border: 1px solid rgba(244,63,94,0.3);
                        border-radius: 10px; padding: 0.6rem 1rem; text-align: center; margin-bottom: 0.5rem;">
                <span style="color: #F43F5E; font-weight: 700;">🔴 RUSH HOUR</span>
                <span style="color: #94A3B8; font-size: 0.8rem;"> — Higher congestion expected</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);
                        border-radius: 10px; padding: 0.6rem 1rem; text-align: center; margin-bottom: 0.5rem;">
                <span style="color: #10B981; font-weight: 700;">🟢 Off-Peak</span>
                <span style="color: #94A3B8; font-size: 0.8rem;"> — Normal traffic flow</span>
            </div>
            """, unsafe_allow_html=True)

        if is_weekend:
            st.markdown("""
            <div style="background: rgba(139,92,246,0.08); border: 1px solid rgba(139,92,246,0.2);
                        border-radius: 10px; padding: 0.4rem 0.8rem; text-align: center;">
                <span style="color: #8B5CF6; font-weight: 600; font-size: 0.85rem;">📅 Weekend</span>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown('<p style="color:#94A3B8; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">Extra Details</p>', unsafe_allow_html=True)
        requires_closure = st.checkbox("Requires Road Closure?")
        st.text_area("Event Description (optional)", height=68,
                     placeholder="e.g., Major water logging near underpass...")

    st.markdown("")

    # ── PREDICT BUTTON ──
    if st.button("🔮 Predict Impact & Generate Action Plan", type="primary", use_container_width=True):
        st.markdown("")
        st.markdown("---")

        # ────────────────────────────────────────────
        # DATA-DRIVEN PREDICTIONS
        # ────────────────────────────────────────────
        # Severity prediction — from data stats
        base_severity_rate = stats.get("severity_rate", {}).get(event_cause, 0.5)
        corridor_severity_rate = stats.get("corridor_severity_rate", {}).get(corridor, 0.5)

        # Combine cause + corridor signals
        severity_proba = 0.6 * base_severity_rate + 0.25 * corridor_severity_rate
        # Rush hour boost
        if is_rush:
            rush_sev = stats.get("rush_severity_rate", 0.65)
            nonrush_sev = stats.get("nonrush_severity_rate", 0.55)
            rush_boost = max(0, rush_sev - nonrush_sev)
            severity_proba += 0.15 * (rush_boost / max(nonrush_sev, 0.01))
        # Weekend adjustment
        if is_weekend:
            severity_proba *= 0.95
        # Closure boost
        if requires_closure:
            severity_proba = min(0.99, severity_proba + 0.1)

        severity_proba = max(0.01, min(0.99, severity_proba))
        severity_label = "High" if severity_proba > 0.5 else "Low"

        # Duration prediction — from data stats (more accurate than ML model for this dataset)
        avg_dur = stats.get("avg_duration", {}).get(event_cause, 2.0)
        med_dur = stats.get("median_duration", {}).get(event_cause, 1.5)
        p10_dur = stats.get("p10_duration", {}).get(event_cause, 0.3)
        p90_dur = stats.get("p90_duration", {}).get(event_cause, 10.0)
        predicted_duration = med_dur * (1.25 if is_rush else 1.0) * (0.9 if is_weekend else 1.0)
        p10_adj = p10_dur * (1.15 if is_rush else 1.0)
        p90_adj = p90_dur * (1.3 if is_rush else 1.0)

        # Closure prediction — from data stats
        closure_rate = stats.get("closure_rate", {}).get(event_cause, 0.1)
        closure_proba = closure_rate
        if requires_closure:
            closure_proba = max(closure_proba, 0.85)
        if is_rush:
            closure_proba = min(0.99, closure_proba * 1.15)
        closure_needed = requires_closure or closure_proba > 0.5

        # ────────────────────────────────────────────
        # RESOURCE RECOMMENDATION (from ResourceRecommender)
        # ────────────────────────────────────────────
        predictions = {
            "severity": severity_label,
            "severity_proba": severity_proba,
            "duration_hours": {
                "mean": avg_dur,
                "p10": p10_adj,
                "p25": p10_adj * 1.5,
                "p50": predicted_duration,
                "p75": p90_adj * 0.6,
                "p90": p90_adj,
            },
            "closure_needed": closure_needed,
            "closure_proba": closure_proba,
        }

        # Get coordinates for the map
        mask = (df["corridor"] == corridor)
        if junction != "unknown":
            mask = mask & (df["junction"] == junction)
        
        valid_coords = df[mask & df["latitude"].notna() & df["longitude"].notna()]
        if not valid_coords.empty:
            event_lat = float(valid_coords["latitude"].iloc[0])
            event_lon = float(valid_coords["longitude"].iloc[0])
        else:
            event_lat = 12.9716
            event_lon = 77.5946

        event_profile = {
            "event_cause": event_cause,
            "corridor": corridor,
            "is_rush_hour": is_rush,
            "is_weekend": is_weekend,
            "is_planned": event_type == "planned",
            "latitude": event_lat,
            "longitude": event_lon,
            "junction": junction if junction != "unknown" else "unknown",
        }

        # Get full resource plan from ResourceRecommender
        if recommender:
            plan = recommender.recommend(event_profile, predictions)
        else:
            plan = None

        # ══════════════════════════════════════════════
        # DISPLAY RESULTS
        # ══════════════════════════════════════════════

        # ── Phase 1: Prediction Results ──
        st.markdown("##### 🎯 Prediction Results")

        pdf_buffer = generate_action_plan_pdf(event_profile, predictions, plan)
        st.download_button(
            label="📄 Download Detailed PDF Report",
            data=pdf_buffer,
            file_name=f"event_action_plan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="primary"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            severity_color = "#F43F5E" if severity_proba > 0.6 else "#F59E0B" if severity_proba > 0.4 else "#10B981"
            severity_emoji = "🔴" if severity_proba > 0.6 else "🟡" if severity_proba > 0.4 else "🟢"
            st.metric(f"{severity_emoji} Severity", severity_label, f"{severity_proba:.0%} confidence")

        with col2:
            st.metric("⏱️ Est. Duration", f"{predicted_duration:.1f}h",
                       f"Range: {p10_adj:.1f}h – {p90_adj:.1f}h")

        with col3:
            closure_emoji = "🚧" if closure_needed else "✅"
            st.metric(f"{closure_emoji} Road Closure",
                       "REQUIRED" if closure_needed else "NOT NEEDED",
                       f"{closure_proba:.0%} probability")

        with col4:
            if plan:
                alert = plan["alert_level"]
                alert_colors = {"GREEN": "🟢", "BLUE": "🔵", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴"}
                st.metric(f"{alert_colors.get(alert['color'], '⚪')} Alert Level",
                           f"Level {alert['level']}",
                           alert["description"][:40])
            else:
                st.metric("⚠️ Alert Level", "N/A", "Recommender not loaded")

        # Severity gauge chart
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=severity_proba * 100,
                domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
                number={"suffix": "%", "font": {"size": 32, "color": "#F1F5F9", "family": "Inter, sans-serif"}},
                delta={"reference": 50, "increasing": {"color": "#F43F5E"}, "decreasing": {"color": "#10B981"}, "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "tickfont": {"color": "#64748B", "size": 10}},
                    "bar": {"color": severity_color, "thickness": 0.25},
                    "bgcolor": "rgba(255,255,255,0.02)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.05)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16,185,129,0.15)"},
                        {"range": [40, 60], "color": "rgba(245,158,11,0.15)"},
                        {"range": [60, 100], "color": "rgba(244,63,94,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 3},
                        "thickness": 0.8,
                        "value": severity_proba * 100,
                    },
                },
                title={"text": "Severity Score", "font": {"size": 16, "color": "#E2E8F0", "family": "Inter, sans-serif"}},
            ))
            fig.update_layout(
                height=260, 
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="white", family="Inter")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Duration distribution chart
            dur_data = {
                "Quantile": ["P10 (Best)", "P25", "P50 (Median)", "P75", "P90 (Worst)"],
                "Hours": [p10_adj, p10_adj * 1.5, predicted_duration, p90_adj * 0.6, p90_adj],
            }
            fig = go.Figure()
            colors = ["#10B981", "#06B6D4", "#6366F1", "#F59E0B", "#F43F5E"]
            fig.add_trace(go.Bar(
                x=dur_data["Hours"],
                y=dur_data["Quantile"],
                orientation="h",
                marker_color=colors,
                text=[f"{h:.1f}h" for h in dur_data["Hours"]],
                textposition="outside",
                textfont=dict(color="#F1F5F9", size=12),
            ))
            fig.update_layout(
                title=dict(text="Duration Forecast (Quantile Range)", font=dict(size=14, color="#94A3B8")),
                height=250, showlegend=False, **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Phase 2: Suggested Diversions ──
        st.markdown("##### 🔄 Suggested Diversion Routes")

        if plan:
            diversion = plan["diversion"]

            # Advisory banner
            if closure_needed:
                st.markdown(f"""
                <div style="background: rgba(244,63,94,0.08); border: 1px solid rgba(244,63,94,0.25);
                            border-radius: 12px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
                    <span style="color: #F43F5E; font-weight: 700;">⛔ {diversion.get('advisory', 'FULL ROAD CLOSURE')}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.2);
                            border-radius: 12px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
                    <span style="color: #F59E0B; font-weight: 700;">⚠️ {diversion.get('advisory', 'PARTIAL DISRUPTION')}</span>
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="diversion-card">
                    <div class="route-label">🟢 Primary Route</div>
                    <div class="route-path">{diversion.get('primary', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="diversion-card" style="border-color: rgba(139,92,246,0.15);
                     background: linear-gradient(135deg, rgba(139,92,246,0.06) 0%, rgba(99,102,241,0.04) 100%);">
                    <div class="route-label" style="color: #8B5CF6;">🔵 Secondary Route</div>
                    <div class="route-path">{diversion.get('secondary', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="diversion-card" style="border-color: rgba(244,63,94,0.15);
                     background: linear-gradient(135deg, rgba(244,63,94,0.06) 0%, rgba(245,158,11,0.04) 100%);">
                    <div class="route-label" style="color: #F43F5E;">🔴 Avoid Zone</div>
                    <div class="route-path">{diversion.get('avoid', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Diversion Map
            st.markdown("###### 🗺️ Route & Diversion Map")
            
            # Dynamic scaling based on severity
            radius_deg = 0.002 + (severity_proba * 0.015)
            dest_radius = radius_deg * 2.0
            
            # Generate Avoid Zone polygon (circle)
            def get_circle(lat, lon, radius, points=40):
                lats, lons = [], []
                for i in range(points + 1):
                    angle = (i / points) * 2 * math.pi
                    lats.append(lat + radius * math.cos(angle))
                    # Adjust longitude for aspect ratio approx at Bengaluru latitude (12.9 deg -> cos ~0.97)
                    lons.append(lon + (radius / 0.97) * math.sin(angle))
                return lats, lons
            
            avoid_lats, avoid_lons = get_circle(event_lat, event_lon, radius_deg)
            
            map_fig = go.Figure()
            
            # 1. Avoid Zone Area (Polygon fill)
            map_fig.add_trace(go.Scattermap(
                lat=avoid_lats, lon=avoid_lons,
                mode="lines",
                fill="toself",
                fillcolor="rgba(244,63,94,0.15)",
                line=dict(color="rgba(244,63,94,0.8)", width=2),
                name="Affected Zone",
                text=[f"Danger Zone<br>Radius: {radius_deg * 111:.1f} km"] * len(avoid_lats),
                hoverinfo="text"
            ))
            
            # 2. Event Center
            map_fig.add_trace(go.Scattermap(
                lat=[event_lat], lon=[event_lon],
                mode="markers",
                marker=dict(size=14, color="#F43F5E", symbol="circle"),
                name="Incident Center",
                text=[f"Incident: {event_cause}"],
                hoverinfo="text"
            ))
            
            # 3. Dynamic Multi-layer Routes (OSRM integration)
            n_lat, n_lon = event_lat + dest_radius, event_lon
            s_lat, s_lon = event_lat - dest_radius, event_lon
            e_lat, e_lon = event_lat, event_lon + (dest_radius / 0.97)
            w_lat, w_lon = event_lat, event_lon - (dest_radius / 0.97)
            
            routes_to_show = []
            
            # Primary always calculated
            e_lats, e_lons = get_osrm_route(event_lon, event_lat, e_lon, e_lat)
            routes_to_show.append(("Primary Escape (East)", e_lats, e_lons, "#10B981", 4))
            
            if severity_proba >= 0.3:
                # Medium Incident
                w_lats, w_lons = get_osrm_route(event_lon, event_lat, w_lon, w_lat)
                routes_to_show.append(("Secondary Escape (West)", w_lats, w_lons, "#8B5CF6", 4))
                
            if severity_proba >= 0.6:
                # Major Incident
                n_lats, n_lons = get_osrm_route(event_lon, event_lat, n_lon, n_lat)
                routes_to_show.append(("Alternative Corridor (North)", n_lats, n_lons, "#F59E0B", 3))
                
            if severity_proba >= 0.8:
                # Extreme Incident
                s_lats, s_lons = get_osrm_route(event_lon, event_lat, s_lon, s_lat)
                routes_to_show.append(("Emergency Corridor (South)", s_lats, s_lons, "#06B6D4", 3))
                
            for name, r_lats, r_lons, color, width in routes_to_show:
                map_fig.add_trace(go.Scattermap(
                    lat=r_lats, lon=r_lons,
                    mode="lines",
                    line=dict(width=width, color=color),
                    name=name,
                    text=[name] * len(r_lats),
                    hoverinfo="text"
                ))
            
            map_fig.update_layout(
                map_style="carto-darkmatter",
                map=dict(
                    center=dict(lat=event_lat, lon=event_lon),
                    zoom=13 - (severity_proba * 1.5),  # Zoom out for higher severity
                ),
                height=450,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    yanchor="top", y=0.95,
                    xanchor="left", x=0.02,
                    bgcolor="rgba(13,17,23,0.7)",
                    font=dict(color="white"),
                    bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning("Resource recommender not available. Check config.yaml.")

        st.markdown("---")

        # ── Phase 3: Resource Deployment Plan ──
        st.markdown("##### 📋 Resource Deployment Plan")

        if plan:
            manpower = plan["manpower"]
            barricading = plan["barricading"]
            cost = plan["estimated_cost"]

            # ── Top-level metrics ──
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("👮 Total Personnel", manpower["total_personnel"])
            with col2:
                st.metric("🚧 Barricades", barricading["barricade_count"])
            with col3:
                st.metric("📡 Shifts Needed", manpower["shifts_needed"])
            with col4:
                st.metric("🔦 Cones", barricading.get("cones_required", 0))
            with col5:
                st.metric("💰 Est. Cost", cost["total_formatted"])

            st.markdown("")

            # ── Detailed breakdown in tabs ──
            tab1, tab2, tab3, tab4 = st.tabs(["👮 Manpower", "🚧 Barricading", "🧰 Equipment", "📡 Communication"])

            with tab1:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Personnel Breakdown**")
                    personnel_data = pd.DataFrame({
                        "Role": ["Traffic Officers", "Constables", "Support Staff"],
                        "Count": [manpower["traffic_officers"], manpower["constables"], manpower["support_staff"]],
                        "Per Shift": [
                            max(1, manpower["traffic_officers"] // manpower["shifts_needed"]),
                            max(1, manpower["constables"] // manpower["shifts_needed"]),
                            max(1, manpower["support_staff"] // manpower["shifts_needed"]),
                        ],
                    })
                    st.dataframe(personnel_data, use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); border-radius: 8px;
                                padding: 0.8rem; margin-top: 0.5rem; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="color: #64748B; font-size: 0.7rem; text-transform: uppercase;
                                    letter-spacing: 0.5px; margin-bottom: 0.3rem;">Calculation Formula</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; font-family: monospace;">
                            {manpower.get('formula', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Personnel donut chart
                    fig = go.Figure(data=[go.Pie(
                        labels=["Traffic Officers", "Constables", "Support Staff"],
                        values=[manpower["traffic_officers"], manpower["constables"], manpower["support_staff"]],
                        hole=0.55,
                        marker_colors=["#6366F1", "#06B6D4", "#8B5CF6"],
                        textinfo="label+value",
                        textfont=dict(size=11, color="white"),
                    )])
                    fig.update_layout(
                        height=280, showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        annotations=[dict(text=f"<b>{manpower['total_personnel']}</b><br>Total",
                                          x=0.5, y=0.5, font_size=16, showarrow=False,
                                          font_color="white")],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    closure_type_labels = {
                        "full_closure": "🔴 Full Road Closure",
                        "partial_closure": "🟡 Partial Closure",
                        "no_closure": "🟢 No Closure Required",
                    }
                    st.markdown(f"**Closure Type:** {closure_type_labels.get(barricading['type'], barricading['type'])}")

                    barricade_items = {
                        "Barricades": str(barricading["barricade_count"]),
                        "Traffic Cones": str(barricading.get("cones_required", 0)),
                        "Signage Boards": str(barricading.get("signage_boards", 0)),
                    }
                    if "temporary_fencing_meters" in barricading:
                        barricade_items["Temp. Fencing"] = f"{barricading['temporary_fencing_meters']}m"
                    if barricading.get("emergency_lighting"):
                        barricade_items["Emergency Lighting"] = "Required"
                    if "reflective_tape_meters" in barricading:
                        barricade_items["Reflective Tape"] = f"{barricading['reflective_tape_meters']}m"

                    bar_df = pd.DataFrame(list(barricade_items.items()), columns=["Item", "Quantity"])
                    st.dataframe(bar_df, use_container_width=True, hide_index=True)

                with col2:
                    # Barricade visualization — use raw numeric values
                    chart_data = {
                        "Barricades": barricading["barricade_count"],
                        "Cones": barricading.get("cones_required", 0),
                        "Signage": barricading.get("signage_boards", 0),
                    }
                    if "temporary_fencing_meters" in barricading:
                        chart_data["Fencing(m)"] = barricading["temporary_fencing_meters"]

                    fig = go.Figure(data=[go.Bar(
                        x=list(chart_data.keys()),
                        y=list(chart_data.values()),
                        marker_color=["#F43F5E", "#F59E0B", "#06B6D4", "#8B5CF6"][:len(chart_data)],
                        text=list(chart_data.values()),
                        textposition="outside",
                        textfont=dict(color="white"),
                    )])
                    fig.update_layout(height=280, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                equipment = plan["equipment"]
                equip_html = ""
                for e in equipment:
                    equip_html += f'<span class="equip-tag">✅ {e}</span>'
                st.markdown(f'<div style="line-height: 2.2;">{equip_html}</div>', unsafe_allow_html=True)

            with tab4:
                comm = plan["communication"]
                priority_color = {
                    "GREEN": "dot-green", "BLUE": "dot-green",
                    "YELLOW": "dot-yellow", "ORANGE": "dot-yellow",
                    "RED": "dot-red",
                }
                dot_class = priority_color.get(comm.get("priority", "GREEN"), "dot-green")
                for action in comm["actions"]:
                    st.markdown(f"""
                    <div class="comm-action">
                        <span class="dot {dot_class}"></span>
                        <span style="color: #CBD5E1; font-size: 0.9rem;">{action}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Cost Breakdown ──
            st.markdown("##### 💰 Cost Breakdown")
            col1, col2 = st.columns([1, 1])
            with col1:
                cost_items = {
                    "Manpower": cost["manpower_cost_inr"],
                    "Barricading": cost["barricade_cost_inr"],
                    "Equipment": cost["equipment_cost_inr"],
                }
                cost_df = pd.DataFrame({
                    "Category": list(cost_items.keys()),
                    "Amount (₹)": [f"₹{v:,.0f}" for v in cost_items.values()],
                })
                st.dataframe(cost_df, use_container_width=True, hide_index=True)
                st.markdown(f"""
                <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
                            border-radius: 10px; padding: 0.8rem; text-align: center; margin-top: 0.5rem;">
                    <span style="color: #94A3B8; font-size: 0.8rem;">Total Estimated Cost</span><br>
                    <span style="color: #F1F5F9; font-size: 1.5rem; font-weight: 800;">{cost['total_formatted']}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=list(cost_items.keys()),
                    values=list(cost_items.values()),
                    hole=0.5,
                    marker_colors=["#6366F1", "#F59E0B", "#06B6D4"],
                    textinfo="label+percent",
                    textfont=dict(size=11, color="white"),
                )])
                fig.update_layout(
                    height=280, showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    annotations=[dict(text=f"<b>{cost['total_formatted']}</b>",
                                      x=0.5, y=0.5, font_size=14, showarrow=False,
                                      font_color="white")],
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            # Fallback when recommender is not available
            st.warning("⚠️ ResourceRecommender not loaded. Showing basic estimates.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👮 Est. Personnel", "4-8")
            with col2:
                st.metric("🚧 Barricades", "6" if requires_closure else "0")
            with col3:
                st.metric("📡 Shifts", "1")


# ──────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────

def render_model_performance():
    """Model performance metrics."""
    st.markdown("""
    <div class="hero-header">
        <h1>📈 Model Performance Dashboard</h1>
        <p>Comprehensive evaluation metrics for all trained ML models</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics is None:
        st.warning("⚠️ No model metrics found. Run the training pipeline first: `python -m src.pipeline`")
        return

    # ── Severity Model ──
    st.markdown("##### 🎯 Impact Severity Model")
    st.markdown('<p style="color:#64748B; font-size:0.8rem;">Stacked ensemble: LightGBM + XGBoost + CatBoost + Random Forest → Logistic Regression</p>', unsafe_allow_html=True)
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

    # Visual bar chart of metrics
    sev_fig = go.Figure(data=[go.Bar(
        x=["ROC-AUC", "F1", "Precision", "Recall"],
        y=[sev.get('roc_auc', 0), sev.get('f1', 0), sev.get('precision', 0), sev.get('recall', 0)],
        marker_color=["#6366F1", "#8B5CF6", "#06B6D4", "#10B981"],
        text=[f"{v:.4f}" for v in [sev.get('roc_auc', 0), sev.get('f1', 0),
              sev.get('precision', 0), sev.get('recall', 0)]],
        textposition="outside",
        textfont=dict(color="white", size=12),
    )])
    sev_fig.update_layout(height=250, yaxis_range=[0.95, 1.01], **PLOTLY_LAYOUT)
    st.plotly_chart(sev_fig, use_container_width=True)

    st.markdown("---")

    # ── Duration Model ──
    st.markdown("##### ⏱️ Duration Prediction Model")
    st.markdown('<p style="color:#64748B; font-size:0.8rem;">LightGBM quantile regression with log-transformed target</p>', unsafe_allow_html=True)
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

    st.markdown("---")

    # ── Closure Model ──
    st.markdown("##### 🚧 Road Closure Model")
    st.markdown('<p style="color:#64748B; font-size:0.8rem;">CatBoost with balanced class weights</p>', unsafe_allow_html=True)
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

    clos_fig = go.Figure(data=[go.Bar(
        x=["ROC-AUC", "F1", "Precision", "Recall"],
        y=[clos.get('roc_auc', 0), clos.get('f1', 0), clos.get('precision', 0), clos.get('recall', 0)],
        marker_color=["#F43F5E", "#F59E0B", "#EC4899", "#14B8A6"],
        text=[f"{v:.4f}" for v in [clos.get('roc_auc', 0), clos.get('f1', 0),
              clos.get('precision', 0), clos.get('recall', 0)]],
        textposition="outside",
        textfont=dict(color="white", size=12),
    )])
    clos_fig.update_layout(height=250, yaxis_range=[0.95, 1.01], **PLOTLY_LAYOUT)
    st.plotly_chart(clos_fig, use_container_width=True)

    st.markdown("---")

    # ── Training Info ──
    st.markdown("##### 📊 Training Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", f"{metrics.get('training_samples', 0):,}")
    with col2:
        st.metric("Test Samples", f"{metrics.get('test_samples', 0):,}")
    with col3:
        st.metric("Total Features", f"{metrics.get('total_features', 0)}")

    # Feature importance plots
    st.markdown("##### 🏆 Feature Importances")
    plot_path = "outputs/plots/11_model_feature_importance.png"
    if os.path.exists(plot_path):
        st.image(plot_path)
    else:
        st.info("Run EDA visualizations to generate feature importance plots")


# ──────────────────────────────────────────────────────────────
# PAGE: POST-EVENT LEARNING
# ──────────────────────────────────────────────────────────────

def render_post_event_learning():
    """Post-event learning analysis."""
    st.markdown("""
    <div class="hero-header">
        <h1>📋 Post-Event Learning System</h1>
        <p>Continuous improvement through systematic analysis of prediction accuracy and bias</p>
    </div>
    """, unsafe_allow_html=True)

    report = load_post_event_report()
    if report is None:
        st.warning("⚠️ No post-event report found. Run the training pipeline first.")
        return

    # Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Events Analyzed", f"{report.get('total_events', 0):,}")
    with col2:
        acc = report.get("severity_accuracy", 0)
        st.metric("Severity Accuracy", f"{acc:.1%}",
                   "Excellent" if acc > 0.95 else "Good" if acc > 0.85 else "Needs work")
    with col3:
        cacc = report.get("closure_accuracy", 0)
        st.metric("Closure Accuracy", f"{cacc:.1%}",
                   "Excellent" if cacc > 0.95 else "Good" if cacc > 0.85 else "Needs work")

    st.markdown("---")

    # Severity bias analysis
    st.markdown("##### 🎯 Severity Prediction Bias by Event Cause")
    sev_bias = report.get("severity_bias_by_cause", {})
    if sev_bias:
        bias_df = pd.DataFrame(sev_bias).T
        bias_df.index.name = "Event Cause"
        bias_df = bias_df.reset_index()

        # Bias chart
        if "bias" in bias_df.columns:
            fig = go.Figure()
            colors = ["#F43F5E" if b > 0.01 else "#10B981" if b < -0.01 else "#6366F1"
                       for b in bias_df["bias"]]
            fig.add_trace(go.Bar(
                x=bias_df["Event Cause"],
                y=bias_df["bias"],
                marker_color=colors,
                text=[f"{b:+.3f}" for b in bias_df["bias"]],
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(
                height=350,
                title=dict(text="Prediction Bias (Positive = Over-predicting severity)",
                           font=dict(size=13, color="#94A3B8")),
                **PLOTLY_LAYOUT,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(bias_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Duration bias
    st.markdown("##### ⏱️ Duration Prediction Bias")
    col1, col2 = st.columns(2)
    with col1:
        if "duration_bias_hours" in report:
            st.metric("Mean Duration Bias", f"{report['duration_bias_hours']:+.2f}h",
                       "Under-predicting" if report["duration_bias_hours"] < 0 else "Over-predicting")
    with col2:
        if "duration_mae_hours" in report:
            st.metric("Duration MAE", f"{report.get('duration_mae_hours', 0):.2f}h")

    dur_bias = report.get("duration_bias_by_cause", {})
    if dur_bias:
        dur_df = pd.DataFrame(dur_bias).T
        dur_df.index.name = "Event Cause"
        dur_df = dur_df.reset_index()

        if "ratio" in dur_df.columns:
            fig = go.Figure()
            colors = ["#F43F5E" if r < 0.5 else "#F59E0B" if r < 0.8 else "#10B981"
                       for r in dur_df["ratio"]]
            fig.add_trace(go.Bar(
                x=dur_df["Event Cause"],
                y=dur_df["ratio"],
                marker_color=colors,
                text=[f"{r:.2f}x" for r in dur_df["ratio"]],
                textposition="outside",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(
                height=350,
                title=dict(text="Prediction/Actual Ratio (1.0 = perfect, <1 = under-predicting)",
                           font=dict(size=13, color="#94A3B8")),
                **PLOTLY_LAYOUT,
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(dur_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Correction factors
    st.markdown("##### 🔧 Auto-Generated Correction Factors")
    corrections = report.get("correction_factors", {})
    if corrections:
        corr_df = pd.DataFrame(list(corrections.items()), columns=["Event Cause", "Correction Factor"])
        corr_df["Direction"] = corr_df["Correction Factor"].apply(
            lambda x: "↑ Increase resources" if x > 1 else "↓ Decrease resources" if x < 1 else "= No change"
        )
        corr_df["Impact"] = corr_df["Correction Factor"].apply(
            lambda x: f"{(x - 1) * 100:+.0f}% adjustment"
        )

        fig = go.Figure(data=[go.Bar(
            x=corr_df["Event Cause"],
            y=corr_df["Correction Factor"],
            marker_color=["#F43F5E" if f > 1 else "#10B981" if f < 1 else "#6366F1"
                           for f in corr_df["Correction Factor"]],
            text=[f"{f:.1f}x" for f in corr_df["Correction Factor"]],
            textposition="outside",
            textfont=dict(color="white"),
        )])
        fig.update_layout(height=300, **PLOTLY_LAYOUT)
        fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(corr_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
