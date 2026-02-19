#!/usr/bin/env python3
"""
AI-Ready Agentic Data Platform - Streamlit Dashboard
Multi-tab analytics dashboard with AI assistant, ML insights, and system health.
"""

import sys
import os
import json
import socket
import glob
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from agent.agent import run_agent, DataQueryEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_DIR, "data")
BRONZE_PATH = os.path.join(DATA_DIR, "bronze", "ecommerce_events")
SILVER_PATH = os.path.join(DATA_DIR, "silver", "ecommerce_events")
GOLD_PATHS = {
    "revenue_per_hour": os.path.join(DATA_DIR, "gold", "revenue_per_hour"),
    "active_users_per_hour": os.path.join(DATA_DIR, "gold", "active_users_per_hour"),
    "conversion_rate": os.path.join(DATA_DIR, "gold", "conversion_rate"),
}
FEATURES_PATH = os.path.join(DATA_DIR, "features", "user_features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")
MODEL_FILE = os.path.join(MODELS_DIR, "purchase_predictor.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")

# Cohesive color palette
COLORS = {
    "primary": "#6366f1",       # indigo-500
    "primary_light": "#818cf8", # indigo-400
    "secondary": "#06b6d4",     # cyan-500
    "success": "#10b981",       # emerald-500
    "warning": "#f59e0b",       # amber-500
    "danger": "#ef4444",        # red-500
    "surface": "#1e1b4b",       # indigo-950
    "text": "#e2e8f0",          # slate-200
}

PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORS = ["#818cf8", "#06b6d4", "#10b981", "#f59e0b", "#f472b6",
                "#a78bfa", "#22d3ee", "#34d399", "#fbbf24", "#fb923c"]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Agentic Data Platform",
    page_icon="https://em-content.zobj.net/source/twitter/408/rocket_1f680.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Main background and fonts ---- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1145 40%, #1e1b4b 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #0f0c29 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* ---- KPI metric cards ---- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.10) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}
div[data-testid="stMetric"] label {
    color: #a5b4fc !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #e0e7ff !important;
    font-weight: 700 !important;
    font-size: 1.65rem !important;
}

/* ---- Tabs styling ---- */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: #a5b4fc !important;
    border-radius: 8px 8px 0 0;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #e0e7ff !important;
}

/* ---- Headings ---- */
.stApp h1 {
    background: linear-gradient(90deg, #818cf8, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
.stApp h2 {
    color: #c7d2fe !important;
    font-weight: 700 !important;
}
.stApp h3 {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}

/* ---- Buttons ---- */
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s ease;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* ---- Expander ---- */
details {
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    background: rgba(15,12,41,0.4) !important;
}

/* ---- Dataframes ---- */
.stDataFrame {
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    overflow: hidden;
}

/* ---- Info / Warning / Success boxes ---- */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ---- Chat messages ---- */
div[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
}

/* ---- Sidebar brand ---- */
.sidebar-brand {
    text-align: center;
    padding: 10px 0 6px 0;
}
.sidebar-brand h1 {
    font-size: 1.4rem !important;
    margin: 0 !important;
}
.sidebar-brand p {
    color: #94a3b8;
    font-size: 0.78rem;
    margin: 2px 0 0 0;
}

/* ---- Pipeline status cards ---- */
.pipeline-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: background 0.2s;
}
.pipeline-card:hover {
    background: rgba(99,102,241,0.14);
}
.pipeline-icon { font-size: 1.5rem; }
.pipeline-info h4 { margin: 0; color: #e0e7ff; font-size: 0.95rem; }
.pipeline-info p  { margin: 0; color: #94a3b8; font-size: 0.78rem; }

/* ---- Health status row ---- */
.health-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,0.15);
    transition: background 0.2s;
}
.health-row:hover { background: rgba(99,102,241,0.08); }
.health-dot {
    width: 12px; height: 12px; border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 8px currentColor;
}
.health-dot.ok  { background: #10b981; color: #10b981; }
.health-dot.warn { background: #f59e0b; color: #f59e0b; }
.health-dot.err { background: #ef4444; color: #ef4444; }
.health-label { color: #e0e7ff; font-weight: 600; font-size: 0.92rem; }
.health-status { color: #94a3b8; font-size: 0.82rem; margin-left: auto; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_delta_table(path: str) -> pd.DataFrame | None:
    """Load a Delta / parquet directory into a pandas DataFrame."""
    try:
        try:
            from deltalake import DeltaTable
            return DeltaTable(path).to_pandas()
        except Exception:
            pass
        parquet_files = glob.glob(os.path.join(path, "*.parquet"))
        if parquet_files:
            return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    except Exception:
        pass
    return None


def dir_last_modified(path: str) -> str | None:
    try:
        times = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                times.append(os.path.getmtime(os.path.join(root, f)))
        if times:
            return datetime.fromtimestamp(max(times)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return None


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _apply_chart_style(fig, height=380):
    """Apply a consistent dark-themed style to any Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7d2fe", family="Inter, sans-serif"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a5b4fc")),
        xaxis=dict(gridcolor="rgba(99,102,241,0.1)", zerolinecolor="rgba(99,102,241,0.15)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.1)", zerolinecolor="rgba(99,102,241,0.15)"),
    )
    return fig


def auto_chart(df: pd.DataFrame, key_prefix: str = ""):
    """Detect column types and render an appropriate Plotly chart."""
    if df.empty:
        st.info("No data to chart.")
        return

    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
                 or "time" in c.lower() or "date" in c.lower() or "hour" in c.lower()]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if len(df) == 1 and len(numeric_cols) == 1:
        st.metric(label=numeric_cols[0], value=f"{df[numeric_cols[0]].iloc[0]:,.2f}")
        return

    if time_cols and numeric_cols:
        tcol = time_cols[0]
        df_sorted = df.sort_values(tcol)
        y_cols = [c for c in numeric_cols if c != tcol][:4]
        if y_cols:
            fig = px.line(df_sorted, x=tcol, y=y_cols, title=f"Trend over {tcol}",
                          color_discrete_sequence=CHART_COLORS)
            _apply_chart_style(fig)
            st.plotly_chart(fig, key=f"{key_prefix}_line")
            return

    if cat_cols and numeric_cols:
        fig = px.bar(df.head(20), x=cat_cols[0], y=numeric_cols[0],
                     title=f"{numeric_cols[0]} by {cat_cols[0]}",
                     color_discrete_sequence=CHART_COLORS)
        _apply_chart_style(fig)
        st.plotly_chart(fig, key=f"{key_prefix}_bar")
        return

    if numeric_cols:
        fig = px.bar(df.head(20), y=numeric_cols[0],
                     title=f"Distribution of {numeric_cols[0]}",
                     color_discrete_sequence=CHART_COLORS)
        _apply_chart_style(fig)
        st.plotly_chart(fig, key=f"{key_prefix}_fallbar")


def render_health_row(label: str, ok: bool, status_text: str):
    dot_cls = "ok" if ok else "err"
    st.markdown(f"""
    <div class="health-row">
        <div class="health-dot {dot_cls}"></div>
        <span class="health-label">{label}</span>
        <span class="health-status">{status_text}</span>
    </div>""", unsafe_allow_html=True)


def render_health_row_warn(label: str, ok: bool, status_text: str):
    dot_cls = "ok" if ok else "warn"
    st.markdown(f"""
    <div class="health-row">
        <div class="health-dot {dot_cls}"></div>
        <span class="health-label">{label}</span>
        <span class="health-status">{status_text}</span>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h1>AI Data Platform</h1>
        <p>Agentic Analytics Engine</p>
    </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("##### Filters")
    date_range = st.date_input("Date range (optional)", value=[], key="date_filter")

    auto_refresh = st.toggle("Auto-refresh (60 s)", value=False)
    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

    st.divider()
    now_str = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Last refresh: {now_str}")


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_gold_table(name: str) -> pd.DataFrame:
    path = GOLD_PATHS.get(name)
    if path:
        df = load_delta_table(path)
        if df is not None:
            return df
    return pd.DataFrame()

@st.cache_data(ttl=120)
def get_features_table() -> pd.DataFrame:
    df = load_delta_table(FEATURES_PATH)
    return df if df is not None else pd.DataFrame()

@st.cache_data(ttl=120)
def get_bronze_table() -> pd.DataFrame:
    df = load_delta_table(BRONZE_PATH)
    return df if df is not None else pd.DataFrame()

@st.cache_data(ttl=120)
def get_silver_table() -> pd.DataFrame:
    df = load_delta_table(SILVER_PATH)
    return df if df is not None else pd.DataFrame()

@st.cache_data(ttl=300)
def get_ml_metrics() -> dict:
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Business Analytics",
    "AI Assistant",
    "Data Explorer",
    "ML & Predictions",
    "Streaming Monitor",
    "Pipeline Status",
    "System Health",
])


# ===== TAB 0 — OVERVIEW ====================================================
with tabs[0]:
    st.header("Overview")
    st.caption(f"Real-time KPIs from Gold layer  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    revenue_df = get_gold_table("revenue_per_hour")
    users_df = get_gold_table("active_users_per_hour")
    conv_df = get_gold_table("conversion_rate")

    # KPI row
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        total_rev = revenue_df["total_revenue"].sum() if not revenue_df.empty and "total_revenue" in revenue_df.columns else 0
        st.metric("Total Revenue", f"${total_rev:,.2f}")
    with k2:
        active = int(users_df["active_users"].sum()) if not users_df.empty and "active_users" in users_df.columns else 0
        st.metric("Active Users", f"{active:,}")
    with k3:
        conv = conv_df["overall_conversion_rate"].mean() * 100 if not conv_df.empty and "overall_conversion_rate" in conv_df.columns else 0
        st.metric("Conversion Rate", f"{conv:.2f}%")
    with k4:
        aov = revenue_df["avg_order_value"].mean() if not revenue_df.empty and "avg_order_value" in revenue_df.columns else 0
        st.metric("Avg Order Value", f"${aov:,.2f}")

    st.markdown("")  # spacer

    # Revenue trend + Activity summary side by side
    chart_left, chart_right = st.columns([3, 2], gap="medium")

    with chart_left:
        if not revenue_df.empty:
            time_col = next((c for c in revenue_df.columns
                             if "hour" in c.lower() or "time" in c.lower() or "date" in c.lower()), None)
            if time_col:
                df_plot = revenue_df.sort_values(time_col)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_plot[time_col], y=df_plot["total_revenue"],
                    mode="lines", name="Revenue",
                    line=dict(color="#818cf8", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(129,140,248,0.15)",
                ))
                fig.update_layout(title="Revenue Trend")
                _apply_chart_style(fig, height=370)
                st.plotly_chart(fig, key="ov_rev_trend")
            else:
                fig = px.bar(revenue_df, y="total_revenue", title="Revenue per Period",
                             color_discrete_sequence=["#818cf8"])
                _apply_chart_style(fig, height=370)
                st.plotly_chart(fig, key="ov_rev_bar")
        else:
            st.info("Revenue data not available. Run the Spark Gold pipeline first.")

    with chart_right:
        if not users_df.empty:
            time_col = next((c for c in users_df.columns
                             if "hour" in c.lower() or "time" in c.lower() or "date" in c.lower()), None)
            if time_col and "active_users" in users_df.columns:
                df_u = users_df.sort_values(time_col)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_u[time_col], y=df_u["active_users"],
                    name="Active Users",
                    marker=dict(color="#06b6d4", cornerradius=4),
                ))
                fig.update_layout(title="Active Users per Hour")
                _apply_chart_style(fig, height=370)
                st.plotly_chart(fig, key="ov_users_bar")
            else:
                st.metric("Total User Sessions", f"{int(users_df['active_users'].sum()):,}" if "active_users" in users_df.columns else "N/A")
        elif not conv_df.empty:
            fig = px.bar(conv_df, y="overall_conversion_rate", title="Conversion Rate per Period",
                         color_discrete_sequence=["#10b981"])
            _apply_chart_style(fig, height=370)
            st.plotly_chart(fig, key="ov_conv_bar")


# ===== TAB 1 — BUSINESS ANALYTICS ==========================================
with tabs[1]:
    st.header("Business Analytics")
    st.caption("Detailed breakdowns from Silver & Gold layers")

    silver_df = get_silver_table()
    if not silver_df.empty:
        # --- Row 1: Revenue by product + Event distribution ---
        r1_left, r1_right = st.columns([3, 2], gap="medium")

        with r1_left:
            st.subheader("Revenue by Product")
            if "product_id" in silver_df.columns and "price" in silver_df.columns:
                purchase_df = silver_df[silver_df["event_type"] == "purchase"] if "event_type" in silver_df.columns else silver_df
                if not purchase_df.empty:
                    rev_by_product = (
                        purchase_df.groupby("product_id")["price"]
                        .sum().reset_index()
                        .rename(columns={"price": "revenue"})
                        .sort_values("revenue", ascending=False).head(15)
                    )
                    rev_by_product["product_id"] = rev_by_product["product_id"].astype(str)
                    fig = px.bar(rev_by_product, x="product_id", y="revenue",
                                 color="revenue",
                                 color_continuous_scale=["#312e81", "#818cf8", "#06b6d4"],
                                 title="Top 15 Products by Revenue")
                    fig.update_layout(coloraxis_showscale=False)
                    _apply_chart_style(fig)
                    st.plotly_chart(fig, key="ba_rev")
                else:
                    st.info("No purchase data found.")
            else:
                st.info("Product/price columns not found in Silver data.")

        with r1_right:
            st.subheader("Event Distribution")
            if "event_type" in silver_df.columns:
                event_dist = silver_df["event_type"].value_counts().reset_index()
                event_dist.columns = ["event_type", "count"]
                fig = px.pie(event_dist, names="event_type", values="count",
                             title="Event Type Split", hole=0.45,
                             color_discrete_sequence=CHART_COLORS)
                _apply_chart_style(fig)
                st.plotly_chart(fig, key="ba_pie")

        # --- Row 2: Top users + Hourly activity ---
        r2_left, r2_right = st.columns(2, gap="medium")

        with r2_left:
            st.subheader("Top Users by Revenue")
            if "user_id" in silver_df.columns and "price" in silver_df.columns:
                purchase_df = silver_df[silver_df["event_type"] == "purchase"] if "event_type" in silver_df.columns else silver_df
                if not purchase_df.empty:
                    top_users = (
                        purchase_df.groupby("user_id")["price"]
                        .sum().reset_index()
                        .rename(columns={"price": "total_spent"})
                        .sort_values("total_spent", ascending=False).head(10)
                    )
                    top_users["user_id"] = top_users["user_id"].astype(str)
                    fig = px.bar(top_users, x="total_spent", y="user_id", orientation="h",
                                 title="Top 10 Spenders",
                                 color="total_spent",
                                 color_continuous_scale=["#312e81", "#818cf8", "#06b6d4"])
                    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    _apply_chart_style(fig, height=400)
                    st.plotly_chart(fig, key="ba_users")

        with r2_right:
            st.subheader("Hourly Activity")
            time_col = next((c for c in silver_df.columns
                             if "hour" in c.lower() or "timestamp" in c.lower() or "time" in c.lower()), None)
            if time_col:
                try:
                    tmp = silver_df.copy()
                    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                    tmp["hour_of_day"] = tmp[time_col].dt.hour
                    hourly = tmp.groupby("hour_of_day").size().reset_index(name="event_count")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=hourly["hour_of_day"], y=hourly["event_count"],
                        marker=dict(
                            color=hourly["event_count"],
                            colorscale=[[0, "#312e81"], [0.5, "#818cf8"], [1, "#06b6d4"]],
                            cornerradius=4,
                        ),
                    ))
                    fig.update_layout(title="Events by Hour of Day",
                                      xaxis_title="Hour", yaxis_title="Events")
                    _apply_chart_style(fig, height=400)
                    st.plotly_chart(fig, key="ba_hourly")
                except Exception:
                    st.info("Could not parse timestamps for hourly chart.")
    else:
        st.info("Silver layer data not available. Run the Spark pipeline first.")


# ===== TAB 2 — AI ASSISTANT ================================================
with tabs[2]:
    st.header("AI Assistant")
    st.caption("Ask natural-language questions — the agent writes SQL, queries Gold tables, and explains results.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sql"):
                    with st.expander("Generated SQL"):
                        st.code(msg["sql"], language="sql")
                if msg.get("data") is not None and not msg["data"].empty:
                    with st.expander("Result Data"):
                        st.dataframe(msg["data"].head(100), hide_index=True)
                    auto_chart(msg["data"], key_prefix=f"chat_{msg.get('idx', 0)}")

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "What is the total revenue?",
            "Show me conversion rates by hour",
            "How many active users do we have?",
            "Which hour had the most purchases?",
        ]
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if cols[i].button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state["_prefill"] = q
                st.rerun()

    prefill = st.session_state.pop("_prefill", None)
    user_input = st.chat_input("Ask a question about your data...")
    active_input = prefill or user_input

    if active_input:
        st.session_state.chat_history.append({"role": "user", "content": active_input})
        with st.chat_message("user"):
            st.markdown(active_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_agent(active_input)
                    summary = result.get("summary", "No response generated.")
                    sql = result.get("sql", "")
                    data = result.get("data", pd.DataFrame())

                    st.markdown(summary)
                    if sql:
                        with st.expander("Generated SQL"):
                            st.code(sql, language="sql")
                    if data is not None and not data.empty:
                        with st.expander("Result Data"):
                            st.dataframe(data.head(100), hide_index=True)
                        auto_chart(data, key_prefix=f"chat_{len(st.session_state.chat_history)}")

                    st.session_state.chat_history.append({
                        "role": "assistant", "content": summary,
                        "sql": sql, "data": data,
                        "idx": len(st.session_state.chat_history),
                    })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# ===== TAB 3 — DATA EXPLORER ===============================================
with tabs[3]:
    st.header("Data Explorer")
    st.caption("Browse any layer of the lakehouse interactively")

    layer_options = {
        "Bronze": BRONZE_PATH,
        "Silver": SILVER_PATH,
        "Gold - Revenue per Hour": GOLD_PATHS["revenue_per_hour"],
        "Gold - Active Users per Hour": GOLD_PATHS["active_users_per_hour"],
        "Gold - Conversion Rate": GOLD_PATHS["conversion_rate"],
        "Feature Table (user_features)": FEATURES_PATH,
    }

    selected_layer = st.selectbox("Select table", list(layer_options.keys()))
    selected_path = layer_options[selected_layer]

    df = load_delta_table(selected_path)
    if df is not None and not df.empty:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.metric("Rows", f"{len(df):,}")
        with m2:
            st.metric("Columns", f"{len(df.columns)}")
        with m3:
            null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            st.metric("Null %", f"{null_pct:.1f}%")
        with m4:
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory", f"{mem_mb:.1f} MB")

        # Schema
        st.subheader("Schema")
        schema_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Null Count": [int(df[c].isnull().sum()) for c in df.columns],
            "Non-Null Count": [int(df[c].notnull().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(schema_df, hide_index=True)

        st.subheader("Data Preview (first 100 rows)")
        st.dataframe(df.head(100), hide_index=True)
    else:
        st.warning(f"No data found at `{selected_path}`. Run the pipeline to generate data.")


# ===== TAB 4 — ML & PREDICTIONS ============================================
with tabs[4]:
    st.header("ML & Predictions")
    st.caption("Purchase-predictor model performance and live inference")

    metrics = get_ml_metrics()
    if metrics:
        # Gauge-style metrics
        st.subheader("Model Performance")
        g1, g2, g3, g4 = st.columns(4, gap="medium")
        metric_pairs = [
            ("Accuracy", metrics.get("accuracy", 0)),
            ("Precision", metrics.get("precision", 0)),
            ("Recall", metrics.get("recall", 0)),
            ("F1 Score", metrics.get("f1_score", 0)),
        ]
        for col, (label, val) in zip([g1, g2, g3, g4], metric_pairs):
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={"text": label, "font": {"color": "#a5b4fc", "size": 14}},
                    number={"font": {"color": "#e0e7ff", "size": 28}, "valueformat": ".3f"},
                    gauge=dict(
                        axis=dict(range=[0, 1], tickcolor="#4b5563"),
                        bar=dict(color="#818cf8"),
                        bgcolor="rgba(30,27,75,0.6)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 0.5], color="rgba(239,68,68,0.15)"),
                            dict(range=[0.5, 0.8], color="rgba(245,158,11,0.15)"),
                            dict(range=[0.8, 1], color="rgba(16,185,129,0.15)"),
                        ],
                    ),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c7d2fe"),
                    height=200, margin=dict(l=20, r=20, t=50, b=10),
                )
                st.plotly_chart(fig, key=f"gauge_{label}")

        extra1, extra2 = st.columns(2, gap="medium")
        with extra1:
            st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
        with extra2:
            st.metric("Training Samples", f"{metrics.get('n_samples', 'N/A')}")

        # Confusion matrix + Feature importance side by side
        cm_col, fi_col = st.columns(2, gap="medium")

        with cm_col:
            cm = metrics.get("confusion_matrix")
            if cm:
                st.subheader("Confusion Matrix")
                cm_array = np.array(cm)
                fig = go.Figure(data=go.Heatmap(
                    z=cm_array,
                    x=["Pred Negative", "Pred Positive"],
                    y=["True Negative", "True Positive"],
                    text=cm_array, texttemplate="%{text}",
                    colorscale=[[0, "#1e1b4b"], [1, "#818cf8"]],
                    showscale=False,
                ))
                fig.update_layout(title="Confusion Matrix")
                _apply_chart_style(fig, height=360)
                st.plotly_chart(fig, key="ml_cm")

        with fi_col:
            fi = metrics.get("feature_importance")
            if fi:
                st.subheader("Feature Importance")
                fi_df = pd.DataFrame(
                    sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True),
                    columns=["Feature", "Importance"],
                )
                fi_df["abs"] = fi_df["Importance"].abs()
                fi_df["color"] = fi_df["Importance"].apply(lambda v: "#10b981" if v >= 0 else "#ef4444")
                fig = go.Figure(go.Bar(
                    x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                    marker=dict(color=fi_df["color"], cornerradius=4),
                ))
                fig.update_layout(title="Coefficient Importance", yaxis=dict(autorange="reversed"))
                _apply_chart_style(fig, height=420)
                st.plotly_chart(fig, key="ml_fi")

        # Training info
        with st.expander("Training metadata"):
            st.write(f"**Training date:** {metrics.get('training_date', 'N/A')}")
            st.write(f"**Features ({metrics.get('n_features', '?')}):** {', '.join(metrics.get('feature_names', []))}")

        # Live prediction
        st.divider()
        st.subheader("Live Prediction")
        user_id_input = st.text_input("Enter user_id to predict purchase likelihood", placeholder="e.g. 42")
        if user_id_input:
            try:
                import joblib
                features_df = get_features_table()
                if features_df.empty:
                    st.warning("Feature table not available.")
                elif not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
                    st.warning("Model or scaler file not found.")
                else:
                    user_id_val = int(user_id_input)
                    user_row = features_df[features_df["user_id"] == user_id_val]
                    if user_row.empty:
                        st.warning(f"User {user_id_val} not found in feature table.")
                    else:
                        model = joblib.load(MODEL_FILE)
                        scaler = joblib.load(SCALER_FILE)
                        feature_names = metrics.get("feature_names", [])
                        X = user_row[feature_names].values
                        X_scaled = scaler.transform(X)
                        pred = model.predict(X_scaled)[0]
                        proba = model.predict_proba(X_scaled)[0]
                        conf = max(proba) * 100
                        if pred == 1:
                            st.success(f"**User {user_id_val}:** Likely Purchaser (confidence {conf:.1f}%)")
                        else:
                            st.warning(f"**User {user_id_val}:** Unlikely Purchaser (confidence {conf:.1f}%)")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("ML metrics file not found. Train the model first.")


# ===== TAB 5 — STREAMING MONITOR ===========================================
with tabs[5]:
    st.header("Streaming Monitor")
    st.caption("Real-time event activity from the Bronze layer")

    bronze_df = get_bronze_table()
    if not bronze_df.empty:
        time_col = next((c for c in bronze_df.columns
                         if "timestamp" in c.lower() or "time" in c.lower()), None)

        # Summary KPIs
        sk1, sk2, sk3 = st.columns(3, gap="medium")
        with sk1:
            st.metric("Total Events", f"{len(bronze_df):,}")
        with sk2:
            n_types = bronze_df["event_type"].nunique() if "event_type" in bronze_df.columns else 0
            st.metric("Event Types", f"{n_types}")
        with sk3:
            n_users = bronze_df["user_id"].nunique() if "user_id" in bronze_df.columns else 0
            st.metric("Unique Users", f"{n_users:,}")

        # Charts row
        c_left, c_right = st.columns([3, 2], gap="medium")

        with c_left:
            st.subheader("Events per Minute")
            if time_col:
                try:
                    tmp = bronze_df.copy()
                    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                    tmp = tmp.dropna(subset=[time_col])
                    tmp["minute"] = tmp[time_col].dt.floor("min")
                    epm = tmp.groupby("minute").size().reset_index(name="event_count").sort_values("minute")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epm["minute"], y=epm["event_count"],
                        mode="lines", line=dict(color="#06b6d4", width=2.5),
                        fill="tozeroy", fillcolor="rgba(6,182,212,0.12)",
                    ))
                    fig.update_layout(title="Event Throughput")
                    _apply_chart_style(fig, height=350)
                    st.plotly_chart(fig, key="sm_epm")
                except Exception:
                    st.info("Could not parse timestamps for events-per-minute chart.")
            else:
                st.info("No timestamp column found in Bronze data.")

        with c_right:
            st.subheader("Event Type Distribution")
            if "event_type" in bronze_df.columns:
                evt_dist = bronze_df["event_type"].value_counts().reset_index()
                evt_dist.columns = ["event_type", "count"]
                fig = px.pie(evt_dist, names="event_type", values="count",
                             hole=0.45, color_discrete_sequence=CHART_COLORS,
                             title="Type Breakdown")
                _apply_chart_style(fig, height=350)
                st.plotly_chart(fig, key="sm_pie")

        # Latest events
        st.subheader("Latest Events")
        if time_col:
            try:
                tmp = bronze_df.copy()
                tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                latest = tmp.sort_values(time_col, ascending=False).head(20)
                st.dataframe(latest, hide_index=True)
            except Exception:
                st.dataframe(bronze_df.tail(20), hide_index=True)
        else:
            st.dataframe(bronze_df.tail(20), hide_index=True)
    else:
        st.info("No Bronze layer data found. Start the streaming pipeline and event generator first.")


# ===== TAB 6 — PIPELINE STATUS =============================================
with tabs[6]:
    st.header("Pipeline Status")
    st.caption("Data layer existence and freshness — no jobs are executed")

    layers = {
        "Bronze Layer":              (BRONZE_PATH, "Raw events from Kafka"),
        "Silver Layer":              (SILVER_PATH, "Cleaned & validated events"),
        "Gold - Revenue per Hour":   (GOLD_PATHS["revenue_per_hour"], "Hourly revenue aggregations"),
        "Gold - Active Users/Hour":  (GOLD_PATHS["active_users_per_hour"], "Hourly user activity"),
        "Gold - Conversion Rate":    (GOLD_PATHS["conversion_rate"], "Funnel conversion metrics"),
        "Feature Table":             (FEATURES_PATH, "Per-user ML feature vectors"),
        "ML Model Artifacts":        (MODELS_DIR, "Trained model, scaler, metrics"),
    }

    for name, (path, desc) in layers.items():
        exists = False
        if os.path.isdir(path):
            exists = any(f for f in os.listdir(path) if not f.startswith("."))
        elif os.path.isfile(path):
            exists = True

        last_mod = dir_last_modified(path) if os.path.isdir(path) else None
        if not last_mod and os.path.isfile(path):
            last_mod = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")

        icon = "checkmark" if exists else "cross"
        status_label = f"Last modified: {last_mod}" if exists and last_mod else ("Available" if exists else "Missing")
        dot = "ok" if exists else "err"
        emoji = "&#x2705;" if exists else "&#x274C;"

        st.markdown(f"""
        <div class="pipeline-card">
            <div class="pipeline-icon">{emoji}</div>
            <div class="pipeline-info">
                <h4>{name}</h4>
                <p>{desc} &mdash; {status_label}</p>
            </div>
        </div>""", unsafe_allow_html=True)


# ===== TAB 7 — SYSTEM HEALTH ===============================================
with tabs[7]:
    st.header("System Health")
    st.caption("Infrastructure component checks (read-only, no restarts)")

    kafka_ok = check_port("localhost", 9092)
    render_health_row("Kafka Broker (port 9092)", kafka_ok,
                      "Running" if kafka_ok else "Not reachable")

    spark_ok = False
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        spark_ok = spark is not None
    except Exception:
        pass
    render_health_row("Spark Session", spark_ok,
                      "Available" if spark_ok else "Not available")

    ollama_ok = check_port("localhost", 11434)
    render_health_row("Ollama LLM Service (port 11434)", ollama_ok,
                      "Running" if ollama_ok else "Not reachable")

    airflow_ok = check_port("localhost", 8793)
    render_health_row_warn("Airflow Scheduler (port 8793)", airflow_ok,
                           "Running" if airflow_ok else "Not detected (optional)")

    delta_ok = False
    try:
        import deltalake
        delta_ok = True
    except ImportError:
        pass
    render_health_row("Delta Lake library", delta_ok,
                      "Installed" if delta_ok else "Not installed")

    # Data directories summary
    st.markdown("")
    st.subheader("Data Directories")
    dir_items = [
        ("Bronze", BRONZE_PATH), ("Silver", SILVER_PATH),
        ("Gold", os.path.join(DATA_DIR, "gold")),
        ("Features", FEATURES_PATH), ("Models", MODELS_DIR),
    ]
    cols = st.columns(len(dir_items), gap="medium")
    for col, (name, path) in zip(cols, dir_items):
        exists = os.path.isdir(path)
        file_count = 0
        if exists:
            for _root, _dirs, files in os.walk(path):
                file_count += len(files)
        with col:
            st.metric(name, f"{file_count} files" if exists else "missing")
