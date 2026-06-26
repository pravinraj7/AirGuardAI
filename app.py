"""
AirGuard AI — Streamlit Frontend
Main Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Path setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Page config — MUST be first Streamlit call
st.set_page_config(
    page_title="AirGuard AI — Air Quality Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend.data_loader import (
    load_city_day, load_pollution_sources, load_city_coordinates,
    load_traffic, get_latest_city_aqi, get_aqi_health_message,
    get_aqi_color, _classify_aqi
)
from backend.ml_models import train_models, forecast_aqi, simulate_digital_twin, load_models
from backend.ai_recommendations import (
    generate_recommendations, generate_full_action_plan,
    get_source_ranking, get_priority, get_smart_action_presets,
    format_population, SMART_ACTION_PRESETS
)
from backend.visualizations import (
    plot_aqi_trend, plot_pollutant_breakdown, plot_top_polluted_cities,
    plot_monthly_heatmap, plot_pollution_sources, plot_multi_city_comparison,
    plot_forecast_chart, plot_digital_twin_gauge, create_folium_heatmap,
    plot_aqi_distribution, plot_model_comparison
)
from streamlit_folium import st_folium

# ─────────────────────────────────────────────
#  Inject custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap');
/* ── Global ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
background: #0a0e1a !important;
font-family: 'Inter', sans-serif !important;
color: #e8eaf0 !important;
}
[data-testid="stSidebar"] {
background: linear-gradient(180deg, #0d1126 0%, #111827 100%) !important;
border-right: 1px solid rgba(0, 212, 255, 0.15) !important;
}
/* ── Header ── */
.airguard-header {
background: linear-gradient(135deg, #0a0e1a 0%, #0d1b3e 50%, #0a0e1a 100%);
border: 1px solid rgba(0, 212, 255, 0.2);
border-radius: 20px;
padding: 30px 40px;
margin-bottom: 24px;
position: relative;
overflow: hidden;
}
.airguard-header::before {
content: '';
position: absolute;
top: -50%;
left: -50%;
width: 200%;
height: 200%;
background: radial-gradient(circle at 30% 30%, rgba(0,212,255,0.08) 0%, transparent 50%),
radial-gradient(circle at 70% 70%, rgba(99,102,241,0.08) 0%, transparent 50%);
pointer-events: none;
}
.airguard-header h1 {
font-family: 'Outfit', sans-serif;
font-size: 2.8rem;
font-weight: 800;
background: linear-gradient(135deg, #00d4ff, #6366f1, #a855f7);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin: 0 0 8px 0;
line-height: 1.1;
}
.airguard-header p {
color: rgba(232,234,240,0.65);
font-size: 1.05rem;
margin: 0;
}
/* ── Metric Cards ── */
.metric-card {
background: linear-gradient(135deg, rgba(13,27,62,0.8) 0%, rgba(17,24,39,0.9) 100%);
border: 1px solid rgba(0, 212, 255, 0.18);
border-radius: 16px;
padding: 20px 24px;
text-align: center;
transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
backdrop-filter: blur(10px);
}
.metric-card:hover {
transform: translateY(-3px);
border-color: rgba(0, 212, 255, 0.45);
box-shadow: 0 12px 40px rgba(0, 212, 255, 0.1);
}
.metric-card .metric-label {
font-size: 0.78rem;
font-weight: 600;
color: rgba(232,234,240,0.5);
letter-spacing: 0.08em;
text-transform: uppercase;
margin-bottom: 8px;
}
.metric-card .metric-value {
font-family: 'Outfit', sans-serif;
font-size: 2.2rem;
font-weight: 800;
line-height: 1;
}
.metric-card .metric-sub {
font-size: 0.82rem;
color: rgba(232,234,240,0.5);
margin-top: 6px;
}
/* ── AQI Badge ── */
.aqi-badge {
display: inline-block;
padding: 4px 14px;
border-radius: 20px;
font-size: 0.85rem;
font-weight: 700;
letter-spacing: 0.05em;
}
/* ── Section Headers ── */
.section-header {
font-family: 'Outfit', sans-serif;
font-size: 1.5rem;
font-weight: 700;
color: #e8eaf0;
margin: 8px 0 20px 0;
padding-bottom: 12px;
border-bottom: 2px solid rgba(0, 212, 255, 0.25);
}
/* ── Health Advisory Card ── */
.advisory-card {
border-radius: 16px;
padding: 24px;
border: 2px solid;
margin: 12px 0;
}
/* ── Glassmorphism panels ── */
.glass-panel {
background: rgba(13, 27, 62, 0.5);
border: 1px solid rgba(0, 212, 255, 0.15);
border-radius: 16px;
padding: 20px;
backdrop-filter: blur(10px);
}
/* ── Sidebar Nav ── */
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }
/* ── Tab styling ── */
[data-baseweb="tab-list"] {
background: rgba(13,27,62,0.5) !important;
border-radius: 12px !important;
padding: 4px !important;
gap: 4px !important;
}
[data-baseweb="tab"] {
border-radius: 8px !important;
color: rgba(232,234,240,0.6) !important;
font-weight: 500 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
background: rgba(0,212,255,0.15) !important;
color: #00d4ff !important;
}
/* ── Streamlit element overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label {
color: rgba(232,234,240,0.75) !important;
font-weight: 500 !important;
}
.stButton > button {
background: linear-gradient(135deg, #00d4ff, #6366f1) !important;
color: white !important;
border: none !important;
border-radius: 10px !important;
font-weight: 600 !important;
font-size: 0.95rem !important;
padding: 0.6rem 1.5rem !important;
transition: all 0.2s !important;
}
.stButton > button:hover {
transform: translateY(-1px) !important;
box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}
/* ── Progress / Alerts ── */
.stAlert { border-radius: 12px !important; }
/* ── Dividers ── */
hr { border-color: rgba(0, 212, 255, 0.12) !important; }
/* ── AI Recommendation Card ── */
.ai-card {
background: linear-gradient(135deg, rgba(13,27,62,0.9) 0%, rgba(17,24,39,0.95) 100%);
border-radius: 20px;
padding: 28px;
border: 1px solid rgba(0, 212, 255, 0.2);
position: relative;
overflow: hidden;
}
.ai-card::before {
content: '';
position: absolute; top: 0; left: 0; right: 0; height: 3px;
background: linear-gradient(90deg, #00d4ff, #6366f1, #a855f7);
}
.ai-priority-badge {
display: inline-flex;
align-items: center;
gap: 6px;
padding: 6px 16px;
border-radius: 30px;
font-size: 0.82rem;
font-weight: 800;
letter-spacing: 0.1em;
text-transform: uppercase;
border: 2px solid;
}
.ai-action-item {
display: flex;
align-items: flex-start;
gap: 10px;
padding: 10px 14px;
background: rgba(255,255,255,0.04);
border-radius: 10px;
margin-bottom: 8px;
border-left: 3px solid rgba(0,212,255,0.4);
font-size: 0.9rem;
color: rgba(232,234,240,0.88);
transition: background 0.2s;
}
.ai-action-item:hover { background: rgba(0,212,255,0.06); }
.ai-xai-item {
display: flex;
align-items: flex-start;
gap: 10px;
padding: 8px 12px;
color: rgba(232,234,240,0.75);
font-size: 0.88rem;
border-bottom: 1px solid rgba(255,255,255,0.05);
}
.ai-impact-bar {
height: 8px;
border-radius: 4px;
background: rgba(255,255,255,0.06);
margin-top: 6px;
overflow: hidden;
}
.ai-preset-btn {
border-radius: 12px !important;
font-size: 0.85rem !important;
font-weight: 600 !important;
padding: 0.5rem 1rem !important;
}
/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Data Loading (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_data():
    df_day = load_city_day()
    df_coords = load_city_coordinates()
    df_traffic = load_traffic()
    df_sources = load_pollution_sources()
    return df_day, df_coords, df_traffic, df_sources


@st.cache_resource(show_spinner=False)
def train_and_cache_models(df_hash):
    df = load_city_day()
    results, features, scaler = train_models(df)
    return results


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='text-align:center; padding: 16px 0 24px 0;'>
<div style='font-size:2.8rem; margin-bottom:6px;'>🌬️</div>
<div style='font-family: Outfit, sans-serif; font-size:1.4rem; font-weight:800; 
background: linear-gradient(135deg,#00d4ff,#6366f1);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
AirGuard AI
</div>
<div style='font-size:0.75rem; color:rgba(232,234,240,0.4); margin-top:4px;'>
Air Quality Intelligence
</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.selectbox(
        "📍 Navigate",
        [
            "🏠 Dashboard",
            "📈 AQI Forecasting",
            "🏭 Pollution Sources",
            "💊 Health Advisory",
            "🏙️ City Comparison",
            "🔧 Digital Twin",
            "🗺️ Interactive Map",
            "🤖 AI Action Engine",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem; color:rgba(232,234,240,0.4); padding: 8px 0;'>
<div style='margin-bottom:8px; font-weight:600; color:rgba(232,234,240,0.6);'>📊 Data Coverage</div>
<div>🏙️ 26 Indian Cities</div>
<div>📅 2015–2020</div>
<div>🔬 14 Pollutants</div>
<div>📈 29,531 Records</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.72rem; color:rgba(232,234,240,0.3); text-align:center;'>
Built for Hackathon 2026<br>
AirGuard AI Team 🚀
</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load Data
# ─────────────────────────────────────────────
with st.spinner("⚙️ Loading air quality data..."):
    df_day, df_coords, df_traffic, df_sources = load_all_data()

CITIES = sorted(df_day["City"].unique().tolist())


# ─────────────────────────────────────────────
#  AI Panel Helpers (reusable across pages)
# ─────────────────────────────────────────────
def render_ai_panel(city: str, aqi: float, df_sources, forecast_aqi=None, sim_result=None, compact=False):
    """Render the full AI Action Recommendation panel."""
    rec = generate_recommendations(city, aqi, df_sources, forecast_aqi, sim_result)
    p = rec["priority"]
    impact = rec["impact"]

    st.markdown(f"""
<div class="ai-card">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px; flex-wrap:wrap; gap:10px;">
<div>
<div style="font-family:Outfit,sans-serif; font-size:1.5rem; font-weight:800; color:#00d4ff; margin-bottom:4px;">
\U0001f916 AI Action Recommendation
</div>
<div style="color:rgba(232,234,240,0.55); font-size:0.9rem;">City: <b style="color:rgba(232,234,240,0.85);">{city}</b> &nbsp;|&nbsp; Current AQI: <b style="color:{p['color']};">{aqi:.0f}</b></div>
</div>
<div class="ai-priority-badge" style="color:{p['color']}; border-color:{p['color']}; background:{p['bg']};">
{p['icon']} {p['level']}
</div>
</div>
<div style="color:rgba(232,234,240,0.5); font-size:0.82rem; margin-bottom:16px; padding:8px 12px; background:rgba(255,255,255,0.03); border-radius:8px;">
{p['description']}
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px;">
<div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:14px;">
<div style="font-size:0.75rem; color:rgba(232,234,240,0.45); font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Primary Source</div>
<div style="font-size:1.3rem; font-weight:800; color:#ff6b6b;">{rec['primary_source']['display']}</div>
<div style="font-size:0.88rem; color:rgba(232,234,240,0.6);">{rec['primary_source']['pct']:.1f}% of total pollution</div>
</div>
<div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:14px;">
<div style="font-size:0.75rem; color:rgba(232,234,240,0.45); font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Secondary Source</div>
<div style="font-size:1.3rem; font-weight:800; color:#ffd93d;">{rec['secondary_source']['display']}</div>
<div style="font-size:0.88rem; color:rgba(232,234,240,0.6);">{rec['secondary_source']['pct']:.1f}% of total pollution</div>
</div>
</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_actions, col_impact = st.columns([3, 2])

    with col_actions:
        st.markdown('<div style="font-family:Outfit,sans-serif;font-size:1.15rem;font-weight:700;color:#e8eaf0;margin-bottom:14px;">&#x2705; Recommended Actions</div>', unsafe_allow_html=True)
        actions_html = ""
        for action in rec["actions"]:
            actions_html += f'<div class="ai-action-item"><span style="color:#00d4ff;font-size:1.1rem;flex-shrink:0;">&#x2713;</span><span>{action}</span></div>'
        st.markdown(actions_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🧠 Why this recommendation? (Explainable AI)", expanded=not compact):
            xai_html = '<div style="background:rgba(13,27,62,0.5);border-radius:12px;padding:4px 0;">'
            for i, reason in enumerate(rec["explainability"]):
                xai_html += f'<div class="ai-xai-item"><span style="color:#6366f1;font-weight:700;flex-shrink:0;">{i+1}.</span><span>{reason}</span></div>'
            xai_html += "</div>"
            st.markdown(xai_html, unsafe_allow_html=True)

    with col_impact:
        st.markdown('<div style="font-family:Outfit,sans-serif;font-size:1.15rem;font-weight:700;color:#e8eaf0;margin-bottom:14px;">&#x1f4ca; Expected Impact</div>', unsafe_allow_html=True)
        for label, val, color in [
            ("AQI Reduction", impact["aqi_reduction_pct"], "#00d4ff"),
            ("PM2.5 Reduction", impact["pm25_reduction_pct"], "#6bcb77"),
            ("Health Risk Reduction", impact["health_risk_reduction_pct"], "#c77dff"),
        ]:
            bar_pct = min(100, val * 2.2)
            st.markdown(f"""
<div style="margin-bottom:14px;">
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
<span style="font-size:0.82rem;color:rgba(232,234,240,0.6);">{label}</span>
<span style="font-weight:800;color:{color};">{val:.1f}%</span>
</div>
<div class="ai-impact-bar"><div style="width:{bar_pct:.0f}%;height:100%;background:linear-gradient(90deg,{color}aa,{color});border-radius:4px;"></div></div>
</div>""", unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;margin-top:8px;text-align:center;">
<div style="font-size:0.75rem;color:rgba(232,234,240,0.45);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Population Benefit</div>
<div style="font-size:2rem;font-weight:800;color:#00d4ff;">{format_population(impact['population_benefit'])}</div>
<div style="font-size:0.8rem;color:rgba(232,234,240,0.45);">citizens see health improvement</div>
</div>
<div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;margin-top:10px;text-align:center;">
<div style="font-size:0.75rem;color:rgba(232,234,240,0.45);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Predicted AQI After Action</div>
<div style="font-size:1.8rem;font-weight:800;color:{get_aqi_color(impact['predicted_aqi'])};">{impact['predicted_aqi']:.0f}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Outfit,sans-serif;font-size:1.15rem;font-weight:700;color:#e8eaf0;margin-bottom:12px;">&#x26a1; One-Click Smart Actions</div>', unsafe_allow_html=True)
    presets = get_smart_action_presets()
    btn_cols = st.columns(len(presets))
    for i, (name, preset) in enumerate(presets.items()):
        with btn_cols[i]:
            is_rec = (name == rec["recommended_preset"])
            label = f"{preset['icon']} {name}" + (" ★" if is_rec else "")
            if st.button(label, key=f"preset_{name}_{city}_{aqi}", width="stretch",
                         help=preset["description"] + (" — Recommended!" if is_rec else "")):
                st.session_state["dt_traffic"] = preset["traffic_reduction"]
                st.session_state["dt_construction"] = preset["construction_reduction"]
                st.session_state["dt_industry"] = preset["industrial_reduction"]
                st.session_state["dt_green"] = preset["green_cover_increase"]
                st.session_state["dt_city"] = city
                st.session_state["dt_preset_applied"] = name
                st.success(f"✅ **{name}** preset applied! Switch to **Digital Twin** page to run the simulation.")


def render_ai_compact_widget(city: str, aqi: float, df_sources):
    """Compact AI recommendation widget for Dashboard."""
    rec = generate_recommendations(city, aqi, df_sources)
    p = rec["priority"]
    impact = rec["impact"]
    st.markdown(f"""
<div class="ai-card" style="padding:20px;">
<div style="font-family:Outfit,sans-serif;font-size:1rem;font-weight:800;color:#00d4ff;margin-bottom:12px;">\U0001f916 AI Recommendation</div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
<div class="ai-priority-badge" style="color:{p['color']};border-color:{p['color']};background:{p['bg']};font-size:0.72rem;padding:4px 12px;">{p['icon']} {p['level']} PRIORITY</div>
<div style="font-size:0.8rem;color:rgba(232,234,240,0.5);">AQI {aqi:.0f}</div>
</div>
<div style="font-size:0.85rem;color:rgba(232,234,240,0.75);margin-bottom:10px;">
<b style="color:#ff6b6b;">Primary:</b> {rec['primary_source']['display']} ({rec['primary_source']['pct']:.0f}%)
</div>
<div class="ai-action-item" style="font-size:0.82rem;margin-bottom:10px;">
<span style="color:#00d4ff;">&#x2713;</span><span>{rec['actions'][0]}</span>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;">
<div style="font-size:0.78rem;color:rgba(232,234,240,0.45);">Expected AQI improvement</div>
<div style="font-weight:800;color:#00d4ff;">{impact['aqi_reduction_pct']:.1f}%</div>
</div>
</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Page: Dashboard
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":

    # Header
    st.markdown("""
<div class="airguard-header">
<h1>🌬️ AirGuard AI</h1>
<p>Real-time Air Quality Intelligence Platform • 26 Indian Cities • ML-Powered Forecasting</p>
</div>
    """, unsafe_allow_html=True)

    # Controls row
    col_city, col_days, col_spacer = st.columns([2, 2, 4])
    with col_city:
        selected_city = st.selectbox("Select City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
    with col_days:
        trend_days = st.selectbox("Trend Period", [30, 60, 90, 180, 365], index=2)

    # ── KPI Cards ──
    city_df = df_day[df_day["City"] == selected_city].dropna(subset=["AQI"]).sort_values("Date")
    latest = city_df.iloc[-1] if not city_df.empty else None

    if latest is not None:
        current_aqi = float(latest["AQI"])
        bucket = str(latest.get("AQI_Bucket", _classify_aqi(current_aqi)))
        aqi_color = get_aqi_color(current_aqi)
        
        # Week comparison
        week_ago = city_df.iloc[-7]["AQI"] if len(city_df) >= 7 else current_aqi
        delta_aqi = current_aqi - float(week_ago)
        
        avg_pm25 = float(city_df["PM2.5"].tail(30).mean()) if "PM2.5" in city_df else 0
        avg_pm10 = float(city_df["PM10"].tail(30).mean()) if "PM10" in city_df else 0
        avg_no2 = float(city_df["NO2"].tail(30).mean()) if "NO2" in city_df else 0
        worst_month = city_df.groupby(city_df["Date"].dt.month)["AQI"].mean().idxmax()
        
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Current AQI</div>
<div class="metric-value" style="color:{aqi_color};">{current_aqi:.0f}</div>
<div class="metric-sub" style="color:{aqi_color};">● {bucket}</div>
</div>""", unsafe_allow_html=True)
        with c2:
            delta_color = "#ff6b6b" if delta_aqi > 0 else "#6bcb77"
            delta_icon = "▲" if delta_aqi > 0 else "▼"
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Week Change</div>
<div class="metric-value" style="color:{delta_color};">{delta_icon} {abs(delta_aqi):.0f}</div>
<div class="metric-sub">vs 7 days ago</div>
</div>""", unsafe_allow_html=True)
        with c3:
            pm25_color = get_aqi_color(avg_pm25 * 4)
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Avg PM2.5 (30d)</div>
<div class="metric-value" style="color:{pm25_color};">{avg_pm25:.1f}</div>
<div class="metric-sub">μg/m³</div>
</div>""", unsafe_allow_html=True)
        with c4:
            pm10_color = get_aqi_color(avg_pm10 * 2)
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Avg PM10 (30d)</div>
<div class="metric-value" style="color:{pm10_color};">{avg_pm10:.1f}</div>
<div class="metric-sub">μg/m³</div>
</div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Worst Month</div>
<div class="metric-value" style="color:#c77dff;">{month_names.get(worst_month,'N/A')}</div>
<div class="metric-sub">historically</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AQI Trend ──
    st.markdown('<div class="section-header">📈 AQI Trend</div>', unsafe_allow_html=True)
    fig_trend = plot_aqi_trend(df_day, selected_city, trend_days)
    st.plotly_chart(fig_trend, width="stretch")

    # ── Pollutant Breakdown + Monthly Heatmap ──
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-header">🔬 Pollutant Profile</div>', unsafe_allow_html=True)
        fig_radar = plot_pollutant_breakdown(df_day, selected_city)
        st.plotly_chart(fig_radar, width="stretch")
    with col_right:
        st.markdown('<div class="section-header">📅 Monthly Heatmap</div>', unsafe_allow_html=True)
        fig_heat = plot_monthly_heatmap(df_day, selected_city)
        st.plotly_chart(fig_heat, width="stretch")

    # ── AQI Distribution + Top Polluted ──
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.markdown('<div class="section-header">📊 AQI Distribution</div>', unsafe_allow_html=True)
        fig_dist = plot_aqi_distribution(df_day, selected_city)
        st.plotly_chart(fig_dist, width="stretch")
    with col_right2:
        st.markdown('<div class="section-header">🏆 Top Polluted Cities</div>', unsafe_allow_html=True)
        fig_top = plot_top_polluted_cities(df_day, 10)
        st.plotly_chart(fig_top, width="stretch")

    # ── AI Recommendation Compact Widget ──
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Action Summary</div>', unsafe_allow_html=True)
    if latest is not None:
        ai_col1, ai_col2, ai_col3 = st.columns([2, 3, 2])
        with ai_col1:
            render_ai_compact_widget(selected_city, current_aqi, df_sources)
        with ai_col2:
            # Source attribution mini bar chart
            ranked = get_source_ranking(df_sources, selected_city)
            src_colors = {"Traffic": "#ff6b6b", "Construction": "#ffd93d", "Industry": "#6bcb77", "Waste Burning": "#4d96ff"}
            src_html = '<div class="glass-panel">'
            src_html += '<div style="font-size:0.85rem;font-weight:700;color:rgba(232,234,240,0.7);margin-bottom:12px;">Pollution Source Attribution</div>'
            for src_key, src_name, pct in ranked:
                color = src_colors.get(src_name, "#fff")
                src_html += f"""
<div style="margin-bottom:10px;">
<div style="display:flex;justify-content:space-between;margin-bottom:4px;">
<span style="font-size:0.82rem;color:rgba(232,234,240,0.7);">{src_name}</span>
<span style="font-weight:700;color:{color};">{pct:.1f}%</span>
</div>
<div style="background:rgba(255,255,255,0.06);border-radius:4px;height:6px;overflow:hidden;">
<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></div>
</div>
</div>"""
            src_html += "</div>"
            st.markdown(src_html, unsafe_allow_html=True)
        with ai_col3:
            priority_info = get_priority(current_aqi)
            st.markdown(f"""
<div class="glass-panel" style="text-align:center;">
<div style="font-size:0.78rem;color:rgba(232,234,240,0.45);font-weight:600;text-transform:uppercase;margin-bottom:8px;">Current Alert Level</div>
<div style="font-size:3rem;margin-bottom:8px;">{priority_info['icon']}</div>
<div class="ai-priority-badge" style="color:{priority_info['color']};border-color:{priority_info['color']};background:{priority_info['bg']};margin:0 auto 12px auto;display:inline-flex;">
{priority_info['level']}
</div>
<div style="font-size:0.78rem;color:rgba(232,234,240,0.5);margin-top:8px;">{priority_info['description']}</div>
</div>
            """, unsafe_allow_html=True)



# ─────────────────────────────────────────────
#  Page: AQI Forecasting
# ─────────────────────────────────────────────
elif page == "📈 AQI Forecasting":
    st.markdown('<div class="airguard-header"><h1>📈 AQI Forecasting</h1><p>XGBoost & Random Forest — 24h, 48h, 72h Predictions</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 4])
    with col1:
        fc_city = st.selectbox("Select City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
        
        st.markdown("### 🤖 Model Training")
        train_btn = st.button("🚀 Train ML Models", width="stretch")
        
        if train_btn:
            with st.spinner("Training XGBoost, Random Forest & Gradient Boosting..."):
                try:
                    results, features, scaler = train_models(df_day)
                    st.session_state["model_results"] = results
                    st.session_state["features"] = features
                    st.success("✅ Models trained!")
                    for mname, mres in results.items():
                        st.markdown(f"**{mname}**: R²={mres['r2']:.3f} | MAE={mres['mae']:.1f}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("### 🔮 Generate Forecast")
        fc_btn = st.button("⚡ Forecast AQI", width="stretch")

    with col2:
        city_data = df_day[df_day["City"] == fc_city].dropna(subset=["AQI"]).sort_values("Date")
        if not city_data.empty:
            current_aqi = float(city_data.iloc[-1]["AQI"])
            aqi_color = get_aqi_color(current_aqi)
            
            st.markdown(f"""
<div class="glass-panel" style="text-align:center; margin-bottom:20px;">
<div style='font-size:0.85rem; color:rgba(232,234,240,0.5); margin-bottom:8px;'>
CURRENT AQI — {fc_city.upper()}
</div>
<div style='font-family:Outfit,sans-serif; font-size:4rem; font-weight:800; color:{aqi_color};'>
{current_aqi:.0f}
</div>
<div style='color:{aqi_color}; font-weight:600;'>
● {_classify_aqi(current_aqi)}
</div>
</div>
            """, unsafe_allow_html=True)

        if fc_btn or "forecast_results" in st.session_state:
            if fc_btn:
                with st.spinner(f"Generating forecast for {fc_city}..."):
                    try:
                        forecasts = forecast_aqi(df_day, fc_city)
                        st.session_state["forecast_results"] = forecasts
                        st.session_state["forecast_city"] = fc_city
                    except Exception as e:
                        st.error(f"Forecast error: {e}")
                        forecasts = {}
            else:
                forecasts = st.session_state.get("forecast_results", {})

            if forecasts:
                fig_fc = plot_forecast_chart(forecasts, fc_city)
                st.plotly_chart(fig_fc, width="stretch")

                st.markdown("### 📋 Forecast Table")
                rows = []
                for model, h_dict in forecasts.items():
                    for h, val in h_dict.items():
                        rows.append({
                            "Model": model,
                            "Horizon": f"{h}h",
                            "Predicted AQI": round(val, 1),
                            "Status": _classify_aqi(val),
                            "Advisory": get_aqi_health_message(val)["icon"] + " " + get_aqi_health_message(val)["level"],
                        })
                
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    hide_index=True,
                )

    # Model comparison
    if "model_results" in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
        fig_compare = plot_model_comparison(st.session_state["model_results"])
        st.plotly_chart(fig_compare, width="stretch")
        
        # Feature importance
        st.markdown('<div class="section-header">🔑 Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
        if "XGBoost" in st.session_state["model_results"]:
            fi = st.session_state["model_results"]["XGBoost"]["feature_importance"]
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            
            import plotly.express as px
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                           color="Importance", color_continuous_scale=["#6366f1", "#00d4ff"],
                           template="plotly_dark")
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_fi, width="stretch")


# ─────────────────────────────────────────────
#  Page: Pollution Sources
# ─────────────────────────────────────────────
elif page == "🏭 Pollution Sources":
    st.markdown('<div class="airguard-header"><h1>🏭 Pollution Source Attribution</h1><p>Traffic • Construction • Industry • Waste Burning</p></div>', unsafe_allow_html=True)

    ps_city = st.selectbox("Select City", sorted(df_sources["city"].unique().tolist()) if "city" in df_sources.columns else CITIES)

    col1, col2 = st.columns(2)
    with col1:
        fig_ps = plot_pollution_sources(df_sources, ps_city)
        st.plotly_chart(fig_ps, width="stretch")
    
    with col2:
        city_src = df_sources[df_sources["city"].str.lower() == ps_city.lower()] if "city" in df_sources.columns else df_sources
        if city_src.empty:
            city_src = df_sources
        
        avg = city_src[["traffic", "construction", "industry", "waste_burning"]].mean().round(1)
        
        st.markdown('<div class="section-header">📊 Source Breakdown</div>', unsafe_allow_html=True)
        
        sources_info = [
            ("🚗 Traffic", avg["traffic"], "#ff6b6b", "Vehicle emissions, exhaust gases, brake dust"),
            ("🏗️ Construction", avg["construction"], "#ffd93d", "Dust, cement particles, machinery emissions"),
            ("🏭 Industry", avg["industry"], "#6bcb77", "Factory smoke, chemical emissions, SO₂"),
            ("🔥 Waste Burning", avg["waste_burning"], "#4d96ff", "Open burning, agricultural residues, plastics"),
        ]
        
        for name, val, color, desc in sources_info:
            st.markdown(f"""
<div class="glass-panel" style="margin-bottom: 12px;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<span style="font-weight:600; font-size:1rem;">{name}</span>
<span style="font-size:1.4rem; font-weight:800; color:{color};">{val:.1f}%</span>
</div>
<div style="background:rgba(255,255,255,0.05); border-radius:6px; height:8px; overflow:hidden;">
<div style="background:{color}; width:{val}%; height:100%; border-radius:6px;"></div>
</div>
<div style="font-size:0.78rem; color:rgba(232,234,240,0.45); margin-top:6px;">{desc}</div>
</div>
            """, unsafe_allow_html=True)

    # Trend over time
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Source Trends Over Time</div>', unsafe_allow_html=True)
    
    import plotly.graph_objects as go
    
    city_trend = df_sources[df_sources["city"].str.lower() == ps_city.lower()] if "city" in df_sources.columns else df_sources
    if not city_trend.empty:
        fig_trend = go.Figure()
        colors = {"traffic": "#ff6b6b", "construction": "#ffd93d", "industry": "#6bcb77", "waste_burning": "#4d96ff"}
        labels = {"traffic": "Traffic", "construction": "Construction", "industry": "Industry", "waste_burning": "Waste Burning"}
        
        for src in ["traffic", "construction", "industry", "waste_burning"]:
            if src in city_trend.columns:
                fig_trend.add_trace(go.Scatter(
                    x=city_trend["date"], y=city_trend[src],
                    mode="lines+markers", name=labels[src],
                    line=dict(color=colors[src], width=2),
                    marker=dict(size=5),
                ))
        
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date", yaxis_title="Contribution (%)",
            legend=dict(font=dict(color="white")),
            height=400,
        )
        st.plotly_chart(fig_trend, width="stretch")

    # ── AI Recommendation Panel ──
    st.markdown("---")
    if not city_trend.empty:
        city_aqi_ps = df_day[df_day["City"] == ps_city].dropna(subset=["AQI"]).sort_values("Date")
        base_aqi_ps = float(city_aqi_ps.iloc[-1]["AQI"]) if not city_aqi_ps.empty else 200.0
        render_ai_panel(ps_city, base_aqi_ps, df_sources)


# ─────────────────────────────────────────────
#  Page: Health Advisory
# ─────────────────────────────────────────────
elif page == "💊 Health Advisory":
    st.markdown('<div class="airguard-header"><h1>💊 Citizen Health Advisory</h1><p>Personalized AQI-based health recommendations for all citizens</p></div>', unsafe_allow_html=True)

    ha_city = st.selectbox("Select Your City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
    
    city_df_ha = df_day[df_day["City"] == ha_city].dropna(subset=["AQI"]).sort_values("Date")
    
    if not city_df_ha.empty:
        latest_ha = city_df_ha.iloc[-1]
        aqi = float(latest_ha["AQI"])
        advisory = get_aqi_health_message(aqi)
        aqi_color = advisory["color"]
        
        # Main advisory card
        st.markdown(f"""
<div class="advisory-card" style="border-color:{aqi_color}; background:linear-gradient(135deg, rgba(13,27,62,0.8), rgba(17,24,39,0.9));">
<div style="display:flex; align-items:center; gap:20px; margin-bottom:16px;">
<div style="font-size:3.5rem;">{advisory.get('icon', '😐')}</div>
<div>
<div style="font-family:Outfit,sans-serif; font-size:2rem; font-weight:800; color:{aqi_color};">
{advisory['level']} — AQI {aqi:.0f}
</div>
<div style="color:rgba(232,234,240,0.7); font-size:1.05rem; margin-top:4px;">
{ha_city} • As of {str(latest_ha['Date'])[:10]}
</div>
</div>
</div>
<div style="font-size:1.05rem; color:rgba(232,234,240,0.85); margin-bottom:12px;">
{advisory['message']}
</div>
<div style="background:rgba(255,255,255,0.06); border-radius:12px; padding:16px; border-left:4px solid {aqi_color};">
<div style="font-weight:600; color:{aqi_color}; margin-bottom:6px;">💡 Recommendation</div>
<div style="color:rgba(232,234,240,0.85);">{advisory['recommendation']}</div>
</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Group-specific advisory
        st.markdown('<div class="section-header">👥 Group-Specific Guidance</div>', unsafe_allow_html=True)
        
        groups = [
            ("👶 Children", aqi, [
                (50, "🟢 Safe for outdoor play"),
                (100, "🟡 Monitor outdoor time"),
                (200, "🟠 Avoid prolonged outdoor play"),
                (300, "🔴 Stay indoors"),
                (500, "🚨 Emergency — stay indoors"),
            ]),
            ("👴 Elderly", aqi, [
                (50, "🟢 Normal activities"),
                (100, "🟡 Limit strenuous activity"),
                (200, "🟠 Wear mask outdoors"),
                (300, "🔴 Avoid going out"),
                (500, "🚨 Seek medical attention"),
            ]),
            ("🫁 Asthma/Respiratory", aqi, [
                (50, "🟢 Minimal risk"),
                (100, "🟡 Carry inhaler"),
                (200, "🟠 Wear N95 mask"),
                (300, "🔴 Use air purifier, stay in"),
                (500, "🚨 Medical emergency possible"),
            ]),
            ("🏃 Athletes", aqi, [
                (50, "🟢 Train freely"),
                (100, "🟡 Avoid high-intensity outdoors"),
                (200, "🟠 Move workouts indoors"),
                (300, "🔴 No outdoor training"),
                (500, "🚨 Cancel all outdoor sports"),
            ]),
        ]

        cols = st.columns(4)
        for idx, (group, aqi_val, thresholds) in enumerate(groups):
            advice = "Moderate caution"
            for thresh, msg in thresholds:
                if aqi_val <= thresh:
                    advice = msg
                    break
            
            with cols[idx]:
                st.markdown(f"""
<div class="glass-panel" style="text-align:center; height:160px; display:flex; flex-direction:column; justify-content:center;">
<div style="font-size:1.8rem; margin-bottom:8px;">{group.split()[0]}</div>
<div style="font-weight:600; margin-bottom:6px; color:rgba(232,234,240,0.8);">{group.split(None, 1)[1] if len(group.split()) > 1 else group}</div>
<div style="font-size:0.85rem; color:rgba(232,234,240,0.65);">{advice}</div>
</div>
                """, unsafe_allow_html=True)

        # Pollutant details
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔬 Pollutant Levels</div>', unsafe_allow_html=True)
        
        pollutants = {
            "PM2.5": (float(latest_ha["PM2.5"]) if not pd.isna(latest_ha["PM2.5"]) else 0, "μg/m³", 60),
            "PM10": (float(latest_ha["PM10"]) if not pd.isna(latest_ha["PM10"]) else 0, "μg/m³", 100),
            "NO2": (float(latest_ha["NO2"]) if not pd.isna(latest_ha["NO2"]) else 0, "μg/m³", 80),
            "SO2": (float(latest_ha["SO2"]) if not pd.isna(latest_ha["SO2"]) else 0, "μg/m³", 80),
            "O3": (float(latest_ha["O3"]) if not pd.isna(latest_ha["O3"]) else 0, "μg/m³", 100),
            "CO": (float(latest_ha["CO"]) if not pd.isna(latest_ha["CO"]) else 0, "mg/m³", 10),
        }

        pcols = st.columns(6)
        for idx, (poll, (val, unit, safe_thresh)) in enumerate(pollutants.items()):
            pct = min(100, (val / safe_thresh) * 100)
            pcolor = "#00e400" if val <= safe_thresh * 0.5 else "#ffff00" if val <= safe_thresh else "#ff0000"
            
            with pcols[idx]:
                st.markdown(f"""
<div class="glass-panel" style="text-align:center;">
<div style="font-size:0.8rem; font-weight:600; color:rgba(232,234,240,0.5); margin-bottom:6px;">{poll}</div>
<div style="font-size:1.6rem; font-weight:800; color:{pcolor};">{val:.1f}</div>
<div style="font-size:0.72rem; color:rgba(232,234,240,0.4);">{unit}</div>
<div style="background:rgba(255,255,255,0.08); border-radius:4px; height:5px; margin-top:8px; overflow:hidden;">
<div style="background:{pcolor}; width:{pct:.0f}%; height:100%;"></div>
</div>
</div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Page: City Comparison
# ─────────────────────────────────────────────
elif page == "🏙️ City Comparison":
    st.markdown('<div class="airguard-header"><h1>🏙️ Multi-City Comparison</h1><p>Compare air quality across Indian cities</p></div>', unsafe_allow_html=True)

    comp_cities = st.multiselect(
        "Select Cities to Compare",
        CITIES,
        default=["Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"][:5],
        max_selections=8,
    )

    if len(comp_cities) >= 2:
        fig_comp = plot_multi_city_comparison(df_day, comp_cities)
        st.plotly_chart(fig_comp, width="stretch")

        # Latest stats table
        st.markdown('<div class="section-header">📊 Latest Stats Comparison</div>', unsafe_allow_html=True)
        
        stats_rows = []
        for city in comp_cities:
            city_df_c = df_day[df_day["City"] == city].dropna(subset=["AQI"]).sort_values("Date")
            if not city_df_c.empty:
                last = city_df_c.iloc[-1]
                avg30 = city_df_c.tail(30)["AQI"].mean()
                stats_rows.append({
                    "City": city,
                    "Latest AQI": round(float(last["AQI"]), 1),
                    "Status": _classify_aqi(float(last["AQI"])),
                    "30-Day Avg": round(avg30, 1),
                    "Max AQI (All)": round(float(city_df_c["AQI"].max()), 1),
                    "Min AQI (All)": round(float(city_df_c["AQI"].min()), 1),
                })
        
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            
            def color_aqi(val):
                try:
                    color = get_aqi_color(float(val))
                    return f"color: {color}; font-weight: bold;"
                except:
                    return ""
            
            styled = stats_df.style.applymap(color_aqi, subset=["Latest AQI", "30-Day Avg", "Max AQI (All)"])
            st.dataframe(styled, width="stretch", hide_index=True)

        # AQI Distribution per city
        st.markdown('<div class="section-header">📦 AQI Distribution Comparison</div>', unsafe_allow_html=True)
        
        import plotly.graph_objects as go
        fig_box = go.Figure()
        palette = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#c77dff", "#ff9f1c", "#ff6b6b"]
        
        for i, city in enumerate(comp_cities):
            city_aqi_data = df_day[df_day["City"] == city]["AQI"].dropna()
            fig_box.add_trace(go.Box(
                y=city_aqi_data,
                name=city,
                marker_color=palette[i % len(palette)],
                boxmean=True,
            ))
        
        fig_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="AQI",
            legend=dict(font=dict(color="white")),
            height=450,
        )
        st.plotly_chart(fig_box, width="stretch")
    else:
        st.info("👆 Please select at least 2 cities to compare")


# ─────────────────────────────────────────────
#  Page: Digital Twin
# ─────────────────────────────────────────────
elif page == "🔧 Digital Twin":
    st.markdown('<div class="airguard-header"><h1>🔧 Digital Twin Simulator</h1><p>Simulate intervention scenarios and predict AQI improvement</p></div>', unsafe_allow_html=True)

    col_ctrl, col_result = st.columns([2, 3])
    
    with col_ctrl:
        st.markdown('<div class="section-header">⚙️ Simulation Controls</div>', unsafe_allow_html=True)
        
        dt_city = st.selectbox("Select City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0)
        
        city_aqi_dt = df_day[df_day["City"] == dt_city].dropna(subset=["AQI"]).sort_values("Date")
        base_aqi = float(city_aqi_dt.iloc[-1]["AQI"]) if not city_aqi_dt.empty else 200.0
        
        st.markdown(f"""
<div class="glass-panel" style="text-align:center; margin-bottom:20px;">
<div style="font-size:0.8rem; color:rgba(232,234,240,0.5); margin-bottom:4px;">CURRENT AQI</div>
<div style="font-size:2.5rem; font-weight:800; color:{get_aqi_color(base_aqi)};">{base_aqi:.0f}</div>
</div>
        """, unsafe_allow_html=True)

        # Check if a preset was triggered from another page
        preset_dt_traffic = st.session_state.get("dt_traffic", 20)
        preset_dt_construction = st.session_state.get("dt_construction", 15)
        preset_dt_industry = st.session_state.get("dt_industry", 10)
        preset_dt_green = st.session_state.get("dt_green", 10)
        preset_name = st.session_state.pop("dt_preset_applied", None)
        
        if preset_name:
            st.info(f"⚡ Smart Action Preset loaded: **{preset_name}**")

        traffic_red = st.slider("🚗 Traffic Reduction %", 0, 100, preset_dt_traffic, 5,
                                 help="Reduce vehicle traffic through restrictions or EV adoption")
        construction_red = st.slider("🏗️ Construction Reduction %", 0, 100, preset_dt_construction, 5,
                                      help="Pause/reduce construction activities")
        industrial_red = st.slider("🏭 Industrial Reduction %", 0, 100, preset_dt_industry, 5,
                                    help="Industrial emission controls & shutdowns")
        green_cover = st.slider("🌳 Green Cover Increase %", 0, 100, preset_dt_green, 5,
                                 help="Tree planting, urban forests, green spaces")

        simulate_btn = st.button("🔮 Run Simulation", width="stretch")

    with col_result:
        if simulate_btn or True:  # Always show results
            sim_result = simulate_digital_twin(
                base_aqi=base_aqi,
                traffic_reduction=traffic_red,
                construction_reduction=construction_red,
                industrial_reduction=industrial_red,
                green_cover_increase=green_cover,
            )

            predicted_aqi = sim_result["predicted_aqi"]
            improvement_pct = sim_result["aqi_improvement_pct"]
            health_red = sim_result["health_risk_reduction_pct"]
            
            st.markdown('<div class="section-header">📊 Simulation Results</div>', unsafe_allow_html=True)
            
            # Gauge charts
            fig_gauge = plot_digital_twin_gauge(base_aqi, predicted_aqi)
            st.plotly_chart(fig_gauge, width="stretch")
            
            # Result metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
<div class="metric-card">
<div class="metric-label">AQI Improvement</div>
<div class="metric-value" style="color:#6bcb77;">↓ {improvement_pct:.1f}%</div>
<div class="metric-sub">{base_aqi:.0f} → {predicted_aqi:.0f}</div>
</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Health Risk Reduction</div>
<div class="metric-value" style="color:#00d4ff;">↓ {health_red:.0f}%</div>
<div class="metric-sub">population benefit</div>
</div>""", unsafe_allow_html=True)
            with c3:
                new_bucket = _classify_aqi(predicted_aqi)
                bucket_color = get_aqi_color(predicted_aqi)
                st.markdown(f"""
<div class="metric-card">
<div class="metric-label">New AQI Status</div>
<div class="metric-value" style="color:{bucket_color}; font-size:1.6rem;">{new_bucket}</div>
<div class="metric-sub" style="color:{bucket_color};">● predicted</div>
</div>""", unsafe_allow_html=True)
            
            # Breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📉 AQI Reduction Breakdown</div>', unsafe_allow_html=True)
            
            breakdown = sim_result["breakdown"]
            breakdown_items = [
                ("🚗 Traffic Control", breakdown["traffic_reduction_aqi"], "#ff6b6b"),
                ("🏗️ Construction Halt", breakdown["construction_reduction_aqi"], "#ffd93d"),
                ("🏭 Industrial Controls", breakdown["industrial_reduction_aqi"], "#6bcb77"),
                ("🌳 Green Cover", breakdown["green_absorption_aqi"], "#4d96ff"),
            ]
            
            total_breakdown = sum(v for _, v, _ in breakdown_items)
            
            for name, val, color in breakdown_items:
                pct = (val / max(total_breakdown, 1)) * 100
                st.markdown(f"""
<div style="margin-bottom:12px;">
<div style="display:flex; justify-content:space-between; margin-bottom:4px;">
<span style="font-size:0.9rem; color:rgba(232,234,240,0.8);">{name}</span>
<span style="font-weight:700; color:{color};">-{val:.1f} AQI</span>
</div>
<div style="background:rgba(255,255,255,0.06); border-radius:6px; height:10px; overflow:hidden;">
<div style="background:linear-gradient(90deg,{color},{color}aa); width:{pct:.1f}%; height:100%; border-radius:6px;"></div>
</div>
</div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            render_ai_panel(dt_city, base_aqi, df_sources, sim_result=sim_result, compact=True)


# ─────────────────────────────────────────────
#  Page: Interactive Map
# ─────────────────────────────────────────────
elif page == "🗺️ Interactive Map":
    st.markdown('<div class="airguard-header"><h1>🗺️ Interactive AQI Map</h1><p>Real-time heatmap across Indian cities using Folium</p></div>', unsafe_allow_html=True)

    # Legend
    st.markdown("""
<div class="glass-panel" style="display:flex; gap:24px; flex-wrap:wrap; margin-bottom:20px; align-items:center;">
<span style="font-weight:600; color:rgba(232,234,240,0.7);">AQI Legend:</span>
<span>🟢 <b style="color:#00e400;">0-50</b> Good</span>
<span>🟡 <b style="color:#9acd32;">51-100</b> Satisfactory</span>
<span>🟡 <b style="color:#ffff00;">101-200</b> Moderate</span>
<span>🟠 <b style="color:#ff7e00;">201-300</b> Poor</span>
<span>🔴 <b style="color:#ff0000;">301-400</b> Very Poor</span>
<span>🟣 <b style="color:#8f3f97;">401+</b> Severe</span>
</div>
    """, unsafe_allow_html=True)

    with st.spinner("🗺️ Rendering map..."):
        try:
            folium_map = create_folium_heatmap(df_day, df_coords)
            st_folium(folium_map, width="stretch")
        except Exception as e:
            st.error(f"Map error: {e}")
            st.info("Showing data table instead:")
            
            latest_all = get_latest_city_aqi(df_day)
            merged_table = latest_all.merge(
                df_coords.rename(columns={"city": "City"} if "city" in df_coords.columns else {}),
                on="City", how="inner"
            )
            st.dataframe(merged_table[["City", "AQI", "AQI_Bucket", "latitude", "longitude"]].round(2),
                        width="stretch", hide_index=True)

    # Data table below map
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 City AQI Summary</div>', unsafe_allow_html=True)
    
    latest_summary = get_latest_city_aqi(df_day)
    st.dataframe(
        latest_summary[["City", "AQI", "AQI_Bucket", "PM2.5", "PM10", "NO2", "SO2", "O3"]]
                     .sort_values("AQI", ascending=False)
                     .round(1)
                     .reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


# ─────────────────────────────────────────────
#  Page: AI Action Engine
# ─────────────────────────────────────────────
elif page == "🤖 AI Action Engine":
    st.markdown("""
<div class="airguard-header">
<h1>🤖 AI Action Recommendation Engine</h1>
<p>AI Decision Support System for City Administrators • Explainable AI</p>
</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 5])
    
    with col1:
        st.markdown('<div class="section-header">🏙️ City Selection</div>', unsafe_allow_html=True)
        ai_city = st.selectbox("Select Target City", CITIES, index=CITIES.index("Delhi") if "Delhi" in CITIES else 0, key="ai_page_city")
        
        city_aqi_ai = df_day[df_day["City"] == ai_city].dropna(subset=["AQI"]).sort_values("Date")
        base_aqi_ai = float(city_aqi_ai.iloc[-1]["AQI"]) if not city_aqi_ai.empty else 200.0
        
        st.markdown(f"""
<div class="glass-panel" style="text-align:center; margin-bottom:20px;">
<div style="font-size:0.8rem; color:rgba(232,234,240,0.5); margin-bottom:4px;">CURRENT AQI</div>
<div style="font-size:3rem; font-weight:800; color:{get_aqi_color(base_aqi_ai)};">{base_aqi_ai:.0f}</div>
</div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🚀 Demo Mode</div>', unsafe_allow_html=True)
        demo_btn = st.button("Generate Smart Action Plan", width="stretch", type="primary")

    with col2:
        if demo_btn:
            with st.spinner("🧠 AI Engine analyzing pollution sources, forecasts, and generating multi-phase action plan..."):
                import time
                time.sleep(1.5)  # UX effect
                plan = generate_full_action_plan(ai_city, df_day, df_sources)
                
                if "error" in plan:
                    st.error(plan["error"])
                else:
                    p = plan["priority"]
                    st.markdown(f"""
<div style="background:linear-gradient(135deg, rgba(13,27,62,0.9) 0%, rgba(17,24,39,0.95) 100%); border-radius:20px; padding:32px; border:1px solid {p['color']}88;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:24px;">
<div>
<h2 style="margin:0; color:#e8eaf0; font-family:Outfit,sans-serif;">{ai_city} — Smart Action Plan</h2>
<div style="color:rgba(232,234,240,0.6); font-size:1.1rem; margin-top:4px;">Comprehensive AI Intervention Strategy</div>
</div>
<div class="ai-priority-badge" style="color:{p['color']}; border-color:{p['color']}; background:{p['bg']}; font-size:1rem; padding:8px 24px;">
{p['icon']} {p['level']} ALERT
</div>
</div>
<div style="display:flex; gap:16px; margin-bottom:28px;">
<div style="background:rgba(255,255,255,0.05); padding:16px; border-radius:12px; flex:1;">
<div style="font-size:0.8rem; color:rgba(232,234,240,0.5); text-transform:uppercase;">Primary Culprit</div>
<div style="font-size:1.6rem; font-weight:800; color:#ff6b6b; margin:4px 0;">{plan['ranked_sources'][0][1]}</div>
<div style="color:rgba(232,234,240,0.7);">{plan['ranked_sources'][0][2]:.1f}% contribution</div>
</div>
<div style="background:rgba(255,255,255,0.05); padding:16px; border-radius:12px; flex:1;">
<div style="font-size:0.8rem; color:rgba(232,234,240,0.5); text-transform:uppercase;">Legal Framework</div>
<div style="font-size:1rem; font-weight:600; color:#6bcb77; margin:4px 0; line-height:1.4;">{plan['policy_framework']}</div>
</div>
<div style="background:rgba(255,255,255,0.05); padding:16px; border-radius:12px; flex:1;">
<div style="font-size:0.8rem; color:rgba(232,234,240,0.5); text-transform:uppercase;">Est. Cost</div>
<div style="font-size:1.6rem; font-weight:800; color:#ffd93d; margin:4px 0;">₹{plan['impact']['estimated_cost_cr']} Cr</div>
</div>
</div>
</div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Implementation timeline
                    st.markdown('<div class="section-header">📅 Multi-Phase Implementation Strategy</div>', unsafe_allow_html=True)
                    
                    for phase in plan["phases"]:
                        st.markdown(f"""
<div style="border-left: 4px solid {phase['phase_color']}; padding-left:20px; margin-bottom:24px;">
<div style="font-family:Outfit,sans-serif; font-size:1.3rem; font-weight:800; color:{phase['phase_color']}; margin-bottom:12px;">{phase['phase']}</div>
                        """, unsafe_allow_html=True)
                        for action in phase["actions"]:
                            st.markdown(f"""
<div class="ai-action-item" style="border-left:none; background:rgba(255,255,255,0.03);">
<span style="color:{phase['phase_color']}; font-size:1.1rem; flex-shrink:0;">■</span>
<span>{action}</span>
</div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    # Overall impact projection
                    st.markdown("---")
                    st.markdown('<div class="section-header">📈 Projected Outcome (30 Days)</div>', unsafe_allow_html=True)
                    impact = plan["impact"]
                    
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    with ic1:
                        st.metric("Total AQI Reduction", f"-{impact['total_aqi_reduction_pct']}%", delta="Success", delta_color="normal")
                    with ic2:
                        st.metric("PM2.5 Reduction", f"-{impact['pm25_reduction_pct']}%", delta="Clean Air", delta_color="normal")
                    with ic3:
                        st.metric("Target AQI", f"{impact['projected_aqi']:.0f}", delta=f"{base_aqi_ai:.0f} currently", delta_color="inverse")
                    with ic4:
                        st.metric("Citizens Protected", f"{format_population(impact['population_benefit'])}", delta="Health Risk ↓", delta_color="normal")

        else:
            # Default view - normal recommendation panel
            render_ai_panel(ai_city, base_aqi_ai, df_sources)
