"""
AirGuard AI - Visualization Module (Charts & Maps)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from typing import Optional, List
from .data_loader import get_aqi_color, _classify_aqi


AQI_COLOR_MAP = {
    "Good": "#00e400",
    "Satisfactory": "#9acd32",
    "Moderate": "#ffff00",
    "Poor": "#ff7e00",
    "Very Poor": "#ff0000",
    "Severe": "#8f3f97",
}

PLOTLY_TEMPLATE = "plotly_dark"


def plot_aqi_trend(df: pd.DataFrame, city: str, days: int = 90) -> go.Figure:
    """Line chart: AQI trend for a city"""
    city_df = df[df["City"] == city].dropna(subset=["AQI"]).sort_values("Date")
    city_df = city_df.tail(days)

    fig = go.Figure()
    
    # AQI line with gradient fill
    fig.add_trace(go.Scatter(
        x=city_df["Date"],
        y=city_df["AQI"],
        mode="lines",
        name="AQI",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.15)",
    ))

    # 7-day rolling average
    city_df["AQI_MA7"] = city_df["AQI"].rolling(7, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=city_df["Date"],
        y=city_df["AQI_MA7"],
        mode="lines",
        name="7-Day Average",
        line=dict(color="#ff6b6b", width=2, dash="dash"),
    ))

    # AQI threshold lines
    for threshold, label, color in [
        (50, "Good", "#00e400"),
        (100, "Satisfactory", "#9acd32"),
        (200, "Moderate", "#ffff00"),
        (300, "Poor", "#ff7e00"),
    ]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color, opacity=0.4,
                      annotation_text=label, annotation_position="right")

    fig.update_layout(
        title=dict(text=f"AQI Trend — {city}", font=dict(size=20, color="white")),
        xaxis_title="Date",
        yaxis_title="AQI",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=400,
    )
    return fig


def plot_pollutant_breakdown(df: pd.DataFrame, city: str) -> go.Figure:
    """Radar chart: pollutant profile for a city"""
    city_df = df[df["City"] == city].dropna(subset=["AQI"])
    if city_df.empty:
        return go.Figure()
    
    last = city_df.sort_values("Date").iloc[-1]
    pollutants = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO", "NOx"]
    values = []
    for p in pollutants:
        val = last.get(p, 0)
        values.append(float(val) if not pd.isna(val) else 0)

    # Normalize to 0-100 scale
    max_vals = [250, 430, 200, 200, 200, 50, 200]
    normalized = [min(100, (v / m) * 100) for v, m in zip(values, max_vals)]

    fig = go.Figure(go.Scatterpolar(
        r=normalized + [normalized[0]],
        theta=pollutants + [pollutants[0]],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.2)",
        line=dict(color="#00d4ff", width=2),
        name="Pollutant Levels",
    ))

    fig.update_layout(
        title=dict(text=f"Pollutant Profile — {city}", font=dict(size=18, color="white")),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="rgba(255,255,255,0.5)"),
            angularaxis=dict(color="rgba(255,255,255,0.7)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


def plot_top_polluted_cities(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bar chart: top N most polluted cities by average AQI"""
    city_aqi = (df.dropna(subset=["AQI"])
                  .groupby("City")["AQI"]
                  .mean()
                  .reset_index()
                  .sort_values("AQI", ascending=True)
                  .tail(top_n))

    colors = [get_aqi_color(v) for v in city_aqi["AQI"]]

    fig = go.Figure(go.Bar(
        x=city_aqi["AQI"],
        y=city_aqi["City"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}" for v in city_aqi["AQI"]],
        textposition="outside",
        textfont=dict(color="white"),
    ))

    fig.update_layout(
        title=dict(text=f"Top {top_n} Most Polluted Cities (Avg AQI)", font=dict(size=18, color="white")),
        xaxis_title="Average AQI",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=10, r=60),
    )
    return fig


def plot_monthly_heatmap(df: pd.DataFrame, city: str) -> go.Figure:
    """Heatmap: monthly AQI averages by year"""
    city_df = df[df["City"] == city].dropna(subset=["AQI"]).copy()
    city_df["Year"] = pd.to_datetime(city_df["Date"]).dt.year
    city_df["Month"] = pd.to_datetime(city_df["Date"]).dt.month

    pivot = city_df.pivot_table(values="AQI", index="Year", columns="Month", aggfunc="mean")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#00e400"], [0.2, "#9acd32"], [0.4, "#ffff00"],
            [0.6, "#ff7e00"], [0.8, "#ff0000"], [1.0, "#8f3f97"]
        ],
        hovertemplate="Year: %{y}<br>Month: %{x}<br>AQI: %{z:.0f}<extra></extra>",
        colorbar=dict(title="AQI", tickfont=dict(color="white"), title_font=dict(color="white")),
    ))

    fig.update_layout(
        title=dict(text=f"Monthly AQI Heatmap — {city}", font=dict(size=18, color="white")),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="white"),
        yaxis=dict(color="white"),
        height=380,
    )
    return fig


def plot_pollution_sources(df_sources: pd.DataFrame, city: str) -> go.Figure:
    """Donut + bar for pollution source attribution"""
    city_data = df_sources[df_sources["city"].str.lower() == city.lower()]
    
    if city_data.empty:
        # Use average
        city_data = df_sources

    sources = ["traffic", "construction", "industry", "waste_burning"]
    values = city_data[sources].mean().values
    labels = ["Traffic", "Construction", "Industry", "Waste Burning"]
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=2)),
        textinfo="label+percent",
        textfont=dict(color="white", size=13),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Pollution Source Attribution — {city}", font=dict(size=18, color="white")),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="white")),
        height=400,
    )
    return fig


def plot_multi_city_comparison(df: pd.DataFrame, cities: List[str]) -> go.Figure:
    """Multi-line comparison of AQI trends across cities"""
    palette = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#c77dff", "#ff9f1c"]
    
    fig = go.Figure()
    
    for i, city in enumerate(cities):
        city_df = df[df["City"] == city].dropna(subset=["AQI"]).sort_values("Date").tail(180)
        city_df["AQI_MA7"] = city_df["AQI"].rolling(7, min_periods=1).mean()
        
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=city_df["Date"],
            y=city_df["AQI_MA7"],
            mode="lines",
            name=city,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=dict(text="Multi-City AQI Comparison (7-Day Average)", font=dict(size=20, color="white")),
        xaxis_title="Date",
        yaxis_title="AQI",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="white")),
        hovermode="x unified",
        height=450,
    )
    return fig


def plot_forecast_chart(forecasts: dict, city: str) -> go.Figure:
    """Bar chart showing forecast results from multiple models"""
    horizons = [24, 48, 72]
    labels = ["24h", "48h", "72h"]
    colors_model = {"XGBoost": "#00d4ff", "RandomForest": "#ff6b6b", "GradientBoosting": "#ffd93d"}

    fig = go.Figure()
    
    for model_name, horizon_dict in forecasts.items():
        values = [horizon_dict.get(h, 0) for h in horizons]
        fig.add_trace(go.Bar(
            name=model_name,
            x=labels,
            y=values,
            marker_color=colors_model.get(model_name, "#ffffff"),
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
            textfont=dict(color="white"),
        ))

    fig.update_layout(
        title=dict(text=f"AQI Forecast — {city}", font=dict(size=18, color="white")),
        xaxis_title="Forecast Horizon",
        yaxis_title="Predicted AQI",
        barmode="group",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="white")),
        height=400,
    )
    return fig


def plot_digital_twin_gauge(base_aqi: float, predicted_aqi: float) -> go.Figure:
    """Gauge chart for digital twin AQI comparison"""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=base_aqi,
        title={"text": "Current AQI", "font": {"color": "white", "size": 16}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor": "white"},
            "bar": {"color": get_aqi_color(base_aqi)},
            "steps": [
                {"range": [0, 50], "color": "rgba(0,228,0,0.2)"},
                {"range": [50, 100], "color": "rgba(154,205,50,0.2)"},
                {"range": [100, 200], "color": "rgba(255,255,0,0.2)"},
                {"range": [200, 300], "color": "rgba(255,126,0,0.2)"},
                {"range": [300, 400], "color": "rgba(255,0,0,0.2)"},
                {"range": [400, 500], "color": "rgba(143,63,151,0.2)"},
            ],
            "bgcolor": "rgba(0,0,0,0)",
        },
        number={"font": {"color": get_aqi_color(base_aqi), "size": 40}},
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_aqi,
        delta={"reference": base_aqi, "valueformat": ".0f", "increasing": {"color": "#ff0000"}, 
               "decreasing": {"color": "#00e400"}},
        title={"text": "Predicted AQI", "font": {"color": "white", "size": 16}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor": "white"},
            "bar": {"color": get_aqi_color(predicted_aqi)},
            "steps": [
                {"range": [0, 50], "color": "rgba(0,228,0,0.2)"},
                {"range": [50, 100], "color": "rgba(154,205,50,0.2)"},
                {"range": [100, 200], "color": "rgba(255,255,0,0.2)"},
                {"range": [200, 300], "color": "rgba(255,126,0,0.2)"},
                {"range": [300, 400], "color": "rgba(255,0,0,0.2)"},
                {"range": [400, 500], "color": "rgba(143,63,151,0.2)"},
            ],
            "bgcolor": "rgba(0,0,0,0)",
        },
        number={"font": {"color": get_aqi_color(predicted_aqi), "size": 40}},
    ), row=1, col=2)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
    )
    return fig


def create_folium_heatmap(df_day: pd.DataFrame, df_coords: pd.DataFrame) -> folium.Map:
    """Interactive Folium heatmap of AQI across cities"""
    # Get latest AQI per city
    latest = (df_day.dropna(subset=["AQI"])
                    .sort_values("Date")
                    .groupby("City")
                    .last()
                    .reset_index()[["City", "AQI", "AQI_Bucket"]])

    # Normalize the city column name in coords to match 'City' in latest df
    df_coords_norm = df_coords.copy()
    # city_coordinates.csv uses lowercase 'city' — rename to 'City' for the merge
    col0 = df_coords_norm.columns[0]
    if col0 != "City":
        df_coords_norm = df_coords_norm.rename(columns={col0: "City"})

    merged = latest.merge(df_coords_norm, on="City", how="inner")


    m = folium.Map(
        location=[20.5937, 78.9629],  # Center of India
        zoom_start=5,
        tiles="CartoDB dark_matter",
    )

    # Heatmap layer
    heat_data = []
    for _, row in merged.iterrows():
        if not pd.isna(row.get("latitude", np.nan)) and not pd.isna(row.get("longitude", np.nan)):
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            aqi = float(row["AQI"])
            heat_data.append([lat, lon, aqi / 500])  # Normalized

    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.4,
            max_zoom=10,
            radius=40,
            blur=25,
            gradient={0.2: "#00e400", 0.4: "#ffff00", 0.6: "#ff7e00", 0.8: "#ff0000", 1.0: "#8f3f97"},
        ).add_to(m)

    # Markers with popups
    for _, row in merged.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            aqi = float(row["AQI"])
            city = row["City"]
            color = get_aqi_color(aqi)
            bucket = row.get("AQI_Bucket", _classify_aqi(aqi))

            popup_html = f"""
            <div style="font-family: Arial; background: #1a1a2e; color: white; padding: 10px; border-radius: 8px; min-width: 180px;">
                <h4 style="margin:0; color:#00d4ff;">{city}</h4>
                <hr style="border-color:#333; margin: 6px 0;">
                <p style="margin:4px 0;"><b>AQI:</b> <span style="color:{color}; font-size:1.3em; font-weight:bold;">{aqi:.0f}</span></p>
                <p style="margin:4px 0;"><b>Status:</b> <span style="color:{color};">{bucket}</span></p>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{city}: AQI {aqi:.0f}",
            ).add_to(m)

        except (ValueError, TypeError, KeyError):
            continue

    return m


def plot_aqi_distribution(df: pd.DataFrame, city: str = None) -> go.Figure:
    """Histogram of AQI distribution"""
    if city:
        data = df[df["City"] == city].dropna(subset=["AQI"])
        title = f"AQI Distribution — {city}"
    else:
        data = df.dropna(subset=["AQI"])
        title = "AQI Distribution — All Cities"

    fig = go.Figure(go.Histogram(
        x=data["AQI"],
        nbinsx=40,
        marker=dict(
            color=data["AQI"],
            colorscale=[
                [0.0, "#00e400"], [0.2, "#9acd32"], [0.4, "#ffff00"],
                [0.6, "#ff7e00"], [0.8, "#ff0000"], [1.0, "#8f3f97"]
            ],
            cmin=0, cmax=500,
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        xaxis_title="AQI",
        yaxis_title="Count",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    return fig


def plot_model_comparison(results: dict) -> go.Figure:
    """Compare model performance metrics"""
    models = list(results.keys())
    mae_vals = [results[m]["mae"] for m in models]
    rmse_vals = [results[m]["rmse"] for m in models]
    r2_vals = [results[m]["r2"] for m in models]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE (lower=better)", "RMSE (lower=better)", "R² (higher=better)"])
    colors = ["#00d4ff", "#ff6b6b", "#ffd93d"]

    fig.add_trace(go.Bar(x=models, y=mae_vals, marker_color=colors, name="MAE", 
                         text=[f"{v:.1f}" for v in mae_vals], textposition="outside",
                         textfont=dict(color="white")), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=rmse_vals, marker_color=colors, name="RMSE",
                         text=[f"{v:.1f}" for v in rmse_vals], textposition="outside",
                         textfont=dict(color="white")), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=r2_vals, marker_color=colors, name="R²",
                         text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
                         textfont=dict(color="white")), row=1, col=3)

    fig.update_layout(
        title=dict(text="Model Performance Comparison", font=dict(size=18, color="white")),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=380,
    )
    return fig
