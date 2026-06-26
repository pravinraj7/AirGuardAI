"""
AirGuard AI — AI Action Recommendation Engine
============================================
Transforms AirGuard AI from analytics into an AI Decision Support System
for city administrators. Generates dynamic, explainable recommendations
based on AQI, pollution source attribution, forecasts, and Digital Twin outputs.
"""

import pandas as pd
import numpy as np
from typing import Optional

# ─── City Population Database (approximate, millions) ─────────────────────
CITY_POPULATIONS = {
    "Delhi": 32941000,
    "Mumbai": 20667656,
    "Kolkata": 14850000,
    "Bengaluru": 13193000,
    "Bangalore": 13193000,
    "Chennai": 10971000,
    "Hyderabad": 10004000,
    "Ahmedabad": 8450000,
    "Pune": 7276000,
    "Surat": 6936000,
    "Jaipur": 4020000,
    "Lucknow": 3765000,
    "Kanpur": 3144000,
    "Nagpur": 2891000,
    "Indore": 2767000,
    "Bhopal": 2118000,
    "Chandigarh": 1161000,
    "Coimbatore": 1601000,
    "Visakhapatnam": 2035000,
    "Patna": 2497000,
    "Talcher": 150000,
    "Amaravati": 100000,
    "Aizawl": 293000,
    "Amritsar": 1183000,
    "Brajrajnagar": 120000,
    "Ernakulam": 677000,
    "Guwahati": 1099000,
    "Jorapokhar": 80000,
    "Kochi": 677000,
    "Shillong": 354000,
    "Talcher": 150000,
    "Thiruvananthapuram": 1687000,
}
DEFAULT_POPULATION = 500000  # fallback

# ─── Recommendation Database (per source) ────────────────────────────────
RECOMMENDATIONS_DB = {
    "traffic": {
        "actions": [
            "Reduce heavy vehicle movement by 30% on major corridors",
            "Restrict diesel trucks during peak hours (7–10 AM, 5–8 PM)",
            "Implement odd-even vehicle scheme on alternate days",
            "Promote electric public transport & bus fleet upgrades",
            "Increase roadside tree cover & green buffer zones",
            "Deploy traffic-free zones in high-pollution hotspots",
            "Enforce strict PUC (Pollution Under Control) checks",
        ],
        "quick_actions": "Reduce speed limits, increase traffic police, enforce idle-free zones",
        "policy": "Issue temporary traffic restriction order under GRAP Stage III",
        "digital_twin": {"traffic_reduction": 30, "construction_reduction": 0, "industrial_reduction": 0, "green_cover_increase": 10},
    },
    "construction": {
        "actions": [
            "Restrict construction in hotspot zones — issue stop-work notices",
            "Mandate water spraying on all active construction sites (3×/day)",
            "Enforce dust suppression barriers & nets at all sites",
            "Limit construction hours to 9 AM – 6 PM only",
            "Temporarily halt non-essential construction projects",
            "Require dust-control compliance certificates for new permits",
            "Deploy mobile dust monitoring units at large sites",
        ],
        "quick_actions": "Immediate halt on top 5 dust-generating sites; deploy water tankers",
        "policy": "Issue construction ban notification under Environment Protection Act",
        "digital_twin": {"traffic_reduction": 10, "construction_reduction": 40, "industrial_reduction": 0, "green_cover_increase": 5},
    },
    "industry": {
        "actions": [
            "Issue emission reduction notices to top 10 industrial units",
            "Enforce 24/7 Continuous Emission Monitoring Systems (CEMS)",
            "Switch industrial boilers to cleaner fuel (PNG/CNG)",
            "Temporary shutdown of non-compliant high-emission units",
            "Mandatory installation of electrostatic precipitators",
            "Restrict industrial production during high-AQI periods",
            "Conduct surprise emission audits across industrial clusters",
        ],
        "quick_actions": "Emergency emission curb notices; industrial cluster alert system activated",
        "policy": "Invoke powers under Air (Prevention and Control of Pollution) Act, 1981",
        "digital_twin": {"traffic_reduction": 0, "construction_reduction": 0, "industrial_reduction": 35, "green_cover_increase": 5},
    },
    "waste_burning": {
        "actions": [
            "Deploy anti-burning enforcement teams in high-risk wards",
            "Increase municipal waste collection frequency to twice daily",
            "Set up centralized bio-waste processing centers",
            "Issue Rs. 5,000 fines for open burning violations",
            "Run community awareness campaigns in identified hotspot areas",
            "Install smoke detection sensors in waste-prone zones",
            "Establish dedicated biomass waste composting facilities",
        ],
        "quick_actions": "Deploy rapid response teams; set up helpline for open-burning complaints",
        "policy": "Issue prohibition order under Solid Waste Management Rules, 2016",
        "digital_twin": {"traffic_reduction": 5, "construction_reduction": 5, "industrial_reduction": 5, "green_cover_increase": 15},
    },
}

# ─── Smart Action Presets (One-Click) ─────────────────────────────────────
SMART_ACTION_PRESETS = {
    "Traffic Control Day": {
        "traffic_reduction": 40,
        "construction_reduction": 10,
        "industrial_reduction": 5,
        "green_cover_increase": 5,
        "description": "Full traffic restriction — odd-even + heavy vehicle ban",
        "icon": "🚗",
        "color": "#ff6b6b",
    },
    "Construction Ban": {
        "traffic_reduction": 10,
        "construction_reduction": 80,
        "industrial_reduction": 0,
        "green_cover_increase": 10,
        "description": "All non-essential construction halted for 48 hours",
        "icon": "🏗️",
        "color": "#ffd93d",
    },
    "Industrial Emission Control": {
        "traffic_reduction": 5,
        "construction_reduction": 10,
        "industrial_reduction": 50,
        "green_cover_increase": 5,
        "description": "CEMS enforcement + production cap on high-polluters",
        "icon": "🏭",
        "color": "#6bcb77",
    },
    "Green City Initiative": {
        "traffic_reduction": 15,
        "construction_reduction": 15,
        "industrial_reduction": 10,
        "green_cover_increase": 40,
        "description": "Mass tree plantation + green corridor creation",
        "icon": "🌳",
        "color": "#4d96ff",
    },
    "Emergency AQI Response": {
        "traffic_reduction": 50,
        "construction_reduction": 70,
        "industrial_reduction": 40,
        "green_cover_increase": 20,
        "description": "All interventions activated — GRAP Stage IV emergency",
        "icon": "🚨",
        "color": "#8f3f97",
    },
}

# ─── Priority Classification ──────────────────────────────────────────────
def get_priority(aqi: float) -> dict:
    """Classify AQI into priority levels with display metadata."""
    if aqi > 300:
        return {
            "level": "CRITICAL",
            "color": "#8f3f97",
            "bg": "rgba(143,63,151,0.15)",
            "border": "#8f3f97",
            "icon": "🚨",
            "badge_color": "#8f3f97",
            "description": "Emergency conditions — immediate government intervention required",
        }
    elif aqi > 200:
        return {
            "level": "HIGH",
            "color": "#ff0000",
            "bg": "rgba(255,0,0,0.1)",
            "border": "#ff0000",
            "icon": "🔴",
            "badge_color": "#ff0000",
            "description": "Serious health risk — urgent administrative action needed",
        }
    elif aqi > 100:
        return {
            "level": "MEDIUM",
            "color": "#ff7e00",
            "bg": "rgba(255,126,0,0.1)",
            "border": "#ff7e00",
            "icon": "🟠",
            "badge_color": "#ff7e00",
            "description": "Elevated pollution — preventive measures recommended",
        }
    else:
        return {
            "level": "LOW",
            "color": "#00e400",
            "bg": "rgba(0,228,0,0.08)",
            "border": "#00e400",
            "icon": "🟢",
            "badge_color": "#00e400",
            "description": "Air quality acceptable — monitoring mode",
        }


def get_source_ranking(df_sources: pd.DataFrame, city: str) -> list:
    """
    Return ranked pollution sources for a city (or average if city not found).
    Returns list of (source_name, display_name, percentage) tuples sorted desc.
    """
    city_data = df_sources[df_sources["city"].str.lower() == city.lower()]
    
    if city_data.empty:
        city_data = df_sources  # fallback to average

    averages = city_data[["traffic", "construction", "industry", "waste_burning"]].mean()
    total = averages.sum()

    # Normalize to 100%
    if total > 0:
        percentages = (averages / total * 100).round(1)
    else:
        percentages = averages

    display_names = {
        "traffic": "Traffic",
        "construction": "Construction",
        "industry": "Industry",
        "waste_burning": "Waste Burning",
    }

    ranked = sorted(
        [(src, display_names[src], float(percentages[src])) for src in averages.index],
        key=lambda x: x[2],
        reverse=True,
    )
    return ranked


def calculate_impact(
    city: str,
    aqi: float,
    primary_source: str,
    primary_pct: float,
    action_reduction_pct: float = 30.0,
) -> dict:
    """
    Estimate expected impact of the recommended action.
    Returns AQI reduction %, PM2.5 reduction %, and population benefit.
    """
    # AQI reduction based on primary source contribution & action intensity
    source_coefficients = {
        "traffic": 0.35,
        "construction": 0.15,
        "industry": 0.30,
        "waste_burning": 0.10,
    }
    coeff = source_coefficients.get(primary_source, 0.20)
    aqi_reduction_pct = (primary_pct / 100) * coeff * action_reduction_pct * 1.5
    aqi_reduction_pct = min(45.0, round(aqi_reduction_pct, 1))

    # PM2.5 is typically reduced ~20% more effectively than AQI
    pm25_reduction_pct = min(55.0, round(aqi_reduction_pct * 1.25, 1))

    # Population benefit — proportion who experience health improvement
    population = CITY_POPULATIONS.get(city, DEFAULT_POPULATION)
    # Assume 30-60% of population in high-pollution areas benefits
    benefit_fraction = 0.35 if aqi > 200 else 0.20 if aqi > 100 else 0.10
    population_benefit = int(population * benefit_fraction * (aqi_reduction_pct / 100) * 3)
    population_benefit = max(5000, population_benefit)

    # Predicted AQI after action
    predicted_aqi = max(10, aqi * (1 - aqi_reduction_pct / 100))

    return {
        "aqi_reduction_pct": aqi_reduction_pct,
        "pm25_reduction_pct": pm25_reduction_pct,
        "population_benefit": population_benefit,
        "predicted_aqi": round(predicted_aqi, 1),
        "health_risk_reduction_pct": min(60.0, round(aqi_reduction_pct * 1.4, 1)),
    }


def generate_explainability(
    city: str,
    aqi: float,
    ranked_sources: list,
    forecast_aqi: Optional[float] = None,
    sim_result: Optional[dict] = None,
) -> list:
    """
    Generate human-readable XAI reasoning bullets for the recommendation.
    """
    primary_src, primary_display, primary_pct = ranked_sources[0]
    secondary_src, secondary_display, secondary_pct = ranked_sources[1] if len(ranked_sources) > 1 else ("", "", 0)

    reasons = [
        f"{primary_display} contributes {primary_pct:.1f}% of total pollution — the dominant source.",
    ]

    if secondary_pct > 15:
        reasons.append(
            f"{secondary_display} is the second-largest source at {secondary_pct:.1f}%, compounding the impact."
        )

    aqi_level = "Severe" if aqi > 400 else "Very Poor" if aqi > 300 else "Poor" if aqi > 200 else "Moderate" if aqi > 100 else "Satisfactory"
    reasons.append(f"Current AQI of {aqi:.0f} is classified as {aqi_level} — affecting the general population.")

    if forecast_aqi is not None:
        trend = "increase" if forecast_aqi > aqi else "decrease"
        change = abs(forecast_aqi - aqi)
        reasons.append(
            f"24-hour forecast shows AQI is expected to {trend} by {change:.0f} points to {forecast_aqi:.0f}."
        )
        if forecast_aqi > aqi:
            reasons.append("Proactive intervention now will prevent worsening conditions tomorrow.")

    if sim_result and "aqi_improvement_pct" in sim_result:
        reasons.append(
            f"Digital Twin simulation confirms a {sim_result['aqi_improvement_pct']:.1f}% AQI improvement "
            f"is achievable through targeted interventions."
        )

    reasons.append(
        f"Targeting {primary_display.lower()} provides the highest cost-effectiveness ratio "
        f"for AQI reduction in {city}."
    )

    return reasons


def generate_recommendations(
    city: str,
    aqi: float,
    df_sources: pd.DataFrame,
    forecast_aqi: Optional[float] = None,
    sim_result: Optional[dict] = None,
) -> dict:
    """
    Main function: Generate full AI recommendation for a city.

    Returns a comprehensive recommendation dict with all UI-ready data.
    """
    ranked_sources = get_source_ranking(df_sources, city)
    primary_src, primary_display, primary_pct = ranked_sources[0]
    secondary_src, secondary_display, secondary_pct = ranked_sources[1] if len(ranked_sources) > 1 else ("", "", 0)

    priority = get_priority(aqi)
    impact = calculate_impact(city, aqi, primary_src, primary_pct)
    reasons = generate_explainability(city, aqi, ranked_sources, forecast_aqi, sim_result)

    # Select top 4 most relevant actions for the primary source
    all_actions = RECOMMENDATIONS_DB[primary_src]["actions"]
    # For high priority, add secondary source's first action too
    selected_actions = all_actions[:4]
    if secondary_src and priority["level"] in ("CRITICAL", "HIGH"):
        selected_actions = all_actions[:3] + [RECOMMENDATIONS_DB[secondary_src]["actions"][0]]

    # Best matching smart preset
    best_preset = _recommend_preset(primary_src, priority["level"])

    return {
        "city": city,
        "current_aqi": round(aqi, 1),
        "priority": priority,
        "ranked_sources": ranked_sources,
        "primary_source": {"name": primary_src, "display": primary_display, "pct": primary_pct},
        "secondary_source": {"name": secondary_src, "display": secondary_display, "pct": secondary_pct},
        "actions": selected_actions,
        "quick_action": RECOMMENDATIONS_DB[primary_src]["quick_action"] if "quick_action" in RECOMMENDATIONS_DB[primary_src] else RECOMMENDATIONS_DB[primary_src]["quick_actions"],
        "policy_reference": RECOMMENDATIONS_DB[primary_src]["policy"],
        "digital_twin_preset": RECOMMENDATIONS_DB[primary_src]["digital_twin"],
        "impact": impact,
        "explainability": reasons,
        "forecast_aqi": forecast_aqi,
        "recommended_preset": best_preset,
        "sim_result": sim_result,
    }


def _recommend_preset(primary_src: str, priority_level: str) -> str:
    """Pick the best one-click preset based on primary source and priority."""
    if priority_level == "CRITICAL":
        return "Emergency AQI Response"
    mapping = {
        "traffic": "Traffic Control Day",
        "construction": "Construction Ban",
        "industry": "Industrial Emission Control",
        "waste_burning": "Green City Initiative",
    }
    return mapping.get(primary_src, "Traffic Control Day")


def generate_full_action_plan(
    city: str,
    df_day: pd.DataFrame,
    df_sources: pd.DataFrame,
    forecast_aqi: Optional[float] = None,
) -> dict:
    """
    Hackathon Demo Mode: Generate a comprehensive intervention strategy
    for the selected city — the flagship AI feature.
    """
    city_df = df_day[df_day["City"] == city].dropna(subset=["AQI"]).sort_values("Date")

    if city_df.empty:
        return {"error": f"No AQI data found for {city}"}

    current_aqi = float(city_df.iloc[-1]["AQI"])
    aqi_7d_avg = float(city_df.tail(7)["AQI"].mean())
    aqi_30d_avg = float(city_df.tail(30)["AQI"].mean())
    aqi_trend = "worsening" if aqi_7d_avg > aqi_30d_avg else "improving"

    ranked_sources = get_source_ranking(df_sources, city)
    primary_rec = generate_recommendations(city, current_aqi, df_sources, forecast_aqi)

    # Multi-phase intervention plan
    phases = [
        {
            "phase": "Immediate (0–24 hrs)",
            "phase_color": "#ff6b6b",
            "actions": [
                f"Activate {primary_rec['priority']['level']} AQI Alert for {city}",
                primary_rec["quick_action"],
                "Deploy mobile AQI monitoring teams to hotspot zones",
                "Issue public health advisory via mobile alerts",
            ],
        },
        {
            "phase": "Short-term (1–7 days)",
            "phase_color": "#ffd93d",
            "actions": primary_rec["actions"][:3] + [
                f"Increase {ranked_sources[1][1]} control measures",
            ] if len(ranked_sources) > 1 else primary_rec["actions"][:4],
        },
        {
            "phase": "Medium-term (7–30 days)",
            "phase_color": "#6bcb77",
            "actions": [
                "Evaluate effectiveness of immediate interventions",
                "Scale successful measures city-wide",
                "Engage industrial units for voluntary emission cuts",
                "Launch public transport incentive program",
                "Plant 10,000 trees in high-pollution corridors",
            ],
        },
        {
            "phase": "Long-term (30+ days)",
            "phase_color": "#4d96ff",
            "actions": [
                "Develop city-wide Clean Air Action Plan (CAAP)",
                "Install permanent IoT air quality sensor network",
                "Transition fleet to electric vehicles (buses/autos)",
                "Establish green buffer zones around industrial areas",
                "Implement real-time AQI-based traffic management",
            ],
        },
    ]

    # KPIs for the plan
    population = CITY_POPULATIONS.get(city, DEFAULT_POPULATION)
    total_aqi_reduction = min(55, primary_rec["impact"]["aqi_reduction_pct"] * 2.5)
    projected_aqi = max(30, current_aqi * (1 - total_aqi_reduction / 100))

    return {
        "city": city,
        "current_aqi": round(current_aqi, 1),
        "aqi_7d_avg": round(aqi_7d_avg, 1),
        "aqi_30d_avg": round(aqi_30d_avg, 1),
        "aqi_trend": aqi_trend,
        "priority": primary_rec["priority"],
        "ranked_sources": ranked_sources,
        "phases": phases,
        "impact": {
            "projected_aqi": round(projected_aqi, 1),
            "total_aqi_reduction_pct": round(total_aqi_reduction, 1),
            "pm25_reduction_pct": round(total_aqi_reduction * 1.2, 1),
            "population_benefit": int(population * 0.25),
            "health_risk_reduction_pct": round(total_aqi_reduction * 1.5, 1),
            "estimated_cost_cr": round(population / 1000000 * 2.5, 1),  # Crores
        },
        "policy_framework": primary_rec["policy_reference"],
        "explainability": primary_rec["explainability"],
        "digital_twin_preset": primary_rec["digital_twin_preset"],
        "forecast_aqi": forecast_aqi,
    }


def get_smart_action_presets() -> dict:
    """Return all smart action presets for the one-click buttons."""
    return SMART_ACTION_PRESETS


def format_population(n: int) -> str:
    """Format population number for display."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)
