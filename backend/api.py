"""
AirGuard AI - FastAPI Backend
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data_loader import (
    load_city_day, load_city_hour, load_traffic,
    load_pollution_sources, load_city_coordinates,
    get_latest_city_aqi, get_aqi_health_message, _classify_aqi
)
from backend.ml_models import train_models, forecast_aqi, simulate_digital_twin

app = FastAPI(
    title="AirGuard AI API",
    description="🌬️ AI-powered Air Quality Intelligence Platform for Indian Cities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Data Cache ---
_cache = {}

def get_data():
    if "city_day" not in _cache:
        _cache["city_day"] = load_city_day()
        _cache["city_coordinates"] = load_city_coordinates()
        _cache["traffic"] = load_traffic()
        _cache["pollution_sources"] = load_pollution_sources()
    return _cache


# ---- Pydantic Models ----

class SimulatorRequest(BaseModel):
    city: str = Field(..., example="Delhi")
    traffic_reduction: float = Field(0.0, ge=0, le=100, description="Traffic reduction %")
    construction_reduction: float = Field(0.0, ge=0, le=100, description="Construction reduction %")
    industrial_reduction: float = Field(0.0, ge=0, le=100, description="Industrial reduction %")
    green_cover_increase: float = Field(0.0, ge=0, le=100, description="Green cover increase %")


class ForecastRequest(BaseModel):
    city: str = Field(..., example="Delhi")
    model: Optional[str] = Field("XGBoost", example="XGBoost")


# ---- Endpoints ----

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "🌬️ AirGuard AI API is running!",
        "version": "1.0.0",
        "endpoints": [
            "/api/cities", "/api/aqi/latest", "/api/aqi/trend",
            "/api/forecast", "/api/pollution-sources", "/api/simulator",
            "/api/health-advisory", "/api/train-models", "/docs"
        ]
    }


@app.get("/api/cities", tags=["Cities"])
def get_cities():
    """Get list of all available cities"""
    data = get_data()
    cities = sorted(data["city_day"]["City"].unique().tolist())
    return {"cities": cities, "count": len(cities)}


@app.get("/api/aqi/latest", tags=["AQI"])
def get_latest_aqi(city: Optional[str] = Query(None, description="Filter by city")):
    """Get latest AQI reading for all cities or a specific city"""
    data = get_data()
    df = data["city_day"]
    
    latest = get_latest_city_aqi(df)
    
    if city:
        latest = latest[latest["City"].str.lower() == city.lower()]
        if latest.empty:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    
    result = latest.fillna(0).to_dict(orient="records")
    
    # Add health advisory
    for r in result:
        advisory = get_aqi_health_message(r.get("AQI", 0))
        r["health_advisory"] = advisory
        r["Date"] = str(r.get("Date", ""))
    
    return {"data": result}


@app.get("/api/aqi/trend", tags=["AQI"])
def get_aqi_trend(
    city: str = Query(..., description="City name"),
    days: int = Query(90, ge=7, le=365, description="Number of days")
):
    """Get AQI trend data for a city"""
    data = get_data()
    df = data["city_day"]
    
    city_df = df[df["City"].str.lower() == city.lower()].dropna(subset=["AQI"])
    
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for city '{city}'")
    
    city_df = city_df.sort_values("Date").tail(days)
    city_df["AQI_MA7"] = city_df["AQI"].rolling(7, min_periods=1).mean()
    
    return {
        "city": city,
        "days": days,
        "data": city_df[["Date", "AQI", "AQI_Bucket", "AQI_MA7", "PM2.5", "PM10", "NO2", "SO2", "O3"]]
                       .fillna(0)
                       .assign(Date=lambda x: x["Date"].astype(str))
                       .to_dict(orient="records")
    }


@app.get("/api/aqi/monthly", tags=["AQI"])
def get_monthly_aqi(city: str = Query(..., description="City name")):
    """Get monthly average AQI by year"""
    data = get_data()
    df = data["city_day"]
    
    city_df = df[df["City"].str.lower() == city.lower()].dropna(subset=["AQI"]).copy()
    city_df["Year"] = city_df["Date"].dt.year
    city_df["Month"] = city_df["Date"].dt.month
    
    monthly = city_df.groupby(["Year", "Month"])["AQI"].mean().reset_index()
    monthly["AQI"] = monthly["AQI"].round(1)
    
    return {"city": city, "data": monthly.to_dict(orient="records")}


@app.get("/api/stats/top-polluted", tags=["Statistics"])
def get_top_polluted(n: int = Query(10, ge=3, le=26)):
    """Get top N most polluted cities"""
    data = get_data()
    df = data["city_day"]
    
    city_stats = (df.dropna(subset=["AQI"])
                    .groupby("City")["AQI"]
                    .agg(["mean", "max", "min", "std"])
                    .reset_index()
                    .rename(columns={"mean": "avg_aqi", "max": "max_aqi", "min": "min_aqi", "std": "std_aqi"})
                    .sort_values("avg_aqi", ascending=False)
                    .head(n)
                    .round(1))
    
    city_stats["bucket"] = city_stats["avg_aqi"].apply(_classify_aqi)
    
    return {"data": city_stats.to_dict(orient="records")}


@app.post("/api/forecast", tags=["Forecasting"])
def get_forecast(request: ForecastRequest):
    """Forecast AQI for next 24h, 48h, 72h using ML models"""
    data = get_data()
    df = data["city_day"]
    
    city_df = df[df["City"].str.lower() == request.city.lower()]
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{request.city}' not found")
    
    try:
        forecasts = forecast_aqi(df, request.city)
        
        # Get current AQI
        latest = city_df.dropna(subset=["AQI"]).sort_values("Date").iloc[-1]
        current_aqi = float(latest["AQI"])
        
        return {
            "city": request.city,
            "current_aqi": current_aqi,
            "current_bucket": _classify_aqi(current_aqi),
            "forecasts": forecasts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


@app.get("/api/pollution-sources", tags=["Pollution Sources"])
def get_pollution_sources(city: Optional[str] = Query(None)):
    """Get pollution source attribution data"""
    data = get_data()
    df = data["pollution_sources"]
    
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    
    if df.empty:
        # Return averaged data
        df = data["pollution_sources"]
    
    avg = df[["traffic", "construction", "industry", "waste_burning"]].mean().round(1)
    
    return {
        "city": city or "Average",
        "sources": {
            "traffic": float(avg["traffic"]),
            "construction": float(avg["construction"]),
            "industry": float(avg["industry"]),
            "waste_burning": float(avg["waste_burning"]),
        }
    }


@app.post("/api/simulator", tags=["Digital Twin"])
def run_simulator(request: SimulatorRequest):
    """
    Digital Twin Simulator: Predict AQI improvement from interventions
    """
    data = get_data()
    df = data["city_day"]
    
    city_df = df[df["City"].str.lower() == request.city.lower()].dropna(subset=["AQI"])
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{request.city}' not found")
    
    current_aqi = float(city_df.sort_values("Date").iloc[-1]["AQI"])
    
    result = simulate_digital_twin(
        base_aqi=current_aqi,
        traffic_reduction=request.traffic_reduction,
        construction_reduction=request.construction_reduction,
        industrial_reduction=request.industrial_reduction,
        green_cover_increase=request.green_cover_increase,
    )
    
    result["city"] = request.city
    result["current_bucket"] = _classify_aqi(current_aqi)
    result["predicted_bucket"] = _classify_aqi(result["predicted_aqi"])
    result["health_advisory"] = get_aqi_health_message(result["predicted_aqi"])
    
    return result


@app.get("/api/health-advisory", tags=["Health"])
def get_health_advisory(city: str = Query(..., description="City name")):
    """Get health advisory for a specific city"""
    data = get_data()
    df = data["city_day"]
    
    city_df = df[df["City"].str.lower() == city.lower()].dropna(subset=["AQI"])
    if city_df.empty:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    
    latest = city_df.sort_values("Date").iloc[-1]
    aqi = float(latest["AQI"])
    
    advisory = get_aqi_health_message(aqi)
    
    return {
        "city": city,
        "date": str(latest["Date"]),
        "aqi": aqi,
        "bucket": _classify_aqi(aqi),
        "advisory": advisory,
        "pollutants": {
            "PM2.5": float(latest["PM2.5"]) if not pd.isna(latest["PM2.5"]) else None,
            "PM10": float(latest["PM10"]) if not pd.isna(latest["PM10"]) else None,
            "NO2": float(latest["NO2"]) if not pd.isna(latest["NO2"]) else None,
            "SO2": float(latest["SO2"]) if not pd.isna(latest["SO2"]) else None,
            "O3": float(latest["O3"]) if not pd.isna(latest["O3"]) else None,
            "CO": float(latest["CO"]) if not pd.isna(latest["CO"]) else None,
        }
    }


@app.get("/api/coordinates", tags=["Maps"])
def get_city_coordinates():
    """Get city coordinates with latest AQI for map rendering"""
    data = get_data()
    df = data["city_day"]
    df_coords = data["city_coordinates"]
    
    latest = get_latest_city_aqi(df)
    
    # Merge coordinates
    df_coords_norm = df_coords.copy()
    if "city" in df_coords_norm.columns:
        df_coords_norm = df_coords_norm.rename(columns={"city": "City"})
    
    merged = latest.merge(df_coords_norm, on="City", how="inner")
    merged["color"] = merged["AQI"].apply(lambda x: get_aqi_color(x) if not pd.isna(x) else "#808080")
    
    from backend.data_loader import get_aqi_color
    
    result = merged.fillna(0).assign(Date=lambda x: x["Date"].astype(str))
    
    return {"data": result.to_dict(orient="records")}


@app.post("/api/train-models", tags=["ML"])
def train_ml_models():
    """Train and save ML models (XGBoost, Random Forest, Gradient Boosting)"""
    data = get_data()
    df = data["city_day"]
    
    try:
        results, features, scaler = train_models(df)
        
        summary = {}
        for name, res in results.items():
            summary[name] = {
                "mae": res["mae"],
                "rmse": res["rmse"],
                "r2": res["r2"],
            }
        
        return {
            "status": "success",
            "message": "Models trained and saved successfully",
            "performance": summary,
            "features_used": features,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
