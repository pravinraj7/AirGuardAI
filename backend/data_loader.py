"""
AirGuard AI - Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_city_day() -> pd.DataFrame:
    """Load and preprocess city_day.csv"""
    df = pd.read_csv(DATA_DIR / "city_day.csv", parse_dates=["Date"])
    
    # Fill missing AQI with interpolation per city
    df = df.sort_values(["City", "Date"])
    
    # Numeric columns
    num_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", 
                "O3", "Benzene", "Toluene", "Xylene", "AQI"]
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Interpolate missing values per city group
    df[num_cols] = df.groupby("City")[num_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    
    # Fill AQI_Bucket from AQI
    df["AQI_Bucket"] = df.apply(
        lambda row: _classify_aqi(row["AQI"]) if pd.isna(row["AQI_Bucket"]) or row["AQI_Bucket"] == "" 
        else row["AQI_Bucket"],
        axis=1
    )
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter
    
    return df


def load_city_hour() -> pd.DataFrame:
    """Load and preprocess city_hour.csv (sampled for performance)"""
    df = pd.read_csv(DATA_DIR / "city_hour.csv", parse_dates=["Datetime"])
    
    num_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                "O3", "Benzene", "Toluene", "Xylene", "AQI"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.sort_values(["City", "Datetime"])
    df["Hour"] = df["Datetime"].dt.hour
    df["Date"] = df["Datetime"].dt.date
    
    return df


def load_traffic() -> pd.DataFrame:
    """Load traffic data"""
    df = pd.read_csv(DATA_DIR / "traffic.csv", parse_dates=["date"])
    df["traffic_density"] = pd.to_numeric(df["traffic_density"], errors="coerce")
    return df


def load_pollution_sources() -> pd.DataFrame:
    """Load pollution sources data"""
    df = pd.read_csv(DATA_DIR / "pollution_sources.csv", parse_dates=["date"])
    for col in ["traffic", "construction", "industry", "waste_burning"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_city_coordinates() -> pd.DataFrame:
    """Load city coordinates"""
    df = pd.read_csv(DATA_DIR / "city_coordinates.csv")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def _classify_aqi(aqi: float) -> str:
    """Classify AQI into bucket categories"""
    if pd.isna(aqi):
        return "Unknown"
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_aqi_color(aqi: float) -> str:
    """Get color for AQI value"""
    if pd.isna(aqi):
        return "#808080"
    elif aqi <= 50:
        return "#00e400"
    elif aqi <= 100:
        return "#9acd32"
    elif aqi <= 200:
        return "#ffff00"
    elif aqi <= 300:
        return "#ff7e00"
    elif aqi <= 400:
        return "#ff0000"
    else:
        return "#8f3f97"


def get_aqi_health_message(aqi: float) -> dict:
    """Return health advisory based on AQI"""
    if pd.isna(aqi):
        return {"level": "Unknown", "color": "#808080", "message": "Data not available", 
                "recommendation": "Please check back later."}
    elif aqi <= 50:
        return {
            "level": "Good",
            "color": "#00e400",
            "message": "Air quality is satisfactory and poses little to no risk.",
            "recommendation": "Enjoy outdoor activities freely! Great day for exercise.",
            "icon": "😊"
        }
    elif aqi <= 100:
        return {
            "level": "Satisfactory",
            "color": "#9acd32",
            "message": "Air quality is acceptable. Minor concern for sensitive groups.",
            "recommendation": "Sensitive individuals should limit prolonged outdoor exertion.",
            "icon": "🙂"
        }
    elif aqi <= 200:
        return {
            "level": "Moderate",
            "color": "#ffff00",
            "message": "Members of sensitive groups may experience health effects.",
            "recommendation": "Wear N95 mask outdoors. Sensitive groups avoid prolonged outdoor activities.",
            "icon": "😐"
        }
    elif aqi <= 300:
        return {
            "level": "Poor",
            "color": "#ff7e00",
            "message": "Everyone may begin to experience adverse health effects.",
            "recommendation": "Wear mask, use air purifiers indoors. Limit outdoor activities.",
            "icon": "😷"
        }
    elif aqi <= 400:
        return {
            "level": "Very Poor",
            "color": "#ff0000",
            "message": "Health alert: everyone may experience serious health effects.",
            "recommendation": "Avoid outdoor activities. Use N99 masks. Keep windows closed.",
            "icon": "🤒"
        }
    else:
        return {
            "level": "Severe",
            "color": "#8f3f97",
            "message": "Health emergency conditions. Entire population affected.",
            "recommendation": "Stay indoors. Use HEPA air purifiers. Seek medical help if needed.",
            "icon": "🚨"
        }


def get_latest_city_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest AQI reading for each city"""
    return (df.dropna(subset=["AQI"])
              .sort_values("Date")
              .groupby("City")
              .last()
              .reset_index()[["City", "Date", "AQI", "AQI_Bucket", "PM2.5", "PM10", "NO2", "SO2", "O3"]])
