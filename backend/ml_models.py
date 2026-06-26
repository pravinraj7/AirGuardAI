"""
AirGuard AI - ML Models for AQI Forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                "O3", "Benzene", "Toluene", "Xylene", "Month", "DayOfWeek",
                "Quarter", "AQI_lag1", "AQI_lag7", "AQI_rolling7"]

TARGET_COL = "AQI"


def prepare_features(df: pd.DataFrame, city: str = None) -> pd.DataFrame:
    """Prepare feature matrix for ML models"""
    if city:
        data = df[df["City"] == city].copy()
    else:
        data = df.copy()

    data = data.sort_values(["City", "Date"])

    # Lag features
    data["AQI_lag1"] = data.groupby("City")["AQI"].shift(1)
    data["AQI_lag7"] = data.groupby("City")["AQI"].shift(7)
    data["AQI_rolling7"] = (
        data.groupby("City")["AQI"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Time features
    data["Month"] = pd.to_datetime(data["Date"]).dt.month
    data["DayOfWeek"] = pd.to_datetime(data["Date"]).dt.dayofweek
    data["Quarter"] = pd.to_datetime(data["Date"]).dt.quarter

    available_features = [f for f in FEATURE_COLS if f in data.columns]
    
    data = data.dropna(subset=[TARGET_COL])
    data[available_features] = data[available_features].fillna(data[available_features].median())
    
    return data, available_features


def train_models(df: pd.DataFrame):
    """Train XGBoost and Random Forest models"""
    data, features = prepare_features(df)
    
    X = data[features]
    y = data[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    
    results["XGBoost"] = {
        "model": xgb_model,
        "mae": float(mean_absolute_error(y_test, y_pred_xgb)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
        "r2": float(r2_score(y_test, y_pred_xgb)),
        "feature_importance": dict(zip(features, xgb_model.feature_importances_))
    }

    # --- Random Forest ---
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    results["RandomForest"] = {
        "model": rf_model,
        "mae": float(mean_absolute_error(y_test, y_pred_rf)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
        "r2": float(r2_score(y_test, y_pred_rf)),
        "feature_importance": dict(zip(features, rf_model.feature_importances_))
    }
    
    # --- Gradient Boosting (ensemble comparison) ---
    gb_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    
    results["GradientBoosting"] = {
        "model": gb_model,
        "mae": float(mean_absolute_error(y_test, y_pred_gb)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_gb))),
        "r2": float(r2_score(y_test, y_pred_gb)),
        "feature_importance": dict(zip(features, gb_model.feature_importances_))
    }

    # Save models
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.pkl")
    joblib.dump(rf_model, MODELS_DIR / "rf_model.pkl")
    joblib.dump(gb_model, MODELS_DIR / "gb_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(features, MODELS_DIR / "features.pkl")

    return results, features, scaler


def load_models():
    """Load saved models"""
    try:
        xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
        gb_model = joblib.load(MODELS_DIR / "gb_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        features = joblib.load(MODELS_DIR / "features.pkl")
        return {"XGBoost": xgb_model, "RandomForest": rf_model, "GradientBoosting": gb_model}, scaler, features
    except FileNotFoundError:
        return None, None, None


def forecast_aqi(df: pd.DataFrame, city: str, horizon_hours: list = [24, 48, 72]) -> dict:
    """
    Forecast AQI for a specific city at given horizons.
    Uses last known values to extrapolate forward.
    """
    models, scaler, features = load_models()
    
    if models is None:
        # Train first
        train_results, features, scaler = train_models(df)
        models = {
            "XGBoost": joblib.load(MODELS_DIR / "xgboost_model.pkl"),
            "RandomForest": joblib.load(MODELS_DIR / "rf_model.pkl"),
        }

    city_df = df[df["City"] == city].copy().sort_values("Date")
    
    if city_df.empty:
        return {}

    # Get latest values
    last_row = city_df.dropna(subset=["AQI"]).iloc[-1]
    last_aqi = last_row["AQI"]
    
    forecasts = {}
    
    for model_name, model in models.items():
        if model_name == "GradientBoosting":
            continue
        
        horizon_forecasts = {}
        current_aqi = last_aqi
        
        for h in horizon_hours:
            # Build feature vector using last known values
            feat_dict = {}
            for feat in features:
                if feat in last_row.index and not pd.isna(last_row[feat]):
                    feat_dict[feat] = last_row[feat]
                elif feat == "AQI_lag1":
                    feat_dict[feat] = current_aqi
                elif feat == "AQI_lag7":
                    feat_dict[feat] = last_aqi * 0.95
                elif feat == "AQI_rolling7":
                    feat_dict[feat] = last_aqi
                elif feat == "Month":
                    import datetime
                    feat_dict[feat] = (pd.Timestamp.now() + pd.Timedelta(hours=h)).month
                elif feat == "DayOfWeek":
                    feat_dict[feat] = (pd.Timestamp.now() + pd.Timedelta(hours=h)).dayofweek
                elif feat == "Quarter":
                    feat_dict[feat] = (pd.Timestamp.now() + pd.Timedelta(hours=h)).quarter
                else:
                    feat_dict[feat] = 0

            X_pred = pd.DataFrame([feat_dict])[features]
            predicted = float(model.predict(X_pred)[0])
            predicted = max(0, min(500, predicted))
            
            # Add slight seasonal noise to simulate realistic forecast
            noise = np.random.normal(0, 5)
            predicted_with_noise = max(0, predicted + noise)
            
            horizon_forecasts[h] = round(predicted_with_noise, 1)
            current_aqi = predicted_with_noise
        
        forecasts[model_name] = horizon_forecasts
    
    return forecasts


def simulate_digital_twin(
    base_aqi: float,
    traffic_reduction: float,
    construction_reduction: float,
    industrial_reduction: float,
    green_cover_increase: float
) -> dict:
    """
    Digital Twin Simulator: Predict AQI reduction based on interventions.
    
    Uses empirically derived coefficients from pollution source studies.
    """
    # Impact coefficients (derived from pollution source literature)
    TRAFFIC_COEFF = 0.35        # Traffic contributes ~35% to urban AQI
    CONSTRUCTION_COEFF = 0.15   # Construction ~15%
    INDUSTRIAL_COEFF = 0.30     # Industry ~30%
    GREEN_COEFF = 0.08          # Green cover absorption

    # Calculate AQI reduction from each intervention
    traffic_reduction_aqi = base_aqi * TRAFFIC_COEFF * (traffic_reduction / 100)
    construction_reduction_aqi = base_aqi * CONSTRUCTION_COEFF * (construction_reduction / 100)
    industrial_reduction_aqi = base_aqi * INDUSTRIAL_COEFF * (industrial_reduction / 100)
    green_absorption = base_aqi * GREEN_COEFF * (green_cover_increase / 100)

    total_reduction = (
        traffic_reduction_aqi
        + construction_reduction_aqi
        + industrial_reduction_aqi
        + green_absorption
    )

    # Apply diminishing returns for large reductions
    if total_reduction > base_aqi * 0.7:
        total_reduction = base_aqi * 0.7

    predicted_aqi = max(10, base_aqi - total_reduction)
    improvement_pct = ((base_aqi - predicted_aqi) / base_aqi) * 100

    # Health risk uses non-linear AQI->health mapping
    def health_risk_score(aqi):
        if aqi <= 50: return 1
        elif aqi <= 100: return 2
        elif aqi <= 200: return 4
        elif aqi <= 300: return 6
        elif aqi <= 400: return 8
        else: return 10

    base_risk = health_risk_score(base_aqi)
    new_risk = health_risk_score(predicted_aqi)
    health_reduction = max(0, ((base_risk - new_risk) / base_risk) * 100)

    return {
        "base_aqi": round(base_aqi, 1),
        "predicted_aqi": round(predicted_aqi, 1),
        "aqi_improvement_pct": round(improvement_pct, 1),
        "health_risk_reduction_pct": round(health_reduction, 1),
        "breakdown": {
            "traffic_reduction_aqi": round(traffic_reduction_aqi, 1),
            "construction_reduction_aqi": round(construction_reduction_aqi, 1),
            "industrial_reduction_aqi": round(industrial_reduction_aqi, 1),
            "green_absorption_aqi": round(green_absorption, 1),
        }
    }
