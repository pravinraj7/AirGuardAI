# 🌬️ AirGuard AI

**AI-powered Air Quality Intelligence Platform for Indian Cities**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **📊 AQI Dashboard** | Real-time AQI, trend analysis, pollutant radar charts, monthly heatmaps |
| **🤖 ML Forecasting** | XGBoost + Random Forest + Gradient Boosting → 24h/48h/72h AQI predictions |
| **🏭 Pollution Sources** | Traffic / Construction / Industry / Waste Burning attribution |
| **💊 Health Advisory** | AQI-based, group-specific health recommendations |
| **🏙️ City Comparison** | Multi-city AQI trends, boxplots, stat tables |
| **🔧 Digital Twin** | Intervention simulator — predict AQI from policy changes |
| **🗺️ Interactive Maps** | Folium heatmaps with city coordinates |
| **⚡ FastAPI Backend** | RESTful API with Swagger docs at `/docs` |
| **🐳 Docker Ready** | Full Docker + Docker Compose support |

---

## 📁 Project Structure

```
AirGuardAI/
├── 📂 data/                    # Raw datasets
│   ├── city_day.csv            # Primary dataset (29,531 rows, 26 cities)
│   ├── city_hour.csv           # Hourly AQI data
│   ├── traffic.csv             # Traffic density data
│   ├── pollution_sources.csv   # Source attribution (traffic/construction/industry/waste)
│   └── city_coordinates.csv    # Lat/lon for 50 cities
│
├── 📂 backend/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading & preprocessing
│   ├── ml_models.py            # XGBoost, RF, GB models + Digital Twin simulator
│   ├── visualizations.py       # Plotly charts + Folium maps
│   └── api.py                  # FastAPI REST endpoints
│
├── 📂 models/                  # Saved ML models (auto-generated)
│   ├── xgboost_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   └── scaler.pkl
│
├── 📂 scripts/
│   ├── train_models.py         # Standalone model training
│   └── explore_data.py         # Data exploration report
│
├── 📂 tests/
│   └── test_airguard.py        # Pytest test suite (25+ tests)
│
├── 📂 .streamlit/
│   └── config.toml             # Dark theme configuration
│
├── app.py                      # 🎯 Main Streamlit Application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image
├── docker-compose.yml          # Multi-service orchestration
└── README.md
```

---

## 📊 Dataset Overview

| File | Records | Coverage |
|------|---------|----------|
| `city_day.csv` | 29,531 rows | 26 cities, 2015–2020 |
| `city_hour.csv` | ~600K rows | 26 cities, hourly |
| `traffic.csv` | 300 rows | 5 cities |
| `pollution_sources.csv` | 300 rows | 15 cities |
| `city_coordinates.csv` | 50 rows | 50 cities |

**Pollutants tracked**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene

---

## ⚡ Quick Start

### Option 1: Local Setup (Recommended)

```bash
# 1. Clone / navigate to project
cd AirGuardAI

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-train ML models
python scripts/train_models.py

# 5. Launch Streamlit app
streamlit run app.py

# 6. (Optional) Launch FastAPI in another terminal
uvicorn backend.api:app --reload --port 8000
```

### Option 2: Docker

```bash
# Build and launch both services
docker-compose up --build

# Streamlit → http://localhost:8501
# FastAPI   → http://localhost:8000
# API Docs  → http://localhost:8000/docs
```

---

## 🌐 FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/cities` | List all cities |
| GET | `/api/aqi/latest` | Latest AQI for all/one city |
| GET | `/api/aqi/trend?city=Delhi&days=90` | AQI trend data |
| GET | `/api/aqi/monthly?city=Delhi` | Monthly averages |
| GET | `/api/stats/top-polluted?n=10` | Top N polluted cities |
| POST | `/api/forecast` | AQI forecast (24h/48h/72h) |
| GET | `/api/pollution-sources?city=Delhi` | Source attribution |
| POST | `/api/simulator` | Digital Twin simulation |
| GET | `/api/health-advisory?city=Delhi` | Health advisory |
| GET | `/api/coordinates` | City coords + AQI for map |
| POST | `/api/train-models` | Train & save ML models |

Full interactive docs at: **http://localhost:8000/docs**

---

## 🤖 ML Models

### Features Used
- **Pollutant measurements**: PM2.5, PM10, NO2, SO2, O3, CO, NOx, Benzene, Toluene, Xylene
- **Lag features**: AQI_lag1, AQI_lag7
- **Rolling average**: AQI_rolling7
- **Time features**: Month, DayOfWeek, Quarter

### Model Performance (typical)
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| XGBoost | ~12–18 | ~25–35 | ~0.92–0.96 |
| Random Forest | ~15–22 | ~28–40 | ~0.89–0.94 |
| Gradient Boosting | ~14–20 | ~27–38 | ~0.90–0.95 |

---

## 🔧 Digital Twin Simulator

The simulator uses empirically derived pollution contribution coefficients:

| Source | Contribution | Coefficient |
|--------|-------------|-------------|
| Traffic | ~35% of urban AQI | 0.35 |
| Industry | ~30% | 0.30 |
| Construction | ~15% | 0.15 |
| Green Cover | Absorbs ~8% | 0.08 |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test class
pytest tests/test_airguard.py::TestDigitalTwin -v
```

---

## 🎨 AQI Index Reference

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0–50 | Good | 🟢 Green | Minimal |
| 51–100 | Satisfactory | 🟡 Yellow-Green | Acceptable |
| 101–200 | Moderate | 🟡 Yellow | Sensitive groups affected |
| 201–300 | Poor | 🟠 Orange | Everyone affected |
| 301–400 | Very Poor | 🔴 Red | Serious health effects |
| 401–500 | Severe | 🟣 Purple | Emergency conditions |

---

## 👥 Team

Built for **Hackathon 2024** — AirGuard AI Team 🚀

---

## 📄 License

MIT License — free to use and modify.
