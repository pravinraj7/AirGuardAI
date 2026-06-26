<<<<<<< HEAD
# 🌬️ AirGuard AI

**AI-powered Air Quality Intelligence Platform for Indian Cities**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)
=======
# 🌬️ AirGuard AI – AI-Powered Urban Air Quality Intelligence Platform

> **An AI-powered decision support platform that helps cities monitor, predict, and reduce air pollution through intelligent analytics, forecasting, and Digital Twin simulation.**

---

## 📌 Problem Statement

Urban air pollution is one of the biggest public health challenges in India. While cities have access to air quality monitoring stations, they often lack intelligent systems that can:

* Predict future AQI
* Identify pollution sources
* Recommend actionable interventions
* Simulate the impact of pollution-control strategies

**AirGuard AI** transforms raw environmental data into actionable intelligence for smart city administrators.
>>>>>>> 72893e3f957b480199c2d71f4b0fa29f452184c1

---

## 🚀 Features

<<<<<<< HEAD
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
=======
### 🏠 Dashboard

* Real-time AQI overview
* PM2.5 & PM10 analytics
* AQI trend visualization
* Top polluted cities
* Pollutant radar charts
* Monthly heatmaps

### 📈 AQI Forecasting

* XGBoost Model
* Random Forest Model
* Gradient Boosting Model
* 24-hour AQI prediction
* 48-hour AQI prediction
* 72-hour AQI prediction
* Feature importance analysis

### 🏭 Pollution Source Attribution

* Traffic contribution analysis
* Construction contribution analysis
* Industrial emission analysis
* Waste burning contribution
* Interactive pie charts and progress indicators

### 💊 Health Advisory

Personalized health recommendations for:

* Children
* Elderly
* Asthma patients
* Outdoor workers
* Athletes

### 🏙️ Multi-City Comparison

* Compare AQI across Indian cities
* Historical AQI trends
* Statistical comparison
* Pollution ranking

### 🔧 Digital Twin Simulator

Simulate pollution control strategies before implementation.

Supports:

* Traffic Reduction
* Construction Activity Reduction
* Industrial Emission Reduction
* Green Cover Increase

Outputs:

* Predicted AQI
* AQI Improvement %
* Health Risk Reduction
* Pollution Reduction Breakdown

### 🤖 AI Action Recommendation Engine

Generate AI-powered intervention strategies including:

* Primary pollution source identification
* Recommended municipal actions
* Expected AQI reduction
* Priority level
* Health impact estimation

### 🗺️ Interactive Air Quality Map

* AQI heatmaps
* Pollution hotspots
* City markers
* Interactive Folium visualization

---

# 🧠 Machine Learning Models

| Model             | Purpose        |
| ----------------- | -------------- |
| XGBoost           | AQI Prediction |
| Random Forest     | AQI Prediction |
| Gradient Boosting | AQI Prediction |

### Model Performance

| Model             | R² Score | MAE  | RMSE |
| ----------------- | -------- | ---- | ---- |
| XGBoost           | 0.9435   | 16.4 | 24.7 |
| Random Forest     | 0.9459   | 15.4 | 24.1 |
| Gradient Boosting | 0.9413   | 16.5 | 25.1 |

---

# 📊 Datasets Used

* city_day.csv
* city_hour.csv
* traffic.csv
* pollution_sources.csv
* city_coordinates.csv
* OpenWeather API

Dataset Size:

* 29,531 Records
* 26 Indian Cities
* 14 Air Pollutants

---

# 🛠️ Technology Stack

## Frontend

* Streamlit

## Backend

* FastAPI

## Machine Learning

* Scikit-learn
* XGBoost
* Random Forest
* Gradient Boosting

## Visualization

* Plotly
* Folium

## Database

* CSV-based Data Processing

## Deployment

* Docker
* Docker Compose

---

# 📁 Project Structure

```text
AirGuardAI/
│
├── app.py
├── backend/
│   ├── api.py
│   ├── data_loader.py
│   ├── ml_models.py
│   └── visualizations.py
│
├── data/
│   ├── city_day.csv
│   ├── city_hour.csv
│   ├── traffic.csv
│   ├── pollution_sources.csv
│   └── city_coordinates.csv
│
├── tests/
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
>>>>>>> 72893e3f957b480199c2d71f4b0fa29f452184c1
└── README.md
```

---

<<<<<<< HEAD
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
=======
# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AirGuardAI.git

cd AirGuardAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Run the FastAPI backend:

```bash
uvicorn backend.api:app --reload --port 8000
```

Open:

Streamlit

```
http://localhost:8501
```

FastAPI Documentation

```
http://localhost:8000/docs
```

---

# 🐳 Docker

Build and run:

```bash
docker-compose up --build
>>>>>>> 72893e3f957b480199c2d71f4b0fa29f452184c1
```

---

<<<<<<< HEAD
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
=======
# 📈 Future Enhancements

* Live AQI sensor integration
* Satellite data integration
* IoT-enabled pollution monitoring
* Mobile application
* AI-powered multilingual chatbot
* Predictive emergency pollution alerts
* Smart traffic signal integration

---

# 📜 License

This project is developed for educational, research, and hackathon purposes.
>>>>>>> 72893e3f957b480199c2d71f4b0fa29f452184c1
