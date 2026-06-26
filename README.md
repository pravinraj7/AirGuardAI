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

---

## 🚀 Features

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
└── README.md
```

---

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
```

---

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
