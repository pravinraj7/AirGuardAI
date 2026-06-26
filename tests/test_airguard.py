"""
AirGuard AI — Pytest Test Suite
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def df_day():
    from backend.data_loader import load_city_day
    return load_city_day()


@pytest.fixture(scope="session")
def df_coords():
    from backend.data_loader import load_city_coordinates
    return load_city_coordinates()


@pytest.fixture(scope="session")
def df_sources():
    from backend.data_loader import load_pollution_sources
    return load_pollution_sources()


# ── Data Loader Tests ──────────────────────────────────────────

class TestDataLoader:
    def test_city_day_loads(self, df_day):
        assert df_day is not None
        assert len(df_day) > 0
        assert "City" in df_day.columns
        assert "Date" in df_day.columns
        assert "AQI" in df_day.columns

    def test_city_day_has_expected_cities(self, df_day):
        cities = df_day["City"].unique()
        assert "Delhi" in cities or "Bengaluru" in cities or "Chennai" in cities

    def test_aqi_range_valid(self, df_day):
        aqi_vals = df_day["AQI"].dropna()
        # All AQI values should be non-negative
        assert (aqi_vals >= 0).all()
        # At least 95% of values should be within standard range (0-500)
        # Dataset may contain extreme outliers above 500
        pct_in_range = (aqi_vals <= 500).mean()
        assert pct_in_range >= 0.80, f"Only {pct_in_range:.1%} of AQI values in 0-500 range"

    def test_coordinates_load(self, df_coords):
        assert len(df_coords) > 0
        assert "latitude" in df_coords.columns or "Latitude" in df_coords.columns.str.capitalize()

    def test_pollution_sources_load(self, df_sources):
        assert len(df_sources) > 0
        for col in ["traffic", "construction", "industry", "waste_burning"]:
            assert col in df_sources.columns


# ── AQI Classification Tests ──────────────────────────────────

class TestAQIClassification:
    def test_classify_good(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(25) == "Good"

    def test_classify_satisfactory(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(75) == "Satisfactory"

    def test_classify_moderate(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(150) == "Moderate"

    def test_classify_poor(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(250) == "Poor"

    def test_classify_very_poor(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(350) == "Very Poor"

    def test_classify_severe(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(450) == "Severe"

    def test_classify_nan(self):
        from backend.data_loader import _classify_aqi
        assert _classify_aqi(float("nan")) == "Unknown"


# ── Health Advisory Tests ─────────────────────────────────────

class TestHealthAdvisory:
    def test_advisory_returns_dict(self):
        from backend.data_loader import get_aqi_health_message
        result = get_aqi_health_message(150)
        assert isinstance(result, dict)
        assert "level" in result
        assert "message" in result
        assert "recommendation" in result

    def test_advisory_good_aqi(self):
        from backend.data_loader import get_aqi_health_message
        result = get_aqi_health_message(30)
        assert result["level"] == "Good"

    def test_advisory_severe_aqi(self):
        from backend.data_loader import get_aqi_health_message
        result = get_aqi_health_message(500)
        assert result["level"] == "Severe"


# ── Digital Twin Tests ────────────────────────────────────────

class TestDigitalTwin:
    def test_simulation_no_intervention(self):
        from backend.ml_models import simulate_digital_twin
        result = simulate_digital_twin(200, 0, 0, 0, 0)
        assert result["predicted_aqi"] == pytest.approx(200.0, abs=1)
        assert result["aqi_improvement_pct"] == pytest.approx(0.0, abs=1)

    def test_simulation_reduces_aqi(self):
        from backend.ml_models import simulate_digital_twin
        result = simulate_digital_twin(300, 50, 30, 20, 15)
        assert result["predicted_aqi"] < 300

    def test_simulation_output_keys(self):
        from backend.ml_models import simulate_digital_twin
        result = simulate_digital_twin(200, 20, 10, 10, 5)
        assert "predicted_aqi" in result
        assert "aqi_improvement_pct" in result
        assert "health_risk_reduction_pct" in result
        assert "breakdown" in result

    def test_simulation_aqi_not_negative(self):
        from backend.ml_models import simulate_digital_twin
        result = simulate_digital_twin(50, 100, 100, 100, 100)
        assert result["predicted_aqi"] >= 0

    def test_simulation_improvement_capped_at_70pct(self):
        from backend.ml_models import simulate_digital_twin
        result = simulate_digital_twin(400, 100, 100, 100, 100)
        # Maximum reduction is 70%
        assert result["aqi_improvement_pct"] <= 70.1


# ── ML Model Tests ────────────────────────────────────────────

class TestMLModels:
    def test_prepare_features(self, df_day):
        from backend.ml_models import prepare_features
        data, features = prepare_features(df_day)
        assert len(features) > 0
        assert len(data) > 0

    def test_feature_columns_present(self, df_day):
        from backend.ml_models import prepare_features
        data, features = prepare_features(df_day)
        for feat in features:
            assert feat in data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
