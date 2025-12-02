"""
Tests for Shadow MLOps API
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should return 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client: TestClient):
        """Health endpoint should return status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_returns_model_info(self, client: TestClient):
        """Health endpoint should return model versions."""
        response = client.get("/health")
        data = response.json()
        assert "models" in data
        assert "champion" in data["models"]
        assert "challenger" in data["models"]


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_valid_request(self, client: TestClient, sample_features: dict):
        """Predict endpoint should return valid response for valid input."""
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200

    def test_predict_returns_required_fields(self, client: TestClient, sample_features: dict):
        """Predict response should contain all required fields."""
        response = client.post("/predict", json=sample_features)
        data = response.json()
        
        required_fields = [
            "request_id",
            "customer_id",
            "churn_probability",
            "churn_prediction",
            "risk_category",
            "model_version",
            "timestamp",
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_probability_range(self, client: TestClient, sample_features: dict):
        """Churn probability should be between 0 and 1."""
        response = client.post("/predict", json=sample_features)
        data = response.json()
        
        assert 0 <= data["churn_probability"] <= 1

    def test_predict_risk_categories(self, client: TestClient, sample_features: dict):
        """Risk category should be one of: low, medium, high."""
        response = client.post("/predict", json=sample_features)
        data = response.json()
        
        assert data["risk_category"] in ["low", "medium", "high"]

    def test_predict_invalid_request(self, client: TestClient):
        """Predict endpoint should return 422 for invalid input."""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_multiple_customers(self, client: TestClient, sample_features: dict):
        """Batch endpoint should handle multiple customers."""
        batch_request = {
            "customers": [
                sample_features,
                {**sample_features, "customer_id": "CUST_002"},
                {**sample_features, "customer_id": "CUST_003"},
            ]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 3


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_returns_200(self, client: TestClient):
        """Metrics endpoint should return 200 status."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_shadow_data(self, client: TestClient):
        """Metrics should contain shadow deployment metrics."""
        response = client.get("/metrics")
        data = response.json()
        
        assert "shadow_deployment_metrics" in data


class TestShadowEndpoints:
    """Tests for shadow deployment endpoints."""

    def test_shadow_logs_returns_200(self, client: TestClient):
        """Shadow logs endpoint should return 200 status."""
        response = client.get("/shadow/logs")
        assert response.status_code == 200

    def test_shadow_comparison_returns_200(self, client: TestClient):
        """Shadow comparison endpoint should return 200 status."""
        response = client.get("/shadow/comparison")
        assert response.status_code == 200


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Create test client."""
    # Import here to avoid issues if dependencies not installed
    try:
        from src.api import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI dependencies not installed")


@pytest.fixture
def sample_features():
    """Sample customer features for testing."""
    return {
        "customer_id": "CUST_TEST_001",
        "days_since_last_login": 7,
        "login_frequency_30d": 12.5,
        "session_duration_avg": 25.0,
        "total_transactions_90d": 8,
        "transaction_value_avg": 150.0,
        "support_tickets_30d": 1,
        "subscription_tenure_days": 365,
        "satisfaction_score": 7.5,
    }
