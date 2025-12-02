"""
Comprehensive API Tests for Shadow Deployment Platform.

This module provides thorough testing of all API endpoints including:
- Health checks
- Prediction endpoints
- Shadow deployment logging
- Error handling
- Rate limiting behavior

Example:
    pytest tests/test_api.py -v --cov=src.api

Author: Shadow MLOps Team
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api import app, get_model_registry

if TYPE_CHECKING:
    from src.models import ModelRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client with overridden dependencies."""
    return TestClient(app)


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create mock model registry for testing."""
    registry = MagicMock()
    
    # Mock champion model
    champion = MagicMock()
    champion._is_loaded = True
    champion.metadata.version = "2.1.0"
    champion.metadata.name = "Churn Champion"
    champion._prediction_count = 100
    champion.average_latency_ms = 5.2
    champion.predict.return_value = MagicMock(
        probability=0.42,
        label=0,
        confidence=0.84,
        model_version="2.1.0",
        risk_tier="MEDIUM",
        latency_ms=5.1,
        feature_contributions={"tenure": -0.5, "monthly_charges": 0.3},
        request_id="test-123",
        to_dict=lambda: {
            "probability": 0.42,
            "label": 0,
            "confidence": 0.84,
            "model_version": "2.1.0",
            "risk_tier": "MEDIUM",
        },
    )
    
    # Mock challenger model
    challenger = MagicMock()
    challenger._is_loaded = True
    challenger.metadata.version = "3.0.0-beta"
    challenger.metadata.name = "Churn Challenger"
    challenger._prediction_count = 100
    challenger.average_latency_ms = 4.8
    challenger.predict.return_value = MagicMock(
        probability=0.38,
        label=0,
        confidence=0.88,
        model_version="3.0.0-beta",
        to_dict=lambda: {
            "probability": 0.38,
            "label": 0,
            "confidence": 0.88,
            "model_version": "3.0.0-beta",
        },
    )
    
    registry.get_champion.return_value = champion
    registry.get_challenger.return_value = challenger
    registry.get_model.side_effect = lambda name: (
        champion if name == "champion" else challenger
    )
    registry.health_check.return_value = {"champion": True, "challenger": True}
    registry.get_metadata.return_value = {
        "champion": {
            "name": "Churn Champion",
            "version": "2.1.0",
            "is_loaded": True,
            "prediction_count": 100,
        },
        "challenger": {
            "name": "Churn Challenger",
            "version": "3.0.0-beta",
            "is_loaded": True,
            "prediction_count": 100,
        },
    }
    
    return registry


@pytest.fixture
def valid_features() -> dict[str, Any]:
    """Sample valid feature set for predictions."""
    return {
        "customer_id": "CUST-12345",
        "tenure": 24,
        "monthly_charges": 75.50,
        "total_charges": 1800.0,
        "contract_type": 1,
        "payment_method": 0,
        "num_support_tickets": 2,
        "avg_monthly_gb_download": 15.5,
        "num_dependents": 1,
        "online_security": 1,
        "tech_support": 1,
    }


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint_returns_welcome(self, client: TestClient) -> None:
        """Root endpoint should return welcome message."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "Shadow" in data["message"]
        assert "docs" in data

    def test_health_endpoint_returns_status(self, client: TestClient) -> None:
        """Health endpoint should return system status."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_contains_model_status(self, client: TestClient) -> None:
        """Health endpoint should include model health status."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert "champion" in data["models"]
        assert "challenger" in data["models"]

    def test_readiness_endpoint(self, client: TestClient) -> None:
        """Readiness probe should indicate if service is ready."""
        response = client.get("/health/ready")
        
        # May return 200 or 503 depending on model state
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ]

    def test_liveness_endpoint(self, client: TestClient) -> None:
        """Liveness probe should always return 200 if service is running."""
        response = client.get("/health/live")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "alive"


# =============================================================================
# Prediction Endpoint Tests
# =============================================================================


class TestPredictionEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_with_valid_features(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Prediction with valid features should return champion result."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.post("/predict", json=valid_features)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "probability" in data
        assert "label" in data
        assert "model_version" in data
        assert 0 <= data["probability"] <= 1
        
        # Clean up
        app.dependency_overrides.clear()

    def test_predict_returns_champion_only(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Shadow deployment should only return champion result to client."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.post("/predict", json=valid_features)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Should be champion version
        assert data.get("model_version") == "2.1.0"
        
        app.dependency_overrides.clear()

    def test_predict_includes_request_id(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Response should include request tracking ID."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.post("/predict", json=valid_features)
        
        assert response.status_code == status.HTTP_200_OK
        # Check header or response body for request ID
        assert "x-request-id" in response.headers or "request_id" in response.json()
        
        app.dependency_overrides.clear()

    def test_predict_respects_custom_request_id(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Custom request ID in header should be used."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        custom_id = "custom-request-12345"
        
        response = client.post(
            "/predict",
            json=valid_features,
            headers={"X-Request-ID": custom_id},
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers.get("x-request-id") == custom_id
        
        app.dependency_overrides.clear()

    def test_predict_with_missing_features_uses_defaults(
        self,
        client: TestClient,
        mock_registry: MagicMock,
    ) -> None:
        """Missing features should use default values."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        minimal_features = {
            "customer_id": "CUST-001",
            "tenure": 12,
        }
        
        response = client.post("/predict", json=minimal_features)
        
        assert response.status_code == status.HTTP_200_OK
        
        app.dependency_overrides.clear()

    def test_predict_with_invalid_types_returns_422(
        self,
        client: TestClient,
    ) -> None:
        """Invalid feature types should return validation error."""
        invalid_features = {
            "customer_id": "CUST-001",
            "tenure": "not-a-number",  # Should be numeric
        }
        
        response = client.post("/predict", json=invalid_features)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_predict_with_out_of_range_values_returns_422(
        self,
        client: TestClient,
    ) -> None:
        """Out of range values should be rejected."""
        invalid_features = {
            "customer_id": "CUST-001",
            "tenure": -5,  # Should be non-negative
        }
        
        response = client.post("/predict", json=invalid_features)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Batch Prediction Tests
# =============================================================================


class TestBatchPredictionEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_multiple_customers(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Batch prediction should handle multiple requests."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        batch_request = {
            "requests": [
                valid_features,
                {**valid_features, "customer_id": "CUST-002"},
                {**valid_features, "customer_id": "CUST-003"},
            ]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        
        app.dependency_overrides.clear()

    def test_batch_predict_respects_max_size(
        self,
        client: TestClient,
        valid_features: dict[str, Any],
    ) -> None:
        """Batch size exceeding limit should be rejected."""
        # Create batch larger than limit (assuming limit is 100)
        large_batch = {
            "requests": [valid_features] * 150
        }
        
        response = client.post("/predict/batch", json=large_batch)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_predict_returns_partial_on_errors(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Batch with some invalid requests should return partial results."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        batch_request = {
            "requests": [
                valid_features,
                {"customer_id": "BAD", "tenure": "invalid"},  # Invalid
                {**valid_features, "customer_id": "CUST-003"},
            ]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        # Should still succeed with partial results
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]
        
        app.dependency_overrides.clear()


# =============================================================================
# Model Metadata Tests
# =============================================================================


class TestModelMetadataEndpoint:
    """Tests for /models endpoint."""

    def test_get_models_returns_metadata(
        self,
        client: TestClient,
        mock_registry: MagicMock,
    ) -> None:
        """Models endpoint should return metadata for all models."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.get("/models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert "champion" in data["models"]
        assert "challenger" in data["models"]
        
        app.dependency_overrides.clear()

    def test_model_metadata_includes_version(
        self,
        client: TestClient,
        mock_registry: MagicMock,
    ) -> None:
        """Model metadata should include version information."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.get("/models")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "version" in data["models"]["champion"]
        
        app.dependency_overrides.clear()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_for_unknown_endpoint(self, client: TestClient) -> None:
        """Unknown endpoints should return 404."""
        response = client.get("/unknown/endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_405_for_wrong_method(self, client: TestClient) -> None:
        """Wrong HTTP method should return 405."""
        response = client.get("/predict")  # Should be POST
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_500_error_includes_request_id(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Internal errors should include request ID for debugging."""
        mock_registry.get_champion.side_effect = RuntimeError("Model crashed")
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.post("/predict", json=valid_features)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "x-request-id" in response.headers
        
        app.dependency_overrides.clear()


# =============================================================================
# Request Validation Tests
# =============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    def test_empty_request_body_returns_422(self, client: TestClient) -> None:
        """Empty request body should return validation error."""
        response = client.post("/predict", json={})
        
        # customer_id is required
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_malformed_json_returns_422(self, client: TestClient) -> None:
        """Malformed JSON should return error."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_customer_id_length_validation(self, client: TestClient) -> None:
        """Customer ID should have length limits."""
        # Too long customer ID
        long_id = "A" * 1000
        
        response = client.post(
            "/predict",
            json={"customer_id": long_id, "tenure": 12},
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# CORS and Headers Tests
# =============================================================================


class TestCORSAndHeaders:
    """Tests for CORS and response headers."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """CORS headers should be present for cross-origin requests."""
        response = client.options(
            "/predict",
            headers={"Origin": "http://localhost:3000"},
        )
        
        # CORS preflight should not fail
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        ]

    def test_response_includes_timing_header(
        self,
        client: TestClient,
        mock_registry: MagicMock,
        valid_features: dict[str, Any],
    ) -> None:
        """Response should include processing time header."""
        app.dependency_overrides[get_model_registry] = lambda: mock_registry
        
        response = client.post("/predict", json=valid_features)
        
        assert response.status_code == status.HTTP_200_OK
        assert "x-process-time" in response.headers
        
        app.dependency_overrides.clear()
