"""
Tests for Champion and Challenger Models
"""

import pytest
import numpy as np


class TestChampionModel:
    """Tests for Champion model."""

    def test_champion_initialization(self, champion_model):
        """Champion model should initialize with correct version."""
        assert champion_model.version == "2.1.0"
        assert champion_model.model_name == "ChampionV2"

    def test_champion_predict_returns_required_fields(self, champion_model, sample_features):
        """Champion prediction should return all required fields."""
        result = champion_model.predict(sample_features)
        
        required_fields = [
            "churn_probability",
            "churn_prediction",
            "model_version",
            "model_name",
            "confidence",
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_champion_probability_range(self, champion_model, sample_features):
        """Churn probability should be between 0 and 1."""
        result = champion_model.predict(sample_features)
        assert 0 <= result["churn_probability"] <= 1

    def test_champion_prediction_is_boolean(self, champion_model, sample_features):
        """Churn prediction should be boolean."""
        result = champion_model.predict(sample_features)
        assert isinstance(result["churn_prediction"], bool)

    def test_champion_consistent_predictions(self, champion_model, sample_features):
        """Model should give consistent predictions for same input."""
        # Note: Model has slight randomness, but predictions should be similar
        results = [champion_model.predict(sample_features) for _ in range(10)]
        probabilities = [r["churn_probability"] for r in results]
        
        # Check variance is reasonable
        assert np.std(probabilities) < 0.1


class TestChallengerModel:
    """Tests for Challenger model."""

    def test_challenger_initialization(self, challenger_model):
        """Challenger model should initialize with correct version."""
        assert challenger_model.version == "3.0.0-beta"
        assert challenger_model.model_name == "ChallengerV3"

    def test_challenger_predict_returns_required_fields(self, challenger_model, sample_features):
        """Challenger prediction should return all required fields."""
        result = challenger_model.predict(sample_features)
        
        required_fields = [
            "churn_probability",
            "churn_prediction",
            "model_version",
            "model_name",
            "confidence",
            "feature_interactions_used",
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_challenger_uses_interactions(self, challenger_model, sample_features):
        """Challenger should use feature interactions."""
        result = challenger_model.predict(sample_features)
        assert result["feature_interactions_used"] is True

    def test_challenger_probability_range(self, challenger_model, sample_features):
        """Churn probability should be between 0 and 1."""
        result = challenger_model.predict(sample_features)
        assert 0 <= result["churn_probability"] <= 1


class TestModelComparison:
    """Tests comparing Champion and Challenger models."""

    def test_models_produce_different_predictions(self, champion_model, challenger_model, sample_features):
        """Champion and Challenger should produce different (but similar) predictions."""
        champion_result = champion_model.predict(sample_features)
        challenger_result = challenger_model.predict(sample_features)
        
        # Predictions should be similar but not identical
        prob_diff = abs(
            champion_result["churn_probability"] - 
            challenger_result["churn_probability"]
        )
        
        # Should be within 0.3 of each other
        assert prob_diff < 0.3

    def test_high_risk_customer_detection(self, champion_model, challenger_model):
        """Both models should identify high-risk customers."""
        high_risk_features = {
            "customer_id": "HIGH_RISK_001",
            "days_since_last_login": 60,  # Haven't logged in for 2 months
            "login_frequency_30d": 1.0,   # Very low login frequency
            "session_duration_avg": 2.0,  # Very short sessions
            "total_transactions_90d": 0,  # No transactions
            "transaction_value_avg": 0.0,
            "support_tickets_30d": 5,     # Many support tickets
            "subscription_tenure_days": 30,  # New customer
            "satisfaction_score": 3.0,    # Low satisfaction
        }
        
        champion_result = champion_model.predict(high_risk_features)
        challenger_result = challenger_model.predict(high_risk_features)
        
        # Both should identify as higher risk
        assert champion_result["churn_probability"] > 0.4
        assert challenger_result["churn_probability"] > 0.4

    def test_low_risk_customer_detection(self, champion_model, challenger_model):
        """Both models should identify low-risk customers."""
        low_risk_features = {
            "customer_id": "LOW_RISK_001",
            "days_since_last_login": 1,    # Just logged in
            "login_frequency_30d": 25.0,   # Very active
            "session_duration_avg": 45.0,  # Long sessions
            "total_transactions_90d": 20,  # Many transactions
            "transaction_value_avg": 500.0,
            "support_tickets_30d": 0,      # No support issues
            "subscription_tenure_days": 1000,  # Long-term customer
            "satisfaction_score": 9.5,     # Very satisfied
        }
        
        champion_result = champion_model.predict(low_risk_features)
        challenger_result = challenger_model.predict(low_risk_features)
        
        # Both should identify as lower risk
        assert champion_result["churn_probability"] < 0.5
        assert challenger_result["churn_probability"] < 0.5


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_register_model(self, model_registry, champion_model):
        """Model registry should register models."""
        model_id = model_registry.register_model(champion_model, role="champion")
        assert model_id is not None
        assert model_registry.current_champion == model_id

    def test_get_champion(self, model_registry, champion_model):
        """Registry should return current champion."""
        model_registry.register_model(champion_model, role="champion")
        retrieved = model_registry.get_champion()
        assert retrieved is not None
        assert retrieved.version == champion_model.version

    def test_promote_challenger(self, model_registry, champion_model, challenger_model):
        """Registry should promote challenger to champion."""
        model_registry.register_model(champion_model, role="champion")
        model_registry.register_model(challenger_model, role="challenger")
        
        old_champion = model_registry.current_champion
        model_registry.promote_challenger()
        
        assert model_registry.current_champion != old_champion
        assert model_registry.current_challenger is None

    def test_list_models(self, model_registry, champion_model, challenger_model):
        """Registry should list all models."""
        model_registry.register_model(champion_model, role="champion")
        model_registry.register_model(challenger_model, role="challenger")
        
        models = model_registry.list_models()
        assert len(models) == 2


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def champion_model():
    """Create champion model instance."""
    from src.models import ChampionModel
    return ChampionModel()


@pytest.fixture
def challenger_model():
    """Create challenger model instance."""
    from src.models import ChallengerModel
    return ChallengerModel()


@pytest.fixture
def model_registry():
    """Create model registry instance."""
    from src.models import ModelRegistry
    return ModelRegistry()


@pytest.fixture
def sample_features():
    """Sample customer features for testing."""
    return {
        "customer_id": "TEST_CUST_001",
        "days_since_last_login": 7,
        "login_frequency_30d": 12.5,
        "session_duration_avg": 25.0,
        "total_transactions_90d": 8,
        "transaction_value_avg": 150.0,
        "support_tickets_30d": 1,
        "subscription_tenure_days": 365,
        "satisfaction_score": 7.5,
    }
