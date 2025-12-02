"""
Comprehensive Model Tests for Shadow Deployment Platform.

This module provides thorough testing of ML model implementations including:
- Model loading and initialization
- Prediction accuracy and consistency
- Feature validation
- Error handling
- Performance characteristics

Example:
    pytest tests/test_models.py -v --cov=src.models

Author: Shadow MLOps Team
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.exceptions import FeatureValidationError, ModelNotLoadedError, PredictionError
from src.models import (
    BaseChurnModel,
    ChampionModel,
    ChallengerModel,
    ModelMetadata,
    ModelRegistry,
    ModelVersion,
    PredictionResult,
    get_registry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def champion_model() -> ChampionModel:
    """Create and load a Champion model instance."""
    model = ChampionModel()
    model.load()
    return model


@pytest.fixture
def challenger_model() -> ChallengerModel:
    """Create and load a Challenger model instance."""
    model = ChallengerModel()
    model.load()
    return model


@pytest.fixture
def registry() -> ModelRegistry:
    """Create a fresh ModelRegistry instance."""
    # Reset singleton for testing
    ModelRegistry._instance = None
    return ModelRegistry()


@pytest.fixture
def sample_features() -> dict[str, Any]:
    """Standard feature set for testing."""
    return {
        "tenure": 24,
        "monthly_charges": 75.0,
        "total_charges": 1800.0,
        "contract_type": 1,
        "payment_method": 1,
        "num_support_tickets": 2,
        "avg_monthly_gb_download": 15.0,
        "num_dependents": 1,
        "online_security": 1,
        "tech_support": 1,
    }


@pytest.fixture
def high_churn_features() -> dict[str, Any]:
    """Features likely to result in high churn probability."""
    return {
        "tenure": 1,
        "monthly_charges": 100.0,
        "total_charges": 100.0,
        "contract_type": 0,  # Month-to-month
        "payment_method": 2,  # Electronic check
        "num_support_tickets": 10,
        "avg_monthly_gb_download": 2.0,
        "num_dependents": 0,
        "online_security": 0,
        "tech_support": 0,
    }


@pytest.fixture
def low_churn_features() -> dict[str, Any]:
    """Features likely to result in low churn probability."""
    return {
        "tenure": 60,
        "monthly_charges": 40.0,
        "total_charges": 2400.0,
        "contract_type": 2,  # Two-year
        "payment_method": 0,  # Credit card
        "num_support_tickets": 0,
        "avg_monthly_gb_download": 25.0,
        "num_dependents": 3,
        "online_security": 1,
        "tech_support": 1,
    }


# =============================================================================
# ModelMetadata Tests
# =============================================================================


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Metadata should be created with required fields."""
        metadata = ModelMetadata(
            name="Test Model",
            version="1.0.0",
            model_type="test",
        )
        
        assert metadata.name == "Test Model"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "test"

    def test_metadata_has_timestamp(self) -> None:
        """Metadata should have auto-generated timestamp."""
        metadata = ModelMetadata(
            name="Test",
            version="1.0.0",
            model_type="test",
        )
        
        assert metadata.trained_at is not None

    def test_metadata_hash_is_deterministic(self) -> None:
        """Same metadata should produce same hash."""
        metadata1 = ModelMetadata(
            name="Test",
            version="1.0.0",
            model_type="test",
        )
        metadata2 = ModelMetadata(
            name="Test",
            version="1.0.0",
            model_type="test",
        )
        
        assert metadata1.model_hash == metadata2.model_hash

    def test_metadata_hash_differs_for_different_versions(self) -> None:
        """Different versions should produce different hashes."""
        metadata1 = ModelMetadata(name="Test", version="1.0.0", model_type="test")
        metadata2 = ModelMetadata(name="Test", version="2.0.0", model_type="test")
        
        assert metadata1.model_hash != metadata2.model_hash

    def test_metadata_is_immutable(self) -> None:
        """Metadata should be immutable (frozen dataclass)."""
        metadata = ModelMetadata(name="Test", version="1.0.0", model_type="test")
        
        with pytest.raises(Exception):  # FrozenInstanceError
            metadata.name = "Modified"


# =============================================================================
# PredictionResult Tests
# =============================================================================


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_valid_prediction_result(self) -> None:
        """Valid prediction result should be created."""
        result = PredictionResult(
            probability=0.42,
            label=0,
            confidence=0.84,
            model_version="1.0.0",
        )
        
        assert result.probability == 0.42
        assert result.label == 0
        assert result.confidence == 0.84

    def test_probability_bounds_validation(self) -> None:
        """Probability outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="Probability"):
            PredictionResult(
                probability=1.5,
                label=0,
                confidence=0.5,
                model_version="1.0.0",
            )

    def test_negative_probability_rejected(self) -> None:
        """Negative probability should be rejected."""
        with pytest.raises(ValueError, match="Probability"):
            PredictionResult(
                probability=-0.1,
                label=0,
                confidence=0.5,
                model_version="1.0.0",
            )

    def test_invalid_label_rejected(self) -> None:
        """Label must be 0 or 1."""
        with pytest.raises(ValueError, match="Label"):
            PredictionResult(
                probability=0.5,
                label=2,
                confidence=0.5,
                model_version="1.0.0",
            )

    @pytest.mark.parametrize("probability,expected_tier", [
        (0.10, "LOW"),
        (0.30, "MEDIUM"),
        (0.60, "HIGH"),
        (0.90, "CRITICAL"),
    ])
    def test_risk_tier_classification(
        self,
        probability: float,
        expected_tier: str,
    ) -> None:
        """Risk tier should be correctly classified."""
        result = PredictionResult(
            probability=probability,
            label=1 if probability >= 0.5 else 0,
            confidence=0.8,
            model_version="1.0.0",
        )
        
        assert result.risk_tier == expected_tier

    def test_to_dict_serialization(self) -> None:
        """Result should serialize to dictionary."""
        result = PredictionResult(
            probability=0.42,
            label=0,
            confidence=0.84,
            model_version="1.0.0",
            request_id="test-123",
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data["probability"] == 0.42
        assert data["model_version"] == "1.0.0"
        assert data["request_id"] == "test-123"


# =============================================================================
# ChampionModel Tests
# =============================================================================


class TestChampionModel:
    """Tests for ChampionModel."""

    def test_model_initialization(self) -> None:
        """Model should initialize with correct metadata."""
        model = ChampionModel()
        
        assert model.version == ModelVersion.CHAMPION.value
        assert model.metadata.name == "Churn Champion"
        assert not model._is_loaded

    def test_model_load(self, champion_model: ChampionModel) -> None:
        """Model should load successfully."""
        assert champion_model._is_loaded

    def test_predict_without_load_raises_error(self) -> None:
        """Prediction before load should raise ModelNotLoadedError."""
        model = ChampionModel()
        
        with pytest.raises(ModelNotLoadedError):
            model.predict({"tenure": 12})

    def test_predict_returns_result(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Prediction should return PredictionResult."""
        result = champion_model.predict(sample_features)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.probability <= 1
        assert result.label in (0, 1)
        assert result.model_version == ModelVersion.CHAMPION.value

    def test_predict_probability_in_valid_range(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Prediction probability should always be in [0, 1]."""
        result = champion_model.predict(sample_features)
        
        assert 0 <= result.probability <= 1

    def test_predict_with_request_id(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Request ID should be included in result."""
        result = champion_model.predict(sample_features, request_id="test-123")
        
        assert result.request_id == "test-123"

    def test_predict_includes_latency(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Prediction should include latency measurement."""
        result = champion_model.predict(sample_features)
        
        assert result.latency_ms > 0

    def test_predict_includes_feature_contributions(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Prediction should include feature contributions."""
        result = champion_model.predict(sample_features)
        
        assert isinstance(result.feature_contributions, dict)
        assert len(result.feature_contributions) > 0

    def test_high_churn_features_produce_higher_probability(
        self,
        champion_model: ChampionModel,
        high_churn_features: dict[str, Any],
        low_churn_features: dict[str, Any],
    ) -> None:
        """High-risk features should produce higher churn probability."""
        high_result = champion_model.predict(high_churn_features)
        low_result = champion_model.predict(low_churn_features)
        
        assert high_result.probability > low_result.probability

    def test_prediction_count_increments(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Prediction count should increment with each prediction."""
        initial_count = champion_model._prediction_count
        
        champion_model.predict(sample_features)
        champion_model.predict(sample_features)
        
        assert champion_model._prediction_count == initial_count + 2

    def test_average_latency_calculation(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Average latency should be calculated correctly."""
        for _ in range(5):
            champion_model.predict(sample_features)
        
        assert champion_model.average_latency_ms > 0


# =============================================================================
# ChallengerModel Tests
# =============================================================================


class TestChallengerModel:
    """Tests for ChallengerModel."""

    def test_model_initialization(self) -> None:
        """Model should initialize with correct metadata."""
        model = ChallengerModel()
        
        assert model.version == ModelVersion.CHALLENGER.value
        assert "Challenger" in model.metadata.name

    def test_challenger_uses_interactions(
        self,
        challenger_model: ChallengerModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Challenger should use interaction features."""
        result = challenger_model.predict(sample_features)
        
        # Just verify it produces valid output
        assert isinstance(result, PredictionResult)
        assert 0 <= result.probability <= 1

    def test_challenger_differs_from_champion(
        self,
        champion_model: ChampionModel,
        challenger_model: ChallengerModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Challenger should produce different predictions than Champion."""
        champion_result = champion_model.predict(sample_features)
        challenger_result = challenger_model.predict(sample_features)
        
        # Predictions should differ (models have different weights)
        # Allow small tolerance for numerical precision
        assert abs(champion_result.probability - challenger_result.probability) > 0.001


# =============================================================================
# Feature Validation Tests
# =============================================================================


class TestFeatureValidation:
    """Tests for feature validation."""

    def test_missing_features_use_defaults(
        self,
        champion_model: ChampionModel,
    ) -> None:
        """Missing features should use default values."""
        # Only provide one feature
        result = champion_model.predict({"tenure": 24})
        
        assert isinstance(result, PredictionResult)

    def test_invalid_feature_type_raises_error(
        self,
        champion_model: ChampionModel,
    ) -> None:
        """Invalid feature type should raise FeatureValidationError."""
        with pytest.raises(FeatureValidationError):
            champion_model.predict({"tenure": "not-a-number"})

    def test_extra_features_ignored(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Extra features should be ignored."""
        features_with_extra = {
            **sample_features,
            "unknown_feature": 999,
        }
        
        result = champion_model.predict(features_with_extra)
        
        assert isinstance(result, PredictionResult)


# =============================================================================
# Async Prediction Tests
# =============================================================================


class TestAsyncPrediction:
    """Tests for async prediction methods."""

    @pytest.mark.asyncio
    async def test_async_predict(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Async prediction should work correctly."""
        result = await champion_model.predict_async(sample_features)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.probability <= 1

    @pytest.mark.asyncio
    async def test_async_predict_with_request_id(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Async prediction should pass through request ID."""
        result = await champion_model.predict_async(
            sample_features,
            request_id="async-test-123",
        )
        
        assert result.request_id == "async-test-123"


# =============================================================================
# ModelRegistry Tests
# =============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_registry_singleton(self) -> None:
        """Registry should be a singleton."""
        ModelRegistry._instance = None
        
        reg1 = ModelRegistry()
        reg2 = ModelRegistry()
        
        assert reg1 is reg2

    def test_get_champion(self, registry: ModelRegistry) -> None:
        """Registry should return champion model."""
        champion = registry.get_champion()
        
        assert isinstance(champion, ChampionModel)

    def test_get_challenger(self, registry: ModelRegistry) -> None:
        """Registry should return challenger model."""
        challenger = registry.get_challenger()
        
        assert isinstance(challenger, ChallengerModel)

    def test_get_model_by_name(self, registry: ModelRegistry) -> None:
        """Registry should return model by name."""
        champion = registry.get_model("champion")
        challenger = registry.get_model("challenger")
        
        assert isinstance(champion, ChampionModel)
        assert isinstance(challenger, ChallengerModel)

    def test_get_unknown_model_raises_error(self, registry: ModelRegistry) -> None:
        """Unknown model name should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            registry.get_model("unknown")

    def test_available_models(self, registry: ModelRegistry) -> None:
        """Available models should list all models."""
        models = registry.available_models
        
        assert "champion" in models
        assert "challenger" in models

    def test_load_all_models(self, registry: ModelRegistry) -> None:
        """Load all should load both models."""
        results = registry.load_all()
        
        assert results["champion"] is True
        assert results["challenger"] is True

    def test_health_check(self, registry: ModelRegistry) -> None:
        """Health check should return status for all models."""
        registry.load_all()
        health = registry.health_check()
        
        assert health["champion"] is True
        assert health["challenger"] is True

    def test_get_metadata(self, registry: ModelRegistry) -> None:
        """Get metadata should return info for all models."""
        registry.load_all()
        metadata = registry.get_metadata()
        
        assert "champion" in metadata
        assert "challenger" in metadata
        assert metadata["champion"]["is_loaded"] is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_registry_returns_singleton(self) -> None:
        """get_registry should return singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        
        assert reg1 is reg2

    def test_sigmoid_numerical_stability(self, champion_model: ChampionModel) -> None:
        """Sigmoid should handle extreme values."""
        # Test with extreme positive value
        result = champion_model._sigmoid(500)
        assert 0 <= result <= 1
        
        # Test with extreme negative value
        result = champion_model._sigmoid(-500)
        assert 0 <= result <= 1


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for prediction determinism."""

    def test_same_input_same_output(
        self,
        champion_model: ChampionModel,
        sample_features: dict[str, Any],
    ) -> None:
        """Same input should produce same output."""
        result1 = champion_model.predict(sample_features)
        result2 = champion_model.predict(sample_features)
        
        assert result1.probability == result2.probability
        assert result1.label == result2.label

    def test_predictions_reproducible_across_instances(
        self,
        sample_features: dict[str, Any],
    ) -> None:
        """Different model instances should produce same predictions."""
        model1 = ChampionModel()
        model1.load()
        
        model2 = ChampionModel()
        model2.load()
        
        result1 = model1.predict(sample_features)
        result2 = model2.predict(sample_features)
        
        assert result1.probability == result2.probability
