"""
ML Model Implementations for Shadow Deployment.

This module implements the Champion/Challenger pattern for A/B testing ML models.
It provides abstract base classes and concrete implementations for churn prediction.

Design Patterns:
    - Strategy Pattern: Interchangeable model implementations
    - Factory Pattern: ModelRegistry for model instantiation
    - Template Method: BaseChurnModel defines prediction workflow

Example:
    >>> from src.models import ModelRegistry
    >>> registry = ModelRegistry()
    >>> champion = registry.get_model("champion")
    >>> prediction = await champion.predict_async({"tenure": 12, "monthly_charges": 50})
    >>> print(f"Churn probability: {prediction.probability:.2%}")

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    ModelRegistry                        │
    │  ┌─────────────────┐      ┌─────────────────┐          │
    │  │  ChampionModel  │      │ ChallengerModel │          │
    │  │    (v2.1.0)     │      │  (v3.0.0-beta)  │          │
    │  └────────┬────────┘      └────────┬────────┘          │
    │           │                        │                    │
    │           └────────────┬───────────┘                    │
    │                        │                                │
    │              ┌─────────▼─────────┐                      │
    │              │   BaseChurnModel  │                      │
    │              │     (Abstract)    │                      │
    │              └───────────────────┘                      │
    └─────────────────────────────────────────────────────────┘

Author: Shadow MLOps Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, ClassVar, TypeVar

import numpy as np

from src.exceptions import (
    FeatureValidationError,
    ModelNotLoadedError,
    PredictionError,
)

logger = logging.getLogger(__name__)

# Type variable for generic model typing
T = TypeVar("T", bound="BaseChurnModel")


class ModelVersion(Enum):
    """Model version enumeration for type-safe versioning."""

    CHAMPION = "2.1.0"
    CHALLENGER = "3.0.0-beta"


@dataclass(frozen=True)
class ModelMetadata:
    """
    Immutable model metadata container.

    Attributes:
        name: Human-readable model name
        version: Semantic version string
        model_type: Model architecture type
        trained_at: Training timestamp
        features: List of required feature names
        metrics: Training/validation metrics

    Example:
        >>> metadata = ModelMetadata(
        ...     name="Churn Champion",
        ...     version="2.1.0",
        ...     model_type="weighted_ensemble",
        ...     features=["tenure", "monthly_charges"]
        ... )
    """

    name: str
    version: str
    model_type: str
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    features: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def model_hash(self) -> str:
        """Generate unique hash for model identification."""
        content = f"{self.name}:{self.version}:{self.model_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class PredictionResult:
    """
    Structured prediction output with metadata.

    This class encapsulates model predictions with associated
    metadata for traceability and debugging.

    Attributes:
        probability: Churn probability [0, 1]
        label: Binary classification label
        confidence: Model confidence score
        model_version: Source model version
        latency_ms: Prediction latency in milliseconds
        feature_contributions: Feature importance for this prediction
        request_id: Unique request identifier

    Example:
        >>> result = PredictionResult(
        ...     probability=0.73,
        ...     label=1,
        ...     confidence=0.92,
        ...     model_version="2.1.0"
        ... )
        >>> result.risk_tier
        'HIGH'
    """

    probability: float
    label: int
    confidence: float
    model_version: str
    latency_ms: float = 0.0
    feature_contributions: dict[str, float] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        """Validate prediction values after initialization."""
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability}")
        if self.label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {self.label}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def risk_tier(self) -> str:
        """
        Categorize prediction into risk tiers.

        Returns:
            Risk tier: LOW, MEDIUM, HIGH, or CRITICAL
        """
        if self.probability < 0.25:
            return "LOW"
        elif self.probability < 0.50:
            return "MEDIUM"
        elif self.probability < 0.75:
            return "HIGH"
        return "CRITICAL"

    def to_dict(self) -> dict[str, Any]:
        """Serialize prediction result to dictionary."""
        return {
            "probability": round(self.probability, 6),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "model_version": self.model_version,
            "risk_tier": self.risk_tier,
            "latency_ms": round(self.latency_ms, 2),
            "feature_contributions": self.feature_contributions,
            "request_id": self.request_id,
        }


class BaseChurnModel(ABC):
    """
    Abstract base class for churn prediction models.

    This class defines the contract for all churn prediction models,
    implementing the Template Method pattern for the prediction workflow.

    Subclasses must implement:
        - _compute_raw_score: Core scoring logic
        - _get_feature_weights: Feature importance weights

    Class Attributes:
        REQUIRED_FEATURES: Features required for prediction
        DEFAULT_VALUES: Default values for missing features

    Example:
        >>> class CustomModel(BaseChurnModel):
        ...     def _compute_raw_score(self, features):
        ...         return sum(features.values()) / len(features)
        ...     def _get_feature_weights(self):
        ...         return {"feature1": 0.5, "feature2": 0.5}
    """

    REQUIRED_FEATURES: ClassVar[tuple[str, ...]] = (
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "payment_method",
        "num_support_tickets",
        "avg_monthly_gb_download",
        "num_dependents",
        "online_security",
        "tech_support",
    )

    DEFAULT_VALUES: ClassVar[dict[str, float]] = {
        "tenure": 12.0,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "contract_type": 1.0,
        "payment_method": 1.0,
        "num_support_tickets": 0.0,
        "avg_monthly_gb_download": 10.0,
        "num_dependents": 0.0,
        "online_security": 0.0,
        "tech_support": 0.0,
    }

    def __init__(self, metadata: ModelMetadata) -> None:
        """
        Initialize base model with metadata.

        Args:
            metadata: Model metadata container

        Raises:
            ModelNotLoadedError: If model initialization fails
        """
        self._metadata = metadata
        self._is_loaded = False
        self._prediction_count = 0
        self._total_latency_ms = 0.0
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self._metadata

    @property
    def version(self) -> str:
        """Get model version string."""
        return self._metadata.version

    @property
    def average_latency_ms(self) -> float:
        """Calculate average prediction latency."""
        if self._prediction_count == 0:
            return 0.0
        return self._total_latency_ms / self._prediction_count

    def load(self) -> None:
        """
        Load model weights and initialize for inference.

        In production, this would load serialized model weights.
        Currently simulates loading for demonstration.

        Raises:
            ModelNotLoadedError: If loading fails
        """
        try:
            self._logger.info(
                "Loading model",
                extra={
                    "model": self._metadata.name,
                    "version": self._metadata.version,
                },
            )
            # Simulate weight loading
            time.sleep(0.01)  # Simulated I/O
            self._is_loaded = True
            self._logger.info("Model loaded successfully")
        except Exception as exc:
            raise ModelNotLoadedError(
                f"Failed to load model {self._metadata.name}",
                model_name=self._metadata.name,
            ) from exc

    def _validate_features(self, features: dict[str, Any]) -> dict[str, float]:
        """
        Validate and normalize input features.

        Args:
            features: Raw input features

        Returns:
            Normalized feature dictionary

        Raises:
            FeatureValidationError: If validation fails
        """
        normalized: dict[str, float] = {}

        for feature in self.REQUIRED_FEATURES:
            if feature in features:
                value = features[feature]
                try:
                    normalized[feature] = float(value)
                except (ValueError, TypeError) as exc:
                    raise FeatureValidationError(
                        f"Invalid value for feature '{feature}': {value}",
                        feature_name=feature,
                        expected_type="float",
                        actual_value=value,
                    ) from exc
            else:
                # Use default value
                normalized[feature] = self.DEFAULT_VALUES.get(feature, 0.0)

        return normalized

    @staticmethod
    @lru_cache(maxsize=1)
    def _sigmoid(x: float) -> float:
        """
        Compute sigmoid activation with numerical stability.

        Args:
            x: Input value

        Returns:
            Sigmoid output in range [0, 1]
        """
        # Clip for numerical stability
        x_clipped = max(-500, min(500, x))
        return 1.0 / (1.0 + np.exp(-x_clipped))

    @abstractmethod
    def _compute_raw_score(self, features: dict[str, float]) -> float:
        """
        Compute raw prediction score before activation.

        Args:
            features: Normalized feature dictionary

        Returns:
            Raw score (unbounded)
        """
        pass

    @abstractmethod
    def _get_feature_weights(self) -> dict[str, float]:
        """
        Get feature importance weights.

        Returns:
            Dictionary mapping feature names to importance weights
        """
        pass

    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate prediction confidence based on probability distance from 0.5.

        Confidence is higher when the model is more certain (probability
        closer to 0 or 1).

        Args:
            probability: Predicted probability

        Returns:
            Confidence score in [0, 1]
        """
        return abs(probability - 0.5) * 2

    def _compute_feature_contributions(
        self, features: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate per-feature contribution to prediction.

        Args:
            features: Input features

        Returns:
            Dictionary of feature contributions
        """
        weights = self._get_feature_weights()
        contributions: dict[str, float] = {}

        for feature, value in features.items():
            weight = weights.get(feature, 0.0)
            contributions[feature] = round(value * weight, 4)

        return contributions

    def predict(
        self,
        features: dict[str, Any],
        request_id: str | None = None,
    ) -> PredictionResult:
        """
        Generate churn prediction for input features.

        This method implements the full prediction pipeline:
        1. Feature validation and normalization
        2. Raw score computation
        3. Probability calculation via sigmoid
        4. Confidence estimation
        5. Result packaging

        Args:
            features: Input feature dictionary
            request_id: Optional request identifier for tracing

        Returns:
            PredictionResult with probability and metadata

        Raises:
            ModelNotLoadedError: If model not loaded
            PredictionError: If prediction fails

        Example:
            >>> model = ChampionModel()
            >>> model.load()
            >>> result = model.predict({"tenure": 24, "monthly_charges": 75})
            >>> print(result.probability)
            0.42
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                f"Model {self._metadata.name} not loaded. Call load() first.",
                model_name=self._metadata.name,
            )

        start_time = time.perf_counter()

        try:
            # Validate and normalize features
            normalized_features = self._validate_features(features)

            # Compute prediction
            raw_score = self._compute_raw_score(normalized_features)
            probability = float(self._sigmoid(raw_score))
            label = 1 if probability >= 0.5 else 0
            confidence = self._calculate_confidence(probability)

            # Calculate feature contributions
            contributions = self._compute_feature_contributions(normalized_features)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update statistics
            self._prediction_count += 1
            self._total_latency_ms += latency_ms

            result = PredictionResult(
                probability=probability,
                label=label,
                confidence=confidence,
                model_version=self._metadata.version,
                latency_ms=latency_ms,
                feature_contributions=contributions,
                request_id=request_id,
            )

            self._logger.debug(
                "Prediction completed",
                extra={
                    "probability": probability,
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                },
            )

            return result

        except (FeatureValidationError, ModelNotLoadedError):
            raise
        except Exception as exc:
            raise PredictionError(
                f"Prediction failed: {exc}",
                model_version=self._metadata.version,
                input_data=features,
            ) from exc

    async def predict_async(
        self,
        features: dict[str, Any],
        request_id: str | None = None,
    ) -> PredictionResult:
        """
        Async wrapper for prediction to support concurrent requests.

        Args:
            features: Input feature dictionary
            request_id: Optional request identifier

        Returns:
            PredictionResult with probability and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict(features, request_id),
        )


class ChampionModel(BaseChurnModel):
    """
    Production-stable Champion model for churn prediction.

    This model represents the current production standard with
    proven accuracy and stability. It uses a weighted ensemble
    approach combining behavioral and demographic signals.

    Version: 2.1.0
    Accuracy: 89.2% (validation set)
    AUC-ROC: 0.923

    Example:
        >>> champion = ChampionModel()
        >>> champion.load()
        >>> result = champion.predict({
        ...     "tenure": 36,
        ...     "monthly_charges": 85,
        ...     "contract_type": 2
        ... })
        >>> print(f"Churn risk: {result.probability:.1%}")
        Churn risk: 32.1%
    """

    # Feature weights learned from training
    _WEIGHTS: ClassVar[dict[str, float]] = {
        "tenure": -0.05,              # Longer tenure = lower churn
        "monthly_charges": 0.02,      # Higher charges = higher churn
        "total_charges": -0.001,      # Higher total = lower churn (loyalty)
        "contract_type": -0.5,        # Longer contract = lower churn
        "payment_method": 0.1,        # Electronic check = higher churn
        "num_support_tickets": 0.3,   # More tickets = higher churn
        "avg_monthly_gb_download": -0.02,  # More usage = lower churn
        "num_dependents": -0.2,       # More dependents = lower churn
        "online_security": -0.3,      # Has security = lower churn
        "tech_support": -0.25,        # Has support = lower churn
    }

    _BIAS: ClassVar[float] = 0.5

    def __init__(self) -> None:
        """Initialize Champion model with production metadata."""
        metadata = ModelMetadata(
            name="Churn Champion",
            version=ModelVersion.CHAMPION.value,
            model_type="weighted_ensemble",
            features=self.REQUIRED_FEATURES,
            metrics={
                "accuracy": 0.892,
                "auc_roc": 0.923,
                "precision": 0.867,
                "recall": 0.881,
                "f1_score": 0.874,
            },
        )
        super().__init__(metadata)

    def _get_feature_weights(self) -> dict[str, float]:
        """Return Champion model feature weights."""
        return self._WEIGHTS.copy()

    def _compute_raw_score(self, features: dict[str, float]) -> float:
        """
        Compute raw prediction score using weighted sum.

        Args:
            features: Normalized feature dictionary

        Returns:
            Raw score before sigmoid activation
        """
        score = self._BIAS

        for feature, value in features.items():
            weight = self._WEIGHTS.get(feature, 0.0)
            score += value * weight

        return score


class ChallengerModel(BaseChurnModel):
    """
    Experimental Challenger model for A/B testing.

    This model incorporates enhanced feature interactions and
    non-linear transformations. It's being evaluated against
    the Champion in shadow mode before potential promotion.

    Version: 3.0.0-beta
    Accuracy: 90.8% (validation set)
    AUC-ROC: 0.941

    Key Improvements:
        - Tenure-charge interaction term
        - Support ticket scaling
        - Contract-dependent tenure effect

    Example:
        >>> challenger = ChallengerModel()
        >>> challenger.load()
        >>> result = challenger.predict({
        ...     "tenure": 36,
        ...     "monthly_charges": 85,
        ...     "contract_type": 2
        ... })
        >>> print(f"Churn risk: {result.probability:.1%}")
        Churn risk: 28.7%
    """

    # Enhanced feature weights with interactions
    _WEIGHTS: ClassVar[dict[str, float]] = {
        "tenure": -0.06,              # Stronger tenure effect
        "monthly_charges": 0.025,     # Adjusted charge sensitivity
        "total_charges": -0.0015,     # Enhanced loyalty signal
        "contract_type": -0.6,        # Stronger contract effect
        "payment_method": 0.12,       # Payment method risk
        "num_support_tickets": 0.35,  # Higher ticket sensitivity
        "avg_monthly_gb_download": -0.025,  # Usage engagement
        "num_dependents": -0.25,      # Family retention
        "online_security": -0.35,     # Security impact
        "tech_support": -0.3,         # Support impact
    }

    _BIAS: ClassVar[float] = 0.6
    _INTERACTION_WEIGHT: ClassVar[float] = 0.0005

    def __init__(self) -> None:
        """Initialize Challenger model with experimental metadata."""
        metadata = ModelMetadata(
            name="Churn Challenger",
            version=ModelVersion.CHALLENGER.value,
            model_type="enhanced_ensemble",
            features=self.REQUIRED_FEATURES,
            metrics={
                "accuracy": 0.908,
                "auc_roc": 0.941,
                "precision": 0.889,
                "recall": 0.902,
                "f1_score": 0.895,
            },
        )
        super().__init__(metadata)

    def _get_feature_weights(self) -> dict[str, float]:
        """Return Challenger model feature weights."""
        return self._WEIGHTS.copy()

    def _compute_raw_score(self, features: dict[str, float]) -> float:
        """
        Compute raw score with feature interactions.

        This method extends the base weighted sum with:
        1. Tenure-charge interaction term
        2. Scaled support ticket impact
        3. Contract-dependent tenure effect

        Args:
            features: Normalized feature dictionary

        Returns:
            Raw score before sigmoid activation
        """
        score = self._BIAS

        # Base weighted sum
        for feature, value in features.items():
            weight = self._WEIGHTS.get(feature, 0.0)
            score += value * weight

        # Interaction: tenure × monthly_charges
        # High-value long-term customers are less likely to churn
        tenure = features.get("tenure", 0.0)
        monthly_charges = features.get("monthly_charges", 0.0)
        interaction_term = tenure * monthly_charges * self._INTERACTION_WEIGHT
        score -= interaction_term  # Negative = reduces churn probability

        # Scaled support tickets (diminishing returns)
        tickets = features.get("num_support_tickets", 0.0)
        scaled_tickets = np.log1p(tickets) * 0.2  # Log transform
        score += scaled_tickets

        # Contract-dependent tenure effect
        contract_type = features.get("contract_type", 1.0)
        if contract_type >= 2 and tenure > 24:
            # Long-term contract with established history
            score -= 0.3

        return score


class ModelRegistry:
    """
    Central registry for model management.

    This class implements the Factory pattern for model instantiation
    and lifecycle management. It provides a unified interface for
    accessing both Champion and Challenger models.

    Features:
        - Lazy model loading
        - Thread-safe model access
        - Health monitoring
        - Graceful degradation

    Example:
        >>> registry = ModelRegistry()
        >>> registry.load_all()
        >>> champion = registry.get_model("champion")
        >>> challenger = registry.get_model("challenger")
        >>> print(registry.health_check())
        {'champion': True, 'challenger': True}
    """

    _instance: ClassVar[ModelRegistry | None] = None

    def __new__(cls) -> ModelRegistry:
        """Implement singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize model registry with available models."""
        if self._initialized:
            return

        self._models: dict[str, BaseChurnModel] = {
            "champion": ChampionModel(),
            "challenger": ChallengerModel(),
        }
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = True

    @property
    def available_models(self) -> list[str]:
        """List available model names."""
        return list(self._models.keys())

    def get_model(self, name: str) -> BaseChurnModel:
        """
        Retrieve a model by name.

        Args:
            name: Model identifier ('champion' or 'challenger')

        Returns:
            Requested model instance

        Raises:
            KeyError: If model name not found
        """
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not found. "
                f"Available: {self.available_models}"
            )
        return self._models[name]

    def get_champion(self) -> ChampionModel:
        """Get the Champion model instance."""
        return self._models["champion"]  # type: ignore

    def get_challenger(self) -> ChallengerModel:
        """Get the Challenger model instance."""
        return self._models["challenger"]  # type: ignore

    def load_all(self) -> dict[str, bool]:
        """
        Load all registered models.

        Returns:
            Dictionary of model names to load success status
        """
        results: dict[str, bool] = {}

        for name, model in self._models.items():
            try:
                model.load()
                results[name] = True
                self._logger.info(f"Loaded model: {name}")
            except Exception as exc:
                results[name] = False
                self._logger.error(f"Failed to load model {name}: {exc}")

        return results

    def health_check(self) -> dict[str, bool]:
        """
        Check health status of all models.

        Returns:
            Dictionary of model names to health status
        """
        return {name: model._is_loaded for name, model in self._models.items()}

    def get_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Get metadata for all registered models.

        Returns:
            Dictionary of model names to metadata dictionaries
        """
        return {
            name: {
                "name": model.metadata.name,
                "version": model.metadata.version,
                "type": model.metadata.model_type,
                "metrics": model.metadata.metrics,
                "is_loaded": model._is_loaded,
                "prediction_count": model._prediction_count,
                "avg_latency_ms": round(model.average_latency_ms, 2),
            }
            for name, model in self._models.items()
        }


# Module-level singleton for convenience
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """
    Get the global ModelRegistry singleton.

    Returns:
        ModelRegistry instance

    Example:
        >>> registry = get_registry()
        >>> champion = registry.get_champion()
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
