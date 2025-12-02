"""
Champion and Challenger Models for Churn Prediction

This module provides model implementations for the shadow deployment pattern.
In production, these would load trained ML models (e.g., from MLflow, pickle files).
For demonstration, these use simplified prediction logic.
"""

import hashlib
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


class BaseChurnModel(ABC):
    """Abstract base class for churn prediction models."""

    def __init__(self, version: str, model_name: str):
        self.version = version
        self.model_name = model_name
        self.load_timestamp = datetime.utcnow().isoformat()
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the model (load weights, setup, etc.)."""
        pass

    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a churn prediction.

        Args:
            features: Dictionary of customer features

        Returns:
            Dictionary with prediction results
        """
        pass

    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """Validate input features have required fields."""
        required_fields = [
            "customer_id",
            "days_since_last_login",
            "login_frequency_30d",
            "session_duration_avg",
            "total_transactions_90d",
            "transaction_value_avg",
            "support_tickets_30d",
            "subscription_tenure_days",
            "satisfaction_score",
        ]
        return all(field in features for field in required_fields)

    def _normalize_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Normalize features for model input."""
        # Extract numeric features
        numeric_features = [
            features.get("days_since_last_login", 0) / 365,  # Normalize to ~1 year
            features.get("login_frequency_30d", 0) / 30,  # Normalize to daily
            features.get("session_duration_avg", 0) / 60,  # Normalize to 1 hour
            features.get("total_transactions_90d", 0) / 100,  # Normalize to 100 txns
            features.get("transaction_value_avg", 0) / 1000,  # Normalize to $1000
            features.get("support_tickets_30d", 0) / 10,  # Normalize to 10 tickets
            features.get("subscription_tenure_days", 0) / 730,  # Normalize to 2 years
            features.get("satisfaction_score", 5) / 10,  # Already 0-10 scale
        ]
        return np.array(numeric_features)


class ChampionModel(BaseChurnModel):
    """
    Champion (Production) Model for Churn Prediction.

    This is the current production model that serves real predictions.
    Uses a weighted scoring approach based on customer behavior metrics.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(version="2.1.0", model_name="ChampionV2")
        self.model_path = model_path

    def _initialize_model(self) -> None:
        """Initialize champion model weights."""
        # Simulated model weights (in production, load from MLflow/pickle)
        self.weights = np.array([
            0.25,   # days_since_last_login (higher = more likely to churn)
            -0.15,  # login_frequency_30d (higher = less likely to churn)
            -0.10,  # session_duration_avg
            -0.20,  # total_transactions_90d
            -0.10,  # transaction_value_avg
            0.15,   # support_tickets_30d (more tickets = more likely to churn)
            -0.10,  # subscription_tenure_days
            -0.25,  # satisfaction_score
        ])
        self.bias = 0.3

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate churn prediction using champion model.

        Uses weighted linear combination with sigmoid activation.
        """
        if not self._validate_features(features):
            raise ValueError("Missing required features for prediction")

        # Normalize and compute prediction
        normalized = self._normalize_features(features)
        raw_score = np.dot(normalized, self.weights) + self.bias

        # Apply sigmoid for probability
        churn_probability = 1 / (1 + np.exp(-raw_score))

        # Add slight noise for realism (simulates model uncertainty)
        churn_probability = np.clip(
            churn_probability + np.random.normal(0, 0.02), 0, 1
        )

        return {
            "churn_probability": float(churn_probability),
            "churn_prediction": bool(churn_probability >= 0.5),
            "model_version": self.version,
            "model_name": self.model_name,
            "confidence": float(abs(churn_probability - 0.5) * 2),  # 0-1 confidence
        }


class ChallengerModel(BaseChurnModel):
    """
    Challenger (Shadow) Model for Churn Prediction.

    This is the new model being tested in shadow mode.
    Uses an updated architecture with different feature weights
    and additional risk factors.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(version="3.0.0-beta", model_name="ChallengerV3")
        self.model_path = model_path

    def _initialize_model(self) -> None:
        """Initialize challenger model with updated weights."""
        # Updated weights reflecting new training (slightly different from champion)
        self.weights = np.array([
            0.30,   # days_since_last_login - increased importance
            -0.20,  # login_frequency_30d - increased importance
            -0.08,  # session_duration_avg
            -0.25,  # total_transactions_90d - increased importance
            -0.12,  # transaction_value_avg
            0.20,   # support_tickets_30d - increased importance
            -0.12,  # subscription_tenure_days
            -0.30,  # satisfaction_score - increased importance
        ])
        self.bias = 0.25

        # Additional interaction terms for challenger model
        self.interaction_weight = 0.1

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate churn prediction using challenger model.

        Uses updated weights and includes feature interactions.
        """
        if not self._validate_features(features):
            raise ValueError("Missing required features for prediction")

        # Normalize and compute base prediction
        normalized = self._normalize_features(features)
        raw_score = np.dot(normalized, self.weights) + self.bias

        # Add interaction term: login frequency * satisfaction score
        # (engaged but unsatisfied customers are at risk)
        login_idx, satisfaction_idx = 1, 7
        interaction = normalized[login_idx] * (1 - normalized[satisfaction_idx])
        raw_score += interaction * self.interaction_weight

        # Apply sigmoid for probability
        churn_probability = 1 / (1 + np.exp(-raw_score))

        # Add slight noise for realism
        churn_probability = np.clip(
            churn_probability + np.random.normal(0, 0.015), 0, 1
        )

        return {
            "churn_probability": float(churn_probability),
            "churn_prediction": bool(churn_probability >= 0.5),
            "model_version": self.version,
            "model_name": self.model_name,
            "confidence": float(abs(churn_probability - 0.5) * 2),
            "feature_interactions_used": True,
        }


class ModelRegistry:
    """
    Model Registry for managing Champion and Challenger models.

    In production, this would integrate with MLflow or similar.
    """

    def __init__(self):
        self.models: Dict[str, BaseChurnModel] = {}
        self.current_champion: Optional[str] = None
        self.current_challenger: Optional[str] = None

    def register_model(
        self,
        model: BaseChurnModel,
        role: str = "candidate",
    ) -> str:
        """Register a model in the registry."""
        model_id = f"{model.model_name}_{model.version}"
        self.models[model_id] = model

        if role == "champion":
            self.current_champion = model_id
        elif role == "challenger":
            self.current_challenger = model_id

        return model_id

    def get_champion(self) -> Optional[BaseChurnModel]:
        """Get the current champion model."""
        if self.current_champion:
            return self.models.get(self.current_champion)
        return None

    def get_challenger(self) -> Optional[BaseChurnModel]:
        """Get the current challenger model."""
        if self.current_challenger:
            return self.models.get(self.current_challenger)
        return None

    def promote_challenger(self) -> bool:
        """Promote challenger to champion."""
        if self.current_challenger:
            self.current_champion = self.current_challenger
            self.current_challenger = None
            return True
        return False

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models."""
        return {
            model_id: {
                "name": model.model_name,
                "version": model.version,
                "role": (
                    "champion"
                    if model_id == self.current_champion
                    else ("challenger" if model_id == self.current_challenger else "candidate")
                ),
                "loaded_at": model.load_timestamp,
            }
            for model_id, model in self.models.items()
        }
