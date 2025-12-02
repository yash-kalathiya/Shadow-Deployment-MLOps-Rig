"""
Custom Exceptions for Shadow MLOps API

This module defines custom exception classes for better error handling
and more informative error responses.
"""

from typing import Any, Dict, Optional


class ShadowMLOpsError(Exception):
    """Base exception for Shadow MLOps application."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


class ModelNotLoadedError(ShadowMLOpsError):
    """Raised when a model is not loaded or available."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Model '{model_name}' is not loaded or unavailable",
            error_code="MODEL_NOT_LOADED",
            status_code=503,
            details=details,
        )


class ModelPredictionError(ShadowMLOpsError):
    """Raised when model prediction fails."""
    
    def __init__(self, model_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Prediction failed for model '{model_name}': {reason}",
            error_code="PREDICTION_FAILED",
            status_code=500,
            details=details,
        )


class FeatureValidationError(ShadowMLOpsError):
    """Raised when input features fail validation."""
    
    def __init__(self, missing_features: list, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Missing required features: {', '.join(missing_features)}",
            error_code="FEATURE_VALIDATION_FAILED",
            status_code=422,
            details={"missing_features": missing_features, **(details or {})},
        )


class RateLimitExceededError(ShadowMLOpsError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after_seconds": retry_after, **(details or {})},
        )


class DriftDetectionError(ShadowMLOpsError):
    """Raised when drift detection fails."""
    
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Drift detection failed: {reason}",
            error_code="DRIFT_DETECTION_FAILED",
            status_code=500,
            details=details,
        )


class FeatureStoreError(ShadowMLOpsError):
    """Raised when feature store operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Feature store {operation} failed: {reason}",
            error_code="FEATURE_STORE_ERROR",
            status_code=503,
            details=details,
        )


class ShadowLogError(ShadowMLOpsError):
    """Raised when shadow logging fails."""
    
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Shadow logging failed: {reason}",
            error_code="SHADOW_LOG_ERROR",
            status_code=500,
            details=details,
        )


class CircuitBreakerOpenError(ShadowMLOpsError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service: str, retry_after: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Service '{service}' is temporarily unavailable (circuit breaker open)",
            error_code="CIRCUIT_BREAKER_OPEN",
            status_code=503,
            details={"service": service, "retry_after_seconds": retry_after, **(details or {})},
        )
