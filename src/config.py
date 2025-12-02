"""
Configuration Management for Shadow MLOps API

This module provides centralized configuration using Pydantic Settings,
supporting environment variables, .env files, and sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    Example: API_HOST=0.0.0.0 API_PORT=8080 python -m src.api
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # Application Settings
    # ==========================================================================
    app_name: str = Field(
        default="Shadow-MLOps API",
        description="Application name for logging and monitoring",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )
    environment: str = Field(
        default="development",
        description="Deployment environment: development, staging, production",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    
    # ==========================================================================
    # API Server Settings
    # ==========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port",
    )
    api_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of API workers",
    )
    
    # ==========================================================================
    # CORS Settings
    # ==========================================================================
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests",
    )
    
    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting",
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Rate limit window in seconds",
    )
    
    # ==========================================================================
    # Model Settings
    # ==========================================================================
    champion_model_version: str = Field(
        default="2.1.0",
        description="Champion model version",
    )
    challenger_model_version: str = Field(
        default="3.0.0-beta",
        description="Challenger model version",
    )
    model_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        description="Model prediction timeout",
    )
    
    # ==========================================================================
    # Shadow Deployment Settings
    # ==========================================================================
    shadow_mode_enabled: bool = Field(
        default=True,
        description="Enable shadow mode predictions",
    )
    shadow_log_max_entries: int = Field(
        default=10000,
        ge=100,
        description="Maximum shadow log entries to retain",
    )
    
    # ==========================================================================
    # Drift Detection Settings
    # ==========================================================================
    drift_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Drift detection threshold",
    )
    drift_check_interval_hours: int = Field(
        default=24,
        ge=1,
        description="Hours between drift checks",
    )
    
    # ==========================================================================
    # Logging Settings
    # ==========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )
    
    # ==========================================================================
    # Feature Store Settings
    # ==========================================================================
    feast_repo_path: Path = Field(
        default=Path("feature_repo"),
        description="Path to Feast feature repository",
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def shadow_log_file(self) -> Path:
        """Get shadow log file path."""
        return self.log_dir / "shadow_logs.json"
    
    @property
    def api_log_file(self) -> Path:
        """Get API log file path."""
        return self.log_dir / "api.log"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience export
settings = get_settings()
