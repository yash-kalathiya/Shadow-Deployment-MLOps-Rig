"""
Pytest Configuration and Shared Fixtures.

This module provides shared fixtures and configuration for all tests
in the Shadow Deployment & Drift Detection Platform.

Example:
    pytest tests/ -v --cov=src --cov=monitoring

Author: Shadow MLOps Team
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Generator

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Automatically mark tests based on location."""
    for item in items:
        # Mark tests in test_integration.py as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark async tests
        if "async" in item.name:
            item.add_marker(pytest.mark.asyncio)


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get the test data directory."""
    return project_root / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def reset_singletons() -> Generator[None, None, None]:
    """Reset singleton instances before each test."""
    from src.models import ModelRegistry
    
    original_instance = ModelRegistry._instance
    yield
    ModelRegistry._instance = original_instance


@pytest.fixture
def env_vars() -> Generator[dict[str, str], None, None]:
    """Fixture to temporarily set environment variables."""
    original_env = os.environ.copy()
    modified_vars: dict[str, str] = {}
    
    class EnvVarSetter(dict):
        def __setitem__(self, key: str, value: str) -> None:
            modified_vars[key] = value
            os.environ[key] = value
    
    env_setter = EnvVarSetter()
    yield env_setter
    
    for key in modified_vars:
        if key in original_env:
            os.environ[key] = original_env[key]
        else:
            del os.environ[key]


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def sample_features() -> dict[str, Any]:
    """Standard feature set for model predictions."""
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
def high_risk_features() -> dict[str, Any]:
    """Features indicating high churn risk."""
    return {
        "tenure": 1,
        "monthly_charges": 100.0,
        "total_charges": 100.0,
        "contract_type": 0,
        "payment_method": 2,
        "num_support_tickets": 10,
        "avg_monthly_gb_download": 2.0,
        "num_dependents": 0,
        "online_security": 0,
        "tech_support": 0,
    }


@pytest.fixture
def low_risk_features() -> dict[str, Any]:
    """Features indicating low churn risk."""
    return {
        "tenure": 60,
        "monthly_charges": 40.0,
        "total_charges": 2400.0,
        "contract_type": 2,
        "payment_method": 0,
        "num_support_tickets": 0,
        "avg_monthly_gb_download": 25.0,
        "num_dependents": 3,
        "online_security": 1,
        "tech_support": 1,
    }


# =============================================================================
# API Fixtures
# =============================================================================


@pytest.fixture
def api_base_url() -> str:
    """Base URL for API tests."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def valid_prediction_request(sample_features: dict[str, Any]) -> dict[str, Any]:
    """Valid prediction request payload."""
    return {
        "customer_id": "CUST-12345",
        **sample_features,
    }


@pytest.fixture
def batch_prediction_request(sample_features: dict[str, Any]) -> dict[str, Any]:
    """Valid batch prediction request payload."""
    return {
        "requests": [
            {"customer_id": f"CUST-{i:05d}", **sample_features}
            for i in range(5)
        ]
    }


# =============================================================================
# Logging Fixtures
# =============================================================================


@pytest.fixture
def capture_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture log output for assertions."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog
