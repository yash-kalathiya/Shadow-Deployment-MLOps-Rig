"""
Tests for Drift Detection Module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_psi_no_drift(self, drift_detector):
        """PSI should be low when distributions are similar."""
        np.random.seed(42)
        reference = np.random.normal(loc=0, scale=1, size=1000)
        current = np.random.normal(loc=0, scale=1, size=1000)
        
        psi = drift_detector.calculate_psi(reference, current)
        assert psi < 0.1, f"PSI should be < 0.1 for similar distributions, got {psi}"

    def test_psi_with_drift(self, drift_detector):
        """PSI should be high when distributions differ significantly."""
        np.random.seed(42)
        reference = np.random.normal(loc=0, scale=1, size=1000)
        current = np.random.normal(loc=3, scale=2, size=1000)  # Shifted distribution
        
        psi = drift_detector.calculate_psi(reference, current)
        assert psi > 0.25, f"PSI should be > 0.25 for different distributions, got {psi}"

    def test_ks_statistic_similar_distributions(self, drift_detector):
        """KS statistic should be low for similar distributions."""
        np.random.seed(42)
        reference = np.random.normal(loc=0, scale=1, size=1000)
        current = np.random.normal(loc=0, scale=1, size=1000)
        
        ks_stat, p_value = drift_detector.calculate_ks_statistic(reference, current)
        assert ks_stat < 0.1, f"KS statistic should be low, got {ks_stat}"
        assert p_value > 0.05, f"P-value should be > 0.05, got {p_value}"

    def test_ks_statistic_different_distributions(self, drift_detector):
        """KS statistic should be high for different distributions."""
        np.random.seed(42)
        reference = np.random.normal(loc=0, scale=1, size=1000)
        current = np.random.normal(loc=5, scale=1, size=1000)  # Shifted mean
        
        ks_stat, p_value = drift_detector.calculate_ks_statistic(reference, current)
        assert ks_stat > 0.3, f"KS statistic should be high, got {ks_stat}"
        assert p_value < 0.05, f"P-value should be < 0.05, got {p_value}"

    def test_detect_drift_no_drift(self, drift_detector, reference_df, current_df_no_drift):
        """Detect drift should return no drift for similar data."""
        results = drift_detector.detect_drift(reference_df, current_df_no_drift)
        
        assert "summary" in results
        assert not results["summary"]["dataset_is_drifted"]

    def test_detect_drift_with_drift(self, drift_detector, reference_df, current_df_with_drift):
        """Detect drift should detect significant drift."""
        results = drift_detector.detect_drift(reference_df, current_df_with_drift)
        
        assert "summary" in results
        assert results["summary"]["dataset_is_drifted"]
        assert results["summary"]["n_drifted_features"] > 0

    def test_detect_drift_feature_results(self, drift_detector, reference_df, current_df_no_drift):
        """Detect drift should return per-feature results."""
        results = drift_detector.detect_drift(reference_df, current_df_no_drift)
        
        assert "features" in results
        for feature_name, feature_result in results["features"].items():
            assert "psi" in feature_result
            assert "ks_statistic" in feature_result
            assert "is_drifted" in feature_result

    def test_generate_report(self, drift_detector, reference_df, current_df_no_drift, tmp_path):
        """Generate report should create JSON file."""
        drift_detector.detect_drift(reference_df, current_df_no_drift)
        
        report_path = tmp_path / "drift_report.json"
        report = drift_detector.generate_report(report_path)
        
        assert report_path.exists()
        assert len(report) > 0


class TestDataGeneration:
    """Tests for data generation functions."""

    def test_generate_reference_data(self):
        """Reference data generation should produce valid DataFrame."""
        from monitoring.detect_drift import generate_reference_data
        
        df = generate_reference_data(n_samples=100)
        
        assert len(df) == 100
        assert "customer_id" in df.columns
        assert "churned" in df.columns

    def test_generate_current_data_no_drift(self):
        """Current data with no drift should be similar to reference."""
        from monitoring.detect_drift import generate_reference_data, generate_current_data
        
        ref_df = generate_reference_data(n_samples=1000)
        cur_df = generate_current_data(n_samples=1000, drift_intensity=0.0)
        
        # Check that mean values are similar
        ref_mean = ref_df["satisfaction_score"].mean()
        cur_mean = cur_df["satisfaction_score"].mean()
        
        assert abs(ref_mean - cur_mean) < 1.0

    def test_generate_current_data_with_drift(self):
        """Current data with drift should differ from reference."""
        from monitoring.detect_drift import generate_reference_data, generate_current_data
        
        ref_df = generate_reference_data(n_samples=1000)
        cur_df = generate_current_data(n_samples=1000, drift_intensity=0.8)
        
        # Check that satisfaction scores differ significantly
        ref_mean = ref_df["satisfaction_score"].mean()
        cur_mean = cur_df["satisfaction_score"].mean()
        
        assert abs(ref_mean - cur_mean) > 1.0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def drift_detector():
    """Create drift detector instance."""
    from monitoring.detect_drift import DriftDetector
    return DriftDetector(threshold=0.3)


@pytest.fixture
def reference_df():
    """Generate reference DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_1": np.random.normal(loc=0, scale=1, size=1000),
        "feature_2": np.random.exponential(scale=5, size=1000),
        "feature_3": np.random.poisson(lam=10, size=1000),
    })


@pytest.fixture
def current_df_no_drift():
    """Generate current DataFrame without drift."""
    np.random.seed(123)
    return pd.DataFrame({
        "feature_1": np.random.normal(loc=0, scale=1, size=1000),
        "feature_2": np.random.exponential(scale=5, size=1000),
        "feature_3": np.random.poisson(lam=10, size=1000),
    })


@pytest.fixture
def current_df_with_drift():
    """Generate current DataFrame with significant drift."""
    np.random.seed(456)
    return pd.DataFrame({
        "feature_1": np.random.normal(loc=5, scale=2, size=1000),  # Shifted
        "feature_2": np.random.exponential(scale=15, size=1000),   # Scaled
        "feature_3": np.random.poisson(lam=25, size=1000),         # Different lambda
    })
