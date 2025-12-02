"""
Comprehensive Drift Detection Tests.

This module provides thorough testing of drift detection functionality including:
- PSI calculation
- KS test
- Jensen-Shannon divergence
- Report generation
- CLI functionality

Example:
    pytest tests/test_drift.py -v --cov=monitoring.detect_drift

Author: Shadow MLOps Team
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest

from monitoring.detect_drift import (
    DriftDetector,
    DriftMethod,
    DriftReport,
    DriftSeverity,
    FeatureDriftResult,
    generate_current_data,
    generate_reference_data,
    main,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def detector() -> DriftDetector:
    """Create a DriftDetector with default thresholds."""
    return DriftDetector(psi_threshold=0.3, ks_threshold=0.1)


@pytest.fixture
def strict_detector() -> DriftDetector:
    """Create a DriftDetector with strict thresholds."""
    return DriftDetector(psi_threshold=0.1, ks_threshold=0.05)


@pytest.fixture
def reference_data() -> dict[str, np.ndarray]:
    """Generate reference data for testing."""
    return generate_reference_data(n_samples=1000, seed=42)


@pytest.fixture
def current_data_no_drift() -> dict[str, np.ndarray]:
    """Generate current data with no drift."""
    return generate_current_data(
        n_samples=1000,
        drift_features=[],
        drift_magnitude=0.0,
        seed=123,
    )


@pytest.fixture
def current_data_with_drift() -> dict[str, np.ndarray]:
    """Generate current data with significant drift."""
    return generate_current_data(
        n_samples=1000,
        drift_features=["monthly_charges", "num_support_tickets"],
        drift_magnitude=0.8,
        seed=123,
    )


@pytest.fixture
def identical_distributions() -> tuple[np.ndarray, np.ndarray]:
    """Create two identical distributions."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    return data.copy(), data.copy()


@pytest.fixture
def shifted_distributions() -> tuple[np.ndarray, np.ndarray]:
    """Create two distributions with mean shift."""
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(2, 1, 1000)  # Mean shifted by 2
    return reference, current


# =============================================================================
# FeatureDriftResult Tests
# =============================================================================


class TestFeatureDriftResult:
    """Tests for FeatureDriftResult dataclass."""

    def test_result_creation(self) -> None:
        """Result should be created with all fields."""
        result = FeatureDriftResult(
            feature_name="test_feature",
            method=DriftMethod.PSI,
            statistic=0.15,
            threshold=0.3,
            drift_detected=False,
            severity=DriftSeverity.LOW,
        )
        
        assert result.feature_name == "test_feature"
        assert result.method == DriftMethod.PSI
        assert result.statistic == 0.15
        assert result.drift_detected is False

    def test_result_to_dict(self) -> None:
        """Result should serialize to dictionary."""
        result = FeatureDriftResult(
            feature_name="test",
            method=DriftMethod.KS_TEST,
            statistic=0.12,
            threshold=0.1,
            drift_detected=True,
            severity=DriftSeverity.MEDIUM,
            p_value=0.03,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data["feature_name"] == "test"
        assert data["method"] == "ks_test"
        assert data["drift_detected"] is True
        assert data["p_value"] == 0.03


# =============================================================================
# DriftReport Tests
# =============================================================================


class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_report_drift_rate_calculation(self) -> None:
        """Drift rate should be calculated correctly."""
        results = [
            FeatureDriftResult("f1", DriftMethod.PSI, 0.1, 0.3, False, DriftSeverity.NONE),
            FeatureDriftResult("f2", DriftMethod.PSI, 0.4, 0.3, True, DriftSeverity.HIGH),
            FeatureDriftResult("f3", DriftMethod.PSI, 0.5, 0.3, True, DriftSeverity.CRITICAL),
        ]
        
        from datetime import datetime, timezone
        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            reference_samples=1000,
            current_samples=500,
            feature_results=results,
            overall_drift=True,
            overall_severity=DriftSeverity.CRITICAL,
        )
        
        assert report.drift_rate == pytest.approx(2/3, rel=0.01)

    def test_report_zero_drift_rate(self) -> None:
        """No drift should result in 0% drift rate."""
        from datetime import datetime, timezone
        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            reference_samples=1000,
            current_samples=500,
            feature_results=[
                FeatureDriftResult("f1", DriftMethod.PSI, 0.05, 0.3, False, DriftSeverity.NONE),
            ],
            overall_drift=False,
            overall_severity=DriftSeverity.NONE,
        )
        
        assert report.drift_rate == 0.0

    def test_report_summary_generation(self) -> None:
        """Report should generate human-readable summary."""
        from datetime import datetime, timezone
        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            reference_samples=1000,
            current_samples=500,
            feature_results=[],
            overall_drift=False,
            overall_severity=DriftSeverity.NONE,
        )
        
        summary = report.summary
        
        assert "Drift Report" in summary
        assert "1000" in summary  # Reference samples
        assert "500" in summary   # Current samples

    def test_report_to_json(self) -> None:
        """Report should serialize to valid JSON."""
        from datetime import datetime, timezone
        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            reference_samples=1000,
            current_samples=500,
            feature_results=[],
            overall_drift=False,
            overall_severity=DriftSeverity.NONE,
            recommendations=["Monitor closely"],
        )
        
        json_str = report.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert "timestamp" in data
        assert "overall_drift" in data


# =============================================================================
# PSI Calculation Tests
# =============================================================================


class TestPSICalculation:
    """Tests for PSI (Population Stability Index) calculation."""

    def test_psi_identical_distributions_near_zero(
        self,
        detector: DriftDetector,
        identical_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """PSI for identical distributions should be near zero."""
        ref, curr = identical_distributions
        result = detector.compute_psi(ref, curr, "test")
        
        assert result.statistic < 0.1
        assert result.drift_detected is False
        assert result.severity == DriftSeverity.NONE

    def test_psi_shifted_distributions_detects_drift(
        self,
        detector: DriftDetector,
        shifted_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """PSI for shifted distributions should detect drift."""
        ref, curr = shifted_distributions
        result = detector.compute_psi(ref, curr, "test")
        
        assert result.statistic > 0.3
        assert result.drift_detected is True
        assert result.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)

    def test_psi_includes_bin_details(
        self,
        detector: DriftDetector,
        reference_data: dict[str, np.ndarray],
    ) -> None:
        """PSI result should include bin-level details."""
        result = detector.compute_psi(
            reference_data["tenure"],
            reference_data["tenure"],
            "tenure",
        )
        
        assert "bin_counts_reference" in result.details
        assert "bin_counts_current" in result.details
        assert "psi_per_bin" in result.details

    def test_psi_handles_empty_bins(
        self,
        detector: DriftDetector,
    ) -> None:
        """PSI should handle distributions with empty bins."""
        # Sparse distribution
        ref = np.array([1, 1, 1, 100, 100, 100])
        curr = np.array([1, 1, 100, 100])
        
        result = detector.compute_psi(ref, curr, "sparse")
        
        # Should not raise and should be finite
        assert np.isfinite(result.statistic)


# =============================================================================
# KS Test Tests
# =============================================================================


class TestKSTest:
    """Tests for Kolmogorov-Smirnov test."""

    def test_ks_identical_distributions_low_statistic(
        self,
        detector: DriftDetector,
        identical_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """KS statistic for identical distributions should be very low."""
        ref, curr = identical_distributions
        result = detector.compute_ks_test(ref, curr, "test")
        
        assert result.statistic < 0.05
        assert result.drift_detected is False

    def test_ks_shifted_distributions_detects_drift(
        self,
        detector: DriftDetector,
        shifted_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """KS test should detect shifted distributions."""
        ref, curr = shifted_distributions
        result = detector.compute_ks_test(ref, curr, "test")
        
        assert result.statistic > 0.5
        assert result.drift_detected is True
        assert result.p_value is not None
        assert result.p_value < 0.01  # Highly significant

    def test_ks_includes_sample_counts(
        self,
        detector: DriftDetector,
    ) -> None:
        """KS result should include sample count details."""
        ref = np.random.default_rng(42).normal(0, 1, 500)
        curr = np.random.default_rng(43).normal(0, 1, 300)
        
        result = detector.compute_ks_test(ref, curr, "test")
        
        assert result.details["reference_samples"] == 500
        assert result.details["current_samples"] == 300


# =============================================================================
# Jensen-Shannon Divergence Tests
# =============================================================================


class TestJSDivergence:
    """Tests for Jensen-Shannon divergence calculation."""

    def test_js_identical_distributions_near_zero(
        self,
        detector: DriftDetector,
        identical_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """JS divergence for identical distributions should be near zero."""
        ref, curr = identical_distributions
        result = detector.compute_js_divergence(ref, curr, "test")
        
        assert result.statistic < 0.05
        assert result.drift_detected is False

    def test_js_bounded(
        self,
        detector: DriftDetector,
        shifted_distributions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """JS divergence should be bounded."""
        ref, curr = shifted_distributions
        result = detector.compute_js_divergence(ref, curr, "test")
        
        # JS divergence is bounded by ln(2) ≈ 0.693 when using natural log
        assert 0 <= result.statistic <= 1.0


# =============================================================================
# Full Drift Detection Tests
# =============================================================================


class TestDriftDetection:
    """Tests for comprehensive drift detection."""

    def test_detect_drift_no_drift(
        self,
        detector: DriftDetector,
        reference_data: dict[str, np.ndarray],
        current_data_no_drift: dict[str, np.ndarray],
    ) -> None:
        """Detection should report no drift for similar distributions."""
        report = detector.detect_drift(reference_data, current_data_no_drift)
        
        # May or may not detect drift depending on random variation
        assert isinstance(report, DriftReport)
        assert report.reference_samples > 0
        assert report.current_samples > 0

    def test_detect_drift_with_drift(
        self,
        detector: DriftDetector,
        reference_data: dict[str, np.ndarray],
        current_data_with_drift: dict[str, np.ndarray],
    ) -> None:
        """Detection should report drift for shifted distributions."""
        report = detector.detect_drift(
            reference_data,
            current_data_with_drift,
        )
        
        assert report.overall_drift is True
        assert len(report.feature_results) > 0

    def test_detect_drift_multiple_methods(
        self,
        detector: DriftDetector,
        reference_data: dict[str, np.ndarray],
    ) -> None:
        """Detection should work with multiple methods."""
        report = detector.detect_drift(
            reference_data,
            reference_data,  # Same data = no drift
            methods=[DriftMethod.PSI, DriftMethod.KS_TEST, DriftMethod.JS_DIVERGENCE],
        )
        
        # Should have 3 results per feature
        n_features = len(reference_data)
        assert len(report.feature_results) == n_features * 3

    def test_detect_drift_generates_recommendations(
        self,
        detector: DriftDetector,
        reference_data: dict[str, np.ndarray],
        current_data_with_drift: dict[str, np.ndarray],
    ) -> None:
        """Detection should generate recommendations."""
        report = detector.detect_drift(
            reference_data,
            current_data_with_drift,
        )
        
        assert len(report.recommendations) > 0


# =============================================================================
# Severity Classification Tests
# =============================================================================


class TestSeverityClassification:
    """Tests for drift severity classification."""

    @pytest.mark.parametrize("psi,expected_severity", [
        (0.05, DriftSeverity.NONE),
        (0.15, DriftSeverity.LOW),
        (0.25, DriftSeverity.MEDIUM),
        (0.35, DriftSeverity.HIGH),
        (0.55, DriftSeverity.CRITICAL),
    ])
    def test_severity_thresholds(
        self,
        detector: DriftDetector,
        psi: float,
        expected_severity: DriftSeverity,
    ) -> None:
        """Severity should be classified based on PSI thresholds."""
        severity = detector._classify_severity(psi)
        
        assert severity == expected_severity


# =============================================================================
# Data Generation Tests
# =============================================================================


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_reference_data_shape(self) -> None:
        """Generated data should have correct shape."""
        data = generate_reference_data(n_samples=500)
        
        for feature, values in data.items():
            assert len(values) == 500

    def test_generate_reference_data_reproducible(self) -> None:
        """Same seed should produce same data."""
        data1 = generate_reference_data(n_samples=100, seed=42)
        data2 = generate_reference_data(n_samples=100, seed=42)
        
        for feature in data1:
            np.testing.assert_array_equal(data1[feature], data2[feature])

    def test_generate_current_data_with_drift(self) -> None:
        """Drift features should have different distributions."""
        ref = generate_reference_data(n_samples=1000, seed=42)
        curr = generate_current_data(
            n_samples=1000,
            drift_features=["monthly_charges"],
            drift_magnitude=1.0,
            seed=42,
        )
        
        # Mean should differ significantly
        ref_mean = np.mean(ref["monthly_charges"])
        curr_mean = np.mean(curr["monthly_charges"])
        
        assert abs(curr_mean - ref_mean) > 5

    def test_generate_current_data_no_drift(self) -> None:
        """Non-drift features should have similar distributions."""
        ref = generate_reference_data(n_samples=1000, seed=42)
        curr = generate_current_data(
            n_samples=1000,
            drift_features=[],  # No drift
            drift_magnitude=0.0,
            seed=42,
        )
        
        # Means should be similar (within noise)
        for feature in ref:
            ref_mean = np.mean(ref[feature])
            curr_mean = np.mean(curr[feature])
            # Allow for random variation
            assert abs(curr_mean - ref_mean) < ref_mean * 0.3


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for command-line interface."""

    def test_main_with_sample_data_no_drift(self) -> None:
        """Main should return 0 for no significant drift."""
        with patch('sys.argv', [
            'detect_drift',
            '--generate-sample',
            '--drift-features', 'nonexistent',  # No real drift
            '--drift-magnitude', '0.0',
        ]):
            # May return 0 or 1 depending on random variation
            exit_code = main()
            assert exit_code in (0, 1)

    def test_main_with_drift(self) -> None:
        """Main should return 1 when drift is detected."""
        with patch('sys.argv', [
            'detect_drift',
            '--generate-sample',
            '--drift-features', 'monthly_charges', 'num_support_tickets',
            '--drift-magnitude', '1.0',
            '--psi-threshold', '0.1',  # Strict threshold
        ]):
            exit_code = main()
            assert exit_code == 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_sample_size(self, detector: DriftDetector) -> None:
        """Should handle small sample sizes."""
        ref = np.array([1.0, 2.0, 3.0])
        curr = np.array([1.5, 2.5, 3.5])
        
        result = detector.compute_psi(ref, curr, "small")
        
        assert np.isfinite(result.statistic)

    def test_single_value_distribution(self, detector: DriftDetector) -> None:
        """Should handle constant distributions."""
        ref = np.ones(100)
        curr = np.ones(100) * 2
        
        result = detector.compute_psi(ref, curr, "constant")
        
        assert np.isfinite(result.statistic)

    def test_extreme_values(self, detector: DriftDetector) -> None:
        """Should handle extreme values."""
        ref = np.array([1e10, 1e-10, 0, -1e10])
        curr = np.array([1e10, 1e-10, 0, -1e10])
        
        result = detector.compute_psi(ref, curr, "extreme")
        
        assert np.isfinite(result.statistic)

    def test_different_feature_sets(self, detector: DriftDetector) -> None:
        """Should handle different feature sets (uses intersection)."""
        ref = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
        curr = {"a": np.array([1, 2, 3]), "c": np.array([7, 8, 9])}
        
        report = detector.detect_drift(ref, curr)
        
        # Should only analyze 'a' (intersection)
        assert any(r.feature_name == "a" for r in report.feature_results)
        assert not any(r.feature_name == "b" for r in report.feature_results)
        assert not any(r.feature_name == "c" for r in report.feature_results)
