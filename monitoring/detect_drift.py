#!/usr/bin/env python3
"""
Statistical Drift Detection Module.

This module implements production-grade drift detection using multiple
statistical methods to identify distribution shifts between reference
(training) and current (production) data.

Supported Methods:
    - PSI (Population Stability Index): Measures distribution shift
    - KS Test (Kolmogorov-Smirnov): Non-parametric distribution comparison
    - Chi-Square Test: Categorical variable drift
    - Jensen-Shannon Divergence: Symmetric KL divergence

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     DriftDetector                               │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
    │  │   PSI Detector  │  │   KS Detector   │  │ Chi-Sq Detector │ │
    │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
    │           │                    │                     │          │
    │           └────────────────────┼─────────────────────┘          │
    │                                │                                 │
    │                    ┌───────────▼───────────┐                    │
    │                    │    DriftReport        │                    │
    │                    │  (Aggregated Results) │                    │
    │                    └───────────────────────┘                    │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    $ python -m monitoring.detect_drift --reference data/reference.csv --current data/current.csv
    $ python -m monitoring.detect_drift --generate-sample

Exit Codes:
    0: No significant drift detected
    1: Drift detected (requires action)
    2: Error in drift detection

Author: Shadow MLOps Team
Version: 2.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Drift severity classification levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftMethod(Enum):
    """Available drift detection methods."""

    PSI = "psi"
    KS_TEST = "ks_test"
    CHI_SQUARE = "chi_square"
    JS_DIVERGENCE = "js_divergence"


@dataclass
class FeatureDriftResult:
    """
    Drift detection result for a single feature.

    Attributes:
        feature_name: Name of the analyzed feature
        method: Detection method used
        statistic: Computed test statistic
        threshold: Threshold for drift determination
        drift_detected: Whether drift exceeds threshold
        severity: Categorized drift severity
        p_value: Statistical p-value (if applicable)
        details: Additional method-specific details
    """

    feature_name: str
    method: DriftMethod
    statistic: float
    threshold: float
    drift_detected: bool
    severity: DriftSeverity
    p_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "feature_name": self.feature_name,
            "method": self.method.value,
            "statistic": round(self.statistic, 6),
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "severity": self.severity.value,
            "p_value": round(self.p_value, 6) if self.p_value else None,
            "details": self.details,
        }


@dataclass
class DriftReport:
    """
    Comprehensive drift detection report.

    Attributes:
        timestamp: Report generation timestamp
        reference_samples: Number of reference samples
        current_samples: Number of current samples
        feature_results: Per-feature drift results
        overall_drift: Whether overall drift detected
        overall_severity: Maximum severity across features
        recommendations: Action recommendations
    """

    timestamp: datetime
    reference_samples: int
    current_samples: int
    feature_results: list[FeatureDriftResult]
    overall_drift: bool
    overall_severity: DriftSeverity
    recommendations: list[str] = field(default_factory=list)

    @property
    def drift_rate(self) -> float:
        """Calculate percentage of features with drift."""
        if not self.feature_results:
            return 0.0
        drifted = sum(1 for r in self.feature_results if r.drift_detected)
        return drifted / len(self.feature_results)

    @property
    def summary(self) -> str:
        """Generate human-readable summary."""
        drifted_features = [r.feature_name for r in self.feature_results if r.drift_detected]
        return (
            f"Drift Report ({self.timestamp.isoformat()})\n"
            f"  Reference samples: {self.reference_samples}\n"
            f"  Current samples: {self.current_samples}\n"
            f"  Overall drift: {self.overall_drift}\n"
            f"  Severity: {self.overall_severity.value}\n"
            f"  Drift rate: {self.drift_rate:.1%}\n"
            f"  Drifted features: {drifted_features or 'None'}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reference_samples": self.reference_samples,
            "current_samples": self.current_samples,
            "feature_results": [r.to_dict() for r in self.feature_results],
            "overall_drift": self.overall_drift,
            "overall_severity": self.overall_severity.value,
            "drift_rate": round(self.drift_rate, 4),
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class DriftDetector:
    """
    Multi-method drift detection engine.

    This class provides a comprehensive drift detection framework
    supporting multiple statistical methods with configurable
    thresholds and severity classification.

    Attributes:
        psi_threshold: PSI threshold for drift detection
        ks_threshold: KS statistic threshold
        num_bins: Number of bins for histogram-based methods

    Example:
        >>> detector = DriftDetector(psi_threshold=0.25)
        >>> report = detector.detect_drift(reference_data, current_data)
        >>> if report.overall_drift:
        ...     print("Drift detected! Action required.")
    """

    # PSI severity thresholds (industry standard)
    PSI_THRESHOLDS = {
        DriftSeverity.LOW: 0.1,
        DriftSeverity.MEDIUM: 0.2,
        DriftSeverity.HIGH: 0.3,
        DriftSeverity.CRITICAL: 0.5,
    }

    def __init__(
        self,
        psi_threshold: float = 0.3,
        ks_threshold: float = 0.1,
        num_bins: int = 10,
    ) -> None:
        """
        Initialize drift detector with thresholds.

        Args:
            psi_threshold: PSI threshold for drift flag
            ks_threshold: KS statistic threshold for drift flag
            num_bins: Number of bins for histogram computation
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.num_bins = num_bins
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _classify_severity(self, psi_value: float) -> DriftSeverity:
        """
        Classify drift severity based on PSI value.

        Args:
            psi_value: Computed PSI value

        Returns:
            Severity classification
        """
        if psi_value < self.PSI_THRESHOLDS[DriftSeverity.LOW]:
            return DriftSeverity.NONE
        elif psi_value < self.PSI_THRESHOLDS[DriftSeverity.MEDIUM]:
            return DriftSeverity.LOW
        elif psi_value < self.PSI_THRESHOLDS[DriftSeverity.HIGH]:
            return DriftSeverity.MEDIUM
        elif psi_value < self.PSI_THRESHOLDS[DriftSeverity.CRITICAL]:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> FeatureDriftResult:
        """
        Compute Population Stability Index (PSI).

        PSI measures the shift in distribution between two datasets.
        It's calculated as: PSI = Σ (actual% - expected%) × ln(actual% / expected%)

        Interpretation:
            PSI < 0.1: No significant change
            0.1 ≤ PSI < 0.2: Slight change, monitor
            0.2 ≤ PSI < 0.3: Moderate change, investigate
            PSI ≥ 0.3: Significant change, action required

        Args:
            reference: Reference distribution (training data)
            current: Current distribution (production data)
            feature_name: Name of the feature being analyzed

        Returns:
            FeatureDriftResult with PSI computation
        """
        # Compute bin edges from combined data for consistent binning
        combined = np.concatenate([reference, current])
        bin_edges = np.histogram_bin_edges(combined, bins=self.num_bins)

        # Compute histograms
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions with Laplace smoothing
        epsilon = 1e-6
        ref_props = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
        curr_props = (curr_counts + epsilon) / (curr_counts.sum() + epsilon * len(curr_counts))

        # Calculate PSI
        psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
        psi = float(np.sum(psi_values))

        severity = self._classify_severity(psi)
        drift_detected = psi >= self.psi_threshold

        return FeatureDriftResult(
            feature_name=feature_name,
            method=DriftMethod.PSI,
            statistic=psi,
            threshold=self.psi_threshold,
            drift_detected=drift_detected,
            severity=severity,
            details={
                "bin_counts_reference": ref_counts.tolist(),
                "bin_counts_current": curr_counts.tolist(),
                "psi_per_bin": psi_values.tolist(),
            },
        )

    def compute_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> FeatureDriftResult:
        """
        Compute Kolmogorov-Smirnov test statistic.

        The KS test measures the maximum absolute difference between
        two cumulative distribution functions (CDFs).

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature

        Returns:
            FeatureDriftResult with KS test results
        """
        # Sort data for CDF computation
        ref_sorted = np.sort(reference)
        curr_sorted = np.sort(current)

        # Combine and sort all unique values
        all_values = np.unique(np.concatenate([ref_sorted, curr_sorted]))

        # Compute CDFs
        ref_cdf = np.searchsorted(ref_sorted, all_values, side="right") / len(ref_sorted)
        curr_cdf = np.searchsorted(curr_sorted, all_values, side="right") / len(curr_sorted)

        # KS statistic is max absolute difference
        ks_statistic = float(np.max(np.abs(ref_cdf - curr_cdf)))

        # Approximate p-value using asymptotic formula
        n1, n2 = len(reference), len(current)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = float(np.exp(-2 * (ks_statistic * en) ** 2))

        drift_detected = ks_statistic >= self.ks_threshold
        severity = (
            DriftSeverity.HIGH if drift_detected and ks_statistic > 0.2
            else DriftSeverity.MEDIUM if drift_detected
            else DriftSeverity.NONE
        )

        return FeatureDriftResult(
            feature_name=feature_name,
            method=DriftMethod.KS_TEST,
            statistic=ks_statistic,
            threshold=self.ks_threshold,
            drift_detected=drift_detected,
            severity=severity,
            p_value=p_value,
            details={
                "reference_samples": n1,
                "current_samples": n2,
            },
        )

    def compute_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> FeatureDriftResult:
        """
        Compute Jensen-Shannon Divergence.

        JS divergence is a symmetric and bounded measure of
        distribution similarity (unlike KL divergence).

        Args:
            reference: Reference distribution
            current: Current distribution
            feature_name: Name of the feature

        Returns:
            FeatureDriftResult with JS divergence
        """
        # Compute histograms
        combined = np.concatenate([reference, current])
        bin_edges = np.histogram_bin_edges(combined, bins=self.num_bins)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalize to probability distributions
        epsilon = 1e-10
        ref_prob = ref_counts / (ref_counts.sum() + epsilon)
        curr_prob = curr_counts / (curr_counts.sum() + epsilon)

        # Average distribution
        avg_prob = (ref_prob + curr_prob) / 2

        # KL divergences
        def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
            mask = (p > 0) & (q > 0)
            return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

        kl_ref_avg = kl_divergence(ref_prob + epsilon, avg_prob + epsilon)
        kl_curr_avg = kl_divergence(curr_prob + epsilon, avg_prob + epsilon)

        # JS divergence (bounded [0, 1] when using log2, [0, ln(2)] with ln)
        js_div = (kl_ref_avg + kl_curr_avg) / 2

        # Threshold at 0.1 (moderate divergence)
        threshold = 0.1
        drift_detected = js_div >= threshold
        severity = (
            DriftSeverity.HIGH if js_div > 0.3
            else DriftSeverity.MEDIUM if js_div > 0.15
            else DriftSeverity.LOW if drift_detected
            else DriftSeverity.NONE
        )

        return FeatureDriftResult(
            feature_name=feature_name,
            method=DriftMethod.JS_DIVERGENCE,
            statistic=js_div,
            threshold=threshold,
            drift_detected=drift_detected,
            severity=severity,
        )

    def detect_drift(
        self,
        reference_data: dict[str, np.ndarray],
        current_data: dict[str, np.ndarray],
        methods: list[DriftMethod] | None = None,
    ) -> DriftReport:
        """
        Perform comprehensive drift detection across all features.

        Args:
            reference_data: Reference data keyed by feature name
            current_data: Current data keyed by feature name
            methods: Detection methods to apply (default: PSI + KS)

        Returns:
            DriftReport with all results and recommendations
        """
        if methods is None:
            methods = [DriftMethod.PSI, DriftMethod.KS_TEST]

        results: list[FeatureDriftResult] = []
        features = set(reference_data.keys()) & set(current_data.keys())

        self._logger.info(f"Analyzing drift for {len(features)} features")

        for feature in sorted(features):
            ref_values = reference_data[feature]
            curr_values = current_data[feature]

            for method in methods:
                if method == DriftMethod.PSI:
                    result = self.compute_psi(ref_values, curr_values, feature)
                elif method == DriftMethod.KS_TEST:
                    result = self.compute_ks_test(ref_values, curr_values, feature)
                elif method == DriftMethod.JS_DIVERGENCE:
                    result = self.compute_js_divergence(ref_values, curr_values, feature)
                else:
                    continue

                results.append(result)

                if result.drift_detected:
                    self._logger.warning(
                        f"Drift detected in '{feature}' using {method.value}: "
                        f"statistic={result.statistic:.4f}, severity={result.severity.value}"
                    )

        # Aggregate results
        overall_drift = any(r.drift_detected for r in results)
        max_severity = max(
            (r.severity for r in results),
            key=lambda s: list(DriftSeverity).index(s),
            default=DriftSeverity.NONE,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(results, max_severity)

        # Get sample counts
        ref_samples = len(next(iter(reference_data.values()), []))
        curr_samples = len(next(iter(current_data.values()), []))

        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            reference_samples=ref_samples,
            current_samples=curr_samples,
            feature_results=results,
            overall_drift=overall_drift,
            overall_severity=max_severity,
            recommendations=recommendations,
        )

        return report

    def _generate_recommendations(
        self,
        results: list[FeatureDriftResult],
        severity: DriftSeverity,
    ) -> list[str]:
        """Generate actionable recommendations based on drift results."""
        recommendations: list[str] = []

        if severity == DriftSeverity.NONE:
            recommendations.append("✓ No significant drift detected. Continue monitoring.")
            return recommendations

        if severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            recommendations.append("⚠️ URGENT: Significant drift detected. Consider model retraining.")

        drifted_features = [r.feature_name for r in results if r.drift_detected]
        if drifted_features:
            unique_features = sorted(set(drifted_features))
            recommendations.append(f"📊 Investigate drifted features: {', '.join(unique_features)}")

        if severity.value in ("medium", "high", "critical"):
            recommendations.append("📈 Increase monitoring frequency for affected features.")
            recommendations.append("🔍 Review data pipeline for potential upstream changes.")

        if severity == DriftSeverity.CRITICAL:
            recommendations.append("🚨 Consider switching to fallback model or champion rollback.")

        return recommendations


def generate_reference_data(
    n_samples: int = 10000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic reference (training) data.

    Creates realistic customer churn data distributions
    for drift detection testing.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary of feature arrays
    """
    rng = np.random.default_rng(seed)

    return {
        "tenure": rng.exponential(scale=20, size=n_samples).clip(1, 72),
        "monthly_charges": rng.normal(loc=65, scale=20, size=n_samples).clip(20, 120),
        "total_charges": rng.normal(loc=2000, scale=1500, size=n_samples).clip(100, 8000),
        "num_support_tickets": rng.poisson(lam=1.5, size=n_samples),
        "avg_monthly_gb_download": rng.gamma(shape=2, scale=10, size=n_samples),
        "contract_type": rng.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2]),
    }


def generate_current_data(
    n_samples: int = 5000,
    drift_features: list[str] | None = None,
    drift_magnitude: float = 0.3,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic current (production) data with optional drift.

    Args:
        n_samples: Number of samples to generate
        drift_features: Features to inject drift into
        drift_magnitude: How much to shift drifted distributions
        seed: Random seed for reproducibility

    Returns:
        Dictionary of feature arrays
    """
    rng = np.random.default_rng(seed)
    drift_features = drift_features or []

    data: dict[str, np.ndarray] = {}

    # Tenure - slight drift toward longer tenure
    drift = drift_magnitude * 5 if "tenure" in drift_features else 0
    data["tenure"] = rng.exponential(scale=20 + drift, size=n_samples).clip(1, 72)

    # Monthly charges - drift toward higher charges
    drift = drift_magnitude * 15 if "monthly_charges" in drift_features else 0
    data["monthly_charges"] = rng.normal(
        loc=65 + drift, scale=20, size=n_samples
    ).clip(20, 130)

    # Total charges
    drift = drift_magnitude * 500 if "total_charges" in drift_features else 0
    data["total_charges"] = rng.normal(
        loc=2000 + drift, scale=1500, size=n_samples
    ).clip(100, 8500)

    # Support tickets - drift toward more tickets
    drift = drift_magnitude * 2 if "num_support_tickets" in drift_features else 0
    data["num_support_tickets"] = rng.poisson(lam=1.5 + drift, size=n_samples)

    # Download usage
    drift = drift_magnitude * 5 if "avg_monthly_gb_download" in drift_features else 0
    data["avg_monthly_gb_download"] = rng.gamma(
        shape=2 + drift, scale=10, size=n_samples
    )

    # Contract type - drift toward month-to-month
    if "contract_type" in drift_features:
        probs = [0.6, 0.25, 0.15]  # More month-to-month
    else:
        probs = [0.5, 0.3, 0.2]
    data["contract_type"] = rng.choice([0, 1, 2], size=n_samples, p=probs)

    return data


def main() -> int:
    """
    Main entry point for drift detection CLI.

    Returns:
        Exit code (0=no drift, 1=drift detected, 2=error)
    """
    parser = argparse.ArgumentParser(
        description="Detect statistical drift between reference and current data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate sample data and detect drift
    python -m monitoring.detect_drift --generate-sample --drift-features tenure monthly_charges

    # Detect drift with custom threshold
    python -m monitoring.detect_drift --generate-sample --psi-threshold 0.2

    # Output report to file
    python -m monitoring.detect_drift --generate-sample --output drift_report.json
        """,
    )

    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate synthetic sample data for testing",
    )
    parser.add_argument(
        "--drift-features",
        nargs="+",
        default=["monthly_charges", "num_support_tickets"],
        help="Features to inject drift into (for sample generation)",
    )
    parser.add_argument(
        "--drift-magnitude",
        type=float,
        default=0.4,
        help="Drift magnitude for sample generation (default: 0.4)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.3,
        help="PSI threshold for drift detection (default: 0.3)",
    )
    parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.1,
        help="KS statistic threshold (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for JSON report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Generate or load data
        if args.generate_sample:
            logger.info("Generating synthetic sample data...")
            reference_data = generate_reference_data()
            current_data = generate_current_data(
                drift_features=args.drift_features,
                drift_magnitude=args.drift_magnitude,
            )
            logger.info(
                f"Generated {len(next(iter(reference_data.values())))} reference samples "
                f"and {len(next(iter(current_data.values())))} current samples"
            )
            logger.info(f"Drift injected into: {args.drift_features}")
        else:
            logger.error("No data source specified. Use --generate-sample or provide data files.")
            return 2

        # Perform drift detection
        detector = DriftDetector(
            psi_threshold=args.psi_threshold,
            ks_threshold=args.ks_threshold,
        )

        logger.info("Running drift detection...")
        report = detector.detect_drift(
            reference_data,
            current_data,
            methods=[DriftMethod.PSI, DriftMethod.KS_TEST],
        )

        # Output results
        print("\n" + "=" * 60)
        print(report.summary)
        print("=" * 60)

        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  {rec}")

        # Save report if output specified
        if args.output:
            args.output.write_text(report.to_json())
            logger.info(f"Report saved to: {args.output}")

        # Return appropriate exit code
        if report.overall_drift:
            logger.warning(
                f"DRIFT DETECTED - Severity: {report.overall_severity.value.upper()}"
            )
            return 1
        else:
            logger.info("No significant drift detected")
            return 0

    except Exception as exc:
        logger.exception(f"Drift detection failed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
