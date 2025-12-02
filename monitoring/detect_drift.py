#!/usr/bin/env python3
"""
Data Drift Detection Module for Shadow MLOps

This module implements drift detection using Evidently AI patterns.
It compares reference (training) data with current (production) data
to identify significant distribution shifts that may require model retraining.

Features:
- Statistical drift detection (KS test, Chi-square, PSI)
- Feature-level drift analysis
- Threshold-based alerting
- Report generation for CI/CD integration

Usage:
    python detect_drift.py --threshold 0.3
    python detect_drift.py --mode evaluation --threshold 0.3
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("drift_detector")

REPORTS_DIR = Path("monitoring/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA GENERATION (Simulated for Demo)
# =============================================================================


def generate_reference_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate reference (training) data distribution.

    This simulates the data distribution the model was trained on.
    In production, this would be loaded from a feature store or data warehouse.
    """
    np.random.seed(seed)

    data = {
        "customer_id": [f"CUST_{i:06d}" for i in range(n_samples)],
        "event_timestamp": [
            datetime.now() - timedelta(days=random.randint(30, 180))
            for _ in range(n_samples)
        ],
        # Engagement features
        "days_since_last_login": np.random.exponential(scale=15, size=n_samples).astype(int),
        "login_frequency_30d": np.random.poisson(lam=12, size=n_samples),
        "session_duration_avg": np.random.gamma(shape=2, scale=15, size=n_samples),
        "page_views_30d": np.random.poisson(lam=50, size=n_samples),
        # Transaction features
        "total_transactions_90d": np.random.poisson(lam=8, size=n_samples),
        "transaction_value_avg": np.random.lognormal(mean=4, sigma=0.8, size=n_samples),
        "transaction_frequency": np.random.gamma(shape=2, scale=0.5, size=n_samples),
        # Support features
        "support_tickets_30d": np.random.poisson(lam=1.5, size=n_samples),
        "complaint_count_90d": np.random.poisson(lam=0.5, size=n_samples),
        # Subscription features
        "subscription_tenure_days": np.random.exponential(scale=365, size=n_samples).astype(int),
        "satisfaction_score": np.clip(np.random.normal(loc=7, scale=1.5, size=n_samples), 1, 10),
        # Target
        "churned": np.random.binomial(n=1, p=0.15, size=n_samples),
    }

    return pd.DataFrame(data)


def generate_current_data(
    n_samples: int = 5000,
    drift_intensity: float = 0.0,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate current (production) data with optional drift.

    Args:
        n_samples: Number of samples to generate
        drift_intensity: Amount of drift to introduce (0.0 = no drift, 1.0 = severe drift)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with current data distribution
    """
    np.random.seed(seed)

    # Apply drift to distribution parameters
    login_scale = 15 + (drift_intensity * 20)  # Drift: customers logging in less
    transaction_lambda = max(1, 8 - (drift_intensity * 5))  # Drift: fewer transactions
    satisfaction_mean = 7 - (drift_intensity * 2)  # Drift: lower satisfaction
    churn_rate = 0.15 + (drift_intensity * 0.25)  # Drift: higher churn

    data = {
        "customer_id": [f"CUST_{i:06d}" for i in range(n_samples)],
        "event_timestamp": [
            datetime.now() - timedelta(days=random.randint(0, 30))
            for _ in range(n_samples)
        ],
        # Engagement features (with drift)
        "days_since_last_login": np.random.exponential(scale=login_scale, size=n_samples).astype(int),
        "login_frequency_30d": np.random.poisson(lam=max(1, 12 - drift_intensity * 4), size=n_samples),
        "session_duration_avg": np.random.gamma(shape=2, scale=15 - drift_intensity * 5, size=n_samples),
        "page_views_30d": np.random.poisson(lam=max(10, 50 - drift_intensity * 20), size=n_samples),
        # Transaction features (with drift)
        "total_transactions_90d": np.random.poisson(lam=transaction_lambda, size=n_samples),
        "transaction_value_avg": np.random.lognormal(mean=4 - drift_intensity * 0.5, sigma=0.8 + drift_intensity * 0.3, size=n_samples),
        "transaction_frequency": np.random.gamma(shape=2, scale=max(0.1, 0.5 - drift_intensity * 0.2), size=n_samples),
        # Support features (with drift)
        "support_tickets_30d": np.random.poisson(lam=1.5 + drift_intensity * 2, size=n_samples),
        "complaint_count_90d": np.random.poisson(lam=0.5 + drift_intensity * 1.5, size=n_samples),
        # Subscription features
        "subscription_tenure_days": np.random.exponential(scale=365, size=n_samples).astype(int),
        "satisfaction_score": np.clip(np.random.normal(loc=satisfaction_mean, scale=1.5, size=n_samples), 1, 10),
        # Target (with drift)
        "churned": np.random.binomial(n=1, p=min(0.9, churn_rate), size=n_samples),
    }

    return pd.DataFrame(data)


# =============================================================================
# DRIFT DETECTION ALGORITHMS
# =============================================================================


class DriftDetector:
    """
    Drift detection using statistical tests and metrics.

    Implements multiple drift detection methods:
    - Kolmogorov-Smirnov (KS) test for numerical features
    - Population Stability Index (PSI) for overall distribution shift
    - Jensen-Shannon divergence for probability distributions
    """

    def __init__(self, threshold: float = 0.3):
        """
        Initialize drift detector.

        Args:
            threshold: Drift score threshold for alerting (0.0 - 1.0)
        """
        self.threshold = threshold
        self.drift_results: Dict[str, Any] = {}

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in distribution between two datasets.
        PSI < 0.1: No significant shift
        PSI 0.1-0.25: Moderate shift, monitor
        PSI > 0.25: Significant shift, action required

        Args:
            reference: Reference data array
            current: Current data array
            n_bins: Number of bins for histogram

        Returns:
            PSI score
        """
        # Create bins from reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate frequencies
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        # Convert to proportions with smoothing
        ref_props = (ref_counts + 1) / (len(reference) + n_bins)
        cur_props = (cur_counts + 1) / (len(current) + n_bins)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    def calculate_ks_statistic(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.

        Args:
            reference: Reference data array
            current: Current data array

        Returns:
            Tuple of (KS statistic, p-value)
        """
        from scipy import stats

        try:
            ks_stat, p_value = stats.ks_2samp(reference, current)
            return float(ks_stat), float(p_value)
        except ImportError:
            # Fallback if scipy not available
            return self._manual_ks_test(reference, current)

    def _manual_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """Manual KS test implementation without scipy."""
        # Combine and sort
        combined = np.concatenate([reference, current])
        combined_sorted = np.sort(combined)

        # Calculate CDFs
        n_ref = len(reference)
        n_cur = len(current)

        ref_cdf = np.searchsorted(np.sort(reference), combined_sorted, side="right") / n_ref
        cur_cdf = np.searchsorted(np.sort(current), combined_sorted, side="right") / n_cur

        # KS statistic is max difference between CDFs
        ks_stat = np.max(np.abs(ref_cdf - cur_cdf))

        # Approximate p-value (simplified)
        n_eff = (n_ref * n_cur) / (n_ref + n_cur)
        p_value = np.exp(-2 * (ks_stat ** 2) * n_eff)

        return float(ks_stat), float(p_value)

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current datasets.

        Args:
            reference_df: Reference (training) data
            current_df: Current (production) data
            feature_columns: Columns to analyze (defaults to numeric columns)

        Returns:
            Dictionary containing drift analysis results
        """
        if feature_columns is None:
            feature_columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude non-feature columns
            exclude_cols = ["customer_id", "event_timestamp", "churned"]
            feature_columns = [c for c in feature_columns if c not in exclude_cols]

        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_reference_samples": len(reference_df),
            "n_current_samples": len(current_df),
            "threshold": self.threshold,
            "features": {},
            "summary": {},
        }

        drift_scores = []
        drifted_features = []

        for col in feature_columns:
            ref_data = reference_df[col].dropna().values
            cur_data = current_df[col].dropna().values

            if len(ref_data) == 0 or len(cur_data) == 0:
                continue

            # Calculate metrics
            psi = self.calculate_psi(ref_data, cur_data)
            ks_stat, ks_pvalue = self.calculate_ks_statistic(ref_data, cur_data)

            # Determine if drifted
            is_drifted = psi > 0.25 or ks_pvalue < 0.05

            results["features"][col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 4),
                "is_drifted": is_drifted,
                "reference_mean": round(float(np.mean(ref_data)), 4),
                "current_mean": round(float(np.mean(cur_data)), 4),
                "mean_shift_pct": round(
                    abs(np.mean(cur_data) - np.mean(ref_data)) / (np.mean(ref_data) + 1e-10) * 100, 2
                ),
            }

            drift_scores.append(psi)
            if is_drifted:
                drifted_features.append(col)

        # Calculate overall drift score
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        dataset_is_drifted = overall_drift_score > self.threshold

        results["summary"] = {
            "overall_drift_score": round(overall_drift_score, 4),
            "dataset_is_drifted": dataset_is_drifted,
            "n_drifted_features": len(drifted_features),
            "drifted_features": drifted_features,
            "drift_threshold": self.threshold,
            "recommendation": (
                "RETRAIN: Significant drift detected. Model retraining recommended."
                if dataset_is_drifted
                else "MONITOR: No significant drift. Continue monitoring."
            ),
        }

        self.drift_results = results
        return results

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate drift detection report.

        Args:
            output_path: Path to save JSON report

        Returns:
            Report as JSON string
        """
        if not self.drift_results:
            raise ValueError("No drift results available. Run detect_drift() first.")

        report_json = json.dumps(self.drift_results, indent=2, default=str)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_json)
            logger.info(f"Drift report saved to {output_path}")

        return report_json


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function for drift detection."""
    parser = argparse.ArgumentParser(
        description="Detect data drift between reference and current datasets"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Drift score threshold for alerting (0.0 - 1.0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detection", "evaluation"],
        default="detection",
        help="Mode: 'detection' for drift check, 'evaluation' for model comparison",
    )
    parser.add_argument(
        "--simulate-drift",
        type=float,
        default=None,
        help="Simulate drift with specified intensity (0.0 - 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for drift report",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Shadow-MLOps Drift Detection")
    logger.info("=" * 60)

    # Determine drift intensity
    if args.simulate_drift is not None:
        drift_intensity = args.simulate_drift
    else:
        # Randomly simulate drift for demo (30% chance of significant drift)
        drift_intensity = random.uniform(0.0, 0.6) if random.random() < 0.3 else random.uniform(0.0, 0.2)

    logger.info(f"Drift intensity: {drift_intensity:.2f}")
    logger.info(f"Threshold: {args.threshold}")

    # Generate datasets
    logger.info("Generating reference dataset...")
    reference_df = generate_reference_data(n_samples=10000)
    logger.info(f"Reference dataset: {len(reference_df)} samples")

    logger.info("Generating current dataset...")
    current_df = generate_current_data(n_samples=5000, drift_intensity=drift_intensity)
    logger.info(f"Current dataset: {len(current_df)} samples")

    # Detect drift
    detector = DriftDetector(threshold=args.threshold)
    logger.info("Running drift detection...")
    results = detector.detect_drift(reference_df, current_df)

    # Output results
    logger.info("-" * 60)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("-" * 60)
    logger.info(f"Overall Drift Score: {results['summary']['overall_drift_score']:.4f}")
    logger.info(f"Dataset Drifted: {results['summary']['dataset_is_drifted']}")
    logger.info(f"Drifted Features: {results['summary']['n_drifted_features']}")

    if results['summary']['drifted_features']:
        logger.info(f"Features with drift: {', '.join(results['summary']['drifted_features'])}")

    logger.info(f"Recommendation: {results['summary']['recommendation']}")
    logger.info("-" * 60)

    # Save report
    output_path = Path(args.output) if args.output else REPORTS_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    detector.generate_report(output_path)

    # Exit with appropriate code for CI/CD
    if results["summary"]["dataset_is_drifted"]:
        logger.warning("⚠️ DRIFT DETECTED - Raising alert for retraining pipeline")
        sys.exit(1)  # Exit with error to trigger retraining
    else:
        logger.info("✅ No significant drift detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
