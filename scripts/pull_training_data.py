#!/usr/bin/env python3
"""
Training Data Extraction Script

Pulls historical feature data from Feast feature store for model training.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_training_data(
    output_path: str,
    start_date: str,
    end_date: str,
    n_samples: int = 50000,
) -> None:
    """
    Generate sample training data for demonstration.
    
    In production, this would query the Feast feature store:
    ```python
    from feast import FeatureStore
    
    store = FeatureStore(repo_path="feature_repo")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=["churn_stats:days_since_last_login", ...],
    ).to_df()
    ```
    """
    logger.info(f"Generating training data: {n_samples} samples")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    np.random.seed(42)
    
    # Generate synthetic training data
    data = {
        "customer_id": [f"CUST_{i:06d}" for i in range(n_samples)],
        "event_timestamp": pd.date_range(start=start_date, end=end_date, periods=n_samples),
        
        # Engagement features
        "days_since_last_login": np.random.exponential(scale=15, size=n_samples).astype(int),
        "login_frequency_30d": np.random.poisson(lam=12, size=n_samples).astype(float),
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
        
        # Target variable
        "churned": np.random.binomial(n=1, p=0.15, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Training data saved to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Churn rate: {df['churned'].mean():.2%}")


def main():
    parser = argparse.ArgumentParser(description="Pull training data from feature store")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples")
    
    args = parser.parse_args()
    
    generate_sample_training_data(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        n_samples=args.samples,
    )


if __name__ == "__main__":
    main()
