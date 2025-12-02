"""
Feast Feature Definitions for Churn Prediction Shadow Deployment

This module defines feature views for the churn prediction system,
enabling both Champion and Challenger models to access consistent
feature sets during shadow deployment.
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# =============================================================================
# DATA SOURCES
# =============================================================================

# Source for customer churn statistics
churn_stats_source = FileSource(
    name="churn_stats_source",
    path="data/churn_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Customer churn statistics data source for model training and inference",
)

# Source for customer demographics
customer_demographics_source = FileSource(
    name="customer_demographics_source",
    path="data/customer_demographics.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Customer demographic information for feature enrichment",
)

# =============================================================================
# ENTITIES
# =============================================================================

customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique identifier for each customer in the churn prediction system",
    join_keys=["customer_id"],
)

# =============================================================================
# FEATURE VIEWS
# =============================================================================

churn_stats_fv = FeatureView(
    name="churn_stats",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        # Engagement Metrics
        Feature(name="days_since_last_login", dtype=Int64),
        Feature(name="login_frequency_30d", dtype=Float32),
        Feature(name="session_duration_avg", dtype=Float32),
        Feature(name="page_views_30d", dtype=Int64),
        
        # Transaction Metrics
        Feature(name="total_transactions_90d", dtype=Int64),
        Feature(name="transaction_value_avg", dtype=Float32),
        Feature(name="transaction_frequency", dtype=Float32),
        Feature(name="days_since_last_transaction", dtype=Int64),
        
        # Support Metrics
        Feature(name="support_tickets_30d", dtype=Int64),
        Feature(name="complaint_count_90d", dtype=Int64),
        Feature(name="avg_resolution_time", dtype=Float32),
        
        # Subscription Metrics
        Feature(name="subscription_tenure_days", dtype=Int64),
        Feature(name="subscription_tier", dtype=String),
        Feature(name="renewal_count", dtype=Int64),
        Feature(name="upgrade_count", dtype=Int64),
        Feature(name="downgrade_count", dtype=Int64),
        
        # Churn Risk Indicators
        Feature(name="churn_risk_score", dtype=Float32),
        Feature(name="engagement_score", dtype=Float32),
        Feature(name="satisfaction_score", dtype=Float32),
    ],
    source=churn_stats_source,
    online=True,
    description="Customer churn statistics for real-time prediction serving",
    tags={
        "team": "mlops",
        "use_case": "churn_prediction",
        "shadow_deployment": "enabled",
    },
)

customer_demographics_fv = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Feature(name="age", dtype=Int64),
        Feature(name="tenure_months", dtype=Int64),
        Feature(name="account_type", dtype=String),
        Feature(name="region", dtype=String),
        Feature(name="acquisition_channel", dtype=String),
        Feature(name="lifetime_value", dtype=Float32),
    ],
    source=customer_demographics_source,
    online=True,
    description="Customer demographic features for model enrichment",
    tags={
        "team": "mlops",
        "use_case": "churn_prediction",
        "pii_level": "low",
    },
)

# =============================================================================
# FEATURE SERVICES (for organized feature retrieval)
# =============================================================================

from feast import FeatureService

churn_prediction_service = FeatureService(
    name="churn_prediction_service",
    features=[
        churn_stats_fv,
        customer_demographics_fv,
    ],
    description="Feature service for churn prediction models (Champion & Challenger)",
    tags={
        "version": "v1.0",
        "shadow_deployment": "enabled",
    },
)
